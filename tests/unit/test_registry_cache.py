"""Registry cache behaviour. Spec 17.3, NFR-003.

A cache miss is free. A cache that serves a STALE registry would inject a constraint the user
revoked, or omit one they just issued, and would do so silently. These tests are mostly about
that second thing.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from scguard.registry.cache import (
    CACHE_PREFIX,
    CachedRegistryStore,
    InMemoryRegistryCache,
    RedisRegistryCache,
)
from scguard.registry.store import (
    InMemoryRegistryStore,
    SCCategory,
    SCStatus,
    Session,
    build_constraint,
)

SESSION = "sess_cache"
TENANT = "tenant_cache"


async def make_store() -> tuple[CachedRegistryStore, InMemoryRegistryStore, InMemoryRegistryCache]:
    inner = InMemoryRegistryStore()
    await inner.create_session(
        Session(
            session_id=SESSION,
            tenant_id=TENANT,
            extractor_model="qwen3.5-9b",
            prompt_hash="sha256:abc",
        )
    )
    cache = InMemoryRegistryCache()
    return CachedRegistryStore(inner, cache), inner, cache


def constraint(text: str, seq: int = 0, category: SCCategory = SCCategory.ACTION):
    return build_constraint(
        session_id=SESSION,
        tenant_id=TENANT,
        seq=seq,
        canonical_text=text,
        category=category,
        source_turn_index=seq,
        token_count=10,
    )


# ---------------------------------------------------------------- hit and miss


async def test_second_read_is_served_from_cache() -> None:
    store, _, cache = await make_store()
    await store.append(constraint("Never send email."))
    await store.active(SESSION)
    await store.active(SESSION)
    assert cache.hits == 1
    assert cache.misses == 1


async def test_append_invalidates_the_cache() -> None:
    """A newly issued constraint must appear on the very next read."""
    store, _, cache = await make_store()
    await store.append(constraint("Never send email.", seq=0))
    first = await store.active(SESSION)
    assert len(first) == 1

    await store.append(constraint("Use metric units.", seq=1, category=SCCategory.PREFERENCE))
    second = await store.active(SESSION)
    assert len(second) == 2, "the cache served a stale registry"
    assert cache.invalidations >= 1


async def test_revocation_invalidates_the_cache() -> None:
    """A revoked constraint must stop being injected immediately."""
    store, _, _ = await make_store()
    row = await store.append(constraint("Never send email."))
    assert len(await store.active(SESSION)) == 1

    await store.set_status(SESSION, row.constraint_id, SCStatus.REVOKED, None)
    assert await store.active(SESSION) == (), "a revoked constraint was still served"


async def test_version_mismatch_is_treated_as_a_miss() -> None:
    """A lost invalidation costs a database read, never a wrong answer."""
    store, inner, _cache = await make_store()
    await store.append(constraint("Never send email.", seq=0))
    await store.active(SESSION)

    # Mutate through the INNER store, so the cache is never told.
    await inner.append(constraint("Use metric units.", seq=1, category=SCCategory.PREFERENCE))
    refreshed = await store.active(SESSION)
    assert len(refreshed) == 2, "the version check failed to catch a lost invalidation"


async def test_cache_is_scoped_per_session() -> None:
    store, inner, _ = await make_store()
    other = "sess_cache_other"
    await inner.create_session(
        Session(session_id=other, tenant_id=TENANT, extractor_model="m", prompt_hash="h")
    )
    await store.append(constraint("Never send email."))
    await store.active(SESSION)
    assert await store.active(other) == ()


async def test_audit_path_is_not_cached() -> None:
    """all_constraints powers audit and transparency, where stale would be worse than slow."""
    store, inner, _ = await make_store()
    row = await store.append(constraint("Never send email."))
    assert len(await store.all_constraints(SESSION)) == 1
    await inner.set_status(SESSION, row.constraint_id, SCStatus.REVOKED, None)
    rows = await store.all_constraints(SESSION)
    assert rows[0].status is SCStatus.REVOKED


# ---------------------------------------------------------------- redis backend


class FakeRedis:
    """Minimal async Redis, with a switch to make every call fail."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.available = True
        self.deletes = 0

    async def get(self, key: str) -> str | None:
        if not self.available:
            raise ConnectionError("redis is down")
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        if not self.available:
            raise ConnectionError("redis is down")
        self.store[key] = value

    async def delete(self, key: str) -> None:
        if not self.available:
            raise ConnectionError("redis is down")
        self.deletes += 1
        self.store.pop(key, None)


def sample_constraint() -> Any:
    return constraint("Never send email.")


async def test_redis_round_trip() -> None:
    client = FakeRedis()
    cache = RedisRegistryCache(client)
    rows = (sample_constraint(),)
    await cache.set(SESSION, 3, rows)
    assert f"{CACHE_PREFIX}{SESSION}" in client.store

    fetched = await cache.get(SESSION, 3)
    assert fetched is not None
    assert fetched[0].canonical_text == "Never send email."
    assert cache.hits == 1


async def test_redis_version_mismatch_is_a_miss() -> None:
    cache = RedisRegistryCache(FakeRedis())
    await cache.set(SESSION, 3, (sample_constraint(),))
    assert await cache.get(SESSION, 4) is None
    assert cache.misses == 1


async def test_redis_outage_degrades_to_a_miss_rather_than_raising() -> None:
    """Falling back to the store yields the RIGHT answer, so degrading here is correct.

    This is deliberately the opposite of the store failure rule, where degrading would yield an
    empty registry and therefore a wrong one.
    """
    client = FakeRedis()
    cache = RedisRegistryCache(client)
    await cache.set(SESSION, 1, (sample_constraint(),))
    client.available = False

    assert await cache.get(SESSION, 1) is None
    assert cache.errors == 1
    await cache.set(SESSION, 1, (sample_constraint(),))  # must not raise
    await cache.invalidate(SESSION)  # must not raise
    assert cache.errors == 3


async def test_corrupt_cache_entry_is_a_miss() -> None:
    client = FakeRedis()
    client.store[f"{CACHE_PREFIX}{SESSION}"] = "not json at all"
    cache = RedisRegistryCache(client)
    assert await cache.get(SESSION, 1) is None


async def test_cache_entry_from_an_older_schema_is_a_miss() -> None:
    """A deploy that changes the row shape must not deserialize into a wrong object."""
    client = FakeRedis()
    client.store[f"{CACHE_PREFIX}{SESSION}"] = json.dumps(
        {"registry_version": 1, "constraints": [{"unexpected": "shape"}]}
    )
    cache = RedisRegistryCache(client)
    assert await cache.get(SESSION, 1) is None


async def test_redis_invalidate_removes_the_entry() -> None:
    client = FakeRedis()
    cache = RedisRegistryCache(client)
    await cache.set(SESSION, 1, (sample_constraint(),))
    await cache.invalidate(SESSION)
    assert client.deletes == 1
    assert await cache.get(SESSION, 1) is None


async def test_redis_connect_requires_the_driver() -> None:
    import sys

    from shared.errors import ConfigError

    previous = sys.modules.pop("redis.asyncio", None)
    sys.modules["redis.asyncio"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(ConfigError, match="requires the redis package"):
            await RedisRegistryCache.connect("redis://localhost:6379/0")
    finally:
        sys.modules.pop("redis.asyncio", None)
        if previous is not None:
            sys.modules["redis.asyncio"] = previous
