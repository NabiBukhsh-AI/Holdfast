"""Registry read cache. Spec 17.3.

Assembly is latency sensitive (NFR-003 targets under 20 ms excluding compactor time) and reads
the same active registry repeatedly, so the read path is cached.

`THE RULE THAT MATTERS` A cache miss is free. A cache that serves a STALE registry is not: it
would inject a constraint the user revoked, or omit one they just issued, and it would do so
silently. So:

- Every mutation invalidates the session's entry. Invalidation is keyed on the session, not
  time based, because a TTL long enough to be useful is long enough to be wrong.
- The cached value carries the `registry_version` it was built from. A version mismatch is
  treated as a miss, so a lost invalidation costs a database read rather than a wrong answer.
- A cache BACKEND failure falls through to the store. That is the one place in this system
  where degrading is correct, because the fallback is the authoritative source rather than
  silence. This is the opposite of the store failure rule, and the difference is deliberate:
  falling back to Postgres yields the right answer, falling back to an empty registry does not.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from scguard.registry.store import (
    RegistryStore,
    SCCategory,
    SCStatus,
    Session,
    SessionConstraint,
)

logger = logging.getLogger(__name__)

CACHE_PREFIX = "scguard:registry:"
# Short by design. The cache is invalidated explicitly on every mutation; the TTL exists only
# to bound the damage from an invalidation that never arrives (a crashed worker, a network
# partition), not as the primary correctness mechanism.
DEFAULT_TTL_SECONDS = 300


class RegistryCache(Protocol):
    """Cache surface. Every method is allowed to fail; none is allowed to lie."""

    async def get(
        self, session_id: str, registry_version: int
    ) -> tuple[SessionConstraint, ...] | None: ...

    async def set(
        self, session_id: str, registry_version: int, constraints: tuple[SessionConstraint, ...]
    ) -> None: ...

    async def invalidate(self, session_id: str) -> None: ...


def _serialize(registry_version: int, constraints: tuple[SessionConstraint, ...]) -> str:
    return json.dumps(
        {
            "registry_version": registry_version,
            "constraints": [c.model_dump(mode="json") for c in constraints],
        }
    )


def _deserialize(raw: str, expected_version: int) -> tuple[SessionConstraint, ...] | None:
    """Return the cached rows, or None when the entry cannot be trusted.

    Every failure mode here returns None rather than raising or guessing: a corrupt entry, an
    unknown field, or a version mismatch all mean the same thing operationally, which is that
    the caller should read the store.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("registry_cache_corrupt_entry")
        return None
    if not isinstance(payload, dict) or payload.get("registry_version") != expected_version:
        return None
    rows = payload.get("constraints")
    if not isinstance(rows, list):
        return None
    try:
        return tuple(SessionConstraint.model_validate(row) for row in rows)
    except Exception:
        # A schema change means the cached shape is from an older deploy. Treat as a miss.
        logger.warning("registry_cache_schema_mismatch")
        return None


class InMemoryRegistryCache:
    """Process local cache. Correct for a single replica, used by dev and CI."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[int, tuple[SessionConstraint, ...]]] = {}
        self.hits = 0
        self.misses = 0
        self.invalidations = 0

    async def get(
        self, session_id: str, registry_version: int
    ) -> tuple[SessionConstraint, ...] | None:
        entry = self._entries.get(session_id)
        if entry is None or entry[0] != registry_version:
            self.misses += 1
            return None
        self.hits += 1
        return entry[1]

    async def set(
        self, session_id: str, registry_version: int, constraints: tuple[SessionConstraint, ...]
    ) -> None:
        self._entries[session_id] = (registry_version, constraints)

    async def invalidate(self, session_id: str) -> None:
        self.invalidations += 1
        self._entries.pop(session_id, None)


class RedisRegistryCache:
    """Redis backed cache, shared across replicas.

    Every Redis call is wrapped: a cache outage degrades this service to uncached reads, which
    is slower and still correct. It must never take the service down, and it must never be
    allowed to answer from a stale entry.
    """

    def __init__(self, client: Any, *, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._client = client
        self._ttl = ttl_seconds
        self.hits = 0
        self.misses = 0
        self.errors = 0

    @classmethod
    async def connect(
        cls, dsn: str, *, ttl_seconds: int = DEFAULT_TTL_SECONDS
    ) -> RedisRegistryCache:
        try:
            import redis.asyncio as redis
        except ImportError as exc:
            from shared.errors import ConfigError

            raise ConfigError(
                "redis.backend=redis requires the redis package. Install the service extra, "
                "or use backend=memory for dev and CI."
            ) from exc
        return cls(redis.from_url(dsn, decode_responses=True), ttl_seconds=ttl_seconds)

    def _key(self, session_id: str) -> str:
        return f"{CACHE_PREFIX}{session_id}"

    async def get(
        self, session_id: str, registry_version: int
    ) -> tuple[SessionConstraint, ...] | None:
        try:
            raw = await self._client.get(self._key(session_id))
        except Exception as exc:
            # Degrading to the store is correct here: the store is authoritative.
            self.errors += 1
            logger.warning("registry_cache_unavailable", extra={"detail": str(exc)})
            return None
        if raw is None:
            self.misses += 1
            return None
        constraints = _deserialize(raw, registry_version)
        if constraints is None:
            self.misses += 1
            return None
        self.hits += 1
        return constraints

    async def set(
        self, session_id: str, registry_version: int, constraints: tuple[SessionConstraint, ...]
    ) -> None:
        try:
            await self._client.set(
                self._key(session_id), _serialize(registry_version, constraints), ex=self._ttl
            )
        except Exception as exc:
            self.errors += 1
            logger.warning("registry_cache_write_failed", extra={"detail": str(exc)})

    async def invalidate(self, session_id: str) -> None:
        try:
            await self._client.delete(self._key(session_id))
        except Exception as exc:
            # A failed invalidation is the dangerous one, so it is logged at error level. The
            # version check on read is what stops it becoming a wrong answer.
            self.errors += 1
            logger.error("registry_cache_invalidate_failed", extra={"detail": str(exc)})


class CachedRegistryStore:
    """Wraps a store with a read cache, invalidating on every mutation.

    Only `active()` is cached. `all_constraints()` is the audit and transparency path, where a
    stale answer would be worse than a slow one, and it is not on the latency sensitive route.
    """

    def __init__(self, inner: RegistryStore, cache: RegistryCache) -> None:
        self._inner = inner
        self._cache = cache

    @property
    def inner(self) -> RegistryStore:
        return self._inner

    async def create_session(self, session: Session) -> Session:
        return await self._inner.create_session(session)

    async def get_session(self, session_id: str) -> Session:
        return await self._inner.get_session(session_id)

    async def next_seq(self, session_id: str) -> int:
        return await self._inner.next_seq(session_id)

    async def append(self, constraint: SessionConstraint) -> SessionConstraint:
        stored = await self._inner.append(constraint)
        await self._cache.invalidate(constraint.session_id)
        return stored

    async def set_status(
        self,
        session_id: str,
        constraint_id: str,
        status: SCStatus,
        superseded_by: str | None = None,
    ) -> SessionConstraint:
        updated = await self._inner.set_status(session_id, constraint_id, status, superseded_by)
        await self._cache.invalidate(session_id)
        return updated

    async def active(self, session_id: str) -> tuple[SessionConstraint, ...]:
        """Cached on (session, registry_version). A version mismatch is treated as a miss."""
        session = await self._inner.get_session(session_id)
        cached = await self._cache.get(session_id, session.registry_version)
        if cached is not None:
            return cached
        constraints = await self._inner.active(session_id)
        await self._cache.set(session_id, session.registry_version, constraints)
        return constraints

    async def all_constraints(self, session_id: str) -> tuple[SessionConstraint, ...]:
        return await self._inner.all_constraints(session_id)

    async def replace_text(self, session_id: str, constraint_id: str, text: str) -> None:
        replace = getattr(self._inner, "replace_text", None)
        if replace is None:
            from scguard.registry.store import AppendOnlyViolationError

            raise AppendOnlyViolationError("session_constraints is append only (FR-080)")
        await replace(session_id, constraint_id, text)


def build_cache(backend: str, dsn: str | None = None) -> RegistryCache:
    if backend == "memory":
        return InMemoryRegistryCache()
    if backend == "redis":
        if not dsn:
            from shared.errors import ConfigError

            raise ConfigError("redis.backend=redis requires a DSN")
        import redis.asyncio as redis

        return RedisRegistryCache(redis.from_url(dsn, decode_responses=True))
    from shared.errors import ConfigError

    raise ConfigError(f"unknown redis backend {backend}")


__all__ = [
    "CACHE_PREFIX",
    "CachedRegistryStore",
    "InMemoryRegistryCache",
    "RedisRegistryCache",
    "RegistryCache",
    "SCCategory",
    "build_cache",
]
