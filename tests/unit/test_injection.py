"""TASK-005 acceptance tests. Spec 6.4, 6.5, 6.6, Algorithm 14.3."""

from __future__ import annotations

import pytest

from compint.core.catalog import SCCatalog
from compint.core.framing import frame
from compint.core.injection import (
    InjectionOutOfRangeError,
    NoUserTurnError,
    inject,
    inject_into_turn,
    injection_locations,
    prepend_into_turn,
)
from compint.core.models import History, InjectionCondition, Message, Role
from compint.core.random_source import RandomSource
from shared.errors import ConfigError


def test_inject_into_turn_appends_with_separator() -> None:
    """Equation 4 read left to right: the SC follows the original user text (A-02)."""
    assert inject_into_turn("Email Sarah.", "Show me drafts first.") == (
        "Email Sarah. Show me drafts first."
    )


def test_prepend_direction_available_for_the_u08_sweep() -> None:
    assert prepend_into_turn("Email Sarah.", "Show me drafts first.") == (
        "Show me drafts first. Email Sarah."
    )


def test_top_middle_bottom_locations(rng: RandomSource) -> None:
    assert injection_locations(InjectionCondition.TOP, 9, rng) == {0}
    assert injection_locations(InjectionCondition.BOTTOM, 9, rng) == {8}


def test_middle_index_convention(rng: RandomSource) -> None:
    """A-03: floor((n-1)/2). On nine user turns Middle selects index 4."""
    assert injection_locations(InjectionCondition.MIDDLE, 9, rng) == {4}


def test_middle_takes_the_lower_of_two_central_turns(rng: RandomSource) -> None:
    """With n even, floor((n-1)/2) selects the LOWER central turn. Recorded in the manifest."""
    assert injection_locations(InjectionCondition.MIDDLE, 10, rng) == {4}


def test_multi_no_replacement(rng: RandomSource) -> None:
    locations = injection_locations(InjectionCondition.MULTI, 9, rng, r=5)
    assert len(locations) == 5
    assert locations <= set(range(9))


def test_multi_clamps_k_to_available_user_turns(rng: RandomSource) -> None:
    """k = min(r, |U^t|). Spec 6.6."""
    assert len(injection_locations(InjectionCondition.MULTI, 3, rng, r=10)) == 3


def test_multi_requires_r_at_least_two(rng: RandomSource) -> None:
    """UNKNOWN U-09: r is not stated for the main grid, so it can never be defaulted."""
    with pytest.raises(ConfigError, match="U-09"):
        injection_locations(InjectionCondition.MULTI, 9, rng, r=None)
    with pytest.raises(ConfigError):
        injection_locations(InjectionCondition.MULTI, 9, rng, r=1)


def test_locations_raise_on_zero_user_turns(rng: RandomSource) -> None:
    with pytest.raises(NoUserTurnError):
        injection_locations(InjectionCondition.TOP, 0, rng)


def test_inject_is_pure(hermes_history: History, catalog: SCCatalog, rng: RandomSource) -> None:
    """Property: H^t is unchanged after many injections. It is reused across the whole grid."""
    before_hash = hermes_history.content_hash()
    before_tokens = hermes_history.token_count
    for i in range(1000):
        sc = catalog.by_id((i % 15) + 1)
        condition = list(InjectionCondition)[i % 4]
        locations = injection_locations(
            condition, hermes_history.n_user_turns, rng.derive(f"i{i}"), r=3
        )
        result = inject(hermes_history, frame(sc), locations, condition=condition)
        assert result.history is not hermes_history
    assert hermes_history.content_hash() == before_hash
    assert hermes_history.token_count == before_tokens


def test_injection_targets_user_turn_index_space(
    hermes_history: History, catalog: SCCatalog
) -> None:
    """FR-022: locations index U^t. Location 1 must land on the SECOND user message."""
    framed = frame(catalog.by_id(2))
    injected = inject(hermes_history, framed, {1})
    second_user_message_index = hermes_history.user_turn_indices[1]
    touched = [
        m.index for m in injected.history.messages if framed.rendered_text in m.content
    ]
    assert touched == [second_user_message_index]


def test_injection_never_touches_a_non_user_message(
    hermes_history: History, catalog: SCCatalog
) -> None:
    framed = frame(catalog.by_id(4))
    injected = inject(hermes_history, framed, {0, 2, 4})
    for message in injected.history.messages:
        if framed.rendered_text in message.content:
            assert message.role is Role.USER


def test_inject_raises_on_out_of_range_location(
    wildchat_history: History, catalog: SCCatalog
) -> None:
    with pytest.raises(InjectionOutOfRangeError):
        inject(wildchat_history, frame(catalog.by_id(1)), {99})


def test_inject_raises_when_there_is_no_user_turn(catalog: SCCatalog) -> None:
    history = History(
        messages=(Message(index=0, role=Role.SYSTEM, content="system only", token_count=2),)
    )
    with pytest.raises(NoUserTurnError):
        inject(history, frame(catalog.by_id(1)), {0})


def test_degenerate_single_user_turn(
    openresearcher_history: History, catalog: SCCatalog, rng: RandomSource
) -> None:
    """FR-023: on |U^t| == 1 all four conditions collapse and the cell is marked DEGENERATE."""
    framed = frame(catalog.by_id(7))
    produced = set()
    for condition in InjectionCondition:
        locations = injection_locations(condition, 1, rng, r=5)
        assert locations == {0}
        injected = inject(openresearcher_history, framed, locations, condition=condition)
        assert injected.degenerate is True
        produced.add(injected.history.content_hash())
    assert len(produced) == 1, "all four conditions must produce the identical history"


def test_inject_raises_when_result_exceeds_the_compactor_window(
    wildchat_history: History, catalog: SCCatalog
) -> None:
    """Algorithm 14.3 failure condition: injection increases total token length."""
    with pytest.raises(ConfigError, match="exceeds the compactor window"):
        inject(
            wildchat_history,
            frame(catalog.by_id(1)),
            {0},
            compactor_max_tokens=wildchat_history.token_count,
        )


def test_injected_history_records_provenance(
    hermes_history: History, catalog: SCCatalog
) -> None:
    injected = inject(
        hermes_history,
        frame(catalog.by_id(3)),
        {0, 4},
        condition=InjectionCondition.MULTI,
        repetition_r=2,
    )
    assert injected.locations == (0, 4)
    assert injected.condition is InjectionCondition.MULTI
    assert injected.repetition_r == 2
    assert injected.separator == " "
