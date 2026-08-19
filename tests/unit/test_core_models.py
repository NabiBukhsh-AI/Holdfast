"""TASK-002 acceptance tests. Spec 6.1."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from compint.core.models import History, Message, Role


def test_user_turn_indices_excludes_tool_and_thinking(hermes_history: History) -> None:
    """U^t counts USER messages only, not the tool and thinking traffic between them."""
    # Table 1 shape: roughly 120 messages against roughly 9 user turns.
    assert hermes_history.n_messages == 118
    assert hermes_history.n_user_turns == 9
    indices = hermes_history.user_turn_indices
    assert len(indices) == 9
    for index in indices:
        assert hermes_history.messages[index].role is Role.USER


def test_user_turn_index_space_differs_from_message_index_space(hermes_history: History) -> None:
    """The distinction is load bearing: injecting over message indices is a different experiment."""
    assert hermes_history.user_turn_indices[1] != 1
    assert hermes_history.user_turn_indices != tuple(range(hermes_history.n_user_turns))


def test_openresearcher_single_user_turn(openresearcher_history: History) -> None:
    """Table 1 reports exactly 1.00 user turns for OpenResearcher."""
    assert openresearcher_history.n_user_turns == 1
    assert openresearcher_history.is_degenerate_for_injection is True


def test_wildchat_history_is_not_degenerate(wildchat_history: History) -> None:
    assert wildchat_history.n_user_turns == 5
    assert wildchat_history.is_degenerate_for_injection is False


def test_history_is_not_mutable(wildchat_history: History) -> None:
    """INV-1: H^t is never mutated in place anywhere in the system."""
    with pytest.raises(ValidationError):
        wildchat_history.messages = ()  # type: ignore[misc]
    with pytest.raises(ValidationError):
        wildchat_history.messages[0].content = "rewritten"  # type: ignore[misc]


def test_history_rejects_non_increasing_indices() -> None:
    messages = (
        Message(index=3, role=Role.USER, content="a", token_count=1),
        Message(index=1, role=Role.USER, content="b", token_count=1),
    )
    with pytest.raises(ValidationError, match="strictly increasing"):
        History(messages=messages)


def test_token_count_is_the_sum_of_cached_counts(wildchat_history: History) -> None:
    """Equation 3 measures token length, not message count."""
    assert wildchat_history.token_count == sum(m.token_count for m in wildchat_history.messages)


def test_render_labels_tool_messages(hermes_history: History) -> None:
    rendered = hermes_history.render()
    assert "[TOOL(search)]" in rendered
    assert "[SYSTEM]" in rendered


def test_content_hash_is_stable_and_content_sensitive(wildchat_history: History) -> None:
    first = wildchat_history.content_hash()
    assert first == wildchat_history.content_hash()
    changed = History(
        messages=tuple(
            m.model_copy(update={"content": m.content + "!"}) if m.index == 0 else m
            for m in wildchat_history.messages
        )
    )
    assert changed.content_hash() != first
