"""The injection operator. TASK-005, Equations 4 and 5, Algorithm 14.3.

    x~^i_U = x^i_U (+) s,   for all i in I                                  (4)
    H^t_{s,I} = Inj(H^t, s, I)                                              (5)

Two properties are load bearing:

1. `locations` index U^t, the USER TURN positions, never raw message indices. On Hermes Agent
   |U^t| is about 9 while |H^t| is about 120 (INFERENCE, spec 6.1, assumption A-04).
2. The operator is PURE. H^t is reused across all 15 SCs and all 4 conditions, so in place
   mutation would corrupt the entire grid (INV-1).
"""

from __future__ import annotations

import logging

from compint.core.models import (
    FramedSC,
    History,
    InjectedHistory,
    InjectionCondition,
    Role,
)
from compint.core.random_source import RandomSource
from shared.errors import ConfigError

logger = logging.getLogger(__name__)


class NoUserTurnError(ConfigError):
    """|U^t| == 0. There is no user turn to inject into."""


class InjectionOutOfRangeError(ConfigError):
    """A requested location does not exist in U^t."""


def inject_into_turn(user_text: str, sc_text: str, separator: str = " ") -> str:
    """Equation 4.

    UNKNOWN: spec 30.2 U-08. The paper does not state whether s is appended or prepended, nor
    the separator. `x^i_U (+) s` read left to right implies append, which also matches the
    paper's own example sentence ordering, where the SC clause follows the task clause
    (assumption A-02). The separator is configurable and recorded in the run manifest, and it
    is the first variable to sweep if reproduction numbers diverge materially.
    """
    return f"{user_text.rstrip()}{separator}{sc_text.lstrip()}"


def prepend_into_turn(user_text: str, sc_text: str, separator: str = " ") -> str:
    """Equation 4 under the prepend reading, available for the U-08 sensitivity sweep."""
    return f"{sc_text.rstrip()}{separator}{user_text.lstrip()}"


def injection_locations(
    condition: InjectionCondition,
    n_user_turns: int,
    rng: RandomSource,
    r: int | None = None,
) -> set[int]:
    """Return indices into U^t for one condition. Spec 6.6.

    Top     I = {0}
    Middle  I = {floor(t/2)}, with t the LAST user turn index, so floor((n-1)/2)
    Bottom  I = {t}
    Multi   k = min(r, |U^t|) indices drawn uniformly WITHOUT replacement, r >= 2
    """
    if n_user_turns < 1:
        raise NoUserTurnError("cannot inject into a history with no user turns")
    match condition:
        case InjectionCondition.TOP:
            return {0}
        case InjectionCondition.MIDDLE:
            # INTERPRETATION NOTE spec 6.6, assumption A-03: with n user turns indexed 0..n-1
            # the last index is n-1, so the median index is floor((n-1)/2). With n even this
            # selects the LOWER of the two central turns. An off by one here shifts every
            # Middle condition result, so the convention is recorded in the manifest.
            return {(n_user_turns - 1) // 2}
        case InjectionCondition.BOTTOM:
            return {n_user_turns - 1}
        case InjectionCondition.MULTI:
            if r is None or r < 2:
                raise ConfigError(
                    "Multi requires a target repetition count r >= 2 (spec 6.6). "
                    "UNKNOWN U-09: the paper does not state r for the main grid."
                )
            k = min(r, n_user_turns)
            return set(rng.sample(range(n_user_turns), k))
    raise ConfigError(f"unhandled injection condition {condition}")


def inject(
    history: History,
    framed_sc: FramedSC,
    locations: set[int],
    *,
    condition: InjectionCondition = InjectionCondition.TOP,
    separator: str = " ",
    direction: str = "append",
    repetition_r: int | None = None,
    compactor_max_tokens: int | None = None,
) -> InjectedHistory:
    """Equation 5. Pure. Returns a NEW History, never mutates the input.

    `locations` are indices into U^t (user turn positions), not raw message indices.
    """
    user_indices = history.user_turn_indices
    if not user_indices:
        raise NoUserTurnError("cannot inject into a history with no user turns")
    valid = set(range(len(user_indices)))
    if not locations <= valid:
        raise InjectionOutOfRangeError(
            f"locations {sorted(locations)} out of range for |U^t|={len(user_indices)}"
        )

    degenerate = history.is_degenerate_for_injection
    if degenerate:
        # FR-023: emit a structured warning. Downstream cells are marked DEGENERATE rather
        # than reported as four apparently independent numbers.
        logger.warning(
            "degenerate_injection",
            extra={
                "n_user_turns": history.n_user_turns,
                "condition": condition.value,
                "sc_id": framed_sc.sc_id,
                "detail": "all four injection conditions collapse when |U^t| == 1",
            },
        )

    target_message_indices = {user_indices[i] for i in sorted(locations)}
    combine = inject_into_turn if direction == "append" else prepend_into_turn
    new_messages = tuple(
        m.model_copy(
            update={
                "content": combine(m.content, framed_sc.rendered_text, separator),
                # The cached count must move with the content it describes, or every
                # downstream length assertion silently measures the pre injection history.
                "token_count": m.token_count + _sc_token_delta(framed_sc, separator),
            }
        )
        if m.index in target_message_indices and m.role is Role.USER
        else m.model_copy()
        for m in history.messages
    )
    injected = History(messages=new_messages)

    if compactor_max_tokens is not None and injected.token_count > compactor_max_tokens:
        # Algorithm 14.3 failure condition: injection increases total token length and on a
        # context already at 1.25 * l_t this can push past the compactor window.
        raise ConfigError(
            f"post injection length {injected.token_count} exceeds the compactor window "
            f"{compactor_max_tokens} (spec 14.3)"
        )

    return InjectedHistory(
        history=injected,
        framed_sc=framed_sc,
        condition=condition,
        locations=tuple(sorted(locations)),
        repetition_r=repetition_r,
        degenerate=degenerate,
        separator=separator,
    )


def _sc_token_delta(framed_sc: FramedSC, separator: str) -> int:
    """Approximate token cost of the injected text, added to the cached message count.

    The exact count depends on the configured tokenizer (U-07). Callers that need an exact
    post injection length recount with that tokenizer; this keeps the cached field
    monotonically correct rather than stale.
    """
    return max(1, len(framed_sc.rendered_text + separator) // 4)
