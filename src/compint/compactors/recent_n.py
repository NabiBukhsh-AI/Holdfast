"""Recent-N truncation compactor. TASK-011, FR-031.

PAPER SPECIFICATION spec 14.4: "turns" here means MESSAGES. This distinction is the whole
explanation of a headline number. With N=5 on Hermes Agent (mean 119.66 messages, 9.04 user
turns) the window frequently contains ZERO user turns, which is why Recent-5 scores 0.4 percent
there while scoring 100 percent on WildChat bottom injection, where user turns are dense.

Implementing this over user turns instead of messages would make Recent-5 look far better
than the paper reports and would quietly break the reproduction.
"""

from __future__ import annotations

import time

from compint.compactors.base import CompactionResult
from compint.core.models import CompactionStatus, History, Role
from compint.core.tokenization import Tokenizer


class RecentNCompactor:
    """Retain the last N messages of H^t, rendered."""

    def __init__(self, n: int, tokenizer: Tokenizer, compactor_id: str | None = None) -> None:
        if n < 1:
            raise ValueError(f"Recent-N requires N >= 1, got {n}")
        self.n = n
        self.id = compactor_id or f"recent_{n}"
        self._tokenizer = tokenizer

    async def compact(self, history: History) -> CompactionResult:
        started = time.perf_counter()
        window = history.messages[-self.n :]
        text = History(messages=window).render() if window else ""
        status = CompactionStatus.OK if text else CompactionStatus.COMPACTION_FAILED
        return CompactionResult(
            text=text,
            compactor_id=self.id,
            model_id="none",
            input_tokens=history.token_count,
            output_tokens=self._tokenizer.count(text),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            status=status,
            raw=text,
        )

    def window_user_turn_count(self, history: History) -> int:
        """Diagnostic: how many user turns the retained window actually contains.

        Reported alongside Recent-N results, because a 0.0 percent retention cell is only
        interpretable next to the fact that the window held no user turn to retain.
        """
        return sum(1 for m in history.messages[-self.n :] if m.role is Role.USER)
