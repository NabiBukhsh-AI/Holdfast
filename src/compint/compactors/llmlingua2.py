"""LLMLingua-2 compactor at a FIXED 500 token budget. TASK-012, FR-032.

PAPER SPECIFICATION spec 11.3 and Appendix B.1: NOT the default probability threshold mode.
The default configuration thresholds a per token retention probability, so output length
scales with input; at 100K input that would exceed 10K output tokens, which is
incommensurable with the LLM compactors and conflicts with the practical motivation for
compaction. 500 tokens is chosen as consistent with the order of magnitude the LLM compactors
produce (Table 11: 301 to 857 tokens).

`ENGINEERING IMPLICATION` spec 3.5: LLMLingua-2's BERT style sliding window is location
invariant, so it scores near zero regardless of injection position. That is a STRUCTURAL
retention failure, not a prompt engineering deficit, and it is one of the strongest arguments
for the registry approach. Nothing in this adapter should try to fix it.
"""

from __future__ import annotations

import time
from typing import Any

from compint.compactors.base import CompactionResult
from compint.core.models import CompactionStatus, History
from compint.core.tokenization import Tokenizer
from shared.errors import ConfigError


class LLMLingua2Compactor:
    """Token level extractive compression with a hard output budget."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        target_tokens: int = 500,
        *,
        model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        compressor: Any | None = None,
        compactor_id: str | None = None,
    ) -> None:
        if target_tokens <= 0:
            raise ConfigError(f"LLMLingua-2 target budget must be positive, got {target_tokens}")
        self.target_tokens = target_tokens
        self.id = compactor_id or f"llmlingua2_t{target_tokens}"
        self.model_id = model_name
        self._tokenizer = tokenizer
        self._compressor = compressor

    def _load(self) -> Any:
        if self._compressor is not None:
            return self._compressor
        try:
            from llmlingua import PromptCompressor
        except ImportError as exc:
            raise ConfigError(
                "the llmlingua package is required for this compactor. Install the research "
                "extra, or inject a stub compressor for tests (spec 23.9 stubs it in CI)."
            ) from exc
        self._compressor = PromptCompressor(model_name=self.model_id, use_llmlingua2=True)
        return self._compressor

    async def compact(self, history: History) -> CompactionResult:
        started = time.perf_counter()
        rendered = history.render()
        compressor = self._load()
        # target_token pins the OUTPUT budget. rate is deliberately not passed: passing both
        # would reintroduce the input proportional behavior this adapter exists to avoid.
        result = compressor.compress_prompt(rendered, target_token=self.target_tokens)
        text = result["compressed_prompt"] if isinstance(result, dict) else str(result)
        status = CompactionStatus.OK if text.strip() else CompactionStatus.COMPACTION_FAILED
        return CompactionResult(
            text=text,
            compactor_id=self.id,
            model_id=self.model_id,
            input_tokens=history.token_count,
            output_tokens=self._tokenizer.count(text),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            status=status,
            raw=text,
        )
