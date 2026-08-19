"""Compactor id to constructor. FR-035.

PAPER SPECIFICATION spec 11.3 and Table 2: the eight configurations the reproduction must
support. Building an LLM compactor here goes through the prompt registry, so an unfetched
compaction prompt fails loudly at construction rather than producing plausible, wrong numbers
later (TASK-001 blocking gate).
"""

from __future__ import annotations

from typing import NamedTuple

from compint.compactors.base import Compactor
from compint.compactors.llm_summarizer import LLMSummarizerCompactor
from compint.compactors.llmlingua2 import LLMLingua2Compactor
from compint.compactors.recent_n import RecentNCompactor
from compint.core.tokenization import Tokenizer
from shared.config import AppConfig
from shared.errors import ConfigError
from shared.llm_client import LLMClient
from shared.prompts import PromptRegistry


class CompactorSpec(NamedTuple):
    """A declared compactor configuration from Table 2."""

    id: str
    family: str  # truncation | extractive | llm
    model_id: str
    prompt_id: str | None


# PAPER SPECIFICATION FR-035. Ids are stable and appear in every result row.
COMPACTOR_SPECS: dict[str, CompactorSpec] = {
    "recent_5": CompactorSpec("recent_5", "truncation", "none", None),
    "llmlingua2_t500": CompactorSpec(
        "llmlingua2_t500", "extractive", "microsoft/llmlingua-2-xlm-roberta-large-meetingbank", None
    ),
    "gpt_oss_120b__anthropic": CompactorSpec(
        "gpt_oss_120b__anthropic", "llm", "gpt-oss-120b", "anthropic"
    ),
    "gpt_oss_120b__pi_mono": CompactorSpec(
        "gpt_oss_120b__pi_mono", "llm", "gpt-oss-120b", "pi_mono"
    ),
    "qwen3_30b_a3b__anthropic": CompactorSpec(
        "qwen3_30b_a3b__anthropic", "llm", "qwen3-30b-a3b", "anthropic"
    ),
    "gemma_4_e4b__anthropic": CompactorSpec(
        "gemma_4_e4b__anthropic", "llm", "gemma-4-e4b", "anthropic"
    ),
    "gpt_5_4_mini__anthropic": CompactorSpec(
        "gpt_5_4_mini__anthropic", "llm", "gpt-5.4-mini", "anthropic"
    ),
    "gpt_5_4_mini__pi_mono": CompactorSpec(
        "gpt_5_4_mini__pi_mono", "llm", "gpt-5.4-mini", "pi_mono"
    ),
    # The paper's own SC targeted ablation (FR-034, TASK-038).
    "gpt_oss_120b__anthropic_sc_targeted": CompactorSpec(
        "gpt_oss_120b__anthropic_sc_targeted", "llm", "gpt-oss-120b", "anthropic_sc_targeted"
    ),
    "qwen3_30b_a3b__anthropic_sc_targeted": CompactorSpec(
        "qwen3_30b_a3b__anthropic_sc_targeted", "llm", "qwen3-30b-a3b", "anthropic_sc_targeted"
    ),
}


def build_compactor(
    compactor_id: str,
    config: AppConfig,
    tokenizer: Tokenizer,
    *,
    client: LLMClient | None = None,
    prompts: PromptRegistry | None = None,
    llmlingua_compressor: object | None = None,
) -> Compactor:
    """Construct one compactor by id. Fails loudly on a missing dependency."""
    spec = COMPACTOR_SPECS.get(compactor_id)
    if spec is None:
        raise ConfigError(
            f"unknown compactor {compactor_id}. Known: {sorted(COMPACTOR_SPECS)}"
        )
    if spec.family == "truncation":
        return RecentNCompactor(config.compaction.recent_n, tokenizer, compactor_id=spec.id)
    if spec.family == "extractive":
        return LLMLingua2Compactor(
            tokenizer,
            config.compaction.llmlingua_target_tokens,
            model_name=spec.model_id,
            compressor=llmlingua_compressor,
            compactor_id=spec.id,
        )
    if spec.family == "llm":
        if client is None:
            raise ConfigError(f"compactor {compactor_id} requires an LLM client")
        if prompts is None:
            raise ConfigError(f"compactor {compactor_id} requires a prompt registry")
        assert spec.prompt_id is not None
        # Raises PromptNotFetchedError when the blocking gate is still closed.
        prompt = prompts.get(spec.prompt_id)
        return LLMSummarizerCompactor(
            client,
            prompt,
            spec.model_id,
            tokenizer,
            temperature=config.compaction.temperature,
            top_p=config.compaction.top_p,
            max_output_tokens=config.compaction.max_output_tokens,
            compactor_id=spec.id,
        )
    raise ConfigError(f"unhandled compactor family {spec.family}")
