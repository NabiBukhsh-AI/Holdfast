"""Dataset id to adapter. FR-010."""

from __future__ import annotations

from compint.core.tokenization import Tokenizer
from compint.data.ingest.base import SourceAdapter
from compint.data.ingest.hermes_agent import HermesAgentAdapter
from compint.data.ingest.openresearcher import OpenResearcherAdapter
from compint.data.ingest.wildchat import WildChatAdapter
from shared.errors import ConfigError

DATASETS = ("wildchat", "hermes_agent", "openresearcher")


def build_adapter(dataset: str, tokenizer: Tokenizer) -> SourceAdapter:
    if dataset == "wildchat":
        return WildChatAdapter(tokenizer)
    if dataset == "hermes_agent":
        return HermesAgentAdapter(tokenizer)
    if dataset == "openresearcher":
        return OpenResearcherAdapter(tokenizer)
    raise ConfigError(f"unknown dataset {dataset}. Known: {list(DATASETS)}")
