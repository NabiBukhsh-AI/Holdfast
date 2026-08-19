"""Error taxonomy shared by both arms.

Execution contract rule 13: fail loudly. This system exists because of a silent failure,
so every error path here is explicit and none of them degrade quietly.
"""

from __future__ import annotations


class HoldFastError(Exception):
    """Base class for every error raised by HoldFast."""


class ConfigError(HoldFastError):
    """Configuration is malformed, missing, or internally inconsistent."""


class UnresolvedUnknownError(ConfigError):
    """A value the paper does not supply was read before being resolved.

    Execution contract rule 3: if something is not specified, it is an UNKNOWN. Expose it
    as a config value with no default and fail loudly if unset. Never invent a value.
    """

    def __init__(self, key: str, unknown_id: str, resolution_path: str) -> None:
        super().__init__(
            f"config value '{key}' is unset and has no safe default. "
            f"This is {unknown_id} in OPEN_QUESTIONS.md. "
            f"Resolution path: {resolution_path}"
        )
        self.key = key
        self.unknown_id = unknown_id
        self.resolution_path = resolution_path


class PromptNotFetchedError(HoldFastError):
    """A prompt that must be fetched from an external source is not present.

    Execution contract rule 4 and TASK-001: do not write a compaction prompt. Fetch it or
    block. A reconstructed prompt produces numbers that look like results and are not.
    """

    def __init__(self, prompt_id: str, unknown_id: str, source_hint: str) -> None:
        super().__init__(
            f"prompt '{prompt_id}' has not been fetched. This is a BLOCKING GATE "
            f"({unknown_id}). Run: python scripts/fetch_prompts.py --confirm. "
            f"Source: {source_hint}. "
            f"Reconstructing this prompt is a spec violation, not a workaround."
        )
        self.prompt_id = prompt_id
        self.unknown_id = unknown_id


class PromptIntegrityError(HoldFastError):
    """A stored prompt's recomputed hash does not match its recorded hash."""


class ProviderError(HoldFastError):
    """The model provider failed in a way the caller must see."""


class ProviderTimeoutError(ProviderError):
    """The model call exceeded its explicit timeout."""


class ProviderRefusalError(ProviderError):
    """The model refused to answer, distinct from a transport failure."""


class ContentFilterError(ProviderError):
    """The provider's content filter rejected the request.

    Spec 6.8: a reproducible operational hazard of WildChat, not an incidental error.
    Counted, reported, and excluded from denominators with the count printed alongside.
    """


class ContextOverflowError(ProviderError):
    """The rendered context exceeded the model's window."""


class CostCeilingExceededError(HoldFastError):
    """A run was refused because its projected cost exceeds the configured ceiling."""


class EmptyEvaluationSetError(HoldFastError):
    """A rate was requested over zero valid records. Never silently return 0.0."""


class BudgetNotConfiguredError(ConfigError):
    """Registry token budget is zero or unset. Spec 14.7: fail at startup."""
