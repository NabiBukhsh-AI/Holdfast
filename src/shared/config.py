"""Configuration loading with explicit UNKNOWN handling.

Spec sections 2 (evidence labels), 30.2 (unknowns), and execution contract rule 3.

Every value the paper does not supply is present here as an explicit `None`. Reading one
through its `require_*` accessor raises UnresolvedUnknownError naming the unknown id and the
resolution path. Nothing in this module invents a value.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from shared.errors import BudgetNotConfiguredError, ConfigError, UnresolvedUnknownError

Mode = Literal["research", "production"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PathsConfig(_Frozen):
    catalog: str = "data/sc_catalog/v1.yaml"
    taxonomy: str = "data/taxonomy/v1.yaml"
    prompts_dir: str = "prompts"
    artifacts_dir: str = "artifacts"


class RandomConfig(_Frozen):
    # UNKNOWN: spec 30.2 U-04. The paper states no seeds, so reproduction is not
    # bit identical. Three seeds are run and the spread reported (E-03).
    seed: int = 20260731


class TokenizationConfig(_Frozen):
    # UNKNOWN: spec 30.2 U-07. 100K tokens is not a tokenizer invariant quantity.
    tokenizer_id: str = "cl100k_base"
    backend: Literal["heuristic", "tiktoken", "huggingface"] = "heuristic"
    heuristic_chars_per_token: float = 4.0

    @field_validator("heuristic_chars_per_token")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("heuristic_chars_per_token must be positive")
        return v


class CatalogConfig(_Frozen):
    version: str = "v1"
    allow_other_category: bool = False  # FR-042: production only


class FramingConfig(_Frozen):
    template_version: str = "v1"
    # PAPER SPECIFICATION: spec 6.7, default for all main experiments.
    default_strength: Literal["preferential", "strict"] = "preferential"
    default_explicitness: Literal["contextualized", "direct"] = "direct"


class InjectionConfig(_Frozen):
    default_condition: Literal["top", "middle", "bottom", "multi"] = "top"
    # UNKNOWN: spec 30.2 U-08, assumption A-02.
    separator: str = " "
    direction: Literal["append", "prepend"] = "append"
    # UNKNOWN: spec 30.2 U-09. No default is legitimate.
    repetition_r: int | None = None
    repetition_sweep: tuple[int, ...] = (1, 5, 10, 15, 20, 25, 30)

    def require_repetition_r(self) -> int:
        """Target repetition count for the Multi condition.

        UNKNOWN: spec 30.2 U-09. The paper sweeps repetition in Appendix E but never states
        the default r used in the main grid.
        """
        if self.repetition_r is None:
            raise UnresolvedUnknownError(
                "injection.repetition_r",
                "U-09",
                "set it explicitly in the experiment config and report the value",
            )
        if self.repetition_r < 2:
            raise ConfigError(
                f"Multi injection requires r >= 2 per spec 6.6, got {self.repetition_r}"
            )
        return self.repetition_r


class CompactionConfig(_Frozen):
    # INFERENCE: spec 6.3, from the paper's 80 percent framing. UNKNOWN U-10.
    alpha_l: float = 0.8
    l_max: int = 128000
    # UNKNOWN: spec 30.2 U-17. Set above 1024 and verify no truncation.
    max_output_tokens: int = 2048
    # UNKNOWN: spec 30.2 U-05, the phrase default hyperparameter is undefined.
    temperature: float = 0.0
    top_p: float = 1.0
    recent_n: int = 5  # PAPER SPECIFICATION FR-031
    llmlingua_target_tokens: int = 500  # PAPER SPECIFICATION FR-032

    @field_validator("alpha_l")
    @classmethod
    def _alpha_range(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError(f"alpha_l must be in (0, 1], got {v}")
        return v


class ContextConfig(_Frozen):
    target_tokens: int = 100000
    n_contexts: int = 50
    knn_k: int = 32
    soft_cap_multiplier: float = 1.25
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_backend: Literal["stub", "huggingface"] = "stub"
    # UNKNOWN: spec 30.2 U-13. Pin before any reported run.
    embedding_revision: str | None = None
    # UNKNOWN: spec 30.2 U-06.
    serialization: str = "role_prefixed_newline"
    # UNKNOWN: spec 30.2 U-14.
    crop_granularity: Literal["message", "token"] = "message"
    dev_contexts: int = 10
    eval_contexts: int = 50

    def require_embedding_revision(self) -> str:
        """UNKNOWN: spec 30.2 U-13. Stitching is not reproducible across model revisions."""
        if self.embedding_revision is None:
            raise UnresolvedUnknownError(
                "context.embedding_revision",
                "U-13",
                "pin the embedding model by commit hash and record it in the manifest",
            )
        return self.embedding_revision


class JudgeConfig(_Frozen):
    model: str = "gpt-5.4"
    temperature: float = 0.0  # UNKNOWN U-12
    timeout_s: float = 120.0
    backend: Literal["stub", "openai_compatible"] = "stub"
    record_normalized_verdict: bool = True


class ProbeConfig(_Frozen):
    model: str = "gpt-oss-120b"
    temperature: float = 0.0
    timeout_s: float = 120.0
    backend: Literal["stub", "openai_compatible"] = "stub"
    # UNKNOWN: spec 30.2 U-11. Position bias is unquantified by the paper.
    option_order: Literal["fixed", "randomized"] = "fixed"
    record_mapping: bool = True


class ExtractorConfig(_Frozen):
    model: str = "qwen3.5-9b"
    thinking: bool = False  # PAPER SPECIFICATION FR-068
    guided_json: bool = False
    temperature: float = 0.0
    timeout_s: float = 30.0
    backend: Literal["stub", "openai_compatible"] = "stub"
    max_retries: int = 2


class RegistryConfig(_Frozen):
    mode: Literal["paper_flat_list", "production"] = "paper_flat_list"
    # ENGINEERING RECOMMENDATION: spec 14.7, assumption A-09.
    budget_tokens: int = 200
    # UNKNOWN: spec 14.6. Must be tuned on a labelled pair set.
    tau_dup: float | None = None
    conflict_detection: bool = False
    tombstoning: bool = False

    def require_budget_tokens(self) -> int:
        """Spec 14.7: budget set to 0 or unset fails loudly at startup, never unbounded."""
        if self.budget_tokens <= 0:
            raise BudgetNotConfiguredError(
                f"registry.budget_tokens must be positive, got {self.budget_tokens}. "
                "Spec 14.7 forbids defaulting to unbounded."
            )
        return self.budget_tokens

    def require_tau_dup(self) -> float:
        """UNKNOWN: spec 14.6. Never hardcode. Sweep and report the ROC."""
        if self.tau_dup is None:
            raise UnresolvedUnknownError(
                "registry.tau_dup",
                "U-18",
                "tune on a labelled duplicate pair set and report the ROC",
            )
        return self.tau_dup


class AssemblyConfig(_Frozen):
    # PAPER SPECIFICATION Eq 10 is bare; delimited is the production default (spec 6.13).
    mode: Literal["bare", "delimited"] = "bare"


class EvaluationConfig(_Frozen):
    min_er_denominator: float = 0.05  # ENGINEERING RECOMMENDATION spec 6.12
    wilson_confidence: float = 0.95
    er_tolerance_pp: float = 0.15


class CostConfig(_Frozen):
    ceiling_usd: float = 50.0
    require_confirm: bool = True
    price_per_1k_input_usd: dict[str, float] = Field(default_factory=dict)
    price_per_1k_output_usd: dict[str, float] = Field(default_factory=dict)


class LLMConfig(_Frozen):
    base_url: str | None = None
    api_key_env: str = "HOLDFAST_API_KEY"
    max_concurrency: int = 8
    connect_timeout_s: float = 10.0


class ServiceConfig(_Frozen):
    host: str = "127.0.0.1"
    port: int = 8080
    drain_timeout_ms: int = 200  # assumption A-11
    queue_max_depth: int = 10000
    rate_limit_per_session_rps: int = 100
    rate_limit_per_tenant_rps: int = 10000
    max_content_bytes: int = 65536
    session_ttl_days: int = 30
    shadow_mode: bool = False  # FR-086


class DatabaseConfig(_Frozen):
    dsn_env: str = "HOLDFAST_PG_DSN"
    backend: Literal["memory", "postgres"] = "memory"


class RedisConfig(_Frozen):
    dsn_env: str = "HOLDFAST_REDIS_DSN"
    backend: Literal["memory", "redis"] = "memory"


class OnlineEvalConfig(_Frozen):
    sample_rate: float = 0.01


class AppConfig(_Frozen):
    """The whole configuration surface. Frozen: nothing mutates config at runtime."""

    mode: Mode = "research"
    experiment: str | None = None
    datasets: tuple[str, ...] = ()
    compactors: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    sc_subset: tuple[int, ...] = ()
    group_by: tuple[str, ...] = ()
    sweep: dict[str, Any] = Field(default_factory=dict)
    report: dict[str, Any] = Field(default_factory=dict)
    human_annotations: str | None = None
    secondary_judge: str | None = None

    paths: PathsConfig = PathsConfig()
    random: RandomConfig = RandomConfig()
    tokenization: TokenizationConfig = TokenizationConfig()
    catalog: CatalogConfig = CatalogConfig()
    framing: FramingConfig = FramingConfig()
    injection: InjectionConfig = InjectionConfig()
    compaction: CompactionConfig = CompactionConfig()
    context: ContextConfig = ContextConfig()
    judge: JudgeConfig = JudgeConfig()
    probe: ProbeConfig = ProbeConfig()
    extractor: ExtractorConfig = ExtractorConfig()
    registry: RegistryConfig = RegistryConfig()
    assembly: AssemblyConfig = AssemblyConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    cost: CostConfig = CostConfig()
    llm: LLMConfig = LLMConfig()
    service: ServiceConfig = ServiceConfig()
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    online_eval: OnlineEvalConfig = OnlineEvalConfig()

    def config_hash(self) -> str:
        """Stable hash of the resolved configuration, stamped into every run manifest."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, default=str)
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def unknowns(self) -> dict[str, Any]:
        """Every UNKNOWN parameter and the value this run chose for it (NFR-017).

        Copied verbatim into the run manifest. A reader must be able to tell which numbers
        depend on a value the paper did not supply.
        """
        return {
            "U-04_seed": self.random.seed,
            "U-05_compactor_temperature": self.compaction.temperature,
            "U-05_compactor_top_p": self.compaction.top_p,
            "U-06_embedding_serialization": self.context.serialization,
            "U-07_tokenizer": self.tokenization.tokenizer_id,
            "U-08_injection_separator": self.injection.separator,
            "U-08_injection_direction": self.injection.direction,
            "U-09_repetition_r": self.injection.repetition_r,
            "U-10_alpha_l": self.compaction.alpha_l,
            "U-11_mcq_option_order": self.probe.option_order,
            "U-12_judge_temperature": self.judge.temperature,
            "U-13_embedding_revision": self.context.embedding_revision,
            "U-14_crop_granularity": self.context.crop_granularity,
            "U-15_framing_template_version": self.framing.template_version,
            "U-17_max_output_tokens": self.compaction.max_output_tokens,
            "U-18_tau_dup": self.registry.tau_dup,
            "A-09_registry_budget_tokens": self.registry.budget_tokens,
            "A-11_drain_timeout_ms": self.service.drain_timeout_ms,
            "assembly_mode": self.assembly.mode,
        }


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml_with_extends(path: Path, seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ConfigError(f"circular extends chain reached {resolved}")
    if not resolved.is_file():
        raise ConfigError(f"config file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping: {resolved}")
    parent_ref = raw.pop("extends", None)
    if parent_ref is None:
        return raw
    if not isinstance(parent_ref, str):
        raise ConfigError(f"extends must be a path string in {resolved}")
    parent = _load_yaml_with_extends(resolved.parent / parent_ref, seen | {resolved})
    return _deep_merge(parent, raw)


def load_config(path: str | Path) -> AppConfig:
    """Load a config file, resolve its extends chain, and validate it strictly.

    Unknown keys are rejected rather than ignored: a typo in a config key that silently
    reverted an experiment to a default would be exactly the class of silent failure this
    system exists to prevent.
    """
    merged = _load_yaml_with_extends(Path(path))
    try:
        return AppConfig.model_validate(merged)
    except Exception as exc:
        raise ConfigError(f"invalid configuration in {path}: {exc}") from exc
