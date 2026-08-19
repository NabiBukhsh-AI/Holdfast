# OPEN_QUESTIONS.md

Every `UNKNOWN / REQUIRES VALIDATION` in the engineering specification, the value this
implementation uses, and how to resolve it properly.

Execution contract rule 3: if something is not specified in the spec and not in the paper, it
is an UNKNOWN. It is exposed as a config value with no default, it fails loudly if unset, and
it is recorded here. Rule 11: assumptions are labelled `A-nn` and recorded in the run
manifest's `unknowns` field.

Every value below is emitted into the run manifest by `AppConfig.unknowns()`, so any number
this system produces can be traced back to the choices that produced it.

## Status legend

| Status | Meaning |
| --- | --- |
| BLOCKING | The reproduction cannot produce valid headline numbers until this is resolved. |
| RESOLVED-BY-CONFIG | A value is set in config, recorded in the manifest, and swept where it matters. |
| UNRESOLVED | No safe value exists. The accessor raises `UnresolvedUnknownError`. |

---

## Blocking unknowns

### U-01 Full text of the Anthropic compaction prompt

- **Status:** BLOCKING. `prompts/compaction/anthropic.v1.yaml` is absent.
- **Impact:** Invalidates Table 2. Every LLM compactor cell using this prompt is unproducible.
- **Chosen value:** None. `PromptRegistry.get("anthropic")` raises `PromptNotFetchedError`.
- **Resolution path:** `python scripts/fetch_prompts.py --confirm`, pulling from the Anthropic
  platform documentation URL cited by the paper (last accessed 20 May 2026) or from the
  reference repository `https://github.com/ZhiqiEliWang/compaction-integrity`.
- **Why it is not worked around:** Execution contract rule 4 and escalation trigger 1. A
  reconstructed compaction prompt produces numbers that look like results and are not.

### U-02 Full text of the pi-mono compaction prompt

- **Status:** BLOCKING. Same mechanism as U-01.
- **Impact:** Invalidates the pi-mono column of Table 2, including the best open source
  result (36.3 percent) and the worst commercial result (6.7 percent).
- **Resolution path:** Fetch from the pi-mono link cited by the paper or the reference repo.

### U-03 Full extraction prompt including few-shot examples

- **Status:** BLOCKING for the extractor claim (Table 4, the mitigation result).
- **Impact:** The 90.3 to 95.6 percent retention claim cannot be reproduced without it.
- **Chosen value:** None. The paper's structured summary of the prompt (SC definition,
  persistence criterion, exclusion rules, three input structure, JSON output contract) is
  encoded as machine checkable requirements in `src/compint/extractor/prompt_builder.py`,
  which VALIDATES a fetched prompt against them. That is a check, not a substitute.
- **Resolution path:** Fetch from the released code of the paper.

---

## Unknowns resolved by config

### U-04 Random seeds

- **Status:** RESOLVED-BY-CONFIG. `random.seed: 20260731`.
- **Consequence:** Reproduction will not be bit identical to the paper.
- **Resolution path:** Run three seeds and report the spread (experiment E-03). The headline
  configuration must carry three seed variance in the final report.

### U-05 Meaning of "default hyperparameter"

- **Status:** RESOLVED-BY-CONFIG. `compaction.temperature: 0.0`, `compaction.top_p: 1.0`.
- **Impact:** High. Determines whether retention is deterministic at all.
- **Resolution path:** Temperature 0 for the headline numbers, plus a sensitivity check at the
  provider defaults. Report both.

### U-06 Conversation to string serialization for embedding

- **Status:** RESOLVED-BY-CONFIG. `context.serialization: role_prefixed_newline`.
- **Impact:** High. A different serialization changes neighbor ranking and therefore changes
  every stitched context.
- **Resolution path:** Sensitivity check on neighbor overlap across two serializations.

### U-07 Tokenizer defining "100K tokens"

- **Status:** RESOLVED-BY-CONFIG. `tokenization.tokenizer_id: cl100k_base`.
- **Impact:** Medium to high. Confounds cross compactor comparison if it varies.
- **Guard:** `compint.core.tokenization.assert_reportable()` refuses the `heuristic` backend
  for a reported run. The heuristic backend exists only so CI runs without model downloads.
- **Resolution path:** Pin one tokenizer per comparison set and record it in the manifest.

### U-08 Injection separator and direction

- **Status:** RESOLVED-BY-CONFIG. `injection.direction: append`, `injection.separator: " "`.
- **Basis:** Assumption A-02, the left to right reading of `x^i_U (+) s`, consistent with the
  paper's own example sentence where the SC clause follows the task clause.
- **Resolution path:** If reproduction numbers diverge materially, this is the FIRST variable
  to sweep. `prepend_into_turn()` is implemented for exactly that sweep.

### U-10 Value of alpha_l

- **Status:** RESOLVED-BY-CONFIG. `compaction.alpha_l: 0.8`.
- **Basis:** INFERENCE from the paper's "approximately 80 percent of a 128K window" framing
  (assumption A-01). The paper never states the constant.
- **Impact:** Low. Dataset construction pins the effective length independently.

### U-11 MCQ option ordering

- **Status:** RESOLVED-BY-CONFIG. Research `fixed`, production `randomized`. Mapping always
  recorded (`probe.record_mapping: true`).
- **Impact:** Medium. If the delta is large, the paper's absolute compliance numbers carry a
  position bias component.
- **Resolution path:** Run both in the reproduction phase and report the delta (E-08).

### U-12 Judge temperature

- **Status:** RESOLVED-BY-CONFIG. `judge.temperature: 0.0`.

### U-14 Cropping granularity in stitching

- **Status:** RESOLVED-BY-CONFIG. `context.crop_granularity: message`.
- **Basis:** The paper says "crop the last sample" without stating message or token boundary.
  Message boundary is chosen because a mid message crop can truncate a user turn and silently
  change `|U^t|`.

### U-15 Composition template when strict and direct framings combine

- **Status:** RESOLVED-BY-CONFIG. `framing.template_version: v1`, pinned by 60 golden strings.
- **Basis:** Figure 2 shows the strict prefix first, then the session scope phrase, with the
  body lowercased after the colon. The paper's literal prefix strings end with a colon and a
  period respectively but the figure merges them with a comma. The template table in
  `src/compint/core/framing.py` IS that decision, made once and versioned.

### U-17 Max output tokens for compactors

- **Status:** RESOLVED-BY-CONFIG. `compaction.max_output_tokens: 2048`.
- **Basis:** Above the 1,024 floor the spec requires, so truncation cannot confound results.
- **Guard:** `CompactionStatus.TRUNCATED` is a distinct terminal state, so truncation is
  visible rather than silently shortening a summary.

---

## Unresolved unknowns

These have NO safe default. The accessor raises `UnresolvedUnknownError` naming the id.

### U-09 Target repetition count `r` for the Multi condition in the main grid

- **Status:** UNRESOLVED. `injection.repetition_r: null` in `configs/base.yaml`.
- **Impact:** Medium. Affects the Multi condition only.
- **Behavior:** `injection_locations(MULTI, ...)` raises unless the experiment config sets it.
  `configs/research/rq3_repetition.yaml` sets `r: 5` explicitly for that sweep alone.
- **Resolution path:** Ask the authors, or report the full sweep instead of a single r.

### U-13 Upstream dataset and model revisions

- **Status:** UNRESOLVED. Every `revision` in `configs/environment.lock.yaml` is `null`.
- **Impact:** Medium. Stitching is deterministic given an embedding model revision and NOT
  reproducible across revisions.
- **Behavior:** `context.require_embedding_revision()` raises.
- **Resolution path:** Pin every model by commit hash before any reported run.

### U-16 Whether the extraction prompt was developed on the same 50 contexts used for reporting

- **Status:** UNRESOLVED and UNRESOLVABLE from the paper.
- **Impact:** HIGH. Determines whether the extractor result is contaminated by prompt
  development on the test set.
- **Mitigation implemented:** A `dev` (10 contexts) and `eval` (50 contexts) split is created
  and the source pool is partitioned BEFORE stitching, so no source conversation can appear in
  both. `dev` results are never reportable. This does not resolve the question about the
  paper's own numbers; it prevents the same contamination here.

### U-18 tau_dup, the semantic duplicate threshold

- **Status:** UNRESOLVED. `registry.tau_dup: null` in base and prod configs.
- **Impact:** Medium. Governs registry deduplication quality in production.
- **Behavior:** `registry.require_tau_dup()` raises. `configs/production/dev.yaml` sets 0.88 as
  a development value only, explicitly marked as needing a sweep before staging.
- **Resolution path:** Tune on a labelled duplicate pair set and report the ROC.

### U-19 Free generation judge prompt wording

- **Status:** RESOLVED-BY-CONFIG with a caveat. The LABEL SET (compliant, non compliant, NEI)
  is `PAPER SPECIFICATION`; the prompt wording is not printed by the paper and is ours.
- **File:** `prompts/judging/free_generation_judge.v1.yaml`, provenance
  `engineering_recommendation`.

### U-20 Separator between C(H^t) and S^t in Equation 10

- **Status:** RESOLVED-BY-CONFIG. `shared.assembly.DEFAULT_JOIN = "\n\n"`.
- **Basis:** The paper writes `(+)` as textual concatenation without specifying the join. Both
  `bare` and `delimited` modes use the same join so the two differ only in markup.

### U-21 MCQ probe template wording

- **Status:** RESOLVED-BY-CONFIG. `prompts/probing/mcq_template.v1.yaml`, provenance
  `engineering_recommendation`. Figure 3's question sentence ("Which behavior is correct given
  your instructions?") is reproduced in the spec and used verbatim; the surrounding rendering
  is ours.

---

## Assumptions carried into the implementation

| Id | Assumption | Where it lives |
| --- | --- | --- |
| A-01 | `alpha_l = 0.8` | `configs/base.yaml` compaction.alpha_l |
| A-02 | Injection appends with a single space | `injection.inject_into_turn()` |
| A-03 | Middle is `floor((n-1)/2)`, the LOWER of two central turns when n is even | `injection.injection_locations()` |
| A-04 | `U^t` indexes user turns, not raw messages | `History.user_turn_indices` |
| A-05 | Retention is judged on the compacted context despite Equation 6's notation | `CompactedContext` type, INV-4 |
| A-09 | Registry budget of 200 tokens | `configs/base.yaml` registry.budget_tokens |
| A-10 | Severity ordering Action > Information > Process > Preference > Output | `data/taxonomy/v1.yaml` severity_order |
| A-11 | Drain timeout of 200 ms | `configs/production/*.yaml` service.drain_timeout_ms |

---

## Experiments the paper does not answer

Recorded here because they change what this system is worth, not merely how it is tuned.

| Id | Experiment | Priority |
| --- | --- | --- |
| E-01 | Base rate of SCs in real conversations | Highest |
| E-02 | Extractor PRECISION (the paper reports recall only) | Highest |
| E-03 | Run to run variance across seeds | High |
| E-04 | Registry budget tuning against task continuity cost | High |
| E-05 | Prompt injection success rate | High |
| E-09 | Retention under REPEATED compaction events | High |
