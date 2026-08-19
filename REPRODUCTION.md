# REPRODUCTION.md

Clean machine to headline results. Every step states what it costs and what gate it must pass.

Nothing in this document spends money or GPU time without an explicit `--confirm` flag.

---

## 0. Prerequisites

| Requirement | Value |
| --- | --- |
| Python | 3.11 or newer |
| Disk | roughly 200 GB for corpora, contexts, and raw responses |
| GPU (offline arm) | One 80 GB card serves the 9B extractor and the embedding model. The 120B compactor needs more; see the source spec section 13.4. |
| GPU (tests) | None. The entire test suite runs on CPU against stubs. |
| Network | Required once for prompt fetching and corpus download. |

```bash
git clone https://github.com/NabiBukhsh-AI/Holdfast.git
cd Holdfast/Code
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"          # Windows
# .venv/bin/python -m pip install -e ".[dev]"            # Linux and macOS
```

For the offline arm add the research extra, which pulls FAISS, LLMLingua-2, datasets, and
Polars:

```bash
.venv/Scripts/python -m pip install -e ".[dev,research]"
```

---

## 1. BLOCKING GATE: fetch the external prompts

```bash
.venv/Scripts/python scripts/fetch_prompts.py --confirm
```

This pulls three artifacts that the source research cites but does not reprint:

| Prompt | Unknown | Destination |
| --- | --- | --- |
| Anthropic compaction prompt | U-01 | `prompts/compaction/anthropic.v1.yaml` |
| pi-mono compaction prompt | U-02 | `prompts/compaction/pi_mono.v1.yaml` |
| SC extraction prompt | U-03 | `prompts/extraction/sc_extractor.v1.yaml` |

**If this fails, stop.** Do not work around it, do not reconstruct a prompt, do not paraphrase
one from the paper's structured summary. A reconstructed compaction prompt invalidates every
comparison in the headline table. This is escalation trigger 1 in the source specification.

Verify:

```bash
.venv/Scripts/python -m pytest tests/unit/test_prompts_and_config.py -q
```

Each fetched file must carry `source_url`, `fetched_at`, and `sha256`, and the recomputed hash
must match the stored one at import.

---

## 2. Arithmetic gate (free, seconds)

Run this before anything expensive. It validates the Effect Retention implementation against
the paper's own published arithmetic.

```bash
.venv/Scripts/python -m pytest tests/unit/test_metrics.py -q
```

Three worked checks from the source specification must pass within 0.15 percentage points:

```
Hermes, gpt-oss-120b (Anthropic):  (0.585 - 0.461) / (0.983 - 0.461) = 23.8%
Hermes, LLMLingua-2 T500:          (0.501 - 0.461) / (0.997 - 0.461) =  7.5%
WildChat, gpt-oss-120b (pi-mono):  (0.512 - 0.512) / (0.951 - 0.512) =  0.0%
```

**If these fail, stop and escalate.** Either the paper's arithmetic or this implementation is
wrong, and both need a human. This is the cheapest correctness gate available and it is why it
runs before a single model is invoked.

---

## 3. Full offline test suite (free, under a minute)

```bash
.venv/Scripts/python -m pytest -q
```

Everything runs against deterministic stubs: the LLM client, the judge, the embedding model,
and the LLM compactors. Recent-N runs for real because it needs no model. The metrics run for
real because they are pure functions and are the highest value CI target.

Gates that must be green before proceeding:

- 60 golden framing strings byte exact
- Injection purity property test (1,000 injections leave `H^t` unchanged)
- Prompt hashes match the pinned fixtures
- Import isolation rule holds

---

## 4. Build the long context environments

```bash
.venv/Scripts/python scripts/build_contexts.py --dataset all --target-tokens 100000 --confirm
```

Costs one embedding pass over the corpora. Produces 50 evaluation contexts and 10 development
contexts per dataset.

Two properties are enforced rather than hoped for:

- **Determinism.** Two runs produce byte identical context sets. Seed selection is the lowest
  indexed remaining sample, never randomized.
- **No leakage.** The source pool is partitioned into `dev` and `eval` BEFORE stitching, so no
  source conversation can appear in both. Prompt iteration on `dev` therefore cannot leak into
  reported numbers.

**Gate:** the Table 1 statistics report must reproduce within tolerance, and OpenResearcher
must show exactly 1.00 user turns per context. If the statistics do not reproduce, the contexts
are wrong and every downstream number is wrong. Escalate rather than proceeding.

---

## 5. Pre-flight cost estimate

```bash
.venv/Scripts/python scripts/estimate_cost.py --config configs/research/rq1_baseline.yaml
```

Prints projected token spend and dollar cost per cell and in total. The runner refuses to
start above `cost.ceiling_usd` without `--confirm`. The source research spent roughly 800 USD
on one 220K experiment, so this gate is not theoretical.

---

## 6. RQ1 baseline

```bash
.venv/Scripts/python scripts/run_experiment.py --config configs/research/rq1_baseline.yaml --confirm
```

Grid: 50 contexts x 15 SCs = 750 instances per compaction condition per dataset, across three
datasets and six compactor configurations.

The run is resumable. A run that dies at hour 80 of 90 resumes from its checkpoint rather than
restarting, keyed on the instance idempotency hash.

---

## 7. Verify against the acceptance thresholds

```bash
.venv/Scripts/python scripts/verify_reproduction.py --run-id <run_id>
```

| Metric class | Tolerance |
| --- | --- |
| Retention cells at 0.0 to 1.0 percent | within 2 percentage points absolute |
| Retention cells above 1.0 percent | within 25 percent relative, or 5 points absolute, whichever is looser |
| Compliance rates | within 5 percentage points absolute |
| Effect Retention recomputed from own components | within 0.15 percentage points |
| Effect Retention versus the paper | within 8 percentage points |
| Extractor retention | within 5 points, and must exceed 85 percent on every dataset |
| Qualitative orderings | must match exactly |

**A reproduction is declared successful when** all qualitative orderings hold, the extractor
exceeds 85 percent on all three datasets, non LLM compactors are below 2 percent everywhere,
and no more than 15 percent of quantitative cells fall outside tolerance, with every out of
tolerance cell individually explained in [DEVIATIONS.md](DEVIATIONS.md).

Missing thresholds on more than 15 percent of cells is a finding, not a bug to grind through.

---

## 8. Remaining suites

| Config | Produces |
| --- | --- |
| `configs/research/rq2_context_length.yaml` | Retention decline across 10K, 50K, 100K |
| `configs/research/rq2_compaction_rate.yaml` | Output length invariance, roughly 182x at 100K |
| `configs/research/rq3_injection_location.yaml` | Location grids at 50K and 100K |
| `configs/research/rq3_framing.yaml` | The full 2x2 framing grid, with marginals derived from it |
| `configs/research/rq3_repetition.yaml` | Repetition sweep, r in 1 to 30 |
| `configs/research/rq3_sc_type.yaml` | Per category retention |
| `configs/research/rq4_extractor.yaml` | The mitigation arm |
| `configs/research/ablation_sc_targeted_prompt.yaml` | SC targeted prompt ablation |
| `configs/research/robustness_free_generation.yaml` | Free generation compliance with NEI bounds |
| `configs/research/validation_judge_agreement.yaml` | Judge agreement and Cohen kappa |

---

## 9. Three seed variance

The source research states no seeds, so bit identical reproduction is impossible. Run the
headline configuration at three seeds and report the spread:

```bash
for seed in 20260731 20260801 20260802; do
  .venv/Scripts/python scripts/run_experiment.py \
      --config configs/research/rq1_baseline.yaml --seed "$seed" --confirm
done
```

Without this, a reproduction cannot distinguish a real discrepancy from run to run noise.

---

## 10. What every run records

Each run emits a manifest capturing the config hash, catalog version, context set version,
model identifiers, every prompt hash, the seed, the git SHA, and **every UNKNOWN parameter with
the value that run chose for it**. Any number this system produces can be traced back to the
choices that produced it.
