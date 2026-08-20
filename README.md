# HoldFast

**Session Constraint integrity under LLM context compaction.**

HoldFast is two deliverables sharing one codebase:

| Deliverable | Nature | Purpose |
| --- | --- | --- |
| **COMPINT** | Offline evaluation suite | Measure Session Constraint loss under context compaction across three long context environments and eight compactor configurations. |
| **SC-GUARD** | Online service | The mitigation, hardened into a deployable sidecar that maintains a session scoped constraint registry and re-injects it at every compaction event. |

COMPINT is the measurement instrument. SC-GUARD is the fix.

## The problem in one paragraph

LLM agent harnesses compact conversation history when the context window fills. Compactors are
optimized for **task continuity**: they preserve the objective, the working state, and the next
steps. Users also issue instructions that constrain **how** the task is executed rather than
**what** it is, for example "confirm with me before any action" or "never write my phone number
into a file". These are **Session Constraints (SCs)**. Because an SC is not the task, a task
centric compactor drops it. The agent then continues working while silently violating an
explicit user instruction.

## The fix in one paragraph

Not a better compaction prompt. **Architectural separation.** A small language model runs
alongside the compactor, reads only user turns, decides whether each turn declares an SC, and
maintains a running registry `S^t`. When compaction fires, the registry is concatenated onto
the compacted summary:

```
H~^t = C(H^t) (+) S^t                                                      (Equation 10)
```

The compactor is unmodified. The primary agent model is unmodified. The extractor is training
free. The registry is never compressed, because the compactor cannot see it.

## Status

364 tests, `mypy --strict` clean, `ruff` clean. The whole suite runs on CPU with no network
and no spend.

| Area | Status |
| --- | --- |
| Core domain models, taxonomy, catalog, framing, injection | Implemented and tested |
| Metrics (Equations 6, 8, 9, Wilson intervals, Cohen kappa) | Implemented, checked against the source's own arithmetic |
| Shared assembly (Equation 10, INV-5, INV-7) | Implemented and tested |
| Compactors: protocol, Recent-N, LLMLingua-2, LLM summarizer | Implemented |
| Data pipeline: ingestion, embedding, exact kNN, splits, stitching, truncation | Implemented and tested |
| Evaluation: retention judge, four compliance conditions, condition caching | Implemented and tested |
| SC extractor: three-input prompt builder, strict parser, research registry | Implemented and tested |
| Experiment runner: grid, manifest, cost gate, resumable checkpoints | Implemented and tested |
| Reporting: Table 2 equivalent and the reproduction verdict | Implemented and tested |
| SC-GUARD: append-only store, budget, audit, dedup, conflicts | Implemented and tested |
| SC-GUARD: queue, workers, assembly service, full API | Implemented and tested |
| Observability: metrics, alerts, runbooks | Implemented and tested |
| Reproduction runs against real models | Blocked on the fetch gate below |

### The fetch gate is closed, deliberately

Three prompts are cited by the source research but not reprinted by it: the Anthropic
compaction prompt, the pi-mono compaction prompt, and the full SC extraction prompt. They must
be **fetched**, never reconstructed. A reconstructed compaction prompt produces numbers that
look like results and are not.

Until `python scripts/fetch_prompts.py --confirm` succeeds:

- `PromptRegistry.get("anthropic")` raises `PromptNotFetchedError`.
- Every LLM compactor and the extractor refuse to construct.
- No Table 2 or Table 4 number is producible.

Everything that does not depend on those prompts is built and tested. See
[OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) U-01, U-02, and U-03.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"      # Windows
# .venv/bin/python -m pip install -e ".[dev]"        # Linux and macOS

.venv/Scripts/python -m pytest -q                    # full suite, no GPU, no network, no spend
```

The test suite runs entirely against deterministic stubs. There is no path by which running
tests spends money or requires a GPU.

## Repository layout

```
configs/          base.yaml plus research/ and production/ overlays; every UNKNOWN is explicit
prompts/          every prompt in the system, versioned and hashed; nothing else may hold one
data/             the 15 SC catalog and the 5 category taxonomy, immutable per version
src/compint/      OFFLINE: the benchmark. May import shared. May NOT import scguard.
src/scguard/      ONLINE: the service.   May import shared. May NOT import compint.
src/shared/       code that MUST be identical between the two arms; assemble() lives here
tests/            unit, integration, e2e, golden, property, security, load
scripts/          fetch_prompts, build_contexts, verify_reproduction, estimate_cost, build_fixtures
migrations/       the scguard schema, with append-only enforced by trigger
docs/runbooks/    one runbook per paging alert
```

The import direction rule is enforced in CI. `src/shared/` exists so that INV-5 (the evaluated
upper bound and the shipped mechanism use the same concatenation code) is guaranteed
structurally rather than by convention.

## Design invariants

These are checkable properties, each backed by a test.

| # | Invariant |
| --- | --- |
| INV-1 | `H^t` is never mutated in place anywhere in the system. |
| INV-2 | The compactor never receives `S^t` as input. |
| INV-3 | The extractor never receives assistant turns as extraction sources. |
| INV-4 | Retention is judged on compacted output, never injected input. |
| INV-5 | `K_ub` and the production assembly use the same concatenation code path. |
| INV-6 | Every rate reported carries its denominator and exclusion counts. |
| INV-7 | The registry is never itself compacted. |

## Engineering principles specific to this system

**Fail loudly.** This system exists because of a silent failure. Any error path that degrades
quietly is a bug regardless of how convenient it is. There are no bare excepts, nothing
defaults to an empty registry, and no unparseable model output is coerced into a verdict.

**Never invent a value the source did not supply.** Unknowns are config values with no
default that raise on access, each carrying its id and its resolution path. See
[OPEN_QUESTIONS.md](OPEN_QUESTIONS.md).

**Research and production are separable.** Every production behavior is toggleable, and a
research run reproduces paper semantics exactly. Deviations are recorded in
[DEVIATIONS.md](DEVIATIONS.md).

**Spend nothing by accident.** Every money or GPU spending path requires an explicit config
flag or `--confirm`.

## Documents

- [REPRODUCTION.md](REPRODUCTION.md) - clean machine to headline results
- [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) - every unknown, its chosen value, its resolution path
- [DEVIATIONS.md](DEVIATIONS.md) - every departure from the source specification

## License

MIT. See [LICENSE](LICENSE).
