# Compaction prompts are FETCHED, never written

This directory is intentionally empty of prompt YAML files in version control.

`PAPER SPECIFICATION` and execution contract rule 4 (spec section 32.1): the Anthropic
compaction prompt, the pi-mono compaction prompt, and the full SC extraction prompt are
cited by the paper but not reprinted. They must be fetched from their cited sources or from
the reference repository. Writing, paraphrasing, or reconstructing them produces numbers
that look like Table 2 results and are not.

Run:

    python scripts/fetch_prompts.py --confirm

Expected files after a successful fetch:

| File | Prompt | Unknown id |
| --- | --- | --- |
| `anthropic.v1.yaml` | Anthropic default compaction prompt | U-01 |
| `pi_mono.v1.yaml` | pi-mono compaction prompt | U-02 |
| `anthropic_sc_targeted.v1.yaml` | Anthropic prompt plus the paper's SC-targeted addendum | U-01 |
| `../extraction/sc_extractor.v1.yaml` | Full SC extraction prompt | U-03 |

Until the fetch succeeds, `shared.prompts.PromptRegistry.get()` raises `PromptNotFetchedError`
for these ids and every LLM compactor and the extractor refuse to run. That is the designed
behavior: the blocking gate in spec section 32.2 is a gate, not a warning.

The SC-targeted addendum sentence IS printed verbatim by the paper (spec section 11.4) and is
stored in `sc_targeted_addendum.v1.yaml`. It is an addendum only. It composes onto the fetched
Anthropic base prompt and is useless on its own.
