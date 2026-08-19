# The extraction prompt is FETCHED, never written

`UNKNOWN / REQUIRES VALIDATION` U-03, spec sections 14.5 and 30.2. The paper gives a
structured summary of the extraction prompt (SC definition, persistence criterion, exclusion
rules, three-input structure, JSON output contract) and states that the full prompt including
few-shot examples lives in the released code. The structured summary is encoded as machine
readable requirements in `src/compint/extractor/prompt_builder.py`, which validates a fetched
prompt against them. It is not a substitute for the prompt text.

Run `python scripts/fetch_prompts.py --confirm` to populate `sc_extractor.v1.yaml`.
