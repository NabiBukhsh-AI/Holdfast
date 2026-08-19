# DEVIATIONS.md

Every departure from `PAPER SPECIFICATION`, what was specified, what was implemented, why,
and the expected effect on results. Execution contract rule 10.

An empty section means no deviation has been taken in that area. Sections are added as the
implementation proceeds, never removed.

---

## D-01 SC 3 probe query is a concrete sentence, not a description

- **Specified:** Appendix Table 12 lists SC 3's probe query as the description
  "Asks the agent to email someone" rather than a literal user utterance. Every other row in
  the table carries a literal query.
- **Implemented:** `data/sc_catalog/v1.yaml` SC 3 uses
  `"Send an email to the team letting them know the release slipped to Friday."`
- **Why:** The probe harness sends the query verbatim as the next user turn (Equation 7). A
  description cannot be sent to a model. The alternative, sending the description literally,
  would test whether the model can respond to a stage direction.
- **Expected effect on results:** Confined to SC 3's compliance cells. The constructed query
  exercises the same behavior the table's compliant and violating options describe (produce a
  draft and stop, versus call the send tool).
- **How to remove this deviation:** Fetch the reference implementation's probe file and
  replace the sentence with the authors' own.

---

## D-02 Retention judge parsing records a normalized verdict alongside the strict one

- **Specified:** The Appendix C.1 judge prompt says "Output only YES or NO." A strict reading
  makes `"YES."` or `"Yes, the constraint is present."` unparseable.
- **Implemented:** Both verdicts are recorded per record: `verdict` under strict matching and
  `normalized_verdict` after stripping trailing punctuation and taking the first token. Only
  the STRICT verdict feeds the reported rate.
- **Why:** `ENGINEERING RECOMMENDATION` spec 14.9. Recording both makes the effect of parser
  leniency measurable rather than assumed. Choosing leniency silently would change the
  headline number by an unknown amount.
- **Expected effect on results:** None on the headline rate, which uses strict parsing. The
  delta between the two is reported as a diagnostic.

---

## D-03 Wilson intervals attached to every reported rate

- **Specified:** The paper reports no uncertainty on any figure.
- **Implemented:** Every rate carries a 95 percent Wilson score interval, its denominator, and
  its exclusion counts (INV-6).
- **Why:** `ENGINEERING RECOMMENDATION` spec 6.11. At N=750 a 17 percent rate carries roughly
  plus or minus 2.7 points, and several of the paper's comparative claims sit inside that
  band. Reporting intervals does not contradict the paper; it makes the reproduction honest
  about which differences are resolvable at the given sample size.
- **Expected effect on results:** No change to point estimates. Adds columns.

---

## D-04 Degenerate injection cells are marked rather than reported as four numbers

- **Specified:** The paper reports injection location results for WildChat and Hermes Agent
  and omits OpenResearcher without stating the mechanism in the results tables.
- **Implemented:** When `|U^t| == 1`, Top, Middle, Bottom, and Multi provably collapse to the
  same single location. The implementation detects this, emits a structured warning, and marks
  the cells `DEGENERATE` (FR-023).
- **Why:** `ENGINEERING RECOMMENDATION`. Reporting four apparently independent numbers that
  are byte identical by construction would misrepresent the grid.
- **Expected effect on results:** None numerically. Changes presentation from four columns to
  one column plus a marker.

---

## D-05 Content filter rejections are a first class terminal state

- **Specified:** The paper notes 15 of 2,000 Gemini samples were blocked as
  `PROHIBITED_CONTENT` and excluded, leaving 1,985.
- **Implemented:** `JUDGE_BLOCKED` and `ProbeStatus.BLOCKED` are distinct terminal states,
  counted, excluded from denominators, and printed alongside every affected metric.
- **Why:** `ENGINEERING RECOMMENDATION` spec 6.8. This is a reproducible operational hazard of
  WildChat, not an incidental error.
- **Expected effect on results:** Denominators shrink slightly on WildChat cells, and by a
  visible, reported amount rather than an invisible one.

---

## D-06 Delimited assembly mode exists alongside bare concatenation

- **Specified:** Equation 10 is bare textual concatenation, `C(H^t) (+) S^t`.
- **Implemented:** Both modes exist behind `assembly.mode`. Research configs use `bare`,
  reproducing the paper literally. Production configs use `delimited`.
- **Why:** `ENGINEERING RECOMMENDATION` spec 6.13. In production, compaction fires repeatedly.
  Without a delimiter the registry block from event n is fed into event n+1 and summarized,
  reintroducing the exact loss the registry exists to prevent.
- **Expected effect on results:** None on research runs, which use `bare`. The two modes share
  one `assemble()` so they cannot drift (INV-5).

---

## Deviations pending

None recorded yet beyond the above. Out of tolerance reproduction cells will be added here
individually once RQ1 has run against real models, per the definition of done in spec 32.3.
