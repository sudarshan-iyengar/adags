## Verdict

MATERIAL DEFECT — not ready to freeze; threshold authority and inference policy need explicit decisions before the textual fixes are sufficient.

Independent judgment; no ARIS skill referenced or used.

| Check | Score | Reason |
|---|---:|---|
| Faithfulness | 5/10 | Silently resolves ambiguous gate authority and adds analysis policy. |
| No substrate code change | 7/10 | Named inputs exist; denominator agrees; downstream verdict handling remains unverified. |
| Reducer preconditions | 3/10 | Changes analysis populations and inherits an incompatible run count. |
| Multi4D gap-content gate | 4/10 | Threshold authority is unresolved; descriptive wording contradicts gating. |
| Reducer completeness | 4/10 | Missing aggregation, invalid-input and incomplete-proof rules. |

## Confirmed defects

1. **Major — §14.2 silently resolves an ambiguous decision, and §14.3 contradicts it.**  
   The principal says “the preconditions that stay” are memory shares and exact zeros, but also says the Multi4D “gap-content gate counts residual or transient rows,” without restating its threshold. Both statements occur in the [plan record, lines 1192–1202](D:/adags/research-wiki/operations/ars-fast-path-plan-2026-09-23.md:1192). The amendment chooses a **0.20 scoring precondition**, then says of the reported shares, including their sum, “None of them decides scoring.” See [§14.2, line 304](D:/adags/research-wiki/operations/cer-multi4d-spec-2026-09-30.md:304) and [§14.3, lines 321–326](D:/adags/research-wiki/operations/cer-multi4d-spec-2026-09-30.md:321).  
   **Fix:** record explicitly whether the sum is a precondition and whether 0.20 survives. Otherwise make it descriptive. If retained, remove the contradictory sentence. This gate checks bank composition: it can reject a memory-gate arm whose ordinary persistent rows paint the gap, and it does not require transient contribution specifically.

2. **Major — §13.4 introduces a new analysis policy rather than merely applying rule 4.**  
   It removes **both arms** from rules 1–3. The frozen record instead specifies three layouts, benchmark counts “out of 9,” pooled do-no-harm, and scene-level failures “never dropped.” See [record, lines 56–68](D:/adags/research-wiki/assets/cer-gates-amendment-2026-10-01/review_prompt.md:56) versus [§13.4, lines 487–494](D:/adags/research-wiki/operations/cer-ex4dgs-spec-2026-09-30.md:487). Rule 2 is explicitly descriptive, despite being called inferential here. Also, a layout mean remains computable from completed scores; its **eligibility for proof**, rather than its computation, is the unresolved issue.  
   **Fix:** obtain a dated decision covering exclusions and their consequences. A simpler reading preserves all nine scene slots and the benchmark denominator, marks ineligible results DESIGN_WITHOUT_POWER, and treats incomplete required proof components as unestablished; subset estimates remain descriptive. Do not certify full-design do-no-harm using an unstated reduced population.

3. **Major — Multi4D inherits an incompatible completeness requirement.**  
   [§14’s introduction, lines 288–289](D:/adags/research-wiki/operations/cer-multi4d-spec-2026-09-30.md:288) imports §13.4, which requires **27 CER scores** ([line 495](D:/adags/research-wiki/operations/cer-ex4dgs-spec-2026-09-30.md:495)). Multi4D’s confirmatory design requires **nine**, with other event types optional ([lines 331–339](D:/adags/research-wiki/operations/cer-multi4d-spec-2026-09-30.md:331)).  
   **Fix:** give Multi4D its own explicit manifest: nine confirmatory CER runs and nine designated base runs. Optional descriptive runs must not block it. Narrow the cross-reference accordingly.

4. **Minor — §14.1 prohibits an accurate contribution description.**  
   “It is never described as residual copies painting the gap” appears at [line 294](D:/adags/research-wiki/operations/cer-multi4d-spec-2026-09-30.md:294). Yet the code creates background residual copies ([`begin`, lines 155–170](D:/adags-cer/multi4d/scene/cer.py:155)) and measures their gap contribution ([line 211](D:/adags-cer/multi4d/scene/cer.py:211)); the supplied smoke record reports a positive share.  
   **Fix:** retain the arm label, but permit measured residual contributions to be reported. Restrict causal claims, not factual contribution reporting.

## Suspected issues

- **Major — stale `pass` may escape into downstream verdicts.** Both writers still require `residual_share_gap >= 0.2` ([Ex4DGS:168–170](D:/adags-cer/ex4dgs/scene/cer.py:168), [Multi4D:216–218](D:/adags-cer/multi4d/scene/cer.py:216)). Neither local launcher branches on it; Multi4D T7 reads row counts, not that verdict. This does **not** establish scorer/worker/report safety.  
  **To settle:** inspect those consumers and test a low-residual record that passes the amended rules. Preserve the old boolean under an explicitly labelled legacy key; use one recomputed verdict everywhere.

## Unverified claims

- The referenced `D:\adags\agent-control\ars\ars_score.py` is absent. Its copy-only behavior and commit enforcement could not be checked.
- Forecast and smoke measurements are supported by supplied excerpts and narrative records, not independently inspected raw job artifacts. Multi4D’s three-bank architecture and ECCV 2026 identification are supported by the [authors’ project page](https://batfacewayne.github.io/Multi4D.io/); that source does not verify these scene-specific measurements.

## Missing

- **Scene-share aggregation:** mean of run ratios or pooled mass ratio? The JSON files do not store every denominator needed for the latter. The named run-level fields and Multi4D’s shared denominator **are present** ([lines 204–213](D:/adags-cer/multi4d/scene/cer.py:204)).
- **Input/status rules:** missing fields, non-finite values, terminal failed runs versus missing scores, and overall proof/Holm handling when an event is DESIGN_WITHOUT_POWER.
- **Conditions on complement shares:** `1 − residual − transient` is ordinary persistent share only with zero memory contribution and positive gap mass. Ex4DGS clamps a zero denominator ([line 158](D:/adags-cer/ex4dgs/scene/cer.py:158)); Multi4D divides directly. Undefined shares need explicit reporting, not an apparent ordinary share of one.