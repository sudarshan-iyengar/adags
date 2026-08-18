# M-2 result — the split, and why the named repair is vacuous (2026-08-18)

Executed against the frozen design
[[elgs-m2-oncomponent-split-design]], which fixed the population,
classification, denominators and decision rule before any outcome
existed. Diagnostic only: no C1–C6 class, census total, coverage figure
or eligibility verdict is changed by anything here.

**Status: BOTH reductions complete. They AGREE EXACTLY — see §8.**
The independent reducer sealed its numbers before experiment 151
returned, so independence held.

## 1. Contract checks — all three exact

The design made these hard void conditions rather than numbers to
reconcile, precisely so a substituted artifact would be caught:

| check | required | computed | delta |
|---|---:|---:|---:|
| true-absence windows | 597 | **597** | 0 |
| low-visibility report-pairs | 1,086,839 | **1,086,839** | 0 |
| total report-pairs over W | 1,240,623 | **1,240,623** | 0 |

Per-sequence window counts match Appendix A exactly. writing_2 was taken
from `…_fix79ae5b7` with the `m1c3fix_census_…` census, and its
recomputed coverage of 0.9243 confirms the corrected substrate — the
writing_2 trap did not fire. Prereg R6 guard: 0 failures in 597 records.
Poker's 171,053 full-window pairs and 79 seeds independently reproduce
figures disclosed in the parent prereg.

**Coverage: all 12 sequences, complete.** No sampling. Cost was far below
the estimate — masks total ~330 MB, tracks ~3.1 GB, 29,901 mask decodes,
about three minutes of CPU.

## 2. The split

| class | count | share |
|---|---:|---:|
| `ON_ELIGIBLE` | 931,410 | **85.70%** |
| `OFF_ELIGIBLE` | 0 | **0.00%** |
| `BACKGROUND` | 155,429 | 14.30% |
| `UNIDENTIFIABLE` | 0 | 0.00% |

`p_on` under every admissible pooling:

| pooling | `p_on` |
|---|---:|
| pooled over reports | **0.8570** |
| unweighted mean over sequences with ≥10 windows (the parent prereg's binding pooling) | **0.8224** |
| leave-scissor-out | 0.9183 |
| unweighted mean over all 12 | 0.8490 |

All four land in the same row and none is within 0.02 of a boundary, so
the conservative tie-break never engages. Sequence range 0.4946
(slice_apple) to 1.0000 (soda).

### Verdict, per the rule frozen before the numbers

> **`p_on >= 0.70` → drift is NOT dominant.**

Pooling-invariant. The 87.60% low-visibility limb is overwhelmingly the
tracker being *right but unconfident*, not the tracker drifting off the
object.

## 3. THE FINDING — the named repair is vacuous

**`v` is a binary flag, not a continuous confidence score.**

Every one of the 1,086,839 low-visibility reports has `v` **exactly
0.0**. Tabulated over six entire tracks artifacts (10.8M reports) by the
independent reducer, `v` takes **exactly two values, ever: 0.0 and 1.0**.

**Verified separately by the primary agent** on `maracas`
(`…/maracas_screen_w0_134_tracks/tracks.json`, pulled and tabulated
locally): 1,608,795 reports, 20,347 `is_miss`, and among the rest
`v ∈ {0.0, 1.0}` only — 734,545 zeros and 853,903 ones, **two distinct
values, no intermediates**. Two independent tabulations, same conclusion.

### What follows, and it is severe

1. **§5's candidate correction (i), "a visibility-threshold change", is
   VACUOUS.** Lowering `0.5` to `0.4`, `0.3`, `0.2` or `0.1` rescues
   **zero** reports. Every threshold in `(0, 1]` partitions the data
   identically. The only "threshold change" that admits anything is
   accepting `v == 0` — which is abolishing the gate, not moving it.
2. **§4.4's `v`-histogram is degenerate by construction.** It could only
   ever have placed 100% in the first bin, and its stated purpose —
   "eligibility under any candidate threshold recomputable offline" — is
   unachievable in principle, not merely unmet.
3. **The scissor/poker eligibility recomputation under correction (i) is
   identically the status quo**: 0.441 and 0.382, both still under the
   0.5 floor. **Correction (i) cannot admit either sequence.** Scissor's
   exclusion is therefore NOT rescued by the correction its exclusion was
   suspected to be an artifact of.
4. **`OFF_ELIGIBLE` is an empty category**, and this was checked rather
   than assumed: ineligible components genuinely exist (632 in pour_tea,
   138 in slice_apple, areas down to 1 px) and the reducer classifies
   against them; no low-visibility pixel ever lands on one. The design's
   hoped-for gradation — "a repair might plausibly rescue the first and
   never the second" — has nothing in the rescuable bucket.

### The tension this creates, stated rather than resolved

The frozen rule selects the row whose consequence column reads "a
census-level threshold correction is admissible". **The same measurement
shows that correction is unexecutable.**

This page does **not** override the frozen rule. `p_on` is what it is and
the verdict stands: drift is not dominant. What is reported is that the
*consequence* attached to that row was written on an assumption about `v`
that the data falsifies.

The repair the evidence actually supports — gate on **component
membership** rather than on `v` — is what the design assigns to its
**middle** row's process: a NEW instrument, needing its own
preregistration and its own adversarial round. It remains census-level
and needs no re-tracking, so the cheap branch survives; but it is a new
instrument, not a constant change, and it must not be slipped in under
the selected row's wording.

**This is a user decision**, and it replaces decision 1 of
[[user-decision-memo-2026-08-18]] with a sharper question: not "census
correction or re-tracking" — re-tracking is not indicated — but "is a
component-membership instrument authorised, with a fresh prereg and
adversarial round?"

## 4. Ambiguities in the frozen text, and how they were resolved

The one that could have changed the answer: **the definition of W**. The
prereg's `ASSOCIATED` requires an eligible component, but its own poker
disclosure ("a *visible* report in some camera of S", 346 frames) invites
a looser visible-only reading. Both were implemented and the choice was
resolved on evidence, not preference: the eligible-component reading
gives 0 R6 guard failures and reproduces 1,240,623 exactly; visible-only
gives 10 guard failures across 5 sequences. The prereg's 346 is a
different, looser statistic — no contradiction.

Immaterial ambiguities, each verified immaterial rather than argued away:
which pooling §5 means (all four agree); rounded vs continuous pixel in
the bounds test (`OUT_OF_DOMAIN` is 0); the `v == 0.5` boundary (never
occurs); track rows with frame gaps (0 occurrences). One gap in the
frozen text: label 0 forced ineligible is implied but never stated.

## 5. Not computed, and why

`anchor_agreement` (§4.5) and the D2 merge/split context (§4.6). Both
carry no decision weight by design. §4.5 additionally needs a
camera-convention axis flip the frozen text never specifies, and emitting
a number that depends on an unstated convention would be worse than
emitting none.

## 6. What this does NOT establish

* **Physical presence.** `ON_ELIGIBLE` means the report landed on an
  eligible foreground component, not that the object was there. The
  frozen M1-A0b audit sample (73 windows) remains emitted and unrun; no
  physical-absence claim is permitted before it returns.
* **That the tracker is correct.** A drifted report can land on an
  eligible component by coincidence.
* **Agreement between reducers.** The primary reducer has not returned.
* **Anything about the occlusion supply**, which stands at 239,545 and is
  untouched here.
* **G-OA's FAIL**, not reopened.

## 7. Next action

Collect experiment 151 and compare against §1–§2. Agreement on three
exact contract checks and a pooling-invariant `p_on` would make this
durable. **Disagreement is the more informative outcome and must be
reported, not reconciled.**


## 8. CROSS-CHECK — the two reductions agree to the digit

Experiment 151 (primary reducer, `dgx`, commit `7465d82`, image
`70a28e3d…`) COMPLETED. Artifact pulled from
`runs/elgs/m2_oncomponent_split_r2/diagnostic.json` and compared against
the independent reduction, which had sealed first.

| quantity | independent | primary | agree |
|---|---:|---:|:---:|
| `ON_ELIGIBLE` | 931,410 | 931,410 | ✔ |
| `OFF_ELIGIBLE` | 0 | 0 | ✔ |
| `BACKGROUND` | 155,429 | 155,429 | ✔ |
| `UNIDENTIFIABLE` | 0 | 0 | ✔ |
| `p_on` | 0.8570 | 0.8569898577434192 | ✔ |
| low-visibility pairs | 1,086,839 | 1,086,839 | ✔ |
| total report-pairs | 1,240,623 | 1,240,623 | ✔ |
| absence windows | 597 | 597 | ✔ |

Two reducers, written independently from the same frozen text, over the
same sealed artifacts, with no shared code — the independent one was
forbidden from reading `build_absence_diagnostic.py` or any
`diagnostic.json`, and sealed before the primary returned. **Every figure
matches.** The verdict in §2 is now durable.

### Third confirmation of the binary-`v` finding

The primary's own `v_histogram` places **all 1,086,839 reports in bin
`[0.0, 0.1)` and exactly zero in `[0.1,0.2)`, `[0.2,0.3)`, `[0.3,0.4)`
and `[0.4,0.5)`.** That is an independent third confirmation, from the
instrument's own machinery, of §3: the histogram the design commissioned
to make thresholds recomputable is degenerate, and no threshold in
`(0, 1]` moves a single report.

### Facts the primary adds that the independent reduction did not report

* **`p_on` is identical at ALL THREE eligibility levels** — 16 px, 64 px
  and 256 px all give `0.8569898577434192` over the same 1,086,839
  reports. The split is therefore **invariant to the component area
  floor**, which removes `component_min_px` as a candidate explanation
  for the result. This is a robustness fact worth more than it looks:
  had `p_on` moved with the floor, the finding would have been an
  artifact of a frozen constant.
* **`restricted_to_C1a_windows` contains ZERO windows** —
  `absence_windows_contributing = 0`, `p_on_primary = None`, with
  `differs_from_unrestricted = True` correctly raised. This is not a
  defect: C1a is the corroborated-true-absence class, and the parent
  diagnostic's headline is that **0 of 597** windows are corroborated. The
  empty restriction reproduces that finding from a different direction.
* Denominators: **363 distinct seed_ids, 26 distinct camera_ids.**
* `anchor_agreement.share = 0.5412` over all 1,086,839 reports
  (`reports_without_an_anchor_label = 0`). **DESCRIPTIVE ONLY** — it
  carries no decision weight by design, and is recorded, not used. The
  independent reducer declined to compute it rather than guess an
  unstated camera-convention axis flip; the primary computes it inside
  the instrument's own convention, so the two are not comparable and no
  agreement is claimed for this row.
* D2 context, unchanged and not an input: 4,210,877 merge events, 22,191
  split events.

### What the cross-check does NOT establish

Agreement between two reducers over the same sealed artifacts confirms
the REDUCTION, not the inputs. Both read the same tracks and the same
masks. If those artifacts were wrong, both would be wrong together —
which is precisely how the 2026-08-13 substrate defect survived an
"exact" recomputation. The three contract checks (597 / 1,086,839 /
1,240,623) are the guard against that, and they passed; the writing_2
trap in Appendix A was the specific substitution they were designed to
catch, and it did not fire.
