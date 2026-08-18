# M-2 — the on/off-component split of low-visibility reports (design)

**FROZEN 2026-08-18, before any outcome was inspected.** Diagnostic only.
No output of this measurement is gate-bearing: it changes no C1–C6 class,
no census total, no eligibility verdict. It exists to decide ONE thing —
whether the absence instrument can be repaired at census level or needs
re-tracking.

Supersedes nothing. Reads [[elgs-absence-diagnostic-result]],
[[elgs-absence-diagnostic-design]], [[elgs-exhaustive-screen-scope]],
[[elgs-cycle2-screening-record]], [[elgs-substrate-remeasurement-result]].

## 1. Why this measurement, and why it is the next step

[[elgs-absence-diagnostic-result]] returned verdict **status_2 (material
defect)** on the DiVa-360 absence instrument: of 597 true-absence windows
across 12 sequences, **zero** are corroborated as genuine full-multiview
disappearance. The dominant limb is `LOW_VISIBILITY` — **1,086,839 of
1,240,623 (camera, frame) report-pairs, 87.60%** — where the tracker
returned a position but with `v < 0.5` and the instrument refused it.

That page then recorded a correction against its own earlier prose:

> `build_absence_diagnostic.report_label` tests `LOW_VISIBILITY` BEFORE
> the on-component test, so a v < 0.5 report is stamped LOW_VISIBILITY
> whether it sits on the identity's component or far away on a wall;
> "in-domain" means only "inside the raster". The 87.60% bucket is
> therefore an **unmeasured mixture** of correct-but-unconfident and
> grossly-drifted reports.

and [[elgs-exhaustive-screen-scope]] drew the operational consequence:

> **If drift dominates, lowering the visibility threshold makes the
> instrument WORSE and a tranche-1 re-evaluation built to that spec is
> wasted.** Therefore: measure the on/off-component split of
> low-visibility reports (one CPU pass over already-sealed artifacts)
> BEFORE specifying the corrected instrument.

This is that pass. It has never been run.

**Verified in code before freezing** (not taken from the prose):
`scripts/build_absence_diagnostic.py:1183` stores
`table[key] = (PRE_LOW_VISIBILITY, -1, -1)` — discarding the `col`/`row`
it computed two lines earlier at `:1175-1176` — and the classification
loop at `:1899-1911` computes `report_label` only when
`precursor == PRE_NEEDS_MASK`. So the component test genuinely never runs
for a low-visibility report. The pixel is recoverable because the tracks
artifact stores `x`/`y` per report regardless of `v`
(`scripts/build_elgs_tracks.py:696-706`).

## 2. Frozen population

**Sequences (12).** Every tranche-1 sequence with a nonzero corrected
true-absence count, per the frozen table in
[[elgs-cycle2-screening-record]] with writing_2 taken at its CORRECTED
value from [[elgs-substrate-remeasurement-result]]:

| sequence | corrected true-absence windows |
|---|---:|
| maracas | 1 |
| tambourine | 18 |
| pour_tea | 73 |
| pan | 11 |
| soda | 1 |
| tea | 13 |
| put_candy | 18 |
| put_fruit | 4 |
| writing_2 | 2 |
| poker | 109 |
| slice_apple | 4 |
| scissor | 343 |
| **total** | **597** |

This list is DERIVED arithmetically from the frozen screening table, not
quoted from a page that lists it. The sum reproducing 597 exactly is the
check that the derivation is right; if the reducer's own window count
disagrees with 597, that is a **contract failure and the run is void**,
not a number to reconcile afterwards.

**Reports.** Every `(seed_id, camera_id, frame)` pair the frozen
instrument stamped `ST_LOW_VISIBILITY` inside those windows. Expected
denominator 1,086,839 of 1,240,623. A disagreement greater than 0 in
either denominator voids the run for the same reason.

**Artifacts.** The CORRECTED tranche-1 substrate only. writing_2 must use
the post-`79ae5b7` conversion; the defective pre-correction artifacts of
[[elgs-substrate-defect-2026-08-13]] are excluded by name, not by
timestamp. Tracks come from
`/apollo/users/sri/proj_adags/data/diva360_derived/<seq>_screen_w0_<N>_tracks/tracks.json`,
censuses from
`/apollo/users/sri/proj_adags/runs/elgs/m1c2_screen_<seq>_r0/census.json`
(all 20 verified present by `rclone lsd`, 2026-08-18).

**Instrument constants.** `configs/elgs/prereg_m1_absence_diagnostic_v1.json`
revision 5, unchanged: `visibility_threshold 0.5`, `component_min_px 64`,
`mask_binarize_threshold 127`. This measurement does not alter them; it
reports what would happen if the first were changed.

## 3. Classification rule

For each low-visibility report, take the pixel `(col, row)` that
`:1175-1176` already computes by `round_half_up` of the tracker's
`x`/`y`, and the connected-component label image
`mask_frame.labels` for that `(camera, frame)` — the SAME `MaskFrame` the
instrument already builds, from the same `load_component_labels`, at the
same eligibility level `li`. No new component machinery is introduced and
none may be: the split must be measured through the instrument's own
correspondence, not a second one.

| class | rule |
|---|---|
| `ON_ELIGIBLE` | `label > 0` and `lookup[li][label]` is true |
| `OFF_ELIGIBLE` | `label > 0` and `lookup[li][label]` is false |
| `BACKGROUND` | `label == 0` |
| `UNIDENTIFIABLE` | the frame's mask is unavailable or undecodable |

Bounds are already enforced upstream: `:1178-1182` stamps
`PRE_OUT_OF_DOMAIN` before the visibility test, so a low-visibility report
is inside the raster by construction. If any nevertheless fails the bounds
check, that is a contract failure, not a fifth class.

**`ON_ELIGIBLE` is the only class that means "the tracker was right but
unconfident".** `OFF_ELIGIBLE` and `BACKGROUND` both mean the report is
not on the thing it claims to track; they are reported separately because
`OFF_ELIGIBLE` (landed on a too-small or otherwise ineligible foreground
component) is a weaker form of drift than `BACKGROUND` (landed on the
scene), and a repair might plausibly rescue the first and never the
second.

**Identity correspondence is NOT manufactured.** The class is defined
against the eligibility lookup the instrument already uses. This
measurement does not ask "is the report on the SAME component the anchor
is on" via any new association — that would use the tracker under
evaluation to certify itself. The anchor-relative question is reported
separately and descriptively in §4's `anchor_agreement` column, computed
from `mask_frame.labels[anchor_row, anchor_col]` which the instrument
already evaluates at `:1865` and `:1895`, and it carries no decision
weight.

**Lineage, merge and split are NOT used.** The D2 mask-component lineage
exists in the reducer, but chaining a low-visibility report through an
IoU lineage would make the answer depend on lineage parameters that were
frozen for a different purpose. The split is per-`(camera, frame)` and
memoryless. Merge/split rates are reported as CONTEXT in §4 from the
existing D2 accumulators, and are not inputs to the decision rule.

## 4. Required outputs

Per sequence and pooled:

1. the four-way class split of low-visibility reports, as counts and as
   shares of the low-visibility total;
2. denominators alongside every share: report-pairs, distinct identities
   (`seed_id`), distinct cameras, and absence windows;
3. the same split restricted to reports inside windows the instrument
   classified as true-absence, if that differs from (1);
4. a joint histogram of `v` against class, on the fixed bin edges
   `[0.0, 0.1, 0.2, 0.3, 0.4, 0.5)`, so eligibility under any candidate
   threshold is recomputable offline without a second pass;
5. `anchor_agreement`: share of low-visibility reports whose label equals
   the anchor's label (descriptive, no decision weight);
6. existing D2 merge and split rates, unchanged, as context;
7. `UNIDENTIFIABLE` count, which must be 0 for a clean run.

## 5. Decision rule — fixed now, before the numbers

Let `p_on` = `ON_ELIGIBLE` / (all low-visibility reports), pooled over the
12 sequences, at the primary eligibility level.

| reading | verdict | consequence |
|---|---|---|
| `p_on >= 0.70` | **drift is NOT dominant** | a census-level threshold correction is admissible. Re-evaluation of tranche 1 costs ~1 CPU-hour and ZERO GPU-hours; no re-tracking. |
| `0.30 < p_on < 0.70` | **mixed** | a threshold change alone is NOT admissible. The corrected instrument must gate on component membership as well as on `v`, which is still census-level but is a NEW instrument and needs its own preregistration and its own adversarial round. |
| `p_on <= 0.30` | **drift dominates** | lowering the threshold makes the instrument WORSE. The census-level branch is REFUTED. Re-tracking, or a different presence instrument, is required — and that is a user decision, not an automatic next step. |

Sequence-level readings are reported but do not override the pooled
verdict; a single sequence may not carry the decision. If the pooled
reading falls within 0.02 of a boundary, the verdict is the **more
conservative** of the two adjacent rows.

**Scissor and poker eligibility** are recomputed under the two candidate
corrections already named in [[elgs-absence-diagnostic-result]] — (i) a
visibility-threshold change, (ii) the never-queried limb — using the §4.4
histogram. Both sequences failed cycle-2 on the coverage floor
(scissor 0.441, poker 0.382, floor 0.5) while scissor cleared the union
return floor (75 >= 12) and poker did not (10 < 12). Because coverage and
the absence limb SHARE the `v >= 0.5` constant, a threshold change moves
coverage too, and the recomputation must move both together or it is
meaningless. **Recomputed eligibility is reported as a CANDIDATE reading
only.** It re-opens no gate. Admitting any sequence requires a fresh
preregistration under a corrected instrument, per
[[elgs-absence-diagnostic-result]].

## 6. What this measurement cannot establish

* **Physical presence.** As with the parent diagnostic, nothing here sees
  the scene. `ON_ELIGIBLE` means "the report landed on an eligible
  foreground component", not "the object was there". The frozen M1-A0b
  audit sample (73 windows) remains emitted and unrun, and no
  physical-absence claim is permitted before it returns.
* **That the tracker is correct.** A drifted report can land on an
  eligible component by coincidence, and a correct report can sit on a
  component the eligibility floor excludes. This measurement bounds the
  repair options; it does not validate the tracker.
* **G-OA's FAIL.** Not reopened.
* **The occlusion supply claim.** Occlusion requires association in >= 2
  cameras and is barely coupled to this constant; per
  [[elgs-absence-diagnostic-result]] it STANDS and is untouched here.

## 7. Execution and independence

CPU-only reduction over sealed artifacts. No tracking is run. No training
is run. Per `AGENTS.md`, a Determined cell occupies one GPU slot even when
CPU-bound, and the slot-hours are reported as such rather than described
as free.

**Independent recomputation is required before the result is durable.** A
fresh-context reducer works from THIS frozen text and the primary
artifacts only, never reading the primary reducer's code or its outputs,
and seals its own numbers before either is disclosed to it. That is the
same discipline that caught the substrate defect and that turned three
would-be positives into honest negatives; it is not optional here because
this measurement is the one that chooses the repair branch.

## 8. Termination

The measurement ends when §4's tables exist for all 12 sequences and
pooled, §5's verdict follows mechanically from `p_on`, and the
independent recomputation either agrees or its disagreement is explained.
No further screening, re-tracking, or tranche-2 work is authorised by this
page under any outcome.
