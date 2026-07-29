# Phase 0 Blinded Visual Forensic Audit — Result (workstream A)

Date: 2026-07-30
Branch: `csvl-vpl-v2-phase0`
Extraction job: `50819631`, COMPLETED `0:0`, 26:31 (dev scene only; no R009
selection, no locked/stress scenes, no cam00)
Package: `$WORK/proj_adags/runs/phase9-depth-visibility-capacity/phase0-audit-v1/`
(36 blinded contact sheets under `sheets/`, synchronized RGB/depth clips under
`clips/`, per-case series under `cases/`, provenance key under `provenance/`)
Reviewer: one fresh-context AI reviewer (Opus), given only `sheets/*.png`,
blinded to category and evidence source; classified all 36 cases before
provenance was revealed. Calibration: all three controlled fixtures were
classified correctly (genuine -> A, static parallax -> B, flicker -> C).
Status: engineering evidence. This audit changes interpretation; it does not
change any recorded census verdict. [[operations/phase0-census-result]] and
[[operations/phase0-census2-result]] stand as written.

## Blinded classification vs provenance (full cross-tabulation)

Taxonomy: A genuine dynamic disocclusion; B static parallax/burial; C margin
flicker; D evidence-time incoherence; E identity/boundary artifact; F camera
edge; G genuine occlusion without certified reveal; H unclear.

| True category (hidden) | n | Reviewer classes |
|---|---|---|
| real-depth certified events | 8 | **A x8** (2 flagged borderline: thin occluder / small margin) |
| shuffle-certified events | 8 | **D x8** (scrambled evidence timestamps identified in every case) |
| censored long occlusions | 6 | B x5, G x1 |
| rejected: grace budget | 3 | G x2, E x1 |
| rejected: below min duration | 3 | A x1, G x1, C x1 |
| near-threshold flicker pairs | 5 | C x5 |
| fixtures (known truth) | 3 | 3/3 correct |

## Principal findings (verified against the cross-tabulation and sheets)

**1. Real-evidence certifications are largely genuine.** All 8 sampled
real-depth certified events were independently judged genuine dynamic
disocclusions — hands, knife, head visibly crossing the marked surface point
with matched evidence timestamps (n=8, unanimous; small sample, single
reviewer). The census-v2 rule, on valid evidence, finds real events.

**2. The census-v2 shuffle control manufactures pseudo-events — the G2 ratio
was measuring control pathology, not signal absence.** All 8 shuffle-certified
cases were identified from imagery alone as evidence-time incoherence:
square-wave depth alternation across scrambled frames while the RGB scene
barely changes. Frame-shuffling does not merely remove temporal structure; for
a transition-triggered rule it *creates* abundant qualifying transitions
(120,076 shuffle events vs 11,540 valid in the audit re-run). The preregistered
rho = valid/shuffle is therefore not a valid noise bound for the v2 rule
family. The CENSUS2_NO_GO verdict stands on its preregistered terms — but its
correct reading is now: *the control was mis-matched for this rule class*,
echoing the Stage-1B lesson (controls must be matched in the statistic that
drives the decision) against this cycle's own design. A future control must be
transition-rate-matched — e.g., a per-camera circular time-shift, which keeps
evidence temporally coherent while destroying alignment with the primitives.

**3. The rule loses genuine events at scale.** 91,411 aborted runs vs 11,540
certifications in the valid pass, and the audit shows why: one rejected case
carried a visibly genuine ~13-frame occlusion scored as a 2-frame run (state
flicker mid-occlusion split the run); two grace-budget rejections were
coherent single occlusions with ragged boundaries; one censored case shows a
clean reveal whose frames were not evaluable (witness/validity gap at reveal
time). Grace budget 1 and strict evaluability are too brittle for real
occlusion statistics.

**4. Censored long runs are mostly static burial, not dynamic events.** 5 of 6
long uncertified occlusion runs are persistent viewpoint burial (primitive
permanently behind nearer structure for that camera) that escaped the
frames-0-29 baseline exclusion because burial began later. Static-parallax
exclusion needs a rolling or full-sequence baseline.

**5. Margin flicker is localized and material-specific.** All 5 near-threshold
cases sit on the window-glass region of camera cam15 — DA3 depth noise on
glass, exactly the surface class the 2026-07-15 qualitative audit flagged.
The rule correctly certifies none of them. Abstention by surface type has
concrete, localizable support.

**6. One identity/boundary artifact (E)**: a marker on a 1-3 px sliver between
sleeve and background — per-primitive granularity is too fine at silhouette
boundaries; supports region-level aggregation for census-v3.

## Audit-process flaws (recorded for any repeat)

- Near-threshold sampling collapsed to one camera/window: three sheets were
  pixel-identical (adjacent primitives on the same pane share the crop
  center). Top-K toggle selection needs a camera/window diversity constraint.
- Two certified cases were the same physical event from two cameras; the 36
  sheets cover fewer independent events than intended.
- One cosmetic rendering defect: a fixture's fourth depth panel used an
  inconsistent color normalization (flagged by the reviewer; does not affect
  any series data).
- Reviewer caveats: a single AI reviewer, not a human; fixture calibration
  3/3 but n is small throughout (8 per certified category). The human
  annotation pilot ([[operations/phase9-annotation-pilot-protocol]]) remains
  the proper instrument for claim-grade reference judgments.

## Implications recorded (no execution authorized here)

- Census-v3, when preregistered, should change: the control (circular
  time-shift, transition-rate-matched), run-merging across short state
  flickers, a rolling static-parallax baseline, region-level aggregation,
  and material-aware abstention. Expected effect: higher genuine yield and a
  control that a genuine signal can actually beat.
- Annotation sign-off decision 2 (candidate-list source): the audit
  strengthens the case for a census-assisted *supplement* — sampled real
  certifications were unanimously genuine — while the uniform core remains
  the unbiased backbone.
- The certified-event pool (11,540 pairs) is a *lower* bound on genuine
  opportunity: the rule demonstrably discards real events (finding 3).

## Links

- [[operations/phase0-census2-result]] (verdict unchanged; interpretation
  revised by finding 2)
- [[operations/phase9-annotation-contract-draft]] (sign-off table)
- [[operations/phase9-annotation-pilot-protocol]] (prepared, not run)
