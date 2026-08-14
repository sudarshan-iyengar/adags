# EL-GS True-Absence Measurement Diagnostic — Design (frozen)

Status: DESIGN FROZEN at `configs/elgs/prereg_m1_absence_diagnostic_v1.json`
**revision 3**. Authority: user directive 2026-08-14 (measurement closure
before further DiVa-360 screening). This page is the durable narrative
record; the prereg JSON is the operative frozen text wherever the two
differ.

Revision history: r1 initial freeze; r2 owner pre-data multi-view
tightening (§5); **r3 repairs of a fresh-context hostile review that
returned REJECTED with nine blocking findings** (§5b). All repairs were
made pre-data: no classification statistic had been computed and no mask
had been decoded at any non-associated (camera, frame) pair.

**This diagnostic changes no prior verdict.** The cycle-1 negative
([[operations/elgs-m1-census-result]]), the cycle-2 DRY
([[operations/elgs-cycle2-screening-record]]), the cycle-3 G-OA valid FAIL
([[operations/elgs-cycle3-gate-result]]) and the corrected-substrate G-R
FAIL ([[operations/elgs-substrate-remeasurement-result]]) all remain FINAL
and are never overwritten. No output of this diagnostic is gate-bearing.

## 1. The question

The frozen census TEXT defines true absence as the disappearance of the
relevant foreground component from every applicable camera
(`prereg_m1_census_v1.json` `eligibility_predicate.true_absence_candidate.
mask_disappearance`). The IMPLEMENTATION
(`scripts/build_m1_census.py::true_absence_and_returns`, via
`build_association`) operationally scores absence when NO camera of the
containing set has an in-domain visible report landing in an eligible
component.

Those coincide when tracking works. They can diverge when it fails: a
component that is still visibly present but no longer tracked satisfies the
implementation's predicate while violating the text's. If that divergence
is material it would simultaneously reduce measured coverage, manufacture
apparent true absences, and concentrate apparent event supply in
low-coverage sequences — which is the shape the data actually shows.

This is a HYPOTHESIS under test, not an established defect.

## 2. Verified state this design rests on (owner-recomputed from primary artifacts)

- Repository HEAD `8e19abc`, equal to `origin/apollo/csvl-vpl-v2-exploratory`.
  The two user-owned files (`research-wiki/deep-dive-prompt.txt`,
  `research-wiki/run-deep-dive.ps1`) are untracked and untouched.
- **The screening table reproduces exactly** from the 20 sealed screening
  censuses under `runs/elgs/m1c2_screen_*_r0/census.json`. Pooled with the
  corrected-substrate rows substituted for writing_2 and xylophone:
  **occlusion 239,545; true-absence 597; coverage ratio-of-sums 0.821201**.
  The durable record's "~240k occlusion opportunities / ~700 true-absence
  candidates" is confirmed as an as-measured (defective-era) figure of
  237,821 / 679; the CORRECTED figure is 239,545 / 597.
- **The reported concentration reproduces exactly.** poker 109 + scissor
  343 + put_candy 18 + pour_tea 73 = **543**; the other sixteen sequences
  = **54**. Those four are indeed the four lowest-coverage sequences
  (0.3824 / 0.4409 / 0.5067 / 0.5908) and they hold 90.95% of all
  true-absence candidates while holding 22.19% of eligible components and
  16.89% of frames.
- **The grouping is post-hoc** and is therefore recorded as a disclosure
  only, never as a decision input. Continuous, rate-normalised analysis
  over all 20 rows (owner-computed): coverage vs absence-per-eligible-
  component pearson r = **-0.765**, spearman rho = **-0.695**; coverage vs
  `return_position_undefined`-per-component r = **-0.724**.
- **The decisive asymmetry**: coverage vs occlusion-per-eligible-component
  is only r = **-0.178**, rho = **-0.429**. The occlusion predicate requires
  the identity to be ASSOCIATED in >= 2 cameras — i.e. tracking WORKING —
  and it is nearly uncorrelated with coverage. The absence predicate
  requires association to FAIL everywhere, and it is strongly
  anticorrelated. A "these are simply occlusion-rich scenes" explanation
  predicts the two should co-vary; they do not. This motivated the
  diagnostic. **It is a correlation, not a mechanism, and is not a decision
  input.**
- **Instrument comparability CONFIRMED.** `scripts/build_m1_census.py`
  changed between the screening commit `1d8f3b0` and the corrected-substrate
  commit `79ae5b7` by exactly one purely ADDITIVE hunk (`by_half` coverage
  tallies, commit `644251f`). No association, no component tally, no gated
  statistic is touched. `scripts/build_elgs_tracks.py` is unchanged since
  `49b5e5e`, which precedes every screening run. All rows come from one
  statistic-bearing code path.
- **Substrate blast radius INDEPENDENTLY REPRODUCED** from the 30 sealed
  `diva360_conversion_provenance.json` files: exactly three conversions
  selected `segmented_ngp.tar.gz` (`writing_2_full_w0_480`,
  `writing_2_screen_w0_239`, `xylophone_screen_w0_306`); all other 27 chose
  `frames_1.tar.gz`. Matches [[operations/elgs-substrate-defect-2026-08-13]]
  exactly. Owner preflight additionally DECODED one mask per in-scope
  conversion (16 conversions): all are 1160x550, matching the declared
  calibration. Zero mismatches.
- **Structural facts that make the hypothesis concrete.** poker: 109 absence
  windows over 79 distinct seeds, while 309 seeds carry occlusion events;
  |S| = 26 (every training camera) for 93/109 windows; 226 bridged
  interruptions; 44 `return_position_undefined`; median window 50 frames.
  DiVa-360 runs at 120 fps, so poker's screened half is 268 frames =
  **2.23 seconds**. The recorded reading is therefore that 79 distinct
  seeds each vanished from all 26 cameras of a surround rig, for a median
  0.42 s, inside 2.2 s of a card-table scene. scissor is the same shape at
  343 windows over 250 seeds. This is the same physical-plausibility
  argument [[operations/elgs-substrate-remeasurement-result]] used against
  writing_2's defective 84 absences, applied to the two sequences that now
  dominate the surviving supply. **Implausibility is a motive for
  measurement, not a measurement.**

## 3. What the diagnostic measures

Four instruments, three of them tracker-free:

- **D1 anchor occupancy** (primary; tracker-free given the sealed anchor).
  Project the record's frozen `ltp` into each camera of the frozen
  containing set and ask whether an eligible foreground component covers it
  (strict pixel, `T1`) or lies within the projected r_site ball
  (`T2`, tolerance `r_site * fl_x / z` — the identical construction the
  frozen R2-prime predicate uses). Consumes only calibration, masks and the
  sealed consensus point.
- **D2 deterministic component lineage** (corroborative; mask-geometric).
  From the last frame the identity WAS associated, follow that mask
  component forward by maximum-IoU chaining. Entry is a tracker SUCCESS;
  propagation consumes no tracker output. Never changes an assigned class.
- **D3 report-status cross-tabulation** (tracker-side; says WHY).
  `NO_REPORT` / `OUT_OF_DOMAIN` / `LOW_VISIBILITY` / `OFF_COMPONENT`.
- **D4 substrate integrity**: decoded vs declared dimensions, every camera.

## 4. The central identity problem, and where the honest boundary is

D1/D2 establish that FOREGROUND covers the frozen anchor. They do NOT
establish that the covering component IS the identity whose absence was
scored. The design refuses to make that slide, and the class semantics say
so explicitly.

Two things make the diagnostic nevertheless decisive for the measurement
question:

1. The frozen predicate's own content is "the component vanishes from all
   applicable cameras". If foreground demonstrably covers the anchor, that
   predicate is not established **whatever the identity is**. This is a
   statement about the instrument's self-consistency and is
   identity-independent.
2. **Multi-view consistency** (revision-2 tightening, below) distinguishes
   "a surface is physically at the anchor" from "something incidentally
   overlaps that image position in one view".

Correspondence is never manufactured by re-running, re-querying or
re-seeding the tracker whose failure is under test. This diagnostic
performs no new tracking.

## 5. Revision-2 owner amendment (pre-data, one-sided tightening)

While preparing the implementation the owner found a defect in revision 1:
**the anchor is by construction STALE.** If an object genuinely leaves, the
anchor is where it WAS, and any later foreground at that site — the
manipulating hand, the table, another object — would satisfy a
single-camera occupancy test and be scored as instrument failure. Revision 1
would have over-attributed genuine absences to tracker failure.

Repair: C2/C3 now require **multi-view** anchor occupancy — at least
`min_occupancy_cameras` (= 2) cameras of S simultaneously satisfying strict
occupancy, with at least one qualifying pair separated by at least the
sequence's frozen angular floor. A physical surface at the anchor projects
to the anchor pixel in every camera of S; a spurious occluder at a
different 3-D location does not. Windows failing this fall to
C5_NON_IDENTIFIABLE.

This tightening makes the "instrument is confounded" classes **strictly
harder** to reach — it moves AGAINST the hypothesis under test. It was made
before any classification statistic existed and before the primary
implementation existed. Sensitivity reading S7 retains
`min_occupancy_cameras = 1` precisely so the effect of this tightening on
the final status is measured and published rather than assumed.

## 5b. Revision-3 hostile-review repairs (pre-data)

A fresh-context hostile review of the frozen design returned **REJECTED
(pre-data, repairable)** with nine blocking findings and seven notes. The
review was substantially correct and its repairs are applied in full. The
three that matter most:

- **B2 — the design as frozen could not execute, and I verified it against
  primary data.** `build_m1_census.true_absence_and_returns` splices
  bridged flicker frames into the scored window
  (`window.extend(reappearance_run)`), and those frames carry association
  in a camera of S **by construction**. The r1/r2 fail-closed rule on
  ASSOCIATED would therefore have aborted the run. Owner verification on
  poker: **85 of 109 windows contain such frames — exactly the 85 windows
  with `bridged_interruptions > 0`** (346 of 6,627 window-frames, 5.22%).
  Repaired by defining the absent-frame set `W(record)` and tallying
  bridged frames instead of aborting on them.
- **B3 — C1, the class carrying the "instrument adequate" verdict, had
  three unbounded false-positive routes.** (i) The anchor is triangulated
  from reports that `build_elgs_tracks.build_artifact` admits with **no
  mask-eligibility test at all**, so under tracker drift P can be the
  triangulation of the very reports whose failure defines the window, and
  `ltp_frame` may equal `first_frame`; (ii) a visual-hull phantom seed has
  no surface anywhere and is guaranteed C1; (iii) foreground below the
  64 px eligibility floor is present but scored absent. Repaired by
  splitting C1 into **C1a / C1b / C1c**, adding a sub-threshold occupancy
  test `T2b`, and requiring per-window anchor-quality outputs. Only C1a
  counts toward adequacy. The independence claim for D1 is correspondingly
  weakened and the limit is now stated in the prereg.
- **B4 — C2, the class carrying the "material defect" verdict, was
  inflated by cameras that could never have associated.** I had already
  found this independently (11.17% of poker's in-window pairs are
  NEVER_QUERIED; 102/109 windows contain at least one such camera), but the
  reviewer's additional repair is the load-bearing one: the winning camera
  `c*` must be drawn only from cameras that were actually queried, else a
  structurally-silent camera can force C2 on its own.

Also repaired: non-exhaustive status rules with two discretion clauses
pre-routing the answer to Status 3 (**B5** — added `status_5_UNRESOLVED`,
deleted every discretion clause, froze the sequence group to the named
minimal prefix `{scissor}`); three of six sensitivity axes provably unable
to move the verdict (**B6** — the decision-relevant grid is now exactly
S2 × S3 × S5 × S7 = 108 combinations, with S1/S4/S6 declared
decision-irrelevant *by construction*); pooled fractions being
arithmetically scissor's fractions (**B7** — three-way pooling with the
sequence-unweighted mean binding); status names asserting more than the
instrument reaches (**B8** — renamed to the occupancy question, with the
mapping to the user-facing four statuses fixed *before* execution, and the
frozen M1-A0b audit sample now a mandatory output); and an incomplete
disclosure (**B9** — `tol_c` and, more importantly, **the author's prior on
the answer** are now on the record).

Two review points are accepted as **standing limits, not repairs**: the r2
multi-view tightening filters incidental image-space overlap but does NOT
address the manipulating hand physically occupying the vacated site; and
exact agreement between the primary implementation and the independent
reducer establishes only that the frozen text is mechanically executable,
never that it measures the intended quantity — B2 is the proof, since both
would have aborted identically.

The reviewer independently re-verified the instrument-comparability claim
and the disclosed arithmetic of §2; both hold.

## 6. Classification and pre-committed decision rules

Six classes under a total, deterministic ordered decision list (prereg
`classification.ordered_decision_list`): C1 genuine-absence-corroborated,
C2 track/report loss, C3 visibility/association failure, C4
substrate/projection, C5 non-identifiable, C6 other (asserted empty; a
non-empty C6 fails the run closed).

Four measurement-closure statuses with **pre-committed** thresholds
(prereg `measurement_closure_decision_rules`), fixed before any outcome
existed and evaluated in the order defect -> adequate -> not-identifiable ->
partially-confounded. Seven sensitivity readings (S1..S7) bracket every
residual semantic freedom; a status that flips across admissible readings
is Status 4 unless a conservative reading is defensible.

## 7. Outcome-blindness is NOT claimed

The prereg's `disclosed_prior_knowledge` block enumerates verbatim
everything the design author already knew: the full per-sequence table, the
pooled totals, the 543/54 concentration, the continuous correlations, the
substrate status, and the structural facts of §2. What IS claimed is that
every unit, predicate, threshold, classification rule and decision rule was
frozen before any classification outcome existed. No mask had been decoded
at any non-associated (camera, frame) pair at freeze time.

## 8. Cost and proportionality

CPU-only; no GPU is required (no tracking, no rendering, no training).
Twelve primary-scope sequences carry at least one window. Each needs at most
one streaming pass over already-converted masks — the same order as the
original census pass, measured at roughly 2-12 minutes CPU per sequence.
Estimate: **2-4 CPU-hours total, parallelisable to ~15-30 minutes wall;
< 200 MB of JSON added; no new download, no new conversion, no new
tracking.**

Cheaper routes were considered and rejected on the record (prereg
`execution_and_verification.cheaper_alternatives_considered`): re-analysis
of the sealed census records cannot answer the question at all, because the
census records no mask state at non-associated pairs; sampling adds error
for negligible saving; the human-audit protocol is slower, costlier, and
still needs this mask evidence (it remains available as a later confirmation
of a subsample).

## 9. Verification protocol

Hostile fresh-context review of the frozen design before execution;
independent fresh-context recomputation written from the frozen text alone,
with the reducer barred from the primary implementation, its tests, its
outputs and every result page until its own results are sealed; exact or
explained comparison; integrity audit of commit, config, image, inputs, task
identity and outputs; every disagreement treated explicitly; negative and
non-identifiable findings preserved.
