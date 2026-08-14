# EL-GS True-Absence Diagnostic — RESULT: material defect (Status 2)

Date: 2026-08-14. Governing frozen protocol:
`configs/elgs/prereg_m1_absence_diagnostic_v1.json` **revision 5**
(sha256 `947cae7260fc34d4600f42461d59bac3ac555a13f69a4c1ad9c786f31310a741`),
frozen, committed and pushed at `7a18173` BEFORE the implementation was
tracked. Design and review chain:
[[operations/elgs-absence-diagnostic-design]].

**No output of this diagnostic is gate-bearing. No prior verdict is
changed. The cycle-3 G-OA valid FAIL is NOT reopened BY ANY OUTPUT OF THIS
DIAGNOSTIC**, and it remains final under the frozen failure policy.

**CORRECTED after final adversarial review — the firewall argument as
first written was too strong.** It said: the sole violation is pour_tea's
per-sequence unscreened-half COVERAGE 0.3748, computed entirely within a
verified-clean conversion, and no window classification can arithmetically
alter it. Both clauses are true and both are beside the point. Coverage is
counted only where a **v >= 0.5** report lands in an eligible component —
the identical constant this diagnostic has just characterised as a
per-point self-occlusion signal. **Coverage is therefore
instrument-dependent, and under a corrected instrument pour_tea's 0.3748
would move UPWARD, toward the 0.5 floor.** "Verified-clean conversion"
answers the SUBSTRATE question, not the INSTRUMENT question. The honest
position: G-OA's FAIL stands as a valid result of the frozen instrument on
that subset and is not reopened here; any future gate under a corrected
instrument would be a NEW gate under a NEW preregistration, never a
reopening of this one. This page previously read in the opposite direction
from [[operations/elgs-exhaustive-screen-scope]] §6, which had the coupling
right; the two now agree.

## Verdict

**status_2_ANCHOR_OCCUPANCY_INCONSISTENT** — mapping, fixed in the frozen
text before execution: *"material defect; a corrected or supplemental
instrument must be defined."*

**Robustness grade: UNANIMOUS. Agreement fraction 1.0 across all 144
decision-relevant readings.** Every contested rule from the four review
rounds is moot on this data:

- `primary_status_under_strict_unanimity` = **status_2** — the superseded
  r4 unanimity rule gives the SAME answer, so the R15 relaxation (the one
  change in this program that relaxed a decisive verdict, and the one the
  author disclosed makes the expected outcome easier) **made no difference
  to the outcome**.
- `primary_status_if_FRAGILE_dominates` = **status_2** — the residual
  FRAGILE/precedence ambiguity resolves identically either way.
- `pooling_disagreement` = **false**; poolings (a) and (b) agree.

## The measurement

597 windows, 12 sequences, all classified. Corrected run = Determined
experiment **69**, commit `4acb0fc`, digest-pinned image
`sha256:a2877f26…`, hopper, `evidence_bearing: true`; artifact
`runs/elgs/m1absdiag_primary_fix4acb0fc_r0/diagnostic.json`
(sha256 `b196b11708865a10…`).

| class | count | share |
|---|---:|---:|
| **C1a genuine absence corroborated** | **0** | **0.0%** |
| C1b sub-threshold foreground only | 0 | 0.0% |
| C1c anchor unsupported | 0 | 0.0% |
| C2 track/report loss | 0 | 0.0% |
| **C3 visibility/association failure** | **577** | **96.6%** |
| C4 substrate/projection (tautology check) | 0 | 0.0% |
| C5 no sustained anchor occupancy | 20 | 3.4% |
| C6 other (asserted empty) | 0 | 0.0% |

**C1a is zero pooled AND in every one of the twelve sequences
individually.** Not one of the 597 scored true-absence windows is
corroborated as genuine full-multiview disappearance.

Pooled (C2+C3): **0.9665** under pooling (a); **0.9169** under the BINDING
pooling (b); **0.9528** leave-scissor-out — so the finding is not an
artifact of the dominant sequence. (The 0.9528 is HAND-CORRECTED, not
instrument-produced: `c_leave_scissor_out_pooled` matched the literal name
`scissor` and so excluded nothing — see defect 2 below.)

### The full 144-cell grid, not just the primary reading

The "UNANIMOUS" grade is unanimity of the STATUS LABEL. The class counts
above are PRIMARY-READING values. Across **all 144 decision-relevant
cells**, computed from the mandatory sensitivity table:

| statistic | min | max |
|---|---:|---:|
| **C1a (genuine absence corroborated)** | **0** | **0** |
| C2 + C3 | 504 (0.8442) | 597 (1.0000) |
| C5 total | 0 (0.0000) | 93 (0.1558) |
| reading_label | status_2 in **144 of 144** cells | |

**C1a is zero in EVERY cell**, including the 0.25×-tolerance reading added
at r5 specifically to make the adequacy class easier to reach, and
including the most demanding corner (0.25×tol, S3 = 0.75, S5 = 16,
S7 = 5), which still yields C2+C3 = 504/597 = 84.4%. So "zero corroborated"
is NOT a primary-reading artifact.

**Denominators differ and must not be conflated.** 96.6% is a share of
WINDOWS. 87.60% / 12.17% are shares of (camera, frame) PAIRS across all
597 windows including the C5 ones. They are not a decomposition of one
another.

### What the absence evidence was actually made of

Report status over all 1,240,623 (camera in S, frame in W) pairs:

| status | count | share |
|---|---:|---:|
| **LOW_VISIBILITY** (tracker gave an in-domain position, flagged v < 0.5) | 1,086,839 | **87.60%** |
| **NEVER_QUERIED** (camera in S with no track row; can never associate) | 151,045 | **12.17%** |
| MISS_TOKEN | 2,606 | 0.21% |
| OFF_COMPONENT | 133 | 0.01% |
| ASSOCIATED | 0 | 0.00% |
| OUT_OF_DOMAIN | 0 | 0.00% (provably empty, N1 — confirmed empirically) |

**CORRECTED after final adversarial review — an earlier version of this
page said the tracker "merely downgraded confidence". THAT WAS NOT
MEASURED and is withdrawn.** `build_absence_diagnostic.report_label` tests
`LOW_VISIBILITY` BEFORE the on-component test, so a v < 0.5 report is
stamped LOW_VISIBILITY whether it sits on the identity's component or far
away on a wall; "in-domain" means only "inside the raster". The 87.60%
bucket is therefore an **unmeasured mixture** of correct-but-unconfident
and grossly-drifted reports. This matters directly for the repair: **if a
material share of that 87.6% is drift, lowering the visibility threshold
makes the instrument WORSE, not better.** The on/off-component split of
low-visibility reports is ONE CPU pass away and MUST be measured before a
corrected instrument is specified.

**C2 = 0** means only that MISS/OUT_OF_DOMAIN never reached a majority at
the winning camera c*; 2,606 miss tokens do exist. It does NOT mean the
tracker never lost the point anywhere.

Since the census's `track_coverage_upper_bound` uses the SAME `v >= 0.5`
threshold, coverage and absence are mechanically coupled through one
constant — part of the coverage/absence correlation recorded in the design
(r = −0.765) is an instrument identity, not a scene fact.

## Independent recomputation

A fresh-context reducer wrote its own reduction from the frozen prereg text
alone, never reading the implementation, its tests, its outputs, or any
result page. It ran all 12 sequences twice, bit-identical.

**Exact agreement on:** all six cells of the report-status cross-tabulation
(151,045 / 2,606 / 1,086,839 / 133 / 0 / 0 over 1,240,623 pairs); the R6
census-reproduction guard (597/597 windows on BOTH conjuncts, and 956
maximal bridged runs matching the census's own `bridged_interruptions`
exactly); the substrate check (312 masks + 312 frames, all 1160×550, zero
mismatches); C1a = 0 pooled and per-sequence; C2 = 0; C4 = C6 = 0;
**status_2**; and **11 of 12 sequences' full C3/C5 split**.

**One divergence, fully localized:** tambourine, one window of 597 (0.17%).
Primary C3=9/C5=9, reducer C3=8/C5=10. No window sits at the 0.5 threshold
(the marginal case is seed 445 ff 71 at m_fraction 0.5333, |M|=8, |W|=15),
so this is not threshold inclusivity but a one-frame difference in
multi-view confirmation. It moves pooling (b) from 0.9169 to 0.9090 — both
far above the 0.50 threshold. It touches neither C1a nor the status.

### A VERIFIED DEFECT the recomputation caught, and its blast radius

The FIRST run (experiment **68**, commit `0b4e374`) carried a real defect:
`run_diagnostic` built `all_windows` in CLI order while `names` was sorted
and the per-sequence aggregation walked the assignment array with a cursor
in sorted order. **40 of 597 windows carried a `class` contradicting their
own recorded `class_detail.m_fraction`.**

- The **per-sequence blocks were WRONG**.
- The **pooled counts were RIGHT** (they never use the cursor): 577/20 in
  both runs.
- **CORRECTED after final review: the pooling-(b) figure was NOT wrong.**
  An earlier version of this page and commit `4acb0fc`'s message both said
  it was. Pooling (b) is emitted from `primary_tally.b_fractions()`, and
  `Tally.per_sequence` is built by `summarize_assignment` keyed off
  `window.sequence` — **never off the cursor** — so it is order-invariant.
  Verified directly against the two sealed artifacts: exp 68 and exp 69
  both give **0.916886**, identical. The 0.9169-vs-0.9090 gap versus the
  independent reducer is entirely the one-window tambourine divergence,
  NOT the misalignment. **Code-derived boundary:** the DECISION path
  (`Tally`, pooled counts, pooling (a)/(b), the status) is order-invariant;
  only the REPORT path (the per-sequence cursor block) was affected. The
  blast-radius method used at the time ("recompute each window from its own
  detail") cannot distinguish those two paths, so the correct half of that
  claim was true by luck and the incorrect half was asserted.
- **No internal check could detect it** — the per-window records and the
  per-sequence blocks derive from the same misaligned array and agreed with
  each other perfectly. **The single-sequence smoke could not detect it
  either**, being trivially aligned; that is exactly why the smoke passed
  and the twelve-sequence run did not.
- It was caught ONLY because the independent reducer's per-sequence splits
  disagreed while its pooled totals agreed.

Fixed at `4acb0fc` with two regression tests, both VERIFIED to fail against
the pre-fix code with the predicted symptom. The corrected artifact has
**0** windows contradicting their own detail. **Experiment 68's artifact is
PRESERVED, not deleted.**

## What this does and does NOT establish

**ESTABLISHED.** For 96.6% of the windows the frozen instrument scored, an
eligible foreground component sustained multi-view-consistent occupancy of
the instrument's own frozen anchor throughout the absent-frame set, while
the tracker's report failed to qualify — overwhelmingly because its
visibility head returned v < 0.5. **The literal frozen predicate — "the
object's fg-mask component vanishes from ALL cameras whose frustum contains
its last triangulated position" — is therefore NOT established for those
windows, whatever the identity is.** That is a statement about the
instrument's self-consistency and is identity-independent.

**NOT ESTABLISHED, and forbidden by the frozen text until the M1-A0b audit
returns.** That the objects were PHYSICALLY present. C2/C3 cannot separate
"the component is still there, untracked" from "the identity left and the
manipulating hand now occupies the vacated site" — the standing
`ACKNOWLEDGED_LIMIT_r3`. The diagnostic emitted the mandatory stratified
**73-window** M1-A0b sample for exactly this question. Running that audit
is NOT in scope for this phase.

**Also bounded.** The anchor is itself triangulated from tracker reports
that `build_elgs_tracks` admits with no mask-eligibility test
(`anchor_provenance_LIMIT_r3`), so D1 is not fully independent of the
tracker in the failure mode under test. No status here may be stated as
tracker adequacy.

### The identity instruments, restricted to the 577 C3 windows

These were computed and, in an earlier version of this page, **not
reported** — the sharpest omission the final review found, because they
are exactly the discriminators for the identity objection.

**Anchor quality.** `ltp_on_eligible_component` (the fraction of applicable
cameras whose anchor projection lies inside an eligible component at the
anchor's OWN source frame): over the 577 C3 windows, **93.9% are >= 0.5**
and 339 are exactly 1.0. So the anchor is on real tracked surface, not in
empty space — the "drifted onto unrelated background" route is bounded.
**Anchor staleness** is median 11 frames but **p90 = 115** (~1 s at
120 fps, max 412), so for roughly a tenth of windows the anchor is
materially stale and the hand-occupies-the-vacated-site confound is live.

**D2 deterministic mask lineage: `LINEAGE_SURVIVES` in 577 of 577**, mean
survived fraction 0.856, `entry_corroborated_any` in 577/577.

**BUT D2 IS NOT DECISIVE, AND MUST NOT BE PRESENTED AS SUCH.** Merge events
total **4,119,545** across 14,346 (window, camera) lineages, with **576 of
577 windows (99.8%) carrying at least one merge**. A merge is precisely the
failure mode: the identity's component merging with an occluder makes a
"surviving" lineage meaningless. There is a benign explanation — hundreds
of co-located seeds on one object generate O(n²) merges by construction —
but **the merge metric as implemented cannot distinguish the benign case
from the malign one**, so D2 corroborates without settling.

**Structural asymmetry, itemised (direction-of-bias disclosure).** The
`ltp_on_eligible_component >= 0.5` gate applies to **C1a only**. A window
with a poor anchor can therefore never be C1a but is freely admitted to
C3. That asymmetry runs in the direction of the author's disclosed prior
and was not itemised until the final review forced it.

## Consequences for screening

- **The current absence instrument may NOT be reused to claim event supply.**
  It may still be used to measure candidate-opportunity UPPER BOUNDS, with
  the disclosure that its absence limb is dominated by a per-point
  visibility flag and by structurally-unqueried cameras.
- **A corrected or supplemental instrument must be defined.** Two named
  mechanisms, both now measured: the `v >= 0.5` visibility limb (87.6%) and
  the never-queried camera set (12.2%).
- **Tranche 1 must be re-evaluated under any corrected instrument** — but
  see the cost fact below.
- **Conversions and tracks are REUSABLE.** The sealed tracks artifacts store
  per-report `v`, so any change to the visibility threshold, association
  rule, absence predicate, camera set or eligibility floor is a
  CENSUS-LEVEL (CPU) change: re-evaluating all 20 tranche-1 sequences costs
  about **1 CPU-hour and ZERO GPU-hours**. Only a QUERY-CONSTRUCTION change
  (querying every frustum-containing camera rather than only mask-positive
  ones — the fix for the 12.2% never-queried limb) would require
  re-tracking.
- Scope and cost for the subsequent exhaustive screen:
  [[operations/elgs-exhaustive-screen-scope]].

## Compute accounting

| | estimated | actual |
|---|---|---|
| GPU computation | none | none (CPU-bound throughout) |
| Slot-hours | 0.5 projected/cell | exp 68 ≈ 0.24, exp 69 ≈ 0.24 |
| Wall | 15–40 min | 14.5 min each |
| Storage added | < 200 MB | ~9.8 MB (2 × 4.9 MB artifacts) |
| New downloads / conversions / tracking | none | none |

Peak RSS 4.7 GB. Superseded work: experiment 68 (defective per-sequence
attribution) — preserved. Claims r0–r2 of `m1absdiag_primary` consumed by
a Git-Bash path-mangled dry-run and two clean dry-runs — preserved, not
deleted, per append-only discipline; the mangling reproduced a failure this
project has recorded before, and the fix is to submit from PowerShell.

## Defects in the frozen text found during execution (recorded, not silently fixed)

1. **`C5_NON_IDENTIFIABLE` and `C5_STRUCTURALLY_SILENT` are co-extensive**
   in the decision list, yet `classification.binding` sums all three C5
   names while `totality` asserts a partition. Taken literally this
   double-counts, and could inflate the C5 limb of status_4 by up to 2×.
   NOT load-bearing here (both are 0). Flagged by the independent reducer.
2. **`c_leave_scissor_out_pooled` matched the literal name `scissor`** and
   so failed to exclude `scissor_screen_w0_561`. Reported-only, never
   binds; the correct value is 0.9528.
3. **The artifact omits the commit** from its provenance block (recoverable
   from the ledger and Determined: `4acb0fc`).
4. The disclosed poker bridged-frame count (346) counted frames with a
   VISIBLE report; the frozen `temporal_unit_W` defines bridged by
   ASSOCIATED, giving **335**. 335 + 11 OFF_COMPONENT = 346. The reducer's
   335 is the value the frozen text specifies and is corroborated by the R6
   guard passing on all 109 poker windows.
