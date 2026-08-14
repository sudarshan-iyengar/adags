# EL-GS Corrected-Substrate Remeasurement — G-R FAILS; tranche 1 has ZERO eligible candidates

Date: 2026-08-13/14. Governing defect record:
[[operations/elgs-substrate-defect-2026-08-13]]. This page records the
remeasurement of the three defective conversions on the CORRECTED image
substrate, with the frozen cycle-2 eligibility predicate and the frozen
cycle-3 G-R gate REAPPLIED UNCHANGED. No threshold, floor, prereg, gate
definition, evaluator or instrument constant was altered; only the input
substrate was corrected.

## Verdicts

**G-R (reactivation precondition; writing_2 unscreened half): FAIL.**
Union returns **0 < 36**. Coverage 0.9340 >= 0.5 passes; the return floor
fails by the full margin. Under the NOTE-2 binding reading the count is
also 0, straddle count 0 — NOTE-2 does not rescue it. The previously
recorded PASS (union 64) is SUPERSEDED: it was produced by the substrate
defect, not by the data.

**Cycle-2 eligibility (union >= 12 AND coverage >= 0.5): BOTH NOT
ELIGIBLE.** writing_2 union 1 < 12; xylophone union 0 < 12. Coverage
passes comfortably for both. **Tranche 1 therefore contains ZERO eligible
candidates** (the other 18 rows are from verified-clean conversions and
were already ineligible).

## Measured statistics (evaluator `79ae5b7`)

| statistic | writing_2 screen (0–239) | xylophone screen (0–306) | writing_2 full (0–480) |
|---|---:|---:|---:|
| occlusion opportunities | 7,171 | 2,899 | 11,972 |
| true-absence candidates | 2 | 0 | 2 |
| primary returns (strict) | 1 | 0 | 1 |
| **union returns** | **1** | **0** | **1** |
| coverage | 0.924308 | 0.778732 | 0.929123 |

writing_2 full, unscreened half: **union returns 0**, coverage 0.933969
(6,266/6,709); second-half occlusion 4,721; second-half true-absence 1;
straddle count 0; per-identity decomposition of unscreened-half returns:
EMPTY.

## Defective vs corrected (the same evaluator, same floors, same window)

| | defective | corrected | direction |
|---|---:|---:|---|
| writing_2 screen union returns | 50 | **1** | collapse |
| writing_2 screen true-absence | 84 | **2** | collapse |
| writing_2 screen coverage | 0.845 | **0.924** | IMPROVED |
| writing_2 screen occlusion | 7,315 | 7,171 | ~unchanged |
| writing_2 2nd-half union returns | 64 | **0** | collapse |
| writing_2 2nd-half coverage | 0.8637 | **0.9340** | IMPROVED |
| xylophone occlusion | 1,031 | **2,899** | ~3x higher |
| xylophone coverage | 0.577 | **0.779** | IMPROVED |
| xylophone union returns | 0 | 0 | unchanged |

**Coverage rose in every case** — corrected registration tracks BETTER,
which is the signature of a genuine fix rather than a second defect.
Occlusion, the predicate most robust to a uniform pixel offset (component
present in >= 2 cameras, absent in >= 1), barely moved for writing_2 and
rose for xylophone. Absence — which requires vanishing from ALL containing
cameras, and is therefore maximally sensitive to mislocated mask
lookups — collapsed 84 -> 2.

**Physical plausibility check.** The defective figures implied a pen
vanishing from EVERY containing camera of a 53-camera surround rig roughly
every three seconds during a writing-and-erasing task. That was never
plausible. Two full-multiview disappearances in 481 frames is.

## Controlled comparison

The corrected and defective tracks artifacts are identical in every
instrument parameter: frames (240 / 481 / 307), realized window, 26
tracking cameras, hull_resolution 96, max_seeds 512, same evaluator, same
frozen boundary conventions. **The only variable that changed is the image
substrate.**

## Verification (standing protocol satisfied)

- **Integrity audit: CLEAN.** All six cells (experiments 62–67) at the
  single pushed commit `79ae5b7`, hopper, digest-pinned image
  `sha256:a2877f26…`, `evidence_bearing: true`, retry 0 — no invalidated
  or retried cells.
- **Independent fresh-context recomputation: EXACT AGREEMENT on all
  seventeen compared numbers.** A fresh worker wrote its own ~330-line
  reduction from the frozen prereg texts alone, never reading
  `build_m1_census.py`, its tests, any `census.json`, or any result page;
  it verified the projection convention empirically (median re-projection
  residual 0.56–1.66 px) rather than by copying projection code; it ran on
  Apollo via `det cmd run` (no experiments, no ledger writes). Derived
  constants independently reproduced: 26 training cameras, 512 seeds,
  rig_radius 5.028491508127949, r_site 0.2514245754063974, angular floor
  0.6296714642878981 rad.
- **Robustness.** Six residual definitional readings were enumerated
  (rig-radius camera set; greedy enumeration resume point; occlusion
  camera set; coverage universe; optical-axis sign; depth convention).
  None is verdict-relevant. Decisive bound: writing_2's FULL frame range
  contains only 2 true-absence candidate windows, and union returns are a
  subset of those by frozen definition, so the unscreened-half union count
  is bounded above by 1 under ANY admissible variant. The 36-floor is
  unreachable.
- Concordance diagnostic (prereg-provided, diagnostic-only): the
  full-range artifact attributes 1 true-absence candidate to the first
  half where the separately-tracked screen artifact yields 2 — two
  independent tracker runs over different windows. Touches neither verdict.

## Consequences

- **The reactivation precondition is NOT restored.** G-R FAILS on the
  corrected substrate.
- **The cycle-3 rescope's premise is VOID.** It rested on writing_2 being
  the unique member of the operational scope predicate (>= 12 union
  returns at >= 0.5 coverage). Corrected, writing_2 does not satisfy that
  predicate, so there was no valid anchor and no valid subset.
- **CC4/G14 empirical support in DiVa-360 tranche 1 is ZERO sequences,
  not one.** The scoping addendum in [[operations/elgs-novelty-record]] is
  corrected accordingly.
- **The cycle-2 DRY outcome STANDS** — fewer than 3 eligible either way —
  and the checkpoint autonomy condition (>= 2 eligible) remains NOT MET,
  so the post-tranche decision remains the user's. xylophone does NOT flip
  it.
- **G-OA's valid FAIL is UNCHANGED and REMAINS FINAL.** Its sole violation
  is pour_tea's per-sequence unscreened-half coverage (0.3748), computed
  entirely within pour_tea, whose conversion is verified clean. Corrected
  writing_2 contributes 2 absences instead of 63, leaving pooled absence
  ~144 >= 36; a FAIL cannot become a PASS. This remeasurement does not
  reopen G-OA.
- **The G13 occlusion/absence supply claim is largely intact.** Of the
  ~700 pooled screened-half absences, writing_2 contributed 84 (now 2) and
  xylophone 0; the other 18 rows are from verified-clean conversions. The
  supply remains large, and this remeasurement does not re-open it.

> **APPEND-ONLY CORRECTION (2026-08-14) — the bullet immediately above is
> HALF SUPERSEDED.** The OCCLUSION half stands. The ABSENCE half does not:
> a frozen, four-times-reviewed diagnostic over all 597 corrected
> true-absence windows returned **status_2 (material defect), UNANIMOUS
> across 144 sensitivity readings**. **ZERO** windows are corroborated as
> genuine full-multiview disappearance; in **96.6%** an eligible foreground
> component sustained multi-view-consistent occupancy of the instrument's
> own anchor while the tracker's report failed to qualify. The evidence is
> **87.6%** per-point visibility flags below 0.5 and **12.2%** cameras
> never queried for that seed. Because track coverage uses the SAME
> `v >= 0.5` threshold, coverage and absence are mechanically coupled
> through one constant.
>
> This does NOT reopen anything on this page: the substrate remeasurement's
> own verdicts (G-R FAILS; both sequences ineligible; DRY stands; G-OA
> unchanged) all turn on RETURNS and COVERAGE, not on the absence class.
> The corrected pooled absence total is **597**, not ~700. See
> [[operations/elgs-absence-diagnostic-result]].

## What is NOT concluded here

- The reactivation MECHANISM is not refuted. What is established is that
  DiVa-360 tranche 1 has not been shown to contain the events required to
  test it — a measurement finding about the dataset subset, not about
  EL-GS's method or mathematics.
- No route decision is taken here (tranche 2, dataset extension, descope,
  or anything else). That is the user's, as it has been since the cycle-2
  checkpoint.
- Compute: ~0.5 GPU-h across experiments 62–67 plus CPU conversion tasks.

## Preservation

All defective artifacts, conversions, tracks, censuses, numbers and the
original recorded verdicts are PRESERVED unchanged. The corrected
conversions and tracks live beside them under the `_fix79ae5b7` suffix;
corrected censuses under `runs/elgs/m1c3fix_census_*_r0/`.
