# EL-GS M1 Census Result — GATE FAILED (final M1 result)

Date: 2026-08-11/12 (UTC). Status: **M1 FAILED at the preregistered
M1-A0 gate** on `same_object_returns_min`. Under the frozen
failure/retry policy ([[operations/elgs-m0-m1-implementation-plan]]
§11.2; `configs/elgs/prereg_m1_census_v1.json` revision 3, reviewer
SIGNED at `72ff97e`), a scientifically valid failure against these
floors is the FINAL M1 result; replacement is possible only under the
verified-defect rule. What follows is a user decision. Full execution
trail: [[operations/elgs-m1-census-record]].

## The gate computation (experiment 14; integrity-audited)

Cell `m1_a0_dev_census`, Determined experiment 14, commit `ec284e1`,
digest-pinned image `sha256:a2877f26cb8528…`, hopper, the SIGNED
revision-3 prereg as the hash-verified named config; inputs = the
three sealed dev tracks artifacts + converted masks + calibration;
in-container `runtime_assertions` passed; the entrypoint, argv,
image, commit, claim, and ledger all match (audit in the census
record). Wall ≈ 12.3 min, ≈ 0.2 GPU-h of the 25 GPU-h census ceiling.

## Result: floors 36 / 36 / 36 / 0.5 applied to POOLED dev-subset values

| statistic (pooled) | evaluator (exp 14) | independent recomputation | floor | verdict |
|---|---|---|---|---|
| occlusion_opportunity_upper_bound | 93,841 | 93,841 (exact match) | ≥ 36 | PASS |
| true_absence_candidate_count | 600 | 600 (exact match) | ≥ 36 | PASS |
| same_object_return_count | **23** | **30** | ≥ 36 | **FAIL** |
| track_coverage_upper_bound | 0.71963 (78,876/109,607) | 0.71963 (identical tallies) | ≥ 0.5 | PASS |

Per-sequence (evaluator / recomputation): occlusion 5,506/5,506
(unlock), 19,015/19,015 (flip_book), 69,320/69,320 (battery);
true-absence 100/100, 0/0, 500/500; same-object returns 17/23, 0/0,
6/7; coverage 0.2706/0.27064, 0.8403/0.84028, 0.7735/0.77345.

## Independent verification (per the preregistered protocol)

A fresh-context worker recomputed all four gated statistics from
PRIMARY inputs (tracks artifacts, masks, calibration) with its own
reduction written from the revision-3 text alone, never reading the
census implementation or its outputs (Determined command
`771264e6-a0da…`, CPU-only, ≈7 min). Its full record — including
twelve named definitional readings where the frozen text left
residual freedom — is preserved in the session artifacts; its bottom
line verbatim: "my independent reduction reproduces a mechanically
executable gate from the revision-3 text, and on the frozen dev
subset it yields 93,841 / 600 / 30 / 0.71963 pooled — FAIL on the
same-object-return floor (30 < 36), PASS on the other three."

**Adjudication of the sole divergence.** The two implementations
differ only on the return-side statistics (SOR 23 vs 30;
return_position_undefined 109 vs 70), traced to residual freedom in
the re-appearance semantics around the frozen `return_position`
anchor (which camera set defines re-appearance; transition-vs-state
at the window opening; end-of-sequence flicker). Under BOTH faithful
readings the pooled count is far below 36, and the recomputation's
distance histogram is cleanly bimodal (returns at 0.03–0.21 vs
relocations at 0.60–0.93 against r_site ≈ 0.21–0.25), so the verdict
is not threshold-knife-edge and is ROBUST to every identified
divergence. No verified implementation, instrumentation,
corrupted-data, or protocol-execution defect was identified against
the gate computation.

> **APPEND-ONLY CORRECTION (2026-08-14) — the ABSENCE statistic on this
> page is now known to be materially confounded.** The gate VERDICT is
> untouched: it failed on `same_object_returns_min` (23-30 vs floor 36),
> a RETURN statistic, and that remains a valid, final, preserved negative.
> What is corrected is the supporting reading below that the dev subset
> holds "600 candidates" of "genuine multi-view absence". A frozen
> diagnostic over the tranche-1 screening windows returned **status_2
> (material defect), UNANIMOUS across 144 readings**: zero of 597 scored
> true-absence windows are corroborated, 96.6% being windows where
> foreground demonstrably occupied the instrument's own anchor while the
> tracker flagged v < 0.5 (87.6% of evidence pairs) or the camera was
> never queried (12.2%). The word "genuine" in the reading below is
> therefore NOT supported for the absence class; "candidate" is.
> The same caution applies to the unlock 0.98 -> 0.24 mean-visibility
> observation, which is a TRACKER-VISIBILITY statistic, not an existence
> one. See [[operations/elgs-absence-diagnostic-result]]. Original text
> preserved unchanged.

## Scientific reading (why the floor failed)

The dev subset is rich in occlusion opportunity (93,841) and genuine
multi-view absence (600 candidates; e.g. unlock's mean tracker
visibility drops from 0.98 in the static prelude to 0.24 during
manipulation), but SAME-OBJECT RETURNS — a true-absence candidate
whose identity re-appears within r_site of its disappearance
position — are scarce: 23–30 pooled, dominated by unlock; flip_book
has none (pages turn, nothing leaves and returns); battery
contributes ≤ 7. A large tally of candidates (70–109) terminated
with re-appearance runs whose consensus was undefined throughout —
under the frozen, reviewer-signed anchor rule these count as
true-absence but never as returns. This frozen choice is
outcome-relevant (≈6 resolved returns would have flipped the floor)
and is recorded as such; it was fixed pre-data by the review chain
and binds.

## Claim limitations (preregistered)

- M1-A0 statistics are model-free candidate-opportunity UPPER
  BOUNDS; meeting a floor is a necessary condition, never evidence
  that that many true events exist (revision-3
  `necessary_condition_note`).
- M1-A0b (audited true absence) did not run; as preregistered it was
  diagnostic-only, and M1 therefore cannot support a claim-grade
  estimate of true-event absence or false-positive prevalence.
- Cells not run after the failed gate: M1-A0b, M1-A, M1-D
  (diagnostics; submission halted at the gate), M1-B/M1-C
  (conditional on an M1-A0 pass by their frozen definitions).

## Compute accounting

Census: ≈ 0.2 GPU-h of the 25 GPU-h ceiling (experiment 14 only).
Preprocessing (separately accounted, no ceiling): ≈ 0.1 GPU-h actual
across experiments 8–13 vs 1.75 projected, hardware and scope in the
experiment ledger.

## Disposition

M1 is a permanently preserved negative under the frozen policy. The
EL-GS scientific chain (M2+) is NOT unblocked. Options that exist
OUTSIDE this plan's scope — a different dataset/sequence selection,
a revised gate under a new preregistration cycle, or abandoning the
direction — are the user's decision, per §11.2 and the plan's §16c
pause conditions.
