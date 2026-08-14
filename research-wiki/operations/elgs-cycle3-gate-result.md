# EL-GS Cycle-3 Gate Result — G-R PASS / G-OA FAIL (split verdicts)

> **APPEND-ONLY SUPERSESSION NOTE (2026-08-13).** The "Material provenance
> finding" below is now a **VERIFIED DEFECT**, and its verdict-robustness
> assessment does not hold. See
> [[operations/elgs-substrate-defect-2026-08-13]].
>
> - **G-R is INVALIDATED PENDING CORRECTED REMEASUREMENT — not FAIL, not
>   PASS.** writing_2's frames came from `segmented_ngp` (1280x720) while
>   its calibration declares 1160x550; geometry-to-image registration is
>   off by ~(107,120) px (IoU 0.082 at the offset the code uses; only
>   14.4% of foreground addressable). Seeds, queries, triangulation and
>   the coverage denominator are all affected, so the frozen artifact
>   cannot be re-scored — it must be re-measured. The corrected outcome is
>   UNKNOWN in either direction.
> - The "INTERNALLY CONSISTENT" argument recorded below (association
>   0.997, reprojection median 2.35 px) is **not** evidence of
>   correctness: a uniform pixel offset predicts exactly those
>   diagnostics. The declared-vs-image bounds sensitivity (union 64 vs 66)
>   re-analysed the frozen artifact and could not test its inputs.
> - **G-OA's valid FAIL is UNAFFECTED and REMAINS FINAL.** Its sole
>   violation (pour_tea per-sequence coverage 0.3748) is computed entirely
>   within pour_tea, whose conversion is clean. Correcting writing_2
>   cannot convert that FAIL into a PASS. This defect must not be used to
>   reopen it.
> - All numbers, artifacts and verdicts recorded below are PRESERVED
>   unchanged as the original record.
>
> **REMEASUREMENT COMPLETE (2026-08-14).** G-R is no longer "pending" —
> it was remeasured on the corrected substrate under the UNCHANGED frozen
> gate and **FAILS**: unscreened-half union returns **0 < 36** (coverage
> 0.9340 passes; straddle count 0, so NOTE-2 does not rescue it). The
> PASS recorded below (union 64) was produced by the substrate defect.
> Independent fresh-context recomputation agreed EXACTLY on all seventeen
> gate-bearing numbers; integrity audit clean. Result:
> [[operations/elgs-substrate-remeasurement-result]]. G-OA's valid FAIL
> below is unchanged and remains final.

Date: 2026-08-12. Governing frozen protocol:
`configs/elgs/prereg_m1_cycle3_gate_v1.json` (SIGNED at `6de4d60`,
zero blocking findings, three binding readings). Design:
[[operations/elgs-cycle3-rescope-design]]. Both verdicts were
recorded only after the standing verification protocol: independent
fresh-context recomputation from primary inputs + integrity audit.

## Verdicts

**G-R (reactivation precondition; writing_2 alone): PASS.**
Unscreened-half (frames 240–480) union returns **64 ≥ 36**;
coverage **0.8637 ≥ 0.5**. Every counted return also passes the
STRICT cycle-1 primary predicate (64 primary — the R2′ loosening
contributed nothing to the pass; per the recomputation, 58 of 64
also matched R2′ and 6 were primary-only). Straddle count 16
(binding reading NOTE-2). Per-identity decomposition: 46 distinct
identities, max 4 returns each, in ~6 temporal bursts — no identity
dominates; the burst structure is consistent with the prereg's
variance disclosure and feeds the frozen temporal-block bootstrap
plan for M2. The unscreened half BEAT the screened half (64 vs 50):
no phase confinement.

**G-OA (occlusion/absence precondition; three dev sequences): valid
FAIL — sole violation: pour_tea unscreened-half coverage 0.3748 <
0.5.** Every other floor passed: pooled occlusion 23,313 ≥ 36;
pooled absence 205 ≥ 36; per-sequence absence 63/47/95 all ≥ 12;
pooled coverage 0.5558 ≥ 0.5; writing_2 0.8637 and tambourine
0.7312 per-sequence coverage. pour_tea's tracker coverage collapsed
in its second half (screened 0.591 → unscreened 0.375) — precisely
the failure mode the per-sequence floor exists to catch (cf. poker
0.382 / scissor 0.441 screening exclusions). Under the frozen
policy this FAIL is FINAL FOR THIS SUBSET for the
occlusion/absence claim family.

## Verification

- Independent recomputation (fresh context, own reduction from the
  frozen texts, primary inputs only; Determined task `a4c63073…`):
  EXACT agreement on every gate-bearing number (union 64, primary
  64, coverage 0.863664, straddle 16, pooled 23,313 / 205 /
  0.555812, pour_tea 0.374846, identical 46-identity multiset).
  Verdicts confirmed robust to the one disclosed ambiguity (below).
- Integrity audit: all seven cells (experiments 55–61) claim pushed
  commit `2b33556`, evidence-bearing, digest-pinned image
  `sha256:a2877f26…`, hopper, exact intended entrypoints,
  runtime_assertions passing; the gate artifact records the three
  signature binding readings and the census sha256s.
- Concordance diagnostic (screened halves from full-range tracks vs
  the sealed half-track table): writing_2 and tambourine near-exact
  (union 50/50, 10/9); pour_tea diverges materially (absence 113 vs
  73; coverage 0.526 vs 0.591) — full-window tracking changes its
  track field, consistent with its weakest-coverage status.

## Material provenance finding (verdict-robust; disclosed)

writing_2's converted scene drew frames from `segmented_ngp`
(1280×720 segmented space) rather than `frames_1`: only
segmented_ngp covered its referenced frame paths, so the
resolution-probe disambiguation never ran (single-covering-archive
path). The chain is INTERNALLY CONSISTENT (reports and masks share
the space; association hit-rate 0.997; consensus reprojection
median 2.35 px) and the recomputation verified BOTH bounds readings
give identical verdicts (declared-bounds union 64 / image-bounds
sensitivity 66; G-R PASS, G-OA FAIL under both). CARRIED FORWARD as
a known evidence-substrate asymmetry: writing_2's tracks come from
segmented imagery while the companions use undistorted frames_1 —
to be resolved (e.g., writing_2 frames_1 investigation or disclosed
substrate policy) BEFORE any M2 training run.

## Standing notes

Model-free upper bounds: floors are necessary conditions, never
evidence of true events. The R2′ loosening disclosure is satisfied
trivially here (primary = union on the pass). The novelty scoping
addendum was appended to [[operations/elgs-novelty-record]] per the
frozen prereg (regardless of verdict). Compute: ~0.7 GPU-h of the
1.5 ceiling across experiments 55–61.

## Consequences (frozen structure)

- **The reactivation precondition (G-R) is RESTORED for the
  rescoped claim family** (the operational scope predicate;
  writing_2 sole member). Starting M2 remains a separate explicit
  user approval.
- **The occlusion/absence precondition (G-OA) FAILED on this
  subset** (final for the subset). The remaining options for the OA
  family are the user's: (a) a follow-on companion re-selection
  cycle (the frozen companion rule applied minus pour_tea yields
  put_candy — 18 screened-half absences at 0.507 coverage; a small
  new prereg round + gate re-run on the new subset); (b) tranche-2
  screening (21 short + 8 long unscreened); (c) R4 dataset
  extension; (d) full R3 descope. The OA event supply itself is not
  in doubt (205 pooled absences, 23k occlusions on THIS subset) —
  the failure is one companion's tracker legibility.
