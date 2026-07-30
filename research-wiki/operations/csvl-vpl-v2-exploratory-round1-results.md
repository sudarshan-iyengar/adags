# CSVL-VPL v2 Exploratory Round 1 — Results (workstream B)

Date: 2026-07-30
Branch: `csvl-vpl-v2-exploratory`
Contract: [[operations/csvl-vpl-v2-exploratory-contract]] (committed at
`932b32b`, corrected at `e584ea3` after the pre-launch verifier; both
pre-date every lane submission; `e584ea3` post-dates only the pre-fix smoke
whose result it cites)
Tier: exploratory, development scene `cut_roasted_beef`, seed 0, single round.
Nothing here is Gate A, Gate B, Phase 0 success, or a disocclusion claim; both
census verdicts stand. Global and dynamic-mask deltas below are NOT evidence
of improved disocclusion (that requires annotated events per the objective).

## Jobs (all verified against sacct)

Smokes `50886783` (pre-fix, 19:55) and `50891508` (post-fix, 21:15); evidence
build `50882303` (12:45); lanes `50896779` L0 (3:16:04), `50896788` L1
(3:24:50), `50896801` L2 (3:20:33), `50896810` L3 (3:29:30), `50896816` L4
(3:28:24), `50896823` L5 (3:28:24) — all of these COMPLETED 0:0. Qualitative
renders: `50968178` FAILED (1:41), `50972304` FAILED (1:43), `50973410`
COMPLETED 0:0 (1:26) — see the tooling-debug chain below. Round-1 GPU cost
21.4 h (within the 24 h expectation).

## Iteration history

- Round 0 (pre-launch, not result-conditioned): fresh-context verifier found
  five blockers; all fixed uniformly before submission (mean-preserving
  exposure denominator — the raw denominator would have amplified capacity
  one-sidedly; contract wording for the time-shift control; per-metric AND
  noise band; instrumentation gaps: peak CUDA memory, protected/pruned
  stable-id samples; donor recency exclusion). Commit `e584ea3`.
- Round 1: the six-lane matrix below, run once, no post-hoc changes.
- Rounds 2-3: NOT used. Decision recorded below.

## Results — `--val` protocol (cam00 test renders at 6000 iterations)

| Lane | Mechanism | PSNR | SSIM | LPIPS | Gaussians |
|---|---|---:|---:|---:|---:|
| L0 | baseline (lifecycle off) | 34.366 | 0.9605 | 0.0524 | 541,662 |
| L1 | E1 protection + occlusion-aware exposure | 34.231 | 0.9613 | 0.0498 | 592,209 |
| L2 | presence-VAD exposure control | 34.399 | **0.9628** | **0.0497** | 599,571 |
| L3 | full E1/E2 lifecycle | 34.306 | 0.9608 | 0.0517 | 591,774 |
| L4 | generic capacity control | 34.020 | 0.9613 | 0.0514 | 593,336 |
| L5 | L3 with time-shifted (misaligned) evidence | **34.479** | 0.9622 | 0.0500 | 593,657 |

training_report protocol (context, never mixed with the above): static-region
PSNR L0 34.827 / L1 34.693 / L2 34.848 / L3 34.768 / L4 34.437 / L5 34.949;
dynamic-mask PSNR L0 25.320 / L1 25.713 / L2 26.037 / L3 25.752 / L4 25.655 /
L5 25.816; static ghost best in protection lanes (L1 0.0950, L5 0.1013, L3
0.1018 vs L0 0.1027).

Historical anchor reproduced: L0 34.366/0.9605/0.0524 vs the 20260619 run's
34.25/0.9610/0.0518 — the harness is consistent.

## Activation validity (a result may not hide behind inactivity)

All lanes pass their contract requirements: L1/L3/L5 protection active well
before iteration 2000 (per-iteration batch counts ranging 211-7,468 with
means ~1.8-2.5k over that window; the maxima are 6,459/6,467/7,468) with
3.4-3.6e8 occluded verdicts over training and abstention active; L3/L4/L5 realized all 9 birth events at 256/256 rows with
zero skips and point-neutral transactions (budget equality logged per event);
L2's presence weights average ~0.33 by cross-lane exposure-mean ratio (not a directly logged field; strong reallocation); peak CUDA
12.8-12.9 GB; per-checkpoint point counts logged for the five
lifecycle lanes (ledger post-densify snapshots ~412k @1000, 515-570k @3000,
saturating toward the 600k cap @6000; L0 writes no ledger — its saved
checkpoints hold 397,405 @1000 and 484,182 @3000, self-limiting at 541,662).
One validity criterion — "exposure denominator differs from raw view counts
for >=1% of primitives" — has no directly logged raw-count baseline and is
satisfied by inference from the verdict histograms (>=2.4% of primitives
occluded-weighted in the first ledger interval alone); disclosed rather than
claimed as instrumented.

## Pre-declared interpretation (applied exactly as contracted)

- **No visibility attribution.** The decisive contrast failed in the
  strongest possible way: the misaligned-evidence control L5 beat the aligned
  L3 on every val metric (+0.17 PSNR, −3.3% LPIPS, +0.0014 SSIM) and even
  posted the best PSNR and static PSNR of the matrix. Whatever the lifecycle
  lanes gain does not depend on the evidence being time-aligned with reality.
- **Occlusion-awareness adds nothing over presence reweighting** at this
  tier — reported as at-least-as-strong-as the pre-declared
  presence-equivalent pattern (whose exact precondition "L2 ~ L1 with both
  > L0" is not met: L2 exceeds L1 by +0.168 dB PSNR and L1 sits below L0):
  L2 matches L1 on LPIPS (0.0497 vs 0.0498) and beats it on SSIM and PSNR,
  so the occlusion-aware evidence coupling underperforms its own
  presence-only control.
- **The full mechanism beats the generic-capacity control at matched
  capacity**: L3 vs L4 is +0.286 dB PSNR with final counts within 0.26%
  (valid per the 2% rule). Attribution caveat (verifier-flagged): L3 and L4
  differ in THREE mechanisms (protection, exposure, birth-target
  construction), so this contrast does not isolate which limb drives the
  gap — and L1 (protection+exposure, no births) at 34.231 < L3 shows the
  internal limbs alone do not explain it either. The pre-declared reading is
  simply that L3 is not capacity-equivalent to generic churn; a
  single-limb attribution would need a dedicated round-2 split that was not
  run.
- **Capacity is disclosed and instructive**: every lifecycle limb (exposure
  reallocation, protection retention vetoes, birth-seeded densification
  pressure) independently pushed lanes toward the 600k cap (+9-11% Gaussians
  vs L0). More capacity did not buy quality — L4 with +52k Gaussians is
  0.35 dB WORSE than L0 globally and −0.39 dB on static regions.
- **Static no-harm**: L5 and L2 pass; L3 marginally fails (−0.059 dB) and L1
  (−0.134) and L4 (−0.39) fail the −0.05 dB bound (training_report protocol
  diagnostic). Protection lanes do reduce static ghost slightly.
- **Mixed metric signs** (PSNR down / LPIPS up for L1 and L3 vs L0) are
  contradictory under the per-metric band rule: neither signal nor null;
  reported as mixed.
- **35 dB aspiration**: not crossed by any lane (max 34.479, L5). Reported
  in context; no SOTA implication.

## Round-2/3 decision: not exercised

A mechanism change responding to L3's own result would burn one of the two
remaining rounds without a hypothesis for why evidence *alignment* should
start mattering in global/val metrics when a wrong-time copy of the same
evidence already performs equally: the failure is structural at this metric
tier, echoing both censuses — the localized events the mechanism targets
occupy too small a pixel fraction for scene-level metrics to reward
alignment. The pre-declared response to this pattern is annotated event-level
evaluation, not another blind iteration. Negative preserved at its tier.

## What these results do and do not support

Support: engineering feasibility of the full automatic pipeline (protection,
exposure, E2 birth, point-neutral transactions, instrumentation) in full
6000-iteration trainings at +2.3% to +6.9% wall-time overhead (mechanism
compute itself is ~0.6%; the rest is capacity/eval-driven); a contract-valid
perceptual signal for the presence-VAD control L2 (LPIPS -5.2% with PSNR
in-band; L1's mixed signs remain classified mixed, not banked as a gain);
the full mechanism beating generic capacity churn at matched counts (limb
attribution not isolated); slight ghost reduction in protection lanes.

Do NOT support: any claim that calibrated visibility evidence improves
reconstruction (L5 refutes it at this tier); any disocclusion claim; any
Gate A/B or Phase 0 statement; generalization beyond cut_roasted_beef seed 0.

## Qualitative artifacts

Checkpoint-aligned panels (GT | render | render+E1-overlay) at iterations
1000/3000/6000 for L0, L3, L5 over three label-free windows (cam05:60,
cam08:175, cam13:250), plus per-window contact sheets and overlay verdict
histograms: `runs/csvl_vpl_v2_exploratory/<run>/qualitative/` (job
`50973410`, COMPLETED 0:0). Tooling-debug chain, recorded: attempt
`50968178` FAILED (checkpoint loaded with `map_location="cpu"` — host
tensors reach the rasterizer, which takes raw data pointers; the trainer
never hit this because it loads without map_location); attempt `50972304`
FAILED (device-string false alarm: "cuda:0" != "cuda" in the new guard).
Both were render-tool-only fixes; no training artifact was touched.
Visual verification of the cam08:175 L3 panel: near-verdict overlay dots
land on visible surfaces (blinds, walls, shelving) and the occluded arc
correctly hugs the region behind the moving person — E1 geometry aligned;
per-frame verdict shares (near ~48%, occluded ~3.3%, not-evaluable ~41%
on dynamic/invalid regions) consistent with census-scale expectations.

## Evidence limitations

Single scene, single seed, no annotated events, two eval protocols kept
separate, capacity deltas vs L0 disclosed but not eliminated, and the
between-run densification variance of this codebase (historical L0-family
runs span 541.7k-559.4k points) means small cross-lane point differences are
partly chaotic. The value of this round is causal structure, not headline
numbers.
