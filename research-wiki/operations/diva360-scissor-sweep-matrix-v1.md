# FROZEN development sweep matrix — DiVa-360 scissor (2026-08-16)

**Status: FROZEN BEFORE ANY SWEEP CELL RAN.** Recorded and committed
prior to submission, per the directive's rule 1 ("Before inspecting
sweep outcomes, freeze a matrix of at most six configurations").

Trigger: the benchmark-aligned baseline (experiment 84,
`configs/elgs/diva360_scissor_bench30.yaml`, commit `bb7bc52`) reached
**best_val PSNR 21.3705 / SSIM 0.9070 at iteration 5000**, below the
24 dB threshold, so the targeted sweep fires.

EXPLORATORY throughout. Nothing here is claim-grade.

## What is held constant across all six cells

Same **seed** (0), **image** (`sudarshaniyengar/adags:apollo-v100-v1`),
**commit** (one pushed commit for the whole matrix), **dataset** (the
30-FPS materialization), **validation schedule**
(`test_iterations 500/1000/2000/3000/4000/5000/6000`), and **hardware
class** (V100, `dgx`). No cell runs on hopper; no H100 result enters
this comparison.

**Scene: the carved development split, not the official one.** Ranking
happens on 5 development cameras held out of the 35 TRAINING cameras;
the model fits on the remaining 30. The six official held-out cameras
(`0, 16, 17, 33, 43, 44`) are absent from BOTH transforms files in that
scene directory, so they are unreachable from any sweep cell and cannot
leak into selection even by accident.

## The six cells

Base config for every cell is experiment 84's
`diva360_scissor_bench30.yaml`. Each cell changes ONLY what its row
says.

| cell | axis | change from base |
|---|---|---|
| **S0** | control | none |
| **S1** | initialization | `points3d.ply` reseeded 20,000 -> 200,000 points, SAME disclosed volume and seed |
| **S2** | densification schedule | `densify_until_iter` 6000 -> 4000 |
| **S3** | capacity | `densify_until_num_points` 600,000 -> 300,000 |
| **S4** | routing | `enable_soft_routing` true -> false |
| **S5** | combination | S1 + S2 together |

Three axes: initialization, densification/capacity, routing — exactly
the three the directive named as priorities. **No learning-rate axis
and no learning-rate values change anywhere in this matrix.**

S5 is chosen A PRIORI as the pair with the strongest independent
evidence (below), NOT after seeing any single-cell result. No adaptive
follow-up cell will be added after outcomes are visible.

## Evidence supporting each axis (all measured this session)

**Initialization.** Experiment 84 pruned **20,000 -> 3,398 -> 3,254**
points by iteration 990 before densification recovered: roughly 84% of
the initial cloud is destroyed. The converter's own docstring calls the
synthesized `points3d.ply` "a coarse smoke-test volume, NOT a
claim-grade initialization", and its measured extent is +/-6.5 world
units against scissor content at about +/-1.2.

S1 tests the direct implication — more seeds in the same volume leave
more survivors after the prune. It is honestly **not** a better prior;
it is more of the same coarse prior. A content-aware initialization
(visual hull from the per-frame masks the converter already writes) is
the natural follow-up and is deliberately NOT in this matrix, because
it is a new component and this sweep is not the place to debut one.

**Densification / capacity.** Experiment 84's validation curve:

| iter | PSNR | SSIM | points |
|---:|---:|---:|---:|
| 500 | 17.8838 | 0.74860 | 20,000 |
| 1000 | 18.5315 | 0.83938 | ~3,254 |
| 2000 | 20.3933 | 0.89091 | ~82,143 |
| 3000 | 20.9610 | 0.90237 | |
| 4000 | 21.2204 | 0.90643 | |
| **5000** | **21.3705** | **0.90698** | 338,528 |
| 6000 | 21.3079 | 0.90488 | 507,178 |

Validation **peaks at 5000 and DECLINES at 6000 on both metrics** while
the point count grows by 169,000. The last 1,000 iterations of
densification made the model worse. S2 (stop densifying at 4000) and S3
(cap at 300k) are two independent mechanisms for the same finding.

**Routing.** Experiment 84 ended at `mean_dynamic_prob 0.99906`,
`percent_near_dynamic 99.99%`, `percent_uncertain 0.010%`, with the
static branch carrying `expected_static_fraction 0.00073`. Soft routing
has collapsed to fully dynamic and is doing no work, while still paying
for a second opacity column and a second rasterized branch. S4 turns it
off.

## Ranking and the single official evaluation

Ranking statistic: **development-split PSNR** (the 5 dev cameras),
best over the frozen validation schedule.

Recorded per cell: realized point count, routing statistics, PSNR,
SSIM, LPIPS, runtime, peak memory.

After the winner is selected, it is **retrained on the full 35-camera
scene** and then evaluated **exactly once** on the six official held-out
cameras with `scripts/eval_diva360_heldout.py`. The retrain is
deliberate and is disclosed as an extra run: a model selected while
fitting 30 cameras is not the model the protocol scores, and evaluating
the 30-camera model directly would understate it for a reason unrelated
to the axis under test.

Every cell's outcome is preserved, including negative ones.
