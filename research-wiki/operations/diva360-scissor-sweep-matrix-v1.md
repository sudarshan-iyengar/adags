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

---

## RESULTS (appended after all six cells completed; matrix above unchanged)

Experiments 92-97, commit `4aded51`, dgx/V100, all six concurrent on the
same 6-slot agent, same seed, image, dataset, validation schedule and
hardware class. Ranking statistic as frozen: development-split PSNR.

| cell | dev PSNR | delta vs S0 | dev SSIM | best iter | points at best | final points | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| **S3 cap 300k** | **23.3310** | **+0.909** | 0.93333 | 6000 | 299,829 | 299,829 | 108.9 min |
| S2 stop densify 4000 | 22.5366 | +0.115 | 0.93108 | 6000 | 205,579 | 205,579 | 96.1 min |
| S0 control | 22.4217 | — | 0.92653 | 6000 | 534,873 | 534,873 | 126.7 min |
| S4 routing off | 21.8221 | -0.600 | 0.92379 | 5000 | 448,317 | 599,841 | 139.7 min |
| S5 init 200k + stop 4000 | 18.9065 | -3.515 | 0.91397 | 5000 | 321,758 | 321,758 | 142.3 min |
| S1 init 200k | 18.4459 | -3.976 | 0.91434 | 5000 | 444,646 | 577,066 | — |

**Winner: S3.** It is better AND cheaper than the control — +0.909 dB
while running 14% faster and holding 235,044 fewer points.

### The capacity axis is confirmed

Both mechanisms on that axis improved on the control, and the ordering
is the one experiment 84's curve predicted when validation fell between
338k and 507k points. The control itself ran to 534,873 points and was
beaten by a model with 299,829. Capping capacity is not a compromise
here; it is the improvement.

### The initialization axis is STRONGLY NEGATIVE, and that is informative

S1 lost **3.976 dB** alone. S5 lost 3.515 and tracked S1 rather than its
beneficial S2 component, so the initialization change dominates anything
the schedule change contributes.

This does NOT weaken the diagnosis that the synthesized initialization is
bad — it sharpens it. The matrix disclosed in advance that S1 is "more of
the same coarse prior, not a better one", and the result says filling a
wrong volume more densely is actively harmful rather than merely useless.
The defect is the VOLUME (+/-6.5 world units against about +/-1.2 of
content), not the seed count. S1 also ended at 577,066 points and
`percent_uncertain 0.3603` — an order of magnitude more routing
uncertainty than any other cell — consistent with 200,000 seeds spread
through empty space producing many rows the model cannot classify.

### Removing collapsed routing HURT

S4 lost **0.600 dB** even though the control's router had collapsed to
`mean_dynamic_prob 0.99931` / `percent_uncertain 0.0099`. With soft
routing off, `mean_dynamic_prob` is exactly 1.0 and `percent_uncertain`
exactly 0.0, the cell hit the 600,000-point cap, and it became the
second-slowest. A collapsed router is evidently not dead weight that can
simply be deleted. Preserved as a negative result; no follow-up cell was
added.

### What these numbers are not

Development-split SSIM (0.926-0.933) sits ABOVE the benchmark baseline's
official-split 0.907 because the five development cameras interpolate
among training views and are an easier set than the official held-out
six. These values rank configurations and nothing else; they are not
comparable to official-split numbers and are never reported as such.

### Not recorded

**LPIPS per cell** and **peak GPU memory per cell** are absent.
`main.py` logs neither — the training loop computes PSNR and SSIM only
(parity audit M3), and it records no GPU memory high-water mark. Point
count stands in for capacity. Neither gap is filled by estimation.

### Submission failures preserved

Experiments 86-91 were the same six cells at retry 0 and all six ERRORed
in about twenty seconds at scene load, on
`assert False, "Could not recognize scene type!"`. Cause: a PowerShell
quoting bug in the submission loop (`--extra-arg=$c[2]` expanded as the
whole array plus a literal `[2]`), so every cell received a malformed
`--source_path`. No GPU work occurred and six claim indices were
consumed. The matrix was unaffected — the failure was submission
plumbing caught by a fail-closed assertion, not a scientific change.

## THE DEVELOPMENT RANKING DID NOT TRANSFER (the decisive result)

The winner was retrained on the full 35-camera official scene
(experiment 98, config `diva360_scissor_bench30_S3cap300k.yaml`, commit
`cf3b34d`) and evaluated ONCE on the six official held-out cameras
(experiment 100, 846 units, AlexNet v0.1 LPIPS).

| | baseline (exp 84) | S3 winner retrained (exp 98/100) | delta |
|---|---:|---:|---:|
| official PSNR | 21.3705 | **21.1967** | **-0.174** |
| official SSIM | 0.90698 | **0.91024** | +0.0033 |
| official LPIPS `[-1,1]` | 0.14685 | **0.14267** | -0.0042 (better) |
| official LPIPS `[0,1]` | 0.12398 | **0.12116** | -0.0028 (better) |
| points | 507,178 | **299,815** | -207,363 |
| best iteration | 5000 | 6000 | |

**S3 won the development split by +0.909 dB and is 0.174 dB WORSE on the
official split.** The ranking did not transfer.

This is precisely the failure the held-out split exists to catch, and the
discipline caught it. `main.py`'s training loop reads
`transforms_test.json` for its validation, which on the FULL scene IS the
official six — so ranking six configurations "the obvious way" would have
selected on the sealed split and reported a +0.9 dB improvement that does
not exist on it. Ranking on 5 development cameras carved from the 35
TRAINING cameras is what made the non-transfer visible instead of
invisible.

Why it plausibly fails: the 5 development cameras interpolate among the
35 training views, while the official six are a different view set. A
capacity reduction that suppresses floaters visible from nearby
interpolated views does not help the official views, which appear to want
the capacity.

### What survives, and what does not

**Does NOT survive:** "capping capacity at 300k improves scissor quality".
On the official split it does not. The +0.909 dB is a development-split
number and is not a quality claim.

**DOES survive:** the capacity finding as an EFFICIENCY result. The
retrained model reaches -0.174 dB PSNR with **207,363 fewer points**
(41% fewer), BETTER SSIM (+0.0033) and BETTER LPIPS on both conventions,
in 14% less training time. That is a real and useful trade, and all three
perceptual/structural metrics move the right way while only PSNR moves
the wrong way — a pattern consistent with fewer floaters and slightly
less raw fidelity.

**Caveat on precision.** The retrain fits 35 cameras where the sweep
cells fit 30, by design. So the -0.174 dB compares two runs differing in
BOTH the capacity cap and the training-camera count. The direction is
clear; the magnitude is not isolated.

### Distance to parity is unchanged in kind

Neither model is near published scissor parity (~25-26 PSNR, ~0.94 SSIM,
LPIPS 0.08-0.10). Best of the two on each metric: PSNR 21.371,
SSIM 0.9102, LPIPS 0.1212-0.1427. Every metric is short, consistently,
and no configuration in this matrix closes a gap of that size.
