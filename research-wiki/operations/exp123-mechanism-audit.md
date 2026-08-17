# Mechanism audit of experiment 123 (2026-08-17)

EXPLORATORY. `scripts/audit_mechanism.py`, experiment 127, on experiment
123's iteration-15000 checkpoint (416,599 points, 141 frames, 846
held-out units). No training state changed. Raw per-point, per-frame and
per-unit rows preserved in the run directory; every aggregate below is
produced by the script's deterministic `reduce_rows()` from those rows.

Companion: experiment 126, the matched training-view evaluation.

## 1. The train/held-out gap — measured on matched sets

Six TRAINING cameras (`cam01, cam09, cam23, cam32, cam39, cam47`) x 141
frames = **846 units**, exactly matching the 846 official held-out units,
same evaluator, same conventions. The official six are excluded from
that scene by construction and cannot leak in.

| | PSNR | SSIM | VGG LPIPS |
|---|---:|---:|---:|
| training views | **26.4871** | 0.94791 | 0.08270 |
| held-out views | 23.1678 | 0.92247 | 0.10507 |
| **gap** | **+3.319 dB** | +0.0254 | -0.0224 |

**This reframes the residual.** Experiment 123's TRAINING-view score
(26.487) is ABOVE the per-frame oracle's HELD-OUT score (25.719 mean /
25.466 excluding frame 0). The model demonstrably represents this scene
to better than 26 dB where it has supervision. So the ~2.3 dB held-out
residual is not a capacity ceiling and not a fitting failure — it is
GENERALIZATION TO UNSEEN VIEWPOINTS.

Stated carefully: the oracle's own train/held-out gap is NOT measured, so
this does not by itself prove the temporal representation generalizes
worse than a per-frame fit. It proves only that fitting capacity is not
the binding constraint. Used for interpretation only; nothing was tuned
on it.

## 2. Routing is fully collapsed

```
mean_dynamic_prob   0.99962      percent_near_dynamic  100.0%
percent_near_static 0.0%         percent_uncertain       0.0%
```

Not one point of 416,599 sits in the uncertain band. The soft router is
carrying no information, consistent with exps 84 (99.906%) and 104
(99.912%) and with the sweep's S4 finding that disabling it nonetheless
COST 0.600 dB. A collapsed router is not free to delete.

## 3. Temporal support has collapsed — the most striking finding

| per point | p1 | p50 | mean | p95 | max |
|---|---:|---:|---:|---:|---:|
| temporal centre | -0.394 | 0.719 | 1.648 | 4.613 | **11.225** |
| temporal scale | 4e-05 | **0.0073** | 0.165 | 0.257 | 555.6 |

The scene spans `time_duration [0, 4.7]`. Temporal centres reach
**11.2**, far outside it. The MEDIAN temporal scale is **0.0073 s** —
about a fifth of one 30-FPS frame interval.

Consequently, active points per frame (temporal marginal >= 0.5):

```
mean 25,658   median 23,792   min 16,514   max 47,214
of 416,599 total  ->  6.2% active at a typical frame
```

**The 4D representation has degenerated toward a per-instant point
cloud.** 416,599 stored primitives, but only ~25.7k are meaningfully
present at any frame. For comparison, the per-frame oracle used
37k-72k points for a SINGLE frame and scored higher. So the dynamic
model is spreading its bank thinly across time and ends up with fewer
effective primitives per instant than the oracle has, while also having
to make them move.

## 4. LoRA rank IS saturated

| | shape | participation ratio | max rank | energy spread |
|---|---|---:|---:|---|
| shared basis | [8, 32, 3] | **7.37** | 8 | 0.262 / 0.196 / 0.147 / 0.121 / 0.103 / 0.077 / 0.058 / 0.036 |
| per-point coefficients | [416599, 8] | 5.98 | 8 | 0.445 / 0.221 / 0.131 / 0.077 / 0.056 / 0.048 / 0.014 / 0.009 |

The participation ratio `(sum s)^2 / sum s^2` is an effective rank
needing no threshold: it equals r when r singular values are equal.

**The shared basis is using 7.37 of 8 available dimensions (92%)**, with
energy spread remarkably evenly across all eight. That is rank
saturation on the shared basis. The coefficients are more concentrated
(5.98 of 8, 75%) but still far from collapsed.

Under the frozen decision rules, "rank saturation" points at rank 8 -> 16
with anchors fixed — CONDITIONAL on the translation oracle showing a
high gain, which has not yet been run.

## 5. Where the error actually is

| region | mean L1 | ratio to overall |
|---|---:|---:|
| whole image | 0.01884 | 1.0x |
| foreground | 0.09810 | **5.2x** |
| mask boundary band (3 px) | 0.23664 | **12.6x** |

Foreground is only 13.7% of pixels. **Error is overwhelmingly a boundary
phenomenon** — the 3-pixel band around the matte transition carries over
twelve times the mean error.

Per-unit PSNR spread: min 13.32, p5 19.01, p50 22.83, p95 27.71, max
29.50.

## 6. Error tracks SPEED, and the endpoints are worst

Frame-level correlations (error aggregated over held-out cameras per
frame, N = 141):

```
L1   vs mean speed         +0.647
PSNR vs mean speed         -0.601
L1   vs active points      +0.467
L1   vs mean strain        -0.242
PSNR vs mean strain        +0.257
point-level strain vs speed +0.330  (N = 20,000)
```

**Error rises with motion.** That is the strongest single correlation in
the audit and it survives in both directions (L1 up, PSNR down).

By temporal bin:

| bin | t range | mean PSNR | mean L1 |
|---|---|---:|---:|
| 0 | 0.00-0.93 | **21.38** | 0.02583 |
| 1 | 0.97-1.87 | 23.92 | 0.01631 |
| 2 | 1.90-2.80 | 24.15 | 0.01524 |
| 3 | 2.83-3.73 | 24.52 | 0.01443 |
| 4 | 3.77-4.67 | **21.93** | 0.02214 |

The first and last bins are **2.5-3.1 dB worse** than the middle. The
temporal model degrades at the sequence ENDPOINTS, which is what a knot
or support problem at the boundary of the anchor grid would look like.

## 7. Strain — high, but its correlation goes the WRONG way, and the measure has a caveat

k-NN strain (median relative neighbour-distance change, neighbour set
fixed at the temporal-centre configuration, scale-free so rigid
translation contributes zero):

```
p1 1.50   p50 22.74   mean 34.25   p95 104.5   max 747.3   (N = 20,000)
```

Those are enormous deformations. **But strain correlates NEGATIVELY with
error (-0.242)**, i.e. high-strain frames are slightly BETTER, which is
the opposite of the "local strain regularizer" hypothesis.

**Caveat that must be read with those numbers.** The strain statistic is
computed over all sampled points at all frames, INCLUDING points whose
temporal marginal is negligible at that frame. Given section 3 — a
median temporal scale of 0.0073 s and only 6.2% of points active per
frame — most contributions come from points that are effectively absent
and whose positions are LoRA extrapolations far outside their support.
So the absolute strain magnitude is not trustworthy as "deformation the
renderer sees", and the negative correlation may be an artifact of that.
A presence-weighted strain is the correct measure and was not
implemented. Recorded as a limitation, not reported as a finding.

## What this audit does and does not select

It does NOT by itself select the next change: the frozen decision rules
all condition on a GAIN from the translation oracle, which has not run.

What it establishes:
* fitting capacity is not the binding constraint (section 1);
* the router is inert but not removable (section 2);
* the temporal representation has collapsed toward per-instant clouds
  (section 3) — this is not in the decision list and may matter more
  than anything in it;
* the shared basis IS rank-saturated (section 4);
* error is a boundary and motion phenomenon (sections 5, 6);
* the sequence endpoints are much worse (section 6);
* the strain hypothesis is NOT supported by this measurement, and the
  measurement itself is flawed in a stated way (section 7).
