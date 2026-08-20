# FROZEN — route-logit-init screen at the 50-frame protocol (2026-08-20)

Status: FROZEN before any cell output. EXPLORATORY. User-proposed
mid-block (2026-08-20); design strengthened by the primary to a
same-commit pair.

## 1. The question

The 2026-08-20 init-order repair revealed every historical run trained
from route logit **4.0** (p_dyn ≈ 0.982) regardless of YAML — the
canonical B0-C deliberately freezes that de-facto value. One-variable
question: **did the defect's forced 4.0 help or hurt, against the
originally intended neutral 0.0 (p_dyn = 0.5)?** Evidence suggesting a
small effect: the ladder B0 router ended at mean p_dyn 0.995 from init
0.982 — but 0.0 lets the router LEARN the split from neutrality, which
the family has never actually run. Worth a ~5.4 slot-h dgx screen, not
an untested ~12 H100-hour 300-frame arm.

## 2. Cells (both fresh at the same commit, seed 0, dgx, admitted V100 image)

| arm | config | materialized init |
|---|---|---|
| **R00** | `configs/n3v/b1_stg_matched_cut_roasted_beef.yaml` UNCHANGED — its `route_logit_init: 0.0` line is effective for the FIRST time | 0.0 |
| **R40** | `configs/n3v/route_init_screen_40.yaml` (verified one-variable diff: 0.0 → 4.0) | 4.0 |

50-frame STG-matched protocol: frames 0–49, 1352×1014, cam00 held out,
batch 2, 6,000 iterations, 600k cap, NO reserved parity — directly
comparable to experiments 181/194. Endpoint: `--val` (pooled+clamped
PSNR, SSIM, LPIPS) on `chkpnt6000`. Scheduling: R00 on the currently
free dgx slot; R40 queues behind the B1-D cells. Hopper untouched.

## 3. Frozen decision rules

1. **Replicate check first**: R40 vs experiment 181/194 (same seed, same
   effective init, different commit). If |Δ pooled PSNR| > 0.1 dB, the
   "training-path-inert code changes" assumption is VIOLATED — flag it
   loudly (it also underpins the B1-D comparator reuse), and the
   0.0-vs-4.0 comparison rests on the same-commit pair only.
2. **The screen**: Δ = R00 − R40 (same commit, same seed). A 300-frame
   0.0 arm is justified only if Δ ≥ +0.15 dB pooled (above the
   0.09–0.17 replicate spread and enough to warrant ~12 H100-hours).
   |Δ| < 0.15 → the de-facto 4.0 stands for the canonical family and
   the question closes at screen tier. Δ ≤ −0.15 → the defect's 4.0
   actively helped; record as a (fortunate) historical accident.
3. One seed per arm; if Δ lands in [0.10, 0.15), a second seed pair is
   the only authorized follow-up. No tuning of intermediate values.

---

## RESULT (2026-08-20, append-only) — the defect's 4.0 actively helped; the question closes at screen tier

Cells: R00 = experiment 215 (val 220), R40 = experiment 216 (val 222),
both commit `5514c66`, seed 0, dgx, admitted V100 image. Endpoint
`--val` pooled+clamped on `chkpnt6000`:

| arm | PSNR | SSIM | LPIPS | points | router end state |
|---|---:|---:|---:|---:|---|
| R40 (4.0) | **33.4871** | 0.95928 | 0.08232 | 599,384 | p_dyn 0.995, 0.59% uncertain |
| R00 (0.0) | **32.9904** | 0.95857 | 0.08273 | 598,990 | **p_dyn 0.916, 24.7% uncertain** |

**Rule 1 (replicate check): PASS, |Δ| = 0.018 dB.** R40 vs experiment
194 (33.5050; the same seed and effective init across ~10 intervening
commits) agree to 0.018 dB pooled+clamped — the training-path-inert
assumption behind the B1-D comparator reuse is now empirically
validated, not just diff-argued.

**Rule 2 (the screen): Δ = R00 − R40 = −0.4967 dB ≤ −0.15.** The
init-order defect's forced 4.0 was a FORTUNATE historical accident:
starting the router at neutral 0.5 leaves ~25% of primitives soft-mixed
at 6,000 iterations and costs ~0.5 dB, with SSIM and LPIPS agreeing in
direction. No 300-frame 0.0 arm is authorized. The canonical family's
explicit `route_logit_init: 4.0` now rests on measurement.

Labels: numbers verified from primary artifacts (validation.json /
summary.json pulled from the run dirs); the "soft-mixed router causes
the deficit" attribution is INFERENCE from the routing histograms (a
per-region decomposition was not run). Claims consumed:
`route_init_screen_r{00,40}` r0 (exps 215/216),
`route_init_r{00,40}_val` r0 (exps 220/222).
