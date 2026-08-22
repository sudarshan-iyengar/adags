# B0-C — canonical 300-frame substrate: RESULTS (2026-08-21)

EXPLORATORY tier. Protocol, schedule derivation and endpoint were frozen
before launch in [[b0c-canonical-300f-2026-08-20]]; nothing there moved.
Training: experiment **208** (`b0c_canonical_crb300` r0, commit
`c080818`, pool `hopper`, NVIDIA H100 PCIe, image `apollo-h100-88ee245`
digest `0d5771…`, seed 0). COMPLETED 36,000/36,000; no stop rule fired.

## 1. The frozen primary endpoint (chkpnt36000, experiment 228)

| quantity (300 held-out cam00 frames, 1352×1014, pooled over clamped renders) | value |
|---|---:|
| **PSNR** | **33.2506** |
| SSIM | 0.95349 |
| LPIPS (alex, torchmetrics normalize=True) | 0.08979 |
| primitives | 599,780 (cap saturated; already 599,573 by iteration 6,000) |

## 2. Convergence curve (descriptive; `--val` pooled+clamped at each saved checkpoint; cam00 selected nothing)

| iter | presentations/unit | PSNR | SSIM | LPIPS | exp |
|---:|---:|---:|---:|---:|---|
| 6,000 | 2.1 | 32.902 | 0.9538 | 0.0991 | 217 |
| 12,000 | 4.2 | **33.508** | 0.9552 | 0.0924 | 223 |
| 18,000 | 6.3 | 33.493 | 0.9545 | 0.0907 | 224 |
| 24,000 | 8.4 | 33.403 | 0.9540 | 0.0899 | 226 |
| 30,000 | 10.5 | 33.171 | 0.9534 | 0.0909 | 227 |
| **36,000** | **12.6** | **33.251** | 0.9535 | 0.0898 | 228 |

Training-time metric agrees: `best_val_iter = 12000`. **The capped model
peaks at ~12k (4.2 presentations/unit) and then REGRESSES ~0.34 dB while
densification churns at the 600k cap; the 30k→36k settle phase recovers
+0.08.** LPIPS, unlike PSNR, improves monotonically to 18k and then
holds — the churn costs PSNR more than perceptual quality. This is a
genuine schedule finding for the capped family: at 300 frames the
600k-capped model does NOT benefit from exposure past ~4 presentations
per unit; the frozen endpoint is reported as frozen, with the peak
disclosed as descriptive.

The 6k row doubles as the answer to "what would the historical 6k/300f
protocol have scored at full raster": **32.90** — against the ~34.0–34.4
the same family recorded at 676×507, i.e. the quarter-raster protocol
inflated the family's numbers by roughly 1.1–1.5 dB.

## 3. Slices and per-frame profile (8-bit saved-render basis; `scripts/slice_metrics.py`)

| slice | pooled PSNR |
|---|---:|
| 0–49 | 33.149 |
| 50–99 | 33.201 |
| 100–149 | 33.291 |
| 150–199 | 33.122 |
| 200–249 | 33.167 |
| 250–299 | **32.955** |

Per-frame: min **32.718**, median 33.160, max 33.452, worst-decile mean
32.862. **Total per-frame spread is 0.73 dB — the model is uniformly
stable across all 300 frames; no catastrophic frame exists.** The worst
decile clusters at frames 249–270 and 297–299 (plus a small 39–51
group): mild late-sequence degradation (~0.2–0.3 dB in the last slice),
not instability.

The 50-frame specialist comparison on the SAME frames: the dedicated
frames-0–49 model reads 33.505 ([[stg-n3v-protocol-parity-2026-08-19]]
Appendix C) vs the single 300-frame model's 33.149 on that slice — the
specialist advantage is **0.36 dB** at matched raster and convention.

## 4. Resource ledger

| item | projected | measured |
|---|---:|---:|
| wall time | ~8 h | **15 h 51 m** (2026-08-20 15:48 → 08-21 07:40 UTC) |
| slot-hours | 9 | ≈ 15.9 |
| rate | 0.8 s/it | 1.08 s/it pre-densification, rising with point count |

Deviation recorded, not smoothed: rasterization cost grows with splat
count and the H100 PCIe advantage over V100 is ~1.5× (bandwidth-bound),
not the ~2× projected. Peak GPU memory was not instrumented by the
trainer; batch-2 V100 precedent (~16 GB at 600k/full raster) bounds it
loosely. Checkpoint-curve evals 217/223/224/226/227/228 ≈ 3 slot-h
additional.

## 5. Verdict on decision item 1 (evidence-labelled)

* **Stable: YES** (verified from primary artifacts: 0.73 dB per-frame
  spread, no NaN/crash, monotone loss curve, no catastrophic frames).
* **Competitive: qualified yes.** 33.25 pooled+clamped across ALL 300
  frames from ONE model, where STG's published 33.52 for this scene is
  six independently trained 50-frame models in series — representation,
  initialization and schedule unmatched, LPIPS convention different.
  The honest gap statement: ADAGS-single-model trails the
  six-specialist STG number by 0.27 dB while covering 6× the temporal
  span per model; its own 50-frame specialist sits at parity (33.505 vs
  33.52). No SOTA claim.
* **The actionable finding is the capacity ceiling**, not exposure: the
  capped model saturates by ~12k and cap churn then erodes PSNR. The
  B0-C-UNCAP companion (experiment 214, running) measures exactly this;
  its section is appended when terminal.

## 6. Bookkeeping

Claims consumed: `b0c_canonical_crb300` r0 (exp 208);
`b0c_canonical_eval{6k,12k,18k,24k,30k,36k}` r0 (exps
217/223/224/226/227/228). Local artifacts:
`data/synthetic/b0c_eval/` (renders, gt, `b0c_36k_slices.json`,
gitignored). The uncap arm: `b0c_uncap_crb300` r0 = exp 214 (running);
`b0c_uncap_eval6k` r0 consumed by a hung submission that never reached
the master (preserved), r1 = exp 221; `b0c_uncap_eval12k` r0 = exp 225;
18k/24k evals submitted as checkpoints landed.

---

## APPENDIX B (2026-08-22, append-only) — B0-C-UNCAP terminal, and a correction to §2's churn attribution

Experiment **214** (`b0c_uncap_crb300` r0, commit `d22e8e2`, hopper,
seed 0) COMPLETED 36,000/36,000; wall **38 h 10 m** (2026-08-20 17:19 →
08-22 07:30 UTC) — 2.4× the capped arm, rasterization scaling with
splat count. Final primitives **2,052,105** (3.42× the cap; growth
1.08M → 1.42M → 1.65M → 1.85M → 2.05M, stopping at the 30k densify
boundary).

Curve (`--val` pooled+clamped; evals 221/225 and the rolling
18k/24k/30k/36k cells; endpoint = exp 232):

| iter | PSNR | SSIM | LPIPS | points |
|---:|---:|---:|---:|---:|
| 6,000 | 33.038 | 0.9548 | 0.0922 | 1,078,395 |
| 12,000 | **33.614** | 0.9578 | 0.0847 | 1,424,077 |
| 18,000 | 33.516 | 0.9572 | 0.0829 | 1,649,295 |
| 24,000 | 33.435 | 0.9565 | 0.0828 | 1,851,495 |
| 30,000 | 33.191 | 0.9557 | 0.0835 | 2,052,105 |
| **36,000 (frozen endpoint)** | **33.074** | 0.9559 | **0.0829** | 2,052,105 |

`best_val_iter = 12000` in this arm too.

**Frozen-endpoint comparison (the one-variable capacity answer):** the
capped arm WINS endpoint PSNR (33.251 vs 33.074) while the uncapped arm
wins SSIM (+0.0024) and LPIPS (−0.0069, ~7.6% relative) at 3.42×
capacity and 2.4× compute. At the descriptive 12k peaks: 33.508 vs
33.614 (+0.106 for capacity) with the LPIPS gap already open.

**CORRECTION to §2's attribution (append-only):** §2 attributed the
post-12k PSNR decline to churn "at the 600k cap". The uncapped arm
reproduces the SAME peak-at-12k-then-decline shape with no cap, and —
stronger — continues losing PSNR (33.19 → 33.07) through the 30k→36k
settle phase where its point count is FROZEN, while SSIM/LPIPS hold or
improve. So the decline is a property of continued optimization under
this family's schedule — densification churn plus late train-view
overfit that is PSNR-specific — not of the capacity cap. The capped
arm's +0.08 settle recovery now reads as the cap LIMITING overfit
capacity rather than churn ending.

**Schedule consequence, for any future 300-frame arm (a NEW spec, not a
retro-fit):** both capacity regimes peak at ~12,000 iterations
(~4.2 presentations/unit); the exposure-matched 36k schedule overshoots
for PSNR in both. An 18k schedule (densify to 12k + settle) would
capture the peak at ~45% of the cost. Disclosed: this observation
derives from held-out-view convergence curves of these two runs; using
it to design a future experiment's schedule is design-from-prior-
evidence and must be stated in that experiment's freeze.

Claims: `b0c_uncap_eval{18k,24k,30k,36k}` r0 consumed (the ledger holds
the exp ids; endpoint = 232).
