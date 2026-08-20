# B1-D donor-mask variant — REJECTED by the frozen gate, on both seeds (2026-08-20)

EXPLORATORY. Lane 2 of the 2026-08-20 block. Frozen change (one
variable): at each packet-birth event, donors are eligible only when
their motion-model centre projects inside the current training view's
dynamic mask (`packet_birth_dynamic_mask_donors`, commit `78b457c`;
14 CPU tests; flag-off bit-identity tested). Everything else —
schedule, seeds, cadence, fraction, ranking-among-eligible, sites,
backprojection, sigma, cap, evaluator — identical to B1.

## 1. Comparator validity (verified, not assumed)

Existing B1 artifacts (experiments 197/200) reused as comparators after
verifying `git diff 22b2dd6..HEAD` and `a798949..22b2dd6` over
`main.py scene/ gaussian_renderer/ utils/ arguments/ elgs/
depth_visibility/` are **empty** — the training path is byte-identical
across the comparator commits and the B1-D commit modulo the flag.

## 2. Cells and machinery health

| cell | exp | best_val pooled | points | eligibility accounting |
|---|---|---:|---:|---|
| B1-D s0 | 210 (val 218) | 33.9483 | 599,434 | 7 events, ~264k eligible/event, **0 shortfalls, 0 skipped**, 19,565 relocated, 642 packets |
| B1-D s1 | 211 (val 219) | 33.8333 | 599,462 | same shape (1,856,927 eligible total, 0 shortfalls) |

The mask rule never starved and never silently fell back — the
machinery worked; the mechanism failed.

## 3. Metrics (val protocol: pooled+clamped; event regions on the frozen masks, 8-bit basis)

B1-D − B1, paired by seed:

| quantity | s0 | s1 | paired mean |
|---|---:|---:|---:|
| global PSNR | **+0.145** | **−0.265** | −0.060 |
| SSIM | −0.0006 | +0.0003 | — |
| LPIPS (alex, norm) | −0.0007 | +0.0002 | — |
| complement | +0.181 | −0.324 | −0.071 |
| event A | −0.054 | −0.364 | −0.209 |
| event B | −0.082 | +0.040 | −0.021 |
| event C | −0.172 | −0.332 | −0.252 |
| **event union** | **−0.145** | **−0.313** | **−0.229** |

Absolute event-union values: B0 31.953/32.066, B1 32.030/32.411,
B1-D 31.885/32.099.

## 4. The frozen gate, applied

| condition | outcome |
|---|---|
| mean global B1-D−B1 ≥ +0.30 | FAIL (−0.060) |
| neither seed negative globally | FAIL (s1 −0.265) |
| complement/static loss ≤ 0.10 dB | FAIL (s1 −0.324) |
| event-union change ≥ 0 on both seeds | **FAIL on BOTH** (−0.145, −0.313) |
| point-neutral budget preserved | pass |

**VERDICT: retain B1, reject B1-D.** No post-hoc tuning of the mask
rule, donor fraction, cadence, or thresholds (frozen prohibition).

## 5. What the negative teaches (labelled)

* **Verified from primary artifacts:** B1-D removes B1's replicated
  event-region benefit on both seeds while global PSNR stays inside the
  ±0.27 seed spread; eligibility was never binding (mask admits ~46% of
  rows; requested 0.5% « eligible).
* **Inference (consistent across both seeds, not independently
  verified):** unrestricted B1 harvests bottom-utility donors largely
  from OUTSIDE the dynamic regions, so relocation acts as a net
  capacity transfer INTO events; restricting donors to the mask makes
  relocation cannibalize the regions it serves. If a future donor rule
  is tried, the pre-registered candidate should be the INVERSE
  (donors eligible only OUTSIDE the dynamic mask) — as a new frozen
  spec, not a tuning of this one.
* **Unresolved:** whether the event-union effect of either variant
  survives at 300 frames; per the block boundary, B1-D is not scaled.

Claims consumed: `ladder_b1d_crb_s{0,1}` r0 (exps 210/211),
`ladder_b1d_s{0,1}_val` r0 (exps 218/219). Local artifacts:
`data/synthetic/lrv3_results/local/b1ds{0,1}/`,
`data/synthetic/ladder_eval/b1ds{0,1}_*`.
