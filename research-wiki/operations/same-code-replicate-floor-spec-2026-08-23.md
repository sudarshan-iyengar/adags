# SPEC (FROZEN) — the same-code replicate floor at the 50-frame 6k
# protocol (2026-08-23, block 2)

EXPLORATORY, `evidence_bearing: false`. Frozen BEFORE any output exists.
Motivated by [[b1f-flow-postmortem-2026-08-23]] §2 and named as the
single next scientific action in
[[paper-path-decision-2026-08-23b]] §4.

## 1. The question

The flow post-mortem measured that two arms differing by two config
lines, executing **identical numerical code** until iteration 1000,
already separate by **0.089–0.236 relative RMS training loss** in the
pre-intervention window — indistinguishable from their post-intervention
separation, with `densify_from_iter: 500` amplifying the divergence
1400×.

That is measured on TRAINING LOSS. **The quantity every comparison in
this family actually uses is the HELD-OUT ENDPOINT metric, and its
same-code replicate spread has never been measured at this protocol.**
Two trajectories can separate and still converge to similar endpoints.

**Question: what is the run-to-run spread of the held-out endpoint
metrics — pooled+clamped PSNR and the frozen event-union PSNR — across
identical cells at a fixed seed?**

This is upstream of every remaining N3V mechanism question. If the
replicate floor is comparable to the measured **0.341 dB** event-union
seed spread, then no two-seed 50-frame comparison in this family can
resolve the effects the ladder was built to measure.

## 2. Design

**Three cells, identical in every respect**: `configs/n3v/ladder_b1_crb.yaml`
(plain B1, the ladder's own comparator), `--seed 0` threaded to the
trainer, pool `hopper`, admitted H100 image `sha256:0d577168…`, 6,000
iterations, frames 0-49, `cam00` sealed, `route_logit_init: 4.0`,
`elgs_reserved_parity: true`, 600k cap, one commit.

They differ **only** in their cell name and retry index. That is the
whole design: the measurement is the spread among them.

Cells `repl_floor_b1_a`, `repl_floor_b1_b`, `repl_floor_b1_c`.

**Experiment 237 is NOT part of the measurement.** It ran at commit
`789595a` and these run at a later commit, so it is a cross-commit
observation reported as a bonus, never as a fourth replicate. (The
intervening commits — the roster repair, the closure repair, the
pre-filter diagnostic — are not expected to touch the plain-B1 training
path, but "not expected" is not "verified equal", and the measurement
does not need it.)

## 3. Endpoints, fixed in advance

Primary: **spread (max − min) across the three cells** of

1. pooled+clamped held-out PSNR at `chkpnt6000` (`main.py --val`);
2. the frozen event-union PSNR from `scripts/event_ray_metrics.py` on
   `configs/n3v/ladder_event_masks_crb0_49.json`, 8-bit saved-render
   basis — the same convention and masks the flow screen used.

Secondary, descriptive: SSIM, LPIPS, realized point count, rows
relocated.

## 4. Interpretation — FIXED BEFORE ANY OUTPUT EXISTS

Let `R` be the event-union replicate spread and recall the measured
event-union **seed** spread of **0.341 dB**.

| outcome | reading | consequence |
|---|---|---|
| **`R` ≥ 0.341** | the replicate floor is at least as large as the seed spread | **The 50-frame two-seed protocol cannot resolve any effect the ladder measured.** Every recorded ladder delta (B1 +0.077/+0.345, B1-D, B1-F, B1-X) is inside the floor. The N3V utility lane needs a different protocol, more replicates, or a different endpoint before any utility claim. No recorded number is retracted; what changes is what they can be said to resolve. |
| **0.10 ≤ `R` < 0.341** | replicate and seed variation are comparable in order | Two-seed comparisons retain only the weak power already claimed, and **every future comparison must report `R` alongside its delta.** The ±0.28 dB figure the ladder era cited is a replicate floor, not a seed spread. |
| **`R` < 0.10** | the substrate's endpoint is reproducible despite chaotic trajectories | Trajectory divergence does **not** propagate to the endpoint. The 0.341 dB seed spread is genuine seed variation, two-seed comparisons keep their stated power, and the flow post-mortem's channel concern is bounded to training-loss dynamics. |

**No threshold here may be moved after the numbers are read.** In
particular a result in the top row does not license re-running at a
different protocol to obtain a smaller floor.

## 5. Cost and termination

Three cells at ~2.5 slot-h each (the measured contended-node cost, not
the 1.9 h that was underestimated last block) ≈ **7.5 slot-h**, plus
three sub-0.5 h evaluation cells. The measurement ends when all three are
terminal and both endpoints are computed. **No further replicate cells
are authorized by this spec under any outcome.**

## 6. What this does NOT do

It does not compare mechanisms, does not reopen B1-D, B1-F or the
deferred 300-frame comparison, and does not bear on the LRV3/LRV4
fixture lanes, whose endpoints are computed on a different protocol.

---

## RESULT (2026-08-23, append-only) — the TOP ROW FIRES

Cells **261 / 262 / 263** (`repl_floor_b1_{a,b,c}` r0, pool `hopper`,
commit `eb293a2`, image `sha256:0d577168…`), all `STATE_COMPLETED`.
Evaluations **265 / 266 / 264** (`_a_val`/`_b_val`/`_c_val`, r0),
`main.py --val` at `chkpnt6000`, all `STATE_COMPLETED`.

**Provenance verified identical across all three** before any metric was
read: `commit`, `image_ref`, `pool`, `seed`, `config_canonical_hash`
(`b5237330f801…`) and `archive_sha256` all compare equal. The three ran
on three different hosts.

### Endpoint 1 — pooled+clamped held-out PSNR (`main.py --val`)

| | a (261) | b (262) | c (263) | **spread** |
|---|---:|---:|---:|---:|
| PSNR | 33.157368 | 33.185623 | 33.559082 | **0.4017** |
| SSIM | 0.958169 | 0.958161 | 0.957208 | 0.000961 |
| LPIPS | 0.082399 | 0.082766 | 0.083123 | 0.000725 |
| points | 599,486 | 599,454 | 599,454 | 32 |

### Endpoint 2 — frozen event union (`scripts/event_ray_metrics.py`)

Scored on `configs/n3v/ladder_event_masks_crb0_49.json`, 8-bit
saved-render basis — the same convention and masks the flow screen used.

| region | a | b | c | **spread** |
|---|---:|---:|---:|---:|
| **all_events_union** | 31.4059 | 31.8043 | 31.9004 | **0.4945** |
| A_hand_press_reveal | 37.6713 | 37.0244 | 38.4369 | 1.4126 |
| B_knife_stroke_reveal | 36.2580 | 35.9051 | 36.3120 | 0.4069 |
| C_tongs_band_reveal | 26.7347 | 27.3877 | 27.2178 | 0.6530 |
| complement | 32.8229 | 32.9124 | 33.3037 | 0.4808 |
| whole_frame | 32.8172 | 32.9081 | 33.2981 | 0.4809 |

### The frozen §4 table, applied

**`R` = 0.4945 dB ≥ 0.341 dB — the TOP ROW FIRES**, at **1.45×** the
measured event-union seed spread.

Its stated consequence, unchanged from the frozen text: *the 50-frame
two-seed protocol cannot resolve any effect the ladder measured.*
Verified exhaustively — **every recorded event-union delta is inside the
floor**:

| recorded delta | value | inside |
|---|---:|---|
| B1 vs B0, seeds 0 / 1 | +0.077 / +0.345 | yes / yes |
| B1-F − B1, seeds 0 / 1 | −0.0881 / +0.1249 | yes / yes |
| B1-X − B1, seeds 0 / 1 | −0.1863 / +0.4136 | yes / yes |
| B1-F − B1-X, seeds 0 / 1 | +0.0982 / −0.2887 | yes / yes |
| B1-D, seeds 0 / 1 | −0.145 / −0.313 | yes / yes |

**No recorded number is retracted.** What changes is what they can be
said to resolve: nothing. The B1 event-region gain (+0.077/+0.345, both
seeds, previously read as "consistent"), the B1-D rejection, and the
flow screen's gate arithmetic are all inside the same-code noise of the
instrument that produced them.

**A stronger reading than the spec anticipated.** The single-region
spread reaches **1.41 dB** on region A, so per-region readings are
noisier still. And the replicate floor (0.4945) *exceeds* the seed spread
(0.341), which means seed-to-seed variation is not the dominant term —
**run-to-run variation at a fixed seed is**, exactly as the flow
post-mortem's trajectory measurement predicted.

### Consequence, applied

**N3V utility scaling is HALTED.** The B0/B1/B2 ladder, the event-union
deltas and the deferred 300-frame comparison are gated behind a protocol
redesign. Per this spec's §6 the LRV3/LRV4 fixture lanes are **not**
gated by this result: they use a different fixture, frame count, camera
set and evaluator, with their own measured same-arm spread of 0.09-0.17
dB against effects of 1.05-2.47 dB.

**No further replicate cells are authorized**, and this result does not
license re-running at a different protocol to obtain a smaller floor.

Measured cost: 3 training cells ≈ 6.7 slot-h (they ran on three separate
hosts, so without the contention the last block measured), plus 3 eval
cells ≈ 0.6 slot-h.
