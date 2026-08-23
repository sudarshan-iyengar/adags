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
