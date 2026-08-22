# FROZEN — B1-F: SEA-RAFT flow as a packet-BIRTH prior (2026-08-23)

Status: **FROZEN before any cell output.** EXPLORATORY,
`evidence_bearing: false`. Schedule ceiling:
[[block-2026-08-23-schedule-amendment]]. Lane-3 asset basis:
[[prior-asset-inventory-2026-08-20]] decision 4 (flow as a BIRTH prior
is the first zero-acquisition prior experiment).

## 1. The question, and why the obvious version of it is not worth running

> Does initializing relocated packet rows with a flow-derived velocity,
> instead of the current zero motion, improve reconstruction over plain
> B1 — and does CORRECT flow separate from WRONG flow?

The obvious flow-birth experiment is flow-gated SITE SELECTION. **It was
rejected on inspection, before any compute.** Birth sites are already
sampled from a residual map that has ALREADY been multiplied by the
dynamic mask (`scene/packet_birth.py`, `_packet_birth`: `residual =
residual * dynamic_mask...` immediately before `sample_residual_sites`).
A flow-magnitude gate on top of that is largely redundant with machinery
the operator already has, and would test "is flow a better dynamic mask"
rather than a birth prior.

What dense flow uniquely adds over a binary dynamic mask is **motion
direction and magnitude**. Relocated rows are currently born motionless:
`_build_packet_target_rows` writes zeros for every motion coefficient
("motion coefficients (poly and LoRA) and the staticness score"). So the
one-variable experiment with genuine information content is **velocity
initialization**, and that is what is frozen here.

## 2. The mechanism (frozen)

At each packet-birth event, for each sampled birth-site pixel `p` in
training camera `c` at timestamp `t`:

1. Read SEA-RAFT forward flow `f(p)` and per-pixel validity `m(p)` for
   camera `c` at the frame of `t` (`forward_t_to_t_plus_1`,
   within-camera, PIXELS).
2. `m(p)` false, or `f(p)` non-finite ⇒ the row **fails closed**: it
   keeps the existing zero-motion initialization. Counted.
3. `p' = p + f(p)`; outside the raster ⇒ fail closed. Counted.
4. Backproject `p` and `p'` through the **same** rendered camera-z
   sampled at `p`, reusing the existing `backproject_camera_z` ray
   machinery ⇒ world points `X0`, `X1`. This is a **constant-depth
   approximation and is disclosed as such**: no render exists at `t+1`
   inside the training loop, so the surface is assumed to stay at the
   same camera-z over one frame.
5. `v = (X1 - X0) / dt_frame`, `dt_frame` = one frame in the trainer's
   time units.
6. Project `v` into the LoRA motion basis. The model computes
   `offset(t) = einsum('nr,nrd->nd', coeff, centered_basis)` with
   `centered_basis = basis(t - t_n) - basis(0)`. For rows born at
   `t_n = t` with probe `dt_p = dt_frame`, `centered_basis` is a
   `(rank, 3)` matrix **identical for every row of the event**, so one
   least-squares solve per event yields every row's coefficients from
   its own target displacement `v · dt_p`.
7. Write those coefficients into `_motion_lora_coeff` in place of zeros.
8. **Magnitude guard, no new constant:** clamp each solved coefficient
   vector's L2 norm to the MEDIAN L2 coefficient norm of the existing
   rows, mirroring the `_median_log_scaling` clamp idiom already used
   for donor scaling at birth. Direction preserved. Counted.

Two flags, both defaulting to current B1 behaviour:
`packet_birth_flow_init` (False ⇒ **bit-identical to B1**, tested) and
`packet_birth_flow_source` ∈ {`correct`, `camera_swapped`}.

## 3. Asset and convention verification (done before freezing, not assumed)

* **Flow exists and is reachable.** `apollo:/apollo/users/sri/
  proj_adags/data/n3v/cut_roasted_beef/flow/` holds **5,980** `.npz`
  files (20 cameras × 299 frame pairs), matching the inventory.
* **The default root finds it.** With `motion_prior_root: ""` (the value
  in `ladder_b1_crb.yaml`), `MotionPriorCache.uses_default_root` is
  True, so `_candidate_roots` yields BOTH `source_path/motion_priors`
  AND `source_path`; `get_track_flow` probes subdirs
  `("track_flows", "flows", "flow", "")`. `<source_path>/flow/
  <image_name>.npz` is therefore found. Verified by reading
  `utils/motion_prior_utils.py`, not by trial.
* **Units are PIXELS.** `normalize_flow_tensor` only reorders axes to
  `(2, H, W)` — it performs no unit change. `resize_flow` rescales
  channel 0 by `new_w/old_w` and channel 1 by `new_h/old_h`, i.e. flow
  remains pixels AT THE TARGET RESOLUTION. Channel 0 = x, channel 1 = y.
  The velocity derivation above is therefore well-posed.
* **`cam00` flow EXISTS in the cache.** The held-out camera's flow files
  are present on disk. The held-out guard in the implementation is
  consequently load-bearing rather than theoretical.

**Disclosed:** at the last training frame the forward flow points to a
frame one step beyond the training window. Its IMAGE is never read and
the held-out dimension is the CAMERA (`cam00`), not time, so this is not
test leakage; it is recorded because the flow field carries information
about a frame outside the declared `time_duration`.

## 4. Arms, pool, and the confound this design refuses to create

Ladder B0/B1/B1-D all trained on **dgx**. Only 1 dgx slot is free this
block; 3 hopper slots are. Comparing a hopper-trained B1-F against the
dgx-recorded plain B1 would confound the flow variable with a
GPU-architecture change, and cross-pool trained-model equivalence is
UNVERIFIED (experiment 207 verified identical CUDA sources and
residuals, which is not the same claim).

**Therefore every arm of this screen trains on the SAME pool, including
a freshly trained plain B1 on that pool.** Six cells:

| arm | flag state | seeds |
|---|---|---|
| B1 (plain, fresh on-pool comparator) | `flow_init: false` | 0, 1 |
| **B1-F** | `flow_init: true`, `flow_source: correct` | 0, 1 |
| **B1-X** (wrong-flow control) | `flow_init: true`, `flow_source: camera_swapped` | 0, 1 |

`camera_swapped` is the project's established wrong-evidence control:
CSVL-VPL Stage 1 measured camera-swapped flow **scoring ABOVE valid
flow** (0.922 vs 0.889). That precedent is exactly why this control is
mandatory and may not be dropped.

Protocol otherwise byte-identical to `ladder_b1_crb.yaml`: frames 0-49,
1352×1014, cam00 sealed, batch 2, **6,000 iterations**, 600k cap,
`route_logit_init: 4.0` explicit, `elgs_reserved_parity: true`,
packet birth interval 500 / fraction 0.005 / sigma_t 1.5 /
window [1000, 4000]. Evaluator: `main.py --val`, pooled+clamped PSNR,
SSIM, LPIPS-alex `normalize=True`; event regions via
`scripts/event_ray_metrics.py` on the frozen
`configs/n3v/ladder_event_masks_crb0_49.json`.

**Free secondary datum:** the fresh on-pool plain-B1 cells are the same
config and seeds as the dgx-recorded experiments 197/200, so their
difference is a direct measurement of the pool/architecture effect at
this protocol. Recorded as an observation, never as a gate.

## 5. Gates, frozen before any output

Primary attribution rule — **B1-F must separate from B1-X.** If correct
flow does not beat wrong flow on the event endpoint, the result is
UNATTRIBUTABLE and the flow birth prior is REJECTED regardless of how
B1-F compares to plain B1.

All of the following must hold to promote:

1. event-ray union delta `B1-F − B1` **≥ 0 on BOTH seeds**;
2. paired mean event effect `B1-F − B1` **> 0**;
3. `B1-F` **>** `B1-X` on the event-ray union endpoint;
4. complement delta paired mean **≥ −0.10 dB** (the existing CCR
   static-region bound);
5. global non-catastrophe: neither seed's global PSNR delta may fall
   below **−0.30 dB** (derived from the measured 50-frame B0 seed spread
   of ±0.28 dB, rounded outward; frozen before output);
6. machinery health: no missing-flow fallback to the asset-error path,
   no invalid-unit pathology, and the per-event funnel must show flow
   actually applied — `flow_sites_valid > 0` at every event.

Zero valid flow sites at any event, or an all-clamped event, is a
MACHINERY finding and is reported as such, never as evidence about
flow.

**No promotion to 300 frames in this block** under any outcome
(amendment; and decision-5's 300-frame pair is separately deferred).

## 6. Expected effect size, stated in advance

Flow here changes only the INITIAL motion coefficients of the ~3.3% of
rows that relocate, which then train for thousands of further
iterations. An initialization effect may wash out entirely. Plain B1 is
already globally neutral (+0.011 paired mean) with a small event gain
(+0.077/+0.345). **A null is a likely and legitimate outcome**, and the
B1-X control is what makes a null informative: `B1-F ≈ B1-X ≈ B1` would
close the BIRTH-prior role for zero-acquisition flow — the last live
zero-acquisition prior experiment — rather than leaving it open.

---

## APPENDIX A (append-only) — asset direction VERIFIED empirically, and the measured motion scale

Nothing in the code can verify that the SEA-RAFT assets are FORWARD rather
than BACKWARD flow — `MotionPriorCache` resolves them by filename only —
and a backward asset would silently initialize REVERSED velocities in the
correct arm while leaving the camera-swapped control unaffected, which is
the worst possible failure because it would corrupt B1-F specifically. The
sealed provenance record asserting `forward_t_to_t_plus_1` lives on
Leonardo and is unreachable from this workstation, so the convention was
tested against the pixels instead.

Test: pull `cam01` frames 0099/0100/0101 and `flow/cam01_0100.npz`;
sample frame 101 and frame 99 at `p + f(p)` and compare each to frame 100,
over masked pixels with `|f| > 0.5 px`.

| reconstruction of frame 100 | mean abs error |
|---|---:|
| **FORWARD:** `I101(p + f(p))` | **0.008032** |
| BACKWARD: `I099(p + f(p))` | 0.024695 |
| no warp: `I101` | 0.015302 |
| no warp: `I099` | 0.017304 |

**VERDICT: the assets are FORWARD flow, `t → t+1`.** Forward is 3.1× better
than backward and improves on the un-warped frame by **47.5%**, so the
field is genuinely explanatory rather than merely small. The npz layout is
confirmed `(H, W, 2)` channel-LAST with a boolean `mask` (99.56% valid),
which `normalize_flow_tensor` permutes to `(2, H, W)`.

**Measured motion scale, and it sharpens §6's expectation from an argument
into a number.** On this frame, flow magnitude percentiles over the valid
mask are **p50 0.057 px, p90 0.436 px, p99 3.076 px, p99.9 3.940 px, max
4.258 px**, and only **8.6%** of pixels move more than 0.5 px.

A birth site's one-frame displacement is therefore at most ~4 px even in
the moving minority, so the flow-derived velocity is a SMALL perturbation
to a row that subsequently trains for thousands of iterations. This is now
a measured pre-registration, not a guess.

**Consequence for execution.** The 1,200-iteration preflight is promoted
from a mechanical check to a DECISION POINT. It crosses the first birth
event at iteration 1,000 and reports the funnel — valid-flow site count,
`flow_realized_ratio_mean`, and the coefficient-norm distribution. The six
training cells are launched only if that funnel shows the mechanism
delivering non-negligible velocities to a meaningful share of sites.
If it does not, the measured funnel is itself the finding, and spending
~11 slot-hours to observe a null already visible in the preflight would be
poor use of a 24 slot-hour ceiling.

## APPENDIX B (append-only) — the magnitude guard is AMENDED, before any output

The guard frozen in §2 item 8 — clamp the solved coefficient's L2 norm to
the MEDIAN coefficient norm of existing rows — is **WITHDRAWN and
replaced**. It was measured, during implementation and before any cell ran,
to require a coefficient norm of **~328** for a realistic one-frame
displacement against a population median of **~1.6**. It would have shrunk
every flow-derived velocity by roughly 200×, so B1-F would have measured
the guard rather than the flow: a vacuous cell.

**Why the original was wrong, recorded because the error class generalizes.**
It was modelled on `_median_log_scaling`, which clamps a donor's SCALING to
the population median. That works because a donor's scaling and the
population's scaling are the same kind of physical quantity. **A LoRA
coefficient is not.** Its magnitude is set by the arbitrary scale of the
learned basis (init 0.01), so the coefficient needed to express a fixed
physical displacement bears no relation to the population's coefficient
distribution. The specification conflated "do not be a population outlier"
with "represent this velocity".

**The amended guard, frozen:**

1. No clamp on the coefficient norm. The minimum-norm least-squares
   solution can never OVERSHOOT the target: `c^T B = d · pinv(B) · B` is the
   projection of `d` onto `B`'s row space, so its norm is at most `|d|`.
   No displacement-space overshoot guard is needed. `pinv`'s rcond
   regularization is retained.
2. Per-row fail-closed if the solved coefficient or the realized
   displacement is non-finite.
3. Input-side outlier rejection: fail closed for a site whose flow
   magnitude exceeds the **99th percentile of that view's valid flow
   magnitudes**. This acts in PIXELS, where the quantity is physical and
   measurable, and can only REMOVE extreme sites — it can never manufacture
   an effect.
4. The funnel must report `flow_coeff_norm_mean`, `flow_coeff_norm_max`
   and `flow_realized_ratio_mean` (realized ÷ target displacement).
   The last makes basis rank-deficiency VISIBLE: far below 1 means the
   basis cannot represent the requested velocities and the cell is
   measuring that, not the flow.

Decided before any cell output, and therefore an amendment rather than a
post-hoc adjustment. The gates in §5 are unchanged.

## 7. Execution priority

This lane is SUPPORTING. It yields GPU precedence to the payload and
timing lanes, which are the paper blockers. Its required outcome is
satisfiable by an implementation-complete, admitted launch packet; the
six cells run only if the ceiling still permits after those lanes are
secured.
