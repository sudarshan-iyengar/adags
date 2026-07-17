# Phase 9 CSVL-ISR v1 Slice B operator and evaluation contract

Status: preregistered design pending independent admission and implementation
Date: 2026-07-15
Parent method: [[operations/phase9-csvl-isr-v1-method]]
Experiment plan: [[operations/phase9-csvl-isr-v1-experiment-plan]]

Slice B asks whether a fixed, point-neutral intervention can represent
intermittently visible surfaces when driven by strong reference or inferred
visibility evidence. It is not a second visibility model. It adds no trainable
network and does not change the rasterizer.

No Slice B run may begin until this contract and its implementation pass
independent review. Confirmatory inferred coupling additionally requires a
genuine Gate A engineering pass. Pre-Gate-A proxy diagnostics are exploratory
and cannot admit B03/B04 or change Gate A thresholds.

## Fixed representation and common checkpoint

- Base: `configs/n3v/fixed_budget_lora_route0_filemask_residual_600k.yaml`.
- Required derivative changes: `densify_until_iter=5000`, save/test iterations
  include 5000 and 6000, capacity mode explicit, hard-static conversion off.
- Gaussian mode: 4D, `rot_4d=false`, LoRA motion, scaffold off, no hard-static
  rows. Any violation fails closed.
- Seed: integer seed 0 for cycle-v1. Current `main.py` applies `--seed` and then
  `safe_state` resets RNGs to zero; implementation must make the CLI seed
  authoritative and add a seed-difference/repeat test before training.
- Train once per scene/seed from initialization through iteration 5000. Hash the
  full checkpoint and branch every lane from that exact checkpoint.
- Ordinary clone/split/prune/reset/topology ends at 5000 for every lane. At the
  start of iteration 5001, after LR update and before render/backprop, execute
  one mode-specific transaction. No topology mutation occurs afterward.
- Pilot endpoint: 5250. Comparable endpoint: 6000. There is no outcome-dependent
  horizon selection.

The matched route0 lane is the no-op continuation of the same iteration-5000
checkpoint with topology disabled. Canonical historical route0 remains a
context baseline but cannot replace this matched lane unless checkpoint,
configuration, code, seed, exposure, topology schedule, and metrics all match.

## Authoritative capacity budget

`dynamic=len(get_xyz)`, `hard_static=len(get_static_xyz)`, and
`total=dynamic+hard_static`. Record final, peak, mean, and cumulative
point-iterations every iteration. The hard-static count must remain zero in v1.
Dynamic, hard-static, and total counts are invariant across the transaction and
total never exceeds 600,000. Primary comparable lanes require final and
integrated counts within +/-2% of matched route0.

Current reporting incorrectly calls the dynamic count total and can leave static
growth uncapped. Repair all reporting and guards before B00; tests include a
nonempty hard-static fixture even though v1 forbids it in experiments.

## Evidence sidecar types

All sidecars are immutable, schema-versioned, and training-camera-only at the
operator boundary. They contain target surfels with world xyz, training time,
geometry-matching linear RGB, model-space RGB from the repository's unchanged
PIL-to-tensor `/255` path, spatial covariance/nearest spacing, source camera
hashes, track/state/confidence, and provenance. Geometry matching may use linear
RGB; Gaussian initialization uses model-space RGB unless every lane changes the
render/loss color space identically.

### Inferred sidecar

Uses only frozen CSVL predictions from transforms-train cameras. Eligible target
tracks have an accepted visible-to-occluded or occluded-to-visible transition,
at least three source cameras, risk <=0.75, and a currently visible target
surfel. Cam00 RGB, annotations, state apertures, and frozen evaluation masks are
prohibited.

### Oracle sidecar

This is the mandatory surface-level oracle-evidence control, not a proxy and not
the reported method. It is constructed only after genuine human annotations
exist. For the frozen cut development-test tracks, use annotated visible rear
polygons in at least two named training cameras, their calibrated K/w2c, and
pinned DA3 optical z to backproject/fuse the corresponding surface. Human
visible/occluded/order/transition labels choose eligible tracks and times.
Cam00 pixels may be used only to define the evaluation event, never target xyz,
color, or a training observation. The oracle sidecar is sealed before any lane
render is inspected and is used only in the oracle-capacity diagnostic. It
cannot tune or supervise the non-oracle full lane.

### Generic capacity-only sidecar

At the same registered iteration and K, choose from all finite visible
training-camera surfels without reading inferred or oracle event/state fields.
Sort by SHA-256 of `(scene,frame,camera,point_id,capacity_seed)` and apply the
same target NMS; take the first K. It may hit event surfaces by chance. It is
rate-matched only by iteration and K, not negatively selected away from events
or distribution-matched with visibility evidence.

### Shuffled sidecar

Start from the inferred eligible target set and sort receivers by canonical
`target_id`. Let `s=capacity_seed+17011`, where `capacity_seed` equals the
registered training seed. For domain UTF-8
`csvl-isr-v1/shuffled-targets\0`, compute `h=SHA256(domain || ASCII(str(s)))`,
interpret the first eight digest bytes as an unsigned big-endian integer, and
set `offset=1+(h mod (N-1))`. Require `N>1`. Receiver `i` keeps its xyz, scale,
confidence, and capacity slot, but receives track/time/model-RGB attributes
from sorted source `(i+offset) mod N`. This exact hash-derived cyclic
permutation has no fixed points and preserves K and the attribute multisets
while destroying surface/time/appearance identity.

## Fixed target and donor counts

- Pilot: `K=256` exactly.
- Comparable runs: `K=2048` exactly.
- Fewer than K eligible targets or donors is a feasibility failure and the
  transaction abstains without partial mutation. Do not fall down a K ladder
  after observing outcomes.
- Target surfels are sorted by descending confidence, then scene/time/track/
  source-camera/point ID. Greedy 3D NMS uses radius equal to initialized spatial
  scale; first K surviving targets are used once.

## Donor eligibility and ranking

Construct one event-blind base donor universe at iteration 5001 before reading
any inferred/oracle event, state, confidence, or target fields. A dynamic slot
enters this universe only when all hold:

1. slot generation age is at least 500 iterations;
2. activated opacity is in the bottom 20% of finite dynamic rows, where stable
   rank rather than interpolated quantiles selects `floor(0.20*N)` rows;
3. at its base xyz it has another finite dynamic row within twice its activated
   maximum spatial scale whose opacity is greater, breaking neighbor ties by
   distance then stable slot ID;
4. all parameter/optimizer/accumulator values are finite.

Rank the base universe by `(activated_opacity, normalized_denom,
-redundant_neighbor_count, stable_slot_id)`, ascending, where
`normalized_denom=denom/max(age,1)`. Capacity-only and null-reset take the
first K without reading event evidence. In inferred/oracle modes only, remove
rows protected by the corresponding high-confidence surface association or an
accepted association within +/-5 frames, then take the first K. Log the base
universe hash, every removed stable ID and reason, and the selected donor hash.

Thus the target, protection, and weighting contributions are part of the
visibility-capacity coupling; they are not silently shared with capacity-only.
The common control is the same event-blind base universe, ranking, row operator,
trigger, and K. Report protection-only diagnostics so any gain is not
misattributed to target initialization alone. Frustum `visibility_filter` is
never used as an occlusion label.

## Exact row initialization

Initialization is identical in capacity-only, oracle, full, and shuffled modes:

- `_xyz`: target world xyz;
- `_features_dc`: `RGB2SH` of coordinate-wise median visible model-space RGB
  from the unchanged PIL-to-tensor `/255` representation;
- `_features_rest`: zeros;
- `_scaling`: three copies of
  `log(clip(0.5*d_nn,0.001*R_scene,0.02*R_scene))`;
- `_rotation`: identity quaternion `(1,0,0,0)`; isotropic scale makes orientation
  immaterial at initialization;
- `_opacity`: `inverse_sigmoid(0.1)`;
- `_t`: target time in `[0,10]`;
- `_scaling_t`: `log(sqrt((time_duration[1]-time_duration[0])/5))`;
- `_route_logit`: configured `route_logit_init`;
- `_motion_v`, `_motion_a`, `_motion_lora_coeff`: zeros;
- `_staticness_score`: zero;
- `_rotation_r`: prohibited because v1 requires `rot_4d=false`;
- scaffold attachments: prohibited because scaffold is off;
- gradients, `xyz_gradient_accum`, `t_gradient_accum`, `denom`, and
  `max_radii2D`: zeros;
- generation increments, stable slot ID remains, created/last-reassigned
  iteration becomes 5001.

Every optimizer-managed per-row tensor is modified in place so `nn.Parameter`
identity and parameter groups remain stable. Reset donor rows of `exp_avg`,
`exp_avg_sq`, and optional AMSGrad buffers to zero. Preserve the tensor-level
Adam step because it cannot be reset per row. Survivor values and moments remain
bitwise unchanged at mutation time.

## Mode-specific transaction

| Mode | Observation weighting | Capacity transaction |
| --- | --- | --- |
| route0 | unchanged | no-op |
| null-reset | unchanged | rewrite selected donor values to exact originals and apply the B02-selected moment policy to the same K event-blind donors; diagnostic only |
| capacity-only | unchanged | generic sidecar, event-blind base donors/operator/K |
| oracle-capacity | unchanged | oracle sidecar, oracle protection over the common base donors/operator/K |
| visibility-only | inferred visible/reveal confidence weighting | no mutation |
| full | inferred confidence weighting | inferred sidecar and protection over the common base donors/operator/K |
| shuffled | inferred-rate weighting with permuted identity | shuffled sidecar and permuted protection over the common base donors/operator/K |

The current base L1 and SSIM terms are scalar-reduced. For visibility modes only,
replace the base L1 scalar with
`sum(weight*mean_channel(abs(render-gt)))/sum(weight)`, where weight is
`1+0.5*confidence` inside accepted visible/reveal training-camera regions and 1
elsewhere. Keep the existing scalar SSIM term, dynamic-ROI term, static
exclusion, flow, and every other loss unchanged. Thus
`(1-lambda_dssim)*weighted_L1 + lambda_dssim*(1-SSIM)` replaces only the base
reconstruction expression. Capacity modes without visibility weighting retain
the original L1 exactly.

The pixel confidence is deterministic raw evidence, not human-calibrated
probability: `c=1-state_risk` for accepted risk <=0.75, clipped to `[0,1]`.
Rasterize every accepted visible or reveal micro-surface into its source
training camera/time with the same point ellipses and ownership rule as CSVL.
At overlaps take maximum c, breaking exact ties by lower track ID; absent support
has c=0. The training weight is exactly `1+0.5*c`.

For shuffled weighting, first seal every inferred per-camera/time confidence
map and sort map keys by `(scene,camera_id,time)`. With the same `s` but domain
UTF-8 `csvl-isr-v1/shuffled-maps\0`, compute `h` and `offset` exactly as above;
require `N>1`, and assign receiver key `i` the entire map array from source key
`(i+offset) mod N`. All arrays must share the declared native shape. This preserves
the multiset of confidence values and per-map compactness while destroying
camera/time identity. Capacity target identity and protection use the separately
declared shuffled target permutation. Both permutations and map hashes are
logged; no re-rasterization from outcome-dependent fields is allowed.

## Transaction and checkpoint protocol

Transaction states are `pre`, `applying`, and `applied`. Its ID hashes source
checkpoint, mode, sidecar, config, seed, K, iteration, and the typed
`optimizer_policy` artifact selected by B02. Pilots use zero moments; B02 emits
exactly `zero` or `coordinatewise_lower_median_v1`. Every B03/B04
capacity-changing lane and B03 null-reset consumes that immutable policy hash.
If median is admitted, null-reset assigns median moments to unchanged donor
values; if zero is admitted, it zeroes them. Thus null-reset always isolates the
selected optimizer-state surgery from geometry reassignment. Save Python, NumPy,
Torch CPU/all-CUDA RNG, sampler, LR scheduler, AMP scaler if present, optimizer,
model, lifecycle state, and transaction state. Write a temporary checkpoint,
validate internal/source hashes, fsync, atomic rename, then fsync parent.

On resume: `pre` re-applies; `applying` restores the validated source and
re-applies; `applied` skips. An orphan temporary file is ignored unless its
internal state and hashes validate, in which case recovery is deterministic.
Inject failure before each state transition in B00.

Store versioned `capacity_state` in the existing nested routing/motion dictionary
rather than changing positional checkpoint tuple length. It contains stable IDs,
generation, created/last-visible/last-reassigned iterations, protection expiry,
utility/visibility accumulators, transaction, controller/config/sidecar hashes,
RNG state, and cumulative counters. Historical checkpoints initialize it
deterministically.

## Gate B evaluation sidecars

Evaluation artifacts are frozen before lane renders open.

- Event interior: every human visible rear polygon on a reveal frame and the
  next four frames (`f..f+4`) while the same track remains visible.
- Event boundary: one-pixel morphological gradient; 4-pixel tolerance is
  primary for localization continuity only.
- Static mask: complement of the union of the pre-existing motion-prior mask and
  every annotated event/foreground polygon, each dilated by a 16-pixel disk;
  erode the complement by an 8-pixel disk. Source mask/config/array hashes are
  sealed. Empty masks make the frame non-evaluable.
- Event PSNR: clamp render and GT to the model's sRGB-coded `[0,1]` space, pool
  squared channel errors over all valid event pixels/frames in one event, and
  use `10*log10(1/max(MSE,1e-12))` with data range 1. Exact-zero MSE is flagged
  separately; its registered finite PSNR is therefore 120 dB.
- Event LPIPS: repository `lpipsPyTorch`, AlexNet, version 0.1, on clamped
  sRGB-coded `[0,1]` inputs. Before outcomes, a Slurm preflight seals hashes of
  AlexNet weights, LPIPS linear weights, and implementation files. Compute each
  layer's learned 1x1 map before spatial mean, area-downsample the binary mask,
  divide weighted map sum by mask mass, sum the five layers, average valid frames
  within event, then unweighted event/scene macros.
- Static metrics use all 300 cam00 test frames, not only event horizons. A
  frame is valid when its frozen static mask has at least 1024 pixels; static
  flicker additionally requires a valid adjacent flow pair. Static PSNR,
  LPIPS, and masked L1 reconstruction no-harm average valid frames within scene.
  A scene needs at least 270 valid PSNR/LPIPS/L1 frames and 269 valid flicker
  pairs; otherwise its static gate is not evaluable. Report event-horizon static
  metrics secondarily. The former `static ghost` name is retired: its canonical
  2% no-harm bound applies to `static_reconstruction_l1`, not to an independent
  ghosting claim.
- Flicker uses declared backward flow `B_{t->t-1}` to bilinearly sample the
  previous render/GT at current pixels with `align_corners=False`. Valid pixels
  require current mask, warped previous mask >0.5, both flow-valid flags, and
  forward/back cycle <=1.5 px. Score sRGB-coded L1 of
  `(R_t-W(R_{t-1}))-(G_t-W(G_{t-1}))`, separately for event/static masks.
- Reveal ghost is distinct from reconstruction error. Backward-warp the
  pre-reveal GT foreground at `f-1` into offsets `f..f+4`. On valid reveal
  pixels score `max(0,L1(R_t,G_t)-L1(R_t,W(G_{f-1})))`; a render retaining the
  old foreground receives positive trail error. Report each offset, then
  unweighted frame/event/scene means.

Finite and missing-unit rules are fail-closed. PSNR uses
`10*log10(1/max(MSE,1e-12))` and records exact-zero MSE separately, so paired
deltas are finite. For an error ratio `(method-base)/base`, base=method=0
maps to 0, while base=0 and method>0 maps to positive infinity and fails an
upper bound. For relative improvement `(base-method)/base`, the same zero/zero
case maps to 0 and base=0, method>0 maps to negative infinity and fails a lower
bound. Nonfinite inputs other than these declared extended results invalidate
the unit.

A frame mask with fewer than one valid pixel is excluded and counted. An event
metric requires at least one valid frame; a reveal-ghost event requires every
declared offset 0--4 or is not evaluable. A scene event macro requires the
method contract's minimum positive tracks plus at least one valid metric unit
per included track. Any missing required metric, under-minimum scene, undefined
macro, or mixture that cannot be reduced by these rules is `not_evaluable` and
cannot pass. Scene and overall macros never silently substitute pooled pixels.

Except for pooled event PSNR as defined above, metrics aggregate valid frames by
arithmetic mean within event, then unweighted event and scene macros; pooled
pixels are secondary. A catastrophic event is PSNR <=-0.50 dB or
LPIPS >=+10% versus matched route0. Report paired event deltas and clustered
bootstrap intervals with 10,000 replicates, seed 20260715.

## Gate B admission and promotion

Pilot B01/B02 asks only feasibility/stability: finite optimization/renders,
exact budget, realized K, zero invariant failures, and no catastrophic global or
static diagnostic at 5250. It does not tune K, weights, or donor rules.

Confirmatory cut B03 requires a genuine A06 engineering pass and the frozen
6000-iteration lanes. Full must satisfy every canonical hard condition versus
route0: mean event PSNR >=+0.20 dB; mean event LPIPS improves by >=5%; a strict
majority of evaluable events improves on both; mean static PSNR delta >=-0.05
dB; mean static LPIPS and static reconstruction L1 each regress by <=2%;
event/static flicker and reveal ghost each regress by <=5% in their declared
scene macro; no
qualitative broad floaters, duplicated surfaces, or persistent ghost layer in
the blinded review; no scene failure; and final/integrated point budgets match. Against each of
capacity-only and visibility-only, full additionally requires mean paired event
PSNR >=+0.05 dB, mean LPIPS >=2% better, and at least half of events better on
both. Shuffled must fail to reproduce at least one half of the full route0 gain:
its PSNR gain <0.5*full PSNR gain or its LPIPS improvement <0.5*full LPIPS
improvement, and it cannot improve a majority of events on both. Exact equality
fails strict control comparisons. Oracle-capacity is attribution, not a
competitor the non-oracle method must beat.

Only a B03 pass freezes a method checksum for all-six-scene B04. On scenes
without human events, B04 evaluates global/static/flicker/quality-budget metrics
and reports event Gate B as not evaluable. Human event claims remain scoped to
cut/flame/sear. No scene may fail its available static no-harm criteria.

Seed-0 all-six completion is the first comparison. Seed expansion requires
completed cut/flame/sear human labels, the same aggregate annotated-scene event
conditions as B03, every available per-scene static bound, no scene-wide
failure on all six, and the same full-versus-control/shuffle conditions. If it
passes, run the same route0/capacity-only/visibility-only/full/shuffled matrix
for seeds 1 and 2. This promotion predicate is fixed before outcomes.

## Permitted optimization repair

The single permitted repair is optimizer-state initialization only. It is
eligible only when B00 passes and every registered K=256 capacity pilot has a
finite pre-transaction state but exceeds 2x the matched route0 reconstruction
loss at any of iterations 5002--5011. A registered decision artifact evaluates
that predicate before either repair lane can run.

The repair retains the exact donor/target/operator/K/trigger/horizon and replaces
zero moments only. For each optimizer tensor and each coordinate independently,
take the lower median over finite, non-donor rows in the event-blind base donor
universe at iteration 5001; assign that vector to each selected donor. Empty or
nonfinite populations make the repair ineligible. Run both capacity-only and
oracle-capacity median-moment pilots from the same source checkpoint. No
geometry, K, opacity, scale, weight, donor, trigger, or horizon change is
allowed. A second repair is prohibited; failure pivots representation/operator.

## Required instrumentation and tests

Log exact dynamic/static/total/final/peak/integrated counts; donor candidates;
protected slots; target tracks; requested/realized K; abstentions; initialization
failures; state-wise capacity; slot generations/ages; moment rows reset; mode/
sidecar/config hashes; memory, wall time, active splats, and invariant failures.
Detailed event ledgers stay under the run directory.

Tests cover eight-slot selection, no-op, null-reset, all per-row tensors,
hard-static budget accounting, seed authority/repeat/difference, moment surgery,
survivor identity, invalid/no-donor/no-target/K=0 paths, temporal initializer,
mode equivalence, checkpoint/failure recovery, resume equivalence, metric masks/
formulas, and one finite Slurm render/gradient smoke.
