# Phase 9 CSVL-ISR v1 method specification

Status: implementation-ready; independently admitted in round 10 (9.1/10)
Date frozen: 2026-07-15
Objective: [[objectives/depth-visibility-capacity-v1]]

Post-outcome note, 2026-07-25: this frozen specification remains the historical
authority for the v1 implementation and its completed outcomes. The corrected
B01 result establishes reassignment-operator stability but not reconstruction
impact. The forward method decision is recorded separately in
[[phase9-post-b01-csvl-vpl-direction]]; this page is not rewritten to imply that
v1 tested persistent surface identity or visibility-guided allocation.

CSVL-ISR means **Calibrated Surface Visibility Ledger with Intermittent-Surface
Reassignment**. This document is the cohesive, implementation-facing method
contract. Raw refinement rounds and independent review history live under the
Phase 9 operational state directory. Machine-readable schemas/configuration
must agree with this page; a mismatch fails closed.

## Scientific boundary

V1 asks whether calibrated training views can predict a surface's first/rear
order in the held-out cam00 view while that surface is reconstructed from at
least three other synchronized cameras, and whether an observed track changes
between visible and occluded states. It does not infer the current position,
continued existence, or order of a deforming surface during a gap in all
cameras. Such a gap may carry `previously_observed_currently_unobserved`; it has
no hidden xyz or order. Flow-based recovery within five frames is
`reappearance`, not reveal.

Gate A and Gate B are independent. Slice A implements only deterministic
visibility evidence and evaluation. It cannot import or mutate the trainer,
Gaussian bank, optimizer, or checkpoint. Slice B is a later, separately reviewed
point-count-neutral reassignment experiment. No label-free or controlled result
can substitute for genuine human labels in real Gate A.

## Data and target exclusion

The active N3V loader uses `transforms_train.json` and `transforms_test.json`.
Live metadata inspection establishes that cam00 is the test camera and cam10 is
a training camera in all six scenes. CSVL parses those files directly and
rejects any transforms-train image/camera record also present in transforms-test.

- Sidecar construction uses RGB, depth, confidence, and flow only from
  transforms-train cameras.
- Cam00 contributes only K, w2c, image dimensions, and time for projection.
- Cam00 RGB and human labels may be opened only by annotation/evaluation after
  prediction artifacts are sealed and hashed.
- Training-camera states are disclosed auxiliary anchors, but each scored
  target camera is leave-one-camera-out through the complete DA3/fusion/witness
  dependency graph; its RGB/calibration may not influence its own state.
- `cut_roasted_beef` is development. `flame_steak` and `sear_steak` remain locked
  labeled transfer. A later admitted representation configuration is frozen
  before the complete six-scene N3V comparison.

Every output carries a transitive dependency manifest. For a scored target
camera c, every consumed DA3 group, source node, fused hypothesis, temporal
edge, and first-visible witness has a complete physical-camera dependency set
that excludes c. Group outputs are generated from the camera universe with c
removed, or filtered to complete group-member sets excluding c; group validity,
diversity, and the two-valid-group minimum are recomputed after exclusion.
Mixing hypotheses whose ancestry contains c is prohibited even when their own
source-camera ID differs. The evaluator recursively verifies this invariant and
rejects missing ancestry. For cam00 it additionally verifies that no cam00 image
hash occurs anywhere in prediction provenance. Registered historical monocular
baselines consume their scored target and are explicitly target-consuming.

## Camera and depth conventions

The adapter applies the active loader's OpenGL camera-to-world Y/Z column flip
and inversion exactly once. It emits column-vector OpenCV matrices with positive
camera z:

`X_cam = R_cw X_world + t_cw`, `u=fx*x/z+cx`, `v=fy*y/z+cy`.

Calibration uses native integer pixel centers. Resizing records the exact
native-to-processed transform and uses bilinear `align_corners=False` sampling.
Nonzero distortion, unsynchronized timestamps beyond `1e-6 s`, rolling shutter,
nonfinite calibration, or nonpositive projection z fails closed.

`R_scene` is the median distance from all transforms-train camera centers to
their coordinate-wise median `C_med`. It is a scene-gauge scale, not meters.

The pinned DA3 code unprojects `K^-1[u,v,1] depth`; its depth is therefore
treated as optical-axis camera z in the supplied extrinsic gauge, not ray
distance, inverse depth, normalized depth, or proven metric depth. Use supplied
K/w2c, `align_to_input_ext_scale=True`, `process_res=504`,
`process_res_method=upper_bound_resize`, `infer_gs=False`,
`use_ray_pose=False`, and `ref_view_strategy=saddle_balanced`. The adapter
must pass all of these arguments explicitly and reject a runtime override. Fit
no scale/shift and perform no per-frame normalization.

### DA3 pin

- checkout commit: `41736238f5bced4debf3f2a12375d2466874866d`;
- model ID: `DA3NESTED-GIANT-LARGE-1.1`;
- `config.json` SHA-256:
  `09adf89474017e717bc05aa86fd3a378708ba8914b036d61874eced328069468`;
- `model.safetensors` expected size: 6,759,558,100 bytes;
- `api.py`: `042ae8f6bfd1e5610100a585ebb00f8cbc3320f9aa9e03a07530fd4deda8d2d8`;
- `input_processor.py`:
  `cdebb6021f5f1001d3aa56261c224a80b9d8431efedfb040537c58a5509410c1`;
- `geometry.py`:
  `3d67391f2eeed57ee5d07387947dea7a5d99fc0ce0a63c6890485183c851c29b`.

Hashing the 6.76 GB weights is a registered CPU Slurm preflight. No inference
may run until its SHA-256, job ID, terminal state, and command are sealed.

### Multiview groups

For anchor a, candidates have optical-axis angle `<=75` degrees and center
distance from a `>=0.02 R_scene`. Start with a. Repeatedly select the candidate
maximizing its minimum center distance to selected cameras, divided by
`R_scene`; ties use lower camera ID. Stop at six. Fewer than six, or second
singular value of centered selected centers `<0.01 R_scene`, invalidates the
group. Every camera/time needs predictions from at least two valid groups.

Weighted median sorts `(value,stable_node_id)` and returns the first value whose
cumulative weight is at least half the total; all-zero weights use the lower
middle value after the same sort. Every coordinate-wise median uses the lower
middle for even n. Scientific manifests are canonical UTF-8 JSON with sorted
keys, compact separators, finite numbers, arrays represented by hash/dtype/shape/
byte-order/semantic key, and scheduler/runtime metadata in a separate envelope.
The canonical encoder is `csvl-cjson-v1`: identity payload floats are lowercase
16-hex-digit IEEE-754 binary64 big-endian strings, integers are JSON integers,
and strings are NFC UTF-8. IDs are lowercase SHA-256 of
`domain_separator + NUL + canonical_payload`. The implementation-freeze
manifest pins the encoder tests, Python/runtime version, and code hash.

The manifest records native/processed image hashes, K/w2c, group/order, exact
crop/resize, flags, code/model/environment pins, returned processed K, and array
hashes. Group-sample IDs use domain `csvl-v1/group-sample` and payload
scene/frame/time/source camera/ordered group member cameras/group ordinal/
processed y/x/image/K/w2c/depth/confidence hashes. Group samples at the same source camera/frame/processed pixel are aggregated
before a source node exists. The accepted source-node payload, domain
`csvl-v1/source-node`, is exactly scene/frame/time, source camera, scored-target
camera, processed y/x, sorted contributing group-sample IDs, sorted union of
complete physical-camera ancestry, weighted-median optical-z hex, arithmetic
mean of finite nonnegative DA3 confidence as binary64 hex, and covariance
semantic hash. Optical z uses nonnegative-confidence weighted median with
ordinary-median fallback. Relative MAD `>0.05` emits only a
`csvl-v1/source-node-rejected` record containing the same ancestry/sample IDs,
aggregate values, `retained=false`, and reason `duplicate_relative_mad`; it
cannot enter fusion. Accepted records contain `retained=true`. This aggregate
source-node ID is the `stable_node_id` everywhere downstream.

## Spatial samples and node uncertainty

Regular processed-image samples begin at `(0,0)` and advance by 8 integer pixels
on each axis. Convert sRGB to linear IEC 61966-2-1 RGB, then gray with
`0.2126 R+0.7152 G+0.0722 B`. Use 3x3 Sobel with OpenCV
`BORDER_REFLECT_101`. Rank candidates by `(-magnitude,y,x)` and greedily retain
points at least 8 pixels apart, capped at `floor(0.25*n_grid)`. Emit grid first
and remove exact duplicates.

A node stores camera/frame/pixel, optical z, world point, linear 5x5 RGB patch,
confidence, and covariance. Set
`sigma_z=max(0.01*abs(z),1.4826*MAD_duplicate_z)` and pixel standard deviation
0.5 on u/v. Transport `diag(0.5^2,0.5^2,sigma_z^2)` through the unprojection and
camera Jacobians. Pair depth uncertainty is the sum of the two transported
camera-z standard deviations.

## Target-independent fusion

For target c, exclude all c nodes. For anchor node i, project into each other
source camera j and find a reciprocal sampled node with positive/in-range z,
projection residual `<=2 px`, depth residual `<=2.5 sigma_z_pair`, and patch
NCC `>=0.60`. Proposal cost is the arithmetic mean of projection/2,
depth/(2.5 sigma), and `(1-NCC)/0.40`, each clipped to `[0,2]`. Select the minimum
tuple `(cost,camera_id,y,x,node_id)` and require reciprocal identity and cost
`<=1`. Keep one match per physical camera and require at least three cameras.

Generate all accepted proposals, sort by `(-camera_count,median_pair_cost,
anchor_id,sorted_node_ids)`, and greedily enforce global source-node exclusivity.
Initialize the point as coordinate-wise world median. Reject nodes once at
diagonal Mahalanobis distance `>2.5`; recompute and retain at least three cameras.
Fused covariance is
`diag((1.4826*MAD_xyz)^2)+mean(transformed_node_covariance)`. A fused point's
linear-RGB coordinate is the coordinate-wise weighted median of constituent
source-node center colors using `w=max(DA3_confidence,0)`; nonfinite confidence
invalidates the node, and an all-zero set uses the declared lower median. Its
patch-color scalar weight is `sum(w)`; an all-zero patch uses the declared lower
median. Fused-hypothesis IDs use domain `csvl-v1/fused` and payload
scene/frame/scored-target/sorted source-node IDs plus the exact retained robust
subset and coordinate/covariance array hashes. Numerical values are clipped to
`1e-12` only inside inverses/eigendecompositions and clipping is recorded.
Within `0.002 R_scene`, retain the proposal with more cameras, lower cost, then
lower anchor ID.

Raw pair risks use the corresponding normalized terms clipped to `[0,1]`.
Robust-distance risk is `min(1,d/2.5)`; support risk is
`clip((4-camera_count)/2,0,1)`. Fused-point risk is the maximum of duplicate,
pair, robust-distance, and support risks. All risks stay in `[0,1]`.

## Micro-surfaces and dense ordering

Estimate each point normal by PCA over its eight nearest fused neighbors within
`0.02 R_scene`, requiring six. Sort neighbors by
`(squared_distance,fused_hypothesis_id)`; exact distance ties therefore never
depend on library iteration order. With eigenvalues
`lambda0<=lambda1<=lambda2`, require `lambda1>=1e-6*R_scene^2`,
`lambda1/max(lambda2,1e-12*R_scene^2)>=0.05`, and
`(lambda1-lambda0)/max(lambda2,1e-12*R_scene^2)>=0.05`; otherwise mark
uncertain. Orient the unique smallest-eigenvalue eigenvector toward the median
supporting-camera center: flip when its dot with
`(C_support_median-X)` is negative; on an exact zero dot, flip if necessary so
the first nonzero component in x/y/z order is positive.
World voxel index is `floor((X-C_med)/(0.01 R_scene))`. Within each voxel
connect points at distance `<=0.01 R_scene`, normal angle `<=30` degrees,
linear-RGB L2 `<=0.15`, and at least two shared physical support cameras.
Deterministic connected components of at least three points are local
micro-surface patches. Patch IDs use domain `csvl-v1/patch` and payload
scene/frame/scored-target/voxel/sorted fused-hypothesis IDs.

Patch centroid/color are coordinate-wise medians (color uses confidence-weighted
median). Patch normal is the normalized sum of oriented normals; norm `<1e-8`
makes all observations uncertain. Patch risk is the nearest-rank 90th percentile
of point risks. Patches are local surfaces, not semantic objects.

After temporal IDs are assigned, rasterize every constituent fused point. Its
ellipse comes from transported point pixel covariance. Eigenvalues descend;
eigenvector signs make the first nonzero element positive. Covariance semi-axis
is `ceil(2.5 sqrt(max(lambda,0)))`. Spacing radius is
`ceil(0.5*d_nn)` to another projected point in the same patch, or 4. Final axes
are `clip(max(covariance_axis,spacing_radius),2,8)`. Include an integer pixel
center when rotated normalized squared distance is `<=1`.

Within a track/pixel keep smallest z, then risk, then patch ID. Between tracks,
depth gaps within
`m=max(0.01*z_front,2.5*(sigma_front+sigma_rear))` are tied/uncertain. Otherwise
sort z. First layer is visible. A later layer is occluded only when it is beyond
m, is first layer in another training-camera prediction at that time, and has
source nodes disjoint from the foreground. Regions are 8-connected accepted
same-track/state pixels; area `<16` becomes uncertain. No interpolation, fill,
dilation, closing, or tile expansion precedes scoring.

## Temporal tracks and states

Flow sidecars must declare source/target camera/image/frame, direction, dt,
dimensions, pixel units, integer-center convention, bilinear sampling,
valid/occlusion semantics, generator revision, and source/array hashes. Ambiguous
legacy direction fails closed.

For every constituent source point at t, sample forward flow and then backward
flow. A reciprocal node-flow match requires cycle `<=1.5 px`, a nearest eligible
destination within 2 px, and the reverse nearest tuple `(distance,y,x,node_id)`.
Nodes are used once. A patch pair needs at least two cameras and three reciprocal
node matches per camera. Per-camera displacement is coordinate-wise median.

Candidate patch pairs require centroid distance `<=0.05 R_scene`, RGB L2
`<=0.20`, and two cameras with valid flow. Aggregate median errors:
reprojection/2, cycle/1.5, appearance/0.40, and 3D-consensus/(0.02 R_scene).
Search cost is `0.4,0.3,0.2,0.1` weighted with terms clipped `[0,2]`; risk is the
max with terms clipped `[0,1]`. Require reciprocal argmin `(cost,patch_id)`,
cost `<=1`, and second/best `>=1.2`.

Any bipartite candidate component with degree >1 on either side is a split/merge:
no identity propagates and involved observations are uncertain. Degree-one
accepted edges propagate the lower prior ID. Unmatched patches get new IDs with domain `csvl-v1/track` and payload
scene/scored-target/first frame/initial patch ID. Propagated IDs retain that
origin; every temporal edge has domain `csvl-v1/track-edge` over the ordered
endpoint patch IDs, flow manifests, and match tuple. Split/merge components hash
their sorted candidate IDs only for provenance and never propagate identity.

A dormant ID retains last visible per-camera projections, RGB, normal, and ID
for at most five frames, never current xyz/order. Chain valid forward/backward
flow through every missing step. Re-identification needs two cameras, endpoint
centroid `<=2 px`, NCC `>=0.60`, normal angle `<=30`, RGB L2 `<=0.15`, reciprocal
cost `<=1`, and second/best `>=1.2`. One-to-one acceptance restores the ID and
marks reappearance/bookended gap; missing frames remain unobserved.

Observation states are visible, occluded, out_of_frustum, invalid, or uncertain.
Track-time aggregate is observed if any camera is visible, unobserved if no
current hypothesis exists, otherwise uncertain when only ambiguous hypotheses
exist. Per-camera reveal is occluded->visible; hide is visible->occluded;
visible-visible/occluded-occluded/out-out is none; all other pairs are uncertain.

## Risk and calibration

All risks are clipped to `[0,1]`: duplicate `relative_MAD/0.05`; reciprocal
projection `e_px/2`; depth residual over `2.5*sigma_z_pair`; appearance
`(1-NCC)/0.40`; robust distance `d/2.5`; support `(4-camera_count)/2`; temporal
reprojection/2, cycle/1.5, appearance/0.40, and 3D error/(0.02 R_scene).
Identity risk is zero without a competitor; otherwise it is
`min(1,1.2/(second_best/best))`. If best=0 and second>0 the ratio is infinity; if
both are zero it is 1 and ambiguous. Missing mandatory terms make the unit
uncertain.

State risk is the maximum applicable point, patch, order, identity, and temporal
risk. Region risk is nearest-rank 90th percentile, index `ceil(0.9*n)-1` after
stable ascending sort. Transition risk is max(previous region,current region,
temporal edge). Visible/occluded order risk is
`min(1,m/max(depth_gap,epsilon))`; without a distinct competitor no ordering
unit is emitted. Default accepted risk is `<=0.75`; report 0.25/0.50/0.75 and
full risk-coverage.

Fit increasing risk-to-binary-error isotonic PAV separately for ordering and
transition on frozen cut calibration only, weighting each track equally. Require
both classes, six event groups, at least 50 order units, and at least 20 positive
plus 20 negative transition units. Otherwise calibration and the tier are not
evaluable. ECE uses 15 stable equal-mass bins and Brier uses mapped error
probability. Report 10,000 track-clustered percentile bootstrap replicates,
seed 20260715, with 2.5/97.5 percentiles; point thresholds remain decisive.

## Human reference protocol

The 54 raw-RGB candidate windows and their cut calibration/development split are
frozen in the Phase 9 annotation manifest. They are disjoint from R009 windows
plus the declared margin. Adjacent inclusive windows share a boundary frame;
de-duplication assigns a physical track to the earlier window. Predictions on a
shared boundary also belong to the earlier window.

The packet shows synchronized raw RGB only. Human fields begin empty. Annotators
never see CSVL, DA3, flow, residuals, or renders. Tables are:

- `track_frames`: track/camera/frame state, a rear polygon whenever visible,
  and an evaluation-only `state_aperture` for every evaluable visible/occluded
  row;
- `ordering_pairs`: directed foreground/rear order with camera/frame;
- `transitions`: track/camera/frame pair and reveal/hide/none/unknown;
- `frame_reviews`: one cam00 row per candidate frame, with nullable
  `spatial_complete`, `no_evaluable_visible_rear_surface`, `unknown_reason`, and
  annotator/adjudication provenance.

Cam00 is reviewed throughout the window. Each positive track also has two named
training cameras reviewed throughout, with rear polygons on every visible frame.
A visible state aperture is exactly the rear polygon. For an occluded row it is
a small human-drawn region on the foreground occluder where synchronized
before/after and source-camera evidence establish that the same rear track lies
behind it. It is not a hidden-shape annotation and never enters inference or
training; ambiguity makes the row unknown. Double-annotation aperture agreement
requires IoU >=0.50.
A positive/evaluable track has two visible rear frames, the same surface in two
training cameras, one valid visible<->occluded transition/order pair, and no
unknown on the transition or its neighbor. Unknown is always allowed. No-event
and unknown windows remain recorded.

`spatial_complete=true` means the entire frame was reviewed, every visible rear
surface belonging to an evaluable event track has a polygon, and no unresolved
spatial unknown remains. A completed no-event frame has an empty reference
union. Any unresolved spatial unknown excludes the whole frame and is counted;
there is no inferred unknown-area pixel mask.

Select the first 12 positives per scene by `(transition_frame,window_start,
canonical_track_id)`; 8-11 is exploratory, fewer than 8 makes a scene not
evaluable. Every one of the 54 candidate windows is assigned to independent
roles A and B before labels; actual identities must be filled and distinct
before either packet opens.

Double annotation is two-stage. In discovery, A and B independently review and
seal the entire window. A frozen matcher links discoveries only within a window
when mean cam00 visible-frame IoU and named-source-anchor IoU are each >=0.30;
weight is `0.75*target+0.25*source`. Assignment maximizes the exact sum of
binary64 weights first; among equal-primary optima it chooses the
lexicographically smallest sorted edge list
`(window_id,A_track_id,B_track_id)` without an epsilon perturbation. Any
degree>1 fragment/merge candidate component goes directly to blind
adjudication. Matched pairs plus every one-sided
discovery form the provenance-preserving union roster. In the roster pass, A
and B independently label every canonical roster track across the predeclared
camera/frame rows; `not_found` is an explicit value. A one-sided discovery is
therefore a disagreement, never silently absent. Row keys are
`(window_id,roster_track_id,camera_id,frame)`; fragment/merge rows remain
unknown until adjudication.

Exact agreement requires neither value is `not_found`, transition within one
frame, matching states/order, and mean polygon IoU `>=0.70`. Discovery recall,
fragment/merge frequency, and roster-pass agreement are reported separately. A
third blind adjudicator resolves disagreements with raw RGB and both sealed
annotations only; absent adjudication, units remain unknown. Thus every roster
track and row has two explicit responses, while discovery misses remain visible
evidence rather than an inflated double-annotation claim.

Boundary/region reference units are exactly visible polygon rows. Ordering,
cross-view, and temporal denominators are explicit non-unknown rows/pairs sealed
before prediction. Hidden polygons are never invented for occluded frames.

Prediction/reference track matching requires mean cam00 visible-frame IoU
`>=0.30` and mean named-source-anchor IoU `>=0.30`, with weight
`0.75*target+0.25*source`. Maximum-weight assignment maximizes the exact binary64 weight sum, then among
equal-primary optima chooses the lexicographically smallest sorted
`(scene,window_id,predicted_track_id,reference_track_id)` edge list; numeric
epsilon perturbations are prohibited. Unmatched references are FN; unmatched
predicted fragments in reference windows are FP.

## Metrics and Gate A

Transitions match one-to-one by type within +/-1 frame, sorted by absolute
offset then predicted/reference frame and ID. Every prediction belongs to one
candidate window. Primary event P/R/F1 macro-averages contributing windows
(any reference or prediction), then scenes; true-negative windows are reported
in specificity, while false-positive-only windows contribute zero. Pooled counts
and positive-track recall are secondary.

Before annotations open, an `event_candidate_track` is a stable cam00 ID with an
accepted visible observation, accepted occluded observation, cam00 reveal/hide,
distinct foreground/rear order, and support from at least three training
cameras. Reappearance alone is insufficient. On spatial-complete frames, the
prediction union is every accepted visible-region pixel from every such track,
including unmatched tracks and fragment siblings. The reference union is every
evaluable visible rear polygon.

Compute region TP/FP/FN and directional boundary-hit/denominator counts per
frame. Pool counts across spatial-complete frames within each candidate window,
then derive P/R/F1/IoU. A window contributes when either pooled union is nonempty;
prediction-only and reference-only windows score zero for the missing side,
while completed empty/empty windows are true negatives and excluded from the
F1/IoU macro. Primary scene metrics are unweighted arithmetic means of
contributing window metrics. Overall summaries are unweighted means of scene
metrics, and every transfer scene must pass. Pooled micro counts and per-track
matched localization are secondary.

Boundary P/R/F1 uses 4 px primary and 2/8 px sensitivities. Ordering score is
signed `(z_rear-z_foreground)/m`; AUROC duplicates each directed order unit with
reversed orientation and score sign. Report conditional accuracy/AUROC and
coverage. Abstention is a recall miss and remains in coverage denominators.

Cross-view inconsistency is camera-specific state error: a pair is wrong when
either predicted visible/occluded state disagrees with its human state or
abstains. Temporal inconsistency is transition-occurrence XOR error on
predeclared adjacent rows, where reveal/hide is occurrence 1 and none is 0;
direction/timing remains in event F1. These metrics never use the method's own
accepted edges as truth. Compactness reports selected
fraction, components/MP, perimeter-squared/(4*pi*area), and localization PR.

Relative inconsistency reduction r passes only when baseline error `E_b>0` and
method error `E_m <= (1-r)E_b`. If `E_b=0`, improvement is undefined and cannot
pass. Event and spatial bootstrap intervals resample candidate windows within
scene; ordering/calibration intervals resample tracks. Both use 10,000
replicates and seed 20260715, reapplying contribution rules per replicate.

Engineering admission on frozen cut development is a conjunction:

- controlled fixtures pass;
- ordering accuracy >=0.70 and AUROC >=0.75 at coverage >=0.60;
- event F1 >=0.45 and recall >=0.60;
- boundary F1 and region IoU each >=0.05 above the matched baseline;
- cross-view and temporal inconsistency each >=15% below the registered
  `R031-MT-support-v1` comparator;
- ordering and transition ECE <=0.15;
- evaluable support for every event and nonzero recall on >=80% of tracks.

After freeze, each of flame_steak and sear_steak must independently satisfy:

- ordering accuracy >=0.75 and AUROC >=0.80 at coverage >=0.70;
- event F1 >=0.60 and recall >=0.70;
- boundary F1 and region IoU each >=0.10 above baseline;
- cross-view and temporal inconsistency each >=25% below comparator;
- both ECE <=0.10 and no annotated event family completely missed.

The finite event-family vocabulary is exactly `reveal` and `hide`, derived
from the non-unknown transition label; `none` is the negative class and
`unknown` is excluded/reported. The claim-grade no-family-miss condition
requires nonzero recall for every vocabulary member represented by at least two
reference events in that scene; otherwise that condition is not evaluable.

Missing, undefined, under-covered, or non-evaluable required metrics cannot pass.

## Registered baselines

R031/R032/R033 retain exact historical code/command/manifest hashes and
parameters in the Phase 9 baseline contract. On cut calibration, enumerate
unique score thresholds plus infinities; choose minimum absolute accepted-
fraction difference from CSVL risk 0.75, breaking ties by higher threshold then
lower baseline ID. Select the strongest by 4-px boundary F1, region IoU, then
R031<R032<R033. Seal before development/transfer.

`R031-MT-support-v1` runs exact R031 independently per annotated camera. Its
dedicated global threshold uses all valid pixels in every non-unknown
cut-calibration state aperture. The target fraction is accepted CSVL aperture
pixels at risk <=0.75 divided by the same valid aperture pixels. Enumerate unique
R031-MT scores plus infinities, minimize absolute selected-fraction difference,
and break ties by higher threshold. Seal population IDs, score/source hashes,
threshold, and fractions. Inside each aperture predict visible iff selected
fraction `>=0.5`, otherwise occluded. Cross-view pair error and temporal
occurrence error use the same predeclared rows. It emits no order and cannot
enter inference. If genuine state apertures are unavailable, the relative
criteria and Gate A are not evaluable.

## Conformance and implementation admission

Static fixtures cover camera round trips, off-axis z/ray distance, target
exclusion, group selection, covariance/fusion/exclusivity, two-layer order,
patch/raster regions, flow direction/cycle, split/merge, gap/reappearance,
risk domain, annotation/matching/FP aggregation, baselines, schemas, hashing,
aggregate/rejected source-node IDs, equal-primary assignment ties, repeated
smallest PCA eigenvalues, exact-zero normal orientation, and failure paths.

The Slurm DA3 conformance has:

- analytic 64x48 K (`fx=50,fy=52,cx=31.5,cy=23.5`), constant z=2, exact center/
  corner projection tolerances (`1e-10` pixel, `1e-12` z) and a translated/yawed
  second camera round trip (`1e-9`);
- cut frame 0; generate one group for every anchor, then choose the lower camera
  ID appearing in at least two group tuples and the first two lexicographic
  tuples containing it; finite depth/confidence >=99%, positive depth >=95%,
  processed-K corner error <=0.5 px and duplicate relative MAD <=0.05. Record
  both repetitions' exact raw array/payload hashes and require numerical
  agreement within absolute/relative `1e-5`; hash inequality is not called a
  repeatability failure when the numerical test passes.

After conformance, one registered production inference writes and seals the
exact raw DA3 sidecar hashes. Every downstream stage reads those immutable
arrays and requires exact input/output provenance hashes; it never reruns DA3
inside fusion/scoring. Regeneration is a new versioned cycle. Deterministic
re-execution tests begin from the same sealed sidecars and require exact
scientific-payload hashes.

Until the Slurm weight hash and both conformance parts pass, real depth inference
fails closed. Until genuine annotations exist, real Gate A remains
`not_evaluable`. Those prerequisites do not block implementing and reviewing
Slice A software or producing the empty annotation packet.

## Deferred Slice B

The first representation intervention, if separately admitted, is one fixed
post-densification in-place preservation/reassignment transaction in the dynamic
4D bank. It must keep total dynamic plus hard-static rows and the integrated
point budget matched, reset every reassigned per-row optimizer moment, preserve
tensor-level step/state, disable other topology changes after the common
checkpoint, and be crash/restart deterministic. Capacity-only, strong-reference,
visibility-only, coupled, and misaligned controls remain required. This page
does not authorize or specify trainer mutation; the Phase 9 plan registers the
review checkpoint that must precede it.
