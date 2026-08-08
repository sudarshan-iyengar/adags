---
type: paper
node_id: paper:mazur2026_4dpm
title: "4D Primitive-Mâché: Glueing Primitives for Persistent 4D Scene Reconstruction"
authors: ["Kirill Mazur", "Marwan Taher", "Andrew J. Davison"]
year: 2026
venue: "CVPR 2026 (Oral)"
external_ids:
  arxiv: "2512.16564"
tags: [dynamic-scene, object-permanence, occlusion, primitives, monocular]
status: deep-dived
---

# 4D Primitive-Mâché: Glueing Primitives for Persistent 4D Scene Reconstruction

**Paper:** https://arxiv.org/abs/2512.16564
**Code:** https://github.com/makezur/4D_PM
**Base method:** Not built on 3D/4D Gaussian Splatting or any deformation-MLP
representation. 4DPM composes three frozen, pretrained front-end priors — π³
(Wang et al. 2025) for feed-forward monocular point-map estimation,
SuperPrimitive (Mazur et al. 2024) for first-frame surface oversegmentation
plus SAMv2 (Ravi et al. 2024) for mask propagation, and AllTracker (Harley et
al. 2025) for dense 2D point tracking — and adds no new learned network of its
own. Its actual contribution is a from-scratch Gauss-Newton SE(3) pose-graph
backend and a hand-designed, non-learned motion-extrapolation /
object-permanence stage layered on top of these priors.

## One-line thesis
Reducing monocular dynamic-scene motion to a sparse set of per-primitive SE(3)
poses (instead of a dense per-pixel or per-Gaussian motion field) turns 4D
reconstruction into a low-dimensional, well-conditioned least-squares problem
that a Gauss-Newton solver can fit from dense 2D correspondences alone — and
because poses are stored per primitive rather than baked into a single
timestep, any previously observed primitive can be remapped and replayed at
every later timestep, even while fully occluded.

## Problem / Gap
Prior monocular/pairwise dynamic reconstruction methods (DUSt3R-style aligners
such as St4Track and POMATO, and trajectory-field methods such as TraceAny)
recover geometry only at the moment a region was observed: once a region
leaves the frame or is occluded, it has no representation at other timestamps,
and pairwise DUSt3R-style methods additionally scale quadratically with frame
count while real dynamic-scene training data "remains extremely sparse."
4DPM targets exactly this gap: making every previously observed piece of
geometry queryable ("replayable") at every timestep, not only the one it was
captured in.

## Method
The frontend runs π³ over the video to get per-keyframe point maps, segments
the first keyframe into primitive masks with SuperPrimitive and propagates
those masks across keyframes with SAMv2, and extracts dense 2D correspondences
between nearby keyframes with AllTracker, filtered per-object and weighted by
per-pixel confidence. Primitives with low initial correspondence residual are
classified static and frozen out of optimization. For each remaining dynamic
object, the backend solves for one SE(3) pose per primitive instance by
minimizing a Huber-robust, confidence-weighted 3D point-alignment energy
across temporally adjacent primitive pairs, using Gauss-Newton with
analytically derived Jacobians (gauge freedom fixed by anchoring the
last-observed primitive's pose to identity). Because every primitive pose is
expressed relative to a shared reference frame, any primitive can then be
projected into any other observed timestamp via a composed SE(3) transform,
giving full 4D replay. For primitives that become fully occluded, a separate
motion-segmentation step links each object to a co-moving "parent" using
inflated-bounding-box spatial contact and Mahalanobis-distance velocity
clustering, and the occluded child inherits the parent's pose trajectory
(transitively through parent chains) instead of being deleted or frozen.

## Assumptions
Input is casual monocular RGB video; the scene is assumed piecewise-rigid, so
non-rigid deformation is out of scope (HO3D hand geometry is approximated as
"largely static," which only approximately holds). The method depends on three
frozen pretrained priors — π³ point maps, SuperPrimitive+SAMv2 masks, and
AllTracker correspondences — being reasonably accurate on the target domain,
which in evaluation and the released configs is tabletop / hand-object /
robot-manipulation scenes (HO3D, a hand-collected multi-object dataset, and
Franka-arm/drawer/fridge scenes). Motion extrapolation for occluded content
further assumes the occluded object maintains a physically consistent
contact or shared-velocity relationship with a still-visible parent object.

## Limitations / Failure Modes
The paper states the system "assumes that each primitive is rigid and thus
cannot represent more intricate non-rigid deformations," and that incremental
mapping over extended sequences "have yet to be explored" — the backend
operates over a fixed observation window rather than online. The
video-segmentation frontend (SAMv2 mask propagation over 50-100 objects)
dominates runtime at roughly 42s, so the system is far from real-time even
though the backend optimization itself is fast (~2-10s). Reported per-scene
F-scores vary substantially (e.g., as low as 0.64 on the hardest Multi-Object
subset "PanStir" versus 0.82-0.90 elsewhere) without an explicit breakdown of
why those scenes are harder.

## Reusable Ingredients
- **Sparse per-primitive SE(3) pose optimization instead of dense motion
  fields** — collapses dynamic motion estimation to a low-dimensional
  Gauss-Newton problem; the paper's own optimizer study shows 10-step
  Gauss-Newton beating 10,000-step Adam on F-score while being ~40x faster.
- **Huber-robust, confidence-weighted point-alignment energy over warped point
  maps** — a generic way to fuse dense 2D tracker output with feed-forward
  monocular point-map priors into rigid relative poses.
- **Composed-SE(3) time remapping** — lets any observed primitive be queried
  at any other timestamp from stored poses alone, without re-running the
  frontend, yielding "replay" behavior from a pose-only representation.
- **Parent-child occlusion motion extrapolation** (inflated-OBB spatial
  contact + Mahalanobis velocity clustering) — a concrete, non-learned
  mechanism for propagating motion to occluded primitives that could be
  adapted as a standalone heuristic for what happens to occluded geometry.
- **Residual-threshold static/dynamic primitive classification** — a cheap
  pre-filter that avoids spending optimization budget on rigid background.

---

### Deep Dive

#### Core Novelty
Relative to prior monocular/DUSt3R-family 4D reconstruction (St4Track,
POMATO), which estimates geometry pairwise or in short windows and only at
observation time, 4DPM's changes are (1) replacing dense/implicit motion
representations with an explicit sparse graph of rigid primitive poses solved
by Gauss-Newton rather than gradient descent, and (2) adding an explicit,
non-learned parent-child motion-extrapolation step so primitives that become
fully occluded still receive a pose estimate at every later timestep. The key
insight: because everyday dynamic content (hands, tools, drawers, objects) is
well approximated as piecewise-rigid, the true degrees of freedom are
per-object SE(3) trajectories rather than per-pixel or per-Gaussian flow,
collapsing the problem to a low-dimensional, well-conditioned least-squares
system a second-order solver fits in seconds.

#### Mathematical Formulation

**Per-object alignment energy (evaluated per dynamic object, during backend
optimization):**
$$E(\mathcal{O}) = \sum_{(i,j)\in\mathcal{T}(\mathcal{O})} \left\| w_{ij} \cdot S_i \cdot S_j^{\wedge} \cdot \left(T_j^{-1}T_i X_i - X_j^{\wedge}\right) \right\|_\rho$$
- $\mathcal{O}$: an object, i.e. the set of primitive instances tracked for
  one rigid part across keyframes.
- $\mathcal{T}(\mathcal{O})$: set of temporally-adjacent primitive-instance
  pairs $(i,j)$ within object $\mathcal{O}$.
- $X_i, X_j$: per-keyframe 3D point maps (from π³) for primitive instances
  $i$ and $j$; $X_j^{\wedge}$ is $X_j$ warped into instance $i$'s pixel frame
  via the dense 2D correspondence.
- $S_i, S_j^{\wedge}$: (warped) binary segmentation masks restricting the
  residual to the shared, valid primitive region.
- $w_{ij}$: per-pixel correspondence confidence weights from the tracker.
- $T_i, T_j \in SE(3)$: the unknown poses mapping each primitive instance to
  the shared (last-observed-frame) coordinate system — the optimization
  variables.
- $\|\cdot\|_\rho$: Huber robust loss, downweighting outlier residuals (e.g.
  from bad correspondences or segmentation drift).

**Total energy (summed over all non-static objects, defines the full backend
objective):**
$$E_{\text{final}} = \sum_i E(\mathcal{O}_i)$$

**Gauss-Newton update (the per-iteration solve of the backend optimizer):**
$$J^\top W J\,\tau = -J^\top W r, \qquad T_i \leftarrow T_i \oplus \tau$$
where $r$ is the stacked residual vector, $W$ the IRLS robust weight matrix
from the Huber kernel, $J$ the analytically-derived Jacobian of the residual
w.r.t. the Lie-algebra perturbation $\tau \in \mathfrak{se}(3) \simeq
\mathbb{R}^6$, and $\oplus$ the SE(3) retraction (left/right update via the
exponential map).

**Time remapping (evaluated post-optimization, to query a primitive observed
at keyframe $p$ at any other observed timestamp $g$):**
$$T^{p \mapsto g} := \left[T(S^g)\right]^{-1} T(S^p)$$
This composes the two primitives' poses (both expressed relative to the
shared last-observed frame) to get the relative transform that places
primitive $p$'s geometry into timestamp $g$'s coordinate frame — this is the
mechanism that produces full 4D "replay."

**Gauge invariance of relative velocity (used to justify comparing motion
across objects with independent, arbitrarily-gauged pose chains during motion
segmentation):**
$$T'(t)^{-1}T'(t-1) = T(t)^{-1}T(t-1)$$
i.e. the relative SE(3) velocity $V = T(t)^{-1}T(t-1)$ between consecutive
poses of the same primitive is invariant to the arbitrary reference-frame
gauge choice, so velocities from different objects/gauges can be compared
directly.

#### Algorithm / Pipeline Changes
1. **Frontend geometry:** Run π³ over the input video to predict per-keyframe
   3D point maps (keyframe interval 15, per the released configs).
2. **Frontend segmentation:** Oversegment the first keyframe into primitives
   with SuperPrimitive; propagate primitive masks to later keyframes with
   SAMv2 within a sliding window (window size 15 in the released configs).
3. **Frontend correspondence:** Run AllTracker between nearby keyframes
   (released configs use a 10-frame support window) to get dense 2D tracks;
   filter to same-object correspondences using the propagated masks and keep
   per-pixel confidence weights $w_{ij}$.
4. **Static/dynamic classification:** Primitives whose initial correspondence
   residual (against a rigid/identity assumption) falls below a threshold are
   classified static and frozen out of the per-object optimization, reducing
   the problem to only genuinely moving primitives.
5. **Backend per-object pose optimization:** For each dynamic object, collect
   its primitive instances across keyframes into the pair set
   $\mathcal{T}(\mathcal{O})$ and solve for one SE(3) pose per instance by
   minimizing $E(\mathcal{O})$ via Gauss-Newton (10-50 iterations), with the
   pose of the primitive instance nearest the last-observed keyframe pinned to
   identity to fix gauge freedom.
6. **Time remapping:** Using the composed-transform formula above, any
   primitive instance's geometry can be projected into any other observed
   timestamp, producing a full persistent 4D reconstruction replayable at
   every timestep rather than only at capture time.
7. **Motion segmentation / occlusion extrapolation:** Build a parent-child
   object graph from (a) spatial contact — oriented bounding boxes (OBBs) per
   object per timestamp, inflated by a padding factor $\alpha=1.1$, linked
   when they intersect (applied transitively for indirect contact chains),
   and (b) velocity clustering — relative SE(3) velocities compared via
   Mahalanobis distance over $\log(V^{-1}W)$ with per-axis (translation,
   rotation) covariance. When a primitive becomes fully occluded, it inherits
   its assigned parent's pose trajectory (transitively through the parent
   chain) instead of being deleted or frozen at its last pose.

#### Key Hyperparameters & Design Choices
- **Optimizer / iterations:** Gauss-Newton, 10-50 iterations (paper Table 3);
  no learning rate applies — this is a per-scene least-squares solve, not
  gradient-descent training, and the frontend priors (π³, SAM2, AllTracker)
  are used frozen/pretrained rather than fine-tuned.
- **Loss weighting for the novel energy term:** only the per-pixel
  correspondence confidence weights $w_{ij}$ and the Huber robustifier; no
  additional weighted auxiliary loss terms are combined with $E(\mathcal{O})$.
- **Huber loss parameter:** not given a numeric value in the paper text; the
  released code's `huber(r, delta=1.345, eps=1e-7)` default is the classical
  statistics default for 95% efficiency under Gaussian noise.
- **Spatial-contact OBB inflation:** $\alpha = 1.1$ (stated in the paper).
- **Velocity-clustering covariance ($\sigma_\tau$, $\sigma_\psi$):** not
  specified numerically in the paper; not confirmed in the inspected code
  (the relevant `motion_segmentation.py` functions expose separate
  `tau_thr=0.1`, `phi_thr=0.1`, `pass_fraction=0.8` constants whose exact
  correspondence to the paper's Mahalanobis covariance formulation could not
  be confirmed from the excerpt inspected).
- **Backend image resolution:** 512×512 (~10s runtime) or 256×256
  (~2.5s, lower accuracy) — explicit efficiency/accuracy trade-off.
- **Evaluation threshold:** 1cm distance threshold for precision/recall/F-score.
- **Architecture dimensions:** not applicable — 4DPM introduces no novel
  neural network; all learned components (π³, SAM2, AllTracker) are
  pre-existing, frozen architectures.
- Released-repo config values not stated in the paper text at all: keyframe
  interval `kf_interval=15`, correspondence `window_size=15`, AllTracker
  `inference_iters=4`, `conf_thr=0.5`, `visibility_thr=0.5`,
  `num_supp_frames=10`; static classifier `residual_thr=0.07`,
  `dynamic_segmentation_thr=0.1`; π³ `confidence_thr=0.1`; frontend
  `min_segment=200` px, `num_pts=300`, `num_pts_active=250`, `pre_erode=1`;
  SAM `box_nms_thresh=0.8`, `iou_threshold=0.5`, `stability_threshold=0.6`.

#### Ablation Summary
No per-component ablation (e.g. removing motion segmentation, or removing
Huber weighting) is presented in the paper. The closest analysis is an
Appendix optimizer/iteration-count study on the backend solver (Table 3),
which is the single most impactful design choice reported:
- Adam, 500 steps: F = 0.6342 (20s)
- Adam, 1,000 steps: F = 0.6474 (40s)
- Adam, 10,000 steps: F = 0.7228 (400s)
- **Gauss-Newton, 10 steps: F = 0.7843 (2s)** — already beats 10,000-step Adam
  while being ~200x faster.
- **Gauss-Newton, 50 steps: F = 0.7948 (10s)** — the adopted final
  configuration, matching the headline Multi-Object result.
Most impactful factor: switching the backend from first-order (Adam) to
second-order (Gauss-Newton) optimization, not any architectural component.

#### Implementation Reality
- **Framework:** PyTorch 2.6 / CUDA 12.4, with custom CUDA kernels
  (`core/src/`, `object_mapper/csrc/`) and the `lietorch` library for SE(3)
  Lie-group math. This is a from-scratch optimization pipeline over frozen
  pretrained priors, not a fork of a Gaussian-splatting or NeRF codebase — no
  differentiable rasterizer is involved.
- **Key files:**
  - `core/dense_optim_GN.py` — main Gauss-Newton backend solver (the
    $E(\mathcal{O})$ minimization).
  - `core/jacobians.py`, `core/hessian.py`, `core/linearisation.py` —
    analytical Jacobian/Hessian construction for the alignment energy.
  - `core/robust_cost.py` — Huber weighting (`delta=1.345`, `eps=1e-7`).
  - `core/is_static.py` — static/dynamic primitive classification.
  - `core/umeyama_tracking.py` — rigid (Umeyama) alignment, likely used for
    pose initialization before Gauss-Newton refinement.
  - `core/second_order.py` — the first-order-vs-second-order optimizer study
    (Table 3).
  - `frontend/pi3/wrapper.py`, `frontend/pi3/kf_runner.py` — π³ point-map
    inference at keyframes.
  - `frontend/segment/{oversegmentation,samv2_tools,active_sampling,filter,
    disjoin,video_matcher}.py` — SuperPrimitive-style oversegmentation, SAMv2
    mask propagation, and active point sampling within masks.
  - `frontend/alltracker/*`, `frontend/tracker/*` — vendored AllTracker
    network plus video-level tracking/segment-matching glue.
  - `object_mapper/motion_segmentation.py` — parent/child grouping logic
    (`compare_motion_increments`, `motion_increments_sides`,
    `cluster_static`, `motion_model`, `filter_motion_grouping`).
  - `object_mapper/motion_extender.py` — propagates a parent's trajectory to
    occluded child primitives; the concrete object-permanence mechanism.
  - `object_mapper/time_mapper.py`, `time_mapping_core.py` — the time
    remapping ($T^{p\mapsto g}$) implementation.
  - `object_mapper/{bbox,bbox_iou,box3d_iou,object_overlaps}.py` — OBB
    construction/intersection for spatial-contact linking.
  - `evals/{eval_franka,metrics,run_eval_sequences}.py` — 1cm
    precision/recall/F-score evaluation harness.
- **Notable implementation details not in the paper text:** the Huber delta
  (1.345) is never given a number in the paper; keyframe interval (15),
  correspondence window (15 keyframes / 10 support frames), and the static
  classifier's two numeric thresholds (`residual_thr=0.07`,
  `dynamic_segmentation_thr=0.1`) are all released-config choices absent from
  the prose, which only describes "freezing objects with high initial
  correspondence residuals" qualitatively. AllTracker is run with only 4
  inference iterations, a speed/accuracy trade-off not discussed in the
  paper. The released configs (`drawer.yaml`, `franka_data.yaml`,
  `fridge.yaml`) target robot-manipulation/tabletop demo scenes rather than
  the exact HO3D/Multi-Object benchmark scenes reported in the paper's
  tables, suggesting the public repo ships qualitative demo configs distinct
  from the internal benchmark-reproduction configs. The config's
  `alignment.gauge_freedom: 'last'` confirms the paper's stated choice of
  anchoring the last-observed primitive's pose to identity.

#### Failure Modes & Limitations
The paper explicitly states the system "assumes that each primitive is rigid
and thus cannot represent more intricate non-rigid deformations," and that
"incremental mapping capabilities, where the scene representation is built
and updated over extended sequences, have yet to be explored" — i.e. no
online/incremental operation over long sequences. It also notes, in
discussing the broader problem setting, that fully general "persistent,
non-rigid dynamic reconstruction remains very challenging even for RGB-D
systems and can only be performed on short sequences in controlled
scenarios," implicitly bounding what monocular piecewise-rigid 4DPM can be
expected to handle. Practically, the video-segmentation frontend (SAMv2 mask
propagation over 50-100 objects) dominates total runtime at ~42s, so despite
a fast 2-10s backend the full system is far from real-time. HO3D evaluation
treats hand geometry as "largely static," an approximation that only
partially holds given hands are genuinely non-rigid. Per-scene F-scores vary
from as low as 0.64 (Multi-Object subset "PanStir") to 0.90, without an
explicit qualitative or quantitative breakdown of what drives the harder
cases.

---

## Relevance to ADAGS

Closest new precedent for explicit object permanence through occlusion at
the primitive level (CVPR 2026 Oral). Differs from ADAGS: monocular,
rigid-primitive correspondence tracking, motion extrapolation — vs calibrated
multiview evidence and a hybrid 3D/4D Gaussian bank. Any
hide-preserve-reveal claim must be positioned against it alongside
[[papers/ramlal2026_persistgs]] and [[papers/zheng2025_gaustar]].

## Connections

- Pressures [[gap_map#G13 - Visibility Events Are Not Smooth Deformation]]
- Pressures [[ideas/event-causal-visibility-gaussians]]

## Sources

- https://arxiv.org/abs/2512.16564
- https://github.com/makezur/4D_PM
