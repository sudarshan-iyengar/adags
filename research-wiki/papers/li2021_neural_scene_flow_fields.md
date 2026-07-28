---
type: paper
node_id: paper:li2021_neural_scene_flow_fields
title: "Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes"
authors: ["Zhengqi Li", "Simon Niklaus", "Noah Snavely", "Oliver Wang"]
year: 2021
venue: "CVPR"
external_ids:
  arxiv: "2011.13084"
  doi: null
  s2: null
tags: ["scene-flow", "dynamic-radiance-field", "disocclusion", "temporal-consistency"]
added: 2026-07-14T22:18:30Z
---

# Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes

**Paper:** https://arxiv.org/abs/2011.13084
**Code:** https://github.com/zhengqili/Neural-Scene-Flow-Fields
**Base method:** NeRF (Mildenhall et al. 2020), extended with a per-time-instant dynamic radiance field plus a separate static NeRF blended via a learned visibility map.

## One-line thesis

Adding a per-Gaussian-analog (per-point) forward/backward 3D scene-flow field to a time-conditioned NeRF, and learning a scalar disocclusion-confidence weight alongside it, lets temporal photometric supervision be reprojected across neighboring frames while automatically downweighting the loss at points where that reprojection is invalid (i.e. at disocclusion/occlusion boundaries) — turning monocular video into a multi-frame multi-view-like constraint without requiring the reprojected content to actually be visible everywhere.

## Problem / Gap

Monocular dynamic view synthesis is severely underconstrained: a single moving camera observes each dynamic 3D point at only one time and one viewpoint, so there is no direct multi-view constraint on non-rigid, time-varying geometry the way there is for a static scene. Prior NeRF variants that simply condition on time (a "NeRF w/ time" baseline) have no mechanism forcing temporally nearby observations of the same physical surface to agree, so they overfit per-frame appearance instead of learning consistent 3D geometry and motion. Naively enforcing temporal photometric consistency by warping through scene flow also fails at disocclusion boundaries, where a 3D point visible at time $i$ may not correspond to any valid point at time $j$, so an unweighted reprojection loss injects incorrect supervision exactly at motion boundaries.

## Method

NSFF represents a dynamic scene with a time-conditioned MLP that, for a 3D point and viewing direction at frame $i$, outputs color, density, forward and backward 3D scene-flow vectors, and forward/backward disocclusion weights. During training, each point is displaced by its scene flow to its predicted position at a neighboring frame $j$, the color/density field at frame $j$ is queried at that displaced location, and the result is volume-rendered and compared to the actual observed frame $j$ image along the same camera ray — with the comparison weighted down by the disocclusion weight so that invalid warps (occluded/disoccluded content) contribute less loss. Cycle consistency between forward and backward flow, scene-flow smoothness/magnitude regularization, and data-driven priors from off-the-shelf single-view depth (MiDaS) and optical flow (RAFT) supervise the flow and geometry, with the data-driven prior terms linearly decayed to zero over training so they only aid initialization. A separate static NeRF is blended with the dynamic field via a learned per-point visibility/blend weight so that static, multi-view-consistent regions can fall back on ordinary multi-view constraints instead of relying on the harder monocular dynamic supervision.

## Assumptions

Requires known camera poses (COLMAP-style calibration) for a single monocular video, and assumes the scene is mostly forward-facing / boundable in normalized device coordinates (the released code operates in NDC space and supports forward-facing scenes only). It also assumes off-the-shelf monocular depth (MiDaS) and optical flow (RAFT) estimates are usable, even if imperfect, as short-lived initialization priors.

## Limitations / Failure Modes

The disocclusion weight mechanism mostly suppresses invalid *loss* at occlusion boundaries; it does not give the model any positive signal or extra capacity to actually reconstruct content that is never observed, so newly revealed or long-hidden surfaces are not guaranteed to be well represented. Quality degrades on longer sequences (the paper notes degradation beyond roughly 2 seconds of motion) and under extreme/fast motion, and the method is reportedly sensitive to degenerate cases where camera motion and object motion are colinear. Per-scene optimization is expensive: ~2 days on 2 V100s (paper) or ~2 days on 4 RTX 2080Ti (released code), with ~6 seconds to render a single 512×288 frame.

## Reusable Ingredients

- **Forward/backward scene-flow-warped temporal reprojection**: displace a 3D point by learned flow to a neighboring frame's field and re-render, giving a temporal photometric constraint without extra cameras.
- **Explicit disocclusion-confidence weight per flow direction**: a learned scalar in $[0,1]$ that downweights reprojection loss at invalid correspondences, with an L1 regularizer pulling it toward 1 so it can't trivially zero out all supervision.
- **Cycle-consistency loss on bidirectional flow**: penalizes $\mathbf{f}_{i\to j}(\mathbf{x}_i) + \mathbf{f}_{j\to i}(\mathbf{x}_{i\to j})$, a cheap self-supervised check that forward/backward flow agree.
- **Decayed data-driven priors**: using single-view depth and optical flow only as an initialization-stage loss (linearly decayed to zero) rather than a permanent constraint, so early optimization is well-conditioned without permanently trusting noisy off-the-shelf priors.
- **Static/dynamic field blending via a learned visibility weight**: lets multi-view-consistent static regions use ordinary multi-view NeRF constraints while only paying the harder monocular-dynamic cost where content actually moves.

---

### Deep Dive

#### Core Novelty
Relative to plain time-conditioned NeRF, NSFF's change is to make the network predict *3D scene flow and a disocclusion confidence* alongside color/density, and to use that flow to warp query points across time so temporally neighboring frames supervise each other. The key insight is that a single moving monocular camera, treated naively, gives no multi-view constraint on dynamic geometry — but if the model also predicts where a point moves to at $t\pm1, t\pm2$, then the *same* camera's frames at other times become pseudo-multi-view observations of that point, provided the loss is aware of where that correspondence is actually valid (the disocclusion weight).

#### Mathematical Formulation

Time-conditioned dynamic field (per point, per frame index $i$), evaluated at every query point before volume rendering:
$$(\mathbf{c}_i, \sigma_i, \mathcal{F}_i, \mathcal{W}_i) = F^{dy}_\Theta(\mathbf{x}, \mathbf{d}, i)$$
where $\mathcal{F}_i = (\mathbf{f}_{i\to i+1}, \mathbf{f}_{i\to i-1})$ is forward/backward 3D scene flow and $\mathcal{W}_i = (w_{i\to i+1}, w_{i\to i-1}) \in [0,1]$ are the paired disocclusion weights.

Temporal point warping (used to look up the field at a neighboring frame $j$ from a point observed at frame $i$):
$$\mathbf{r}_{i\to j}(t) = \mathbf{r}_i(t) + \mathbf{f}_{i\to j}(\mathbf{r}_i(t))$$
The warped point is then queried in frame $j$'s field, $\sigma^j(\mathbf{r}_{i\to j}(t))\, \mathbf{c}^j(\mathbf{r}_{i\to j}(t), \mathbf{d}_i)$, and volume-rendered — this is the mechanism that turns a neighboring frame into a supervisory signal for the current frame's ray.

Disocclusion-weighted rendered accumulation, used to modulate the photometric loss (Eq. 8 region):
$$\hat{W}_{j\to i}(\mathbf{r}_i) = \int T_j(t)\, \sigma_j(\mathbf{r}_{i\to j}(t))\, w_{i\to j}(\mathbf{r}_i(t))\, dt$$
Low values indicate the warp is not trustworthy (occlusion/disocclusion), so the corresponding photometric term is downweighted; an L1 regularizer additionally pulls $w$ toward 1 to prevent the network from trivially discounting all supervision.

Cycle-consistency loss on bidirectional flow (Eq. 9), evaluated as a training-time regularizer, not at render time:
$$\mathcal{L}_{cyc} = \sum w_{i\to j}\, \lVert \mathbf{f}_{i\to j}(\mathbf{x}_i) + \mathbf{f}_{j\to i}(\mathbf{x}_{i\to j}) \rVert_1$$

Combined training objective (Eq. 12-15 region):
$$\mathcal{L} = \mathcal{L}_{cb} + \mathcal{L}_{pho} + \beta_{cyc}\mathcal{L}_{cyc} + \beta_{data}\mathcal{L}_{data} + \beta_{reg}\mathcal{L}_{reg}$$
where $\mathcal{L}_{pho}$ is the disocclusion-weighted temporal photometric term, $\mathcal{L}_{data}$ bundles the single-view-depth and optical-flow data priors (with $\beta_z = 2$ weighting the depth term, and the whole $\mathcal{L}_{data}$ term linearly decayed to zero during training), and $\mathcal{L}_{reg}$ bundles the flow smoothness/magnitude and disocclusion-weight-toward-1 regularizers. A separate static-field blend weight $v(t)$ combines the dynamic and a standalone static NeRF so static, multi-view-consistent content can be explained by ordinary multi-view constraints.

#### Algorithm / Pipeline Changes
1. Precompute per-frame camera poses (COLMAP), monocular depth (MiDaS) and optical flow (RAFT) for the input monocular video — used only as decayed initialization priors, not fixed geometry.
2. At each training step, sample a ray at frame $i$; query the dynamic MLP $F^{dy}_\Theta(\mathbf{x}, \mathbf{d}, i)$ at points along the ray to get color, density, forward/backward scene flow, and disocclusion weights.
3. For each neighbor $j$ in the temporal window $\mathcal{N}(i) = \{i, i\pm1, i\pm2\}$ (flow "chained" across steps for the $\pm2$ case), warp sampled points via $\mathbf{r}_{i\to j}(t) = \mathbf{r}_i(t) + \mathbf{f}_{i\to j}(\mathbf{r}_i(t))$ and re-query the field at frame $j$ to render a reprojected image of ray $i$ as seen through frame $j$'s geometry/appearance.
4. Compare the reprojected render to the true frame-$i$ observation, weighting the per-ray loss by the accumulated disocclusion weight $\hat{W}_{j\to i}(\mathbf{r}_i)$ so invalid (occluded/disoccluded) correspondences contribute less.
5. Add cycle-consistency, scene-flow smoothness/magnitude, and disocclusion-weight-toward-1 regularization terms (all pure training-time losses, no change to inference).
6. Add decayed data-driven terms: reproject scene-flow-displaced 3D points to 2D and compare against RAFT optical flow (geometric consistency), and compare rendered depth to MiDaS depth with a scale-shift-invariant loss; both terms' weight is linearly annealed to zero over training.
7. At render/inference time, blend the dynamic field's output with a separately trained static NeRF using the learned blend weight $v(t)$, then volume-render as usual — the scene-flow and disocclusion machinery is training-time-only supervision, not part of the final rendering pipeline itself.

#### Key Hyperparameters & Design Choices
- Optimizer: Adam, learning rate $5\times10^{-4}$.
- Ray sampling: 128 points per ray (paper); released code exposes `N_samples` and recommends 256-512 for higher-resolution rendering.
- Time input normalized to $i \in [0,1]$.
- Temporal neighborhood $\mathcal{N}(i) = \{i, i\pm1, i\pm2\}$, with a `chain_sf` flag in the released code toggling 3-frame vs. 5-frame consistency.
- Depth-loss weight $\beta_z = 2$; data-prior loss weights in the released code default to $(w_{depth}, w_{optical\_flow}) \in \{(0.4, 0.2), (0.2, 0.1)\}$ ("usually work the best").
- Data-driven prior terms ($\mathcal{L}_{data}$) linearly decay to zero during training (decay duration controlled by `decay_iteration` in the released code) — used for initialization only.
- Network width: paper does not give exact MLP layer/hidden-dim counts in the main text (deferred to supplemental material, not accessed here); released code exposes a `netwidth` parameter, with 512 suggested for longer sequences.
- Positional encoding: standard NeRF-style positional encoding on inputs, with scenes parameterized in normalized device coordinates; exact frequency counts not specified in the accessible text.
- Depth prior model: MiDaS (Ranftl et al., cited as [58] in the paper). Optical flow prior model: RAFT (per the released code; not necessarily named in the main paper text).
- Training cost: ~2 days on 2 NVIDIA V100 GPUs (paper) / ~2 days on 4 NVIDIA RTX 2080Ti GPUs (released code) per scene; ~6 seconds to render one 512×288 frame.

#### Ablation Summary
(Dynamic-only regions, SSIM↑ / LPIPS↓, Table 3 — component removed relative to the full model)
1. NeRF w/ time (baseline, no scene flow/disocclusion machinery at all): 0.630 / 0.159.
2. w/o $\mathcal{L}_z$ (single-view depth prior): 0.710 / 0.132.
3. w/o $\mathcal{L}_{geo}$ (optical-flow geometric consistency prior): 0.713 / 0.139.
4. w/o $\mathcal{L}_{cyc}$ (cycle consistency): 0.731 / 0.115.
5. w/o $\mathcal{L}_{reg}$ (flow smoothness/magnitude regularization): 0.751 / 0.110.
6. w/o $\mathcal{W}_i$ (disocclusion weights): 0.754 / 0.112.
7. **Full model w/ static-field blending: 0.758 / 0.097 (best).**
The single largest jump is baseline → adding any scene-flow machinery at all (0.630 → ≥0.710 SSIM), i.e. the core scene-flow-warped temporal reprojection mechanism itself is the dominant contribution; each individual auxiliary term (depth prior, geometric prior, cycle consistency, regularization, disocclusion weights) contributes a comparatively small additional increment on top of that.

#### Implementation Reality
- **Framework:** PyTorch, custom NeRF-family codebase (not built on the original TensorFlow NeRF release).
- **Key files:** the public repo (`zhengqili/Neural-Scene-Flow-Fields`) includes `run_midas.py` (precomputes MiDaS single-view depth) and `run_flows_video.py` (precomputes RAFT optical flow) as offline preprocessing scripts feeding the decayed data-driven priors.
- **Notable implementation details:** the released code operates exclusively in normalized device coordinates and is documented as supporting forward-facing scenes only, a scene-type restriction not emphasized in the main paper text. The README notes the current default branch includes "improvements for monocular videos in the wild" beyond what's described in the paper, with a separate branch kept matching the paper's original description exactly — i.e. paper and default released code are not identical. Depth/flow priors (MiDaS/RAFT) are named concretely in the code/README though the main paper text only cites the depth method by reference number, not name.

#### Failure Modes & Limitations
Cannot extrapolate genuinely unseen disoccluded content — the disocclusion weight suppresses bad loss but supplies no positive reconstruction signal for surfaces never observed. Performance degrades on sequences longer than roughly 2 seconds and under extreme motion. Sensitive to degenerate configurations where camera motion and object motion are colinear (scene flow becomes unobservable/ambiguous along the epipolar direction). Substantial per-scene compute cost (multi-day training, seconds-per-frame rendering) limits scalability.

---

## Open Questions

Can calibrated multiview N3V observations produce surface visibility evidence without learning a full scene-flow field first?

## Claims

None yet.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

It provides the temporal 3D comparison missing from R031's unwarped same-pixel depth subtraction.
