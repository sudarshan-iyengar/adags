---
type: paper
node_id: paper:zhang2026_vad_gs
title: "Visibility-Aware Densification for 3D Gaussian Splatting in Dynamic Urban Scenes"
authors: ["Yikang Zhang", "Rui Fan"]
year: 2026
venue: "CVPR"
external_ids:
  arxiv: "2510.09364"
  doi: null
  s2: null
tags: ["visibility", "densification", "multi-view-stereo", "geometry-completion"]
added: 2026-07-14T22:18:30Z
---

# Visibility-Aware Densification for 3D Gaussian Splatting in Dynamic Urban Scenes

**Paper:** https://arxiv.org/abs/2510.09364
**Code:** https://github.com/YikangZhang1641/VAD-GS
**Base method:** 3D Gaussian Splatting for dynamic urban scenes (StreetGaussians-style static/dynamic decomposition for Waymo, DriveStudio for nuScenes), itself built on 3D Gaussian Splatting (Kerbl et al.); compared against PVG, OmniRe, and StreetGaussians as baselines.

## One-line thesis

Rasterizing an already-initialized point cloud's voxels via z-buffering exposes exactly which pixels have no reliable underlying geometry (Gaussian-rendered depth absent or far behind voxel depth); reconstructing those specific regions with patch-match MVS from a diversity-selected subset of supporting views lets VAD-GS create new Gaussians at missing structures, rather than relying on clone/split densification that can only propagate from primitives that already exist.

## Problem / Gap

In multi-camera autonomous-driving capture, adjacent synchronous cameras overlap by less than 15%, LiDAR has blind spots beyond its vertical range, and low-texture surfaces give SfM too little signal — so the initial point cloud (and the Gaussians grown from it) leaves entire surfaces uninitialized. Concretely: standard 3DGS densification (clone/split, as in GeoTexDensifier, DNGaussian) can only grow from positions where a primitive already exists, so it cannot manufacture geometry in a region with zero coverage; camera rays through such a region hit nothing, and the resulting photometric error incorrectly propagates gradients to unrelated, already-visible Gaussians instead of the missing surface. GaussianPro's single-camera temporal propagation is also scoped to static scenes and does not handle multi-camera or dynamic-object structure.

## Method

VAD-GS voxelizes the initialized point cloud and rasterizes per-view voxel depth via z-buffering, then compares it against the current Gaussian-rendered depth at each pixel; pixels where Gaussian depth is missing or the depth ratio to voxel depth exceeds 1.1 are flagged as unreliable. For each flagged voxel region, a diversity-aware score selects a top-$k$ ($k=4$) subset of supporting camera-time views via maximum-weight $k$-clique optimization, favoring views that see the same voxels but differ in lateral offset and viewing angle while minimizing along-axis (longitudinal) displacement. Patch-match MVS then reconstructs local depth/normal planes from that view subset — using rasterized voxel visibility as an instance-level static/dynamic segmentation prompt so patch matching does not mix static background and rigid dynamic-object geometry — and the recovered surface points seed new Gaussians directly in the missing regions. Geometric losses (normal-angle and depth-consistency terms against the MVS output) supervise the new and existing Gaussians alongside the photometric loss.

## Assumptions

Calibrated multi-camera driving rigs with known intrinsics/extrinsics, LiDAR-backed or SfM initialization with per-point visibility/track metadata (COLMAP TRACK table), instance segmentation (via SAM) to separate dynamic objects from static background, and locally rigid or planar surfaces so patch-match MVS's local-plane assumption holds for both static structure and rigid dynamic objects (e.g. vehicles).

## Limitations / Failure Modes

The paper explicitly reports failure on non-rigid entities: pedestrians violate the patch-match rigidity assumption and are called out as an unresolved case despite being "inevitable" in urban scenes. It also fails on visually ambiguous thin/reflective structures — wire fences and glass produce dense LiDAR returns that don't correspond to what the images actually show as occluded vs. visible, and the paper flags correctly modeling occlusion in such regions as open future work. Performance also drops on scenes with near-linear ego-motion (nuScenes Scene 05), where minimal cross-camera baseline diversity limits how much new geometry the view-selection/MVS step can recover.

## Reusable Ingredients

- **Voxel z-buffer visibility flag**: comparing rasterized voxel depth to rendered Gaussian depth (ratio threshold 1.1, or missing Gaussian depth) to localize exactly where existing capacity fails to explain the scene, independent of photometric error.
- **Diversity-aware supporting-view selection**: a closed-form score trading off shared-voxel count, lateral vs. longitudinal camera offset, and angular difference, solved via max-weight $k$-clique, to pick MVS input views that add stereo baseline rather than redundant viewpoints.
- **Visibility-conditioned instance segmentation for MVS**: using existing 3D visibility/occupancy as a segmentation prompt so patch matching never blends static and dynamic-object geometry.
- **Opacity-fraction re-initialization trigger**: re-running the visibility/MVS pipeline when depth ratio ≥ 1.1 OR more than 25% of an instance's pixels have accumulated opacity < 0.7, as a concrete, non-photometric trigger for when existing capacity is insufficient.
- **Geometry-first evaluation discipline**: treating PSNR/SSIM/LPIPS as necessary but not sufficient, and explicitly ablating a geometric-loss term to show it suppresses artifacts under viewpoint deviation the photometric loss alone would miss.

---

### Deep Dive

#### Core Novelty
Prior densification (clone/split) and prior occlusion-aware pruning both operate only on primitives that already exist — they redistribute or remove capacity but cannot originate a Gaussian where none exists. VAD-GS's change is to detect "this pixel/voxel has no reliable Gaussian explaining it" as a first-class signal (via a voxel-vs-render depth-ratio test) and, only for those flagged regions, run a separate reconstruction path (diversity-selected MVS) whose output is fed in as *new* Gaussian seeds rather than as a densification direction for existing ones. The insight is that the missing-geometry problem is a visibility/coverage problem, not a photometric-error problem, so it needs its own detection signal (z-buffer disagreement) and its own capacity-creation mechanism (MVS-seeded initialization), decoupled from the standard gradient-driven densification loop.

#### Mathematical Formulation

Gaussian primitive and rendering (standard 3DGS, restated as the substrate being extended):
$$G(\mathbf{x}) = \exp\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\mathbf{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right), \qquad C = \sum_{i \in \mathcal{N}} \mathbf{c}_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)$$
$\boldsymbol{\mu}$ is the primitive center, $\mathbf{\Sigma}$ the (scale/quaternion-parameterized) covariance; evaluated per-pixel during standard alpha-blended rasterization.

Diversity score for candidate supporting view $S$ against reference view $R$ (Eq. 3), evaluated once per candidate view pair when a voxel region is flagged for re-initialization:
$$s = \frac{N}{\mathbf{d}_R^T \mathbf{d}_S} \cdot \frac{\sqrt{t_x^2 + t_y^2}}{|t_z| + \epsilon} \cdot \sin\theta$$
- $N$: number of voxels visible in both $R$ and $S$.
- $\mathbf{d}_R, \mathbf{d}_S$: per-voxel distance vectors from the voxel set to $R$ and $S$ respectively (their inner product penalizes views that are effectively co-located/co-aligned with the reference).
- $\mathbf{t} = (t_x, t_y, t_z)^T$: relative translation from $R$ to $S$; the ratio rewards lateral (stereo-useful) displacement and penalizes longitudinal (along-viewing-axis, stereo-useless) displacement.
- $\theta$: relative viewing-angle difference between $R$ and $S$; larger angular diversity increases the score.
- $\epsilon$: stabilizer against division by zero when $t_z \to 0$.
The top-$k$ ($k=4$) supporting-view subset per flagged region is chosen by maximum-weight $k$-clique optimization over pairwise scores $s$.

Patch-match local-plane model and cross-view projection (Eq. 4), evaluated per candidate pixel during MVS reconstruction over the selected $k$ supporting views:
$$z\,\mathbf{n}^T\mathbf{K}^{-1}\tilde{\mathbf{p}} + d = 0, \qquad \tilde{\mathbf{p}}' \simeq \mathbf{K}\left(\mathbf{R} - \frac{\mathbf{t}\mathbf{n}^T}{d}\right)\mathbf{K}^{-1}\tilde{\mathbf{p}}$$
- $z$: depth at pixel $\mathbf{p} = (u,v)^T$; $\mathbf{n}$: local surface normal; $d$: plane-to-camera-origin distance; $\mathbf{K}$: camera intrinsics; $\tilde{\mathbf{p}}$: homogeneous pixel coordinate; $[\mathbf{R},\mathbf{t}]$: relative pose from reference to supporting view. The first equation defines the local planar-patch hypothesis at a pixel; the second reprojects that hypothesis into a supporting view to score photometric/geometric consistency, iteratively propagated from neighboring pixels' hypotheses (standard PatchMatch MVS propagation).

Depth-ratio re-initialization test (per pixel, checked at fixed intervals during training): flag if Gaussian-rendered depth is absent, or
$$\frac{d_{\text{gaussian}}}{d_{\text{voxel}}} \ge 1.1$$

Total training loss (evaluated after rendering, each training step):
$$\mathcal{L} = \mathcal{L}_{\text{color}} + \lambda_{\text{normal}}\mathcal{L}_{\text{normal}} + \lambda_{\text{hard}}\mathcal{L}_{\text{hard}} + \lambda_{\text{soft}}\mathcal{L}_{\text{soft}}$$
$\mathcal{L}_{\text{color}}$ is the standard photometric term; $\mathcal{L}_{\text{normal}}$ penalizes angular deviation between rendered and MVS-derived normals; $\mathcal{L}_{\text{hard}}$ and $\mathcal{L}_{\text{soft}}$ are depth-consistency terms against MVS depth under fixed vs. learned opacity respectively — these are the geometric supervision that ties new MVS-seeded Gaussians (and neighboring existing ones) back to the reconstructed surface.

#### Algorithm / Pipeline Changes
1. **Voxelize** the current initialized point cloud (from SfM/LiDAR); at training checkpoints, rasterize per-view voxel depth via z-buffering and build a 2D pixel→voxel index map.
2. **Flag unreliable regions**: per pixel, compare Gaussian-rendered depth to voxel depth; flag if Gaussian depth is missing or $d_{\text{gaussian}}/d_{\text{voxel}} \ge 1.1$. Separately, flag an instance for re-initialization if $>25\%$ of its pixels have accumulated opacity $< 0.7$.
3. **Select supporting views** for each flagged region: compute the diversity score $s$ (Eq. 3) over candidate camera-time pairs, then pick the top-$k=4$ subset via maximum-weight $k$-clique optimization (replaces naive nearest-view or all-view MVS input selection).
4. **Segment static vs. dynamic** using rasterized voxel visibility as an instance-level prompt (fed to/aligned with SAM-based instance masks) so the following MVS step never mixes static-background and rigid-dynamic-object geometry.
5. **Patch-match MVS reconstruction**: within each segmented region, jointly optimize local plane hypotheses (Eq. 4) via photometric + geometric consistency across the $k$ selected views, propagating hypotheses from neighboring pixels.
6. **Seed new Gaussians** directly from the recovered MVS surface points in the flagged regions (this is the step that differs from clone/split — it creates primitives at positions with no prior Gaussian, rather than duplicating/splitting an existing one).
7. **Re-run periodically**: the voxel-visibility re-initialization (steps 1–6) repeats every 5 complete sampling cycles (~48ms per operation, integrated via CUDA) rather than running once at initialization only, so newly-exposed missing regions discovered later in training are also patched.
8. Downstream: newly seeded and existing Gaussians are jointly optimized under the combined loss $\mathcal{L}$ through the normal 3DGS rasterization/backprop loop.

#### Key Hyperparameters & Design Choices
- Depth-ratio re-initialization threshold: $d_{\text{gaussian}}/d_{\text{voxel}} \ge 1.1$.
- Opacity-based instance re-initialization trigger: $>25\%$ of instance pixels with accumulated opacity $< 0.7$.
- Supporting-view subset size: $k = 4$, chosen via maximum-weight $k$-clique optimization.
- Re-initialization cadence: every 5 complete view-sampling cycles, ~48ms per operation (CUDA-integrated).
- Loss weights: $\lambda_{\text{normal}} = 0.02$; $\lambda_{\text{hard}} = 0.1$, disabled after 80% of training (switches to soft/learned-opacity depth term only in the later phase); $\lambda_{\text{soft}}$ value not specified in paper.
- Views sampled without replacement during training to reduce photometric overfitting.
- Test protocol: every 4th frame (all associated camera views) held out.
- Hardware: single NVIDIA RTX 4090.
- Voxel size, $k$-clique edge/score threshold, and patch-match window size: not specified in paper.

#### Ablation Summary
(nuScenes, Table 3; PSNR / PSNR* (dynamic-limited) / SSIM / LPIPS)
1. **w/o voxel visibility reasoning**: 23.79 / 22.75 / 0.753 / 0.215 — the largest drop from the complete model, confirming voxel-vs-render depth disagreement is the most load-bearing component (the mechanism that finds missing geometry in the first place).
2. **w/o view selection**: 23.92 / 22.83 / 0.757 / 0.212 — smaller drop; paper attributes remaining artifacts to floaters from poorly-chosen supporting views in sparse regions.
3. **w/o geometric losses**: 24.59 / 22.78 / 0.764 / 0.194 — full-frame PSNR/LPIPS are close to or better than the complete model, but PSNR* (dynamic-object-focused) and visual inspection under large viewpoint deviation are worse; geometric losses are framed as an overfitting suppressor rather than a raw-metric booster.
4. **Complete VAD-GS**: 24.51 / 23.16 / 0.765 / 0.199.
Voxel visibility reasoning is flagged as the single most impactful component — removing it degrades every metric, unlike removing view selection or geometric losses individually.

#### Implementation Reality
- **Framework:** PyTorch/CUDA, extending StreetGaussians (for Waymo) and DriveStudio (for nuScenes) — both themselves 3DGS-family static/dynamic urban codebases — plus off-the-shelf Depth Anything V2, DSINE (normals), and SAM for auxiliary priors/segmentation.
- **Key files:** `train.py` (main training loop), `render.py` (rendering/output generation), `metrics.py` (evaluation), `configs/example/*.yaml` (per-scene experiment configs, e.g. `nuscenes_train_000.yaml`, specifying dataset paths and training parameters).
- **Notable implementation details:** the repo relies on Depth Anything V2 and DSINE as auxiliary monocular depth/normal predictors feeding into the geometric losses/MVS pipeline — this specific auxiliary-model choice is not detailed in the paper's method text as reviewed; data preprocessing follows StreetGaussians/DriveStudio conventions (10Hz frame processing, per-frame LiDAR depth maps, ego pose hierarchies) inherited wholesale from those base codebases rather than being VAD-GS-specific engineering.

#### Failure Modes & Limitations
Explicitly fails on non-rigid pedestrians, which violate the patch-match MVS rigidity assumption yet are called "inevitable" in urban scenes — no mitigation is proposed, just flagged as unsolved. Wire fences and glass surfaces cause visually ambiguous occlusion: LiDAR returns dense points there, but images show the region as unoccluded, and the paper calls correct occlusion modeling in such regions "an important direction for future research." Scenes with near-linear ego-motion (nuScenes Scene 05) reduce cross-camera geometric diversity available to the view-selection step, lowering reconstruction quality relative to scenes with more varied trajectories.

---

## Relevance to ADAGS

This is the closest competing visibility-to-capacity work. ADAGS novelty must be non-oracle, non-rigid multiview-temporal visibility and hidden-surface memory rather than generic visibility-aware densification.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Sources

- https://arxiv.org/abs/2510.09364
- https://github.com/YikangZhang1641/VAD-GS
- https://mias.group/VAD-GS/
