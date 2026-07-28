---
type: paper
node_id: paper:lei2025_mosca
title: "MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds"
authors: ["Jiahui Lei", "Yijia Weng", "Adam W. Harley", "Leonidas Guibas", "Kostas Daniilidis"]
year: 2025
venue: "CVPR"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: [dynamic-gs, motion-scaffold, priors]
status: deep-dived
---

# MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds

**Paper:** https://arxiv.org/abs/2405.17421
**Code:** https://github.com/JiahuiLei/MoSca
**Base method:** 3D Gaussian Splatting, with camera pose/focal length recovered via a custom tracklet-based bundle adjustment (no COLMAP or external SLAM required).

## One-line thesis

A sparse graph of trajectory nodes ("Motion Scaffold"), each carrying a per-timestep SE(3) transform interpolated via dual-quaternion blending, supplies a compact, ARAP-regularized deformation field that every dynamic Gaussian is skinned to; this replaces per-Gaussian or MLP-warp deformation with a low-rank, physically-rigid motion prior recoverable from monocular casual video.

## Problem / Gap

Monocular 4D reconstruction from casual video is severely underconstrained: there is no multi-view stereo cue at a single timestep, and camera parameters are usually unknown. Prior dynamic-GS methods that warp Gaussians with a continuous per-point MLP deformation field over-smooth motion discontinuities and have no explicit rigidity prior, so they degrade under sparse, single-view supervision. MoSca targets this by exploiting the physical prior that real-world motion is largely rigid, smooth, and low-rank, and by not assuming known camera poses.

## Method

MoSca first extracts 2D foundation-model priors (monocular depth, long-term 2D point tracks, and RAFT-based epipolar error maps that flag dynamic regions) from the raw video. It then solves camera poses/focal length via a lightweight tracklet-based bundle adjustment over confident static-background tracks (no COLMAP). A sparse set of trajectory nodes is instantiated and optimized purely geometrically (ARAP + temporal smoothness losses on node SE(3) trajectories) using the lifted 3D tracks, before any photometric optimization. Dynamic 3D Gaussians are then initialized from back-projected depth at every frame, each assigned RBF skinning weights to nearby scaffold nodes (plus a learned per-Gaussian correction), and fused into a single canonical set of Gaussians per query time via dual-quaternion blending of the relevant nodes' transforms; a separate static background Gaussian set is rendered unwarped. The whole system is finished with photometric (RGB, depth, track) optimization jointly with the geometric regularizers.

## Assumptions

Monocular RGB video with unknown intrinsics/extrinsics; scene deformation is assumed low-rank, smooth, and locally rigid; long-term 2D tracks must stay largely reliable across the video; background is assumed mostly static so it can bootstrap camera bundle adjustment.

## Limitations / Failure Modes

Quality is bottlenecked by the foundation-model priors: track and depth errors propagate directly since MoSca does not have an independent geometry signal to fall back on. Only regions visible at some point in the video can be reconstructed — permanently occluded geometry is not hallucinated (no diffusion prior). Non-rigid low-level radiometric effects (shadows, reflections, liquids, exposure changes) are not explainable by a deformation field and show up as artifacts, mostly in the background. Fast or highly complex motion is implicitly harder because the sparse-node smoothness/low-rank assumption breaks down.

## Reusable Ingredients

- **Trajectory-curve KNN graph topology** — connects scaffold nodes by max-distance-over-time between their translation trajectories rather than static spatial proximity, so the graph reflects motion similarity, not just rest-pose distance.
- **Dual-quaternion blending (DQB) for SE(3) interpolation** — interpolates multiple rigid transforms while staying exactly on the SE(3) manifold, avoiding matrix-blending artifacts.
- **RBF skinning weights with a learned per-Gaussian correction** — base weight is a Gaussian RBF of distance to node/radius, refined by a small learned offset `Δw_j` per Gaussian, letting individual Gaussians deviate from the coarse scaffold assignment.
- **Two-phase optimization order (geometry-only, then photometric)** — the scaffold's node trajectories are first fit to lifted 3D tracks with only ARAP/velocity/acceleration losses, before any rendering loss is introduced, decoupling motion-structure recovery from appearance fitting.
- **Scale-invariant depth alignment loss for bundle adjustment** — `|x/y − 1| + |y/x − 1|` term lets monocular depth from different frames be reconciled during pose solving without assuming absolute scale.
- **Tracklet-based bundle adjustment as a COLMAP-free camera solver** — recovers poses and focal length directly from 2D track reprojection + depth consistency on static points.

---

### Deep Dive

#### Core Novelty
Relative to MLP-deformation dynamic GS baselines, MoSca replaces the deformation field with an explicit sparse graph of per-timestep SE(3) node trajectories, regularized by ARAP and temporal-smoothness losses and interpolated with dual-quaternion blending. The key insight: because the graph topology is built from trajectory-curve distance (not rest-pose distance) and optimized geometrically before photometric fitting, the motion field is forced to be low-rank and physically rigid independent of appearance, which is claimed to be more robust to the sparse, single-view supervision available from casual video than a continuous per-point MLP warp.

#### Mathematical Formulation

Node definition, evaluated once per scaffold node over all timesteps (state variable, not a per-step computation):
$$\mathbf{v}(m) = ([\mathbf{Q}_1(m), \mathbf{Q}_2(m), \ldots, \mathbf{Q}_T(m)],\, r(m))$$
where $\mathbf{Q}_t(m) \in SE(3)$ is node $m$'s rigid transform at frame $t$ and $r(m) \in \mathbb{R}^+$ is its RBF influence radius. $M$ is the total node count (≪ scene point count).

Graph topology, computed once to build the scaffold's edge set:
$$\mathcal{E}(m) = \mathrm{KNN}_{n}\left[D_{\text{curve}}(m,n)\right], \qquad D_{\text{curve}}(m,n) = \max_{t=1,\ldots,T} \lVert \mathbf{t}_t(m) - \mathbf{t}_t(n) \rVert$$
Edges connect nodes whose translation trajectories stay close across *all* timesteps, so topology reflects motion similarity rather than rest-pose proximity.

Dual-quaternion blending, used every time a set of node transforms must be interpolated into one SE(3) element (deformation field evaluation, per query):
$$\mathrm{DQB}(\{(w_i, \mathbf{Q}_i)\}_{i=1}^L) = \frac{\sum_{i=1}^L w_i \hat{\mathbf{q}}_i}{\lVert \sum_{i=1}^L w_i \hat{\mathbf{q}}_i \rVert_{DQ}} \in SE(3)$$
where $\hat{\mathbf{q}}_i$ is the dual-quaternion form of $\mathbf{Q}_i$ and $w_i$ are skinning weights.

Deformation of a query point from source to destination time, evaluated per Gaussian per query time before rasterization:
$$\mathcal{W}(\mathbf{x}, \mathbf{w}; t_{src}, t_{dst}) = \mathrm{DQB}(\{w_i, \Delta\mathbf{Q}(i)\}_{i \in \mathcal{E}(m^*)}), \qquad \Delta\mathbf{Q}(i) = \mathbf{Q}^{(i)}_{t_{dst}}\left(\mathbf{Q}^{(i)}_{t_{src}}\right)^{-1}$$
$m^*$ is the node nearest $\mathbf{x}$ at $t_{src}$; only its local edge neighborhood $\mathcal{E}(m^*)$ contributes.

Skinning weight, computed per (Gaussian, node) pair from an RBF around each node:
$$w_i(\mathbf{x}, t_{src}) = \exp\!\left(-\frac{\lVert \mathbf{x} - \mathbf{t}^{(i)}_{t_{src}} \rVert_2^2}{2\, r(i)}\right) \in \mathbb{R}^+$$

ARAP loss over the (possibly sub-sampled) node topology pyramid $\hat{\mathcal{E}}$, applied as a geometric regularizer during the node-fitting stage and jointly during final photometric optimization:
$$\mathcal{L}_{\text{arap}} = \sum_{t=1}^{T}\sum_{m=1}^{M}\sum_{n \in \hat{\mathcal{E}}(m)} \lambda_l \Big|\lVert \mathbf{t}_t(m)-\mathbf{t}_t(n)\rVert - \lVert \mathbf{t}_{t+\Delta}(m)-\mathbf{t}_{t+\Delta}(n)\rVert\Big| + \lambda_c \lVert \mathbf{Q}_t^{-1(n)}\mathbf{t}_t(m) - \mathbf{Q}_{t+\Delta}^{-1(n)}\mathbf{t}_{t+\Delta}(m) \rVert$$
First term preserves inter-node edge length across time (rigidity); second term preserves each node's position in a neighbor's local frame.

Temporal smoothness (velocity and acceleration) losses on node trajectories, applied at the same stage as ARAP:
$$\mathcal{L}_{\text{vel}} = \sum_{t,m} \lVert \mathbf{t}_t(m)-\mathbf{t}_{t+1}(m)\rVert + \lVert \log(\mathbf{R}_t(m)\mathbf{R}_{t+1}^{-1}(m)) \rVert_F$$
$$\mathcal{L}_{\text{acc}} = \sum_{t,m} \lVert \mathbf{t}_t(m)-2\mathbf{t}_{t+1}(m)+\mathbf{t}_{t+2}(m)\rVert + \big|\,\lVert\log(\mathbf{R}_t\mathbf{R}_{t+1}^{-1})\rVert_F - \lVert\log(\mathbf{R}_{t+1}\mathbf{R}_{t+2}^{-1})\rVert_F\,\big|$$
Combined into $\mathcal{L}_{\text{geo}} = \lambda_{\text{arap}}\mathcal{L}_{\text{arap}} + \lambda_{\text{acc}}\mathcal{L}_{\text{acc}} + \lambda_{\text{vel}}\mathcal{L}_{\text{vel}}$, optimized before photometric terms are introduced.

Gaussian attributes and fused scene, evaluated per Gaussian per query time immediately before rasterization:
$$\mathcal{G} = \{(\mu_j, R_j, s_j, o_j, c_j;\, t^{ref}_j, \Delta\mathbf{w}_j)\}_{j=1}^N$$
$$\mathcal{G}(t) = \{(\mathbf{T}_j(t)\mu_j,\, \mathbf{T}_j(t)R_j,\, s_j, o_j, c_j)\,|\, \mathbf{T}_j(t) = \mathcal{W}(\mu_j, \mathbf{w}(\mu_j, t^{ref}_j) + \Delta\mathbf{w}_j;\, t^{ref}_j, t)\}_{j=1}^N$$
$\Delta\mathbf{w}_j \in \mathbb{R}^K$ is a learned per-Gaussian correction to the RBF skinning weights. Static background Gaussians $\mathcal{H}$ are rendered unwarped; the final rendered scene is $\mathcal{G}(t) \cup \mathcal{H}$.

Camera bundle adjustment losses, used only during the pose-solving stage (before scaffold/Gaussian optimization):
$$\mathcal{L}_{\text{proj}} = \sum_{i \in |\mathcal{T}_{\text{static}}|} \sum_{a,b} (v^{(i)}_a v^{(i)}_b) \left\lVert \pi_{\mathbf{K}}\!\left(\mathbf{W}_b^{-1}\mathbf{W}_a \pi_{\mathbf{K}}^{-1}(p^{(i)}_a, D_a[p^{(i)}_a])\right) - p^{(i)}_b \right\rVert$$
$$\mathcal{L}_z = \sum_{i \in |\mathcal{T}_{\text{static}}|}\sum_{a,b} (v^{(i)}_a v^{(i)}_b)\, D_{\text{scale-inv}}\!\left(\left[\mathbf{W}_b^{-1}\mathbf{W}_a \pi_{\mathbf{K}}^{-1}(p^{(i)}_a, D_a[p^{(i)}_a])\right]_z,\, D_b[p^{(i)}_b]\right)$$
with $D_{\text{scale-inv}}(x,y) = |x/y - 1| + |y/x - 1|$, and $\mathcal{L}_{\text{BA}} = \lambda_{\text{proj}}\mathcal{L}_{\text{proj}} + \lambda_z \mathcal{L}_z$. 3D lifting of a 2D track uses visibility-gated back-projection with linear interpolation across occluded spans:
$$\mathbf{h}_t = \begin{cases} \mathbf{W}_t \pi_{\mathbf{K}}^{-1}(p_t, D_t[p_t]) & v_t = 1 \\ \mathrm{LinearInterp}(\mathbf{h}_{\text{left}}, \mathbf{h}_{\text{right}}) & v_t = 0 \end{cases}$$

Final training objective, applied jointly in the photometric optimization stage:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{rgb}}\mathcal{L}_{\text{rgb}} + \lambda_{\text{dep}}\mathcal{L}_{\text{dep}} + \lambda_{\text{track}}\mathcal{L}_{\text{track}} + \lambda_{\text{arap}}\mathcal{L}_{\text{arap}} + \lambda_{\text{acc}}\mathcal{L}_{\text{acc}} + \lambda_{\text{vel}}\mathcal{L}_{\text{vel}}$$

#### Algorithm / Pipeline Changes
1. Run 2D foundation models on the input video: monocular depth (ZoeDepth/Metric3D-v2/UniDepth), long-term 2D point tracks (BootsTAPIR/CoTracker-v3/SpaTracker), and RAFT-based epipolar error maps to flag likely-dynamic pixels.
2. Solve camera intrinsics/extrinsics with tracklet-based bundle adjustment ($\mathcal{L}_{\text{BA}}$) restricted to confident static-background tracks — no COLMAP/SLAM dependency.
3. Lift 2D tracks to 3D via visibility-gated back-projection (Eq. 8), instantiate a sparse set of scaffold nodes, build the trajectory-curve KNN graph, and optimize node SE(3) trajectories with $\mathcal{L}_{\text{geo}}$ only (rotations, plus translations for invisible/occluded intervals) — no rendering loss yet.
4. Initialize dynamic 3D Gaussians from back-projected depth at every frame (reference timestep $t^{ref}_j$ recorded per Gaussian); initialize a separate static background Gaussian set $\mathcal{H}$.
5. Attach each dynamic Gaussian to nearby scaffold nodes via RBF skinning weights plus a learned correction $\Delta\mathbf{w}_j$.
6. At render time, fuse all dynamic Gaussians into the query timestep via $\mathcal{W}$ (DQB over the local node neighborhood), union with the static background set, and rasterize.
7. Jointly optimize Gaussian attributes and scaffold node trajectories with $\mathcal{L}_{\text{total}}$ (RGB + depth + track rendering losses plus the geometric regularizers), replacing the single-stage photometric-only optimization used by MLP-deformation baselines.

#### Key Hyperparameters & Design Choices
- Node count: nodes are uniformly resampled by curve distance; exact count/threshold not specified in the main paper (reported result: ~3,177 MoSca nodes vs. ~106,596 foreground Gaussians, a ~46× compression ratio).
- Initial node RBF radius $r_{\text{init}}$: referenced as predefined; exact value stated to be in supplemental material, not extracted here.
- Topology KNN parameter $K$: not specified in the main paper.
- Loss weights $\lambda_{\text{proj}}, \lambda_z, \lambda_l, \lambda_c, \lambda_{\text{rgb}}, \lambda_{\text{dep}}, \lambda_{\text{track}}, \lambda_{\text{arap}}, \lambda_{\text{acc}}, \lambda_{\text{vel}}$: not specified in the main paper (noted as in supplemental material).
- Per-Gaussian skinning correction $\Delta\mathbf{w}_j$ dimensionality: $\mathbb{R}^K$ (tied to node neighborhood size); no further architecture detail given.
- Reported system stats (Table 7): 37.8 FPS inference at 2× resolution on DyCheck.
- Not specified in paper: learning rates/schedules for any component.

#### Ablation Summary
DyCheck, mPSNR / mSSIM / mLPIPS (higher PSNR/SSIM better, lower LPIPS better):
- Full model: 19.32 / 0.706 / 0.264
- No photometric optimization: 13.71 / 0.480 / 0.763 (**largest drop — photometric optimization is the single most impactful component**)
- Fuse only 4 frames (vs. all frames): 16.96 / 0.663 / 0.344
- No geometric optimization (ARAP/temporal losses): 18.85 / 0.693 / 0.287
- No dual-quaternion blending: 19.18 / 0.701 / 0.276

#### Implementation Reality
- **Framework:** PyTorch, CUDA 11.8; optional JAX support (`jax_requirements.txt`). README notes a TODO to replace the current rendering backend with GSplat.
- **Key files:** `lib_mosca` / `lib_moca` implement the core scaffold, camera pose solving, and deformation field; `lib_prior` wraps the third-party foundation models and Gaussian Splatting rendering; `lib_render` holds the rasterization backend. Top-level scripts: `mosca_precompute.py` (runs 2D foundation models — flow, tracking, depth), `lite_moca_reconstruct.py` (tracklet-based bundle adjustment for camera pose), `mosca_reconstruct.py` (fits the full 4D scaffold + Gaussian scene).
- **Notable implementation details:** Configuration is driven by YAML profiles (e.g. `profile/demo/demo_prep.yaml`, `profile/demo/demo_fit.yaml`); the README states the system needs per-scene parameter tuning, but specific hyperparameter values are not exposed at the README level. Supported prior models in code: BootsTAPIR, CoTracker, SpaTracker, RAFT for tracking/flow; DepthCrafter, Metric3D-v2, UniDepth for depth (broader than the three depth models named in the main paper text).

#### Failure Modes & Limitations
Reconstruction quality is bounded by the accuracy of the 2D tracking and depth foundation models it depends on. Only scene content visible at some point in the video can be recovered — permanently occluded regions are not hallucinated. Effects that are not explainable by rigid/smooth deformation (shadows, reflections, liquids, exposure changes) produce artifacts, primarily in background regions. The sparsity/low-rank/smoothness assumptions underlying the scaffold implicitly limit robustness to very fast or highly complex (non-rigid, high-frequency) motion.

---

## Relevance to ADAGS

Core comparison for `idea:track-prior-scaffold-motion`.

## Connections

## Sources

- https://arxiv.org/abs/2405.17421
- https://github.com/JiahuiLei/MoSca
- https://openaccess.thecvf.com/content/CVPR2025/papers/Lei_MoSca_Dynamic_Gaussian_Fusion_from_Casual_Videos_via_4D_Motion_CVPR_2025_paper.pdf
