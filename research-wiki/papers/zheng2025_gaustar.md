---
type: paper
node_id: paper:zheng2025_gaustar
title: "GauSTAR: Gaussian Surface Tracking and Reconstruction"
authors: ["Chengwei Zheng", "et al. (ETH AIT)"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2501.10283"
tags: [dynamic-gs, surface-tracking, topology-change, multiview]
status: deep-dived
---

# GauSTAR: Gaussian Surface Tracking and Reconstruction

**Paper:** https://arxiv.org/abs/2501.10283 (v3, CVPR 2025; authors: Chengwei Zheng, Lixin Xue, Juan Zarate, Jie Song)
**Code:** https://github.com/eth-ait/GauSTAR (populated, C++/Python, MIT-adjacent license file present)
**Base method:** 3D Gaussian Splatting (Kerbl et al. 2023) + SuGaR-style mesh-bound Gaussians (Guédon & Lepetit 2024) for the geometry/rendering half; HumanRF (Işık et al. 2023) for initial mesh generation; RAFT (Teed & Deng 2020) for 2D optical flow used in scene-flow initialization.

## One-line thesis

Binding Gaussians rigidly to mesh faces gives temporally consistent tracking under smooth deformation, but a per-face "unbinding weight" — combining 3DGS positional-gradient magnitude with per-face RGB/depth reconstruction error — lets GauSTAR selectively detach Gaussians from the mesh exactly where topology changes (appear/disappear/split), and only there, then regenerates mesh surface from the freed Gaussians via TSDF re-fusion.

## Problem / Gap

Prior dynamic-surface trackers force an all-or-nothing choice: fixed-topology template trackers (e.g., mesh-bound Gaussian trackers, classical non-rigid mesh tracking) preserve correspondence but cannot represent surfaces appearing, disappearing, or splitting, so quality degrades wherever real topology change occurs. Fully unconstrained per-frame reconstruction methods (independent per-frame 3DGS/mesh extraction, methods like Dynamic 3D Gaussians and PhysAvatar) can represent any topology but produce temporally inconsistent geometry and lose face correspondence across frames entirely, so there is no usable tracking signal. GauSTAR targets the middle: keep template tracking wherever topology is actually stable, and only pay the re-creation cost in the specific regions that change.

## Method

GauSTAR represents dynamic surfaces as "Gaussian Surfaces": triangle meshes with N=6 Gaussians per face, positioned by fixed barycentric coordinates on each face and oriented so the Gaussian's local z-axis aligns with the face normal (z-scale fixed to a small predefined value), so Gaussian pose is fully determined by the mesh face pose. Per frame, a surface-aware 3D scene-flow field (built by reprojecting RAFT 2D flow through per-view depth, filtered by multi-view visibility/bi-directional consistency, then propagated and smoothed across the mesh) initializes vertex motion for the next frame. The mesh is then refined under multi-view RGB-D photometric/geometric losses and mesh regularizers (this is the "fixed-topology" tracking mode). Simultaneously, a per-face unbinding weight is computed from 3DGS-style positional gradient magnitude plus per-face RGB and depth reconstruction error; faces/Gaussians whose weight exceeds a threshold have their rigid mesh-binding constraint relaxed (an explicit per-Gaussian ΔR, Δt is introduced and only lightly regularized there), letting Gaussians drift independently of the mesh in topology-changing regions. Finally, a re-meshing stage renders multi-view depth from the optimized (partially unbound) Gaussians, TSDF-fuses it into a new local mesh only in high-unbinding-weight regions, stitches it to the retained stable mesh via nearest-boundary-vertex correspondence plus edge-flip/hole-fill cleanup, and the updated mesh re-enters the tracking loop for the next frame.

## Assumptions

Requires synchronized multi-view RGB-D (or RGB + derivable per-view depth) captures of a dynamic scene — the authors' own dataset uses 52 RGB + 52 IR cameras at 3004x4092, 30 fps, with depth derived from IR laser point clouds. An initial mesh from any multi-view reconstruction method (HumanRF, in their pipeline) is required to bootstrap tracking. The method assumes topology changes are spatially localized (affecting a subset of faces at a time) rather than global/scene-wide.

## Limitations / Failure Modes

The paper explicitly flags degraded handling of "complex or sudden topology changes, such as when a new person suddenly enters the scene" — i.e., large, non-local topology events are harder than the gradual local splitting/merging the method is designed around. It requires multi-view capture rigs, which the authors state "restricts its applicability in general public scenarios" (no monocular/casual-capture claim). Transparent and specular surfaces are named as a general weakness ("pose challenges for most surface reconstruction methods"), inherited rather than solved by this method. The ablation shows each component is load-bearing: removing unbinding costs -2.57 dB PSNR and roughly +2.4 cm 3D tracking error (ATE); removing re-meshing costs -2.1 dB and +1.63 cm; removing scene-flow initialization costs -1.95 dB PSNR but a much larger +6.11 cm 3D ATE — i.e., tracking accuracy is far more dependent on the scene-flow initialization than rendering quality is.

## Reusable Ingredients

- **Per-face unbinding weight (gradient + reconstruction-error signal)**: reuses the existing 3DGS densification gradient statistic as a topology-change detector rather than only a densification trigger — a way to get "is this region under-modeled" for free from optimizer-internal signals.
- **Rigid-binding-with-escape-hatch parameterization**: keep primitives rigidly attached to a structure (mesh face) by default, but expose a low-weight-regularized ΔR/Δt residual that only activates where a per-region error signal is high — a general pattern for "protect by default, relax locally on evidence."
- **Surface-aware scene flow from 2D optical flow + multi-view depth reprojection**: cheap, off-the-shelf-model-based 3D motion initialization (RAFT + depth reprojection + bi-directional consistency filtering + neighbor propagation/smoothing) usable as a tracking initializer without any learned 3D flow network.
- **Boundary-matched local re-meshing**: only re-fuse (TSDF) and re-mesh the regions flagged as changed, then stitch to the untouched mesh via nearest-boundary-vertex correspondence — avoids full-scene re-meshing every frame.

---

### Deep Dive

#### Core Novelty

Relative to SuGaR-style mesh-bound Gaussians (which assume a fixed mesh topology for the whole sequence) and to unconstrained per-frame Gaussian/mesh reconstruction (which has no cross-frame correspondence), GauSTAR's change is making the bind/unbind decision itself a per-face, per-frame, data-driven quantity rather than a global setting. The key insight is that the same gradient signal 3DGS already computes to decide where to densify is also informative about where the current mesh topology is failing to explain the data — reusing it as a topology-change detector avoids needing a separate learned "change" head, and coupling it with photometric/depth error keeps the detector grounded in reconstruction quality rather than gradient magnitude alone.

#### Mathematical Formulation

Gaussian definition (per-Gaussian, standard 3DGS form, evaluated at rasterization):
$$G(\mathbf{x}) = \sigma(\alpha) \cdot \exp\left(-\tfrac{1}{2}(\mathbf{x}-\mathbf{p})^{\top}\Sigma^{-1}(\mathbf{x}-\mathbf{p})\right), \qquad \Sigma = \mathbf{R S S}^{\top}\mathbf{R}^{\top}$$
where $\alpha$ is opacity, $\mathbf{p}$ position, $\mathbf{S}$ scale, $\mathbf{R}$ rotation, $\Sigma$ covariance. Standard 3DGS; included because $\mathbf{p}$ and $\mathbf{R}$ are the quantities constrained by mesh-binding below.

Gaussian position on a face (per-Gaussian, evaluated once per face update, before rasterization):
$$\mathbf{p} = b_1\mathbf{v}_1 + b_2\mathbf{v}_2 + b_3\mathbf{v}_3$$
fixed barycentric weights $b_1,b_2,b_3$ over the face's three vertices $\mathbf{v}_1,\mathbf{v}_2,\mathbf{v}_3$; this is what makes a bound Gaussian's pose fully determined by mesh vertex motion. The Gaussian's local z-axis is constrained to the face normal and z-scale fixed to a small predefined $\delta$ (value not specified in paper).

Surface-aware scene-flow smoothing (per-vertex, applied after raw 3D flow is computed from 2D-flow reprojection through depth, before it is used to initialize next-frame vertex positions):
$$\mathcal{F}'(v) = \frac{1}{|\mathbf{N}(v)|}\sum_{u \in \mathbf{N}(v)} w(u,v)\,\mathcal{F}(u)$$
$\mathbf{N}(v)$ is the neighborhood of vertex $v$ on the mesh, $w(u,v)$ a surface-distance-based weight (exact form not given in the accessible text). This is a mesh-graph smoothing/propagation pass over per-vertex raw flow estimates, run iteratively (the released code runs `mesh_vert_propagate`/`mesh_color_smoothing` for 5 iterations).

Unbinding weight (per-face, computed each optimization step or at fixed intervals during fixed-topology refinement, used to gate both the binding regularizer below and the re-meshing trigger):
$$\mathcal{W}(f) = \mathcal{G}_{\text{pos}}(f) + \lambda_{\text{rgb}}\mathcal{L}_{\text{rgb}}(f) + \lambda_{\text{depth}}\mathcal{L}_{\text{depth}}(f)$$
$\mathcal{G}_{\text{pos}}(f)$ is a positional-gradient-magnitude term "inspired by adaptive density control in 3DGS" (i.e., reuses the same accumulated view-space positional gradient statistic 3DGS uses to decide where to densify) aggregated per face; $\mathcal{L}_{\text{rgb}}(f)$, $\mathcal{L}_{\text{depth}}(f)$ are per-face photometric and depth reconstruction errors; $\lambda_{\text{rgb}}, \lambda_{\text{depth}}$ weight them. $\mathcal{W}(f)$ is capped at 1.0.

Unbinding regularizer (per unbound Gaussian $g$ belonging to face $f_g$, added to the training loss during fixed-topology refinement):
$$\mathcal{L}_{\text{unb}}(g) = (1-\mathcal{W}(f_g))\left(\|\Delta\mathbf{R}(g)-\mathbf{I}\|_1 + \lambda_t\|\Delta\mathbf{t}(g)\|_1\right)$$
$\Delta\mathbf{R}(g), \Delta\mathbf{t}(g)$ are learned per-Gaussian rotation/translation residuals layered on top of the rigid mesh-binding pose; the $(1-\mathcal{W}(f_g))$ factor means high unbinding weight (high suspected topology change) directly reduces the penalty for the Gaussian drifting away from its rigid mesh-bound pose, which is the mechanism that lets Gaussians "escape" the mesh where topology is changing. In the released code this maps to `loose_bind_factor_r` and `loose_bind_factor_t`.

Standard L1+SSIM RGB loss, L1 depth loss, L1 mask loss, plus mesh regularizers (normal smoothing across neighboring faces, face-area preservation against initial areas, and a spherical-harmonics-based temporal color-consistency term) are used but not modified from standard form beyond being computed per-face/per-mesh; not transcribed here per instructions.

#### Algorithm / Pipeline Changes

1. **Scene-flow warping (Sec 3.2)**: For frame t to t+1, project previous-frame mesh vertices into each camera view, advect the 2D pixel position with RAFT optical flow, reproject the advected pixel back to 3D using that camera's depth map (requires at least `min_observe`=4 camera views agreeing), filter with bi-directional consistency (pixel and depth thresholds) and a face-normal/view-angle cutoff, then propagate to unmeasured vertices via k-NN (k=8) averaging and smooth with the neighbor-weighted pass above. Produces the initial vertex positions for the next frame's optimization.
2. **Fixed-topology reconstruction (Sec 3.3)**: With mesh topology frozen, jointly optimize vertex positions and per-Gaussian appearance (opacity, SH color, scale, rotation offsets) against multi-view RGB-D under the photometric/depth/mask losses plus mesh regularizers. This is where Gaussians remain rigidly bound to their face via the barycentric formula unless unbound (step 3).
3. **Adaptive Gaussian unbinding (Sec 3.4)**: Each face's unbinding weight $\mathcal{W}(f)$ is computed from accumulated positional gradient plus RGB/depth error. Faces above a threshold get their Gaussians' rigid-binding regularizer relaxed (per the $\mathcal{L}_{\text{unb}}$ formula), letting per-Gaussian $\Delta\mathbf{R},\Delta\mathbf{t}$ grow; new Gaussians are duplicated for faces whose weight exceeds threshold (topology-change candidate regions get extra capacity, analogous to 3DGS densification but keyed to the unbinding signal instead of only gradient magnitude).
4. **Surface re-meshing (Sec 3.5)**: Render depth maps from the current (partially unbound) Gaussian surface across the multi-view rig, TSDF-fuse them into a new local mesh only within regions where unbinding weight exceeds threshold, discard/replace the old mesh geometry there, stitch the new patch to the retained stable mesh by matching each boundary vertex to its nearest boundary vertex on the other side, then clean up with edge-flipping and hole-filling. The updated mesh becomes the topology for the next frame's step 1.

This entire 4-step loop repeats per frame; the "tracking" output is the sequence of meshes with per-face correspondence preserved wherever unbinding never triggered.

#### Key Hyperparameters & Design Choices

From the paper text (values largely deferred to unread supplementary material):
- Gaussians per face: N = 6.
- Gaussian z-scale $\delta$: "small predefined value" — not specified in paper body.
- $\lambda_{\text{rgb}}, \lambda_{\text{depth}}, \lambda_t, \lambda_{\text{SSIM}}$: referenced symbolically, numeric values not given in the main text (paper defers to supplementary materials, not accessed).
- Unbinding threshold and densification-duplication threshold: described as thresholds on $\mathcal{W}(f)$, no numeric value stated in paper body.
- Capture setup (not a model hyperparameter but a reproduction-relevant constant): 52 RGB + 52 IR cameras, 3004x4092 resolution, 30 fps.

From the released code (`gaustar_trainers/refine.py`, `gaussian_splatting/arguments/__init__.py`) — these are implementation defaults, not necessarily the paper's reported numbers, and are not all confirmed to match the paper's stated results:
- Position LR: init 1.6e-4, final 1.6e-6, delay_mult 0.01, max_steps 30,000; feature LR 2.5e-3; opacity LR 0.05; scaling LR 0.005; rotation LR 0.001 (standard 3DGS schedule, unmodified).
- `lambda_dssim` (RGB L1/SSIM mix) = 0.2 (standard 3DGS default).
- `surface_mesh_laplacian_smoothing_factor` = 5.0; `area_reg_loss_factor` = 0.1.
- `cfg.mask_loss_factor` = 1; `cfg.depth_loss_factor` = 0.1; `cfg.sh_reg_loss_factor` = 1 (temporal color-consistency weight).
- `cfg.loose_bind_factor_t` = 100, `cfg.loose_bind_factor_r` = 1 — these are the code-side realization of the $\lambda_t$ split between translation and rotation residual penalties in $\mathcal{L}_{\text{unb}}$, and show translation residuals are penalized 100x more per-unit than rotation residuals.
- `cfg.min_opacity` = 0.8 — opacity floor, not mentioned in the paper text.
- `loose_bind_from` = 1000 (iteration at which unbinding is first allowed).
- Densification: `densify_from_iter` = 99 (code) vs. base 3DGS default 500/50000 depending on file; `densify_until_iter` = 7000; interval 200; `densify_grad_threshold` = 0.0001 * 0.4 (tightened relative to base 3DGS's 0.0002 default) — code applies a stricter/earlier densification schedule than vanilla 3DGS.
- Gaussian scale clamps: `max_gaussian_scale` = 0.003 or `ref_edge_len * 0.8`; `min_gaussian_scale` = 0.0003 or `ref_edge_len * 0.08`.
- Scene-flow warping (`gaustar_tools/warp_mesh.py`): `min_observe` = 4 views; `knn_K` = 8; `cmr_view_max_cos` = -0.5; `max_move_dist` = 0.2; TSDF `voxel_size` = 0.04; `bi_direct_pix_threshold` = 4; `bi_direct_depth_threshold` = 0.004; smoothing/propagation run for 5 iterations.

Where the paper states a symbol but not a value, this is marked "Not specified in paper" above rather than filled from the code, since the code default is not confirmed to be the value used to produce the paper's reported numbers.

#### Ablation Summary

From Table 2 (full method vs. component removed), full GauSTAR: PSNR 31.87 dB, SSIM 0.952, LPIPS 0.102, Chamfer Distance 0.237 cm, F-Score 0.980, 3D ATE 0.45 cm, 2D ATE 2.03 cm.

- **Scene-flow initialization is the single most impactful component for tracking accuracy**: removing it costs only -1.95 dB PSNR but +6.11 cm 3D ATE (0.45 → 6.56 cm), a >14x increase in tracking error — far larger than the tracking-error cost of removing either other component. This is the standout finding: rendering quality is fairly robust to flow-initialization removal, but tracking correctness collapses without it.
- **Unbinding**: removing it costs -2.57 dB PSNR (largest rendering-quality drop of the three), F-Score -0.042 (0.980 → 0.938), 3D ATE +2.4 cm (0.45 → 2.85 cm).
- **Re-meshing**: removing it costs -2.1 dB PSNR, F-Score -0.044 (0.980 → 0.936), 3D ATE +1.63 cm (0.45 → 2.08 cm).

Ranked by PSNR impact: unbinding (-2.57 dB) > re-meshing (-2.1 dB) > scene flow (-1.95 dB). Ranked by tracking (3D ATE) impact: scene flow (+6.11 cm) >> re-meshing (+1.63 cm) ≈ unbinding (+2.4 cm). The paper's headline comparison against baselines (Table 1, PSNR): GauSTAR 31.87 dB vs. HumanRF 30.59 dB, 2D Gaussian Splatting 30.17 dB, Dynamic 3D Gaussians 27.61 dB.

#### Implementation Reality

- **Framework:** PyTorch, extending the official 3D Gaussian Splatting rasterizer/repo structure (`gaussian_splatting/` mirrors the canonical `graphdeco-inria/gaussian-splatting` layout: `arguments/`, `gaussian_renderer/`, `scene/`, `submodules/`, `lpipsPyTorch/`) plus a SuGaR-derived mesh-Gaussian binding layer (`gaustar_scene/sugar_model.py`, `sugar_compositor.py`, `sugar_densifier.py`, `sugar_optimizer.py`). Repo is primarily flagged C++ on GitHub (custom CUDA rasterizer submodules) with a Python training/data-processing layer on top. Data processing embeds a full copy of HumanRF (for initial mesh generation) and RAFT (for optical flow).
- **Key files:** `train_seq.py`/`render_seq.py` (top-level sequential train/render entry points, run frame-by-frame over the pipeline described above); `gaustar_trainers/refine.py` and `refined_mesh.py` (the fixed-topology refinement + unbinding training loop, containing the actual loss-weight and threshold constants); `gaustar_scene/` (mesh-bound Gaussian model, compositor, densifier, optimizer — the "Gaussian Surface" representation); `gaustar_tools/warp_mesh.py` (scene-flow computation/smoothing) and `tracking_util.py`; `data_process/ahq2gaustar.py` (dataset conversion from the public ActorsHQ format, indicating ActorsHQ is used as a public evaluation dataset alongside the authors' own 52-camera capture rig).
- **Notable implementation details differing from or absent in the paper text:** the paper does not give numeric values for $\lambda_{\text{rgb}}, \lambda_{\text{depth}}, \lambda_t$, or the unbinding threshold, but the code exposes them as `cfg.depth_loss_factor=0.1`, `cfg.loose_bind_factor_t=100`, `cfg.loose_bind_factor_r=1`, with unbinding only enabled after iteration 1000 (`loose_bind_from`) — none of these are stated in the paper body. The code also applies a `cfg.min_opacity=0.8` floor and a densification gradient threshold tightened to `0.0001*0.4` versus vanilla 3DGS's `0.0002` default, neither mentioned in the paper. The repo supports both the authors' own multi-view rig data and the public ActorsHQ dataset via a conversion script, which is not emphasized in the accessible paper text (paper text only describes the authors' own capture setup).

#### Failure Modes & Limitations

The paper states GauSTAR struggles with "complex or sudden topology changes, such as when a new person suddenly enters the scene" — i.e., large non-local topology events, as opposed to the gradual/local surface splitting and appearance the unbinding mechanism is designed to localize. It requires multi-view capture rigs, explicitly limiting applicability "in general public scenarios" (no casual/monocular capture claim is made). Transparent and specular surfaces are named as a shared weakness with "most surface reconstruction methods," not specifically solved by this method.

---

## Relevance to ADAGS

The cleanest published foil for CSVL-VPL v2: GauSTAR's answer to surface
appearance/disappearance is unbind-and-re-create, while the ADAGS thesis is
hide-and-reveal-the-same-primitives (preservation, not recreation). It is
also a multiview persistent-surface-identity precedent (mesh-bound), so any
"persistent surface identity in multiview dynamic capture" claim must be
positioned against it. No occlusion-evidence-driven capacity lifecycle, no
uncertainty/abstention per the abstract.

## Connections

- Foil for [[operations/phase9-csvl-vpl-v2-direction]]
- Pressures [[ideas/event-causal-visibility-gaussians]]

## Sources

- https://arxiv.org/abs/2501.10283 (paper, v3)
- https://arxiv.org/html/2501.10283v3 (full HTML text)
- https://github.com/eth-ait/GauSTAR (code repository)
- https://eth-ait.github.io/GauSTAR/ (project page)
