---
type: paper
node_id: paper:gao2026_proxy_gs
title: "Proxy-GS: Efficient 3D Gaussian Splatting via Proxy Mesh"
authors: ["Yuanyuan Gao", "Yuning Gong", "Yifei Liu", "Jingfeng Li", "Zhihang Zhong", "Dingwen Zhang", "Yanci Zhang", "Dan Xu", "Xiao Sun"]
year: 2026
venue: "CVPR"
external_ids:
  arxiv: "2509.24421"
  doi: null
  s2: null
tags: ["occlusion", "proxy-depth", "densification", "structured-gaussians"]
added: 2026-07-14T23:36:29Z
---

# Proxy-GS: Efficient 3D Gaussian Splatting via Proxy Mesh

**Paper:** https://arxiv.org/abs/2509.24421 (published as "Proxy-GS: Unified Occlusion Priors for Training and Inference in Structured 3D Gaussian Splatting", CVPR 2026 Oral / Best Paper Candidate)
**Code:** https://github.com/Visionary-Laboratory/Proxy-GS
**Base method:** MLP-based structured 3D Gaussian Splatting — specifically Octree-GS and Scaffold-GS (anchor-based, MLP-decoded Gaussian attributes), itself built on 3D Gaussian Splatting (Kerbl et al.).

## One-line thesis

A lightweight proxy mesh, rasterized into a sub-millisecond depth map, gives training and rendering a shared occlusion prior — culling anchors/Gaussians that lie behind the proxy surface at render time, and pulling densification toward the proxy surface during training — so the two stages no longer disagree about what is visible.

## Problem / Gap

MLP-based structured 3DGS variants (Scaffold-GS, Octree-GS) improve fidelity in large scenes but pay a heavy per-anchor MLP-decoding cost at render time, and standard pruning/LOD schemes don't know which anchors are actually occluded from the current view. Concretely: anchors sitting behind other geometry (e.g. inside a building block, behind a wall) still get decoded and rasterized every frame because densification and pruning are driven purely by RGB reconstruction error, with no visibility signal — so heavily occluded regions accumulate redundant capacity that never contributes to the rendered image but still costs compute.

## Method

Proxy-GS builds a coarse proxy mesh from existing per-scene geometry (COLMAP for outdoor scenes, MapAnything for indoor scenes), simplifies it with QEM edge-collapse, and partitions it into clusters for hierarchical (Hi-Z + Early-Z) visibility culling. At render time, this proxy is rasterized via hardware rendering into a per-view depth map (1000×1000 in <1ms); anchors are projected to NDC space, and any anchor whose camera-space depth lies behind the proxy depth map (plus a safety margin γ) at its projected pixel is culled before MLP decoding and rasterization. During training, the same proxy depth is used two ways: (1) filtering is applied consistently so training and inference see the same occlusion state, and (2) high rendering-error image patches are back-projected onto the proxy mesh surface to seed new anchors directly on geometrically plausible surfaces, with a proxy-aligned voxel grid capping anchor count per cell to prevent redundant growth.

## Assumptions

Assumes a reasonably accurate static per-scene proxy mesh can be extracted upfront from an existing reconstruction pipeline (COLMAP/MapAnything) and stays valid for the whole scene — i.e. a static, non-deforming, large-scale (typically outdoor/urban) scene where standard SfM/MVS geometry is obtainable and where anchors don't move relative to the world across training.

## Limitations / Failure Modes

The paper reports only marginal gains in low-occlusion scenes (Berlin: 275 FPS for Proxy-GS vs. 263 FPS for Octree-GS — the occlusion prior has little to cull). Quality depends on proxy mesh fidelity: the method tolerates <5% vertex noise with limited degradation, but larger perturbations disrupt occlusion boundaries and hurt results. It also depends on reliable upstream geometric reconstruction, which the paper notes is harder to obtain in texture-less indoor environments even with MapAnything. The method targets static large scenes; it does not model non-rigid temporal occlusion/reveal or preserve any per-surface identity through a hidden interval.

## Reusable Ingredients

- **Hardware-rasterized proxy depth map**: sub-millisecond, high-resolution (1000×1000) occlusion depth from a coarse mesh — cheap enough to compute per-view every frame.
- **Shared train/inference occlusion prior**: applying the *same* culling rule during training and at test time, rather than only at render time, to avoid train/test mismatch (this is the paper's single biggest measured effect — see ablation).
- **Safety margin on occlusion depth (γ)**: a small additive offset on the proxy depth threshold that trades off false culling against redundant capacity.
- **Error-driven, proxy-anchored densification**: back-projecting high per-patch rendering error onto a known surface (rather than growing along raw gradient/position heuristics) to keep new capacity geometrically grounded.
- **Grid-capped anchor insertion**: a spatial hash/grid with a per-cell anchor cap (`κ[c(a)] < K`) as a simple redundancy control independent of the occlusion signal itself.

---

### Deep Dive

#### Core Novelty
The paper's change relative to Octree-GS/Scaffold-GS is not a new Gaussian representation but a new *auxiliary signal*: a fast, hardware-rasterized proxy-mesh depth map consumed identically during training and inference. The key insight is that prior occlusion-aware pruning only applied culling at test time, which decouples what the optimizer sees (all anchors, dense supervision) from what the renderer keeps (culled anchors) — causing a quality drop when culling is added post-hoc. By injecting the same occlusion prior into densification and training-time filtering, the anchor set that gets optimized is consistent with the anchor set that gets rendered.

#### Mathematical Formulation

Point culling test (per anchor, per view, before rasterization):
- Project anchor to normalized device coordinates $(x_{ndc}, y_{ndc}, z_{ndc})$, and derive homogeneous camera-space depth $z_h$.
- Discard points behind/at the camera: $z_h \le \tau$, with $\tau = 10^{-4}$.
- Map NDC $(x_{ndc}, y_{ndc})$ to pixel indices $(x_{pix}, y_{pix})$ using image width $W$, height $H$.
- Convert the hardware depth buffer value at $(x_{pix}, y_{pix})$ to linear camera-space depth $d_{mesh}(x_{pix}, y_{pix})$ using the near/far planes $n, f$.
- Cull condition: an anchor is removed if its own camera-space depth is greater (farther) than the proxy depth plus a safety margin:
$$\hat{d}(x_{pix}, y_{pix}) = d_{mesh}(x_{pix}, y_{pix}) + \gamma$$
  where $\gamma = 0.3$ (empirically chosen; ablated over {0.1, 0.3, 0.6, 1.0}). Evaluated per-anchor, per-view, before MLP decoding/rasterization — this is the shared filter applied at both training and inference.

Proxy-guided densification trigger (per image patch, during training):
- Let $\ell_P$ be a patch's rendering error and $\bar{\ell}$ the mean error over the frame. A patch is selected for new-anchor generation if
$$\ell_P > \tau, \quad \tau = 3\bar{\ell}$$
- Selected high-error pixels are back-projected onto the proxy mesh surface (not placed at arbitrary depth) to produce new anchor positions.

Grid-capped insertion (redundancy control on new anchors):
- With grid cell size $h$ and grid origin $b_{min}$, each candidate anchor $a$ maps to a cell index
$$c(a) = \left\lfloor \frac{a - b_{min}}{h} \right\rfloor \in \mathbb{Z}^3$$
- Insert $a$ only if the current anchor count in that cell, $\kappa[c(a)]$, is below a cap $K$ (i.e. `insert a if κ[c(a)] < K`).

Mesh simplification (QEM edge-collapse, used to build the lightweight proxy):
- Partitioned quadric matrix $Q' = \begin{bmatrix} A & b \\ b^\top & c \end{bmatrix}$; optimal edge-collapse target position minimizes
$$x^* = \arg\min_x \left( x^\top A x + 2 b^\top x + c \right)$$

#### Algorithm / Pipeline Changes
1. **Offline proxy construction**: reconstruct scene geometry (COLMAP outdoor / MapAnything indoor) → simplify via QEM edge-collapse with feature/boundary preservation → partition into clusters for hierarchical visibility (Hi-Z culling with Early-Z).
2. **Per-view proxy depth rendering**: hardware-rasterize the simplified proxy mesh to a depth buffer at 1000×1000 resolution (<1ms) for the current camera pose. In the released code this is precomputed offline via `mesh_render.py` into cached `.npy` depth maps consumed by `train.py --depth_npy_dir --ply_mesh`, rather than recomputed on the fly every step.
3. **Anchor/Gaussian culling (replaces plain frustum culling)**: for every anchor, project to NDC, discard near-camera invalid points ($z_h \le 10^{-4}$), look up proxy depth at the projected pixel, and cull if the anchor's depth exceeds $d_{mesh} + \gamma$. Applied identically whether the current forward pass is a training step or a test-time render — this is the pipeline stage that differs from prior occlusion-culling work (which only culled at inference).
4. **Proxy-guided densification (augments/replaces standard densification)**: after computing per-pixel rendering error, aggregate into patches, flag patches with $\ell_P > 3\bar{\ell}$, back-project their pixels onto the proxy mesh to get 3D positions for new anchors (instead of cloning/splitting existing anchors along positional gradients).
5. **Grid-capped anchor insertion**: before finalizing a new anchor from step 4, check its cell occupancy $\kappa[c(a)]$ against cap $K$; skip insertion if the cell is already full.
6. Downstream: decoded Gaussians from surviving anchors proceed through the normal Octree-GS/Scaffold-GS MLP-decoding and rasterization pipeline, now over a reduced (culled) and more surface-aligned anchor set.

#### Key Hyperparameters & Design Choices
- Near-plane validity threshold $\tau = 10^{-4}$ (for $z_h$).
- Occlusion safety margin $\gamma = 0.3$ (ablated: 0.1, 0.3, 0.6, 1.0 — 0.3 chosen as the balance point on the Small City dataset).
- Densification error-patch threshold multiplier $\tau = 3\bar{\ell}$ (3× mean frame error).
- Grid cell size $h$ and per-cell anchor cap $K$: exact numeric values not specified in paper (reported structurally as `c(a) = ⌊(a−b_min)/h⌋`, cap $K$, but no concrete $h$/$K$ given).
- Baseline densification gradient threshold reduced to $10^{-4}$ for non-MLP method comparisons (fairness adjustment, not a Proxy-GS hyperparameter per se).
- Training: 40,000 iterations, single NVIDIA A100-40GB for training; inference benchmarked on consumer RTX 4090.
- Built on top of the Octree-GS framework.

#### Ablation Summary
(Block 5, MatrixCity; PSNR / FPS / avg. anchors, from Table 3)
1. Baseline Octree-GS: 21.41 PSNR, 48 FPS, 719k anchors.
2. + Inference-only proxy filtering: 19.06 PSNR, 165 FPS, 82k anchors — **−2.35 dB PSNR drop despite 3× FPS gain**, from train/test occlusion-filtering inconsistency. This is flagged as the most impactful (and most cautionary) result: naive test-time-only culling actively hurts quality.
3. + Training-time proxy filtering added (consistent train/test occlusion): 21.50 PSNR, 147 FPS, 93k anchors — recovers above baseline PSNR while keeping most of the speedup.
4. + Proxy-guided densification added: 21.68 PSNR, 143 FPS, 106k anchors — best quality, comparable FPS to (3).
Additional: safety-margin ablation (Table 10, Small City) shows $\gamma=0.1$ introduces visible artifacts (too little culling margin), while larger $\gamma$ over-grows anchors and reduces FPS; $\gamma=0.3$ is the reported optimum. Depth-acquisition method comparison (Table 6): the proxy-based depth path reaches 151 FPS vs. 54 FPS (3DGS depth extraction) and 32 FPS (nvdiffrast).

#### Implementation Reality
- **Framework:** PyTorch + CUDA, extending the Octree-GS/Scaffold-GS codebase; custom CUDA extensions `diff-gaussian-rasterization` and `simple-knn` as submodules (standard 3DGS-family dependencies). An optional Vulkan-CUDA interop backend (`ProxyGS-Vulkan-Cuda-Interop/`) is provided for real-time inference via `render_real.py`.
- **Key files:** `mesh_render.py` (rasterizes the proxy mesh into per-view depth `.npy` caches — this precomputation step is not spelled out as a distinct offline stage in the paper's method description); `train.py` (accepts `--depth_npy_dir` and `--ply_mesh` to wire the cached proxy depth into training-time filtering and densification); `Mesh2DepthHelper/` (proxy depth/mesh handling); `gaussian_renderer/` vs. `gaussian_renderer_inference/` (separate rendering code paths for train vs. test, presumably because the inference path integrates the Vulkan-CUDA interop for speed); `pose_block/` (appears to handle spatial/pose decomposition for large scenes like MatrixCity, not discussed in the paper text obtained).
- **Notable implementation details:** the repo precomputes and caches proxy depth maps per view offline (`mesh_render.py` → `.npy`) rather than rasterizing the proxy live inside the training loop every step as the paper's "<1ms per view" framing might suggest to a naive reader; the existence of separate train/inference renderer modules and an optional Vulkan backend are engineering details not mentioned in the paper itself.

#### Failure Modes & Limitations
Marginal benefit in low-occlusion scenes (Berlin: 275 vs. 263 FPS — proxy culling has little to remove). Robust to small proxy-mesh vertex noise (<5%) but degrades once noise disrupts occlusion boundaries. Depends on a reconstruction pipeline (COLMAP/MapAnything) producing usable geometry; the paper notes this is harder in texture-less indoor scenes. Scoped to static large scenes — no treatment of non-rigid motion, temporal occlusion/reveal, or surface identity persistence through occluded intervals.

---

## Relevance to ADAGS

Proxy-GS is a direct precedent for using occlusion depth to change densification. ADAGS cannot claim visibility-guided capacity as new in general; its working distinction must concern calibrated non-rigid reveal state, hidden-surface learning, uncertainty, and budget-neutral preservation/reassignment.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Sources

- https://arxiv.org/abs/2509.24421
- https://github.com/Visionary-Laboratory/Proxy-GS
