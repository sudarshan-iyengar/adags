---
type: paper
node_id: paper:lin2025_depth_anything_3
title: "Depth Anything 3: Recovering the Visual Space from Any Views"
authors: ["Haotong Lin", "Sili Chen", "Junhao Liew", "Donny Y. Chen", "Zhenyu Li", "Guang Shi", "Jiashi Feng", "Bingyi Kang"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2511.10647"
  doi: null
  s2: null
tags: ["depth", "any-view-geometry", "pose-conditioning", "foundation-model"]
added: 2026-07-14T22:18:30Z
status: deep-dived
---

# Depth Anything 3: Recovering the Visual Space from Any Views

**Paper:** https://arxiv.org/abs/2511.10647
**Code:** https://github.com/ByteDance-Seed/Depth-Anything-3
**Base method:** Vanilla DINOv2 ViT backbone (no multi-branch specialization), successor to Depth Anything 2 (monocular) and directly benchmarked against VGGT (multi-view geometry transformer).

## One-line thesis

A single plain ViT backbone predicting a unified per-pixel depth-and-ray representation, with an optional lightweight camera-token conditioning path, is sufficient to recover consistent geometry from any number of views (1 to many) without specialized multi-branch architectures or an explicit point-cloud/camera-pose auxiliary head.

## Problem / Gap

Prior visual-geometry foundation models (e.g., VGGT-style architectures) stack multiple specialized branches (separate heads/streams for depth, point maps, and camera pose) and use two-ViT designs, which the paper shows is architecturally wasteful: their own ablation finds a VGGT-style 2×ViT-B stack scores 3.72 Auc3 versus 39.2 Auc3 for a single ViT-L with the depth-ray head under matched conditions (90% relative degradation). Existing monocular depth models (Depth Anything 2) cannot consume multiple views or known camera poses to improve consistency, and multi-view stereo/pose pipelines are usually separate systems from monocular depth estimation.

## Method

DA3 takes 1-18+ images, tokenizes each with a shared DINOv2 backbone, and applies a two-phase attention schedule: the first `Ls` transformer layers run within-image self-attention only, and the remaining `Lg` layers alternate between within-view and cross-view (token-rearranged) self-attention, so a single image collapses naturally to monocular depth estimation while multiple images get cross-view consistency. A dual-DPT head decodes each token stream into a per-pixel depth map and a per-pixel 6D ray map (origin + unnormalized direction) rather than separate depth/point-cloud/camera heads. If known camera intrinsics/extrinsics are supplied, a lightweight MLP camera encoder produces a per-image camera token that is prepended to the patch tokens and participates in all attention (used during training with probability 0.2); if poses are absent, a shared learnable placeholder token is used instead. Explicit 3D points and camera parameters are then recovered post hoc from the depth+ray output via closed-form geometry (homography/RQ decomposition), not via separate learned heads.

## Assumptions

Assumes broad, largely rigid-scene public training data (68 sources: synthetic renders, LiDAR, COLMAP reconstructions, real captures) is sufficient for the geometry prior; dynamic content is not explicitly modeled as a first-class signal, and non-rigid/temporal correspondence is outside the trained objective. Assumes camera pose conditioning, when available, is provided in a compatible calibrated intrinsics/extrinsics format.

## Limitations / Failure Modes

The paper's own ablations show performance is architecture- and scale-sensitive: dropping the dual-DPT head for a generic head costs 86% of Auc3 (39.2 → 5.59), and removing teacher pseudo-label supervision on synthetic data costs 71% (39.2 → 11.2), meaning the released numbers depend heavily on training-recipe choices not just the depth-ray representation itself. Metric depth accuracy still trails specialized single-purpose models (UniDepthv2) on NYUv2/KITTI, and pose-conditioning gains plateau on datasets with limited video-sequence diversity (7Scenes). The paper does not explicitly discuss dynamic/non-rigid scene failure modes; ADAGS's own R031 usage found DA3 does not by itself provide occlusion-state or foreground/background ordering reasoning.

## Reusable Ingredients

- **Depth-ray unified output (M ∈ ℝ^(H×W×6), r=(t,d))** — replaces separate depth/point-cloud/camera heads with one representation that closed-form geometry can recover cameras from, avoiding multi-head training instability.
- **Optional camera-token conditioning** — a per-image MLP-encoded pose token prepended to patch tokens, trained with 0.2 dropout probability, giving a single model that works with or without known cameras at negligible (~0.1%) extra compute.
- **Staggered within-view/cross-view attention schedule (Ls:Lg ≈ 2:1)** — lets one architecture handle both monocular and multi-view input without branching.
- **RANSAC scale-shift teacher-to-student depth alignment** — `(ŝ,t̂) = argmin Σ mp(s·D̃p + t − Dp)²` for bootstrapping noisy metric ground truth from a relative-depth teacher, reusable wherever noisy metric GT needs calibration against a cleaner relative signal.
- **Constant-token-count dynamic batching across resolutions** — enables training/inference across mixed aspect ratios and view counts without re-tuning batch size per resolution.

---

### Deep Dive

#### Core Novelty
Relative to VGGT-style multi-branch visual geometry transformers, DA3's change is architectural collapse: one plain ViT, one unified depth+ray output head, and pose information injected as an optional token rather than a separate conditioning pathway or camera-regression head. The key insight is that camera parameters and 3D points are *derivable* from a dense per-pixel ray field via classical geometry, so the network only needs to learn a single consistent field rather than multiple redundant, potentially conflicting outputs (depth, explicit point cloud, and camera pose regression) that in the ablation actively hurt accuracy when kept as separate heads/branches.

#### Mathematical Formulation
- **Ray map**: $M \in \mathbb{R}^{H\times W \times 6}$, per pixel $r = (t, d)$ where $t \in \mathbb{R}^3$ is the ray origin (camera center) and $d \in \mathbb{R}^3$ is an *unnormalized* ray direction (its magnitude encodes projection scale). Evaluated per-pixel at the dual-DPT head output.
- **3D point recovery**: $P = t + D(u,v) \cdot d$ (element-wise), i.e. depth times ray direction plus origin — computed post-hoc from the depth and ray maps, not learned directly.
- **Camera recovery**: cameras are recovered from the ray map via a homography $H = KR$ solved by Direct Linear Transform; the camera center is the average of per-pixel ray origins, and intrinsics/rotation come from an RQ decomposition of $H$. This runs as a closed-form post-processing step after the network forward pass, not inside training.
- **Camera token**: $c_i = \mathcal{E}_c(f_i, q_i, t_i)$, a lightweight MLP encoding of (focal, quaternion, translation) into a token prepended to image $i$'s patch tokens, when pose is known; a shared learnable token $c_l$ substitutes when pose is unknown. Applied before the transformer's attention layers.
- **Total training loss** (per training step):
$$L = L_D(\hat D, D) + L_M(\hat R, M) + L_P(\hat D \odot d + t, P) + \beta L_C(\hat c, v) + \alpha L_{grad}(\hat D, D)$$
  with $\alpha=1$, $\beta=1$.
- **Depth loss**: $L_D = \frac{1}{Z_\Omega}\sum_{p\in\Omega} m_p\left(D_{c,p}|\hat D_p - D_p| - \lambda_c \log D_{c,p}\right)$ — an uncertainty/confidence-weighted L1 term ($D_{c,p}$ is a per-pixel confidence, $\lambda_c$ regularizes confidence from collapsing to zero), evaluated after rendering the depth head output.
- **Gradient loss**: $L_{grad} = \|\nabla_x \hat D - \nabla_x D\|_1 + \|\nabla_y \hat D - \nabla_y D\|_1$, encourages sharp depth edges.
- **Teacher supervision loss** (synthetic-data pretraining only): $L_T = \alpha L_{grad} + L_{gl} + L_N + L_{sky} + L_{obj}$, $\alpha = 0.5$, combining a global-local depth loss, a distance-weighted surface-normal loss, and sky/object mask losses.
- **Teacher-to-student scale-shift alignment**: $(\hat s, \hat t) = \arg\min \sum_{p\in\Omega} m_p (s\tilde D_p + t - D_p)^2$, solved via RANSAC; the aligned teacher depth $D_{T\to M} = \hat s \tilde D + \hat t$ substitutes for noisy metric ground truth after step 120k of training.

#### Algorithm / Pipeline Changes
1. Tokenize each of 1-18+ input images independently with a shared DINOv2 ViT patch embedding.
2. Run the first `Ls` transformer layers (ratio `Ls:Lg ≈ 2:1` of total depth `L`, e.g. `L=12` for the base model) as within-image self-attention only.
3. If camera poses/intrinsics are supplied for an image, encode them via a small MLP into a camera token and prepend it to that image's patch-token sequence (train-time inclusion probability 0.2); otherwise prepend a shared learnable placeholder token. This happens before the remaining transformer layers.
4. Run the remaining `Lg` layers, alternating between within-view self-attention and cross-view self-attention (via token rearrangement across the image batch), so information mixes across views.
5. Feed each image's final token stream through a dual-DPT decoder head that outputs a per-pixel depth map and a per-pixel 6D ray map jointly (this replaces separate depth/point/camera heads used in prior multi-branch designs).
6. Post-hoc, non-learned: recover explicit 3D points via $P = t + D \cdot d$, and recover camera intrinsics/extrinsics via DLT homography solve + RQ decomposition on the ray map, when explicit camera parameters are needed downstream (e.g. for 3DGS).
7. For 3D Gaussian Splatting outputs, predict Gaussian parameters in local camera space, then unproject/scale to world space using either the supplied poses or the model's own predicted poses (pose-adaptive design).
8. Training-time-only: for the first 120k of 200k total steps, supervise depth against ground truth directly; after 120k steps, transition to teacher-pseudo-label supervision (RANSAC-aligned) on the synthetic-data subset.

#### Key Hyperparameters & Design Choices
- Backbone: vanilla DINOv2 ViT, model sizes Small (0.03B), Base (0.11B), Large (0.36B), Giant (1.1B).
- Attention layer split: `Ls:Lg ≈ 2:1`, e.g. `L=12` total layers for the base-scale config.
- Camera-token conditioning dropout: pose supplied with probability 0.2 during training.
- Loss weights: $\alpha = 1$ (main gradient loss), $\beta = 1$ (camera loss), teacher loss $\alpha = 0.5$.
- Training: 128 H100 GPUs, 200k total steps, 8k warmup steps, peak learning rate $2\times10^{-4}$.
- Base training resolution: 504×504 (chosen for divisibility by 2,3,4,6,9,14 to support multiple aspect ratios); also trained at 504×378, 896×504, etc.
- View count during training: 2-18 views per batch item.
- Supervision transition point: ground-truth → teacher pseudo-labels at step 120k of 200k.
- Training data: 68 sources (synthetic, LiDAR, COLMAP, real), including Objaverse (505k scenes), Trellis (557k), AriaSyntheticEnvironments (99k), Co3Dv2 (30k), ScanNet++, ARKitScenes, and others; teacher model itself trained on 20 synthetic datasets.

#### Ablation Summary
(Auc3 on HiRoom benchmark; deltas relative to full proposed configuration at 39.2 unless noted)
- **Dual-DPT head vs. generic head**: 39.2 → 5.59 (**-86%, single most impactful component**).
- **Teacher pseudo-label supervision vs. none**: 39.2 → 11.2 (-71%).
- **Single ViT-L architecture vs. VGGT-style 2×ViT-B stack**: 39.2 vs. 3.72 (-90% for the stacked alternative).
- **Partial (staggered) cross-view attention vs. full alternation attention**: 39.2 vs. 24.7 (-37% for full alternation).
- **Depth+Ray representation vs. Depth+PointCloud+Camera heads**: 48.7 vs. 9.1 Auc3 on the depth-ray-sufficiency comparison (-81% for the multi-head alternative); adding an auxiliary camera head on top of depth+ray gave no further benefit.
- **Pose conditioning present vs. absent**: 65.8 (w/o poses) → 73.8 (w/ poses), a +12% gain, on the metric where this comparison was reported.
- **Model scale** (not an ablation of a component but of capacity): Giant 1.1B = 80.3, Large 0.36B = 58.7, Base 0.11B = 19.0, Small 0.03B = 9.49 Auc3 on HiRoom.

#### Implementation Reality
- **Framework:** PyTorch, custom repository (not a fork of gaussian-splatting or another public codebase); ships its own CLI and API docs.
- **Key files:** `src/depth_anything_3/model/da3.py` contains the core model definition (backbone, attention scheduling, dual-DPT head, camera token logic). The repo also ships `da3_streaming/` for sliding-window streaming inference on long videos (<12GB GPU memory), and `docs/API.md` / `docs/CLI.md` documenting the known-camera-conditioning and multi-view API surface referenced in ADAGS's R031 usage.
- **Notable implementation details:** A November 2025 update added DA3-Streaming specifically for ultra-long video sequences via sliding-window inference — this streaming mode is not described in the arXiv paper's main method section and is a repo-only addition. Model weights are released at four scales (Small/Base/Large/Giant), trained exclusively on public academic data per the repo's stated licensing scope.

#### Failure Modes & Limitations
The paper does not explicitly analyze dynamic-scene or non-rigid failure modes. Reported quantitative weaknesses: metric depth still trails specialized single-task models (UniDepthv2) on NYUv2/KITTI; pose-conditioning benefit plateaus on 7Scenes, attributed to limited video-sequence diversity in that benchmark; and the large ablation deltas (e.g. -86% without the dual-DPT head, -71% without teacher pseudo-labels) indicate the released model's accuracy is tightly coupled to specific training-recipe choices rather than being robust to architectural or supervision substitutions.

---

## Relevance to This Project

R031 used the checkpoint but omitted its camera-conditioning and common-geometry capabilities. DA3 remains a useful uncertainty-bearing cue, not an occlusion oracle.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Sources

- https://arxiv.org/abs/2511.10647
- https://arxiv.org/html/2511.10647
- https://github.com/ByteDance-Seed/Depth-Anything-3
