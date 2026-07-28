---
type: paper
node_id: paper:liu2025_occlugaussian
title: "OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering"
authors: ["Shiyong Liu", "Xiao Tang", "Zhihao Li", "Yingfan He", "Chongjie Ye", "Jianzhuang Liu", "Binxiao Huang", "Shunbo Zhou", "Xiaofei Wu"]
year: 2025
venue: "ICCV"
external_ids:
  arxiv: "2503.16177"
  doi: null
  s2: null
tags: ["occlusion", "camera-covisibility", "large-scenes", "scene-partitioning"]
added: 2026-07-14T23:36:29Z
---

# OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering

**Paper:** https://arxiv.org/abs/2503.16177
**Code:** Not found (project page lists "Code (coming soon)" at https://github.com/OccluGaussian/OccluGaussian; no release as of the deep-dive date)
**Base method:** 3D Gaussian Splatting (Kerbl et al. 2023), with divide-and-conquer large-scene partitioning in the style of VastGaussian (whose appearance-modeling module it directly adopts), compared against CityGaussian, Hierarchical-GS, and DOGS.

## One-line thesis

Partitioning a large static scene by camera co-visibility (via spectral clustering of an attributed camera graph) instead of by spatial position/grid yields regions whose cameras are mutually informative, which produces better per-region reconstructions and enables cheap region-level Gaussian culling at render time.

## Problem / Gap

Prior large-scene 3DGS pipelines (VastGaussian, CityGaussian, Grid-NeRF-style methods) partition scenes purely by camera position or a spatial grid. This is occlusion-agnostic: a region can group cameras that are physically close but see almost disjoint content (e.g., separated by a wall or building), so each camera in the region contributes little to reconstructing what the others see. The result is weak per-region supervision and degraded quality in occluded, cluttered large scenes.

## Method

OccluGaussian builds an attributed graph over training cameras where edge weight equals the number of matched SfM feature points shared between camera pairs (a co-visibility signal) and node features are positionally-encoded 3D camera coordinates. It applies spectral graph clustering (via graph-Laplacian smoothing/graph convolution followed by similarity-matrix clustering) to this graph to partition cameras into regions with strong internal co-visibility, then iteratively splits oversized and merges undersized clusters to keep region camera counts balanced. Each region is reconstructed independently using three camera subsets — base (inside the region), extended (outside cameras that still see the region), and border (occluded cameras facing the region, added purely to constrain Gaussian shape near boundaries) — after which out-of-region Gaussians are trimmed and regions are merged into one model. At render time, a precomputed per-region visibility mask culls Gaussians that never contributed rendering weight to that region's training cameras, with a border-shrinking region-subdivision trick to further tighten the mask and speed up rasterization without touching quality.

## Assumptions

The scene is static and large-scale, with enough SfM feature correspondences between camera pairs to build a meaningful co-visibility graph, and it assumes independently-optimized spatial partitions can be seamlessly merged into one coherent global Gaussian field.

## Limitations / Failure Modes

The authors state that camera clustering starts from a fixed initial cluster count (K=10), which worked in their experiments but "may be insufficient for extremely large scenes." The method is entirely offline/static-scene: it reasons about which cameras co-observe geometry, not about per-ray or per-timestep occlusion, so it has no notion of temporal reveal/occlusion events, surface identity, or a capacity lifecycle — it culls already-trained Gaussians for rendering speed rather than deciding what to grow, protect, or retire during optimization.

## Reusable Ingredients

- **Attributed camera co-visibility graph** (SfM-matched-feature-count edge weights + positionally-encoded camera-position node features) — a general way to quantify which observations are mutually informative about the same content.
- **Spectral clustering with balance-driven cluster count refinement** — split-oversized/merge-undersized iteration to keep partitions balanced without hand-tuning K per scene.
- **Base/extended/border camera role split** — explicitly separating "cameras that constrain shape at a boundary" from "cameras that supervise appearance" is a reusable pattern for any spatially- or temporally-partitioned training scheme.
- **Post-hoc visibility-mask culling from accumulated rasterization weight** (threshold 0.01) — a cheap, training-free way to determine which primitives are irrelevant to a given viewpoint/region for render-time acceleration.

---

### Deep Dive

#### Core Novelty
Relative to VastGaussian/CityGaussian-style spatial partitioning, OccluGaussian replaces position/grid-based camera assignment with a co-visibility-driven spectral clustering of an attributed camera graph, and adds a region-based rendering (RBR) culling step derived from accumulated per-Gaussian rasterization weight. The insight is that spatial proximity is a poor proxy for "these cameras jointly constrain the same geometry" in occluded large scenes — co-visibility (shared matched features) is the more direct signal, and once regions are defined this way, the same visibility evidence used to build regions can be reused to cull invisible Gaussians per region at render time.

#### Mathematical Formulation

**Symmetric normalized graph Laplacian**, computed once over the camera graph $G=(V,E,X)$:
$$L_s = I - D^{-1/2} A D^{-1/2}$$
where $A$ is the weighted adjacency matrix (edge weight = matched-feature count between camera pairs) and $D$ is its degree matrix. Evaluated once, before clustering, over the full training-camera set.

**Graph convolution / low-pass filtering** of camera features $X$ (3D camera position with positional encoding):
$$\bar{f} = G \cdot f,\quad G = \left(I - \tfrac{1}{2}L_s\right)^r$$
producing smoothed features $\bar{Y} = GX$; $r$ is the number of filtering iterations. Evaluated before spectral clustering to make co-visible cameras' features more similar.

**Similarity matrix for clustering:**
$$\bar{I} = \frac{|H| + |H^\top|}{2},\quad H = \bar{Y}\bar{Y}^\top$$
Symmetrized affinity matrix fed into standard spectral clustering to produce the initial region assignment.

**Cluster balance constraint:** clusters are iteratively split/merged until every region's camera count falls in
$$[M_c - \sigma_c M_c,\; M_c + \sigma_c M_c],\quad \sigma_c = 0.5$$
where $M_c$ is the mean camera count per region. Applied after initial spectral clustering, before per-region training.

**Region visibility mask:** for region $R_j$, mask $M_j = \{m_{ij} \in \{0,1\}\}$ over Gaussians $i$, set via
$$m_{ij} = 1 \iff \text{accumulated rasterization weight of Gaussian } i \text{ over cameras in } R_j > 0.01$$
Computed post-training per region (using both forward and backside renders for coverage), and used at inference to cull Gaussians whose mask bit is 0 for the viewpoint's region.

**Region subdivision boundary shrink:** interior/border sub-region boundary lines are shrunk inward by
$$0.1 \cdot d_{\max}$$
where $d_{\max}$ is the maximal inter-camera distance in the region, producing tighter (smaller) visibility masks for the interior sub-region to further cull Gaussians without quality loss.

#### Algorithm / Pipeline Changes
1. Build attributed camera graph $G=(V,E,X)$ from SfM: nodes = training cameras, edge weight = number of matched feature points between camera pairs, node feature = positionally-encoded 3D camera position.
2. Compute normalized Laplacian $L_s$, apply $r$-step graph-convolution smoothing to get filtered features $\bar Y$, form similarity matrix $\bar I$, run spectral clustering to get an initial $K$-way camera partition (initial $K=10$).
3. Iteratively split clusters that are too large and merge/remove clusters that are too small until camera counts per region satisfy the $\sigma_c=0.5$ balance bound; final region counts differ from $K$ per scene (e.g., 5 for Gallery/Canteen, 8 for Berlin, 7 for NYC).
4. For each resulting region, assign three camera roles: base (physically inside), extended (outside but co-visible with region content), border (occluded/boundary-facing, included only to constrain Gaussian geometry near edges and prevent elongated floaters).
5. Train each region independently with standard 3DGS optimization (90,000 iterations, densification from iteration 1,500 to 45,000 every 100 iterations), using VastGaussian's appearance-modeling module; loss is unmodified standard 3DGS $L_1$ + D-SSIM.
6. After per-region optimization, discard Gaussians that fall outside the region's spatial boundary, then merge all regions into one global Gaussian model.
7. Post-training, for each region compute a per-Gaussian visibility mask from accumulated rasterization weight (threshold 0.01) across that region's cameras (front and back), then shrink region-interior boundaries by $0.1\,d_{\max}$ to produce an additional, tighter interior sub-region mask.
8. At render time, determine which region the current viewpoint belongs to, fetch that region's (sub-)mask, and cull all Gaussians with mask bit 0 before rasterization — this replaces "render the full merged model" with "render only the region-visible subset."

#### Key Hyperparameters & Design Choices
- Initial cluster count $K = 10$ (then adaptively refined per scene; e.g. final counts 5–8).
- Cluster balance tolerance $\sigma_c = 0.5$ around mean camera count $M_c$.
- Visibility/culling weight threshold: $0.01$ accumulated rasterization contribution.
- Region-interior boundary shrink: $0.1 \cdot d_{\max}$.
- Training: 90,000 iterations per region; densification active from iteration 1,500 to 45,000 at 100-iteration intervals.
- Graph-convolution filtering steps $r$: not specified in paper.
- Loss weights: unmodified standard 3DGS $L_1$ + D-SSIM weighting (not changed from Kerbl et al.).

#### Ablation Summary
- **Region-based rendering (RBR) + region subdivision (RSD), Gallery scene:** no culling/subdivision = 189.52 FPS → culling only = 271.79 FPS → culling + subdivision (full) = 288.94 FPS, with PSNR/SSIM held constant (25.81 / 0.903) — i.e., the culling and subdivision steps are purely a rendering-speed contribution (≈52% faster than no culling) with no measured quality cost. This is the single most impactful ablated component for the paper's speed claim.
- **Division strategy (Table 6, Canteen/Berlin), all else equal:** VastGaussian-style division 24.60/29.48 PSNR vs. CityGaussian-style division 24.16/28.68 PSNR vs. OccluGaussian's co-visibility division 25.25/30.37 PSNR — co-visibility-based division is the most impactful contributor to reconstruction quality, worth roughly +0.65–1.69 dB over the next-best division strategy on these two scenes.
- **Initial cluster number K robustness (Table 7, NYC):** K=7, 10, 15 all yield ~30.9–31.35 PSNR with minimal variance, indicating the method is not highly sensitive to the exact initial K given the adaptive balance refinement.

#### Failure Modes & Limitations
The paper states the fixed initial cluster number ($K=10$) "works well in our experiments but may be insufficient for extremely large scenes," flagged as future work. Beyond this stated limitation, the method's border-camera mechanism is explicitly needed to prevent elongated/floater Gaussians near region boundaries — implying that without it, per-region independent optimization produces boundary artifacts.

---

## Relevance to This Project

OccluGaussian is an occlusion-aware Gaussian precedent and prevents broad novelty wording. Its static scene-division mechanism is nevertheless distinct from Route 1's proposed dynamic surface-state and budget-reassignment hypothesis.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Sources

- https://arxiv.org/abs/2503.16177
- https://occlugaussian.github.io
- https://openaccess.thecvf.com/content/ICCV2025/html/Liu_OccluGaussian_Occlusion-Aware_Gaussian_Splatting_for_Large_Scene_Reconstruction_and_Rendering_ICCV_2025_paper.html
