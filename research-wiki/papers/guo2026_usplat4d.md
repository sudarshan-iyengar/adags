---
type: paper
node_id: paper:guo2026_usplat4d
title: "Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction"
authors: ["Fengzhi Guo", "Chih-Chuan Hsu", "Sihao Ding", "Cheng Zhang"]
year: 2026
venue: "ICLR"
external_ids:
  arxiv: "2510.12768"
  doi: null
  s2: null
tags: ["uncertainty", "dynamic-gaussians", "occlusion", "motion-propagation"]
added: 2026-07-14T22:18:30Z
---

# USplat4D: Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction

**Paper:** https://arxiv.org/abs/2510.12768
**Code:** https://github.com/TAMU-Visual-AI/usplat4d
**Base method:** Post-hoc refinement layer over an existing per-Gaussian dynamic reconstruction (the paper adapts both Shape of Motion / "SoM" (Wang et al. 2025a) and MoSca (Lei et al. 2025) as the underlying model whose motion it refines), itself built on 3D Gaussian Splatting.

## One-line thesis

Per-Gaussian uncertainty, estimated from how well each Gaussian's rendered pixels have converged, can single out a sparse set of reliably-observed "key" Gaussians and propagate their motion — via an uncertainty-weighted kNN graph and dual-quaternion blending — to the remaining poorly-observed Gaussians, instead of optimizing every primitive's motion with equal weight.

## Problem / Gap

Monocular dynamic Gaussian Splatting pipelines (Shape of Motion, MoSca) fit per-Gaussian motion trajectories directly from a single video, but treat every Gaussian identically during optimization even though some are seen repeatedly and unoccluded while others are glimpsed rarely, from oblique angles, or under occlusion. Concretely, a Gaussian on a surface patch that's occluded for most of the sequence gets the same gradient-driven motion update as a well-observed one, so its trajectory drifts and produces geometry artifacts and poor extreme-viewpoint synthesis — because nothing in the objective distinguishes "well-constrained" from "poorly-constrained" primitives.

## Method

USplat4D is a refinement stage applied on top of a pretrained per-Gaussian dynamic GS model (SoM or MoSca). For each Gaussian at each timestep it estimates a scalar uncertainty from the photometric-loss curvature at convergence, converts it into a depth-aware anisotropic 3D covariance-like uncertainty ellipsoid, then uses a 3D voxel grid to select a small fraction (~2%) of Gaussians as "key nodes" — those with a long enough run of low uncertainty. Key nodes are connected into a spatio-temporal graph using uncertainty-weighted (Mahalanobis-distance) kNN, and every non-key Gaussian is assigned to its nearest key node under the same uncertainty-weighted distance. Motion is then re-optimized: key nodes keep an uncertainty-weighted fit to their original trajectory, while non-key nodes are additionally pulled toward a dual-quaternion blend of their assigned key nodes' rigid transforms, so uncertain primitives inherit motion from reliable anchors rather than fitting noisy per-Gaussian gradients directly.

## Assumptions

Assumes a pretrained monocular dynamic Gaussian model (SoM/MoSca) already provides a usable per-Gaussian initialization and per-Gaussian motion trajectories $\mathbf{p}_{i,t}^o$ to refine, and that repeated/well-converged photometric observation is a valid proxy for "which primitives are geometrically reliable." Assumes scene topology is roughly stable enough for a single spatio-temporal graph (built from key-node positions) to remain meaningful across the sequence.

## Limitations / Failure Modes

The paper's Appendix E ("Challenging Cases") reports degradation on textureless regions, where high photometric uncertainty makes it hard to select trustworthy key nodes/anchors; on fast motion, where temporal discretization of the graph is too coarse to track rapid displacement; and on strongly deforming objects, where the graph-construction assumption of relatively stable topology breaks down. It is a post-hoc reorganization of an existing primitive set's motion optimization — it does not infer occlusion/reveal geometry, create hidden-surface capacity, or add/remove primitives.

## Reusable Ingredients

- **Convergence-gated scalar uncertainty**: derive per-Gaussian, per-timestep uncertainty from the curvature of the photometric loss at its optimum, with a fallback penalty value for pixels that never converged — a cheap way to get an uncertainty signal without a learned network.
- **Depth-aware anisotropic uncertainty ellipsoid**: lift a scalar uncertainty into a 3D ellipsoid via axis scale factors and the camera rotation, so uncertainty is directionally informed by viewing geometry.
- **Key/non-key primitive split via long-run low uncertainty**: use a temporal persistence criterion (≥5 consecutive low-uncertainty frames), not just a single-frame threshold, to select reliable anchors.
- **Uncertainty-weighted (Mahalanobis) kNN graph**: build correspondence/propagation graphs using distances weighted by combined uncertainty covariances rather than plain Euclidean distance.
- **Dual quaternion blending for motion propagation**: interpolate rigid SE(3) motion from multiple anchor nodes onto a dependent point, weighted by graph edge weights.

---

### Deep Dive

#### Core Novelty
Relative to SoM/MoSca, USplat4D does not change the Gaussian representation or the base motion-fitting objective; it inserts an uncertainty-estimation and graph-propagation stage that reweights and reorganizes how motion is optimized after the base model is (pre)trained. The key insight is that photometric convergence quality is a usable, unsupervised proxy for per-primitive reliability, and that letting reliable ("key") primitives anchor the motion of unreliable ones via a distance metric that itself accounts for uncertainty (Mahalanobis rather than Euclidean) avoids propagating motion through noisy, poorly-observed intermediaries.

#### Mathematical Formulation

Scalar per-Gaussian, per-frame uncertainty, from the local-minimum curvature of the photometric loss:
$$\sigma_{i,t}^{2}=\left(\sum_{h \in \Omega_{i,t}}(T_{i,t}^{h}\alpha_{i})^{2}\right)^{-1}$$
where $\Omega_{i,t}$ is the set of pixels Gaussian $i$ contributes to at frame $t$, $T_{i,t}^h$ is its transmittance at pixel $h$, and $\alpha_i$ its opacity. Lower accumulated (transmittance·opacity)² mass → higher uncertainty. Evaluated per-Gaussian, per-frame, after the base model's rendering pass.

Convergence gating:
$$u_{i,t}=\mathds{1}_{i,t}\,\sigma_{i,t}^{2}+(1-\mathds{1}_{i,t})\,\phi$$
$\mathds{1}_{i,t}$ indicates whether pixel/Gaussian $i$'s color error at frame $t$ is below a threshold $\eta_c$ (i.e. "converged"); unconverged primitives are assigned a fixed large penalty $\phi$ instead of the (potentially misleadingly small) curvature estimate.

Depth-aware anisotropic uncertainty (lifts scalar $u_{i,t}$ into a 3D ellipsoid in world space):
$$\mathbf{U}_{i,t}=\mathbf{R}_{wc}\,\mathbf{U}_{c}\,\mathbf{R}_{wc}^{\mathsf{T}}, \quad \mathbf{U}_{c}=\text{diag}(r_{x}u_{i,t},\,r_{y}u_{i,t},\,r_{z}u_{i,t})$$
where $\mathbf{R}_{wc}$ is the world-from-camera rotation and $r_x,r_y,r_z$ are fixed axis scale factors (depth axis typically scaled differently from image-plane axes). Computed per-Gaussian, per-frame, before graph construction.

Key-node graph edges (uncertainty-weighted kNN among key nodes $\mathcal{V}_k$):
$$\mathcal{E}_{i}=\text{$k$NN}_{j \in \mathcal{V}_{k} \setminus \{i\}}\left(\left\|\mathbf{p}_{i,\hat{t}}-\mathbf{p}_{j,\hat{t}}\right\|_{(\mathbf{U}_{w,\hat{t},i}+\mathbf{U}_{w,\hat{t},j})}\right)$$
i.e. nearest neighbors under a Mahalanobis-type norm using the summed world-space uncertainty covariances of the two candidate nodes, at a representative time $\hat t$.

Non-key node assignment to its nearest key node, summed over all frames:
$$j=\arg\min_{l \in \mathcal{V}_{k}}\sum_{t=0}^{T-1}\left\|\mathbf{p}_{i,t}-\mathbf{p}_{l,t}\right\|_{(\mathbf{U}_{w,t,i}+\mathbf{U}_{w,t,l})}$$

Key-node loss (fit to original trajectory, uncertainty-weighted, plus an appendix-defined motion regularizer):
$$\mathcal{L}^{\text{key}}=\sum_{t=0}^{T-1}\sum_{i \in \mathcal{V}_{k}}\left\|\mathbf{p}_{i,t}-\mathbf{p}_{i,t}^{\mathrm{o}}\right\|_{\mathbf{U}_{w,t,i}^{-1}}+\mathcal{L}^{\text{motion,key}}$$
(inverse-uncertainty weighting: confident nodes are held closer to their original position $\mathbf{p}_{i,t}^o$ from the base model.)

Non-key node loss, combining a fit term and a dual-quaternion-blended propagation term:
$$\mathcal{L}^{\text{non-key}}=\sum_{t=0}^{T-1}\sum_{i \in \mathcal{V}_{n}}\left\|\mathbf{p}_{i,t}-\mathbf{p}_{i,t}^{\mathrm{o}}\right\|_{\mathbf{U}_{w,i}^{-1}}+\sum_{t=0}^{T-1}\sum_{i \in \mathcal{V}_{n}}\left\|\mathbf{p}_{i,t}-\mathbf{p}_{i,t}^{\text{DQB}}\right\|_{\mathbf{U}_{w,i}^{-1}}+\mathcal{L}^{\text{motion,non-key}}$$
where $\mathbf{p}_{i,t}^{\text{DQB}}$ is the dual-quaternion blend (below) of the rigid transforms of $i$'s connected key nodes.

Dual quaternion blend for a non-key node's propagated pose:
$$(\mathbf{p}_{i,t}^{\text{DQB}}, \mathbf{q}_{i,t}^{\text{DQB}}) = \text{DQB}(\{(w_{ij}, \mathbf{T}_{j,t})\}_{j \in \mathcal{E}_i})$$
where $w_{ij}$ are normalized graph edge weights and $\mathbf{T}_{j,t} \in SE(3)$ are the key nodes' rigid motions at frame $t$ (blending itself follows Kavan et al. 2007's dual quaternion skinning, not re-derived in the paper).

Total loss combining rendering and the two graph-based terms:
$$\mathcal{L}^{\text{total}}=\mathcal{L}^{\text{rgb}}+\mathcal{L}^{\text{key}}+\mathcal{L}^{\text{non-key}}$$

#### Algorithm / Pipeline Changes
1. **Pretrain/obtain a base dynamic Gaussian model** (SoM or MoSca) on the monocular video, yielding per-Gaussian positions/trajectories $\mathbf{p}_{i,t}^o$ — this is unchanged from the base method.
2. **Per-Gaussian uncertainty estimation**: after (or during) base-model convergence, compute $\sigma_{i,t}^2$ from accumulated transmittance-weighted opacity per pixel, gate it through the convergence indicator to get $u_{i,t}$, then lift to the anisotropic world-space covariance $\mathbf{U}_{i,t}$.
3. **Key-node selection**: voxelize 3D space; within each voxel/region select Gaussians whose uncertainty stays below threshold for ≥5 consecutive frames as key nodes $\mathcal{V}_k$, targeting an overall key:non-key ratio near 1:49 (~2%).
4. **Graph construction**: connect key nodes via uncertainty-weighted kNN ($\mathcal{E}_i$); assign every non-key node $i \in \mathcal{V}_n$ to its nearest key node(s) by the summed-covariance Mahalanobis distance over all frames.
5. **Community detection**: apply spectral clustering over the graph to group nodes into motion-coherent instances (used to structure/regularize the propagation, per the architecture notes).
6. **Motion re-optimization**: jointly optimize $\mathcal{L}^{\text{total}}$ — key nodes fit their own original trajectory (inverse-uncertainty weighted) plus a motion regularizer; non-key nodes are pulled toward both their own original trajectory and the dual-quaternion-blended pose propagated from their connected key nodes. This replaces/augments the base model's original uniform per-Gaussian motion optimization.
7. **Render** with the refined per-Gaussian motion for novel-view/extreme-view synthesis — downstream rasterization is unchanged from the base 3DGS pipeline.

#### Key Hyperparameters & Design Choices
- Key/non-key node ratio: ~2% key nodes (1:49 ratio); paper sweeps 0.5%–4%.
- Key-node persistence threshold: ≥5 consecutive frames of low uncertainty.
- Color-error convergence threshold $\eta_c$: not specified (paper states "$\eta_c > 0$" with no numeric value given in the accessible text).
- Unconverged-pixel uncertainty penalty $\phi$: not specified (described only as "a large constant").
- kNN neighbor count $k$ (in UA-kkNN): not specified in the accessible paper text.
- Anisotropic scale factors $r_x, r_y, r_z$: not specified numerically.
- Learning rates, optimizer, iteration counts, and loss weights ($\lambda$ terms) for $\mathcal{L}^{\text{total}}$: not specified in the accessible main text (paper refers to Appendix A/B for full loss/training details, not available in the fetched source).
- MLP/network architecture for motion propagation or community detection: not applicable — the paper states community detection uses spectral clustering and motion propagation uses dual quaternion blending, not a learned network.

#### Ablation Summary
(Table 3, DyCheck benchmark, PSNR / SSIM / LPIPS; full model = 19.63 / 0.716 / 0.25)
1. **Full model**: 19.63 / 0.716 / 0.25.
2. **(a) w/o key-node uncertainty**: 18.86 / 0.688 / 0.28 — **−0.77 dB PSNR, the largest single drop**, flagged as the most impactful component (i.e., using photometric-convergence uncertainty to pick reliable anchors matters more than any other piece).
3. **(c) w/o loss weighting** (uncertainty-weighted Mahalanobis terms replaced with unweighted): 19.08 / 0.681 / 0.25 — −0.55 dB PSNR.
4. **(b) w/o UA-kkNN** (graph built with plain Euclidean kNN instead of uncertainty-weighted): 19.50 / 0.711 / 0.26 — −0.13 dB PSNR.
5. **(d) w/o 3D gridization** (key-node selection without voxel-grid spatial partitioning): 19.50 / 0.712 / 0.25 — −0.13 dB PSNR.

Headline results: DyCheck (Table 1, 7 scenes, 2× resolution) — MoSca baseline 19.32/0.706/0.26 vs. USplat4D 19.63/0.716/0.25 (+0.31 dB PSNR overall). Objaverse extreme-view synthesis (Table 2, 120°–180° angular range) — USplat4D over MoSca: +0.42 dB PSNR, +0.011 SSIM, −0.04 LPIPS, indicating the method's benefit concentrates at viewpoints far from the training cameras rather than at nominal views.

#### Implementation Reality
- **Framework:** PyTorch 2.1.2 + CUDA 12.1, Python 3.10; built on a custom fork of `gsplat` with uncertainty estimation added into the rasterizer, plus `xformers`, `pytorch3d`, PyG (PyTorch Geometric, likely for the graph/kNN operations), and `nvdiffrast`. Verified on H100 (sm_90).
- **Key files:** `USplat4d_vMoSca/` (main reconstruction pipeline and preprocessing); `MoSca_mask/` (a MoSca fork adding foreground/background masking, used for initialization); `external/gsplat` (custom rasterizer fork computing per-Gaussian uncertainty); `lib_ugraph/` (the uncertainty-graph construction and propagation — the original USplat4D contribution); `lib_usplat4d_prep/` (preprocessing utilities).
- **Notable implementation details:** the released pipeline is two-stage in practice — a full MoSca initialization/reconstruction run, then a separate uncertainty-guided refinement pass — rather than a single end-to-end optimization, which is not obviously stated as a hard two-stage split in the paper's method description. Run-directory naming in the repo (e.g. `dr0.001_thr0.5_vmax_contrib`) implies a dropout-style rate `dr=0.001` and a threshold `thr=0.5` used somewhere in the pipeline, but neither is tied explicitly back to $\eta_c$, $\phi$, or $k$ from the paper's notation, so the correspondence is not confirmed.

#### Failure Modes & Limitations
Per Appendix E ("Challenging Cases"): textureless regions produce uniformly high uncertainty, making it hard to identify trustworthy key nodes; fast motion exposes the temporal discretization of the graph (frame-to-frame key-node persistence and kNN assignment lag behind rapid displacement); strongly deforming objects violate the implicit assumption that the spatio-temporal graph's topology stays relatively stable over the sequence. No numeric per-scene metric drops for these cases were available in the fetched text.

---

## Relevance to This Project

It motivates uncertainty as a first-class Gate A output while showing that uncertainty-weighted optimization alone does not solve Gate B.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Sources

- https://arxiv.org/abs/2510.12768
- https://tamu-visual-ai.github.io/usplat4d/
- https://github.com/TAMU-Visual-AI/usplat4d
