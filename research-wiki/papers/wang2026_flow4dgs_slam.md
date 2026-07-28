---
type: paper
node_id: paper:wang2026_flow4dgs_slam
title: "Flow4DGS-SLAM: Optical Flow-Guided 4D Gaussian Splatting SLAM"
authors: ["Yunsong Wang", "Gim Hee Lee"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2604.22339"
tags: [dynamic-gs, slam, optical-flow, dynamic-scenes]
status: deep-dived
---

# Flow4DGS-SLAM: Optical Flow-Guided 4D Gaussian Splatting SLAM

**Paper:** https://arxiv.org/abs/2604.22339
**Code:** https://github.com/wangys16/Flow4DGS-SLAM (listed on project page, marked "Coming soon" — no source available at time of review)
**Base method:** 4DGS-SLAM (sparse control points + MLP deformation fields, category-based dynamic segmentation) as the primary baseline being replaced; also compares against MonoGS (monocular static 3DGS-SLAM) and SplaTAM (dense RGB-D 3DGS tracking that treats dynamic content as outliers).

## One-line thesis

Camera-ego-motion-decomposed optical flow can generate a category-agnostic dynamic mask and directly drive explicit per-keyframe 3D Gaussian center propagation, replacing an MLP deformation field with flow-warped positions plus a lightweight Gaussian-Mixture-Model (GMM) opacity/rotation schedule — cutting per-step mapping cost from ~110s (4DGS-SLAM) to 6.3s while improving both tracking (ATE) and rendering (PSNR).

## Problem / Gap

4DGS-SLAM models dynamic content with sparse control points driven by MLP-based deformation fields and relies on category-specific (semantic-class) segmentation to isolate dynamic objects; this is computationally expensive (~110s per mapping step) and fails on general dynamic scenes with objects that don't belong to a known category, or with complex interactions such as people leaving and re-entering the view. SplaTAM and MonoGS assume static scenes and treat any dynamic pixels as outliers/noise to be masked out rather than reconstructed, so moving content is simply discarded rather than modeled.

## Method

Optical flow (RAFT) between consecutive frames is decomposed into a camera-induced rigid-motion component, solved via an image-Jacobian least-squares fit over pixels marked static by a semantic mask (YOLOv9); residual flow that a rigid camera motion cannot explain is thresholded (median + k·MAD) to produce a category-agnostic dynamic mask, which is unioned with the semantic mask. Dynamic Gaussians keep explicit 3D center positions at each keyframe, linearly interpolated between keyframes and updated at new keyframes by projecting each Gaussian into the previous keyframe, applying the estimated optical flow, and unprojecting using depth — with a KNN-weighted local-rigidity smoothing term over neighboring Gaussians. Temporal opacity and rotation are modeled continuously (not just at keyframes) via a K=3-component Gaussian Mixture Model over time, so a Gaussian's visibility/orientation blends smoothly between the keyframes it's active at. New Gaussians are inserted adaptively by backwarping motion masks with backward optical flow to find newly revealed dynamic regions and sparsely sampling new points there. Tracking and mapping alternate within a sliding keyframe window (size 8), with combined color, depth, flow, mask, and isotropic-shape loss terms.

## Assumptions

RGB-D input with known camera intrinsics, run online/real-time in a keyframe-based SLAM loop; requires off-the-shelf optical flow (RAFT) and semantic segmentation (YOLOv9) as auxiliary priors, and assumes most pixels the semantic mask marks static are in fact static (the category-agnostic residual only refines around that prior).

## Limitations / Failure Modes

The paper does not include an explicit limitations section; failure modes must be inferred. It states 4DGS-SLAM struggles with complex dynamics such as people leaving and re-entering the view, which the category-agnostic flow-residual mask is meant to address, but Flow4DGS-SLAM's own robustness to this case is not separately quantified. The method's dynamic-Gaussian propagation is only as good as RAFT flow, so fast motion, occlusion, or textureless regions (where flow is unreliable) are implicit risk factors. KNN-based local-rigidity smoothing over dynamic Gaussians will tend to oversmooth genuinely non-rigid deformation (e.g., cloth, fluids) since it penalizes divergence from neighbors' motion.

## Reusable Ingredients

- Camera-ego-motion decomposition of optical flow (image-Jacobian least-squares with Cauchy weighting) to isolate a residual flow field attributable only to independently moving objects, decoupled from a semantic-class prior.
- Median + k·MAD residual thresholding for a category-agnostic dynamic mask — a robust-statistics alternative to fixed thresholds or learned classifiers.
- Explicit per-keyframe Gaussian center propagation via project → apply 2D flow → unproject, instead of an MLP deformation field, as a cheaper mechanism for tracking known dynamic geometry across time.
- GMM-parameterized continuous-time opacity/rotation blending between sparse keyframe observations, giving smooth temporal attributes without per-frame optimization.
- Backward-flow backwarping of the dynamic mask to localize newly-revealed regions for targeted adaptive Gaussian insertion (a flow-driven disocclusion/reveal detector).

---

### Deep Dive

#### Core Novelty

Relative to 4DGS-SLAM, the paper replaces (a) an MLP deformation field with explicit, flow-propagated 3D Gaussian centers, and (b) category-specific semantic segmentation with a category-agnostic dynamic mask derived from the residual between observed optical flow and flow predicted by rigid camera ego-motion alone. The insight is that once camera motion is factored out of the flow field, whatever flow remains must originate from independently moving scene content — regardless of its semantic category — and that residual flow can directly warp Gaussian positions rather than being fit implicitly by a network, which is both cheaper and generalizes past a fixed object taxonomy.

#### Mathematical Formulation

Camera-induced (rigid, static-scene) flow field, evaluated per pixel during tracking to fit camera ego-motion:
$$F(u,v) = J(x)\,\xi$$
where $J(x)$ is the image Jacobian mapping a 6-DoF camera twist $\xi = (\rho, \theta)$ (translation $\rho$, rotation $\theta$) to predicted 2D pixel flow at pixel $(u,v)$, $x$ its back-projected 3D point. Solved via iteratively reweighted least squares with Cauchy weights over pixels the semantic mask marks static.

Category-agnostic dynamic mask, evaluated once ego-motion is estimated, to flag pixels flow cannot explain as rigid camera motion:
$$\mathcal{M}_{ca}(u,v) = \mathbb{1}\big(r(u,v) > \mathrm{median}(r) + k\cdot \mathrm{MAD}(r)\big), \quad r(u,v) = \lVert F(u,v) - \hat F(u,v)\rVert_2$$
with $\mathrm{MAD}(r) = \mathrm{median}_i\lvert r_i - \mathrm{median}_j(r_j)\rvert$. The final dynamic mask is $\mathcal{M}_{dy} = \mathcal{M}_s \cup \mathcal{M}_{ca}$ (union with the semantic mask $\mathcal{M}_s$).

Continuous-time opacity for a dynamic Gaussian $i$, evaluated per-frame before rasterization:
$$m_i(t) = 1 - \exp\!\Big(-A_i \sum_k w_{i,k}\, \mathcal{N}(\hat t;\ \mu_{i,k}, \tau_{i,k}^2)\Big), \qquad \sigma_i(t) = \sigma_i \cdot m_i(t)$$
i.e. a $K$-component ($K{=}3$) Gaussian Mixture Model over normalized time $\hat t$ modulates the base opacity $\sigma_i$; rotation is likewise blended by quaternion interpolation weighted by the same GMM activations.

Scene-flow-propagated Gaussian center update at new keyframe $k$, evaluated at each keyframe transition for every existing dynamic Gaussian:
$$\Delta X_i^k = R_k^\top\big(D_i^k K^{-1}\bar u_i^k - t_k\big) - x_i^{k-1}$$
where $D_i^k$ is the depth at the Gaussian's flow-shifted 2D projection $\bar u_i^k$, $K$ the camera intrinsics, $(R_k, t_k)$ the keyframe pose, and $x_i^{k-1}$ the Gaussian's previous 3D position — i.e. project to the prior keyframe, offset by 2D optical flow, unproject with depth at the new location, and difference against the old position.

KNN local-rigidity smoothing weight between dynamic Gaussians $i,j$, applied when regularizing the propagated centers within a local neighborhood:
$$w_{ij}^{knn} = \frac{\mathcal{N}\big(\lVert x_j^{k-1} - x_i^{k-1}\rVert_2;\ 0,\ \tau_{knn}^2\big)}{\sum_l \mathcal{N}\big(\lVert x_l^{k-1} - x_i^{k-1}\rVert_2;\ 0,\ \tau_{knn}^2\big)}$$
a Gaussian-kernel weighted average over the K nearest neighbors' motion, penalizing deviation from locally consistent (near-rigid) motion.

Mapping and tracking loss composites (standard color/depth/flow/mask/isotropic terms combined, not individually novel):
$$\mathcal{L}_{map} = \lambda_1 \mathcal{L}_c + \lambda_2 \mathcal{L}_d + \lambda_f \mathcal{L}_f + \lambda_m \mathcal{L}_m + \lambda_{iso}\mathcal{L}_{iso}, \qquad \mathcal{L}_{track} = \tfrac{1}{|V|}\sum_v M_v(u)\big[\lambda_1 \mathcal{L}_1(\hat C) + \lambda_2 \mathcal{L}_1(\hat D)\big]$$

#### Algorithm / Pipeline Changes

1. Estimate camera ego-motion twist $\xi$ by robust least-squares fit of the rigid-flow model $J(x)\xi$ to observed RAFT flow, restricted to pixels the semantic mask ($\mathcal{M}_s$, from YOLOv9) marks static.
2. Compute per-pixel residual $r(u,v)$ between observed and rigid-predicted flow; threshold via median+k·MAD to get $\mathcal{M}_{ca}$; union with $\mathcal{M}_s$ to get the final dynamic mask $\mathcal{M}_{dy}$, replacing 4DGS-SLAM's purely category-based segmentation.
3. Within a sliding window of 8 keyframes, alternate tracking (camera pose optimization against static Gaussians, using $\mathcal{L}_{track}$) and mapping (Gaussian attribute optimization, using $\mathcal{L}_{map}$), 50 mapping iterations per keyframe.
4. For each dynamic Gaussian at a new keyframe: project to previous keyframe pose, shift by the estimated 2D optical flow, unproject using the new keyframe's depth map, and set $\Delta X_i^k$ as the position update (Section 3 formula); apply KNN-weighted rigidity smoothing across neighboring dynamic Gaussians. This replaces the MLP deformation-field query used by 4DGS-SLAM/SC-GS.
5. Opacity and rotation for a dynamic Gaussian at arbitrary query time $t$ are read out from a per-Gaussian $K{=}3$-component GMM over time rather than stored per-frame, evaluated before rasterization.
6. Backwarp the current dynamic mask using backward optical flow to identify pixels that are newly dynamic/revealed but not yet covered by existing Gaussians; randomly sample new Gaussian seeds there at an insertion density of $1/D_{init}$.
7. After SLAM completes, a color-only refinement stage runs 1500 additional iterations to sharpen appearance.

#### Key Hyperparameters & Design Choices

- Mapping iterations per keyframe: 50.
- Tracking iterations: 100–200, dataset-dependent.
- Sliding keyframe window size: 8.
- Optical-flow loss ($\mathcal{L}_f$) applied only in the last 25 iterations of an optimization round.
- GMM components per dynamic Gaussian: $K=3$.
- Post-hoc color refinement: 1500 iterations.
- Loss weights $\lambda_1, \lambda_2, \lambda_f, \lambda_m, \lambda_{iso}$: not specified in the main text ("please refer to our supplementary material").
- Dynamic-mask threshold multiplier $k$ (median + k·MAD): not specified in paper.
- Adaptive-insertion density factor $D_{init}$: value not specified beyond the $1/D_{init}$ sampling-rate form.

#### Ablation Summary

Ablation on TUM fr3/walk_xyz (ATE in cm, lower is better; PSNR in dB, higher is better), relative to the full method's ATE 2.5cm / PSNR 24.60dB:

- w/o Adaptive Insertion: ATE 3.4cm (+0.9cm), PSNR 23.53dB (−1.07dB) — **largest degradation, the single most impactful component**.
- w/o Flow Propagation: ATE 2.6cm (+0.1cm), PSNR 23.91dB (−0.69dB).
- w/o GMM (temporal opacity/rotation): ATE 2.7cm (+0.2cm), PSNR 24.04dB (−0.56dB).
- w/o Motion Decomposition (i.e. no category-agnostic mask): ATE 2.7cm (+0.2cm), PSNR 24.40dB (−0.20dB).
- w/o KNN smoothing: ATE 2.5cm (+0.0cm), PSNR 24.47dB (−0.13dB) — smallest effect.

#### Failure Modes & Limitations

The paper does not present a dedicated limitations section. It motivates the category-agnostic mask by noting 4DGS-SLAM (the baseline) struggles with complex dynamics such as people leaving and re-entering the view, but does not separately quantify how well Flow4DGS-SLAM itself handles this case. No discussion is given of behavior under fast motion, long-duration occlusion, or textureless regions, despite the method's direct dependence on RAFT optical-flow quality for both the dynamic mask and the Gaussian-center propagation.

---

## Relevance to ADAGS

Adds to the evidence that "add optical flow" is no longer enough. ADAGS's contribution must be reliability-gated flow or a targeted diagnostic showing when flow helps.

## Connections

- Addresses [[gap_map#G8 - Flow Supervision Needs Reliability Gating]]
- Addresses [[gap_map#G2 - Static/Dynamic Leakage Is A Representation And Evaluation Problem]]

## Sources

- https://arxiv.org/abs/2604.22339
