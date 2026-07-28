---
type: paper
node_id: paper:li2025_4dgs_slam
title: "WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments"
authors: ["Jianhao Zheng", "Zihan Zhu", "Valentin Bieri", "Marc Pollefeys", "Songyou Peng", "Iro Armeni"]
year: 2025
venue: "CVPR"
tags: [dynamic-gs, slam, static-dynamic]
status: deep-dived
---

# WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments

**Paper:** https://arxiv.org/abs/2504.03886
**Code:** https://github.com/GradientSpaces/WildGS-SLAM
**Base method:** DROID-SLAM (optical-flow-based tracking with a differentiable dense bundle adjustment layer) + MonoGS/Splat-SLAM-style 3D Gaussian Splatting mapping backbones, with uncertainty prediction adapted from NeRF-On-the-go / WildGaussians.

Note: the full ablation table (per-component PSNR/SSIM/LPIPS/ATE deltas) and the verbatim "Failure Cases" subsection are in a later supplementary section of the HTML source that could not be retrieved in full through the fetch tool (content was truncated). The core method, equations, and main-text limitations below were extracted directly from the paper; anything not confirmed is marked "Not specified in paper."

## One-line thesis

An online, self-supervised per-pixel uncertainty (predicted by a shallow MLP from DINOv2 features, trained purely from rendering residuals with no ground-truth dynamic masks) is used to down-weight likely-dynamic pixels inside both the dense bundle adjustment tracking objective and the Gaussian-splatting rendering losses, letting a monocular SLAM system track and map static structure while ignoring unknown-class moving distractors.

## Problem / Gap

Prior monocular Gaussian-Splatting SLAM systems (e.g., MonoGS, Splat-SLAM) assume a static scene; when moving objects (people, pets, other traffic) enter the field of view, their pixels get baked into the Gaussian map and corrupt both pose tracking (via DBA) and the photometric/depth losses used for mapping. Existing dynamic-SLAM approaches typically rely on semantic segmentation (fixed class lists) or explicit motion masks, which fail to generalize to arbitrary, unknown dynamic object categories in the wild. WildGS-SLAM instead needs a purely geometric/photometric signal that flags "unreliable/dynamic" pixels without any semantic prior.

## Method

The system runs an online pipeline over streamed monocular RGB frames: dense features from a 3D-aware fine-tuned DINOv2 backbone are fed to a shallow per-pixel uncertainty MLP that is trained incrementally alongside the map. This predicted uncertainty (β) is inserted as a per-pixel weighting term into two places: (1) the DROID-SLAM-style dense bundle adjustment (DBA) optimization over poses and depths, where residuals for pixels with high uncertainty are down-weighted via a Mahalanobis-style `/β²` term; and (2) the 3D Gaussian Splatting rendering losses (color + depth) used to optimize the Gaussian map, again divided by `β²`. Metric3D v2 monocular depth supplies a metric-depth regularization/consistency signal, and a multi-view depth-consistency check (reprojection agreement across a threshold ε plus DINOv2 feature-cosine-similarity threshold γ) helps flag dynamic regions independent of the learned uncertainty. The uncertainty MLP and the Gaussian map (P and G) are optimized with detached gradients relative to each other — i.e., independently — with keyframes inserted at a fixed interval (every 8 frames) and an initial window of frames (12 keyframes) processed before uncertainty weighting becomes active.

## Assumptions

Monocular RGB input with known camera calibration (for the projection function `Π_c`); the scene contains a dominant static structure plus optionally unknown-class moving distractors (no assumption of a specific object class or a segmentation prior); requires sufficient inter-frame covisibility/motion for frame-graph construction (i.e., standard SLAM baseline/motion assumptions), and treats only the static portion of the scene as the reconstruction target — dynamic content is removed, not reconstructed.

## Limitations / Failure Modes

The paper explicitly notes that the map (G) and the uncertainty MLP (P) are optimized independently via gradient detachment, adding coordination complexity rather than a single joint objective. Higher-resolution uncertainty maps are possible but traded off against computational efficiency. The online-trained MLP is noted to be less accurate "especially during early stages of tracking," i.e., before enough frames have been observed to calibrate the per-pixel uncertainty. A dedicated "Failure Cases" subsection is referenced in the paper's supplementary section but its full text could not be retrieved through the tool used here; treat specific quantitative failure-mode numbers (e.g., degradation with large dynamic-object screen proportion, fast motion, textureless regions) as unconfirmed pending direct access to the PDF/supplementary material.

## Reusable Ingredients

- **Self-supervised per-pixel uncertainty from rendering residuals** — trains a lightweight MLP to predict "how trustworthy is this pixel" using only photometric/depth reconstruction error as supervision, with no ground-truth dynamic masks or semantic segmentation required.
- **Uncertainty as a `/β²` down-weighting term in both tracking and mapping losses** — a single scalar field is reused to protect two different optimization objectives (pose/depth bundle adjustment and Gaussian rendering losses) from dynamic-content corruption.
- **DINOv2 (3D-aware fine-tuned variant) as the feature backbone for uncertainty prediction** — leverages a pretrained foundation model's dense features instead of training a feature extractor from scratch.
- **Multi-view depth-consistency + feature-cosine-similarity masking (Eq. 8)** — a purely geometric/appearance cross-check (reprojection agreement across views plus feature similarity) that flags likely-dynamic regions independent of the learned uncertainty, usable as an auxiliary/complementary signal.
- **Gradient-detached alternating optimization of map and auxiliary predictor** — decouples an auxiliary per-pixel predictor's training from the primary geometry optimization to avoid one destabilizing the other.

---

### Deep Dive

#### Core Novelty
Relative to prior static-scene Gaussian-Splatting SLAM (MonoGS, Splat-SLAM) and prior semantic-segmentation-based dynamic SLAM, WildGS-SLAM's change is to replace any explicit object-class/motion-mask mechanism with a single learned, self-supervised scalar uncertainty field that is trained online purely from rendering/geometric residuals and then reused to re-weight both the DBA tracking objective and the Gaussian-map rendering losses. The key insight is that a system does not need to know *what* is moving (no semantic prior, no fixed class list) if it can instead learn, per pixel and per frame, *how much to trust* the reconstruction residual there — generalizing to unknown dynamic object categories.

#### Mathematical Formulation

Gaussian rendering follows the standard 3DGS alpha-compositing formulation:
$$\alpha_i = o_i \exp\left(-\tfrac{1}{2}(x' - \mu_i')^\top \Sigma_i'^{-1} (x' - \mu_i')\right)$$
$$\hat{I} = \sum_{i \in \mathcal{G}'} c_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j), \qquad \hat{D} = \sum_{i \in \mathcal{G}'} \hat{d}_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)$$
where $o_i$ is opacity, $\mu_i', \Sigma_i'$ are the projected 2D Gaussian mean/covariance, and $c_i, \hat d_i$ are per-Gaussian color and depth. Evaluated per-pixel at rasterization time (standard 3DGS, not novel).

Depth loss (per-pixel, standard L1 against rendered vs. reference/metric depth $\tilde D_i$):
$$\mathcal{L}_{depth} = |\hat D_i - \tilde D_i|_1$$

**Uncertainty loss** (novel — trains the uncertainty MLP; evaluated per keyframe after rendering):
$$\mathcal{L}_{uncer} = \frac{\mathcal{L}_{SSIM}' + \lambda_1 \mathcal{L}_{uncer\_D}}{\beta_i^2} + \lambda_2 \mathcal{L}_{reg\_V} + \lambda_3 \mathcal{L}_{reg\_U}$$
where $\beta_i$ is the per-pixel predicted uncertainty, $\mathcal{L}_{SSIM}'$ is a rendering-consistency (SSIM-based) term, $\mathcal{L}_{uncer\_D}$ is the L1 loss between rendered depth and metric depth (Eq. 3's depth loss reused as the signal that drives uncertainty), $\mathcal{L}_{reg\_V}$ regularizes uncertainty to be low-variance across pixels with similar (DINOv2) features, and $\mathcal{L}_{reg\_U} = \log(\beta_i)$ is a log-barrier term preventing uncertainty from growing unboundedly (which would otherwise let the model trivially zero out all supervision).

**Dense bundle adjustment objective** (novel weighting inserted into DROID-SLAM-style tracking; evaluated during pose/depth tracking optimization):
$$\arg\min_{\omega,d} \sum_{(i,j)\in E} \left\| \tilde p_{ij} - \Pi_c\big(\omega_j^{-1}\omega_i \Pi_c^{-1}(p_i, d_i)\big) \right\|^2_{\Sigma_{ij}/\beta_i^2} + \lambda_4 \sum_{i \in V} \left\| M_i \left(d_i - \tfrac{1}{\tilde D_i}\right)\right\|^2$$
where $\omega_i$ are camera poses, $d_i$ inverse depths, $\Pi_c$ the calibrated camera projection, $\tilde p_{ij}$ optical-flow correspondences, $\Sigma_{ij}$ the standard DBA confidence covariance, and $\beta_i$ the same learned per-pixel uncertainty dividing the Mahalanobis weighting — this is the mechanism that suppresses dynamic-pixel influence on pose estimation. $M_i$ is a mask/weight applied to the metric-depth disparity regularization term.

**Rendering (mapping) loss** (novel weighting inserted into 3DGS optimization; evaluated per training iteration on keyframes):
$$\mathcal{L}_{render} = \frac{\lambda_5 \mathcal{L}_{color} + \lambda_6 \mathcal{L}_{depth}}{\beta^2} + \lambda_7 \mathcal{L}_{iso}$$
$$\mathcal{L}_{color} = (1-\lambda_{ssim})\|\hat I - I\|_1 + \lambda_{ssim}\mathcal{L}_{ssim}$$
Uncertainty again divides the reconstruction terms so high-uncertainty (likely dynamic) pixels contribute less gradient to Gaussian attribute updates. $\mathcal{L}_{iso}$ is an isotropy regularizer on Gaussian shape (not uncertainty-weighted).

**Dynamic-region flagging (Eq. 8)** — a multi-view depth-consistency check: a pixel is flagged as inconsistent/dynamic if its reprojected depth disagrees with other views beyond a threshold $\varepsilon$, cross-checked against a DINOv2 feature-cosine-similarity threshold $\gamma$ to confirm correspondence validity. Used as an auxiliary geometric consistency signal alongside the learned uncertainty.

#### Algorithm / Pipeline Changes
1. Stream RGB frames; extract dense features per frame using a fine-tuned, 3D-aware DINOv2 backbone.
2. Feed DINOv2 features to a shallow per-pixel uncertainty MLP to produce $\beta$, a per-pixel scalar uncertainty map, at each keyframe.
3. Insert keyframes at a fixed interval (every 8 frames); for the first 12 keyframes, run tracking/mapping without uncertainty weighting active (warm-up), then activate uncertainty weighting.
4. Run DROID-SLAM-style dense bundle adjustment (Eq. 5) over the frame graph, dividing the flow-correspondence residual by $\beta_i^2$ so uncertain (likely dynamic) pixels contribute less to pose/inverse-depth updates; a metric-depth disparity regularization term (from Metric3D v2) is added with weight $\lambda_4$.
5. Optimize the 3D Gaussian map using the rendering loss (Eq. 6), again dividing color+depth reconstruction terms by $\beta^2$.
6. Train the uncertainty MLP using Eq. 4, with gradients detached from the Gaussian map parameters (P and G optimized independently) so neither optimization directly back-propagates into the other's parameters within the same step.
7. Apply the multi-view depth-consistency mask (Eq. 8) as an auxiliary geometric cross-check for dynamic-region detection, independent of the learned uncertainty channel.
8. (Referenced but not confirmed in retrieved text) A final global bundle adjustment / ablation pass is described in a later supplementary section (Sec. 8).

#### Key Hyperparameters & Design Choices
- Loss weights $\lambda_1$–$\lambda_7$, $\lambda_{ssim}$: Not specified in paper (referenced only symbolically in the retrieved text; numeric values were in supplementary material not retrieved).
- Uncertainty MLP architecture (layer count, hidden dimension, activation): Not specified in paper — described only as "a shallow multi-layer perceptron."
- DINOv2 variant: a fine-tuned, 3D-aware variant from Yue et al. 2025 ("injects 3D awareness").
- Depth prior: Metric3D v2 (monocular metric depth).
- Keyframe interval: every 8 frames.
- Warm-up: first 12 keyframes processed before uncertainty weighting is activated.
- Multi-view consistency thresholds $\varepsilon$ (depth) and $\gamma$ (feature cosine similarity): Not specified in paper (present in Eq. 8 but numeric values not retrieved).
- Learning rates, optimizer, batch/window sizes: Not specified in paper (not present in retrieved text).

#### Implementation Reality
- **Framework:** PyTorch, CUDA 11.8, building on third-party CUDA extensions.
- **Key files/dirs (from repo structure):** `thirdparty/lietorch/` (Lie-group pose operations, shared with DROID-SLAM lineage), `thirdparty/diff-gaussian-rasterization-w-pose/` (custom Gaussian rasterizer supporting pose gradients), `thirdparty/simple-knn/` (KNN acceleration for Gaussian densification/init). Experiment configs live under `configs/Dynamic/<dataset>/` (e.g., `Wild_SLAM_Mocap`, `Bonn`, `TUM_RGBD`), with a `configs/Custom/custom_template.yaml` for new sequences.
- **Notable implementation details:** the public README does not document the uncertainty MLP's architecture or the specific loss-weight values — these must be read from the config YAMLs or source directly rather than the README; this could not be further verified without cloning and reading the actual config/source files.

#### Failure Modes & Limitations
The map (Gaussians, G) and the uncertainty predictor (P) are optimized independently via gradient detachment, which the paper flags as a source of coordination complexity relative to a single joint objective. Higher-resolution uncertainty maps are possible but cost more compute, indicating a resolution/efficiency trade-off was made. The online-trained uncertainty MLP is explicitly noted to be less reliable "especially during early stages of tracking" (i.e., before it has seen enough residual signal to calibrate). A dedicated "Failure Cases" subsection exists in the paper's supplementary material (Sec. 8) but its content could not be retrieved through the tool used for this extraction — treat any claim about specific failure scenarios (e.g., proportion of frame occupied by dynamic content, fast-motion or textureless-region degradation) as unconfirmed until read directly from the PDF.

## Relevance to ADAGS

Adds prior art pressure on dynamic-environment separation, though the task differs from N3V rendering.

## Connections

## Sources

- https://arxiv.org/abs/2504.03886
