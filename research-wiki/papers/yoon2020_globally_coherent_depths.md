---
type: paper
node_id: paper:yoon2020_globally_coherent_depths
title: "Novel View Synthesis of Dynamic Scenes with Globally Coherent Depths from a Monocular Camera"
authors: ["Jae Shin Yoon", "Kihwan Kim", "Orazio Gallo", "Hyun Soo Park", "Jan Kautz"]
year: 2020
venue: "CVPR"
external_ids:
  arxiv: "2004.01294"
  doi: null
  s2: null
tags: ["dynamic-depth", "scale-alignment", "multi-view-stereo", "scene-flow"]
added: 2026-07-14T22:18:30Z
status: deep-dived
---

# Novel View Synthesis of Dynamic Scenes with Globally Coherent Depths from a Monocular Camera

**Paper:** https://arxiv.org/abs/2004.01294
**Code:** Not found
**Base method:** Monocular relative-depth prediction (MiDaS / Lasinger et al. 2019) + classical multi-view stereo (COLMAP PixelwiseMVS, Schönberger et al. 2016), fused by a purpose-built depth fusion network; view synthesis via explicit 3D depth-based warping plus an adversarially-trained in-painting/blending network (Context-Encoders-style, Pathak et al. 2016). Predates Gaussian splatting/NeRF-style volumetric rendering entirely — this is an image-space warp-and-blend pipeline.

## One-line thesis

A per-scene-frame nonlinear scale-correction network can upgrade complete-but-view-variant monocular depth to match incomplete-but-view-invariant multi-view-stereo depth on static regions while preserving monocular relative-depth structure and minimal 3D scene flow on dynamic regions, yielding one coherent, complete depth field usable for warping-based novel view synthesis of non-rigid content.

## Problem / Gap

Multi-view stereo (MVS) gives geometrically consistent depth across views but only where correspondence exists — it is systematically incomplete on moving foreground content and degrades as camera baseline increases. Single-view (monocular) depth predictors (e.g. MonoDepth/MiDaS) are dense and complete but each frame's depth is estimated independently and only up to an unknown, per-frame scale/shift, so comparing depth values across time or view at the "same" pixel is meaningless — exactly the failure this project's own R031 heuristic (independent per-frame percentile-normalized DA3 depth, compared at the same raw pixel without warping) reproduces. Sparse2Dense-style depth completion (Mal & Karaman 2018) also fails here because it does not have an independent, view-invariant relative-depth signal to fall back on for the foreground, so its dynamic-region depth is reported as "completely incorrect."

## Method

Given a reference frame and neighboring views, the pipeline first computes per-frame monocular depth (DSV) and MVS depth (DMV), then feeds both plus the RGB image into a Depth Fusion Network (DFNet) that outputs one fused, complete depth map per frame. DFNet is trained self-supervised (no ground truth) with four losses: match DMV on static pixels, preserve DSV's scale-invariant relative gradients on dynamic pixels, minimize 3D scene flow (reprojected motion consistency) between optical-flow-corresponded pixels across two nearby frames, and a Laplacian smoothness regularizer. The fused depths are then used for bidirectional 3D warping of both a foreground (dynamic, mask-selected) region and a background (static, multi-view-aggregated) region into the target virtual view/time, and a second network (DeepBlender) fills missing/disoccluded regions and removes warping artifacts via a learned, adversarially-trained blending residual.

## Assumptions

Requires calibrated camera poses (via SfM) for all source views, a working per-view MVS reconstruction, per-frame monocular relative depth, dense optical flow between temporally/view-adjacent frames, and a foreground/dynamic-content segmentation mask (manually specified in this paper's evaluation, though the authors note automatic segmentation could substitute).

## Limitations / Failure Modes

The paper reports its own explicit limitations: DFNet degrades when the angular baseline between neighboring views exceeds roughly 45°, since this reduces the overlap of dynamic content needed for the scene-flow and relative-depth constraints; heavily cluttered scenes with many foreground/background objects (people, thin poles, trees) produce noisy warps from unresolved depth discontinuities; the whole pipeline fails outright when SfM camera calibration fails (e.g. scenes dominated by dynamic content); and a completely failed foreground mask produces visible artifacts such as object fragmentation or "afterimages" (Fig. 7). On the two most background-dominated test scenes (Umbrella, Teddybear) plain MVS actually beats the fused method on whole-scene RMSE, because background pixels dominate the metric there.

## Reusable Ingredients

- A learned, per-scene *nonlinear* scale-correction function trained self-supervised against an independent view-invariant depth source, instead of assuming a single global affine (scale+shift) correction.
- Scale-invariant multi-scale relative-depth-gradient loss ($g(D;x,\Delta x)$ with $\Delta x \in \{1,2,4,8,16\}$) as a way to transfer monocular structure without transferring monocular scale.
- Treating "3D scene flow should be minimal/locally consistent" as a *training signal* for depth (reprojecting optical-flow-corresponded pixels into 3D and penalizing their distance) rather than only as a diagnostic.
- Foreground/background-separated warping with a learned blending residual to avoid pixel-mixing at object boundaries, rather than a single unified warp+composite step.
- Synthetic pretraining recipe for a self-supervised fusion network: pretrain on synthetic data with ground-truth depth/flow/mask, but inject partial foreground depth removal and 5%-variance depth noise every iteration to simulate real-data imperfection before self-supervised fine-tuning.

---

### Deep Dive

#### Core Novelty
Relative to treating MVS and monocular depth as competing/alternative depth sources, this paper's actual change is a learned nonlinear scale-correction function $\psi$ that is trained per-scene, self-supervised, to reconcile the two: it forces $\psi$ applied to monocular depth to match MVS depth exactly where MVS is valid (static regions), while only preserving the monocular depth's *relative* gradient structure (not absolute values) where MVS is missing (dynamic regions), and additionally regularizes the implied 3D motion of dynamic points to be small and locally consistent. The insight is that a single global affine fit (scale+bias) between monocular and MVS depth is provably wrong once the scene is non-rigid (Eq. 5-6 in the paper), so the correction must be a spatially-varying, nonlinear function learned from a network rather than solved in closed form.

#### Mathematical Formulation

Depth upgrade (per-frame, applied to raw monocular depth before use):
$$\hat{\mathbf{D}}^{r_t} = \psi(\mathbf{D}_s^{r_t})$$
$\psi$ is the (nonlinear, network-realized) scale correction function; $\mathbf{D}_s^{r_t}$ is monocular (DSV) depth at view/time index $r_t$. For a static scene this would reduce to affine ($\mathbf{D}_m = \alpha \mathbf{D}_s + \beta$), but the paper asserts this affine form breaks down under scene motion, motivating a learned nonlinear $\psi$.

Static-region depth-matching constraint (evaluated on background pixels $x \notin \mathcal{M}^{r_t}$):
$$\mathbf{D}_m^r(x) \approx \psi(\mathbf{D}_s^{r_t}(x))$$

Dynamic-region relative-structure-preservation constraint (evaluated on dynamic pixels $x \in \mathcal{M}^{r_t}$), using the scale-invariant relative gradient
$$g(D; x, \Delta x) = \frac{D(x+\Delta x) - D(x)}{|D(x+\Delta x)| + |D(x)|}$$
computed at multi-scale offsets $\Delta x \in \{1,2,4,8,16\}$ so both local and global structure are constrained:
$$g(\mathbf{D}_s^{r_t}(x)) \approx g(\psi(\mathbf{D}_s^{r_t}(x)))$$

3D scene-flow minimality constraint (evaluated between optical-flow-corresponded pixels in a reference frame $r_t$ and a neighboring frame $n_t$, over all pixels):
$$p(x; \mathbf{D}^{r_t}, \Pi^{r_t}) \approx p(\mathbf{F}_{r_t \to n_t}(x); \mathbf{D}^{n_t}, \Pi^{n_t})$$
where $\mathbf{F}_{r_t\to n_t}$ is dense optical flow (PWC-Net) from frame $r_t$ to $n_t$, and $p(x; D, \Pi) = \psi(D(x))\,\mathbf{R}^\top \mathbf{K}^{-1}\tilde{x} + \mathbf{C}$ back-projects pixel $x$ to a 3D world point using depth $D$ and the camera's rotation $\mathbf{R}$, intrinsics $\mathbf{K}$, and optical center $\mathbf{C}$ from projection matrix $\Pi$.

DFNet training loss (self-supervised, per fused-depth output, before rasterization/warping):
$$L(\mathbf{w}) = L_g + \lambda_l L_l + \lambda_s L_s + \lambda_e L_e$$
- $L_g = \lVert \hat{\mathbf{D}}^{r_t}(x) - \mathbf{D}_m^{r_t}(x)\rVert$ for $x \notin \mathcal{M}^{r_t}$ (static/MVS-matching term — reported as the single most critical term).
- $L_l = \lVert g(\hat{\mathbf{D}}^{r_t}(x)) - g(\mathbf{D}_s^{r_t}(x))\rVert$ for $x \in \mathcal{M}^{r_t}$ (dynamic relative-structure term).
- $L_s = \lVert p(x;\mathbf{D}^{r_t},\Pi^{r_t}) - p(\mathbf{F}_{r_t\to n_t}(x);\mathbf{D}^{n_t},\Pi^{n_t})\rVert$ (3D scene-flow term, all pixels), with $n_t = r_t \pm 2$.
- $L_e = \lVert \nabla^2 \hat{\mathbf{D}}^{r_t}(x)\rVert^2 + \lambda_f \lVert \nabla^2 \hat{\mathbf{D}}^{r_t}(\bar{x})\rVert^2$ for $x \notin \mathcal{M}^{r_t}$, $\bar{x}\in\mathcal{M}^{r_t}$ — Laplacian smoothness, weighted separately for static vs. dynamic regions by $\lambda_f$.

View synthesis compositing (per virtual view/time, after warping, before/at the blending network):
$$\phi(\mathbf{J}_*^v, \mathbf{J}^{v,t}; \mathcal{M}^v) = \mathbf{J}_*^v(x) + \mathbf{J}^{v,t}(y) + \tilde\phi_\theta(\mathbf{J}_*^v, \mathbf{J}^{v,t})$$
for $x \notin \mathcal{M}^{v,t}$ (background pixel) and $y \in \mathcal{M}^{v,t}$ (dynamic-content pixel); $\mathbf{J}_*^v$ is the globally-aggregated warped static background (shortest-baseline source view wins per pixel), $\mathbf{J}^{v,t}$ is the warped dynamic content from time $t$, and $\tilde\phi_\theta$ (the DeepBlender network) predicts the residual that fills unseen/disoccluded regions and removes warp artifacts.

DeepBlender training loss (self-supervised with synthetic holes/noise, evaluated on the blended output):
$$L(\mathbf{w}_\theta) = L_{\text{rec}} + \lambda_{\text{adv}} L_{\text{adv}}$$
where $L_{\text{rec}}$ is reconstruction error against the (synthetically corrupted) ground truth and $L_{\text{adv}}$ is a standard adversarial loss (Pathak et al. 2016 formulation).

#### Algorithm / Pipeline Changes
1. Calibrate all source-view cameras via SfM (COLMAP-style structure-from-motion).
2. Per frame, compute monocular depth $\mathbf{D}_s^{r_t}$ (Lasinger et al. 2019 monocular predictor) and MVS depth $\mathbf{D}_m^{r_t}$ (Schönberger et al. 2016 pixelwise MVS); both are converted to normalized inverse depth to avoid scale confusion, with the fused depth's final scale recovered from the original MVS scale afterward.
3. Extract a foreground/dynamic mask per frame (GrabCut-style interactive segmentation in this paper; noted as substitutable by an automatic salient-object detector).
4. Compute dense optical flow between the reference frame and each of $n_t = r_t \pm 2$ neighboring frames (PWC-Net, with forward-backward consistency filtering for outliers).
5. Feed $(\mathbf{D}_s^{r_t}, \mathbf{D}_m^{r_t}, \mathbf{I}^{r_t})$ into DFNet (shared encoder for DSV/DMV features, skip connections into a depth-generator decoder) to produce fused depth $\hat{\mathbf{D}}^{r_t}$, trained/fine-tuned with the four-term loss above.
6. For each target virtual view/time, 3D-warp the static background from all source views bidirectionally (checking forward-backward consistency to avoid holes) and pick the shortest-camera-baseline source per pixel to build $\mathbf{J}_*^v$; separately 3D-warp the masked dynamic foreground from the single chosen source time to get $\mathbf{J}^{v,t}$. Depth is refined per-warp with a bilateral weighted median filter.
7. Feed $(\mathbf{J}_*^v, \mathbf{J}^{v,t})$ into DeepBlender (feature extraction from both streams, decoder with skip connections) to predict the blending residual $\tilde\phi_\theta$; final image is background + foreground + residual, composited separately by region to avoid boundary pixel-mixing.

#### Key Hyperparameters & Design Choices
- Multi-scale relative-gradient offsets: $\Delta x = \{1,2,4,8,16\}$.
- Scene-flow neighbor window: $n_t = r_t \pm 2$ (±2 camera views).
- DFNet pretraining: on a synthetic dataset (Lv et al. 2018, "Learning Rigidity in Dynamic Scenes") providing ground-truth depth/flow/mask; foreground depth is partially removed and Gaussian noise at 5% of variance is injected every training iteration to simulate real-data imperfection.
- DeepBlender pretraining: on a video object segmentation dataset (DAVIS/Perazzi et al. 2016), with synthetic holes/seams generated via mask morphology + superpixels, and up to 30-pixel-thick image-border removal.
- Loss weights $\lambda_l, \lambda_s, \lambda_e, \lambda_f, \lambda_{\text{adv}}$: exact numeric values not specified in the paper (only that they "control the importance of each loss").
- Optical flow: PWC-Net (Sun et al. 2018), with forward-backward consistency for outlier rejection.
- Camera rig for quantitative evaluation: 12-camera GoPro Black Edition rig, two height levels of 6 cameras each, 0.22 m inter-camera baseline.
- Capture: hand-held qualitative sequences at 1920×1080/60Hz (Samsung Galaxy Note 10), evaluated at half resolution.

#### Ablation Summary
(Table 1, RMSE in meters, "whole scene / dynamic-content-only," averaged over 8 scenes — lower is better)
1. Full DFNet: 0.20 / 0.70 (best on both metrics).
2. DFNet$-L_s$ (no scene-flow term): 0.26 / 0.91.
3. DFNet$-L_e$ (no Laplacian regularizer): 0.26 / 0.87.
4. DFNet$-L_l$ (no relative-gradient term): 0.28 / 1.53 — large jump on the dynamic-only metric, showing $L_l$ is what keeps dynamic-content depth from collapsing.
5. DFNet$-L_g$ (no MVS-matching term): 1.18 / 1.10 — **by far the largest degradation**; the paper identifies $L_g$ (matching MVS on static regions) as the single most critical self-supervision signal, since it anchors the fused depth's scale globally and everything else is refined relative to it.
Ablation on full pipeline for view synthesis (Table 2, perceptual similarity / optical-flow-error pixels, averaged): removing DeepBlender (DFNet+B3W-DeepBlender) raises perceptual distance from 0.15 to 0.21, confirming the blending/in-painting stage materially improves visual plausibility; replacing bidirectional 3D warping with grid-wise as-similar-as-possible warping (DFNet+ASAPW) is worse (0.18 vs. 0.15) than the paper's own bidirectional 3D warping (B3W), showing depth-based pixel warping outperforms affine grid warping when the depth is accurate.

#### Failure Modes & Limitations
Explicitly stated by the authors: performance drops when the angular baseline between neighboring source views exceeds roughly 45°, reducing dynamic-content overlap needed for scene-flow/relative-depth supervision; heavy scene clutter (many people, thin poles, trees) causes noisy warps from unresolved depth discontinuities; SfM calibration failure (e.g., scenes dominated by dynamic content, per Lv et al. 2018's rigidity-learning dataset) causes outright pipeline failure; and a completely failed foreground segmentation mask produces visible artifacts (object fragmentation, "afterimages"). Quantitatively, plain MVS beats the fused method on whole-scene RMSE for the two most background-dominated scenes (Umbrella, Teddybear), since background pixels dominate that metric there even though DFNet still wins on the dynamic-content-only metric.

---

## Relevance to This Project

It directly explains why R031's independent percentile normalization removed the temporal meaning needed for visibility inference.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Sources

https://arxiv.org/abs/2004.01294
