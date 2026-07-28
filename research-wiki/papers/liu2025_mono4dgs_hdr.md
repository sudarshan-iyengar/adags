---
type: paper
node_id: paper:liu2025_mono4dgs_hdr
title: "Mono4DGS-HDR: High Dynamic Range 4D Gaussian Splatting from Alternating-exposure Monocular Videos"
authors: ["Jinfeng Liu", "Lingtong Kong", "Mi Zhou", "Jinwen Chen", "Dan Xu"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2510.18489"
tags: [dynamic-gs, monocular, hdr, exposure, robustness]
status: deep-dived
---

# Mono4DGS-HDR: High Dynamic Range 4D Gaussian Splatting from Alternating-exposure Monocular Videos

**Paper:** https://arxiv.org/abs/2510.18489
**Code:** https://github.com/LiuJF1226/Mono4DGS-HDR (repo exists but as of this deep dive contains only a README/TODO — "Release data and code" is unchecked; no implementation files published yet)
**Base method:** Spline-parameterized dynamic 3D Gaussians following SplineGS (Park et al., 2025) and Li et al. (2024a) for per-Gaussian trajectory functions, plus the orthographic video-space Gaussian representation from SaV (Sun et al., 2024), extended with an HDR irradiance/tone-mapping formulation.

## One-line thesis

A two-stage pipeline first fits fully dynamic Gaussians in a pose-free orthographic "video space" to recover clean HDR appearance and motion from alternating-exposure frames, then lifts and refines those Gaussians into world space with real camera poses — deferring the hard pose/geometry problem until after HDR appearance and motion are already stable, and using a rendered-flow-based temporal luminance loss (not optical-flow priors) to keep per-frame HDR brightness consistent.

## Problem / Gap

Alternating-exposure monocular capture (e.g., short/long/short/... frame exposures) breaks the constant-brightness assumption that photometric reprojection losses in prior dynamic Gaussian methods (MoSca, SplineGS, GFlow) rely on, so directly applying them to HDR reconstruction produces poor geometry and flickering appearance. Off-the-shelf 2D priors (depth, tracks, optical flow) are themselves noisy, and because dynamic Gaussians float around true object surfaces during joint pose+appearance+motion optimization, naively adding HDR estimation on top compounds instability into temporally inconsistent radiance.

## Method

Stage 1 optimizes a "video Gaussian" representation directly in orthographic camera-coordinate space (normalized pixel coordinates $(x^v,y^v)\in[-1,1]^2$ plus depth $z^v$), which removes camera-pose estimation from the loop entirely and lets the system first recover a stable per-frame HDR video and motion trajectories under 2D-prior supervision (depth, sparse tracks, dense flow) and a temporal luminance regularizer. Stage 2 lifts these video Gaussians into world space via a closed-form/optimized transform (occlusion-aware dynamic/static classification plus a 2D-covariance-matching re-scaling step), then jointly refines world-space Gaussians and camera poses (bundle-adjustment-style) using both the standard photometric loss and a dense HDR photometric reprojection loss derived from the Stage-1 HDR video. Each Gaussian carries an HDR irradiance value $e$ that is converted to a displayable color via a learned logarithmic tone-mapper conditioned on exposure time. Gaussian position follows a cubic Hermite spline over control points and rotation follows a cubic polynomial in $t$; scale, opacity, and base color are time-invariant.

## Assumptions

Input is a single unposed monocular LDR video with frames captured at alternating, known exposure durations (2 or 3 exposure levels cycling), static camera intrinsics, and no multi-view constraint. Camera poses for the world-space stage are recovered via bundle adjustment, not assumed known.

## Limitations / Failure Modes

The paper documents a dedicated "Challenging Conditions" appendix (A.4) for scene/capture regimes where reconstruction degrades, though full detail wasn't recoverable from the fetched excerpt. Explicitly stated constraints: test-time camera poses can only be interpolated between training frames (no extrapolation), and per-Gaussian scale/opacity/color are modeled as strictly time-invariant, so any true photometric change at a fixed surface point over time must be explained through the HDR irradiance/tone-mapping path rather than appearance parameters. The method also inherits the reliability limits of its foundation-model priors (DepthCrafter, SpatialTracker, RAFT), which the paper notes may fail under extreme low light or highly dynamic content.

## Reusable Ingredients

- **Pose-free "video space" first stage**: fit an orthographic-projection Gaussian representation before ever estimating camera poses, decoupling appearance/motion recovery from pose/geometry recovery — reduces compounding of errors across simultaneously-optimized unknowns.
- **Rendered-flow (not prior-flow) temporal consistency loss**: warps the model's own rendered output between frames using its own rendered flow, then compares against a normalized-difference formulation that cancels absolute radiance scale — useful for any setting where raw photometric consistency loss is invalidated by a per-frame nuisance variable (here, exposure).
- **Occlusion-gated dynamic/static classification**: thresholding video-space depth against a smoothed/aggregate depth map to detect occlusion before deciding whether a Gaussian's positional change reflects true motion vs. depth-order occlusion — directly relevant to any ADAGS surface-visibility/occlusion ledger.
- **2D-covariance-invariance re-fitting for world-space scale initialization**: rather than directly copying an initial scale from one coordinate space to another, solve a small optimization that matches the *projected* 2D covariance footprint, preserving perceived Gaussian size across a representation change.

---

### Deep Dive

#### Core Novelty
Relative to prior monocular dynamic-GS baselines (MoSca, SplineGS, GFlow) and prior HDR-GS work (GaussHDR, HDR-HexPlane), the paper's actual change is architectural sequencing plus one new loss: (1) split the problem into a pose-free orthographic "video Gaussian" stage and a subsequent world-space refinement stage instead of jointly estimating pose, geometry, motion, and (now) HDR appearance all at once, and (2) add a temporal luminance regularization term computed from the model's own rendered flow (rather than an external optical-flow prior) so that per-frame absolute-scale HDR irradiance stays temporally consistent without being confounded by the alternating exposure times. The insight is that alternating exposure breaks the standard constant-brightness photometric loss, so stabilizing HDR appearance/motion first — before pose estimation, which itself depends on photometric consistency — avoids letting exposure-induced brightness noise corrupt pose/geometry convergence.

#### Mathematical Formulation

Dynamic/static classification with occlusion gating (evaluated once per Gaussian per frame, before the video→world transform):
$$N_d = \sum_t \mathbb{I}\big[M_t(x^v_t, y^v_t)\cdot(1-o_t) = 1\big], \qquad o_t = \mathbb{I}\big[z^v_t > \tilde{D}_t(x^v_t, y^v_t)\big]$$
where $M_t$ is a per-pixel dynamic mask, $z^v_t$ is the Gaussian's video-space depth at frame $t$, $\tilde D_t$ is an aggregated/reference depth map, and $o_t$ flags the Gaussian as occluded (so it should not count as evidence of true dynamic motion) when its depth lies behind the reference surface. A threshold of 0.1 on the resulting fraction decides the final dynamic/static label.

2D-covariance-invariance scale re-fitting (solved once via gradient descent, ~1000 iterations, at video→world lifting time, before Stage-2 joint refinement):
$$\min_{S^w} \sum_t \left\| \Sigma'^{\,v}_t - \Sigma'^{\,w}_t \right\|_2$$
where $S^w$ is the world-space Gaussian scale being solved for, and $\Sigma'^v_t$, $\Sigma'^w_t$ are the 2D projected covariances of the same Gaussian under the video-space and world-space camera models at frame $t$ — matching these keeps the Gaussian's apparent screen-space footprint consistent across the representation change.

Temporal Luminance Regularization (evaluated as a loss term after rendering, each training step in both stages):
$$\mathcal{L}_{tlr} = \left\| V_{t\to t-1} \odot \frac{\tilde H_{t-1\to t} - \tilde H_t}{\tilde H_{t-1\to t} + \tilde H_t} \right\|_1$$
where $\tilde H_t$ is the rendered HDR image at frame $t$, $\tilde H_{t-1\to t}$ is the HDR render from frame $t-1$ warped forward using the model's own rendered scene flow $\tilde F_{t\to t'}$ (not an external optical-flow prior), and $V_{t\to t-1}$ is a validity mask. Normalizing the difference by the sum cancels the absolute-irradiance scale, so the term penalizes relative temporal luminance inconsistency rather than being confounded by irradiance magnitude.

HDR color/tone-mapping (evaluated per-Gaussian, at rendering time, replacing the plain RGB color attribute of standard 3DGS):
$$c = \phi\big(\log(e \cdot \Delta t)\big)$$
where $e \in [0,+\infty)$ is the Gaussian's stored HDR irradiance, $\Delta t$ is the exposure time of the frame being rendered, and $\phi$ is a learned tone-mapping function mapping log-domain exposed irradiance to a displayable LDR color.

Gaussian trajectory parameterization (evaluated per-Gaussian, per-frame, before rasterization): position follows a cubic Hermite spline over $N_c$ control points (with $N_c$ set by sampling one control point every 4 frames), and rotation follows a cubic polynomial in normalized time, $r(t) = \sum_{j=0}^{3} a_j t^j$; scale, opacity, and base color are held time-invariant per Gaussian.

Overall training objective (sum of per-term losses, each stage):
$$\mathcal{L} = \lambda_{rgb}\mathcal{L}_{rgb} + \lambda_{ue}\mathcal{L}_{ue} + \lambda_{dep}\mathcal{L}_{dep} + \lambda_{track}\mathcal{L}_{track} + \lambda_{arap}\mathcal{L}_{arap} + \lambda_{vel}\mathcal{L}_{vel} + \lambda_{acc}\mathcal{L}_{acc} + \lambda_{tlr}\mathcal{L}_{tlr} + \lambda_{pr}\mathcal{L}_{pr}$$

#### Algorithm / Pipeline Changes
1. Precompute 2D priors on the raw alternating-exposure video using off-the-shelf models: per-frame depth (DepthCrafter), sparse long-range 2D tracks (SpatialTracker), and dense optical flow between same-exposure frame pairs (RAFT).
2. Stage 1 — initialize and optimize "video Gaussians" in orthographic camera-coordinate space using normalized pixel coordinates $(x^v,y^v)\in[-1,1]^2$ and depth $z^v$; optimize for 4K iterations against the 2D priors, an HDR-aware photometric/tone-mapping loss, and $\mathcal{L}_{tlr}$. No camera pose is estimated in this stage — orthographic projection replaces the usual perspective-projection Jacobian in rasterization.
3. Video-to-world transform — classify each Gaussian dynamic vs. static using the occlusion-gated rule above, then re-fit each Gaussian's world-space scale via the 2D-covariance-invariance optimization (~1000 gradient-descent iterations, ~1 minute total) so screen-space footprint is preserved across the coordinate-space change.
4. Stage 2 — jointly refine world-space Gaussian parameters and camera extrinsics/intrinsics (bundle-adjustment-style) for 11K iterations, now supervised additionally by a dense HDR photometric reprojection loss that uses the Stage-1-recovered HDR video as a supervisory signal for geometry and pose (replacing/augmenting the standard single-exposure photometric loss used in prior monocular dynamic-GS pipelines).
5. At render/test time, HDR irradiance per Gaussian is tone-mapped through $\phi(\log(e\cdot\Delta t))$ to produce the displayed LDR color at a chosen exposure; test-frame camera poses are obtained by interpolating between neighboring training-frame poses (no extrapolation supported).

#### Key Hyperparameters & Design Choices
- Stage 1 iterations: 4K; Stage 2 iterations: 11K; total: 15K iterations per scene.
- Control-point sampling for position spline: every 4 frames ($N_s=4$), reported as optimal in ablation.
- Rotation: cubic polynomial (degree 3) in quaternion space, $j=0..3$.
- Dynamic/static occlusion threshold: 0.1 (on the fraction defining $N_d$).
- 2D-covariance-invariance re-fitting: ~1000 gradient-descent iterations, ~1 minute wall-clock.
- Loss weights $\lambda_{rgb}, \lambda_{ue}, \lambda_{dep}, \lambda_{track}, \lambda_{arap}, \lambda_{vel}, \lambda_{acc}, \lambda_{tlr}, \lambda_{pr}$: Not specified in the accessible text (deferred to an implementation-details appendix not recovered in this pass).
- Tone-mapper $\phi$ architecture (layers, hidden dim, activation): Not specified in the accessible text.
- Total training time: ~1.5 hours per scene; rendering speed 161 FPS.

#### Ablation Summary
Ranked by PSNR impact (Syn-Exp-3 benchmark, HDR metric):
1. **Video Gaussian initialization removed: −1.13 dB** — largest single contributor; supplies rotation, scaling, opacity, and color priors that Stage 2 otherwise lacks. Most impactful component.
2. 2D covariance invariance removed: −0.44 dB — direct scale inheritance across coordinate spaces fails without this re-fit.
3. HDR photometric loss removed: −0.37 dB — dense HDR supervision benefits camera/geometry refinement.
4. Occlusion handling removed: −0.35 dB — causes static/dynamic misclassification in occluded regions.
5. Temporal luminance regularization removed: only −0.02 dB PSNR, but temporal-stability metric TAE worsens from 0.067 to 0.082 — the term's benefit is temporal consistency, not per-frame accuracy, and would be invisible to a PSNR-only ablation read.

#### Implementation Reality
- **Framework:** Not verifiable — the public GitHub repository (https://github.com/LiuJF1226/Mono4DGS-HDR) currently contains only a README, LICENSE, and .gitignore; "Release data and code" is listed as an outstanding TODO item, so no source files are available to inspect.
- No further implementation details can be reported without speculation.

#### Failure Modes & Limitations
Test-time camera poses are limited to interpolation between training frames — the method cannot extrapolate to novel poses outside the training trajectory's temporal span. Per-Gaussian scale, opacity, and base color are strictly time-invariant, so the model has no direct mechanism for true photometric surface change over time other than through the shared HDR irradiance/tone-mapping pathway. The pipeline's ceiling is bounded by its foundation-model priors (DepthCrafter, SpatialTracker, RAFT), which the paper notes are less reliable in extreme low light or under highly dynamic content. The paper reserves a dedicated appendix subsection ("Challenging Conditions," A.4) for further scene-level failure cases, but its full content was not recoverable from the fetched source.

## Relevance to ADAGS

Useful as a reminder that cooking scenes may mix motion blur, specularities, flame/lighting, and exposure artifacts. ADAGS diagnostics should avoid attributing all dynamic-region error to motion representation.

## Connections

- Addresses [[gap_map#G7 - A Benchmark/Diagnostic Claim Is Necessary]]

## Sources

- https://arxiv.org/abs/2510.18489
