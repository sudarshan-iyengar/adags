---
type: paper
node_id: paper:shih2025_prior_enhanced_gs
title: "Prior-Enhanced Gaussian Splatting for Dynamic Scene Reconstruction from Casual Video"
authors: ["Meng-Li Shih", "Ying-Huan Chen", "Yu-Lun Liu", "Brian Curless"]
year: 2025
venue: "SIGGRAPH Asia"
external_ids:
  arxiv: "2512.11356"
tags: [dynamic-gs, priors, tracks, masks, depth, scaffold]
status: deep-dived
---

# Prior-Enhanced Gaussian Splatting for Dynamic Scene Reconstruction from Casual Video

**Paper:** https://arxiv.org/abs/2512.11356
**Code:** Not found (project page lists code as "Coming Soon")
**Base method:** MoSca (Motion Scaffold, Lei et al. 2024) for lifting-to-3D and scaffold-node motion representation; Dynamic Gaussian Splatting (DGS) as the underlying scene representation; MegaSAM for video depth/pose consistency optimization, which this paper's object-depth loss modifies.

## One-line thesis

Off-the-shelf priors (video segmentation, depth, 2D tracks) fed into Dynamic Gaussian Splatting are unreliable specifically on thin/fast-moving structures and under self-occlusion, so the paper redesigns the prior-extraction stage itself (mask-guided depth refinement, skeleton-sampled re-identified tracks) and adds two reconstruction-time losses (virtual-view depth, scaffold-projection) that anchor motion-scaffold nodes to those improved tracks rather than proposing any new scene representation.

## Problem / Gap

Prior monocular dynamic-GS pipelines (Shape of Motion, MoSca) consume video depth and 2D tracks from generic foundation models largely unchanged, but those priors are unreliable exactly where dynamic reconstruction needs them most: video depth loses consistency on thin structures (limbs, utensils), and generic point trackers lose tracks through self-occlusion and produce few samples on thin/fast-moving parts because sampling is not mask- or skeleton-aware. The resulting scaffold nodes drift off objects and floaters appear in novel views that were never directly supervised.

## Method

A three-stage pipeline. Stage 1 (Initialization) builds object-level dynamic masks by combining video segmentation with epipolar-error maps that pick up thin structures segmentation misses, then uses those masks to refine MegaSAM's video depth via an object-level scale/shift loss, and finally produces 2D tracks by drawing 1/6 of samples from the medial-axis skeleton of each dynamic mask (rather than uniform sampling) with mask-guided re-identification to recover tracks lost to occlusion and a divergence-based filter to discard self-occluded tracks. Stage 2 (Lifting to 3D) is identical to MoSca: back-projects dynamic pixels to initialize 3D Gaussians and promotes 2D tracks into motion-scaffold nodes. Stage 3 (Reconstruction) trains with the standard DGS/MoSca losses (RGB, depth, per-Gaussian track re-projection, ARAP/velocity/acceleration regularization on scaffold nodes) plus two new terms: a virtual-view depth loss that renders from a synthetically translated camera to penalize floaters never visible from the real training view, and a scaffold-projection loss that directly penalizes 2D distance between a projected scaffold node and its corresponding track, preventing nodes from drifting off thin or fast-moving geometry.

## Assumptions

Monocular, casually captured RGB video (no calibrated multi-view rig, no known camera poses) is the primary target, though the method is also evaluated on a multi-view rig (NVIDIA dataset) using a single training view. It assumes a MoSca-style motion-scaffold representation is already in place and that dynamic content can be isolated via video segmentation plus epipolar-error masking.

## Limitations / Failure Modes

The paper states the method reproduces motion blur and focus blur already present in the input video (no deblurring), and leaves regions never observed in any input frame empty (no hallucination/inpainting) — the authors suggest video generative models as future work for both. No scene-specific quantitative failure case (e.g., a named scene with a large PSNR drop) is reported in the accessible text.

## Reusable Ingredients

- **Skeleton-guided track sampling** — draws a fixed fraction of 2D tracks from the medial-axis skeleton of a dynamic mask (dilated 5px) instead of uniform sampling, concentrating correspondence supervision on thin structures.
- **Mask-guided track re-identification** — recovers tracks that go missing across self-occlusion by re-associating within the object mask, rather than discarding the trajectory.
- **Self-occlusion filtering by short-window divergence** — resamples a track over a small temporal window ([t-2, t+2]) and discards it if re-sampled position diverges beyond a pixel threshold, a cheap occlusion-confidence signal without an explicit visibility network.
- **Object-level depth scale/shift refinement loss** — refines globally-consistent video depth per dynamic object with its own learned affine (scale, shift) rather than one global affine, letting thin/fast objects deviate from the background's depth calibration.
- **Virtual-view depth loss** — renders depth from a synthetically translated camera (translation proportional to scene depth) purely to suppress floaters that are invisible from the real training trajectory but would be visible under novel-view rendering.
- **Scaffold-to-track 2D projection loss** — directly ties a motion representation's control nodes to observed 2D tracks in image space, rather than relying solely on 3D consistency regularizers (ARAP, velocity) to keep nodes on-surface.

---

### Deep Dive

#### Core Novelty

The paper's contribution is entirely in the *prior stage*, not the representation or optimizer: it treats "which pixels get segmented as dynamic," "how depth is scale-aligned," and "which pixels get tracked and how tracks survive occlusion" as first-class design problems specific to thin, fast, self-occluding structures, then closes the loop with two losses that make the reconstruction stage actually use the improved priors as hard 2D/depth constraints (virtual-view depth, scaffold-projection) instead of only as soft initialization.

#### Mathematical Formulation

**Object-level depth refinement loss** (applied during MegaSAM-style video depth/pose consistency optimization, before lifting to 3D):

$$L_{\text{depth}}^{\text{object}} = \frac{1}{T|\Omega|}\sum_{o=1}^{O}\sum_{t=0}^{T}\sum_{p\in\Omega}M_t^{(o)}(p)\,\big|D_t(p)-(\alpha_t^{(o)}\tilde{D}_t(p)+\beta_t^{(o)})\big|$$

Computes an L1 depth-consistency term per dynamic object $o$, restricted to pixels $p$ where the object's mask $M_t^{(o)}(p)$ is active. $D_t(p)$ is the current optimized depth, $\tilde{D}_t(p)$ is a reference/prior depth, and $(\alpha_t^{(o)}, \beta_t^{(o)})$ is a per-object, per-frame learned affine (scale, shift) that aligns the reference depth to the current estimate independently for each dynamic object, rather than sharing one global affine across the whole frame. Evaluated during Stage 1 depth refinement, prior to Gaussian initialization.

**Standard per-Gaussian track loss** (Stage 3, reconstruction):

$$L_{\text{track}}^{\text{gaussian}} = \big\|u_t^{(n)} + \hat{F}_{t\to t'}[u_t^{(n)}] - u_{t'}^{(n)}\big\|_2$$

Penalizes the discrepancy between a track's observed 2D position at $t'$, $u_{t'}^{(n)}$, and its position at $t$ warped forward by the rendered/estimated flow field $\hat{F}_{t\to t'}$.

**Virtual-view depth loss** (novel, Stage 3, evaluated as a rendering-based loss term after rendering from a synthetic camera):

$$L_{\text{depth}}^{\text{virtual}} = \big\|\hat{D}_t^{\text{virtual}} - D_t^{\text{virtual}}\big\|_1$$

The virtual camera is generated by translating the training camera in the image plane by up to $0.18\cdot\text{median}(D_t)$ (i.e., a baseline scaled to scene depth), and $D_t^{\text{virtual}}$ is a reference depth for that virtual view (constructed from the refined video depth via reprojection). This penalizes Gaussians that render plausible depth from the real training view but would produce floaters/incorrect depth from a nearby unobserved viewpoint.

**Scaffold-projection loss** (novel, Stage 3, evaluated after projecting scaffold nodes into the camera, as a 2D loss term):

$$L_{\text{track}}^{\text{scaffold}} = \|v_{2D} - u_t^{(n)}\|_2$$

$v_{2D}$ is the 2D projection of a motion-scaffold node $v^{(m)}$ into the training camera; $u_t^{(n)}$ is its corresponding observed 2D track point. This directly constrains the scaffold node's position/motion in image space, independent of the 3D ARAP/velocity/acceleration regularizers already used on scaffold nodes.

#### Algorithm / Pipeline Changes

1. **Dynamic object mask selection (Stage 1):** Combine per-frame video-segmentation masks $S_t^{(l)}$ with epipolar-error maps $E_t$ (pixels with high epipolar error under estimated camera motion indicate non-rigid/independently-moving content, catching thin structures segmentation misses). Two-pass filtering keeps a segment only if it covers ≥5% of the total moving-surface area (τ_salient-style threshold) and ≥20% of the segment's own area is classified as in-motion.
2. **Depth refinement (Stage 1):** Run MegaSAM's consistency optimization with the added $L_{\text{depth}}^{\text{object}}$ term (per-object affine alignment) so thin dynamic structures get their own scale/shift instead of inheriting the global-scene affine.
3. **Mask-guided point tracking (Stage 1):** Sample 2D tracks from two pools — 3,000 skeleton-sampled tracks (from the medial axis of the dynamic mask, dilated 5px) and 16,384 uniformly-sampled tracks (19,384 total). Apply mask-guided re-identification to reconnect tracks broken by occlusion using the dynamic mask as an association cue. Apply self-occlusion filtering: resample each track over a $[t-2, t+2]$ window and discard it if its resampled position diverges from the original by more than $\tau_{\text{self-occ}} = 10$ px.
4. **Lifting to 3D (Stage 2):** Unchanged from MoSca — back-project dynamic pixels using refined depth to initialize 3D Gaussians; promote the (now higher-quality) 2D tracks into motion-scaffold nodes $v^{(m)}$.
5. **Reconstruction (Stage 3):** Train with standard RGB ($L_1$ + 0.1·SSIM), depth ($L_1$), per-Gaussian track, and scaffold ARAP/velocity/acceleration regularization losses, plus the two new terms $L_{\text{depth}}^{\text{virtual}}$ and $L_{\text{track}}^{\text{scaffold}}$ added directly to the total training objective (relative loss weights not stated).
6. Video/trajectory estimation (Stage 1 track/depth work) runs at a downsampled resolution — longer side = 512 px — independent of the final rendering resolution.

#### Key Hyperparameters & Design Choices

- $\tau_{\text{salient}} = 0.05$ — minimum fraction of total moving-surface area a mask segment must cover to be kept.
- $\tau_{\text{appearance}} = 0.2$ — appearance-change threshold (role in the two-pass filter not further specified in extracted text).
- $\tau_{\text{self-occ}} = 10$ px — max allowed divergence over the $[t-2,t+2]$ resample window before a track is discarded as self-occluded.
- Virtual camera translation magnitude: $0.18 \cdot \text{median}(D_t)$, applied in the image plane.
- Skeleton-mask dilation: 5 px.
- Track sampling: 3,000 skeleton tracks + 16,384 uniform tracks = 19,384 total tracks per video.
- Trajectory-estimation resolution: longer side = 512 px.
- Per-video pipeline stage timings: dynamic mask selection ~17 min, pose estimation & depth refinement ~11 min, point tracking ~12 min, lifting to 3D ~5 min, reconstruction ~25 min.
- RGB loss weighting: $L_1$ + 0.1·SSIM (standard, not novel).
- Loss weights for $L_{\text{depth}}^{\text{object}}$, $L_{\text{depth}}^{\text{virtual}}$, and $L_{\text{track}}^{\text{scaffold}}$ relative to the total objective: Not specified in paper.
- Network/MLP architecture details: Not applicable — the paper's novel components are losses and data-preparation heuristics, not new learned modules.

#### Ablation Summary

From Table 3 on iPhone DyCheck (pose-free RGB protocol):

| Configuration | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Full method | 17.63 | 0.648 | 0.268 |
| w/o $L_{\text{track}}^{\text{scaffold}}$ | 17.61 | 0.637 | 0.270 |
| w/o Mask-Guided Track (re-ID/skeleton sampling) | 17.55 | 0.632 | 0.274 |
| w/o $L_{\text{depth}}^{\text{virtual}}$ | 17.64 | 0.630 | 0.277 |

Most impactful component: mask-guided tracking (largest PSNR/SSIM drop when removed, −0.08 dB PSNR, −0.016 SSIM), reported by the authors as having the largest *visual* impact on thin regions specifically. The virtual-view depth loss shows the largest LPIPS degradation when removed (0.268 → 0.277), framed by the authors as critical for floater removal even though PSNR/SSIM barely move (0.268→ actually PSNR ticks up slightly without it, 17.64 vs 17.63, showing this loss trades a marginal average-pixel metric for perceptual/floater quality). Scaffold-projection loss shows the smallest numeric ablation delta but is reported as visually important for limb continuity — numeric metrics likely undercount its effect since averaged pixel error is insensitive to a few drifting limb regions.

#### Failure Modes & Limitations

The paper explicitly states the method reproduces motion blur and defocus blur already present in the input casual video (it does not deblur), and leaves scene regions that are never observed in any input frame empty rather than hallucinating content. The authors propose integrating video generative models as future work to address both deblurring and missing-content hallucination. No accessible per-scene worst-case quantitative failure (e.g., a specific scene with a large metric drop, comparable to ADAGS's own "sear steak" tracking) was found in the retrieved text.

---

## Relevance to ADAGS

Closest literature pressure on ADAGS's current motion-prior branch.

## Connections

## Sources

- https://arxiv.org/abs/2512.11356
- https://arxiv.org/html/2512.11356
- https://priorenhancedgaussian.github.io/
