---
type: paper
node_id: paper:nazarczuk2025_vidar
title: "ViDAR: Video Diffusion-Aware 4D Reconstruction From Monocular Inputs"
authors: ["Michal Nazarczuk", "Sibi Catley-Chandar", "Thomas Tanay", "Zhensong Zhang", "Gregory Slabaugh", "Eduardo Pérez-Pellitero"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2506.18792"
tags: [dynamic-scene, diffusion-prior, evaluation, dynamic-region-metrics, monocular]
status: deep-dived
---

# ViDAR: Video Diffusion-Aware 4D Reconstruction From Monocular Inputs

**Paper:** https://arxiv.org/abs/2506.18792
**Code:** https://github.com/vidar-4d/ViDAR
**Base method:** MoSca (Lei et al., motion-scaffold monocular 4D Gaussian reconstruction) for the initial static/dynamic Gaussian reconstruction — the released code is a direct fork of the MoSca repository (`lib_mosca`, `lib_moca` carried over almost verbatim) — plus Stable Diffusion XL (SDXL), personalized per-scene via DreamBooth/LoRA, as the appearance-enhancement prior.

## One-line thesis
A DreamBooth-personalized SDXL model, run as a multistep image-to-image denoiser on renders from sampled novel camera poses, supplies pseudo-multi-view supervision for a MoSca-style dynamic Gaussian reconstruction; restricting that supervision to Track-Anything dynamic regions and jointly optimizing the (noisy, interpolated) sampled camera poses in a separate loss pass is what prevents the spatio-temporally inconsistent diffusion outputs from degrading reconstruction quality.

## Problem / Gap
Regularization-based monocular 4D methods (MoSca, Shape of Motion) achieve geometrically compact reconstructions but fall short on photorealistic appearance because casual monocular video provides only single-view supervision in dynamic regions. Diffusion-prior approaches (ReconFusion, DpDy, the concurrent CAT4D) inject stronger appearance detail but their outputs are not spatio-temporally consistent frame-to-frame; naively using such outputs as training supervision causes the reconstruction to either regress toward a blurry mean radiance value or overfit to individual, temporally inconsistent frames.

## Method
MoSca reconstructs static Gaussians 𝒢_s and dynamic Gaussians 𝒢_d (𝒢 = 𝒢_d ∪ 𝒢_s) from the input video, with Track-Anything masks substituted for MoSca's own epipolar-error dynamic/static classification. A DreamBooth/LoRA pass personalizes SDXL on the input video's frames. New camera poses are sampled per timestep from noised interpolations of the input trajectory; the fitted MoSca model renders degraded images from these views, and the personalized SDXL model enhances each one via multistep image-to-image (noise the render's latent for k steps, then denoise) rather than full generation or Score Distillation Sampling. These enhanced pseudo-views retrain the reconstruction under two masked losses: a dynamic-region-only loss (L1 + perceptual + SSIM) that updates the Gaussians and the original input camera poses, and a full-frame loss that updates only the sampled camera poses, which are themselves noisy and need re-alignment to scene geometry.

## Assumptions
Casual single-camera monocular video of a dynamic scene (evaluated on the DyCheck iPhone benchmark), with MoSca's own prior stack (monocular depth, optical flow, 2D tracking) assumed reliable enough to bootstrap the initial static/dynamic Gaussian split and motion scaffold, and Track Anything assumed to give a usable per-frame dynamic-object segmentation.

## Limitations / Failure Modes
The paper's stated limitation: "ViDAR limits the scope of diffusion to enhancing rendered images, which are limited by the initial accuracy of the 4D reconstruction, thus, cannot repair major geometrical artefacts." It also reports that CAT4D achieves a better LPIPS score, attributed to CAT4D trading spatio-temporal consistency (visible in the supplementary video, and reflected in worse PSNR/SSIM) for per-frame perceptual detail.

## Reusable Ingredients
- Multistep img2img diffusion-as-enhancer (noise a render's latent for k steps, then denoise, instead of full SDS/generation) — injects detail into a degraded render while preserving coarse structure/geometry.
- Dynamic-region-only masked supervision from an existing reconstruction's rendered/predicted pair — stops low-consistency generative supervision from washing out already well-constrained static regions.
- Two-pass, two-loss training split per iteration (one backward pass updates only the sampled camera poses, a second updates the scene + input camera poses) — decouples "fix the pseudo-view's alignment" from "fix the scene" when the supervision source is itself spatially noisy.
- Extreme-baseline + noised-interpolation camera sampling around a fitted mean/sphere of the input trajectory — a scene-agnostic recipe for turning one input trajectory into a diverse pseudo-multi-view camera set.
- Dynamic-region evaluation protocol (a dynamic mask used in place of the standard co-visibility mask, plus a reported per-scene overlap statistic between the two) — reusable diagnostic for showing that a benchmark's default masked metric is static-biased.

---

### Deep Dive

#### Core Novelty
Relative to MoSca (its literal code base) and to diffusion-enhanced reconstruction more broadly (ReconFusion, CAT4D), ViDAR changes three things: (1) it personalizes the diffusion model per scene via DreamBooth/LoRA rather than using a generic prior; (2) it applies the diffusion enhancement only to degraded renders from *sampled novel views of an already-fitted reconstruction*, via multistep img2img rather than SDS or from-scratch generation; and (3) it masks the resulting pseudo-GT supervision to dynamic regions only, while jointly optimizing the noisy sampled camera poses in a separate loss pass. The key insight is that diffusion's spatio-temporal inconsistency is concentrated in fine detail/texture and is more damaging in already-multi-view-observed static regions than in genuinely under-observed dynamic regions — so routing the diffusion signal exclusively to dynamic regions, and treating the sampled-camera pose as a free variable that absorbs pseudo-GT misalignment, avoids the mean-collapse/overfitting failure mode of naive diffusion supervision.

#### Mathematical Formulation

Enhancement step (evaluated once per sampled camera per timestep, offline, before diffusion-aware reconstruction training):
$$x_0 = \mathcal{E}(R_{m,t}), \quad x_0 \rightarrow x_k \ (\text{add } k \text{ noise steps, Discrete Euler scheduler}), \quad x_k \rightarrow \hat{x}_0 \ (\text{denoise } k \text{ steps}), \quad E_{m,t} = \mathcal{D}(\hat{x}_0)$$
$R_{m,t}$ = MoSca render from sampled camera $m$ at time $t$; $\mathcal{E}/\mathcal{D}$ = SDXL VAE encoder/decoder; $E_{m,t}$ = resulting enhanced pseudo-GT image for that (camera, time) pair.

Dynamic reconstruction loss (evaluated per training iteration on dynamic-masked renders; backpropagated to update $\mathcal{G}$ and $\mathcal{C}_{inp}$):
$$\mathcal{L}_{dyn} = |E_{m,t}^{dyn} - \hat{I}_{m,t}^{dyn}|_1 + \lambda_p |E_{m,t}^{dyn} - \hat{I}_{m,t}^{dyn}|_{vgg} + \lambda_s |E_{m,t}^{dyn} - \hat{I}_{m,t}^{dyn}|_{ssim}$$
where $E_{m,t}^{dyn} = E_{m,t} \odot D_{m,t}$, $\hat{I}_{m,t}^{dyn} = \hat{I}_{m,t} \odot D_{m,t}$, $D_{m,t}$ = Track-Anything dynamic mask for that render, $\odot$ = elementwise product, $\hat{I}_{m,t}$ = current model's rendered prediction, $\lambda_p = \lambda_s = 0.1$.

Camera loss (evaluated per training iteration on the *unmasked* full image; backpropagated separately, updates only $\mathcal{C}_{sample}$):
$$\mathcal{L}_{cam} = |E_{m,t} - \hat{I}_{m,t}|_1 + \lambda_p |E_{m,t} - \hat{I}_{m,t}|_{vgg} + \lambda_s |E_{m,t} - \hat{I}_{m,t}|_{ssim}$$
Same loss form as $\mathcal{L}_{dyn}$, but unmasked, and gradient-routed only to the sampled camera poses — coarse static structure is judged a more reliable localization signal than the dynamic region alone.

**PSNR-D / SSIM-D / LPIPS-D definition (the paper's evaluation contribution).** Standard DyCheck practice (Gao et al. 2022; used by MoSca, Shape of Motion) computes PSNR, SSIM, LPIPS restricted to a *co-visibility mask* (pixels visible across enough views/timesteps to have real supervision), denoted with a "-m" suffix. ViDAR computes a second, independent per-frame *dynamic mask* $D_t$ from Track Anything for each test scene, and reports the same PSNR/SSIM/LPIPS formulas but with $D_t$ used **in place of** the co-visibility mask — denoted with a "-D" suffix. The underlying metric math is unmodified DyCheck/`ml-pgdvs` code: for PSNR/LPIPS the plain mean over the masked MSE / per-pixel LPIPS map is replaced by a mask-weighted mean, `masked_mean(x, mask) = (x*mask).sum() / mask.sum()`; SSIM uses the identical substitution applied inside a partial-convolution-style masked Gaussian filter (11×11, σ=1.5, k1=0.01, k2=0.03) rather than a plain box/Gaussian filter. The paper additionally reports (Table 3) the **intersection** between the co-visibility mask and the dynamic mask, as a percentage of the co-visibility mask's area, purely as a diagnostic to argue that co-visibility-masked ("-m") metrics are static-biased — this intersection statistic is *not* the mask actually used to compute the "-D" metrics (confirmed in the released code; see Implementation Reality below).

Table 3 — "Intersection of co-visibility mask with dynamic regions with respect to co-visibility mask area" (DyCheck iPhone scenes):

| Scene | Dyn/Co-vis Intersection (%) |
|---|---|
| apple | 4.42 |
| block | 27.46 |
| paper-windmill | 3.58 |
| space-out | 20.63 |
| spin | 19.76 |
| teddy | 81.33 |
| wheel | 24.65 |
| **mean** | **25.97** |

Exact paper quote: "We find that on average only 26% of the co-visibility masked pixels correspond to the dynamic region. Some scenes such as apple and paper-windmill have an intersection as low as 4%." Dataset: DyCheck (iPhone), 7 scenes with ground-truth test views (of 14 total captured scenes; the other 7 lack test-view ground truth and are qualitative-only).

#### Algorithm / Pipeline Changes
1. Run MoSca end-to-end on the input video, but replace its epipolar-error-based dynamic/static classification with Track-Anything masks $D_t$ when reconstructing the static Gaussians $\mathcal{G}_s$ and building the motion scaffold (this specifically targets floater artifacts from background leaking into the dynamic Gaussian set).
2. Fine-tune SDXL on the input video's frames with DreamBooth/LoRA (diffusers implementation) at the input resolution, so a fixed trigger token reproduces the scene's appearance.
3. Camera sampling: fit a mean camera pose and an approximating sphere to the input trajectory; pick the two input views with the largest longitudinal displacement on that sphere as "extreme" views. Per timestep, sample $M=18$ new cameras: 4 from noised means of random input-view pairs, 12 from noised weighted blends of a random input view and one of the two extreme views (blend weight toward the extreme view, and camera-noise amplitude, both increase over the sampling schedule), 2 as the extreme views themselves.
4. Render all $M$ sampled cameras at all timesteps with the fitted MoSca model to get degraded images $R_{m,t}$; the full set is generated once up front (not on-the-fly) to allow reuse and reduce GPU memory pressure.
5. Enhance each $R_{m,t}$ with the personalized SDXL via image-to-image (encode → add $k$ noise steps → denoise $k$ steps → decode) to obtain $E_{m,t}$.
6. Diffusion-aware reconstruction: extend MoSca's photometric training from 8,000 to 40,000 iterations. At each iteration, run two forward/backward passes: pass 1 renders 2 of the sampled cameras matching the current timestep, computes the mean $\mathcal{L}_{cam}$ over them, and updates only $\mathcal{C}_{sample}$; pass 2 re-renders with the updated sampled poses, computes $\mathcal{L}_{dyn}$ (dynamic-masked) in addition to MoSca's existing losses, and updates the Gaussians $\mathcal{G}$ and the input camera poses $\mathcal{C}_{inp}$.
7. Output: standard Gaussian-splatting rasterization of $\mathcal{G} = \mathcal{G}_d \cup \mathcal{G}_s$ for novel-view rendering, evaluated at both half- and full-resolution.

#### Key Hyperparameters & Design Choices
- DreamBooth/LoRA on SDXL: 5,000 training iterations (diffusers default is 500; raised because the scene has 400+ frames vs. the 5–40 images typical personalization is tuned for); resolution matched to input, 720×960. From the released training script/config (not stated in the paper text): base model `stabilityai/stable-diffusion-xl-base-1.0`, VAE `madebyollin/sdxl-vae-fp16-fix`, LoRA rank 4 (diffusers script default, not overridden), learning rate 1e-4, constant LR schedule, no warmup, batch size 1, gradient accumulation 4, instance prompt `"a photo of sks"`.
- Image-to-image enhancement (repo defaults, not given numerically in the paper): `strength=0.4`, `num_inference_steps=30` — i.e., the paper's $k$ corresponds to roughly $0.4 \times 30 \approx 12$ of the 30 scheduler steps; SDXL Img2Img pipeline, prompt `"a photo of sks"`, images resized to 960×720 before enhancement.
- Camera sampling: $M=18$ new cameras/timestep for most scenes; the released per-scene config for the `wheel` scene overrides this to 11, with its own separate training config file — not disclosed in the paper text.
- Loss weights: $\lambda_p = \lambda_s = 0.1$ for both $\mathcal{L}_{dyn}$ and $\mathcal{L}_{cam}$ (paper). The repo confirms the "$|\cdot|_{vgg}$" term is implemented as an LPIPS network call with `lpips_lambda=0.1` matching $\lambda_p$, not a raw VGG feature-distance loss as the notation suggests.
- Training length: 40,000 total photometric iterations (up from MoSca's 8,000-iteration baseline). Per iteration, 2 of the 18 sampled cameras are used (`n_aug_cams_to_use=2`, matches "two sampled cameras" in the paper text).
- Sampled camera pose optimization window: the released config restricts $\mathcal{C}_{sample}$ updates to iterations 2,000–6,000 of the 40,000 (`aug_optim_cam_after_steps=2000`, `stop_training_cam_opt_steps=6000`) — not stated in the paper.
- Dynamic mask binarization threshold: 0.8, applied to a soft Track-Anything-derived confidence map (`dynamic_mask_threshold` in the training config; `vidar_generate_masks.py --threshold 0.8`) — not given as a numeric value in the paper text.
- Noise scheduler: Discrete Euler (paper-stated, standard SDXL default).

#### Ablation Summary
Table 4, full-resolution DyCheck, co-visibility-masked metrics (PSNR-m / SSIM-m / LPIPS-m):
- Full ViDAR: 19.00 / 0.6672 / 0.3623
- w/o Sampled Camera Optimization (SO): 18.39 / 0.6514 / 0.4040 → **−0.61 dB PSNR, the largest single-component PSNR drop** — the paper attributes this to geometric inconsistency from misaligned pseudo-view poses.
- w/o Dynamic Reconstruction (DR): 18.93 / 0.6274 / 0.4497 → only −0.07 dB PSNR but the largest SSIM/LPIPS degradation (−0.040 SSIM, +0.087 LPIPS) — the paper attributes this to blurring from convergence toward a mean radiance value once dynamic masking is removed.
- w/o Tracking-based Gaussian Classification (TGS): 18.88 / 0.6651 / 0.3693 → smallest individual PSNR drop (−0.12 dB); the paper attributes its benefit to reduced floater artifacts rather than raw PSNR.
- w/o SO + DR + TGS (naive diffusion supervision, all three components removed): 18.46 / 0.6075 / 0.4656 → worst SSIM/LPIPS of all rows, confirming this is the actual failure mode the three mechanisms jointly address.
- Single most impactful component by PSNR: **Sampled Camera Pose Optimization**. Most impactful by SSIM/LPIPS: **Dynamic Reconstruction**.

#### Implementation Reality
- **Framework:** PyTorch; the released repo is a direct fork/extension of the MoSca reconstruction codebase (`lib_mosca`, `lib_moca` directories carried over almost verbatim) plus HuggingFace `diffusers` for SDXL DreamBooth-LoRA personalization and Img2Img enhancement. MIT-licensed; per the repo's citation block, accepted at NeurIPS 2025.
- **Key files:**
  - `lib_mosca/photo_recon_vidar.py` — the diffusion-aware reconstruction training loop (the two-pass $\mathcal{L}_{cam}$/$\mathcal{L}_{dyn}$ optimization described above).
  - `lib_mosca/gs_utils/loss_helper_vidar.py` — a generic masked `compute_rgb_loss` (L1 + SSIM + optional LPIPS) reused for both $\mathcal{L}_{dyn}$ (dynamic mask) and $\mathcal{L}_{cam}$ (full-frame mask) by passing different `sup_mask` arguments.
  - `vidar_sample_cameras.py`, `vidar_render_extra_cams.py` — extreme-pose + noised-interpolation camera sampling and rendering.
  - `train_dreambooth_lora_sdxl.py`, `vidar_enhance_extra_renders.py` — DreamBooth/LoRA personalization and the multistep img2img enhancement pass.
  - `vidar_generate_masks.py` — thresholds a precomputed soft dynamic-segmentation map into the binary $D_{m,t}$ mask used by $\mathcal{L}_{dyn}$.
  - `eval_utils/dycheck_metrics.py` — the shared masked PSNR/SSIM/LPIPS implementation (adapted from the DyCheck benchmark / Apple's `ml-pgdvs`), used unchanged for both the co-visibility-masked ("-m") and dynamic-masked ("-D") metrics.
  - `eval_utils/eval_dyncheck.py`, `vidar_evaluate.py` — evaluation driver that calls the masked-metric functions twice per scene: once against `test_covisible` (→ "-m" metrics) and once against `test_masks` (the Track-Anything dynamic mask → "-D"/dynamic metrics).
- **Notable implementation details differing from or unstated in the paper:**
  - Table 3's co-visibility/dynamic-mask "intersection" statistic (25.97% mean) is diagnostic-only. The actual PSNR-D/SSIM-D/LPIPS-D metrics in Table 2 are computed with the Track-Anything dynamic mask used **in place of**, not intersected with, the co-visibility mask: `vidar_evaluate.py` calls the identical `eval_dycheck_no_viz` function once with `gt_mask_dir=test_covisible` and once with `gt_mask_dir=test_masks`, with no combination of the two masks.
  - The masked PSNR/SSIM/LPIPS functions are literally standard PSNR/SSIM/LPIPS with a soft-mask-weighted mean substituted for the plain mean — inherited essentially verbatim from the DyCheck/`ml-pgdvs` codebase (file header explicitly credits both), not a ViDAR-specific formula.
  - The dynamic mask $D_{m,t}$ used during training is produced by thresholding a soft segmentation map at 0.8; the README points users to a pre-baked HuggingFace dataset of these masks rather than shipping a live Track-Anything integration in the training loop.
  - The "$|\cdot|_{vgg}$" perceptual term in $\mathcal{L}_{dyn}$/$\mathcal{L}_{cam}$ is implemented as an LPIPS network call (`lpips_model`, weight `lpips_lambda=0.1`), not a bare VGG feature-distance loss.
  - Sampled camera pose optimization is not applied for the full 40,000 iterations as the equations might imply; the shipped config restricts it to iterations 2,000–6,000.
  - Per-scene training config overrides exist that the paper does not mention (e.g., the `wheel` scene uses a separate config with `total_aug_cams=11` instead of 18).

#### Failure Modes & Limitations
The paper states diffusion is used only to *enhance* rendered images, so output quality remains bounded by the initial MoSca reconstruction's geometric accuracy — ViDAR "cannot repair major geometrical artefacts" already present in the base reconstruction. It also reports that CAT4D obtains a better LPIPS score, attributed to CAT4D sacrificing spatio-temporal consistency (visible in the supplementary video and reflected in worse PSNR/SSIM) for per-frame perceptual quality. Quantitatively, ViDAR's PSNR margin over the next-best baseline (MoSca) is smaller than its margin over all other baselines: "a minimum of 1dB improvement over all methods, except for MoSca where we average 0.94dB and 0.56dB higher in dynamic and co-visibility masked regions respectively."

---

## Relevance to ADAGS

External, quantified corroboration of G7 ("global PSNR is insufficient;
dynamic-region diagnostics required") — directly citable precedent for
dynamic-mask metrics as first-class evaluation. Any ADAGS event/dynamic
evaluation protocol should cite and extend this precedent rather than claim
the critique as novel.

## Connections

- Supports [[gap_map#G7 - A Benchmark/Diagnostic Claim Is Necessary]]

## Sources

- https://arxiv.org/abs/2506.18792
