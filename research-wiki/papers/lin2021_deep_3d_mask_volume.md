---
type: paper
node_id: paper:lin2021_deep_3d_mask_volume
title: "View Synthesis of Dynamic Scenes based on Deep 3D Mask Volume"
authors: ["Kai-En Lin", "Guowei Yang", "Lei Xiao", "Feng Liu", "Ravi Ramamoorthi"]
year: 2021
venue: "ICCV"
external_ids:
  arxiv: "2108.13408"
  doi: null
  s2: null
tags: ["layered-representation", "disocclusion", "dynamic-view-synthesis", "background-memory"]
added: 2026-07-14T22:18:30Z
status: deep-dived
---

# View Synthesis of Dynamic Scenes based on Deep 3D Mask Volume

> Full arXiv HTML is not published for this paper (2021, pre-HTML-conversion era) and the arXiv/CVF/PDF sources exceeded fetchable size or returned 403. This deep dive is built from: the arXiv/project-page abstract, the UCSD project page, a `researchgate`-indexed abstract giving the published TPAMI DOI, targeted web-search snippets that surfaced verbatim ablation numbers and a limitation sentence from the paper body, and the **official GitHub implementation** (`ken2576/deep-3dmask`) — specifically `README.md`, `train_mask/net.py`, `train_mask/module.py`, `train_mask/losses.py`, and the training config files, which were fetched directly. Architecture and loss details below are code-verified; exact in-paper equation notation could not be transcribed verbatim, so equations are reconstructed from the code and flagged as such.

**Paper:** https://arxiv.org/abs/2108.13408 (extended/journal version: TPAMI, DOI 10.1109/TPAMI.2023.3289333)
**Code:** https://github.com/ken2576/deep-3dmask
**Base method:** Multiplane Image (MPI) view synthesis via 3D-CNN plane-sweep networks (Zhou et al. 2018 *Stereo Magnification*; Flynn et al. 2019 *DeepView*-style deep plane-sweep prediction), extended to dynamic scenes with a second, static "background" MPI and a learned 3D mask volume that blends the two per depth-voxel.

## One-line thesis

Predicting a *second* MPI for the static background (accumulated from frames where it was visible) and a learned 3D mask volume to blend it against the instantaneous foreground MPI per depth-voxel lets the renderer swap in real background content on disocclusion, instead of hallucinating it frame-by-frame — eliminating the flicker that framewise inpainting/MPI methods produce when the occluder moves.

## Problem / Gap

Prior binocular/multi-view dynamic view synthesis (e.g., Local Light Field Fusion-style per-frame MPI prediction, Broxton et al. 2020 immersive light field video) treats every frame independently, so when a foreground object moves and uncovers background, the network must invent that region from a single frame's stereo cue. Because the guess is uncorrelated across frames, the disoccluded region flickers and produces ghosting (the paper cites Mildenhall et al. 2019-style baselines showing blurred bushes and ghosted silhouettes of objects held by people in front of the camera). 2D masking approaches that try to gate this in image space cannot represent the problem volumetrically, so they cannot separate "this pixel is background revealed at this depth" from "this pixel is occluder" when both project to overlapping image regions across time.

## Method

The pipeline runs two MPI predictions per stereo frame pair from a fixed 3D-CNN encoder-decoder (`MPINet3d`): one MPI for the **instantaneous** (foreground/dynamic) content from the current stereo pair, and one MPI for the **background**, built from a background plate accumulated/observed across the video where that region was unoccluded. A second, identically-shaped network (`MaskNet3d`) takes a 12-channel **temporal plane-sweep volume** — the instantaneous left image, the right image warped to each depth plane, the background left image, and the background right image warped to each depth plane — and predicts a per-voxel mask volume over the same 32 depth planes. That mask blends the foreground and background MPI color layers per depth-voxel before standard alpha over-compositing renders the final novel view. The MPI network is pretrained on RealEstate10K (static real-estate video, self-supervised static-scene view synthesis) and then frozen/fine-tuned while the mask network is trained on the paper's own dynamic-scene dataset.

## Assumptions

Captures binocular (2-view) or multi-view video from **static** cameras only, with a scene decomposable into one dynamic occluder layer and one time-invariant static background layer; it assumes the true background becomes unoccluded and observable at some point during the video so a background plate can be accumulated, and that camera baseline/geometry is known (LLFF-style pose format) for plane-sweep warping.

## Limitations / Failure Modes

The paper reports ambiguity when the scene contains texture-less structure oriented parallel to the camera baseline (e.g., beams, handrails), because plane-sweep stereo cues degenerate in that configuration. Some scenes show ghosting artifacts in the *extracted background* plate itself (imperfect background accumulation propagates into every later disocclusion). The two-layer (single dynamic occluder + single static background) decomposition is a structural assumption, not a learned generality — it does not extend to a dynamic object occluding another dynamic object, or to a hidden surface that is itself non-rigid/deforming rather than static.

## Reusable Ingredients

- **Dual-MPI decomposition (instantaneous vs. accumulated-background):** separates "what's here right now" from "what's normally here," reused/composited only where evidence says it's hidden — directly relevant to hidden-surface memory design.
- **Temporal plane-sweep volume (TPSV) as mask-network input:** stacking current-frame and background-plate views (both warped to candidate depth planes) gives the mask network explicit multi-hypothesis depth evidence rather than a single RGB frame.
- **Per-depth-voxel (not per-pixel) blending mask:** the mask lives in the same 3D volume as the MPI, so blending decisions are made before compositing/occlusion resolution, not as a 2D post-hoc gate.
- **Two-stage training (pretrain geometry network on abundant static data, then train the novel component against it):** MPI network pretrained on RealEstate10K before the mask network ever sees dynamic-scene supervision, isolating the novel component's training signal.
- **Sparsity loss on mask/alpha volumes:** an L1 penalty on mean absolute volumetric values, encouraging the mask to commit rather than blend everywhere.

---

### Deep Dive

#### Core Novelty
Relative to single-MPI dynamic-scene baselines, this paper adds (a) a second MPI branch fed by an accumulated background plate instead of only the current frame, and (b) a dedicated 3D CNN that predicts a *volumetric* (per depth-voxel) blending mask between the two MPIs, rather than gating in image space after rendering. The key insight is that disocclusion is a depth-and-time-dependent event, so the decision of "use background vs. foreground content here" needs the same 3D structure as the geometry itself — a 2D post-hoc mask cannot correctly resolve cases where foreground and background project to the same pixel at different depths across frames.

#### Mathematical Formulation
Exact paper notation was not accessible; the following is reconstructed from the official code (`train_mask/net.py`, `module.py`) and is code-verified rather than transcribed from the PDF.

- **Depth plane sampling:** $d_k$ for $k = 1 \ldots 32$ are sampled at inverse-linear (disparity-linear) spacing between a scene-dependent near/far bound — standard MPI plane placement, evaluated once per clip before plane-sweep warping.
- **Plane-sweep warp:** each source view is homography-warped into the reference camera frame at every depth plane $d_k$ using known intrinsics/extrinsics, producing the plane-sweep volume that both `MPINet3d` and `MaskNet3d` consume. Evaluated per depth plane, before the 3D-CNN encoder.
- **MPI prediction:** $\text{MPINet3d}(\cdot): \mathbb{R}^{6} \to (\alpha_k, w_k)$ predicts, per depth plane $k$, an alpha/opacity channel (sigmoid) and blending weights (softmax) used to form RGB color layers — run once for the instantaneous plane-sweep volume and once for the background plane-sweep volume (two forward passes, shared weights).
- **Mask prediction:** $\text{MaskNet3d}(\cdot): \mathbb{R}^{12} \to M_k \in [0,1]$, a per-depth-voxel mask (sigmoid output, single channel) from the 12-channel TPSV (instantaneous-left, instantaneous-right-warped, background-left, background-right-warped, 3 channels each). Evaluated after both MPIs exist, before compositing.
- **Volumetric blend (reconstructed, not verbatim):**
$$C_k = M_k \odot C_k^{\text{fg}} + (1 - M_k) \odot C_k^{\text{bg}}$$
  blending the per-depth-plane color layers of the foreground and background MPIs using the predicted mask; per the paper's own description the alpha used for final compositing is taken from the instantaneous (foreground) MPI's alpha volume rather than a separately blended alpha — exact handling of alpha blending at mask boundaries is **not specified in accessible sources**.
- **Over-compositing:** standard back-to-front alpha compositing of $\{C_k, \alpha_k\}$ over $k=1\ldots32$ to render the final RGB image and foreground alpha — this is the unmodified MPI rendering operator, evaluated last, at the target (novel) camera viewpoint after re-warping the composited MPI.

#### Algorithm / Pipeline Changes
1. Accumulate a background plate per scene by observing frames where each background region is unoccluded (data preparation, not a learned step).
2. Pretrain `MPINet3d` (a 3D-CNN encoder-decoder, GroupNorm + ReLU, with skip connections, 6-channel input i.e. left+right stereo, 2-channel output i.e. alpha + blend weight) on RealEstate10K for static-scene view synthesis (400 epochs per `configs/train_realestate10k.txt`).
3. Freeze/reuse that checkpoint to run two forward passes per training sample: one on the instantaneous stereo plane-sweep volume, one on the background-plate plane-sweep volume, yielding two 32-plane MPIs.
4. Build the 12-channel TPSV (instantaneous L/R-warped + background L/R-warped) and run it through `MaskNet3d` (identical architecture to `MPINet3d` but 12-channel input, 1-channel sigmoid output) to get a 32-plane mask volume.
5. Blend the two MPIs' color layers per depth-voxel with the predicted mask (Eq. above); take alpha from the instantaneous MPI.
6. Over-composite to the source viewpoint for loss computation, and re-warp to arbitrary novel viewpoints at inference/rendering time (`render_llff_video.py`).
7. Train `MaskNet3d` end-to-end (400,000 steps, config `train_mask.txt`) against the frozen/pretrained MPI network using the combined loss below, at 360×640 resolution.

#### Key Hyperparameters & Design Choices
- Depth planes: 32 (both MPI and mask volumes).
- MPI network input/output channels: 6 in (stereo pair) → 2 out (alpha, blend weight) per plane.
- Mask network input/output channels: 12 in (TPSV) → 1 out (mask) per plane.
- MPI pretraining: RealEstate10K, `num_epochs = 400`, loss type `vgg`.
- Mask training: `num_steps = 400000`, image resolution `[360, 640]`, loss type `vgg_only`.
- Combined loss weights (from `CombinedLoss` in `losses.py`): VGG perceptual (`VggBNLoss`) weight **1.0**, `MaskLoss` weight **0.25**, `SparseLoss` weight **0.10**.
- `VggBNLoss` per-layer weights (VGG19-BN features, 5 layers): **2.6, 4.8, 3.7, 5.6, 1.5**.
- `MaskLoss` dilates the ground-truth mask with a convolution kernel, default **5×5**, before penalizing foreground/background alpha.
- Learning rate, optimizer, and batch size: **not specified in accessible sources** (config files fetched did not expose these fields; would require `opt.py`/`train.py` internals not retrieved).

#### Ablation Summary
From a web-search-surfaced excerpt of the paper's loss ablation (PSNR, higher is better):
- Full method ("Ours", rendering/VGG loss only): **26.22 dB** — best.
- + sparsity loss ($\mathcal{L}_s$): **26.18 dB** (−0.04 dB).
- + sparsity loss + explicit mask supervision loss: **26.09 dB** (−0.13 dB vs. full method).

The paper's own interpretation: relying on the rendering (VGG) loss alone gives better temporal consistency and slightly better visual quality than adding explicit sparsity/mask-supervision terms — i.e., the auxiliary losses each cost a small amount of PSNR even though they were presumably added to improve mask crispness/temporal stability qualitatively. Most impactful single factor per this table: removing the rendering-loss-only training and adding direct mask supervision costs the most (−0.13 dB total). Full ablation table (all rows/metrics, SSIM/LPIPS if reported) was not accessible — this is a partial extract.

#### Implementation Reality
- **Framework:** PyTorch, custom repo (not built on a public gaussian-splatting or NeRF codebase — this predates 3DGS).
- **Key files:**
  - `train_mask/net.py` — `RenderNet`, the top-level module orchestrating plane generation, plane-sweep warping, dual MPI prediction, mask prediction, blending, and over-compositing.
  - `train_mask/module.py` — `MPINet3d` and `MaskNet3d`, both 3D-CNN encoder-decoders (conv1–conv4 encoder with GroupNorm+ReLU, stride-2 downsampling every other block; decoder upsamples ×2 with skip concatenation at each of 3 stages; final MPI output applies sigmoid to alpha and softmax to blend weights, final mask output applies sigmoid).
  - `train_mask/losses.py` — `VggBNLoss`, `MaskedL1Loss`, `FgbgL1Loss`/`FgbgVGGLoss`, `MaskLoss`, `SparseLoss`, `CombinedLoss`, and a `loss_dict` registry for config-driven loss selection.
  - `train_mask/homography.py`, `train_mask/projector.py` — plane-sweep warping and re-projection utilities.
- **Notable implementation details:** the two-stage split (pretrain MPI on RealEstate10K, then train the mask network against a frozen/fine-tuned MPI checkpoint) is enforced structurally by separate `train_mpi/` and `train_mask/` directories and separate config files, not mentioned as a named technique in the abstract-level material reviewed. The mask network reuses the exact same `MPINet3d`/`MaskNet3d`-style architecture for both the MPI and mask branches (same encoder-decoder shape, different channel counts), which is a code-level design choice not obviously implied by the paper's abstract description.

#### Failure Modes & Limitations
- Texture-less structure parallel to the camera baseline (beams, handrails) creates depth/mask ambiguity, since plane-sweep stereo matching degenerates for such geometry regardless of the mask network.
- The accumulated background plate can itself contain ghosting artifacts in some scenes, which then propagates into every subsequent disocclusion rendered from it — the method has no mechanism to detect or correct a bad background plate at inference time.
- Static-background assumption is structural, not a soft prior: the architecture has exactly one foreground MPI slot and one background MPI slot, so it cannot represent more than two depth layers or a background that itself moves.

---

## Relevance to This Project

It is the clearest layered precedent for the desired hidden-surface memory, while its static-background assumption marks the ADAGS novelty boundary.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Sources

- https://arxiv.org/abs/2108.13408
- https://github.com/ken2576/deep-3dmask
- https://cseweb.ucsd.edu/~viscomp/projects/ICCV21Deep/
- https://cseweb.ucsd.edu/~viscomp/projects/ICCV21Deep/assets/deep_iccv.pdf (paper PDF, not directly fetchable — size-limited)
