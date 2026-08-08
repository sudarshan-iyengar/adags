---
type: paper
node_id: paper:zoomers2026_nvgs
title: "NVGS: Neural Visibility for Occlusion Culling in 3D Gaussian Splatting"
authors: ["Zoomers et al."]
year: 2026
venue: "CVPR"
external_ids:
  arxiv: "2511.19202"
tags: [static-gs, visibility, occlusion-culling, learned-visibility]
status: deep-dived
---

# NVGS: Neural Visibility for Occlusion Culling in 3D Gaussian Splatting

**Paper:** https://arxiv.org/abs/2511.19202
**Code:** https://github.com/UHasselt-DigitalFutureLab/NVGSTrainer (offline visibility extraction + MLP training) and https://github.com/UHasselt-DigitalFutureLab/NVGSViewer (instanced rasterizer / viewer); project page https://brent-zoomers.github.io/nvgs/
**Base method:** 3D Gaussian Splatting (Kerbl et al. 2023) tile-based rasterization pipeline, extended with a custom instanced software rasterizer. The paper's direct point of comparison and closest prior method is V3DG (multi-resolution spatial clustering for composed 3DGS scenes); it also positions against mesh-style LoD adaptations for 3DGS (H3DG, LODGE, Octree-GS, FLoD, OccluGaussian).

## One-line thesis

A small MLP shared across every instance of a given asset learns that asset's continuous, viewpoint-dependent Gaussian-level visibility (nonzero rendering contribution), letting a render-time instanced rasterizer discard occluded Gaussians before instancing/rasterization — cutting VRAM 3-4x and raising FPS for large composited 3DGS scenes without the storage duplication that cluster-based LoD (V3DG) requires.

## Problem / Gap

Triangle-mesh occlusion culling and LoD techniques don't transfer to 3DGS because Gaussians are semi-transparent: a Gaussian behind another primitive can still contribute to the final pixel color, so a binary front/back occlusion test is invalid. The closest prior method, V3DG, works around this by building spatial-proximity clusters at multiple LoD levels per asset, but this roughly doubles per-asset storage and still leaves a large VRAM footprint (13.4-20.4GB across the paper's composed test scenes) because it stores redundant cluster geometry rather than a compact visibility signal. Other LoD adaptations for 3DGS (H3DG's chunk division, LODGE's distance-based LoD levels, Octree-GS's voxel-grid neural anchors, FLoD's independently-trained representation levels, OccluGaussian's occlusion-aware chunk partitioning) similarly rely on discrete, precomputed spatial structures rather than a learned continuous visibility function.

## Method

For each asset, NVGS samples ~2000 camera viewpoints around the object via Fibonacci-sphere sampling (with randomized distance-scaled offsets and extra auxiliary views near LoD/near-plane transitions to counter 3DGS "popping" artifacts), renders the asset, and records each Gaussian's per-pixel contribution $C_{G,p} = \alpha_p \cdot T$ (opacity times accumulated transmittance) as a ground-truth visibility label. A small shared visibility MLP (2 hidden layers, 32 neurons) is trained with a frequency-weighted binary cross-entropy loss on a fixed 16-dimensional input (normalized Gaussian mean, normalized view direction, distance, camera forward vector, plus a 6-D learned embedding of the Gaussian's opacity/scale/rotation produced by a same-sized secondary MLP) to predict visibility from arbitrary viewpoints. At render time, a custom instanced software rasterizer frustum-culls each asset instance, queries the shared visibility MLP per surviving Gaussian using a distance/FoV-normalized viewing direction, discards Gaussians predicted invisible, and only instances the survivors into per-tile lists before handing off to a standard tiled 3DGS rasterizer.

## Assumptions

Assumes 3DGS content is organized as discrete, individually-trained "assets" placed as repeated instances within a larger composed scene (e.g., many copies of the same tree or crowd member) — the shared MLP is trained once per asset and reused across all its instances, which is what avoids V3DG's per-cluster storage duplication. Assumes a trained, static 3DGS point cloud already exists (this is a post-hoc render-time culling layer, not a scene-reconstruction or training-time method), and that per-asset visibility can be adequately captured by ~2000 synthetic training viewpoints sampled around the object.

## Limitations / Failure Modes

The paper states MLP accuracy depends on Gaussian count and the complexity of the asset's visibility function, singling out high-frequency visibility patterns (e.g., tree crowns) as needing extra overhead for robustness. Validation is per-asset; the paper does not claim the same properties are verified end-to-end on full composed scenes at scale. NVGS cannot compete with true LoD methods at far viewing distances, since it only culls occluded Gaussians rather than reducing the per-Gaussian count the way LoD does. The optional radius-clipping component is a genuine speed/quality trade rather than a free win: in the ablation it recovers ~8 FPS (42.02 -> 50.59) but roughly doubles FLIP error (0.0031 -> 0.0067).

## Reusable Ingredients

- Per-pixel $\alpha \cdot T$ contribution as a visibility ground-truth label — turns any already-trained 3DGS asset into a labeled visibility dataset with no manual annotation, by simply logging the standard rasterizer's alpha-compositing terms during rendering.
- Fibonacci-sphere + randomized-offset + auxiliary-cone-view camera sampling — a general recipe for building a de-biased, popping-artifact-aware viewpoint dataset around an isolated 3D asset.
- 16-dimensional, Tensor-Core-aligned MLP input packing (drop frequency encoding, replace it with a small learned per-primitive embedding) — a general pattern for keeping small per-primitive MLPs cheap enough to query at rasterization time.
- Frequency-weighted BCE loss for imbalanced binary visibility/occupancy labels.
- Distance/FoV re-normalization formula for reusing a viewpoint-conditioned MLP trained under one camera configuration at inference time under a different camera configuration (focal length, screen coverage).

---

### Deep Dive

#### Core Novelty

NVGS replaces V3DG's explicit multi-resolution spatial clustering (store several discrete LoD copies of an asset) with an implicit, continuous, learned visibility function shared across every instance of that asset. The key insight is that visibility in 3DGS is inherently a soft, continuous quantity (a Gaussian's alpha-weighted transmittance contribution) rather than a binary mesh-style occlusion boolean, so a small regression/classification MLP is a more natural and more compact fit than bucketed LoD levels or spatial clusters — and because the MLP is shared per-asset rather than duplicated per-cluster or per-instance, storage does not scale with the number of times that asset appears in the scene.

#### Mathematical Formulation

- Training-view distance placement (evaluated once per asset, during offline dataset generation, to pick near/far camera distances for view sampling):
$$d = \frac{r}{\tan(\theta/2) \cdot p}$$
where $d$ is the camera distance, $r$ is the asset's half-diagonal bounding length, $\theta$ is the camera field of view, and $p$ is the target screen-coverage percentage (paper uses $p=90\%$ for the near distance and $p=5\%$ for the far distance).

- Gaussian visibility contribution (defines the ground-truth label used to supervise the visibility MLP; computed during the offline rendering passes that build the training set, not at inference time):
$$C_{G,p} = \alpha_p \cdot T$$
where $\alpha_p$ is the Gaussian's opacity at pixel $p$ and $T$ is the accumulated transmittance up to that Gaussian in the compositing order; a Gaussian is labeled visible for a given view iff its summed $C_{G,p}$ over the image is nonzero.

- Render-time distance normalization (evaluated per Gaussian, per frame, immediately before the visibility MLP query, to correct for FoV/focal-length mismatches between the training cameras and the actual render camera):
$$d_t = d_r \cdot \frac{f_t}{f_r} \cdot \frac{1}{s}$$
where subscript $t$ denotes the training-time camera parameterization and $r$ the render-time camera, $f$ is focal length, and $s$ is a scaling factor; the result is then normalized to $[-1, 1]$ using the asset's precomputed per-asset min/max training distance before being fed to the MLP.

#### Algorithm / Pipeline Changes

Offline, per-asset dataset construction:
1. Preprocess the trained 3DGS asset: remove Gaussians with opacity below $1/255$; center the asset for uniform processing.
2. Compute near/far training-camera distances via $d = r/(\tan(\theta/2)\cdot p)$ with $p = 90\%$ (near) and $p = 5\%$ (far) screen coverage.
3. Sample 2000 camera positions around the asset using Fibonacci-sphere sampling; apply random offsets scaled by distance (zero offset at the minimum distance) to avoid center bias.
4. Add auxiliary views on cones (5° angular resolution) rotated toward the camera, specifically to sample the near-plane/LoD transition region where 3DGS popping artifacts occur.
5. Render each sampled view with the standard 3DGS rasterizer and log $C_{G,p} = \alpha_p \cdot T$ per Gaussian per view as the visibility training label.

MLP training (replaces/augments nothing in the base rasterizer; runs entirely offline):
6. Build a 16-D input per (Gaussian, view) sample: normalized mean (3), normalized view direction (3), distance (1), camera forward vector (3), plus a 6-D embedding from a secondary MLP (same 2-layer/32-neuron architecture) that encodes the Gaussian's opacity, scale, and rotation.
7. Train the primary visibility MLP (2 hidden layers, 32 neurons/layer, ReLU, single output) with Adam, batch size $2^{19}$ samples drawn across views and Gaussians, LR $2\times10^{-3} \to 2\times10^{-4}$ under cosine warm-up for the first 20% of steps followed by exponential decay, using a frequency-weighted binary cross-entropy loss. Resulting checkpoint is 18KB.

Render-time instanced rasterization (new pipeline stage inserted before the standard tiled rasterizer):
8. Precompute the secondary (Gaussian-embedding) MLP output once per rendering session, since it does not depend on the camera.
9. Per frame, per asset instance: frustum-cull Gaussians.
10. For surviving Gaussians, make a distance-based decision on whether the visibility-MLP query is worth its cost at all (skipped for Gaussians where culling isn't beneficial).
11. Convert render-camera position/direction into the asset's local, training-normalized space using the distance/FoV correction and per-asset min/max normalization.
12. Query the shared visibility MLP per Gaussian and discard those predicted invisible.
13. Instance only the surviving Gaussians into per-tile lists; optionally apply radius clipping using the 2D covariance determinant at 1σ.
14. Rasterize survivors with a standard tiled 3DGS rasterizer (skip fragments with $\alpha < 1/255$; early-terminate rays at transmittance $< 10^{-4}$).

#### Key Hyperparameters & Design Choices

- Visibility MLP: 2 hidden layers, 32 neurons/layer, ReLU activations, 1 output.
- Secondary (Gaussian-embedding) MLP: same 2-layer/32-neuron architecture, 6-D output.
- MLP input dimensionality: fixed at 16 (deliberately no frequency encoding), chosen to be optimal for Tensor Core execution.
- Camera sampling: 2000 viewpoints/asset via Fibonacci-sphere sampling.
- Near/far training-view screen coverage: 90% (near) / 5% (far).
- Auxiliary-view cone resolution: 5°.
- Batch size: $2^{19}$ samples.
- Optimizer: Adam; LR $2\times10^{-3} \to 2\times10^{-4}$; cosine warm-up for first 20% of iterations, then exponential decay.
- Visibility-MLP checkpoint size: 18KB.
- Opacity culling threshold: $\alpha < 1/255$.
- Transmittance early-termination threshold: $10^{-4}$.
- Radius clipping: optional, based on the 2D covariance determinant at 1σ.
- Total training iteration count / epochs: Not specified in paper (only batch size and schedule shape are given).
- Per-asset build time (visibility extraction + MLP training combined): 2-4 minutes, versus 7-9 minutes for V3DG.

#### Ablation Summary

Component-by-component ablation (Table 3; FPS, FLIP error, and VRAM measured with each component added cumulatively):
1. **FoV correction — largest single-component impact, flagged by the paper itself.** Going from "+ Auxiliary views" (FLIP 0.0113) to "Full + FoV correction" (FLIP 0.0031) is a ~3.6x reduction in FLIP error, at a substantial FPS cost (53.23 -> 42.02) and VRAM increase (2.98GB -> 3.88GB).
2. Auxiliary views: FLIP improves 0.0142 -> 0.0113 over "+ Random offset," at a cost of ~6.5 FPS (59.75 -> 53.23) and +0.26GB VRAM.
3. Random offset: FLIP improves 0.0168 -> 0.0142 over Fibonacci-only sampling, at a cost of ~1.9 FPS.
4. Fibonacci vs. LongLat camera sampling: marginal improvement (FLIP 0.0170 -> 0.0168, FPS 61.04 -> 61.61) — a nearly free swap.
5. Radius clipping (optional, added last): trades quality for speed rather than improving both — FPS recovers from 42.02 to 50.59 but FLIP roughly doubles from 0.0031 to 0.0067.

#### Implementation Reality

- **Framework:** PyTorch + CUDA (CUDA Toolkit 12.8 recommended). The trainer repository builds on a 3DGS-style training codebase with submodules for `diff-gaussian-rasterization`, `simple-knn`, `fused-ssim`, `FasterGS`, and `tiny-cuda-nn` (used for the visibility/embedding MLPs). The viewer repository integrates with the NeRFICG framework and ships its own `NVGSCudaBackend` CUDA extension for the instanced rasterizer and MLP query.
- **Key files:** Trainer — `run_asset.py` (single-asset training), `run_all_assets.py` (batch training across a dataset), `run_preprocessing.py` (data prep), `run_precompute_contr.py` (the offline $C_{G,p}$ visibility-contribution extraction step), `run_training.py` (MLP training loop), with code organized under `nvgs/`, `gaussian_renderers/`, `models/`, `scene/`. Viewer — `Model.py`, `Renderer.py`, `Trainer.py`, `flip_loss.py` (FLIP metric implementation used for evaluation), and `NVGSCudaBackend/`.
- **Notable implementation details not stated in the paper:** the trainer repo documents refinements beyond the preprint — smaller tile sizes during visibility extraction and an optional `FasterGS` renderer path. The viewer supports rendering a mix of neural-visibility-enabled assets and plain (unculled) 3DGS assets in the same scene, and claims to handle 100M+ Gaussian scenes via instance reuse rather than duplication. The viewer repo is MIT-licensed.
- Caveat: repository contents were inspected via automated summarization of the READMEs rather than a full manual read of source files, so exact file-level logic (e.g., precise CUDA kernel structure) was not independently verified line-by-line.

#### Failure Modes & Limitations

Quoted from the paper's stated limitations: "The performance of the MLP depends on the number of Gaussians and the complexity of their visibility functions, which may require additional overhead to ensure robustness for complex objects." The paper explicitly flags high-frequency visibility functions (its example: tree crowns) as challenging. It also notes the method is demonstrated on individual assets rather than verified identically at full-scene scale, that combining occlusion culling with true LoD (as is standard for mesh-based scenes) is left as future work rather than solved here, and that further engineering optimization (e.g., CUDA Streams) is likely still available.

---

## Relevance to ADAGS

Occupies the phrase "learned per-Gaussian visibility function". Static
scenes, render-time culling only — no lifecycle, no temporal state, no
preservation. Constrains the wording of the deferred Route 2 (learned
visibility field) in [[objectives/depth-visibility-capacity-v1]]: if that
route is ever activated it must be positioned against NVGS.

## Connections

- Constrains Route 2 wording in [[objectives/depth-visibility-capacity-v1]]

## Sources

- https://arxiv.org/abs/2511.19202
