---
type: paper
node_id: paper:song2025_coda4dgs
title: "CoDa: Context-aware Deformable Gaussian Splatting for Dynamic Scene Reconstruction"
authors: ["Jiwon Song", "Seung-Mok Lee", "Honggyu An", "Seung-Hwan Baek"]
year: 2025
venue: "ICCV"
tags: [dynamic-gs, context-aware, driving]
status: deep-dived
---

# CoDa-4DGS: Dynamic Gaussian Splatting with Context and Deformation Awareness for Autonomous Driving

**Paper:** https://arxiv.org/abs/2503.06744
**Code:** https://github.com/Chenwei-Liang/CoDa-4DGS
**Base method:** 4D Gaussian Splatting with HexPlane deformation field (Wu et al. 2024), extended with Feature-3DGS-style semantic feature distillation and StreetGaussian-style multi-frame LiDAR initialization for driving scenes. Benchmarked against S3Gaussian.

Note: the frontmatter author list ("Jiwon Song, Seung-Mok Lee, Honggyu An, Seung-Hwan Baek") does not match the authors found on arXiv/GitHub/ICCV for arXiv 2503.06744, which lists Rui Song, Chenwei Liang, Yan Xia, Walter Zimmer, Hu Cao, Holger Caesar, Andreas Festag, Alois Knoll. The frontmatter is preserved unedited per instructions, but this mismatch should be checked before citing.

## One-line thesis

CoDa-4DGS adds a self-supervised semantic-context feature and an explicit temporal-deformation feature to each Gaussian, then fuses both through a gated Deformation Compensation Network (DCN) whose sigmoid mask suppresses the residual correction for Gaussians it infers are static (e.g. sky, ground) — turning a single monolithic HexPlane deformation field into a deformation-plus-selective-correction pipeline that resists hallucinating motion on background content.

## Problem / Gap

Vanilla 4DGS/HexPlane and driving-scene variants like S3Gaussian deform every Gaussian from one shared deformation field conditioned only on position and time, which struggles in autonomous-driving capture: scenes mix large static backgrounds with dynamic foreground objects under a single roughly-linear ego-trajectory with weak multi-view constraint, so the field cannot reliably tell static from dynamic content and imparts spurious motion to things like sky and ground. It also degrades for dynamic objects that just entered the scene, since the deformation field has no prior-frame history for them to condition on.

## Method

Each Gaussian is augmented with three extra feature channels: a 128-d semantic context feature `f_seg`, self-supervised by rasterizing it and comparing (cosine similarity) against 2D foundation-model segmentation features (LSeg or SAM) in the style of Feature-3DGS; a temporal-deformation feature `f_def`, which is literally the raw HexPlane deformation output `ΔG = F(G,t)` retained as a feature rather than only applied as the final delta; and a 64-d sinusoidal time embedding `f_time`. These three are aggregated and passed into a Deformation Compensation Network (DCN) with two heads — an MLP `φ_p` that predicts a residual compensation vector, and a sigmoid filter `φ_s` that predicts a per-Gaussian relevance gate — whose elementwise product is added on top of the Gaussian's already-deformed attributes. The scene is initialized from LiDAR point clouds aggregated across all frames (StreetGaussian-style) rather than single-frame SfM.

## Assumptions

Multi-camera autonomous-driving capture with LiDAR available for scene initialization, a roughly linear/simple ego-vehicle trajectory, and availability of a 2D semantic segmentation foundation model producing usable per-frame pseudo-labels for self-supervision.

## Limitations / Failure Modes

The paper reports a 2.1x computational-complexity increase from expanding the per-Gaussian feature dimensionality (62 → 190 concatenated dims feeding the DCN) and a parameter-count increase from the added DCN and semantic rasterizer (36.4M vs. 35.8M baseline params). It targets large-scale outdoor driving scenes with LiDAR and roughly linear camera trajectories; domain emphasis (autonomous driving, Waymo/KITTI/NOTR) differs substantially from N3V cooking scenes — no handheld/indoor multi-view results are reported.

## Reusable Ingredients

- **Self-supervised semantic feature distillation into Gaussians** (Feature-3DGS-style cosine-similarity loss against a 2D foundation model's rendered feature map) — adds semantic context without manual labels.
- **Gated deformation compensation**: an MLP-predicted residual multiplied by a learned sigmoid mask, letting the model suppress corrections for Gaussians it infers are static — directly relevant to any static/dynamic separation problem.
- **Reusing a deformation field's raw output as an input feature** to a downstream correction network, rather than only consuming it as the final applied delta.
- **Sinusoidal encoding of a binary sparse time signal** (not just raw scalar time) to capture multi-scale temporal patterns.
- **Multi-frame-aggregated LiDAR initialization** for mixed static/dynamic scenes, instead of single-frame SfM point clouds.

---

### Deep Dive

#### Core Novelty

Relative to vanilla HexPlane-based 4DGS, CoDa-4DGS does not change how the base deformation is computed; it adds a second-stage correction network (the DCN) that consumes semantic context and the deformation field's own output as features, and applies its correction through a learned per-Gaussian gate rather than unconditionally. The key insight is that gating (not just added network capacity) is what lets the model distinguish "this Gaussian's motion is real" from "this Gaussian is static background being perturbed by field noise."

#### Mathematical Formulation

- Semantic feature association and supervision (per-Gaussian, rasterized then compared to a 2D foundation model's feature map via cosine similarity, following Feature-3DGS): $f_{seg}$ is a learned 128-d attribute per Gaussian; loss term $\mathcal{L}_f$ penalizes divergence from the 2D teacher features after rasterization.
- Temporal deformation feature (evaluated per-Gaussian, per-frame, as part of the HexPlane deformation query): $f_{def} \leftarrow \Delta\mathcal{G} = \mathcal{F}(\mathcal{G}, t)$, i.e., the standard HexPlane deformation network's output is retained as an explicit feature vector (dimension $N \times 62$) instead of being consumed only as the applied delta.
- Time embedding (sinusoidal, transformer-style): $f_{time} = \sin(\tau / 10000^{2i/d})$, where $\tau$ is a binary sparse time signal and $i$ indexes the embedding dimension ($f_{time} \in \mathbb{R}^{N \times 64}$).
- Deformation Compensation Network (applied per-Gaussian, after the base HexPlane deformation, before rasterization): $\mathcal{G}^t \leftarrow \mathcal{G}^t + \phi_p(f_{time}, f_{def}, f_{con}) \otimes \phi_s(f_{time}, f_{def}, f_{con})$, where $\phi_p$ is an MLP predicting a compensation vector, $\phi_s$ is a sigmoid-activated filter/gate network, and $\otimes$ is elementwise multiplication. ($f_{con}$ appears to denote the context feature, i.e. $f_{seg}$; the paper's notation is not fully disambiguated between the two symbols.)
- Total training loss: $\mathcal{L} = \lambda_{rgb}\mathcal{L}_{rgb} + \lambda_{d\text{-}ssim}\mathcal{L}_{d\text{-}ssim} + \lambda_{tv}\mathcal{L}_{tv} + \lambda_{depth}\mathcal{L}_{depth} + \lambda_f\mathcal{L}_f$ (standard RGB/D-SSIM/TV/depth terms plus the semantic feature term $\mathcal{L}_f$).

#### Algorithm / Pipeline Changes

1. Initialize Gaussians from LiDAR points aggregated across all frames (StreetGaussian-style), replacing single-frame SfM initialization.
2. Attach a learnable 128-d semantic feature `f_seg` to every Gaussian in addition to standard 3DGS attributes.
3. At render time for frame $t$, query the HexPlane deformation network $\mathcal{F}(\mathcal{G}, t)$ to get the canonical deformation $\Delta\mathcal{G}$; apply it as usual, but also keep it as feature `f_def` (dim $N \times 62$).
4. Compute a 64-d sinusoidal time embedding `f_time` from a binary sparse time signal.
5. Concatenate/aggregate `f_time`, `f_def`, and the context feature, and feed into the DCN: MLP head `φ_p` produces a compensation vector, sigmoid head `φ_s` produces a per-Gaussian gate; multiply and add the result to the already-deformed Gaussian attributes `G^t`. This is a new correction stage inserted after the standard deformation step and before rasterization.
6. Rasterize both an RGB image and a semantic feature map; supervise the feature map with cosine similarity against 2D foundation-model (LSeg/SAM) output on the same view, and supervise RGB with the standard photometric/depth/TV losses.
7. Train for 50,000 steps with the combined loss.

#### Key Hyperparameters & Design Choices

- Semantic feature dimension: 128.
- Temporal deformation feature dimension: $N \times 62$ (tied to spherical-harmonics coefficient count, k=48, plus other Gaussian attributes).
- Time embedding dimension: $N \times 64$.
- Training length: 50,000 steps.
- Learning rate: initial $1.6\times10^{-3}$, decaying to $1.6\times10^{-4}$ (decay schedule shape not specified).
- Loss weights: $\lambda_{rgb}=1$, $\lambda_{d\text{-}ssim}=0.2$, $\lambda_{tv}=1$, $\lambda_{depth}=0.5$, $\lambda_f=1$.
- DCN MLP depth/hidden dimensions: Not specified in paper.
- Gate network (`φ_s`) architecture details: Not specified in paper.
- Batch size / GPU configuration: Not specified in paper.

#### Ablation Summary

- **DCN gating mechanism (Table 4, Waymo PSNR):** no DCN = 31.71 dB → deeper MLP only, no gate = 31.75 dB → full gated DCN = 32.98 dB. The gate, not added MLP capacity, is responsible for essentially the entire ~1.27 dB gain (deeper MLP alone contributes only +0.04 dB). **This is the single most impactful component.**
- **Feature-type ablation (Table 5):** combined time+context+deformation-awareness = 32.86 dB vs. component-only variants ranging 32.69–32.80 dB — each feature contributes, but individually the deltas are small (roughly 0.06–0.17 dB per removed component); the gain comes from combining all three.
- **Overall (Table 1, Waymo):** full method 33.65 dB / 0.919 SSIM / 0.078 LPIPS vs. S3Gaussian 32.16 dB / 0.915 / 0.101 and vanilla 4DGS 31.02 dB / 0.901 / 0.136.

#### Implementation Reality

- **Framework:** PyTorch; repo explicitly states it is "developed based on 3D Gaussian Splatting, 4D Gaussians, Feature 3DGS and S3Gaussian."
- **Key files:** `gaussian_renderer/` (rasterization including the semantic feature channel), `scene/` (scene/data handling), `lseg_encoder/` (LSeg semantic feature extraction, the 2D teacher for `f_seg`), `lpipsPyTorch/` (perceptual loss), `submodules/diff-gaussian-rasterization` and `submodules/simple-knn` (external CUDA dependencies), `main_train.py` (training entry point).
- **Notable implementation details:** training is configured via `arguments/config.yaml` (source_path, semantic_feature_path, model_path, start_checkpoint, eval_only); dataset preparation is Waymo-specific with a precomputed semantic-feature-map path, i.e. semantic teacher features are precomputed offline rather than run online during training.

#### Failure Modes & Limitations

The paper's own stated limitation is compute/parameter overhead: expanding per-Gaussian feature dimensionality from 62 to 190 (after concatenating `f_time`, `f_def`, `f_con` for the DCN) increases computational complexity by roughly 2.1x, and the added DCN plus semantic rasterizer increase parameter count from 35.8M to 36.4M. The method is scoped to large-scale driving scenes with LiDAR initialization and roughly linear ego-trajectories; it does not report results outside this domain (no indoor/handheld/multi-view-rig evaluation).

---

## Relevance to ADAGS

Supports ADAGS's need for localized dynamic context beyond a single global LoRA basis.

## Connections

## Sources

- https://arxiv.org/abs/2503.06744
- https://github.com/Chenwei-Liang/CoDa-4DGS
- https://rruisong.github.io/publications/CoDa-4DGS/
- https://openaccess.thecvf.com/content/ICCV2025/html/Song_CoDa-4DGS_Dynamic_Gaussian_Splatting_with_Context_and_Deformation_Awareness_for_ICCV_2025_paper.html
