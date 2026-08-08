---
type: paper
node_id: paper:mubashshira2026_cloth_hugs
title: "CLOTH-HUGS: Cloth Aware Human Gaussian Splatting"
authors: ["Sadia Mubashshira", "Nazanin Amini", "Kevin Desai"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2604.15875"
tags: [gaussian-splatting, layered, human-avatar, occlusion-order]
status: deep-dived
---

# CLOTH-HUGS: Cloth Aware Human Gaussian Splatting

**Paper:** https://arxiv.org/abs/2604.15875
**Code:** Not found (no GitHub repo, project page, or code release located in the paper, Papers With Code, or GitHub search as of this deep-dive)
**Base method:** 3D Gaussian Splatting (Kerbl et al. 2023) + HUGS (Human Gaussian Splats, apple/ml-hugs-style monolithic body-cloth avatar); cloth mesh priors from SNUG; SMPL body model and skeleton.

## One-line thesis

Splitting a monocular clothed-human avatar into two separately-optimized Gaussian layers (body, cloth) that share a canonical triplane space but are composited at render time via a depth-derived visibility matte lets loose garments deform independently of the body surface, instead of being absorbed into one monolithic skin-and-cloth Gaussian set.

## Problem / Gap

Prior monocular human Gaussian avatars (e.g., HUGS) represent body and clothing as a single Gaussian population driven by SMPL skinning, which forces cloth to move rigidly with the nearest bone and cannot express loose-garment dynamics (skirts, jackets) that separate from the body surface. These methods also lack any physics-informed supervision for cloth shape, so garment geometry drifts under fast or non-rigid motion and produces double layers or holes at occlusion boundaries between body and cloth.

## Method

Cloth-HUGS encodes body and cloth Gaussians in a shared canonical triplane (256x256x32 per plane), decoded per-Gaussian by three MLPs (appearance via spherical harmonics, geometry corrections, and deformation/LBS weights). Cloth Gaussians are initialized from SNUG-simulated garment meshes rather than sampled from the body surface, giving them independent topology. Both layers are deformed from canonical to posed space using SMPL-driven linear blend skinning (24 joints, softmax-normalized weights), with cloth LBS weights regularized toward weights transferred from the underlying SMPL body. Rendering is a two-pass, depth-aware composite: pass 1 jointly rasterizes body and scene Gaussians to produce a base image, pass 2 rasterizes cloth Gaussians and derives a per-pixel visibility matte from depth, and the final image blends cloth over the base using that matte. Training combines photometric losses with cloth-specific physics regularizers (LBS-weight matching, simulation-mesh alignment, ARAP shape preservation, silhouette mask consistency).

## Assumptions

Monocular RGB video of a single performer with known/estimated SMPL pose per frame; requires a pre-fit or estimated SMPL body model and a SNUG-generated simulated cloth mesh as an initialization/regularization target, i.e., it assumes access to a physics cloth simulator or precomputed simulated garment geometry, not just raw video.

## Limitations / Failure Modes

The paper's available text does not contain an explicit, itemized limitations section; the deep-dive fetch found no dedicated failure-mode discussion in the HTML version. Structurally implied constraints (not stated as an explicit "Limitations" section by the authors): dependence on SNUG cloth-simulation priors and accurate SMPL pose estimates, and no stated handling for garments SNUG cannot simulate (e.g., very loose or multi-layer clothing) since cloth Gaussians are initialized from and regularized toward that simulated mesh.

## Reusable Ingredients

- **Depth-aware two-pass visibility-matte compositing** (`I_final = I_cloth ⊙ V_cloth + I_base ⊙ (1−V_cloth)`): a cheap way to enforce a fixed occlusion order between two Gaussian populations without sorting all Gaussians together.
- **Cross-layer LBS-weight regularization** (`L_cloth-lbs`): ties a secondary layer's learned skinning weights to weights transferred from a primary layer's known skeleton, useful whenever a new Gaussian group needs to inherit articulation from an already-rigged one.
- **Simulation-alignment loss via bidirectional Chamfer + Geman-McClure**: a robust-to-outliers way to pull a Gaussian point set toward a target mesh (here a physics-simulated cloth mesh) without hard point correspondence.
- **ARAP variance regularizer on Gaussian positions**: keeps local structure of a deforming Gaussian layer rigid-ish using only pairwise edge-length variance, cheaper than full ARAP energy.

---

### Deep Dive

#### Core Novelty

Relative to monolithic body+cloth avatars (HUGS), Cloth-HUGS's change is architectural disentanglement: cloth gets its own Gaussian population, initialized from a physics-simulated mesh (SNUG) instead of being sampled as part of the body surface, and is composited on top of a jointly-rendered body+scene pass using a depth-based visibility matte rather than being blended into one global sort. The insight is that giving cloth independent geometry initialization and its own regularization targets (simulated mesh, ARAP, mask) lets it deviate from body-surface motion, which a single shared Gaussian set under one skinning field cannot do.

#### Mathematical Formulation

- **Cloth LBS regularization** (Eq. 1): $\mathcal{L}_{\text{cloth-lbs}} = \lambda_{\text{cloth-lbs}} \| W_{\text{cloth}} - W_{\text{cloth}}^{gt} \|_2^2$, $\lambda=1000.0$. $W_{\text{cloth}}$ is the learned per-Gaussian linear-blend-skinning weight vector (24 SMPL joints) for cloth Gaussians; $W_{\text{cloth}}^{gt}$ is a target obtained by transferring the underlying SMPL body's skinning weights to the cloth. Evaluated per-Gaussian during canonical-to-posed deformation, before rasterization.
- **Simulation alignment** (Eq. 2): a bidirectional Chamfer distance between cloth Gaussian positions and the SNUG-simulated cloth mesh vertices, passed through a Geman-McClure robust kernel; $\lambda_{\text{sim}}=1.0$. Computed on posed cloth Gaussian centers each training step to pull them toward physically simulated garment geometry, robust to simulation-mesh outliers/topology mismatch.
- **ARAP regularization** (Eq. 3): $\mathcal{L}_{\text{ARAP}} = \lambda_{\text{ARAP}} \, \text{Var}(\{\|v_i - v_j\|_2 : (i,j) \in \mathcal{E}\})$, $\lambda=0.5$. $\mathcal{E}$ is a set of neighbor-edges among cloth Gaussians/mesh vertices; the loss penalizes variance in pairwise edge lengths across pose changes, i.e., an as-rigid-as-possible proxy computed without solving a full rotation-fitting ARAP energy. Applied per training iteration on posed cloth geometry.
- **Mask consistency** (Eq. 4): $\mathcal{L}_{\text{mask}} = \lambda_{\text{mask}} \frac{1}{|N|} \| M_{\text{render}} - M_{\text{gt}} \|_2^2$, $\lambda=1.0$. $M_{\text{render}}$ is the rendered silhouette (alpha) from the composited image, $M_{\text{gt}}$ the ground-truth human/cloth mask; $N$ the pixel count. Evaluated after the full two-pass render, as a loss term on the final composited alpha.
- **Combined loss** (Eq. 5-6): reconstruction terms L1 (weight 0.8), SSIM (weight 0.2), LPIPS (weight 1.0), summed with the four physics terms above. Standard photometric-loss weighting is not otherwise modified.
- **Depth-aware compositing** (Eq. 7): $I_{\text{final}} = I_{\text{cloth}} \odot V_{\text{cloth}} + I_{\text{base}} \odot (1 - V_{\text{cloth}})$. $I_{\text{base}}$ is the pass-1 joint render of body+scene Gaussians; $I_{\text{cloth}}$ is the pass-2 render of cloth Gaussians alone; $V_{\text{cloth}}$ is a per-pixel visibility matte derived from comparing rendered depth between the two passes (cloth wins where it is in front). Evaluated once per frame, after both rasterization passes, as the final image-formation step — this is the mechanism that enforces occlusion order between layers, not a per-Gaussian state.

#### Algorithm / Pipeline Changes

1. Initialize body Gaussians $\mathcal{G}^{canon}_B$ from SMPL surface in canonical (T-pose-like) space; initialize cloth Gaussians $\mathcal{G}^{canon}_C$ from a SNUG-simulated garment mesh in the same canonical space.
2. Encode both layers' canonical positions into a shared triplane feature grid (256x256x32 per plane, 3 planes).
3. Decode per-Gaussian attributes via three MLPs: appearance (spherical-harmonics coefficients), geometry (position/rotation/scale corrections, $\mu_{\text{def}} = \mu_{\text{canon}} + \Delta\mu$ plus rotation/scale deltas), and deformation (per-joint LBS weights + pose-dependent offsets).
4. Apply SMPL-driven linear blend skinning (24 joints, softmax-normalized weights) to map each layer from canonical to posed/world space for the current frame's pose parameters.
5. Regularize cloth deformation with the four physics losses (cloth-LBS matching to transferred SMPL weights, simulation-mesh Chamfer alignment, ARAP edge-variance, mask consistency) computed on the posed cloth Gaussians / rendered output each step.
6. Render pass 1: rasterize body Gaussians + scene Gaussians jointly (standard alpha-blended 3DGS rasterization) to get $I_{\text{base}}$ and its depth buffer.
7. Render pass 2: rasterize cloth Gaussians alone to get $I_{\text{cloth}}$ and its depth buffer; derive visibility matte $V_{\text{cloth}}$ from the depth comparison between the two passes.
8. Composite: $I_{\text{final}} = I_{\text{cloth}} \odot V_{\text{cloth}} + I_{\text{base}} \odot (1-V_{\text{cloth}})$ (Eq. 7); this replaces single-pass monolithic rasterization used by HUGS-style baselines.
9. Backpropagate combined photometric + physics losses (step 5's terms plus reconstruction L1/SSIM/LPIPS on $I_{\text{final}}$) through both MLP decoders and per-Gaussian parameters.

Each frame at inference/training time re-runs steps 3-8 from scratch from the current pose; no learned or cached state is carried from one frame's render to the next (see Ablation/state note below).

#### Key Hyperparameters & Design Choices

- Cloth LBS regularization weight $\lambda_{\text{cloth-lbs}} = 1000.0$
- Simulation-alignment weight $\lambda_{\text{sim}} = 1.0$
- ARAP weight $\lambda_{\text{ARAP}} = 0.5$
- Mask-consistency weight $\lambda_{\text{mask}} = 1.0$
- Reconstruction loss weights: L1 = 0.8, SSIM = 0.2, LPIPS = 1.0
- Training iterations: 20,000 at 512x512 resolution
- Position learning rate: $1.6\times10^{-4} \rightarrow 1.6\times10^{-6}$ (decayed)
- Rotation / scale / opacity learning rates: $1.0\times10^{-3}$, $5.0\times10^{-3}$, $5.0\times10^{-2}$ (fixed, not decayed)
- Triplane feature resolution: 256x256x32 channels per plane, 3 planes
- MLP architecture (layer count, hidden dims): Not specified in paper
- Number of Gaussians per layer / total: Not specified in paper
- Training time: ~40 minutes on an NVIDIA L40S GPU
- Inference speed: exceeds 60 FPS

#### Ablation Summary

From Table 4 (human-only NeuMan crops; full model PSNR 18.812 / SSIM 0.675 / LPIPS 0.160):
- **w/o Cloth LBS Reg.**: PSNR 18.203 (−0.609 dB vs. full), SSIM 0.642, LPIPS 0.185 — largest single-component drop, i.e., the cloth LBS-weight regularization is the single most impactful ablated component.
- **w/o Physics Losses** (all four removed): PSNR 18.358 (−0.454 dB), SSIM 0.648, LPIPS 0.182 — removing all physics terms together hurts less than removing cloth-LBS alone is large relative to the others, underscoring LBS regularization's outsized share of the physics-loss benefit.
- **ARAP Only** (other physics terms removed): PSNR 18.607 (−0.205 dB), LPIPS 0.168.
- **Simulation Only**: PSNR 18.437 (−0.375 dB), LPIPS 0.168.
- **Mask Only**: PSNR 18.603 (−0.209 dB), LPIPS 0.165.
All four physics components and the depth-aware compositing contribute positively; cloth-LBS regularization is flagged as the most impactful individual term.

#### Implementation Reality

No repository was found in the paper text, Papers With Code, or GitHub search (queries for "Cloth-HUGS" / "CLOTH-HUGS" surfaced only unrelated "HUGS" projects, e.g., apple/ml-hugs and hyzhou404/HUGS). This section is omitted per instructions beyond noting the absence — no implementation details can be verified against code.

#### Failure Modes & Limitations

The accessible HTML version of the paper does not contain an explicit itemized limitations section; no specific failure-case numbers (e.g., degraded scenes or garment types) were found in the fetched text.

---

## Relevance to ADAGS

Closest partial occupant of "occlusion-ordered layered dynamic GS": genuine
ordered Gaussian layers, but fixed 3-layer avatar-specific structure with
no temporal hidden-state persistence. Bounds the layered-GS dead-end claim
in [[operations/sota-sweep-2026-08]]; cite when positioning any
order/memory conjunction claim.

## Connections

- Pressures [[gap_map#G13 - Visibility Events Are Not Smooth Deformation]]

## Sources

- https://arxiv.org/abs/2604.15875
