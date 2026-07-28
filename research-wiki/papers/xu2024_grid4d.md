---
type: paper
node_id: paper:xu2024_grid4d
title: "Grid4D: 4D Decomposed Hash Encoding for High-fidelity Dynamic Gaussian Splatting"
authors: ["Zhen Xu", "Jiabo Ye", "Yuanbo Xiangli", "Xiaopeng Zhang", "Lingxi Xie"]
year: 2024
venue: "NeurIPS"
external_ids:
  arxiv: "2410.20815"
tags: [dynamic-gs, hash-encoding, high-fidelity]
status: deep-dived
---

# Grid4D: 4D Decomposed Hash Encoding for High-fidelity Dynamic Gaussian Splatting

**Paper:** https://arxiv.org/abs/2410.20815
**Code:** https://github.com/JiaweiXu8/Grid4D
**Base method:** 3D Gaussian Splatting + deformation-field dynamic extension (Deformable-3D-Gaussians / D-NeRF-style canonical-space warp), with multiresolution hash encoding (Instant-NGP style) replacing the plane-decomposed encodings of 4D-GS/HexPlane/K-Planes.

## One-line thesis

Replacing the six shared 2D feature planes of prior 4D-GS deformation encoders with one 3D spatial hash grid plus three 3D spatio-temporal hash grids, fused by a learned directional-attention gate, removes the low-rank plane-decomposition assumption that causes feature collisions between Gaussians sharing coordinate pairs, giving sharper deformation prediction at similar or lower memory cost.

## Problem / Gap

Plane-based dynamic GS methods (4D-GS, HexPlane, K-Planes) encode a 4D (x,y,z,t) query by summing/multiplying features looked up from six 2D planes (xy, xz, yz, xt, yt, zt). This is a low-rank factorization of the 4D feature volume: any two Gaussians that share a coordinate pair (e.g. identical y,z) collide and receive identical features on the (y,z), (y,t), and (z,t) planes, corrupting their predicted deformation. Fully implicit MLP deformation fields (DeformGS/Deformable-3D-Gaussians) avoid the collision problem but are "over-smooth" and fail on detailed, complex, or fast-moving content because a single continuous MLP cannot represent high-frequency spatial-temporal variation well.

## Method

Canonical-space 3D Gaussians (position μ, scale S, rotation R, opacity σ, SH color c) are initialized as in standard 3DGS/Deformable-3DGS. For a query Gaussian center at time t, a single 3D hash grid encodes spatial features from (x,y,z), and three separate 3D hash grids encode spatio-temporal features from (x,y,t), (y,z,t), (x,z,t) — this reduces the encoding's spatial complexity from the naive O(n⁴) 4D grid to O(n³) per grid. The spatial features are passed through a small MLP/sigmoid to produce a directional attention score in [-1, 1] (not the usual [0,1]), which gates (element-wise multiplies) the aggregated temporal features before decoding. A shared decoder MLP maps the fused feature to a rigid rotation/translation and residual scale/rotation offset, which is applied to the canonical Gaussian to get its deformed state at time t. A smooth regularization loss penalizes large feature differences between a query point and a small random spatio-temporal perturbation of it, discouraging chaotic deformation predictions.

## Assumptions

Assumes a canonical-space deformation-field formulation (single rest-state Gaussian set warped per frame, à la Deformable-3D-Gaussians/D-NeRF), i.e. topology-stable, non-appearing/disappearing dynamic content, evaluated on monocular (D-NeRF synthetic, HyperNeRF) and multi-view (Neu3D/N3V-style) captures with roughly known camera poses.

## Limitations / Failure Modes

The authors report no training-speed improvement over DeformGS/Deformable-3D-Gaussians — the smooth regularization requires encoding each sample twice, and higher-fidelity deformation prediction leads to more Gaussians (more render/optimization cost). They show explicit qualitative failure cases ("artifacts") on scenes with large and complex motions (paper's Figure 9). On HyperNeRF/Neu3D, imprecise estimated camera poses degrade quantitative metrics even where renders look visually sharper, and some individual HyperNeRF scenes score lower PSNR than the 4D-GS baseline despite improved qualitative detail.

## Reusable Ingredients

- **Decomposed 4D hash encoding (1 spatial 3D grid + 3 spatio-temporal 3D grids)** — captures full 4D variation without the plane-decomposition low-rank collision problem, at O(n³) rather than O(n⁴) cost.
- **Directional attention gate in [-1, 1]** — lets a fused feature flip sign/cancel rather than only scale down, so opposing deformation contributions from different temporal grids can be represented; a plain [0,1] sigmoid gate measurably underperforms in their ablation.
- **Neighborhood smooth regularization on hash features** — penalizes feature divergence under small random (x,y,z,t) perturbations, checkpoint-cheap way to suppress non-smooth deformation without adding a rendering-loss term.

---

### Deep Dive

#### Core Novelty
Grid4D keeps the 3DGS + canonical-space deformation-MLP pipeline of Deformable-3D-Gaussians but swaps its (plane-based or single-MLP) 4D positional encoder for a decomposed hash-grid encoder: 1 spatial 3D hash grid + 3 spatio-temporal 3D hash grids, fused via a learned directional-attention gate instead of naive concatenation/multiplication. The key insight is that 2D-plane decompositions implicitly assume the 4D deformation field is low-rank/separable along each axis pair, which is false whenever two different Gaussians share a coordinate pair; moving to 3D grids removes one axis of ambiguity per grid (three coordinates must all match to collide, not two), and the attention gate lets the network weight/cancel spatial vs. temporal contributions per query rather than always summing them.

#### Mathematical Formulation
- **Multiresolution hash encoding (per grid), Eq. 3-4:** grid resolution at level $l$ follows a geometric progression $N_l = \lfloor N_{\min} \cdot b^l \rfloor$, and each level's per-voxel-corner index is computed by hashing $h_l(\mathbf{x}_l) = \left(\bigoplus_i x_i \cdot \pi_i\right) \bmod T_l$, where $\pi_i$ are large fixed primes used for the spatial hash (standard Instant-NGP hash function) and $T_l$ is the hash table size at level $l$. Evaluated once per query point per grid (spatial grid on (x,y,z); three temporal grids on (x,y,t), (y,z,t), (x,z,t)), before the decoder.
- **Directional attention gate, Eq. 5-6:**
  $$\mathbf{a} = 2\cdot\Phi(h_{xyz}) - 1, \qquad \mathbf{h} = \mathbf{a} \odot f_t(G_{xyt}, G_{yzt}, G_{xzt})$$
  where $h_{xyz} = f_s \circ G_{xyz}(x,y,z)$ is the spatial feature (after a small learned projection $f_s$), $\Phi$ is a sigmoid, and $f_t$ aggregates the three temporal hash-grid outputs. The $2\Phi(\cdot)-1$ form maps the gate to $[-1,1]$ rather than the conventional $[0,1]$, so temporal features can be sign-flipped, not just attenuated. Evaluated per-Gaussian, per-frame, right before the deformation decoder.
- **Deformation application (post-decode):** decoder MLP $D(\mathbf{h}) \to \{R_x, T_x, \Delta r, \Delta s\}$; canonical Gaussian attributes are updated as $\mu' = R_x \mu + T_x$, $S' = S + \Delta s$, $R' = R + \Delta r$. This happens per-Gaussian, per-frame, before rasterization, replacing the deformation-MLP output stage of Deformable-3D-Gaussians.
- **Smooth regularization loss, Eq. 8:**
  $$\mathcal{L}_r = \left\| G_{xyzt}(x,y,z,t) - G_{xyzt}(x+\epsilon_x, y+\epsilon_y, z+\epsilon_z, t+\epsilon_t) \right\|_2^2$$
  with small random perturbations $\epsilon$; encourages local smoothness of the combined encoding. Computed as an auxiliary loss term alongside the rendering loss (requires a second encoder forward pass on the perturbed sample).
- **Total loss, Eq. 9:** $\mathcal{L} = (1-\lambda_c)\mathcal{L}_1 + \lambda_c \mathcal{L}_{\text{D-SSIM}} + \lambda_r \mathcal{L}_r$.

#### Algorithm / Pipeline Changes
1. Initialize canonical 3D Gaussians (position/scale/rotation/opacity/SH) as in vanilla 3DGS, using COLMAP points (or `points.npy` for HyperNeRF scenes).
2. For each Gaussian at render time $t$, query the spatial hash grid $G_{xyz}(x,y,z)$ and project through $f_s$ to get $h_{xyz}$; this replaces the "spatial branch" of a plane-based or MLP-based deformation encoder.
3. Query the three spatio-temporal hash grids $G_{xyt}, G_{yzt}, G_{xzt}$ and aggregate with $f_t$ to get the raw temporal feature.
4. Compute the directional attention gate from $h_{xyz}$ (Eq. 5) and apply it element-wise to the temporal feature (Eq. 6) to get the fused feature $\mathbf{h}$.
5. Feed $\mathbf{h}$ through a shared multi-head decoder MLP $D$ that outputs rigid rotation/translation $\{R_x, T_x\}$ and residual scale/rotation offsets $\{\Delta s, \Delta r\}$.
6. Apply the deformation to canonical Gaussian attributes ($\mu' = R_x\mu + T_x$, $S'=S+\Delta s$, $R'=R+\Delta r$) before rasterization — this is a drop-in replacement for the deformation-application stage in Deformable-3D-Gaussians, only the encoder/decoder feeding it changes.
7. During training, additionally sample a small random spatio-temporal perturbation of each query point, re-run the encoder, and add the smooth-regularization loss $\mathcal{L}_r$ (Eq. 8) to the standard photometric loss.

#### Key Hyperparameters & Design Choices
- Spatial hash grid resolution: 16 → 2048 across 16 levels.
- Temporal (spatio-temporal) hash grid levels: up to $L=32$.
- Hash table size cap: $2^{19}$ entries per level.
- Feature dimension per hash-grid voxel: 2.
- Spatial feature projection $f_s$ and temporal aggregation $f_t$: each a single fully-connected layer + activation.
- Decoder MLP depth: 0 hidden layers (D-NeRF/synthetic setting) vs. 2 hidden layers (HyperNeRF/real-world setting); width 256.
- Loss weights: $\lambda_c = 0.2$ (D-SSIM weight), $\lambda_r = 0.5$ (smooth-regularization weight).
- Optimizer: Adam, $\beta=(0.9, 0.999)$.
- Time-axis grid resolution set to roughly "between a half and a quarter" of the number of time samples (exact formula not specified in paper).
- Not specified in paper: learning rate values/schedule for the hash-grid parameters and decoder MLP, exact warmup iteration counts, densification interaction with the deformation field.

#### Ablation Summary
D-NeRF dataset ablations (full model = 42.00 dB PSNR):
- **w/o decomposition** (collapse back toward plane/shared encoding): 28.45 dB PSNR (**−13.55 dB — by far the most impactful component**), 0.949 SSIM, 0.055 LPIPS.
- w/o smooth regularization ($\mathcal{L}_r$): 39.47 dB PSNR (−2.53 dB), 0.991 SSIM, 0.012 LPIPS.
- w/o attention gate (presumably plain concatenation/sum of spatial+temporal features): 41.37 dB PSNR (−0.63 dB), 0.993 SSIM, 0.009 LPIPS.
- w/o directional range (gate restricted to conventional [0,1] instead of [-1,1]): 41.32 dB PSNR (−0.68 dB), 0.993 SSIM, 0.009 LPIPS.
- Full Grid4D: 42.00 dB PSNR, 0.994 SSIM, 0.008 LPIPS.
- Reported mean improvement over 4D-GS baseline (across evaluated datasets): +6.59 dB PSNR, +0.009 SSIM, −0.013 LPIPS.
The 4D decomposition itself (vs. a lower-rank/plane-style shared encoding) dwarfs every other component's contribution.

#### Implementation Reality
- **Framework:** PyTorch; repo states it is "adapted from Gaussian Splatting, Deformable-3D-Gaussians," with the hash-encoder implementation "heavily based on ObjectSDF++" (a custom CUDA/PyTorch multiresolution hash-encoding implementation, not tiny-cuda-nn directly).
- **Key files:** `/hashencoder/` (4D decomposed hash encoding — spatial + 3 spatio-temporal grids); `/gaussian_renderer/` (rendering pipeline integration); `train.py` (training orchestration); per-scene config files under `./arguments/<dataset>/<scene>.py`.
- **Notable implementation details:** Gaussian initialization uses `points.npy` for HyperNeRF scenes and a downsampled COLMAP point cloud (`points3D_downsample2.ply`) for Neu3D-style multi-view scenes — i.e. different initialization sources per dataset family, not stated as a unified procedure in the paper. Exact MLP depths/widths per dataset, densification schedule, and learning-rate schedule are configuration-driven per scene rather than fixed constants, and are not fully exposed without reading the individual argument files.

#### Failure Modes & Limitations
No training-speed gain over DeformGS/Deformable-3D-Gaussians is reported despite the quality gain — the double encoder pass for smooth regularization and the tendency toward more Gaussians both add cost. The paper shows explicit qualitative artifacts on scenes with large, complex motion (Figure 9). On HyperNeRF and Neu3D, imprecise estimated camera poses are cited as degrading quantitative scores even when renders are visually sharper, and some individual HyperNeRF scenes underperform the 4D-GS baseline on PSNR despite better perceptual quality.

## Relevance to ADAGS

Competitor if ADAGS claims LoRA/scaffold representation quality.

## Connections

## Sources

- https://arxiv.org/abs/2410.20815
- https://jiaweixu8.github.io/Grid4D-web/
- https://github.com/JiaweiXu8/Grid4D
