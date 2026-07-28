---
type: paper
node_id: paper:rai2026_packuv
title: "PackUV: Packed Gaussian UV Maps for 4D Volumetric Video"
authors: ["Aashish Rai", "Angela Xing", "Anushka Agarwal", "Xiaoyan Cong", "Zekun Li", "Tao Lu", "Aayush Prakash", "Srinath Sridhar"]
year: 2026
venue: "CVPR"
external_ids:
  arxiv: "2602.23040"
  doi: null
  s2: null
tags: ["4d-gaussians", "disocclusion", "temporal-consistency", "uv-representation"]
added: 2026-07-14T23:36:29Z
status: deep-dived
---

# PackUV: Packed Gaussian UV Maps for 4D Volumetric Video

**Paper:** https://arxiv.org/abs/2602.23040
**Code:** https://github.com/aashishrai3799/packuv (repo exists but as of the 2026-03 release contains only the paper/project-page README — no implementation published yet)
**Base method:** 3D Gaussian Splatting (Kerbl et al. 2023), directly optimized in UV space rather than fit-then-project like prior UVGS.

## One-line thesis

Optimizing Gaussian attributes directly inside a structured, multi-layer UV atlas (instead of fitting 3DGS per frame and projecting to UV afterward) removes the lossy post-hoc projection step and lets flow-guided keyframing plus per-Gaussian dynamic/static labeling keep long multi-view video sequences temporally coherent while staying compatible with standard video codecs.

## Problem / Gap

Prior UVGS-style representations project an already-optimized 3DGS scene into a UV map post-hoc, which the paper calls "lossy and computationally redundant." Separately, per-frame or short-window 3DGS/4DGS fitting drifts over long sequences and "struggle[s] to operate on videos longer than a few seconds," degrading further under large motion and disocclusion because deformation-style methods have no explicit mechanism for reintroducing surfaces that reappear after being hidden.

## Method

PackUV-GS parameterizes every Gaussian's position by spherical angles mapped to discrete UV coordinates, storing radial distance, rotation, scale, opacity, and color at each UV pixel across up to K stacked layers, with layers packed into a single pyramid-downsampled atlas for codec compatibility. Optical-flow magnitude peaks (with a minimum separation) select keyframes, and frames with high drift, occlusions/disocclusions, or appearance breaks are additionally promoted to keyframes; every non-keyframe initializes its Gaussians from the immediately preceding frame's fitted state. A per-Gaussian dynamic/static label is computed by projecting each Gaussian's 2D covariance ellipse into every camera view and testing overlap (Mahalanobis distance) against a per-view motion mask, OR-ing the result across views; static Gaussians (label 0) have their gradients zeroed so only dynamic regions update. UV-aware density control prunes children whose recomputed UV coordinates are invalid and caps population by keeping only the top-K-opacity Gaussians per UV pixel.

## Assumptions

Requires dense synchronized multi-view video capture (50+ cameras in the paper's own dataset), a reliable off-the-shelf optical-flow estimator to build motion masks and detect keyframe triggers, and a fixed per-scene UV atlas resolution/layer count decided in advance. Child Gaussians inherit their parent's dynamic/static label rather than having it re-derived.

## Limitations / Failure Modes

Keyframe promotion reacts to flow magnitude, occlusion/disocclusion, or appearance breaks — it does not infer explicit foreground/background surface order or persistent hidden-surface identity, so it cannot say *which* surface is occluding *which*. Sequential per-frame initialization from the predecessor can propagate fitting errors forward through a sequence. The ablation shows removing keyframing alone costs 6.46 dB PSNR, the largest drop of any component, indicating heavy reliance on flow-triggered keyframe placement rather than a more principled occlusion model. The representation and capture regime (50+ synchronized cameras, codec-packed storage) is far denser than ADAGS's sparse calibrated N3V setup.

## Reusable Ingredients

- **Flow-magnitude-peak keyframe selection with minimum separation** — cheap, unsupervised way to pick temporal anchor frames without a learned scene-change detector.
- **Multiview-aggregated per-Gaussian dynamic/static mask (OR across views via covariance-ellipse/motion-mask overlap)** — a concrete recipe for turning 2D flow masks into a 3D per-primitive label.
- **Gradient freezing on static-labeled primitives** — budget-neutral way to stop static Gaussians from absorbing dynamic-region gradient noise, directly relevant to ADAGS's static-preservation goal.
- **Direct optimization in a structured target representation instead of fit-then-project** — general lesson that post-hoc projection into any auxiliary structure (UV atlas, or an ADAGS surface ledger) loses information relative to optimizing in that structure from the start.
- **Top-K-per-cell population cap for density control** — a simple, hard capacity ceiling per spatial bucket that could be adapted to a per-surface-slot budget.

---

### Deep Dive

#### Core Novelty

Relative to prior UVGS (post-hoc projection of a finished 3DGS into a UV map), PackUV-GS optimizes Gaussian attributes *inside* the UV atlas from the start of training, for every frame of a long sequence, rather than treating UV-packing as an export step. The key insight is that projection-after-fitting is lossy and that a fixed structured layout (fixed resolution, fixed K layers, pyramid-downsampled and packed into one atlas) can instead directly host the optimization variables, which simultaneously buys temporal coherence (via keyframe-anchored sequential initialization) and video-codec compatibility (via a stable per-pixel/per-layer array structure).

#### Mathematical Formulation

UV coordinate mapping (Eq. 1), evaluated once per Gaussian to place it in the atlas before optimization/rasterization:
$$u_i = \left\lfloor \frac{\pi + \theta_i}{2\pi} \times M \right\rfloor, \quad v_i = \left\lfloor \frac{\phi_i}{\pi} \times N \right\rfloor$$
where $\theta_i, \phi_i$ are the azimuthal and polar angles of Gaussian $i$'s position in spherical coordinates, and $M, N$ are the atlas's width/height at the current pyramid level.

UV storage (Eq. 2), the per-cell data layout the optimizer writes to:
$$U[u_i, v_i, k] = g_i = \{\rho_i, r_i, s_i, o_i, c_i\} \in \mathbb{R}^D$$
storing radial distance $\rho_i$, rotation $r_i$, scale $s_i$, opacity $o_i$, and color $c_i$ for the Gaussian occupying layer $k$ at that UV cell.

Pyramid downsampling across layers (halves width or height on alternating layers), used when packing all $K$ layers into one atlas:
$$(M_k, N_k) = \begin{cases} (M_0, N_0) & k = 0 \\ (M_{k-1}, N_{k-1}/2) & k \text{ odd} \\ (M_{k-1}/2, N_{k-1}) & k \text{ even} \end{cases}$$

2D covariance projection (Eq. 3), evaluated per Gaussian per camera view during dynamic/static labeling:
$$\Sigma^{3D}_{i,\text{cam}} = T_c \Sigma^{3D}_i T_c^T, \qquad \Sigma^{2D}_{i,c} = J_c \Sigma^{3D}_{i,\text{cam}} J_c^T$$
where $T_c$ is the camera extrinsic transform and $J_c$ the projection Jacobian for camera $c$; this gives the projected 2D ellipse used to test overlap with that camera's motion mask.

A pixel $\mathbf{p}$ is inside a Gaussian's projected ellipse when its Mahalanobis distance to the ellipse satisfies $d^2(\mathbf{p}; \mathbf{m}_{i,c}, \Sigma^{2D}_{i,c}) \le 9$ (a fixed 3-sigma-equivalent cutoff), used to decide which pixels of view $c$'s optical-flow motion mask $M^c_t$ "belong" to Gaussian $i$.

Dynamic mask aggregation (Eq. 4), computed per Gaussian after per-view overlap testing, before gradient computation:
$$D_{i,c} = \bigvee_{p \in E_{i,c}} M^c_t(p), \qquad D_i = \bigvee_{c \in C} D_{i,c}$$
where $E_{i,c}$ is the set of pixels inside Gaussian $i$'s ellipse in view $c$; $D_i \in \{0,1\}$ is the final per-Gaussian dynamic label, ORed across all cameras $C$.

Gradient freezing, applied at each backward pass:
$$\nabla_{\theta_i} L \leftarrow D_i \cdot \nabla_{\theta_i} L$$
zeroing all attribute gradients for static ($D_i = 0$) Gaussians so only dynamic-labeled primitives update.

Auxiliary regularizers added to the total loss: scale regularization (Eq. 5) $L_{scale} = \mathbb{E}_i[\max\{0, \max(s_i) - s_{max}\}]^2$ and opacity regularization (Eq. 6) $L_{opacity} = \mathbb{E}_i[\alpha_i(1-\alpha_i)]$, combined as $L = L_{photo} + L_{depth} + \lambda_{scale} L_{scale} + \lambda_{opacity} L_{opacity}$, with $L_{photo} = (1-\lambda_{ssim})\lVert \hat I^c_t - I^c_t \rVert_1 + \lambda_{ssim}(1 - \mathrm{SSIM}(\hat I^c_t, I^c_t))$.

#### Algorithm / Pipeline Changes

1. Convert each Gaussian's 3D position to spherical angles and discretize to a UV cell (Eq. 1); assign it to one of $K$ atlas layers at that cell (Eq. 2). Runs once at Gaussian creation/re-densification, before optimization.
2. Build the multi-scale pyramid atlas by halving width or height on alternating layers and packing all $K$ layers into a single quadtree-style image (reported 88.5% packing efficiency) — this is the storage/streaming stage, decoupled from optimization.
3. Select keyframes: compute optical-flow magnitude $M(t)$ per video, take the top $m{-}1$ magnitude peaks with a minimum separation $\theta$, and use the first frame of each resulting segment as a keyframe; additionally promote any frame with high drift, occlusion/disocclusion, or an appearance break.
4. Initialize each frame's Gaussians from its temporal predecessor's fitted state: $\mathcal{G}(t) \leftarrow \mathrm{Update}(\mathcal{G}(t-1))$ for both keyframes and transition frames — replaces from-scratch or purely-deformation-field initialization used by prior per-frame/warp methods.
5. For dynamic/static labeling, project each Gaussian's 3D covariance into every camera view (Eq. 3), test which pixels fall inside its ellipse via the Mahalanobis cutoff, OR that against each view's flow-derived motion mask, then OR across all views to get one binary label per Gaussian (Eq. 4). Runs each optimization step (or on a schedule) before the backward pass.
6. Zero gradients for static-labeled Gaussians so only dynamic Gaussians receive attribute updates that step.
7. Density control: after densification (clone/split assumed standard 3DGS, not detailed further), recompute each child's UV coordinates and prune any that fail the valid-UV-projection test (Eq. 1 mapping); separately, at each UV pixel retain only the top-$K$ Gaussians ranked by opacity, pruning the rest ("Max-K UV Pruning") to bound per-cell population.

#### Key Hyperparameters & Design Choices

- Base atlas resolution: $M_0 = N_0 = 1024$.
- UV layers per pixel: $K = 8$.
- Keyframe minimum separation $\theta$: reported as 30 (units/exact definition of the flow-magnitude-peak separation not further specified in the excerpt).
- Loss weights: $\lambda_{scale} = \lambda_{opacity} = 0.0001$.
- Quantization: 8-bit for $\{s, r, \alpha, c\}$; position uses 16-bit split across two 8-bit channels.
- Mahalanobis overlap cutoff: $d^2 \le 9$.
- Optical-flow magnitude threshold $\tau$ and mask dilation radius $r$: not specified (paper mentions them without numeric values in the accessible excerpt).
- Optimizer settings, learning-rate schedule, warmup: not specified in paper.

#### Ablation Summary

From Table 3 (PackUV-2B sequence), relative to full method (PSNR 27.41 / SSIM 0.84 / LPIPS 0.28):
- **w/o Keyframe: 20.95 dB (−6.46 dB)** — largest single drop; keyframing is the most impactful component.
- w/o UV Optim (i.e., post-hoc projection instead of direct UV-space optimization): 23.81 dB (−3.60 dB).
- w/o Labeling (no dynamic/static split): 25.42 dB (−1.99 dB).
- No Atlas (layers not packed into one atlas): 27.43 dB (+0.02 dB) — negligible cost from packing.
- w/o Codec (no FFV1-style compression): 27.41 dB (0.00 dB) — no quality cost from codec compatibility.
- No LPO: 27.52 dB (+0.11 dB) — negligible effect.

#### Implementation Reality

- **Framework:** Not available — the public repo (github.com/aashishrai3799/packuv, 63 stars, 2 commits as of the March 2026 release) currently contains only a README pointing to the paper and project page; no source code has been published yet.
- **Key files:** None published.
- **Notable implementation details:** None available; cannot compare paper claims to code.

#### Failure Modes & Limitations

Prior deformation-field methods "struggle to operate on videos longer than a few seconds" and "struggle with large motions and disocclusions" — the gap this paper targets. Existing multi-view video datasets are "largely restricted to frontal cameras" with limited motion/disocclusion coverage, motivating the paper's own 50+ camera, 100-sequence, 2-billion-frame PackUV-2B dataset. A dedicated "Limitations and Future Work" section (Section 15) is referenced in the paper's structure but its content was not retrievable from the accessible HTML excerpt.

## Relevance to This Project

PackUV-GS prevents any broad claim that temporal consistency or disocclusion handling is absent from Gaussian representations. ADAGS must test a narrower surface-order-to-budget-reassignment hypothesis.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Sources

- arXiv: https://arxiv.org/abs/2602.23040
- Project page: https://ivl.cs.brown.edu/packuv/
- Code (README-only as of 2026-03): https://github.com/aashishrai3799/packuv
