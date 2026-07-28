---
type: paper
node_id: paper:zhou2026_motionscale
title: "MotionScale: Reconstructing Appearance, Geometry, and Motion of Dynamic Scenes with Scalable 4D Gaussian Splatting"
authors: ["Haoran Zhou", "Gim Hee Lee"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2603.29296"
tags: [dynamic-gs, 4dgs, scalable-motion, geometry, monocular]
status: deep-dived
---

# MotionScale: Reconstructing Appearance, Geometry, and Motion of Dynamic Scenes with Scalable 4D Gaussian Splatting

**Paper:** https://arxiv.org/abs/2603.29296
**Code:** https://github.com/hrzhou2/motion-scale
**Base method:** Monocular 4D Gaussian Splatting with per-Gaussian motion bases, in the lineage of Shape of Motion (used as the paper's main baseline); relies on off-the-shelf priors π³ (depth/pose), SAM2 (masks), and CoTracker3 (2D tracking).

Note: the paper's main text (arXiv HTML) omits most quantitative hyperparameters
and defers them to supplementary material, which was not accessible. The
GitHub repo exists and was reachable, but the fetched config file
(`configs/davis/default.yaml`) contained only paths/data-source settings, not
model hyperparameters. Numeric details below are marked "Not specified" where
the paper/repo did not state them.

## One-line thesis

Grouping dynamic Gaussians into spatially coherent clusters — each governed by one global rigid transform plus a small set of shared local non-rigid basis transforms that Gaussians blend via learned weights — lets a monocular 4DGS system scale to larger scenes and longer sequences without the geometric drift that per-Gaussian or single-global-basis motion models accumulate.

## Problem / Gap

Monocular dynamic reconstruction is supervised almost entirely by view-dependent photometric signals, which under-constrain 3D structure for moving content; methods that lean on 2D tracking priors instead (e.g. Shape of Motion-style approaches) accumulate temporal drift over long sequences, producing geometric collapse and inconsistent trajectories as the sequence grows. A single global motion basis (or one MLP field) also cannot capture spatially localized non-rigid deformation, and naive per-Gaussian motion has no mechanism to stay temporally or spatially consistent as scene size and duration grow.

## Method

MotionScale partitions dynamic Gaussians into K disjoint clusters (initialized via K-means on 3D tracked points in the canonical frame). Each cluster k has a global SE(3) rigid transform per frame and a bank of B local non-rigid basis transforms per frame; every Gaussian in the cluster carries a learned per-Gaussian weight vector that blends the local bases before the cluster's global transform is applied. Reconstruction runs as progressive optimization: a background-extension stage projects existing Gaussians into newly seen frames and seeds unobserved regions from monocular depth while jointly refining camera extrinsics, and a three-stage foreground-propagation pass (one-directional alignment on new frames, bidirectional short-term consistency within a propagation window, then full-sequence long-term refinement) supervises motion with tracking and depth losses. An adaptive control step runs HDBSCAN on 3D trajectories within a cluster and splits clusters whose trajectory divergence exceeds a distance threshold, duplicating parameters into the new cluster. Dedicated "shadow Gaussians" — background primitives that move with a dynamic object but receive only photometric supervision, no geometric/motion loss — absorb effects like shadows or dynamic-object-induced appearance change without corrupting motion learning.

## Assumptions

Monocular RGB video input, non-rigid but locally-coherent dynamic content (clusterable into rigid-plus-local-deformation groups), and availability of off-the-shelf monocular depth/pose (π³), instance/object masks (SAM2), and 2D point tracks (CoTracker3) as supervision priors.

## Limitations / Failure Modes

The paper does not include a dedicated limitations section and the accessible HTML text states no explicit failure cases. This is itself a gap: no quantitative breakdown of where clustering or the propagation windows fail (e.g. thin/textureless structures, tracker dropout, topology changes) is given in the main text — only the ablation table's aggregate deltas hint at which components carry the most weight.

## Reusable Ingredients

- **Cluster-centric motion decomposition** (global rigid transform per cluster + shared local non-rigid bases blended per-Gaussian) — captures fine-grained deformation while keeping per-cluster parameters low, versus one global basis or unconstrained per-Gaussian motion.
- **Adaptive cluster splitting via HDBSCAN on 3D trajectories** — detects motion-inconsistent groups mid-training and grows capacity only where trajectories actually diverge, rather than fixing K up front.
- **Progressive/staged propagation** (one-directional → bidirectional short-term → full-sequence long-term) — a curriculum for introducing new frames that limits how far tracking/depth supervision has to reach before consistency is established.
- **Shadow Gaussians** — a dedicated primitive class that explains photometric-only, motion-correlated appearance effects (e.g. shadows cast by moving foreground) without letting them pollute the geometric/motion loss terms.
- **Photometric background extension with camera refinement** — separates "grow the static/background scaffold into newly seen regions" from "propagate foreground motion," each with its own supervision recipe.

---

### Deep Dive

#### Core Novelty
Relative to prior monocular 4DGS motion models that use either one global deformation field/basis or fully independent per-Gaussian motion, MotionScale's novelty is the two-level motion decomposition (per-cluster global SE(3) + shared local basis bank, blended per-Gaussian) combined with data-driven cluster splitting. The key insight is that motion coherence should be enforced at the granularity of physically-coherent parts (clusters), not the whole scene or single Gaussians — this bounds the degrees of freedom enough to resist drift while still being expressive enough for non-rigid local deformation, and the HDBSCAN-based splitting lets the number of parts grow only where trajectories actually diverge instead of being fixed a priori.

#### Mathematical Formulation

Per-Gaussian position at time $t$ (evaluated per-Gaussian, before rasterization, every frame):

$$\boldsymbol{\mu}_i^t = \mathbf{R}_{k,g}^t\left(\mathbf{R}_{i,\ell}^t \boldsymbol{\mu}_i^0 + \mathbf{t}_{i,\ell}^t\right) + \mathbf{t}_{k,g}^t$$

where for Gaussian $i$ belonging to cluster $k$: $\boldsymbol{\mu}_i^0$ is the canonical (rest-frame) position; $(\mathbf{R}_{i,\ell}^t, \mathbf{t}_{i,\ell}^t)$ is the Gaussian's local non-rigid transform at time $t$, obtained by blending the cluster's $B$ local basis transforms with the Gaussian's learned weight vector $\mathbf{w}_i$; and $(\mathbf{R}_{k,g}^t, \mathbf{t}_{k,g}^t)$ is cluster $k$'s single global rigid (SE(3)) transform at time $t$, shared by every Gaussian in the cluster. Rotations are parameterized with a 6D continuous representation.

Tracking loss (evaluated after projecting Gaussian motion to image space, within the foreground-propagation stages):

$$\mathcal{L}_{\text{track}} = \frac{1}{|I_t|}\sum \left\| \hat{\mathbf{U}}_{t\to t'}(\mathbf{p}) - \mathbf{U}_{t\to t'}(\mathbf{p}) \right\|$$

where $\hat{\mathbf{U}}_{t\to t'}$ is the rendered/predicted 2D flow-like correspondence from frame $t$ to $t'$ and $\mathbf{U}_{t\to t'}$ is the CoTracker3-supplied 2D track target for pixel $\mathbf{p}$.

Depth loss (same evaluation point, paired with the tracking loss):

$$\mathcal{L}_{\text{depth}} = \frac{1}{|I_t|}\sum \left\| \hat{\mathbf{D}}_{t\to t'}(\mathbf{p}) - \mathbf{D}_{t\to t'}(\mathbf{p}) \right\|$$

comparing rendered depth correspondence against the π³-derived depth prior. An as-rigid-as-possible (ARAP) regularization term is also used in the long-term refinement stage to keep local basis deformation physically plausible, but its exact formula and weight are not given in the accessible main text (deferred to supplementary material).

#### Algorithm / Pipeline Changes

1. Run π³ to get monocular depth and camera poses, SAM2 for object/foreground masks, and CoTracker3 for 2D point tracks — all precomputed priors, not learned jointly.
2. Initialize canonical Gaussians and cluster assignment via K-means on 3D tracked points (K clusters); initialize each cluster's global transform via Procrustes analysis between consecutive point clouds; initialize local basis transforms to identity.
3. **Background extension stage**: for each new temporal window, project existing (static/background) Gaussians into the newly added frames, identify regions with no coverage, sample new Gaussians there from the monocular depth prior, and jointly refine camera extrinsics using a photometric loss.
4. **Foreground propagation stage**, run per new batch of $T_{\text{new}}$ frames, in three sequential sub-stages:
   a. Initial alignment: one-directional tracking loss only; optimizes only the new frames' motion-basis parameters (cluster/basis params for earlier frames are frozen).
   b. Short-term consistency: bidirectional tracking loss within a propagation window of $T_{\text{prop}}$ frames.
   c. Long-term refinement: full-sequence supervision (tracking + depth + RGB photometric + ARAP), jointly optimizing all parameters seen so far.
5. **Adaptive control**: after some optimization progress, run HDBSCAN on each cluster's set of per-Gaussian 3D trajectories; if the intra-cluster trajectory divergence exceeds a predefined distance threshold, mark the cluster motion-inconsistent and split it — the new cluster is initialized by duplicating the parent cluster's parameters.
6. **Shadow Gaussians**: a separate primitive population attached to move with dynamic clusters but supervised only by photometric (RGB) loss, with no tracking/depth/ARAP terms — absorbs shadow/appearance artifacts correlated with motion without affecting the geometric motion fit.
7. Standard 3DGS/4DGS rasterization renders each frame from the per-Gaussian time-$t$ position computed in step 1 of the Mathematical Formulation above.

Input/output shapes, exact frame-window sizes ($T_{\text{init}}$, $T_{\text{new}}$, $T_{\text{prop}}$), and K/B values are referenced symbolically in the paper but not numerically specified in the accessible text.

#### Key Hyperparameters & Design Choices

- Number of clusters $K$: Not specified in paper (initialized via K-means; no default count given).
- Number of local basis transforms $B$ per cluster: Not specified in paper.
- $T_{\text{init}}$ (initial temporal window length): Not specified in paper.
- $T_{\text{new}}$ (progressive batch size in frames): Not specified in paper.
- $T_{\text{prop}}$ (bidirectional propagation window): Not specified in paper.
- HDBSCAN distance threshold for triggering cluster splits: Not specified in paper.
- Loss weights ($\lambda$) for tracking, depth, RGB photometric, and ARAP terms: Not specified in paper.
- Learning rates/schedules for cluster global transforms, local basis transforms, and per-Gaussian blend weights: Not specified in paper.
- Rotation parameterization: 6D continuous representation (stated).
- Cluster initialization: K-means on 3D tracked points in canonical frame (stated).
- Global transform initialization: Procrustes analysis between consecutive point clouds (stated).
- Local basis initialization: identity transforms (stated).
- Hardware/framework: single NVIDIA RTX 4090, PyTorch (stated).
- Priors: π³ (depth/pose), SAM2 (masks), CoTracker3 (2D tracking) (stated).

The paper explicitly defers full hyperparameter and training-schedule details to supplementary material, which was not accessible during this review.

#### Ablation Summary

From Table 3 (DyCheck dataset; PSNR/SSIM/LPIPS plus tracking metrics AJ, average-$\delta$, occlusion accuracy OA):

| Configuration | PSNR | Δ vs. full |
|---|---|---|
| Full method | 17.98 | — |
| **Global bases (fixed, no clustering)** | 16.70 | **−1.28 dB** |
| w/o Shadow Gaussians | 16.26 | −1.72 dB (largest PSNR drop, though not a like-for-like ablation of the motion model itself) |
| w/o FG Propagation | 16.97 | −1.01 dB |
| w/o Adaptive Control (no HDBSCAN splitting) | 17.21 | −0.77 dB |
| w/o Pose Refinement | 17.45 | −0.53 dB |

By raw PSNR delta, removing Shadow Gaussians hurts most (−1.72 dB), but among the paper's own framing the cluster-based motion decomposition is called out as the central result: replacing per-cluster local bases with a single fixed global basis set ("Global Bases (fixed)") costs 1.28 dB and is explicitly the comparison the paper uses to argue localized motion bases capture fine-grained deformation better than a global field. Adaptive (HDBSCAN-driven) cluster splitting contributes a further 0.77 dB over a static cluster count.

#### Implementation Reality

- **Framework:** PyTorch with CUDA; builds `gsplat` from source during installation (per `install.sh`).
- **Key files (from repo structure):** `run_training.py` (training entry point), `run_rendering.py` (interactive viewer), `configs/davis/default.yaml` (per-dataset config — only contains directory paths and data-source type flags, not model hyperparameters), `preproc/README.md` (data preprocessing instructions for depth/mask/track priors).
- **Notable implementation details:** The fetched default config for the DAVIS dataset setup contains no numeric hyperparameters (K, B, loss weights, learning rates, thresholds) — these appear to live in code defaults or a separate config not covered by the file fetched during this review. This means the paper's reproducibility currently rests on inspecting the training script directly rather than a documented config; treat any specific numeric value not listed above as unverified until the code is read directly.

#### Failure Modes & Limitations

The paper does not include a dedicated limitations discussion in the accessible text, and no explicit qualitative failure cases (e.g. specific scene types, occlusion duration, or texture conditions that break the method) are stated. The only quantitative signal on where the method is weak is indirect, via the ablation table: removing Shadow Gaussians causes the largest PSNR regression, suggesting shadow/appearance artifacts correlated with dynamic motion are a significant source of error the base motion model cannot otherwise absorb, and that scenes without a clean way to isolate such effects may be harder for the method.

---

## Relevance to ADAGS

ADAGS should avoid claiming that its residual motion alone solves temporal consistency. MotionScale is a natural comparator for whether ADAGS improves local fast motion while preserving global scene geometry.

## Connections

- Addresses [[gap_map#G6 - Single Global Motion Models Are A Known Weakness]]
- Addresses [[gap_map#G7 - A Benchmark/Diagnostic Claim Is Necessary]]

## Sources

- https://arxiv.org/abs/2603.29296
