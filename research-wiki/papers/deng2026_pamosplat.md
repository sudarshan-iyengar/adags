---
type: paper
node_id: paper:deng2026_pamosplat
title: "PaMoSplat: Part-Aware Motion-Guided Gaussian Splatting for Dynamic Scene Reconstruction"
authors: ["Yinan Deng", "Jianyu Dou", "Jiahui Wang", "Jingyu Zhao", "Yi Yang", "Yufeng Yue"]
year: 2026
venue: "IEEE TCSVT"
external_ids:
  arxiv: "2605.10307"
  doi: "10.1109/TCSVT.2026.3691475"
tags: [dynamic-gs, part-aware, flow-guidance, motion]
status: deep-dived
---

# PaMoSplat: Part-Aware Motion-Guided Gaussian Splatting for Dynamic Scene Reconstruction

**Paper:** https://arxiv.org/abs/2605.10307
**Code:** https://github.com/BIT-DYN/pamosplat
**Base method:** 3D Gaussian Splatting (Kerbl et al. 2023) + Dynamic 3D Gaussians / D-3DGS (Luiten et al. 2023) as the per-timestep deformation baseline it is compared against and structurally extends; uses SAM for 2D segmentation and RAFT for optical flow as frozen off-the-shelf priors.

## One-line thesis

Multi-view 2D segmentation masks, lifted into 3D via co-visibility graph clustering, define rigid "parts" that each get one 6-DoF pose per timestep; optical flow supervises a differential-evolution search that warm-starts each part's rigid motion before per-Gaussian gradient refinement, replacing independent per-Gaussian motion estimation with a part-level structural prior.

## Problem / Gap

Prior per-timestep dynamic 3DGS methods (e.g., D-3DGS/Luiten et al. 2023) optimize each Gaussian's translation/rotation independently frame-to-frame with only a local rigidity regularizer, so there is no structural unit larger than a single Gaussian to constrain motion. Under large or fast inter-frame displacement (e.g., a tennis swing, a thrown ball) this independent optimization falls into poor local minima, producing high tracking error and blurred/duplicated geometry, since there is no coarse initialization step and no shared identity across the Gaussians that belong to one rigid or quasi-rigid object part.

## Method

At t=0, standard 3DGS Gaussians are initialized with an added per-Gaussian discrete part ID; multi-view 2D masks are lifted to 3D Gaussians via depth-guided pixel-Gaussian correspondence (0.1 m threshold), and a co-visibility graph over Gaussians (edge weight = cross-view mask co-visibility ratio) is partitioned with the Louvain community-detection algorithm to yield part clusters. At each subsequent timestep, RAFT computes forward/backward multi-view optical flow, and for each part a differential-evolution (DE) optimizer searches a 6-DoF rigid transform (bounded ±0.2 m translation, ±20° rotation) that best explains the observed flow under the part's projected mask, seeded by a part-inertia prior (translation = previous frame's velocity; rotation = SVD-based relative rotation between the last two frames' anchor-point correlation matrices). This DE-estimated pose warm-starts per-Gaussian gradient-based refinement for an adaptive number of iterations (scaled by how many pixels the part covers), trained with an image loss, a flow-supervised photometric loss weighted by flow magnitude, and a learnable per-part internal rigidity loss that only penalizes anchor-distance drift once it is stable enough to trust (asymmetric growth/decay rates).

## Assumptions

Requires synchronized multi-view RGB video (5-27 cameras across the paper's four benchmarks) with known camera intrinsics/extrinsics, and assumes the full set of scene parts/objects is present and segmentable at the initial timestamp — motion is estimated per-part across time from a fixed part decomposition established once at t=0.

## Limitations / Failure Modes

The paper states the method cannot handle monocular capture (multi-view is required for both the mask-lifting initialization and per-view flow supervision) and cannot model objects that appear only after the initial timestamp, since parts are only created once at t=0. Per-timestep sequential training introduces temporal dependency (errors/drift can propagate forward across timesteps). Far-field regions and complex lighting degrade rendering quality (paper's Figure 19). The ablation shows that removing the DE warm-start is the single most damaging change (tracking accuracy drops from 100% to 81.78% and MTE roughly triples on the Tennis scene), and removing the local/anchor rigidity loss is catastrophic for tracking (MTE 33.56 cm, tracking accuracy 19.56%), indicating the method is fragile without both the DE initialization and the rigidity regularizer.

## Reusable Ingredients

- **Co-visibility graph clustering (Louvain) over lifted 2D masks** — turns noisy multi-view 2D segmentation into a single coherent 3D part decomposition without requiring any single view to be complete.
- **Differential-evolution warm-start from optical flow** — a gradient-free global search over a small bounded 6-DoF space that avoids the bad local minima gradient descent falls into for large inter-frame motion, before handing off to gradient-based refinement.
- **Part-inertia prior (constant-velocity translation + SVD relative-rotation)** — a cheap, model-free per-part motion prior that seeds and constrains the search space each frame.
- **Adaptive per-part iteration budget scaled by pixel coverage** — allocates more optimization steps to parts that occupy more of the image, an implicit visibility/salience-aware compute allocation.
- **Asymmetric learnable rigidity loss (fast growth, slow decay of the trust weight)** — lets a rigidity constraint self-calibrate its own confidence over training instead of using a fixed regularization weight.

---

### Deep Dive

#### Core Novelty

Relative to per-Gaussian independent motion optimization (D-3DGS), PaMoSplat inserts an explicit intermediate structural layer — the "part" — with its own discrete identity, its own rigid pose per timestep, and its own optimization loop (DE search seeded by inertia) before any per-Gaussian gradient update happens. The key insight is that treating flow-consistent, co-visible clusters of Gaussians as one rigid unit turns an ill-conditioned per-Gaussian search (many parameters, sparse per-Gaussian flow signal) into a well-conditioned low-dimensional search (6 DoF per part, dense aggregated flow signal), which is tractable for a population-based global optimizer (DE) rather than only gradient descent.

#### Mathematical Formulation

- **Part motion estimation (per part $p_i$, per timestep $t$)**, solved by differential evolution before gradient refinement:
$$
\min_{\Delta^{p_i}, \Omega^{p_i}} \sum_{v} \left\| \hat{\mathcal{O}}_{v,t}\!\left[m^{p_i}_{v,t-1}\right] - \mathcal{O}_{v,t}\!\left[m^{p_i}_{v,t-1}\right] \right\|
$$
  where $\Delta^{p_i}$/$\Omega^{p_i}$ are the candidate translation/rotation of part $p_i$, $\mathcal{O}_{v,t}$ is the RAFT-observed optical flow at view $v$/time $t$, $\hat{\mathcal{O}}_{v,t}$ is the flow implied by projecting the candidate-transformed part, and $m^{p_i}_{v,t-1}$ is the part's 2D mask in view $v$ at the previous frame. Evaluated once per timestep per part, before any per-Gaussian gradient step.

- **Part inertia prior**, used both to seed the DE population and to bias the search:
$$
\Theta^{p_i}_{\text{tran}} = \mathcal{C}^{p_i}_{t-1} - \mathcal{C}^{p_i}_{t-2}, \qquad \Theta^{p_i}_{\text{rota}} = UV^{T} \text{ where } U\Sigma V^{T} = \mathrm{SVD}\!\left(O^{T}_{t-2}O_{t-1}\right)
$$
  $\mathcal{C}^{p_i}_t$ is the part centroid at time $t$; $O_t$ is the matrix of anchor-point offsets from centroid at time $t$. Translation is a constant-velocity extrapolation; rotation is the closest rigid rotation (via SVD/Procrustes) aligning the previous two frames' anchor configurations.

- **Adaptive iteration count**, sets per-part gradient-refinement steps as a function of visible pixel footprint:
$$
\Upsilon_t = \mathrm{clip}(\epsilon \cdot d_{\text{pixel}},\ \Upsilon_{\min},\ \Upsilon_{\max}), \quad \epsilon = 10^5,\ \Upsilon_{\min}=1500,\ \Upsilon_{\max}=2000
$$
  $d_{\text{pixel}}$ is presumably a normalized pixel-coverage fraction for the part (exact definition/units not fully specified beyond the scaling constant). Evaluated once per part per timestep, before gradient refinement begins.

- **Flow-supervised photometric loss** (per view, per timestep), weights the image loss by where flow is large, i.e. focuses supervision on moving regions:
$$
\mathcal{L}_{\mathcal{O}} = \left| (\hat{\mathcal{I}}_{v,t} - \mathcal{I}_{v,t}) \cdot \mathrm{Norm}\!\left(\|\mathcal{O}_{v,t}\| + \|\mathcal{O}_{v,t}^{-}\|\right) \right|
$$
  $\hat{\mathcal{I}}_{v,t}$/$\mathcal{I}_{v,t}$ are rendered/observed images, $\mathcal{O}_{v,t}$/$\mathcal{O}_{v,t}^{-}$ are forward/backward flow. Evaluated as a per-pixel render loss term each training iteration.

- **Learnable internal rigidity loss**, per part, with a self-calibrating trust weight $\mathcal{W}_t$:
$$
\mathcal{L}_{\text{part-rigid}} = \mathcal{W}_t \left| D_t - D_{t-1} \right|
$$
  $D_t$ is the (anchor-pair) distance structure of the part's Gaussians at time $t$; $\mathcal{W}_t$ grows at rate $\alpha=0.02$ and decays at rate $\beta=0.2$ ($\alpha < \beta$) as a function of observed distance stability (threshold $\delta=10^{-3}$ m), so the constraint is only trusted once measured internal distances have proven stable, and is quickly relaxed again if they destabilize.

- **Combined training objective**:
$$
\mathcal{L}_{\text{Img}} = \lambda_c \mathcal{L}_1 + \lambda_s \mathcal{L}_D + \lambda_o \mathcal{L}_{\mathcal{O}}
$$
  ($\mathcal{L}_D$ is presumably the local/anchor rigidity term ablated as "w/o $\mathcal{L}_{\text{loc-rigid}}$"; loss weights not specified in the extracted text.)

#### Algorithm / Pipeline Changes

1. **t=0 only:** Standard 3DGS initialization, adding a discrete part-ID attribute $p_i \in \mathbb{Z}$ to every Gaussian (alongside center $\mu$, quaternion $q$, RGB $c$, scale $s$, opacity $o$).
2. **t=0 only:** Depth-guided pixel-Gaussian correspondence (0.1 m depth threshold) lifts multi-view 2D masks (from SAM) to per-Gaussian mask membership.
3. **t=0 only:** Build a Gaussian co-visibility graph with edge weights = cross-view mask co-visibility ratio; run Louvain community detection to assign final part IDs. This fixes the part decomposition for the whole sequence.
4. **Per timestep t>0, per part:** Run RAFT to get forward/backward flow per view; run the part-inertia prior to get a seed pose; run differential evolution (population 2, max 25 generations) over the bounded 6-DoF space, scored by the flow-matching objective above, to get a warm-start rigid pose for the part.
5. **Per timestep t>0, per part:** Compute the adaptive iteration budget $\Upsilon_t$ from the part's pixel footprint.
6. **Per timestep t>0:** Run standard 3DGS gradient-based optimization for $\Upsilon_t$ iterations per part, initialized from the DE pose, using the combined loss ($\mathcal{L}_1$ + rigidity $\mathcal{L}_D$ + flow-weighted $\mathcal{L}_{\mathcal{O}}$ + learnable $\mathcal{L}_{\text{part-rigid}}$), refining individual per-Gaussian attributes within the part.
7. Downstream (not a training-time step): part IDs plus optional TAP-caption/SBERT-embedding association enable language-queryable part-level 4D editing.

#### Key Hyperparameters & Design Choices

- Depth-guided correspondence threshold: 0.1 m.
- DE population size: 2; max generations: 25.
- DE search bounds: translation ±0.2 m, rotation ±20°.
- Adaptive iteration count: $\epsilon = 10^5$, $\Upsilon_{\min}=1500$, $\Upsilon_{\max}=2000$.
- Rigidity stability threshold $\delta = 10^{-3}$ m; rigidity-weight growth rate $\alpha=0.02$; decay rate $\beta=0.2$ (constraint $\alpha<\beta$).
- Loss weights $\lambda_c, \lambda_s, \lambda_o$: not specified in the extracted text.
- Part-clustering algorithm: Louvain (community detection), no resolution parameter reported.
- Segmentation backbone: SAM (paper reports SAM2 substitution has minimal impact).
- Flow backbone: RAFT (paper reports SEA-RAFT substitution changes results by <1%).

#### Ablation Summary

Tennis scene, full vs. component removed (PSNR / SSIM / LPIPS / MTE / tracking accuracy):

- **Full model:** 29.53 dB / 0.915 / 0.108 / 1.94 cm / 100.00%
- w/o $\mathcal{L}_{\text{loc-rigid}}$ (local/anchor rigidity): 29.22 dB / 0.918 / 0.103 / **33.56 cm** / **19.56%** — by far the largest degradation; this is the single most impactful component.
- w/o DE (differential-evolution warm start): 28.64 dB / 0.914 / 0.110 / 5.47 cm / 81.78% — second-largest impact, confirms the flow-guided global search matters most for tracking, less for photometric quality.
- w/o Inertia: 29.34 dB / 0.914 / 0.108 / 2.68 cm / 98.66%.
- w/o $\mathcal{L}_{\mathcal{O}}$ (flow-supervised loss): 29.12 dB / 0.915 / 0.108 / 2.13 cm / 100.00%.
- w/o $\mathcal{L}_{\text{part-rigid}}$ (learnable rigidity): 29.51 dB / 0.918 / 0.100 / 1.98 cm / 97.15%.

The local/anchor rigidity loss ($\mathcal{L}_{\text{loc-rigid}}$) is the dominant contributor — removing it causes near-total tracking failure — while PSNR/SSIM/LPIPS are comparatively insensitive to any single ablated component, showing that tracking accuracy (not photometric quality) is where this method's mechanisms do their work.

#### Implementation Reality

- **Framework:** PyTorch, extends a 3DGS codebase with a custom `diff-gaussian-rasterization-w-depth` rasterizer (depth-aware rasterization, needed for the depth-guided pixel-Gaussian correspondence step).
- **Key files:** `train.py` (main training loop), `gen_mask/` (SAM-based 2D segmentation mask generation, run as a preprocessing stage), `gen_flow/` (RAFT-based optical flow generation, also preprocessing), `visualize.py` (interactive/offline rendering), `external.py`/`helpers.py` (utilities), `colormap.py` (visualization).
- **Notable implementation details:** the released pipeline pulls in additional pretrained models beyond what the core method equations require — SAM (segmentation), TAP (captioning), and SBERT (`sentence-transformers`, semantic embeddings) — which support the paper's stated downstream "part-level 4D scene editing" / language-queryable parts capability rather than the core reconstruction/tracking method itself. As of inspection, the authors note self-captured dataset release is still pending ("will be releasing it soon"), so the training data for one of the four benchmark scenes is not yet public even though the code is.

#### Failure Modes & Limitations

Cannot operate on monocular video (both mask-lifting and flow-based part motion estimation need multi-view input). New objects/parts entering the scene after t=0 are not modeled, since the part set is fixed at initialization. Sequential per-timestep training creates forward temporal dependency, so errors at one timestep can affect later ones. Far-field regions under complex lighting show degraded rendering quality (paper's Figure 19). The ablation further shows the method is highly reliant on the local rigidity loss specifically — without it, tracking accuracy collapses to 19.56% and MTE balloons to 33.56 cm on the Tennis scene, indicating the part-rigid structural assumption is load-bearing rather than a minor refinement.

---

## Relevance to ADAGS

Direct competitor to revisiting part-aware reversible routing.

## Connections

## Sources

- arXiv: https://arxiv.org/abs/2605.10307
- Project page: https://pamosplat.github.io/
- Code: https://github.com/BIT-DYN/pamosplat
