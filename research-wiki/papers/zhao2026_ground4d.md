---
type: paper
node_id: paper:zhao2026_ground4d
title: "Ground4D: Consistency-Aware 4D Reconstruction from Monocular Video"
authors: ["Qing Zhao", "Weijian Deng", "Pengxu Wei", "Liang Lin"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2606.28828"
tags: [dynamic-reconstruction, monocular, geometry-consistency, 3d-foundation-models, dynamic-gs]
status: deep-dived
---

# Ground4D: Consistency-Aware 4D Reconstruction from Monocular Video

**Paper:** https://arxiv.org/abs/2606.28828
**Code:** Not found. (A GitHub repo named "Ground4D" exists at github.com/wsnbws/Ground4D but is an unrelated off-road-scene-reconstruction project — a naming collision, not this paper's code.)
**Base method:** VGGT (Visual Geometry Grounded Transformer) as a frozen, training-free 3D foundation model for geometry/pose initialization, combined with MoSca (dynamic Gaussian fusion via 4D motion scaffolds) as the deformable Gaussian representation and optimizer that is refined on top.

## One-line thesis

Suppressing Key/Value contributions from dynamic tokens (while keeping their Query vectors) in the first few global attention layers of a frozen 3D foundation model (VGGT) lets it produce multi-view-consistent static geometry and camera poses directly from monocular video containing moving objects, and using this geometry as a consistency-loss target — evaluated at both real and synthesized virtual viewpoints — keeps a MoSca-style dynamic Gaussian optimization from drifting into 3D-inconsistent solutions that photometric loss alone cannot detect.

## Problem / Gap

MoSca and similar dynamic Gaussian methods are optimized with photometric loss from a single monocular camera trajectory, so multiple different underlying 4D geometries can render correctly along that one trajectory while disagreeing with each other in 3D over time (depth/scale ambiguity, drift in moving regions). Separately, naively running 3D foundation models (VGGT, DUSt3R, MonST3R) on video containing moving objects corrupts the recovered static geometry and camera poses, because these models are pretrained under a rigid, static multi-view assumption that moving content violates.

## Method

Stage 1 runs VGGT once, training-free, over the full monocular frame sequence, but in the first `L_s=5` global self-attention layers it restricts Key/Value computation to tokens labeled static by an external segmentation (SAM), while leaving Query vectors unrestricted for all tokens — this stops moving-object tokens from polluting the static scene's internal representations without blocking them from still attending outward. This produces per-frame camera poses, an aggregated static point cloud, per-frame dynamic point clouds, and a multi-view-consistent depth map reprojectable to arbitrary viewpoints. Stage 2 initializes a MoSca-style deformable Gaussian scene (sparse SE(3) motion-scaffold nodes) and optimizes it with the usual photometric loss plus a geometry-consistency loss that penalizes disagreement between rendered depth and the Stage-1 projected depth, evaluated at the real training views and, after 2,000 iterations, also at synthesized "virtual" camera viewpoints slightly perturbed from each training view. Continuous-time rendering at arbitrary timestamps comes from B-spline interpolation over the nearest scaffold-node SE(3) transforms.

## Assumptions

Input is casual single-camera (monocular) video, not multi-view capture; the method assumes an external dynamic/static segmentation (SAM) is available to identify which tokens/pixels are dynamic, and that VGGT's pretrained rigid multi-view prior remains a valid proxy for the static background even in scenes containing moving content.

## Limitations / Failure Modes

The paper states VGGT processing "may become computationally demanding for very long video sequences," with no incremental/keyframe scheme yet in place. Dynamic/static separation currently depends on SAM as an external, decoupled model rather than something integrated into the foundation model itself — the authors explicitly flag this as future work ("future work could instead leverage intrinsic representations within the foundation model to achieve tighter integration"). The token-masking ablation shows the method is fragile to layer choice: masking all attention layers rather than only layers 1–5 quadruples ATE (0.010 → 0.040) and worsens Chamfer distance (0.254 → 0.414), so "mask more for safety" actively hurts.

## Reusable Ingredients

- **Dynamic token masking (mask K/V, keep Q, shallow layers only)** — a training-free way to make a rigid-scene 3D foundation transformer robust to moving objects, without retraining or per-frame independent inference.
- **Virtual/synthesized-viewpoint consistency supervision** — perturb the camera pose slightly off the training trajectory and supervise geometry there too, using foundation-model projected depth as a pseudo-label where no real image exists.
- **Two-stage decoupled pipeline** — frozen foundation geometry used first for initialization, then again as an ongoing consistency-loss target during separate dynamic Gaussian optimization, rather than only as a one-shot initializer.
- **B-spline SE(3) blending over sparse motion-scaffold nodes** — continuous-time deformation query for arbitrary-timestamp rendering.

---

### Deep Dive

#### Core Novelty

Relative to running VGGT (or DUSt3R/MonST3R-style foundation models) as-is, Ground4D adds selective dynamic-token masking in shallow global-attention layers so the frozen model itself becomes robust to moving content, instead of requiring scene-specific retraining or discarding dynamic frames. Relative to MoSca, it adds a geometry-consistency loss sourced from this foundation geometry and evaluated at synthesized virtual viewpoints — not just the observed training views — which constrains 4D solutions that would otherwise satisfy photometric loss at the training camera trajectory alone but disagree with themselves in 3D.

#### Mathematical Formulation

**Eq. 1 — Dynamic token masking** (applied inside VGGT's global self-attention, layers 1 through `L_s=5`, before rasterization/reconstruction begins):
$$\tilde{K}_t^{(l)} = \{K_t^{(l)}(k) \mid t_t^{(l)}(k) \in S\}, \qquad \tilde{V}_t^{(l)} = \{V_t^{(l)}(k) \mid t_t^{(l)}(k) \in S\}$$
where $K_t^{(l)}(k)$, $V_t^{(l)}(k)$ are the Key/Value vectors for token $k$ at frame $t$, layer $l$; $S$ is the set of static tokens (from the SAM-derived mask). Query vectors are left unrestricted for all tokens (dynamic included), so dynamic tokens can still attend out but cannot inject their Key/Value into other tokens' attention.

**Eq. 2 — Geometry-consistency loss** (evaluated after rendering, as a loss term over both real and synthesized viewpoints during Stage-2 optimization):
$$\mathcal{L}_{gc} = \sum_t \lVert \tilde{D}_t - \hat{D}_t \rVert + \lambda_s \sum_t \sum_k \lVert \tilde{D}_{t,k} - \hat{D}_{t,k} \rVert$$
$\tilde{D}_t$ is Gaussian-rendered depth at frame $t$; $\hat{D}_t$ is the Stage-1 foundation-model depth projected to that frame; the second term repeats this at $k$-th synthesized virtual viewpoints; $\lambda_s = 0.5$ weights the virtual-view term.

**Eq. 3 — Total training objective**:
$$\mathcal{L} = \mathcal{L}_{rgb} + \lambda_{gc}\mathcal{L}_{gc} + \lambda_r(\mathcal{L}_{arap} + \mathcal{L}_{acc} + \mathcal{L}_{vel})$$
with $\lambda_{gc}=0.1$, $\lambda_r=0.01$; $\mathcal{L}_{rgb}$ is the standard photometric term, ARAP/acceleration/velocity are scaffold regularizers inherited from MoSca.

**Eq. 4 — Continuous-time scaffold-node pose** (queried at render time for arbitrary-timestamp rendering):
$$Q_\tau^{(m)} = \sum_{i \in N(\tau)} B_i(\tau) C_i^{(m)}$$
$Q_\tau^{(m)}$ is the interpolated SE(3) pose of motion-scaffold node $m$ at continuous time $\tau$; $N(\tau)$ is the set of neighboring discrete control times; $B_i(\tau)$ are B-spline basis weights; $C_i^{(m)}$ are the node's discrete control SE(3) transforms.

#### Algorithm / Pipeline Changes

1. Run VGGT once, training-free (no fine-tuning), over the full monocular frame sequence.
2. Obtain per-frame SAM masks labeling tokens/pixels as static vs. dynamic.
3. In VGGT's global attention layers 1–5, restrict Key/Value per Eq. 1 to static tokens only; leave layers 6+ and all Query vectors unmodified.
4. Extract camera poses $\{g_t\}_{t=1}^T$, an aggregated static point cloud, and per-frame dynamic point clouds from VGGT's output.
5. Project the recovered 3D points to build reprojectable dense depth $\hat{D}$ at each frame (and at arbitrary queried viewpoints).
6. Initialize a MoSca-style dynamic Gaussian scene with sparse SE(3) motion-scaffold nodes from the recovered geometry and poses.
7. Optimize for 10,000 iterations with $\mathcal{L}_{rgb}$ active from the start.
8. After 2,000 iterations, begin sampling $K=1$ synthesized virtual camera per training view (~5° rotation, 0.1× translation perturbation) and activate the virtual-view term of $\mathcal{L}_{gc}$.
9. Apply $\mathcal{L}_{gc}$ (real + virtual) at weight $\lambda_{gc}=0.1$ and scaffold regularizers at $\lambda_r=0.01$ throughout optimization.
10. At render/query time, obtain the Gaussian pose at an arbitrary timestamp via B-spline blending of nearby scaffold-node SE(3) transforms (Eq. 4).

#### Key Hyperparameters & Design Choices

- Total optimization steps: 10,000
- Token-masking scope: first $L_s=5$ global attention layers of VGGT
- Virtual-view count: $K=1$ per training view, introduced after 2,000 iterations
- Virtual-view camera perturbation: ~5° rotation, 0.1× translation ratio
- $\lambda_s = 0.5$ (virtual-view term weight inside $\mathcal{L}_{gc}$)
- $\lambda_{gc} = 0.1$ (geometry-consistency loss weight in total objective)
- $\lambda_r = 0.01$ (regularization weight, covers ARAP + velocity + acceleration terms)
- Dynamic/static segmentation source: SAM
- Hardware: single NVIDIA A6000 GPU; VGGT used frozen, no fine-tuning
- Deformation-network architecture dimensions: Not specified in paper (inherits MoSca's motion-scaffold representation; no new MLP dims are introduced)

#### Ablation Summary

- **Token-masking layer range is the most impactful and most fragile choice.** No masking: ATE 0.012, Chamfer 0.262. Masking layers 1–5 (chosen): ATE 0.010, Chamfer 0.254. Masking all layers: ATE 0.040, Chamfer 0.414 — a 4× degradation in pose accuracy versus the chosen range, showing more masking is not safer.
- **Geometry-consistency supervision, added incrementally:** photometric-only baseline 18.72 mPSNR / 0.261 Chamfer → + real-training-view geometry term 18.91 mPSNR / 0.237 Chamfer → + virtual-view term (full method) 19.07 mPSNR / 0.226 Chamfer. The virtual-view term produces the largest single Chamfer-distance improvement.
- **Mask source (SAM vs. an internal VGGT-derived dynamic signal):** comparable accuracy/completeness (0.136 / 0.977 with SAM), indicating the method's gains come mainly from the geometry-consistency modeling rather than from which segmenter supplies the dynamic mask.
- **Virtual-view sampling count/perturbation:** $K=1$ with 5°/0.1× perturbation was found optimal (19.07 mPSNR / 0.226 Chamfer distance) among tested settings.

#### Failure Modes & Limitations

The paper states VGGT processing "may become computationally demanding for very long video sequences," with key-frame selection or incremental geometry propagation left as future work. Dynamic/static separation depends on SAM as an external, decoupled model rather than an integration native to the foundation model — flagged explicitly as a direction for tighter future integration. The ablation shows the method degrades sharply (4× worse ATE) if the shallow-layer-only masking design is not respected.

## Relevance to ADAGS

This paper widens the redo beyond ADAGS: if geometry consistency is the bottleneck, reliability-gated masks/flow alone may be too local. ADAGS can still use Ground4D as a comparator or as a source of geometry-consistency cues for counterfactual prior-usefulness routing.

## Connections

- Addresses [[gap_map#G12 - Feedforward 4D Models Raise The Baseline]]
- Adds pressure to geometry-consistency gaps in dynamic Gaussian reconstruction.
- Inspires [[ideas/counterfactual-prior-usefulness-routing]]

## Sources

- https://arxiv.org/abs/2606.28828
