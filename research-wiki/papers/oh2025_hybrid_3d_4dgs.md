---
type: paper
node_id: paper:oh2025_hybrid_3d_4dgs
title: "Hybrid 3D-4D Gaussian Splatting for Fast Dynamic Scene Representation"
authors: ["Seungjun Oh", "Younggeun Lee", "Hyejin Jeon", "Eunbyung Park"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2505.13215"
tags: [dynamic-gs, hybrid-3d-4d, fast-representation, baseline]
status: deep-dived
---

# Hybrid 3D-4D Gaussian Splatting for Fast Dynamic Scene Representation

**Paper:** https://arxiv.org/abs/2505.13215
**Code:** https://github.com/ohsngjun/3D-4DGS
**Base method:** 4D Gaussian Splatting (Yang et al. 2024, "Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting"), with the differentiable-rasterizer backward pass borrowed from Taming-3DGS.

## One-line thesis

Most 4D Gaussians in a trained 4DGS scene end up with a very large temporal scale (i.e. they are effectively static), so thresholding each Gaussian's learned temporal-scale parameter and converting the ones above threshold into plain 3D Gaussians (dropping the time axis) removes wasted 4D compute/memory without hurting quality, since those Gaussians were never using their temporal capacity.

## Problem / Gap

4DGS (Yang et al. 2024) represents every Gaussian with a full 4D mean/covariance (3 spatial + 1 temporal) regardless of whether the underlying content moves. In real captures the large majority of Gaussians land on static background, but each still pays for a 4×4 rotation, temporal scale/mean, and the marginalization math needed to slice a spatio-temporal Gaussian at query time t. This makes training slow (hours) and memory-heavy, and the paper shows empirically (their Figure 2 temporal-scale histogram) that most Gaussians already have near-degenerate temporal scale after a short warmup, i.e. the 4D parameterization is wasted capacity, not a modeling necessity.

## Method

Training starts with a standard 4DGS optimization for a short warmup (500 iterations). After warmup, at each conversion step the method inspects every Gaussian's exponentiated temporal-scale parameter exp(s_{t,i}) against a fixed threshold τ; Gaussians above τ (i.e. temporally near-constant) are converted in place into ordinary 3D Gaussians by discarding the temporal mean, projecting the 4×4 rotation down to a 3×3 spatial rotation, and re-deriving a 3D quaternion from that submatrix. The remaining Gaussians stay 4D and keep training with full spatio-temporal marginalization. A custom CUDA rasterizer renders both populations in one unified pass: 4D Gaussians are sliced at the query time t into transient 3D Gaussians, merged with the true 3D (converted) Gaussians, and alpha-composited back-to-front together. Densification is run separately for the 3D and 4D populations, and (unlike vanilla 3DGS/4DGS) periodic opacity resets are disabled because they were found to disrupt the spatio-temporal optimization.

## Assumptions

Assumes calibrated multi-view video (N3V-style / Technicolor-style rigs) with a fixed background and a 4DGS-style continuous deformation representation as the starting point; it assumes the static/dynamic split is well-approximated by a single global scalar threshold on temporal scale rather than requiring per-region or learned classification.

## Limitations / Failure Modes

The authors state the scale-thresholding heuristic is dataset/duration-dependent — they need different τ for 10s N3V clips (τ=3), 40s sequences (τ=6), and Technicolor (τ=1) — and suggest it "could be refined, potentially using learning-based or data-driven methods," i.e. the hard global threshold is acknowledged as brittle. They also note no specialized 4D densification strategy is used, leaving redundancy/memory further reducible, and that their simpler per-scene initialization forgoes the frame-by-frame COLMAP initialization other baselines use (which costs ~20 extra minutes but can improve geometry).

## Reusable Ingredients

- **Temporal-scale thresholding as a static/dynamic classifier**: use each Gaussian's learned temporal-scale parameter exp(s_t) directly as a cheap, no-extra-network signal for static vs. dynamic classification, rather than learning a separate mask/head.
- **In-place representation conversion (4D→3D) rather than pruning**: instead of deleting redundant Gaussians, demote them to a cheaper parameterization, preserving their learned appearance/geometry while dropping only the now-unused temporal DOF.
- **Disabling opacity resets for spatio-temporal optimization**: periodic opacity resets (standard in 3DGS-style training) can be actively harmful once a temporal dimension is involved; ablation shows +0.73 dB and better LPIPS from disabling them.
- **Unified dual-population rasterization**: a single rasterizer pass that slices 4D Gaussians into transient 3D Gaussians at render time and composites them together with true static 3D Gaussians, avoiding two separate render passes.

---

### Deep Dive

#### Core Novelty

Relative to 4DGS, the paper's only structural change is a training-time conversion operator that migrates individual Gaussians from the 4D parameter family to the 3D parameter family once their temporal scale indicates they are effectively time-invariant. The key insight is that this is a capacity-reallocation move, not a quality trade-off: since a converted Gaussian's temporal extent already covered the whole sequence, slicing it at any query time yields (approximately) the same spatial Gaussian, so dropping the temporal parameters loses negligible information while removing the marginalization cost for every future forward/backward pass and render.

#### Mathematical Formulation

**Static/dynamic classification** (evaluated per Gaussian, periodically during training, before rasterization):
$$\text{static}_i = \mathbb{1}\left[\exp(s_{t,i}) > \tau\right]$$
where $s_{t,i}$ is the (log-space) temporal-scale parameter of Gaussian $i$'s 4D covariance, and $\tau$ is a fixed, dataset-specific scalar threshold chosen by inspecting the valley in the histogram of $\exp(s_{t,i})$ values.

**4D→3D conversion** (applied once per Gaussian at the moment it crosses threshold, before it continues training as a 3D Gaussian):
- Mean: keep spatial mean $\mu_x$, discard temporal mean $\mu_t$ from $\mu_{4D} = (\mu_x, \mu_t)$.
- Rotation: extract the 3×3 spatial block $R_{3D}$ from the 4×4 rotation $R_{4D}$ (which, at the static limit, reduces to $R_{4D} = \begin{pmatrix} R_{3D} & 0 \\ 0^\top & 1\end{pmatrix}$), then convert to a unit quaternion:
$$w = \tfrac{1}{2}\sqrt{1+\mathrm{tr}(R_{3D})}, \quad x = \frac{R_{3D}(3,2)-R_{3D}(2,3)}{4w}, \quad y = \frac{R_{3D}(1,3)-R_{3D}(3,1)}{4w}, \quad z = \frac{R_{3D}(2,1)-R_{3D}(1,2)}{4w}$$
- The resulting static Gaussian is re-parameterized as $(\mu_x, q_{3D}, s_x, s_y, s_z, \sigma, \text{SH})$ — a standard 3DGS Gaussian.

**4D-to-time-t marginalization** used for the still-4D population at render time (standard 4DGS slicing, evaluated per-Gaussian, per query time $t$, before projection):
$$\mu_{xyz|t} = \mu_{1:3} + \Sigma_{1:3,4}\,\Sigma_{4,4}^{-1}(t-\mu_t), \qquad \Sigma_{xyz|t} = \Sigma_{1:3,1:3} - \Sigma_{1:3,4}\,\Sigma_{4,4}^{-1}\Sigma_{4,1:3}$$

#### Algorithm / Pipeline Changes

1. Initialize and train a standard 4DGS scene for 500 iterations (warmup) with no conversion.
2. Every fixed interval thereafter (up to iteration 15,000), evaluate $\exp(s_{t,i}) > \tau$ for every Gaussian still in 4D form.
3. For each Gaussian crossing threshold, run the 4D→3D conversion (drop $\mu_t$, project rotation, re-derive quaternion) and move it into the 3D Gaussian set; it is no longer touched by 4D-specific optimization/marginalization from that point on.
4. Densification runs on the 3D and 4D populations separately, every 100 iterations, up to iteration 15,000.
5. Opacity resets, standard in 3DGS/4DGS training, are disabled entirely for the whole run.
6. At render time, for query time $t$: 4D Gaussians are sliced via the marginalization above into transient 3D Gaussians; these are merged with the true (converted) static 3D Gaussians; the unified set is projected to screen space, tile/depth-keyed, sorted, and alpha-composited back-to-front in one CUDA rasterizer pass (Algorithm 1 in the paper).
7. Total training: 6,000 iterations for 10s N3V-style scenes, 20,000 iterations for 40s sequences; batch size 4 (N3V) / 2 (Technicolor).

#### Key Hyperparameters & Design Choices

- Temporal-scale threshold $\tau$: 3 for 10-second N3V sequences, 6 for 40-second sequences, 1 for Technicolor (dataset/duration-specific, hand-set from a histogram valley, not learned).
- Warmup before any conversion: 500 iterations of pure 4DGS.
- Densification window: every 100 iterations, up to iteration 15,000, run separately per population.
- Opacity resets: disabled (default 3DGS/4DGS periodic reset is turned off).
- Training length: 6,000 iters (10s scenes) / 20,000 iters (40s scenes).
- Batch size: 4 (N3V), 2 (Technicolor).
- MLP/network dims: Not specified in paper — the method has no learned classifier network; thresholding is a closed-form comparison, not a trained head.
- Loss weights: Not specified in paper (loss formulation is inherited from base 4DGS; the paper does not report modified loss weights).

#### Ablation Summary

Reported on N3V (Table 4 in paper):

- **Scale threshold τ** (most impactful reported ablation): τ=3.0 (default) gives 32.25 dB with 843,175 4D + 229,707 3D Gaussians; τ=2.5 drops to 31.37 dB (670,807 4D / 276,265 3D — too aggressive conversion, quality loss); τ=3.5 gives 31.98 dB (913,927 4D / 184,548 3D — keeps more in 4D, doesn't help). τ=3.0 is the peak; both directions hurt PSNR.
- **Opacity resets**: disabling resets improves PSNR from 31.52 → 32.25 dB (+0.73 dB) and LPIPS from 0.1016 → 0.0970. Flagged as the second lever but with a fixed, unambiguous direction (always disable), whereas τ requires per-dataset tuning.

#### Implementation Reality

- **Framework:** PyTorch + custom CUDA, built directly on the 4DGS (Yang et al. 2024) codebase, reusing the differentiable-rasterization backward pass from Taming-3DGS.
- **Key files (repo structure, exact conversion-logic file not confirmed from available access):** `gaussian_renderer/` (rendering/rasterization integration), `diff-gaussian-rasterization/` (CUDA rasterizer, unified 3D+4D compositing), `scene/` (scene/data handling), `pointops2/` and `simple-knn/` (point-cloud ops, likely used for densification/KNN), `configs/n3v/default.yaml` (N3V hyperparameters incl. τ), `scripts/n3v2blender.py` (N3V dataset preprocessing), `main.py` (training entry point, single-sequence and batch modes).
- **Notable implementation details:** Repo is described as a "drop-in replacement" for existing 4DGS pipelines. Specific hyperparameter values inside `configs/n3v/default.yaml` and any code-vs-paper discrepancies could not be confirmed from the accessible repo summary — flagged as unverified rather than guessed.

#### Failure Modes & Limitations

The paper explicitly flags the hand-set, dataset/duration-dependent threshold τ as a weak point ("could be refined, potentially using learning-based or data-driven methods") — it is not a single universal value and must be re-tuned per capture duration/dataset. It also flags the absence of a specialized 4D-aware densification strategy as leaving redundancy/memory further reducible, and notes its simpler initialization (vs. frame-by-frame COLMAP init used by some baselines) trades ~20 minutes of extra setup time for potentially weaker geometry init.

## Relevance to ADAGS

This is foundational prior art for the local ADAGS codebase and should be treated as a baseline/ancestor, not merely background. ADAGS novelty must be the reversible LoRA route, priors, diagnostics, or targeted residual allocation on top of this family.

## Connections

- Addresses [[gap_map#G2 - Static/Dynamic Leakage Is A Representation And Evaluation Problem]]
- Addresses [[gap_map#G5 - Capacity Allocation Must Be Matched And Dynamic-Aware]]

## Sources

- https://arxiv.org/abs/2505.13215
