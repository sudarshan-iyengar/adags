---
type: paper
node_id: paper:gao2024_hicom
title: "HiCoM: Hierarchical Coherent Motion for Dynamic Streamable Scenes with 3D Gaussian Splatting"
authors: ["Rui Gao", "Lu Chen", "Zhao Wang", "Lan Xu"]
year: 2024
venue: "NeurIPS"
external_ids:
  arxiv: "2411.07541"
tags: [dynamic-gs, hierarchical-motion, coherent-motion]
status: deep-dived
---

# HiCoM: Hierarchical Coherent Motion for Dynamic Streamable Scenes with 3D Gaussian Splatting

**Paper:** https://arxiv.org/abs/2411.07541
**Code:** https://github.com/gqk/HiCoM
**Base method:** 3D Gaussian Splatting (Kerbl et al. 2023), in the online/streaming per-frame regime pioneered by 3DGStream; compared against 4DGaussians and Dynamic3DGS as related online/offline dynamic-GS baselines.

## One-line thesis

Sharing a small set of per-region rigid motion parameters across all Gaussians in a spatial cell (a coarse-to-fine hierarchy of cells), instead of learning per-Gaussian deformation independently each frame, gives frame-to-frame motion enough spatial coherence to converge in ~100 steps/frame while avoiding the overfitting and storage blowup that per-Gaussian online updates cause.

## Problem / Gap

Streaming/online dynamic-GS reconstruction (e.g. 3DGStream-style per-frame updates) learns motion and density updates independently for every incoming frame, which overfits to the sparse camera views available at each timestep and lets the Gaussian count grow unbounded, driving up training time, storage, and transmission cost. Prior online methods also treat per-Gaussian motion as unconstrained, so nearby Gaussians on the same physical surface can drift incoherently under limited multi-view supervision.

## Method

HiCoM first trains a single high-quality 3DGS on the initial frame using a perturbation-smoothing strategy (adding Gaussian noise to Gaussian positions during optimization) to curb overfitting to sparse initial views. For subsequent frames it partitions Gaussians into cubic spatial regions at multiple hierarchy levels (coarse to fine) and learns one shared rigid motion (translation + rotation) per region per level; each Gaussian's actual per-frame motion is the sum of the region motions across all levels containing it. Only ~100 steps of motion learning plus ~100 steps of lightweight densification/pruning ("continual refinement") are run per frame, with motion parameters warm-started from the previous frame's solution. An optional parallel-training mode processes k frames simultaneously against a shared reference frame to trade memory for wall-clock throughput.

## Assumptions

Requires calibrated multi-view (multi-camera) video input (evaluated on N3DV and Meet Room, both multi-camera studio/desktop capture rigs), an initial frame dense enough to fit a good static 3DGS, and scene motion that is locally piecewise-rigid enough to be approximated by shared per-region rigid transforms rather than fully deformable per-Gaussian fields.

## Limitations / Failure Modes

The paper reports that reconstruction quality is bottlenecked by the initial frame's 3DGS — errors in that first fit propagate through all later frames. Because each frame is optimized from the previous frame's state without global reoptimization, the method is susceptible to error accumulation over long sequences. Evaluation is restricted to indoor multi-camera capture (N3DV, Meet Room); outdoor or sparser-view generalization is untested. These are acknowledged as common issues for the online-learning-from-streaming-video paradigm generally, not fixes unique to HiCoM.

## Reusable Ingredients

- **Perturbation smoothing** — adding position noise during initial-frame 3DGS training to prevent overfitting to sparse initial camera views.
- **Hierarchical shared-motion regions** — grouping Gaussians into a coarse-to-fine cell hierarchy and summing per-level shared rigid motions, giving spatial coherence "for free" without per-Gaussian motion supervision.
- **Warm-started per-frame optimization** — initializing frame t+1's motion parameters from a fraction (0.6x) of frame t's solved values, cutting per-frame convergence steps.
- **Parallel multi-frame training against a shared reference** — trading memory for throughput by batching k frames against one anchor frame instead of strictly sequential per-frame updates.

---

### Deep Dive

#### Core Novelty

Relative to per-frame independent motion fitting (3DGStream-style), HiCoM's change is to tie Gaussian motion parameters together spatially: instead of every Gaussian owning its own free per-frame transform, a hierarchy of spatial cells each own one shared rigid motion, and a Gaussian's realized motion is the sum of the motions of every cell (at every hierarchy level) that contains it. The insight is that under sparse multi-view supervision, unconstrained per-Gaussian motion is underdetermined and overfits; forcing spatial neighbors to share motion parameters regularizes the problem and lets each frame converge in far fewer optimization steps (~100).

#### Mathematical Formulation

Initial-frame perturbation smoothing (applied per-Gaussian, during initial 3DGS training, before rasterization each step):
$$\mu_p = \mu + \lambda_{noise} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)$$
where $\mu$ is a Gaussian's mean position, $\mu_p$ the perturbed position actually rasterized that step, and $\lambda_{noise}$ a fixed noise scale (0.01).

Region assignment (evaluated once per hierarchy level, per frame, before motion optimization):
$$r = \left\lfloor \frac{\mu}{e} + 0.5 \right\rfloor \cdot e$$
where $\mu$ is the Gaussian's position, $e$ is the region (cell) size for that hierarchy level, and $r$ is the resulting region center the Gaussian is bound to. Finer levels use smaller $e$ (each level subdivides the previous by $2^3$).

Composed per-frame motion (per-Gaussian, applied before rasterization each frame):
$$\Delta\mu_g = \sum_{l=1}^{L} \Delta\mu^l, \qquad \Delta q_g = \sum_{l=1}^{L} \Delta q^l$$
where $L$ is the number of hierarchy levels (3), $\Delta\mu^l$/$\Delta q^l$ are the shared translation/rotation-quaternion offsets of the level-$l$ region containing the Gaussian, and $\Delta\mu_g$/$\Delta q_g$ are the Gaussian's final applied position/rotation deltas for that frame.

#### Algorithm / Pipeline Changes

1. Fit a single static 3DGS on the initial frame for 15k steps (N3DV) or 10k steps (Meet Room), with densification/splitting halted after 5k steps and perturbation smoothing ($\lambda_{noise}=0.01$) applied throughout to reduce overfitting to sparse initial views.
2. For each subsequent frame: assign every existing Gaussian to a region at each of 3 hierarchy levels via the region-assignment equation (cell size $e$ shrinking by $2^3$ per level, cap of 55 Gaussians/region before finer subdivision is triggered).
3. Optimize the shared per-region motion parameters ($\Delta\mu^l$, $\Delta q^l$) for $E_m = 100$ steps against that frame's multi-view images, warm-started at $0.6\times$ the previous frame's solved region-motion values; sum across levels to get each Gaussian's applied $\Delta\mu_g$, $\Delta q_g$.
4. Run "continual refinement" for $E_r = 100$ steps: clone Gaussians in regions whose accumulated position gradients exceed a threshold (new geometry/disocclusion handling), then prune low-opacity duplicates to keep Gaussian count compact; densification is triggered every 40 steps within this window.
5. (Optional) Parallel-training mode processes $k$ frames concurrently, all optimized against the same reference frame's 3DGS state rather than strictly chaining frame-by-frame, reducing wall-clock at the cost of memory.
6. Standard 3DGS photometric loss is used for both the initial frame and per-frame motion/refinement optimization: $L = (1-\lambda)L_1 + \lambda L_{D\text{-}SSIM}$ with $\lambda = 0.2$ (unmodified from vanilla 3DGS).

#### Key Hyperparameters & Design Choices

- Perturbation noise scale $\lambda_{noise} = 0.01$, chosen empirically from $\{0.001, 0.01, 0.1\}$.
- Initial-frame training steps: 15k (N3DV), 10k (Meet Room); splitting/densification halted at 5k steps.
- Motion hierarchy levels $L = 3$ (default); each subsequent level subdivides regions by $2^3$.
- Max Gaussians per region before subdivision: 55.
- Per-frame motion learning steps $E_m = 100$; per-frame refinement steps $E_r = 100$.
- Motion parameters warm-started at $0.6\times$ previous frame's values.
- Densification interval during refinement: every 40 steps.
- Photometric loss weight $\lambda = 0.2$ (standard 3DGS D-SSIM weighting, unmodified).
- Continual-refinement clone gradient threshold: not specified (numeric value not given in accessible text).

#### Ablation Summary

- **Motion learning (hierarchical shared motion) is the most impactful component**: removing it drops PSNR by ~2.3 dB on most scenes.
- **Perturbation smoothing**: removing it drops PSNR by ~1.4 dB on Coffee Martini.
- **Continual refinement**: removing it causes a slight PSNR decrease but with longer training times (refinement mainly helps efficiency, not just quality).
- **Hierarchy depth**: 3 levels is optimal; additional levels give diminishing returns (Table 4).
- **Motion-step convergence**: 100 motion steps/frame is sufficient; more steps show diminishing returns (Table 3).

#### Implementation Reality

- **Framework:** Python, built on "LibGS" (the authors' own Gaussian Splatting library), not directly on the original 3DGS reference repo; explicitly cites 3DGStream, 4DGaussians, and Dynamic3DGS as related/inspiring implementations.
- **Key files:** `main.py` (entry point), `config/*.yaml` (per-scene configs, e.g. `dynerf.yaml`), `pipeline/hicom/` (core algorithm implementation — motion hierarchy, perturbation smoothing, continual refinement live here per the repo layout).
- **Notable implementation details:** Not determinable from the README/page alone — the actual source under `pipeline/hicom/` was not accessible via the fetch, so no paper-vs-code discrepancies can be confirmed.

#### Failure Modes & Limitations

Initial-frame 3DGS quality bottlenecks all downstream frames since per-frame updates only refine relative to the previous state rather than re-optimizing globally. Sequential per-frame optimization without global bundle adjustment risks error accumulation over long sequences. All quantitative evaluation is on indoor multi-camera rigs (N3DV, Meet Room); the paper does not test outdoor scenes or sparser/monocular capture.

## Relevance to ADAGS

Supports investigating specialized residual motion only where LoRA route0 fails.

## Connections

## Sources

- arXiv: https://arxiv.org/abs/2411.07541
- Code: https://github.com/gqk/HiCoM
