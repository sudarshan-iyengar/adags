---
type: paper
node_id: paper:wu2026_rigs
title: "RiGS: Rigid-aware 4D Gaussian Splatting from a Single Monocular Video"
authors: ["Chenyu Wu", "Wanhua Li", "Zhu-Tian Chen", "Hanspeter Pfister"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2605.23672"
tags: [dynamic-gs, 4dgs, rigidity, monocular, temporal-consistency]
status: deep-dived
---

# RiGS: Rigid-aware 4D Gaussian Splatting from a Single Monocular Video

**Paper:** https://arxiv.org/abs/2605.23672
**Code:** https://github.com/ladvu/RiGS (CC BY 4.0)
**Base method:** 3D/2D Gaussian Splatting rasterization (2DGS backend via the `gsplat` library) combined with a shared SE(3) motion-basis deformation field in the style of prior monocular dynamic-GS work (e.g. Shape-of-Motion/MoSca-style basis trajectories), extended here with an explicit three-way rigid/transient/static primitive split.

## One-line thesis

Splitting dynamic Gaussians into a long-duration "rigid" population (driven by a small shared bank of SE(3) motion bases) and a short-duration "transient" population (driven by per-Gaussian linear velocity), with a learned soft temporal-existence gate and a training-time conversion rule that reassigns short-lived rigid Gaussians to the transient pool, resolves the bimodal duration statistics observed in single-basis rigid representations and lets one model cover both coherent long-range motion and brief high-frequency deformation without either underfitting fast motion or overfitting noise into spurious long-lived primitives.

## Problem / Gap

Prior monocular dynamic-GS methods model all non-rigid motion with a single global deformation mechanism (either a continuous MLP warp field or one shared bank of SE(3) trajectory bases). The authors empirically show that when such rigid/basis-driven Gaussians are fit with a learned temporal existence duration, the distribution of fitted durations is bimodal — a peak at long durations (genuinely rigid, persistent structure) and a separate sharp peak at short durations (Gaussians being misused to patch fast or non-rigid motion that the shared basis cannot represent). This means a single rigid/basis representation is being forced to cover two physically different regimes at once, degrading fine, fast-moving detail (e.g. facial expressions, judo/dog-agility motion in DAVIS) while also risking overfit artifacts.

## Method

RiGS maintains three explicit Gaussian populations — static {g^s}, rigid {g^r}, and transient {g^t} — rather than one deformation field. Rigid Gaussians move via a per-scene shared bank of K learned SE(3) transformation bases, blended per-Gaussian by a learned normalized weight vector; transient Gaussians instead move via a per-Gaussian learnable linear velocity, giving them unconstrained local motion at the cost of not sharing structure with other Gaussians. Both dynamic types carry a soft sigmoid temporal-existence gate (center, duration, sharpness) that fades their opacity in/out over time instead of hard-cutting visibility. Training proceeds in stages (static/rigid warm-up, then dynamic conversion, then joint optimization), during which rigid Gaussians whose fitted temporal duration falls below a threshold are converted into transient Gaussians — directly operationalizing the bimodal-duration observation as a routing rule. Supervision combines photometric/SSIM losses gated by an object-wise (not pixel-wise) dynamic mask, depth/normal geometric losses, and a correspondence loss that lifts 2D optical flow and long-range 2D tracks into 3D scene flow to directly supervise each Gaussian's rendered 3D trajectory.

## Assumptions

Single monocular video with known per-frame camera intrinsics/extrinsics; requires off-the-shelf metric depth, forward/backward optical flow, and 2D point tracks (from foundation models: ViPE/MoCa, MegaSaM, SEA-RAFT, BootsTAP, SAM2) as precomputed inputs, not learned jointly. Assumes the scene decomposes into a static background plus a set of segmentable dynamic objects, and that motion within the scene genuinely spans both a smooth/rigid regime and a fast/local regime (the two-population design is motivated by, and tuned to, that assumption).

## Limitations / Failure Modes

The paper's own diagnostic — the bimodal duration histogram — implies that rigid/basis-constrained Gaussians remain fundamentally unable to model "regions with large deformation or fast, non-rigid motion" on their own, which is exactly why the transient pool and the duration-triggered conversion rule are necessary; residual misrouting between the two pools is an implicit failure mode. Monocular video supplies only "limited structural cues," so the method leans on external depth/normal supervision rather than resolving ambiguity from images alone. Ablations show removing rigid Gaussians causes large empty (unreconstructed) regions, while removing transient Gaussians "significantly reduces fine details," confirming each pool covers a failure mode the other cannot.

## Reusable Ingredients

- **Bimodal temporal-duration diagnostic**: fit a soft temporal-existence gate per Gaussian and inspect the distribution of fitted durations as a signal for whether a single motion representation is being overloaded with two different motion regimes.
- **Duration-triggered population conversion**: a concrete, threshold-based rule (β^r below threshold → convert rigid Gaussian to transient) for reassigning primitives between representations mid-training, rather than fixing population membership at init.
- **Object-wise (not pixel-wise) dynamic mask via aggregated Sampson error**: aggregating per-pixel Sampson/flow-uncertainty scores over a segmented object's mask before thresholding, shown in ablation to beat pixel-wise Sampson masking (+1.43 dB).
- **3D scene-flow lifting from 2D flow + depth via warp-and-difference**, used as a direct supervisory signal on rendered per-Gaussian velocity rather than only on 2D reprojected flow.
- **Soft sigmoid temporal-existence gate** (center/duration/sharpness parameterization) as a differentiable alternative to hard temporal cropping/pruning of transient content.

---

### Deep Dive

#### Core Novelty

Relative to single-mechanism monocular dynamic-GS baselines (one deformation MLP, or one shared SE(3) basis bank for all dynamic Gaussians), RiGS's change is architectural: it keeps two structurally different, explicitly typed dynamic populations (shared-basis "rigid" vs. per-Gaussian-velocity "transient") alive simultaneously and lets Gaussians migrate between them during training based on a fitted temporal-duration statistic. The key insight is that "rigid vs. non-rigid" is not a fixed scene-level property to segment once, but a per-primitive, time-varying property best discovered by watching how long a shared-basis fit remains stable for that primitive.

#### Mathematical Formulation

Object-wise dynamic mask (per object $i$, frame $t$; used to gate photometric loss to actually-dynamic pixels):
$$s_{i,t} = \frac{1}{\sum_{\mathbf{p}\in\mathcal{P}_{i,t}} w_t(\mathbf{p})} \sum_{\mathbf{p}\in\mathcal{P}_{i,t}} w_t(\mathbf{p})\, e_t(\mathbf{p})$$
where $\mathcal{P}_{i,t}$ is object $i$'s pixel set in frame $t$, $w_t(\mathbf{p})$ is a flow-uncertainty/occlusion-derived weight, and $e_t(\mathbf{p})$ is the Sampson (epipolar) error. Aggregated over motion frames $\mathcal{F}_i$ (those exceeding a temporal threshold $\varepsilon^{temp}$):
$$s_i = \frac{1}{|\mathcal{F}_i|}\sum_{t\in\mathcal{F}_i} s_{i,t}, \qquad \mathbf{M}_t^{dyn} = \bigcup_{i:\, s_i > \varepsilon^{dyn}} \mathbf{M}_{i,t}^{obj}$$
Evaluated once per object as a preprocessing/labeling step before training, then used to weight $\mathcal{L}_{mask} = \mathrm{BCE}(\hat{\mathbf{M}}, \mathbf{M}^{dyn})$ each training step.

Rigid Gaussian motion (per-Gaussian, evaluated every frame before rasterization): a shared bank of $K$ per-frame SE(3) bases $\{\mathbf{T}_{j,t}\}_{j=1}^K$ is blended by a learned per-Gaussian weight vector $\mathbf{w}$ (constrained $\lVert\mathbf{w}\rVert_2^2=1$):
$$\boldsymbol{\mu}^r(t) = \Big(\sum_{j=1}^K w_j \mathbf{T}_{j,t}\Big)\cdot \boldsymbol{\mu}^r, \qquad \mathbf{R}^r(t) = \Big(\sum_{j=1}^K w_j \mathbf{T}_{j,t}\Big)_{3\times 3}\cdot \mathbf{R}^r$$
Transient Gaussian motion (per-Gaussian, unconstrained): position evolves by a learnable constant velocity $\mathbf{v}$,
$$\boldsymbol{\mu}^t(t) = \boldsymbol{\mu}^t + \mathbf{v}(t-\gamma^t)$$
Shared soft temporal-existence gate applied to both rigid and transient opacity, evaluated per-frame before compositing:
$$o(t) = o\cdot \sigma\big(\alpha(\beta - |t-\gamma|)\big)$$
with $\alpha$ = boundary sharpness, $\beta$ = temporal duration (half-width), $\gamma$ = temporal center — this is the quantity whose fitted per-Gaussian $\beta^r$ produces the bimodal-duration diagnostic and drives the conversion rule.

3D scene-flow lifting (computed once per frame pair from precomputed depth/flow, used as a supervisory target, not a model component):
$$\mathbf{v}_t^{fwd} = \mathcal{W}\big(\pi_{t+1}^{-1}(\mathbf{D}_{t+1}), \mathbf{F}_t^{fwd}\big) - \pi_t^{-1}(\mathbf{D}_t), \qquad \mathbf{v}_t^{bwd} = \pi_t^{-1}(\mathbf{D}_t) - \mathcal{W}\big(\pi_{t-1}^{-1}(\mathbf{D}_{t-1}), \mathbf{F}_t^{bwd}\big)$$
where $\mathcal{W}$ warps a point cloud by a 2D flow field and $\pi^{-1}$ unprojects a depth map using that frame's $\mathbf{K}, \mathbf{E}$. Supervises rendered velocity as $\mathcal{L}_{flow} = \lVert\hat{\mathbf{v}}^{fwd}-\mathbf{v}^{fwd}\rVert_1 + \lVert\hat{\mathbf{v}}^{bwd}-\mathbf{v}^{bwd}\rVert_1$ (a per-step training loss term, not evaluated at render time for inference).

Regularization on rigid Gaussians (loss term, discourages both needle-shaped covariances and overuse of long rigid durations that would resist conversion):
$$\mathcal{L}_{reg} = \frac{1}{N}\sum_{i=1}^N \Big(\lambda_\beta \cdot \frac{1}{\beta_i^r} + \lambda_s \cdot \mathrm{var}(\mathbf{s}_i)\Big)$$

#### Algorithm / Pipeline Changes

1. Preprocess monocular video with ViPE/MoCa (depth-aligned poses + metric depth), SEA-RAFT (forward/backward optical flow), BootsTAP (long-range 2D tracks), and SAM2 (object segmentation, modified to detect new instances) — all offline, before any Gaussian optimization.
2. Compute object-wise dynamic mask $\mathbf{M}_t^{dyn}$ per the aggregated-Sampson-error formula above; this labels which segmented objects/pixels are treated as dynamic for the rest of training.
3. Lift 2D flow + depth to 3D scene flow per frame pair (warp-then-difference, forward and backward) — precomputed supervisory targets.
4. **Warm-up phase**: initialize and optimize only static Gaussians, rigid Gaussians, and the shared SE(3) motion-basis bank (3K steps static + 12K rigid on Nvidia Dynamic Scenes; 5K + 40K on DyCheck iPhone).
5. **Conversion step**: any rigid Gaussian whose fitted $\beta^r$ falls below a threshold (2.0) is reassigned into the transient population (re-parameterized with a learnable velocity $\mathbf{v}$ instead of basis weights $\mathbf{w}$).
6. **Joint training phase**: static, rigid, and transient Gaussians are all optimized together; the rigid→transient conversion check continues to run periodically during this phase ("dynamic conversion"), not just once.
7. Per training step, render with the standard 2DGS/gsplat rasterizer using the combined static+rigid(t)+transient(t) Gaussian set, and apply photometric ($\mathcal{L}_{photo}$, mask-gated), geometric ($\mathcal{L}_{geom}$: scale/translation-invariant depth + cosine normal loss), correspondence ($\mathcal{L}_{cor}$: rendered-3D-trajectory vs. track loss + $\mathcal{L}_{flow}$), and regularization ($\mathcal{L}_{reg}$) losses.
8. At inference, render selectable component subsets (fused/static/dynamic/transient/rigid) — the type split is preserved through to rendering/visualization, not just training.

#### Key Hyperparameters & Design Choices

- $K$ (SE(3) motion bases): 10, all scenes.
- $\beta^r$ conversion threshold: 2.0.
- Soft-gate sharpness $\alpha$: 3.0.
- Temporal thresholds: $\varepsilon^{temp} = 10^{-4}$; $\varepsilon^{dyn} = \max(s_i)/4$ (adaptive per scene).
- Loss weights: $\lambda_{ssim}=0.1$, $\lambda_a=0.5$ (mask), $\lambda_{depth}=0.05$, $\lambda_{normal}=0.05$, $\lambda_{track}=2.0$, $\lambda_{flow}=0.01$, $\lambda_\beta=0.5$, $\lambda_s=0.5$.
- Learning rates: position 1.6e-4, scale 5e-3, rotation 1e-3, opacity 5e-2, color 1e-2, $\beta$ 1e-3, $\gamma$ 1e-3, motion weights $\mathbf{w}$ 1e-2, SE(3) bases $\mathbf{T}$ 1e-4.
- Training length: 30K iterations (Nvidia Dynamic Scenes), 100K (DyCheck iPhone); ~30 min for a 270-frame, 480×360 video at 30K iterations; ~160 FPS inference.
- Fundamental-matrix estimation: LMEDS over 10,000 sampled correspondences (used to compute Sampson error for the dynamic mask).
- Depth source: Metric3D (Nvidia dataset) or provided LiDAR depth (DyCheck), both pose-aligned via MoCa.

#### Ablation Summary

(Nvidia Dynamic Scene Dataset, average over scenes; full method: PSNR 27.43, SSIM 0.879, LPIPS 0.051)

- **Object-wise vs. pixel-wise Sampson mask**: pixel-wise variant drops to 26.00 PSNR / 0.069 LPIPS (−1.43 dB) — the single largest ablation delta, flagging the object-wise aggregation as the most impactful individual component.
- **Without transient Gaussians**: 26.87 PSNR / 0.063 LPIPS (−0.56 dB) — "significantly reduces fine details."
- **Without rigid Gaussians**: 27.11 PSNR / 0.055 LPIPS (−0.32 dB) — "causes large empty regions."
- **Without scene-flow supervision**: 27.27 PSNR / 0.052 LPIPS (−0.16 dB) — smallest measured contribution among the tested components.

#### Implementation Reality

- **Framework:** PyTorch/CUDA 12.x; rasterization via the `gsplat` library with a 2D Gaussian Splatting (2DGS) rendering backend; Adam optimizer for all Gaussian parameters and motion bases.
- **Key files (per repo structure):** `src/main.py` (entry point), `src/config/config.py` (configuration/Tyro CLI flags including `--num_fg`, `--num_motion_bases`, `--pose_opt`, `--test_time_pose_opt`), `src/run_viewer.py` (interactive viewer), `dependencies/vipe/` (modified ViPE submodule with integrated dynamic-mask prediction folded into preprocessing), `scripts/` (video/frame conversion and TAPIR/BootsTAPIR track-inference utilities).
- **Notable implementation details not fully specified in the paper text:** the public README does not itself name which rasterizer variant (gsplat vs. a custom 2DGS fork) is the default — this was corroborated from the paper text rather than the repo; the CLI exposes render-component selection (fused/static/dynamic/transient/rigid) and multiple camera-trajectory render modes (arc, lemniscate, spiral, wander, fixed) that are not discussed as contributions in the paper itself, suggesting they are visualization/evaluation tooling rather than method components.

#### Failure Modes & Limitations

The bimodal temporal-duration histogram is itself presented as evidence that a single rigid/basis representation "make[s] these methods less effective in modeling regions with large deformation or fast, non-rigid motion" — i.e., the two-population split is a direct response to a diagnosed, quantified failure mode rather than a hypothetical one. Monocular capture supplies "limited structural cues," which is why the method depends on external depth/normal/flow/track supervision rather than resolving geometry from photometric signal alone; failures in those upstream estimators (depth, flow, tracks, segmentation) would propagate directly since they are precomputed, not jointly refined.

## Relevance to ADAGS

Directly pressures ADAGS's reversible route0 plus residual-motion story. A publishable ADAGS variant should explain whether route0 corresponds to coherent rigid-ish motion, and whether scaffold/part residuals are reserved for local non-rigid failures.

## Connections

- Addresses [[gap_map#G2 - Static/Dynamic Leakage Is A Representation And Evaluation Problem]]
- Addresses [[gap_map#G6 - Single Global Motion Models Are A Known Weakness]]
- Pressures [[ideas/part-aware-reversible-routing]]

## Sources

- https://arxiv.org/abs/2605.23672
- https://github.com/ladvu/RiGS
