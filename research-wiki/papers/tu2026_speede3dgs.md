---
type: paper
node_id: paper:tu2026_speede3dgs
title: "SpeeDe3DGS: Speeded-up Dynamic 3D Gaussian with Motion-aware Pruning and Spatial-temporally Consistent Densification"
authors: ["Xiaochen Tu", "Mingqiao Ye", "Yansong Tang", "Jingwei Huang", "Zilong Huang", "Zhiqiang Shen", "Yibo Yang", "Zhiyu Xiang"]
year: 2026
venue: "CVPR"
external_ids:
  arxiv: "2506.07917"
tags: [dynamic-gs, pruning, densification, efficiency]
status: deep-dived
---

> Note on metadata: the arXiv record for 2506.07917 lists the title as
> "SpeeDe3DGS: Speedy Deformable 3D Gaussian Splatting with Temporal Pruning
> and Motion Grouping" and authors Allen Tu, Haiyang Ying, Alex Hanson,
> Yonghan Lee, Tom Goldstein, Matthias Zwicker — not the title/author list
> currently in this file's frontmatter. Frontmatter is preserved verbatim per
> process rules; flagging the discrepancy for correction.

# SpeeDe3DGS: Speedy Deformable 3D Gaussian Splatting with Temporal Pruning and Motion Grouping

**Paper:** https://arxiv.org/abs/2506.07917
**Code:** https://github.com/tuallen/speede3dgs
**Base method:** Deformable 3D Gaussians (Yang et al., a shared MLP deformation field conditioned on position + time; [[papers/... ]] not yet in wiki) for the base representation, plus a pruning strategy adapted from Speedy-Splat (Hanson et al.) for the sensitivity-based removal criterion.

## One-line thesis

Per-Gaussian temporal-sensitivity gradients (aggregated over all training views and timesteps, and re-probed under synthetic timestamp jitter) identify which Gaussians and which deformation-MLP calls are actually load-bearing, letting most Gaussians be pruned and many of the survivors' per-frame neural deformations be replaced by a shared per-group rigid SE(3) transform — without a quality loss on held-out views.

## Problem / Gap

Deformation-MLP dynamic 3DGS methods (e.g. Deformable 3D Gaussians) query a neural field per-Gaussian at every frame to get position/rotation offsets, which captures complex non-rigid motion well but is computationally expensive: every rendered frame pays for a full MLP forward pass per surviving Gaussian, and standard 3DGS densification/opacity pruning is not aware of temporal redundancy, so many Gaussians end up encoding motion that contributes little to any frame's reconstruction loss, and no existing mechanism distinguishes genuinely non-rigid regions (where per-Gaussian neural deformation is worth its cost) from locally-rigid regions (where one shared transform per group would suffice).

## Method

Trains a standard Deformable-3DGS model, then at iteration 6,000 begins periodically pruning Gaussians using Temporal Sensitivity Pruning (TSP): an aggregated squared-gradient sensitivity score of the rendering loss with respect to each Gaussian's contribution, summed over all training poses and timesteps, with low-scoring Gaussians removed. Temporal Sensitivity Sampling (TSS) perturbs the timestamp fed into the deformation MLP with annealed Gaussian noise before computing this sensitivity, which surfaces "floater" Gaussians that look fine at exact training timestamps but are unstable at nearby unseen times. After densification ends, GroupFlow clusters the surviving Gaussians into a fixed number of trajectory groups (via farthest-point sampling of control points plus a trajectory-distance assignment), fits one rigid SE(3) transform per group per timestep (Umeyama alignment), and replaces the expensive per-Gaussian MLP deformation call with this shared per-group rigid transform for applicable regions.

## Assumptions

Assumes a monocular dynamic-scene setup already trained with a Deformable-3DGS-style neural motion field (position+time → offset MLP) as the base model; assumes scenes are evaluated on MonoDyGauBench-style benchmarks (NeRF-DS, D-NeRF, HyperNeRF, Nerfies, and other monocular dynamic NeRF scenes) where much of the motion is either near-static or locally rigid enough that a bounded number (J=2048) of rigid motion groups can approximate it.

## Limitations / Failure Modes

The paper reports "minor quality degradation... under extreme pruning ratios," and states GroupFlow's shared per-group rigid transform "preserves quality in most scenes through locally rigid motion, but highly deformable regions may lose fidelity when the number of motion groups is limited" — i.e., GroupFlow specifically degrades on non-rigid/topology-changing regions where no rigid group approximation fits. On the NeRF-DS ablation table, adding GroupFlow alone (no pruning) already costs some quality (PSNR 23.80→23.54, LPIPS 0.1781→0.1892 vs. the DeformableGS baseline), and the full TSP+TSS+GroupFlow combination trades a further slice of LPIPS (0.1781→0.1901) for a 10.68× FPS gain and 11.91× fewer Gaussians. The authors also note the method adds no explicit learned motion prior and is framed as "complementary to prior-driven or motion-aware 3DGS frameworks" rather than a replacement for them.

## Reusable Ingredients

- **Temporal Sensitivity Pruning (TSP):** aggregate squared gradient of rendering loss w.r.t. each Gaussian's rendered contribution, summed over all training views and timestamps, as a temporally-global importance score for pruning — generalizes single-frame opacity/gradient pruning to dynamic scenes.
- **Temporal Sensitivity Sampling (TSS):** inject annealed Gaussian noise into the timestamp input before computing sensitivity, to expose Gaussians that are stable at seen timestamps but unstable ("floaters") at nearby unseen ones.
- **GroupFlow motion grouping:** farthest-point-sampled control points + trajectory-distance assignment (blend of std and mean of per-timestep positional distance) to cluster Gaussians into rigid motion groups, then Umeyama-aligned per-group SE(3) fit per timestep — a cheap way to replace per-primitive neural deformation with shared rigid motion where locally valid.
- **Soft/hard pruning schedule split:** heavier pruning (60%) during active densification vs. lighter pruning (30%) after densification ends — separates exploratory capacity growth from final capacity trimming.

---

### Deep Dive

#### Core Novelty

Relative to Deformable-3DGS, this paper adds no new representation — it adds two orthogonal efficiency mechanisms applied post-hoc to an already-trained deformation-field model: (1) a temporally-aggregated, jitter-probed sensitivity score that finds Gaussians whose removal would not measurably change any training-time-conditioned render, letting most of the point cloud be pruned; and (2) a rigidity-clustering step (GroupFlow) that finds where the expensive per-Gaussian MLP call can be replaced by one shared rigid transform per cluster per frame, cutting the dominant per-frame inference cost. The key insight is that in typical monocular dynamic benchmarks most scene volume is either not visually load-bearing over time, or moves rigidly with a small number of coherent groups — so a per-Gaussian continuous deformation field is over-parameterized for most of the scene, and cheap post-hoc sensitivity/rigidity analysis can find and exploit that overparameterization without retraining the deformation field itself.

#### Mathematical Formulation

Temporal Sensitivity Pruning score (evaluated per Gaussian, aggregated over the full training set of poses and timesteps, used to rank Gaussians for pruning):
$$\tilde{U}_{\mathcal{G}_i} \approx \sum_{(\phi,t)\in\mathcal{P}_{gt}} \left(\nabla_{g_i} I_{\mathcal{G}_t}(\phi)\right)^2$$
where $I_{\mathcal{G}_t}(\phi)$ is the rendered image of the deformed Gaussians at time $t$ from training pose $\phi$, $g_i$ is Gaussian $i$'s contribution to the render, and $\nabla_{g_i}$ is obtained from the differentiable renderer's backward pass. Low-$\tilde{U}$ Gaussians are pruned.

Temporal Sensitivity Sampling perturbation (applied to the timestamp fed into the deformation MLP before computing the sensitivity score above, at training iteration $i$):
$$\mathcal{X}(i) = \mathcal{N}(0,1)\cdot\beta\cdot\Delta t\cdot\left(1-\frac{i}{\tau}\right)$$
with $\beta=0.1$ the perturbation magnitude, $\Delta t$ the frame interval, and $\tau=20{,}000$ the annealing period — the perturbation shrinks linearly to zero as training progresses.

GroupFlow trajectory-similarity assignment (evaluated once, after densification, to assign each Gaussian $i$ to a control point $j$):
$$S_{i,j} = \lambda_r \cdot \mathrm{std}_t\!\left(\lVert \mu_i^t - h_j^t \rVert\right) + (1-\lambda_r)\cdot \mathrm{mean}_t\!\left(\lVert \mu_i^t - h_j^t \rVert\right)$$
with $\lambda_r = 0.5$, $\mu_i^t$ the position of Gaussian $i$ at time $t$, and $h_j^t$ the position of control point $j$ at time $t$ (control points selected by farthest point sampling on Gaussian means at $t=0$). Each Gaussian is assigned to its lowest-$S$ control point.

GroupFlow per-group rigid transform fit (evaluated per group $j$, per timestep $t>0$, via Umeyama alignment over up to $N_{max}=100$ sampled group members):
$$\arg\min_{R_j^t, T_j^t} \sum_{\mu_i \in \mathcal{M}_{samp}^j} \left\lVert \mu_i^t - \left(R_j^t(\mu_i^0 - h_j^0) + h_j^0 + T_j^t\right)\right\rVert^2$$
The resulting $[R_j^t \mid T_j^t]$ is applied to both the position and rotation of every Gaussian in group $j$ at time $t$, replacing that Gaussian's per-frame MLP deformation call.

#### Algorithm / Pipeline Changes

1. Train Deformable-3DGS normally from iteration 0.
2. Starting at iteration 6,000, every 3,000 iterations: compute the TSP sensitivity score $\tilde{U}_{\mathcal{G}_i}$ per Gaussian by aggregating squared render-loss gradients over all training poses/timestamps (optionally with TSS timestamp jitter applied to expose floaters), then prune — soft pruning removes 60% of candidates while densification is still active, hard pruning removes 30% after densification ends (post iteration 15,000).
3. TSS is applied during the same sensitivity computation by adding $\mathcal{X}(i)$ to the timestamp before it is fed to the deformation MLP, so the sensitivity probe sees slightly-off-training-time motion states rather than only exact training timestamps.
4. After densification ends (iteration 15,000), run GroupFlow once: farthest-point-sample $J=2048$ control points from Gaussian means at $t=0$; assign every remaining Gaussian to a control point via the $S_{i,j}$ trajectory-similarity score; for each group and each subsequent timestep, solve the Umeyama rigid-alignment least-squares problem over up to 100 sampled member positions to get $[R_j^t\mid T_j^t]$.
5. At render time, apply each Gaussian's group's shared rigid transform for position/rotation instead of invoking the per-Gaussian deformation MLP, cutting the dominant per-frame neural-inference cost while keeping the (now much smaller) pruned Gaussian set.

#### Key Hyperparameters & Design Choices

- Total training iterations: 30,000 (test/eval at up to 40,000 per README).
- TSP start iteration: 6,000; TSP interval: every 3,000 iterations.
- Soft pruning ratio (during densification): 60%. Hard pruning ratio (after densification): 30%.
- TSS perturbation magnitude $\beta = 0.1$; annealing period $\tau = 20{,}000$.
- GroupFlow trajectory-similarity weight $\lambda_r = 0.5$.
- Number of GroupFlow motion groups $J = 2048$.
- Max samples per group for Umeyama fit $N_{max} = 100$.
- GroupFlow applied starting at iteration 15,000 (after densification ends).
- Loss weights for any novel loss terms: Not specified in paper (TSP/TSS are pruning/sampling procedures on the existing photometric loss, not new loss terms; GroupFlow is a closed-form alignment, not a learned/weighted loss).

#### Ablation Summary

NeRF-DS dataset, relative to DeformableGS baseline (PSNR 23.80, SSIM 0.8503, LPIPS 0.1781, 54.37 FPS, 132.22K Gaussians, 1523.83s train):
- **+TSP:** PSNR 23.78 (−0.02), 6.38× FPS, 12.13× fewer Gaussians, 2.05× faster training — pruning alone recovers most of the speedup with essentially no PSNR cost. **Single most impactful component** for the FPS/Gaussian-count reduction.
- **+TSP+TSS:** PSNR 23.81 (+0.01 vs. baseline), 6.35× FPS, 11.95× fewer Gaussians, 2.03× faster training — TSS gives a marginal quality bump (removing floaters) at essentially no added cost.
- **+GroupFlow alone (no pruning):** PSNR 23.54 (−0.26), LPIPS 0.1892 (worse), 8.58× FPS, no Gaussian reduction, 1.84× faster training — GroupFlow's own FPS gain comes from skipping per-Gaussian MLP calls, but it costs some quality on its own.
- **+TSP+TSS+GroupFlow (full method):** PSNR 23.66 (−0.14 vs. baseline), LPIPS 0.1901 (worst of all variants), 10.68× FPS, 11.91× fewer Gaussians, 2.44× faster training — the largest combined speedup, at a modest but nonzero quality cost concentrated in LPIPS/perceptual quality rather than PSNR.
- On the full MonoDyGauBench (50 scenes): pruning alone gives 6.78× rendering speedup at maintained quality; pruning + GroupFlow gives 13.71× rendering speedup and 2.53× training speedup.

#### Implementation Reality

- **Framework:** PyTorch (2.8.0, CUDA 12.8 in the published environment), built directly on top of the Deformable-3D-Gaussians codebase, with the pruning strategy adapted from the Speedy-Splat repo.
- **Key files:** `train.py` exposes the `--use_tss` flag for Temporal Sensitivity Sampling; `gaussian_renderer/` contains the `--gflow_flag`-controlled GroupFlow logic and the pruning strategy inherited from Speedy-Splat. Detailed algorithmic logic is not exposed in README-level documentation — file-level correspondence to specific equations was not independently verified beyond the flag names.
- **Notable implementation details:** the README frames pruning as reusing Speedy-Splat's pruning mechanism rather than a from-scratch implementation, which is not obvious from the paper's TSP description alone. Test-time evaluation iteration (40,000) exceeds the 30,000-iteration training schedule, suggesting evaluation checkpoints beyond nominal training length; this was not explained further in the sources checked.

#### Failure Modes & Limitations

The paper states quality degrades under "extreme pruning ratios," and separately that GroupFlow's rigid-per-group approximation loses fidelity in "highly deformable regions" when the fixed group count ($J=2048$) is insufficient to capture the true degrees of freedom of the motion — i.e., failure is concentrated in non-rigid/topologically-changing content, not in mostly-rigid or near-static regions. The ablation table shows this concretely: LPIPS is monotonically worse than the DeformableGS baseline at every combination that includes GroupFlow (0.1892 and 0.1901 vs. 0.1781 baseline), even though PSNR stays roughly flat — indicating the quality cost shows up as perceptual/structural degradation rather than raw pixel error, consistent with locally-rigid approximation of genuinely non-rigid motion.

## Relevance to ADAGS

Important comparator for fixed-budget and capacity-allocation claims.

## Connections

## Sources

- arXiv: https://arxiv.org/abs/2506.07917
- Project page: https://speede3dgs.github.io/
- Code: https://github.com/tuallen/speede3dgs

