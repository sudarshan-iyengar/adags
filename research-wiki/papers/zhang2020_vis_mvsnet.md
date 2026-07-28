---
type: paper
node_id: paper:zhang2020_vis_mvsnet
title: "Visibility-aware Multi-view Stereo Network"
authors: ["Jingyang Zhang", "Yao Yao", "Shiwei Li", "Zixin Luo", "Tian Fang"]
year: 2020
venue: "BMVC"
external_ids:
  arxiv: "2008.07928"
  doi: null
  s2: null
tags: ["multi-view-stereo", "visibility", "uncertainty", "occlusion"]
added: 2026-07-14T22:18:30Z
status: deep-dived
---

# Visibility-aware Multi-view Stereo Network

**Paper:** https://arxiv.org/abs/2008.07928
**Code:** https://github.com/jzhangbs/Vis-MVSNet
**Base method:** Cascade/coarse-to-fine learned MVS networks (e.g. CasMVSNet-style pipelines) built on the MVSNet plane-sweep cost-volume paradigm (Yao et al. 2018), with pairwise (two-view) cost volumes instead of a single aggregated N-view volume.

## One-line thesis

A network can learn per-pixel matching uncertainty for free (no occlusion labels, just a depth-supervised Laplacian likelihood), and using that uncertainty as a per-pixel, per-source-view fusion weight suppresses the contribution of occluded/mismatched source views before they corrupt the aggregated cost volume.

## Problem / Gap

Learned MVS networks (MVSNet and its cascade successors) build the multiview cost volume by directly averaging or variance-pooling per-source-view matching costs across all `N` source views, treating occluded and visible source pixels identically. Because occluded pixels contribute wrong/noisy matching costs, naive aggregation gets *worse*, not better, as more source views are added — the paper explicitly shows the variance-based baseline's accuracy degrading with increasing `N_v`, whereas classical MVS pipelines explicitly estimate and exclude occluded views via visibility reasoning.

## Method

Vis-MVSNet keeps the MVSNet-style coarse-to-fine (3-stage) cost-volume backbone but replaces same-view aggregation with a two-step pairwise-then-fused pipeline. First, a 2D UNet extracts multiscale features per image; for each of `N_v` source views, a pairwise cost volume is built against the reference view via group-wise correlation and regularized independently, producing a pairwise depth map (via soft-argmax) and a pairwise per-pixel uncertainty jointly, trained with a Laplacian negative-log-likelihood loss. Second, the pairwise *latent* volumes (not the depth maps) are fused into one N-view volume using inverse-uncertainty weighting, so pairs with high predicted uncertainty (indicating occlusion or bad matches) contribute less to the fused volume. The fused volume is regularized again and regressed to the final depth map, and the whole stage repeats at three progressively finer resolutions, each stage's fused depth initializing the next.

## Assumptions

Calibrated, posed multiview images of a static scene at the reconstruction instant (standard MVS setup); depth ground truth is available for supervision, but no occlusion/visibility ground truth is required since uncertainty is learned implicitly through the depth-fitting likelihood.

## Limitations / Failure Modes

The paper's own ablation shows the "no coarse-to-fine" single-stage variant already improves over averaging/max-pooling baselines but the full 3-stage version is needed for the best numbers (L1 0.908 → 0.759 on BlendedMVS val), meaning the uncertainty-fusion idea alone is not the whole story — resolution refinement matters too. The method suppresses occluded source evidence at inference time but has no mechanism to track *which* surface was occluded over time, infer persistent hidden-surface state, remember it across frames, or allocate representation capacity to it — it only produces a single static-instant depth map per forward pass.

## Reusable Ingredients

- **Learned matching uncertainty without occlusion labels**: derive per-pixel uncertainty from the entropy/spread of the depth-hypothesis probability distribution, trained jointly with depth under a Laplacian NLL — no manual occlusion annotation needed.
- **Uncertainty-weighted fusion**: fuse multiple pairwise (or multi-source) latent representations by inverse-uncertainty weighting rather than uniform averaging, so unreliable/occluded evidence is automatically downweighted rather than hard-masked.
- **Pairwise-then-fuse decomposition**: compute per-source-pair cost volumes and depth/uncertainty independently before fusing, isolating each source view's failure (occlusion, mismatch) so it doesn't propagate into every other view's estimate.
- **Coarse-to-fine cascade with per-stage loss weighting**: three depth-hypothesis resolutions (32/16/8 hypotheses) with increasing per-stage loss weight (0.5/1.0/2.0), refining depth range progressively.

---

### Deep Dive

#### Core Novelty

Relative to prior aggregation strategies (simple averaging, variance pooling, or max pooling of per-view matching costs), Vis-MVSNet's change is to (1) predict a per-pixel *uncertainty* alongside each pairwise depth estimate via a joint Laplacian likelihood, and (2) use `1/uncertainty` as the fusion weight when combining pairwise latent volumes into the multiview volume. The key insight is that matching uncertainty is a good proxy for visibility/occlusion — a source pixel that is actually occluded produces a poor, high-entropy match against the reference view, which the network learns to flag as high-uncertainty even without ever being told "this pixel is occluded." This turns visibility reasoning into a byproduct of depth regression rather than a separately supervised task.

#### Mathematical Formulation

- Depth regression via soft-argmax over `N_d` depth hypotheses, evaluated per stage on the fused probability volume:
$$D_i = \sum_{j=1}^{N_d} d_j \cdot P_{i,j}$$
  where $D_i$ is the regressed depth at pixel $i$, $d_j$ is the $j$-th depth hypothesis, and $P_{i,j}$ is the softmax probability at that hypothesis.

- Matching entropy, computed per pixel from the pairwise probability distribution before it is converted to uncertainty:
$$H_i = \sum_{j=1}^{N_d} -P_{i,j} \log P_{i,j}$$

- Uncertainty is not regressed directly from entropy by a fixed formula; instead the network predicts a log-uncertainty $S_i = \log U_i$ directly (for numerical stability), jointly with depth, and the two are tied together by a Laplacian-likelihood loss (evaluated per source-view pair, at each of the 3 cascade stages, before fusion):
$$L_i^{\text{joint}} = \frac{1}{U_i}\left|D_i - D_{\text{gt},i}\right| + \log U_i$$
  This is the negative log-likelihood of the ground-truth depth under a Laplacian distribution centered at the predicted depth with scale $U_i$ — minimizing it simultaneously fits depth and calibrates uncertainty to the actual per-pixel error the pairwise match produces.

- Visibility-aware fusion of the pairwise latent cost volumes $V_i$ (indexed by source view $i$) into the fused multiview volume $V$, evaluated per pixel/voxel after all pairwise volumes are regularized, before the fused volume is regularized a second time and regressed to final depth:
$$V = \left(\sum_i \frac{1}{\exp(S_i)}\right)^{-1} \sum_i \frac{1}{\exp(S_i)} \cdot V_i$$
  Here $S_i = \log U_i$ is the predicted log-uncertainty for source view $i$ at that pixel, so $\exp(S_i) = U_i$ and the weight $1/U_i$ downweights high-uncertainty (likely occluded/mismatched) source views.

- Total training loss (Eq. 5), summed over the 3 cascade stages $k$ with stage weights $\lambda_1,\lambda_2,\lambda_3 = 0.5, 1.0, 2.0$:
$$L = \sum_{k=1}^{3} \lambda_k \left[L_{1,k}^{\text{final}} + \frac{1}{N_v}\sum_i \left(L_{1,k,i}^{\text{pair}} + L_{k,i}^{\text{joint}}\right)\right]$$
  combining the final fused-depth L1 loss, the per-pair depth L1 loss, and the per-pair joint (Laplacian) loss, at every stage.

#### Algorithm / Pipeline Changes

1. A shared 2D UNet extracts multiscale image features for the reference view and all `N_v` source views.
2. Per cascade stage `k` (3 stages, coarse to fine, `N_d,1/2/3 = 32/16/8` depth hypotheses): for each source view `i`, build a pairwise cost volume against the reference view via group-wise correlation over the current stage's depth hypotheses.
3. Regularize each pairwise cost volume independently (own 3D CNN regularization), then regress a pairwise depth map (soft-argmax, Eq. above) and a pairwise log-uncertainty $S_i$ jointly from the regularized pairwise volume/probability distribution.
4. Fuse all `N_v` pairwise latent volumes into one multiview volume via inverse-uncertainty weighting (fusion equation above) — this step replaces the naive variance-pooling/averaging aggregation used in prior cascade MVS networks.
5. Regularize the fused volume again (separate 3D CNN) and regress the final per-stage depth map via soft-argmax.
6. The refined depth map from stage `k` initializes the depth-hypothesis range/interval for stage `k+1`; repeat steps 2-5 at increasing resolution.
7. At inference (not training), post-process the final depth map with photometric and geometric consistency filtering across views, then apply median depth fusion before generating the output point cloud. Filtering uses a visibility-count threshold `N_f` and per-stage probability thresholds `p_t`.

#### Key Hyperparameters & Design Choices

- Optimizer: Adam, initial learning rate 0.001, halved at iterations 100k/120k/140k.
- Batch size: 2, trained on a single Nvidia GTX 1080Ti; 160k total iterations.
- Training input resolution: 640×512; training output resolution: 320×256.
- Depth hypothesis counts per stage: `N_d,1=32`, `N_d,2=16`, `N_d,3=8`.
- Per-stage loss weights: $\lambda_1=0.5$, $\lambda_2=1.0$, $\lambda_3=2.0$.
- Training data: BlendedMVS (113 scenes, 16,904 samples) plus the DTU training set.
- Inference on Tanks & Temples: input 1920×1080, `N_v=7` source views, filtering `N_f=4`, per-stage probability thresholds `p_t,1/2/3 = 0.8, 0.7, 0.8`.
- Inference on DTU: input 1600×1200, `N_v=5`, depth range [425mm, 905mm], filtering `N_f=2`, `p_t=0.6` at all stages.
- Uncertainty supervision requires no occlusion/visibility ground truth — it falls out of the joint Laplacian depth loss.

#### Ablation Summary

BlendedMVS validation set, `N_v=7` source views, lower L1/higher `<1%`/`<3%` accuracy is better:

| Fusion strategy | L1 | <1% acc | <3% acc |
|---|---|---|---|
| Variance-based baseline | 1.50 | 79.31 | 92.25 |
| Simple averaging | 0.999 | 83.03 | 94.95 |
| Max pooling | 0.956 | 84.71 | 95.19 |
| Proposed uncertainty fusion, single-stage (no coarse-to-fine) | 0.908 | 85.35 | 95.48 |
| **Full method (uncertainty fusion + 3-stage coarse-to-fine)** | **0.759** | **90.86** | **96.05** |

The single most impactful factor reported is the combination: uncertainty-weighted fusion alone (no coarse-to-fine) already beats all non-uncertainty baselines, but coarse-to-fine refinement adds a further large jump (L1 0.908 → 0.759, roughly another 16% relative reduction) on top of it. The paper also notes qualitatively that variance-based aggregation *degrades* as `N_v` increases, while the proposed fusion continues to improve — this trend, not just the absolute numbers, is presented as the central evidence for the visibility-aware design.

#### Implementation Reality

- **Framework:** PyTorch (tested on 1.4.0), custom cascade-MVS codebase (not a fork of a specific public MVSNet repo), using `apex` for synchronized batch norm, OpenCV, and Open3D for point-cloud I/O.
- **Key files:** `core/` holds the network (feature extraction, pairwise cost volume, uncertainty head, fusion module, regularization); `train.py`/`val.py`/`test.py` are the entry points; `fusion.py` implements the post-hoc photometric/geometric consistency filtering and depth fusion; `colmap2mvsnet.py` converts COLMAP camera parameters into the format the network expects.
- **Notable implementation details not obvious from the paper:** the public re-implementation explicitly differs from the original internal (Altizure) version used for the paper's reported numbers. Default inference hyperparameters in the repo (`max_d=256`, `--vthresh 4 --pthresh .8,.7,.8`, resize to roughly 1280×720) require manually ensuring `max_d * interval_scale` matches the depth count baked into the camera file — an easy-to-miss consistency requirement the paper doesn't discuss.

#### Failure Modes & Limitations

The paper reports that naive variance/averaging fusion baselines get systematically worse as the number of source views `N_v` increases, which is the failure mode it is explicitly designed to fix — but it does not report a case where its own uncertainty-based fusion breaks down (e.g. very sparse view counts, textureless regions, or extreme occlusion fractions are not analyzed as separate failure regimes in the accessible material).

---

## Relevance to This Project

It supplies the calibrated multiview visibility principle absent from R031-R033 and warns against indiscriminate camera aggregation.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Sources

- https://arxiv.org/abs/2008.07928
- https://github.com/jzhangbs/Vis-MVSNet
