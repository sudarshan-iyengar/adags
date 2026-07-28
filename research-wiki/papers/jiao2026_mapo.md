---
type: paper
node_id: paper:jiao2026_mapo
title: "MAPo: Motion-Aware Partitioning of Deformable 3D Gaussian Splatting for High-Fidelity Dynamic Scene Reconstruction"
authors: ["Han Jiao", "Jiakai Sun", "Yexing Xu", "Lei Zhao", "Wei Xing", "Huaizhong Lin"]
year: 2026
venue: "CVPR"
external_ids:
  arxiv: "2508.19786"
tags: [dynamic-gs, motion-partitioning, fast-motion]
status: deep-dived
---

# MAPo: Motion-Aware Partitioning of Deformable 3D Gaussian Splatting

**Paper:** https://arxiv.org/abs/2508.19786
**Code:** Not found. No official GitHub repository was located via the arXiv page, Papers With Code, or direct GitHub search. The paper states its implementation "builds upon the E-D3DGS codebase" with full details deferred to an appendix, but does not link a public release.
**Base method:** E-D3DGS (Embedding-based Dual Deformation 3D Gaussian Splatting) — learnable per-Gaussian embeddings combined with coarse and fine deformation networks. Rendering itself is standard 3DGS alpha-compositing.

## One-line thesis

A per-Gaussian dynamic score computed from historical trajectory spread drives two separate interventions: high-motion Gaussians get their temporal window recursively split and their deformation network duplicated per sub-window (so no single network has to fit conflicting motion patterns), while low-motion Gaussians are frozen to a single static attribute set and skipped from deformation inference entirely (cutting redundant compute).

## Problem / Gap

E-D3DGS and related deformation-MLP methods (D3DGS, 4DGaussians, Ex4DGS, DN-4DGS) use one canonical Gaussian set and one globally shared deformation network across the whole sequence, forcing that network to find parameters that fit all motion patterns simultaneously — this produces a temporal-averaging effect that blurs regions with complex or fast motion. Separately, these methods run every Gaussian through the deformation network at every frame even when a Gaussian's region is static, wasting compute. SWinGS addressed dynamics via coarse, window-level partitioning but needed heavy pre/post-processing and leaned on 2D priors like optical flow.

## Method

MAPo scores every 3D Gaussian's dynamic-ness from its recorded historical positions (max displacement and position variance, combined via a harmonic mean into a score in [0,1]). Gaussians whose score exceeds a level-specific threshold are recursively bisected in time: the Gaussian's temporal segment is split at the midpoint, the original keeps the first half and advances a partition level, a duplicate is created for the second half, and the deformation network is cloned into two segment-specific copies — repeating until the score drops below threshold or a max level is reached. Gaussians whose score falls below a separate static threshold are decoupled from the deformation network altogether: their attributes are baked once from a network evaluation at a random timestep, then optimized directly without further network calls. A cross-frame consistency loss is added near partition boundaries (within 5 frames) to prevent visible seams between adjacent temporal segments' Gaussian sets.

## Assumptions

Multi-view dynamic-scene video with dense camera coverage (20 cameras for N3DV, 13 for MeetRoom) at 30 FPS, sufficient temporal and spatial coverage to build a canonical Gaussian representation, and a training setup that already tracks per-Gaussian historical positions over many iterations (m = 300 recorded positions per Gaussian) before dynamic scores are computed.

## Limitations / Failure Modes

The authors report diminishing quality returns as max partition level increases past 3 (level 3 → 5 gains only ~0.06 dB PSNR on flame_salmon_frag3) while storage and training time keep climbing steadily (level 0: 44MB/1h13m vs level 5: 103MB/2h40m). Using only the current-frame consistency loss (`L_current`) without ground-truth supervision at boundaries lets adjacent-segment Gaussians converge to an over-smoothed, blurred compromise state rather than a sharp one. Partitioning also increases per-scene storage and training cost relative to the unpartitioned baseline (e.g. +21MB, and up to +tens of minutes training time depending on configuration).

## Reusable Ingredients

- **Trajectory-spread dynamic score** — normalizes per-Gaussian max-displacement and position-variance via dataset percentile ranks, then fuses with a harmonic mean so a Gaussian must score high on *both* signals to be classified dynamic; usable as a general per-primitive motion-intensity metric independent of MAPo's partitioning scheme.
- **Static-Gaussian deformation bypass** — freeze low-motion Gaussians' attributes after one network evaluation and skip them from all future deformation-network calls, directly reducing rendering/training compute for static content.
- **Recursive temporal bisection with network duplication** — a generic way to let capacity (here, a whole deformation network) scale locally with motion complexity rather than uniformly across the scene.
- **Boundary-local cross-frame consistency loss** — a two-term loss (segment-vs-segment agreement + ground-truth anchoring) applied only within a small frame window around partition seams, avoiding the cost of applying it scene-wide while still suppressing visible discontinuities.

---

### Deep Dive

#### Core Novelty

MAPo's change relative to E-D3DGS is architectural rather than a new loss or representation: it lets the *number of deformation-network instances* vary spatially and temporally based on measured motion intensity, instead of using one shared network for the whole scene and sequence. The key insight is that a single network's capacity is spent uniformly, so regions with genuinely conflicting motion (different objects/parts moving differently within the same temporal window) get under-fit and blur; splitting the temporal domain per-Gaussian lets each network specialize on a narrower, more consistent motion pattern, while Gaussians that never move at all are removed from the optimization problem's iterative cost entirely.

#### Mathematical Formulation

Per-Gaussian motion statistics from m = 300 recorded historical positions $\mu_{i1},\dots,\mu_{im}$:

$$r_i = \|\max_j \mu_{ij} - \min_j \mu_{ij}\|$$

Maximum spatial displacement across the recorded history — evaluated once per Gaussian before partitioning decisions are made.

$$v_i = \frac{1}{m}\sum_{j=1}^{m} \|\mu_{ij} - \bar\mu_i\|^2$$

Position variance around the mean position $\bar\mu_i$ — same evaluation point as $r_i$.

Percentile normalization (rank-based, so scores are dataset-relative rather than absolute):

$$\tilde r_i = \frac{1}{100}\sum_{k=1}^{100} \mathbb{1}(r_i \ge q_r(k)), \qquad \tilde v_i = \frac{1}{100}\sum_{k=1}^{100} \mathbb{1}(v_i \ge q_v(k))$$

where $q_r(k)$/$q_v(k)$ are the $k$-th percentile values of $r$/$v$ over all Gaussians.

Harmonic-mean fusion into the dynamic score, evaluated per Gaussian before the partition/static decision:

$$S_i = \frac{2}{\dfrac{1}{\tilde r_i+\epsilon} + \dfrac{1}{\tilde v_i+\epsilon}}, \qquad \epsilon = 10^{-6}$$

The harmonic mean penalizes Gaussians that are high on only one of the two signals, requiring both displacement and variance to be high for a high fused score.

Cross-frame consistency loss, applied only to training views within 5 frames of a partition boundary, added to the standard photometric loss:

$$L_{current} = \|I_t(G_t, V) - I_t(G_t', V)\|_1$$

Compares renderings of the same frame $t$ and viewpoint $V$ using the two adjacent temporal segments' Gaussian sets ($G_t$ from the segment that owns frame $t$, $G_t'$ from the neighboring segment) — enforces agreement between segments at the seam.

$$L_{gt} = \|I_t(G_t', V) - I^{GT}\|_1$$

Anchors the neighboring segment's rendering directly to ground truth, preventing the two segments from converging to a mutually-consistent but blurred compromise.

$$L_{cross} = 0.5 \cdot L_{current} + L_{gt}$$

Combined boundary loss term added to the overall training objective.

Base deformation call (inherited from E-D3DGS, unchanged by MAPo except that it is invoked once per temporal segment's own network copy instead of once globally):

$$(\Delta\mu, \Delta q, \Delta s, \Delta\alpha, \Delta_{sh}) = \mathcal{F}(z_g, z_t^c) + \mathcal{F}_\theta(z_g, z_t^f)$$

$z_g$ is the learnable per-Gaussian embedding, $z_t^c$/$z_t^f$ are coarse/fine temporal embeddings, $\mathcal{F}$/$\mathcal{F}_\theta$ are the coarse/fine deformation networks.

#### Algorithm / Pipeline Changes

1. During/after an initial training phase, record each Gaussian's position over $m = 300$ steps/frames to build its history $\{\mu_{ij}\}$.
2. Compute $r_i$, $v_i$, percentile-normalize to $\tilde r_i, \tilde v_i$, and fuse into dynamic score $S_i$ (harmonic mean) for every Gaussian.
3. Classify each Gaussian: if $S_i < \tau_{static}$, mark static; else if $S_i \ge \tau_l$ at the Gaussian's current partition level $l$, mark for partitioning; otherwise leave as-is.
4. For a Gaussian marked for partitioning at level $l$ covering $[t_{start}, t_{end}]$: split at $t_{mid} = (t_{start}+t_{end})/2$; the original Gaussian keeps $[t_{start}, t_{mid}]$ and advances to level $l+1$; a duplicate Gaussian is created for $[t_{mid}, t_{end}]$; the deformation network is cloned so segment $[t_{start}, t_{mid}]$ and segment $[t_{mid}, t_{end}]$ each get their own network instance (initialized from the pre-split network). Recurse up to a max level (ablated at 3, tested up to 5).
5. For a Gaussian marked static: evaluate the deformation network once at a randomly chosen timestep, bake the resulting $(\Delta\mu,\Delta q,\Delta s,\Delta\alpha,\Delta_{sh})$ into that Gaussian's attributes, and remove it from all future per-frame deformation-network forward passes — it remains optimizable directly (not frozen), just not re-derived from the network.
6. During rendering/training for any frame $t$ near a partition boundary (within 5 frames), additionally render with both the owning segment's Gaussians and the neighboring segment's Gaussians to compute $L_{current}$ and $L_{gt}$, and add $L_{cross}$ to the training loss.
7. Standard 3DGS alpha-compositing rasterizes the resulting (partitioned + static) Gaussian set per frame; no change to the rendering equation itself.

#### Key Hyperparameters & Design Choices

- Historical positions recorded per Gaussian: $m = 300$.
- Numerical stabilizer: $\epsilon = 10^{-6}$.
- Maximum partition level used for main results: 3 (chosen as the quality/cost knee; ablated 0–5).
- Level-specific dynamic thresholds $\tau_l$: not specified in paper (percentile-derived but exact cutoff values not given).
- Static threshold $\tau_{static}$: not specified in paper.
- Cross-frame consistency loss weights: 0.5 on $L_{current}$, 1.0 (implicit) on $L_{gt}$ — i.e. $L_{cross} = 0.5 L_{current} + L_{gt}$.
- Boundary window for applying $L_{cross}$: ±5 frames around each partition seam.
- Training details (optimizer, learning rates, iteration count): not specified in the extracted text — deferred to the paper's appendix, which was not accessible.

#### Ablation Summary

MeetRoom dataset, cumulative additions over the E-D3DGS baseline (PSNR 26.24 dB, SSIM 0.896, LPIPS 0.081):

- **+L_gt / full method (largest single PSNR gain): +0.48 dB** PSNR, +0.007 SSIM, −0.015 LPIPS over baseline — the single most impactful configuration reported, i.e. the complete method including ground-truth boundary anchoring.
- +Static-Gaussian partitioning alone: +0.36 dB PSNR, and notably *reduces* training time by 24 minutes and *increases* FPS by 2.33, showing the compute-saving effect is real, not just a quality effect.
- +Variance-based dynamic partitioning (max level 1.2, presumably a partial/early config): +0.39 dB PSNR.
- +Max-displacement-only partitioning: +0.28 dB PSNR (weakest of the individual signals, consistent with the harmonic-mean design rationale that a single motion signal is less reliable).
- +L_current alone: +0.25 dB PSNR, −18 min training time.
- Temporal partitioning alone increases the boundary-region temporal-consistency metric (tOF) from 0.074 to 0.084 (worse) until $L_current$/$L_gt$ are added, which bring it back down to 0.072 (better than baseline) — confirms that partitioning without the cross-frame loss introduces visible seam artifacts.
- Max partition level ablation (flame_salmon_frag3): PSNR rises monotonically from 29.93 (level 0) to 30.36 (level 5), but the level 3→5 gain (+0.06 dB) costs +33MB storage and +44 min training versus the 0→3 gain (+0.37 dB), motivating level 3 as the reported default.

#### Failure Modes & Limitations

The paper explicitly reports diminishing PSNR returns beyond partition level 3 while storage and training time continue to scale up (level 0: 44MB/1h13m/95.21 FPS → level 5: 103MB/2h40m/57.05 FPS, for only +0.43 dB total PSNR gain across all 5 levels). It also identifies an over-smoothing failure mode when relying on $L_{current}$ alone: without ground-truth anchoring at the boundary, adjacent temporal segments can optimize toward a mutually consistent but blurred state rather than a sharp, accurate one, which is why $L_{gt}$ is required in the final loss.

---

## Relevance to ADAGS

Directly overlaps with the claim that ADAGS should sharpen fast-moving cooking regions.

## Connections

## Sources

- https://arxiv.org/abs/2508.19786
