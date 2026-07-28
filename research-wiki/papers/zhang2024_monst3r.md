---
type: paper
node_id: paper:zhang2024_monst3r
title: "MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion"
authors: ["Junyi Zhang", "Charles Herrmann", "Junhwa Hur", "Varun Jampani", "Trevor Darrell", "Forrester Cole", "Deqing Sun", "Ming-Hsuan Yang"]
year: 2024
venue: "ICLR 2025"
external_ids:
  arxiv: "2410.03825"
  doi: null
  s2: null
tags: ["dynamic-geometry", "pointmaps", "video-depth", "camera-pose"]
added: 2026-07-14T22:18:30Z
---

# MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion

**Paper:** https://arxiv.org/abs/2410.03825
**Code:** https://github.com/Junyi42/monst3r
**Base method:** DUSt3R (Wang et al. 2024) pointmap regression, fine-tuned for per-timestep dynamic pointmaps; downstream pose/flow estimation follows DUSt3R's PnP-based pipeline.

## One-line thesis

Fine-tuning only DUSt3R's decoder and prediction heads (encoder frozen) on a small mix of dynamic-scene datasets makes a static pairwise pointmap regressor emit per-timestep pointmaps that stay valid under scene motion, and a lightweight sliding-window global optimization with a static-region flow-projection term then stitches these per-frame pointmaps and camera poses into a consistent video-scale point cloud — without ever explicitly modeling object motion or trajectories.

## Problem / Gap

DUSt3R and other static multiview stereo/pointmap methods assume a rigid scene between an image pair, so when a pair contains moving or deforming content the pairwise pointmap regression and the downstream pose solve are corrupted by the non-rigid pixels. Conversely, independent per-frame monocular depth (e.g. from a video-depth model) has no shared 3D coordinate frame across time, so nothing enforces that the same physical surface at two timesteps lands at consistent 3D coordinates. Concretely, DUSt3R's own confidence-weighted global alignment has no mechanism to distinguish "this region disagrees with rigid geometry because of camera-pose error" from "this region disagrees because it physically moved," so dynamic content silently degrades both the pointmap and the recovered camera trajectory.

## Method

MonST3R keeps DUSt3R's architecture (ViT encoder + cross-attention transformer decoder + pointmap/confidence heads) and fine-tunes only the decoder and heads on four datasets with ground-truth depth/pose and varying dynamic-content fractions (PointOdyssey, TartanAir, Spring, Waymo), sampling frame pairs at strides 1-9 weighted toward larger motion. At inference, each frame pair still produces per-pair pointmaps/confidences exactly as in DUSt3R, but the pointmaps are now trained to represent the scene "as it is at each timestep" rather than assuming a shared rigid structure. A sliding temporal window (default 9 frames) limits pairwise computation for long videos; within the window, camera pose/intrinsics are solved per frame via PnP+RANSAC on the predicted 2D-3D correspondences, and a confidence-thresholded comparison between camera-motion-induced flow and the network's estimated flow identifies static regions. A joint optimization then aligns all per-pair pointmaps into one world-coordinate video pointmap using DUSt3R's standard alignment loss plus two new terms: a camera-trajectory smoothness loss and a flow-projection loss that forces the global camera+geometry solution to reproduce optical flow specifically in the detected static regions.

## Assumptions

Assumes access to synchronized dynamic-scene training data with ground-truth depth and camera pose (used only for fine-tuning, not at inference) and, at inference, a monocular video where a reasonable fraction of each frame is static/rigid so the flow-projection term and static-region detector have signal to lock the camera trajectory to. Also assumes a small sliding window (default 9 frames) is sufficient to resolve local dynamic geometry and that camera motion is smooth enough for the trajectory-smoothness prior to be a valid regularizer.

## Limitations / Failure Modes

The small sliding window is explicitly reported as vulnerable to long-term occlusion, since a surface that leaves and re-enters the window has no mechanism carrying identity or geometry across the gap. The method struggles out-of-distribution on open-field/large-scale scenes with little rigid structure to anchor the static-region flow term, and foreground (dynamic) object geometry, while improved over DUSt3R, remains the harder and less accurate part of the reconstruction. The trajectory-smoothness loss assumes smooth camera motion and can fail under abrupt camera movement; dynamic (time-varying) intrinsics are only theoretically supported and are reported to need careful hyperparameter tuning to work at all.

## Reusable Ingredients

- **Decoder/head-only fine-tuning of a pretrained pointmap regressor**: adapts a static-scene geometry foundation model to dynamic scenes cheaply, preserving the frozen encoder's geometric prior while requiring only a modest, mixed synthetic/real dynamic dataset.
- **Camera-motion-vs-estimated-flow static-region detector**: a confidence-thresholded L1 comparison between the optical flow implied by camera motion alone and the network's own estimated flow, used to identify which pixels are safe to use as rigid/static evidence.
- **Flow-projection loss restricted to static regions**: rather than trusting flow everywhere, the global optimization only uses optical-flow agreement where the static-region mask says it's valid — a cheap way to keep dynamic content from corrupting camera-pose recovery.
- **Sliding-window global optimization for video-scale pointmaps**: bounds the O(frames²) pairwise cost of DUSt3R-style global alignment for long videos while still enforcing cross-frame consistency within each window.
- **Asymmetric multi-dataset sampling by dynamic-content fraction**: deliberately over/under-sampling datasets by how much scene motion they contain, rather than uniform mixing, to bias fine-tuning toward the harder dynamic case.

---

### Deep Dive

#### Core Novelty
Relative to DUSt3R, MonST3R's change is narrow and deliberately "simple": it does not add any motion-specific architecture (no scene-flow field, no per-point trajectory, no explicit object segmentation) but instead (1) fine-tunes DUSt3R's existing decoder/heads so the same pointmap representation stays valid per-timestep under motion, and (2) adds two optimization-time loss terms — trajectory smoothness and static-region-gated flow projection — to a slightly modified version of DUSt3R's existing global alignment. The insight is that DUSt3R's pointmap representation does not inherently require rigidity; what breaks it under motion is only the training distribution (all-rigid pairs) and the optimization's blind trust in flow/correspondence everywhere, both of which can be fixed without new machinery.

#### Mathematical Formulation

Pairwise pointmap/confidence prediction (per frame pair $(t, t')$, from the fine-tuned network, evaluated for every pair in the current sliding window before optimization):
$$\mathbf{X}^{t;tt'}, \mathbf{X}^{t';tt'} \in \mathbb{R}^{H\times W \times 3}, \qquad \mathbf{C}^{t;tt'}, \mathbf{C}^{t';tt'}$$
same output structure as DUSt3R; points are expressed in frame $t$'s camera coordinate frame.

Relative camera pose via confidence-filtered PnP (downstream of pointmap prediction, per frame pair):
$$R^*, T^* = \arg\min_{R,T} \sum_i \big\| \mathbf{x}_i - \pi\big(\mathbf{K}^{t'}(R\mathbf{X}_i^{t';tt'} + T)\big) \big\|^2$$
using only correspondences $i$ whose confidence $\mathbf{C}_i^{t';tt'}$ exceeds a threshold $\alpha$, solved with RANSAC for robustness.

Static-region mask (per frame pair, before global optimization), comparing camera-motion-only flow to the network's estimated flow with a smooth-L1 norm:
$$\mathbf{S}^{t\to t'} = \big[\, \alpha > \| F^{cam}_{t\to t'} - F^{est}_{t\to t'} \|_{L1} \,\big]$$

Global joint objective over the sliding window (evaluated during optimization, not at inference feed-forward time):
$$\hat{X} = \arg\min_{X, P_W, \sigma} \; \mathcal{L}_{align} + w_{smooth}\,\mathcal{L}_{smooth} + w_{flow}\,\mathcal{L}_{flow}$$

Alignment term (DUSt3R's original loss, unmodified in form), aligning each pairwise pointmap to the shared world pointmap $\mathbf{X}^t$ via a per-pair rigid transform $\mathbf{P}^{t;e}$ and scale $\sigma^e$:
$$\mathcal{L}_{align} = \sum \big\| \mathbf{C}^{t;e} \cdot (\mathbf{X}^t - \sigma^e \mathbf{P}^{t;e}\mathbf{X}^{t;e}) \big\|_1$$

Camera-trajectory smoothness term (new), penalizing rotation and translation change between consecutive frames:
$$\mathcal{L}_{smooth} = \sum_t \big( \| R^{t\top}R^{t+1} - I \|_F + \| T^{t+1} - T^t \|_2 \big)$$

Static-region flow-projection term (new), forcing the global camera/geometry solution's implied flow to match estimated flow only where the static mask is active:
$$\mathcal{L}_{flow} = \sum \big\| \mathbf{S}^{global} \cdot (F^{global}_{cam} - F^{est}) \big\|_1$$
enabled only once the average flow error drops below 20 (i.e. after the optimization has roughly converged), with the motion/static mask itself re-updated whenever per-pixel error exceeds 50.

#### Algorithm / Pipeline Changes
1. **Fine-tuning (offline, one-time)**: freeze the DUSt3R ViT encoder; fine-tune only the decoder and pointmap/confidence prediction heads on frame pairs (strides 1-9, weighted toward larger motion) drawn asymmetrically from PointOdyssey, TartanAir, Spring, and Waymo, with images downsampled to a 512px max dimension and random field-of-view/scale augmentation on center crops.
2. **Per-pair inference (replaces nothing — same as DUSt3R)**: for each frame pair in the current sliding window of size $w$ (default 9), run the fine-tuned network to get pointmaps and confidences in frame $t$'s coordinate frame.
3. **Pose/intrinsics extraction (same DUSt3R downstream step, now over dynamic-tolerant pointmaps)**: solve focal length per frame from the pointmap, and solve relative pose via confidence-filtered PnP+RANSAC.
4. **Static-region detection (new step, inserted before global optimization)**: compute flow implied by the estimated camera motion alone, compare to the network's estimated optical flow, and threshold to produce a per-pixel static mask $\mathbf{S}^{t\to t'}$.
5. **Sliding-window global optimization (modifies DUSt3R's global alignment)**: jointly optimize world pointmaps $X$, per-pair rigid poses $P_W$, and scales $\sigma$ over the window using $\mathcal{L}_{align} + w_{smooth}\mathcal{L}_{smooth} + w_{flow}\mathcal{L}_{flow}$; run for 300 iterations at learning rate 0.01 (~1 minute for 60 frames on a single GPU). The flow term activates once average flow error is below 20 and its static mask is refreshed when per-pixel error exceeds 50.
6. **Video depth extraction (new downstream output)**: read per-frame depth directly off the optimized global pointmaps, now parameterized consistently by per-frame camera pose and per-frame depth maps across the whole video.

#### Key Hyperparameters & Design Choices
- Fine-tuning: AdamW, learning rate $5\times10^{-5}$, 25 epochs, 20,000 image pairs/epoch, 2× RTX 6000 GPUs, ~1 day total.
- Frame-pair sampling stride: 1-9, weighted toward pairs with larger motion.
- Training data mix: PointOdyssey (~200k frames, ~50% dynamic), TartanAir (~1M frames, 0% dynamic), Spring (~6k frames, ~5% dynamic), Waymo (~160k frames, ~20% dynamic); sampling deliberately over-weights PointOdyssey and de-weights TartanAir/Waymo.
- Image resolution: downsampled to 512px max dimension.
- Fine-tuning strategy ablated as best: "decoder & heads" (vs. full-model or heads-only fine-tuning).
- Sliding window size $w = 9$ frames (default).
- Global optimization: 300 iterations, learning rate 0.01, loss weights $w_{smooth} = 0.01$, $w_{flow} = 0.01$; ~1 minute for a 60-frame video on a single GPU.
- Flow-projection loss activation threshold: average flow error $< 20$; static/motion mask refresh threshold: per-pixel error $> 50$.
- PnP correspondence confidence threshold $\alpha$: exact numeric value not specified in the extracted text.

#### Ablation Summary
(Table 5 region, components relative to the full method)
1. Training-data composition: all four datasets combined performs best; PointOdyssey + TartanAir + Spring is called out as the strongest subset combination — dynamic-heavy PointOdyssey is flagged as the most important single dataset.
2. Fine-tuning scope: "decoder & heads" fine-tuning outperforms fine-tuning the full network or only the heads.
3. Inference-time components: both the flow-projection loss and the trajectory-smoothness loss are reported to improve camera pose accuracy, and the static-region mask is called out as important to the flow term working correctly.
No single delta-metric ranking table was recoverable from the accessible text; the qualitative ranking above is as reported.

#### Implementation Reality
- **Framework:** PyTorch, extending the official DUSt3R codebase (ViT-Base CroCo-pretrained encoder + cross-attention decoder); optional custom CUDA kernels for RoPE positional embeddings (`croco/models/curope/`).
- **Key files:** `optimizer.py` implements the sliding-window global optimization (both a batchified and a non-batchified variant); `demo.py` is the main inference entry point; `viser/visualizer_monst3r.py` provides interactive 4D visualization.
- **Notable implementation details:** the non-batchified optimizer variant (added after the paper) reduces VRAM from ~33GB to ~23GB for a 65-frame 16:9 video, and a separate fully feed-forward real-time mode was added post-paper that trades quality for speed under a small-camera-motion constraint — neither is described in the paper text itself. Ground-truth dynamic masks can be substituted via a `--use_gt_mask` evaluation flag, meaning the released code's default dynamic-mask behavior is swappable in a way not detailed in the main paper.

#### Failure Modes & Limitations
Explicitly reported to be vulnerable to long-term occlusion because the small sliding window (default 9 frames) has no mechanism to carry geometry or identity for content that leaves and re-enters the window. Struggles on out-of-distribution scenes such as open fields with little rigid structure for the static-region flow term to lock onto. Foreground/dynamic-object geometry is improved relative to DUSt3R but remains less accurate than static/background geometry. The trajectory-smoothness prior assumes smooth camera motion and is expected to degrade under abrupt camera movement. Dynamic (per-frame-varying) camera intrinsics are only theoretically supported and reportedly require careful hyperparameter tuning to work in practice.

---

## Open Questions

Can calibrated N3V cameras make the alignment simpler and produce reliable surface visibility states?

## Claims

None yet.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

It is a direct alternative to R031's independently normalized, same-pixel temporal depth comparison.
