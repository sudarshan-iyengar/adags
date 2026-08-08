---
type: paper
node_id: paper:xiao2026_mope
title: "MoPe: Motion Permanence for Robust Monocular Gaussian Mapping in Dynamic Environments"
authors: ["Qixin Xiao"]
year: 2026
venue: "RSS 2026 Workshop"
external_ids:
  arxiv: "2606.29237"
tags: [gaussian-mapping, permanence, occlusion, slam]
status: deep-dived
---

# MoPe: Motion Permanence for Robust Monocular Gaussian Mapping in Dynamic Environments

**Paper:** https://arxiv.org/abs/2606.29237
**Code:** https://github.com/chloeqxq/MoPe
**Base method:** WildGS-SLAM — monocular Gaussian-splatting SLAM that replaces binary dynamic masking with a continuous, per-pixel, per-frame uncertainty field predicted from DINOv2 features (tracking front end derived from DROID-SLAM; monocular depth from Metric3D V2). MoPe extends WildGS-SLAM with a temporal-memory layer; it does not replace the uncertainty predictor itself.

## One-line thesis
MoPe replaces WildGS-SLAM's per-frame, memoryless dynamic-uncertainty estimate with a persistent per-pixel Bayesian log-odds state that is propagated forward each frame via SE(3)+depth warping and fused with new evidence, so a region keeps its "dynamic" identity across occlusion or a temporary pause instead of collapsing back to "static" the instant a single frame's evidence looks static.

## Problem / Gap
Uncertainty-weighted monocular Gaussian SLAM systems such as WildGS-SLAM predict a continuous per-pixel uncertainty field from DINOv2 features independently at every frame, with no memory across frames. When a pedestrian pauses or is briefly occluded, that frame's evidence looks static, so the memoryless weighting immediately treats the region as static and lets it re-enter tracking residuals and be inserted into the map — producing ghosting/duplicate geometry once the object moves again or is revealed. Prior temporal approaches either operate on discrete object-track/segmentation masks (DGS-SLAM, DyGS-SLAM; RGD-SLAM's Extended Kalman Filter confined to segmentation masks) or inject only a timestamp positional embedding for view synthesis (UP-SLAM's "temporal encoder"), rather than propagating an explicit continuous dynamic-identity state; VarSplat bakes uncertainty into the map only as a passive byproduct.

## Method
MoPe converts WildGS-SLAM's per-pixel uncertainty β_t(u) into a bounded confidence weight and an uncertainty-based dynamic-ness measurement, optionally combined (via max) with a semantic-prior measurement from an off-the-shelf segmentation model to get a per-pixel measurement m_t(u). The previous frame's log-odds dynamic-posterior map is warped into the current frame via projective SE(3)+depth warping (relative camera pose, current depth), which only propagates to pixels with a valid correspondence — this is the mechanism's occlusion handling. The warped prior is fused with the new measurement in log-odds space with a fixed gain and clipped to a bound, then converted back to a probability (the persistent "dynamic posterior"). This posterior then down-weights tracking residuals, reweights the mapping loss, gates whether new Gaussians are inserted at high-dynamic-posterior pixels (fail-closed insertion), and — at keyframes — drives progressive opacity decay of existing Gaussians with high accumulated dynamic-observation history.

## Assumptions
Monocular RGB input with a metric/scale-consistent depth estimate (Metric3D V2 in the released code) and camera poses from a tracking front end (DROID-SLAM-derived); assumes depth+pose-based per-pixel warping is accurate enough to propagate a belief state across frames; assumes the underlying WildGS-SLAM DINOv2 uncertainty predictor as a given, unmodified input signal; the semantic prior is optional and assumes access to an off-the-shelf segmentation model producing per-pixel/mask scores for likely-transient classes (e.g., person).

## Limitations / Failure Modes
The paper reports the method trades completeness for cleanliness: suppressing/gating insertion in uncertain regions reduces Gaussian insertion density and slightly lowers PSNR/SSIM on some sequences, with explicit regressions on ANYmal2 (a continuously walking/moving robot), Stones, and Table1 (near-static clutter) — "regimes in which a persistent dynamic prior offers little benefit." The SE(3)+depth propagation step "also relies on reasonably accurate depth," so depth error directly degrades the warp used to carry the log-odds posterior forward.

## Reusable Ingredients
- Bounded log-odds Bayesian fusion (an occupancy-grid-style update) applied to a per-pixel *dynamic/semantic* state instead of geometric occupancy — a general pattern for giving any per-frame classifier persistent memory across frames.
- Depth+pose projective warping as the propagation operator for a persistent belief map, where pixels lacking a valid correspondence simply receive no propagated prior (occlusion handled by omission, not explicit modeling).
- Fail-closed insertion gating: withhold new-primitive creation wherever a persistent dynamic/uncertainty score exceeds a threshold, rather than filtering primitives after they are created.
- Accumulated per-primitive "dynamic-observation ratio" (dynamic observations / total valid observations across keyframes) as a criterion for post-hoc, gradual opacity decay rather than hard deletion.

---

### Deep Dive

#### Core Novelty
Relative to WildGS-SLAM's per-frame DINOv2-based uncertainty field, MoPe's only structural addition is a persistent state variable — a per-pixel log-odds map — that is warped forward each frame and Bayesian-fused with new evidence, plus four consumption points (tracking weight, mapping loss, insertion gate, opacity decay) that read this persistent posterior instead of the raw instantaneous uncertainty. The key insight: "dynamic-ness" is a property of an object's motion *history*, not of a single frame's appearance, so a momentarily static-looking frame should not immediately overwrite an accumulated dynamic belief. Log-odds fusion with a bounded clip gives exactly this: slow, evidence-weighted updates that still permit eventual recovery to "static" once motion evidence is genuinely gone.

#### Mathematical Formulation

$$w_t(u) = \text{clip}\left(\frac{0.5}{\beta_t(u)^2},\ w_{min},\ 1\right) \quad \text{[Eq. 1]}$$
Converts WildGS-SLAM's inherited per-pixel uncertainty $\beta_t(u)$ (DINOv2-feature-based, unmodified from the baseline) into a bounded confidence weight in $[w_{min}, 1]$. Evaluated per-pixel, per-frame, before fusion.

$$d_t^{unc}(u) = 1 - w_t(u) \quad \text{[Eq. 2]}$$
Turns the confidence weight into an uncertainty-based dynamic-ness measurement.

$$m_t(u) = \max\left(d_t^{unc}(u),\ d_t^{sem}(u)\right) \quad \text{[Eq. 3]}$$
Combines the geometric/appearance-uncertainty measurement with an optional semantic-prior measurement $d_t^{sem}(u)$ (a soft floor derived from an off-the-shelf segmentation model's transient-class likelihood, not a hard replacement for geometry). Either signal alone can raise the combined measurement.

$$\tilde p_{t-1\to t}(u) = \mathcal{W}\left(p_{t-1};\, T_{t-1\to t},\, D_t\right)(u) \quad \text{[Eq. 4]}$$
Projective SE(3)+depth warp of the previous frame's dynamic-posterior probability map $p_{t-1}$ into the current frame, using relative camera pose $T_{t-1\to t}$ and current depth $D_t$. Evaluated once per frame, before fusion. Pixels without a valid warp source (occluded or out-of-frame in the previous view) receive no propagated prior — this is the paper's occlusion-handling mechanism: $\mathcal{W}$ simply skips propagation for invalid correspondences rather than explicitly modeling occlusion, so occluded regions fall back to current-frame measurement (and clip-bounded decay) only.

$$L_t(u) = \text{clip}\left(\tilde L_{t-1\to t}(u) + \gamma \cdot \text{logit}(m_t(u)),\ -c,\ c\right) \quad \text{[Eq. 5]}$$
$$p_t(u) = \sigma(L_t(u)) \quad \text{[Eq. 6]}$$
Bounded log-odds Bayesian fusion: the warped prior probability is converted to log-odds $\tilde L_{t-1\to t}$, a gain-scaled ($\gamma$) logit of the new measurement is added, and the result is clipped to $[-c, c]$ to prevent saturation. $L_t(u)$ — equivalently $p_t(u)$ after the sigmoid — **is the persisted state**: a single log-odds (or probability) scalar per pixel, carried forward to frame $t+1$ as $p_{t-1}$ in Eq. 4. The clip bound $c$ is what allows a region to recover to static once motion evidence stops, while preventing a single misleading frame from erasing accumulated history.

$$\tilde\omega_t(u) = (1 - p_t(u)) \cdot \omega_t(u) \quad \text{[Eq. 7]}$$
Down-weights the tracking residual weight $\omega_t(u)$ by the complement of the dynamic posterior. Evaluated inside pose-tracking optimization each frame.

$$\mathcal{L}_{map} = \sum_u \frac{\mathcal{L}_{rgb}(u) + \lambda_d \cdot \mathcal{L}_{depth}(u)}{\tilde\beta_t(u)^2} + \lambda_u \cdot \mathcal{L}_{unc} \quad \text{[Eq. 8]}$$
Mapping objective, per-pixel-weighted by a temporally-informed uncertainty $\tilde\beta_t(u)$ (a persistent-posterior-modulated version of $\beta_t$); $\lambda_d, \lambda_u$ are loss weights for the depth and uncertainty terms. Evaluated as the photometric/geometric optimization objective for Gaussian parameters. $\mathcal{L}_{unc}$'s exact form is not defined in the accessible paper text.

$$s_t(u) = \max\left(p_t(u),\ \eta \cdot M_t^{sem}(u)\right) \quad \text{[Eq. 9]}$$
$$M_t^{keep}(u) = \mathbb{1}\left[s_t(u) < \tau_{ins}\right] \quad \text{[Eq. 10]}$$
Insertion score combining the dynamic posterior with a down-weighted ($\eta$) semantic support mask $M_t^{sem}(u)$ (a per-pixel transient-object-likelihood mask; not formally defined beyond this use in the accessible text). The binary keep-mask allows new Gaussians to be spawned only at pixels below the insertion threshold $\tau_{ins}$ — a fail-closed gate evaluated before Gaussian insertion/densification at each mapping step.

$$r_i^{dyn} = h_i^{dyn} / h_i^{obs} \quad \text{[Eq. 11]}$$
Per-Gaussian accumulated dynamic-observation ratio across keyframes: $h_i^{dyn}$ = count of observations of Gaussian $i$ flagged dynamic, $h_i^{obs}$ = total valid observations of Gaussian $i$. Evaluated by accumulating evidence at each keyframe.

$$\alpha_i \leftarrow \max(\rho \cdot \alpha_i,\ \alpha_{min}) \quad \text{[Eq. 12]}$$
Progressive geometric opacity decay applied, at post-cleanup/keyframe steps, to Gaussians that have both sufficient multi-view support and a high accumulated dynamic ratio $r_i^{dyn}$; $\rho$ is the decay factor and $\alpha_{min}$ a floor that avoids fully deleting the primitive outright.

#### Algorithm / Pipeline Changes
1. At each new frame $t$, compute WildGS-SLAM's per-pixel uncertainty $\beta_t(u)$ unchanged (DINOv2-feature-based prediction from the baseline).
2. Convert to bounded confidence weight $w_t(u)$ and uncertainty-dynamic measurement $d_t^{unc}(u)$ (Eq. 1–2).
3. If the semantic prior is enabled, compute $d_t^{sem}(u)$ from an off-the-shelf segmentation model and combine via max to get $m_t(u)$ (Eq. 3).
4. Warp the previous frame's persistent posterior $p_{t-1}$ into frame $t$ using the relative pose $T_{t-1\to t}$ and current depth $D_t$ via projective warp $\mathcal{W}$ (Eq. 4); pixels lacking valid correspondence get no propagated value.
5. Fuse warped log-odds with the gain-scaled logit of $m_t(u)$, clip to $[-c,c]$ to get $L_t(u)$ (Eq. 5); convert to $p_t(u)$ via sigmoid (Eq. 6). This map is the state carried to frame $t+1$.
6. Tracking: down-weight per-pixel residuals by $(1-p_t(u))$ before pose optimization (Eq. 7).
7. Mapping: weight the photometric+depth loss per-pixel by $\tilde\beta_t(u)$ and add the uncertainty loss term (Eq. 8).
8. Gaussian insertion: compute insertion score $s_t(u)$ (Eq. 9) and keep-mask $M_t^{keep}(u)$ (Eq. 10); new Gaussians are spawned/densified only where the mask is 1 — insertion is fail-closed (default is *not* to insert) rather than filtered post hoc.
9. At keyframes, for each existing Gaussian $i$, accumulate dynamic/total observation counts across re-observations and compute $r_i^{dyn}$ (Eq. 11).
10. Post-cleanup: for Gaussians with sufficient multi-view support and high $r_i^{dyn}$, apply geometric opacity decay (Eq. 12), progressively attenuating — not immediately deleting — persistent dynamic Gaussians.

#### Key Hyperparameters & Design Choices
- $\gamma$ (log-odds measurement gain): Not specified in paper — stated only that it is "held fixed across all sequences, with no per-sequence tuning."
- $c$ (log-odds clip bound): Not specified in paper.
- $\tau_{ins}$ (insertion threshold): Table VI sweeps $\tau_{ins} \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$ and reports $\tau_{ins} \in [0.3, 0.7]$ as the best-performing range, but the single operating value actually used is not stated.
- $\rho$ (opacity decay factor), $\eta$ (semantic-floor weight), $w_{min}$ (min confidence-weight clip): Not specified in paper.
- $\lambda_d$, $\lambda_u$ (mapping loss weights for depth and uncertainty terms, Eq. 8): Not specified in paper.
- $\mathcal{L}_{unc}$ functional form: Not specified in the accessible paper text.
- $\beta_t(u)$ / DINOv2 uncertainty-network architecture: Inherited unmodified from the WildGS-SLAM baseline; not re-derived or re-specified in this paper.
- Released code (`configs/wildgs_slam.yaml`, `mapping.uncertainty_params.temporal_params`) exposes a *different* parameterization than the paper's symbols — see Implementation Reality below for the actual shipped values.

#### Ablation Summary
On the iPhone_wandering sequence (Mean/Tail BG PSNR↑, Mean/Tail BG MAE↓):

| Variant | Mean PSNR | Mean MAE | Tail PSNR | Tail MAE |
|---|---|---|---|---|
| Temporal Fusion only | 18.364 | 19.365 | 18.026 | 20.862 |
| + Semantic Prior | 18.396 | 19.367 | 18.102 | 20.649 |
| + Insertion Gating | 18.371 | 19.405 | 18.072 | 20.759 |
| + Opacity Decay | 18.354 | 19.441 | 18.045 | 20.829 |
| **Full MoPe** | **18.410** | **19.183** | **18.418** | **20.021** |

Only the full combination beats "Temporal Fusion only" on every metric; each single add-on in isolation sometimes performs *worse* than temporal fusion alone on at least one metric (e.g., "+ Opacity Decay" has the worst Mean PSNR of the table). The paper explicitly frames this as components being "complementary rather than independently additive" — no single component can be flagged as the most impactful from this table, and this ablation does not include a true no-persistence baseline (that comparison instead appears in the main-results tracking-accuracy gains: ATE RMSE improves 17.4% on Wild-SLAM, 15.6% on Bonn, 15.2% on TUM relative to the non-temporal baseline).

#### Implementation Reality
- **Framework:** PyTorch, extending the WildGS-SLAM codebase, which itself integrates DROID-SLAM (tracking), Metric3D V2 (monocular depth), and `diff-gaussian-rasterization` (rendering).
- **Key files:**
  - `src/utils/dyn_uncertainty/temporal_fusion.py` — implements the SE(3)-consistent warping and log-odds fusion (Eq. 4–6).
  - `src/depth_video.py` — integrates the temporal posterior into pose tracking (Eq. 7) with Bayesian averaging and semantic person-prior support.
  - `src/mapper.py` — memory-aware mapping: insertion-mask gating (Eq. 9–10) and Gaussian-level cleanup via evidence accumulation and opacity decay (Eq. 11–12).
- **Notable implementation details not in the paper:**
  - Temporal fusion is not an unconditional pipeline change — it is gated behind independent boolean flags (`tracking_use_temporal_posterior`, `mapping_use_temporal_posterior`, `person_prior_activate`, `insertion_mask_activate`, `post_cleanup_opacity_decay_activate`), all `False` in the root config (`configs/wildgs_slam.yaml`) and enabled per-scene in the `Dynamic`/`Custom` config subdirectories.
  - The config exposes `fusion_mode: log_odds` as a named option, implying the codebase supports (or once supported) alternative fusion modes beyond the one equation set the paper presents.
  - The shipped default `temporal_params` block uses a hysteresis-style parameterization that does **not** map cleanly onto the paper's single gain/clip ($\gamma$, $c$) symbols: `decay: 0.95`, `on_threshold: 0.55`, `off_threshold: 0.35`, `release_threshold: 0.60`, `evidence_decay: 0.90`, `min_parallax_px: 1.5`, `min_weight: 0.05`. This suggests the released implementation is a superset or variant of the log-odds formulation described in the text, with separate on/off/release thresholds and a `min_parallax_px` warp-validity gate not mentioned in the paper.
  - License: Apache 2.0. At time of check: ~3 stars, ~15 commits, with a populated README covering installation, quick-start demos, and Wild-SLAM MoCap benchmark reproduction.

#### Failure Modes & Limitations
The paper states the method "buys temporal consistency at the price of conservatism": suppressing or skipping uncertain regions reduces Gaussian insertion density and "can... slightly lower... PSNR and SSIM on some sequences." It names explicit regressions on ANYmal2 (a continuously walking robot), Stones, and Table1 (near-static clutter) — "regimes in which a persistent dynamic prior offers little benefit." It also states the propagation step "relies on reasonably accurate depth," i.e., depth error directly degrades the SE(3)+depth warp that carries the log-odds posterior forward. Future work is scoped to adaptive Bayesian parameters and lightweight attention mechanisms to handle depth error and unstructured environments.

---

## Relevance to ADAGS

Persistent-state-through-occlusion in Gaussian mapping — in spirit the
nearest 2026 permanence mechanism, but the persisted quantity is a
dynamicness classification for mapping robustness, not surface
geometry/appearance for reconstruction. Cite and differentiate.

## Connections

- Pressures [[gap_map#G13 - Visibility Events Are Not Smooth Deformation]]

## Sources

- https://arxiv.org/abs/2606.29237
