---
type: paper
node_id: paper:wu2026_consistent_instance_field
title: "Consistent Instance Field for Dynamic Scene Understanding"
authors: ["Junyi Wu", "Van Nguyen Nguyen", "Benjamin Planche", "Jiachen Tao", "Changchang Sun", "Zhongpai Gao", "Zhenghao Zhao", "Anwesa Choudhuri", "Gengyu Zhang", "Meng Zheng", "Feiran Wang", "Terrence Chen", "Yan Yan", "Ziyan Wu"]
year: 2026
venue: "CVPR 2026 (reported; arXiv Dec 2025)"
external_ids:
  arxiv: "2512.14126"
tags: [dynamic-gs, instance-identity, occupancy, existence-vs-visibility]
status: deep-dived
---

# Consistent Instance Field for Dynamic Scene Understanding

**Paper:** https://arxiv.org/abs/2512.14126
**Code:** Not found (no GitHub link on arXiv abstract page, no HTML/TeX source repo pointer, no PapersWithCode or Hugging Face Papers entry located)
**Base method:** Deformable 3D Gaussian Splatting with a time-conditioned MLP deformation field, following SC-GS (Shi et al., CVPR 2024) and Per-Gaussian Embedding-Based Deformation (Bae et al., ECCV 2024), with the persistent-primitive framing following Dynamic 3D Gaussians / "tracking by persistent dynamic view synthesis" (Luiten et al., 3DV 2024).

## One-line thesis
Factoring each space-time point's instance-membership probability into a separate occupancy term `π(x,t)` and a conditional identity distribution `p(x,t,k)` — rather than tying identity directly to per-Gaussian radiance/opacity as prior instance-Gaussian methods do — removes the visibility bias that corrupts identity estimates for partially-occluded or momentarily-invisible instances.

## Problem / Gap
Prior instance/panoptic Gaussian-splatting methods (e.g., Gaussian Grouping, SA4D, VLGS-style approaches) attach identity labels or semantic features directly to each Gaussian's rendered radiance/opacity channel, so a Gaussian's contribution to an instance's identity estimate is entangled with how often and how strongly it happens to be visible. The paper states this directly: prior works "tie identity to radiance and remain vulnerable to visibility bias" without "explicitly modeling persistent object existence in space-time." Concretely, a rarely-visible or frequently-occluded Gaussian belonging to instance k gets systematically under-weighted in identity aggregation relative to a Gaussian that happens to face the camera often, degrading instance consistency across views and time.

## Method
Each Gaussian primitive carries the usual geometric/appearance attributes (position, rotation, scale, color, opacity) plus two new per-primitive quantities: an occupancy probability `π_i` and a K-way identity distribution `(p_i^1, ..., p_i^K)`. Instance masks are rendered per-pixel by alpha-compositing `π_i · P_i(u,v,t) · p_i^k` along the ray (an instance-specific transmittance/rendering equation separate from the RGB rendering equation), and trained against pseudo-ground-truth 2D instance masks produced automatically by the DEVA video tracker (no manual annotation). A visibility-biased raw identity estimate is computed per Gaussian by aggregating normalized per-pixel rendering weights against the GT mask label across all frames/views, then a learned per-Gaussian, per-instance calibration factor `m_i^k` reweights this estimate to correct for the fact that some Gaussians are seen far less than others. Periodically during training, "Instance-Guided Resampling" duplicates Gaussians in semantically active regions (using the joint occupancy-identity response `γ_i^k = π_i · p_i^k` to decide where to add capacity) while splitting the parent's opacity and occupancy volume-preservingly among the new copies.

## Assumptions
Multiview or monocular RGB video of a dynamic scene with roughly persistent (non-amorphous) object instances that can be represented by a bounded, evolving set of deformable Gaussian primitives; the method assumes DEVA-derived 2D instance masks are available (or automatically generated) per frame/view as pseudo-ground-truth supervision, and that a fixed camera-calibrated Gaussian-splatting reconstruction pipeline (deformable-Gaussian + time-conditioned MLP) is already in place as the geometry/appearance backbone.

## Limitations / Failure Modes
The paper's own limitations section (Appendix C.1) states that "scenes involving amorphous or continuously evolving materials (e.g., smoke or liquids) lack stable structure and may not be faithfully represented through persistent Gaussian primitives" — i.e., the persistent-identity assumption breaks down for non-rigid, topologically unstable matter. It also notes that "residual cross-view inconsistencies or missing annotations under severe occlusion can still bias identity estimation," meaning the calibration mechanism does not fully eliminate visibility bias when occlusion is severe or DEVA pseudo-masks are wrong/missing. The ablation shows removing calibration is the single most damaging change (see Ablation Summary), confirming visibility bias remains a live failure mode without it.

## Reusable Ingredients
- **Occupancy/identity factorization** (`γ = π · p`): separates "is something here" from "which instance is it," letting a system reason about existence and identity independently instead of conflating them with opacity/radiance.
- **Visibility-bias calibration via learned per-primitive factors** (`m_i^k`): a lightweight, trainable correction that reweights aggregated per-primitive evidence to counteract systematic under-sampling of rarely-visible primitives — reusable anywhere per-primitive statistics are aggregated unevenly across views/frames.
- **Volume-preserving opacity/occupancy split on resampling** (`α_new = 1-(1-α)^{1/(n+1)}`): a general recipe for cloning a primitive into n+1 children without changing the primitive's net contribution to the rendering integral.
- **Automatic pseudo-label pipeline via DEVA**: removes manual annotation from the instance-supervision loop, usable as a drop-in mask source for any Gaussian-splatting instance/panoptic task.

---

### Deep Dive

#### Core Novelty
Relative to prior instance-aware Gaussian methods that store a single identity/semantic vector per Gaussian directly coupled to its rendered opacity, this paper adds an explicit second probabilistic axis — occupancy `π(x,t)` — so that "this point is occupied" and "this point belongs to instance k" are two separate learned quantities whose product gives the instance-rendering weight. The key insight is that this factorization is what makes visibility-bias calibration mathematically well-posed: the calibration step (Eq. 8) operates on the identity distribution `p` after removing the occupancy/visibility component, rather than trying to reweight a single entangled radiance-identity signal.

#### Mathematical Formulation
- **Joint occupancy-identity distribution** (Eq. 1-2), evaluated per space-time point/per-Gaussian:
  $$\gamma(\mathbf{x},t,k) = P(E=1, K=k \mid \mathbf{x},t) = \pi(\mathbf{x},t)\cdot p(\mathbf{x},t,k)$$
  `π(x,t) ∈ [0,1]` is the occupancy probability (something physically exists at this point/time); `p(x,t,k)` is a conditional distribution over the K instance identities with `Σ_k p(x,t,k) = 1`; `γ` is the joint probability that instance k occupies that point.

- **Per-pixel instance rendering** (Eq. 4), evaluated at rasterization time, one pass per output instance channel:
  $$\mathbf{M}_k(u,v,t) = \sum_i T_i^{\text{inst}}(u,v,t)\cdot \pi_i \cdot P_i(u,v,t)\cdot p_i^k$$
  where `P_i(u,v,t)` is the Gaussian's projected 2D density/kernel weight at pixel `(u,v)` and time `t`, and `T_i^{inst}` is the instance-specific transmittance:
  $$T_i^{\text{inst}}(u,v,t) = \prod_{j<i}\left(1-\pi_j P_j(u,v,t)\right)$$
  Note this alpha-compositing uses `π_j` (occupancy) in place of the usual opacity `α_j`, i.e., a parallel compositing pass to the RGB one.

- **Instance Identity Estimation — visibility-biased raw estimate** (Eq. 5-7), computed by aggregating rendering statistics across all training frames/views for each Gaussian:
  $$w_i(u,v,t) = \frac{T_i \alpha_i P_i(u,v,t)}{\sum_j T_j \alpha_j P_j(u,v,t)}$$
  ($w_i$ = Gaussian i's normalized fractional contribution to a pixel's rendered color — this uses ordinary RGB opacity `α_i`, not occupancy)
  $$\tilde p_i^k = \frac{\sum_{t,(u,v)} \mathbb{1}[\mathbf{M}_t(u,v)=k]\cdot w_i(u,v,t)}{\sum_{t,(u,v)} w_i(u,v,t)}, \qquad \hat p_i^k = \frac{\tilde p_i^k}{\sum_{k'}\tilde p_i^{k'}}$$
  `M_t(u,v)` is the DEVA pseudo-ground-truth instance label at that pixel/frame; `\hat p_i^k` is the raw (visibility-biased) per-Gaussian identity estimate before calibration.

- **Calibration** (Eq. 8), applied per-Gaussian after each raw-estimate aggregation pass, before the identity distribution is used downstream:
  $$p_i^k = \frac{\hat p_i^k \cdot m_i^k}{\sum_{k'} \hat p_i^{k'}\cdot m_i^{k'}}$$
  `m_i^k > 0` are learnable calibration factors (one per Gaussian, per instance) that absorb systematic visibility bias — i.e., they upweight instances/Gaussians that were under-sampled due to low visibility.

- **Instance-Guided Resampling — volume-preserving split** (Eq. 10-11), applied when a Gaussian is cloned into `n+1` copies (selection driven by the joint response `γ_i^k = π_i p_i^k` identifying semantically active regions needing capacity):
  $$\alpha^{\text{new}} = 1-(1-\alpha^{\text{src}})^{1/(n+1)}, \qquad \pi^{\text{new}} = 1-(1-\pi^{\text{src}})^{1/(n+1)}$$
  This ensures splitting a primitive into n+1 children does not change its net compositing contribution (same identity as standard Gaussian-splatting densification split, applied here to both opacity and the new occupancy channel).

- **Total loss** (Eq. 12), applied at the end of each forward/rasterization pass:
  $$\mathcal{L} = \mathcal{L}_{\text{rgb}} + \lambda_{\text{inst}}\mathcal{L}_{\text{inst}}, \qquad \mathcal{L}_{\text{rgb}} = \lVert C^{\text{rendered}} - C^{\text{gt}}\rVert_1, \qquad \mathcal{L}_{\text{inst}} = -\sum_{u,v,t}\sum_k \mathbf{M}_k^{\text{gt}}\log \mathbf{M}_k^{\text{rendered}}$$
  (`L_inst` is a standard per-pixel cross-entropy against the DEVA pseudo-masks; no separate occupancy sparsity/regularization loss is used).

#### Algorithm / Pipeline Changes
1. Initialize/train a deformable 3D Gaussian Splatting backbone (time-conditioned MLP deformation) with standard RGB reconstruction — reported as 10,000 iterations.
2. Generate per-frame, per-view 2D instance pseudo-masks automatically via the DEVA video tracker (temporally consistent instance segmentation); for Neu3D's multiview captures, per-view masks are harmonized into a merged pseudo-monocular label sequence (Appendix A.1).
3. Augment every Gaussian with two new learnable attributes: occupancy `π_i ∈ [0,1]` and a K-dimensional identity vector `(p_i^1,...,p_i^K)` (softmax-normalized); K is set per-scene from the number of instances discovered in the masks, not fixed globally.
4. Run an instance-segmentation training stage (reported as 3,000 additional iterations) that renders the parallel instance-transmittance compositing pass (Eq. 4) alongside the RGB pass and supervises it with `L_inst` against the DEVA masks.
5. Periodically during this stage, aggregate visibility-biased raw identity estimates per Gaussian (Eq. 5-7) from accumulated rendering weights, then apply the learned calibration correction (Eq. 8) to de-bias them.
6. Periodically run Instance-Guided Resampling: identify high joint-response (`γ_i^k`) regions and clone Gaussians there, using the volume-preserving opacity/occupancy split (Eq. 10-11) so cloning does not perturb existing rendered output; resampling rate is 1% of all Gaussians per round for HyperNeRF, 5% for Neu3D.
7. At inference, open-vocabulary 4D querying is performed by pairing rendered instance masks with Grounded DINO for text-to-instance grounding (no CLIP-aligned per-Gaussian embedding is trained; querying is done via the discrete instance channel plus an external open-vocabulary detector).

#### Key Hyperparameters & Design Choices
- Reconstruction stage: 10,000 iterations. Instance-segmentation stage: 3,000 iterations (paper does not describe these as separately gated/frozen stages beyond this iteration split).
- Learning rate: 0.01 for occupancy and instance-identity calibration parameters; all other (backbone) parameters use the base Deformable Gaussian Splatting defaults. No warmup or LR schedule is reported for the new components.
- `λ_inst` (instance loss weight): 0.01 on HyperNeRF, 0.005 on Neu3D.
- Instance-Guided Resampling rate: 1% of all Gaussians (HyperNeRF), 5% of all Gaussians (Neu3D).
- K (number of instances): determined per-scene from the input masks, not a fixed global hyperparameter.
- Deformation-MLP architecture (layers/hidden dims): Not specified in paper (inherited unchanged from the cited backbone works, not re-detailed here).
- Instance-embedding dimension for open-vocabulary querying: Not applicable — no CLIP-aligned embedding is trained; open-vocabulary querying uses the discrete K-way instance channel plus Grounded DINO.
- Hardware: single NVIDIA A40 GPU.

#### Ablation Summary
From Table 3, on the HyperNeRF "split-cookie" scene (metric deltas vs. the Full model at mAcc-pix 97.93 / mAcc-inst 90.40 / mIoU 86.03 / PSNR 32.42):
- **Without calibration** (drop the Eq. 8 correction): mIoU 78.16 (**-7.87**), mAcc-inst 82.65 (-7.75), PSNR drops sharply to 26.73 (-5.69) — the single most impactful component, confirming visibility-bias calibration is load-bearing for both segmentation and reconstruction quality.
- **Constant occupancy** (fix `π=0.02` instead of learning it): mIoU 80.80 (-5.23).
- **Opacity-as-occupancy** (reuse RGB opacity `α` in place of a separate learned `π`, i.e., collapse the factorization): mIoU 82.34 (-3.69) — directly evidences that decoupling occupancy from radiance/opacity helps.
- **Without resampling**: mIoU 82.82 (-3.21) — smallest effect of the four ablated components, but still consistently negative across all four metrics.

#### Failure Modes & Limitations
The paper states in Appendix C.1 that "scenes involving amorphous or continuously evolving materials (e.g., smoke or liquids) lack stable structure and may not be faithfully represented through persistent Gaussian primitives," and separately that "residual cross-view inconsistencies or missing annotations under severe occlusion can still bias identity estimation" — i.e., the calibration mechanism reduces but does not eliminate visibility bias when occlusion is severe or when DEVA pseudo-labels are inconsistent or absent across views.

---

## Relevance to ADAGS

The cleanest formal existence-vs-persistence factorization found in the
sweep — but the persisted quantity is SEMANTIC instance identity, not
occluded-surface geometry/appearance, and hidden-surface recall on
disocclusion is untested. Mandatory citation for any identity-bearing
representation claim; the delta is geometric/appearance hidden-state
memory and reconstruction (not understanding) evaluation.

## Connections

- Pressures [[gap_map#G13 - Visibility Events Are Not Smooth Deformation]]
- Pressures [[gap_map#G14 - Detail Needs Identity-Conserving Promotion Rules]]

## Sources

- https://arxiv.org/abs/2512.14126
