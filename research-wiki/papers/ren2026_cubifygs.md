---
type: paper
node_id: paper:ren2026_cubifygs
title: "CubifyGS: Object-Centric 3D Gaussian Splatting for Lifelong Dynamic Scene Maintenance"
authors: ["Bohan Ren", "Dianyi Yang", "Shiyang Liu", "Yu Gao", "Jiadong Tang", "Zhilin Lai", "Yi Yang", "Mengyin Fu"]
year: 2026
venue: "IROS 2026"
external_ids:
  arxiv: "2606.28720"
tags: [gaussian-splatting, object-assets, identity-reuse, robotics, lifelong]
status: deep-dived
---

# CubifyGS: Object-Centric 3D Gaussian Splatting for Lifelong Dynamic Scene Maintenance

**Paper:** https://arxiv.org/abs/2606.28720
**Code:** Not found
**Base method:** MonoGS (Matsuki et al., 2024) — a monocular/RGB-D SLAM-Gaussian-Splatting system built on 3DGS (Kerbl et al., 2023) — extended with an object-centric asset-management layer; object proposals are supplied by CubifyAnything (Lazarow et al., 2025), an off-the-shelf 3D object detector for RGB-D streams.

## One-line thesis
Restructuring a Gaussian map from a flat primitive bag into per-object Gaussian "assets" that are pruned wholesale on disappearance and retrieved-and-rigidly-realigned wholesale on reappearance/relocation avoids re-optimizing photometric primitives from scratch, fixing the ghosting and slow-recovery failure that primitive-level online 3DGS updates exhibit under rigid object rearrangement.

## Problem / Gap
Online/incremental 3DGS mapping systems (e.g., MonoGS-style SLAM-Gaussian pipelines) update primitives locally wherever new photometric error appears. When a rigid object is moved or removed in a lifelong/robotics setting, this primitive-level update leaves stale Gaussians ("ghosting") at the old location and requires many gradient steps to regrow correct geometry/appearance at the new location or to inpaint the void — recovery is slow and quality is degraded during the transient. The paper targets this specific failure mode of rigid rearrangement in long-duration, repeatedly-revisited scenes, not general non-rigid dynamics.

## Method
Each detected object is tracked as a first-class entity 𝒪ᵢ = (𝒢ᵢ, ℱᵢ, Bᵢ) — its own Gaussian primitive set, a multi-view DINO feature bank over 12 canonical viewpoints, and a gravity-aligned 3D bounding box — rather than being absorbed into one undifferentiated scene-level Gaussian cloud. Per-frame 3D box proposals from CubifyAnything are linked to tracked instances via a two-layer hierarchical association (volumetric IoU, then 2D projection IoU). A per-object existence-probability state machine (hit/free/occluded ray-casting statistics) governs whether an object is confirmed present, decayed as possibly-vanished, or pruned outright — pruning removes every Gaussian inside the object's bounding box, leaving a geometric void that is then repaired by an event-triggered, ROI-mask-weighted photometric loss. Objects that accumulate enough view coverage and gradient-stable geometry are promoted into a global asset library; when a previously-seen object reappears (moved or in a new session), it is retrieved from the library by DINO feature cosine similarity, coarsely posed via canonical-viewpoint matching, then finely posed by freezing the Gaussian parameters and optimizing only a rigid SE(3) transform against the new photometric observations.

## Assumptions
Requires calibrated RGB-D input with known camera poses (built on a SLAM-Gaussian backbone) and a 3D object detector (CubifyAnything) that can propose gravity-aligned boxes per frame. Assumes objects move/disappear/reappear as whole rigid bodies (rearrangement), not through non-rigid deformation, articulation, or partial occlusion of a still-present object's shape.

## Limitations / Failure Modes
The paper states the current study "focuses on rigid object rearrangement with reusable object assets, and the promoted assets are reused as fixed templates" — i.e., an asset's Gaussians are never updated once promoted, so appearance/geometry changes to an object after promotion (wear, lighting change, partial modification) are not captured on reuse. Incremental asset updates and cross-session asset merging are named as unimplemented future work. The real-world transfer sequence (Bonn_kidnapping_box2, from the ReFusion dataset) lacks high-fidelity 3D box ground truth and is therefore excluded from the paper's own perception (3D mAP) evaluation, i.e., the method's detection/association accuracy is not claim-grade validated on real data, only its rendering transfer is shown.

## Reusable Ingredients
- **Hierarchical box-to-track association** (volumetric IoU gate, then 2D-projection IoU gate) — cheap two-stage instance association usable for any per-object 3D tracking layer on top of a detector stream.
- **Existence-probability state machine from ray-casting statistics** (hit/free/occluded ray fractions driving a bounded probability with asymmetric increment/decay) — a lightweight, non-learned way to decide presence/absence/vanished status per tracked entity without a neural temporal model.
- **Promotion gate on view coverage + gradient stability** — a concrete, checkable criterion for "this object's reconstruction is good enough to freeze and reuse," transferable to any bank of per-entity primitive subsets.
- **Freeze-geometry, optimize-pose-only re-registration** — when reusing a previously reconstructed asset, only a rigid transform is optimized against new photometric evidence (Gaussian parameters frozen), which is far cheaper and more stable than re-optimizing appearance/geometry jointly.
- **Event-triggered ROI-weighted photometric loss** (uniform weight elsewhere, large weight γ_focus inside the affected region) for fast local inpainting after a prune, rather than uniform-weight full-scene re-optimization.

---

### Deep Dive

#### Core Novelty
Relative to MonoGS-style online 3DGS SLAM, CubifyGS's change is architectural: it partitions the primitive population by object identity (𝒢ᵢ per object rather than one shared cloud) and replaces primitive-level gradient updates for rearrangement with three discrete, non-differentiable-at-the-primitive-level operations — retrieve (feature match against a library), transform (rigid pose-only optimization), and prune (hard removal of all primitives inside a stale bounding box). The key insight is that rigid rearrangement is a low-dimensional event (one SE(3) transform per object) and treating it as such avoids re-spending photometric optimization budget on geometry/appearance that is already known and simply moved.

#### Mathematical Formulation
- **Eq. 1 — Object instance representation:** 𝒪ᵢ = (𝒢ᵢ, ℱᵢ, Bᵢ), where 𝒢ᵢ is the object's own set of 3D Gaussians (mean μ∈ℝ³, covariance Σ, opacity α, SH color coefficients 𝐜), ℱᵢ is a multi-view DINO feature bank sampled at M=12 canonical viewpoints, and Bᵢ is a gravity-aligned 3D bounding box. This is the data structure the rest of the pipeline operates on — evaluated/maintained continuously during tracking, not a loss term.
- **Eq. 5 — Existence probability update (per object, per frame/step):**
  $$p_t^{(i)} = \begin{cases} \min(1,\, p_{t-1}^{(i)} + \delta_{hit}) & \text{if } N_{hit}/N > \tau_{hit} \\ p_{t-1}^{(i)} \cdot \lambda_{free} & \text{if } N_{free}/N > \tau_{free} \wedge N_{occ}/N < \tau_{occ} \\ p_{t-1}^{(i)} \cdot \lambda_{decay} & \text{otherwise} \end{cases}$$
  where $N_{hit}$, $N_{free}$, $N_{occ}$ are counts (out of $N$ cast rays/samples against the object's box) landing on the object surface, passing through empty space where the object should be, or being occluded, respectively; $\tau_{hit}, \tau_{free}, \tau_{occ}$ are fraction thresholds; $\delta_{hit}$ is an additive confirmation increment; $\lambda_{free}, \lambda_{decay}$ are multiplicative decay factors (<1). Evaluated per tracked object at each association step, outside the rendering graph — this is a bookkeeping/state-machine computation, not a differentiable term. When $p_t^{(i)}$ falls below a vanish threshold $\tau_{vanish}$ after a grace period, the object is flagged "Vanished" and its Gaussians are pruned.
  - This is the mechanism that answers the temporal-presence question: it is a **scalar occupancy-confidence state per object per timestep**, not a continuous learned function of time and not a per-Gaussian quantity — it is entity-level and rule-based (thresholds/decay constants), not a network output.
- **Eq. 8 — Asset promotion criterion:** $\mathbb{I}_{promote} = \mathbb{1}(C_{view} > \tau_v \wedge \nabla_{stable} \le \epsilon)$, where $C_{view}$ is the angular azimuth coverage span observed for the object and $\nabla_{stable}$ is a gradient-convergence measure over K consecutive frames. Evaluated once per tracked object to gate insertion into the global asset library; after promotion the object's $\mathcal{G}_i$ is frozen as a fixed template.
- **Fine-alignment pose optimization (Sec. III-C2):** pose gradient approximated as $\partial \mathcal{L}_{pho}/\partial T \approx \sum_i \partial \mathcal{L}_{pho}/\partial C \cdot \partial C/\partial \alpha_i \cdot \partial \alpha_i/\partial m_i \cdot \partial m_i/\partial T$ — the photometric loss gradient is backpropagated only through the rigid transform $T$ applied to each Gaussian mean $m_i$ (via projected pixel color $C$ and opacity/alpha compositing $\alpha_i$), with all other Gaussian parameters (covariance, opacity, SH coefficients) held frozen. This is the one place gradients flow, and they flow only into a 6-DoF rigid transform, confirming reuse-by-retrieval is pose-only differentiable, not full-asset differentiable.
- **Adaptive/inpainting loss:** $L_{adapt} = \sum_u \Omega(u)\cdot[(1-\lambda)L_1(u) + \lambda L_{D\text{-}SSIM}(u)]$, where $\Omega(u) = \gamma_{focus}$ (≫1) inside the affected ROI (a pruned void or newly placed object) and $1$ elsewhere. Standard L1/D-SSIM combination, reweighted spatially; evaluated after rendering, as the loss driving post-prune/post-placement local re-optimization of the surrounding (non-asset) scene Gaussians.

#### Algorithm / Pipeline Changes
1. Per RGB-D frame, run CubifyAnything to produce 3D bounding-box object proposals.
2. Associate proposals to existing tracked instances with a two-layer hierarchical matcher: Layer 1 gates candidates by 3D volumetric IoU; Layer 2 refines/disambiguates by 2D image-plane projection IoU.
3. For each tracked object, update ray-cast hit/free/occluded counts against its bounding box and update existence probability $p_t^{(i)}$ via Eq. 5.
4. If $p_t^{(i)}$ stays below $\tau_{vanish}$ past a grace period, mark the object "Vanished," and prune — hard-delete every Gaussian primitive whose mean falls inside $B_i$ — leaving a geometric void.
5. Run the ROI-weighted adaptive loss $L_{adapt}$ (event-triggered, focused optimization with $\Omega(u)=\gamma_{focus}$ inside the void/placement region) to let neighboring/background Gaussians fill or adjust rapidly, instead of uniform full-scene optimization.
6. Independently, for each confirmed-present object accumulating enough angular view coverage and gradient stability (Eq. 8), promote it: freeze its current $\mathcal{G}_i$ as a template and store it plus its multi-view DINO feature bank $\mathcal{F}_i$ (12 canonical viewpoints) in a global asset library, replacing what would otherwise be continued online optimization of that object.
7. When a new/moved object is detected that resembles a library entry, retrieve by cosine similarity between the current object's visual feature and the library's per-asset multi-view feature banks.
8. Coarse-align: pick the best-matching canonical viewpoint to seed an initial rigid pose (constrained to ≤±15° angular error).
9. Fine-align: freeze all Gaussian parameters of the retrieved template and run pose-only optimization (Adam) against the new photometric observations for 1,000 iterations total — 700 coarse-stage iterations at rotation LR 1×10⁻², translation LR 5×10⁻³, then 300 fine-stage iterations at rotation LR 6×10⁻³, translation LR 3×10⁻³.
10. Instantiate the retrieved template's Gaussians into world space at the optimized pose via rigid transform — this replaces what a primitive-level online system would do by re-densifying/re-optimizing appearance and geometry at the new location from scratch.

#### Key Hyperparameters & Design Choices
- Canonical viewpoints per asset feature bank (M): 12.
- Fine-alignment optimization: 1,000 iterations total, split 700 (coarse stage) + 300 (fine stage).
- Coarse-stage Adam LR: rotation 1×10⁻², translation 5×10⁻³.
- Fine-stage Adam LR: rotation 6×10⁻³, translation 3×10⁻³.
- Coarse-alignment angular constraint: ≤±15° error from canonical-viewpoint matching.
- Evaluation window for post-rearrangement recovery: 10 seconds, sampled at 30-iteration intervals.
- $\gamma_{focus}$ (ROI loss weight): stated only as "≫1"; exact value not specified in paper.
- $\tau_{hit}, \tau_{free}, \tau_{occ}, \delta_{hit}, \lambda_{free}, \lambda_{decay}, \tau_{vanish}$, grace-period length, $\tau_v$ (view-coverage promotion threshold), $\epsilon$ (gradient-stability bound), $K$ (consecutive-frame window for stability), $N$ (ray/sample count for hit/free/occ statistics), and the ROI temporal window width $w$: Not specified in paper.
- Main scene-level 3DGS/MonoGS optimization schedule (learning rates, densification schedule, iteration counts): Not specified in paper — only the object re-registration/fine-alignment schedule above is given; base MonoGS settings are implied but not restated.

#### Ablation Summary
Table IV reports only two named components, each removed individually (numbers shown are PSNR / SSIM / LPIPS on the Livingroom-1 / Kitchen-1 scenes):
- **Full method (FDO + GAL):** Livingroom-1 20.50 / 0.546 / 0.476; Kitchen-1 21.18 / 0.463 / 0.434.
- **GAL only (no Focus-Driven Optimization):** Livingroom-1 17.58 / 0.511 / 0.503; Kitchen-1 20.54 / 0.456 / 0.441 — removing FDO costs 2.92 dB PSNR on Livingroom-1.
- **FDO only (no Global Asset Library, "cold start"):** Livingroom-1 15.98 / 0.502 / 0.571; Kitchen-1 20.11 / 0.458 / 0.488 — removing GAL costs 4.52 dB PSNR on Livingroom-1, the single most impactful component in this ablation.
No other ablation rows (e.g., isolating the association layers, the existence-probability state machine, or the pose-only fine-alignment step) are reported.

#### Failure Modes & Limitations
The paper explicitly scopes itself to "rigid object rearrangement with reusable object assets," with "promoted assets... reused as fixed templates" — meaning any appearance/geometry change to an object after it is promoted into the library is not reflected on reuse (no incremental asset update). Cross-session asset merging is named as future work, implying the current library does not reconcile duplicate or drifting asset copies across sessions. The only real-world sequence tested (Bonn_kidnapping_box2, from the ReFusion dataset) lacks high-fidelity 3D box ground truth, so it is used only for qualitative/rendering transfer, not for the paper's own perception (3D mAP) evaluation — real-world detection/association accuracy is therefore not claim-grade demonstrated.

---

## Relevance to ADAGS

The closest 2026 identity-REUSE precedent: reusable Gaussian assets with
persistent identity. Delta for any ADAGS identity representation: CubifyGS
performs discrete asset maintenance across robotic sessions/edits with
retrieval operations, not differentiable multi-interval temporal presence
within one capture with gradient-pooled appearance. Mandatory citation.

## Connections

- Pressures [[gap_map#G14 - Detail Needs Identity-Conserving Promotion Rules]]

## Sources

- https://arxiv.org/abs/2606.28720
