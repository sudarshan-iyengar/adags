---
type: paper
node_id: paper:nguyen2026_tsa
title: "TSA: Temporal Slot Activation for Persistent Object-Centric Video Representation"
authors: ["Duc Nguyen", "Sieu Tran", "Hao Vo", "Khoa Vo", "Duy Minh Ho Nguyen", "Nghi D. Q. Bui", "Anh Nguyen", "Long Mai", "Ngan Le"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2606.13714"
tags: [object-centric, slots, occlusion, activation, video]
status: deep-dived
---

# TSA: Temporal Slot Activation for Persistent Object-Centric Video Representation

**Paper:** https://arxiv.org/abs/2606.13714
**Code:** Not found (paper states "The source code will be made publicly [available]" but no repository link is present in the paper, on arXiv, or via GitHub/Papers-With-Code search as of 2026-08-08).
**Base method:** Recurrent video slot attention (SAVi-style pipeline: frozen visual encoder → Temporal Query Transitioner → Slot Attention → Transformer decoder), formalized against the unconditional-propagation baselines VideoSAUR, SlotContrast (Manasyan et al. 2025), and RandSF.Q.

## One-line thesis
A single learned per-slot, per-frame activation score, computed without any visibility supervision, jointly gates a slot's state update (freeze vs. absorb current evidence) and its decoder attention weight (suppress vs. render), so that an occluded object's slot stops drifting and stops leaking into reconstruction at the same time instead of independently in each pathway.

## Problem / Gap
Existing recurrent video slot-attention methods (VideoSAUR, SlotContrast, RandSF.Q) use unconditional slot propagation: every slot is refreshed by slot attention against current-frame features and contributes to decoding at every frame, regardless of whether its object is actually visible (Eq. 1-2 baseline). The paper isolates two concrete failure pathways this causes: update-induced state drift, where an occluded slot's state is overwritten by attention to unrelated visible content (accumulating error, Eq. 3), and decoder-induced reconstruction interference, where the still-participating occluded slot receives reconstruction-loss gradients (Eq. 4) that pull its representation toward whatever currently occupies its spatial region — together producing identity switches when the object reappears.

## Method
Each frame, a frozen DINOv2 ViT-S/14 encoder produces patch features; a Temporal Query Transitioner conditions per-slot queries on the previous slot states, and Slot Attention (3 iterations on the first frame, 1 iteration on subsequent frames) refines each query against current-frame features into a per-slot candidate state S̃_{k,t}. A 2-layer MLP activation estimator Φ_act reads that candidate together with a per-slot GRU-maintained temporal memory M_{k,t-1} and outputs a scalar activation score α_{k,t} ∈ (0,1), trained end-to-end with no occlusion/visibility labels. That single scalar then gates two pathways with the same value: it convexly blends the candidate into the new state (freezing toward S_{k,t-1} as α→0), and it is added in log-space to the slot's decoder cross-attention logits before the softmax over slots (suppressing an inactive slot's reconstruction contribution). An autoregressive Transformer decoder (4 layers, 4 heads) then reconstructs DINOv2 feature tokens from the resulting slot-weighted attention.

## Assumptions
Assumes a slot-attention-style object-centric decomposition with a fixed, dataset-specific slot budget K (11-24) is an appropriate scene model, and that a frozen pretrained DINOv2 backbone supplies enough semantic/boundary signal to reconstruct (features, not RGB). Targets 2D monocular video only — synthetic (MOVi-C, MOVi-E) and real-world (YouTube-VIS HQ, OVIS) clips up to 500 frames — with no multi-view input, no 3D geometry, and no explicit occlusion or visibility ground truth at train time.

## Limitations / Failure Modes
The paper flags the fixed per-dataset slot budget K as an open limitation, with scene-adaptive slot allocation left unsolved. It notes dependence on the frozen DINOv2 backbone and suggests optical flow or depth could sharpen object boundaries further. It explicitly states that gradual appearance changes from deformation or scale variation over long sequences remain "an orthogonal open challenge" that activation-gated persistence does not address — the mechanism handles disappearance/reappearance, not slow visual drift of a still-visible object.

## Reusable Ingredients
- Shared-scalar dual gating: one learned value gates both a state-update pathway (convex blend, Eq. 7) and a decoder-attention pathway (log-bias, Eq. 8), so the two channels can never disagree about whether an entity is currently "there."
- GRU-accumulated temporal memory as the *input* to the activation/visibility estimator (not just the gated state itself) — ablation shows this beats gating with access to only the raw previous state.
- Log-bias attention suppression: adding log(α) to pre-softmax attention logits gives a soft gate the effect of multiplying post-softmax attention weight by α without any renormalization bookkeeping.
- Usage + sparsity regularizer pair on an unsupervised soft gate: a mean-activation penalty (L_usage) prevents always-on collapse; a value-sparsity penalty (L_sparse) discourages ambiguous mid-range (~0.5) gating.
- Decomposing a persistence mechanism's ablation into two independent failure pathways (state-drift vs. decoder-interference) as a design pattern for isolating which half of a gate actually matters.

---

### Deep Dive

#### Core Novelty
TSA replaces the unconditional recurrent slot update (every slot always absorbs current-frame evidence and always renders, Eq. 1-2) with a single learned scalar α_{k,t} per slot per frame that gates both pathways at once: Eq. 7 convexly blends the slot-attention candidate into the new state, and Eq. 8 additively biases that same slot's decoder attention logits in log-space. The key insight is that "object exists" and "object currently visible" are different quantities that unconditional propagation conflates; using one shared value to gate both the memory and the rendering pathway keeps them consistent — an inactive slot is simultaneously frozen (won't be corrupted by unrelated visible content) and silent (won't leak stale content into the reconstruction), rather than independently drifting in one channel while still rendering in the other.

#### Mathematical Formulation

**Baseline unconditional update (Eq. 1)** — evaluated per slot, per frame, before TSA's gating is introduced:
$$\mathbf{S}_{k,t} = SA(\mathbf{f}_t, \mathbf{q}_{k,t}) = U_\theta(\mathbf{f}_t, T_\phi(\mathbf{S}_{t-1}, \mathbf{f}_t))$$
where $T_\phi$ is the Temporal Query Transitioner producing query $\mathbf{q}_{k,t}$ from the previous slot states $\mathbf{S}_{t-1}$ and current features $\mathbf{f}_t$, and $U_\theta$ is Slot Attention's competitive cross-attention refinement.

**Baseline unconditional decoder attention (Eq. 2)** — per output token $n$, per slot $k$, per frame, inside the decoder's cross-attention:
$$z_{k,n,t} = \frac{1}{\sqrt{d}}\left(\mathbf{q}^d_n \cdot \mathbf{k}^d(\mathbf{S}_{k,t})\right), \qquad A^d_{k,n,t} = \frac{\exp(z_{k,n,t})}{\sum_{j=1}^{K}\exp(z_{j,n,t})}$$
$d$ is the attention key/query dimension, $\mathbf{q}^d_n$ a decoder output query, $\mathbf{k}^d(\cdot)$ the key projection of a slot. This is unconditional: $A^d_{k,n,t} > 0$ for every slot regardless of visibility.

**Activation score prediction (Eq. 6)** — evaluated once per slot per frame, after Slot Attention produces the candidate but before it is committed to the state:
$$\alpha_{k,t} = \sigma\left(\Phi_{act}(\tilde{\mathbf{S}}_{k,t}, \mathbf{M}_{k,t-1})\right)$$
where $\tilde{\mathbf{S}}_{k,t} = U_\theta(\mathbf{f}_t; \mathbf{q}_{k,t})$ is the slot-attention *candidate* (256-dim, not yet gated into the state) and $\mathbf{M}_{k,t-1}$ is the slot's temporal memory (64-dim) from the previous frame. $\Phi_{act}$ is a 2-layer MLP over the concatenated 320-dim (256+64) input, no visibility label used as supervision.

**Activation-gated state update (Eq. 7)** — replaces Eq. 1, evaluated per slot per frame:
$$\mathbf{S}_{k,t} = \alpha_{k,t}\,\tilde{\mathbf{S}}_{k,t} + (1-\alpha_{k,t})\,\mathbf{S}_{k,t-1}$$
A convex blend: as $\alpha_{k,t}\to 0$ the slot's state is anchored to $\mathbf{S}_{k,t-1}$ and current-frame evidence is fully rejected; as $\alpha_{k,t}\to 1$ it reduces to the unconditional update.

**Activation-gated decoder participation (Eq. 8)** — replaces Eq. 2, evaluated per output token per slot per frame, inside decoder cross-attention:
$$A^d_{k,n,t} = \mathrm{softmax}_k\!\left(z_{k,n,t} + \log \alpha_{k,t}\right) = \frac{\alpha_{k,t}\exp(z_{k,n,t})}{\sum_{j=1}^{K}\alpha_{j,t}\exp(z_{j,n,t})}$$
The log-bias makes the softmax numerator proportional to $\alpha_{k,t}$, i.e. equivalent to multiplying the unconditional attention weight by $\alpha_{k,t}$ pre-renormalization — the same scalar computed in Eq. 6 as used in Eq. 7.

**Temporal Context Encoder** — a single-layer GRU ($d_h=64$) updates memory for the *next* frame's activation decision, run per slot per frame after the state update:
$$\mathbf{M}_{k,t} = \Psi_{tce}(\mathbf{M}_{k,t-1}, \mathbf{S}_{k,t})$$

**Training objective (Eq. 10-11)** — evaluated once per training step over the full clip:
$$\mathcal{L} = \mathcal{L}_{recon} + \lambda_{ssc}\mathcal{L}_{ssc} + \lambda_{reg}\mathcal{L}_{reg}, \qquad \mathcal{L}_{reg} = \mathcal{L}_{usage} + \beta\,\mathcal{L}_{sparse}$$
$\mathcal{L}_{usage}$ penalizes mean activation across slots/frames (prevents always-on collapse); $\mathcal{L}_{sparse}$ penalizes intermediate ($\approx 0.5$) activation values (prevents ambiguous gating). $\mathcal{L}_{ssc}$ is the slot-sustaining/slot-consistency contrastive loss adapted from Manasyan et al. (2025); its exact form is not re-derived in this paper's main text.

#### Algorithm / Pipeline Changes
1. Frozen DINOv2 ViT-S/14 encodes frame $t$ into 384-dim patch features $\mathbf{f}_t$. (Unchanged from baseline VSA pipeline.)
2. Temporal Query Transitioner $T_\phi$ takes previous slot states $\mathbf{S}_{t-1}$ (K slots × 256-dim) and $\mathbf{f}_t$, produces per-slot query $\mathbf{q}_{k,t}$.
3. Slot Attention $U_\theta$ (competitive cross-attention; 3 iterations on frame 0, 1 iteration on subsequent frames) refines $\mathbf{q}_{k,t}$ against $\mathbf{f}_t$ into candidate state $\tilde{\mathbf{S}}_{k,t}$ (256-dim). This step is unchanged from baseline VSA, but its output is no longer committed directly to the state.
4. **New:** Slot Activation Estimator $\Phi_{act}$ (2-layer MLP, 320→128 dims, input = concat of $\tilde{\mathbf{S}}_{k,t}$ and $\mathbf{M}_{k,t-1}$) computes scalar $\alpha_{k,t}\in(0,1)$ per slot (Eq. 6).
5. **New:** State update is replaced by the activation-gated convex blend (Eq. 7), producing $\mathbf{S}_{k,t}$.
6. **New:** Temporal Context Encoder (single-layer GRU, $d_h=64$) updates per-slot memory $\mathbf{M}_{k,t}$ from $\mathbf{S}_{k,t}$, feeding step 4 on the next frame.
7. **Modified:** Autoregressive Transformer decoder (4 layers, 4 heads) cross-attends each output feature token to the K slots; attention logits get the $\log\alpha_{k,t}$ bias (Eq. 8) added before the softmax over slots, replacing the baseline's unconditional softmax (Eq. 2). Decoder reconstructs DINOv2 feature tokens, not RGB.
8. Loss: reconstruction (feature-space) + $\lambda_{ssc}\mathcal{L}_{ssc}$ (slot-consistency contrastive term) + $\lambda_{reg}\mathcal{L}_{reg}$ (usage + sparsity regularization on $\alpha$), backpropagated end-to-end with no visibility/occlusion supervision anywhere in the loop.

#### Key Hyperparameters & Design Choices
- Slot dimension: 256
- Slot Activation Estimator $\Phi_{act}$: 2-layer MLP, 320 → 128 (input is concat of 256-dim candidate + 64-dim memory)
- Slot Attention iterations: 3 on first frame, 1 on subsequent frames
- Temporal Context Encoder: single-layer GRU, hidden dim $d_h=64$
- Decoder: autoregressive Transformer, 4 layers, 4 attention heads, reconstructs DINOv2 features
- Visual encoder: frozen DINOv2 ViT-S/14 (384-dim features)
- Slot count K (dataset-specific): MOVi-C = 11, MOVi-E = 24, YouTube-VIS HQ = 7, OVIS = 22
- Optimizer: Adam, 50,000 training steps, batch size 8 video clips
- Learning rate: $5\times10^{-5}$ with 2,500-step warmup
- Gradient norm clipping: 0.05
- $\lambda_{ssc}$: 0.5 (constant across datasets)
- $\lambda_{reg}$: reported range 0.03-0.24 across datasets; exact per-dataset value not specified in the extracted text
- $\beta$ (sparsity weight within $\mathcal{L}_{reg}$): reported range 0.042-0.30 across datasets; exact per-dataset value not specified
- Regularization schedule: two-stage (warmup then linear ramp), doubled in duration for OVIS "due to longer occlusion dynamics"; exact step counts not specified

#### Ablation Summary (YouTube-VIS HQ; ARIfg / HOTA)
- Gating pathway: decoder-gating alone (60.8 / 21.7) is markedly worse than state-update gating alone (76.1 / 40.0); combining both (full model, 77.6 / 44.6) still adds +1.5 ARIfg / +4.6 HOTA over state-gating alone — **state-update gating is the dominant single pathway**, but the two are not redundant.
- Regularization: no regularization causes outright collapse (57.1 ARIfg); usage-only regularization recovers most of the gap (76.1 / 41.9); adding sparsity on top reaches the full model (77.6 / 44.6) — **usage regularization is necessary to prevent collapse**, sparsity is a smaller refinement (+1.5 ARIfg / +2.7 HOTA).
- Temporal memory: no memory input to $\Phi_{act}$ gives 61.7 / 20.1; using only the raw previous state $\mathbf{S}_{k,t-1}$ gives 72.8 / 39.8 (+11.1 ARIfg / +19.7 HOTA); the full GRU-accumulated memory $\mathbf{M}_{k,t-1}$ reaches 77.6 / 44.6 (a further +4.8 / +4.8) — **the single largest ablation delta among functioning configurations is going from no temporal context to any temporal context**, with accumulated memory beating raw previous state.

#### Failure Modes & Limitations
The paper reports a fixed, dataset-tuned slot budget K as an unresolved limitation — no scene-adaptive slot allocation mechanism is provided. It notes reliance on a frozen DINOv2 backbone and suggests optical flow or depth signals could further sharpen object boundaries, implying current boundary quality is backbone-limited. It explicitly separates the occlusion problem it solves from a different, unsolved problem: gradual appearance change from deformation or scale variation over long sequences, which it calls "an orthogonal open challenge" not addressed by activation-gated persistence — the mechanism preserves identity across disappearance/reappearance but does not track slow visual drift in an object that remains continuously visible.

---

## Relevance to ADAGS

Strongest adjacent-field statement of the existence-vs-visibility
conflation and its identity-switch consequence — supports the ADAGS
deficiency diagnosis. Also a novelty pressure: any "activation-gated
primitive" framing must differentiate from TSA (2D slots, learned
amortized model) — the ADAGS delta must live in the differentiable scene
representation itself, not in gating machinery.

## Connections

- Pressures [[gap_map#G13 - Visibility Events Are Not Smooth Deformation]]

## Sources

- https://arxiv.org/abs/2606.13714
