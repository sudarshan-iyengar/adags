---
type: paper
node_id: paper:jin2026_moegs
title: "MoE-GS: Mixture of Experts for Dynamic Gaussian Splatting"
authors: ["In-Hwan Jin", "Hyeongju Mun", "Joonsoo Kim", "Kugjin Yun", "Kyeongbo Kong"]
year: 2026
venue: "ICLR"
external_ids:
  arxiv: "2510.19210"
tags: [dynamic-gs, mixture-of-experts, motion-specialization]
status: deep-dived
---

# MoE-GS: Mixture of Experts for Dynamic Gaussian Splatting

**Paper:** https://arxiv.org/abs/2510.19210
**Code:** https://github.com/cvsp-lab/MoE-GS
**Base method:** Ensembles five existing dynamic-GS deformation experts unchanged
— HexPlane-based 4DGaussians (Wu et al.), per-Gaussian embedding E-D3DGS,
reformulation-based STG, interpolation-based Ex4DGS, and polynomial-trajectory
models. MoE-GS itself contributes only the router, the single-pass rendering
merge, the gate-aware pruning, and the distillation stage on top of these
frozen/independently-trained experts.

## One-line thesis

No single deformation prior (HexPlane grid, per-Gaussian embedding,
polynomial trajectory, or interpolation-based warp) is uniformly best across
scenes, regions, or time, so routing per-pixel per-frame among several
independently-trained experts via a differentiably-rasterized, per-Gaussian
learned weight recovers most of an oracle-best-expert upper bound while
staying close to single-expert render cost after pruning and distillation.

## Problem / Gap

Prior dynamic-GS methods each encode a fixed motion-representation bias:
HexPlane-based 4DGaussians smooths over sharp local motion, per-Gaussian
embedding methods (E-D3DGS) can drift without enough temporal regularization,
polynomial/interpolation methods (Ex4DGS, STG) fit local trajectories well but
generalize poorly to regions with a different motion regime. Benchmarks show
the best-performing method changes scene-to-scene, region-to-region within a
scene, and frame-to-frame — i.e., no single expert dominates, so committing to
one method's inductive bias leaves quality on the table exactly where its bias
doesn't match local scene dynamics.

## Method

MoE-GS trains N existing dynamic-GS experts (N=2..4, e.g., Ex4DGS+STG,
+E-D3DGS, +4DGaussians) independently to convergence in Stage 1, then freezes
them. Stage 2 duplicates each Gaussian, attaches a small learned per-Gaussian
weight vector (scalar, view-direction-conditioned, and time-conditioned
components), and rasterizes these weights into pixel-space embeddings via the
same differentiable splatting used for color. A lightweight MLP refines these
splatted embeddings per-pixel, and a softmax over experts produces the final
per-pixel gating weights that blend each expert's rendered color at that
pixel. Only the router (per-Gaussian weights + MLP) is trained in Stage 2;
experts are frozen. A single-pass rendering trick merges all experts' Gaussians
into one rasterization call using one-hot expert-identity vectors, gate-aware
pruning drops Gaussians whose per-Gaussian weight has low gradient influence
on the gate, and an optional distillation stage retrains each expert alone
using MoE-blended output as a pseudo-label so a single expert can approximate
the ensemble at deployment.

## Assumptions

Assumes each expert can already be trained to convergence on the target scene
(multi-view calibrated capture, per N3V/Technicolor-style dynamic multi-view
datasets) and that expert checkpoints are available/frozen before router
training. Assumes the heterogeneity that matters is capturable by existing
deformation-prior categories (grid/HexPlane, per-Gaussian embedding,
polynomial, interpolation) rather than requiring a new motion representation.

## Limitations / Failure Modes

The paper states that increased model capacity (multiple frozen experts plus
router) and reduced FPS are inherent to the MoE architecture; distillation and
pruning reduce but do not eliminate this cost (from 747 MB / 36 FPS unoptimized
to 270 MB / 68 FPS fully optimized for N=3, still carrying N=3 experts' worth
of underlying capacity before distillation). Router quality is sensitive to
architecture: a naive per-pixel-only router underperforms (31.12 PSNR) and a
naive volume-only router underperforms (32.05 PSNR) versus the proposed
volume-aware pixel router (33.23 PSNR), indicating the routing signal itself
is a nontrivial design problem, not solved by any obvious router.

## Reusable Ingredients

- **Differentiable weight splatting for per-pixel routing**: attach a learned
  weight vector to each Gaussian (duplicating it, replacing color with the
  weight), rasterize it exactly like color, then refine in pixel space with a
  small MLP — gives a spatially and temporally coherent per-pixel gate without
  a separate image-space network needing its own multi-view consistency.
- **Single-pass multi-expert rendering via one-hot identity vectors**: merge
  all experts' Gaussians into one rasterization call, tagging each with a
  one-hot expert-id vector so per-expert color channels fall out of a single
  alpha-compositing pass instead of N separate render passes.
- **Gate-aware pruning by gradient sensitivity**: score each Gaussian by the
  gradient magnitude of the gating output w.r.t. its own per-Gaussian weight,
  averaged over a dataset of views; prune Gaussians below a threshold — a
  pruning criterion tied to *routing relevance* rather than opacity/size.
- **Two-stage freeze-then-route training**: train each candidate method to
  full independent convergence first, then train only a lightweight router
  with everything else frozen — avoids one fast-converging expert dominating
  gradients if trained jointly from scratch.
- **Self-distillation from ensemble to single expert**: retrain a single
  expert using the MoE's blended render as a pseudo-label (weighted by that
  expert's own routing share) to recover most of the ensemble's quality gain
  in a single-expert-cost deployment model.

---

### Deep Dive

#### Core Novelty
MoE-GS does not propose a new deformation representation; its novelty is the
**volume-aware pixel router**: a way to combine per-Gaussian (3D-consistent)
and per-pixel (fine spatial/view-dependent) routing signals via differentiable
rasterization, so that gating is both spatially/temporally coherent (inherited
from the 3D Gaussian structure) and locally precise (refined per-pixel). The
key insight is that routing purely in 3D (per-Gaussian classification) is too
coarse and unstable, while routing purely in 2D pixel space discards
multi-view/temporal consistency — splatting a learned 3D attribute into pixel
space and refining it there gets both.

#### Mathematical Formulation
Per-Gaussian weight attached to Gaussian $i$:
$$w_i^{per} = \left[w_i,\ w_i^{dir},\ t \cdot w_i^{time}\right]^T$$
where $w_i$ is a static scalar component, $w_i^{dir}$ is modulated by the
current viewing direction, and $w_i^{time}$ is scaled by frame time $t$; this
vector is rasterized (evaluated per-Gaussian, splatted like color before the
final compositing/blend stage) to give pixel-aligned embeddings
$w_{2D}(u), w_{2D}^{dir}(u), w_{2D}^{time}(u)$ at pixel $u$.

Pixel-space refinement (evaluated post-rasterization, per pixel):
$$R'(u) = w_{2D}(u) + \Phi\big(w_{2D}^{dir}(u),\, w_{2D}^{time}(u),\, r(u)\big)$$
where $\Phi$ is a lightweight MLP and $r(u)$ is (implied) a per-pixel render
feature/residual input alongside the splatted weights.

Expert gate (softmax over the $N$ experts' refined logits, per pixel):
$$I_{MoE}(u) = \sum_{k=1}^{N} G'_k(u) \cdot I_{E_k}(u), \qquad G'_k(u) = \mathrm{Softmax}\big(R'_k(u)\big)$$
where $I_{E_k}(u)$ is expert $k$'s rendered color at pixel $u$ and $G'_k(u)$
its gate weight — this is the final blended output, computed after each
expert's independent render.

Single-pass merged rendering (replaces $N$ separate alpha-compositing passes
with one, per pixel):
$$C_k(u) = \sum_{j=1}^{M} T_j(u)\, \alpha_j(u)\, c_j \cdot (e_j)_k$$
where $M$ is the total Gaussian count across all experts merged, $T_j(u)$ is
accumulated transmittance, $\alpha_j(u)$ opacity, $c_j$ color, and
$e_j \in \mathbb{R}^K$ a one-hot vector marking which expert Gaussian $j$
belongs to — so $C_k(u)$ isolates expert $k$'s contribution from a single
merged rasterization pass.

Gate-aware pruning importance score (computed post Stage-2 training, offline,
before pruning; averaged over a held-out view set $D$):
$$\mathcal{E}_i = \frac{1}{|D|} \sum_{v \in D} \left\| \frac{\partial G'_k(v)}{\partial w_i^{per}(v)} \right\|$$
Gaussian $i$ is pruned if $\mathcal{E}_i < \tau$ for a threshold $\tau$.

Distillation loss (Stage 3, per expert $k$, retraining that expert alone):
$$L_k^{KD} = \lambda \cdot L\big(G'_k \cdot I_{E_k},\, G'_k \cdot I_{GT}\big) + (1-\lambda) \cdot L\big((1-G'_k)\cdot I_{E_k},\, (1-G'_k)\cdot I_{MoE}\big)$$
blending ground-truth supervision (weighted by the expert's own routing share)
with MoE-output pseudo-label supervision (weighted by the complement) so the
distilled single expert learns to approximate the ensemble specifically in
the regions/frames where the ensemble routed away from it.

#### Algorithm / Pipeline Changes
1. **Stage 1 — independent expert training**: train each of $N$ candidate
   dynamic-GS methods (chosen from HexPlane/4DGaussians, E-D3DGS, STG,
   Ex4DGS, polynomial models) to full convergence on the target scene, fully
   independently, with no shared parameters.
2. **Stage 2 — router training, experts frozen**: duplicate every Gaussian
   across all $N$ frozen experts; attach the learnable $w_i^{per}$ vector to
   each; rasterize color (from the frozen expert) and the weight vector in
   parallel; refine the splatted weight via MLP $\Phi$ to get $R'(u)$; softmax
   to get $G'_k(u)$; composite $I_{MoE}(u)$; backprop the standard
   L1+SSIM rendering loss only into $w_i^{per}$ and $\Phi$ (experts stay
   frozen). This stage replaces what would otherwise be a naive fixed-weight
   or heuristic ensemble average.
3. **Efficiency pass — single-pass rendering**: merge all experts' Gaussians
   into one rasterizer call tagged with one-hot expert-id vectors $e_j$,
   replacing $N$ separate rasterization passes with one that yields all
   $C_k(u)$ simultaneously.
4. **Efficiency pass — gate-aware pruning**: after Stage 2 converges, compute
   $\mathcal{E}_i$ over a view set and drop Gaussians below threshold $\tau$;
   this runs once, offline, before/instead of standard opacity-based pruning.
5. **Stage 3 (optional) — distillation**: for deployment, retrain a single
   chosen expert from scratch (or fine-tune) using $L_k^{KD}$ so it alone
   approximates the frozen ensemble's blended output, removing the need to
   keep all $N$ experts resident at inference time.

#### Key Hyperparameters & Design Choices
- Number of experts evaluated: N=2 (Ex4DGS+STG), N=3 (+E-D3DGS), N=4
  (+4DGaussians).
- Router learning rate for the shared MLP $\Phi$ and $w_i^{time}$: 0.05.
- Router learning rate for $w_i$ and $w_i^{dir}$: 0.5.
- Optimizer: RAdam.
- Per-Gaussian weight initialization: direction/time components initialized
  near zero (static component's init value not specified in paper).
- Pruning threshold $\tau$: not specified in paper.
- Distillation weight $\lambda$: not specified in paper.
- MLP $\Phi$ architecture (layers, hidden dim): not specified in paper.
- Training iterations for distillation/Stage 3: not specified in paper
  (render evaluation script shows checkpoints at 2000/5000 iterations).

#### Ablation Summary
Router architecture (Table 4, PSNR/SSIM):
- Pixel-only router: 31.12 dB / 0.952 — weakest.
- Volume-only router: 32.05 dB / 0.951.
- **Volume-aware pixel router (proposed): 33.23 dB / 0.954 — most impactful
  single component**, i.e., the core routing mechanism itself is the largest
  lever, not just an incremental refinement.

Efficiency optimizations (Table 5, N=3 experts):
- No optimization: 32.54 dB, 36 FPS, 747 MB.
- Single-pass rendering only: 33.23 dB, 40 FPS, 270 MB (quality actually rises
  alongside memory drop, and enables pruning to be effective).
- Pruning only: 32.54 dB, 60 FPS, 747 MB.
- Both combined: 33.23 dB, 68 FPS, 270 MB — full efficiency stack recovers
  best quality at ~2x FPS and ~3x lower memory than unoptimized.

Expert training budget (Table 6, N=3): at only 20% of full per-expert training
budget, MoE-GS already reaches 32.60 dB, exceeding fully-converged single
experts (~31.5-32.3 dB) — the ensemble/routing gain is large enough to beat
fully-trained single baselines even with partially-trained experts.

Distillation (Table 7): consistent per-expert gains from MoE-guided
supervision, e.g., E-D3DGS on Technicolor improves 32.88 → 33.67 dB when
retrained with $L_k^{KD}$ versus standard ground-truth-only training.

#### Implementation Reality
- **Framework:** PyTorch 2.0.1, extends the standard 3D/4D Gaussian Splatting
  CUDA-rasterizer codebase family (submodules/thirdparty dirs present); tested
  on NVIDIA RTX A6000.
- **Key files:** `train_E3.py`, `train_E4.py`, `train_E3_tech.py` are the
  Stage-2 router-training entry points for 3- and 4-expert configurations
  (N3V vs Technicolor variants); `render_E3.py`/`render_E4.py` render at
  specified checkpoint iterations; `configs/N3V/*.json` and
  `configs/techni/*.json` hold per-scene configuration; `gaussian_renderer/`
  holds rendering logic and is the most likely location of the single-pass
  merge and weight-splatting code, though the exact file implementing the
  router MLP $\Phi$ and gate-aware pruning was not identified from the
  available documentation.
- **Notable implementation details:** as of this review, the repository's own
  TODO list marks "Expert Training Scripts," pruning, and distillation as
  "Coming Soon" — only the Stage-2 router training/rendering code appears
  fully released; Stage-1 expert training, gate-aware pruning, and the
  distillation stage described in the paper are not yet public in this repo.

#### Failure Modes & Limitations
The paper explicitly frames added capacity and reduced FPS as inherent to the
MoE approach — running $N$ experts (even merged into one rasterization pass)
costs more memory/compute than any single expert, and pruning/distillation
mitigate rather than remove this. No scene-specific or motion-type-specific
failure cases (e.g., specific N3V/Technicolor scenes where routing fails) were
surfaced in the accessible sources.

---

## Relevance to ADAGS

Direct pressure on single LoRA basis; supports part/expert reversible routing as a research direction.

## Connections

## Sources

- https://arxiv.org/abs/2510.19210
- https://arxiv.org/html/2510.19210
- https://github.com/cvsp-lab/MoE-GS
- https://paper.pnu-cvsp.com/MoE-GS
