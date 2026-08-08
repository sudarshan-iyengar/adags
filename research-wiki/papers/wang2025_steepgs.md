---
type: paper
node_id: paper:wang2025_steepgs
title: "Steepest Descent Density Control for Compact 3D Gaussian Splatting"
authors: ["Peihao Wang", "Yuehao Wang", "Dilin Wang", "Sreyas Mohan", "Zhiwen Fan", "Lemeng Wu", "Ruisi Cai", "Yu-Ying Yeh", "Zhangyang Wang", "Qiang Liu", "Rakesh Ranjan"]
year: 2025
venue: "CVPR 2025"
external_ids:
  arxiv: "2505.05587"
tags: [static-gs, densification, theory, optimization]
status: deep-dived
---

# Steepest Descent Density Control for Compact 3D Gaussian Splatting

Extracted from the arXiv HTML rendering (arxiv.org/html/2505.05587), the arXiv
abstract page, and the official GitHub repository
(facebookresearch/SteepGS), all via automated fetches. No CVPR-typeset
PDF/LaTeX source was directly parsed, so equation subscript/superscript
nesting below has been reconstructed from HTML text extraction and should be
treated as faithful-but-not-camera-ready; numeric hyperparameters were
cross-checked against the released code defaults where possible and are
flagged individually below when they come from code rather than the paper
text.

**Paper:** https://arxiv.org/abs/2505.05587
**Code:** https://github.com/facebookresearch/SteepGS
**Base method:** 3D Gaussian Splatting (Kerbl et al. 2023), specifically its
Adaptive Density Control (ADC) clone/split heuristic.

## One-line thesis
Splitting an under-optimized Gaussian is only loss-reducing when that
Gaussian sits at a saddle point of the photometric loss (not a local
minimum); the paper derives, from a second-order Taylor expansion of the
loss around split parameters under a trust-region constraint, that exactly
two equally-weighted offspring displaced along ± the minimum eigenvector of
a per-Gaussian "splitting matrix" achieve the steepest possible loss
decrease, replacing 3DGS's gradient-magnitude split heuristic with a
closed-form optimal rule for whether, where, and with what opacity to split.

## Problem / Gap
Original 3DGS ADC decides which Gaussians to split/clone by thresholding the
average magnitude of view-space position gradients, then routes by a scale
threshold, with offspring position sampled randomly from the parent's own
scale distribution and opacity copied or naively halved — none of this is
tied to whether the split actually reduces loss. This produces redundant
points: many splits do not reduce loss, inflating point count, memory, and
render time without commensurate quality gain. Prior heuristic revisions
(Bulò et al.'s modified splitting criterion, Kheradmand et al.'s
opacity-distribution sampling / 3DGS-MCMC) give only limited improvement
because, per the paper, "the densification process is not well understood."

## Method
The paper models densification as: replace a converged (stationary-gradient)
Gaussian with `m` offspring parameterized by per-offspring parameter
perturbations `δ_j` and mixture weights `w_j`, then take a second-order
Taylor expansion of the full-image photometric loss around the parent's
parameters. Because the parent's own gradient has vanished, the first-order
term drops out, leaving a purely quadratic term governed by a new
per-Gaussian "splitting matrix" `S(θ)` — a pixel-averaged,
loss-gradient-weighted Hessian of the Gaussian's projected opacity response.
Solving the resulting constrained optimization (offspring displacement
bounded by a trust region, weights summing to one) in closed form shows
splitting reduces loss only when `S` has a negative eigenvalue (the Gaussian
is at a saddle, not a minimum), and when it does, the optimal move is exactly
two equal-weight offspring displaced by ± the eigenvector of the most
negative eigenvalue, with opacity analytically halved. This criterion and
placement rule replace 3DGS's gradient-threshold split; the released code
computes `S` from reused forward-pass intermediates and gates it with an
additional "gradient ≈ 0" stationarity test before applying the saddle
condition, then routes through the same clone-vs-split scale-based branching
as vanilla 3DGS.

## Assumptions
Same regime as 3DGS: static, multi-view captured scenes with known camera
poses (SfM) and photometric-loss-only optimization (no depth/flow/semantic
priors). The theory assumes the loss is well approximated locally by a
second-order (quadratic) expansion — i.e., the trust-region radius is small
enough that cubic and higher terms are negligible. The splitting matrix is
computed over position parameters only (3×3, `dimΘ=3`); scale, rotation,
opacity, and SH coefficients continue to be optimized by ordinary gradient
descent and are only indirectly affected (via the derived halving/scaling
rules applied at the moment of a position-triggered split).

## Limitations / Failure Modes
The accessible paper text has no dedicated limitations/failure-case
discussion or per-scene-category breakdown; the clearest weakness signal is
quantitative. On Mip-NeRF 360, SteepGS trades away real quality for
compactness: PSNR drops ~0.3 dB (29.037 → 28.734) and LPIPS worsens by
+0.028 (0.183 → 0.211) alongside the ~52% point-count reduction, i.e. the
hardest (unbounded, 360°) benchmark shows a real perceptual-quality cost. By
contrast, on Deep Blending it slightly improves PSNR/SSIM (+0.27 dB, +0.001
SSIM) at ~54% fewer points, so the quality/compactness trade-off is
scene-dependent and not uniformly free. No discussion is given of dynamic
scenes, occlusion, or the extra compute/memory cost of accumulating and
eigendecomposing the splitting matrix beyond the claim that it reuses
forward-pass intermediates.

## Reusable Ingredients
- **Splitting matrix `S(θ)`**: a per-primitive, loss-gradient-weighted
  Hessian of the primitive's pixel-response function; its eigenstructure
  diagnoses whether locally perturbing/duplicating that primitive can reduce
  loss — a generic "is this primitive stuck at a saddle point" test
  applicable to any gradient-optimized primitive-based scene representation.
- **Closed-form steepest split direction**: displacing offspring along ± the
  minimum eigenvector of a local Hessian-like matrix is a general
  non-convex-descent trick for any mitosis-style parameter-growth operator.
- **Minimal-offspring-count argument**: a trust-region Taylor argument
  proving 2 offspring is provably sufficient (more offspring give no
  additional first-order benefit) — a reusable template for bounding "how
  many children should this growth operator spawn."
- **Reuse of forward-pass intermediates** (opacity, projected mean/covariance)
  to make an extra per-primitive Hessian/eigendecomposition step nearly free
  — a general pattern for adding second-order diagnostics to a rasterization
  pipeline without a second forward pass.

---

### Deep Dive

#### Core Novelty
Replaces 3DGS's empirical "average positional gradient magnitude exceeds a
threshold" split trigger with a provably loss-decreasing trigger, plus a
closed-form offspring count, placement direction, and opacity normalization,
all derived from a single second-order Taylor expansion of the rendering
loss. The mechanistic insight: once a Gaussian's own gradient has vanished
(it is locally converged), the only way splitting it can still lower loss is
if it sits at a saddle rather than a minimum of the loss landscape — exactly
detectable as "the local splitting-matrix has a negative eigenvalue" — and
the optimal remedy is to split along that negative-curvature direction
rather than along a randomly sampled offset.

#### Mathematical Formulation
Loss after replacing Gaussian `i` (parameters `θ^(i)`) with `m_i` offspring
(`ϑ_j^(i) = θ^(i) + δ_j^(i)`, weights `w_j^(i)`), Taylor-expanded around the
parent parameters and evaluated per-Gaussian at each densification step:

$$\mathcal{L}(\vartheta, w) = \mathcal{L}(\theta) + \nabla_\theta \mathcal{L}(\theta)^\top \mu + \tfrac12 \mu^\top \nabla^2_\theta \mathcal{L}(\theta)\mu + \tfrac12\sum_i\sum_j w_j^{(i)}\, \delta_j^{(i)\top} S^{(i)}(\theta)\, \delta_j^{(i)} + O(\varepsilon^3)$$

The first two correction terms are the ordinary first/second-order change
from moving the parent's own parameters; for a converged (stationary-
gradient) Gaussian, `∇_θ L(θ) = 0` and this vanishes, leaving only the last,
purely split-controlled term.

**Splitting matrix** (the paper's central novel quantity), accumulated during
training's backward pass and evaluated/eigendecomposed at each densification
step:

$$S^{(i)}(\theta) = \mathbb{E}\!\left[\frac{\partial \ell}{\partial \sigma}\,\Pi(x;\theta^{(i)})\, \nabla^2_{\theta^{(i)}} \sigma\,\Pi(x;\theta^{(i)})\right]$$

An expectation over rendered pixels `x` of the per-pixel loss-gradient
weighted Hessian of the Gaussian's projected opacity response `σΠ(x;θ)` with
respect to its own parameters — i.e., how curved the loss is in the space of
"where would a duplicate of this Gaussian want to sit."

**Splitting characteristic function** — the only loss term the split decision
controls:

$$\Delta^{(i)}(\delta^{(i)}, w^{(i)}; \theta) \triangleq \tfrac12 \sum_j w_j^{(i)}\, \delta_j^{(i)\top} S^{(i)}(\theta)\, \delta_j^{(i)}$$

**Necessary/sufficient condition** for splitting to reduce loss (evaluated
once per Gaussian per densification step, gating "should this Gaussian
split"): `λ_min(S^{(i)}(θ)) < 0`, i.e. `S^{(i)}` is not positive
semi-definite.

**Constrained optimum.** With a trust region of radius `ε` on offspring
displacement and weights on the simplex:

$$\min_{\vartheta,w} \mathcal{L}(\vartheta,w) \quad \text{s.t.} \quad \lVert \vartheta_j^{(i)}-\theta^{(i)}\rVert_2 \le \varepsilon,\ \ \sum_j w_j^{(i)} = 1$$

closed-form solution: if `λ_min(S^{(i)}) ≥ 0`, set `m_i* = 1` (no split);
else set `m_i* = 2`, `w_1^{(i)*} = w_2^{(i)*} = 1/2`, and

$$\delta_1^{(i)*} = \varepsilon \cdot v_{\min}(S^{(i)}(\theta)), \qquad \delta_2^{(i)*} = -\varepsilon \cdot v_{\min}(S^{(i)}(\theta))$$

where `v_min` is the unit eigenvector of the most negative eigenvalue. This
is evaluated per-Gaussian at densification time and replaces 3DGS's
clone/split decision and offset-sampling step. The `w* = 1/2` split weight is
what the paper's abstract calls "an analytical solution for normalizing
offspring opacity" — offspring opacity is set to half the parent's rather
than copied or heuristically adjusted.

**Tractable Hessian approximation** used to compute `S` efficiently
(position-only, `dimΘ = 3`), evaluated per-pixel during rasterization and
reusing forward-pass intermediates so the extra Hessian requires no second
forward pass:

$$\nabla^2_\theta\, \sigma\,\Pi(x;\theta^{(i)}) \approx \sigma^{(i)}\,\Upsilon\Upsilon^\top - \sigma^{(i)}\,P^\top \Pi(\Sigma^{(i)})^{-1} P, \qquad \Upsilon \triangleq P^\top \Pi(\Sigma^{(i)})^{-1}\big(x - \Pi(p^{(i)})\big)$$

where `P` is the 2D projection Jacobian and `Π(Σ)`, `Π(p)` the projected
covariance and mean.

#### Algorithm / Pipeline Changes
1. During ordinary backward passes, accumulate a per-Gaussian estimate of the
   3×3 (position-only) splitting matrix `S^(i)` into a persistent buffer
   (`xyz_splitting_mat_accum` in the released code), using one of three
   interchangeable estimators exposed as `S_estimator ∈ {partial, approx,
   inv_covar}` (default `inv_covar`).
2. Every `densification_interval` steps (default 100), starting at
   `densify_from_iter` (default 500) and continuing until
   `densify_until_iter` (default 15,000 in code — not restated in the
   accessible paper text): batch-eigendecompose each Gaussian's accumulated
   3×3 splitting matrix, keeping only the least eigenvalue/eigenvector pair.
3. Build the candidate-for-splitting mask by combining (a) the original ADC
   view-space gradient-norm test against `densify_grad_threshold` (0.0002,
   3DGS's original default) — repurposed in this codebase as a "gradient ≈ 0
   / stationary" test rather than the original "large gradient" trigger —
   AND (b) the saddle test `S_eigvals < densify_S_threshold` (default
   `-1e-6`). The released code exposes each condition as an independently
   toggleable flag (`stationary`, `no_saddle`, `no_eig_cond`, `no_uncertain`,
   `no_eig_upd`) for ablating design choices.
4. Route selected Gaussians into split vs. clone exactly as vanilla 3DGS
   does, by comparing `max(scaling)` against `percent_dense * scene_extent`
   (`percent_dense = 0.01`, unchanged): large Gaussians go through
   `densify_and_split_steepest`; small Gaussians are cloned.
5. `densify_and_split_steepest` creates `N=2` offspring per selected
   Gaussian: new positions are the parent position ± a magnitude sampled
   from the parent's own scaling distribution, but displaced along the
   steepest-descent eigenvector `v_min(S)` instead of a random direction; new
   scale uses `scaling_inverse_activation(get_scaling.repeat(N,1) /
   (0.8*N))` — the same `1/(0.8N)` reduction factor as vanilla 3DGS's split
   (unchanged, not itself a new contribution); new opacity, when the
   `no_div_opacity` flag is unset, is `inverse_sigmoid(get_opacity * 0.5)` —
   implementing the paper's `w* = 1/2` analytic halving — and falls back to a
   raw `repeat` of the parent's opacity (no halving) when that flag is set.
6. Clone-routed (small) Gaussians are displaced in place along ± the same
   steepest-descent eigenvector, scaled by the current learning rate rather
   than a scale-sampled magnitude.
7. The whole path is opt-in via `--densify_strategy steepest` (vs. `adc` to
   recover stock 3DGS); loss, optimizer, opacity reset every
   `opacity_reset_interval` (3000 steps, unchanged), and pruning are
   otherwise unmodified from 3DGS.

#### Key Hyperparameters & Design Choices
- Densification cadence: every 100 steps starting at step 500
  (`densify_from_iter=500`, `densification_interval=100`) — stated
  explicitly in the paper's §5.1 implementation details and matches the code
  default.
- Saddle/eigenvalue threshold: `densify_S_threshold = -1e-6` — stated
  explicitly in the paper ("the threshold for the smallest eigenvalues of
  splitting matrices is chosen as −1e−6") and matches the code default.
- Densification stop iteration: `densify_until_iter = 15,000` — from code
  defaults only; not confirmed restated in the paper text that was
  accessible, so treat as code-derived.
- View-space gradient/stationarity threshold: `densify_grad_threshold =
  0.0002` — this is 3DGS's original default, reused here as one of the
  AND-ed gating conditions; not called out as a newly tuned value by the
  paper.
- Clone/split routing threshold: `percent_dense = 0.01` — inherited
  unchanged from 3DGS.
- Opacity-reset interval: 3000 steps — inherited unchanged from 3DGS.
- Splitting-matrix estimator: `S_estimator = "inv_covar"` by default
  (`partial` and `approx` alternatives exist in code, not selected as
  default).
- Offspring scale-reduction factor `1/(0.8·N)` with `N=2`: inherited
  unchanged from vanilla 3DGS's split operator — the paper's own derived
  "offspring magnitude" contribution is specifically the analytic opacity
  halving (`w*=1/2`), not this position-scale factor.
- Trust-region radius `ε`: appears symbolically in the theorem (offspring
  displaced by `ε·v_min`) but no explicit numeric value is given in the
  accessible paper text — the paper states "all other hyper-parameters are
  kept the same with 3DGS's default settings," implying `ε` is realized
  implicitly through the existing scale-sampled / learning-rate-scaled
  displacement magnitudes at the implementation level. Not specified as an
  explicit numeric constant — not guessing a value.
- Hardware: single NVIDIA V100 GPU per scene. No other training
  hyperparameters (base learning rates, total iteration count, loss term
  weights) are stated as changed from 3DGS defaults.

Ablation Summary: omitted — no dedicated ablation table with quantitative
per-component deltas (e.g. PSNR/point-count when disabling the saddle
condition, the stationarity gate, or opacity halving) could be located in the
accessible paper text, despite the released code exposing toggle flags
(`no_saddle`, `no_eig_cond`, `stationary`, `no_uncertain`, `no_div_opacity`)
that suggest such ablations were run internally.

#### Implementation Reality
- **Framework:** PyTorch + CUDA, forked directly from the official 3DGS
  reference implementation — reuses its `diff-gaussian-rasterization` and
  `simple-knn` submodules (PyTorch 1.12.1+cu116 per the repo's setup
  instructions).
- **Key files:** `scene/gaussian_model.py` holds essentially all the novel
  logic — splitting-matrix accumulation (`xyz_splitting_mat_accum`, updated
  from `splitting_mats.grad`), batched eigendecomposition
  (`eigh_in_batch(..., least_k=1)`), the candidate-mask construction
  combining gradient/stationary/saddle tests, and
  `densify_and_split_steepest` (offspring position/scale/opacity). `arguments/
  __init__.py` defines the new CLI surface (`densify_strategy`,
  `densify_S_threshold`, `S_estimator`, and the option-string flags).
  `train.py` is the driving loop; `full_eval.py`/`metrics.py` are the
  standard 3DGS evaluation scripts, unchanged.
- **Notable implementation details not obvious from the paper text:** (1)
  the released code still gates splitting on the original 3DGS view-space
  gradient threshold, repurposed as a "gradient ≈ 0 / stationary" filter
  ANDed with the new saddle-eigenvalue test — the paper's math only strictly
  requires `λ_min(S) < 0`, so this gradient pre-filter is an engineering
  addition, not something derived in the theory section. (2) The offspring
  scale reduction (`1/(0.8N)`) is literally vanilla 3DGS's split formula,
  unchanged — only the *direction* of displacement and the *opacity* value
  are the paper's derived contributions, which is easy to mis-read from the
  abstract's "magnitude of offspring Gaussians should be halved" language.
  (3) The analytically-derived opacity halving is implemented as a toggle
  (`no_div_opacity` reverts to vanilla copy-without-halving) rather than an
  unconditional default. (4) Three interchangeable estimators for the
  splitting matrix ship in code (`partial`, `approx`, `inv_covar`) though the
  paper's main derivation centers on one formulation; `inv_covar` is the code
  default.

---

## Relevance to ADAGS

Primary theory anchor for any claim that gradient-threshold densification is
biased or suboptimal. Static-scene scope: does not treat temporal presence,
per-view occlusion, or dynamic under-densification — the delta an ADAGS
densification claim would need. Cite for the "densification as optimization"
framing rather than reuse directly.

## Connections

- Supports theory framing for [[gap_map#G5 - Capacity Allocation Must Be Matched And Dynamic-Aware]]

## Sources

- https://arxiv.org/abs/2505.05587
