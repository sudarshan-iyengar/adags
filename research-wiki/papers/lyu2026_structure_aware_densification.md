---
type: paper
node_id: paper:lyu2026_structure_aware_densification
title: "Faster 3D Gaussian Splatting Convergence via Structure-Aware Densification"
authors: ["Linjie Lyu", "Ayush Tewari", "Jianchun Chen", "Thomas Leimkühler", "Christian Theobalt"]
year: 2026
venue: "SIGGRAPH 2026"
external_ids:
  arxiv: "2604.28016"
tags: [static-gs, densification, frequency, aliasing]
status: deep-dived
---

# Faster 3D Gaussian Splatting Convergence via Structure-Aware Densification

**Paper:** https://arxiv.org/abs/2604.28016
**Code:** https://github.com/LinjieLyu/SADGS
**Base method:** 3D Gaussian Splatting (Kerbl et al. 2023), with its standard adaptive
density control replaced. Also folds in AbsGS (Ye et al. 2024) gradient-based
empty-region densification as a complementary path, and is compared against
Taming-3DGS (Mallick et al. 2024), FastGS (Ren et al. 2026), DashGaussian
(Chen et al. 2025), Mini-Splatting (Fang & Wang 2024), Speedy-Splat (Hanson
et al. 2025a), and Mip-Splatting (Yu et al. 2024, contrasted as an
anti-aliasing/low-pass approach rather than a densification approach).

## One-line thesis
Standard 3DGS positional-gradient densification conflates geometric
misplacement with frequency aliasing, so it reactively discovers under-resolved
Gaussians only after many split-and-retrain cycles; explicitly comparing each
Gaussian's projected screen-space extent to the locally estimated dominant
texture wavelength yields a per-axis violation score that determines the
correct anisotropic split factor analytically in one step, letting training
converge in 3k-7k iterations instead of 30k.

## Problem / Gap
Standard 3DGS adaptive density control thresholds the screen-space positional
gradient to decide when to split/clone a Gaussian, but this signal does not
distinguish "Gaussian is in the wrong place/shape" from "Gaussian is correctly
placed but too large to resolve local high-frequency texture" — a Gaussian
that fully covers a textured region but is too coarse produces blur while
generating only small positional gradients, so gradient thresholding is slow
to fix it. Because standard splitting is isotropic (one Gaussian becomes two),
a region that needs 16x denser sampling requires roughly four successive
split-and-train cycles (2^4=16), each spanning hundreds of iterations, which
is why standard 3DGS training runs ~30k iterations.

## Method
The method precomputes a multi-scale structure tensor / Laplacian scale-space
pyramid over each training image to estimate the locally dominant texture
frequency (equivalently, a minimum wavelength) at every pixel. During
training, each Gaussian's three principal axes are projected to screen space
and compared against the locally sampled minimum wavelength to produce a
per-axis frequency-violation ratio η; η>1 means the Gaussian's projected
extent exceeds the wavelength of the texture it must represent, i.e. it is
under-resolved. Per-axis violation votes are aggregated across the views seen
since the last densification pass, and only Gaussians whose high-violation (or
low-violation) vote fraction crosses a consistency threshold are split (or
pruned), which suppresses noisy single-view decisions. Splitting is anisotropic
and one-shot: the number of children along each axis is computed directly from
η via a concave mapping (n=⌈√η⌉) rather than discovered through repeated
binary splits, and this frequency-driven path runs alongside a retained
AbsGS-style gradient-based path that fills empty regions the frequency
criterion alone would miss.

## Assumptions
Operates on the standard 3DGS setting: static, multi-view calibrated scenes
with COLMAP camera poses and photometric (RGB) supervision, extending the
standard point-cloud Gaussian initialization. It assumes local image content
has a well-defined dominant spatial frequency/wavelength that a Gaussian
multi-scale pyramid plus structure-tensor eigenanalysis can estimate, and that
enough training views are available per region to make the multiview
consistency vote meaningful.

## Limitations / Failure Modes
The paper reports PSNR slightly below competing methods because 3k training
iterations (vs. 30k) leave Gaussian positions not fully converged, producing
small positional offsets that manifest as high-frequency rendering errors
that penalize PSNR more than perceptual metrics. The frequency-aware criterion
also tends to over-densify relative to budget-constrained baselines (12-53%
more Gaussians without multiview consistency gating) because η is matched to
the peak local frequency across a Gaussian's whole footprint even when only
part of that footprint (e.g., an edge) needs the finer resolution. Stochastic
jitter sampling used to estimate per-Gaussian frequency requirements
introduces minor run-to-run variance.

## Reusable Ingredients
- **Multi-scale structure-tensor + Laplacian scale-space frequency
  estimation** — a general local-dominant-frequency/wavelength estimator per
  training image, usable to diagnose whether any splatting-style
  representation is too coarse for the content it covers.
- **Per-primitive per-axis frequency-violation metric (η)** — decouples
  "primitive too coarse for local content" from positional-gradient signals,
  giving a direct, analytic resolution diagnostic instead of an indirect,
  reactive one.
- **Concave-mapped one-shot anisotropic splitting (n=⌈√η⌉)** — computes the
  required child count per axis directly instead of discovering it through
  repeated binary splits, cutting densification wall-clock/iteration cost.
- **Multiview-consistency vote gating before acting on a per-view signal** —
  a general noise-reduction pattern (fraction-of-views-agreeing threshold)
  applicable to any per-view per-primitive decision in multi-view training.
- **Rescaling the position-learning-rate schedule to the shortened iteration
  budget** — ablation shows this is necessary and highly impactful whenever
  training iteration count is cut; reusing an LR schedule tuned for 30k
  iterations at 3k iterations causes large quality loss.

---

### Deep Dive

#### Core Novelty
Relative to standard 3DGS adaptive density control, this paper replaces the
reactive positional-gradient threshold with a proactive frequency-domain
diagnostic: it directly compares a Gaussian's projected screen-space extent
against the locally estimated minimum image wavelength (from a multiscale
structure-tensor/Laplacian pyramid) and uses the resulting per-axis ratio η to
compute, analytically and in one shot, how many children are needed along
each axis. The key insight is that positional gradients conflate two distinct
failure causes (wrong position/shape vs. insufficient resolution); isolating
the resolution-insufficiency cause lets the method compute the correct
subdivision factor directly instead of discovering it iteratively over many
split-and-retrain cycles, which is what collapses the standard 30k-iteration
schedule down to 3k-7k iterations.

#### Mathematical Formulation

**Projected Gaussian axes** (evaluated per-Gaussian, before rasterization,
to obtain the screen-space extent used by the violation metric):
$$v_k = JWR\,S_k,\quad k \in \{x, y, z\}$$
where $\Sigma = RSS^TR^T$ is the Gaussian covariance ($S=\mathrm{diag}(s)$,
$s=(s_x,s_y,s_z)$ the scale vector, $R$ the rotation), $S_k$ is the $k$-th
column of $S$, $W$ is the view transform, and $J$ is the projection Jacobian.

**Structure tensor at scale $\sigma$** (computed per training image, per
scale level, as part of pipeline precomputation):
$$S_\sigma = G_\rho * (\nabla I_\sigma \nabla I_\sigma^T), \quad I_\sigma = G_\sigma * I$$
with component form $S_{xx}=\sum_c I_{x,c}^2$, $S_{xy}=\sum_c I_{x,c}I_{y,c}$,
$S_{yy}=\sum_c I_{y,c}^2$ (summed over color channels $c$); $\rho$ is the
integration scale.

**Multi-scale aggregation** (precomputed per training image, once, before
training; produces the per-pixel dominant-frequency estimate used later):
$$\hat{S}_l = \frac{S_l}{\mathrm{tr}(S_l)+\epsilon} \qquad
E_l = \|I_{l-1}-I_l\|_2 \qquad
\bar{S} = \frac{\sum_{l=0}^{L} E_l^{\gamma}\,\omega_l^2\,\hat{S}_l}{\sum_{l=0}^{L} E_l^{\gamma}+\epsilon}$$
$\hat{S}_l$ normalizes the tensor at level $l$; $E_l$ is the band-pass energy
between successive pyramid levels (a Laplacian-pyramid difference); $\gamma$
controls how sharply higher-energy bands dominate the aggregate; $\omega_l$ is
the spatial frequency associated with level $l$; $\bar{S}$ is the final
aggregated structure tensor at a pixel.

**Minimum local wavelength** (derived from $\bar S$'s dominant eigenvalue
$\lambda_1$):
$$\Lambda_{\min} = \frac{1}{\sqrt{\lambda_1}+\epsilon}$$

**Frequency violation metric** (evaluated per-Gaussian per-axis, each time
violation statistics are updated for a sampled view — this is the paper's
central novel quantity):
$$\eta_k = \frac{\|v_k\|_2}{\Lambda_{\min}}$$
$\eta_k>1$ means the Gaussian's projected extent along axis $k$ exceeds the
local texture's minimum wavelength, i.e. the Gaussian is too coarse to
resolve that content. An ablated alternative,
$\eta_k^{(\mathrm{proj})}=\sqrt{S_{xx}u_k^2+2S_{xy}u_kv_k+S_{yy}v_k^2}$
(projecting the structure tensor itself onto the Gaussian's axis direction
$(u_k,v_k)$), is theoretically more principled but performs worse in practice
(9.3% worse LPIPS on Mip-NeRF360), which the paper attributes to axis
rotations not yet being well-optimized early in training.

**Multiview consistency gating** (evaluated after accumulating per-axis
high/low violation votes across the views seen since the last densification
event; gates whether the split/prune below actually fires):
$$\frac{N_{\text{high}}}{N_{\text{total}}} > \tau_{\text{split}} \quad\Rightarrow\quad \text{split axis}$$
$$\frac{N_{\text{low}}}{N_{\text{total}}} > \tau_{\text{prune}} \ \text{and}\ \alpha < \tau_\alpha \quad\Rightarrow\quad \text{prune}$$
with $\tau_{\text{split}}=\tau_{\text{prune}}=0.8$ and $\tau_\alpha=0.1$;
per-view votes are classified "high" when $\eta>1$ and "low" when $\eta<0.1$.

**Split factor and children generation** (evaluated per axis of a Gaussian
selected for structure-aware splitting; replaces the standard binary-split
step):
$$n = \lceil \sqrt{\eta} \rceil$$
$$\mu_{\text{child}}^{(i,j,k)} = \mu_{\text{parent}} + R_{\text{parent}}\cdot(s_{\text{parent}}\odot g^{(i,j,k)}), \qquad
s_{\text{child}} = s_{\text{parent}} \oslash (n_x,n_y,n_z)$$
$g^{(i,j,k)}$ enumerates grid coordinates in a unit cube at resolution
$n_x\times n_y\times n_z$; $\odot$/$\oslash$ are elementwise
multiply/divide. The square-root (concave) mapping from $\eta$ to $n$ was
chosen empirically (ablated against $\eta^{0.25}$ and $\eta^{0.75}$ mappings)
to avoid excessive splitting.

#### Algorithm / Pipeline Changes
1. Before training, precompute the multi-scale structure-tensor pyramid (5
   levels, $\sigma_l=1.5^l$ for $l=0,\dots,4$, integration scale
   $\rho_l=3\sigma_l$) for every training image and derive per-pixel
   $\Lambda_{\min}$; this is a one-time, GPU-vectorized cost of about 0.7s
   per dataset.
2. Every 500 iterations, for each visible Gaussian in each sampled view:
   project its 3 principal axes to screen space (projected-axis equation
   above), stochastically jitter-sample pixel locations under the Gaussian's
   screen footprint, look up $\Lambda_{\min}$ there, compute $\eta_k$ per
   axis, and accumulate per-Gaussian per-axis running high/low vote counts.
3. After the view pass, apply multiview-consistency gating: Gaussians whose
   per-axis high-vote fraction exceeds $\tau_{\text{split}}=0.8$ are marked
   for structure-aware split; Gaussians whose low-vote fraction exceeds
   $\tau_{\text{prune}}=0.8$ with opacity $\alpha<0.1$ are marked for
   pruning.
4. Marked Gaussians are split anisotropically in one shot: per-axis split
   factors $n_x,n_y,n_z=\lceil\sqrt{\eta_x,\eta_y,\eta_z}\rceil$ generate an
   $n_x\times n_y\times n_z$ grid of children replacing the parent — this
   substitutes for the standard iterative binary-split-and-retrain loop.
5. In parallel, the standard/AbsGS gradient-based densify-and-clone path
   still runs every 100 iterations to populate empty/under-covered regions
   the frequency criterion does not address.
6. Total training iterations are cut to 3,000 (Mip-NeRF360, Deep Blending) or
   7,000 (Tanks & Temples, which has a wider camera distribution), versus the
   standard 30,000, with `position_lr_max_steps` rescaled to match (3k or 7k
   instead of 30k) — omitting this rescale is the largest single ablated
   quality regression reported (see Ablation Summary).
7. Batch size is set to 2 for indoor scenes (Mip-NeRF360, Deep Blending) and
   1 for outdoor scenes (Tanks & Temples) to improve background coverage
   under the compressed schedule.
8. The photometric loss keeps the standard L1+SSIM combination, plus an
   additional L2 term the paper states is added for faster convergence (exact
   weight not specified in the extracted text).

#### Key Hyperparameters & Design Choices
- Scale-space levels: $L=4$, i.e. $l=0,\dots,4$; $\sigma_l=1.5^l$
  ($\sigma_0=1,\ \sigma_1=1.5,\ \sigma_2=2.25,\ \sigma_3=3.375,\ \sigma_4=5.0625$);
  integration scale $\rho_l=3\sigma_l$.
- Energy exponent $\gamma=3.0$ in the aggregation formula.
- Split-factor mapping exponent $p=0.5$ (i.e. $n=\lceil\eta^{0.5}\rceil$),
  chosen from an ablation over $p\in\{0.25,0.5,0.75\}$.
- Multiview consistency thresholds: $\tau_{\text{split}}=\tau_{\text{prune}}=0.8$.
- Per-view $\eta$ classification thresholds: "high" at $\eta>1$, "low" at
  $\eta<0.1$.
- Opacity prune threshold: $\alpha<0.1$ (combined with the low-vote fraction
  condition).
- Structure-aware densification cadence: every 500 iterations. AbsGS-style
  empty-region densification cadence: every 100 iterations.
- Structure-tensor precomputation overhead: ~0.7s per dataset.
- Training iterations: 3,000 (Mip-NeRF360, Deep Blending); 7,000 (Tanks &
  Temples).
- `position_lr_max_steps`: rescaled to match the iteration budget (3k or 7k)
  rather than the standard 30k.
- Batch size: 2 (Mip-NeRF360, Deep Blending, indoor); 1 (Tanks & Temples,
  outdoor).
- Loss weight for the added L2 term: Not specified in paper.
- Normalization epsilon values in the structure-tensor/wavelength formulas
  (the $\epsilon$ terms in $\hat S_l$ and $\Lambda_{\min}$): exact numeric
  values Not specified in the paper text extracted (the reference
  implementation uses $10^{-6}$ and $10^{-5}$ respectively, but this is a
  repository detail, not confirmed against the paper's own stated values).
- Hardware used for all reported timings: Nvidia H100.

#### Ablation Summary
- **Position-LR schedule not rescaled to the shortened iteration budget**:
  largest measured single-change regression — PSNR −1.24 dB, LPIPS +0.064 on
  Mip-NeRF360 when the original 30k-iteration schedule is kept at 3k
  iterations. This is flagged as the most impactful factor by delta size, but
  it is a necessary implementation-matching fix rather than a novel
  algorithmic component.
- **η (wavelength-based) vs. η^(proj) (structure-tensor-projection-based)**:
  the simpler wavelength-ratio η used in the final method beats the
  theoretically stricter projection-based alternative by 9.3% LPIPS on
  Mip-NeRF360 — the single most impactful ablation among the paper's actual
  novel-mechanism choices.
- **Multiview consistency gating removed**: marginally better quality
  (LPIPS −0.001) but 12-53% more Gaussians and longer training time; kept
  because it offers a favorable quality/efficiency trade-off.
- **Split-factor exponent $p$** (Table 3, sweep over $\{0.25,0.5,0.75\}$):
  $p=0.5$ is the reported sweet spot (best LPIPS 0.197 at a moderate 4.1M
  Gaussians); $p=0.75$ raises training time to 72.4s for only marginal
  quality gain.

#### Implementation Reality
- **Framework:** PyTorch, extending the standard 3D Gaussian Splatting
  codebase, with three custom CUDA submodules:
  `diff-gaussian-rasterization_structgs` (modified rasterizer),
  `simple-knn`, and `fused-ssim`.
- **Key files:**
  - `utils/freq_utils.py` — frequency-violation computation; implements both
    the wavelength-based η and the projection-based η^(proj) as selectable
    modes (`eta_compute_mode`), and calls a `get_structure_tensor_torch`
    helper (defined elsewhere, in `loss_utils`) for the structure-tensor
    computation itself.
  - `scene/gaussian_model.py` — houses the densification/pruning logic:
    `densify_and_split_structgs()` (analytic anisotropic splitting),
    `densify_and_clone_structgs()`, `densify_and_prune_structgs()` (main
    pruning pipeline), `final_prune_structgs()`, `expand_undersized_gs()`
    (see below), plus the per-Gaussian per-axis running vote counters
    (`eta_high_count`/`eta_high_sum_3ch`, `eta_mid_count`/`eta_mid_sum_3ch`,
    `eta_low_count`). A legacy standard stochastic `densify_and_split()` path
    also remains in the file.
  - `train.py` — main training loop; `run_train.sh` — per-dataset
    hyperparameter configuration (rather than separate config files).
- **Notable implementation details** (extracted from repository inspection;
  treat as indicative rather than verbatim-verified against the paper text,
  since these come from reading the released code, not the manuscript):
  - The split factor in code is computed as
    `ks = ceil(sqrt(clamp(eta, min=1.0)))` and then clamped to the range
    `[1, 8]` per axis — the paper's description does not mention an explicit
    per-axis cap of 8 children.
  - A configurable `scale_power` argument (`args.ks_scale_power`, default
    1.0) generalizes the children-scale divisor to
    `sigma_new = sigma_old / k^scale_power`, i.e. the paper's stated
    $s_{\text{child}}=s_{\text{parent}}\oslash n$ is the $k^1$ special case.
  - An `expand_undersized_gs()` function applies a direct analytic
    correction, `log_scale_new = log_scale_old - 0.5*log(eta)`, which forces
    η to exactly 1.0 for undersized Gaussians rather than only relying on the
    discrete grid-split — this analytic-expansion path is not described in
    the extracted paper text.
  - `eta_low_count` is declared alongside the other vote counters but does
    not appear to be actively populated in the visible densify/prune code
    path, suggesting the low-violation (pruning) branch may be wired
    elsewhere or only partially implemented in the released version.
  - Opacity is capped at 0.8 inside `densify_and_prune_structgs()`, a detail
    not mentioned in the paper excerpt.
  - Two epsilon constants appear in the reference implementation (roughly
    $10^{-6}$ for covariance/Cholesky stabilization and $10^{-5}$ for the
    wavelength division), which the paper's own text (as extracted) does not
    state explicitly.

#### Failure Modes & Limitations
Quoted/paraphrased from the paper's own limitations discussion: PSNR is
"slightly lower than some competing methods," attributed to the 3k-vs-30k
iteration reduction leaving Gaussian positions under-converged, which
produces small positional offsets that "manifest as high-frequency rendering
errors." The frequency-aware criterion "tends to produce a higher number of
Gaussians compared to budget-constrained methods" because η is matched to the
peak local frequency across a Gaussian's entire footprint even when "not all
regions within a Gaussian require such fine resolution—only the edges or
texture boundaries do." The stochastic jitter sampling used to estimate
per-Gaussian frequency requirements "introduces minor variance across
training runs."

---

## Relevance to ADAGS

Third distinct static-scene account of densification failure (aliasing).
Relevant to G11 (representation frequency as a sharpness axis) and to
dynamic-region blur: high-frequency dynamic texture may be under-split for
frequency reasons independent of temporal presence. Distinguish any ADAGS
claim from this frequency mechanism.

## Connections

- Pressures [[gap_map#G11 - Representation Frequency Is A New Sharpness Axis]]

## Sources

- https://arxiv.org/abs/2604.28016
