## 1. File-access attestation

(a) **YES.**

(b) Verbatim line 58 of [interpolations.py](/D:/adags/refine-logs/review-2026-09-21/src/ex4dgs/utils/interpolations.py:58):

```python
    opacity_ = m.pow(2) / (v.exp() + var_min/2.36).pow(2)
```

(c) Median `G-U` contrast on endpoint `P2`, analysis set `itt`: **2.4890 dB**. This is the value in [absfix-stage2-report.md:123](/D:/adags/research-wiki/assets/absfix-stage2-report.md:123).

## 2. Verdict on Claude’s plan

Overall verdict: **the experimental discipline is mostly right, but the proposed technical contribution is not yet strong enough.** A plain Ex4DGS notch is incremental and only suppresses the returning object; it does not explicitly reconstruct what should replace that object during the absence. I recommend retaining the benchmark plan but replacing the plain notch with a coupled persistent-memory/gap-residual mechanism.

### What is right

- Running each baseline at its published configuration and scoring everything with one evaluator is correct. The project record already requires identical cameras, windows, masks and metric implementations [here](/D:/adags/research-wiki/operations/neurips-workshop-paper-handoff-2026-09-21.md:117).
- STG, Ex4DGS and Multi4D are the correct minimum baseline set.
- The q=0 exact-nesting requirement is excellent. The ordered-notch design has a verified interior-boundary gradient and exact q=0 equality [here](/D:/adags/research-wiki/operations/ex4dgs-multi-interval-plan-v0.2-2026-09-21.md:74).
- Claude correctly noticed that Ex4DGS’s nominally static bank has no temporal opacity: static opacity is just `sigmoid(_opacity)` [c_gaussian_model.py:355](/D:/adags/refine-logs/review-2026-09-21/src/ex4dgs/scene/c_gaussian_model.py:355), whereas only the dynamic bank receives `time_bigaussian` [c_gaussian_model.py:363](/D:/adags/refine-logs/review-2026-09-21/src/ex4dgs/scene/c_gaussian_model.py:363).
- Independent SAM 2 masks, framewise human audit, q=0 controls, montages before numbers and a mechanism-exercised precondition are all essential.
- The compute estimate is directionally right that allocation is not limiting; wall-clock scheduling is.

### Confirmed problems

1. **The existing result is not an all-endpoint success.** The frozen verdict is `P1 NOT_MET`, while `P2 CLAIM_CONDITIONS_MET` [report:231–232](/D:/adags/research-wiki/assets/absfix-stage2-report.md:231). The descriptive 12/12 positive pattern must follow—not replace—the frozen verdict.

2. **The “0.34% of pixel-times” number is not the arithmetic for these fixtures.** That value comes from an earlier utility-scaling event union [query_pack.md:606](/D:/adags/research-wiki/query_pack.md:606). With the supplied 5,100-pixel, 25/300-frame fixture figures, the fraction is **0.0310%**, not 0.34%. Details are in §4.

3. **A plain absence notch is weak novelty.** SharpTimeGS already introduces learnable flat-top lifespan modulation; Flow4DGS-SLAM models temporal opacity with a GMM; FreeTimeGS permits primitives to appear at arbitrary times; MoRel uses learnable opacity blending across temporal anchors. The exact ordered “lifespan minus one notch” may be a new implementation detail, but it is not a strong paper-level conceptual novelty by itself. [SharpTimeGS](https://arxiv.org/abs/2602.02989), [Flow4DGS-SLAM](https://arxiv.org/abs/2604.22339), [FreeTimeGS](https://arxiv.org/abs/2506.05348), [MoRel](https://arxiv.org/abs/2512.09270).

4. **The latest local plan and Claude’s plan conflict.** The current v0.3 record says “Multi4D for the paper” and demotes Ex4DGS to cheap falsification because its cohort mean is 1.06 dB below the published figure [v0.3:174–180](/D:/adags/research-wiki/operations/ex4dgs-multi-interval-plan-v0.3-2026-09-21.md:174). Claude adopts Ex4DGS without addressing that evidence. I ultimately still recommend Ex4DGS for the first method implementation—but because of deadline and code simplicity, not because the reproduction weakness disappeared.

5. **Three or six repetitions are not three or six independent seeds.** Ex4DGS’s six nominally identical runs have sd 0.3564 dB and a 0.9128 dB range [reproduction:262–288](/D:/adags/research-wiki/operations/ex4dgs-n3v-leonardo-reproduction-2026-09-19.md:262). Multi4D’s 0.3445 dB spread is likewise fixed-seed CUDA nondeterminism [reproduction:281](/D:/adags/research-wiki/operations/multi4d-n3v-leonardo-reproduction-2026-09-18.md:281). The paper must call them repeated trainings, not seed diversity.

6. **Gate 1 has a logical error.** If Ex4DGS and Multi4D already perform well on both gap and return, the proposed “published methods fail on return” benchmark thesis is refuted; it does not remain a benchmark-plus-diagnostic story unchanged.

7. **The attribution proposal is underspecified.** Ex4DGS exposes a three-vector auxiliary VJP at [renderer:66–120](/D:/adags/refine-logs/review-2026-09-21/src/ex4dgs/gaussian_renderer/__init__.py:66), but “inject mask × acc and read αT” must first pass finite-difference, mass-conservation and dominant-primitive tests. Attribution is also primarily a diagnostic, not sufficient novelty.

8. **The plan lacks a frozen statistical design for the new method.** Three scenes—not twelve prefixes—are the independent experimental units. Prefixes and repeated trainings are nested observations.

9. **The venue page-limit ambiguity is not material.** The current CFP says long papers up to eight pages and extended abstracts up to four pages, followed by “Reference and Appendix are not included”; it also confirms double-blind OpenReview review. The natural reading covers both formats. [PhysWorldAI CFP](https://physworld-org.github.io/physworld.github.io/cfp/).

### Suspected problems requiring measurement

- Claude is right that the bottle may remain in Ex4DGS’s static bank, because promotion is displacement-driven [c_gaussian_model.py:1147–1163](/D:/adags/refine-logs/review-2026-09-21/src/ex4dgs/scene/c_gaussian_model.py:1147). But the bottle’s actual bank membership is **unverified** until a contribution census is run.
- A target-only notch may improve the gap by revealing existing background, but the EL-GS render-time result warns against assuming that. In that model, the same rows painted the bottle outside the gap and the counterfactual background inside it; suppressing them cost 5.9 dB in P1 [absence-fixture-lane:309–326](/D:/adags/research-wiki/operations/absence-fixture-lane-2026-09-10.md:309).
- Reusing `colmap_0` is plausible for `cut_roasted_beef`, but source-frame identity, resolution and point-cloud provenance still need an exact check.
- Sep 26 is an optimistic deadline for all baseline results given two converters, common-metric validation and poor 1-GPU scheduling.

## 3. The workshop bar

### PersistGS

PersistGS is **not an ECCV workshop paper**. It appeared in the **CVPR 2026 Workshops**, specifically the Generative 3D Reconstruction workshop, pp. 4687–4696. [Official CVF record](https://openaccess.thecvf.com/content/CVPR2026W/GenRecon3d/html/Ramlal_PersistGS_Differentiable_Physics_for_Object_Permanence_in_4D_Gaussian_Splatting_CVPRW_2026_paper.html).

Its bar is respectable but narrow:

- Three synthetic rigid-ball scenes.
- Five training cameras and two held-out evaluation cameras, 512×512 at 60 fps.
- Object-region PSNR/SSIM/LPIPS during occlusion plus trajectory RMSE.
- GT-trajectory upper bound, linear interpolation, constant velocity and no-physics baselines.
- Differentiable rigid simulation, a centroid-silhouette loss, five physics initializations per scene and several ablations.
- Mean 17.15 dB during occlusion, +2.46 dB over constant velocity and 0.19 dB below the GT-trajectory upper bound. [PersistGS experiment details](https://arxiv.org/html/2606.03479).

That establishes an important lesson: a workshop paper need not cover many scenes if the problem is sharply defined, the mechanism is nontrivial, and the evaluation has meaningful upper bounds, baselines and ablations.

### Other calibration papers

| Work | Verified workshop status | Relevant bar |
|---|---|---|
| [CTRL-GS](https://arxiv.org/abs/2505.18306) | 4D Vision Workshop, CVPR 2025 | A real new representation—video/segment/frame residual hierarchy—evaluated on established datasets, not merely a diagnostic. |
| [4D Object-Mover](https://virtualhumans.mpi-inf.mpg.de/4D_Object_Mover/) | 4D Vision Workshop, CVPR 2025 | A clearly scoped new task and mechanism, with diffusion-guided object motion and comparisons on D-NeRF scenes. |
| [Learning Physics from RGB Videos](https://mikel-zhobro.github.io/) | 4D Vision Workshop, CVPR 2025 | A functioning 3DGS/point-transformer simulator precursor. Detailed evaluation is unavailable from the accessible page. |
| PackUV | Accepted to 4D Vision Workshop, CVPR 2026 | A Gaussian representation contribution. Acceptance is verified, but the workshop page says camera-ready PDFs are still forthcoming. |
| Lang4DGS / SP4R | Accepted to 4D Vision Workshop, CVPR 2026 | The titles establish language-driven and sparse-view 4DGS scope; technical depth cannot yet be assessed from the unavailable PDFs. |

The CVPR 2026 workshop officially lists PackUV, Lang4DGS, SP4R, LoGeR and Splannequin, while noting that camera-ready PDFs are forthcoming. [Official accepted-paper list](https://4dvisionworkshop.github.io/).

### Where this project sits now

Today, it is **below the strong-accept bar as a complete paper**, despite unusually strong controls.

It is already stronger than PersistGS in some respects:

- Real captured multiview footage rather than only simulation.
- Three scenes × four prefixes.
- Wrong-membership and timing shams.
- An authored absence/return intervention rather than an ordinary occlusion.
- A striking training-time/render-time sign reversal.
- Scene-clustered and independently re-scorable endpoints.

But it is weaker in the load-bearing respect:

- No new method currently wins both the gap and return.
- Ex4DGS and Multi4D have not been run on the fixtures.
- Unified SSIM/LPIPS results do not exist.
- The independent mask rescore is unfinished.
- The strongest frozen endpoint is mixed: P1 NOT_MET, P2 met.
- The evidence is three highly related cooking scenes with the same object class.
- The plain notch is less conceptually distinct than PersistGS’s differentiable physics mechanism.

My present score for the evidence package alone is **4/10, weak reject**. With the complementary mechanism below and successful gates, it can plausibly become **7/10, accept**. No honest collaborator can promise definite acceptance.

## 4. Benchmark arithmetic and headline claim

The image contains:

\[
1352\times1014=1{,}370{,}928\ \text{pixels}.
\]

For a 5,100-pixel region:

\[
f_{\text{spatial}}=\frac{5100}{1{,}370{,}928}
=0.0037201=0.3720\%.
\]

If it matters for 25 of 300 frames:

\[
f_{\text{pixel-time}}
=f_{\text{spatial}}\frac{25}{300}
=0.000310009
=0.03100\%.
\]

If error density were uniform, making that region perfect changes a 300-frame globally pooled PSNR by only

\[
10\log_{10}\frac{1}{1-f_{\text{pixel-time}}}
=0.00135\ \text{dB}.
\]

The often-repeated ≈0.02 dB figure is approximately the **25-frame-only spatial calculation**:

\[
10\log_{10}\frac{1}{1-f_{\text{spatial}}}
=0.01619\ \text{dB},
\]

not the full 300-frame pixel-time calculation.

More precisely, the maximum gain depends on the region’s share of total squared error, not just its area:

\[
\Delta_\max=10\log_{10}\frac{E_{\text{all}}}{E_{\text{all}}-E_{\text{event}}}.
\]

Using the twelve ungated P1 and H2 values from [absfix-stage2-paired.json](/D:/adags/research-wiki/assets/absfix-stage2-paired.json) as an approximate regional/global error-density correction gives a maximum **0.0023–0.0062 dB**, median **0.0034 dB**. This is approximate because P1 uses the event-region pooling convention, but it confirms the conclusion: whole-scene PSNR cannot carry the paper.

### What “beat the benchmark” should mean

1. **Achievable by Oct 15:** Pareto improvement on the fixture endpoints against STG, Ex4DGS and Multi4D rerun at their own settings:

   - P1 gap-core PSNR/SSIM/LPIPS.
   - P2 first-return PSNR/SSIM/LPIPS.
   - S1 settled-return secondary endpoint.
   - C1 and H2 do-no-harm checks.
   - Contribution retention as a mechanism endpoint.

2. **Achievable as a safety result:** Standard whole-scene N3V metrics showing the proposed method does not materially damage ordinary rendering. This should be described as no-harm, not SOTA.

3. **Not realistically achievable from this mechanism by Oct 15:** A convincing new whole-scene N3V SOTA claim over all published methods. That requires a stronger general base reconstruction method, a broad scene suite and effects far beyond a 0.03%-of-pixel-times intervention.

The headline quantitative claim should therefore be:

> The method Pareto-dominates the strongest baselines on both authored-absence quality and first-return quality, while preserving ordinary-region and whole-scene quality.

Do not average P1 and P2 into one flattering score. Show a two-dimensional gap–return plot and require dominance on both axes.

## 5. Ranked novel ideas

### Audit of ideas already in the plan

| Existing idea | Novelty verdict |
|---|---|
| Ordered presence notch on Ex4DGS | **Low to medium.** Exact nesting and boundary gradients are useful engineering, but flat-top lifespan and multi-modal temporal opacity already exist in SharpTimeGS and Flow4DGS-SLAM. |
| Multi-interval temporal presence | **Low alone.** Flow4DGS-SLAM’s GMM is a direct collision. |
| Event-window PSNR/SSIM/LPIPS | **Low alone.** PersistGS already uses object-region metrics during occlusion. The paired gap/return protocol is more distinctive. |
| Authored counterfactual absence fixtures | **Medium benchmark novelty.** I found no prior 4DGS benchmark with the same full-multiview removal, return, timing shams and wrong-membership controls. |
| Training-time versus render-time sign flip | **Medium-high empirical novelty**, if repeated beyond one calibration prefix and explained mechanistically. |
| Contribution-mass attribution | **Low as a diagnostic.** Alpha/transmittance attribution is natural in splatting. It becomes more interesting when used for cohort construction or a randomized training loss. |

### 1. Complementary Episode Residuals — recommended core method

**Idea.** Preserve a long-lived object representation, but give the authored absence a separate disposable residual representation. The two are driven by the same ordered event state.

Let \(n(t;b_0,b_1)\in[0,1]\) be a smooth flat-top indicator of the absence interval and \(q\in[0,1]\) the intervention strength. For the pre-gap object-memory cohort \(K\),

\[
\alpha_i^{\mathrm{mem}}(t)
=\alpha_i^0(t)\,[1-q\,n(t)],\qquad i\in K.
\]

Add a small residual cohort \(R\), initialized from the largest gap-region contributors or residual backprojections:

\[
\alpha_j^{\mathrm{res}}(t)
=q\,n(t)\,\sigma(o_j),\qquad j\in R.
\]

Thus:

- \(q=0\) reproduces the base model exactly.
- During the gap, memory primitives disappear and residual primitives reconstruct the revealed/background content.
- At return, residual support becomes exactly zero and the original object-memory cohort reappears unchanged.

This is stronger than a notch because it assigns the two incompatible visual roles to separate parameters. It directly addresses the observed fact that, in EL-GS, the same rows painted the bottle outside the gap and the counterfactual background inside it.

**Ex4DGS attachment.**

- Construct \(K\) and \(R\) using the auxiliary VJP exposed at [renderer:66–120](/D:/adags/refine-logs/review-2026-09-21/src/ex4dgs/gaussian_renderer/__init__.py:66).
- Apply the memory factor immediately after `get_opacity_at_t` [renderer:62–64](/D:/adags/refine-logs/review-2026-09-21/src/ex4dgs/gaussian_renderer/__init__.py:62), before the rasterizer receives opacities at lines 100–109.
- Append residual primitives using the existing dynamic-tensor/optimizer path modeled by [c_gaussian_model.py:1191–1210](/D:/adags/refine-logs/review-2026-09-21/src/ex4dgs/scene/c_gaussian_model.py:1191).
- Disable densification, pruning and static→dynamic promotion during the short residual fine-tune, avoiding lineage propagation through [train.py:217–240](/D:/adags/refine-logs/review-2026-09-21/src/ex4dgs/train.py:217).
- Both static and dynamic object rows are covered because the factor is applied to the already-concatenated opacity vector. No CUDA modification is required.

**Why it may be new.** Closest work:

- [SharpTimeGS](https://arxiv.org/abs/2602.02989): flat-top lifespan per primitive, but no persistent-memory/gap-residual complement.
- [Flow4DGS-SLAM](https://arxiv.org/abs/2604.22339): GMM temporal opacity, but no shared object event or paired residual role.
- [Multi4D](https://arxiv.org/abs/2606.22197): persistent and transient branches compete for ordinary photometric residuals, but are not forced into complementary states for absence and return.
- [CTRL-GS](https://arxiv.org/abs/2505.18306): video/segment/frame residual hierarchy, but no object-memory preservation or complementary absence switch.
- [CubifyGS](https://arxiv.org/abs/2606.28720): appearance/disappearance with asset retrieval and explicit pruning, not a nested differentiable gap/return representation.
- [OP2GS](https://arxiv.org/abs/2605.20044): dual opacity separates visual existence from instance-mask occupancy, not two temporal rendering roles.

I found no paper combining an exact-nested object-memory gate with a gap-only residual Gaussian cohort and evaluating the coupled representation on absence **and** return. That is promising, but not a proof of worldwide novelty.

**Transplanted idea.** This is the analogue of a video codec’s persistent reference state plus short-lived residual frames, or a hidden semi-Markov model with complementary present/absent emissions.

**Expected effect.**

- P1: residual cohort provides STG-like short-lived capacity.
- P2/S1: original memory cohort returns without being repurposed.
- C1/H2: residual is exactly zero outside the event.
- It is the only proposed mechanism here that plausibly obtains both STG’s gap quality and the current gate’s return advantage.

**Cost.** Five to seven engineering days. Approximately 5–10 GPU-h for oracle probes, 18 GPU-h for a five-arm × three-repeat screen on one scene, and ≈65 GPU-h for three arms × six repetitions × three scenes. Total ≈85–100 GPU-h.

**Non-vacuity.**

- \(K\) must explain at least 50% of pre-gap target-region contribution mass.
- \(R\) must explain at least 20% of gap-region mass after fine-tuning.
- \(q\ge0.8\) or at least one exact-zero memory frame must occur.
- q=0 rendering must match the base path within numerical tolerance.
- The residual cohort must be absent in P2/S1 and C1.

**Kill rule.** On `cut_roasted_beef`, with oracle membership and interval:

- Kill if P1 improves by <2.0 dB over base.
- Kill if it closes <50% of the base-to-STG P1 deficit.
- Kill if P2 falls by >0.3 dB.
- Kill if the residual produces visible content outside the gap.

**Risks.** Novelty collision with CTRL-GS/Multi4D; unstable residual initialization; authored background artifacts; parameter-budget criticism. Include target-only notch, residual-only, q=0 and time-shifted residual ablations.

### 2. Interventional gap–return benchmark

**Mechanism.** Treat training-time versus render-time gating as a causal intervention, not an anecdote. Cross:

- training-time versus post-hoc render-time gating;
- correct versus shifted interval;
- correct versus wrong membership;
- object-memory-only versus memory-plus-residual.

Score the vector \((P1,P2,S1,C1,H2)\), never a single average.

The existing evaluator already provides frozen event definitions and pooled MSE-based PSNR at [realdata_gate_analysis.py:76–129](/D:/adags/scripts/realdata_gate_analysis.py:76). The same-checkpoint on/off intervention is implemented in [eval_n3v_gated.py:1–44](/D:/adags/scripts/eval_n3v_gated.py:1).

**Novelty.** PersistGS measures only occlusion-region reconstruction. I found no 4DGS benchmark combining:

- full-multiview authored absence;
- first-return and settled-return endpoints;
- wrong-membership and timing shams;
- same-checkpoint render-time intervention;
- independent re-segmentation;
- contribution-level non-vacuity.

The novelty is the protocol as a whole, not “event-window PSNR.”

**Transplanted idea.** Factorial causal intervention and negative-control design.

**Effect.** No direct metric gain. It converts a local observation into defensible evidence and exposes methods that optimize P1 by sacrificing P2.

**Cost.** Three to four engineering days for adapters, common SSIM/LPIPS validation and figures. Compute overlaps with the required baseline runs: ≈95–120 new GPU-h, mostly STG on `cut_roasted_beef`.

**Non-vacuity.** Every arm must visibly exercise its event; masks must contain the edited object/background; every method must render the exact held-out camera and frame indices.

**Kill rule.** The benchmark-only paper becomes weak if:

- independent masks reverse the effect direction;
- at least two strong baselines dominate both P1 and P2;
- the training/render sign flip does not reproduce outside its original prefix;
- method outputs are visually indistinguishable inside the scored regions.

**Risks.** Only three related scenes, authored rather than physical absence, mask dependence, and reviewers calling it an evaluation note rather than a method paper. This is the fallback contribution, not the preferred sole contribution.

### 3. Dormant-landmark lifecycle protection

**Mechanism.** During the inferred absent state, memory primitives do not merely receive zero opacity; their parameters and lifecycle statistics sleep:

\[
\nabla_{\theta_i}L=0,\quad
m_i^{\text{Adam}},v_i^{\text{Adam}}\ \text{unchanged},\quad
i\notin\text{prune/densify candidates},\qquad i\in K.
\]

They resume at the first return frame. Gap gradients are routed only to ordinary or residual support.

**Attachment.** In Ex4DGS, mask the error accumulators and densification statistics around [train.py:215–234](/D:/adags/refine-logs/review-2026-09-21/src/ex4dgs/train.py:215), protect sleeping rows from `prune_invisible` at lines 237–240, and mask their optimizer gradients before [train.py:243–251](/D:/adags/refine-logs/review-2026-09-21/src/ex4dgs/train.py:243).

**Novelty.** [Splannequin](https://arxiv.org/abs/2512.05113) detects hidden/defective states and temporally anchors them, so “protect hidden Gaussians” alone is not novel. The narrower possible novelty is state-controlled optimizer **and lifecycle** dormancy paired with a separate gap renderer. [MaskGaussian](https://arxiv.org/abs/2412.20522) is also a collision around probabilistic existence, although it deliberately keeps gradients flowing to masked primitives.

**Transplanted idea.** Dormant landmarks in SLAM and parameter isolation in continual learning.

**Expected effect.** P2/S1 and primitive-retention metrics; almost no direct P1 benefit. It can support idea 1 but cannot replace it.

**Cost.** Two to three days, 15–25 GPU-h.

**Non-vacuity.** At least 20% of target contribution must come from rows that would otherwise be pruned, promoted or materially altered during the gap.

**Kill rule.** Do not implement if base target-contribution retention already exceeds 90%, or if oracle dormancy changes P2 by <0.5 dB.

**Risks.** Strong collision with Splannequin; preserving wrong primitives; optimizer-state complexity. Make this conditional on the attribution census.

### 4. Randomized contribution-cycle regularization — stretch

**Mechanism.** Give each frozen primitive/root \(r\) a resampled Rademacher code \(h_r\in\{-1,+1\}^3\). Render code-weighted αT contribution in pre-gap and return masks:

\[
\widehat L_{\mathrm{cycle}}
=\frac13\left\|
\sum_r h_r(p_r^{\mathrm{pre}}-p_r^{\mathrm{return}})
\right\|_2^2.
\]

Its expectation is the squared distance between high-dimensional contribution distributions:

\[
\mathbb E_h[\widehat L_{\mathrm{cycle}}]
=\sum_r(p_r^{\mathrm{pre}}-p_r^{\mathrm{return}})^2.
\]

Use resampled codes for training and fixed multiple draws only for diagnosis. A return-side code permutation is the matched negative control.

**Attachment.** A second Ex4DGS rasterization can replace the auxiliary `dir3D` input at [renderer:66–106](/D:/adags/refine-logs/review-2026-09-21/src/ex4dgs/gaussian_renderer/__init__.py:66) with the codes. Freeze row lifecycle during this stage so “identity” really is stable.

**Novelty.** [SA4D](https://arxiv.org/abs/2407.04504) learns temporal identity features; [TRiGS](https://arxiv.org/abs/2604.00538) preserves long-term identity with continuous motion. I found no prior using an unbiased randomized sketch of rendered αT contribution distributions as a return-cycle loss. That is the genuinely unusual part—not attribution itself.

**Transplanted idea.** CountSketch/Hutchinson random projection combined with cycle consistency.

**Expected effect.** P2, contribution retention and re-identification; little direct P1 benefit. It cannot produce STG-like gap quality without idea 1.

**Cost.** Four to six engineering days; a second render makes training roughly 2–3× more expensive. Expect 20–30 GPU-h for screening and >60 GPU-h for credible confirmation.

**Non-vacuity.** Baseline pre/return contribution distributions must differ materially, and the estimator must distinguish true from permuted identity reliably over repeated hash draws.

**Kill rule.**

- Kill if 16-draw estimator CV exceeds 0.25.
- Kill if it fails to rank true correspondence above permutation in ≥95% of probes.
- Kill if one-scene P2 gain is <0.5 dB and contribution retention improves <15 points.

**Risks.** High variance, extra rendering cost, accidental regularization toward the wrong spatial correspondence and insufficient time. Do not put this on the critical path.

### 5. Multi4D return loop closure — do not schedule unless the oracle demands it

**Mechanism.** Treat the pre-gap persistent foreground as a keyframe map. At return, estimate a small object-level SE(3) residual aligning the deformed persistent cohort to the returning target, while leaving transient detail independent.

In Multi4D, this would attach after foreground deformation at `gaussian_renderer/__init__.py:459–462`, before foreground opacity at `:468`. Persistent FG gating must also cover standalone render `:198`, mask render `:314`, and hybrid render `:468`; the local durable map identifies the mask path as a silent-bypass risk [multi4d-code-map:130–134](/D:/adags/research-wiki/operations/multi4d-code-map-2026-09-19.md:130). The transient branch is already hard-culled by its temporal marginal [forward.cu:281–285](/D:/adags/refine-logs/review-2026-09-21/src/multi4d/cuda_rasterizer/forward.cu:281).

**Novelty.** This is heavily threatened:

- PersistGS already applies visible-frame residual translations when the object re-emerges.
- CubifyGS retrieves and transforms persistent Gaussian assets.
- TRiGS preserves identity with continuous SE(3).
- MoRel uses anchor relays and bidirectional temporal blending.

Only a return-censored, loop-closure formulation would be distinguishable.

**Transplanted idea.** Object-level SLAM loop closure and keyframe memory.

**Expected effect.** P2 on genuinely moved or trajectory-shifted returns, not P1. The current authored fixture restores the original video after frame 89, so the residual may correctly be nearly zero.

**Cost.** Six to nine days, 50–80 GPU-h.

**Kill rule.** Run an oracle rigid-alignment probe first. If it improves return PSNR by <0.5 dB, stop immediately.

**Risk.** Likely vacuous on the current fixtures and too close to prior work. It is not a mid-October core idea.

## 6. Revised Sep 22–Oct 15 plan

I recommend **Ex4DGS for the first method implementation**, Multi4D as a reproduced baseline, and no Multi4D method port unless Ex4DGS fails early but Multi4D’s FG/TR census is strongly non-vacuous. Ex4DGS’s partial reproduction prevents a global-SOTA claim, but its single Python opacity junction and ≈1.2 GPU-h runtime make it the only substrate that leaves enough time for rigorous confirmation.

| Date | Work and verification | Gate / kill |
|---|---|---|
| **Sep 22** | Freeze the paper claim: P1 and P2 are co-primary; C1/H2 are harm checks; no whole-frame SOTA promise. Freeze common PSNR pooling, SSIM and LPIPS conventions. Principal confirms diff-review coverage and audit owner. | No experiments until claim/evaluator text is frozen. |
| **Sep 22–23** | Build Ex4DGS and Multi4D fixture converters. Golden-test the common evaluator against each native evaluator and hand-check ten rendered crops. Prepare STG `cut_roasted_beef` six-segment jobs. | Any camera/frame/resolution mismatch blocks baseline submission. |
| **Sep 23–24** | One Ex4DGS `cut_roasted_beef` baseline; run contribution VJP checks, bank census and an oracle Complementary Episode Residual probe. | **Gate A:** \(K\ge50\%\) pre-gap target mass; residual support can reach ≥20% gap mass; oracle P1 +2 dB, P2 loss ≤0.3. Failure kills the method before a substrate diff. |
| **Sep 24–27** | Complete published-setting baselines: Ex4DGS 3 repeats × 3 scenes; Multi4D 3 × 3; STG on `cut_roasted_beef`. Evaluate every completed model immediately with montages. | If two published baselines dominate both P1 and P2, drop the “failure on return” thesis. |
| **Sep 25–27** | One small Ex4DGS diff implementing q=0 exact nesting, memory gating and gap-only residual support. Disable row lifecycle in the short fine-tune. Principal reviews the diff before submission. | Tests: q=0 image equality, both banks covered, residual zero outside gap, checkpoint round-trip. |
| **Sep 28–30** | Single-scene screen: q=0 base, target-only notch, residual-only, complementary method, shifted-interval sham; three repeated trainings each. Inspect crops before tables. | **Gate B:** method exceeds base P1 by ≥1 dB, target-only notch by ≥0.75 dB, loses ≤0.3 dB P2, and has correct visuals in at least 2/3 repeats. Otherwise kill the method and pivot to the diagnostic paper. |
| **Oct 1–4** | Confirm base, target-only and complementary arms: 6 repeats × 3 scenes. Run original-unedited N3V no-harm cells. Generate independent SAM 2 masks and human-audit all P1/P2/S1 frames. | **Gate C:** effect larger than the 0.62 dB Ex4DGS resolution warning; same direction in ≥5/6 repetitions per scene; independent masks preserve direction; no catastrophic scene. |
| **Oct 5** | Freeze final scientific verdict. A strong headline requires closing ≥50% of the STG P1 deficit while retaining ≥80% of the present G-over-STG P2 advantage, or direct P1/P2 Pareto dominance on at least 2/3 scenes. | If unmet, no benchmark-beating language. Submit only the benchmark/causal result if it remains independently strong. |
| **Oct 5–7** | Contribution attribution, q=0/notch/residual ablations, parameter and runtime table, gap–return Pareto plot, temporal strips and failure cases. Add lifecycle dormancy only if the census proves target drift/pruning. | No sketch loss or loop-closure work unless all core results are already frozen. |
| **Oct 8** | Results and title freeze. Produce every table from machine-readable outputs. Full eight-page draft. | No new exploratory experiments after this date. |
| **Oct 9–11** | Adversarial review, novelty update through Oct 8, citation audit, claim-to-artifact audit, anonymity and limitation review. | Every number must trace to an artifact; every negative or skipped check is reported. |
| **Oct 12** | Scientific freeze. Only bug fixes that do not change the claim set. | Any result-changing bug reopens Gate C. |
| **Oct 13–15** | Final revision, figures, appendix, reproducibility checklist, style and OpenReview checks. Paper done Oct 15. | Submit-ready artifact, not “draft nearly complete.” |

### Compute

Conservative new compute:

- Ex4DGS baselines: 10.8 GPU-h.
- Multi4D baselines: 24.3 GPU-h.
- STG `cut_roasted_beef`: roughly 60–84 GPU-h if four prefixes each require six 50-frame segments.
- Oracle and screen: 20–30 GPU-h.
- Three-arm confirmation: ≈65 GPU-h.
- Original-scene no-harm and diagnostics: 25–40 GPU-h.

Total: approximately **205–255 GPU-h**, around 1.3–1.6% of 15,500 GPU-h. The important scheduler requests are therefore short walltimes: ≈2 h Ex4DGS, ≈4 h Multi4D, and separate ≈4 h STG segment jobs. Do not rely on resume.

### Decisions the principal must make on Sep 22

1. Approve replacing the plain notch with Complementary Episode Residuals.
2. Confirm that every external-substrate diff remains principal-reviewed, with a same-day review cadence.
3. Accept that the headline is event-local Pareto dominance, not whole-scene SOTA.
4. Assign the approximately 129-frame independent-mask audit.
5. Commit to certificate renewals while baseline and confirmation chains are active.
6. Decide whether to add one held-out, different-object fixture. My recommendation: only if the existing assembler can finish it in ≤1 day after Gate B; otherwise protect the core schedule.

## 7. Mock NeurIPS workshop review

**Conditional score: 7/10 — Accept. Confidence: 4/5.**

**Summary.** The paper identifies a previously under-evaluated tradeoff in dynamic Gaussian representations: temporal opacity can suppress an absent object but discard its return, whereas persistent gating preserves return but leaves poor support for the revealed scene. It introduces authored multiview absence-and-return fixtures and Complementary Episode Residuals, which pair persistent object memory with gap-only residual support.

**Strengths**

- Sharp, physically meaningful failure mode rather than another small average NVS gain.
- Method follows directly from the measured train-time/render-time sign reversal.
- Exact q=0 nesting and strong timing/membership negative controls.
- Multiple published substrates evaluated under one metric implementation.
- Gap and return reported separately, avoiding dilution by whole-frame metrics.
- Real multiview footage, independent masks, visual audits and mechanism-level contribution evidence.
- Simple enough to understand and reproduce.

**Weaknesses**

- Absence is authored by segmentation-guided editing, not naturally captured physical exit.
- Three scenes are closely related and apparently share the same object class.
- Membership and timing remain externally supplied or estimated from task-specific masks.
- Ex4DGS reproduction is below its published mean, weakening any broad quality claim.
- Only three independent scene units; repeated trainings do not replace scene diversity.
- The residual branch is conceptually adjacent to CTRL-GS and Multi4D, so the exact complementary coupling must be emphasized and ablated.
- No evidence yet that the method generalizes to multiple absence intervals or trajectory-changing returns.

**What moves it to 8/10**

- Pareto dominance over STG, Ex4DGS and Multi4D on both P1 and P2.
- One held-out fixture with a different object and scene type.
- A natural physical absence-and-return example.
- Independent masks preserving the effect.
- A successful Multi4D port showing the mechanism is substrate-independent.
- Clear evidence that target-only notch fails while the complementary residual succeeds.

**If submitted with Claude’s current plan and only a plain notch:** **4–5/10, weak reject**. The evaluation would be interesting, but the technical novelty and gap-closing mechanism would be insufficiently differentiated.

## 8. UNVERIFIED

- No literature search can prove a negative. I found no exact prior for Complementary Episode Residuals or the randomized αT sketch, but an unindexed or newly released collision may exist.
- Detailed technical/evaluation content for the 2026 4D Vision workshop’s LoGeR, PackUV, Lang4DGS and SP4R papers is unverified because the official page says camera-ready PDFs are forthcoming.
- The PhysWorldAI CFP is verified, but the final non-archival portal state and downloadable style package were not independently exercised.
- Ex4DGS bottle membership in static versus dynamic banks is unverified.
- Ex4DGS and Multi4D fixture ingestion, `colmap_0` reuse and the flame/sear `fused.ply` inventory are unverified.
- The contribution VJP has not yet been validated as exact αT mass by finite differences and conservation tests.
- Complementary Episode Residuals’ P1/P2 effects, engineering duration and residual-bank capacity are predictions, not results.
- STG on `cut_roasted_beef` is not complete.
- The supplied 5,100-pixel ROI is approximate; the arithmetic above is exact conditional on that approximation.
- Generalization from authored absence to naturally occurring physical absence-and-return remains unverified.

I made no local or remote file changes, submitted no jobs, and used Leonardo only for read-only inspection of Multi4D at HEAD `c483e82cd8daa1b4fdc460ed45882fc2fe7a22b0`.