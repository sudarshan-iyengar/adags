# Cross-model review round 2 — constructive rescoping (2026-09-21)

**Reviewer:** `gpt-5.6-sol`, `xhigh`, same thread as
[round 1](ex4dgs-multi-interval-review-round1-2026-09-21.md) (thread continuity
was used deliberately so it would build on its own critique).
**Outcome: a reframe of the project, a reversal of the substrate choice, and
one genuinely novel technical device.**

Acted on in [`plan v0.3`](ex4dgs-multi-interval-plan-v0.3-2026-09-21.md).

## Why this round was asked for

The project lead pushed back on round 1: it is easier to kill a direction than
to build one; the goal is **object permanence in 4DGS**; the novelty scare is
de-prioritised ("there are 50 papers doing static-dynamic separation — there is
always a place as long as you beat the benchmarks"); the metrics may not be
PSNR/SSIM/LPIPS at all but **4D multi-object tracking, object identification
and re-ID**; the win could be in training time, primitive count, FPS or
tracking rather than photometry; and prior findings point at **fast motion and
low-quality frame regions** as where the headroom is.

## Attestation (verified)

(a) YES. (b) line 161 of `train.py` = `loss += opt.motion_reg * diff1.norm(dim=-1).mean()`.
(c) `extracton_interval` = `3980`. Both confirmed locally before the review was
read.

## The reframe

> Learn a **persistent object memory** that survives periods without
> observations and **regenerates disposable render support** on return.

Concretely: a compact **root** carries identity, semantics, canonical
appearance and long-term state; **leaves** (densified children, transient
primitives) carry episode-local detail, hold a `parent_root_id`, and are free
to split, die and respawn.

**This answers the round-1 kill argument.** Identity no longer requires a
tensor row to survive — a copied id cannot fake continuity, because the claim
is about root contribution mass and downstream semantic behaviour rather than
row survival.

## The substrate reversal — and an error of mine

Round 1 chose Ex4DGS over Multi4D on the gradient-path argument: Multi4D's
temporal gate is in CUDA and hard-culls at `marginal_t > 0.05` with zero
gradient outside. **That is true of the transient bank only.** Multi4D's
*persistent foreground* opacity is ordinary PyTorch — and
[our own code map](multi4d-code-map-2026-09-19.md) §5 already said so, listing
`gaussian_renderer/__init__.py:468/:198/:314` as attachment points and stating
that a new parameter "gets gradient with **no CUDA change**".

I over-generalised a property of TR to the whole substrate and recommended
against the substrate that actually reproduces. **Multi4D becomes the headline
substrate; Ex4DGS stays as the cheap falsification probe.**

Multi4D fits the reframe unusually well: FG is 20–25k persistent primitives
against 198–252k transient, and its semantic stage distils features onto the
**persistent subset only**. The separation the project needs already exists and
is unexploited.

## The one new technical device: hashed lineage sketches

Give each root a reproducible Rademacher code `h_r ∈ {-1,+1}^3`, render the
code-weighted contribution through the **three auxiliary channels the renderer
already carries** for `[acc, L1, SSIM]`, and penalise the pre/post-gap
discrepancy. Because

```
E_h[ (1/3) || sum_r h_r (p_pre_r - p_post_r) ||^2 ] = sum_r (p_pre_r - p_post_r)^2
```

this trains a high-dimensional lineage-distribution objective with **no one-hot
identity buffer and no CUDA change**. One-hot would need ~64k channels; this
needs three.

**Verified numerically before adoption:** the estimator is unbiased and
converges to the true value. **Caveat recorded rather than glossed:** at 3
channels the per-draw relative error is ~33%, so it is a high-variance unbiased
estimator — fine as an SGD gradient with hashes resampled per step, not fine as
a low-noise measurement. Matched negative control: permute the codes on the
return side.

## Metric ranking (preregistration targets, not results)

Headline candidates: **return segmentation / re-ID (+5–20 J&F)** and
**persistent-root contribution retention (+15–30 pp)**. Promising secondary:
high-motion/disocclusion strata (+0.5–1.5 dB).

Named traps: whole-frame PSNR (0–0.05 dB), mAcc (flat, background-dominated),
FPS (0–5%), reconstruction iterations (flat or worse), standard Neu3D mIoU
(diluted).

**Instrument defect:** Multi4D's `eval_seg.py` computes full-image pixel
accuracy and **silently skips missing predictions** — fail-open. Must be made
fail-closed before any semantic number is evidence.

## Benchmark position

**No turn-key benchmark supplies calibrated multiview 4D reconstruction +
certified all-camera disappearance + return + identity association.**
Neu3D-Mask is a semantic harness with binary masks, not proof of permanence.
BEHAVE is proposed for a certified-occlusion census; InterCap as fallback;
MOSE/OVIS are monocular and external-baseline only. **A small certified
physical benchmark may itself be part of the contribution.**

## Not verified

The Multi4D upstream claims (semantic stage excluding TR, `eval_seg.py`
fail-open, the 16-dimension semantic default) were read by the reviewer from
the upstream repository. **The Leonardo certificate expired during this round,
so they were not re-verified against our pinned checkout.** They must be
checked before anything is built on them; the README corroborates the
persistent-subset distillation, which is the load-bearing one.

## Full text

Verbatim below.

---

(a) Can you read files? YES.

(b)
```python
            loss += opt.motion_reg * diff1.norm(dim=-1).mean()
```

(c) `"extracton_interval"` is `3980`.

## Bottom line

The strongest formulation is:

> A compact persistent object root carries identity, semantics, canonical appearance, and long-term state; disposable render-detail Gaussians may split, die, and respawn as descendants of that root.

This is better than requiring the same Gaussian row to survive. It matches Multi4D’s architecture unusually well: semantic training already excludes the huge transient bank and learns features only on persistent foreground/background Gaussians ([exact reproduced commit](https://github.com/BatFaceWayne/Multi4D/blob/c483e82cd8daa1b4fdc460ed45882fc2fe7a22b0/train_semantic.py#L1407-L1509)). Multi4D has only about 20–25k persistent foreground Gaussians but 198–252k transient Gaussians ([reproduction results](/D:/adags/research-wiki/operations/multi4d-n3v-leonardo-reproduction-2026-09-18.md:252)). That division is not merely an implementation detail; it is the representation you need.

Use Ex4DGS as the cheap mechanism probe. Use Multi4D for the headline result.

## Q1. Winnable formulations

The score below is a planning prior: \(P(\text{win}) \times \text{evidence strength}/\text{GPU-weeks}\). It is useful for ordering work, not a scientific statistic.

| Rank | Formulation | Mechanism | Primary win | Benchmark | Planning score |
|---|---|---|---|---|---:|
| 1 | Ex4DGS notch + lineage-sketch cycle | Adopted ordered lifespan-minus-notch gate; attach seeded random codes to root lineages and require the same contribution distribution before and after the gap | Fast proof that identity-preserving reactivation is learnable | Three authored fixtures; return-withheld masks; event ROI | ≈5.6 |
| 2 | Multi4D persistent root + transient leaves | Identity/semantic/canonical state lives on FG roots; TR descendants carry temporary detail and inherit root ID; root visibility uses the notch gate | Return re-ID, semantic transfer, association, possibly lower transient count | Return-withheld Neu3D-Mask protocol, then certified physical gaps | ≈4.3 |
| 3 | Persistent root + episode-specific pose residual | Same root and appearance, but each visible episode receives a small SE(3)/deformation re-entry residual | Fast-motion return accuracy and lower recovery latency | High-motion/disocclusion strata; 2D/3D tracks | ≈2.5 |
| 4 | Permanence-aware lifecycle/compression | On return, reuse an existing root and grow new leaves from it instead of creating unrelated roots; prune leaves aggressively | Primitive count, storage, semantic-training cost | Matched-quality count/bytes and wall time | ≈2.2 |

The first ranks highest only because it is cheap. The second is the actual paper formulation.

### 1. Ex4DGS notch plus randomized lineage sketches

The existing dummy three-channel render pathway should become a trainable identity signal, not merely an attribution tool. Round 1 already identified its contribution-weighted VJP use ([v0.2, line 103](/D:/adags/research-wiki/operations/ex4dgs-multi-interval-plan-v0.2-2026-09-21.md:103)).

Assign every root lineage a reproducible Rademacher code \(h_r\in\{-1,+1\}^3\). Render the code-weighted contribution distribution in pre-gap and post-gap target ROIs and penalize their discrepancy. Across independently resampled hashes,

\[
E_h\left[\frac{1}{3}\left\|\sum_r h_r(p_r^{pre}-p_r^{post})\right\|^2\right]
=
\sum_r (p_r^{pre}-p_r^{post})^2.
\]

That approximates a high-dimensional lineage-distribution loss without one-hot lineage buffers or CUDA changes. A return-side code permutation is the matched negative control.

This gives you a precise result: “The representation preserved the object’s root contribution distribution,” not merely “the image looked similar.”

### 2. Multi4D persistent roots with transient leaves

Implement the notch/presence function on FG roots, not on TR. The round-1 objection to Multi4D concerned the TR CUDA marginal, whose forward and backward both hard-cull below 0.05 ([code map, line 67](/D:/adags/research-wiki/operations/multi4d-code-map-2026-09-19.md:67)). FG opacity is ordinary PyTorch and accepts a learned presence multiplier without CUDA modification ([code map, line 128](/D:/adags/research-wiki/operations/multi4d-code-map-2026-09-19.md:128)).

Each FG root should own:

- `root_id`;
- semantic feature;
- canonical appearance;
- long-term motion state;
- visibility/lifespan-minus-notch parameters.

Each transient or densified child should own:

- `parent_root_id`;
- episode/local appearance residual;
- short-term motion/detail;
- ordinary birth/death metadata.

The parent identity remains stable even when its rendering support changes. This is a much more defensible definition of object permanence than exact tensor-row survival.

### 3. Shared identity, episode-specific return pose

A visibility gate alone cannot fix a primitive that returns at the wrong location. Give each visible episode a low-capacity pose/deformation offset relative to the canonical root. Regularize appearance and semantic identity to remain shared while allowing motion state to jump across an unobserved interval.

This directly targets the empirical fact that return is harder than absence. It is likely to help fast motion more than adding further flexibility to the gate.

### 4. Permanence as compression

Prevent new root creation within a matched target region after return when an eligible dormant root exists. New Gaussian detail may still be born, but it must attach to that root.

This formulation can win on count/storage only if it changes lifecycle policy. A notch gate alone adds parameters and will not reduce primitive count.

## Q2. Which axis is most likely to win?

The best headline axis is post-gap identity transfer: return J&F/mIoU, association accuracy, re-ID success, and persistent-root contribution.

The effect sizes below are preregistration targets, not claimed results.

| Axis | What must change | Plausible result | Trap assessment |
|---|---|---:|---|
| Return segmentation/re-ID | Same root/semantic state must reappear without post-gap target supervision | +5–20 J&F or IoU points on a deliberately hard censored-return split; materially fewer swaps | Best headline |
| Persistent-root contribution | Return rendering must come from pre-gap roots or their descendants | +15–30 percentage points root-mass retention | Strong mechanism endpoint |
| High-motion/disocclusion quality | Episode pose or trajectory state must correct return geometry | +0.5–1.5 dB within predefined hard strata | Promising secondary |
| Temporal consistency | Shared state must reduce incorrect post-gap motion/appearance | 5–20% lower visibility-aware warped error | Useful, but flow errors can dominate |
| Primitive count/storage | Dormant roots must replace unrelated root births; leaves must be pruned | 10–30% target-region reduction; perhaps 5–15% scene-wide | Real only with lifecycle changes |
| Semantic training time | Fewer persistent identities/features or faster convergence | Perhaps 10–30% | Multi4D already excludes TR, so large gains are unlikely |
| Reconstruction iterations | Persistent roots must reduce relearning after return | Likely flat or 5–15% worse initially | Trap unless carefully demonstrated |
| FPS | Fewer rasterized leaves, not just better identities | Probably 0–5% whole-scene | Likely flat |
| Standard Neu3D mIoU | Existing clips must contain hard enough return events | Probably 0–2 points | Likely diluted |
| mAcc | Foreground result must affect a background-dominated pixel average | Essentially flat | Do not headline |
| Whole-frame PSNR | Tiny event regions must move a global average | Frequently 0–0.05 dB | Definite trap |

Multi4D’s evaluator computes full-image pixel accuracy and silently skips missing predictions, so its mAcc can be both background-dominated and fail-open ([evaluator](https://github.com/BatFaceWayne/Multi4D/blob/c483e82cd8daa1b4fdc460ed45882fc2fe7a22b0/eval_seg.py#L17-L45)). Fix that evaluator before treating semantic numbers as evidence.

## Q3. Is segmentation/tracking the right pivot?

Yes—but Neu3D-Mask is a semantic evaluation harness, not proof of object permanence.

The official [Mask-Benchmark dataset](https://huggingface.co/datasets/yunjinli/Mask-Benchmark) supplies binary segmentation masks. TRASE learns a feature field from SAM-derived masks and evaluates dynamic segmentation ([TRASE paper](https://arxiv.org/abs/2411.19290)). It does not provide certified all-camera disappearance intervals, instance association annotations, or an identity-switch protocol.

### The exact Neu3D experiment

For each authored fixture:

1. Train the reconstruction normally.
2. Train target semantics using target masks only before the gap. Remove every post-gap target mask and overlapping pseudo-label.
3. Select the target root/cluster once using the exact pre-gap GT mask—no text query.
4. Freeze that selection across the gap.
5. Render held-out cameras with occlusion-aware rendering.
6. Fail closed if any expected frame or prediction is missing.
7. Score:
   - first-return J&F;
   - J&F at return +1/+5/+10/+30 frames;
   - recovery latency;
   - gap false-positive area;
   - target–distractor swap rate;
   - IDF1 or HOTA association accuracy where distractors exist;
   - fraction of return contribution from pre-gap roots and descendants.

Multi4D’s default selected-only render can show selected Gaussians through unselected occluders; `--occlusion_aware` is needed for modal visible-mask evaluation ([render code](https://github.com/BatFaceWayne/Multi4D/blob/c483e82cd8daa1b4fdc460ed45882fc2fe7a22b0/render_cluster_by_text.py#L1474-L1518)).

Minimum arms:

- M0: shipped Multi4D, post-gap semantic supervision censored.
- M1: persistent-root notch mechanism.
- M2: identical visibility and parameter budget, but pre/post episodes receive different root identities.
- M3: M1 plus lineage-sketch cycle loss.
- M4: M1 plus episode-specific return pose, if the oracle geometry test says location is the failure.
- External: TRASE and a strong first-mask 2D VOS baseline under identical supervision.

M2 is load-bearing. If M1 and M2 perform equally, persistence did not cause the win; capacity or visibility did.

### Failure modes

The dangerous ones are:

- post-gap pseudo-label leakage;
- choosing/re-querying the cluster after return;
- no distractor, making “re-ID” trivial;
- target encoded mainly in TR, which semantic training excludes;
- KNN semantic smoothing fusing touching objects;
- visible-mask versus amodal-mask mismatch;
- missing predictions silently omitted;
- authored removal artifacts serving as an appearance cue;
- an object hidden in one camera but visible in another being called “absence.”

### Genuine-occlusion benchmark

First run a visibility census on [BEHAVE](https://virtualhumans.mpi-inf.mpg.de/behave/). It provides calibrated synchronized RGB-D views, masks, and registered human/object geometry, allowing you to certify that an object remains in the capture volume while being depth-occluded in every camera ([CVPR paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Bhatnagar_BEHAVE_Dataset_and_Method_for_Tracking_Human_Object_Interactions_CVPR_2022_paper.pdf)).

Require:

- target visible in at least two cameras before and after;
- effectively zero depth-tested visible support in every camera during the gap;
- sustained gap, not one-frame mask noise;
- a same-category or visually confusing distractor;
- synchronized montage review.

If BEHAVE lacks enough certified events, inspect InterCap next. [Dynamic 3D Gaussians](https://dynamic3dgaussians.github.io/) and [TAPVid-3D](https://arxiv.org/abs/2407.05921) provide useful 2D/3D tracking protocols, but do not automatically certify complete multiview disappearance. [MOSE](https://mose.video/MOSEv1/) and [OVIS](https://arxiv.org/abs/2102.01558) contain real disappearances but are monocular; use them as external tracking baselines, not direct 4DGS proof.

I do not see a turn-key benchmark that simultaneously supplies calibrated multiview 4D reconstruction, certified all-camera disappearance, return, and identity association. A small physical benchmark may itself become part of the contribution.

## Q4. Fast motion and low-quality regions

Build a preregistered failure atlas rather than selecting “bad pixels” after seeing treatment results.

Define strata using input/GT quantities or an independent baseline scout:

- Motion: forward-flow magnitude; preferably residualized against camera motion.
- Occlusion boundary: frozen GT mask dilation/erosion band.
- Disocclusion: previously occluded pixels becoming visible, cross-checked by forward/backward flow consistency.
- Return: fixed post-gap windows of 1, 5, 10, and 30 frames.
- Specularity: cross-view/view-direction photometric variance, validated separately because it is easily confounded with geometry error.
- Baseline-hard: error quantiles from an independent scout run, never from the treatment seeds.

Dense flow already exists for all six N3V scenes ([project record](/D:/adags/research-wiki/query_pack.md:912)).

For each stratum \(s\), pool squared error over pixels before converting to PSNR:

\[
\mathrm{MSE}_s =
\frac{\sum_{f,p} m_s(f,p)\,\|I-\hat I\|^2}
{\sum_{f,p}m_s(f,p)},\qquad
\mathrm{PSNR}_s=-10\log_{10}\mathrm{MSE}_s.
\]

Do not average per-frame PSNR, and do not treat pixels or frames as independent replicates. The experimental unit is the run within a scene.

Report:

- stratum PSNR/SSIM/crop-LPIPS;
- boundary F;
- visibility-aware warped error;
- tracking AJ/OA or endpoint error;
- performance-versus-motion-decile curve;
- worst-decile or CVaR error;
- ordinary/interior noninferiority.

A defensible targeted claim requires:

1. a larger improvement in high-motion/disocclusion strata than in low-motion interiors;
2. direct evidence that the proposed root/pose mechanism was active there;
3. no material degradation outside the target stratum.

## Q5. Substrate choice

Use Multi4D for the paper. Use Ex4DGS for rapid falsification and independent replication.

Multi4D wins the substrate decision because:

- it reproduced 3/3, with mean peak 33.576 dB ([results](/D:/adags/research-wiki/operations/multi4d-n3v-leonardo-reproduction-2026-09-18.md:244));
- it already separates persistent identity-capable FG from disposable TR;
- it has an existing semantic pipeline and benchmark;
- its persistent FG bank is compact;
- the root-level presence mechanism can be implemented before rasterization in PyTorch, including the critical FG mask-render path ([code map](/D:/adags/research-wiki/operations/multi4d-code-map-2026-09-19.md:132)).

Ex4DGS remains valuable because its seven-line temporal gate and auxiliary render channel make mechanism iteration cheap. But its 1.06 dB mean baseline deficit prevents it from being the sole headline substrate.

Do not use “Ex4DGS mechanism, Multi4D evaluation” without porting the mechanism. That would be two unrelated experiments. The same root/notch/identity principle must exist in Multi4D.

Before Multi4D inference:

- add explicit seed plumbing;
- establish `q=0` parity;
- reproduce the semantic stage separately;
- resolve the semantic-dimension discrepancy—the current code defaults to 16 dimensions ([source](https://github.com/BatFaceWayne/Multi4D/blob/c483e82cd8daa1b4fdc460ed45882fc2fe7a22b0/train_semantic.py#L2079-L2087));
- account for the observed 0.3445 dB nondeterministic spread ([reproduction record](/D:/adags/research-wiki/operations/multi4d-n3v-leonardo-reproduction-2026-09-18.md:281)).

## Q6. Minimal two-week experiment

### Days 1–3: make the test trustworthy

- Reproduce the shipped Multi4D semantic result on one scene.
- Make evaluation fail closed.
- Add return-supervision censoring and exact pre-gap mask selection.
- Add explicit seeds.
- Implement root/descendant contribution logging.
- Run a BEHAVE/InterCap visibility census.
- Run an oracle test: manually select the correct pre-gap root cohort and ask whether its deformed geometry lands on the returning object.  
  - If yes: identity/visibility is the likely bottleneck.
  - If no: implement the episode-specific return-pose candidate first.

### Days 4–7: single-seed candidate screen

One authored scene, preferably `cut_roasted_beef`:

- S0: q=0/root-unmodified baseline.
- S1: persistent-root notch.
- S2: S1 plus lineage-sketch cycle.
- S3: S1 plus episode return-pose residual.
- S4: episode-split identity control.
- S5: S1 with return-time unrelated-root birth suppressed, as the compression probe.

Evaluate semantic return, root contribution, high-motion ROI quality, ordinary-region quality, primitive count, and runtime. Use this only to select one candidate—not to make claims.

### Days 8–14: paired confirmation

Take the winning candidate, q=0 baseline, and episode-split control.

Run:

- 2 scenes: `cut_roasted_beef` plus one challenge scene chosen in advance by highest p90 flow or worst independent-scout return quality;
- 3 paired seeds;
- 3 arms.

That is 18 reconstructions. At 2.7 GPU-h each, reconstruction costs 48.6 GPU-h. With semantic training and the screen, the complete funnel is approximately 120–145 GPU-h, under one GPU-week.

### Go/no-go rule

Continue to six-week paper-scale work only if the winner satisfies all relevant conditions:

- return J&F: at least +5 points against both q=0 and episode-split controls;
- root/descendant contribution: at least +15 percentage points;
- return-region PSNR: at least +0.5 dB, or a comparably clear tracking improvement;
- standard Neu3D-Mask mIoU: no worse than −1 point;
- ordinary-region PSNR: no worse than −0.2 dB;
- gap false positives: no increase;
- paired direction positive in at least 5 of 6 scene-seed units;
- compression claim, if pursued: at least 15% fewer dynamic primitives/bytes at matched quality.

With \(n=3\), these are screening rules, not hypothesis-test thresholds. Use the measured variance to power the remaining experiment.

If no candidate clears one headline endpoint plus the mechanism check, stop the representation project. A failure atlas or benchmark contribution may remain worthwhile, but “object permanence improves 4DGS” would not be alive.

## Q7. The opportunity not previously named

The hidden opportunity is that Multi4D has already separated identity memory from rendering detail—but does not exploit that separation explicitly.

Its semantic stage constructs the scene without transient Gaussians and freezes reconstruction geometry while learning semantics on FG/BG alone ([source](https://github.com/BatFaceWayne/Multi4D/blob/c483e82cd8daa1b4fdc460ed45882fc2fe7a22b0/train_semantic.py#L1407-L1509)). Yet TR contains the majority of dynamic render primitives. This means:

> The compact persistent foreground bank can be treated as an object-memory layer, while transient Gaussians become disposable observations or render leaves.

That reframes the project from “make Gaussian support disconnected in time” to:

> “Learn a persistent object memory that survives periods without observations and regenerates disposable render support on return.”

The second half of the opportunity is the randomized lineage-sketch loss. Round 1 used the auxiliary channel only for contribution measurement; using hashed root codes makes contribution identity trainable with no one-hot feature field and no CUDA rewrite.

Together—persistent roots plus hashed contribution supervision—this is substantially stronger than another temporal gate. It offers a single mechanism connecting re-ID, segmentation, fast return, compression, and interpretable identity attribution.