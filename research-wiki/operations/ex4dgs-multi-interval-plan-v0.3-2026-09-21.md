# Object permanence in 4D Gaussian splatting — plan v0.3 (2026-09-21)

**Status: DRAFT. Nothing implemented.** Supersedes
[`v0.2`](ex4dgs-multi-interval-plan-v0.2-2026-09-21.md) (and v0.1). Earlier
versions are left standing and unedited.

Round-2 constructive review:
[`ex4dgs-multi-interval-review-round2-2026-09-21`](ex4dgs-multi-interval-review-round2-2026-09-21.md).

**The reframe, in one sentence.** The project stops trying to make a Gaussian's
temporal support disconnected, and instead tries to

> **learn a persistent object memory that survives periods without observations
> and regenerates disposable render support on return.**

---

## 0. Three corrections carried in

**(a) My substrate recommendation was wrong, and the error was mine.** I
rejected Multi4D on the grounds that its temporal gate is in CUDA and
hard-culls at `marginal_t > 0.05` with zero gradient outside. That is true —
**of the transient bank only**. Multi4D's *persistent foreground* opacity is
ordinary PyTorch, and my own code map already said so: §5 lists
`gaussian_renderer/__init__.py:468/:198/:314` as attachment points and states
"a new parameter gets gradient with **no CUDA change**". I over-generalised a
property of TR to the whole substrate and recommended against the one that
reproduces. **Multi4D is now the headline substrate.**

**(b) The goal is object permanence, not multi-interval presence.** Disconnected
support was a means; it is not the contribution.

**(c) The headline metric is not PSNR.** The event union is ~0.34% of
pixel-times, so whole-frame PSNR is a *trap*, not a weak endpoint.

---

## 1. The formulation

**Persistent roots, disposable leaves.**

* A compact **root** carries identity: `root_id`, semantic feature, canonical
  appearance, long-term motion state, and a presence/lifespan gate.
* **Leaves** — densified children and transient primitives — carry episode-local
  detail, hold a `parent_root_id`, and are **free to split, die and respawn**.

This directly answers the round-1 kill argument. Identity no longer requires a
tensor row to survive; it requires a *root* to survive while its rendering
support is explicitly disposable. A copied id can no longer fake continuity,
because the claim is about root contribution mass and downstream semantic
behaviour, not row survival.

**Why Multi4D fits unusually well.** It has already separated identity memory
from rendering detail and does not exploit it: FG is 20–25k persistent
primitives against 198–252k transient, and the semantic stage distils features
onto the **persistent subset only**, excluding TR. That separation is the
representation, sitting unused.

---

## 2. The one new mechanism worth its weight: hashed lineage sketches

Round 2's genuinely novel technical piece, and the analogue to Multi4D's reuse
of SH channels that this project was looking for.

Give each root a reproducible Rademacher code `h_r ∈ {-1,+1}^3`. Render the
code-weighted contribution through the **three auxiliary channels the renderer
already carries** for `[acc, L1, SSIM]`, and penalise the discrepancy between
pre-gap and post-gap contribution distributions. Because

```
E_h[ (1/3) || sum_r h_r (p_pre_r - p_post_r) ||^2 ]  =  sum_r (p_pre_r - p_post_r)^2
```

this estimates a high-dimensional lineage-distribution loss **with no one-hot
identity buffer and no CUDA change**.

**Verified numerically before adoption:** the estimator is unbiased (converges
to the true value as hash draws increase). **Caveat to state, not gloss:** with
3 channels the per-draw relative error is ~33%, so it is a *high-variance*
unbiased estimator — acceptable as an SGD gradient with hashes resampled each
step, but it must not be read as a low-noise measurement.

Matched negative control: **permute the codes on the return side**.

One-hot would need a channel per root (~64k). The sketch needs three.

---

## 3. Metrics — what can actually win, and what is a trap

Preregistration targets, not results.

| axis | plausible effect | verdict |
|---|---|---|
| **return segmentation / re-ID** (J&F, IDF1/HOTA) | **+5–20 points** | **headline** |
| **persistent-root contribution retention** | **+15–30 pp** | **mechanism endpoint** |
| high-motion / disocclusion strata PSNR | +0.5–1.5 dB | promising secondary |
| temporal consistency (warped error) | 5–20% | useful; flow error can dominate |
| primitive count / storage | 10–30% in-region | real **only** with lifecycle change |
| semantic training time | 10–30% | Multi4D already excludes TR — unlikely |
| reconstruction iterations | flat or worse | **trap** |
| FPS | 0–5% | **likely flat** |
| standard Neu3D mIoU | 0–2 points | **diluted** |
| mAcc | ~flat | **do not headline** |
| whole-frame PSNR | 0–0.05 dB | **definite trap** |

**Instrument defect to fix first:** Multi4D's `eval_seg.py` computes full-image
pixel accuracy and **silently skips missing predictions** — background-dominated
and fail-open. It must be made fail-closed before any semantic number counts as
evidence. *(Flagged by the reviewer from the upstream source; not re-verified
locally — see §7.)*

---

## 4. The evaluation that carries the claim

**Return-censored 4D segmentation.** For each authored fixture:

1. Train reconstruction normally.
2. Train target semantics on **pre-gap masks only**; remove every post-gap
   target mask and overlapping pseudo-label.
3. Select the target root/cluster **once**, from the exact pre-gap GT mask — no
   text query, no re-querying after return.
4. **Freeze** that selection across the gap.
5. Render held-out cameras **with `--occlusion_aware`** (the default
   selected-only render can show selected Gaussians through unselected
   occluders).
6. **Fail closed** on any missing frame or prediction.
7. Score: first-return J&F; J&F at return +1/+5/+10/+30; recovery latency; gap
   false-positive area; target–distractor swap rate; IDF1/HOTA where
   distractors exist; fraction of return contribution from pre-gap roots and
   their descendants.

**Arms.** M0 shipped Multi4D (censored supervision) · M1 persistent-root gate ·
**M2 identical visibility and parameter budget but different root identities
pre/post** · M3 = M1 + sketch cycle loss · M4 = M1 + episode return-pose
residual.

**M2 is load-bearing.** If M1 ≈ M2, persistence did not cause the win —
capacity or visibility did.

**Failure modes to guard:** post-gap pseudo-label leakage; re-querying the
cluster after return; no distractor (making re-ID trivial); target encoded
mainly in TR, which semantic training excludes; KNN smoothing fusing touching
objects; visible-vs-amodal mask mismatch; authored-removal artefacts acting as
an appearance cue; an object hidden in one camera but visible in another being
called "absence".

---

## 5. Fast motion and bad regions — a preregistered failure atlas

Strata defined from **inputs, GT, or an independent scout run** — never from
treatment results: flow magnitude (residualised against camera motion);
occlusion-boundary band; disocclusion (forward/backward flow consistent);
return windows 1/5/10/30; specularity (cross-view photometric variance);
baseline-hard quantiles from the scout.

Pool **squared error over pixels before** converting to PSNR:
`MSE_s = Σ m_s·||I−Î||² / Σ m_s`. Do **not** average per-frame PSNR, and do not
treat pixels or frames as replicates — the unit is the run within a scene.

A targeted claim needs all three: larger gain in hard strata than in low-motion
interiors; direct evidence the mechanism was active there; no material
degradation elsewhere.

SEA-RAFT dense flow already exists for all six N3V scenes.

---

## 6. Substrate split

**Multi4D for the paper** — reproduces 3/3 (mean peak 33.576), separates
persistent FG from disposable TR, ships the semantic pipeline and benchmark,
compact FG bank, and the root gate attaches in PyTorch including the FG
mask-render path.

**Ex4DGS for cheap falsification** — seven-line gate, auxiliary channel, 1.2
GPU-h per run. Its 1.06 dB mean deficit disqualifies it as the sole headline.

**Do not** pair "Ex4DGS mechanism, Multi4D evaluation" without porting the
mechanism; that is two unrelated experiments. Before Multi4D work: add seed
plumbing, establish `q=0` parity, reproduce the semantic stage separately,
resolve the 16-dimension semantic default, and respect the measured 0.3445 dB
nondeterministic spread.

---

## 7. Benchmark honesty, and a possible contribution

**No turn-key benchmark supplies all of: calibrated multiview 4D
reconstruction, certified all-camera disappearance, return, and identity
association.** Neu3D-Mask is a semantic harness with binary masks — not proof
of permanence.

Run a **visibility census on BEHAVE** (calibrated synchronised RGB-D, masks,
registered human/object geometry) to certify events where an object stays in
the capture volume while depth-occluded in *every* camera. Require: visible in
≥2 cameras before and after; ~zero depth-tested visible support in every camera
during the gap; sustained gap; a confusing distractor; synchronised montage
review. InterCap next if BEHAVE is thin. MOSE/OVIS are monocular — external
baselines only.

**A small certified physical benchmark may itself be part of the
contribution.**

---

## 8. Two-week screen, then decide

**Days 1–3 — make the test trustworthy.** Reproduce Multi4D's semantic result
on one scene; make evaluation fail-closed; add return-censoring and exact
pre-gap selection; add seeds; add root/descendant contribution logging; run the
BEHAVE census; **run the oracle test** — manually select the correct pre-gap
root cohort and ask whether its deformed geometry lands on the returning
object. *Yes* → identity/visibility is the bottleneck. *No* → build the
episode return-pose residual first.

**Days 4–7 — single-seed screen**, `cut_roasted_beef`: S0 baseline · S1 root
notch · S2 +sketch · S3 +return pose · S4 episode-split control · S5
birth-suppression compression probe. Selection only; no claims.

**Days 8–14 — paired confirmation.** Winner + baseline + episode-split control,
2 scenes (the second chosen in advance by highest p90 flow), 3 paired seeds =
18 reconstructions ≈ 48.6 GPU-h; whole funnel ≈ 120–145 GPU-h, under a
GPU-week.

**Go/no-go — all must hold:** return J&F ≥ +5 against *both* q=0 and
episode-split; root contribution ≥ +15 pp; return-region PSNR ≥ +0.5 dB or a
comparable tracking gain; Neu3D-Mask mIoU ≥ −1; ordinary-region PSNR ≥ −0.2 dB;
no increase in gap false positives; paired direction positive in ≥5 of 6
scene-seed units. At n=3 these are **screening rules, not hypothesis tests**.

**If no candidate clears one headline endpoint plus the mechanism check, stop
the representation project.** A failure atlas or benchmark may still be worth
publishing; "object permanence improves 4DGS" would not be alive.

---

## 9. What is still unverified

* The Multi4D source claims in §3 and §6 (semantic stage excluding TR at
  `train_semantic.py:1407-1509`, `eval_seg.py` fail-open, the 16-dim semantic
  default) come from the reviewer reading the upstream repository. **I could
  not re-verify them locally: the Leonardo certificate expired during this
  round.** They must be checked against `$W/repo/multi4d` before anything is
  built on them. The README corroborates the persistent-subset distillation,
  which is the load-bearing one.
* Effect sizes in §3 are preregistration targets with no evidence behind them.
* BEHAVE has not been inspected; whether it contains enough certified
  all-camera disappearance events is unknown.
