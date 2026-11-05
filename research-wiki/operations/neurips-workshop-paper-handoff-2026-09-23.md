# NeurIPS 2026 workshop paper — handoff (2026-09-23, revision 3)

**Read this first in a fresh session.** It supersedes
[revision 2](neurips-workshop-paper-handoff-2026-09-21.md), which is still
accurate for everything it records; this page adds what the 2026-09-21/23 session
established and corrects four numbers. Nothing here is a decision the principal
has made unless it says so.

**State: NOTHING HAS RUN.** No job was submitted, no substrate code was changed,
no fixture was converted. The session read the record, verified the venue, and
ran a two-round exchange with `gpt-5.6-sol` (xhigh). **Five decisions in §6 are
blocking** and were put to the principal on 2026-09-21; no answer had arrived by
the end of the session.

---

## 0. What has not changed (from revision 2 §0)

One paper, **8 pages, NeurIPS 2026 PhysWorldAI, done ~Oct 15, submitted by
Oct 29**. Nothing is reserved for the CVPR/ICCV paper. **No title yet** — it
follows the results. Start from the authored counterfactual fixtures. Baselines
run at their own published settings under one common evaluator (the principal
rejected matched-schedule retraining). The honesty conditions of revision 2 §2
stand verbatim: "authored counterfactual" wording, the frozen NOT_MET verdict
first, scene-clustered reporting, independent mask re-scoring.

## 1. Venue — verified 2026-09-21/22, additions to revision 2 §1

| item | value | source |
|---|---|---|
| style | **NeurIPS 2026 style** | CFP, fetched 2026-09-21 |
| review | **double-blind, via OpenReview** | same |
| portal | archival group exists; the **non-archival venue "will be open soon"** — still not verified as open | same |
| page limit | long papers up to 8 pages; the clause "(Reference and Appendix are not included)" sits on the *extended-abstract* line | same |
| dual submission | "permitted if authors comply with those venues' policies" | same |

**Open item:** ask the organisers whether references and appendix are excluded
from the 8 pages. Sol judged the natural reading to cover both formats; this is
not material to planning, but the answer is needed before layout.

## 2. Four numbers this session corrected

1. **The event is ~0.031% of pixel-times on these fixtures, not 0.34%.**
   5,100 px of 1,352×1,014 = 0.372% spatially, × 25/300 frames = **0.0310%**.
   The 0.34% figure in the record belongs to an earlier utility-scaling event
   union (`query_pack.md`, 2026-08-24 block). Consequence: a *perfect* gap moves
   300-frame pooled whole-frame PSNR by **≈0.0013 dB** under uniform error
   density, and **0.0023–0.0062 dB** (median 0.0034) when corrected by the
   measured P1/H2 error densities in `absfix-stage2-paired.json`. The
   often-quoted "≈0.02 dB" is the 25-frame-only spatial calculation.
   **Whole-scene PSNR/SSIM/LPIPS cannot carry this paper. The principal's
   expectation that fixing the gap raises overall metrics is arithmetically
   false and must not be designed around.**
2. **PersistGS is a CVPR 2026 workshop paper** (Generative 3D Reconstruction,
   CVPRW 2026, pp. 4687–4696), not ECCV. Verified on arXiv 2606.03479's comments
   field and the CVF record. Its bar: three synthetic rigid-ball scenes, 5 train
   + 2 held-out cameras at 512², object-region PSNR/SSIM/LPIPS during occlusion
   plus trajectory RMSE, physics-free baselines and a GT-trajectory upper bound.
   **No code or scene release was found.** Its object persists *behind an
   occluder*; our authored gap removes the object from the world — the opposite
   physical state, so PersistGS is related work and at most an occlusion control,
   never the direct absence baseline.
3. **STG on `cut_roasted_beef` is 15–21 GPU-h for ONE seed** (6 × 50-frame
   segments at 2.5–3.5 GPU-h), **30–42 GPU-h to match the two seeds run on
   flame/sear**. Revision 2's "15–21" is the one-seed figure; Sol's round-1
   "60–84" multiplied by four prefixes, which are evaluations, not trainings.
4. **Ex4DGS's static bank has no temporal opacity** (`get_opacity` =
   `sigmoid(_opacity)`, `c_gaussian_model.py:355-357`) and static→dynamic
   promotion is displacement-driven (`:1147-1163`). The fixture target (a wine
   bottle) is static, so **any presence mechanism must attach to the
   already-concatenated opacity vector covering both banks**, or it is vacuous.
   Which bank actually paints the bottle is **UNVERIFIED** until a contribution
   census runs.

## 3. The Sol exchange — what it concluded

Two rounds, `gpt-5.6-sol` xhigh, full record tracked at
[`assets/neurips-plan-sol-2026-09-21/`](../assets/neurips-plan-sol-2026-09-21/)
(`prompt_round1_full.md`, `sol_round1.md`, `prompt_round2.md`, `sol_round2.md`,
`claude_plan_v1.md` = the plan it reviewed). Both rounds passed a file-access
attestation (verbatim `interpolations.py:58`; P2 `G-U` median 2.4890 dB from the
stage-2 report). Claude re-verified the load-bearing citations: PersistGS venue,
CTRL-GS (CVPRW 2025), Splannequin (WACV 2026), OP2GS, MoRel (CVPR 2026),
Kubric-5D's repo, and every Ex4DGS/ADAGS `file:line` quoted below.

**(a) The plain notch is not enough novelty.** SharpTimeGS (flat-top lifespans)
and Flow4DGS-SLAM (GMM temporal opacity) already occupy multi-modal per-primitive
temporal opacity. Sol scored the notch-only plan **4–5/10, weak reject**.

**(b) Its proposed core method: Complementary Episode Residuals (CER).** Two
cohorts driven by one event state, on Ex4DGS:

* a persistent **memory cohort** `K` (the object's rows, both banks):
  `α_i(t) = α_i^0(t) · [1 − q·n(t)]`;
* a **gap-only residual cohort** `R`, born from the largest gap-region
  contributors: `α_j(t) = q·n(t)·σ(o_j)` — exactly zero outside the gap;
* `q = 0` reproduces unmodified Ex4DGS exactly; `n(t)` is the smooth flat-top
  absence indicator (ordered boundaries, verified interior gradient).

The rationale is this project's own measurement: on the fixture the *same* rows
paint the bottle outside the gap and the counterfactual background inside it, so
suppressing them cost 5.9 dB at render time
([absence-fixture-lane-2026-09-10](absence-fixture-lane-2026-09-10.md) §5). CER
gives the two incompatible jobs to two cohorts. It is the only candidate that
could plausibly reach STG's gap quality *and* keep the gate's return advantage.
Attachment: after `get_opacity_at_t` (`gaussian_renderer/__init__.py:62-64`),
before the rasterizer call at `:100-109`; residual rows appended through the
existing dynamic-tensor path (`c_gaussian_model.py:1191-1210`); lifecycle
(densify/prune/promotion) disabled during the short residual stage. **No CUDA
change.** Closest work Sol found and distinguished: CTRL-GS, Multi4D, SharpTimeGS,
CubifyGS, OP2GS. Cost 5–7 engineering days, ≈85–100 GPU-h. Conditional score
**7/10, accept**; today's evidence ≈5/10.

**(c) Ranked alternatives**, all below CER: the interventional gap–return
benchmark as the fallback contribution; lifecycle dormancy for the absent cohort
(collides with Splannequin, WACV 2026 — only worth it if a census shows the
target's rows are pruned or drift during the gap); the randomised αT
contribution-sketch cycle loss (stretch, high variance, 2–3× render cost);
Multi4D return loop closure (**do not schedule** — likely vacuous, since the
fixture restores the original footage after frame 89, so the return pose is
unchanged).

**(d) Datasets (round 2).** Running "all datasets" would weaken the paper.
Recommended: the three N3V fixtures (core), a small claim-grade synthetic suite,
the existing real-occlusion null (specificity control), **one** cheap Panoptic
Sports census (drop by Sep 25 if no certified event). Skip Technicolor (no raw
footage locally; its licence requires approval to modify data), ImViD, DiVa-360,
D-NeRF (single-camera construction — cannot show all-camera absence) and
MOVi/Kubric-5D (good post-deadline, days of setup now). The other three N3V
scenes are a stretch continuity test, not three more mandatory fixtures.

**(e) Event taxonomy (round 2).** One event type is not enough, and **the real
editor cannot produce the others** — verified in code: `sa4d_absence_edit_render.py`
has one `--window A B`, no transform/insertion/relight/multi-gap interface
(`:2496-2587`), and the assembler copies original frames outside that single
interval. So variety must come from the synthetic renderer. Minimum set for
Oct 15: (1) three real authored same-pose fixtures, (2) the real-occlusion null,
(3) synthetic same-pose return, (4) synthetic moved/translated return,
(5) synthetic "another object moves through the vacated region during the gap" —
the sharpest test that the residual cohort does more than switch opacity off,
(6) a two-level gap-duration ablation at equal pre/post observation supply.
Changed appearance, look-alikes, non-rigid targets and multiple gaps → limitations.
A moved return raises the priority of *testing* relocation, not of building
loop closure: if same-pose passes and moved return fails, try an oracle episode-2
SE(3) transform first.

**(f) Synthetic first — for mechanism debugging only.** Use the existing analytic
renderer (`scripts/build_synthetic_reveal_scene.py`: deterministic 20-camera,
60-frame ray trace, exact identity buffers, Blender layout EL-GS reads natively;
`scripts/eval_lrv1_event.py` already separates return / ghost / ordinary /
whole-frame). It is labelled **exploratory, not claim-grade** in its own header,
so promotion needs a frozen spec, held-out cameras, repetitions and preregistered
metrics. No Blender/Kubric/`bpy` is installed locally and none should be
installed for this deadline. An Ex4DGS-format exporter for synthetic data is a
moderate task and **must not block the real-fixture lane**.

**(g) Gates.** Sol agreed with Claude that the primary kill comparison is
**unmodified Ex4DGS vs Ex4DGS+CER** on the same data, schedule and evaluator —
not STG, which is confounded (six independent 50-frame specialists at 25k, and
its control window is already 0.4–1.2 dB above our arms; the record states no
pairing or floor applies to it). Approaching STG's gap quality is a
paper-strength stretch, not a kill condition.

## 4. The headline claim to aim for

> The method Pareto-dominates the strongest baselines on **both** authored-absence
> (gap) and first-return quality, while preserving ordinary-region and whole-scene
> quality.

Show a two-axis gap-vs-return plot; never average P1 and P2 into one score.
Whole-scene metrics appear only in a do-no-harm role. A whole-scene N3V SOTA
claim is not achievable from a mechanism acting on 0.03% of pixel-times.

## 5. Plan as it stands (rebased to 2026-09-23; one day already lost to the
pending decisions)

| dates | work | gate / kill |
|---|---|---|
| **Sep 23** | Principal's five decisions (§6). Freeze the claim set and the common evaluator conventions (PSNR pooling/clamping, one SSIM, one LPIPS) before any run. Renew the CINECA certificate. | No experiment until the claim and evaluator text are frozen. |
| Sep 23–25 | Ex4DGS and Multi4D fixture converters (reuse `colmap_0` for crb — frame 0 is byte-identical to the original scene, **verify before relying on it**; Multi4D needs `fused.ply` per scene — crb exists, flame/sear **unverified**). Golden-test the common evaluator against each native evaluator. Submit unmodified Ex4DGS + Multi4D on the three fixtures and STG seed 1 on crb. Measure the Ex4DGS event-window replicate floor **before** reading any method outcome. Extend the synthetic renderer with moved return and gap-replacement. One CPU-light Panoptic census. | **Panoptic kill Sep 25:** no certified candidate → drop it. **Converter gate:** any camera/frame/resolution mismatch blocks submission. |
| Sep 25–27 | CER as **one small review-gated Ex4DGS diff** covering both banks. Tests first: `q=0` image equality with the unmodified path, contribution-mass non-vacuity, residual exactly zero outside the gap, checkpoint round-trip. | Principal reviews the diff before anything trains. |
| Sep 27–29 | Screen: base / target-only notch / residual-only / CER / shifted-interval sham on one real fixture plus the three synthetic conditions. **Montages of the scored region on every arm before any table.** | **Method kill:** P1 must beat unmodified Ex4DGS beyond the frozen floor with P2 non-inferior on ≥2 of 3 real scenes. Moved-return failure → oracle episode-2 transform diagnostic only. |
| Sep 30–Oct 3 | Confirmation: 3 repetitions × base/CER × 3 scenes; finish Multi4D; STG seed 2 if wall-clock allows; ablations; independent SAM 2 masks + human audit of every P1/P2/S1 frame; frozen reducer re-run unchanged. | **No new datasets after Sep 30.** |
| Oct 4 | Data freeze. Every figure and table generated from frozen artefacts. | Decide: "tradeoff removed", "tradeoff reduced", or benchmark-only. |
| Oct 5–8 | Full 8-page draft; title chosen from the results. | No exploratory runs after Oct 8. |
| Oct 9–11 | Adversarial review; claim-to-artefact audit; citation and venue audit; anonymity check. | Every number traces to an artefact. |
| Oct 12 | Empirical freeze. | Result-changing bug reopens the confirmation gate. |
| Oct 13–15 | Final revision, appendix, reproducibility, submission package. | Done Oct 15. |

**Compute:** ≈90–120 GPU-h total (STG 15–42, Ex4DGS base+method 21.6, Multi4D
24.3, synthetic ≤21.6, screens 5–10). ≈1% of the ≈15,500 GPU-h remaining.
**Wall-clock and the 1-GPU queue are the constraint, not budget** — shortest safe
walltimes, measure a rate on `boost_qos_dbg` first, never rely on resume (unusable
on both substrates).

## 6. BLOCKING decisions (put to the principal 2026-09-21, unanswered)

1. **Method:** adopt CER on Ex4DGS as the core mechanism, with the plain notch
   kept as an ablation arm?
2. **Headline:** accept event-local Pareto dominance (gap **and** return) instead
   of whole-scene SOTA, given §2.1?
3. **STG seeds:** one seed on crb (15–21 GPU-h) or two to match flame/sear
   (30–42)?
4. **Control model:** does the review-gated-diff rule (revision 2 §7) also cover
   ADAGS's own EL-GS code? Never stated either way.
5. **Mask audit:** who performs the ≈129-frame independent SAM 2 audit (every
   frame of P1, P2, S1 across three scenes)?

Sol adds a sixth, lower-stakes one: whether to author one extra fixture with a
*different object* — recommended only if the existing assembler can finish it in
≤1 day after the method gate.

## 7. Traps and carry-overs

Revision 2 §5 stands in full (cohort means not best-of-n; montage before table;
a frozen rule needs a frozen precondition; check a knob is read before varying it;
noise floors are fixed-seed lower bounds — Multi4D 0.3445 dB, Ex4DGS sd 0.3564 dB
at n=6 with a 0.9128 dB range; a reviewer's upstream line numbers have been wrong).
Add from this session:

* **Repetitions are not seeds.** Neither substrate exposes `--seed`; call them
  repeated trainings in the paper.
* **Sol's round-1 Gate A ("close ≥50% of the base-to-STG P1 deficit") was
  withdrawn** in round 2 as a kill rule. Do not reinstate it.
* **Codex usage limits are real.** Round 2 died mid-answer on a quota limit and
  had to be re-run after the reset; budget for that when planning a review.

## 8. What to read, in order

1. This page, then revision 2 (`neurips-workshop-paper-handoff-2026-09-21.md`)
   for the venue, the evidence table and the §5 traps.
2. [`assets/neurips-plan-sol-2026-09-21/sol_round1.md`](../assets/neurips-plan-sol-2026-09-21/sol_round1.md)
   and `sol_round2.md` — the CER mechanism, the novelty audit, the dataset triage
   and the event taxonomy, in full.
3. The revision-2 §8 list, unchanged: the stage-2 verdict + `absfix-stage2-paired.json`
   (generate tables from the JSON), the fixture lane, the real-occlusion lane, the
   Ex4DGS multi-interval plan v0.3 and its two reviews, both code maps, both
   reproduction pages, the thesis and its kill-argument, the literature check,
   and `query_pack.md`.

## 9. Operational notes

* **Leonardo was unreachable at the end of this session** — Sol's SSH attempt was
  refused on port 22 and the certificate needs renewing (2FA, principal only).
  Nothing was verified on Leonardo this session; every Leonardo-side fact in §5
  marked *unverified* stays that way.
* **Re-opening the Sol thread:** `codex exec resume 01a0c514-f272-7a62-a215-4ab1a4e0c32b
  -m gpt-5.6-sol -c model_reasoning_effort="xhigh" -c tools.web_search=true
  --skip-git-repo-check -o <out.md> -` with the prompt on **stdin**, run from
  `D:\adags` in PowerShell. The thread carries both rounds' context.
* Nothing in this session was committed. This page and
  `assets/neurips-plan-sol-2026-09-21/` are untracked additions awaiting the
  principal's review.
