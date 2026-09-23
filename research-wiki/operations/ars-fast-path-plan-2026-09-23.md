# ARS fast path: proof against the benchmarks first (plan revision 3, 2026-09-23)

**Status: FINAL, the agreed active plan (principal, 2026-09-23).** Nothing has
trained and nothing is committed. Execution follows §8 step by step.

This page supersedes the **scope, tiers, compute and schedule** of
[revision 2](absence-return-suite-and-baseline-lanes-plan-2026-09-23.md). Revision
2 is kept as the **ideal design** and as the **spec** for everything below marked
"kept (rev 2 §x)". Its snapshot is `refine-logs/ars-review-2026-09-23/plan_snapshot_r2.md`,
sha256 `f3cce49c…`. Where the two differ, this page wins.

## 0. What the principal changed (2026-09-23)

1. **Evidence hierarchy flipped.**
   - **ARS is the primary evidence** for the movement/return taxonomy and for the
     method's causal claims.
   - **The real authored fixtures are a secondary realism check**, for same-pose
     absence and return only. Caveat: their gap ground truth is an **SA4D teacher
     render** (authored counterfactual), and their scoring masks are DEVA- and
     construction-derived ([fixture lane](absence-fixture-lane-2026-09-10.md)).
   - **Rationale:** the workshop bar. PersistGS, CVPRW 2026, was accepted with three
     synthetic scenes (handoff rev 3 §2.2). ARS ground truth is exact.
2. **Order of work:** first, proof that the method beats the benchmarks on ARS; then
   breadth; then real-data validation, including a later self-captured dataset if
   the proof holds.
3. **Decisions:** the 1352 route is accepted; STG on crb runs **both seeds**; Panoptic
   is **not** copied to `D:` for now.
4. **Method confirmed: CER on Ex4DGS**, with the plain notch as an ablation and the
   declared inputs of §4. The Multi4D port was added as the conditional item §4b.

## 1. Kept, because it protects correctness

| kept | spec |
|---|---|
| designed reference clock (A\*, B\*); phases for moving returns | rev 2 §3 |
| analytic masks: front-most identity and target-alone projection at a declared pose. **No shadow masks**, because shadows are deferred (§2) | rev 2 §5 |
| V1–V7 and V9–V11 under the applicability matrix; **V8 kept but reduced** to identity only, 20 (camera, frame) pairs per scene, written by an independent agent from the frozen spec | rev 2 §6 |
| collision and free-space check (V10) | rev 2 §4 |
| frozen spec, trajectory hashes and per-scene manifests, all **before any method run** | rev 2 §1.7 |
| montages of every scored region, viewed before any table | rev 2 §1.7 |
| common evaluator with frozen PSNR/SSIM/LPIPS conventions and golden tests | rev 2 §8.3 |
| native preprocessing as an admission gate; no oracle-init fallback | rev 2 §4 |
| Multi4D final iteration as primary; no conditional repetitions | rev 2 §8.1 |
| STG segment-aware reporting | rev 2 §8.2 |
| method contribution gates (non-vacuity) per ARS scene before any method arm is scored | rev 2 §6 |
| canonical frame store plus per-method symlink adapters; instrumented cam00-leakage assertion | rev 2 §4, §6 |
| real N3V rigs, 300 frames at 30 fps, 1352×1014, cam00 held out, 2×2 supersampling, object-frame textures | rev 2 §1, §4 |

## 2. Cut or deferred

| item | status | why it is safe to defer | saves |
|---|---|---|---|
| **cast shadows** and shadow-caster masks | deferred to the ideal | realism only. Absence without shadows is still exact; declared as a limitation | ≈ 0.5 day of renderer and mask work |
| **Z0/Z1 twin trainings** | deferred | interpretive ceilings only, not needed to beat a benchmark | 6 scenes, ≈ 7 GPU-h, V-checks |
| **X1 (carrier), CMP, K1, R1, G2, N2** | deferred | the carrier is the most expensive renderer feature; the rest are coverage extensions | ≈ 0.5–1 day, 6 scenes |
| **O1 on L2/L3** | cut (O1 kept on L1, wave 2) | specificity is already shown on real data by the occlusion null; one synthetic layout suffices for a control | 2 scenes |
| **secondary two-view novel-view audit** | deferred | results are stated cam00-scoped, as in the N3V protocol | a render-only script and GT views |
| **pose endpoint** (silhouette IoU/centroid) | deferred | the P2 and V regions already score a moved return | evaluator work |
| **flame/sear unedited twins** (6 Ex4DGS runs) | cut | the real fixtures are secondary; the crb unedited ×3 runs stay as the route golden test | ≈ 7 GPU-h |
| **Panoptic census** | **deferred entirely**, including any Leonardo CPU job | synthetic is primary, and the census prior was low | ≈ 0.5 day |
| **real-fixture SAM 2 mask audit** (handoff §6.5) | deferred until real validation | the fixtures are secondary | principal's time |
| **ordinary-dynamics variety** | reduced: one moving body and one rotating object per layout, with different parameters per layout | do-no-harm on motion still exists | renderer time |

## 3. Scope

**Wave 1, "proof": 9 scenes.**

- Conditions: **A0** (same pose), **R2** (moved elsewhere) and **G1** (another object
  crosses the empty spot).
- Each on three layouts, which use the real rigs of crb, flame and sear:
  - L1: label bottle;
  - L2: carton;
  - L3: mug.
- This is the three-synthetic-scenes bar, times three event types.

**Wave 2, "breadth": 7 scenes, on L1 only.**

- **O1:** cover, occlusion control.
- **N1:** permanent departure.
- **M1:** moving target drives out and back.
- **R3:** rotated return.
- **D10 / D30 / D90:** equal pre/post supply.

**Freeze timing.** Wave 2 is rendered, checked and frozen **by Sep 29**, so the
handoff's "no new datasets after Sep 30" holds. It trains after the proof readout.

**Renderer feature set, a new `scripts/build_absence_suite_scene.py`:**

- **Primitives:** sphere, oriented box, capped cylinder, textured plane, each with a
  per-frame SE(3) pose.
- **Textures:** procedural, in the object frame.
- **Shading:** Lambertian plus ambient, with no cast shadows; 2×2 supersampling.
- **Buffers:** front-most identity, and target-alone projection.
- **Checks built in:** a collision check, and a base-world render for V6.

**Why the N3V rig and the analytic renderer (principal's question, 2026-09-23).**

*ARS is fully synthetic.* Only the camera matrices come from the real N3V rigs;
no real pixel is used.

**The rig is fixed by the baselines, not by us.** None of the three can load a
Blender / NeRF-synthetic scene. Loaders verified on Leonardo 2026-09-23:

- Ex4DGS `sceneLoadTypeCallbacks` = Colmap, Technicolor, Neural3DVideo;
- Multi4D = dynerf, technicolor, nerfies (monocular HyperNeRF);
- STG = Colmap, Technicolor, Immersive, Colmapmv, with configs `n3d_*`, `techni_*`, `im_*`.

A synthetic scene must therefore look like an N3V capture for them to run with
their published configs and no loader code. Borrowing the fixtures' own rigs also
means ARS and the real fixtures differ **only in content**.

**The renderer is a speed choice.** Blender, `bpy` and Kubric are **not installed**
(verified locally 2026-09-23), and a Blender pipeline would need:

- the install and asset downloads;
- N3V-rig import;
- deterministic rendering;
- exact masks, via object-index passes.

That is an estimated +1.5–3 engineering days (**UNVERIFIED**), which would move
the proof readout to about Oct 1–2.

**Post-proof option, not scheduled:** re-render the same frozen event specs in
Blender, with real assets and shadows. The specs are renderer-independent (rig,
objects and trajectory JSON), so this is a re-render, not a redesign. It is also
the natural seed of a later self-captured / self-built dataset. After Sep 30 it
counts as a new dataset under handoff rev 3's rule.

**Preferred route for that re-render: the Kubric-5D generator.** Checked on the
public README, 2026-09-23 (not run):

- Google Scanned Objects on HDRI backgrounds;
- exact segmentation, depth and flow from the raw generator output;
- `bpy==3.4.0` under Python 3.10, with no Docker;
- generator code CC BY-NC 4.0; data CC BY 4.0.

It would be driven with our N3V rig, 300 frames, 1352×1014 and scripted event
trajectories, and exported through our own N3V exporter.

**Not usable as-is:**

- The published Kubric-5D scenes contain no absence/return events (objects are
  thrown in).
- The published archives ship RGB and cameras only, with no masks.
- The rig is 16 cameras on a 15 m sphere, 60 frames at 24 fps, 576×384. That is
  outside all three baselines' N3V regime.
- AnyView-DVS's `anyview/kubric_dataset.py` feeds a 2B-parameter diffusion
  transformer (clips of at most 41 frames). It is not a bridge into Ex4DGS,
  Multi4D or STG.
- Cycles cost at 1352×1014 × 6,000 frames per scene is **UNVERIFIED** and likely
  far above the analytic renderer's.

## 4. Arms, and the rule that counts as proof

**Method: CER on Ex4DGS**, confirmed by the principal on 2026-09-23 (handoff
decision 1). The plain notch is kept as an ablation.

**Declared inputs** that the baselines do not receive:

- the designed clock (A\*, B\*);
- the target's 2D masks in the **training cameras only**, never cam00.

The paper must state this asymmetry. An estimated-window arm is deferred to the
ideal. On the real fixtures, estimated windows (T1) matched oracle windows to
0.04 dB ([fixture lane](absence-fixture-lane-2026-09-10.md)).

**Every wave-1 scene runs:**

- Ex4DGS base ×3;
- CER ×3;
- Multi4D ×1, final iteration;
- STG ×1 (seed 0), segment-aware.

These are repeated trainings, not seeds.

**Proof rule, frozen at S0 before any CER output exists.** It stands unless the
principal amends it by Sep 25.

1. **Causal (confirmatory): CER vs Ex4DGS base.**
   - **Endpoints:** P1 (gap) and P2 (return) on each of A0, R2 and G1. That is 6
     endpoints.
   - **Model:** pooled over three layouts, 9 vs 9 runs; two-way fixed effects
     (arm + layout); Holm correction over the 6 endpoints.
   - **A condition passes** when its P1 improvement is Holm-significant **and** its
     P2 is non-inferior: the one-sided 95% lower bound of CER − base exceeds −0.25 dB.
   - **Causal proof:** at least 2 of the 3 conditions pass, one of them A0.
2. **Against the benchmarks (descriptive, pre-declared).**
   - On each scene, CER (the mean of 3 runs) **Pareto-dominates** a benchmark when
     it is ≥ that benchmark on **both** P1 and P2.
   - The count out of 9 is reported for each of STG and Multi4D, and also against
     the Ex4DGS base mean.
   - **Benchmark proof:** CER Pareto-dominates STG **and** Multi4D on **≥ 6 of 9**
     scenes, including at least one scene of each condition.
3. **Do-no-harm.** Whole-frame (H2) and ordinary-region PSNR, CER − base, have a
   one-sided 95% lower bound > −0.2 dB, pooled.
4. **Preconditions that block scoring.** A method arm is scored only if it passes
   the contribution gates on that scene. A failure is reported as
   **DESIGN_WITHOUT_POWER** for that scene, never dropped.

**Honest risk.** On the real fixtures, STG rendered the gap 3.4–7.5 dB better than
the earlier gate while losing the return ([stage-2 verdict](absfix-wave2-stage2-verdict-2026-09-15.md)).
The uncertain part of the proof is **P1 against STG**, and it may fail. If causal
proof passes but benchmark proof fails, the paper stays a benchmark plus a
diagnostic; that is handoff rev 3 §5's Oct 4 decision.

## 4b. Conditional wave-2 item: port CER to Multi4D's persistent bank (added 2026-09-23, principal)

**Why.** Ex4DGS was chosen for speed and engineering safety, not as the better model:

- a single Python opacity junction;
- ≈ 1.2 vs 2.7 GPU-h per run;
- an evaluation stack validated to 0.0016 dB.

The record's history of that choice is in
[multi-interval review round 2](ex4dgs-multi-interval-review-round2-2026-09-21.md)
§"substrate reversal" and [sol_round1.md:397](../assets/neurips-plan-sol-2026-09-21/sol_round1.md).

Multi4D reproduced its paper (3/3 inside the band), is quieter (0.3445 dB range at
n = 3) and newer (ECCV 2026). A port shows that the mechanism is
**substrate-independent**, and it answers the objection that we picked the weaker
substrate. Sol listed a successful Multi4D port as one of the moves from 7/10 to
8/10 (sol_round1.md:469).

**Entry gate.** All three conditions must hold. They are checked in order, and
they concern the setup, not a score:

1. **The Ex4DGS causal proof passes** at the Mon Sep 28 readout (§4 rule 1).
2. **Multi4D's persistent bank actually paints the target (census, non-vacuity).**
   On the wave-1 Multi4D base runs of A0, persistent FG + BG must carry
   **≥ 50% of the target-region contribution mass** over the pre-gap frames,
   **in every layout**.
   - If the transient bank paints the target, gating the persistent bank is
     vacuous. That outcome is reported as a finding about Multi4D's architecture,
     not as a port failure.
   - The 50% threshold is **proposed here and frozen at S0**.
3. **The port diff reaches the principal by Thu Oct 1.** If it does not, the port
   is deferred past the data freeze, to the ideal plan and the camera-ready.

**Design, reusing the Multi4D code map §5; verify first at the pinned commit `c483e82`:**

- **Gate all three render paths:** standalone `gaussian_renderer/__init__.py:198`,
  mask render `:314` and hybrid `:468`.
- **`:314` is the recorded silent bypass.** A test renders through each path and
  asserts identical gating.
- **The transient bank is not gated.** Its temporal cull is in CUDA
  (`forward.cu:281-285`).
- **The gap-only residual cohort goes in the persistent PyTorch path**, since the
  transient bank cannot host it.
- **Lifecycle:**
  - densify/prune and the FG→TR lift are disabled for the residual cohort during
    its stage, because the lift **copies rows and leaves the donor alive**
    (query pack, 2026-09-21);
  - every run is one uninterrupted job, because resume restores the FG branch only.

**Tests before any training:**

- `q = 0` gives byte-identical images to unmodified Multi4D, on all three render paths.
- The contribution non-vacuity gate passes.
- The residual is exactly zero outside the gap.
- A checkpoint round-trip reproduces the model.
- The diff is **review-gated**, the same control model as the Ex4DGS diff.

**Runs:**

- **Confirmatory:** CER-Multi4D vs Multi4D base on **A0 across three layouts**, 3 vs
  3 per layout. The base reuses the wave-1 run plus 2 new runs. The model is the
  same pooled blocked one as §4 rule 1, over P1 and P2 with Holm correction.
- **Descriptive, only if the queue allows:** CER-Multi4D ×1 on R2 and G1 per layout.
- **Checkpoint policy:** Multi4D **final iteration** is primary, for both arms.

**Cost:**

- ≈ 2–3 engineering days.
- **≈ 40 GPU-h** for the core: 6 new base runs and 9 CER runs at 2.7 GPU-h each.
- **≈ 57 GPU-h** with the descriptive R2 and G1 runs.

**Head start at zero GPU, taken regardless of the proof:**

- On Sun 27, while CER-Ex4DGS trains, verify the code-map attachment points on
  Leonardo at `c483e82`.
- Write the FG/TR census, and run it on the wave-1 Multi4D A0 models.

The census is informative on its own terms: it shows which bank paints an object
that leaves and returns.

## 5. Schedule

| date | ARS | method (CER) | real fixtures (secondary) | gate |
|---|---|---|---|---|
| **Wed 23 PM** | renderer core | read Ex4DGS; write the CER test plan | Stage 0 data check; upload flame/sear `fused.ply`; 1352 adapters; STG crb seeds 0+1 chain submitted | — |
| **Thu 24** | L1 wave 1 rendered; V-checks; montages; Ex4DGS / Multi4D / STG adapters; an independent agent writes V8; evaluator written, golden-tested on stage-2 renders | CER implemented behind a flag; tests: `q=0` image equality, contribution non-vacuity, residual exactly zero outside the gap, checkpoint round-trip | Stage 1 golden (crb unedited ×3, dbg rate check first); Multi4D golden | **G-route** |
| **Fri 25** | V7/V9 on Leonardo; **wave-1 spec frozen (S0)**; wave-1 baselines submitted (27 Ex4DGS, 9 Multi4D, 9 STG scenes); L2/L3 wave 1 rendered first if L1 slips | **CER diff to the principal (evening)** | fixtures: Ex4DGS ×3, Multi4D ×1 | **S0**; proof rule frozen |
| **Sat 26** | baselines landing; replicate spreads recorded before any CER output (G-floor) | principal reviews → "submit" → CER ×3 on the 9 scenes | results in | **G-floor** |
| **Sun 27** | wave 2 rendered and checked | CER results land; montages; contribution gates. **Multi4D port head start (zero GPU):** verify attachment points at `c483e82`; FG/TR census on the wave-1 Multi4D A0 models | — | port gate 2 (census) |
| **Mon 28** | **proof readout** (analysis-ready tables; the principal reads) | — | — | **proof / no proof** |
| **Tue 29** | wave 2 frozen | — | — | no new datasets after Sep 30 |
| **Sep 29 – Oct 3** | wave 2: base ×3 + CER ×3; STG and Multi4D ×1; notch ablation on A0 ×3 layouts | **if the port gate passes:** Multi4D port implemented and tested Sep 29–30; **principal reviews the port diff by Thu Oct 1**; CER-Multi4D runs Oct 1–3 (§4b) | CER ×3 on the fixtures (same-pose realism check) | port gate 3 (review by Oct 1) |
| **Oct 4** | data freeze (handoff rev 3 §5) | | | |

**The critical path** runs: renderer (Thu) → adapters and V7 (Fri) → CER review
(Fri night / Sat) → CER runs (Sat) → readout (Mon).

**If the renderer slips,** L1's three scenes go first and L2/L3 follow within 24 h.
The pooled confirmatory test waits for all three layouts, so the readout would move
to Tue 29.

## 6. Compute

| block | runs | GPU-h |
|---|---|---|
| wave 1 | Ex4DGS base 27 + CER 27 (1.2 each); Multi4D 9 (2.7); STG 9 scenes (≈ 6 each, measured) | 32 + 32 + 24 + 54 ≈ **142** |
| wave 2 | base + CER 42; notch ablation 9; Multi4D 7; STG 7 scenes | 50 + 11 + 19 + 42 ≈ **122** |
| real fixtures | Ex4DGS golden 3 + fixtures 9; Multi4D golden 1 + fixtures 3; STG crb 2 seeds; later CER on fixtures 9 | 4 + 11 + 3 + 8 + 12 + 11 ≈ **49** |
| conditional: CER port to Multi4D (§4b) | base 6 + CER 9 on A0 (2.7 each); optional R2/G1 ×1 per layout (6) | ≈ **40** (≈ **57** with R2/G1) |
| **total** | | **≈ 313 GPU-h ≈ 2%** of ≈ 15,500 without the port; **≈ 353–370 GPU-h ≈ 2.3–2.4%** with it |

**Queue load for wave 1:** 63 Ex4DGS/Multi4D jobs plus ≈ 81 STG jobs (9 scenes ×
[6 train + prep + render + collect]). Rules:

- use the shortest safe walltime (measured rate × 1.3);
- check the rate on `boost_qos_dbg` first;
- never rely on resume.

**Revision 2's ideal** was ≈ 225 GPU-h for **baselines only**, with the method arms
extra and the full benchmark proof around Oct 3. The fast path spends a similar
amount but puts the proof on **Sep 28**.

## 7. What remains with the principal

**Resolved 2026-09-23:**

- method = CER on Ex4DGS;
- the 1352 route;
- STG on crb, both seeds;
- no Panoptic copy;
- the Multi4D port as a conditional item.

**Still with the principal:**

1. **CER diff review: Fri 25 evening or Sat 26 morning.** Substrate diffs are
   review-gated, and this review is on the critical path to the Mon 28 readout.
2. **Proof rule** (§4, including §4b's 50% census threshold): it freezes at S0 on
   Fri 25 unless amended.
3. **Keep the CINECA certificate alive** through the weekend submissions.
4. **Only if the Multi4D port triggers:** a port-diff review by Thu Oct 1.
5. **The readout on Mon 28 is yours to read.** Substrate control model: I deliver
   analysis-ready tables and montages, not conclusions.

## 8. Execution sequence, one step at a time

Steps within a group can run in parallel:

- the Leonardo steps E1–E3;
- the local steps E4–E5;
- the method step E8, which runs alongside E4–E7.

Each step ends with a stated check and a short report. **No job is submitted
without the step's check passing, and every job ID goes into the ledger
immediately.**

| step | work | done when (check) |
|---|---|---|
| **E1** real-fixture prep (Leonardo) | Stage 0 data-equivalence check (5 frames, ffmpeg vs cv2 decode, same PIL halving); upload flame/sear `fused.ply` and hash-check; build 1352-route adapter trees for the 3 fixtures and unedited crb; verify each fixture's frame 0 is byte-identical to the unedited frame 0 | Stage 0 max/mean abs-diff reported; hashes match `bbc5988f…` / `6a013735…`; trees pass a loader check (camera count, frame count, 1352×1014) |
| **E2** STG crb, both seeds | adapt the stage-2 STG chain (prep → 6 × train → render → collect) to crb; submit with dependencies | 2 × (prep + 6 train + 6 render + collect) job IDs in the ledger; later, both `f_box_profile.json` exist and are hashed |
| **E3** Ex4DGS route golden | `boost_qos_dbg` rate check; Stage 1: crb unedited ×3 on the 1352 route; Multi4D golden ×1 | **G-route:** \|mean − 32.6731\| ≤ 0.50 dB; Multi4D inside 33.4–34.0 at its reproduction's reading rule |
| **E4** common evaluator | `scripts/eval_event_windows.py` with the frozen conventions (rev 2 §8.3) | analytic unit tests pass; reproduces `event_region_frame_profile.py` P1/P2 on stage-2 renders to the measured tolerance |
| **E5** ARS renderer and wave-1 specs | `scripts/build_absence_suite_scene.py` (§3 feature set); frozen JSON specs for A0, R2, G1 × L1–L3; render L1, then L2/L3 | V1–V6, V10 and V11 pass under the applicability matrix; montages viewed; manifests written |
| **E6** ARS adapters and preprocessing | Ex4DGS/Multi4D/STG adapter trees; native `colmap_0`, dense `fused.ply`, STG prep; **V8** written by an independent subagent from the spec text only; **V7/V9** on Leonardo | V7 reprojection ≤ 0.5 px per method; V8 ≥ 99.9% identity agreement; V9 cam00 never reaches the optimizer; native preprocessing succeeds on all 9 scenes (admission gate) |
| **E7** freeze and baseline submission | **S0:** spec hash, manifests, proof rule and census threshold frozen; submit wave-1 baselines (27 Ex4DGS, 9 Multi4D, 9 STG scenes); after G-route, the real-fixture baselines (Ex4DGS ×3 × 3, Multi4D ×1 × 3) | every job ID in the ledger; the frozen hashes recorded in this page's append-only log |
| **E8** CER implementation (parallel to E4–E7) | CER behind a flag in Ex4DGS; tests: `q=0` byte-identical images, contribution non-vacuity, residual exactly zero outside the gap, checkpoint round-trip | all tests pass; **diff delivered to the principal** (Fri 25 evening) |
| **E9** G-floor, then CER runs | base replicate spreads per scene recorded **before any CER output exists**; A0 power recomputed; after the principal's "submit", CER ×3 on the 9 scenes | spreads logged with hashes; CER job IDs in the ledger |
| **E10** proof readout (Mon 28) | montages of every scored region and arm; contribution gates; the proof-rule tables from the frozen evaluator | tables and montages delivered; the principal reads |
| **E11** wave 2 and extensions (Sep 29 – Oct 3) | wave-2 scenes frozen by Sep 29; base/CER ×3, STG/Multi4D ×1, notch ablation; Multi4D port if §4b's gate passes; CER ×3 on the real fixtures | as E5–E10 per item |
| **E12** data freeze (Oct 4) | every figure and table regenerated from frozen artefacts | handoff rev 3 §5 |
