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

## 9. Execution log (append-only)

### 2026-09-23 evening: E1–E3

Leonardo state for this plan lives in `$W/agent-control/ars/`:

- `ledger.txt`, the ledger for every job below;
- `checks/`, the check reports;
- `sbatch/`, `configs/`, and the scripts, whose sha256 are in the session record.

`$W = /leonardo_work/EUHPC_D36_068/sri/proj_adags`.

**E1 Stage 0 (CPU, crb, 5 frames: cam00 f0/75/299, cam07 f60, cam15 f150).**
Report `checks/stage0_crb.json`, sha256 `24b40b4c…`.

- **Decoding is identical.** ffmpeg and cv2 decodes are bit-identical (max abs diff 0).
- **A premise in rev 2 §8.1 is wrong.** It says both routes halve with "the same PIL
  call and default filter". Ex4DGS's `PILtoTorch` passes `resample=2` (**bilinear**,
  `utils/general_utils.py:23`), while the fixture builder's `img.resize()` uses PIL's
  default (bicubic). The halving filter is the whole route difference:

  | route difference | max abs diff | mean abs diff | PSNR between routes |
  |---|---|---|---|
  | Ex4DGS 1352 vs native | 33/255 | 0.38/255 | ≥ 47.8 dB |
  | Multi4D 1352 vs native (its extraction uses LANCZOS) | 18/255 | 0.20/255 | ≥ 53.2 dB |

- **Same-size resizes are exact copies** (Pillow 12.3). On the 1352 route, both
  methods train and evaluate on the fixtures' own pixels.

**E1 `fused.ply`: no upload needed.** The premise that flame/sear exist only
locally is wrong: `data_staging/multi4d/_fused_ply/{flame,sear}_steak.ply` were
uploaded 2026-09-18. Their sha256 `bbc5988f…` / `6a013735…` equal the local
`D:\adags\dynerf_fused` copies (all six scenes match).

**E1 trees.** `data_staging/{ex4dgs,multi4d}/<name>_1352`:

- built for the unedited crb and the three fixtures;
- symlinks into the 1352 frame stores;
- `colmap_0` intrinsics halved from the native `colmap_0`;
- flame/sear `colmap_0` built by job 58478098: cv2 native frame 0, then Ex4DGS
  `pre_n3d_colmap.py` unmodified.

Loader checks, through each method's own reader, pass on all 4 trees × 2 methods:

- Ex4DGS: 19/20 train cameras + cam00, 300 frames each, raster 1352×1014; on crb,
  FoV, R and T equal to the native tree exactly;
- Multi4D: focal = `poses_bounds` focal / 2, no extraction triggered.

Fixture frame 0 is byte-identical to the unedited frame 0 in every camera:
crb 20/20, flame 21/21, sear 21/21.

**E2 submitted.** 32 jobs, ids 58478205–58478249: 6 prep, 12 train, 12 render,
2 collect. The chain is stage 2's, with walltime set to the measured max × 1.3.

**E3.**

- **dbg rate check (58478131).** 12.76 it/s on the 1352 route vs 12.47 on the
  native route over the same span. It TIMED OUT at the cap, as expected.
- **Ex4DGS golden ×3:** 58479271–58479273, walltime 1:40, with `--save_img`.
- **Multi4D golden ×1:** 58478135, walltime 3:40.
- **G-route: pending.**

**E2 result (22:05).** All 32 jobs COMPLETED; training took at most 48:22.

| seed | renders | `f_box_profile.json` sha256 |
|---|---|---|
| 0 | 300/300 | `e164580c…` |
| 1 | 300/300 | `614f1893…` |

**E3 G-route result (22:05; `checks/g_route.json`, sha256 `adcab961…`).**

**Ex4DGS: PASS.**

- Runs: 32.7737 / 32.4163 / 33.0964.
- Mean 32.7621 (sd 0.3402, range 0.6801) against the native n = 6 mean 32.6731.
- Δ = +0.089 dB, inside the 0.50 dB tolerance.
- Exit 0, commit `7fac649`, tracked diff 0.

**Multi4D: FAIL.**

- Peak 32.1748 at iteration 16,500 (`best/` is the same).
- Final 31.3849 at 20,000.
- The band is 33.4–34.0; the reproduction peaks were 33.4226 / 33.7672 / 33.5383.
- So the peak is 1.25 dB below the lowest reproduction run, while that run-to-run
  range is 0.3445 dB (n = 3, fixed seed).
- Exit 0, commit `c483e82`, clean.

Diagnostics, read-only:

- The tree differs from the native staging only in its frames (`fused.ply` and
  `poses_bounds.npy` are the same files).
- The evaluation curve matches the reproduction runs at iterations 500 and 1000,
  then sits 1.2–1.6 dB below all three from 13,500 on.
- The background bank is 141k vs 150–153k; the transient bank 188k vs 198–252k.

Whether this is the halving filter (Stage 0: ≥ 53.2 dB between routes) or run
variance cannot be told from n = 1. **Stopped for the principal.**

### 2026-09-23/24 night: Multi4D disambiguation (principal-approved) and E4

**Multi4D route disambiguation.** Chosen by the principal after the G-route FAIL.

- 58486985: a second 1352-route golden.
- 58486986: a native-staging rerun (rep 4).
- **Hybrid fixture trees built, CPU, all pass the Multi4D loader check.**
  `data_staging/multi4d/<fixture>_hybrid` holds Multi4D's own cv2 + LANCZOS
  frames, with the fixture's pixels pasted inside each edited file.
  - crb: 600 composites; flame and sear: 630 each.
  - Edit masks cover 6.7k–14.5k px per frame.
  - Outside the masks, the native and fixture frames differ by at most 16–18/255.
  - The first crb build (58487051) failed closed before writing anything: the
    fixture manifest lists only its margin-window files. Rebuilt as 58488724, with
    every unlisted file now verified sha-identical to the unedited frame.
- New unedited native stagings (58487049/50): `data_staging/multi4d/{flame,sear}_steak`.

**E4, the common evaluator (uncommitted).**

- **Files:**
  - `scripts/eval_event_windows.py`, the evaluator, with rev 2 §8.3's conventions;
  - `scripts/make_window_package_absfix.py`, the fixture window packages, windows
    from spec v2 plus T;
  - `tests/test_eval_event_windows.py`.
- **Where it ran:** on Leonardo from `agent-control/ars/eval/tree`, which is
  `git archive 3a104ab` plus these three files.
- **Unit tests:** 12 pass locally with LPIPS skipped (no torch); all 13 pass on
  Leonardo.
- **Golden 1 (58488677): the stage-2 table of record, from the saved renders.**
  - 84 of 104 cells re-scored. The other 20, the wave-1 crb cells, have no
    profile at their recorded run dir, since `runs/realdata_gate` was archived.
  - All six endpoints P1/P2/S1/H1/H2/C1 match `absfix-stage2-paired.json`
    within max |Δ| **6.8e-7 dB**.
  - The 6 STG runs match the reducer's pooling within 4.9e-7 dB.
  - The package GT is byte-identical to each scene's cell GT.
  - Output: `out/golden_record.json`, sha256 `79c54c4a…`.
- **Golden 2 (58488678): whole frame against each native evaluator, on the same
  PNGs.** Output: `out/golden_native.json`, sha256 `c7fe313b…`.

  | method | PSNR, mine − native | SSIM, mine − native | LPIPS, mine − native |
  |---|---|---|---|
  | Ex4DGS | −0.010 mean, max \|Δ\| 0.0117 dB | −0.00095 | +0.045 |
  | Multi4D | −0.009, max 0.0117 dB | +0.019 | +0.043 |
  | STG | −0.013, max 0.0174 dB | +0.018 | +0.039 |

  - **The PSNR offset is 8-bit quantisation:** the natives score float renders.
  - **LPIPS: all three natives are half-contrast.** Ex4DGS and STG use the vendored
    `lpipsPyTorch`, whose `z_score` applies LPIPS's own [-1,1] shift and scale to
    [0,1] inputs. Multi4D calls `lpips` on [0,1] without `normalize`.
  - **SSIM** was probed on a dbg GPU (58492995):
    - the evaluator equals the GraphDECO formula in fp32 on CPU and on GPU
      without TF32, to 1e-5, under both torch 2.5.1 and 2.1.2;
    - TF32 moves SSIM by +0.004 to +0.005 under cudnn 9 and not at all under cudnn 8;
    - STG's and Multi4D's natives sit 0.019 **below** the exact value on their own
      saved PNGs, cause **not identified**.
- **Frozen tolerances:**
  - reproduction of the frozen reference chain: |Δ| ≤ 1e-5 dB (measured 6.8e-7);
  - whole-frame PSNR against a native evaluator on the same PNGs: per-frame
    |Δ| ≤ 0.025 dB (measured 0.0174).
  - SSIM and LPIPS are not gated against natives. Cross-method SSIM and LPIPS
    come only from this evaluator.
- **E4 check: PASS.** The analytic unit tests pass, and the evaluator reproduces
  `event_region_frame_profile.py` + reducer P1/P2 (and S1/H1/H2/C1) on the
  stage-2 renders within the measured tolerance.

**Multi4D disambiguation, first result (06:48).**

- **1352 golden rep2 (58486985):**
  - peak **33.4959** at iteration 12,500, **inside** the 33.4–34.0 band;
  - final **30.8244** at 20,000;
  - `best/` 32.7039.
- **The two 1352 runs disagree on the peak reading.** The peaks are 32.17 and 33.50,
  1.32 dB apart, against a 0.3445 dB range for the native reproduction at n = 3.
- **They agree on the final-iteration reading, which the plan makes primary.** Both
  finals, 31.38 and 30.82, sit 1.4–2.7 dB below the reproduction's finals of
  32.785 / 33.540 / 32.858.
- **Native rerun 58486986:** running since 05:03; not yet read, because the CINECA
  certificate expired at 06:53.

### 2026-09-24: E5 renderer built; two bottle-layout checkpoints; E6 COLMAP admission on the bottle layout

**Renderer (uncommitted):** `scripts/build_absence_suite_scene.py` (specs, render, gt, check,
montage, manifest), `tests/test_build_absence_suite_scene.py` (10 tests pass, including a
byte-identity test of the optimised render against brute force), and the nine wave-1 specs in
`configs/ars/wave1/`. `build_synthetic_reveal_scene.py` is untouched. Outputs are outside the repo,
in `D:\adags-ars\checkpoint*`.

**Checkpoint 1 → principal's changes.** The first look had a thin cook and a busy high-frequency
wall. The principal asked for:
- a cook with human proportions (torso ~4 units wide, two-joint arms working over the back of the
  table);
- a calm, large-scale wall (no fine speckle) with sparse strong features: a window with a blind, a
  poster, a shelf;
- COLMAP admission run together with every render (E5 and E6 hand in hand).

They accepted:
- resting contacts in the collision check (V10);
- the pass-through occupant (G1) being present all sequence;
- the region definitions, subject to revision.

Frame PNGs dropped from 1.6 MB to 0.5 MB.

**Checkpoint 2, bottle layout (L1), all three events (same-pose A0, moved R2, pass-through G1):**
visibility, absence, move, pass-through coverage, collision/contact, pixel denominators, no-leak
(on frames 0 and 50–100 in all 20 cameras) and fast-vs-brute all pass. Collision/contact passes on
all nine specs after two fixes it caught in the carton/mug layouts: a shelf box poking 0.02 into the
wall, a 0.044 gap, and the mug-layout cook's head 0.07 from the window sill.

**Nearest visible surface ≥ 7.7 units in every camera of every layout.** Ex4DGS culls Gaussians
nearer than `near = 4.0` (CUDA `min_depth`; code map). Real N3V's nearest surfaces are 5.6–7.8. A
NEAR check (floor 5.0) is now in the specs.

**COLMAP admission (E6), jobs 58543652 / 58545179 / 58545180, each tool's own preprocessing:**

| scene | Ex4DGS `colmap_0` | Multi4D `fused.ply` | STG frames 50–99 |
|---|---|---|---|
| bottle, same-pose | 9,169 pts, 20/20 cams, 0.32 px | 198,470 pts | 50/50 triangulated |
| bottle, moved | 9,178, 20/20, 0.32 px | 198,549 | 50/50 |
| bottle, pass-through | 9,698, 20/20, 0.31 px | 200,376 | 50/50 |

- Real crb references: 8,206 sparse and 386,905 dense.
- Against the exact geometry, 93–94% of sparse and 95% of dense points lie within 0.10 units of a
  true surface (dense median 0.008).
- The bottle carries 310 sparse and ~2,600 dense points.
- STG finds points on the bottle exactly in the frames where it is present (~265 vs 0).
- The ~6% off-surface points are ordinary mismatches: reprojection error 1.1 px vs 0.14 px
  overall, and shorter tracks.
- Multi4D's cloud follows the 4DGaussians `colmap.sh llff` commands and flags, with poses from
  Ex4DGS's converter. 4DGaussians' `llff2colmap.py` is not on Leonardo.

**Side questions (Blender; must ARS look like N3V?).** Memo and an independent gpt-5.6-sol xhigh
review are in `research-wiki/assets/ars-side-questions-2026-09-24/`. The review's verdict was
MATERIAL DEFECT. Its catches, all verified:
- the aggregate `all_pass` failed open on missing checks (fixed: fail-closed);
- Ex4DGS's effective near cull is 4.0, not the 0.01 projection near (measured safe; checked now);
- STG's published protocol is the first 50 frames, so six tiled segments are an adaptation;
- the physical N3V rig is a control choice, not a loader requirement;
- the repeated trainings of the proof rule are pseudoreplicates of three scenes.

It also suggested a textured "revealed backplane". Measured: one third of the bottle's gap region
in cam00 is calm wall with pixel std 1–2.

### 2026-09-24 evening: principal's decisions after checkpoint 2 (recorded before S0)

- **Go:** render all 9 wave-1 scenes at full resolution. Each scene gets three COLMAP admission
  jobs, one per tool (Ex4DGS `colmap_0`, Multi4D `fused.ply`, STG per-frame on all 300 frames),
  scored against admission criteria A1–A4 as declared in `analyse_colmap.py` (checkpoint 2).
  Stop and ask only if a check fails.
- **Textured band behind the table: not now.** Revisit only if results look distorted by the
  calm wall behind the targets.
- **Shadows stay deferred.**

**Proof-rule amendment (§4), before the S0 freeze.** §4 stands as written, with two additions:

1. **Scope of the claim.** The causal test (CER vs Ex4DGS base) treats repeated trainings as
   measuring training noise on these specific scenes. Every claim is worded as holding "on these
   nine synthetic scenes", never for scenes in general.
2. **Consistency across layouts.** An event type (same-pose, moved, pass-through) passes only if,
   in addition to the §4 criteria, CER's mean gap improvement over Ex4DGS base is positive in each
   of the three layouts (bottle, carton, mug) separately. Per-layout differences are reported for
   every endpoint.

**Combining future scenes.** Before any runs start on added layouts, the real fixtures or our own
capture, the rule for combining them is written into this log. The real fixtures are always
reported separately and never pooled with the synthetic scenes.

**Wave 2 addition: a dense-initialisation variant**, labelled as a variant and not part of the proof.
- **Scene:** ARS-L1-A0 (bottle, same-pose return).
- **Runs:** Ex4DGS base ×3 and CER ×3, each started from the scene's Multi4D `fused.ply` (dense
  COLMAP of frame 0) instead of the sparse `colmap_0` points. Compute ≈ 6 × 1.2 GPU-h.
- **Precondition (setup, not score):** the logged initial Gaussian count of every run equals the
  point count of the dense cloud it was given. The loader path that reads the initial cloud is
  verified in Ex4DGS's code before submission. If the precondition fails, the variant is reported
  as not exercised.
- **Robustness criterion, declared before any result exists.** Let Δ_dense = mean(CER) −
  mean(base) under dense init (3 vs 3), and Δ_sparse the same difference from the wave-1 runs on
  the same scene. The result is called robust to initialisation iff all three hold:
  - Δ_dense on the gap window is > 0;
  - Δ_dense on the return window is > −0.25 dB (the §4 non-inferiority margin);
  - Δ_dense and Δ_sparse on the gap window have the same sign.

  The report is descriptive only: all 12 per-run values and both Δs for every endpoint, with no
  significance test at 3 vs 3.

**Independent identity check (V8), 2026-09-24 22:35.**
- **Setup:** an independent agent wrote `D:\adags-ars\verifier\verify_identity.py` (sha256
  `415a1b3f…`) from `configs/ars/ARS_SPEC_FORMAT.md` and the spec/rig/identity files only. It
  confirmed it never opened the renderer or its tests.
- **Checkpoint-2 bottle scenes (20 random camera-frame pairs each, seeds 1001–1003):** 60/60 full
  frames agree on every pixel, edges included. Its planted-defect controls show the declared
  criterion (≥ 99.9% agreement away from edges) would miss a half-pixel centre shift and a reversed
  joint order; overall agreement catches both.
- **Wave 1, declared now, before any wave-1 run:** a scene passes the identity check iff
  away-from-edge agreement ≥ 0.999 on every pair and pooled, AND overall agreement ≥ 0.9999 pooled.

### 2026-09-25 03:15: wave 1 rendered in full; every scene check passes; COLMAP admission misses one declared criterion on 3 of 2,700 frames

**Rendered.** All 9 wave-1 scenes at full resolution: 20 or 21 cameras × 300 frames, ~3.2 GB each.
Stored in `D:\adags-ars\wave1\<scene>`, each with a MANIFEST (sha256 of every file). Uploaded to
`/leonardo_scratch/large/userexternal/siyengar/ars/wave1/` and verified file by file against the
MANIFEST.

**Scene checks: all pass on all 9, with no applicable check missing.**

| check | result |
|---|---|
| visibility | held-out camera 4,985–5,839 px; target in 19/19 or 20/20 training cameras |
| absence | 0 target pixels anywhere over frames 60–89 |
| move | IoU 0 |
| pass-through coverage | ≥ 4 frames at ≥ 50% coverage; 77–100% of the former volume crossed |
| no-leak | 26–98 M changed pixels per scene, 0 outside the event objects' outlines (+1 px) |
| nearest surface | 7.72–8.06 units, above Ex4DGS's 4.0 cull and above STG's per-rig near plane of 5.35–7.42 |
| collision / contact | pass |
| pixel denominators | pass |
| fast-vs-brute | 8/8 sampled frames byte-identical |

**Independent identity verifier (V8), criterion strengthened before the runs:** 9/9 scenes pass,
100.0000% overall pixel agreement on 20 random camera-frame pairs each, and every identity file it
read matched the MANIFEST.

**COLMAP admission (each tool's own preprocessing):**
- **Ex4DGS `colmap_0`:** all cameras registered; 8,365–11,062 points; 0.30–0.37 px reprojection
  error.
- **Multi4D `fused.ply`:** 194,626–204,473 points.
- **STG:** all 300 frames triangulated in every scene.
- **Two transient COLMAP aborts** ("Camera model does not exist" in two-view verification) were
  retried: L1-G1, and L2-R2 with resume. Later jobs retry inside the job.

**Loader checks (E1 scripts) and V7:**
- Ex4DGS and Multi4D loaders pass on all 9: cam00 held out, 300 frames, 1352×1014, tensors equal to
  the files.
- The 1,000-point reprojection through each method's own camera representation passes for Ex4DGS,
  Multi4D and STG on all 9 (max 1.3e-10 px).

**V9, static half:** passes on all 9. No adapter-tree link resolves into `gt/`. The instrumented
half (log what the optimizer actually consumes) has NOT been run.

**The one failure: admission criterion A3.** A3 is declared in `analyse_colmap.py`: ≥ 90% of a
product's points within 0.10 units of a true surface. It is missed on 3 of 2,700 STG frames, all
gap frames of the mug layout:
- L3-A0 frame 89: 89.75%;
- L3-R2 frames 60 and 89: 89.60% at lowest.

Context:
- Every other STG frame is ≥ 90.16%. Per-scene medians are 91.1–92.0% (mug), 92.2–93.1% (bottle)
  and 93.5–93.7% (carton).
- The off-surface points are ordinary mismatches: reprojection error ~1.2 px against ~0.15 px for
  on-surface points.
- All cameras are registered, and there are correctly no points on the absent target.
- The misses fall where the mug's ~300 accurate points are absent. The pass-through scene, whose
  occupant fills that spot, has none.

Reported to the principal (stop-and-ask rule). Not yet decided.

**Infrastructure defects found and fixed during the run (none touched a scene):**
- Git Bash `tar` read `D:/…` as a remote host (`--force-local`).
- The shared COLMAP container binds only `/leonardo_work`. An ARS wrapper also binds
  `/leonardo_scratch`. Jobs 58564657/58/60 failed on this and were resubmitted.
- The STG walltime was raised 45 → 60 min, since a 21-camera scene needs up to 48 min.
- `v7_reproject.py` was missing STG's `thirdparty/gaussian_splatting` import path.
- The first static leakage script had an incomplete allow-list and was too slow; it was replaced by
  `v9_static.py`.

### 2026-09-26: principal's decision on the admission criterion A3 (logged exception)

**Decision (principal):** admit all 9 wave-1 scenes as they are, and log the A3 miss as a documented
exception.

**The exception:**
- **Criterion:** A3, ≥ 90% of a COLMAP product's points within 0.10 units of a true surface,
  declared in checkpoint 2.
- **Where it is missed:** STG per-frame COLMAP on 3 of 2,700 frames, all gap frames of the mug
  layout: ARS-L3-A0 frame 89 (89.75%); ARS-L3-R2 frames 60 and 89 (89.60% at lowest).
- **Everything else holds on those frames:** every camera registered, no points on the absent
  target, and the off-surface points are ordinary mismatches.
- **Not changed:** the criterion itself, the scenes, and the COLMAP products. Future scenes are held
  to A3 as declared, and any new exception is logged the same way.

**Next:** the instrumented half of the leakage check (V9) runs next, per method on one scene.

**V9, instrumented half, 2026-09-26: PASS for all three methods on ARS-L1-A0.**

**How it was run.** `agent-control/ars/v9_run.py` runs each method's own `train.py` unmodified (via
runpy) with runtime wrappers only:
- It wraps the method's render function(s), recording the camera and whether autograd is on.
- For Ex4DGS and STG it also wraps the scene's train/test camera lists. Ex4DGS reads each step's GT
  image from that list's paths.
- cam00 is identified by its camera centre, within 1e-3 of the rig's cam00; rig cameras are ≥ 0.98
  apart.
- The runs are short and labelled non-evidence; only the iteration count was shortened (config
  copies in `agent-control/ars/configs/v9/`).

**Pass rule:**
- no cam00 render with autograd;
- no cam00 in any training camera list;
- ≥ 100 gradient renders from ≥ 10 cameras;
- a positive control showing the detector recognises the method's own cam00 objects.

| method (job) | renders with gradients | cam00 with gradients | positive control |
|---|---|---|---|
| Ex4DGS (58679159, 4,500 it) | 4,500 from 19 training cameras | 0; 10 training lists (209–589 cameras), 0 cam00 | 111/111 of its test cameras recognised as cam00 |
| STG (58679160, 3,000 it, segment 0–49) | 6,000 from 19 | 0; training list of 950, 0 cam00 | 50/50 test cameras recognised |
| Multi4D (58679161, coarse 300 + fine 3,500) | 30,400 across all four render paths, from 19 | 0 | 3,300 cam00 renders, all gradient-free, all from its periodic evaluation |

Multi4D's evaluation of cam00 selects its `best/` checkpoint; that is test-camera selection, not
optimiser leakage. The final iteration remains the primary reading (rev 2 §8.1).

With this, E6's done-when is met on wave 1:
- COLMAP admission (with the A3 exception);
- loader checks and V7;
- V8;
- V9, static and instrumented.

**Next:** E7 (S0 freeze and wave-1 baseline submission), on the principal's go.

### 2026-09-28: S0 freeze of wave 1 (principal's go), recorded before any baseline or CER output exists

Glossary for this entry: same-pose return (A0), moved return (R2), pass-through (G1); bottle, carton and mug
layouts (L1, L2, L3); gap window (P1), return window (P2), whole frame (H2), ordinary region (ORD); independent
identity check (V8); leakage check (V9).

**Frozen scenes.** Nine spec files in `configs/ars/wave1/`, rendered by `scripts/build_absence_suite_scene.py`
(sha256 `ea827198f08128b29b459f8f0c1299750dad920dabb32bdb646163bfeba24bef`, uncommitted; every MANIFEST records this
generator hash, git head `c45a5c3`). Each spec file is byte-identical to the scene's `event_spec.json`. The MANIFEST
of every scratch copy on Leonardo (`/leonardo_scratch/large/userexternal/siyengar/ars/wave1/<scene>`) was re-hashed
today and equals the local one.

| scene | spec file sha256 | trajectory sha256 | MANIFEST sha256 | files |
|---|---|---|---|---|
| ARS-L1-A0 (bottle, same-pose) | `c88bb84a5cc01956d11e278b455a8fec88e175c9a4637fe6da7f56608e52d7e7` | `543b4a70023fb970881a2dec7f3176c793e54c203d53a39a4740a8d13fb3d790` | `a73e6b0a89ac05783897b821382ea673a049dd1d572b724be1907f8c34cdfb47` | 7,247 |
| ARS-L1-R2 (bottle, moved) | `38413fa5941f99f8a08e4d53b02a370f262f8d522db38a4d0def9b191de000f2` | `016174ae4966e56bea8451cf3951f82dac2331b04777ea1d031b836f74f52bf9` | `1468c2eeadef7fae755d784d009bdb45aeb5b3bd025a0d5b2071c9d99dfb870b` | 7,547 |
| ARS-L1-G1 (bottle, pass-through) | `68f258d6b07995c24b2db63b82d5c138628b8ab4dd6f3befc365bd9db270d7db` | `47b965fda82ab1badb1c2cdecfb04a00715898b1899587a86d84e3d5b5842bb0` | `c6a9ea764981c5a0e31e0d4c3d2aa40b9a58ae37248e0af811368ce35ef09b1f` | 7,547 |
| ARS-L2-A0 (carton, same-pose) | `349aeb896e122413a4e7a4e15b9f07ce60c058311714c0f107991c6022b6bbb0` | `fde58ed09773273544c24c9995c46e698b787b6b3434f4fb89ac852dbce055b5` | `6d686ccb7121d8cee0c16294b97ee6aeb94b878d8b2da83f31408a5e8d4df664` | 7,549 |
| ARS-L2-R2 (carton, moved) | `ae74d72f8582e85b4bf0183a0eac55aeba7331fced362e263eb4fbc118100ff2` | `afb4982a7bf7663da93a697813d72a346f8b58d772c40923a008786f60d19af2` | `2e1350cd786d2d24d2997cc6123a529ffe28a12c9e0c37af3895bf3797dd7afd` | 7,849 |
| ARS-L2-G1 (carton, pass-through) | `66025a6427f5137fcca065e800de9a29699e4451f72b9d522336f510951aa8e0` | `da7b40980494351f46eaf95c220c4aaa43eb0964e0558f091fca8dad3bf98e82` | `6e65f6f6d45b9a3b4456dab05b3de5d61e5c8388869dcfe46285aa1a5c17d171` | 7,849 |
| ARS-L3-A0 (mug, same-pose) | `00cbc5c6e017e99c5a796357ff2c47aea21fa4d557e3defe710f09f361ab7d72` | `7da9f4aa83ef7324e5354743b4873469c2a847a4d2ec938d4f3a4fe2f7e3c39d` | `d996e3643f2d7f544ff4306572425eb438800b618e060a74cce9b5a991f77d3d` | 7,549 |
| ARS-L3-R2 (mug, moved) | `146363ee10ba91af48b3449c464e3df97b23ee2a085b297ec27223ca2f642fa0` | `517835d010c9e3c688aba684bd44e0ac994f7865862ee8f2fb9acbd487d0609a` | `652cfe620b3c8e6c0833c15d55c410e8241587ca62df525e34cb50bfe4dab14d` | 7,849 |
| ARS-L3-G1 (mug, pass-through) | `96d8a9f01be6fd0f0fef1cc24fb92a9f3bff6fc3faecaa9b1b1c05ebb1856e1a` | `4509bc477e29d245774171eace3a515982caed0762c51ba97fe1423452c9b74e` | `854adfe750e8c9f5e2ed5be493d90178ba3ff598bc77bfa5b0c8e630cfa89873` | 7,849 |

**Frozen scoring inputs.** Each scene's `gt/window_package.json` (in its MANIFEST) defines the regions and windows the
proof rule reads; sha256: L1-A0 `2521e275…`, L1-R2 `4b961874…`, L1-G1 `a1d592e7…`, L2-A0 `6ab155ba…`, L2-R2
`720d5c76…`, L2-G1 `db24769a…`, L3-A0 `73dbd575…`, L3-R2 `30da089e…`, L3-G1 `27e128f3…`. The common evaluator is
`scripts/eval_event_windows.py` sha256 `30f0e4b467e360ae4e3836232a01632e68b67eead3ee639bd12935590e760404` (identical
locally and in the E4-validated tree on Leonardo). Also frozen: `configs/ars/ARS_SPEC_FORMAT.md` `183011cd…`, the
renderer tests `d625e962…`, the evaluator tests `9c88baff…`.

**Frozen rules (as amended; nothing below changes after this line without a new dated entry).**

1. **Causal proof, CER against Ex4DGS base.** Six endpoints: the gap window and the return window, for each of
   same-pose, moved and pass-through. Pooled over the three layouts, 9 runs against 9; two-way fixed effects (arm +
   layout); Holm correction over the 6 endpoints. An event type passes when (a) its gap improvement is
   Holm-significant, (b) the one-sided 95% lower bound of CER − base on its return window exceeds −0.25 dB, and
   (c) *(amendment 2026-09-24)* CER's mean gap improvement over base is positive in each of the three layouts
   separately. Causal proof: at least 2 of the 3 event types pass, one of them same-pose.
2. **Against the benchmarks (descriptive).** Per scene, CER (mean of 3 runs) Pareto-dominates a benchmark when it is
   ≥ that benchmark on both gap and return windows. Counts out of 9 are reported against STG, Multi4D and the Ex4DGS
   base mean. Benchmark proof: CER dominates both STG and Multi4D on ≥ 6 of 9 scenes, including at least one of each
   event type.
3. **Do-no-harm.** Whole-frame and ordinary-region PSNR, CER − base, one-sided 95% lower bound > −0.2 dB, pooled.
4. **Preconditions.** A method arm is scored on a scene only if it passes that scene's contribution gates; a failure
   is reported as DESIGN_WITHOUT_POWER for that scene, never dropped.
5. **Scope** *(amendment 2026-09-24)*: repeated trainings measure training noise on these scenes; every claim holds
   "on these nine synthetic scenes". Per-layout differences are reported for every endpoint. The real fixtures are
   always reported separately and never pooled with synthetic scenes; any other combination is written here before
   its runs start.
6. **Multi4D port census threshold (§4b gate 2):** persistent foreground + background carry ≥ 50% of the
   target-region contribution mass over the pre-gap frames, in every layout, on the wave-1 Multi4D same-pose runs.
7. **Independent identity check, strengthened criterion (2026-09-24):** a scene passes iff away-from-edge agreement
   ≥ 0.999 on every pair and pooled, and overall agreement ≥ 0.9999 pooled. Measured on wave 1: 9/9 at 100.0000%.
8. **Dense-initialisation variant (wave 2, not part of the proof):** precondition and robustness criterion exactly as
   written in the 2026-09-24 evening entry.
9. **COLMAP criterion A3:** unchanged; the wave-1 miss (3 of 2,700 STG frames, mug-layout gap frames) stands as the
   logged exception of 2026-09-26.

**Readings frozen with the baselines.** Ex4DGS: `render.py` at 40,000 iterations, cam00. Multi4D: final iteration
20,000 is primary (`best/` is not read). STG seed 0: the 25,000-iteration checkpoint of each 50-frame segment, scored
with `--segment_len 50` so no window value pools two segment models. Every run is scored only after a check that the
method's own saved cam00 ground truth equals the frozen frame on every pixel for all 300 frames (render file *f* is
frame *f*).

**E7 submission, 2026-09-28 (after the S0 entry above).** Every job ID is in `agent-control/ars/ledger.txt` and
`jids/`, written at submission.

- **Rate checks first, on the debug queue, on the carton pass-through scene (the most sparse points, 21 cameras):**

  | method | synthetic scene | real-beef reference | walltime (reference job × 1.3) |
  |---|---|---|---|
  | Ex4DGS (58902948) | 20,000 it in 20:45; 24,700 in 25:26 | 24,530 in 32:02 | 1:11:18 → **1:35** |
  | Multi4D (58902549) | coarse 1,500 in 13:23; fine 990 in 10:35 | 13:24; 11:21 | 2:48:03 → **3:40** |
  | STG (58905317), segment 50–99 | 25,000 in 27:12 | 20,000 in 28:05 | train 49:11 → **1:05**; render 4:11 → **0:06** |

  The first Ex4DGS check (58902548) failed in 32 s on `import torch` on one node; the environment verified intact
  against the torch wheel's own record, and the retry ran normally. Logged in the ledger as a transient node fault.
- **Submitted:** Ex4DGS base × 3 on each of the 9 scenes (27 jobs, render at 40,000 for cam00 in the same job);
  Multi4D × 1 on each scene (9); STG seed 0 on each scene, six 50-frame segments each (54 training + 54 render jobs).
- **Multi4D final-iteration renders.** Multi4D writes cam00 images only for its `best/` checkpoint and has no render
  script. `agent-control/ars/m4d_final_render.py` runs Multi4D's own `train.py` unmodified and, after Multi4D's own
  evaluation at iteration 20,000, renders the test camera with the same call and saves the PNGs. Smoke test at
  iteration 500: 300 frames, mean PSNR 28.174176 against Multi4D's own 28.1741758.
- **Scoring.** `agent-control/ars/ars_score.py` scores a run only after checking its provenance (exit codes, pinned
  commit, clean tree, config hash), the 300 renders at 1352×1014, and that the method's own saved cam00 ground truth
  equals the frozen frame on every pixel for all 300 frames. It then runs the common evaluator. The serial queue
  admits only 10 queued jobs per user, so the 10 per-run scoring jobs first queued were cancelled before they started.
  Two self-resubmitting scoring workers (`ars_score_worker.py`, jobs `ars_w1_scoreworker_A1`/`B1`) now score each run
  as its training completes; failed trainings are marked, never scored.

**Scoring defects found and fixed, 2026-09-28 afternoon (all in my scoring instrument; no run was affected).**

1. **Ex4DGS saves no ground-truth images.** `render.py` creates `gt/` but writes only renders, so the first alignment
   check (method GT equals frozen GT) found nothing on the first 8 runs. Replaced for Ex4DGS by a comparison with
   Ex4DGS's own per-frame PSNR record (`all_metrics.json`) within the E4 tolerance of 0.025 dB, with a positive
   control that a one-frame shift fails.
2. **8-bit rounding.** Ex4DGS scores its float render; the evaluator reads the saved PNG. Rounding adds a nearly
   constant squared error, Q = 1/(12·255²), so the PSNR offset grows with quality: 0.005–0.011 dB near 31 dB, 0.04–0.06
   dB near 40 dB. The E4 tolerance was calibrated on real footage near 33 dB. The predicted offset
   −10·log10(1 + Q/MSE) matched the observed one within 0.0015 dB (31 dB run) and 0.0127 dB (40 dB run); neighbouring
   frames sit ≥ 0.137 dB away (median 2.28 dB). The check now subtracts the predicted offset before the 0.025 dB test.
3. **Two scoring workers claimed one run at once** (a claim whose owner had not yet written its job id was treated as
   stale). The collision hit the carton pass-through runs 2 and 3; one good score was renamed by the second worker.
   Fixed: such claims are live, takeover is atomic, and scores are written to a private file and moved into place.
   Both runs are rescored; the 16 scores that existed then were re-validated.
4. **Serial-queue limits.** `lrd_all_serial` allows 10 queued jobs, 2 running and 30,800 MB per user, so per-run scoring
   jobs could not be queued; scoring runs in self-resubmitting workers (two on the serial queue, two more on GPU-partition
   nodes scoring on CPU).

**Open, for the principal:** the corrected alignment check still misses on 6 and 20 frames of the Ex4DGS runs 2 and 3
on the mug moved-return scene, by 0.0003–0.0025 dB over the tolerance, on gap frames near 43 dB. On those frames the
correct frame's discrepancy is 0.016–0.028 dB and the frames one or two away are 2.6–8.7 dB off, so the file-to-frame
mapping is not in doubt; the rounding model is approximate at that quality. The two runs are not scored until decided.

**Principal's decisions, 2026-09-28 afternoon.** (1) Ex4DGS alignment check: a frame outside the rounding-corrected
0.025 dB tolerance still passes when its discrepancy is at least 1 dB below that of every frozen frame within ±2
frames. Applied to all Ex4DGS runs; the runs scored earlier passed the absolute test on every frame. On the mug
moved-return runs 2 and 3 it passed exactly the 6 and 20 gap frames, with the nearest neighbour ≥ 3.6 dB away.
(2) Commit of the E5–E7 work: done as `3f5efd2`, before this entry.

### 2026-09-28 evening: Ex4DGS base replicate spreads (G-floor, E9), recorded before any CER output exists

No CER run has been submitted and no CER score exists (`scores/wave1/` holds only `ex4dgs`, `multi4d`, `stg`; the
reducer refuses to run otherwise). Source: the 27 verified scores `agent-control/ars/scores/wave1/ex4dgs/*.json`
(their sha256 are inside the output), reduced by `agent-control/ars/g_floor.py`. Output
`agent-control/ars/checks/wave1/g_floor.json`, sha256 `99c548697c79253c3b3951bd839bf539d596a7e5389935415ed04b0093e4816f`.

PSNR in dB of runs 1 / 2 / 3, then the sample standard deviation (sd) and range. Gap window = the ghost region in the
middle of the absence; return window = the target just after it comes back; whole frame = all pixels, all 300 frames;
static background = everything outside the moving and event objects; ordinary region = the other moving objects.

| scene | gap window: runs; sd; range | return window: runs; sd; range | whole frame: runs | static background: runs | ordinary region: runs |
|---|---|---|---|---|---|
| bottle, same-pose | 26.53 / 27.05 / 26.55; 0.30; 0.52 | 23.04 / 23.27 / 23.01; 0.14; 0.26 | 31.39 / 38.73 / 40.55 | 31.31 / 40.11 / 43.03 | 34.38 / 34.16 / 34.51 |
| bottle, moved | 35.73 / 36.52 / 35.10; 0.72; 1.43 | 19.34 / 20.60 / 20.60; 0.72; 1.25 | 40.21 / 38.88 / 38.43 | 43.15 / 40.78 / 39.96 | 34.53 / 34.15 / 34.47 |
| bottle, pass-through | 23.16 / 23.33 / 22.52; 0.43; 0.81 | 23.44 / 24.39 / 24.19; 0.50; 0.95 | 39.74 / 39.11 / 39.76 | 43.38 / 42.02 / 43.40 | 34.33 / 33.97 / 34.40 |
| carton, same-pose | 27.64 / 28.89 / 28.58; 0.65; 1.25 | 32.77 / 32.12 / 31.41; 0.68; 1.37 | 40.89 / 41.14 / 27.68 | 42.36 / 42.95 / 27.41 | 34.67 / 34.31 / 34.35 |
| carton, moved | 35.68 / 36.18 / 35.93; 0.25; 0.50 | 23.87 / 23.82 / 23.75; 0.06; 0.12 | 39.13 / 41.22 / 41.22 | 40.19 / 43.48 / 43.43 | 34.50 / 34.50 / 34.63 |
| carton, pass-through | 19.77 / 19.52 / 20.05; 0.26; 0.52 | 32.54 / 31.92 / 32.34; 0.32; 0.62 | 32.66 / 39.45 / 38.77 | 32.78 / 43.56 / 41.98 | 34.55 / 34.29 / 34.39 |
| mug, same-pose | 27.46 / 28.13 / 29.49; 1.03; 2.02 | 26.17 / 25.64 / 26.22; 0.32; 0.59 | 41.98 / 27.12 / 37.49 | 43.86 / 26.86 / 37.78 | 36.14 / 35.19 / 36.31 |
| mug, moved | 36.52 / 36.06 / 36.16; 0.24; 0.45 | 20.59 / 20.50 / 20.06; 0.28; 0.53 | 40.82 / 41.40 / 41.32 | 43.05 / 43.91 / 43.97 | 36.36 / 36.30 / 36.26 |
| mug, pass-through | 22.70 / 22.13 / 22.99; 0.44; 0.87 | 26.53 / 26.25 / 26.41; 0.14; 0.28 | 40.46 / 34.12 / 40.84 | 43.33 / 34.24 / 44.17 | 36.24 / 35.98 / 36.20 |

Descriptive facts, not interpreted here:

- The gap-window spread is 0.24–1.03 dB sd (range 0.45–2.02 dB). The return-window spread is 0.06–0.72 dB sd (range
  0.12–1.37 dB).
- On 5 of 9 scenes, one of the three runs sits 6–15 dB below the other two on the whole frame and on the static
  background, while its gap, return and ordinary-region values stay close to its siblings'. The runs are: bottle
  same-pose run 1 (9.2 dB below the best sibling), carton same-pose run 3 (13.5), carton pass-through run 1 (6.8), mug
  same-pose run 2 (14.9; its run 3 is a further 4.5 dB below run 1), mug pass-through run 2 (6.7). The do-no-harm
  rule (§4 rule 3) reads the whole frame and the ordinary region.
- Every one of these runs passed provenance, output and alignment checks; none was dropped or rerun.

### 2026-09-29: diagnosis of the Ex4DGS whole-frame drops (non-evidence diagnostics; nothing re-scored or dropped)

- **Where.** In every flagged run the damage is one static region seen from the held-out camera: the dark panes of
  the window behind the cook, rendered as light grey cloud-like blobs (on the mug same-pose run 2 also the poster).
  Everything else matches the good siblings. The drop is flat over all 300 frames (a static-model defect, not the
  time machinery). Figures: `D:\adags-ars\wave1\diag\` (per-frame curves, bad-vs-good montage).
- **Training views are unaffected.** End-of-training PSNR on the training cameras and the static/dynamic Gaussian
  counts of the bad runs equal their siblings' (e.g. bottle same-pose 41.34 vs 41.38/41.29; 29,697 vs 29,726/30,541
  static Gaussians).
- **The trained model is bad, not the render.** Re-rendering the saved 40k models of three bad runs on three other
  nodes reproduces their scores to 2e-5 dB (jobs 59019955/58/62).
- **Node trend.** All 5 Ex4DGS runs trained on `lrdn0066` are degraded (4 severe, 1 by 2.2 dB); of the 22 runs on
  other nodes, 2 are (mug same-pose runs 2 and 3, on `lrdn3254` and `lrdn0088`). Under no node effect, all five
  `lrdn0066` runs falling among the 7 degraded runs has probability ≈ 2.6e-4. No Multi4D or STG job ran there.
  This is a trend, not a proven hardware fault; the decisive test (repeat trainings pinned to `lrdn0066`) is proposed,
  not run.
- **Rendering is not bit-deterministic either:** the same saved model rendered on two GPUs differs by up to 36/255 at
  single pixels.
- **Visual run reports** now exist for all 45 runs (`runs/<method>/<run>/report/`: stats.json, per-frame curves,
  key-frame montages with error maps and zooms) and one comparison image per scene
  (`agent-control/ars/reports/wave1/`); the scoring worker writes one for every run it scores from now on.

### 2026-09-29 evening: baseline completion batch (principal: CER will be built on both Ex4DGS and Multi4D)

Submitted so that no baseline needs rerunning once CER exists; every job ID is in the ledger.

1. **Multi4D runs 2 and 3 on all 9 wave-1 scenes** (18 jobs, 59036033–59036051): the same design as Ex4DGS (3
   repeated trainings per scene), so §4 rule 1 applies unchanged to CER on Multi4D. Identical to run 1.
2. **Ex4DGS and the `lrdn0066` question.**
   - Replacements on other nodes for the 5 runs trained on `lrdn0066` (59036052–56). The originals stay on the
     record; whether a replacement stands in for its original in the proof is the principal's decision after the
     node test.
   - Node test: 3 trainings of the bottle pass-through scene pinned to `lrdn0066` (59036057/58/60). That scene's
     three existing runs, on other nodes, are within 0.66 dB of each other.
   - Every other job now excludes `lrdn0066`.
3. **Dense-initialisation variant (wave 2, bottle same-pose), base arm.** A trap was caught before submission.
   Ex4DGS divides its sparse points by a scene normalisation radius when it builds `points3D.ply`
   (`scene/dataset_readers.py:499-503`), so a dense cloud copied in unscaled would have loaded at the wrong scale while
   still passing the frozen precondition (initial count = dense count). `agent-control/ars/build_dense_tree.py`
   therefore:
   - recovers the radius from the sparse ply against the bin (1.0000 on this rig, consistent to 1e-7);
   - checks that the dense cloud sits on the sparse points after normalisation (median nearest-point distance 0.023,
     scene extent 64.5);
   - writes the 198,546 points in those units (`checks/wave2/dense_tree_ARS-L1-A0.json`).

   A debug-queue speed check (59036061) runs before the 3 base runs.

**Dense-initialisation base runs submitted** (59036768/70/72). The speed check (59036061) logged 198,546 initial
points, equal to the dense cloud, so the frozen precondition holds; its training speed equals the sparse runs'.

### 2026-09-30: principal's decisions (recorded before any CER code exists)

1. **CER is built on both Ex4DGS and Multi4D, in parallel.** This changes §4b, which made the Multi4D port wait for
   the Ex4DGS causal proof (§4b gate 1). Gate 1 no longer applies.
2. **The Multi4D census runs first.** §4b gate 2, with the 50% threshold frozen at S0: persistent foreground +
   background must carry ≥ 50% of the target-region contribution mass over the pre-gap frames, in every layout. It
   runs on the existing wave-1 Multi4D run 1 of the three same-pose scenes (bottle, carton, mug). CER-Multi4D is built
   only if the census passes.
3. **CER code lives in a separate git worktree per substrate, on a `cer` branch.** The pinned baseline checkouts
   (`repo/ex4dgs` @ `7fac649`, `repo/multi4d` @ `c483e82`) are never modified.
4. **Wave 2 is on hold until the wave-1 proof reads out.** The "no new datasets after Sep 30" rule is to be revisited
   then.

Note (mine, not part of the decisions): the wave-2 dense-initialisation base runs submitted on 2026-09-29
(59036768/70/72) were already running when wave 2 was put on hold; they are left to finish and be scored.

### 2026-09-30 01:50: Multi4D census (§4b gate 2, frozen rule 6): PASS in every layout

Wave-1 Multi4D run 1 of the three same-pose scenes, final iteration 20,000, pre-gap frames 0–59, training cameras
only (cam00 never read). Share of the target-region contribution mass (Σ αT inside the target's front-most mask),
measured by rendering Multi4D's own final hybrid pass with each bank's colour set to a one-hot indicator through the SH
constant term (the hybrid rasterizer takes no precomputed colours for the persistent set):

| same-pose scene | persistent foreground | static background | transient | foreground + background | ≥ 0.50 | per camera min / median / max |
|---|---|---|---|---|---|---|
| bottle | 0.0002 | 0.9904 | 0.0094 | 0.9906 | pass | 0.986 / 0.991 / 0.992 |
| carton | 0.0056 | 0.9848 | 0.0096 | 0.9904 | pass | 0.971 / 0.992 / 0.992 |
| mug | 0.0001 | 0.9833 | 0.0166 | 0.9834 | pass | 0.958 / 0.984 / 0.986 |

- Self-checks: re-rendering cam00 at frames 0/30/59/150 reproduces the saved final renders byte for byte on all three
  scenes; the three bank channels sum to the rendered alpha within 1.01e-6; permuting the bank-to-channel assignment
  permutes the masses exactly. The carton scene's cam01 sees no target pixel before the gap; its 60 camera-frames carry
  no mass and are excluded.
- Jobs: 59049323 failed in 19 s on the script's own config comparison (`saving_folder`, an output path that no model or
  render code reads); 59049518 is the run of record. Script `agent-control/ars/m4d_census.py` sha256 `98d6e782…`;
  outputs `agent-control/ars/checks/wave1/m4d_census_ARS-L{1,2,3}-A0.json` (`81fc5c54…`, `5b03bd9e…`, `2db216a9…`) and
  `m4d_census_summary.json` (`ae6d45f4…`, `gate_pass_all_layouts: true`).
- **Descriptive fact that shapes the port (not interpreted further):** the target is painted by the static background
  bank, not the persistent foreground bank (≤ 0.6%). CER on Multi4D therefore gates background rows at both background
  render sites (standalone and hybrid) in addition to the three foreground paths §4b names.

### 2026-09-30 03:40: CER built on both substrates; Ex4DGS diff ready for the principal's review; Multi4D blocked on an upstream defect

Glossary: memory cohort = existing rows that paint the target before it leaves (switched off in the gap); residual
cohort = new gap-only rows; gap = frames 60–89.

**Ex4DGS (branch `cer`, worktree `repo/ex4dgs-cer`; pinned `repo/ex4dgs` untouched at 7fac649, 0 porcelain).**
- Spec [cer-ex4dgs-spec-2026-09-30](cer-ex4dgs-spec-2026-09-30.md) rev 2, after GPT-6.1 Sol (high, material defect, 7
  findings) and Fable 5.1 (ready after fixes) on rev 1; code reviewed by both (Sol: material defect in tests and one
  guard; Fable: ready after one launcher fix). Every finding applied; reviews in `assets/cer-ex4dgs-2026-09-30/`.
- Commits, tests before code: f37694c tests → cd5ef1e code → 73e9b65 tests strengthened → f71c1b0 code fixes. Diff for
  review: `assets/cer-ex4dgs-2026-09-30/cer_ex4dgs_7fac649..f71c1b0.diff` (substrate: 48 changed lines in 5 files, plus
  `scene/cer.py` 157 lines and `test_cer.py`). No CUDA change.
- Tests on Leonardo, bottle same-pose, wave-1 base run 2: on the tests-only commit T0/T1 pass and T2 fails with
  ImportError (59050085); on f71c1b0 every test passes (59051055): renders with q = 0 byte-identical to unmodified
  Ex4DGS on all 300 frames with 299 residual rows present; exact factors; renders outside the gap byte-identical at
  q = 1; zero rendering gradient to the switched-off cohort (positive control at q = 0); birth leaves every existing
  row and all 15 RAdam groups untouched; instrument conservation 5.7e-5; checkpoint round trip incl. optimizer state;
  ply fallback and a CER directory without CER state both refuse; prune alignment; non-finite opacity caught.
- Membership forecast on the nine wave-1 run-1 base models (59050133): memory share 0.88–0.95 of the pre-gap target
  mass (mostly dynamic-bank rows), 217–434 residual rows, donors paint 0.36–0.81 of the gap region at birth.
- T7 integration (shortened 17,200 iterations, not evidence; cd5ef1e 59050242/59050304 and f71c1b0 59051056/59051057):
  both pass; render.py's cam00 image equals a direct gated render. **Descriptive: in both smoke runs the residual share of
  the gap region ended at 0.10 and 0.09, below the pre-declared 0.20 gate**; memory 0.92–0.93 before, 0.91 after; upstream
  pruning at 17,101 removed about 70% of the memory rows while the rest still carried the target.
- Scoring deployed: `ars_score.py` method `ex4dgs_cer` accepts only commit f71c1b0 on branch `cer` with a clean worktree
  at training and at render and the 40,000 render iteration, and carries `cer_gates.json` into the score (every run
  scored; rule 4 in the reducer); 27 CER runs added to `ars_score_worker.py`. Negative check on the T7 run: refused
  (render iteration 17,200), nothing written. Originals kept as `*.pre_cer.py`. Launcher
  `agent-control/ars/sbatch/train_ex4dgs_cer.sbatch`. **Nothing submitted; waits for "submit".**

**Multi4D.** Census PASS (entry above). Spec [cer-multi4d-spec-2026-09-30](cer-multi4d-spec-2026-09-30.md) rev 2
after Sol and Fable; worktree `repo/multi4d-cer` (pinned `repo/multi4d` untouched at c483e82); commits bf838c6 tests →
c5ac11e code; own JIT extensions directory `build/torch_extensions_multi4d_cer`. Tests: on bf838c6 T0/T1 pass, T2 fails
with ImportError (59051214); on c5ac11e (59051536) T0/T1 pass and **`begin` stops on its per-render conservation
assertion: the contribution mass is exactly 0 against 1.37e6 on the first render**.

**Upstream defect found (Multi4D hybrid rasterizer, not CER):** `max_contrib` is allocated
(`diff_gaussian_rasterization_hybrid/cuda_rasterizer/rasterizer_impl.cu:189`) and passed to the forward kernel but never
written (the forward writes only `n_contrib`, `forward.cu:722`), while the launched backward returns early for every
bucket with `bucket_idx·32 ≥ max_contrib[tile]` (`backward.cu:1229`). Every gradient of the hybrid render therefore
depends on uninitialised memory: zero-filled memory drops a whole tile (here all tiles), large garbage computes it. This
affects Multi4D training itself, baselines included; its size in training is not measured. Decision put to the principal.

### 2026-10-01: principal's decisions on CER (recorded before any CER run is submitted)

1. **Residual gate frozen (Ex4DGS and Multi4D):** the primary threshold of the "residual active inside the gap" gate is
   **0.20** (residual share of the gap-region contribution mass over the gap window, training cameras, final model), as
   declared in the CER specs. **Stated with it: two shortened smoke runs (T7, not evidence) were already seen, with
   residual shares 0.10 (cd5ef1e) and 0.09 (f71c1b0).** The reducer also reports the verdicts at 0.10, 0.20 and 0.30 as a
   descriptive table; only 0.20 is primary. The other gate thresholds stand as declared (memory ≥ 0.50 before the gap
   and after the return; exact zeros as guards).
2. **Residual donors measured on the memory-gated model (Fable F1): confirmed.**
3. **Multi4D instrument, option (a):** the training kernel stays unmodified in both arms; a separately built copy of
   the hybrid rasterizer that writes `max_contrib` is used only inside the contribution-mass instrument, with the
   per-render conservation check kept. It is a CUDA change on the `cer` branch, in its own commit, through the same
   two-model code review.
4. **The `max_contrib` defect's effect on the baselines:** not now. Later, on the principal's yes, as a diagnostic only
   (fraction of tiles skipped in a real training process at a few iterations); no reruns.
5. **Multi4D round-trip test:** replaced as proposed (saved CER state equals the in-memory state; a saved CER model
   refuses to reload).
6. **Before "submit" on Ex4DGS:** (a) forecast the donor count and gap mass by bank (static / dynamic) on the three
   pass-through scenes, since residual copies are frozen snapshots at frame 74 and a moving donor would leave a
   stationary ghost; report, no design change; (b) a montage of the smoke renders at frames 59, 74, 90 beside the base
   run, gap region zoomed; (c) whether the base arm loses a similar share of the same memory rows to upstream pruning
   over 30k–40k (descriptive, from existing checkpoints).
7. **Wave-2 ablations, recorded, not now:** residual initialisation (low opacity at birth; a donor→copy swap in the gap;
   dynamic copies for moving donors); start iteration (20k vs 30k, with the label plumbing an earlier start needs); the
   0.5 majority rule. The doubled layer at birth (donor and copy both render in the gap) is a known design point, not a
   bug.

### 2026-10-01 01:20: Ex4DGS CER diagnostics requested before "submit" (descriptive, not evidence)

Job 59077300 (script `agent-control/ars/cer/cer_diag.py`, run on the `cer` worktree at f71c1b0; outputs
`agent-control/ars/cer/diag/*.json`). Scene units; the scene extent is ≈ 6.5.

**(a) Pass-through scenes, residual donors by bank** (wave-1 base run 1 at 30k, `begin()` as in training):

| pass-through scene | donors static / dynamic | gap-region mass carried by donors, static / dynamic | gap-region mass of all rows, static / dynamic | dynamic donors' largest move across the gap: median / 90th pct / max |
|---|---|---|---|---|
| bottle | 15 / 255 | 0.001 / 0.360 | 0.031 / 0.969 | 0.31 / 1.48 / 3.44 |
| carton | 4 / 239 | 0.003 / 0.743 | 0.152 / 0.848 | 0.27 / 1.26 / 3.83 |
| mug | 2 / 369 | 0.001 / 0.746 | 0.097 / 0.903 | 0.34 / 1.77 / 4.91 |

On the pass-through scenes almost every donor is a dynamic row, and many move during the gap by up to ≈ 0.5–0.75 of
the scene extent; the residual copies are static snapshots at frame 74. Reported only; no design change (principal).

**(b) Pruning of the memory rows in the base arm, 30k → 40k** (27 wave-1 base runs; memory rows by the `begin()` rule on
chkpnt30000, ungated). `prune_invisible` reads per-row timestamps frozen after 30k and `prune_small` a running minimum
radius, so rows with timestamp < 0 or minimum radius < 5 px at 30k are certainly removed by 40k; the row counts at 40k
bound the rest, but the bound is loose (5–9k dynamic removals per run are not explained by the 30k state, so the upper
bound exceeds 100%). Certainly removed share of the memory rows, lower bounds:

| bank | across the 27 runs |
|---|---|
| dynamic memory rows | 0.44–0.77 (same-pose 0.56–0.76, moved 0.44–0.63, pass-through 0.57–0.77) |
| static memory rows | 0.00–0.21 |

For comparison, the CER smoke run lost 71% of its dynamic memory rows and 16% of its static ones after `begin`. The base
arm therefore loses at least a similar share of the same rows; the loss is upstream behaviour, not caused by CER. Not
measured: how much target mass the removed rows carried (the smoke run's remaining memory rows still carried 0.92 of it).

### 2026-10-01 02:20: pass-through residuals, option (a) taken (principal's recommendation); spec amendment under review

- **Decision:** residual copies become verbatim copies in the donor's own bank (dynamic donors into the dynamic bank with
  their keyframes, so a copy moves exactly as its donor; static donors copied with their displacement), gated to the gap,
  instead of rev 2's static snapshot at frame 74. Spec decision proposed in the amendment: a dynamic copy keeps its own
  temporal opacity inside the gap. Option (b), keeping the snapshot, was rejected because it would knowingly handicap the
  pass-through third of the proof.
- **Amendment:** [cer-ex4dgs-spec-amendment-a-2026-10-01](cer-ex4dgs-spec-amendment-a-2026-10-01.md), sent alone to both
  reviewers; then tests → code → code review → tests, as before. The Ex4DGS diff under the principal's review
  (`f71c1b0`) therefore changes in `begin()`'s birth step only; the scorer pin moves with the final commit.
- **Requested descriptive check:** donor-by-bank forecast on the same-pose and moved-return scenes, job 59078436.

### 2026-10-01 03:05: Ex4DGS donor forecast on all nine scenes; Multi4D residual-cohort forecast (decision for the principal)

**Ex4DGS, donors by bank** (job 59078436 for same-pose and moved-return, 59077300 for pass-through; base run 1 at 30k):
every scene has donors in both banks, mostly dynamic (static / dynamic: same-pose 19/351, 7/243, 3/355; moved-return
18/199, 36/398, 34/189; pass-through 15/255, 4/239, 2/369, bottle / carton / mug), and the dynamic donors move during the
gap on every condition (median of each donor's largest move 0.26–0.52, maximum 1.8–4.9 scene units, extent ≈ 6.5). This
motivated amendment (a) beyond the pass-through scenes.

**Multi4D, residual cohort** (wave-1 Multi4D run 1, `begin()` at fine 16,000; jobs 59078553/59078592 bottle, 59078606
carton and mug): with the memory rows off, the **transient bank paints 0.88–0.96 of the gap region** (bottle 0.96, carton
0.88, mug 0.91); background rows 0.02–0.10, foreground 0.001–0.07. Background donors (the only donors the Multi4D spec
copies) are **6, 0 and 1 rows** on the three same-pose scenes, carrying ≤ 0.0004 of the gap-region mass, while transient
rows meeting the donor rule carry 0.87–0.90. Multi4D already represents the revealed background with transient rows, so
on Multi4D CER reduces to its memory gate; the frozen residual gate (≥ 0.20 of the gap region) is forecast to fail on all
three same-pose scenes, which under rule 4 would make every CER-Multi4D scene DESIGN_WITHOUT_POWER. Not interpreted
further; the choice is the principal's (the Multi4D port is otherwise implemented and its tests pass except one test bug,
fixed in 00357aa, not yet re-run).

### 2026-10-01 04:10: CER test status on both substrates (nothing submitted)

- **Ex4DGS, amendment (a) implemented and tested.** Commits 7e3ccf7 / 319b1e4 (tests) and 4a92369 / 5b82ae7 (code) on
  `cer`, after the amendment's spec review (Sol, Fable; rev 2) and code review (Sol: material defect in the tests,
  fixed; Fable: ready). Tests-only commit: T2c failed with KeyError as designed (59078802). On 5b82ae7 every test passes
  (59078863), incl. q = 0 byte identity with residual rows in both banks (12 static, 287 dynamic), copies equal to their
  donors in every parameter and evaluated input (also for begin() driven to static-only / dynamic-only / mixed donors
  across the bank boundary), the donor-swap diagnostic 0 differing values, exact birth state, round trip and pruning with
  residuals in both banks. T7 smoke (59078864/65): passes; descriptive gates: memory 0.93 / 0.92, residual share 0.060
  (residual rows 12 / 284 at birth, 8 / 80 after the 17,101 pruning). Scorer pin moved to 5b82ae7. Review diffs:
  `assets/cer-ex4dgs-2026-09-30/cer_ex4dgs_7fac649..5b82ae7.diff` and `..._f71c1b0..5b82ae7_amendment_a.diff`.
- **Multi4D.** Commits through 7d8c5e2 (code) and 00357aa (tests), incl. the CUDA instrument commit 33215f1 and both code
  reviews (Sol: material defect in tests and plumbing, fixed; Fable: code ready). Tests: T0–T2 (59078492), T2b–T3
  (59078553), T4–T4b (59078592; the training build breaks conservation in a fresh process, the instrument build
  conserves), T5–T6 (59079481, after a test-ordering fix; the first run 59078600 failed on that test bug) all pass; T7
  (59078493/94, smoke schedule) passes, gates descriptive: memory 0.94 / 0.94, residual share 0.024. The residual-cohort
  question above is the open decision.
- **Queue incidents, recorded:** a JIT build lock left by a job I cancelled mid-build (59077345) blocked 59078050 (timed
  out) and 59078060 (failed after 35 min); the lock was removed and the extensions are now built once by a dedicated job
  (`sbatch/cer_m4d_build.sbatch`) before any user of them.

### 2026-10-01 (later): principal's decisions on the contribution gates and on Multi4D (recorded before any real CER run)

Glossary: memory gate = the memory cohort switched off inside the gap; residual copies = the gap-only copies of the
donors; gap content = whatever paints the gap region during the gap.

1. **(a) The residual share becomes descriptive, not a design-without-power gate (amendment, proposed before any real
   CER run).** The share of the gap-region mass carried by the residual copies is reported as the attribution between
   the memory gate and the residual copies; it no longer decides whether a scene is scored. The preconditions that stay
   (plan §4 rule 4): memory share ≥ 0.50 of the target mass before the gap and after the return, and the exact zeros
   (memory mass in the gap, residual mass outside it). **Stated with it:** the residual shares of the shortened smoke
   runs (integration tests, not evidence) were already seen, 0.06–0.10 on Ex4DGS (0.10 and 0.09 with the rev-2
   snapshot birth, 0.06 with amendment (a)) and 0.024 on Multi4D. This supersedes the 2026-10-01 freeze of the residual
   gate at 0.20 as a precondition; the reducer's 0.10 / 0.20 / 0.30 table stays, as a description.
2. **(b) Multi4D is treated as a memory-gate arm:** the exact memory gate on its persistent banks, with Multi4D's native
   transient bank as the gap content. On Multi4D the gap-content gate counts residual or transient rows (the census and
   forecast show the transient bank paints 0.88–0.96 of the gap region once the memory rows are off).
3. **(c) Exploratory Multi4D work** (for example transient donors, or gating the transient bank) comes only after the
   confirmatory design is frozen, and is labelled exploratory.

Not yet done (open items in [cer-handoff-2026-10-01](cer-handoff-2026-10-01.md)): writing (a) and (b) into both CER specs,
the code that reports instead of gating, and the reducer.

### 2026-10-01 afternoon: gate amendment written and reviewed; Multi4D scoring, the reducer and both submission scripts ready (nothing submitted)

Glossary: memory gate = the memory cohort switched off inside the gap; residual copies = gap-only copies of the donors;
gap content = whatever paints the gap region during the gap; design-without-power = a scene or rule whose preconditions
fail, reported, never dropped.

1. **The gate amendment** is appended to both specs: [Ex4DGS §13](cer-ex4dgs-spec-2026-09-30.md),
   [Multi4D §14](cer-multi4d-spec-2026-09-30.md).
   - Rev 1 went alone to both reviewers. Fable 5.1: ready after fixes (13 findings). GPT-6.1 Sol: material defect.
     Rev 2 absorbs every finding. Reviews, prompt and rev 1 text are in `assets/cer-gates-amendment-2026-10-01/`.
   - **No substrate code change.** Both `cer_gates.json` already store every field the amended rules need. The old
     `pass` field (which still requires a residual share of 0.20) is never read; the reducer recomputes the verdict and
     uses the old field only as an integrity check. The Ex4DGS runs therefore go on the reviewed commit 5b82ae7, and the
     scorer pin is unchanged.
2. **New fact about Multi4D, from the shortened smoke run** (integration test, not evidence; `cer_t7_m4d_r4`):
   - the transient bank paints 0.001 of the gap region at CER start and 0.0009 at the end;
   - the gap region is painted by the persistent foreground (0.45) and background (0.55) rows at start;
   - the residual copies carry 0.024 at the end.

   The forecast on full-schedule base models said 0.88–0.96 transient. The smoke schedule gives the transient bank 500
   iterations after the lift, against 6,000 in a real run, so the two may not disagree on real runs; that is not
   measured. It bears directly on decision (b).
3. **Two proposals wait for the principal** (amendment rev 2, marked as not frozen):
   - **The Multi4D gap-content gate:** read as descriptive, not a precondition. Both reviewers recommend this; a
     precondition at 0.20 would make every Multi4D scene design-without-power if the smoke value holds.
   - **How a scene that fails the preconditions enters the proof (Ex4DGS 13.4).** No gate changes a denominator, the
     Holm family (always 6; 2 on Multi4D) or a threshold. An event type with such a scene is design-without-power. Do-no-
     harm is certified only on all nine scenes. Rule 2 stays out of 9.
4. **Multi4D scoring installed on Leonardo:** method `multi4d_cer` in `ars_score.py` and 9 worker rows (three same-pose
   scenes × 3). Previous versions are kept as `*.pre_m4dcer.py`.
   - The method accepts only commit 00357aa on `cer`, with the worktree clean before and after, the full schedule, the
     scored scene as `cer_scene`, and exactly 40 fine evaluations.
   - Checks run there: the smoke run is refused; a doctored fake run reaches the evaluator; six negative variants are
     each refused; the base path is unchanged.
5. **Reducer** `agent-control/ars/cer_reduce.py`, sha256 `4abbc88a…`. It refuses partial or non-finite inputs and an
   inconsistent stored `pass`. It needs an explicit base-run choice (originals or the `lrdn0066` replacements; still
   open) and a reference to the plan-log entry recording the principal's acceptance of the policy.
   - **Statistical reading, written down here before any CER output exists:**
     - alpha 0.05, one-sided tests;
     - the gap window is a superiority test, the return window a non-inferiority test at −0.25 dB;
     - the six-test Holm family contains both;
     - the return clause uses the unadjusted 95% bound, as frozen;
     - rule 1 uses arm + layout fixed effects (18 runs per event type); rule 3 uses arm + scene fixed effects.
   - It also reports per-layout differences with Welch intervals on every endpoint, and the residual-share (and on
     Multi4D the gap-content) table at 0.10 / 0.20 / 0.30.
   - Tests: 10 pass locally and on Leonardo; three deliberate mutations are each caught. Code review by Sol (one
     reviewer) is running.
6. **Submission scripts** `agent-control/ars/submit_cer_ex4dgs.sh` (27 runs, 02:00 walltime) and `submit_cer_multi4d.sh`
   (9 runs, 03:40). Each runs a fail-closed preflight and passes its dry run; neither has submitted anything.
   - Ex4DGS preflight: commit, clean worktree, scorer pin, `lrdn0066` excluded, run ids free, inputs present.
   - Multi4D additionally refuses until the design is recorded frozen, and checks for built extensions and no
     stale build lock.
   - The Multi4D launcher differs from the base launcher only in the declared CER points.
7. **Code review of the scoring, reducer and submission work** (one reviewer, GPT-6.1 Sol; three rounds; records in
   `assets/cer-gates-amendment-2026-10-01/code_review_*`). Rounds 1 and 2 each returned a material defect. Every
   confirmed finding is fixed, tested and installed on Leonardo:
   - **CER failures** are declared to the reducer explicitly (job id and reason) and make their scene
     design-without-power; any other missing score still refuses.
   - **Replacements** must be infrastructure-classified records. Labels and underlying run directories must be
     distinct, and each score's method and scene must match the plan.
   - **Final-render record (Multi4D):** the scorer reconciles it against Multi4D's own iteration-20,000 evaluation and
     its 300 per-frame values, instead of trusting its discrepancy field.
   - **Launchers:** both now require the reviewed commit and a clean worktree at job start, checked before anything is
     created; the submission scripts export the pin. The Multi4D script also forces the smoke overrides empty and
     checks the recorded census pass.
   - **Inputs:** the preflights check the window package, at least 19 identity files and 300 held-out frames.
   - **Tests and checks:** 12 reducer tests; seven deliberate mutations caught. On Leonardo: the launchers refuse a
     wrong or empty pin with no run directory created; the scorer refuses inconsistent, NaN or edited render records.
   - **Not adopted:** immutable code snapshots. The worktrees change only by a manual pull, which the hand-off forbids
     while a job runs; the pin closes the queue window. The round-3 fixes were not sent back for a fourth round.
   - **Left to the principal** (Sol): the statistical reading. The return bound is unadjusted as frozen, not Holm;
     rule 3 is blocked by scene; Multi4D run 1 is the benchmark as frozen.

### 2026-10-01 evening: principal's decisions on the reducer, the statistics, the base runs and Multi4D (recorded before any CER run is submitted)

1. **Reducer policy accepted:** Ex4DGS spec §13.4–13.5 and Multi4D spec §14, rev 2.
   - No gate changes a count, the Holm family or a threshold.
   - A scene is design-without-power if any of its three CER runs fails the checks, and so is its event type.
   - Do-no-harm is certified only on all scenes of the design.
   - Benchmark counts stay out of 9.
   - CER failures are declared explicitly; replacements are infrastructure-only.

   The Multi4D gap-content share (residual + transient) is therefore descriptive, not a precondition (§14.2 reading
   confirmed).
2. **Statistical reading:**
   - alpha 0.05, one-sided tests;
   - gap window: superiority test, Holm over the six endpoints (two on Multi4D);
   - return window: the unadjusted one-sided 95% lower bound must exceed −0.25 dB, as frozen; its Holm-adjusted p is
     reported descriptively;
   - do-no-harm: arm + scene fixed effects, pooled; the one-sided 95% lower bound must exceed −0.2 dB on the whole
     frame and on the ordinary region;
   - benchmark: Multi4D run 1, as frozen; the mean of its three runs is reported descriptively.
3. **Ex4DGS base runs for the proof:** the replacements for the five runs trained on `lrdn0066`. Every base run trained
   on `lrdn0066` is replaced, matching the CER arm's node exclusion; no other run is replaced. Rules 1 and 3 are also
   reported with the original runs as a sensitivity analysis.
4. **Multi4D:** the confirmatory design (§14 rev 2) is accepted and frozen by this entry.
5. **Go:** submit the 27 Ex4DGS CER runs; the 9 confirmatory Multi4D CER runs may also be submitted.

**Submitted, 2026-10-01 ~16:55 (after the entry above).** Every job id is in `agent-control/ars/ledger.txt` and `jids/`.

- **CER on Ex4DGS, confirmatory:** 27 runs, 59107748–59107774 (`ars_w1_cer_<scene>_rep<r>`), commit 5b82ae7, walltime
  2:00, `lrdn0066` excluded, launcher pinned to the commit at job start.
- **CER on Multi4D, confirmatory (memory-gate arm):** 9 runs, 59107775–59107783 (`ars_w1_m4dcer_<scene>_rep<r>`), commit
  00357aa, walltime 3:40, same exclusions and pin.

All 36 were pending (Priority) at submission. The scoring workers score each run as it completes. They pick up the
Multi4D CER rows at their next resubmission.

**The reducer now also reports rules 1 and 3 with the original `lrdn0066` base runs** (`sensitivity_original_base`;
the replacements are the runs of record). sha256 `55d9a1b4…`; 13 tests pass on Leonardo.

**2026-10-01 21:32, principal's request:** the 9 Multi4D CER jobs (59107775–59107783) were cancelled while still pending.
None had started and none had created a run directory; each cancellation is a ledger line. They were resubmitted
unchanged under the job names `…_r2`, as 59122506–59122516, with the same run ids, commit, walltime and exclusion.
`submit_cer_multi4d.sh` gained `SUFFIX`; with it set, the script requires every replaced job to have been cancelled
before it started. The resubmission's priority is 132,631, against about 133,038 for the cancelled jobs; that is the
age lost.

### 2026-10-02 afternoon: CER training and scoring done; principal's decision on the Ex4DGS alignment check

- **Training:** all 27 Ex4DGS CER runs completed, exit 0, 54–63 min each, between 03:54 and 10:36. All 9 Multi4D CER runs
  (`_r2`) completed, 2:36–2:47 each.
- **Preconditions:** from the gate files, read before any score was opened. They hold on all 18 same-pose and
  pass-through Ex4DGS runs. All 9 moved-return runs fail "memory active after the return" (memory share 0.000–0.002):
  the target returns at a new pose. Under 13.4 the three moved-return scenes and the moved-return event type are
  design-without-power.
- **Scoring:** the scoring workers were stopped from 2026-10-01 21:34 to 2026-10-02 11:27 at the principal's request.
  Scored: Ex4DGS CER 26 of 27, Multi4D CER 9 of 9.
- **The one refusal:** carton moved-return run 3 failed the Ex4DGS alignment check on gap frames 76 and 77. The
  discrepancy against the correct frame was 0.0253 / 0.0250 dB; against frames ±1 it was 0.85 / 1.25 and 0.70 / 1.04 dB.
  Both the absolute 0.025 dB test and the 1 dB relative rule missed: the scene barely changes between mid-gap frames.
- **Principal's decision (before any CER score was opened):** the relative rule becomes a ratio. A frame outside the
  rounding-corrected tolerance passes when its discrepancy is at least 10 times smaller than that of every frozen frame
  within ±2. This replaces the 1 dB margin of 2026-09-28 for all Ex4DGS scoring.
  - Runs scored earlier are unaffected: they passed the absolute test, or the 1 dB margin, which the ratio also
    passes in every recorded case, since the neighbour ratio was at least 30 times.
  - The refused run and the two dense-start runs refused on the same check (wave 2, bottle same-pose runs 2 and 3) are
    rescored under it.

**Multi4D reading (memory gate on Multi4D, confirmatory, same-pose; 9 CER runs against the 9 wave-1 Multi4D base runs):
PASS.** Reducer `cer_reduce.py` sha256 `ea2565c6…`; output `agent-control/ars/checks/wave1/cer_reduce_multi4d.json`,
sha256 `b0ca8cee82b59d615a17a3828778c156eca047ec1328313a2f97926f66898023`.

- **Gap window:** CER − base +4.25 dB (se 0.84, one-sided 95% lower bound +2.78; Holm-adjusted p 1.7e-4 over the 2
  endpoints).
- **Return window:** −0.01 dB (lower bound −0.18 > −0.25, non-inferior).
- **Gap gain per layout:** bottle +4.79, carton +4.00, mug +3.97; positive in each.
- **Do-no-harm:** whole frame +0.03 (bound −0.03); ordinary region −0.02 (bound −0.14); PASS.
- **Rule 4:** every run meets the preconditions. Memory share is 0.91–0.95 before the gap and after the return; the
  exact zeros hold.
- **Descriptive:**
  - the bottle layout's gap difference has a wide Welch interval (−5.7, +15.3), so its runs vary a lot; carton and mug
    lie within +3.2 to +4.8;
  - the return-window differences per layout run from −0.22 to +0.15, every interval spanning 0;
  - the gap content is the transient bank (0.23–0.30 of the gap region) and the residual copies (0.0001–0.0017).
- **Reducer defect found on this first real run:** writing the output failed on a numpy boolean (the return bound).
  Fixed: the bound is cast to float, and the tests now check the JSON output. The failed attempt wrote nothing.

### 2026-10-02 evening: the CER proof reading on Ex4DGS (wave 1)

- **Inputs:**
  - all 27 Ex4DGS CER scores. The last, carton moved-return run 3, was scored under the ratio rule (frames 76/77 passed
    at ratios 33 and 28) by job 59227700 on a compute node in the debug QOS: the login-node workers were CPU-starved,
    one got 20 s of CPU in 27 min;
  - the 27 Ex4DGS base runs, with the five `lrdn0066` runs replaced, as the runs of record;
  - STG seed 0 and Multi4D run 1 as the benchmarks.
- **Tool and output:** reducer `cer_reduce.py` sha256 `ea2565c6…`, scorer `54c21928…`. Output
  `agent-control/ars/checks/wave1/cer_reduce_ex4dgs.json`, sha256
  `d479db1c44bdb0d0d86d84041b1249ec3d447b576a9b294cc997949236a16b54`.

**Rule 1 (causal): PASS.**

| event type | verdict | gap window: CER − base (lower bound, Holm p) | return window (lower bound) | gap gain per layout (bottle / carton / mug) |
|---|---|---|---|---|
| same-pose | PASS | +9.99 dB (+8.80, 1.7e-9) | +0.71 (+0.37) | +13.1 / +7.6 / +9.2 |
| pass-through | PASS | +2.52 dB (+1.77, 9.4e-5) | +0.62 (+0.27) | +4.5 / +0.8 / +2.4 |
| moved return | DESIGN_WITHOUT_POWER (memory after the return = 0 on all 9 runs) | descriptive only | descriptive only | +5.2 / +1.6 / +3.4 |

**Sensitivity, with the original `lrdn0066` base runs:** rule 1 is still PASS. Same-pose +9.89 dB, pass-through
+2.62 dB; return windows +0.78 and +0.62.

**Rule 3 (do-no-harm): DESIGN_WITHOUT_POWER**, because the moved-return scenes fail the preconditions. On the remaining 6
scenes, descriptively:

- **Ordinary region:** −0.09 dB, lower bound −0.22.
- **Whole frame:** −2.41 dB, lower bound −4.73 (−0.83 / −3.63 with the original base runs).
- **What drives the whole-frame number:** single runs whose static background collapses, the Ex4DGS defect recorded on
  2026-09-29. The static-background PSNR falls to 23–34 dB against 39–44 dB in their siblings.
  - On the same-pose and pass-through scenes this hits 6 of 18 CER runs: bottle same-pose run 1, all three mug
    same-pose runs, carton pass-through run 2, mug pass-through run 2.
  - It hits 3 of 18 base runs: carton same-pose run 3 of the replacement set, mug same-pose runs 2 and 3.
  - Not interpreted; not examined visually yet.

**Rule 2 (benchmarks, descriptive): benchmark proof FAILS.**

- **Counts:** CER Pareto-dominates STG on 0 of 9 scenes and Multi4D run 1 on 0 of 9 (3 design-without-power). It
  dominates the Ex4DGS base mean on 6 of 9.
- **Gap window:** STG is above CER on every scene. Multi4D is below CER on the same-pose scenes (bottle 22.9 against
  39.7) and above it on the moved-return scenes.
- **Return window:** CER is below both benchmarks on every scene (for example bottle same-pose 23.9 against STG 33.3
  and Multi4D 34.7). That is inherited from the Ex4DGS base (23.1 there).
- This is the outcome plan §4's "honest risk" named before any run: causal proof passes, benchmark proof fails, so the
  paper stays a benchmark plus a diagnostic.

**Descriptive:** the residual share of the gap content is ≥ 0.10 on 18 of 27 runs, ≥ 0.20 on 2, ≥ 0.30 on 0.

### 2026-10-06: visual check of the CER renders, and the Ex4DGS static-background collapse investigation (step 1)

**Montages** from the existing renders (principal's request; looking material, never a score) are in
`research-wiki/assets/cer-wave1-montages-2026-10-06/`, with the script that made them. They cover the six same-pose and
pass-through scenes:

- every Ex4DGS base and CER run at the mid-gap frame;
- STG, Multi4D, Ex4DGS and Ex4DGS+CER side by side before the gap, mid-gap and at the return, with zoomed crops from the
  scenes' own masks and each tile's window PSNR.

Observations, descriptive:

- **The collapse looks the same as on 2026-09-29:** dark window panes become grey cloud-like blobs, and sometimes the
  poster fades.
- **The return-window shortfall of both Ex4DGS arms against STG and Multi4D is already there before the gap.** On the
  pre-gap target window, Ex4DGS is about 9 dB below STG and Multi4D (mug same-pose 26.1 / 26.8 against 35.5 / 35.0).
  Ex4DGS renders the target texture worse throughout; CER does not change that.
- **On pass-through scenes, the object crossing the gap is smeared by both Ex4DGS arms**, which bounds CER's
  pass-through gain.

**Inventory of the collapse:** 21 of 65 Ex4DGS runs on record have a static-background PSNR more than 3 dB below the
best run on the same scene. Base 9/27, CER 8/27, `lrdn0066` replacements 2/5, dense-start 1/3, node test 1/3. It
happens on many nodes, worst on mug same-pose. Severity ranges from 3 to 20 dB. Script
`agent-control/ars/cer/collapse_inventory.py`.

**Step 1: 30k against 40k** (jobs 59505486/87/685, debug QOS; script `agent-control/ars/cer/collapse_ckpt.py`).

- **Method:** pinned Ex4DGS 7fac649, from scratch copies; held-out camera, frames 0 / 75 / 150 / 299,
  static-background PSNR. The 21 collapsed runs plus 8 healthy controls.
- **Results:**
  - 18 of 21 collapsed runs are healthy at 30,000 (39–44 dB) and collapse between 30,000 and 40,000.
  - 3 are already collapsed at 30,000 (bottle same-pose CER run 1: 24 dB; mug same-pose base run 2 and CER run 1:
    29–30 dB) and drop a further ~2 dB by 40,000.
  - The 8 controls stay at about 43 dB.
- **What this establishes:**
  - CER does not cause the collapse. Before 30,000 the CER runs are plain Ex4DGS, and after 30,000 base and CER
    collapse at similar rates.
  - The usual onset is the final phase, after densification ends, when only optimisation and the per-epoch pruning (at
    34,201 and 39,901) run.

**Steps 2 and 3 of the collapse investigation (jobs 59506005, 59506599; scripts `agent-control/ars/cer/collapse_step2.py`
and `collapse_step3.py`; outputs `cer/collapse/step2.json`, `step3.json`; descriptive, no training).**

- **The collapsed region R:** cam00 at frame 150, static-background pixels whose error grew by more than 0.05 between
  the 30k and 40k renders. Healthy controls use the union of their collapsed siblings' regions on the same scene.
- **Step 2, pruning certain at 30k:**
  - Rows already certain to be pruned at 30k (`min_radii2D < 5` or `xyz_error_min_timestamp < 0`) paint 0.1–3% of R
    in every run, collapsed or control. One run reaches 12%, on a 198-px region.
  - So rows certain to be pruned at 30k are not the cause.
  - Not covered: `min_radii2D` is a running minimum, so rows that cross 5 px after 30k are not visible to this test.
- **Step 3, what paints R at 30k and at 40k (base runs; the pinned code cannot load a CER 40k checkpoint):**
  - **Collapsed runs:** the dynamic-bank share of R falls sharply. Values: 0.85→0.33, 0.64→0.24, 0.63→0.30, 0.51→0.29,
    0.67→0.44, 0.43→0.22, 0.83→0.64. The static rows that remain collapse to a single depth (q10 = q50 = q90, e.g.
    18.413), so one large static Gaussian behind the window now paints most of R.
  - **Healthy controls:** the dynamic share is unchanged: 0.72→0.72, 0.20→0.20, 0.72→0.75, 0.23→0.22.
  - **Depths:** the dynamic rows that remain sit at the same depths at 30k and 40k, so this is not forward drift.
- **Reading, not yet proven:** in the collapsed runs, the dynamic-bank rows that drew the window for the held-out camera
  vanish between 30k and 40k, and a single large static Gaussian shows through (the grey "cloud").
  - The likeliest remover is the per-epoch `prune_small` at 34,201 / 39,901. Its running-minimum radius can fall below
    5 px after 30k, for example because dynamic positions keep a learning rate that is never decayed.
  - The checkpoints cannot show it, because rows have no identity across 30k→40k.
  - Base and CER run the same pruning, consistent with their similar collapse rates.
- **Noted, not changed:** upstream `prune_invisible` computes a `logical_or` for the dynamic mask and discards the
  result (dead code, c_gaussian_model.py:1078-1081).

### 2026-10-06 afternoon: principal's decisions: a one-model STG comparison, and an instrumented collapse probe (recorded before either runs)

**1. One-model STG (descriptive, an addition after the frozen STG result was seen).** It never replaces the frozen
6 × 50-frame benchmark of record, and every report labels it as added after the fact.

- **Design:** one STG model for all 300 frames, trained once on one tuning scene: bottle same-pose (ARS-L1-A0).
- **Unchanged:**
  - STG pinned code 427abfc and its published "full" configuration; `duration` 300 instead of 50;
  - schedules exactly as published: position learning rate decays over 30,000, densification until 9,000;
  - one continuous run to 150,000 iterations, saving models at 25k, 50k, 100k and 150k.
- **Two technical deviations, both lossless:**
  - **`--gtisint8 1`** (STG's own option): the training images stay on the GPU as 8-bit. The default float32 copy of
    20 cameras × 300 frames would need about 99 GB against the A100's 64 GB. The source frames are 8-bit PNGs, so
    nothing is lost.
  - **A separate staging view:** a fresh `colmap_0` whose `sparse/0` is a real directory, so STG's merged-points file
    (`points3D_total300.ply`) is not written into the shared wave-1 staging.
- **Iteration-count rule, frozen here before any checkpoint is scored:**
  - Each checkpoint is rendered for cam00 and scored with the common evaluator on the gap and return windows.
  - The frozen count N* is the smallest checkpoint whose next checkpoint improves neither window by more than 0.2 dB.
    If it is still improving at 150k, N* = 150k and that is stated.
  - N* is then used unchanged on the other scenes.
  - The tuning scene's own numbers are reported as tuning-scene numbers.
- **First step:** a debug-queue check of speed and GPU memory at 300 frames, before the long run is committed.

**2. Instrumented collapse probe (Ex4DGS, mug same-pose, 3 runs).**

- **Code:** pinned 7fac649, unmodified, with the same config and tree as the wave-1 base runs. A runtime wrapper (like
  `m4d_final_render.py`) runs Ex4DGS's own `train.py`.
- **What the wrapper records** on the held-out camera at frame 150:
  - **every 500 iterations from 29,000 to 40,000:** static-background PSNR, and the window region's contribution
    mass by bank;
  - **at every pruning call after 30,000** (`prune_invisible` and `prune_small` separately): the share of that mass
    carried by the rows the call removes, by bank.
- **The window region:** the union of the collapsed regions of the five collapsed mug same-pose runs (base runs 2 and 3,
  CER runs 1–3), frozen as a file before the probe runs.
- **What it may change:** the extra renders consume no random numbers and touch no parameter gradient. They add GPU
  work, so the runs' nondeterministic outcome may differ from an unprobed run; that is acceptable for a diagnostic.
- **Labels:** exploratory diagnostic runs, outside the proof (`runs/ex4dgs_probe/`).

**Submitted, 2026-10-06 afternoon (after the entry above; every job id is in the ledger).**

- **One-model STG.**
  - **Speed check (59512893, debug, stopped at its 30-minute cap by design):**
    - merging the 300-frame starting points and loading took about 11 minutes;
    - training ran at about 12.5 iterations per second;
    - GPU memory peaked at 27.6 GB of 64 GB (steady 26.8 GB) with `--gtisint8 1`; host memory 52 GB.
  - **The STG checkout carries the same two modified files as the frozen wave-1 STG runs** (the project's seed patch in
    `general_utils.py` and a one-line `test.py` change), so the code matches wave 1.
  - **Long run:** 59516063, ARS-L1-A0, 150k iterations, walltime 5:00. Then 59516065 renders and scores the 25k / 50k
    / 100k / 150k checkpoints (`agent-control/ars/cer/score_stg300.py`: STG's own cam00 ground truth equal to the frozen
    frame on every pixel, then the common evaluator with no segmenting).
- **Collapse probe.**
  - **Smoke 1 (59512895):** the wrapper recorded its checks, but no pruning pass fell in 10k–12k.
  - **Smoke 2 (59516073):** FAILED. With the probe started at 3k, frame 150 was outside Ex4DGS's growing training
    window, and Ex4DGS's own position lookup refused it. The probe now records such checks as skipped.
  - **Smoke 3 (59516581):** passed. Both pruning hooks fired at 8,600 with their share of W measured (`prune_invisible`
    104 rows / 0.000; `prune_small` 25,384 rows, 23,018 of them dynamic / 0.011).
  - **Probe runs:** 59517846–48 (`runs/ex4dgs_probe/probe_ARS-L3-A0_rep1..3`).

### 2026-10-06 evening: Multi4D+CER montages and what each Multi4D bank paints (descriptive, looking material, never a score)

**Material** (principal's request), in `research-wiki/assets/cer-wave1-montages-2026-10-06/`, with the scripts that made it:

- `m4d_methods_<scene>.jpg`: STG, Multi4D, Multi4D+CER, Ex4DGS, Ex4DGS+CER on the three same-pose scenes at frames 57, 63, 75,
  87 and 92. The tiles carry the window PSNR / SSIM / LPIPS from the verified scores. Multi-run arms show the run closest
  to the arm's mean gap PSNR.
- `m4d_runs_<scene>.jpg`: all 3 Multi4D base and all 3 Multi4D+CER runs at gap frames 63, 75 and 87, with error maps.
- `m4d_banks_<scene>.jpg`: cam00 renders of the persistent banks alone and of the transient bank alone, at frames 57 and 75.
  - Made by `agent-control/ars/cer/cer_m4d_banks.py`: jobs 59520175, 59520180 and 59520588, debug QOS, CER worktree 00357aa
    read-only (clean before and after). The montage jobs were 59520589 and 59520854.
  - A saved CER model refuses Multi4D's reload by design. The script loads the point clouds from scratch links without
    `cer_state.pt`, re-attaches the saved labels, clock and strength, and arms the gate.
  - **Faithfulness:** the full re-render reproduces each run's saved final cam00 render with 0 differing 8-bit values,
    on all 5 frames of all 18 runs, gap frames included.

**Observations, descriptive:**

- **Base Multi4D never removes the target.** Its persistent banks alone still render the whole object at mid-gap, in every run.
  - The transient bank paints a target-shaped patch of background colour over the object: alone, its alpha is
    0.88–0.99 inside the ghost mask at frames 63, 75 and 87, against 0.09–0.46 on the target at frame 57.
  - The base ghost is whatever this patch fails to cover.
  - The worst case is bottle run 1: the cap and label show through, and its gap PSNR is 22.9 dB against 30.7 for runs 2
    and 3. Bottle's wide Welch interval comes from that run.
- **With CER, the persistent banks alone render background in the gap.** The object is gone, and the ordinary persistent
  rows paint what was behind it, slightly blurred where the object stood. Residual copies are negligible (≤ 0.002).
- **The transient patch persists after CER, at a lower alpha** (0.20–0.64 in the ghost mask), as scattered fragments.
  These fragments are now the main remaining gap error:
  - bottle CER runs 2 and 3 show tilted, streaky smears (32.1 and 32.6 dB);
  - carton and mug CER runs show speckle only (34.3–35.5 dB).
- **Before the gap (frame 57), the persistent banks alone render the target exactly as the full render does,** in every run.
  This is consistent with the census and with the memory share of 0.91–0.95.
- **Gap window, mug, representative runs:** STG 43.2, Ex4DGS+CER 37.7, Multi4D+CER 35.3, Multi4D 31.4, Ex4DGS 28.1 dB.
  **Return window:** the Multi4D arms (34.7–34.8) are at or above STG (33.8), and both Ex4DGS arms are about 9 dB lower
  (25.6–26.4).

**Collapse probe result (59517846–48, mug same-pose, 3 runs, 43–44 min each; exploratory diagnostic).**

- **Pruning calls after 30k:** 30,820 and 36,820 in these runs. The epoch boundaries differ from the 34,201 / 39,901
  quoted earlier.
  - `prune_small` removes 15,115–16,326 dynamic rows at 30,820 (about 35% of the dynamic bank) and about 3,000 at
    36,820.
  - The rows removed carry only 0.2–0.5% of the window region's mass, in every run. `prune_invisible` removes ≤ 17 rows,
    carrying 0.000.
- **Two runs stay healthy:** static background 43.7 dB at 40k, with the dynamic share of W rising slightly (0.50→0.55,
  0.33→0.41).
- **Run 3 collapses:**
  - the static background is 43.96 dB at 30k and 43.77 at 30.5k;
  - then it declines steadily over the remaining iterations, from 43.12 at 31k through 40.65 at 32k and 37.45 at 34k to
    36.24 at 40k;
  - no step coincides with either pruning call (37.74 just before the one at 36,820, and 37.66 just after);
  - the dynamic share of W falls slowly, 0.55→0.47.
- **Reading:**
  - **The pruning hypothesis is refuted:** the collapse is not caused by pruning removing the rows that paint the
    window.
  - **It forms gradually through optimisation after densification stops at 30k:** the existing rows that paint the
    window for the held-out camera degrade step by step, with nothing added to compensate.
  - It occurred in 1 of 3 probed runs here, against 5 of 6 among the earlier runs of this scene. Small numbers; the
    probe's extra GPU work also changes each run's nondeterministic path.
- **Not yet known:** which parameter of those rows drifts (opacity, scale, position, colour). That needs the rows' parameters recorded through
  30k–40k with an alignment across the two pruning calls.

### 2026-10-06 evening: principal's decisions after the substrate debate (recorded before the probe runs)

Debate record: `assets/cer-wave1-montages-2026-10-06/substrate_debate_2026-10-06.md` (Claude Opus 5.5 vs Fable 5.1, two rounds,
consensus: Multi4D the headline substrate, Ex4DGS+CER the replication).

1. **STG (principal):** STG trains six independent 50-frame models. That is a protocol that does not scale to long
   scenes, and the principal judges it not a meaningful comparison for a persistent-memory method. How STG enters the
   paper is to be settled in the writing; the frozen benchmark rule 2 and its result stay on the record unchanged.
2. **Schedule probe on Multi4D+CER: approved.** Exploratory, never evidence, never pooled with the confirmatory runs.
   - **Design:** 3 runs on bottle same-pose (ARS-L1-A0). Branch `cer` at 00357aa, unchanged, with one change: fine
     iterations 20,000 → 30,000, by command line (`--iterations 30000`). Gate still at fine 16,000.
   - **Checked before submission:** `position_lr_max_steps` stays 20,000, so the schedule up to the gate is identical to
     the confirmatory runs. Every densify, prune, lift and opacity-reset call is gated below 16,000 (or 10,000), so the
     CER sentinels cannot fire after 20,000. From 20,000 to 30,000 the position learning rate sits at its floor and no
     row can be added.
   - **Renders:** cam00 at 20,000 and 30,000 within each run (`RENDER_AT="20000 30000"`).
   - **Run ids:** `runs/multi4d_cer/probe30k_ARS-L1-A0_m4dcer_rep{1,2,3}`.
   - **Reading:** at 30,000, the ghost-region PSNR at frames 63/75/87 pooled, for the persistent banks alone
     (`cer_m4d_banks.py`) and for the full render. Also the full-render gap and return windows with the common
     evaluator at 20,000 and 30,000.
   - **Decision rule, frozen here (asymmetric, after the debate):**
     - **≥ 40 dB persistent-alone in 2 of 3 runs:** the gap deficit was training time. Next is a confirmatory longer
       schedule with a matched longer base.
     - **Otherwise:** unresolved between capacity (no rows can be added after 16,000) and the learning-rate floor. This
       reading does NOT establish "capacity". The early-gate port is the route that addresses both.
     - **Anchor:** STG's bottle gap is 44.85 dB, the confirmatory Multi4D+CER bottle gap is 32.9 dB, and Multi4D's
       ordinary static background is 44.4 dB.
3. **Early-gate code: approved for writing and review, not for submission.**
   - **Pipeline:** Opus orchestrates and writes the plan; Codex (gpt-6.1-sol) reviews the plan to consensus; a Sonnet 5.5
     agent writes the code; Codex reviews the code (scientific, engineering, tensor shapes, CUDA), iterated until the
     orchestrator is satisfied.
   - **Isolation:** a separate branch and worktree, never touching `cer` or `repo/multi4d-cer`. A separate torch
     extensions directory, so no running job's compiled extension is ever rebuilt.
   - **Submission:** the principal reviews the code and approves any training submission.

**One-model STG on the tuning scene (bottle same-pose): checkpoint scores and the frozen iteration count.**

- **Runs:** training 59516063 completed (3:17, exit 0). Evaluation 59536343: the first evaluation (59516065) failed
  because `lpips` is missing in the STG environment, and the scorer now runs in the ARS scoring environment; a mistaken
  resubmission (59536312) was cancelled while pending.
- **Scores:** `agent-control/ars/scores/stg300/ARS-L1-A0_stg300_<N>.json`. STG's own cam00 ground truth equals the
  frozen frames on every pixel, at every checkpoint.

| iteration | gap | return | pre-gap target | whole frame |
|---|---|---|---|---|
| 25,000 | 25.05 | 28.74 | 31.34 | 39.11 |
| 50,000 | 26.25 | 33.25 | 33.63 | 40.65 |
| 100,000 | 29.65 | 33.88 | 35.03 | 41.67 |
| 150,000 | 31.01 | 34.81 | 35.01 | 41.95 |

- **Rule applied as frozen:**
  - every step improves at least one window by more than 0.2 dB (+1.20 / +4.51, +3.40 / +0.63, +1.36 / +0.93);
  - so **N* = 150,000**, with both windows **still improving at the last checkpoint**;
  - this is the count for the other scenes unless the principal amends it before they run.
- **Descriptive comparison on this scene (tuning-scene numbers), gap / return:**
  - one-model STG 31.0 / 34.8;
  - frozen 6 × 50 STG 44.9 / 33.3;
  - Multi4D+CER 32.9 / 34.7;
  - Ex4DGS+CER 39.7 / 23.9.
- **Reading:** the 50-frame segmentation accounts for most of STG's gap advantage on this scene; as one model it is
  about 14 dB lower on the gap and 1.5 dB higher on the return.

**Principal's decision (2026-10-06 evening):** run the other eight wave-1 scenes as one-model STG at 150,000 iterations,
as frozen. The bottle run is not extended. Each run saves only the 150k model, which is rendered and scored as on the
tuning scene. The result stays descriptive, beside the frozen 6 × 50 STG benchmark of record.

### 2026-10-06 late evening: early-gate CER on Multi4D written, reviewed and tested; nothing submitted for training

The full record is in [cer-multi4d-early-gate-plan-2026-10-06](cer-multi4d-early-gate-plan-2026-10-06.md): plan rev 4
(sections 1–9) and the implementation record (section 10).

- **Plan.** Three Codex plan rounds reached consensus. Rev 1 had proposed periodic relabelling; it was dropped for
  memory labels chosen once at begin and inherited.
- **Code.** Branch `cer-early`, code unchanged since `96e92ee`, tests at `7b30ef8`. Five Codex code-review rounds; every
  finding was fixed except the two escalated below.
- **GPU tests (not evidence):** E0–E9 all PASS on Leonardo. Two integration runs on the shortened schedule exited 0.
- **Escalated:** the inherited hybrid `clamped` buffer defect, which corrupts transient clamp flags in every Multi4D run,
  and the unwritten `max_contrib` buffer.
- **Forecast F1** is answered structurally: under the real schedule the transient bank has 1 row at fine 3,000.
- **Waiting for the principal:** code review, the frozen confirmatory design (9 runs, ≈ 27 GPU-h), the scorer method, and
  "submit".

### 2026-10-06 night: schedule probe read (exploratory, never evidence)

**Runs.**

- Bottle same-pose, Multi4D+CER `cer@00357aa`, fine iterations 30,000 by command line, gate at 16,000.
- Training jobs 59534383–85, 3:28–3:35 each; readouts 59536782/87/90.
- The 30k re-renders reproduce each run's saved render exactly.
- Outputs: `agent-control/ars/cer/probe30k/rep*/readout.json`.

**Window PSNR (dB) from the common evaluator, 20,000 → 30,000 within each run:**

| run | gap | return | whole frame |
|---|---|---|---|
| 1 | 37.52 → 40.93 | 34.75 → 34.84 | 43.79 → 43.91 |
| 2 | 35.25 → 40.32 | 34.70 → 34.93 | 43.61 → 43.83 |
| 3 | 37.11 → 40.90 | 34.87 → 34.97 | 43.64 → 43.78 |

**Ghost region, frames 63/75/87 pooled, at 30,000:**

| run | full render | persistent banks alone |
|---|---|---|
| 1 | 40.85 | 38.25 |
| 2 | 40.28 | 38.83 |
| 3 | 40.89 | 39.72 |

**The frozen rule's reading:** persistent-alone reaches 40 dB in 0 of 3 runs. That means "unresolved between capacity (no
rows after 16,000) and the learning-rate floor"; it does not establish capacity. The early-gate code (written today, not
submitted) is the route that addresses both.

**Descriptive:**

- **Training longer helps the gap within every run** by +3.4 / +5.1 / +3.8 dB, with the return and whole frame unchanged
  or slightly better.
- **The transient bank now helps.** At 30,000 the full render is above persistent-alone, the reverse of carton at 20,000.
- **The 30k numbers break the equal-budget rule** (spec §2.2). Any confirmatory use needs a matched 30k base.
- **Not explained:** the probe runs' 20,000-iteration gap (37.5 / 35.3 / 37.1, mean 36.6) is above the confirmatory bottle
  CER runs' (34.1 / 32.1 / 32.6, mean 32.9).
  - The code, config and schedule are identical up to 20,000, except that iteration 20,000 takes an optimizer step in
    the probe and not in the confirmatory runs.
  - Possible explanations, untested: run-to-run variance on this scene (the bottle base runs ranged 22.9–30.7), and the
    uninitialised `max_contrib` buffer.
  - The within-run 20k → 30k comparison is the valid reading.

### 2026-10-07: principal's decisions on the early gate (recorded before any early-gate confirmatory run is submitted)

1. **The confirmatory design of [cer-multi4d-early-gate-plan-2026-10-06](cer-multi4d-early-gate-plan-2026-10-06.md)
   section 6 is approved and frozen by this entry.**
   - **Runs:** the early arm (`cer-early@7b30ef8`, code unchanged since 96e92ee; `cer_begin_iter = 3000`; published
     schedule; freeze at 16,000; final iteration 20,000), 3 runs on each same-pose scene (bottle, carton, mug).
   - **Comparison:** against the nine wave-1 Multi4D base runs 1–3.
   - **Rule 1 as in spec §14.4:**
     - gap improvement Holm-significant over 2 endpoints;
     - return window one-sided 95% lower bound > −0.25 dB;
     - mean gap gain positive in every layout.
   - **Rule 3:** do-no-harm on the whole frame and the ordinary region, lower bound > −0.2 dB, pooled.
   - **Rule 4 (preconditions, from `cer_gates.json`):**
     - memory share ≥ 0.50 before the gap and after the return;
     - exact zeros;
     - `frozen == true`;
     - begin 3,000 with 0 lifts before it.
   - **Descriptive:**
     - against the 16k arm (memory gate on Multi4D, 2026-10-02) and the 30k probe;
     - the transient share;
     - drift;
     - persistent-alone renders.
   - **Statistical reading:** as recorded on 2026-10-01 evening.
   - **Cost:** ≈ 27 GPU-h.
2. **Submit the nine runs.**
3. **The `cer-early` commits may be pushed.** The substrate repository's only remote is the Leonardo repository, where
   `cer-early` already stands at 7b30ef8 (fast-forwarded by bundle during testing).

**Prepared for the submission:**

- **Scorer:** `ars_score.py` method `multi4d_cer_early`, pinned to 7b30ef8 and begin 3,000. It requires:
  - a clean `cer-early` tree, unchanged `cer` and pinned trees;
  - begin 3,000 as passed and as resolved in `cfg_args`;
  - exactly 40 fine evaluations;
  - an early begin with 0 lifts, `frozen` true, and the drift report present.

  The diff is `D:/adags-cer/agent/ars_score_m4dearly.diff`; the previous version is kept as `ars_score.pre_m4dearly.py`.
- **Submission script:** `submit_cer_multi4d_early.sh`, fail-closed preflight, walltime 3:50.

**Submitted, 2026-10-07 00:39 (after the entry above).** Every job ID is in `agent-control/ars/ledger.txt` and `jids/`.

- **Runs:** early-gate CER on Multi4D, confirmatory, 9 runs, 59556516–59556525
  (`runs/multi4d_cer_early/ars_w1_<scene>_m4dearly_rep<r>`).
- **Settings:** commit `cer-early@7b30ef8`, begin 3,000, published schedule, walltime 3:50, `lrdn0066` excluded. The
  launcher is pinned to the commit and to the clean trees at job start.
- **Queue:** all pending (Priority) at submission.
- **Preflight passed:**
  - the early, `cer` and pinned trees at their pins and clean;
  - the scorer pinned to the same commit and begin;
  - the extensions built; the run IDs free.
- **Before submission:** the scorer method was installed on Leonardo (sha256 `f88f7687…`). As a negative test it refused
  the shortened integration run `e9_smoke96` and wrote no score.
- **The push:** the substrate repository's only remote is the Leonardo repository, where `cer-early` stands at
  7b30ef8, so nothing was left to push.

### 2026-10-07 early morning: reducer extended to the early gate (installed on Leonardo only as a candidate)

- **Change.** `cer_reduce.py` gained `--substrate multi4d_early`: the 9 early-arm runs against the 9 Multi4D base runs,
  with rules 1, 3 and 4 exactly as on Multi4D.
  - Rule 4 adds the frozen early preconditions: gates mode early, begin_iter 3000, frozen true; begin mode early;
    lifts_before_begin 0 (non-bool ints).
  - Descriptive only: the early arm against the 16k arm (loaded apart from the confirmatory inputs), and the drift
    shares. Neither part ever refuses or decides anything.
- **Hardened during review, all substrates.**
  - Confirmatory PSNRs and gate fields are type- and range-checked: no booleans; shares in [0, 1] ± 1e-6; masses ≥ 0;
    PSNR 0–1000 dB.
  - A design-without-power subset contrast and the Ex4DGS sensitivity re-run report "unavailable" instead of blocking.
  - Output is strict JSON. Only a verdict's own evidence must be finite; other non-finite values are replaced by None
    and their paths listed.
- **Tests:** 17 pass locally (14 originally; the test runner now reseeds per test).
- **Compatibility.** The candidate (sha256 `3d6b8a95…`) reproduces both readings of record,
  `checks/wave1/cer_reduce_multi4d.json` and `cer_reduce_ex4dgs.json`, with 0 differences on the real scores. It is at
  `agent-control/ars/cer/cer_reduce_m4dearly_candidate.py`.
- **Review.** Codex rounds 1–7 each returned MATERIAL DEFECT, and every finding was fixed. Records are in
  `assets/cer-early-gate-2026-10-06/reducer_review_*`.
- **Open.** Round 8 could not run: the Codex usage limit resets at 05:45. The candidate replaces `cer_reduce.py` on
  Leonardo only after that review. The previous reducer stays in place, and local masters are kept as
  `*.pre_m4dearly.py`.

**One-model STG, all nine scenes at 150k (descriptive; evaluations 59545509–31 all COMPLETED; scores in
`agent-control/ars/scores/stg300/`).** Gap / return windows, dB:

| scene | one-model STG | STG 6 × 50 (frozen) | Ex4DGS+CER (mean of 3) | Multi4D run 1 |
|---|---|---|---|---|
| bottle same-pose (tuning) | 31.01 / 34.81 | 44.85 / 33.26 | 39.67 / 23.92 | 22.92 / 34.68 |
| carton same-pose | 34.16 / 37.12 | 47.61 / 36.92 | 35.86 / 33.15 | 30.29 / 36.81 |
| mug same-pose | 34.91 / 35.44 | 43.15 / 33.77 | 37.59 / 26.49 | 31.67 / 34.90 |
| bottle moved | 47.10 / 34.26 | 46.56 / 33.07 | 41.02 / 19.91 | 43.09 / 30.72 |
| carton moved | 50.33 / 37.50 | 47.04 / 37.88 | 37.56 / 24.09 | 40.77 / 35.74 |
| mug moved | 45.85 / 34.84 | 43.01 / 34.12 | 39.60 / 20.35 | 41.73 / 32.71 |
| bottle pass-through | 23.86 / 35.84 | 33.49 / 33.13 | 27.47 / 24.97 | 25.84 / 34.60 |
| carton pass-through | 21.08 / 37.40 | 26.99 / 36.77 | 20.68 / 32.58 | 22.06 / 37.43 |
| mug pass-through | 25.59 / 36.09 | 29.52 / 33.47 | 25.10 / 26.99 | 25.60 / 34.51 |

**Descriptive reading:**

- **Same-pose:** as one model, STG's gap falls 8–14 dB, below Ex4DGS+CER on all three scenes.
- **Pass-through:** it falls 4–10 dB, below Ex4DGS+CER on bottle, and about level on carton and mug.
- **Moved return:** it rises by 0.5–3 dB.
- **Return window:** one-model STG is at or above the 6 × 50 model on 7 of 9 scenes, and above Ex4DGS+CER everywhere.
- **Not computed yet:** the per-scene Pareto counts against Ex4DGS+CER and Multi4D+CER.

### 2026-10-07: principal's decisions: Multi4D+CER becomes the primary substrate for the comparisons; it is extended to the moved-return and pass-through scenes, and one-model STG gets replicates (recorded before any of these runs)

1. **Multi4D+CER on the six moved-return and pass-through scenes, 3 runs each (18 runs).**
   - **Unchanged setup:** commit 00357aa, the published schedule, CER from fine 16,000, final iteration 20,000, against
     the existing wave-1 Multi4D base runs 1–3 of each scene.
   - **The same-pose confirmatory reading of 2026-10-02 stands unchanged**, with its Holm family of 2. It is not
     re-read with the new scenes.
   - **The new scenes get their own reading, frozen here:**
     - per event type (moved return, pass-through): the same model (arm + layout fixed effects, 9 v 9) and the same
       three clauses as plan §4 rule 1;
     - Holm over these 4 new endpoints (2 event types × gap / return);
     - preconditions as in Multi4D spec §14.2, applied per scene as in Ex4DGS 13.4 (a moved-return scene may be
       design-without-power if the memory rows do not follow the target);
     - do-no-harm over the scenes that meet the preconditions, reported descriptively only, since the full-design
       certificate was the same-pose one.
2. **One-model STG replicates:** two more runs on each same-pose scene (6 runs). Same frozen 150k, same seed 0, so they
   differ only by GPU nondeterminism, like every other arm's repeated trainings. Run ids `ars_w1_<scene>_stg300_rep2/3`.
   The 150k models are scored as before. They stay descriptive.

**Submitted, 2026-10-07 ~10:06 (after the entry above; every job id is in the ledger).**

- **Multi4D+CER moved-return and pass-through runs:** 59590875–59590931 (18 runs; `submit_cer_multi4d.sh` with
  `SCENES` for the six scenes; the preflight passed; walltime 3:40).
- **One-model STG replicates:** 59590932/34/36/38/40/42, each followed by its scoring job (…33/35/37/39/41/43).
- **Scoring workers:** A18 and B18 (59590945/46), with the worker list extended by the 18 new Multi4D+CER rows. The
  previous versions are kept as `*.pre_m4dext.*`, and a diff confirms only those rows and the `SCENES` override changed.
- **Observed, not mine:** another session is running and scoring Multi4D "early-gate" runs (method
  `multi4d_cer_early`) from its own worktree, `repo/multi4d-cer-early` (branch `cer-early`, 7b30ef8). It does not touch
  `repo/multi4d-cer` (still 00357aa, clean), so this work and that work do not interfere. Its jobs were left alone.

### 2026-10-07 morning: early-gate reading (confirmatory, design frozen 2026-10-07)

**Training.** All 9 runs completed between 01:12 and 05:05, each 2:34–2:41 with exit 0 and trees unchanged.
Preconditions, read from the gate files before any score was opened:

- memory share 0.905–0.956 before the gap and after the return;
- memory mass in the gap and residual mass outside it exactly 0;
- begin 3,000 with 0 lifts before it; frozen true;
- **transient share of the gap region 0.000–0.042**, against 0.18–0.32 in the 16k arm;
- gap-only copies 0.06–0.33 of the gap region.

**Scoring.**

- The first 9 jobs (59590448–59590804) failed before scoring: the folder `scores/wave1/multi4d_cer_early` did not exist,
  so the claim could not be taken. This was recorded in the ledger.
- The folder was created and the runs rescored as `_r2`; all 9 were scored.

**Reducer.**

- Codex round 8 returned SOUND WITH QUALIFICATIONS. The reducer was then installed: `agent-control/ars/cer_reduce.py`
  sha256 `3d6b8a95…`, with the previous one kept as `cer_reduce.pre_m4dearly.py`. Its 17 tests pass on Leonardo.
- **Retained compatibility report** (`agent-control/ars/cer/reduce_compat_2026-10-07/report.json`, copied to
  `assets/cer-early-gate-2026-10-06/reducer_compat_report_2026-10-07.json`, with its script): the installed command line
  reproduces both readings of record with 0 differences and identical input hashes.

**Output:** `checks/wave1/cer_reduce_multi4d_early.json`, sha256 `dffc2001179182e594bbf5ced1e51e870e6b68bbaa3ffd4a156fedb985f22bb6`.

**Rule 1 (causal, early gate against base Multi4D): FAIL, on clause (b) only.**

| clause | value | met |
|---|---|---|
| (a) gap superiority | +9.24 dB (se 1.14, lower bound +7.24, Holm p < 1e-6) | yes |
| (b) return non-inferiority | −0.10 dB, one-sided 95% lower bound −0.32 against the frozen −0.25 | **no** |
| (c) gap gain positive in every layout | bottle +13.4, carton +6.7, mug +7.6 dB | yes |

**Rule 3 (do-no-harm): PASS.** Whole frame +0.06 (lower bound −0.002); ordinary region +0.02 (lower bound −0.13).

**Rule 4:** OK on all three scenes.

**Per scene, window PSNR (dB), mean of 3 runs:**

| scene | gap: early / base / 16k arm | return: early / base / 16k arm |
|---|---|---|
| bottle | 41.54 / 28.14 / 32.93 | 34.57 / 34.69 / 34.73 |
| carton | 37.17 / 30.46 / 34.45 | 36.98 / 37.08 / 37.24 |
| mug | 38.93 / 31.33 / 35.30 | 34.66 / 34.74 / 34.52 |

**Descriptive, never a verdict:**

- **Early minus the 16k arm,** pooled: gap +4.99 dB (lower bound +3.71), return −0.09 (lower bound −0.31), whole frame
  +0.03.
- **Run-to-run spread is small:** the early arm's runs lie within 0.6 dB of each other in the gap on every scene.
- **STG's gap**, for reference, is 44.85 / 47.61 / 43.15 dB.
- **The return-clause miss,** stated without reinterpreting the rule: the point estimate (−0.10 dB) lies inside
  Multi4D's 0.34 dB fixed-seed floor, but n = 3 per layout does not bound it above −0.25. The 16k arm's return, by the
  same rule on the same base runs, was −0.01 dB with a lower bound of −0.18.

### 2026-10-16: why the early gate's return window reads −0.10 dB (descriptive diagnostics, no training)

Asked by the principal: identify the cause, model first. Jobs 60724773 and 60724911 (debug QOS), renders and gate files only.
Outputs are in `agent-control/ars/cer/return_diag/`; the scripts are `target_edge_interior.py` and `interior_by_bank.py`.

1. **It is not a return defect; the return is the best of the three arms.**
   - The early arm scores the same at every time the object is present: pre-gap 30–57, return 92–99, 100–109 and
     230–259 (bottle 34.57 throughout).
   - Its offset against base is the same before the gap (−0.10 / −0.18 / −0.21) as long after it (−0.18 / −0.20 / −0.24).
   - At the first return frames 90–91, which lie outside the scored return window, the early arm is +4.9 / +5.4 / +6.1 dB
     above base. It has no ramp; base starts at 26–29 dB at frame 90, the 16k arm at 32–37.
2. **The deficit is in the object's interior, not its edges.**
   - Interior (mask eroded 4 px), early minus base: −0.19 / −0.29 / −0.15 dB.
   - Edge ring: −0.01 / −0.01 / −0.16 dB. The mug alone loses at the edge too.
   - The colour bias is about 0.001 and negligible.
3. **Two sources, by bank** (bank-alone renders at pre-gap frames 40 and 57; all 18 re-renders reproduce the saved
   renders exactly).
   - **(a) Transient detail is absent in the early arm.**
     - In base runs the transient bank covers the object interior with alpha 0.05–0.10 (bottle), 0.12–0.23 (carton)
       and 0.07–0.52 (mug).
     - It adds detail in some runs: carton +0.31; mug +0.60 and +0.27 dB. On the bottle it hurts in all three runs
       (−1.68 / −0.11 / −0.28).
     - In the early arm it covers 0.000–0.053 and adds nothing.
     - This follows the foreground memory rows at fine 3,000: bottle 0–20, carton 229–288, mug 72–129. The transient
       bank is seeded by lifting foreground rows, and the early gate forbids lifting labelled rows (D4). That is
       consistent with the lift exclusion cutting off the transient detail path. **Not proven:** the lift lineage is not
       tracked.
   - **(b) The persistent banks themselves are slightly worse in the early arm** on bottle and carton: persistent-alone
     interior −0.28 / −0.16 dB against base's persistent banks. On the mug they are better (+0.20).
     - Cause unestablished. Candidates are the gated rows' diluted densification statistics (D7) and fewer gradient
       views for memory rows over 17,000 iterations.
     - The 16k arm's object quality was about equal to base's (pre-gap +0.08 / +0.09 / −0.36).


### 2026-10-16 evening: new lane, failure-mode discovery (plan only; nothing trained)

- **Why.** The principal asked for a plan to find a specific failure of the SOTA methods in a deliberately chosen
  setting (few cameras or one moving camera, where hidden surfaces must be remembered), with stop rules and a
  fight-for-it step, instead of choosing directions by elimination.
- **Record:** [failure-mode-discovery-plan-2026-10-16](failure-mode-discovery-plan-2026-10-16.md) (rev 2, after a
  Codex gpt-6.1-sol review that returned MATERIAL DEFECT on rev 1; all points folded in).
- **Measured today (CPU, exact, static background of the suite):** with 19-21 training cameras the "remembered"
  class is empty or near-empty, at 6 cameras about 0.002-0.004 %, at 3 cameras 0.05-0.10 % on the camera line and
  about 1 % on a probe arc; most remembered pixels are seen again later. The first frame after the object returns
  is 3-11 dB below the steady return in STG, Multi4D and Ex4DGS (45 local score files), but the fixture switches the
  object's existence on in one frame, so its real-footage relevance is open.
- **Dates:** hypothesis list closes 23 Oct; evidence checkpoint 24 Oct; hard stop 28 Oct ("no admissible CVPR claim"
  is an allowed outcome). About 85 GPU-h, all waiting for the principal's approval. The early-gate memory-gate lane
  is not touched.

### 2026-10-16 evening: the two sources of the early arm's object deficit, tested (exploratory; no training of an arm)

Brief from the principal (2026-10-16): (A) plan, review, implement and test a fix for the missing transient detail (memory rows
lift donors again, their transient copies gated as memory); prepare, not submit, a comparison. (B) Measure whether memory rows
are densified less in the early arm before any densification change. The record is
[cer-multi4d-early-tr-plan-2026-10-16](cer-multi4d-early-tr-plan-2026-10-16.md) (plan rev 5, sections 1–13; implementation,
tests and premise check, section 14). Reviews: `assets/cer-early-tr-2026-10-16/`. Every job id below is in the ledger.

**Task A, the fix (transient copies of memory rows gated as memory).**

- **Plan:** five revisions; Codex rounds 1–4 each MATERIAL DEFECT (6, 3, 3, 2 findings), all fixed; round 5 (21:31) PLAN READY.
  The implementation went ahead during the Codex usage-limit wait (17:00–21:28), after round 4 had found nothing in task A.
  Code review round 1 (same call): READY WITH FIXES (the new launcher's exit-code order, F9's tree check; both fixed).
- **Code:** branch `cer-early-tr`, code at `5afa119` (tests to `b5254ee`), flag `cer_transient_gate` (0 = the early arm, shown
  bit-identical by F0). GPU tests: F0–F4, F6–F8 PASS; F9 integration PASS; F5/F5c gate clauses PASS (exact zeros for every
  witness in all nine parameter groups) with positive-control failures: one part explained by the escalated `clamped` buffer
  (they vanish with the static set out of view), and 10 rows with entirely zero training-build gradient, cause not
  established and not isolated from the code by a paired control (attribution unresolved).
- **The premise is weakened (Codex: QUALIFIED).** On the early trajectory the lift exclusion met few labelled candidates: 0–45
  candidate occurrences per run over 139 selections (bottle 0/0/0, carton 45/12/16, mug 1/0/6). This is low exposure, not a
  bound on the fix's effect (a restored pool also changes the random draw and later training). In base runs the transient rows
  that paint the object interior before the gap are mostly centred in time inside the gap (mug 0.83–1.00, bottle 0.52–0.99 at
  frame 57, carton 0.10–1.00 of their mass), and 0.43–1.00 of that mass is from rows with some contribution in the gap region:
  consistent with time tails of gap rows, not shown to be an occluder or to cause the PSNR difference. The 2026-10-16 entry's
  "most likely mechanism" (the lift exclusion) is unestablished.
- **Recommendation:** defer the comparison (≈ 26 GPU-h), as a prioritisation. Nothing was submitted for training.

**Task B, densification of memory rows (bottle same-pose, one run each, exploratory diagnostics).**

- **Probe:** `agent-control/ars/densify_probe.py` (runpy around `cer-early@7b30ef8` `train.py`), its test (17 checks PASS, jobs
  60728104/60728741/60728898/60729394), reader `densify_read.py`. Runs: early arm 60729395 (2:49), shadow base 60729396 (2:32;
  CER off, the memory rule applied at fine 3,000 without arming the gate). Output
  `agent-control/ars/cer/densify_probe/densify_read_bottle_r1.json`.
- **Background memory rows (3,349 of the early arm's 3,388 memory rows at 3,000), steps 3,100–15,900, early / shadow:** mean average gradient 1.011×
  (median 1.026×); clone selections 5 / 5; split 0 / 0; rows at 16,000 2,652 / 2,667 (0.994×); visibility count per window
  99.6 / 99.6. Increments that carry no gradient for a gated row (every view of the batch that saw the row was a gap view) are
  1.8% of the count, not ≈ 10%; deleting them raises the average by 1.9% and changes no selection.
- **Other background rows:** gradient 1.02×; clones 213 / 208; splits 668 / 850 (0.79×).
- **Foreground memory rows:** 39 (early) / 93 (shadow) at 3,000. In the early arm they appear in only 2 densification windows
  (3,100 and 3,200), with mean average gradient about 1.1e-7, and are gone by 3,300; the shadow base has them in all 69
  windows (mean 1.4e-4), splits 54 of them and keeps 18 to 16,000. A small cohort (39 of 3,388 memory rows in the early arm).
- **Reading (corrected after Codex round 5):** background memory rows are densified about as often in the early arm as in the
  shadow base (similar gradient, clone 5/5, split 0/0, rows 0.99×). The early arm's foreground memory rows vanish within
  300 iterations of begin (2 windows against 69, so the gradient ratio compares unequal windows); why they vanish (pruning
  after near-zero gradient, or something else) is not shown. The deletion
  sensitivity removes only all-zero increments; the gradient a gated row misses in mixed batches (one gap view, one non-gap
  view; gradients summed before the norm, `train.py:855`) is not measured. So the diluted-statistics candidate is small for
  background memory rows; the early loss of the foreground memory rows and the mixed-batch loss are open, and the image impact of either is not
  measured. No densification change is proposed yet. Unverified: one run per arm, one scene; the two runs already differ at
  fine 3,000 by GPU nondeterminism.

**Cost.** Task B 5.9 GPU-h (probe tests 0.5, runs 5.4). Task A tests 3.3 GPU-h (F9 1.1 on lprod). Premise check 0.05 GPU-h.

**For the principal.** (1) Whether to drop the fix (recommended) or still run the comparison. (2) An observation with unresolved attribution: some transient rows get no training-build gradient although they render,
and transient gradients differ between fresh processes. (3) Where the object deficit comes from remains open: the transient part is consistent with gap rows' time tails; for the
persistent part (bottle −0.28, carton −0.16, mug +0.20 dB) the candidates left are the early loss of the foreground memory
rows and the gradient missed in mixed batches; neither is measured against image quality.

- **2026-10-16 late, revision 4 of that plan (its section 12, principal's decision):** the few-camera claim is about
  methods published for few cameras (4C4D at its published cameras 1, 10, 13, 20; GC-4DGS; Multi4D's own low-view
  setting), with Multi4D's dense setting, Ex4DGS and STG as labelled reference rows; first GPU block about 66 GPU-h,
  not yet approved or submitted.

- **2026-10-17 ~00:00, failure-mode discovery lane:** the principal approved revision 4's first GPU block (about 66 GPU-h),
  with environments and trees on scratch, after a short Codex check of its sections 12-13 (record: that page, section 14).

- 2026-10-19: failure-mode discovery, sections 24-25: real footage confirms Multi4D monocular ghosting (3/3); the
  kill-argument FAILs the broad claim; Spacetime Gaussians leaves no real-footage ghost at 20 cameras, so the claim
  narrows to monocular/low-view static-background methods, real-data-first (`failure-mode-discovery-plan-2026-10-16.md`
  section 25).
