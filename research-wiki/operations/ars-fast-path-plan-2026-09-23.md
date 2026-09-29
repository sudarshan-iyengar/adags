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
