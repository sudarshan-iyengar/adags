# Absence–return synthetic suite + parallel baseline lanes — plan (2026-09-23, revision 2)

> **Superseded as the ACTIVE plan by [revision 3, the fast path](ars-fast-path-plan-2026-09-23.md)
> (2026-09-23, principal).** ARS is now the primary evidence and the real fixtures
> are a secondary same-pose realism check. This page is retained as the **ideal
> design** and as the **spec** for every mechanism the fast path keeps. Its scope,
> tiers, compute and schedule are replaced there.

**Status: PLAN. Nothing has trained; nothing is committed.** This expands
[handoff rev 3](neurips-workshop-paper-handoff-2026-09-23.md) §5, row "Sep 23–25":
*extend the synthetic renderer with moved return and something moving through the
gap, while Ex4DGS, Multi4D and STG run on the real fixtures and the Panoptic check
happens.* Every number below cites a record, is marked **measured 2026-09-23**, or
is marked **UNVERIFIED**.

**Revision 2 folds in an independent adversarial design review** (gpt-5.6-sol,
xhigh). That review ran on a pinned snapshot of revision 1 and returned **MATERIAL
DEFECT**: 2 fatal, 9 major and 4 minor findings. The artefacts are in
`refine-logs/ars-review-2026-09-23/`:

- the revision-1 snapshot, `plan_snapshot_r1.md`, sha256 `ed8beec7…`;
- the review, `sol_review_r2.md`;
- the raw Leonardo probe output, `leonardo_probe_2026-09-23.txt`.

A round-1 review had audited only the review request, not the design. The
disposition of every finding is in §15.

The suite is called **ARS** (absence–return suite). Scene IDs read
`ARS-<layout>-<condition>`, e.g. `ARS-L1-R2`.

---

## 0. Deliverables and their completion tests

| # | deliverable | done when |
|---|---|---|
| D1 | ARS synthetic suite | the spec and per-condition trajectory JSONs are frozen by hash, and every scene has a sha256 manifest; the preconditions in §6 pass per their applicability matrix, including a non-empty denominator for every primary endpoint; montages of every scored region have been viewed |
| D2 | common evaluator | one script scores any method's cam00 renders on real and synthetic windows, with every convention frozen (§8.3), and its golden tests pass |
| D3 | real-fixture baselines | both route golden checks pass (§8.1); Ex4DGS ×3 and Multi4D ×1 have run on all three fixtures; the Ex4DGS event-window replicate spread is recorded **before any method output is opened** |
| D4 | STG on crb | seeds 0 and 1 (≈ 6 GPU-h each, measured §8.2) are rendered and collected into the flame/sear layout |
| D5 | Panoptic census | a candidate list, contact sheets and a go/kill decision are recorded by Sep 25 EOD |

## 1. Why the current renderer cannot do this, and the design rules

**The existing LRV renderer** is `scripts/build_synthetic_reveal_scene.py`.

- **Scene:** one sphere event object.
- **Capture:** a 20-camera surround ring at 400×300, 60 frames at 6 fps.
- **Rendering:** no shadows, no antialiasing.
- **Dynamics:** the world is static apart from the event.
- **Output:** EL-GS Blender layout only.
- **CLI:** seven options (`:334-358`). The hazards page's "six" predates `--allow-short-return`.
- **Status:** labelled exploratory in its own header (`:3-4`).

Its recorded hazards ([lrv3-fixture-hazards](lrv3-fixture-hazards-2026-08-23.md)):

- The return is identical in every respect, so any pose payload is vacuous.
- Scene constants are frozen.
- `inf` leaks into per-frame PSNR means.

**No baseline can run it with its published hyperparameters.**

- Ex4DGS and Multi4D train on the 300-frame, 30-fps N3V protocol with cam00 held out.
- STG trains six independent 50-frame models over the same 300-frame sequence.

**Design rules:**

1. **Match the real N3V capture protocol.** ARS renders with the real camera poses
   and intrinsics from the local `poses_bounds.npy`:
   - crb: 20 cameras;
   - flame and sear: 21 each;
   - measured: native 2028×2704, f ≈ 1462.

   Other protocol settings:
   - 300 frames at 30 fps;
   - 1352×1014, the raster all three baselines train at;
   - cam00 held out.

   Every baseline keeps its **published hyperparameters**, and every
   compatibility adaptation is enumerated:
   - the 1352 route (§8.1);
   - STG's seed-exposure patch and `sk_ssim` fix (STG pin `:248-253`), so STG is
     "protocol-matched with compatibility and seed-exposure patches", not "unmodified";
   - per-method symlink adapters (§4).
2. **Exact analytic ground truth.** Scored regions come from the renderer's own
   buffers, not from any segmentation model:
   - the actual scene's front-most identity;
   - a shadow-caster buffer;
   - target-alone projections at a declared pose.

   Full counterfactual *worlds* are used only where they are physically valid
   (§5). Revision 1's image-difference twins were invalid in G1 and X1, where an
   occupant or carrier would interpenetrate the kept target (review F3).
3. **One factor at a time from an anchor.** The anchor is the synthetic twin of the
   real fixtures: the same absent frames [60,89] and the same window arithmetic.
   The anchor is replicated across three layouts. Extensions run on one layout.
4. **The suite must be able to falsify the method.** It includes:
   - an occlusion control where a presence mechanism must be inert (O1);
   - a permanent departure where memory must not invent a return (N1);
   - a substitution where the old appearance must not come back (N2, deferred).
5. **Physical realism where it is cheap and bears on absence:**
   - hard cast shadows (absence removes the shadow too);
   - 2×2 supersampling;
   - ordinary dynamics in every frame;
   - a carrier for physical exit;
   - textures fixed to the object;
   - every trajectory collision-checked.
6. **Replication by independent layouts:** three rigs and three target shapes.
   ARS stays secondary evidence; the real fixtures stay primary.
7. **Frozen before any run.** The spec hash, trajectory hashes and manifest hashes
   are fixed first.
   - Preconditions concern the setup, never the score, and have an explicit
     applicability matrix.
   - An independent verifier checks the renderer.
   - Montages are viewed before any table.
   - Every frozen condition is reported whatever its outcome.
8. **No leakage.** The optimizer must never consume a cam00 frame (an instrumented
   assertion, §6). Identity buffers, masks, windows and poses live outside every
   method tree.

## 2. The event space considered

| axis | included | excluded (reason → where it goes) |
|---|---|---|
| **A. What leaves** | static rigid target in three shapes (label bottle, carton, mug); moving rigid target (M1) | non-rigid or articulated: needs a deformable renderer, and the real dog target was unremovable ([fixture lane](absence-fixture-lane-2026-09-10.md) §1.1) → limitations; transparent or specular: Lambertian renderer, while the real bottle is glass → limitation; multi-object departure → future work |
| **B. How it leaves** | instantaneous removal (how the real fixtures are authored); carried out of every frustum by a carrier (X1); driven out laterally under its own motion (M1) | leaving *via* occlusion: images cannot tell whether it left. This is the identifiability boundary, stated as such |
| **C. How long** | 10 / 30 / 90 frames with equal pre- and post-return target supply (D family) | — |
| **D. What happens in the vacated space** | nothing; a distinct occupant crosses the vacated volume (G1); a stand-in is set down and removed (G2, deferred) | a look-alike passes: needs re-identification → limitations; lighting change → out of scope |
| **E. How it returns** | same pose; moved elsewhere with disjoint footprints (R2); rotated 90° (R3); re-enters while moving (M1); small overlapping shift (R1, deferred). As controls: never returns (N1); a different object returns (N2, deferred) | same object with changed appearance: needs episode-conditioned appearance → limitations; partial return of a multi-object set → future work |
| **F. Visibility pathway** | absent from the world; present but enclosed by an opaque cover (O1, the PersistGS state) | partial-camera occlusion is view dropout, covered by the real-occlusion null ([realdata lane](realdata-gating-lane-2026-09-09.md)) |
| **G. Nuisance** | three layouts; ordinary dynamics; boundary phase relative to Ex4DGS's 10-frame keyframes (K1) | — |

## 3. The conditions

**Designed clock.** Each condition's frozen JSON declares a **designed reference
clock (A\*, B\*)**, independent of visibility. It also declares the target's pose
schedule, and for every event object its presence interval and trajectory.

- Windows are computed from the designed clock.
- Rendered visibility only **checks** it: V2 (§6) requires the measured
  all-camera absence interval to equal [A\*, B\*] exactly for instant conditions,
  and within ±2 frames for X1 and M1.
- This removes revision 1's fatal defect, where "B = last invisible frame" made
  B = 299 in N1 (review F1).

**Phases for the non-instant returns** (X1, M1). Measured, then frozen at build:

- E_out: last cam00-visible frame before the exit;
- E_in: first cam00-visible frame after the return;
- S: settle frame, i.e. carrier release for X1 or the first frame at steady
  velocity inside the frustum for M1.

Their return windows use E_in and S, not B (§5, review F2).

| ID | physical scenario | design (A\*, B\*) and change vs A0 | question it answers | tier · layouts |
|---|---|---|---|---|
| **A0** | removed and put back exactly (analogue of the authored real fixtures) | (60, 89); same pose | exact-GT replica of the real fixtures, with no mask noise; carries the confirmatory contrast (§5) | T1 · L1 L2 L3 |
| **R2** | put back somewhere else | (60, 89); instant return at 90 to a disjoint pose on the table | does the method recover the object at a new place **and** leave the old spot empty? | T1 · L1 L2 L3 |
| **G1** | something passes through the empty spot | (60, 89); a distinct occupant crosses the vacated volume, overlapping the old footprint only inside the gap | does the gap representation carry *new* content where the target was, or only switch the target off? | T1 · L1 L2 L3 |
| **O1** | enclosed by an opaque cover but still there | (60, 89) is the **cover** interval. An opaque box, open at the bottom, is lowered over the target [45,59], encloses it [60,89] and is lifted [90,104]. The target never moves | specificity: occlusion ≠ absence | T1 · L1 L2 L3 |
| **Z0, Z1** | never taken / never there | A0's clock | physically valid twins of A0 **only**: per-method ceilings for A0's P2 and P1 regions | T1 · L1 L2 L3 (Ex4DGS base ×1) |
| **N1** | taken away for good | (60, 89) with no return. B\* marks where A0 would have returned | memory must not invent a return | T2 · L1 |
| **M1** | a moving object drives out and back in | (60, 89) global absence; the target moves on a line, exits laterally and re-enters | the dynamic-bank case; the real fixtures only test a static target | T2 · L1 |
| **X1** | a carrier picks it up, carries it out of view, brings it back and sets it down | (60, 89) global absence; lift [45,59], return [90,104]; same final pose | physical exit and entry: a moving target, partial-camera visibility, carrier occlusion. Ex4DGS promotes static→dynamic by displacement (`c_gaussian_model.py:1147-1163`) | T2 · L1 |
| **R3** | put back turned | (60, 89); 90° yaw; the label faces away from cam00 | is appearance tied to the object frame? | T2 · L1 |
| **K1** | the anchor off the keyframe grid | (65, 94) | alignment sensitivity: the real fixtures and T1 are aligned with Ex4DGS keyframes | T2 · L1 |
| **D10, D30, D90** | brief / 1-s / long absence, with equal supply | A\* = 60, B\* = 69 / 89 / 149. The target is present 60 frames before the gap and exactly 150 frames after the return, then removed permanently (at 220 / 240 / 300→none) | duration sensitivity with pre/post supply held equal (review F7). Conclusions are drawn only within the D family | T2 · L1 |
| **R1, G2, N2, CMP** | slight shift / brief stand-in / replaced by a different object / carried out, crossed, returned elsewhere | as in §2 | coverage extensions | **T3 (deferred)** · L1: rendered and frozen; trained only if T2 lands by Oct 1 |

**Counts:**

- T1: 18 scenes, i.e. 4 conditions plus 2 twins, on each of 3 layouts.
- T2: 8 scenes on L1.
- T3: 4 scenes on L1.
- Total: **30 scenes**.

**D-family timing.** The target in D90 is present 150 frames after its return at
150 and is never removed within the sequence. D10 removes it at 220 and D30 at
240. The late removals fall after every scored window of that scene, which ends
by B\*+20. The C1 and H2 do-no-harm regions exclude the late-removal pixels.

**Two structural facts to disclose:**

- T1 events are aligned with Ex4DGS keyframes (60 and 90), like the real fixtures.
  The headline is therefore scoped to aligned events, with K1 as the sensitivity test.
- STG's segment boundaries at 50 and 100 cut some windows. STG reporting is split
  at segment boundaries (§8.2).

## 4. Scene construction

**Layouts:**

| | rig | target (cam00 footprint 4,000–6,000 px; real bottle ≈ 5,100 px, handoff rev 3 §2.1) | ordinary dynamics (identical in every condition of the layout) |
|---|---|---|---|
| L1 | crb (20 cams) | label bottle: capped cylinder, h/d ≈ 3.5, label on one side | a "cook proxy" (torso and arm capsules moving behind the table); a turntable object |
| L2 | flame_steak (21) | carton: box 2 : 1.5 : 1 with distinct faces | cook proxy; a pendulum |
| L3 | sear_steak (21) | mug: cylinder plus box handle | cook proxy; a sliding disc |

**Static world:** textured wall and table (procedural, seeded), 3–4 props, a
directional key light with hard shadows, and ambient light. The target is placed
near the real bottle's cam00 image position.

**Motion:** every trajectory is analytic and piecewise smooth, in a frozen
per-condition JSON, so any frame can be re-rendered.

**Renderer:** a new file, `scripts/build_absence_suite_scene.py`.
`build_synthetic_reveal_scene.py` is **not touched**, because LRV1–4 depend on it.

- **Primitives:** sphere, oriented box, capped cylinder, textured plane, each with
  a per-frame SE(3) pose.
- **Shading:** object-frame textures; a hard-shadow ray to the key light; 2×2
  supersampling with a box filter.
- **Numerics:** float64, 8-bit output rounded by +0.5; seeded.
- **Speed:** the static background is cached per camera, and per-frame work is
  limited to the dynamic objects' screen boxes and their shadows.
- **Buffers emitted per pixel:** front-most identity and **shadow-caster identity**.

**Cost basis, measured 2026-09-23:**

- The existing renderer takes 0.74 s per 1352×1014 frame (5 primitives, no
  shadows, no antialiasing).
- PNGs are 0.06–0.11 MB per frame.
- The workstation has 28 cores and 364 GB free on `D:`.
- Minutes per scene with caching: **UNVERIFIED**.

**Collision and free space (V10).** At every frame, every pair of solids must be
separated by at least a declared clearance. The one designed contact is the
carrier gripping the target, and there the clearance is ≥ 0. The cover must
enclose the target without intersecting it.

**Outputs:**

- **Canonical frame store:** `frames/camXX/NNNN.png` for every camera, cam00 included.
- **Per-method adapter trees** of symlinks into the store (review F5):
  - Ex4DGS: `<scene>/camXX/<frame>.png`, `poses_bounds.npy`, `colmap_0/`;
  - Multi4D: `camXX/images/`, the `cam*.mp4` inventory (symlinked placeholders,
    as for STG), `poses_bounds.npy`, `fused.ply`;
  - STG: its own prep layout;
  - EL-GS (optional): the Blender layout.
- **GT package**, never inside an adapter tree:
  - cam00 scoring masks per window;
  - the cam00 identity and shadow-caster buffers;
  - identities for every camera, compressed (`np.savez_compressed` of the int16
    buffers). The verifier regenerates anything else on demand. Revision 1 would
    have stored ≈ 16 GB of raw int16 identities per scene (review F11);
  - per-frame object poses and presence;
  - `event_spec.json` (the designed clock, phases and windows);
  - a MANIFEST with the sha256 of every file, the generator commit and the spec hash.
- **Initialisation: each method's native preprocessing, with the GT poses.**
  Success is an **admission gate**: if COLMAP fails on a scene, the scene's
  texture is fixed and the scene is rebuilt before freeze. **No oracle-init
  fallback** in any reported baseline result (review F6). GT-sample
  initialisation may appear only as a separately labelled diagnostic.
  - Ex4DGS `colmap_0`: sparse, frame 0.
  - Multi4D `fused.ply`: dense COLMAP on frame 0 (Multi4D README).
  - STG: sparse per-frame points.

## 5. Ground truth, windows and metrics (pre-registered)

**Windows,** from the designed clock with the real-fixture arithmetic ([spec v2](absfix-wave2-spec-v2-2026-09-11.md) §4):

- P1 [A\*+3, B\*−2]
- T [B\*+1, B\*+2] (transition frames, reported)
- P2 [B\*+3, B\*+10]
- S1 [B\*+11, B\*+20]
- H1 [A\*−30, A\*−3]
- C1 [230,259] under the real admission rule
- H2 = all frames

For X1 and M1: T = [E_in, E_in+1], P2 = [E_in+2, E_in+9], S1 = [S, S+9].

**Regions,** cam00 per frame, all analytic:

| region | definition | used in |
|---|---|---|
| **ghost** | target-alone projection at its pre-gap pose, plus that pose's shadow on the static world. Scored against the **actual** GT, which is background, or the occupant in G1 | gap frames |
| **return** | actual identity == target, plus pixels whose shadow caster == target | after the return |
| **vacated-after-return V** | pre-gap-pose ghost region minus the current target and shadow region | R2, N1 (and R1, CMP); frames [B\*+1, B\*+20] |
| **occupant** | occupant identity inside the pre-gap ghost region, scored **only on frames where it is non-empty** (review F2) | G1 |
| **N2 (deferred)** | substitute region (actual identity == substitute); original-target suppression region = pre-gap ghost region minus substitute region, designed ≥ 1,000 px | N2 |
| **ordinary-dynamic, static complement, whole frame** | as named | do-no-harm |
| **cover** | cover identity | O1 |

**Z0 and Z1 use A0's windows and regions.** They are twins of A0 only, because
they are physically valid there (review F3).

**Denominator rule (V11).** Every primary endpoint must have ≥ 5,000 pixel-times
in its window, computed at build. A condition that fails it is redesigned before
freeze, never scored with an empty or tiny region.

**Primary endpoints:**

| condition | primary endpoints |
|---|---|
| A0 | P1, P2 |
| R2 | P2, V |
| G1 | occupant, P2 |
| O1 | P2, H1 |
| N1 | V |
| M1, X1 | P2, T |
| R3 | P2 |
| K1 | P1, P2 |
| D family | P1, P2 |
| N2 | substitute, suppression |

**Pose endpoint, secondary, for R2, R3, M1 and X1** (review "missing"):

- **Silhouette:** pixels inside a dilated union of the old and new footprints where
  |render − target-free GT| > 0.1, where the target-free GT is rendered analytically.
- **Scores:** IoU and centroid error (px) against the GT return silhouette, over P2 and S1.
- The same image-only rule applies to every method.

**Statistics** (review F8):

- **One confirmatory contrast family:** Ex4DGS + method vs Ex4DGS base on **A0,
  pooled over the three layouts**. That is 9 vs 9 runs (3 per layout per arm),
  analysed as a two-way fixed-effects model (arm + layout) on **P1 and P2**, with
  Holm correction over the two endpoints.
  - For power, the whole-frame Ex4DGS sd is 0.3564 dB (n = 6). That gives
    SE ≈ 0.17 dB for 9 vs 9, and a minimum resolvable effect of ≈ 0.5 dB at 80% power.
  - **The event-window sd is unknown** until G-floor, and on the real fixtures
    EL-GS's P1 paired sd was 1.59 dB (stage-2 verdict). The power statement is
    recomputed at G-floor. If it fails, repetitions are added **to A0 only**,
    before any method output is read.
  - Layout × arm interaction is reported descriptively.
- **Everything else is exploratory:**
  - both arms' raw per-run values are shown, with the Welch mean difference and a 95% CI;
  - no "beyond the replicate range" rule;
  - 3 vs 3 per condition resolves only effects of about 1 dB or more, and claims are scoped accordingly.
- **Scope by tier:** T1 conditions span three layouts. T2 and T3 conditions are
  single-layout findings. The three repetitions apply to the Ex4DGS base and
  method arms only.
- **Baselines** are one-run descriptive. Nothing is conditionally repeated after
  seeing a result.
- **Evaluation view:** cam00-scoped, as the N3V protocol is. Two extra synthetic
  evaluation views (outside the rig, preregistered) are scored for the Ex4DGS
  base and method arms as a secondary novel-view audit, via a render-only script.

## 6. Preconditions: applicability matrix

All are evaluated on the setup before any training. A failure blocks that scene.

| check | A0 | R2 | G1 | O1 | Z0 | Z1 | N1 | M1 | X1 | R3 | K1 | D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **V1** target visibility in present frames: cam00 3,000–8,000 px and > 0 in ≥ 15 training cameras | ✓ | ✓ | ✓ | outside the cover interval | ✓ | — | before A\* | steady-state frames only | frames outside lift/return | ✓ | ✓ | ✓ |
| **V2** world absence (target not in the scene graph) **and** zero pixels in every camera over [A\*, B\*]; exact for instant conditions, ±2 for M1/X1 | ✓ | ✓ | ✓ | — | — | — | from A\* to end | ±2 | ±2 | ✓ | ✓ | ✓ |
| **V3** target in the scene graph **and** zero pixels in every camera over [A\*, B\*] | — | — | — | ✓ | — | — | — | — | — | — | — | — |
| **V4** pose change visible: R2 IoU = 0 and the new footprint in range; R3 mean \|ΔRGB\| between the rotated and unrotated target-alone renders ≥ 0.1 over the footprint | — | ✓ | — | — | — | — | — | — | — | ✓ | — | — |
| **V5** occupant covers ≥ 50% of the ghost region on ≥ 3 frames and passes through ≥ 50% of the former volume | — | — | ✓ | — | — | — | — | — | — | — | — | — |
| **V6** every pixel where the scene differs from the **base world** (static world plus ordinary dynamics only) lies inside the a-priori support: the analytic silhouettes and shadows of that scene's event objects, dilated by 1 px | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **V7** loader golden test through each method's own reader: camera count, frame count, shape, time normalisation; 1,000 GT points reprojected within 0.5 px | all scenes, per method | | | | | | | | | | | |
| **V8** independent verifier (written from the frozen spec text only): identity and depth for 50 random (camera, frame) pairs, ≥ 99.9% agreement away from edges, depth within 1e-4 | all | | | | | | | | | | | |
| **V9** leakage: an instrumented run logs the image names the optimizer consumes and asserts that no cam00 frame appears; no GT-package path is reachable from an adapter tree | per method, on one scene | | | | | | | | | | | |
| **V10** collision and free space (§4) | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **V11** non-empty denominators for every primary endpoint (§5) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**These are fixture checks. They cannot show that a method's mechanism was
exercised** (review F4; gap_map 2026-08-24). The method lane's contribution gates
must therefore pass **per ARS scene** before any method arm is scored:

- persistent-cohort mass is active before the gap and after the return;
- it is suppressed inside the gap;
- the residual is active inside the gap;
- both Ex4DGS banks are covered.

Those gates belong to the method lane (handoff rev 3 §5, Sep 25–27) and are
cross-referenced here.

## 7. Runs on ARS in this step (baselines only; method arms are the method lane)

| tier | runs | GPU-h |
|---|---|---|
| **T1a**, needed by the Sep 27 screen | Ex4DGS base ×3 on L1 A0, R2, G1, O1 (12) and ×1 on Z0, Z1 (2) | ≈ 17 |
| **T1b** | the same on L2 and L3 (28 runs) | ≈ 34 |
| T1 descriptive | Multi4D ×1 on the 12 T1 event scenes; STG ×1 seed on A0, R2 and G1 of all 3 layouts (9 scenes) | ≈ 32 + ≈ 54 |
| T2 | Ex4DGS base ×3 on 8 L1 scenes (24 runs) | ≈ 29 |
| T3 | Ex4DGS base ×3 on 4 scenes (12 runs), only if T2 lands by Oct 1 | ≈ 14 |

**Rates:**

- Ex4DGS 1.2 GPU-h per run ([repro](ex4dgs-n3v-leonardo-reproduction-2026-09-19.md)).
- Multi4D 2.7 ([repro](multi4d-n3v-leonardo-reproduction-2026-09-18.md)).
- STG ≈ 6 per scene and seed (§8.2).

T1a is deliberately small (14 runs), so the Sep 27 screen is credible (review
F11). Multi4D and STG on ARS are descriptive and do not gate the screen.

## 8. The parallel lanes

### 8.1 Real fixtures: Ex4DGS and Multi4D

**Route. This needs the principal.** The fixtures exist only at 1352×1014, while
the recorded Ex4DGS route wants native frames and `resolution: 2`
(Ex4DGS pin `:118-120`).

**Recommendation: the 1352 route.**

- **What it is:** 1352 frames, `resolution: 1`, and `colmap_0` intrinsics halved.
- **Precedent:** this is already the recorded STG route, whose pin says it "MATCHES
  the published effective raster".
- **Where Ex4DGS reads intrinsics:** from `colmap_<start>`, and `poses_bounds.npy`
  is "parsed then discarded" at train time (Ex4DGS pin).
- **No code change needed:** its raster is the loaded size divided by `resolution`
  (`scene/cameras.py:201-202`, verified on Leonardo).
- **`colmap_0` is built from the native unedited frame 0:**
  - crb reuses the existing one at `data_staging/ex4dgs/cut_roasted_beef/colmap_0`;
  - flame and sear get one each;
  - intrinsics are then halved.

  The point cloud is therefore identical to the published route's.

**Stage 0, data equivalence (CPU, zero GPU).** Both routes halve with the same PIL
call and default filter:

- the fixture builder uses `img.resize((w//2, h//2))` (`scripts/n3v2blender.py:265-273`);
- Ex4DGS uses `pil_image.resize(resolution)` in `PILtoTorch`;
- a same-size PIL resize is a copy.

The only possible difference is the H.264 decoder's colour conversion: ffmpeg for
the fixtures, cv2 for Ex4DGS. Load five frames through both paths and report the
max and mean absolute difference.

**Stage 1, training (coarse).**

- Ex4DGS runs the 1352 route on the **unedited** crb, three times.
- It is compared with the native record at n = 6: mean 32.6731, sd 0.3564.
- **Pass** iff |Δmean| ≤ 0.50 dB, about 2 SE of a 3-vs-6 difference.
- These runs double as the never-left twins of the real fixtures.

**Multi4D:**

- one run on the unedited crb against its frozen band 33.4–34.0;
- extraction is skipped when images exist (`scene/neural_3D_dataset_NDC.py:72`, verified).

**Fallback if the route fails:** native frames outside the edit silhouette and the
upsampled edit inside it, with its own golden test (about one more day).

**Runs after the checks:**

- Ex4DGS ×3 per fixture: 9 runs.
- The unedited flame and sear ×3 each, as twins: 6 runs.
- Multi4D ×1 per fixture: 3 runs. This is **fixed**, with no conditional
  repetitions (review F9).

**Multi4D checkpoint policy, frozen:**

- **primary** = the final iteration;
- secondary = native `best/`, labelled "test-camera-selected"
  ([Multi4D repro](multi4d-n3v-leonardo-reproduction-2026-09-18.md) §peak-vs-best).

**Preconditions:**

- Fixture frame 0 is byte-identical to the unedited frame 0.
- flame and sear `poses_bounds.npy` have 21 rows (measured locally), and Leonardo
  holds `cam00–cam20.mp4` for both.
- `fused.ply` on Leonardo exists only for crb (`data_staging/multi4d/`, `df3572b2…`,
  identical to the local copy). The local flame (`bbc5988f…`) and sear
  (`6a013735…`) copies are uploaded and hash-checked.

**Replicate spread:** P1, P2 and S1 from the three base repetitions per fixture,
recorded **before any method output is opened**.

**Note for the method lane.** Review F8's objection, that "beyond a 3-run range"
is not valid inference, also applies to handoff rev 3's real-fixture kill rule. It
is flagged here and not edited there.

### 8.2 STG on crb, and STG reporting everywhere

**Correction to the record, measured from `sacct`.** The raw output is in
`refine-logs/ars-review-2026-09-23/leonardo_probe_2026-09-23.txt`.

| stage | elapsed per job | notes |
|---|---|---|
| training | **46:04–49:11** | all 24 stage-2 jobs; 1 GPU, 128 GB |
| prep | 5–6 min per segment | — |
| render | 2–4 min per segment | — |

- **One seed on one scene costs ≈ 5–6 GPU-h.** That supersedes handoff rev 3 §2.3's
  "15–21 GPU-h for ONE seed", which is left unedited.
- **Recommend both crb seeds now** (≈ 12 GPU-h).
- Scripts are present in `$W/agent-control/realdata/stg/`.

**Segment-aware reporting** (review F10). STG trains independent 50-frame models,
so:

- its descriptive comparison uses only windows inside one segment: T, P1 and P2
  lie in [50,99];
- S1, V, H1 and recovery curves are split at the 50 and 100 boundaries;
- values from different specialists are never pooled or differenced.

### 8.3 Common evaluator (no number is read before it exists)

`scripts/eval_event_windows.py`, a new file.

- **Input:** a directory of cam00 renders plus a GT/window package, real or synthetic.
- **Output:** JSON per region and window.

**Frozen conventions** (review F12):

- **Renders:** read back from the saved 8-bit PNG and clamped.
- **PSNR:** pooled, with MSE summed over pixels, channels and frames first.
  Zero MSE is emitted as `mse: 0, psnr: null, exact_match: true`. An empty
  region is emitted as `NA` with its count.
- **SSIM:**
  - an 11×11 Gaussian window, σ = 1.5, K1 = 0.01, K2 = 0.03, data_range 1,
    population (not sample) covariance;
  - per channel, then the RGB mean;
  - the full-frame map with zero padding at frame borders, averaged over the
    region's pixels (windows may extend outside the region);
  - frames pooled by pixel weight.
- **LPIPS:**
  - AlexNet v0.1, with inputs mapped to [−1,1];
  - computed per frame on the region's bounding box, padded by 16 px and to at
    least 64×64, clipped to the frame;
  - the frame mean over the window;
  - frames whose region is under 100 px are skipped, with the count reported;
  - no method's vendored LPIPS is used.

**Golden tests:**

- Whole-frame PSNR is compared against each native evaluator on the same PNGs,
  with the quantisation offset measured first and the tolerance then frozen.
- P1/P2 PSNR reproduces `event_region_frame_profile.py` on the stage-2 renders.
- Unit tests use analytic MSE.

### 8.4 Panoptic Sports census (CPU-light; kill Sep 25 EOD)

**Facts** (verified 2026-09-21/23):

- six sub-sequences of **one** capture;
- 31 cameras, 150 frames at 30 fps, 640×360 JPEG;
- `seg/` covers 27 cameras and is missing for the test cameras 0, 10, 15 and 30;
  it is foreground/background only;
- 716 MB, at `/leonardo_work/EUHPC_D21_034/proj_adags/data/panopticsports`.

Masks can nominate candidates but cannot certify one. **Prior: low** (my judgement).

**Steps:**

1. Copy to `D:`, or run a CPU Slurm job.
2. Find candidates:
   - components that disappear from every masked camera for ≥ 30 frames, with
     ≥ 10 present frames before and after in ≥ 2 cameras;
   - a raw-RGB differencing pass over all 31 cameras.
3. The principal adjudicates contact sheets of all 31 cameras on raw RGB.
4. SAM 2 (`D:\sam2`) is the second method.

**Kill:** if no candidate survives step 3 by Sep 25 EOD, drop Panoptic and record the negative.

**Caveat:** one capture, so two events would count as one scene under the
two-paper rule ([two-paper Sol r2](../assets/two-paper-2026-09-21/sol_round2.md) `:220`).

## 9. Schedule

| date | ARS | real fixtures | STG crb | Panoptic | gate |
|---|---|---|---|---|---|
| **Wed 23** | spec r2 (this page); renderer core | evaluator; converters; upload flame/sear `fused.ply`; Stage 0 | both seeds: prep submitted | — | principal: route + scope (§11) |
| **Thu 24** | L1 T1a rendered; V1–V6, V10, V11; montages; an independent agent writes V8 | Stage 1 golden (dbg rate check first); Multi4D golden | train + render | copy; candidates and contact sheets | **G-route** |
| **Fri 25** | V7 and V9 on Leonardo; **spec frozen**; T1a submitted (14 runs); L2/L3 rendered | fixtures ×3 Ex4DGS + ×1 Multi4D; unedited twins | collect | **kill/go EOD** | **S0** |
| **Sat–Sun 26–27** | T1a results; T1b and T2 submitted | results in | — | — | **G-floor:** event-window spreads recorded and the A0 power recomputed before any method output is opened |
| **Sep 27–29** | the method screen consumes T1a | method screen | — | — | method kill rule (handoff rev 3 §5) |

If V7 fails for Ex4DGS by Sep 25 EOD, the exporter is fixed while the real lane
proceeds. ARS never blocks the real lane.

## 10. Compute and storage

**Compute:**

- Real fixtures ≈ 44 GPU-h: Ex4DGS 21.6, Multi4D 10.8, STG crb two seeds ≈ 12.
- ARS baselines ≈ 180 GPU-h (§7).
- Total ≈ **225 GPU-h ≈ 1.5%** of ≈ 15,500.
- The 1-GPU queue is the constraint: use the shortest safe walltime (measured
  rate × 1.3), check the rate on `boost_qos_dbg` first, and never rely on resume.

**Storage:**

- ARS ≈ 0.5–1 GB per scene: PNGs plus compressed identities (UNVERIFIED until the
  first scene is built). That is ≈ 15–30 GB in total.
- STG prep ≈ 11 GB per scene for 9 scenes.
- Both go on `$SCRATCH`. `$WORK` is at 89.9% of 4 TB (measured).
- Scenes can be regenerated from the generator commit and the spec. GT packages
  and manifests are archived.

## 11. Needs the principal

1. **Keep the CINECA certificate alive.** Leonardo answered at 13:04 and 13:42 CEST
   today; the 12-h window was not read. Tell me when you last renewed.
2. **Accept the 1352 route**, gated by the Stage 0 and Stage 1 checks. It is
   already the recorded STG route.
3. **STG on crb:** both seeds now (≈ 12 GPU-h, measured).
4. **OK to copy Panoptic (716 MB)** to `D:`.
5. **Scope:** 30 scenes, tiered as above; T3 is deferred by default. I proceed
   unless redirected.

The real-fixture mask audit (handoff rev 3 §6.5) is unchanged. ARS needs none.

## 12. Risks

| risk | caught by |
|---|---|
| **External validity:** three hand-authored procedural layouts, cam00-scoped. The reviewer names this as **the objection most likely to remain** | the real fixtures stay primary; three rigs and shapes; the secondary two-view audit; stated as a limitation, never argued away |
| "the suite was tuned to the method" | frozen before any method run; falsifying controls (O1, N1, R2, M1); every condition reported |
| empty or ill-posed regions | the designed clock; V11 denominators; the applicability matrix |
| renderer or loader bugs | V6, V7, V8, V10 |
| a mechanism never exercised | the method lane's contribution gates per ARS scene (§6) |
| underpowered claims | one confirmatory 9-vs-9 contrast; power recomputed at G-floor; everything else exploratory with CIs |
| STG segment artefacts | segment-split reporting |
| test-camera selection | Multi4D final iteration is primary |
| queue latency | T1a is 14 runs; dbg rate check; shortest safe walltimes |

## 13. Verified 2026-09-23

Read-only Leonardo probes at 13:05 and 13:42 CEST; raw output in `refine-logs/ars-review-2026-09-23/`.

**Fixture trees** (images, MANIFEST sha256):

| fixture | images | MANIFEST sha256 |
|---|---|---|
| crb | 6,000 | `b8daf2eab97d4c39…` (**not previously in the repo**) |
| flame v3 | 6,300 | `a0a9c369…` (matches spec v2) |
| sear v3 | 6,300 | `e7811266…` (matches spec v2) |

**Other facts:**

- The unedited 1352 frames exist under D21 `data/n3v/<scene>/images/` (crb 6,000, flame 6,300).
- Queue: no jobs.
- Quota: `$WORK` at 89.9% of 4 TB.
- STG timings (§8.2), `fused.ply` and `colmap_0` locations (§8.1).
- The Ex4DGS reader (`dataset_readers.py:520`) reads intrinsics from `colmap_<start>`,
  hardcodes near/far 0.01/300, and splits on `"cam00"`.
- The raster rule (`cameras.py:201-202`) and Multi4D's extraction skip (`:72`)
  and load-time LANCZOS resize (`:121`).
- Panoptic inventory (§8.4).

## 14. UNVERIFIED

- The certificate's validity window.
- Ex4DGS 1352-route equivalence (Stage 0 and Stage 1 decide it).
- ARS render speed with shadows, supersampling and about 15 primitives.
- Scene size with compressed identities.
- Whether native COLMAP succeeds on the procedural textures (an admission gate).
- Ex4DGS, Multi4D and STG iteration rates on ARS scenes.
- The event-window replicate sd of Ex4DGS (G-floor).
- Whether any Panoptic sequence contains a certifiable event.

## 15. Disposition of the round-2 review (snapshot r1 `ed8beec7…`)

| # | finding (severity) | disposition | where |
|---|---|---|---|
| F1 | invalid event clocks in N1/N2/Z0/Z1 (fatal) | **accepted**: designed clock (A\*, B\*); N2 endpoints replaced; Z0/Z1 use A0 windows | §3, §5 |
| F2 | empty or wrong-phase masks in X1/M1/G1 (fatal) | **accepted**: E_in/S phases; occupant scored only where present; V11 denominators | §3, §5, §6 |
| F3 | twins interpenetrate in G1/X1; M1 has no single return pose (major) | **accepted**: analytic target-alone and shadow-caster masks; world twins only for A0; V10 collision | §1, §4, §5 |
| F4 | preconditions inconsistent; no mechanism-exercise check (major) | **accepted**: applicability matrix; world vs visibility split; V6 against a base world; method contribution gates per scene | §6 |
| F5 | cam00 leakage rule incompatible with the layouts (major) | **accepted**: canonical store plus adapters; instrumented optimizer-input assertion | §4, §6 |
| F6 | oracle-init fallback unfair (major) | **accepted**: native preprocessing is an admission gate; oracle init is diagnostic only | §4 |
| F7 | duration ablation confounded (major) | **accepted**: D family with equal 60 pre / 150 post supply | §3 |
| F8 | "beyond the base range" invalid; underpowered (major) | **accepted**: one confirmatory 9-vs-9 blocked contrast with Holm; the rest exploratory with CIs; power recomputed at G-floor. Flagged for the real-fixture kill rule | §5, §8.1 |
| F9 | Multi4D test-selected checkpoint; post-outcome repetitions (major) | **accepted**: final iteration primary; fixed one run | §8.1 |
| F10 | STG segment contamination (major) | **accepted**: segment-split reporting | §8.2 |
| F11 | schedule and identity storage not credible (major) | **accepted, modified**: T1a cut to 14 Ex4DGS runs; compressed identities. **Not** fully adopted: the reviewer's cut would drop R3 and one of X1/M1, but R3, X1 and M1 stay in T2 because they are the movement types the principal asked for, and they are cheap (Ex4DGS only). R1, G2, N2 and CMP are deferred to T3 | §3, §4, §7 |
| F12 | evaluator conventions unspecified (minor) | **accepted**: parameters frozen; JSON schema for exact and empty | §8.3 |
| F13 | stale facts: seven CLI options; STG not "unmodified"; STG trains 50-frame units (minor) | **accepted** | §1 |
| S1 | Ex4DGS keyframe alignment not counterbalanced (suspected) | **partly**: the headline is scoped to aligned events and K1 tests sensitivity. Counterbalancing T1 across layouts would break the match to the real fixtures and push windows across STG segments | §3 |
| S2 | cam00-only evidence (suspected) | **accepted**: cam00-scoped wording plus a secondary two-view audit for the Ex4DGS arms | §5 |
| S3 | O1 lid may not hide the target (suspected) | **accepted**: an enclosing opaque box, with V3 and V10 | §3, §6 |
| S4 | 0.01 dB golden tolerance vs 8-bit quantisation (minor) | **accepted**: measure the offset, then freeze | §8.3 |
| M | pose endpoint; one central contrast; applicability matrix; collision checks (missing) | **accepted** | §5, §6 |
| — | residual objection: external validity | **stands**; stated as a limitation | §12 |
