---
title: "Wave-2 preparation block (2026-09-11/12) — freeze list for spec v2.0.0, fixture repair, estimator visualisation, S2 setup; no training"
date: 2026-09-12
evidence_bearing: false
---

# Wave-2 preparation block (2026-09-11 → 2026-09-12)

EXPLORATORY, `evidence_bearing: false`. No wave-2 training was authorised
or run. This page records what was built, measured and frozen so that the
freeze list of [[absfix-wave2-spec-v2-2026-09-11]] §11.7 / §13.7 can be
read against artefacts. Every number is tied to a Leonardo job id, a
commit, or a file. Lanes ran in parallel (A primary; B, B4, C, S2 as
subagents); every subagent result was re-read against its artefacts by
the primary before it entered this page.

## 1. Outcome in one table

| item | state | evidence |
|---|---|---|
| Spec v2.0.0 | approved §11; amended §13 (two S2 clicks, frozen construction field, deferred ablations, follow-up); admission blocks, construction fields, instrument hashes filled | spec page §13.1–13.8; JSON hashes in §13 |
| Fixture repair | DONE for both confirmatory scenes (_v3): cap removed on all 21 cameras; root cause was the build-only IQR box, not the camera threshold | jobs 57382785/87 (flame_steak), 57382813/14 (sear_steak); construction_v3.json per scene |
| Admission | BOTH scenes admitted under §11.1 | jobs 57377609/10; `absfix-<scene>-admission.json` |
| Click packet | delivered (frames 50 and 92, 20 cameras per scene); clicks NOT placed (user decision) | `absfix-<scene>-clickpacket-f{50,92}.jpg`, `absfix-s2-clicks-template.json` |
| `--val` gate fix | integration test PASSED on Leonardo to 4 decimals | job 57378507; commits bfb8cbb, 7ba33df |
| v2 reducer | implemented, fail-closed, 55 new tests + wave-1 reproduction fixture | commit 34cc193 |
| sham draws | implemented, 33 tests | commit 2bc5acf |
| T1 visualisation | script + 23 tests; two montages | commit deff517; job 57377939 |
| S2 (SAM2) | installed and pinned; driver + 27 tests; smoke and click probe | commit 1fc8392; jobs 57383718, 57383950 |
| External baseline X | NOT ADMISSIBLE (no released code); STG proposed as replacement, pending the user | literature check by web fetch, 2026-09-12 |
| Lines still pending | S2 clicks (user); S2 masks; programs and prefix checkpoints (need prefix training, T1 and votes, which are GPU work outside this block's authorisation); X decision; final commit hash | §13.7 |

## 2. The fixture repair, in order (Lane B)

1. **First reading, wrong.** The 2026-09-11 build left the bottle cap
   floating on all 21 cameras of `flame_steak`. Diagnostic 57267459 had
   shown the cap band is inside the edit mask and IS modified, so the
   cause was a row-selection failure. The obvious hypothesis (raise
   recall by lowering `--mask_min_cams`) was tried as a ladder (jobs
   57377527/28/29/41): it changed nothing, and at `mask_min_cams 1` it
   damaged the background (867 red px on cam08). The cap is not a
   separate DEVA id (probe 57377286: the ids above the silhouette are
   whole-scene background segments).
2. **The instrument was misread.** The ladder compared PREVIEW
   composites against a BUILD composite. Re-measured on like for like
   (`capband_prev_vs_build.json`): the 8-camera preview is already
   cap-free (0 red cap-band pixels on cam00/08/15 at frames 60/75/89);
   the 8-camera build at the identical vote is not (283/178/272). Carry
   as method, again: compare the artefact that will be trained on, not
   its preview.
3. **Root cause.** `run_build` applies a post-vote IQR outlier box
   (`points_inside_convex_hull`, factor 1.0, on the undeformed centres)
   that `run_preview` does not; it dropped 240 of 2,230 voted rows on
   `flame_steak` and 134 of 2,434 on `sear_steak`. Cap-band object-alpha
   counts identify the dropped rows as the cap (flame_steak cam00 677 →
   441, cam15 829 → 580, cam08 484 → 328; sear_steak cam00 674 → 674,
   which is why its cap survived as a speck only).
4. **Fix.** Commit 3d95d3e exposes the factor as `--hull_outlier_factor`
   (default unchanged; ≤ 0 disables the box; recorded in
   `edit_params.json` and the provenance). Rebuilt both scenes with the
   frozen rule of spec §13.6 (`mask_min_cams 8`, box disabled, all else
   as wave 1, `--event_prefix BOTTLE`): build cap-band counts 0/0/0 on
   every checked camera and frame; `iqr_box.applied = false`; rows
   removed 2,227 / 2,431 (vote 2,230 / 2,434, radius filter −3 each);
   verifier exhaustive PASS (6,300 images per scene: 5,670 sha-identical,
   630 edited, all diffs inside mask + 14 px, raw tree hash unchanged);
   events `BOTTLE_{absence_gap [63,87], return_early [92,99], return_late
   [100,109], pre [30,57], control [230,259]}`; cam00 event boxes
   [957,753,1027,965] / [957,754,1027,963]; manifest sha256
   `a0a9c369…42d7e` / `e7811266…5c322`. The mc=7 "_v2" builds are
   mis-labelled repairs, preserved, never trained on.
5. **A construction-quality note the paper must carry.** Disabling the
   box also removes the ~237 non-cap rows it had spared, and on
   `flame_steak` some of those rows painted the counterfactual
   background: the _v3 and _v1 cam00 frame-75 composites differ on 5,682
   px (flame_steak) and 6,252 px (sear_steak), all inside the event box,
   and a knife segment revealed behind the flame_steak bottle is crisp in
   _v1 and blurred in _v3 (zoom sheets
   `runs/realdata/absfix2/<scene>/zoom_f75_orig_v1_v3.png`). This is the
   2026-09-10 shared-row mechanism appearing in the CONSTRUCTION: SA4D's
   rows near the bottle carry both bottle and background. It is inside
   P1's teacher region, so every arm shares it; it lowers the ceiling of
   P1 on flame_steak and is a declared limitation of that fixture, not a
   difference between arms. Montages viewed by the primary:
   `research-wiki/assets/absfix-<scene>-cam-montage-v3.jpg`.

## 3. Admission evidence (Lane B jobs 57377609/10)

Both scenes: 21 cameras, frames 40–109, worst min/median 0.9803
(flame_steak cam04) / 0.9765 (sear_steak cam14); 0 frames below 0.60 ×
median; 0 below 2,000 px; 0 missing masks; one component per camera at
frames 50/75/100; control window [230,259] passes on both medians (worst
0.9893 / 0.9915). cam00 evaluation ids 65 / 67 (mask jobs 57252526 /
57252529, id-map jobs 57254952 / 57254953); mechanism boxes
[957,753,1027,965] / [957,754,1027,962]. Files
`research-wiki/assets/absfix-<scene>-admission.json`.

## 4. The `--val` gate repair verified on Leonardo (Lane C, job 57378507)

`tests/test_validation_elgs_gate.py` integration class on
`absfix_oracle_s0` (300 frames, cam00): `main.py --val` at commit 7ba33df
reproduces `eval_n3v_gated.py --restore_state` on every whole-frame and
per-event per-frame PSNR to 4 decimals (pooled 34.5512 / 32.3415 /
32.7718 / 32.0405 / 32.0022 on the five BOTTLE events; gate live,
7,747 gated rows). A first attempt (57377233) failed on a harness defect
only (the test handed `--val` a model path that did not exist; `Scene`
copies `input.ply` there without creating it); fixed by creating the
directory, nothing compared changed. Wave-2 scoring still uses
`eval_n3v_gated.py --restore_state` only (spec §11.3).

## 5. Reducer, sham draws, visualisation, S2 (Lanes C, B4, S2)

* **Reducer** (commit 34cc193): everything behind `is_spec_v2()`; Claim
  A / Claim B per §11.5, both endpoints, per-pair harm guards, membership
  and T1 preconditions from integer sidecars, zero-overlap assertion,
  cell preconditions, scene precedence, calibration scene reported
  separately, every ratio with numerator and denominator. 55 tests; the
  wave-1 v1.2.0 output is reproduced from a tracked 796 KB fixture (exact
  JSON equality with three machine-path fields replaced and floats at
  1e-9 relative, because numpy's `mean` over 300 doubles differs in the
  last bit between machines; a companion test shows 1e-6 is caught). The
  primary read `evaluate_claim_v2` and `combine_scene_verdicts_v2`
  against §11.5. Found by this work: the frozen JSON had omitted the
  per-scene bounding-box `event_name`; added as a pending-key fill
  (§13.6).
* **Sham draws** (commit 2bc5acf): GWRONGMEM_A/B uniform, count-matched,
  zero overlap, seeds 1000/2000 + prefix seed; GWRONGMEM_L greedy by
  contribution within ±10% of the truth mass, refusing when unreachable;
  counts sidecar per draw; the vote script gained an off-by-default
  `--emit_row_weights` dump. 33 tests.
* **T1 visualisation** (commit deff517; CPU job 57377939, 26 s): every
  voxel cell projected on cam00 across the gap, coloured by outcome, the
  fired cells' windows, and per-fired-group strips. The report schema
  stores no per-frame series (`estimate_episodes.py` writes only the
  decisions), so the strips show per-camera onset/offset markers and say
  so on the figure. Real occlusion: 1 gated cell (g1033, 1,483 rows) on
  the beef, estimated [159,187] vs curated [158,187]. Fixture prefix 0:
  2 gated cells (g971 [60,89], g1023 [60,91]) on the bottle's counter.
  Montages `research-wiki/assets/t1-viz-*.jpg`, viewed by the primary.
* **S2** (commit 1fc8392): SAM2 pinned at 2b90b9f5…, checkpoint
  `sam2.1_hiera_large.pt` sha256 `2647878d…dd318`, venv `envs/sam2`,
  torch 2.5.1+cu121; the PEP-517 editable install silently produced no
  `_C` extension (rebuilt in place), and a first smoke job completed
  while silently skipping post-processing on a `GLIBCXX` load failure
  (fixed with the env preload plus a hard import guard). Smoke 57383718:
  masks at the DEVA raster 1352 × 1014. Click probe 57383950 on cam15
  frame 50: shoulder click → whole bottle (7,152 px); label click →
  label only (3,517 px, would fail the 0.70 IoU precondition); low-body
  click → the wine glass in front. The user's clicks are not placed by
  the probe.

## 6. External baseline

FreeTimeGS has no released training code (an explicit release request on
the authors' framework repository has been open, unanswered, for more
than a year); FreeTimeGS++'s repositories return 404; RetimeGS has no
code. An unofficial reimplementation exists (OpsiClear-4DGS/
FreeTimeGsVanilla, commit 911dcf41…, AGPL, COLMAP input, no parity
claim) and is not a citable baseline. Recorded as NOT RUN with reason;
SpacetimeGaussians (public code, per-primitive temporal opacity,
published 33.52 on `cut_roasted_beef`) is proposed as the replacement,
pending the user.

## 7. Cost and what was skipped

GPU: two SA4D builds at ~31 min each, one 21-min integration test, two
short SAM2 jobs, four short montage/probe jobs, the retired mc=7 builds
(~1 h). No training. Skipped or not done: S2 clicks (user); S2 masks;
prefix training, T1 and votes on the new scenes (GPU, not authorised in
this block); the X decision. A pre-existing failing test
(`tests/test_gate_cell_precondition.py::ConsumerContract`, a ninth field
`arm_kind` lifted by `read_precondition`) was noticed and not touched.
