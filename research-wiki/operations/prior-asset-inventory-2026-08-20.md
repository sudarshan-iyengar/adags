# Prior-asset inventory and the first prior-assisted experiment design (2026-08-20)

Operational record, static curation tier. Part A: inventory of prior /
auxiliary-signal assets already resident in the repository or on Apollo,
usable with ZERO new acquisition, downloads, or credentials. Produced by
a bounded read-only worker and reviewed by the primary; each row names
its source and location so any entry can be re-verified. Part B (by the
primary): the frozen design of the smallest prior-assisted synthetic
comparison — DEFINED, NOT RUN, and not authorized to run by this page.

---

# PART A — inventory
# Lane-3 prior / auxiliary-signal inventory (READ-ONLY sweep, 2026-08-20)

Scope: `D:\adags` working tree + `apollo:/apollo/users/sri/proj_adags` (rclone,
listing + small reads only). Nothing written outside this scratchpad; no weights
downloaded; no `det` submission; no multi-hundred-MB pull.

Class legend: **(1)** external learned prior Â· **(2)** geometric preprocessing
(SfM/MVS/COLMAP) Â· **(3)** reconstruction-derived signal.

Leonardo (`$WORK/proj_adags`, `/leonardo_work/EUHPC_D21_034/...`) is **not
reachable from this workstation**; every artifact whose only materialization is
a Leonardo path is marked `NOT REACHABLE` even where its manifest is in-repo.

---

## (a) Optical flow

| Artifact | Class | Source/model | Coverage | Conventions | Confidence/validity | Storage + size | Usable now? |
|---|---|---|---|---|---|---|---|
| `data/n3v/<scene>/flow/<cam>_<ffff>.npz` | **1** | **SEA-RAFT** (`repo/SEA-RAFT/generate_dataset_flow.py`, Leonardo-side generator; revision stamped per sealed record) | **All 6 N3V scenes.** cut_roasted_beef: 5,980 files = **20 cams Ã— 299 frames**; cook_spinach / flame_steak / sear_steak 6,279 (21 cams); flame_salmon_1 5,681 (19); coffee_martini 5,382 (18) | `flow` = float32 `(H,W,2)`, **pixels at source resolution**, integer pixel centers, sampling `bilinear_align_corners_false`; direction **`forward_t_to_t_plus_1`**, **within one camera only** (no cross-camera pairs); `dt` = target.time âˆ’ source.time | `mask` = bool `(H,W)`, `true_means_sample_is_valid`; occlusion semantics `true_means_not_occluded`; per-file `valid_pixel_fraction` recorded by `_build_flow_record` | `apollo:.../data/n3v/cut_roasted_beef/flow`, **56.28 GiB / 5,980 objects**. Per-file sha256 recorded only inside the sealed P02 v6 record (Leonardo) â€” no hash pulled here | **YES, zero conversion.** `MotionPriorCache.get_track_flow` already probes subdir `flow` with `.npz` and selects key `"flow"`, mask key `"mask"` (`utils/motion_prior_utils.py:370-399, 447-475`). Runs in-container on Apollo; nothing to fetch. Verified end-to-end by `scripts/flow_plumbing_smoke.py` and the four `n3v_flow_plumbing_smoke` runs (2026-08-17) |
| Sealed P02 v6 flow record (`depth-visibility-flow-schema-v1`) | 1 (metadata) | `scripts/run_phase9_depth_visibility.py::action_adapt_flow`, config `configs/depth_visibility/phase9_flow_sidecar_cycle_v6.json` | **cut_roasted_beef only** (`ProvenanceError` refuses every other scene) | as above | per-record `npz_sha256`, `flow_contiguous_sha256`, `mask_contiguous_sha256`, source RGB hashes | sealed under Leonardo `$WORK` | **NO** â€” NOT REACHABLE. The raw `flow/` npz on Apollo is usable without it; only the hash-chain of provenance is missing |
| Rendered flow (`pkg["flow"]` from the rasterizer) | **3** | trainer's own renderer, `--enable_rendered_flow`, loss `lambda_track_flow` | any scene/camera/frame the trainer loads | screen-space pixels, same raster as render | none (it is a prediction) | none (computed live) | **YES**, always available; VJP verified on both H100 and V100 (`flow_vjp_*_verify` runs) |

## (b) Short-range point tracks

| Artifact | Class | Source | Coverage | Notes | Usable now? |
|---|---|---|---|---|---|
| None dedicated. | â€” | â€” | â€” | Short-range correspondence is only available **implicitly**, by chaining the SEA-RAFT forward flow above (tâ†’t+1, within-camera). | **YES via chaining** the flow; no separate short-track asset exists for N3V or LRV3 |

## (c) Long-range point tracks

| Artifact | Class | Source/model | Coverage | Conventions | Confidence/visibility semantics | Storage + size | Usable now? |
|---|---|---|---|---|---|---|---|
| `data/diva360_derived/<seq>_tracks/{tracks.json,tracks_shift.json,tracks_shuffle.json,MANIFEST.json}` | **1** | **CoTracker3** offline (`scaled_offline.pth`, sha256 `2670d456â€¦ce7834`), `v2=False, offline=True, window_len=60`; temporal chunking `chunk_len 512 / overlap 64`, first-writer-wins stitching, applied when T>512 | **25 DiVa-360 sequences** (battery, chess, flip_book, jenga, keyboard_mouse, kindle, maracas, music_box, pan, piano, poker, pour_tea Ã—2, put_candy, put_fruit, scissor Ã—2, slice_apple, soda, tambourine Ã—2, tea, unlock Ã—2, writing_1, writing_2 Ã—2, xylophone) + `_fix79ae5b7` re-builds for writing_2/xylophone. **NO N3V sequence, NO LRV3.** Example scissor: 26 training cams, 9 held-out (id â‰¡ 0 mod 4), frames 0â€“561 @ 120 fps, 512 seeds | Schema `elgs-tracks-artifact-v1` (`elgs/tracks_schema.py`). Reports are per (seed, camera, frame): `{frame, time, v, x, y}` or `{frame, is_miss:true}`. `x,y` = **pixel coords in that camera's undistorted raster**; `frame` = DiVa-360 frame index, `time` = index/fps. Seeds are 3D **visual-hull surface voxels** at the query frame (world coords), `n_cam â‰¥ 2` required. `consensus[seed][frame]` = IRLS-DLT Huber (4 px) triangulated world point or `null` when < 2 visible cameras | `v` âˆˆ [0,1] = CoTracker3 visibility logit clipped; **the `v â‰¥ 0.5` threshold is what feeds triangulation**. Reliability: `diagnostics.fb_rms_px` per (seed,camera) from a forward-backward re-query, `reproj_rms_px` per seed; frozen mapping `r_u = clip(min(1 âˆ’ fb/8, 1 âˆ’ reproj/8), 0.05, 1.0)`. **`v` is stored per report**, so instrument re-thresholding costs ~1 CPU-hour and 0 GPU-hours (query_pack 2026-08-14) | e.g. `scissor_screen_w0_561_tracks` = **1.875 GiB / 4 files**; `tracks.json` alone **675,049,735 B**. sha256 of every file recorded in the sealed `MANIFEST.json` (`files_sha256`, `config_sha256 ef80dc78â€¦`) | **YES for DiVa-360, in-container** (`elgs/tracks_loader.load_sealed_tracks`, arg `--elgs_tracks_dir`, `arguments/__init__.py:251`). Ten fail-closed checks incl. sequence-identity and `transforms_train.json` digest binding. **Do NOT rclone-pull** â€” 675 MB/file. **NOT usable for cut_roasted_beef or LRV3: no such artifact exists** |
| CoTracker3 weights | 1 | Meta CoTracker3 `scaled_offline.pth` | â€” | â€” | â€” | `apollo:.../data/tracker_weights/cotracker3/` (+ `MANIFEST.sha256`) | **YES â€” already on Apollo storage.** Building N3V/LRV3 tracks needs **no download**, only GPU time via `scripts/build_elgs_tracks.py`. Licence not stated in-repo (upstream CoTracker3 is CC-BY-NC 4.0 â€” a claim-grade use would need that checked) |
| `build_elgs_tracks.py` shift / shuffle controls | 1 | derived from the primary artifact | same as primary | `+10` frame shift (out-of-window dropped); seed-identity permutation, RNG seed `20260811` | hash-tied to the primary in the manifest | alongside primary | **YES** as falsification arms; `tracks_loader` **refuses** them as evidence (`control_artifact`) |

## (d) Multiview correspondences

| Artifact | Class | Source | Coverage | Conventions | Usable now? |
|---|---|---|---|---|---|
| Common-seed identity in the DiVa-360 tracks | 1 | CoTracker3 + visual-hull seeding | 25 DiVa-360 seqs | Correspondence is **by construction**: all per-camera tracks sharing a `seed_id` are the same 3D surface point. Not a matcher output | **YES** (DiVa-360 only) |
| COLMAP `points3d.ply` | **2** | COLMAP SfM + dense MVS, **frame 0 only** | All 6 N3V scenes + LRV1/2/3. Vertex counts: cut_roasted_beef **366,366** (confirmed from PLY header), cook_spinach 383,105, flame_steak 387,503, flame_salmon_1 338,077, coffee_martini 294,950, sear_steak (binary header, not parsed), LRV3 50,000 | `binary_little_endian`, x/y/z + nx/ny/nz + uchar rgb. **Single time instant; no track ids, no per-point visibility, no reprojection residuals retained** | **YES** locally (`D:\adags\data\<scene>\points3d.ply`, 8â€“10 MB each) and on Apollo. This is the trainer's init cloud |
| `transforms_train.json` / `transforms_test.json` / `poses_bounds.npy` | 2 | COLMAP calibration | all 6 N3V + LRV1/2/3 | Blender/OpenGL c2w in transforms; `opengl_c2w_to_opencv_w2c` converts. Static rig asserted by `build_elgs_tracks._static_rig_matrix` | **YES**, local |
| `data/imvid/sparse*/frame_{000000,000150,000299}/model_{in,out,txt}` | 2 | COLMAP sparse models, Immersive Video pilot | **Immersive/imvid only, 3 frames** | COLMAP text/bin model | **YES** but wrong dataset for the next experiment |
| **Per-frame or time-channeled N3V point clouds** | â€” | â€” | **NONE FOUND** anywhere (local, Apollo `data/`, Apollo `runs/`) | â€” | **NO** |

## (e) Triangulated track geometry

| Artifact | Class | Source | Coverage | Conventions | Usable now? |
|---|---|---|---|---|---|
| `tracks.json â†’ consensus[seed_id][frame]` | 1 (learned tracks) â†’ geometric solve | robust consensus triangulation: IRLS DLT, 3 iterations, Huber 4.0 px, over cameras with `v â‰¥ 0.5` | 25 DiVa-360 seqs; â‰¤512 seeds Ã— full frame window | `{"frame": float, "point": [x,y,z] world | null, "n_cam": int, "reproj_rms": float}`. `null` whenever < 2 visible cameras â€” **this is the only 3D-unknown token** | **YES** (DiVa-360, in-container) |
| Per-seed `reproj_rms_px` | derived | mean over frames | same | pixels | **YES** |
| `data/diva360_derived/scissor_oracle_hull/frame_{0,80,â€¦,560}` | 2/3 hybrid | visual-hull carving (`scripts/build_visual_hull_points.py`) at 8 instants | scissor only | world-frame hull point clouds | **YES** (DiVa-360 only) |
| Triangulated track geometry for **N3V / LRV3** | â€” | â€” | **NONE** | â€” | **NO** |

## (f) Visibility / confidence signals

| Artifact | Class | Source | Coverage | Semantics | Usable now? |
|---|---|---|---|---|---|
| Track `v` per report | **1** | CoTracker3 visibility head, clipped to [0,1] | DiVa-360 only | Higher = more visible. **`v â‰¥ 0.5` is used for both coverage and true-absence**, so the two are partly an instrument identity (query_pack 2026-08-14 measurement closure: 0/597 true-absence windows corroborated; C2 true-track-loss = 0 everywhere) | **YES**, but *the absence instrument has a UNANIMOUS material defect verdict* â€” do not build a presence claim on it without the recorded correction |
| `is_miss` token | 1 | pipeline fail-close on out-of-domain tracker positions | DiVa-360 | Distinct from low `v`: means the tracker left the image domain | **YES** |
| `fb_rms_px`, `reproj_rms_px`, `r_u` | 1 â†’ derived | forward-backward re-query + consensus residual | DiVa-360 | frozen mapping in the manifest | **YES** |
| SEA-RAFT `mask` channel | **1** | SEA-RAFT | all 6 N3V scenes, all cams | `true_means_sample_is_valid`; also documented as `true_means_not_occluded` | **YES** â€” the only per-pixel N3V visibility-ish signal available without new compute |
| Rendered transmittance / rendered depth | **3** | the trainer's own rasterizer | anything the trainer loads | model-internal; the E1-int / observability route | **YES**, always |
| `dynamic_mask_from_residual` | **3** | `|pred âˆ’ gt|` mean over channels, quantile threshold (default 0.85), dilate 2 | anything the trainer loads | binary; computed live, **not cached to disk** when residual-derived | **YES**, always (`--dynamic_mask_from_residual`) |
| R026 boundary support masks | 3 | route0 boundary support selection | cut_roasted_beef, **66 support frames**, cam00 only (e.g. `cam00_0048`, 65 px, fraction 1.9e-4) | manifest in-repo (`refine-logs/hide_reveal_poc/r026_m2_boundary_support/â€¦`), consumable via `--event_boundary_support_manifest` | **PARTIAL** â€” manifest present; the referenced `support_masks/<scene>/<img>.png` files are **NOT in the repo** and **NOT on Apollo** â†’ NOT REACHABLE |

## (g) Semantic / instance features

| Artifact | Class | Source | Coverage | Semantics | Usable now? |
|---|---|---|---|---|---|
| `data/synthetic/lrv3/gt_identity/<cam>_f<nnn>.npy` | **oracle GT** (not 1/2/3 â€” synthetic ground truth) | `scripts/build_synthetic_reveal_scene.py` ray-cast | **LRV3 test cams only: 2, 7, 12, 17 Ã— frames 0â€“59 = 240 files** (identical structure in lrv1, lrv2) | int16 `(300,400)` **front-most identity buffer**. `EVENT_OBJECT_ID = 100`, plus `GROUND_ID`, `BACKGROUND_ID`, per-distractor ids | **YES**, local, tiny. Held-out views only by design ("reveal-mask source") â€” train-camera identity buffers do **not** exist but are cheaply re-renderable on CPU from the frozen generator |
| `data/synthetic/lrv3/event_spec.json` | oracle GT | same generator | LRV3, all 20 cams Ã— 60 frames | Presence intervals: episode_1 `[0,29]`, gap `[30,56]`, episode_2 `[57,59]`; `event_object` centre `[0.7,0.1,0.35]` r=0.2; per-test-view **exact visible pixel count per frame** (e.g. cam02 8570 â†’ 0 â†’ 8570) | **YES** â€” this is the LRV3 presence/birth oracle |
| DiVa-360 fg/bg `masks/` (per converted seq) | 2/3 | dataset-provided / derived foreground masks under `diva360_derived/<seq>/masks` | 25 DiVa-360 seqs | binarized at grayscale > 127; the visual-hull seed constructor's only input | **YES** (DiVa-360) |
| `motion_priors/masks/cam00_*.png` | **3** | `scripts/build_motion_priors.py` â€” **temporal-change heuristic, NOT a learned net**: max of |gray(t)âˆ’gray(tÂ±1)|, box-smoothed radius 2, thresholded at quantile 0.85 (floor 0.05) | **cut_roasted_beef, flame_steak, sear_steak â€” cam00 ONLY, 300 frames each** | binary PNG, 0/255 | **YES** on Apollo (`data/n3v/<scene>/motion_priors/masks/`), but cam00 is the **held-out** camera for the N3V protocol â†’ near-useless for training-time use; regenerable for all cams in minutes |
| Panoptic `seg/` masks | â€” | PanopticSports convention (`_find_panoptic_seg_mask`) | **not present for N3V** (only `configs/panopticsports/smoke.yaml` exists) | â€” | **NO** |
| Any SAM / DINO / semantic-feature artifact | â€” | â€” | **NONE FOUND** | â€” | **NO** |

## Depth priors (adjacent, asked implicitly)

| Artifact | Class | Source | Coverage | Usable now? |
|---|---|---|---|---|
| DA3 weights `DA3NESTED-GIANT-LARGE-1.1` (`model.safetensors`, ~6.76 GB expected) | 1 | Depth Anything 3 | â€” | **Present on `apollo:.../models/depth-anything/`.** Code checkout `apollo:.../repo/depth-anything-3` pinned to commit `41736238â€¦` with 3 file-level sha pins (`depth_visibility/da3_adapter.py`). Inference args frozen: `process_res 504`, `upper_bound_resize`, `align_to_input_ext_scale`, `infer_gs False`. **Usable in-container with zero download**; running it is new GPU compute |
| R031 DA3 depth sidecars | 1 | DA3, `generated_by: infer-da3-depth`, 2026-07-09 | **900 frames = 3 scenes (cut_roasted_beef, flame_steak, sear_steak) Ã— cam00 Ã— frames 0â€“299**; `n_written: 900` | **NO â€” NOT REACHABLE.** Only the manifest (`refine-logs/depth_occlusion_support/r031_da3_depth_full/da3_depth_manifest.json`, 288 KB) is in the repo; the `depth_npz/<scene>/<img>.npz` arrays live under Leonardo `$WORK` and are absent from Apollo. Also **cam00-only and known camera-confounded** (query_pack "Corrected prior evidence": R031-R033 used only cam00, per-frame normalization, no warping) |
| Rendered depth | **3** | renderer | anything | **YES**, always |

## Reconstruction-derived event/candidate manifests (all class 3, all in-repo, all tiny)

| Artifact | Coverage | Content | Usable now? |
|---|---|---|---|
| `configs/n3v/ladder_event_masks_crb0_49.json` | **cut_roasted_beef, cam00, frames 0â€“49**, raster 1352Ã—1014 | 3 hand-frozen reveal rects: A_hand_press `[655,845,745,905]` frames [10,18]âˆª[29,43]; B_knife_stroke `[700,880,790,955]` [34,39]; C_tongs_band `[795,845,890,955]` [34,39]. Schema `ccr-event-ray-masks-v1`, frozen before any ladder output was read | **YES** â€” this is the current CCR scoring mask, already used by `scripts/event_ray_metrics.py` |
| `refine-logs/hide_reveal_poc/r020_â€¦/nonoracle_candidate_manifest.json` | cut_roasted_beef | **72 windows** with `crop_xyxy`, `frame_start/frame_end`, and 7 candidate score terms (dynamic_motion_mean, flicker_motion_mean, mask_boundary_mean, motion_mask_mean, static_delta_motion, score_mean/peak) | **YES** â€” directly loadable via `--event_candidate_manifest` (`MotionPriorCache._load_event_candidates`) |
| R018/R019/R025/R027/R029-R030/R036-R037 manifests + validations | cut_roasted_beef | successive candidate/boundary/visibility-event refinements, each with a decision memo recording a **failed** checkpoint-backed event test | **YES to read**; all are recorded negatives |
| `refine-logs/depth_occlusion_support/r031|r032|r033_*` | cut_roasted_beef (+ sear_steak, flame_steak in the depth manifest) | `depth_occlusion_support_components.csv` (2.3â€“2.4 MB), support manifests, `support_overlap_windows.csv` | **YES to read**, but the underlying depth is cam00-confounded; treat as the recorded negative, not as signal |
| `data/synthetic/ladder_eval/b{0,1}s{0,1}_event.json` + `*_renders/` + `gt/` | cut_roasted_beef frames 0â€“49, cam00 raster | per-region pooled PSNR for B0/B1 Ã— seeds 0/1, 50 GT frames + 50 renders per arm, 228 MB total | **YES**, local |

---

## Usable NOW for `cut_roasted_beef` (zero acquisition, zero credentials, zero weight download)

1. **SEA-RAFT dense forward optical flow â€” 20 cameras Ã— 299 frame pairs, 56.28 GiB on Apollo, already in the exact npz layout `MotionPriorCache` probes.** Direction `tâ†’t+1`, within-camera, pixels at 1352Ã—1014, plus a boolean validity mask per pixel. This is by far the strongest untapped prior and it needs **no new compute at all** â€” only `motion_prior_root` pointing at the scene root (the default already yields `<scene>/flow`). Verified consumable by the four 2026-08-17 `n3v_flow_plumbing_smoke` runs.
2. **COLMAP frame-0 SfM+dense cloud, 366,366 points** (local, 9.9 MB). Class 2, no time channel.
3. **Full calibration** (`transforms_train/test.json`, `poses_bounds.npy`), static rig, 20 cams Ã— 300 frames of undistorted PNGs (6,000 images on Apollo).
4. **Reconstruction-derived, always-on:** `dynamic_mask_from_residual`, rendered depth, rendered flow (`--enable_rendered_flow`, VJP verified on H100 and V100).
5. **Frozen event geometry:** `ladder_event_masks_crb0_49.json` (3 reveal rects, frames 0â€“49) and the R020 72-window candidate manifest with crop boxes.
6. **cam00-only temporal-change masks** (300 frames) â€” cam00 is held out, so this is diagnostic-only; the generator (`build_motion_priors.py`) regenerates all cams in minutes if wanted.
7. **CoTracker3 weights and DA3 weights are already resident on Apollo** â€” building N3V point tracks or N3V depth is GPU time, never a download.

**NOT available for cut_roasted_beef:** any point tracks (short or long range), any triangulated track geometry, any per-frame/time-channeled point cloud, any semantic/instance labels, any reachable DA3 depth (the 300 cam00 frames exist only on Leonardo and are camera-confounded), and the R026 support-mask PNGs.

## Usable NOW for the synthetic LRV3 fixture

1. **`event_spec.json`** â€” exact presence intervals (ep1 `[0,29]`, gap `[30,56]`, ep2 `[57,59]`), event-object world sphere (centre `[0.7,0.1,0.35]`, r 0.2), 16 train / 4 test cams, 400Ã—300 @ 6 fps, and **per-test-view exact visible pixel counts per frame**.
2. **`gt_identity/` â€” 240 int16 front-most identity buffers** (test cams 2,7,12,17 Ã— 60 frames), `EVENT_OBJECT_ID = 100`. Perfect oracle instance/semantic labels and per-pixel occlusion ordering, for held-out views.
3. **COLMAP-convention 50,000-point `points3d.ply`** + `transforms_train/test.json`.
4. **`configs/lrv3/oracle_correct.json` / `oracle_shift2.json` / `oracle_wrong.json`** â€” frozen episode oracles including the deliberate 2-frame-mistiming arm.
5. Reconstruction-derived signals (residual mask, rendered depth, rendered flow) as for any scene.

**NOT available for LRV3:** nothing track-like at all â€” **no optical flow, no point tracks, no visibility scores, no CoTracker3 artifact**. Train-camera identity buffers are also absent (test views only), though the frozen generator `scripts/build_synthetic_reveal_scene.py` re-renders them deterministically on CPU.

## Biggest gap

**Point-track evidence and dense flow are on disjoint datasets.** Every long-range track, visibility score, and triangulated 3D trajectory the project owns is CoTracker3-on-DiVa-360 (25 seqs, ~1.9 GiB each) â€” and its visibility instrument carries a unanimous material-defect verdict. Every dense flow field is SEA-RAFT-on-N3V. **cut_roasted_beef has flow but no tracks; LRV3 has neither.** Closing it for cut_roasted_beef costs GPU time only (CoTracker3 weights + the builder are both already on Apollo), but `build_elgs_tracks.py` currently hard-requires a DiVa-360-style converted scene: per-camera `masks/<cam>/<frame>.png` for visual-hull seeding and `elgs.diva360_schema` held-out/path conventions. N3V supplies neither â€” it has no per-camera foreground masks (only cam00 temporal-change masks) and a different split rule. **That adapter, not compute, is the binding constraint.**

## Which prior role the existing assets support FIRST, with zero acquisition

**The B1 birth prior.** SEA-RAFT flow on cut_roasted_beef is complete (20/20 cameras, all 299 frame pairs), pixel-accurate, carries its own per-pixel validity mask, is multi-view (independent per camera over the same instants, so agreement across cameras is checkable), and is already wired into the trainer's data path â€” a birth prior wants exactly "where did surface move / appear in this frame, in this camera, how trustworthy", which is the flow-plus-validity pair verbatim. The B2 lineage prior wants **identity across an absence gap**, i.e. long-range tracks with visibility states â€” the one thing cut_roasted_beef does not have and cannot get without first writing the N3V mask/seed adapter. So: **B1 birth prior now on flow; B2 lineage prior blocked behind an N3V tracks-builder adapter.**

For LRV3 the ordering inverts: the oracle `event_spec` + `gt_identity` buffers give an exact lineage/identity prior for the held-out views and nothing at all for birth â€” which is consistent with the fixture's role as an identity-claim substrate only.

---

# PART B — the smallest prior-assisted synthetic comparison (frozen design, NOT run)

Target fixture: LRV3 (identity ground truth exists; no priors exist for
it yet, so the flow arm requires ONE preprocessing cell rendering
SEA-RAFT-style flow is NOT available -- instead the fixture's own
analytic renderer can emit exact ground-truth flow, which is the honest
synthetic analogue and removes flow-model error from the comparison; a
real-flow variant on cut_roasted_beef comes only after the synthetic
verdict).

Arms, all passed through the IDENTICAL reconstruction-based acceptance
(the unchanged CCR certificate), plus one arm that bypasses it:

| arm | proposal source | acceptance |
|---|---|---|
| P-desc | descriptor-only (current CCR proposer) | reconstruction certificate |
| P-track | ground-truth flow/track chains only (no appearance) | reconstruction certificate |
| P-comb | track chains gated by descriptor consistency | reconstruction certificate |
| P-trust | the P-track proposals | NONE - link trusted as proposed |

Frozen roles: this is a LINEAGE-prior experiment only (cross-gap pair
proposal). The BIRTH-prior role (SEA-RAFT flow improving B1 sites and
velocities on cut_roasted_beef, zero acquisition, already plumbed via
MotionPriorCache.get_track_flow) is deliberately NOT combined with it;
whichever role runs first runs alone. Scoring: Claim-A identity metrics
(proposal recall, accepted precision, wrong accepts, abstentions, the
opportunity denominator over authored eligible returns) plus reserved
reconstruction deltas. Any external-tracker variant on real data must be
described as tracker-assisted lineage proposal/validation, never as
reconstruction-only identity discovery.

Precondition, from Lane 4: this experiment is worth running only if the
B2 edit survives (or is replaced after) oracle-correct falsification --
a better proposer cannot rescue an ineffective edit.
