---
title: Counterfactual-absence fixture from SA4D-edited N3V footage, and Lane A (deferred-seeding retest of the real occlusion)
date: 2026-09-10
evidence_bearing: false
---

# Counterfactual-absence fixture lane (2026-09-10) — an oracle-controlled diagnostic

EXPLORATORY, `evidence_bearing: false`. LABEL: the fixture is a
**teacher-rendered counterfactual absence** — real N3V footage in which one
segmented object is removed for a frame window by compositing an SA4D
render of the scene without that object's Gaussians into the real frames
inside the object's mask. The edited dataset is its own ground truth for
every arm; it is not evidence about the physical world, and the real
occlusion result ([[realdata-gating-lane-2026-09-09]] §4: −3.4 dB at
render time) stays the headline real-footage number.

Plan approved 2026-09-10 after an external Codex review (24 defects,
dispositions in the plan; the review's model override `gpt-6-astra` is not
available in the installed Codex, the default model at high reasoning was
used). Sections are appended as results return; nothing is rewritten.

## 0. Decisions before any job

| item | value |
|---|---|
| scene / object | `cut_roasted_beef`, the dog on the stool; cam15 DEVA id **95** (identified on the id montage, frames 60 and 240; the stool is a separate id) |
| window | 30 absent frames, chosen at low mask velocity from candidates [40,69], [60,89], [230,259]; guard frames a±2, b±2 excluded from primaries |
| edit | composite: real pixels outside the object mask, SA4D background render inside (dilate 6 px, feather 2 px); only window frames edited; hole fraction (unsupported background) REPORTED per camera and frame, not a stop |
| evaluation masks | DEVA on cam00 (job 57085944, COMPLETED, 300 id maps; cam00 had never been labelled and is held out of SA4D training) + manual audit; the dog is cam00 id **125** at frames 60 and 240 (stool seat 123); per-frame supported eroded core / ring / object ROIs |
| membership truth | rendered-contribution vote with the construction masks (pre-gap and return anchors) |
| training design | four independent ungated 6k prefixes; paired continuations per prefix: U, G-oracle, G-est, G-mis, G-wrongmem, G-ones; descriptive statistics at n = 4 pairs |
| code-path control (G-ones) | the loader refuses an empty gap list and a gap outside [0, 10 s] (probe 2026-09-10: "at least one absence gap is required"; "boundaries not strictly increasing"), so G-ones = the same membership with a gap at the end of the take, outside every scored window; a 6-frame gap (292–297) was refused at run time (inverse target 0.10 s below the 0.167 s floor after the 2-frame edge inset), so the gap is **286–297** (12 frames): EL-GS machinery and the marginal replacement active, the gate exercised only on unscored frames |

## Lane A — deferred-seeding retest of the real occlusion (runs in parallel)

Four ungated 6k prefixes on the un-edited scene, `configs/n3v/b0c_crb300_6k_rp_prefix.yaml`
(= `b0c_crb300_12k_rp.yaml` with `iterations: 6_000`), seeds 0–3: jobs
57085931, 57085937, 57085938, 57085942 (submitted 2026-09-10, label
`laneA`). Per prefix: the membership vote on that prefix's own
`chkpnt6000.pth` with the T1 cell as the authored seed box
(`--seed_bbox3d -0.385 -0.403 -2.927 -0.054 0.311 -2.505`, the 16-grid cell
2944 that T1 gated on the ivv 6k model; recorded as authored), 8 cameras,
mass rule, cap 0.5, emitting row_ids programs for gaps 159–187 (G),
118–147 (G-mis) and 292–297 (G-ones); then four continuations to 12,000
resumed from the same checkpoint (U, G, G-mis, G-ones), configs derived
from the committed templates with only the program path changed (sha256
recorded per cell). Endpoints and analysis as the 2026-09-09 spec, paired
by prefix.

## 1. Phase 0 — preview of the id-95 deletion (job 57088491, COMPLETED 9 min)

`scripts/sa4d_absence_edit_render.py --mode preview --ids 95 --cameras 0 15 8
--frames 25-105,215-275` on the SA4D cut_roasted_beef model (iteration
14000, identity reference cam15). Findings, read before any window was
chosen:

* the per-timestamp row set is **2,773 rows** with the notebook rule
  (argmax | softmax > 0.001), **1,854** at softmax > 0.05, **1,579** argmax
  only; the identity field is time-varying and the sets differ frame to
  frame (Jaccard recorded in `per_timestamp`);
* the object-only render covers **~236k px on cam00** at frame 25 (bbox
  [0,380,652,1013]) — a quarter of the frame — as a diffuse smear, and the
  same on cam15 (`preview/cam15_0060_obj.png`): SA4D represents the
  fluffy dog with large, low-opacity Gaussians whose tails, hidden in the
  full render, dominate an object-only render; the soft limb also admits
  stool/background rows, and the background-without-object render
  (`preview/cam00_0060_bg.png`) loses the stool under the dog;
* the mask-hole fraction reported against that inflated footprint is 0
  and is meaningless until the footprint is fixed.

Design consequences (recorded before the next preview): (i) the edit
region is the SEGMENTER's silhouette per camera (DEVA id harmonised per
camera; cam00 id 125, cam15 id 95), not the object-only alpha, which is
kept only as a diagnostic; (ii) the row set is filtered before rendering:
argmax-only, a scale cap relative to the base set's median largest scale,
a per-axis percentile box, and a multi-view DEVA-mask consistency vote
(a row must project inside the harmonised dog id in ≥ K training cameras
at the anchor frames); survivors after each filter are reported.

### 1.1 The dog cannot be removed cleanly from SA4D's model (previews 57093079, 57093627, 57094791, 57095922)

Row filters were added and run in four previews (all recorded in
`runs/realdata/absfix/preview_ids95_*`): argmax-only + scale cap 3× + 2%
percentile box + multi-view DEVA-mask consistency (a row's deformed centre
must project inside the harmonised dog id in ≥ 8 of 19 training cameras
at frames 40/60/89/240; the analytic projection agreed with the
rasterizer's `points2d` to 3e-5 px). Results:

| base row set | survivors | rendered footprint on cam00 | alpha cover of the cam15 silhouette | edit by eye |
|---|---:|---:|---:|---|
| identity argmax (1,577) | 677 | ~25k px, 99.6% inside the DEVA dog on cam15 | 0.52 | dog mostly gone, white haze remains |
| identity argmax ∨ soft > 0.001 (2,727) | 723 | ~25k px | 0.52 | same haze |
| ALL rows (131,615), hull of the masks | 784 (6 after a miscalibrated depth slab, not used) | — | 0.01–0.03 | — |

So the rows whose centres lie inside the dog's multi-view hull are the
~700–800 the identity field already found; the haze that remains comes
from large Gaussians centred on the stool and wall whose extent covers
the dog (SA4D fits the fluffy white toy with diffuse primitives shared
with its surroundings). No centre- or identity-based selection can
remove the dog without also removing its surroundings; a
contribution-based selection would remove the stool with it. **The dog
is dropped as the edit target**; the fixture needs a rigid, sharply
represented object. Candidates previewed next: bottles (cam15 ids 63,
117, 154, 136), which are tall and thin (parallax exposes the wall behind
them) and static.

### 1.2 Bottles: identity removal leaves ghosts, hull removal works (previews 57096329, 57097283)

Four bottle ids (cam15 63, 117, 154, 136) under the filtered identity
selection: crisp rendered footprints (fit to the segmenter 0.94–0.98 on
cam15) but every background-without-object render still shows a
translucent ghost of the bottle — SA4D's identity field labels only part
of each object's Gaussians. Hull mode (`--base_rows all`, every row whose
deformed centre projects inside the harmonised DEVA id in ≥ 8 of 19
training cameras at four anchor frames): **1,716 rows for the wine
bottle (id 117), 1,827 for the oil bottle (id 63)**, and both objects are
GONE from the background render; what remains is a dark smear at the
bottle's base (the contact patch and the wall/table region no camera
observed) and, for the oil bottle, a blurred reconstruction of the
grinders behind it. The depth-slab "visible-only" test is not used (it
removed 778 of 784 rows in the dog trial and is miscalibrated).

**Edit target: the wine bottle, cam15 id 117, cam00 id 136** (rigid,
static, tall, dark and distinct on the held-out camera at the right of
the cutting board, background = wall and shelf edge, parallax-exposed).
Window candidate [60,89] (margins 40–109, disjoint from the F event).

### 1.3 Wine bottle, silhouette region, frames 40–110 (preview 57098650) → BUILD (job 57102816)

Hull mode gave **1,720 removed rows**; the harmonised DEVA ids for all 19
training cameras are recorded in `preview_id117_deva/preview.json`; cam00
id 136 explicit. Per camera over frames 40–110: silhouette 6,978–7,077 px
on cam00 (7,243–7,399 on cam15, 4,482–4,617 on cam08), alpha cover of the
silhouette 0.96 / 0.98 / 0.93, silhouette velocity ≤ 0.05 px/frame (the
object is static). Edit composites (`preview/camXX_FFFF_edit.png`): the
bottle is gone on every camera and the wall behind is revealed; the one
artefact is a dark vertical smear at the bottle's position (the region no
camera ever observed) and a blurred patch at the base — consistent across
cameras because it comes from one 3D model. **Window frozen: absent
frames [60,89]**, margins 40–109, control window [230,259] (bottle visible,
disjoint from both the edit margins and the F event). Build submitted
with `--base_rows all --mask_consistency (8 of 19 cams at 45/60/75/89/104)
--edit_region deva --dilate 6 --feather 2`, followed in the same job by
`scripts/build_absence_fixture_scene.py` (derived root
`$W/data_derived/absfix/cut_roasted_beef_absfix_60_89`, cam00 evaluation
ROIs from DEVA id 136) and `scripts/verify_absence_fixture.py`.

### Lane A — prefixes done, chains submitted (03:20 CEST)

Prefixes COMPLETED (2h02–2h08 each): `runs/laneA/20260910_0111*_cut_roasted_beef_uprefix6k_s{0..3}`.
The first vote attempt (jobs 57092331/57092336) emitted the G and G-mis
programs (prefix 0: **3,211 members** of 599,547, LOCO Jaccard ≥ 0.994)
and failed only on the 6-frame G-ones gap (floor); its continuations were
cancelled by dependency. Re-chained with the 12-frame gap: votes
57108427 (s0), 57108432 (s1), 57108448 (s2); continuations U/G/G-mis/G-ones
57108428–57108431 (s0), 57108433/38/41/42 (s1), 57108450/51/56/57 (s2);
seed 3 queued after the votes clear the 20-job cap. Derived per-prefix
gated configs under `agent-control/realdata/laneA/configs/` (template
`elgs_local_crb300_12k.yaml`, only the program path differs; sha256
recorded by each cell in `inputs.sha256`).

## 2. Build, assembly and verification (jobs 57102816, 57109482, 57111086)

Build: 20 cameras × 70 frames (40–109), 1,720 removed rows, hull mode,
silhouette edit region; outputs under `runs/realdata/absfix/build_id117_60_89/`
(`images_edited/` 600 frames, `construction_masks/` 1,400, `visible_object/`,
`support/`, `alpha_obj/`, `null_composite/`, `truth3d/`, `rowset.json`,
`hole_fraction.json`, `edit_params.json`, `sa4d_provenance.json`).
Assembly (`scripts/build_absence_fixture_scene.py --copy`; hardlinks are
impossible across the D21_034/D36_068 filesystems): derived scene
`$W/data_derived/absfix/cut_roasted_beef_absfix_60_89/` — 5,400 frames
copied byte-identical, 600 edited, 6.7 GB; cam00 evaluation ROIs
(`absfix/evaluation_rois/{core,ring,object}`) from DEVA cam00 id 136
intersected with the dilated construction mask; `absence_event_masks.json`
(events BOTTLE_absence_gap [63,87], return early [92,99] and late
[100,109], pre [30,57], control [230,259]); `MANIFEST.absence_edit.json`.
Verifier: **PASS** after one fix (it had read the renderer's parameters
from the wrong layout and used a 2 px slack; the correct slack is dilate 6
+ three feather sigmas + 2 = 14 px). QA montage
`build_id117_60_89/qa_montage.jpg`.

Fixture prefixes (four ungated 6k cells on the edited scene, seeds 0–3,
`b0c_crb300_6k_rp_prefix.yaml`, label `absfix`): jobs 57111855–57111858.
