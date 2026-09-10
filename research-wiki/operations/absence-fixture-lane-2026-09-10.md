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

## 3. Lane A RESULT — deferred seeding binds membership, and the gate still harms the occlusion return (16 cells, 07:40 CEST)

Cells: four ungated 6k prefixes (jobs 57085931/37/38/42; held-out 6k PSNR
33.186 / 33.688 / 33.660 / 33.181) and, from each prefix, four
continuations to 12k — U (`b0c_crb300_12k_rp.yaml`), G, G-mis, G-ones
(`elgs_local_crb300_12k.yaml` with the per-prefix program). Jobs
57108428–31 (s0), 57108433/38/41/42 (s1), 57108450/51/56/57 (s2),
57109619/23/27/28 (s3). Every cell trained, evaluated and profiled;
eight cells (all G-mis and G-ones) reported FAILED only because the chain
passed the G program as the precondition reference and the extractor
fails closed on a program mismatch — re-run with the program each cell
trained with (job 57136959, all rc=0). Analysis:
`scripts/realdata_gate_analysis.py --paired --wave 1` (default spec
v1.0.0, F-box windows) → `runs/realdata/laneA_analysis/wave1_paired.json`
(sha256 `d3d6d11f…a509a`), manifest sha256 `d0b000b9…4ae83`.

**Membership is bound this time.** Seeding on the restored 6k cloud gated
3,017–3,234 rows per prefix (2026-09-09 lane: 389 on the sparse initial
cloud); 2,994–3,219 survive to 12k; 637–662 of them lie in the F-box at
frame 150; `frames_presence_zero` 27 for G and G-mis, 9 for G-ones.
Every arm passes the mechanism precondition (`ME = yes`, 16/16).

**The gate harms the occlusion event in every pair.** Paired
within-prefix contrasts (dB, median over 4 pairs; min..max; sign
consistency):

| contrast | P1 gap [161,185] | P2 return [190,197] | S1 [198,207] | H1 pre [128,155] | H2 whole | C1 [230,259] |
|---|---|---|---|---|---|---|
| G − U | −0.222 (−0.320..−0.053) 4/4 | −0.464 (−0.778..−0.227) 4/4 | −0.356 4/4 | −0.135 4/4 | +0.004 3/4 | −0.143 4/4 |
| G-mis − U | −0.082 4/4 | −0.363 4/4 | −0.308 4/4 | −0.166 3/4 | +0.004 3/4 | −0.279 3/4 |
| G-ones − U | −0.122 4/4 | −0.774 (−1.246..−0.405) 4/4 | −0.631 4/4 | −0.385 4/4 | −0.014 4/4 | −0.344 4/4 |
| G − G-mis | −0.140 (−0.205..−0.045) 4/4 | −0.101 2/4 | −0.047 2/4 | +0.031 3/4 | +0.003 3/4 | +0.136 3/4 |

Verdict: **CLAIM_CONDITIONS_NOT_MET on P1 and P2** (no G−U pair is over
the floor in the right direction; the sham contrasts are "clean" only
because everything is negative). Paired sd 0.11 dB (P1) / 0.23 dB (P2);
the sizing rule asks for 8 pairs at δ = 0.30, which is moot given the
sign.

Reading. (i) Whole-frame is unchanged (|G−U| ≤ 0.008 dB), so the cost is
confined to the ~3,000 gated rows' box, as designed. (ii) The G−G-mis
contrast on P1 is the decisive one: gating the SAME rows at the CORRECT
time is 0.14 dB worse in the gap window than gating them at a shifted
time, in all four pairs — the occlusion window is where the rows are
needed to paint the hand and blade, and switching them off there exposes
what is behind. This is the training-time version of the −3.4 dB
render-time result of 2026-09-09: binding membership at 6k does not
separate the roles, because the rows the vote finds under the beef are
the rows the occluder also uses. (iii) **G-ones is not a null arm.** Its
program has a 12-frame gap at 286–297 (the emitter refuses gaps below
0.1667 s, so an all-present program cannot be written); the rows it
gates lose their learned temporal marginal for the rest of the sequence
and it is the WORST arm on P2/S1/H1/C1 (−0.77 dB on P2). So the
"replicate floor |U − G-ones|" the paired analysis reports (0.12 / 0.77
dB) is a code-path effect, not chaos; the G-mis − U contrast (−0.08 on
P1) is the better floor estimate here. (iv) Lane A's question — "can the
gate act on the occlusion return once membership is bound?" — is
answered **no** on this event: P2 is harmed by 0.23–0.78 dB in every
pair. The absence fixture (§4) is the only remaining route to a positive
number on real footage.

Cost: 4 × 2.1 h prefixes + 16 × ~2.5 h continuations + 8 × 9 min
preconditions ≈ 50 A100-h.

## 4. Fixture Phase 3 — prefixes, blind T1 (jobs 57111855–58, 57132256)

Prefixes on the edited scene (`b0c_crb300_6k_rp_prefix.yaml`, seeds 0–3,
label `absfix`): held-out 6k PSNR **33.083 / 33.506 / 33.376 / 32.924**
against 33.186 / 33.688 / 33.660 / 33.181 on the un-edited scene at the
same seeds — the edit costs the substrate 0.10–0.28 dB whole-frame (the
bottle's counterfactual background is teacher-rendered, so cam00 has a
harder target there). The four cells reported FAILED only in their
post-step: the assembler writes `absence_event_masks.json` under
`absfix/`, the cell template looked for it at the scene root; training,
eval and checkpoints are intact (templates fixed; the 6k profiles are
produced by the vote jobs instead).

**Assembler defect found and fixed (commit aa7c97e).** It read the
editor's margin from the top level of `edit_params.json` (0) instead of
`edit_params.args.margin` (20), so the cam00 evaluation ROIs covered only
[60,89]. Regenerated for the full render range [40,109] with the same
`build_rois` (the 30 existing frames matched byte-for-byte; 120 files
added; recorded in `absfix/MANIFEST.rois_regen.json`). Core ROI ≈ 5,100
px, ring ≈ 5,050 px, object ≈ 7,000 px on every frame.

**Blind T1 on prefix 0** (`estimate_episodes.py`, frames 35–114, 4 cameras,
16 cells over the [1,99] percentile box → 1,245 groups, `--skip-scoring`;
job 57132256, 2.0 h; program sha256 `2a491638…dafcc`, report
`b5399abf…3dc20`): **2 of 1,245 groups gated, both with offset frame 60 =
the authored A; onsets 90 (= authored B+1, exact) and 92 (two frames
late); zero false activations.** The two cells (keys 2482, 2737) are
adjacent (x 9–10, y 11, z 1–2 of 16), together 6,692 rows at estimation.
Frozen parse (declared before the run): seed bbox = the union of the
gated cells, `[-1.0604, 1.7081, -2.5094] .. [-0.4000, 2.4219, -1.6656]`;
estimated gap = min offset .. max onset − 1 = **[60, 91]**, temporal IoU
with the authored [60,89] = 30/32 = **0.9375**. Estimand (ii)'s timing leg
therefore survives on real footage (the 2026-09-09 real-occlusion T1 was
159/188 against a curated 158–187; here the authored truth is exact).

Votes chained on the T1 seed for every prefix (jobs 57150910–13:
truth/mis/ones from the construction masks, est from the DEVA ids with
the T1 gap, wrongmem = a count-matched random row set with the authored
gap) and render-time gate evals on prefix 0 with the oracle and the
estimated program (57150914/15).

## 5. Membership votes, the render-time diagnostic, the review, the freeze (08:20 CEST)

**Votes (jobs 57150910–13, per prefix).** Seed = T1's two cells; anchors
50–57 and return frames 92–99; all 19 training cameras; `--id_rule mass`,
cap 0.5. Truth (construction masks) and estimated (DEVA ids) membership:
**7,004 / 6,981 / 6,858 / 6,813 rows, identical row sets in every prefix
(precision = recall = 1.0)**. This is by construction, not an achievement:
the edit region was the DEVA silhouette (`--edit_region deva`), so the
construction masks ARE the DEVA masks (8,806 px on cam01 frame 50 in
both trees) and the two votes see the same pixels. **G-est therefore
differs from G-oracle only by the gap (60–91 vs 60–89); the membership
leg of estimand (ii) is not tested by this fixture.** The T1-independent
membership number is T1's own cells against the truth set: precision
0.771, recall 0.737, Jaccard 0.605 (prefix 0). G-wrongmem = a
count-matched random row set (overlap with the truth set 1.0–1.3%),
authored gap. G-ones = truth rows, gap 286–297. Scores under
`runs/realdata/absfix/membership_scores/`.

**Render-time gate on prefix 0 (jobs 57150914/15, `eval_n3v_gated.py`,
frames 30–110, cam00).** Precondition block: 7,004 gated rows
(1.17% of the cloud), exact absence on 28 of 30 declared gap frames
(the two edge frames carry the ramp). Readings (pooled PSNR, dB):

| arm | P1 core [63,87] | box gap [63,87] | P2 [92,99] | S1 [100,109] | H1 [30,57] | whole [30,110] |
|---|---|---|---|---|---|---|
| ungated 6k model | 31.22 | 33.12 | 27.34 | 31.00 | 29.80 | 32.75 |
| + gate (oracle program) | 25.32 | 28.29 | **30.50** | 31.22 | **30.63** | 32.74 |
| + gate (estimated program) | 25.32 | 28.29 | 30.50 | 31.22 | 30.63 | 32.73 |

Two readings. (i) **The ungated 6k model already renders the authored
absence** (31.2 dB on the core during the gap, above its own pre-gap
box), and switching the bottle rows off at render time costs 5.9 dB
there: the same rows paint the counterfactual background in the gap
(4D colour) — the shared-row mechanism of 2026-09-09 again, now on a
genuine absence. (ii) **The gate repairs the return at render time:
+3.2 dB on P2 and +0.8 dB on H1.** The learned temporal marginal of the
bottle rows has been dragged down around the gap (the ungated return
sits 3.7 dB below its settled value), and forcing full presence in the
two episodes restores it. Nothing was trained here; the comparison
below asks whether training WITH the gate keeps (ii) and removes (i).

**Codex adversarial review of the frozen spec** (default model,
reasoning high, read-only; 15 items). Accepted and folded into
`configs/n3v/absfix_gate_spec_v1.json` v1.2.0 (sha256
`0d58b32c…d810`) and `scripts/realdata_gate_analysis.py` (commit
6b368d2; every new behaviour sits behind a spec key so the frozen
2026-09-09 output is unchanged, 100 tests): event name
`BOTTLE_absence_gap` (was the F event); anchors frozen A=60, B=89,
CA=230, CB=259 with **B redefined as the last absent frame** (the 1.1.0
prose said "first frame of the return", which would have shifted every
window by one); the claim floor is the fixed 0.5 dB, `|U−GONES|` and
`|U−GMIS|` are reported descriptively (G-ones is a late-gap sham with a
real code-path cost, not a null arm); only P1 carries a verdict; the
operative verdict is read from the mechanism-exercised set; GMIS,
GWRONGMEM and GEST pairs are required on every G-U prefix; every P1
frame must carry a non-empty core mask (a missing ROI is no longer a
silently dropped frame); per-frame MSE diagnostics (fraction of frames
with lower G MSE, count of exact-zero frames, max PSNR, pooled ΔMSE);
reserved units required from every cell; the mechanism predicate now
requires zero-presence frames inside [63,87] (G, G-est, G-wrongmem) and
the bottle box [964,748,1034,952] at frame 75 (the extractor is run with
`--fbox … --fbox_frame 75`); G-wrongmem/G-ones passing is labelled "gate
code path exercised"; sizing informational and P1-only; PAIRED_MIN_PAIRS
4; the membership scorer labels its envelope IoU and lists per-group
gaps. Not adopted: a GEST direction condition in the claim (GEST is
reported against G as the estimator's share; it cannot fail
independently here since its rows equal G's). The provenance concern
(a mislabelled run directory) is already enforced upstream: the
extractor refuses a program the cell did not train with, which is
exactly what tripped the Lane A post-steps.

**Freeze** (`runs/realdata/absfix/freeze_v1.txt`, sha256
`02516960…8b678`, written before the first continuation was submitted):
commit 6b368d2, spec, seven scripts, three base configs and the 20
derived per-prefix configs, the 20 programs, the T1 artefacts, the four
prefix checkpoints, the fixture manifests, the per-kind ROI digests.

**Continuations submitted 08:23 CEST:** prefixes 0–2 × {U, G-oracle,
G-est, G-mis, G-wrongmem, G-ones} = jobs 57153546–57153568; prefix 3's
six follow when the 20-job cap allows.

## 6. INSTRUMENT DEFECT FOUND AT THE READING STAGE: `--val` renders every EL-GS cell with its gate OFF (11:30 CEST)

The first look at the fixture cells was a montage of the cam00 bottle crop
(`research-wiki/assets/absfix-prefix0-cam00-montage.jpg`: rows GT, U,
G-oracle, G-est, G-wrongmem, G-mis, G-ones of prefix 0; columns frames 45,
62, 75, 88, 95, 105). Inside the gap the G-oracle and G-est rows show a
**full, sharp bottle** — sharper than the faint ghost U renders — although
their gated rows are exactly the bottle rows the gate drives to zero.
Read in code: `main.py validation()` restores the checkpoint and renders
with `render` **without calling `setup_elgs`**; `gaussians.restore` only
stashes `_pending_elgs_state`, the runtime is attached nowhere else
(`elgs/trainer_hooks.py:302` is the only assignment), so `elgs_active`
is False in the renderer and every row goes through the ordinary temporal
marginal. **Every `--val` metric of every EL-GS cell in this project on
N3V — the 2026-09-09 lane, Lane A above, and the interim fixture numbers
— was rendered with the gate off.** The synthetic Lane B positive is
unaffected (`scripts/eval_lrv1_event.py` calls `setup_elgs`), and so are
the preconditions (`gate_cell_precondition.py` attaches the runtime and
reads presence from it). The U cells are unaffected. What the gated
cells' `--val` numbers measured is a model rendered in a configuration
it was not trained in: under the total gate the gated rows' presence is
the episode function, so their temporal-marginal parameters are
unexercised from 6k onward, and at `--val` those stale marginals decide
what the bottle rows paint.

Interim fixture reading under the defect (prefixes 0–2, kept as an
append-only record, NOT a result): P1 core G−U −10.7 dB, G-est−U −12.5,
G-mis−U −1.9, G-ones−U −2.1, G-wrongmem−U −12.2; P2 G−U −3.4, G-mis/
G-ones −7.3; H1 G-mis/G-ones −6.0. These numbers describe the gate-off
render of gated models and are superseded by §7.

Repair (commit fbf4693): `scripts/eval_n3v_gated.py --restore_state`
takes `setup_elgs`'s restore branch (the checkpoint's own `elgs_state`),
proves the restored intervals equal `--program` by lineage key (the same
check the precondition extractor uses), and scores the model gate-on
and gate-off over frames 0–299 on cam00. Re-evaluation template
`agent-control/realdata/reeval_gated.sbatch`: per cell the gate-off
`--val` profile is preserved as `f_box_profile_val_nogate.json`, the
gate-on profile becomes `f_box_profile.json`, and the evaluator's own
gate-off render is profiled to `f_box_profile_gateoff_check.json` (it
must reproduce the `--val` numbers). Jobs 57178206/07 (Lane A, 12 gated
cells), 57178209/10 (fixture prefixes 0–2, 15 gated cells); prefix 3's
five follow. Both paired analyses are re-run afterwards; §3's Lane A
table is superseded by §7 as well.

Carry as METHOD: a montage of the scored region on every arm is not
optional decoration; it is the cheapest precondition there is. Here it
caught in one glance what the per-cell precondition (which reads the
live runtime, not the rendered pixels) could not.

## 7. Gate-faithful results (re-evaluation jobs 57178206/07/09/10, 57194307; every cell re-rendered with its own gate over frames 0–299 on cam00)

**Reproduction check passed for all 27 re-evaluated cells:** the
evaluator's gate-off render reproduces the `--val` profile to four
decimals (e.g. Lane A G s0 whole-frame 32.9608 / 32.9608; box [161,185]
35.857 / 35.857), so the §6 diagnosis is exact. Every Lane A re-eval ran
in restore mode with the gate live: 2,994–3,219 gated rows, 27 exact-
absence frames (9 for G-ones), `program_match` confirmed by lineage key.

### 7A. Lane A corrected: the gate is a NULL on the real occlusion, not a harm

`runs/realdata/laneA_analysis_gateon/wave1_paired.json` (sha256
`3213b69a…42a9c`, same manifest as §3). Paired medians over 4 prefixes
(dB; sign consistency):

| contrast | P1 gap [161,185] | P2 return [190,197] | S1 | H1 pre | H2 whole | C1 [230,259] |
|---|---|---|---|---|---|---|
| G − U | −0.074 (4/4) | +0.050 (3/4) | +0.018 | +0.044 (4/4) | +0.006 | +0.042 (4/4) |
| G-mis − U | −0.036 | −0.016 | −0.031 | −0.146 (4/4) | +0.005 | −0.235 (4/4) |
| G-ones − U | −0.021 | −0.068 (4/4) | −0.074 (4/4) | −0.040 | −0.006 | −0.016 |
| G − G-mis | −0.066 | +0.073 | +0.049 | +0.187 (4/4) | +0.004 | +0.276 (4/4) |

Within-prefix `|U − G-ones|` floor: 0.040 (P1) / 0.068 (P2) dB; paired
sd 0.066 / 0.032 dB. Verdict NOT_MET on both primaries (no G−U pair
reaches +0.5 dB; on P1 the sham is not "clean" only because G-mis−U is
also slightly negative). Reading: the §3 harms (−0.22 / −0.46 dB) were
entirely the gate-off artefact. With the gate applied, G and U agree
inside ~0.1 dB everywhere, and the one sham signal is where it should be:
G-mis, which gates the beef rows while the beef is visible, loses 0.24
dB in its own window (C1) and 0.15 dB pre-occlusion, in all four pairs.
On the beef box during the occlusion the gated rows are behind the hand,
so gating them changes the render by nothing measurable (the gate-on and
gate-off box values coincide to three decimals on every G cell). Lane
A's answer to "can the gate act on the occlusion return once membership
is bound?" is therefore **no effect either way**: a real occlusion offers
the presence gate nothing to remove and nothing to restore.


### 7B. The fixture comparison, gate-faithful (24 cells, 4 prefixes × 6 arms, 12k; analysis at 17:10 CEST)

`runs/realdata/absfix/analysis_wave1/wave1_paired.json` sha256
`64a32d41…98b3`, manifest `a0e69d9a…9a40`, spec v1.2.0 `0d58b32c…d810`,
analysis code at commit f954603; every cell complete, every gated cell
passes the mechanism predicate with the bottle box at frame 75 and its
zero-presence frames inside [63,87] (G-mis/G-ones exempt by design);
no blocking error, no warning.

Per-cell pooled PSNR (dB), cam00:

| prefix | arm | P1 core [63,87] | P2 return [92,99] | S1 [100,109] | H1 pre [30,57] | H2 whole | C1 [230,259] |
|---|---|---|---|---|---|---|---|
| 0 | U | 31.09 | 29.11 | 31.71 | 30.77 | 33.36 | 32.29 |
| 0 | G-oracle | 34.24 | 32.34 | 32.77 | 32.04 | 33.38 | 32.00 |
| 0 | G-est | 34.08 | 32.39 | 32.73 | 32.02 | 33.26 | 31.97 |
| 0 | G-mis | 26.71 | 27.36 | 31.22 | 29.58 | 33.34 | 32.21 |
| 0 | G-wrongmem | 31.86 | 29.35 | 31.82 | 30.95 | 33.38 | 32.25 |
| 0 | G-ones | 26.71 | 27.42 | 31.28 | 29.54 | 33.38 | 32.29 |
| 1 | U | 31.36 | 29.06 | 32.44 | 31.50 | 33.73 | 32.08 |
| 1 | G-oracle | 36.28 | 32.10 | 32.88 | 32.35 | 33.72 | 31.86 |
| 1 | G-est | 36.29 | 32.13 | 32.87 | 32.37 | 33.72 | 31.86 |
| 1 | G-mis | 27.12 | 26.96 | 31.42 | 30.15 | 33.71 | 31.70 |
| 1 | G-wrongmem | 32.08 | 29.96 | 32.42 | 31.73 | 33.72 | 32.04 |
| 1 | G-ones | 26.95 | 27.06 | 31.50 | 30.20 | 33.70 | 31.98 |
| 2 | U | 30.77 | 30.00 | 32.67 | 31.27 | 33.19 | 31.91 |
| 2 | G-oracle | 35.02 | 32.61 | 32.97 | 32.45 | 33.13 | 31.76 |
| 2 | G-est | 34.92 | 32.65 | 33.00 | 32.48 | 33.18 | 31.79 |
| 2 | G-mis | 24.80 | 28.27 | 31.72 | 30.92 | 33.12 | 32.26 |
| 2 | G-wrongmem | 31.35 | 30.32 | 32.71 | 31.47 | 33.22 | 32.01 |
| 2 | G-ones | 24.75 | 28.45 | 31.81 | 31.00 | 33.15 | 31.91 |
| 3 | U | 30.91 | 29.90 | 32.20 | 31.57 | 33.23 | 32.11 |
| 3 | G-oracle | 34.98 | 32.58 | 32.91 | 32.79 | 33.21 | 31.88 |
| 3 | G-est | 35.09 | 32.37 | 32.83 | 32.75 | 33.21 | 31.93 |
| 3 | G-mis | 26.13 | 28.67 | 31.07 | 30.53 | 33.20 | 31.56 |
| 3 | G-wrongmem | 31.09 | 30.23 | 32.20 | 31.53 | 33.21 | 31.98 |
| 3 | G-ones | 26.19 | 28.80 | 31.23 | 30.66 | 33.22 | 32.18 |

Paired within-prefix contrasts (dB; median, min..max over the 4 pairs,
sign consistency):

| contrast | P1 ghost core | P2 return | S1 | H1 pre | H2 whole | C1 control |
|---|---|---|---|---|---|---|
| G-oracle − U | **+4.16 (3.16..4.92) 4/4** | **+2.86 (2.61..3.24) 4/4** | +0.58 (0.30..1.07) 4/4 | +1.20 (0.84..1.27) 4/4 | −0.01 | −0.22 (−0.28..−0.16) 4/4 |
| G-est − U | +4.16 (3.00..4.93) 4/4 | +2.86 (2.47..3.28) 4/4 | +0.53 4/4 | +1.20 4/4 | −0.01 | −0.20 4/4 |
| G-mis − U | −4.58 (−5.98..−4.24) 4/4 | −1.73 4/4 | −0.99 4/4 | −1.12 4/4 | −0.02 | −0.23 3/4 |
| G-ones − U | −4.57 (−6.03..−4.38) 4/4 | −1.61 4/4 | −0.90 4/4 | −1.07 4/4 | −0.02 | 0.00 |
| G-wrongmem − U | **+0.65 (0.17..0.78) 4/4** | +0.33 (0.25..0.90) 4/4 | +0.02 | +0.19 | +0.01 | −0.04 |
| G-oracle − G-est | +0.04 (−0.11..0.16) | −0.04 | +0.03 | 0.00 | 0.00 | −0.02 |
| G-oracle − G-wrongmem | +3.78 (2.38..4.20) 4/4 | +2.32 (2.14..2.99) 4/4 | +0.59 4/4 | +1.04 4/4 | 0.00 | −0.21 4/4 |
| G-oracle − G-mis | +9.01 (7.53..10.22) 4/4 | +4.66 4/4 | +1.51 4/4 | +2.23 4/4 | +0.01 | −0.02 |

Per-frame diagnostics on P1 (descriptive): G-oracle has the lower core
MSE than U on **25 of 25 frames in every pair**; no frame reaches exact
zero error in any arm (the teacher background is never rendered
bit-exactly); max per-frame core PSNR 34.4 / 36.5 / 35.2 / 35.0 (G)
against 32.5 / 32.2 / 31.9 / 32.2 (U).

**Frozen verdict: NOT_MET.** Every G−U pair on P1 clears the 0.5 dB
floor by a factor of six or more, but the pre-registered claim also
requires no sham pair to clear it, and G-wrongmem − U exceeds +0.5 dB
on P1 in three of four pairs (0.77, 0.72, 0.58; the fourth 0.17). The
rule was frozen before any 12k number existed and is applied as
written. Sizing (informational): paired sd 0.73 dB on P1, n2 = 58 pairs
at δ = 0.30; irrelevant at this effect size.

Reading, in order of weight.

1. **The training-time gate delivers the method's claim on real
   footage under an authored absence**: exact absence in the ghost
   window (+4.2 dB on the supported core, 25/25 frames in 4/4 pairs)
   AND a faithful return (+2.9 dB on [92,99], +0.6 dB settled) AND a
   better pre-gap object (+1.2 dB), with whole-frame unchanged. The
   render-time diagnostic of §5 (P1 −5.9 dB on the 6k ungated model)
   is reversed by training with the gate on: the rows behind the
   bottle learn the counterfactual background when the bottle rows
   cannot paint it. The two timing shams (same rows, gap moved) lose
   4.6 dB in the ghost window and 1.7 dB at the return in every pair,
   so the effect is the window, not the code path or the row set.
2. **The random-membership sham is not a null either**, and it is what
   the frozen rule trips on: gating 7,000 random rows (≈1% of them
   bottle rows) during [60,89] gains 0.17–0.77 dB on the core and
   0.25–0.90 dB at the return. The magnitude-matched control shows
   that part of the effect is generic (any 7,000 rows removed during
   the gap take some ghost with them, and the ~70 true bottle rows in
   the random set are not nothing), and the membership-specific share
   is the G − G-wrongmem contrast: **+3.8 dB (2.4..4.2) on P1 and
   +2.3 dB (2.1..3.0) on P2, 4/4.** The rule asked the sham to be
   quiet; it was audible. That is the correct outcome of a control,
   and the verdict stands as NOT_MET under this spec. A wave-2 spec
   could re-declare the claim on the membership-specific contrast, but
   that is a new frozen spec, not a re-reading of this one.
3. **Estimated timing costs nothing here:** G-est (T1's gap [60,91]
   plus the same rows) matches G-oracle to 0.04 dB on P1 and −0.04 on
   P2 in every pair; the two extra gap frames (90, 91) cost 0.2 dB at
   most on the return in one pair. Membership estimation was NOT
   tested (§5: the construction masks are the DEVA silhouettes).
4. **Costs outside the event:** C1 (the untouched control window)
   loses 0.16–0.28 dB in every G and G-est pair; whole-frame is
   unchanged. That is the price of forcing full presence on 7,000 rows
   for 270 frames under the total gate (their temporal marginal is
   overridden), and it is the same size as the G-mis harm there.
5. **Scope, restated:** an oracle-controlled diagnostic on a
   teacher-rendered counterfactual absence (`evidence_bearing:
   false`); the ground truth inside [60,89] is SA4D's background
   render, the red cap was left floating by the edit, and the object
   was hand-selected. Nothing here is evidence about the physical
   world, and the 2026-09-09 render-time negative on the real
   occlusion (−3.4 dB) and Lane A's null (§7A) stay in front of it:
   the gate helps exactly when the object is absent from every camera
   and does nothing when it is merely occluded.

Cost of the fixture lane: 4 prefixes × 2.0 h + T1 2.0 h + votes 4 ×
0.13 h + gate evals 0.5 h + 24 continuations × ~2.5 h + re-evaluation
32 × 0.27 h ≈ 78 A100-h; Lane A ≈ 50 A100-h plus 3.3 h re-evaluation.
