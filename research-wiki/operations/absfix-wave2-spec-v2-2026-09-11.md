---
title: "SPEC v2.0.0 (DRAFT for adversarial review) — wave-2 counterfactual-absence comparison on three N3V scenes; primary = the membership-specific contrast"
date: 2026-09-11
evidence_bearing: false
status: DRAFT — not frozen; no cell may be submitted under it
---

# Spec v2.0.0 — wave-2 counterfactual-absence comparison (DRAFT)

A NEW frozen spec, not a re-reading of v1.2.0
(`configs/n3v/absfix_gate_spec_v1.json`). Wave 1's verdict under v1.2.0
is NOT_MET and stays NOT_MET ([[absence-fixture-lane-2026-09-10]] §7B).
This spec exists because every unresolved point of
[[paper-thesis-v1-kill-argument-2026-09-11]] must become an arm, a
precondition, or an explicit out-of-scope line before any wave-2 cell
trains. Written before any wave-2 prefix exists on any new scene. The
machine-readable instance is `configs/n3v/absfix_gate_spec_v2.json`
(same anchors, endpoints and keys; new keys listed in §9).

## 0. Scope, stated first (kill-argument P_1, P_4, P_5)

* **What is claimed if the rule fires:** on real N3V footage with an
  AUTHORED absence (an SA4D counterfactual edit), training a
  per-primitive multi-interval presence gate on the correct row set and
  window lowers the ghost-core error during the gap and raises the
  return quality, and that gain is SPECIFIC to the membership (a
  count-matched random row set with the same window does not deliver
  it) and SPECIFIC to the window (the same rows with the window moved
  do not deliver it). The claim is descriptive at n = 4 pairs per
  scene and requires the pattern in EVERY scene.
* **Out of scope, stated so it cannot be read in later:** discovery of
  PHYSICAL absence in unedited footage is not claimed; the
  absence-versus-occlusion decision is not claimed (the real-occlusion
  null of [[absence-fixture-lane-2026-09-10]] §7A is reported beside the
  fixture result as evidence that the gate is inert on an occlusion);
  no claim is made about ImViD or any dataset not trained here.
* **Wording rule.** The estimator is a VISIBILITY-GAP estimator (it
  finds a window in which a voxel group's contribution to the training
  views vanishes). "Exact-zero presence" is the mechanism predicate
  (§5); "zero-error frames" is a reported count and is never asserted
  for the fixture, whose gap ground truth is a teacher render.

## 1. Scenes, objects, windows

| scene | object | construction id (cam15 DEVA) | absent frames | margins | control window |
|---|---|---|---|---|---|
| `cut_roasted_beef` (wave 1, reused) | wine bottle | 117 (cam00 136) | [60,89] | 40–109 | [230,259] |
| `flame_steak` | wine bottle | 79 (Lane B, 2026-09-11) | [60,89] default | 40–109 | chosen per §1.1 |
| `sear_steak` | wine bottle | 114 (Lane B, 2026-09-11) | [60,89] default | 40–109 | chosen per §1.1 |

**§1.1 Freezing rule for the new scenes.** Object id, window, margins,
control window, the cam00 evaluation id and the mechanism box are
written into the JSON spec BEFORE the first prefix trains on that
scene, from the Lane B preview/montage evidence alone (per-camera
silhouette area over 40–109 must show the bottle unoccluded in all 20
cameras; the control window must show it visible, unedited and
disjoint from the margins). If either bottle fails the visibility
test, the scene is REPORTED as not admitted and no substitute object
is chosen inside this spec.

**§1.2 Cross-fitting (P_2).** Three roles use three different sources:
construction masks (cam15-referenced DEVA harmonised over 19 training
cameras; SA4D's model) define the EDIT; the truth membership is the
rendered-contribution vote of the EL-GS prefix cloud against those
construction masks (the same instrument as wave 1); the EVALUATION
ROIs come from the held-out cam00 segmenter, which is never used in
construction or membership. The G-est-mem arm (§2) adds a fourth,
independent source for membership.

## 2. Arms (per prefix; every arm is a continuation of the same ungated 6k prefix to 12k)

| arm | rows | gap | role |
|---|---|---|---|
| U | none (reserved parity on) | — | ungated comparator |
| G | truth vote | authored | intervention |
| G-est | truth vote | T1's estimated gap | estimated TIMING share (as wave 1) |
| **G-est-mem** (NEW) | **independent-segmenter vote (§3)** | T1's estimated gap | fully estimated arm: timing AND membership from training-view instruments that never read the construction masks |
| G-mis | truth vote | control window | timing sham (magnitude-matched, wrong time) |
| **G-wrongmem-A** (NEW draw) | count-matched random rows, **zero overlap with the truth set** | authored | membership sham, draw A: the PRIMARY comparator |
| **G-wrongmem-B** (NEW) | second independent zero-overlap draw | authored | membership sham, draw B: the sham spread |
| G-ones | truth vote | late gap [286,297] | code-path sham (as wave 1; a late-gap sham, not a null) |

Eight continuations per prefix, four prefixes per scene, three scenes:
96 cells at 12k, plus prefixes, T1, votes and gate-faithful
re-evaluation (§7). Wave-1 cells on `cut_roasted_beef` are REUSED for
U, G, G-est, G-mis, G-ones; its G-wrongmem (1.0–1.3% truth overlap) is
NOT reused as draw A: two fresh zero-overlap draws are trained on each
existing prefix (8 cells).

**External baseline arm X (P_7).** One compatible per-primitive
temporal-opacity method trained from scratch on each derived scene for
its own published schedule, two seeds per scene, scored with the same
`f_box_profile` on the same windows. Candidate: FreeTimeGS (code
public) — SharpTimeGS has no released training code. X is DESCRIPTIVE
(it cannot be paired by prefix): the paper reports X's P1/P2 beside U
and G. If X cannot be admitted (code, licence, environment), the paper
says NOT RUN; X is never silently dropped and never substituted by our
own ungated arm.

## 3. The independent membership instrument for G-est-mem (P_3)

* **Segmenter S2** must not be DEVA-on-SA4D-labels and must not read
  the construction masks, the SA4D model, or any wave-1 artefact.
  Frozen choice: SAM2 video propagation on each training camera from
  ONE positive click per camera placed by the primary at anchor frame
  50 (clicks recorded in the JSON spec before any vote runs); masks
  over frames 40–109 and the return anchors 92–99.
* **Vote**: the wave-1 rendered-contribution vote (`--id_rule mass`,
  cap 0.5, all 19 training cameras, anchors 50–57 and 92–99) run with
  the S2 masks in place of the construction masks; nothing else
  changes.
* **Precondition, evaluated on the prefix cloud BEFORE any score, per
  prefix, per scene:** against the truth vote of the same prefix,
  per-row **precision ≥ 0.80 AND recall ≥ 0.70**, set size between
  0.5× and 2.0× the truth set, and 2D IoU between S2 and the
  construction mask ≥ 0.70 on cam15 at frames 50 and 95. A prefix
  failing any clause is trained anyway (so the score exists on the
  record) but its G-est-mem pair is EXCLUDED from the
  mechanism-exercised set and that scene's G-est-mem leg is reported
  DESIGN_WITHOUT_POWER; thresholds are never relaxed after a score is
  read. The precision/recall of every prefix is reported whether or
  not it passes.
* **What G-est-mem can and cannot show.** It tests whether a
  construction-independent training-view membership exists at the
  quality the gate needs on THIS kind of object (rigid, static, tall,
  parallax-exposed). It does not test membership on the dog-class of
  object that wave 1 could not even edit (§1.1 of the lane page).

## 4. Endpoints (unchanged from v1.2.0 except the primaries)

P1 ghost core `roi:core` [A+3, B−2]; P2 return [B+3, B+10]; S1 settled
[B+11, B+20]; H1 pre-gap [A−30, A−3]; H2 whole frame; C1 control
[CA, CB]. Anchors A = first absent frame, B = LAST absent frame (v1.2.0
semantics). Per-frame masks required on every P1 frame.

## 5. Preconditions per gated cell (from the SETUP, before any score)

As v1.2.0: gated-row count and box test at the scene's mechanism box
and frame (frame 75, box from the cam00 silhouette union padded 8 px,
written into the JSON before training), zero-presence frames inside
[A+3, B−2] for G, G-est, G-est-mem, G-wrongmem-A/B; G-mis and G-ones
exempt from the window test only; reserved units reported by every
cell and equal across arms within a prefix; `program_match` by lineage
key on the restored state. **Every gated cell is scored ONLY through
`scripts/eval_n3v_gated.py --restore_state`** (or the repaired
`main.py --val` once its integration test has passed on Leonardo,
[[minimal-path-config-2026-09-11]]); a `--val` profile without a
gate-on reproduction check is not a result.

**Zero-overlap draws.** G-wrongmem-A/B row sets are drawn uniformly
from rows OUTSIDE the truth set, count-matched to it; the overlap is
asserted 0 and recorded. Wave 1's ~1% overlap is the reason.

## 6. Analysis and verdict (P_6, P_7)

`scripts/realdata_gate_analysis.py --spec --paired`, extended behind
NEW spec keys so that the frozen v1.2.0 output is unchanged (§9).

* **Primary contrasts:** `G − GWRONGMEM_A` on **P1** and on **P2**.
  Per scene: MET iff every complete pair (4 required) exceeds the fixed
  floor **+0.5 dB** on BOTH P1 and P2, AND the timing-specificity
  contrast `G − GMIS` exceeds +0.5 dB on P1 in every pair, AND the harm
  guard holds (median `U − G` on H2 ≤ 0.15 dB and on H1 ≤ 0.15 dB).
* **Overall verdict:** CLAIM_CONDITIONS_MET iff every admitted scene is
  MET; PARTIAL (reported, not claimed) if at least one but not all;
  NOT_MET if none; DESIGN_WITHOUT_POWER if any scene lacks four
  complete `U/G/GWRONGMEM_A/GMIS` pairs or fails §5 in the
  mechanism-exercised set. The scene is the unit of generalisation;
  pairs within a scene share the event, the teacher and the masks.
* **Secondary (descriptive, every scene):** `G − U` (the gross
  effect); `GWRONGMEM_A − U` and `GWRONGMEM_B − U` (the generic-deletion
  share, with the A/B spread as its replicate floor); `G − GEST` (timing
  estimation share); `G − GESTMEM` and `GESTMEM − U` (the fully
  estimated arm, read only on prefixes passing §3);
  `GONES − U` and `|U − GONES|` (code-path cost); S1, C1, per-frame
  MSE diagnostics and zero-error counts; X beside U and G.
* **Statistics:** descriptive at n = 4 pairs per scene (median, range,
  sign consistency, paired-t interval labelled descriptive); no
  p-value, no equivalence claim; sizing informational.
* **Reading rules:** a G-est-mem leg that fails §3 is reported as
  "membership instrument not admitted on this prefix", never as
  "learned membership does not work"; PARTIAL is reported as PARTIAL;
  the C1 cost and the H1/H2 guards are reported in every scene
  regardless of verdict.

## 7. Cost (Leonardo A100-h, from wave-1 measurements)

Per new scene: 4 prefixes × 2.1 h + T1 2.0 h + votes (truth, est,
S2) 0.5 h + 32 continuations × 2.5 h + re-evaluation 32 × 0.27 h ≈
**101 h**; two scenes ≈ 202 h. `cut_roasted_beef` additions: 8
zero-overlap draws + 4 G-est-mem ≈ 12 × 2.77 h ≈ 33 h. X: FreeTimeGS
at its published schedule, ~6 cells ≈ 30–60 h (unknown until admitted).
Total ≈ 270–300 A100-h against ~131k h remaining; not binding.

## 8. Freeze list (the spec is FROZEN only when every line has a value)

| item | value |
|---|---|
| this page, sha256 | pending (written after the review is folded in) |
| `configs/n3v/absfix_gate_spec_v2.json`, sha256 | pending |
| repository commit | pending |
| analysis extension (`realdata_gate_analysis.py` v2 keys), sha256 | pending — must land with tests that reproduce the wave-1 v1.2.0 output byte-for-byte |
| per-scene object/window/box block in the JSON, for each new scene | pending Lane B evidence (§1.1) |
| S2 click coordinates per camera per scene | pending |
| S2 masks manifest, sha256 | pending |
| programs (truth, est, est-mem, mis, wrongmem-A, wrongmem-B, ones) per prefix, sha256 | pending |
| prefix checkpoints, sha256 | pending |
| evaluation ROI digests per scene | pending |
| X: method, commit, config, schedule, or NOT RUN with reason | pending |

No wave-2 continuation may be submitted while any line reads pending.

## 9. New spec keys (for the JSON instance and the analysis extension)

`spec_version: "2.0.0"`; `scenes: [...]` with per-scene anchors, ids,
`MECHANISM_FBOX`, `MECHANISM_FBOX_FRAME`; `arms` adds `GESTMEM`,
`GWRONGMEM_A`, `GWRONGMEM_B` (GWRONGMEM of v1 retired);
`PAIRED_PRIMARY_CONTRASTS: [["G","GWRONGMEM_A"]]`;
`PAIRED_CLAIM_ENDPOINTS: ["P1","P2"]` (both required);
`PAIRED_TIMING_CONTRAST: ["G","GMIS"]` on P1;
`PAIRED_HARM_GUARD: {"H1": 0.15, "H2": 0.15}`;
`PAIRED_SCENE_RULE: "all_scenes"`;
`MEMBERSHIP_PRECONDITION: {"precision": 0.80, "recall": 0.70,
"size_ratio": [0.5, 2.0], "mask_iou_cam15": 0.70, "frames": [50, 95]}`;
`WRONGMEM_OVERLAP_MAX: 0`; `EXTERNAL_BASELINE: {"method": "FreeTimeGS",
"seeds": 2, "status": "pending"}`; `WORDING: {"estimator":
"visibility-gap", "fixture_absence": "exact-zero presence"}`.

## 10. Codex adversarial review (gpt-5.6-sol, reasoning high, thread 01a08dec…) — 20 defects, verdict DO NOT FREEZE, and every disposition

The §0–§9 draft above is preserved as reviewed. Every accepted
disposition is BINDING and is restated in §11; where §11 and §0–§9
disagree, §11 rules.

| # | defect (paraphrased) | disposition |
|---|---|---|
| 1 | the claim could fire with G worse than U | ACCEPTED: every `G − U` pair must exceed +0.5 dB on P1 and P2 |
| 2 | scene admission is selective; "every admitted scene" can be vacuous | ACCEPTED: all predeclared scenes are required; any non-admission → DESIGN_WITHOUT_POWER |
| 3 | draw B cannot fail the claim | ACCEPTED: `G − A` AND `G − B` > +0.5 dB in every pair on both endpoints; sampler seeds and row hashes frozen before training |
| 4 | count matching is not magnitude matching | ACCEPTED IN PART: a third, LOCAL contribution-matched sham (G-wrongmem-L, §11.2) is added; the count-matched draws stay as the random baseline |
| 5 | independent membership is non-operative | ACCEPTED: the fully estimated arm carries its own claim-bearing verdict (Claim B, §11.5); the thesis records P_3 as "a claim-bearing test with a fail-closed precondition", not as resolved |
| 6 | "cross-fitting" does not remove construction circularity | ACCEPTED: the truth vote is renamed construction-derived membership; inference restricted accordingly; evaluation-mask provenance frozen |
| 7 | `cut_roasted_beef` is not fresh confirmation | ACCEPTED: the confirmatory verdict is computed on the two fresh scenes; `cut_roasted_beef` is the calibration scene, reported separately with its post-hoc-added controls |
| 8 | the analyser cannot compute the verdict | ACCEPTED: a fail-closed spec-driven reducer with tests for every branch is a freeze-list prerequisite |
| 9 | mechanism preconditions unfrozen; shams exempt | ACCEPTED: per-arm thresholds frozen (§11.4); zero-presence frames required inside each arm's OWN gap interior, GMIS and GONES included |
| 10 | ratios without n | ACCEPTED: every ratio carries its integer numerator and denominator; zero denominators fail closed |
| 11 | admission, windows, boxes discretionary | ACCEPTED: numeric admission rule, deterministic control-window rule, box construction and coordinate semantics frozen (§11.1) |
| 12 | SAM2 unfrozen | ACCEPTED: checkpoint hash, config, propagation, click rule, no-retry policy frozen (§11.3) |
| 13 | T1 has no success precondition | ACCEPTED: exactly one interval, temporal IoU ≥ 0.80, onset/offset error ≤ 3 frames, else GEST/GESTMEM legs DWP |
| 14 | prose overstates sham failure | ACCEPTED: wording "at least 0.5 dB incremental benefit over each sham" |
| 15 | window specificity only on P1 | ACCEPTED: `G − GMIS` > +0.5 dB on P1 and P2 |
| 16 | harm guard permissive; C1 harm omitted | ACCEPTED: per-pair ceilings on H1/H2; C1 cost predeclared and a general no-harm claim prohibited |
| 17 | external baseline optional | ACCEPTED AS STATED: competitiveness is OUT OF SCOPE for the verdict; X is a paper requirement with its own frozen admission item and is reported NOT RUN if not admitted; P_7's baseline half is therefore only partially disposed |
| 18 | verdict categories overlap | ACCEPTED: precedence DWP > CLAIM > PARTIAL > NOT_MET (§11.5) |
| 19 | evaluator choice unfrozen | ACCEPTED: one hashed evaluator, `scripts/eval_n3v_gated.py --restore_state`; the repaired `main.py --val` is not used for wave-2 scoring |
| 20 | cost understated | ACCEPTED: ~105 A100-h per new scene before contingency; X unvalidated |

## 11. Spec v2.0.0 — consolidated binding rules after the review

### 11.1 Scenes, admission, windows, boxes

* Predeclared scenes: `cut_roasted_beef` (CALIBRATION; wave-1 cells plus
  the new controls; reported separately, never part of the confirmatory
  verdict), `flame_steak` and `sear_steak` (CONFIRMATORY; both
  required).
* **Admission rule (numeric, from Lane B's per-camera silhouette-area
  table over frames 40–109, before any prefix trains):** on every one of
  the 20 cameras and every frame in 40–109 the bottle's harmonised
  silhouette area is ≥ 60% of that camera's median area over 40–109 and
  ≥ 2,000 px; the id is unique (one connected component per camera at
  frames 50, 75, 100). Failure on either scene → that scene is not
  admitted → the wave's overall verdict is DESIGN_WITHOUT_POWER (no
  substitute object, no substitute window under this spec).
* Absent frames [60,89] (A = 60, B = 89, B = LAST absent frame), margins
  40–109 on both new scenes; not a default but the frozen value.
* **Control window rule:** [230,259] if it satisfies the admission
  criterion on every camera; otherwise the LATEST 30-frame window
  ending ≤ 299 and starting ≥ 130 that does; recorded in the JSON before
  training.
* **Mechanism box:** the union of the cam00 evaluation silhouette over
  frames 63–87, padded 8 px, half-open `[x0, y0, x1, y1)` in render-raster
  pixels; `MECHANISM_FBOX_FRAME = 75`. Evaluation silhouettes come from
  the cam00 DEVA id maps (cam00 is held out of SA4D training and of
  every vote); the id and the job that produced the maps are recorded.
* Ratios everywhere carry integer numerators and denominators.

### 11.2 Arms per prefix (nine continuations)

U; G (construction-derived membership, authored gap); G-est (same
rows, T1 gap); G-est-mem (S2 membership, T1 gap); G-mis (same rows,
control window); G-wrongmem-A and G-wrongmem-B (count-matched,
zero-overlap uniform draws over rows OUTSIDE the construction-derived
set; sampler seed = 1000 × draw index + prefix seed; row hashes
recorded before training); **G-wrongmem-L (NEW, the local
contribution-matched sham):** rows outside the construction-derived set
whose deformed centre projects inside the dilated (+20 px) construction
silhouette on ≥ 8 of 19 training cameras at frames 50 and 95, drawn
greedily by descending rendered contribution until the set's total
contribution mass at frame 50 on cam15 is within ±10% of the
construction-derived set's, recorded with its counts; G-ones (late gap
[286,297]). Wave-1 `cut_roasted_beef` G-wrongmem (1.0–1.3% overlap) is
retired from the verdict and reported as history.

### 11.3 Instruments

* **T1 (visibility-gap estimator)** as wave 1 (16 cells over the [1,99]
  percentile box, 4 training cameras, frames 35–114, `--skip-scoring`).
  **Precondition:** exactly one estimated interval; temporal IoU with
  [60,89] ≥ 0.80; |onset error| ≤ 3 and |offset error| ≤ 3 frames.
  Failure → the GEST and GESTMEM legs of that prefix are
  DESIGN_WITHOUT_POWER; G and the shams are unaffected.
* **Construction-derived membership** (was "truth"): the rendered-
  contribution vote (`--id_rule mass`, cap 0.5, 19 training cameras,
  anchors 50–57 and 92–99) against the construction masks.
* **S2 membership for G-est-mem:** SAM2 (checkpoint file and sha256,
  config, and code commit recorded in the JSON before any mask is
  produced), per training camera, forward propagation from frame 50 to
  109 and a second forward run from frame 92 to 99, seeded by ONE
  positive click per camera placed by the primary on the bottle at
  frame 50 from the raw frame only, coordinates written into the JSON
  BEFORE inference, no retries; a camera whose mask at frame 50 has area
  < 500 px or > 5× the median over cameras is dropped from the vote and
  listed; the vote requires ≥ 12 remaining cameras. Then the same vote
  with S2 masks in place of the construction masks.
  **Precondition per prefix (evaluated before any score):** against the
  construction-derived set of the same prefix, TP/FP/FN counts giving
  precision ≥ 0.80 and recall ≥ 0.70; predicted/truth size ratio in
  [0.5, 2.0]; 2D IoU (pixel intersection/union) between S2 and the
  construction mask on cam15 ≥ 0.70 at frames 50 and 95. A failing
  prefix trains anyway; its G-est-mem pair is excluded from the
  mechanism-exercised set and counted toward Claim B's DWP.
* **Evaluator:** `scripts/eval_n3v_gated.py --restore_state` at the
  frozen hash, over frames 0–299 on cam00, gate-on and gate-off
  profiles, `program_match` by lineage key. Any correction creates an
  append-only version and requires re-rendering every arm of every
  scene.

### 11.4 Preconditions per gated cell (setup, before any score)

Gated rows surviving at 12k ≥ 1,000 (n/N recorded); gated rows whose
projection at frame 75 on cam00 lies inside the mechanism box ≥ 100
(for G-mis and G-ones the same count is taken at the midpoint of THEIR
gap); ≥ 1 frame with presence exactly 0 on a gated row inside the
arm's OWN gap interior (edges excluded), for every gated arm including
G-mis and G-ones; reserved units reported and equal across arms within
a prefix; `program_match` true. A failing cell is reported, included in
the ITT tables, and excluded from the mechanism-exercised set that
carries the verdict.

### 11.5 Analysis, verdicts, precedence

Floor: fixed +0.5 dB. Endpoints P1 (`roi:core`, [63,87]) and P2
([92,99]) both claim-bearing; S1, H1, H2, C1 descriptive except as
guards. All contrasts are within-prefix pairs; per scene the
mechanism-exercised set must contain 4 complete pairs for every
contrast the rule names.

**Claim A (membership- and window-specific benefit of the gate with
construction-derived membership), per confirmatory scene, MET iff in
EVERY pair and on BOTH P1 and P2:** `G − U` > 0.5; `G − GWRONGMEM_A` >
0.5; `G − GWRONGMEM_B` > 0.5; `G − GWRONGMEM_L` > 0.5; `G − GMIS` > 0.5;
AND the harm guards hold in every pair: `U − G` ≤ 0.15 dB on H1 and on
H2; `U − G` on C1 ≤ 0.35 dB (the predeclared accepted cost; no general
no-harm claim is permitted while C1 > 0).

**Claim B (the fully estimated gate), per confirmatory scene, MET iff
all 4 prefixes pass the S2 precondition and the T1 precondition, and in
every pair on both endpoints:** `GESTMEM − U` > 0.5, `GESTMEM −
GWRONGMEM_A` > 0.5, `GESTMEM − GWRONGMEM_L` > 0.5, with the same harm
guards. Fewer than 4 admitted prefixes → DESIGN_WITHOUT_POWER for
Claim B on that scene, reported as "membership instrument not
admitted", never as a negative.

**Precedence (per claim, over the two confirmatory scenes):** any
required-scene non-admission, incomplete pair set, or precondition
failure that leaves < 4 pairs → DESIGN_WITHOUT_POWER; else both scenes
MET → CLAIM_CONDITIONS_MET; else one MET → PARTIAL (reported, not
claimed); else NOT_MET. `cut_roasted_beef` receives the same per-scene
reading labelled CALIBRATION.

**Descriptive, every scene:** `GWRONGMEM_{A,B,L} − U` (the
generic-deletion share and its A/B spread); `G − GEST` (timing
estimation share); `G − GESTMEM`; `GONES − U`, `|U − GONES|`; S1; C1;
per-frame MSE diagnostics and zero-error counts; X beside U and G.
Statistics descriptive at n = 4 (median, range, sign consistency,
paired-t interval labelled descriptive); no p-value; sizing
informational. Wording: "G provides at least 0.5 dB incremental
benefit over each sham"; the estimator is a visibility-gap estimator;
the fixture shows exact-zero presence, not zero error.

**External baseline X:** competitiveness is out of scope for both
claims. X (FreeTimeGS at its published schedule, 2 seeds per scene,
scored with the same profile) is a paper requirement with its own
freeze-list line; if not admitted it is reported NOT RUN with the
reason.

### 11.6 Cost (revised)

Per new scene ≈ 105 A100-h (4 prefixes 8.4 h; T1 2.0 h; three votes
and the S2 masks ≈ 1.5 h; 36 continuations × 2.5 h = 90 h; re-eval 36 ×
0.27 h ≈ 10 h; precondition extraction ≈ 1 h) → ≈ 210 h for the two
confirmatory scenes; `cut_roasted_beef` additions (A, B, L, est-mem on
4 prefixes = 16 cells) ≈ 45 h; X unvalidated (30–60 h if admitted).
Total ≈ 260–320 A100-h plus SAM2 integration; not binding on the
allocation.

### 11.7 Freeze list (amended; the spec is FROZEN only when every line has a value)

| item | value |
|---|---|
| spec text (this page) sha256 | see §12 |
| `configs/n3v/absfix_gate_spec_v2.json` sha256 | see §12 |
| repository commit carrying both | pending (commit after the user's decision) |
| spec-driven fail-closed reducer in `scripts/realdata_gate_analysis.py` with branch tests, sha256 | pending — prerequisite; wave-1 v1.2.0 output must reproduce byte-for-byte |
| G-wrongmem-L draw script, sha256 | pending |
| per-scene admission block (areas table, control window, box, cam00 id and job) | pending Lane B |
| SAM2 checkpoint sha256, config, commit; click coordinates per camera per scene | pending |
| S2 mask manifest sha256 | pending |
| programs per prefix (8 gated arms), sha256 | pending |
| prefix checkpoints, sha256 | pending |
| evaluator `scripts/eval_n3v_gated.py` sha256 | pending |
| X: method, commit, config, schedule, adapter, or NOT RUN with reason | pending |

**No wave-2 continuation may be submitted while any line reads
pending.** This block authorises no training.

## 12. Freeze record (2026-09-11)

| item | sha256 |
|---|---|
| this page, sections 0-11 (the file as it stood before this section was appended) | `21d20ecd52cec13b8fc0d99247b9a840731163e077a452015455171cc4da0856` |
| `configs/n3v/absfix_gate_spec_v2.json` | `eb2c5e6f523eaf8433b067aa5a89c188d3be7132cbab625505055a94a9881d3e` |

STATUS: **TEXT FROZEN v2.0.0**; the freeze list in section 11.7 is OPEN (reducer, draw script, per-scene admission blocks, SAM2 items, programs, prefixes, evaluator hash, X). No wave-2 cell may be submitted until every line has a value; this block authorises no training. Uncommitted pending the user's decision.

## 13. Amendments after the user's approval (2026-09-12, append-only; binding together with §11)

The user approved §11 as written on 2026-09-12 and directed the
following amendments. Where §13 and §11 differ, §13 rules.

### 13.1 S2 seeding takes two recorded clicks per camera

§11.3 seeded SAM2 with one click at frame 50. Amended: **one positive
click per training camera at frame 50 AND one at frame 92**, both
placed by the user from the RAW frames (the Lane B click packet,
`research-wiki/assets/absfix-<scene>-clickpacket-f{50,92}.jpg`), both
written into `configs/n3v/absfix_gate_spec_v2.json` under `S2.clicks`
BEFORE any SAM2 inference, no retries. The frame-50 click seeds the
forward propagation 50 → 109; the frame-92 click seeds the forward
propagation 92 → 99. The camera-drop rule of §11.3 (mask area at the
seed frame < 500 px or > 5× the median over cameras) is applied at
EACH seed frame separately; a camera dropped at either seed frame is
dropped from the vote and listed; the vote still requires ≥ 12
remaining cameras.

### 13.2 The hull-selection rule of fixture construction is a frozen per-scene field

Wave 1 recorded the construction rule in prose only (hull mode,
`--base_rows all`, mask consistency 8 of 19 training cameras at anchor
frames 45/60/75/89/104, DEVA-harmonised cam15 id, `--edit_region deva
--dilate 6 --feather 2`). The 2026-09-11 flame_steak build showed that
this rule can leave part of the object (the bottle cap's Gaussians) out
of the removed row set, so the rule is now a **frozen per-scene field**
`CONSTRUCTION` in the JSON with: `base_rows`, `mask_min_cams`,
`mask_frames`, `mask_source` (the harmonised DEVA id and any extra id
added to the removal set), `edit_region`, `dilate`, `feather`,
`rows_removed`, `build_dir`, `derived_root`, and `preview_evidence`
(the job id and the per-camera cap-band changed-pixel counts that
justified the choice). Rules:

* the two CONFIRMATORY scenes must share ONE rule; the repaired
  flame_steak rule is applied identically to sear_steak (user decision:
  rebuild sear_steak for consistency; the 2026-09-11 `mask_min_cams 8`
  builds of both scenes are preserved, superseded, and never trained
  on);
* `cut_roasted_beef` keeps its wave-1 rule (it is the calibration
  scene); if the confirmatory rule differs from it, the difference and
  the reason are recorded here, and the calibration scene's cap
  remnant stays a declared limitation of the calibration fixture only;
* the field is written from Lane B's preview evidence BEFORE the
  scene's first prefix trains; any later change is a new derived root
  and a new §13 entry, never an edit of the frozen value.

### 13.3 Deferred ablations (recorded, NOT in this block)

* **Deferred-seeding sweep.** Seeding iteration versus bound rows: on
  the real occlusion, seeding at iteration 0 bound 389 rows of 366k
  (grown to ~500 of 600k by 12k) whereas seeding at 6k bound ~3,100
  ([[realdata-gating-lane-2026-09-09]] §6-v2 results;
  [[absence-fixture-lane-2026-09-10]] §3). A post-wave-2 ablation over
  the seeding iteration (0, 2k, 4k, 6k, 8k) on one confirmatory scene
  would measure how much of the gate's effect depends on WHEN
  membership binds. Not in this block.
* **Dense initialisation (VGGT / Depth Anything 3).** Replacing the
  COLMAP sparse initial cloud with a dense feed-forward reconstruction
  (VGGT, or the read-only `depth-anything-3` utility checkout) before
  any densification, on the same protocol, to test whether the
  ghost-core and return effects survive a substrate whose object rows
  are dense from iteration 0. Post-wave-2 ablation; not in this block.

### 13.4 Follow-up recorded, nothing more

Absence-versus-occlusion separation stays OUT OF SCOPE (§0). The
recorded follow-up is the model-transmittance idea of the EL-GS v8
design ([[elgs-method]]): renderer-conditioned censored evidence, in
which a family-present, query-source-excluded transmittance computed by
the current scene model decides whether a training view could have
observed the group at all, so that a vanished contribution is read as
absence only when the model says the group was observable. It is a
design note, not a plan, and no cell is authorised for it.

### 13.5 Hash record for §0–§13.4

Appended below by the freeze step (this page with the record line
removed, and the JSON after the §13 edits). Filling a pending key in
the JSON (admission blocks, CONSTRUCTION fields, S2 clicks, freeze-list
hashes) is the only permitted JSON edit and each filling is recorded as
a further §13 line with the new JSON hash.

| item | sha256 |
|---|---|
| this page, sections 0-13.5 (before this record) | `492bffb285d536b8902fe625efdaf3105ab64b250aa0f9c3fb5da9f1cb315a5d` |
| `configs/n3v/absfix_gate_spec_v2.json` after the section-13 edits | `6f39916ed550e53546039d12a8a71374ab6865deec727225e69550b6f98377c6` |
| `configs/n3v/absfix_gate_spec_v2.json` after line-ending normalisation to LF (content unchanged; the previous line hashed a CRLF working copy) | `735ec3060e5899f25f941a2dd512fb2abf7dfe83745cbff218c118cae0e64195` |

### 13.6 Admission blocks filled, the repair's root cause, and the confirmatory construction rule (2026-09-12)

**Admission (Lane B jobs 57377609/57377610; evidence files
`research-wiki/assets/absfix-{flame_steak,sear_steak}-admission.json`):
BOTH confirmatory scenes ADMITTED under §11.1.** 21 cameras, frames
40–109: worst min/median 0.9803 (flame_steak cam04) and 0.9765
(sear_steak cam14); 0 frames below 0.60 × median, 0 below 2,000 px
(smallest minimum area 4,934 / 4,977 px on cam10), 0 missing masks; one
connected component on every camera at frames 50/75/100; control window
[230,259] passes on its own median and on the main-window median (worst
0.9893 / 0.9915), no fallback. cam00 evaluation ids 65 (flame_steak; mask
job 57252526, id-map job 57254952) and 67 (sear_steak; 57252529 /
57254953); raster 1352 × 1014; mechanism boxes (union of the cam00
silhouette over 63–87, +8 px, half-open) **[957,753,1027,965]** and
**[957,754,1027,962]**; bbox at frame 75 [965,761,1018,957] /
[965,763,1018,953]. Written into the JSON (CA/CB, boxes, ids, jobs,
`admission` block, `event_name` rule: every scene is assembled with
`--event_prefix BOTTLE`, so the bounding-box event is
`BOTTLE_absence_gap`; the reducer needs this name and the frozen JSON
had omitted it, found by Lane C).

**Repair root cause (Lane B, jobs 57377286, 57377527/28/29/41,
57378195/97; `capband_prev_vs_build.json`).** The cap is not a separate
DEVA id (the silhouette includes it). Lowering `mask_min_cams` changed
nothing: the 8-camera PREVIEW is already cap-free (0 red cap-band pixels
on cam00/08/15 at frames 60/75/89), the BUILD at the same vote is not
(283/178/272). `run_build` applies a post-vote IQR outlier box
(`points_inside_convex_hull`, factor 1.0) that `run_preview` does not;
it dropped 240 of 2,230 voted rows on flame_steak and 134 of 2,434 on
sear_steak, and cap-band object-alpha counts identify the dropped rows as
the cap (flame_steak cam00 677 → 441, cam15 829 → 580, cam08 484 → 328;
sear_steak cam00 674 → 674). The mc=7 "_v2" builds of both scenes are
mis-labelled repairs (an mc=7 variant), preserved and never trained on.

**Confirmatory construction rule (frozen here, applied identically to
both scenes, build suffix _v3):** `base_rows all`, `mask_min_cams 8`
of the training cameras (the wave-1 value), `mask_frames
45/60/75/89/104`, DEVA harmonised cam15 id (79 / 114), no extra id,
`edit_region deva`, `dilate 6`, `feather 2`, and
**`--hull_outlier_factor 0` (the post-vote IQR box DISABLED)**, with the
radius filter unchanged. Difference from `cut_roasted_beef` (factor
1.0): recorded here with the reason above; the calibration scene keeps
its wave-1 build. Code: commit exposing the knob (this session), pushed
before the rebuild. The JSON's CONSTRUCTION fields for the two scenes are
filled from the _v3 build outputs.

| item | sha256 |
|---|---|
| `configs/n3v/absfix_gate_spec_v2.json` after the admission fill | `f8c68072e7bc7e5da0d12761fc7faa024138d78816ca11bb231071a4ecf762b8` |

### 13.7 The _v3 construction record and the flame_steak background note (2026-09-12)

Both confirmatory fixtures were rebuilt under the §13.6 rule (Lane B jobs
57382785/87 flame_steak, 57382813/14 sear_steak; records
`research-wiki/assets/absfix-<scene>-construction-v3.json`, copied into
the JSON's `CONSTRUCTION` fields): `iqr_box.applied = false`; rows
removed 2,227 / 2,431 (vote 2,230 / 2,434; radius filter −3 each);
build cap-band red pixels 0/0/0 on cam00/08/15 at frames 60/75/89 (the
originals carry 263–278); verifier exhaustive PASS on 6,300 images per
scene (5,670 sha-identical, 630 edited, all diffs inside mask + 14 px,
raw tree unchanged); events `BOTTLE_*` as in wave 1; cam00 event boxes
[957,753,1027,965] / [957,754,1027,963]; MANIFEST sha256
`a0a9c3690e56085b96105d9465182950146c0986ac4345ab4a458666aa242d7e` /
`e781126645a2d0d898dfa01195f564e5295d5f4fe455a80caad2ae15b7d3c322`.
Derived roots `data_derived/absfix/{flame_steak,sear_steak}_absfix_60_89_v3`.
Montages viewed by the primary: the cap is gone on all 21 cameras in
both scenes.

**Declared limitation of the flame_steak fixture (inside P1's teacher
region, shared by every arm):** the rows the disabled box had spared
also painted the counterfactual background, so the _v3 and _v1 cam00
frame-75 composites differ on 5,682 px (flame_steak) and 6,252 px
(sear_steak), all inside the event box, and a knife segment revealed
behind the flame_steak bottle is crisp in _v1 and blurred in _v3 (zoom
sheets `runs/realdata/absfix2/<scene>/zoom_f75_orig_v1_v3.png`). This
is the shared-row mechanism of [[absence-fixture-lane-2026-09-10]] §5
appearing in the construction; it lowers the P1 ceiling on flame_steak
for all arms equally and is not a difference between arms. Recorded,
not repaired: any further construction change would be a new derived
root and a new §13 entry.

### 13.8 Freeze list — filled as far as this block can fill it (2026-09-12)

Read against §11.7. "user" = the user's decision; "go" = needs the
prefix training, T1 and votes that this block was not authorised to run.

| item | value |
|---|---|
| spec text, sha256 | recorded in the hash record below (this page with the record lines removed) |
| `configs/n3v/absfix_gate_spec_v2.json`, sha256 | recorded below |
| repository commit carrying both | the commit that adds this section (see `git log`); Leonardo fast-forwarded to it before any cell |
| reducer `scripts/realdata_gate_analysis.py` | commit 34cc193, sha256 `a44dcc333b921fd502f9bb661593c689f1bcdc694ce23840ac3e8862a207b64f`; 55 v2 tests + wave-1 reproduction fixture |
| sham draw script `scripts/draw_membership_shams.py` | commit 2bc5acf, sha256 `0758048da4f3c8d33e9f73e945c7345b548e418e1f0d0fdf344cd8df258e9485`; vote script with `--emit_row_weights` sha256 `69d1a2b7fd7ac4890ec1ff65ff30fc38c4d397044b9d4aa5cde15495f48f5905` |
| per-scene admission blocks | FILLED (§13.6; both scenes admitted; jobs 57377609/10) |
| per-scene construction fields | FILLED (§13.7; _v3 builds; `--hull_outlier_factor 0`; `sa4d_absence_edit_render.py` sha256 `7648de041d845ef9ab580ed817200ecad0fa0ec5befec20b580ca81e7e4076b0` at commit 3d95d3e; assembler sha256 `15994bbf29b0a7881ee756f0da35e0072a8003c78ba7676bb03380c3f10c71f8`) |
| SAM2 checkpoint sha256, config, commit | FILLED: `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`, `configs/sam2.1/sam2.1_hiera_l.yaml`, facebookresearch/sam2 `2b90b9f5ceec907a1c18123530e92e794ad901a4`; driver `scripts/s2_sam2_propagate.py` sha256 `4f874ed4160c6d9c7863d928889b4de5d80ec2273b71961f0e92e36f8028f752` (commit 1fc8392); smoke 57383718, probe 57383950 |
| S2 click coordinates per camera per scene | **PENDING (user)**: packet delivered (`absfix-<scene>-clickpacket-f{50,92}.jpg`), template `absfix-s2-clicks-template.json`; the probe's guidance (shoulder above the label) is information, not a placement |
| S2 mask manifest sha256 | PENDING (after the clicks; one GPU inference job per scene, ~minutes) |
| T1 estimator | `scripts/estimate_episodes.py` sha256 `c90416df0901708350171a065afd2622d413f80d86f95c81bdd6f8639b56362d` (unchanged since wave 1); T1 runs on the new scenes PENDING (go) |
| programs per prefix (8 gated arms), sha256 | PENDING (go): needs the four ungated 6k prefixes per scene, T1, the three votes and the sham draws |
| prefix checkpoints, sha256 | PENDING (go): 8 new prefixes (2 scenes × 4 seeds, ~2.1 h each) |
| evaluator `scripts/eval_n3v_gated.py` sha256 | FILLED: `4bdf427fa59264125843b0523b4466b2200383eb76eacf6d196679681cc1b927`; `main.py --val` verified equal to 4 decimals (job 57378507) but not used for scoring |
| X external baseline | NOT ADMISSIBLE as FreeTimeGS (no released training code; ++ 404; RetimeGS no code); **PENDING (user)**: accept SpacetimeGaussians as the descriptive baseline, or record X as NOT RUN |
| pre-existing test failure noticed | `tests/test_gate_cell_precondition.py::ConsumerContract::test_read_precondition_recovers_all_eight_fields` (a ninth field `arm_kind`); not touched; not a freeze item |

Three pending kinds remain: the user's clicks and the X decision, and
the GPU prerequisites (prefixes, T1, votes) that produce the programs
and checkpoint hashes. The programs cannot exist before the prefixes
train, so "every line has a value" is reachable only after a "go" that
covers the prefixes, T1 and votes as a first stage, with continuations
withheld until the last hash is recorded.

| item | sha256 |
|---|---|
| this page, sections 0-13.8 (before this record) | `e5a6cfa56232762c11d34d433711f1fb159225093dfdb9d07b4c1bb715347a09` |
| `configs/n3v/absfix_gate_spec_v2.json` after the S2/evaluator/baseline/construction fills | `062c8ebd0495aa28e776a1aa52554e3b5745d35580985d7be2d4623b11f955be` |

### 13.9 S2 click procedure amended, one bounded re-seed, X fixed to SpacetimeGaussians, stage 1 authorised (user decision 2026-09-13; appended BEFORE any SAM2 inference)

**(a) Who places the clicks.** §13.1 had the user place the clicks. Amended:
the S2 clicks are placed by the PRIMARY (the orchestrating agent), not by
the user, from click-packet sheets REGENERATED WITHOUT the cyan
construction boxes; the placer never views a DEVA or SA4D mask at any
point of the placement. The crop windows of the regenerated sheets reuse
the packet's per-camera origins (a window around the bottle; no mask or
box is drawn inside it) so that the sheets remain comparable; the click
is placed on the raw pixels only. One click per training camera at frame
50 and one at frame 92, on the dark glass shoulder above the label (the
2026-09-12 probe showed a shoulder click returns the whole bottle, a
label click only the label, a low-body click the wine glass). All
coordinates (2 × 20 cameras × 2 scenes = 80) are written into
`S2.clicks` in the JSON BEFORE inference, with the sheet files they were
read from named beside them.

**(b) One bounded re-seed.** After the first propagation run, every
camera's SAM2 mask is checked ONCE, by eye, against the RAW frame only
(never against a DEVA/SA4D mask), with the single criterion "the mask
covers the bottle from cap to base". A camera whose mask fails the
criterion at a seed frame receives exactly ONE second click at that seed
frame; the second click replaces the first for that camera and frame;
propagation is re-run for that camera only; no third attempt exists. The
manifest records, per scene and per seed frame, which cameras were
re-seeded, both click coordinates, and the count; the count is reported
beside the precondition. The precondition thresholds of §11.3 are
unchanged (precision ≥ 0.80, recall ≥ 0.70, size ratio [0.5, 2.0], cam15
IoU ≥ 0.70 at frames 50 and 95); a camera that still fails after its
second click is kept in the vote unless the §11.3 camera-drop rule
removes it.

**(c) X = SpacetimeGaussians** (STG; public code; per-primitive temporal
opacity; published `cut_roasted_beef` 33.52), confirmed by the user.
Descriptive only, outside both claims; two seeds per derived confirmatory
scene at STG's published N3V schedule, scored with the same profile;
runs alongside stage 2. The pinned commit and schedule are recorded in
the JSON (`EXTERNAL_BASELINE`) and in the hash record below.

**(d) Stage 1 authorised (no gated cell):** the eight ungated 6k prefixes
on the `_v3` derived roots of flame_steak and sear_steak (four seeds
each, `b0c_crb300_6k_rp_prefix.yaml`), T1 per prefix, the
construction-derived vote per prefix, the row-weight dump, the three
sham draws per prefix, SAM2 propagation under (a)–(b), the S2 vote per
prefix, both preconditions, and every hash of §13.8. Stage 2
(continuations) waits for the user's second "go" after the fully valued
freeze list is shown.

**(e) In parallel, CPU only:** a plate-candidate census on all three
scenes for a possible v3 fixture (a moving object whose absent-window
footprint is object-free and hand-free at some other time on every
camera), scoring each DEVA id's segmentation stability across the
window as well as plate availability; ranked montage for the user's
choice; the user decides the object. Recorded here as intent; a plate or
insertion fixture is a NEW frozen spec (v3) with its own kill-argument
pass. The bottle fixtures stay as the static-object case. The SA4D
background fine-tune is not done.

| item | sha256 |
|---|---|
| this page, sections 0-13.9 (before this record) | `af83c8813fbd8895902bcb5141afda5d6d5da1f00e4c72b0ff60b73901a19cea` |
| `configs/n3v/absfix_gate_spec_v2.json` after the 13.9 edits (STG commit and clicks still pending) | `e7790bc7af3bae55a13d12281c4ddfe1b9eff6c5a787e83c765bc243789d4a7a` |

### 13.10 S2 clicks recorded before inference (2026-09-13)

All clicks placed by the primary under 13.9(a) from the box-free sheets `research-wiki/assets/absfix-<scene>-clickpacket-nobox-f{50,92}.jpg` (no DEVA/SA4D mask viewed): 20 cameras × 2 frames on flame_steak and sear_steak, 19 × 2 on cut_roasted_beef (cam04 absent from its derived scene), 118 clicks, one per camera per seed frame on the dark glass shoulder above the label; the bottle is static, so the frame-92 click repeats the frame-50 position after the frame-92 tile was viewed. Written to `S2.clicks` in the JSON with `S2.clicks_sha256`; no SAM2 inference has run on them yet.

| item | sha256 |
|---|---|
| `configs/n3v/absfix_gate_spec_v2.json` with the S2 clicks | `b333a914a8727360f3b270efcc033b7f4ec96f481e0a8b9b9bca8bed94c9668b` |

### 13.11 S2 run 1 and the bounded re-seed (2026-09-13)

Run 1 (jobs 57485941/42/43, driver on the recorded clicks): kept 19/20, 19/20, 19/19 cameras; cam19 dropped by the area rule on both _v3 scenes (55,670 / 55,986 px: the click had landed on the wall left of the bottle). By-eye check under 13.9(b) on the raw-frame sheets `s2-<scene>-run1-check-f{50,92}.jpg`, criterion "cap to base": failures = flame_steak cam19, sear_steak cam19 and cam09 (neck and shoulder only, 1,770 px), cut_roasted_beef cam17 (neck only, 1,997 px); every other camera passes at both seed frames. Second clicks (one per camera per seed frame, recorded in `S2.reseeds` and substituted in `S2.clicks` with the first click kept beside them): cam19 → [1058,822] on both scenes; sear cam09 → [805,908]; crb cam17 → [906,766]. Re-seed counts: flame_steak 1 camera × 2 frames, sear_steak 2 × 2, cut_roasted_beef 1 × 2. No third attempt exists.

| item | sha256 |
|---|---|
| `configs/n3v/absfix_gate_spec_v2.json` after the re-seed record | `54bf5ae0f223d0607ef5efd762acd34357b338892de77cb61174339a4ae30adc` |

### 13.12 X pinned; plate census NEGATIVE, insertion fixture selected for v3 (2026-09-13)

**X = SpacetimeGaussians pinned:** commit `427abfc58309a4a5213843dd673fb22c4529306c` (MIT with the Gaussian-Splatting use limitation); published full schedule 30,000 iterations, batch 2, densify 500–9,000, metric at 25,000, one GPU, per-frame COLMAP init; on the derived 1352 × 1014 frames the config uses `resolution: 1` (the published `resolution: 2` on native 2704 × 2028 gives the same raster; verified intrinsics f = 731.03). Installed (five CUDA extensions, job 57485725), two recorded patches (SSIM import; a seed flag, since STG hard-codes seed 0), flame_steak frames 0–49 preprocessed (57485385), 200-iteration dry run and cam00 render verified at the raster (57488278, 57489307). Estimated 30–42 A100-h for 2 seeds × 2 scenes; runs with stage 2; descriptive only.

**Plate census (jobs 57485507, 57486262, 57487805; CPU): NEGATIVE on all three scenes.** Rows passing stability and size / plate-possible / passing the cam15 plate test: cut_roasted_beef 1,026 / 8 / 0; flame_steak 1,331 / 9 / 5; sear_steak 958 / 0 / 0. The one id passing the cam15 plate test (flame_steak 127) is the flame above the pan and its cam00 counterpart is the cook (montage viewed). Binding cause: DEVA segment granularity differs across cameras by 10–30×, so a small object cannot be tested for plates on 21 cameras by this route. Consequence, per the user's 2026-09-13 rule: the INSERTION fixture is the v3 candidate (new frozen spec with its own kill-argument); nothing about wave 2 changes.

| item | sha256 |
|---|---|
| `configs/n3v/absfix_gate_spec_v2.json` after the X pin and census record | `0e80b11709889b7933f8bdfdbfd6c5acd4ac7a046925dfe0379b350437fcd19c` |

### 13.13 Stage-1 facts recorded before any score (2026-09-13, morning)

* **The local contribution-matched sham (G-wrongmem-L, §11.2) is NOT constructible as frozen.** On the calibration prefix the locally eligible non-truth rows (deformed centre inside the +20 px construction silhouette on ≥ 8 cameras at frames 50 and 95) carry 767.3 units of contribution mass at frame 50 on cam15 against 5,496.9 for the construction-derived set, so the ±10% mass band is unreachable and `draw_membership_shams.py` refuses it (smoke job 57487916, rc 2). The cause is structural: the vote assigned the high-contribution rows to the truth set, so the local remainder cannot carry the truth mass. Draws A and B are emitted by a separate fallback (`ab_only/`); no threshold was changed; the L draw was never emitted. Disposition is the user's: see the recommendation in the block report (a local, count-matched, zero-overlap draw greedy by contribution, with its mass ratio reported descriptively, replaces the mass-matched L; the user decides before stage 2).
* Also found: `draw_membership_shams.py --gap` compares against [offset, onset] = [60, 90] rather than the absent window [60, 89] its help text names; the stage-1 chain omits `--gap` so the shams inherit the truth program's gap unchanged. To be fixed in code before stage 2 (no scientific effect).
* **S2 final masks (run 1 + one bounded re-seed):** every camera kept after the re-seed: flame_steak 20/20 (cam19 re-seeded), sear_steak 20/20 (cam09, cam19), cut_roasted_beef 19/19 (cam17); re-seed counts per seed frame 1/1, 2/2, 1/1.
* **Membership precondition on the calibration scene, prefixes 0–2 (S2 vote vs construction-derived vote, jobs 57488990/57492151/57489429):** precision 1.0000 / 0.9996 / 0.9999; recall 0.9890 / 0.9904 / 0.9882 (TP 6,927 / 6,914 / 6,777; FP 0 / 3 / 1; FN 77 / 67 / 81; truth_n 7,004 / 6,981 / 6,858); size ratio 0.989 / 0.991 / 0.988; cam15 IoU 0.9386 (f50) and 0.9349 (f95). All three PASS §11.3. Prefix 3 pending (its S2 vote stalled twice at import and is being resubmitted).
* **Chain state at 03:55 CEST:** flame_steak prefixes 0–3 and sear_steak prefix 1 COMPLETED (~2 h each); sear_steak prefixes 0, 2, 3 hung with empty run dirs for 3 h 45 and are being cancelled and resubmitted with their dependents; T1 running on the completed prefixes; votes, eligibility, draws, S2 votes and preconditions queued by dependency.

### 13.14 Run-2 outcomes by eye (2026-09-13)

Second clicks, checked once against the raw-frame sheets `s2-<scene>-run2-check-reseed.jpg`: flame_steak cam19 and sear_steak cam19 now cover the whole bottle (7,381 / 7,654 px at frame 50); sear_steak cam09 selected the countertop (23,627 px) and cut_roasted_beef cam17 the label only (2,590 px). Both failures are KEPT in their votes under 13.9(b) (the area rule does not remove them; no third attempt). The membership precondition is the guard: on the calibration scene it PASSED on prefixes 0–2 with cam17's label-only mask included (§13.13). Recorded in the JSON under `S2.run2` and `S2.membership_precondition_results`.

| item | sha256 |
|---|---|
| `configs/n3v/absfix_gate_spec_v2.json` after the run-2 record | `8e286d90fb2ba1f763ac4fb709a4185cace02f3b6888f6ec19dfb514cca5639d` |

### 13.15 Two clarifications made before any score (2026-09-13, 04:40 CEST)

* **The vote's spatial seed is an authored per-scene 3D box; T1 is never an input to the vote.** The stage-1 chain seeded the calibration scene's votes from the wave-1 authored box (`--seed_bbox3d -1.0604 1.7081 -2.5094 -0.4 2.4219 -1.6656`) but the two confirmatory scenes' votes from each prefix's own T1 program. On flame_steak prefix 3 the T1 leading group estimated offset 62 / onset 94 and both votes admitted zero rows (jobs 57485406, 57488908, `ContractError`), which would have made G itself depend on T1. §11.3 gives T1 one role, the TIMING of G-est and G-est-mem. Frozen now: every vote on every scene is seeded from an authored box derived from that scene's construction the same way as the calibration box (the box is a construction-derived region of interest shared by the construction-derived vote and the S2 vote; the S2 independence claim of §11.3 concerns the MASKS, which the box does not touch), recorded in `seed_boxes.json` with its derivation and a projection sanity check; the eight confirmatory-scene vote chains are cancelled and resubmitted under this rule; T1-seeded outputs are kept, never used.
* **The T1 precondition's "exactly one estimated interval" is read on the POOLED program.** T1 gates several voxel groups whose gaps differ by a frame or two (calibration prefix: [60,89] and [60,91]); the program emitted and trained with is their union (wave 1's G-est used [60,91]). The per-group table is reported; the pass/fail is computed on the pooled interval (`pass_union_reading`). The reducer's per-group reading of the calibration prefix (2 intervals → fail) is therefore not the binding one. No threshold changes: pooled IoU ≥ 0.80, |onset error| ≤ 3, |offset error| ≤ 3. On flame_steak prefix 3 the pooled window is expected to fail the onset bound (94 vs 90); if so its G-est / G-est-mem legs are DESIGN_WITHOUT_POWER as §11.3 says, and G is unaffected.

### 13.16 Correction to 13.15, the seed-box rule adopted, and the sear_steak S2-vote consequence (2026-09-13, 06:10 CEST)

* **Correction (append-only): the calibration box was NOT construction-derived.** [[absence-fixture-lane-2026-09-10]] §4 records it as the union of the two voxel cells blind T1 gated on prefix 0 (job 57132256), reused as an authored constant for all four prefixes. §13.15's premise was wrong; its rule stands with that correction: the calibration scene keeps its wave-1 box (its votes already ran with it), and the confirmatory scenes use a T1-free, construction-derived rule: rows of a reference prefix cloud whose deformed centre projects inside the undilated construction silhouette on every training camera at frames 50 and 95, box = per-axis [1,99] percentile of their canonical xyz padded 0.05 per face, authored once per scene from prefix 1. Calibration of the rule against the wave-1 box on the same prefix and masks: in-box rows inside the S2 cam15 frame-50 mask 0.9340 (rule) vs 0.9363 (wave-1 box); flame_steak 0.9050, sear_steak 0.9281; the rule on flame_steak prefix 3 reproduces the prefix-1 box within 0.0033 per face. A first candidate rule (+20 px, ≥ 8 cameras, job 57508186) failed its own acceptance (0.332 in-mask) and was replaced before any box was used (job 57508564). Boxes and checks in the JSON (`VOTE_SEED_BOXES`) and `seed_boxes.json` (sha256 5fefd68c…). All eight confirmatory vote chains resubmitted under this rule into `*_seedbox` dirs (ids in `chain_seedbox.txt`); the T1-seeded outputs are kept and never used.
* **Result of the re-seeding:** flame_steak prefix 3, whose T1-seeded votes admitted zero rows, now admits 6,641; flame s1/s2 6,584 / 6,648; sear s1 7,434; LOCO Jaccard ≥ 0.998 on all. S2 votes on flame_steak prefixes 0/2/3 admitted 6,434 / 6,515 / 6,497; membership precondition on flame_steak prefix 2 PASSES (precision 1.0000, recall 0.9800, TP 6,515, FP 0, FN 133, truth 6,648).
* **sear_steak S2 vote refuses on every prefix** (`ContractError: no DEVA id met the mass bar for cam09 frame 50`, jobs 57488910 and 57509087; independent of the seed box). Cause: the cam09 countertop mask kept under 13.9(b). Under §11.5 as frozen, Claim B on sear_steak reads DESIGN_WITHOUT_POWER ("membership instrument not admitted"); Claim A is unaffected. Whether to amend the camera-drop rule (a mask-area plausibility band would have removed a 2.95× median mask that the 5× rule kept) is the user's decision; it is recorded here as a v3 lesson either way.
* The calibration scene's fourth membership precondition also passes (prefix 3: precision 0.9996, recall 0.9905, IoU 0.94; job 57504225): all four calibration prefixes admit the S2 instrument.
* The L-limb refusal recurs on flame_steak (eligible local mass 2,220.5 vs truth 5,724.2 on prefix 1), consistent with §13.13.

| item | sha256 |
|---|---|
| `configs/n3v/absfix_gate_spec_v2.json` after the seed-box and precondition records | `3c83f7a7ecbf99d272ac676614bec4c11fdbfb64cda5c0362ec9860e22095b83` |

### 13.17 Freeze list, fully valued as far as stage 1 can value it (2026-09-13, 08:40 CEST)

Stage 1 is terminal (all chains; warden and collector reports in
[[absfix-wave2-stage1-2026-09-13]]). Machine record:
`research-wiki/assets/absfix-stage1-record.json` (tracked from this
commit), table `absfix-stage1-record.md`, seed boxes
`absfix-seed-boxes.json`; the JSON carries the same values under
`FREEZE_LIST_STATUS`.

| item | value |
|---|---|
| spec text / JSON | hashes in the record line below; commit = the one carrying this section |
| reducer | `a44dcc33…` (34cc193) |
| draw script | `0758048d…` (2bc5acf, ran stage 1) and `442231ae…` (3f09120, `--gap` fix; identical outputs since draws omit `--gap`) |
| vote script | `69d1a2b7…` |
| evaluator | `4bdf427f…` |
| T1 estimator | `c90416df…`; 8/8 reports and programs hashed; pooled windows flame [60,89] / [60,90] / [60,89] / [60,93], sear [60,89] / [60,89] / [57,106] / [60,90] |
| admission and construction blocks | FILLED (13.6, 13.7) |
| SAM2 | checkpoint `2647878d…`, config `sam2.1_hiera_l.yaml`, commit `2b90b9f5…`, driver `4f874ed4…`; 118 clicks + 4 re-seeds; final manifests hashed (flame `f0fba4a7…`, sear `23601321…`, crb `ab0feb99…`) |
| seed boxes | `5fefd68c…` (13.16) |
| prefix checkpoints | 8/8 hashed |
| G / G-mis / G-ones programs | 8/8 hashed; members flame 6,582 / 6,584 / 6,648 / 6,641, sear 7,676 / 7,434 / 7,656 / 7,536 |
| G-wrongmem-A / B | 8/8 hashed; count-matched, overlap 0 |
| **G-wrongmem-L** | **0/8, not constructible as frozen** (eligible local mass 0.37–0.41 of truth); **user decision pending** |
| G-est-mem programs | flame 4/4, calibration 4/4, **sear 0/4** (S2 vote refuses on cam09, 13.16) |
| membership precondition | flame 4/4 PASS; calibration 4/4 PASS; sear not computable |
| T1 precondition (pooled) | flame s0–s2 PASS, s3 FAIL; sear s0, s1 PASS, s2 FAIL ([57,106]), s3 PASS on the pooled reading; calibration 4/4 PASS |
| Claim B admission under §11.5 | **DWP on flame_steak (prefix 3) and on sear_steak (instrument not admitted)**; calibration admitted |
| X | SpacetimeGaussians `427abfc5…`; trains with stage 2 |

**Every line the frozen rules can value is valued. Three lines are
decisions, not measurements, and stage 2 waits for them:** (1) the
G-wrongmem-L rule; (2) whether Claim B stays DESIGN_WITHOUT_POWER on
both confirmatory scenes as §11.5 reads, with GESTMEM arms still trained
where admitted and reported descriptively; (3) the stage-2 go.

| item | sha256 |
|---|---|
| this page, sections 0-13.17 (before this record) | `1a4e612f3cd1df53d3226009d1410bcecf70197fe9f966e1097bff2e3d41d95c` |
| `configs/n3v/absfix_gate_spec_v2.json` with FREEZE_LIST_STATUS | `11341cad22823e012888060037d05912868270df772cf0b6fdc3624ee506406b` |

### 13.18 §11.2 amended: G-wrongmem-L keeps mass matching and relaxes locality by radius (user decision 2026-09-13; appended BEFORE any L program is drawn)

§11.2 defined G-wrongmem-L as rows OUTSIDE the construction-derived set
that project inside the +20 px construction silhouette on ≥ 8 cameras,
drawn greedily by descending contribution until the mass at frame 50 on
cam15 is within ±10% of the truth mass. §13.13 and §13.17 recorded that
this set cannot reach the band on any of the 8 confirmatory prefixes:
the locally eligible non-truth rows carry only 0.37–0.41 of the truth
mass (flame 0.41 / 0.39 / 0.39 / 0.37; sear 0.38 / 0.40 / 0.37 / 0.40).
**That number is a finding and stays on the record: the construction-
derived vote captures most of the paint around the object.**

The user rejected a count-matched local replacement because it gives up
the magnitude matching the Codex review asked for (a count-matched local
draw removes less than half the paint, so a win over it could be read as
"removing more paint near the object"). **Amended rule, binding:**

* G-wrongmem-L = non-truth rows taken in order of 3D distance from the
  centroid of the truth set (canonical `_xyz`, the frame the vote and the
  seed boxes use), greedy by descending contribution, until the
  contribution mass at frame 50 on cam15 is within ±10% of the truth
  mass. Concretely: the radius R is the smallest distance at which the
  non-truth rows within R carry at least (1 − 0.10) × the truth mass;
  within R the rows are taken by descending contribution (ties by row
  index), skipping a row that would overshoot (1 + 0.10) × the truth
  mass, stopping once the lower edge is reached. Zero overlap with the
  truth set is asserted. The draw is deterministic (no seed).
* Recorded per prefix in the counts sidecar: the radius reached, the
  number of non-truth rows within it, the drawn count and mass, the
  truth mass, and the local-only mass ratio of the retired rule
  (0.37–0.41) as the finding. Draws A and B are unchanged.
* The reducer's Claim A rule (§11.5) is unchanged: `G − GWRONGMEM_L` on
  both endpoints in every pair.
* Code: `scripts/draw_membership_shams.py` gains the radius mode (the
  retired local mode is kept, off by default) with tests; the `--gap`
  check defect of §13.13 was fixed separately in commit 3f09120; the A/B
  programs of stage 1 are re-emitted under the fixed script with `--gap
  60 89` and their hashes compared to the stage-1 hashes (expected
  identical; the draws are seeded and the fix touched only the check).

| item | sha256 |
|---|---|
| this page, sections 0-13.18 (before this record) | `24501477306d6a8948b922b8657b855cfcde9700bee3ca0547d7f537ce16e5b3` |

### 13.19 Recount against the frozen boxes, the audit of T1-derived quantities, and why the radius L rule also refuses (2026-09-13, 14:40 CEST)

**Audit of §13.15 (user request).** Exactly one quantity in the stage-1
setup was T1-derived: the calibration scene's vote seed box
`[-1.0604, 1.7081, -2.5094] .. [-0.4, 2.4219, -1.6656]`, which is the
union of the two voxel cells blind T1 gated on wave-1 prefix 0 (job
57132256) and was reused as an authored constant for the four calibration
votes in wave 1 and in stage 1 (both the construction-derived and the S2
votes). No confirmatory-scene quantity is T1-derived: their seed boxes
follow the construction-derived rule of §13.16; the frozen mechanism
boxes of every scene are the cam00 silhouette unions of §13.6 (JSON
`MECHANISM_FBOX`), which never read T1. §11.5 is NOT amended to accept
the T1-derived box; instead every program of every prefix was recounted
against the frozen boxes.

**Recount (jobs 57556801/02/04; `research-wiki/assets/absfix-box-recount.md`;
integer counts, rows gated / rows whose deformed centre projects inside the
frozen box on cam00 at frame 75, or at 245 / 291 for G-mis / G-ones):**
G, G-est, G-est-mem (where it exists), G-mis and G-ones put ≥ 0.9996 of
their rows inside the box on every prefix of every scene (all PASS the
≥ 100 clause; e.g. sear_steak G 7,676 / 7,434 / 7,656 / 7,536 rows,
≥ 0.9996 in box). So the calibration reading survives the audit: its
programs pass the frozen box exactly as the confirmatory ones do.
**G-est-mem on sear_steak stays MISSING** (no S2 program: the vote refuses
on cam09, §13.16), so Claim B on sear_steak stays DESIGN_WITHOUT_POWER;
the recount could not admit it because the instrument itself is not
admitted, which is a mask question, not a box question.

**The count-matched uniform shams fail the box clause by construction:**
A and B place 46–103 rows in the box (23 of 24 cells below 100; sear s3 A
= 103). A uniform draw over the cloud is non-local by definition; §11.4
applies an object-locality test to a control whose purpose is not to be
the object. Recorded here as a category error in §11.4 for the user to
dispose of (recommended: A and B are exempt from the in-box clause only,
keeping the row-count and zero-presence clauses, as G-mis and G-ones are
exempt from the window clause). Not changed.

**The radius L rule of §13.18 ALSO refuses on all 12 prefixes, and
locality is not the cause.** With the frozen measure (each row's rendered
contribution INSIDE the object silhouette on cam15 at frame 50, the
archive column `w_in`), all non-truth rows of the whole cloud together
carry only 0.21–0.48 of the truth mass (calibration 0.21–0.25, flame
0.45–0.48, sear 0.44–0.46), so the ±10% band is unreachable at any
radius (verbatim: "all non-truth rows together carry 2732.7 of
contribution mass against the truth set's 5667.53; the +-10% band
[5100.78, 6234.28] is unreachable at any radius"). This is a property of
the measure: only rows that paint the object's silhouette carry mass in
it, and the vote assigned those rows to the truth set. The retired
local-only ratios reproduce (flame 0.41 / 0.39 / 0.39 / 0.37, sear
0.38 / 0.40 / 0.37 / 0.40).

**Feasibility on the total-contribution measure (`w_total`, each row's
rendered contribution to the whole cam15 view at frame 50; login-node
numpy, three prefixes):** the truth set's `w_total` is 5,831 / 5,744 /
5,681 (flame s1 / sear s1 / calibration s0), of which 98% is inside its
silhouette; the non-truth cloud carries 168–172× that mass; the radius
at which non-truth rows reach 0.9× the truth mass is **0.707 / 0.722 /
0.740 units** from the truth centroid, enclosing 5,597 / 6,148 / 4,928
non-truth rows, against truth rows whose own distances have median 0.20
and maximum 0.53. A mass-matched draw on `w_total` therefore exists, is
as local as the cloud allows (a shell just outside the bottle), and
removes the same amount of PAINT from the view as the truth set, which
is the magnitude the Codex review asked to match. The decision to change
the measure is the user's; the script gained `--contribution_key` so the
measure is an explicit, recorded choice (every counts sidecar names it).

**Script defect found and fixed (commit after this section):** the xyz
loader flattened arrays, so `--l_mode radius --xyz` refused every input
from the CLI ("xyz has shape (1798695,)"); the radius tests passed
because they bypassed the loader. Fixed with CLI-path tests.

**A/B re-emission under the fixed `--gap` check (job 57556521):** row
sets identical to stage 1 on all 16 programs (`row_ids_sha256` equal);
file hashes differ by exactly one key, `source.wrongmem.emitted_by`
(stage 1's fallback wrote it, `draw_all` does not). The stage-1 A/B
programs stay the frozen ones.

Calibration inputs added for the 16 calibration cells: row weights
(votes re-run with the dump, member sets identical to wave 1 on all four:
7,004 / 6,981 / 6,858 / 6,813), local eligibility, canonical xyz (sha256
equal to each T1 program's `cloud.xyz_sha256`, 12/12).

| item | sha256 |
|---|---|
| this page, sections 0-13.19 (before this record) | `c55a66905f81f0f071d0aa5e3c007b69145b569c225d932f79eae85b9b05e253` |

### 13.20 Provisional L draws on the total-contribution measure (2026-09-13, 15:20 CEST; NOT frozen, decision pending)

Job 57559108 (CPU, 39 s), script version `4ea716ad…` (commit cb6dd1c),
`--contribution_key w_total --l_mode radius --gap 60 89`, outputs in
`draws_prefix<S>_v3prov/` (never used for training unless the user
freezes the measure); summary `research-wiki/assets/absfix-draws-v3prov.json`.

| scene | prefix | radius reached (units) | non-truth rows within | rows drawn | truth rows | mass ratio | overlap | local-only ratio of the §11.2 rule on `w_total` |
|---|---|---|---|---|---|---|---|---|
| flame_steak | 0 / 1 / 2 / 3 | 0.708 / 0.707 / 0.711 / 0.706 | 5,588 / 5,597 / 5,677 / 5,680 | 2,223 / 2,211 / 2,144 / 2,224 | 6,582 / 6,584 / 6,648 / 6,641 | 0.900 ×4 | 0 ×4 | 1.60 / 1.47 / 1.42 / 1.44 |
| sear_steak | 0 / 1 / 2 / 3 | 0.720 / 0.722 / 0.718 / 0.717 | 6,035 / 6,148 / 6,213 / 6,298 | 2,361 / 2,316 / 2,602 / 2,219 | 7,676 / 7,434 / 7,656 / 7,536 | 0.900 ×4 | 0 ×4 | 1.29 / 1.33 / 1.38 / 1.35 |
| cut_roasted_beef | 0 / 1 / 2 / 3 | 0.740 / 0.739 / 0.735 / 0.731 | 4,928 / 4,954 / 4,852 / 5,046 | 1,546 / 1,524 / 1,668 / 1,503 | 7,004 / 6,981 / 6,858 / 6,813 | 0.900 ×4 | 0 ×4 | 1.12 / 0.95 / 1.02 / 1.12 |

Readings, before any decision: (i) on `w_total` the mass-matched sham
exists on 12/12 prefixes, within 0.71–0.74 units of the truth centroid
(the truth rows themselves lie within 0.53), i.e. a shell just outside
the bottle; the greedy walk stops at the lower edge of the band (0.900)
because the highest-contribution rows are taken first; (ii) **the
ORIGINAL §11.2 local rule is itself constructible on `w_total`**: the
locally eligible non-truth rows carry 0.95–1.60 of the truth mass on
every prefix (the ≥ 0.90 edge is reachable on all 12), so locality can be
KEPT and only the measure changed; (iii) the retired `w_in` measure is
the reason for both refusals, not the geometry.

The decision on the record for the user: which measure (`w_total`,
recommended: the paint the rows contribute to the view) and which
locality rule (the original local rule on `w_total`, recommended, since
it keeps the review's locality and magnitude together; the radius rule
stays as the fallback). Until frozen, no L program enters the freeze
list.

| item | sha256 |
|---|---|
| this page, sections 0-13.20 (before this record) | `8526974f6a8047a8f307488ab37e8cf3699b572180c235e1f0738db081759462` |

### 13.21 The four decisions (user, 2026-09-14), the stage-2 setup they unlock, one conflict found by the setup counts (decision 5, PENDING), and the intentions declared before any stage-2 score exists

Cut-off: 2026-09-15T00:30:00+02:00 (CEST). Every stage-2 preparation
job named here is in `agent-control/realdata/jobs/ledger.txt` with a
reason; the full sha256 of every artefact named here is in the tracked
machine record `research-wiki/assets/absfix-stage2-freeze.json`
(abbreviated in prose only). No stage-2 cell has been trained; nothing
in this section reads a score. Commits: the EXECUTION commit of the
draws, votes and recounts below is `bb7ebf66e14fae53f4dbbfa1ed90c0af4913df27`
(HEAD when they ran); the AMENDMENT commit is the one carrying this
section (its id is written into the freeze record and the next
section). Two fresh-context Codex reviews (gpt-5.6-sol) of the draft
were folded in before it was appended; their blocking points and
dispositions are listed at the end.

**Decision 1 — G-wrongmem-L: the ORIGINAL local rule of §11.2 on the
total-paint measure.** §11.2 is amended append-only: the rule stays
"non-truth rows whose deformed centre projects inside the +20 px
construction silhouette on ≥ 8 of 19 training cameras at frames 50 and
95, taken greedily by descending rendered contribution (a row that would
overshoot the upper edge is skipped) until the mass is within ±10% of
the construction-derived set's", and the MEASURE is `w_total` (each
row's rendered contribution to the whole cam15 view at frame 50) instead
of `w_in` (its contribution inside the object silhouette), for the
reason of §13.19: a magnitude defined by the treated region cannot be
matched by rows outside it. The radius rule of §13.18 is retired to a
recorded fallback; no radius program was ever used for training. Draw:
job 57752673 (CPU, 82 s), tracked CLI `scripts/draw_membership_shams.py
--l_mode local --contribution_key w_total --gap 60 89` at the execution
commit (script sha256 `4ea716ad…`, introduced in cb6dd1c), outputs ONLY
in the new dirs `runs/realdata/absfix2/<scene>/draws_prefix<S>_final/`.
Constructible on 12/12 prefixes. The script re-emits A and B alongside;
their ROW SETS are identical to the frozen stage-1 A/B on all 24
(`row_ids_sha256` equal, checked in the job log) and the frozen files
(`draws_prefix<S>_seedbox/ab_only/` on the confirmatory scenes,
`draws_prefix<S>_v2/` on the calibration scene) stay the training files.

| scene | prefix | L rows drawn | truth rows | locally eligible non-truth rows | mass ratio draw/truth (`w_total`, cam15 f50) | `program_gwrongmem_l.json` sha256 |
|---|---|---:|---:|---:|---:|---|
| cut_roasted_beef | 0 / 1 / 2 / 3 | 325 / 773 / 504 / 329 | 7,004 / 6,981 / 6,858 / 6,813 | 4,492 / 4,595 / 4,465 / 4,745 | 0.9005 / 0.9001 / 0.9001 / 0.9002 | `4efc0e06…` / `6f153dd4…` / `b6828eea…` / `45622d95…` |
| flame_steak | 0 / 1 / 2 / 3 | 152 / 224 / 255 / 225 | 6,582 / 6,584 / 6,648 / 6,641 | 6,551 / 6,549 / 6,503 / 6,479 | 0.9010 / 0.9004 / 0.9000 / 0.9011 | `aa06f9f9…` / `13e78f37…` / `5e4abd26…` / `9734be6a…` |
| sear_steak | 0 / 1 / 2 / 3 | 381 / 325 / 256 / 276 | 7,676 / 7,434 / 7,656 / 7,536 | 6,824 / 6,941 / 6,942 / 7,134 | 0.9004 / 0.9001 / 0.9001 / 0.9002 | `2f3461d9…` / `b8fc63dc…` / `254e386a…` / `48f74e2c…` |

Property of the rule, recorded before any score: the locally eligible
rows are the high-paint shell around the bottle and the walk is greedy
by contribution, so the band is reached with 9–43× fewer rows than the
truth set (7,004/325 = 21.6 … 6,582/152 = 43.3; smallest 6,981/773 =
9.0). What L matches is the baseline whole-view rendered contribution in
cam15 at frame 50 within ±10%; it is not a claim about paint removed in
the scored crop, during the gap, or in other cameras. `G − GWRONGMEM_L`
therefore separates correct membership from "a local, contribution-
matched wrong set"; it does not separate row count, which A and B do.
These programs are the ones that decision 5 is about; whether they, or
the option-(iv) redraw, are the training programs is the user's call.

**Decision 2 — A and B are exempt from the ≥ 100-rows-in-box clause of
§11.4 ONLY.** Every other clause of §11.4 (gated rows at 12k ≥ 1,000; ≥ 1
zero-presence frame inside the arm's own gap; reserved units equal;
`program_match`) applies to them unchanged; their in-box counts are
recorded descriptively. Implementation: the reducer applied the in-box
clause to every gated arm with no per-arm mechanism, so
`scripts/realdata_gate_analysis.py::mechanism_exercised_v2` gains guards
driven by the spec INSTANCE, never by code constants:
`CELL_PRECONDITION.fbox_min_exempt_arms` (`["GWRONGMEM_A",
"GWRONGMEM_B"]` in the JSON) and `CELL_PRECONDITION.rows_min_exempt_arms`
(EMPTY in the JSON; the key exists so that decision 5 can be a JSON
edit). Rules of the guards: an exemption waives a THRESHOLD, never a
MEASUREMENT — a missing, boolean, negative or non-integer count fails
for every arm; the exempt lists are validated against the spec's arm
list (an unknown arm raises); the reason string of an exempt arm names
the waived clause and carries both counts, on failure as well as on
pass.

In the same change the reducer stops TRUSTING the extractor for the
timing shams: `CELL_PRECONDITION.fbox_frame_by_arm` (`GMIS: 244, GONES:
291`, the floor of each gap's midpoint — the values the reducer's own
frozen test fixture of 34cc193 already used; the stage-1 setup recount
used 245 for GMIS, a one-frame difference in a descriptive count, noted
and not repeated) and `zero_window_by_arm` (`GMIS: [231,258], GONES:
[287,296]`, the gap interiors with the edges excluded, as §11.4 reads)
are now checked by the reducer for GMIS and GONES exactly as the [63,87]
window (the P1 core, §11.5) and frame 75 are checked for every other
gated arm; the pre-13.21 "note and trust" path remains only for spec
instances that do not name them. Consequence for the calibration scene:
the wave-1 GMIS/GONES cells were extracted at frame 75, so their
preconditions are re-extracted at 244 / 291 on their frozen
`chkpnt12000.pth` with the unchanged extractor into new directories
(`runs/realdata/absfix2/cut_roasted_beef/wave1_cells/<tag>/`, beside
symlinks to the frozen profile; job 57761259 in the ledger; the wave-1 run dirs are not
written), and the calibration manifest points at those directories for
GMIS and GONES. Two facts about the precondition record: (i) the
extractor's field `gated_rows_fbox_frame150` is a LEGACY NAME — it holds
the in-box count at the cell's `--fbox_frame`, whose value is recorded
in `detail.fbox.frame` and read by the reducer as `fbox_frame`; (ii)
`frames_presence_zero_list` carries the zero-presence frame ids, and it
is that list, not the scalar count, that the window tests read.

Tests: six new branch tests (in-box exemption for A/B with L unexempt;
row-count exemption keeping zero-presence; missing counts fail despite
exemption; unknown arm refused; GMIS/GONES validated in their own gap
and refused at the wrong frame or the wrong gap; the stale
`test_a_scene_whose_anchors_are_still_pending_is_dwp` — which asserted
that the SHIPPED instance leaves the confirmatory anchors pending, false
since §13.6 filled them, failing at the execution commit before any edit
of this block — now constructs the pending state explicitly); the
wave-1 reproduction fixture and every other reducer test pass (154 in
the two reducer suites).

**Decision 3 — Claim B is DESIGN_WITHOUT_POWER on both confirmatory
scenes as §11.5 reads.** The camera-drop rule of §13.9(b)/§11.3 is not
touched. GESTMEM is trained where its program exists (flame_steak 4/4,
cut_roasted_beef 4/4) and reported descriptively; sear_steak has no S2
program (0/4, §13.16), so the confirmatory cell count is 68, not the 72
of the handover (36 flame + 32 sear).

**Decision 4 — stage 2 GO, with the setup below.** NOT submitted: this
section is shown to the user with its hashes first, and decision 5 is
open. Adaptive inputs are disclosed: the setup rules below were written
after wave 1's cells and stage 1's counts were seen; none reads a
stage-2 score.

* *Cells:* 68 confirmatory continuations (2 scenes × 4 prefixes × 9 arms
  minus 4 sear GESTMEM) + 16 calibration additions (GESTMEM, A, B, L on
  the four wave-1 prefixes of `cut_roasted_beef`) = 84, each the prefix's
  `chkpnt6000.pth` → 12k under the arm's program. The calibration scene's
  U, G, GEST, GMIS and GONES are the wave-1 cells (gate-faithful
  re-evaluations of 2026-09-10; GMIS/GONES with the re-extracted
  preconditions above) and enter the manifest as such.
* *G-est programs for the confirmatory prefixes* (jobs 57753670 / 672 /
  674 / 678 flame s0–s3, 57753680 / 683 / 685 / 690 sear s0–s3, ~4.5 min
  GPU each): built exactly as wave 1 built `program_est.json` — the
  stage-1 vote invocation (authored seed box, construction masks,
  `--id_rule mass`) with `--gap_frames` = T1's pooled window read from
  the frozen T1 program and asserted against the `--export` value —
  written to the NEW dirs `votes_prefix<S>_est/`; the row set is
  asserted IDENTICAL to the frozen `program_oracle.json` on 8/8
  (n = 599,568–599,686). Windows and sha256: flame [60,89] `7d1ecee0…`,
  [60,90] `518aa136…`, [60,89] `6c277d32…`, [60,93] `223cbd11…`; sear
  [60,89] `cc8501f6…`, [60,89] `9f889670…`, [57,106] `41ff4d81…`, [60,90]
  `daf99005…`.
* *Scored checkpoint and scoring path, every arm:* the scored checkpoint
  is `chkpnt12000.pth`; `chkpnt_best.pth` (written by `main.py` at the
  best in-training test PSNR) is diagnostic only and is never scored.
  Gated arms: `scripts/eval_n3v_gated.py --restore_state` (the
  checkpoint's own `elgs_state`, `program_match` by lineage key), scored
  from the GATED render. U: the same evaluator in fresh mode with the G
  program (the only mode an ungated checkpoint admits), scored from the
  UNGATED render (its gated render is the render-time-gate diagnostic of
  §5, descriptive). **Path cross-check on EVERY arm:** the wave-1
  `main.py --val` pass of the same `chkpnt12000.pth` (no EL-GS setup,
  gate off) must reproduce, on every numeric leaf to 5e-5 with IDENTICAL
  leaf-key sets and no non-finite value, the evaluator's ungated
  render's window profile — for U the scored profile, for gated arms
  the gate-off check profile (`f_box_profile_gateoff_check.json`); a
  mismatch FAILS the cell (`path_crosscheck.json`, exit 9), reported,
  never repaired in place. This proves the fresh/restore asymmetry inert
  on every cell, not only on U. The `--val` PNGs are written to the
  scratch area (`/leonardo_scratch/fast/EUHPC_D36_068/sri/stage2_val/`,
  purged by policy, not evidence) with a per-file sha256 manifest kept in
  the run dir; the evaluator's renders stay in the work area. Evaluator
  sha256 `4bdf427f…` (unchanged since fbf4693).
* *Cell template* `agent-control/realdata/absfix2/stage2/stage2_cell.sbatch`
  (same shape as wave 1's continuation template) with three recorded
  differences: (1) the `--val` pass is a cross-check, its renders on
  scratch (the evaluator's render is scored; saves 1.2 GB of work-area
  disk per cell); (2) `ADAGS_SAVE_ITERATIONS=12000`, a new OFF-BY-DEFAULT
  launcher variable in `scripts/run_leonardo.sh`, validated by
  `^[1-9][0-9]*( [1-9][0-9]*)*$` and asserted equal to `12000` by the
  cell, which appends `--save_iterations 12000` after the `--wandb_tags`
  list (`train` there is a W&B tag from which the mode is inferred, not a
  positional). Verified against the REAL parser: `main.py`'s own
  parser-definition block, executed as is, parses the launcher's argv to
  `wandb_tags=[scene, tag, 'train']`, `save_iterations=[12000]`, and
  without the option to the default `[3000, 6000, 9000, 10000, 12000,
  14000, 15000]`; the launcher already records the resolved argv per run
  (`printf %q` into `meta/`). The only consumer of the schedule in
  `main.py` is the `if iteration in saving_iterations:` block (a point
  cloud and checkpoint write), so the training path is untouched and the
  9k / 10k files are simply not written (~1 GB per cell); (3) the U
  scoring path above. Every cell refuses to run unless the repo HEAD
  equals the frozen commit and the tree is clean; inputs (config,
  evaluator config, program, prefix checkpoint) are hashed into the run
  dir before training and outputs after. Preconditions use the frozen
  per-scene `MECHANISM_FBOX` and frame 75 (244 / 291 for GMIS / GONES)
  on `chkpnt12000.pth`.
* *Configs:* U runs the tracked `configs/n3v/b0c_crb300_12k_rp.yaml`
  unchanged; every gated cell runs a per-cell copy of
  `configs/n3v/elgs_local_crb300_12k.yaml` that differs in the
  `elgs_oracle_episodes` line only (asserted by diff at plan time).
  Plan and input hashes: `stage2/stage2_plan.json`, `stage2/
  stage2_inputs.sha256` (252 lines); `--go` refuses if any input hash
  moved since the plan. The freeze record names, per cell, the config,
  program and prefix-checkpoint hashes. The plan is regenerated after
  decision 5 and refused if its L paths disagree with the chosen option.
* *Order of reading and the montage rubric:* the collector computes and
  hashes the reducer's inputs and the reducer's output BEFORE any montage
  is viewed; the per-prefix montage of the scored crop on every arm
  (`stage2_montage.py`: rows GT / U ungated / U render-time gate / the
  gated arms; frames 40, 63, 75, 87, 92, 99, 109, 245, 291; the frozen
  box ± 60 px; labelled) is then viewed as technical QC whose ONLY
  permitted consequences are (a) recording a pixel-reproducible
  instrument defect (a gated arm rendered gate-off, a wrong frame, a
  wrong crop) and (b) a DIAGNOSTIC identical re-run of the affected
  step, both ledgered. The original reducer result stays authoritative
  unless a pre-declared automatic integrity check (`program_match`, the
  path cross-check, the precondition) invalidated the artefact before
  scoring; if the re-run reproduces the defect the cell is reported
  failed; any repair needing changed code, config, crop, frame or program
  is a separately declared stage and cannot replace a stage-2 result.
  The reducer is `scripts/realdata_gate_analysis.py --spec configs/n3v/
  absfix_gate_spec_v2.json --paired --wave 2` on the collector's
  manifest.
* *Warden* (`stage2_warden.sh`, login node, 5-min period, policy frozen
  here): HUNG = RUNNING with a 0-byte `.out` for longer than the shortest
  elapsed time of a COMPLETED stage-2 cell of the SAME scene whose run
  dir holds `chkpnt12000.pth` (a verified sibling), floor 3,600 s,
  fallback 12,600 s before any verified sibling exists → cancel and
  resubmit the identical recorded sbatch line; NODE_FAIL / BOOT_FAIL →
  one identical resubmission; at most 2 automatic resubmissions per cell
  IN TOTAL across both causes; FAILED / TIMEOUT / OOM / externally
  CANCELLED → logged, never auto-resubmitted, no code or config change
  during the stage, reported as failed in the manifest if it stays
  failed. Every event is ledgered.
* *X = STG:* the chain of `README_stage2.md` verbatim (11 prep, 24
  training, 24 render, 4 collect-and-profile jobs) with Slurm `afterok`
  dependencies; the collector reindexes the six 50-frame segments into
  the evaluator's absolute-frame layout beside the derived scene's own
  cam00 frames and profiles them with `scripts/event_region_frame_profile.py`
  on the same masks and ROIs. Descriptive only, outside the reducer.
* *Disk:* a wave-1 cell occupied 3.7–4.5 GB; a stage-2 cell is ≈ 2.4 GB
  in the work area (`chkpnt12000` 0.52, `chkpnt_best` 0.52, `point_cloud`
  0.43, `gated_eval_12000` 0.85) plus 1.2 GB of `--val` renders on
  scratch, so 84 cells ≈ 200 GB work area + 100 GB scratch, and STG
  ≈ 47 GB, against ≈ 300 GB free on a 4 TB project quota shared with
  another member (`cindata` 92.9 %) and ≈ 180 GB free on scratch. The
  eight composite dirs the user named (`preview_*` and `build_*_v2` on
  both scenes; no `_v1` exists; 9.16 GB) are being copied to
  `D:\adags-archive\leonardo\runs\realdata\absfix2\<scene>\` with
  per-file sha256 manifests written on Leonardo before the copy
  (`stage2/composite_manifest_*.sha256`, 8 files, 15,540 entries) and are
  removed from Leonardo only after the copy verifies against them; the
  `_v3` builds and the training roots are untouched. That alone leaves a
  thin margin; moving `runs/realdata_gate` (119 GB, the 2026-09-09 lane,
  every number on the record) the same way is proposed — the user's
  call.

**Decision 5 — PENDING, found by the setup counts before any score: the
L programs of decision 1 do not meet the nominal thresholds of two §11.4
clauses that were written for ~7,000-row programs.** The clauses are
measured on the 12k cell, so this is an expectation, not a failure: (a)
gated rows at 12k ≥ 1,000 — L has 152–773 rows at seeding, and in wave 1
the family ids grew 7,004 → 7,747 (G, +10.6%), 6,981 → 7,595 (G),
7,004 → 7,726 (GEST), 7,004 → 7,210 (GMIS) and SHRANK 7,004 → 6,849 /
6,981 → 6,906 (wrongmem, −2.2 % / −1.1 %) from seeding to 12k, so no L
cell can be expected to reach 1,000; (b) rows in the mechanism box
≥ 100 — the round-4 setup recount, the authoritative one for this
section (jobs 57753922 / 925 / 927; the 6k prefix checkpoint, deformed
centres projected on held-out cam00 at frame 75 inside the frozen
`MECHANISM_FBOX`; identical to round 3 except that GWRONGMEM_L is read
from `_final`; round-3 reports preserved under
`box_recount_round3_L_radius_prov/`) gives L in-box / gated =
cut_roasted_beef 190/325, 500/773, 301/504, 185/329; flame_steak
**79/152**, 118/224, 131/255, 110/225; sear_steak 194/381, 171/325,
137/256, 157/276 (52–65 % of each L set is in the box; flame s0 is under
the count). Under §11.5 Claim A requires `G − GWRONGMEM_L` in every pair,
so an L cell excluded from the mechanism-exercised set makes Claim A
DESIGN_WITHOUT_POWER on that scene whatever G does — a control's
precondition deciding the treatment claim, the §13.19 category error
again. Options, recorded before any cell trains, with their
constructibility computed on the frozen inputs (login node, CPU):

(i) exempt GWRONGMEM_L from the row-count and in-box clauses
(`rows_min_exempt_arms` and `fbox_min_exempt_arms` gain
`"GWRONGMEM_L"`). Acceptable ONLY with L-specific substitute floors
frozen in the same entry — final gated rows ≥ 0.5 × seeded rows and
final in-box rows ≥ 50 (rationale: the worst wave-1 shrink is 2.2 %, so
0.5 is a generous retention floor; 50 is half the object arms' in-box
floor) — because a bare waiver would let an L cell with zero surviving
or zero in-box rows pass; those floors would need a reducer key and
test of their own, and Claim A's interpretation would state that its L
endpoint is guaranteed only under those floors.
(ii) keep §11.4 verbatim and accept that Claim A will read DWP through L
on any scene where an L cell ends below the thresholds.
(iii) fall back to the §13.18 radius rule (2,144–2,602 rows; in-box not
recounted; locality given up).
(iv) RECOMMENDED — a deterministic LOCAL draw constructed WITHOUT any
§11.4 exemption, every §11.4 clause still evaluated on the 12k cell:
same frozen eligible pool, zero overlap, rows visited by descending
`w_total` (ties by ascending frozen row id), a row ACCEPTED iff
mass + w ≤ 1.00 × truth (an overshooting row is skipped permanently, the
walk continues), STOP after the first acceptance at which mass ≥ 0.90 ×
truth AND n ≥ 1,100 both hold, REFUSED if the pool is exhausted first;
the 1,100 gives a 10 % margin over the ≥ 1,000 clause against the ≤ 2.2 %
wave-1 shrink and is frozen now, not chosen after any L outcome; the cap
at 1.00 keeps L's dose at or below G's (the walk lands at 1.000 wherever
the pool allows); a construction ACCEPTANCE condition, in-box ≥ 100 rows
at frame 75 on the 6k prefix, is recounted before training and refuses
the draw otherwise. Constructible on 12/12 on the frozen inputs: n = 1,100
on every prefix; mass ratio 1.000000 on 11 prefixes and 0.946 on
cut_roasted_beef s1 (its whole eligible pool carries 0.95 of the truth
mass, §13.20); of the 1,100 rows, 221–938 carry positive `w_total` at
cam15 f50 and the rest are locally eligible rows with ZERO contribution
in that view — the ≥ 1,000-row criterion is therefore satisfied by
geometric membership (inside the dilated silhouette on ≥ 8 cameras),
not by 1,000 paint-carrying rows, and the paint dose of L is the mass
ratio; whether those rows paint in other cameras or frames is not
measured by the frozen measure and is stated as such. Requires
`--l_mode local_floor --l_min_rows 1100 --l_cap 1.00` in the draw
script (tracked, with tests, executable pseudocode in the entry), a
redraw, the acceptance recount (minutes), and a §13.22 entry with the
new L hashes and a regenerated plan.
Nothing is submitted until the user decides; the G / GEST / GESTMEM /
GMIS / GONES setup counts are unaffected (≥ 0.9996 of their rows in the
box on 12/12), A/B are 46–103 as in §13.19.

**Declared intentions, before any stage-2 score is read.** These are
NOT frozen pre-registrations: each names the rule and the arm, none is
implemented or run in the stage-2 task, and each becomes a frozen spec
section with its own parameters, hashes and kill-argument BEFORE it
runs. What is fixed now is that they were declared before any score
existed; anything about them declared after a score is read is post hoc
and will be labelled so.

a. **GESTMEM-S3** — descriptive additional arm on the 8 confirmatory
   prefixes: T1's pooled gap + membership from SAM 3 through the SAME
   vote (`realdata_membership_vote.py`, `--id_rule mass`, same seed box,
   same anchors) and the SAME §11.3 preconditions, concept prompt "wine
   bottle", highest-scoring instance per training camera; the SAM 3.1
   checkpoint from `facebook/sam3` (file sha256), config, code commit,
   the exact prompt string, the no-detection rule and the tie-break are
   recorded in this JSON BEFORE any mask is produced. It cannot rescue
   Claim B under v2.0.0 (a different instrument from the frozen S2); it
   is reported beside GESTMEM. Separate session after stage 2.
b. **Deferred ablations, calibration prefixes, four cells each:**
   (b1) *product gate* — presence MULTIPLIES the learned temporal
   marginal of a gated row instead of replacing it (targets the C1
   control-window cost of −0.16..−0.28 dB,
   [[absence-fixture-lane-2026-09-10]] §7B reading 4); requires a
   flag-gated renderer change that lands on a branch with a verified
   empty diff when the flag is off, and only after every stage-2 cell has
   finished; (b2) *free boundaries* — `elgs_a_lr > 0` (value to be
   frozen), intervals initialised from T1's window, everything else as
   G.
c. **Optional descriptive ablation, calibration only: graded membership**
   — row sets of 25 / 50 / 75 % object rows, count-matched, true window
   (12 cells). Whether it runs is decided after the verdict; if it runs
   it is labelled post-verdict exploratory.

In parallel, no effect on the chain (CPU/web only): the Charge dataset
(arXiv 2512.13639) is public under CC BY 4.0 on Hugging Face
(`charge-benchmark`, 8 scenes, Dense 25 + 16 cameras, per-frame uint16
per-mesh-part segmentation). On scene 050_0130 the per-id census finds
26 full-multiview absence-and-return events, ALL small rig sub-parts
(1–306 px peak footprint) under self-occlusion, and every large-
footprint id present in 93/93 frames; the census of the remaining seven
scenes with a grouped-object test was interrupted (API limit) and is
resumed. Interim answer to the user's Sep-25 question: no scene-level
exit-and-return found yet. To be promoted to its own page.

Also in this commit: `minimal-path-config-2026-09-11` §2 gains the
`visibility_event_manifest` / `hide_reveal_*` rows (must stay unset).

*Codex reviews (gpt-5.6-sol, fresh thread, two passes on the draft) —
blocking points and dispositions:* seed-vs-12k conflation in decision 5
→ reworded as an expectation with the wave-1 growth factors and the
recount's checkpoint/frame/box named; fail-open exemptions → missing
counts fail for every arm, lists validated, waivers named on failure,
option (i) given substitute floors, option (iv) constructed and measured
with an acceptance condition and a frozen margin; GMIS/GONES trusted
rather than validated → arm-specific frame and own-gap window in the
instance, reducer checks them, wave-1 calibration shams re-extracted;
U-only cross-check → path cross-check on every arm with identical key
sets and non-finite failure; launcher passthrough → typed, validated,
asserted, tested against the real parser, argv recorded; provenance →
full-hash machine record, commits labelled, recount round named; items
a–c → declared intentions; `w_total` overclaim reworded; montage
authority and re-run semantics fixed; warden retry semantics fixed;
9–43× corrected; ISO cut-off. Not adopted: an L construction requiring
1,000 paint-carrying rows (the pool does not contain them, §13.20:
0.95–1.60 × truth mass in total) — the geometric reading is stated
instead.

| item | sha256 |
|---|---|
| this page, sections 0–13.20 (before this record) | `6fed1bfb015266c8aadda1aacf9cc7b095398d9bc459f87890e40e82e9da1334` |
| `configs/n3v/absfix_gate_spec_v2.json` before this record | `19217882d6cb8494a1b90e4108a9dad0635719e8a12b0581cd1a6a2f6e14e6a6` |
| `configs/n3v/absfix_gate_spec_v2.json` after the 13.21 edits | `55d9f50de2480c5fd523a80be499d883e431ce59c84936bce665a574bf219ec0` |
| `research-wiki/assets/absfix-stage2-freeze.json` | regenerated at the amendment commit and committed immediately after it; its sha256 and the amendment commit id are in that follow-up commit and in the ledger |
| `scripts/realdata_gate_analysis.py` (exemption guards, own-gap validation) | `7f8b86aef10fd9cd5b28522edb6691907ff0c426dae6802794f197b2cad8ecc7` |
| `tests/test_realdata_gate_analysis_v2.py` | `7b37bd0d50f7b35f339ffb81d972592993d4308161b1dddf8f6711d637e642f9` |
| `scripts/run_leonardo.sh` (`ADAGS_SAVE_ITERATIONS`) | `226cd824d6ee4c8c7a6cfd925efd56fa1e8beecdf5d45cfa3a4b4020c5a908a2` |
| stage-2 scripts (`stage2_cell.sbatch`, `submit_stage2.py`, `stage2_warden.sh`, `stage2_montage.py`, `stage2_collect.py`, `u_val_crosscheck.py`, `stg_chain.sh`, `stg_collect.sbatch`, `collect_cam00.py`, `draws_final.sbatch`, `vote_est.sbatch`, `precond_wave1_shams.sbatch`) | in the freeze record |
| amendment commit carrying this section | recorded in the follow-up commit that adds the freeze record, and in the ledger |

### 13.22 Decision 5 resolved: option (iv), the local-floor L draw; the setup re-frozen; stage-2 GO authorised, submission follows this commit (user decisions 2026-09-15; written 2026-09-15 00:20–01:30 CEST)

Nothing in this section reads a score. Every job named here is in
`agent-control/realdata/jobs/ledger.txt` with a reason. Every EXISTING
freeze-relevant input named here (programs, counts, recount reports,
preconditions, plan, per-cell configs, scripts, templates, manifests,
the tracked spec files) is hashed in full in the regenerated
`research-wiki/assets/absfix-stage2-freeze.json`, with these precise
exceptions: the record contains commit A (this section's commit) but
cannot contain commit F or its own hash, both of which go into F's
commit message and the ledger line that records F; artefacts that do
not yet exist (submission scripts, cell run directories and their
manifests, the Windows copy of `realdata_gate`) are hashed when they
are created, in the ledger and the next section. Hashes are
abbreviated in prose.

**The user's decisions (2026-09-15):** option (iv) for G-wrongmem-L;
`runs/realdata_gate` (127.1 GB, 29,552 files) is APPROVED for archival
to the Windows archive (the copy to scratch_large is under way; the D:
transfer, its verification and the removal follow, in that order);
stage 2 GO. No STG or stage-2 training job is submitted by this commit
(the preparation jobs named below already ran); the submission ids go
to the ledger and to the next section. A fresh-context Codex review
(gpt-5.6-sol, high) of this text was folded in before it was committed
(dispositions at the end).

**§11.2 amended (append-only): G-wrongmem-L = the local-floor draw.**
Same frozen eligible pool as decision 1 (deformed centre inside the
+20 px construction silhouette on ≥ 8 of 19 training cameras at frames
50 and 95), zero overlap with the construction-derived set, measure
`w_total` (each row's rendered contribution to the whole cam15 view at
frame 50). Rows are visited by descending `w_total`, ties by ascending
frozen row id (the pre-densification integer index); a row is ACCEPTED
iff the running mass stays ≤ 1.00 × the truth mass (an overshooting row
is skipped permanently; the walk continues); the walk STOPS at the first
acceptance at which mass ≥ 0.90 × truth AND n ≥ 1,100 both hold, and is
REFUSED if the pool is exhausted first. Comparisons inclusive, float64,
the accepted set sorted by row id. Executable-equivalent pseudocode and
seven tests: `scripts/draw_membership_shams.py::
local_floor_contribution_matched_draw` (CLI `--l_mode local_floor
--l_min_rows 1100 --l_cap 1.0`), commit 846ee1a, script sha256
`bff08d90…03c5`, tests `c69114c1…ed77` (48 pass locally and on
Leonardo); the `local` and `radius` modes are unchanged. The 1,100 is
the §11.4 row floor with a 10 % margin against the ≤ 2.2 % wave-1
shrink; the cap keeps L's baseline contribution at or below G's. Rows
accepted once no remaining positive-contribution row fits under the cap
carry zero `w_total` at the measured view and satisfy the count by
geometric membership only; the sidecar
records `rows_with_positive_contribution` and
`rows_with_zero_contribution`. Construction ACCEPTANCE: the round-5
recount (rows of the L program inside the frozen mechanism box at frame
75, deformed centres from the 6k prefix checkpoint `chkpnt6000.pth`; a
construction check, distinct from the 12k cell precondition) must give
≥ 100 for every L program; a prefix below it is REFUSED and reported,
never relaxed.
Every §11.4 clause is still evaluated on the 12k cell with NO exemption
(`rows_min_exempt_arms` stays empty); an L cell that fails there is
excluded and can still make Claim A DWP. The decision-1 draws
(`draws_prefix<S>_final/`, job 57752673) are preserved unchanged as the
recorded fallback and are not used for training.

**Draw: job 57765868** (CPU, 76 s, HEAD 846ee1a), inputs identical to
job 57752673, outputs ONLY in the new `draws_prefix<S>_iv/`; 12/12
constructed, 0 refused. A and B re-emitted alongside are row-identical
to the frozen stage-1 A/B on all 24 (`row_ids_sha256` equal, checked in
the job log); the frozen A/B files stay the training files. **Recount
round 5: jobs 57766941 (cut_roasted_beef) / 57766942 (flame_steak) /
57766945 (sear_steak)** (GPU, 2 min each; round-4 reports preserved
byte-for-byte under `box_recount_round4_L_final/`; the twelve round-5
report hashes are in the freeze record), L read from `_iv` via
`L_DRAWS_SUFFIX=_iv`; every other arm's count is unchanged from round 4
(GMIS included, see below).

| scene | prefix | L rows | rows with positive `w_total` | mass ratio (draw/truth) | L in-box at f75 (6k prefix) | `program_gwrongmem_l.json` sha256 |
|---|---|---:|---:|---:|---:|---|
| cut_roasted_beef | 0 / 1 / 2 / 3 | 1,100 × 4 | 504 / 1,100 / 938 / 527 | 1.000000 / 0.936142 / 1.000000 / 1.000000 | 838 / 738 / 741 / 820 | `5b721776…` / `f486c25a…` / `aa568d70…` / `5a38c912…` |
| flame_steak | 0 / 1 / 2 / 3 | 1,100 × 4 | 221 / 316 / 363 / 327 | 1.000000 × 4 | 793 / 777 / 739 / 765 | `bd681c7a…` / `164f8994…` / `d4eb208b…` / `bd61eb36…` |
| sear_steak | 0 / 1 / 2 / 3 | 1,100 × 4 | 539 / 460 / 377 / 398 | 1.000000 × 4 | 741 / 793 / 806 / 796 | `efd757d5…` / `2f2b74f3…` / `0ccae8fd…` / `a8a438b6…` |

Acceptance: in-box ≥ 100 on 12/12 (minimum 738), so no prefix is
refused. Truth rows 6,582–7,676; eligible non-truth rows 4,465–7,134;
overlap 0 on 12/12. **Pre-run discrepancy against the §13.21 dry
computation, disclosed and diagnosed.** §13.21 wrote "mass ratio
1.000000 on 11 prefixes and 0.946 on cut_roasted_beef s1 (its whole
eligible pool carries 0.95 of the truth mass, §13.20); of the 1,100
rows, 221–938 carry positive `w_total`". Measured: cut_roasted_beef s1
stops at 1,100 rows with mass 5,436.38 against a truth mass of 5,807.21
(ratio 0.936142), and every one of its 1,100 rows carries positive
`w_total`; the other 11 prefixes are consistent with §13.21's stated
ratio (1.000000) and positive-row range (221–938). Diagnosis:
on s1 the cap is never binding, so the walk stops on the row floor at
the 1,100th row with 3,495 eligible rows unvisited; the §13.21 figure
0.946 is the mass of the WHOLE eligible pool (0.946368 of truth,
measured from the frozen `w_total` and eligibility arrays on
2026-09-15), i.e. the value the walk would reach only if it continued
past the row floor, and the "221–938" range described the 11 capped
prefixes and silently omitted s1. Both are approximation errors of the
dry text, not stale inputs (the job's `contribution_truth` and
`eligible_n` equal the decision-1 sidecar's) and not an implementation
difference (the rule stops at the first acceptance where both floors
hold, which on s1 is the row floor). No parameter was changed after
seeing the job's values; the job's values are the record. On the 11
capped prefixes 221–938 rows carry positive contribution and the rest
(162–879) are geometric members with zero `w_total` at cam15 f50, as
the rule foresees; the reducer reads none of these counts.

**GMIS box frame (this section fixes a recount/spec disagreement before
any score exists).** §13.21 and `CELL_PRECONDITION.fbox_frame_by_arm`
freeze the GMIS in-box frame at **244** (the floor of the [230,259]
midpoint, the reducer fixture's value); the stage-1 recount helper
`stage1/box_recount.py` used 245 through round 4. It now uses 244
(backup `box_recount_round4_gmis245.py`, ledger line, no other change).
Two explicit results: at frame 244 (round 5) the GMIS in-box count
equals the GMIS gated-row count on 10 of 12 programs (6,582–7,676) and
is one row short on two, cut_roasted_beef s3 (6,812 of 6,813) and
sear_steak s3 (7,535 of 7,536); in the preserved frame-245 reports
(round 4) every one of the 12 in-box counts is identical to its
frame-244 value, the same two prefixes one row short. The wave-1
re-extraction and `submit_stage2.py` already used 244.

**Calibration timing shams re-extracted at their own frame: job
57761925** (resubmission of 57761259, which failed after 84 s because
the extractor needs the cell's `meta/train.log`; the sbatch now links
`meta/`, `cfg_args`, `cameras.json`, `input.ply` and `inputs.sha256`
from the frozen wave-1 run dir, which is not written): the eight
`wave1_cells/absfix_{mis,ones}_s{0..3}/precondition.json` at frames
244 / 291 on the frozen `chkpnt12000.pth`; GMIS 7,002–7,210 gated rows
in box with 27 zero-presence frames, GONES 7,312–7,326 with 9; every
hash in the freeze record; the calibration manifest points at these
directories for GMIS and GONES.

**Pre-registrations (user item a).** The four declared intentions
(GESTMEM-S3 with SAM 3; the product gate; free boundaries; graded
membership) are already in §13.21 as rule-only declarations and in the
JSON's `PREREGISTERED_2026_09_14`; nothing is added or implemented
here.

**Training-path diff between the wave-1 commit and the frozen commit
(user item b).** Wave 1 trained at `6b368d2` (freeze_v1). Between it
and this section's commit exactly one commit touches `main.py`,
`scene/`, `gaussian_renderer/` or `elgs/`: `bfb8cbb` (2026-09-11),
+52/−5 lines in `main.py` and `scene/gaussian_model.py`;
`gaussian_renderer/` and `elgs/` are untouched. In `main.py` the change
is confined to `validation()` (the `--val` path now restores the EL-GS
runtime); the training loop is not touched, and `--val` is not the
scored path of any stage-2 cell (it is the cross-check pass, gate-off on
U and gate-on on gated arms, whose window profile must reproduce the
evaluator's scored profile to 5e-5 on every arm; see the template
correction below). In
`scene/gaussian_model.py` the constructor now declares
`_appearance_source_idx = None` and `_appearance_share_mode = "dc"`
and the two render-time reads use the attribute instead of
`getattr(..., None)`; with no appearance edit installed (every training
lane) both branches evaluate to the same `None` test and the same
tensors. Outside those four paths the repo-wide diff `6b368d2..HEAD`
(excluding `research-wiki/`, `refine-logs/`, `tests/`) touches only
the spec JSON, evaluation- and preparation-side scripts under `scripts/`
(evaluator, reducer, draws, vote, census, viz, preconditions) and
`scripts/run_leonardo.sh` (+16 lines: the `ADAGS_SAVE_ITERATIONS`
variable of §13.21, which changes which checkpoints are WRITTEN, not
what is computed); `configs/` (other than the spec JSON), `utils/`,
`arguments/` and the CUDA submodules are byte-identical, and the cells
run in the same pinned venv (`exp_index/leonardo_env.sh`). Not compared:
installed packages beyond the pinned venv, driver and node; no
deterministic replay was run. **The claim is therefore: no
training-path change is identified on the inspected training-reachable
code and configuration between 6b368d2 and the frozen commit**, not a
proof of bit-identity. The calibration scene pairs wave-1 cells
(6b368d2) with wave-2 cells (the frozen commit) under that statement;
the per-cell path cross-check guards the evaluation side only.

**Freeze list (JSON edits in this commit):** `arm_roles.GWRONGMEM_L` →
the rule above; `FREEZE_LIST_STATUS.programs_wrongmem_L` → the 12 `_iv`
programs (hashes above); `pending_user_decisions` → `[]`; `stage2_go` →
the user's go of 2026-09-15; `box_recount_round5` added;
`SHAM_DRAWS.script_iv` added; `STAGE_2.frozen_commit` set to the
literal object `{"id": null, "authoritative_source": "the LAST ledger
line matching '^FROZEN_COMMIT F = <40 hex>'", "written_to":
["stage2/submit/<tag>.sh", "<run_dir>/inputs.sha256 (frozen_commit=)"],
"validation": "submit_stage2.py --go refuses unless HEAD ==
--frozen_commit == that ledger value and the tree is clean; every cell
re-asserts HEAD == FROZEN_COMMIT at start"}` (the submitter's ledger
check is a 13.22 addition, `patch_submit_frozen.py`, hashed in the
record); `STAGE_2.disk` updated;
`submit_stage2.py` L path → `draws_prefix<S>_iv/program_gwrongmem_l.json`. Plan regenerated at 846ee1a: 84 planned
cells, sear GESTMEM 0/4 MISSING as in §13.16; against the §13.21 plan
(preserved as `stage2_plan_13_21.json`) exactly the 12 L cells differ
and only in program path, program hash and the per-cell YAML hash that
embeds the path (36 field differences, `plan_diff.out`); `stage2_plan.json`
`5ab25096…`, `stage2_inputs.sha256` `312e5223…`. One template change,
recorded: the `--val` cross-check renders go to
`/leonardo_scratch/large/userexternal/siyengar/proj_adags/stage2_val/`
instead of scratch_fast, because the scratch_fast project quota stood
at 841 GB of 1 TB shared with other users and 84 × 1.2 GB would have
taken it to 94 %; nothing evidence-bearing lives there (the sha256
manifest of the PNGs stays in the run dir). **A second template
correction, found by reading the template against commit bfb8cbb
before any cell ran:** §13.21 described the `--val` cross-check pass as
"gate off" on every arm and compared it with the evaluator's gate-off
render; since bfb8cbb `main.py --val` restores the EL-GS runtime whenever
the config sets `elgs_enable`, which every gated per-cell YAML does, so
on a gated arm the `--val` render is GATE-ON and the §13.21 comparison
would have failed on every gated cell (exit 9 after the scored outputs
were written). The template now compares the `--val` profile with the
SCORED profile on every arm (U: ungated vs gate-off `--val`; gated arms:
gated vs gate-on `--val`). This checks that `main.py`'s restore path and
the evaluator's `--restore_state` path agree on the render that is
scored; the trade is stated: the per-cell gate-off identity on gated
cells that §13.21 described is no longer compared, so the guard is a
different one, not strictly stronger. `u_val_crosscheck.py`, its 5e-5
tolerance and the fail-on-mismatch rule are unchanged. The tolerance
is not tight in practice: on the 20 wave-1 gated cells the evaluator's
gate-off profile (`f_box_profile_gateoff_check.json`, 2026-09-10
re-evaluation) and the gate-off `--val` profile
(`f_box_profile_val_nogate.json`, produced by the wave-1 `--val` pass at
commit 6b368d2, before bfb8cbb, when `--val` never restored the EL-GS
runtime) agree to `max_abs_diff = 0` on all 3,254 numeric leaves
(login-node check, 2026-09-15 00:35). The gate-on pairing has no
precedent. **Frozen
consequence of a mismatch:** the cell exits 9 after its scored files are
written; the collector marks a cell complete only if its
`path_crosscheck.json` exists with `pass = true` (`stage2_collect.py`
sha256 `1a9deba9…`, the §13.21 collector kept as
`stage2_collect_13_21.py`), so a mismatching cell is FAILED, excluded
from the reducer manifest and listed, and a pair it removes makes the
affected claim DWP under the existing §11 rules. If EVERY gated cell
mismatches, reduction halts and a diagnosis stage is declared
separately; frozen now: that diagnosis may read only job logs,
configurations, `precondition.json` and `path_crosscheck.json`; the
scored renders, the window profiles, the montages and every table stay
unread; no failed cell is reinstated in this wave; any repaired rerun
is a separately pre-registered wave with its own section. Template
sha256 `27cfe279…` (the §13.21 template preserved as
`stage2_cell_13_21.sbatch`). The cell template runs
`gate_cell_precondition.py` on every cell before the evaluator (user
item g, verified by reading the template).

**Disk (user items e, f).** Composites: all 8 directories
(`flame_steak/{build_id79_60_89_v2, preview_id79_deva, preview_id79_mc1/5/6/7}`,
`sear_steak/{build_id114_60_89_v2, preview_id114_deva}`, 14,996 files)
are on `D:\adags-archive\leonardo\runs\realdata\absfix2\` and verified
with `sha256sum -c` against their manifests (the last,
`sear_steak/build_id114_60_89_v2`, 12,498 files, verified 2026-09-15
00:05 CEST, rc 0); manifests also under `D:\adags-archive\leonardo\
manifests\`. `runs/realdata_gate`: **archive-copy job 57767135**
(CPU) writes the per-file manifest, rsyncs to scratch_large
(`/leonardo_scratch/large/userexternal/siyengar/proj_adags/runs/
realdata_gate`) and verifies that copy against the manifest; it
DELETES NOTHING and scratch_large is not the archive. After it
verifies, the scratch copy is transferred to
`D:\adags-archive\leonardo\runs\realdata_gate\` and verified there
with `sha256sum -c` against the same manifest; only then is the
removal of the verified originals (the 8 composite directories and
`realdata_gate`) proposed to the user as explicit commands, which wait
for approval; no job deletes. D: had 483 GB free; the archive would
grow to ≈ 136 GB, under the 400 GB the user set. Footprint, binary
units throughout: work quota 3.716 of 4 TiB used (project-wide,
shared with other users), ≈ 291 GiB free now; per cell 1,890 MiB ≈
1.85 GiB on work (two 520 MiB checkpoints, `chkpnt12000` and
`chkpnt_best`, plus the 850 MiB evaluator renders; the 1.2 GB `--val`
renders go to scratch_large) → 84 cells ≈ 155 GiB; STG ≈ 44 GiB
(the §13.21 estimate of 47 GB, 21.6 GB of it COLMAP output); total
≈ 199 GiB against ≈ 291 GiB free: current margin ≈ 92 GiB, projected
≈ 216 GiB once the two removals (≈ 118 GiB + ≈ 8 GiB) have been
approved and verified. The chain is to be submitted on the current
margin; the warden's status file carries `cindata` each cycle.

**Order of execution after this commit** (each id to the ledger with a
reason), with the commit scheme made non-self-referential: call this
section's commit A and the follow-up commit F. The freeze record is
regenerated on Leonardo at A and names A as `amendment_commit_13_22`
(its parent); F adds that record as
`research-wiki/assets/absfix-stage2-freeze.json` and changes no other
tracked file (verified by `git diff --stat A F`). **F is the
FROZEN_COMMIT every cell asserts.** F's id is not written into any
tracked file of F (a file cannot name the commit that first contains
it); it is written FIRST to the ledger as a line of the exact form
`FROZEN_COMMIT F = <40 hex>` (the authoritative source), then to the
sbatch line of every cell (`submit/<tag>.sh`) and to every cell's
`inputs.sha256` (`frozen_commit=`), and it is quoted in the next
section together with the submission ids. Validation is mechanical:
`submit_stage2.py --go` refuses unless `HEAD`, `--frozen_commit` and
the last such ledger line are one and the same id and the tree is
clean; every cell re-asserts `HEAD == FROZEN_COMMIT` and a clean tree
before training; so no two parts of the record can name different
commits. Then: `stg_chain.sh --go` (63 jobs, dependencies);
`submit_stage2.py --go --frozen_commit F` (84 cells); warden under
nohup; then §11.8's order of reading: reduce, hash the output, view
every montage, read the table.

| item | sha256 |
|---|---|
| this page, sections 0–13.21 (the file before this section) | `c087c9ee4528f8a3af4680c1500b6218d982ad893adee68064f2fbc854214893` |
| `configs/n3v/absfix_gate_spec_v2.json` after the 13.22 edits | `18aae16ae3a4772ddcee4f2ee2f801a232e2ecfcae0dde6f7c918e685ce23c16` |
| `scripts/draw_membership_shams.py` (local_floor, commit 846ee1a) | `bff08d90d58b4823c7d02939b2e2d7ec6b24d34deb89cb8543de26cbc83e03c5` |
| `tests/test_draw_membership_shams.py` | `c69114c11e8252d17bc3aee59779792028d90c06bde840d9a724c82f79ffed77` |
| `stage2/stage2_plan.json` / `stage2_inputs.sha256` | `5ab250965c9bf2ccf…` / `312e5223602e9f64c…` |
| `stage2/stage2_cell.sbatch` (scratch_large, scored-profile cross-check) | `27cfe279a98ce57d521cda7d77cb36520ee87ac19863bffb88e0b898435ad4bd` |
| `stage2/stage2_collect.py` (cross-check part of completeness) | `1a9deba9fba57a253940cb049212e498522ab552d706a99bfe40b7fec8e94e55` |
| `research-wiki/assets/absfix-stage2-freeze.json` (regenerated at A) | committed in F; its sha256 is in F's commit message and in the ledger line that records F |
| commit A (this section) | named as `amendment_commit_13_22` inside the freeze record and in the ledger |
| commit F (freeze record; FROZEN_COMMIT of every cell) | ledger, every `submit/<tag>.sh`, every cell's `inputs.sha256`, and the next section |

*Codex review (gpt-5.6-sol, reasoning high, fresh thread, two passes on
this text before commit) — blocking points and dispositions.* Pass 1
(8 blocking, all accepted): placeholders and unknowable commit ids →
the A/F scheme above, the record hash in F's message and ledger, the
opening claim narrowed to existing inputs with precise exceptions;
self-referential frozen commit → the record names A, F is ledger-first,
mechanically validated by the submitter and every cell; "go executed" /
"submitted" → "GO authorised, submission follows", ids in the next
section; "moved to the Windows archive" → "approved for archival",
scratch_large named as not the archive, D: verification before any
removal, margins labelled current vs projected; "numerical training
path unchanged" → narrowed to "no change identified on the inspected
training-reachable code and configuration", scope and omissions
enumerated; cross-check "proves … stronger" → the trade stated, frozen
consequence of a mismatch added (collector hashed); dry-computation
discrepancy → measured diagnosis (whole-pool mass 0.946368 vs walk
0.936142), labelled approximation errors of the dry text; ambiguous
GMIS sentence → two explicit results with prefixes and paired counts.
Pass 1 non-blocking, adopted: binary units throughout; "after the cap
is reached" reworded; `chkpnt6000.pth` construction check distinguished
from the 12k precondition; recount job-to-scene mapping and report
hashes in the record. Pass 2 (3 blocking, all accepted): the opening
"every artefact hashed" claim → existing inputs only, exceptions listed;
`STAGE_2.frozen_commit` → a literal object with the authoritative
ledger locator and a submitter check that refuses any disagreement
among HEAD, `--frozen_commit` and the ledger; the all-gated-mismatch
branch → diagnosis limited to logs, configs, precondition and
cross-check files, scored artefacts unread, no reinstatement in this
wave, repaired reruns a new pre-registered wave. Pass 2 non-blocking,
adopted: "no STG or stage-2 training job is submitted by this commit";
"consistent with" instead of "agree exactly"; provenance of the wave-1
gate-off `--val` profiles stated. Not adopted: none.

### 13.23 Stage 2 SUBMITTED (2026-09-15, 00:45–00:57 CEST): commits A and F, the job ids, the warden; the Leonardo checkout is pinned at F until the chain ends

Append-only record of the execution declared in §13.22; no score is
read here. Commit A (§13.22) = `ca9dead395a120842f6b2394094abee4339190a1`
(reducer and draw tests pass on Leonardo at A: 202). The freeze record
was regenerated at A (345 entries, sha256
`db3b4f0247b5aed122eff5dfeef56c35fd32fdcd933ce9f0acc457321314d5c7`) and
committed as `research-wiki/assets/absfix-stage2-freeze.json` in
**commit F = `dc7b301ee0993f8093015965931cce3666c41bba`**; `git diff
--stat A F` is that one file (2,700 lines) and the asset is
byte-identical to the record on Leonardo. Ledger line 391:
`FROZEN_COMMIT F = dc7b301ee0993f8093015965931cce3666c41bba`; the
submitter's check printed "frozen commit validated: HEAD ==
--frozen_commit == ledger". One note on the plan: `--plan` was re-run
at A after the submitter gained the ledger check (`patch_submit_frozen.py`),
which rewrote `stage2_plan.json`'s `repo_head` and timestamp fields
(sha256 now `423c4520…`, the value in the freeze record); the 84 cells
and every input hash are identical to the plan quoted in §13.22 and
`stage2_inputs.sha256` is unchanged (`312e5223…`).

**Submitted (each with a ledger line and a reason):** STG chain, 63
jobs 57769509–57769763 with `afterok` dependencies (prep 11: 57769509–
57769525; train/render 48; collect 4: 57769568, 57769648, 57769674,
57769763; the full name→id map is in `stage2/jids_stg/`). Stage-2
cells, 84 jobs 57769768–57770106 (`stage2/jids/<tag>.txt`, sbatch lines
in `stage2/submit/<tag>.sh`, each carrying `FROZEN_COMMIT=F`), in the
order flame_steak (36), sear_steak (32), cut_roasted_beef (16). Warden
started 00:50:23 CEST on login02, pid 2385710, policy §13.21 plus a
`cindata` line per cycle. At 00:57 CEST 45 cells and 7 STG jobs were
RUNNING, 40 cells and 52 STG jobs PENDING; work quota 3.7 of 4 TiB.
Archive-copy job 57767135 was still copying (manifest of 29,552 files
written 00:39; a copy of the manifest is at
`D:\adags-archive\leonardo\manifests\archive_manifest_realdata_gate.sha256`,
sha256 `08768f23…`).

**Operational rule, binding for the chain:** every cell asserts
`HEAD == F` and a clean tree at its start, and the warden's identical
resubmissions do the same, so the Leonardo checkout stays at F until
the last cell (including any resubmission) has started; commits made
after F (this section included) are pushed but NOT pulled on Leonardo
before then. Reduction (§11.8 order) runs at F or at a later commit
whose diff against F touches no reducer input, stated when it happens.

### 13.24 Declared repair stage (2026-09-15, 03:20 CEST): every U cell dies in the evaluator's fresh mode before any profile exists; U is scored through the wave-1 `--val` path

Declared and committed BEFORE any U profile exists and before any
score of any arm has been read. At 03:14 CEST the chain watch reported
the first terminal cells: `absfix2_flame_steak_u_s0/s2/s3` and
`absfix2_sear_steak_u_s1` (jobs 57769768, 57769863, 57769922,
57770011) FAILED after 2 h 12–17 min each. Their logs are identical in
kind: `train rc=0`, `precondition rc=0`, then
`scripts/eval_n3v_gated.py` (fresh mode, G program) raised
`ContractError: episode program v2: row_ids membership was computed on
a 599,6xx-row cloud but seeding sees 599,5xx rows. A fresh
create_from_pcd run never reproduces a trained cloud`. Cause: the U
arm's evaluator call was designed in §13.21 to seed the G program on
the trained U checkpoint in fresh mode so that its UNGATED render could
be scored beside a render-time-gate diagnostic; the G program's
membership is `row_ids` on the 6k prefix cloud, and the 12k U cloud has
a different row count after densification, which the contract refuses
by design (the same refusal protects every gated arm from a mislabelled
program). Wave 1 never exercised this path: its U cells were scored by
`main.py --val`. The remaining four U cells (`flame_steak_u_s1`,
`sear_steak_u_s0/s2/s3`) will reach the same line and fail identically;
no U cell has an `f_box_profile.json`. Gated arms are unaffected: they
run the evaluator in `--restore_state` mode on their own `elgs_state`.
The warden logged the failures and, by policy, resubmitted nothing.

**Repair, declared here as its own stage (§11.8 / §13.21 "any repair
needing changed code … is a separately declared stage"):**
`stage2/u_score.sbatch`, one GPU job per U cell on its EXISTING run
directory: asserts `HEAD == F` and a clean tree, the frozen commit in
the cell's `inputs.sha256`, `chkpnt12000.pth` and `precondition.json`
present and NO `f_box_profile.json` present; runs `main.py --val` on
that checkpoint through `scripts/run_leonardo.sh eval` with the cell's
own U config (`b0c_crb300_12k_rp.yaml`; U carries no gate, so this is
its plain ungated render, the identical path wave 1 scored its U cells
with, on a training path with no identified change since 6b368d2, §13.22
item b); profiles it with `event_region_frame_profile.py` exactly as the
cell would have (`f_box_profile_val.json`), copies that file to
`f_box_profile.json` (the scored file), writes the PNG manifest and a
`u_score.json` provenance record (`scored_from: "main.py --val"`, job,
frozen commit, hashes) and appends to `outputs.sha256`. NOT PRODUCED for
U, and recorded as such: the evaluator render, the render-time-gate
diagnostic and the U path cross-check (a single path cannot be
cross-checked; the gated arms' cross-check stands). `stage2_collect.py`
now marks a U cell complete iff `chkpnt12000.pth`, `precondition.json`,
`f_box_profile.json` and `u_score.json` with `scored_from ==
"main.py --val"` exist (`patch_collect_u.py`); gated arms keep the
§13.22 rule. Nothing else changes: no training, no program, no config,
no crop, no frame; frozen commit F unchanged; the eight failed cell jobs
stay FAILED in the ledger and the warden log; the repair jobs are
submitted with `afterany` on the four U cells still running so that
their training output is scored as soon as they finish. Job ids, the
sbatch and collector hashes and every ledger line are recorded when
they exist (ledger, then the next section).

Consequence for the record: the U scored render comes from `main.py
--val` while every gated arm's comes from `eval_n3v_gated.py
--restore_state`; on every gated cell the two paths are proved equal to
5e-5 on the gate-on render by the §13.22 cross-check, and on wave 1 the
gate-off pairing was exact (`max_abs_diff = 0`), which is the evidence
that the two renderers are the same renderer. This is stated as a
recorded asymmetry, not hidden.

### 13.25 State of the chain read at 12:20 CEST 2026-09-15: every gated cell complete on disk but marked FAILED by the template's last line; two cells timed out on slow nodes; U repair done; STG done

Read from Slurm states, job logs and run-directory listings only; no
window profile, table or montage has been opened.

**Slurm says 82 FAILED (exit 1:0) and 2 TIMEOUT; the files say 74 gated
cells are complete.** Every one of the 74 gated cells that did not time
out holds `chkpnt12000.pth`, `precondition.json`, the evaluator's gated
and gate-off renders and their profiles (`f_box_profile.json`,
`f_box_profile_gateoff_check.json`), the `--val` profile, the PNG
manifest and `path_crosscheck.json` with `pass = true`; the 74
cross-checks all report `max_abs_diff = 0` on 3,254 leaves, so the
gate-on `--val` render and the evaluator's `--restore_state` render are
identical on every gated cell (the §13.22 "no precedent" pairing now
has 74 exact agreements). The FAILED state comes from the LAST line of
`stage2_cell.sbatch`: `sha256sum … $RUN_DIR/meta/*cmd* >>
outputs.sha256`, whose glob matches nothing (the launcher names those
files `command_train_*.sh` / `command_eval_*.sh`), and
`exp_index/leonardo_env.sh`, which the template sources after training,
sets `-euo pipefail`; the non-zero status ended the script with exit 1
after `PRECONDITION …` was printed and before `CELL DONE`. The effect
is one missing line in `outputs.sha256` (the `meta/` command files) and
the Slurm state; no output is affected. The collector's completeness
test is file-based (§13.22, §13.24) and records the Slurm state beside
it; nothing is resubmitted for this. The eight U FAILED states are the
§13.24 evaluator refusal; all eight U cells were scored by
`u_score.sbatch` (jobs 57781496–57781506, COMPLETED, 8–11 min each,
`f_box_profile.json` + `u_score.json` present).

**Two cells timed out at 6 h on slow nodes:** `flame_steak_gones_s1`
(57769857, lrdn2013: `train rc=0` at 06:22, precondition at 06:37, the
evaluator cut) and `flame_steak_gwrongmem_l_s3` (57769994, lrdn3263:
`train rc=0` at 06:56, cut inside the precondition step). Training took
5.4–5.9 h against 2.3 h on every other node (the same identical sbatch
line, the same frozen inputs); both run directories hold the
`chkpnt12000.pth` written by that line. **Declared continuation:**
`stage2/gated_score.sbatch`, the post-training part of the template
unchanged in substance (precondition kept if present, evaluator
`--restore_state`, profiles, `--val` cross-check against the scored
profile, outputs; its final hashing line lists only files that exist),
run once per cell on the existing run directory under the same
`HEAD == F` and clean-tree assertions and refusing to overwrite an
existing profile. No training, no program, no config, no crop, no frame
change. The warden's policy (no automatic resubmission of TIMEOUT)
stands; this is the declared manual step it foresees. Job ids in the
ledger and the next section.

**STG (X, descriptive):** all 63 jobs COMPLETED; the four collect jobs
wrote `runs/stg/{flame_steak,sear_steak}_v3_full_seed{0,1}/
f_box_profile.json` (unread).

**Disk:** work quota rose from 3.716 to 3.9 of 4 TiB (97.5 %); the
three `cells/` trees hold 162 GB (33 + 68 + 61), as estimated; the
`--val` renders (96 GB) sit on scratch_large. The `realdata_gate`
transfer to D: was interrupted at 43 GB when the previous session ended
(the copy on scratch_large is intact and verified); it is resumed as a
manifest-driven delta and verified on D: before any removal is
proposed.

**Order of reading, unchanged:** when the two continuations finish,
`stage2_collect.py` (file-based completeness, Slurm state recorded), the
reducer with its output hashed, then every montage, then the table.
