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
