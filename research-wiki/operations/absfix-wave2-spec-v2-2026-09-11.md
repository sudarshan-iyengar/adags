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
