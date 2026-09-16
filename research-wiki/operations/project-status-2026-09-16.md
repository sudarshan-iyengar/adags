---
title: "Project status and hand-off (2026-09-16) — the episodic-presence CVPR 2027 project after wave-2 stage 2: what is established, what is unresolved, what exists, and the five structuring decisions ahead"
date: 2026-09-16
evidence_bearing: false
status: status document for a new agent; nothing here authorises a cell
---

# Project status and hand-off (2026-09-16)

Read this first, then `AGENTS.md` (operating policy), `research-wiki/query_pack.md`
(compressed memory, newest block first), `research-wiki/gap_map.md`, and the pages
named below. Every number here is copied from a page named beside it; the
authoritative records are those pages, not this one. The user is about to add
new information on datasets, comparison targets and baselines; the questions in
§5 are framed so that information can be slotted in without re-deriving the
record.

## 0. The project in one paragraph

Working title "Occlusion is not absence: per-primitive multi-interval presence
in 4D Gaussians" ([[paper-thesis-v1-2026-09-11]], attacked in
[[paper-thesis-v1-kill-argument-2026-09-11]]). The mechanism, EL-GS, gives each
4D Gaussian a multi-interval presence with exact zeros (a training-time
"presence gate"), infers the absence window from training views alone (the T1
estimator) and decides membership (which primitives are the absent object) by a
multi-camera vote over segmentation masks (construction-derived, or SAM 2 from
one click per camera). The evaluation is an authored absence: an SA4D edit
removes a wine bottle from frames 60–89 on every camera of an N3V cooking scene,
and event-window PSNR is read on the held-out camera (ghost core P1, return P2,
plus pre-gap, settled, whole-frame and control windows) under a frozen paired
spec with sham arms (wrong timing, wrong membership). Wave 2's stage 2
(2026-09-15, [[absfix-wave2-stage2-verdict-2026-09-15]]) is the first
confirmatory test on two locked scenes: **the return benefit is real and large
everywhere (8 of 8 pairs, +1.7 to +3.2 dB over the ungated model and over every
sham); the ghost-core benefit is real but its membership-specific share is
scene-dependent and misses the frozen all-pairs 0.50 dB rule on flame_steak
(one pair at +0.44, six sham pairs under the floor), so Claim A reads NOT_MET on
both scenes and Claim B (fully estimated gate) is DESIGN_WITHOUT_POWER as
pre-decided.** The descriptive fact that outranks the verdict: SpacetimeGaussians
(STG), which has a per-Gaussian temporal opacity and no gate, renders the
authored absence core 3.4–7.5 dB better than the gated arm, at whole-frame
quality within +0.4–1.2 dB of ours, while sitting below the gated arm on the
return on seven of eight prefixes. The paper's thesis has to absorb that before
anything else is built.

## 1. Constraints and calendar

| item | value | source |
|---|---|---|
| Venue | CVPR 2027, papers **2026-11-16 AoE**, supplementary 2026-11-23 | [[paper-plan-assessment-2026-09-14]] §0 |
| Days left | 61 | arithmetic |
| Compute | Leonardo `EUHPC_D36_068`, ≈ 127k local hours left at 2026-09-14; a 12k 300-frame cell is 2.3 h (up to 5.9 h on a slow node); compute is not the constraint | assessment §0, verdict §4 |
| Wall-time constraints | 12-h SSH certificate (jobs and the warden survive it; new logins do not); 6-h cell wall clock; work quota 4 TiB at **97.5 %** until the approved removals run | [[leonardo-ssh-quoting]], verdict §4 |
| Discipline | frozen spec before any cell; every reading rule has a precondition on the setup; montages before tables; DWP is "unresolved", never "negative"; append-only records; every job id in the ledger with a reason | `AGENTS.md`, [[cvpr2027-deadline-and-research-structuring]] |
| User's structuring rule | simplest implementable idea that shows excellent results first; earn complexity through results (isolate → headroom → representative real data → engineer against observed failures → competitive evaluation); "unresolved" over "failed" | memory note 2026-09-14 |

## 2. Evidence ledger (what is established, what is not)

Positives (each with its boundary):

1. **Exact-zero absence and a better return from a training-time total presence
   gate** on an authored absence, real footage: wave 1 on `cut_roasted_beef`
   +4.16 dB ghost core / +2.86 dB return, 4/4 pairs, gate-faithful renders
   ([[absence-fixture-lane-2026-09-10]] §7B); wave 2 on two locked scenes,
   return +1.7 to +3.2 dB in 8/8 pairs over U and every sham; ghost core over U
   in 8/8 pairs (+0.44 to +2.8) ([[absfix-wave2-stage2-verdict-2026-09-15]] §1).
   Post-hoc gating of the same rows is the opposite (−5.9 dB in the core on the
   ungated model): a gate must be trained with, not applied after.
2. **Window inference from training views (T1) is exact where it fires:**
   LRV3 exact, real occlusion within one frame, the authored window recovered
   within one frame on 6 of 8 confirmatory prefixes, zero false activations;
   fails visibly on 2 of 8 (flame prefix 3, sear prefix 2), and when it fails
   the return collapses (GEST P2 23.6 / 22.6 dB) ([[absfix-wave2-stage1-2026-09-13]],
   verdict §2). Recall of the object's voxel groups is the weakness
   ([[nonoracle-episode-timing-result-2026-08-23]]).
3. **Membership from one SAM 2 click per camera reproduces construction
   membership** at precision ≥ 0.9996 / recall 0.978–0.990 on 8 of 8 admitted
   prefixes ([[absfix-wave2-stage1-2026-09-13]]); the fully estimated arm
   (GESTMEM) equals G within 0.21 dB wherever admitted (verdict §2).
4. **Substrate at STG parity** on the ungated task (33.51 vs 33.52 dB on
   `cut_roasted_beef` 0–49 at a quarter of the schedule,
   [[stg-n3v-protocol-parity-2026-08-19]]).

Negatives and nulls (kept, not to be re-run):

5. **A real occlusion offers the gate nothing** (render-time −3.4 dB in the
   occluded box; trained with the gate, every contrast within 0.1 dB;
   [[realdata-gating-lane-2026-09-09]], [[absence-fixture-lane-2026-09-10]] §2).
   Occlusion ≠ absence at the row level: a 4D primitive with broad temporal
   support is shared between object and occluder.
6. **Membership by spatial partition is refuted** (precision 0.045, recall
   0.179 on the seeding cloud; partial membership is worse than no gate;
   [[lrv3-membership-diagnostic-2026-08-23]]).
7. **No public ≥ 8-camera capture supplies documented absence-and-return:**
   DiVa-360 0 of 597 windows, ImViD census closed, Charge negative at scene
   level on 6 of 8 scenes (2 not censused) ([[literature-check-absence-return-2026-09-10]]
   §2, [[charge-absence-census-2026-09-15]]). The plate (moving-object,
   real-background) fixture is negative on all three scenes (DEVA granularity).
8. **Wrong-membership shams on the ghost core** come within 0.50 dB of the
   gate on flame_steak (six of 24 confirmatory sham pairs): on that scene most
   of the ghost-core gain is not membership-specific. It is +0.4–2.7 dB on
   sear_steak and +2.7–5.2 dB on the calibration scene (verdict §0).
9. **STG's temporal opacity reaches the gap core without any gate** (3.4–7.5
   dB above our gated arm on P1; below it on P2 on 7 of 8 prefixes; verdict
   §2.1). Confounds stated there (six 50-frame models, 25k iterations).
10. Earlier lines closed: CCR consolidation payloads (no headroom), flow birth
    prior (rejected against the camera-swap control), N3V utility scaling
    (variance study: unaffordable), CSVL-VPL v1 (three no-gos). See the
    query pack blocks of 2026-07-29 to 2026-08-24.

Unresolved (DWP), not negative: the fully estimated gate on the confirmatory
scenes (Claim B: T1 miss on flame 3, S2 refusal on sear); physical absence
(P1 of the kill-argument: authored ≠ physical); the C1 control-window cost
(−0.2 dB on the calibration scene, within ±0.14 dB on the confirmatory ones).

## 3. What exists (code, data, infrastructure)

**Repository `D:\adags` / Leonardo `$W/repo/adags`, branch
`apollo/csvl-vpl-v2-exploratory`, HEAD `3a104ab` (2026-09-15).**

* `elgs/` — the EL-GS runtime: `presence.py` / `intervals.py` /
  `families.py` (multi-interval presence per primitive, families, exact
  zeros), `trainer_hooks.py` (`setup_elgs`, seeding of a program onto a cloud,
  reserved units, the lineage-key proof `check_program_matches_runtime`),
  `runtime.py`, plus the older structural-search machinery (`search.py`,
  `energy.py`, `ops.py`, `evidence*.py`, `tracks_*`, `diva360_schema.py`)
  that no live lane uses.
* Fixture and instrument scripts: `scripts/sa4d_absence_edit_render.py`
  (authored absence), `scripts/realdata_membership_vote.py` (multi-camera
  membership vote, `--emit_row_weights`), `scripts/s2_sam2_propagate.py`
  (SAM 2 masks from one click), `scripts/t1_estimator_viz.py` and the T1
  estimator, `scripts/wave2_preconditions.py` (membership and T1
  preconditions), `scripts/draw_membership_shams.py` (A/B/L shams; local-floor
  L, §13.22), `scripts/gate_cell_precondition.py` (per-cell precondition),
  `scripts/eval_n3v_gated.py` (the only authorised scorer for gated cells,
  `--restore_state`), `scripts/event_region_frame_profile.py`,
  `scripts/realdata_gate_analysis.py` (spec-driven reducer, 158 tests after
  the 2026-09-15 repairs), `scripts/plate_census.py`.
* Frozen spec instance `configs/n3v/absfix_gate_spec_v2.json` (page §11 is
  binding where they disagree); cell configs `configs/n3v/elgs_local_crb300_12k*.yaml`,
  `b0c_crb300_12k_rp.yaml` (ungated with reserved parity).
* Leonardo transient state (never committed): `$W/agent-control/realdata/`
  — `jobs/ledger.txt` (every job id with a reason), `jobs/cells.txt`,
  `absfix2/stage1/`, `absfix2/stage2/` (cell template, U-scoring and
  continuation templates, collector, montage, warden, submitter, freeze
  record, manifests), `stg/` (STG install and READMEs). Outputs under
  `$W/runs/realdata/absfix2/` (cells, votes, draws, preconditions,
  `analysis_stage2/`), `$W/runs/stg/`, derived scenes under
  `$W/data_derived/absfix/{cut_roasted_beef_absfix_60_89, flame_steak_absfix_60_89_v3, sear_steak_absfix_60_89_v3}`.
* Data on disk: N3V (six scenes, source under the old D21_034 project);
  the three authored-absence derived scenes; DEVA all-camera masks and SAM 2
  final masks for the three scenes; SA4D checkout and env; STG pinned at
  427abfc5 with its env; ImViD partial download (rate-limited, 21 files);
  DiVa-360 tracks and conversions (reusable); Charge metadata and one scene's
  segmentation (partial).
* Windows archive `D:\adags-archive\leonardo\` (composites 8 dirs,
  `runs/realdata_gate`, manifests), all verified against per-file manifests
  (`research-wiki/assets/absfix-archive-manifests.sha256`).
* Tracked evidence assets: `research-wiki/assets/absfix-stage2-*` (reducer
  output, report, manifest, inputs, 12 montages), `absfix-stage2-freeze.json`,
  `absfix-stage1-record.json`.

## 4. Operational items open right now

1. **Removal of verified originals awaits the user's approval**
   (`runs/realdata_gate`, 119 GB; eight composite dirs, 8.8 GB); the exact
   commands are in the 2026-09-15 session's last message and the ledger says
   nothing was removed. Work quota 97.5 % until then.
2. The scratch_large copy of `realdata_gate` (127 GB) is a second copy,
   subject to purge; not proposed for removal.
3. Leonardo checkout is at `3a104ab`, clean; no job queued. Templates on
   Leonardo differ from the §13.21 versions in the ways §13.22–13.25 record
   (kept beside them as `*_13_21.*`).
4. Declared, not started (§13.21): GESTMEM-S3 (SAM 3 membership), product
   gate, free boundaries, graded membership. The insertion fixture was not
   started (user instruction 2026-09-15).

## 5. The five structuring decisions ahead

The user holds new information for each of these. Below is what the record
already says, the options that are live, and what a decision needs to name.
None of it is decided here.

### 5.1 Which dataset carries the final outcome

What the record says: no public multi-view video dataset documents absence and
return (§2 item 7); the authored SA4D absence on N3V is the only real-footage
evaluation with exact ground truth in every window, and it is one object, one
gap, three scenes, four prefixes each, one seed per prefix. Charge is
rendered, dense (25 + 16 cameras), has per-frame segmentation, and its census
found no scene-level exit-and-return in 6 of 8 scenes (2 not censused, 4.3 GB
segmentation over the cap). ImViD has long takes (a full Opera take is
15,215 frames) and no annotation; DiVa-360 has none corroborated.

Options: (a) stay with the authored N3V fixture and add scenes/objects/seeds;
(b) author absences on a denser rig (Charge or ImViD) where ground truth per
camera is exact; (c) a physical capture is out of reach before the deadline;
(d) a new dataset the user knows of.

A decision needs: camera count (≥ 8 has been the bar), an absence that is
full-multiview with a return of the same object, per-camera ground truth in
the window, a licence, size against a 4 TiB quota at 97.5 %, and an honest
statement of whether the absence is authored or captured (the kill-argument's
P1).

### 5.2 Datasets and metrics that show a general improvement

What the record says: on the ungated task the substrate is at STG parity on
one N3V segment (§2 item 4); no general-quality claim exists on more scenes;
event-window PSNR is the only metric the reducer implements (P1/P2/S1/H1/H2/C1
with per-frame masks and ROIs); LPIPS/SSIM are computed by the `--val` path
but not reduced; model size is recorded per cell (`final_points`, checkpoint
≈ 520 MiB at 600k rows) but never claimed; no tracking or segmentation
evaluation exists in the repository.

Options the user named: reconstruction quality on standard benchmarks (N3V
six scenes, Technicolor, MeetRoom, Immersive), model size / training time,
point tracking, segmentation. Each needs a baseline that reports it and a
frozen protocol; the ones closest to what exists are N3V full-benchmark PSNR
/ SSIM / LPIPS (the `--val` path already produces them pooled and clamped,
[[n3v-baseline-registry-2026-08-20]] lists the traps) and size (rows,
checkpoint bytes, iterations to a target).

What a decision needs: for each metric, the baseline's published protocol,
whether its code exists (FreeTimeGS-family: no released training code;
SharpTimeGS: no repository; STG, 4DGaussians, Ex4DGS-style methods: code),
and the evidence boundary a one-seed comparison can carry (measured seed
spread 0.64 dB whole-frame at the 50-frame protocol; same-code replicate floor
of the same order).

### 5.3 New related work to read

Read and recorded: PersistGS (owns the problem statement, must be contrasted
first), FreeTimeGS / FreeTimeGS++ / RetimeGS (no released training code),
SharpTimeGS (no repository), CubifyGS (object-level asset reuse), SA4D (read,
used), Spacetime Gaussian Grouping (read), TAD-GS, VAD-GS, CEC-4DGS, 4D
Scaffold-GS, STG (pinned and now measured), SAM 3 (released; the click-free
membership instrument, not yet installed), Charge (CVPR 2026, censused),
MemoBench (wrong shape). Pages: [[literature-check-absence-return-2026-09-10]],
[[spacetime-gaussian-grouping-read-2026-08-24]], [[sa4d-read-2026-08-24]],
[[ccr-novelty-record-2026-08-20]], `research-wiki/papers/`.

Not read or not checked: whatever the user now has; anything on temporal
opacity / lifespan primitives that already achieves exact absence (the STG
fact makes this the first thing to search), per-primitive tracking or
segmentation benchmarks for 4DGS, and failure-analysis benchmarks (none found
in the 2026-09-10 check). A reading should be recorded as a `papers/` page
with mechanism, code availability and which of our claims it touches.

### 5.4 Baseline choice and what to carry over from EL-GS / ADAGS

The two questions are separable. Baseline: STG is pinned, installed, run on
the fixtures and now measured; its temporal opacity is also the strongest
competitor on the gap core. Whatever the user picks, the record says: use the
published schedule verbatim, pin the commit, run it on the same derived scenes
with the same held-out camera, score it with the same profiler, and report it
descriptively unless it is inside a paired frozen design.

Portable pieces, in order of how much they have been proven:

1. the evaluation instrument: authored-absence fixture pipeline (SA4D edit,
   hull outlier control, per-camera verification), event-window profiler,
   spec-driven reducer with preconditions, shams and the montage-before-table
   rule — this transfers to any baseline unchanged;
2. the T1 visibility-gap estimator (training views only, zero false
   activations where it fires) — representation-agnostic in principle; on a
   temporal-opacity baseline it would set a window, not a gate;
3. the membership vote (multi-camera masks → row set; SAM 2 one-click or a
   click-free SAM 3 prompt) — transfers to any per-primitive representation;
4. the training-time presence gate itself — the piece whose benefit the STG
   fact puts in question on the gap core and supports on the return; the
   return is where a gate on a temporal-opacity baseline would have to prove
   itself;
5. reserved-unit parity, lineage-key proofs, ledger/warden discipline —
   engineering, portable.

What a decision needs: the baseline's representation of time (does it already
have a per-primitive support with exact zeros?), its code, its schedule cost
per cell, and which of the four pieces above it lacks.

### 5.5 Whether and how "episodic" earns its place

What "episodic" has meant here: one primitive, several disjoint presence
intervals with exact zeros, and the same primitive returning with its own
trained content (no respawn, no donor clone). What the evidence says about
its worth so far: on real occlusions, nothing to gain (item 5); on authored
absences, the gap core is reachable by temporal opacity without any gate
(item 9) while the return is where the gate wins and the shams lose (items
1, 8, 9); membership, not timing, is the binding difficulty (items 2, 3, 6,
8); the fully estimated version is admitted on the calibration scene and on
three of four flame prefixes, unresolved elsewhere.

A decision tree the record supports: if the chosen baseline already renders
exact absence in the gap (as STG does here), the episodic claim narrows to
"return with the same primitives at the pre-gap quality" and must be tested
against that baseline's own return; if the baseline ghosts (as ours does),
the wave-2 tables stand as the mechanism evidence and the open work is
membership and the ghost-core margin on scenes like flame_steak. In either
case the simplest implementable version is a window (T1) plus a membership
set (vote) applied as a presence schedule, evaluated on the return window
first; complexity (product gate, free boundaries, multi-episode chains) is
earned only if the return shows headroom the baseline lacks.

## 6. Hand-off checklist for a new agent

1. Read `AGENTS.md`, this page, `query_pack.md` (2026-09-15 block first),
   [[absfix-wave2-stage2-verdict-2026-09-15]], spec v2 §11 and §13.21–13.31,
   [[paper-plan-assessment-2026-09-14]], [[paper-thesis-v1-2026-09-11]] with
   its kill-argument.
2. Leonardo access is PowerShell-only SSH with a 12-h user certificate
   ([[leonardo-ssh-quoting]]); the live tree is
   `/leonardo_work/EUHPC_D36_068/sri/proj_adags` ([[leonardo-d36-layout]]);
   every job goes through Slurm, every job id into the ledger with a reason.
3. Do not re-reduce stage 2, do not resubmit any stage-2 cell, do not pull a
   commit onto Leonardo while cells that assert a frozen commit are pending.
4. Before any new cell: a frozen spec section with a precondition on the
   accepted set, a kill-argument pass, hashes in a freeze record, and a
   reducer test that uses one real extractor output and one real multi-scene
   manifest (the 2026-09-15 lesson).
5. Known traps: `main.py --val` is gate-on for `elgs_enable` configs since
   bfb8cbb; U cells cannot be scored by `eval_n3v_gated.py` fresh mode with a
   row-id program; `leonardo_env.sh` sets `set -e`; a `| head` after a long
   producer kills a remote script with SIGPIPE; the Bash tool refuses heredocs
   containing backticks (write scripts with the Write tool).
6. Nothing may be deleted on Leonardo or D: without the user's explicit
   approval of the exact command.
