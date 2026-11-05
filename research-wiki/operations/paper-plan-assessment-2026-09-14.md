---
title: "Paper-plan assessment (2026-09-14) — what remains for CVPR 2027, the four decisions, the step-back options, and the research-structuring rules"
date: 2026-09-14
evidence_bearing: false
status: assessment (not a spec; nothing here authorises a cell)
---

# Paper-plan assessment (2026-09-14)

Input: [[handover-2026-09-14]], [[absfix-wave2-spec-v2-2026-09-11]] §11 and
§13.9–13.20, [[absfix-wave2-stage1-2026-09-13]],
[[paper-thesis-v1-2026-09-11]] with its §8 amendments,
[[paper-thesis-v1-kill-argument-2026-09-11]],
[[literature-check-absence-return-2026-09-10]],
[[absence-fixture-lane-2026-09-10]] §7, [[minimal-path-config-2026-09-11]],
plus a code inventory and a Leonardo probe made today. No new number is
produced here; every number is copied from the page named beside it.

## 0. Facts verified today (2026-09-14)

| item | value | source |
|---|---|---|
| Venue deadline | CVPR 2027 papers **2026-11-16 AoE** (fixed); supplementary 2026-11-23 | cvpr.thecvf.com/Conferences/2027/Dates |
| Days remaining | 63 (9 weeks) | arithmetic |
| Leonardo checkout | `bb7ebf6`, clean, equal to the Windows HEAD | `git log` on both |
| Queue | empty; no absfix job running or pending | `squeue` |
| Allocation `EUHPC_D36_068` | 144,000 local-h total, 16,454 consumed, ≈127.5k left; month 11,835 / 5,219 consumed | `saldo -b` |
| Work-area disk | 3.7 TB of 4 TB (92.9 %) | `cindata` |
| Stage-2 cost estimate | 260–320 A100-h (spec §11.6) + STG 30–42 | spec §11.6, §13.12 |
| STG baseline | pinned, installed, one scene preprocessed (frames 0–49), dry run done, **no training run** | `agent-control/realdata/stg/README_stage2.md` |
| SAM 3 | released (SAM 3.1 checkpoints 2026-03-27); text/exemplar concept prompts segment ALL instances of a concept in images and video; decoupled detector–tracker; checkpoints gated behind a Hugging Face access request | github.com/facebookresearch/sam3 |
| SharpTimeGS code | project page carries only the arXiv link; no repository | liaozhanfeng.github.io/SharpTimeGS |
| Charge dataset (CVPR 2026) | 8 benchmark scenes of 93–653 frames, Dense split 25 train + 16 test cameras, 2048×858 at 96 fps, per-frame object segmentation, depth, flow; public-release URL NOT verified in this pass | arXiv 2512.13639 |

Compute is not a constraint on anything below. Wall time, the 12-h
certificate, disk, and the writing calendar are.

## 1. Where the paper stands, claim by claim

The thesis ([[paper-thesis-v1-2026-09-11]] §8 as amended) makes four
claims. What each has, and what it still lacks for a CVPR submission:

| claim | evidence on the record | what stage 2 adds | what is still missing after stage 2 |
|---|---|---|---|
| C1 training-time total presence gate: exact-zero presence in the gap, better return; render-time gating is the opposite | one scene, 4 prefixes: +4.16 dB ghost core, +2.86 dB return, 4/4; membership-specific +3.78 / +2.32; render-time −5.9 dB; real-occlusion null | the same contrast on two further scenes with the membership-specific primary, zero-overlap random shams, the local sham, and STG descriptive | a ghost-window ground truth that is not a teacher render (kill point P1/P5); the reason for the −0.2 dB control-window cost |
| C2 visibility-gap estimator (T1) from training views, zero false activations | LRV3 exact; real occlusion exact to 1 frame; fixture [60,91] vs [60,89]; stage 1: 6 of 8 within one frame, 2 visible failures | G-est on 12 prefixes | nothing essential; report the two failures as recall limits |
| C3 no public ≥ 8-camera capture supplies absence-and-return; authored fixture + event-window metric is the evaluation | literature check (promoted); DiVa 0/597; N3V one occlusion | two more authored scenes | a check of Charge before asserting "none exists" (the literature page itself says so) |
| C4 membership is the open problem, quantified | LRV3 spatial-partition refuted; SAM2 one-click vote reproduces construction membership at precision ≥ 0.9996 / recall 0.978–0.990 on 8/8 admitted prefixes | GESTMEM descriptive on flame + calibration; Claim B DWP on both confirmatory scenes as frozen | a fully estimated arm that is admitted on a confirmatory scene (the sear cam09 countertop mask and the flame s3 late window are admission failures, not estimation failures) |

The kill-argument's seven points and their current state: P2, P3, P4,
P5, P6, P7 are each disposed by an arm, precondition, wording rule or
scene rule in spec v2.0.0 and are answered by stage 2 if it returns
MET on both confirmatory scenes. **P1 (authored ≠ physical) is out of
scope by decision and remains the paper's largest exposure.** Section 4
lists what could narrow it in the time available.

## 2. The four decisions (recommendations; the user decides)

1. **L-sham measure and rule.** Adopt the original local rule on
   `w_total` (constructible 12/12 at 0.95–1.60 × truth mass, §13.20) and
   record the achieved mass ratio in every sidecar. Reason: the L sham
   exists to exclude "the gain comes from removing paint near the object";
   it must be measured where the control acts (the whole view), which is
   the §13.19 lesson. Expected outcome: L loses on P1 (it removes real
   background paint during the gap); that is the correct behaviour of
   this control and should be said in the paper as what it excludes.
   Amend §11.2 append-only, draw with `--contribution_key w_total
   --l_mode local`, hash, add to the freeze list.
2. **A/B exemption from the in-box clause.** Exempt A and B from the
   ≥ 100-rows-in-box clause only; keep every other clause; report their
   in-box counts descriptively. A precondition written for the treatment
   cannot be applied verbatim to a control defined by non-locality.
3. **Claim B = DWP on both confirmatory scenes.** Accept, as frozen. Do
   not amend the camera-drop rule (post hoc). GESTMEM trains where its
   program exists and is reported descriptively on flame (4 prefixes)
   and calibration (4). Separately, and BEFORE any stage-2 score is
   read, declare a §13.21 pre-registered additional arm (see §4.1) so
   that an admitted fully-estimated arm on sear_steak can still exist
   without touching the frozen S2 instrument.
4. **Stage-2 go.** Yes, now. It is the critical path: 72 + 16
   continuations + STG, ≈ 300 A100-h, 2–3 days of wall time with a
   warden. Nothing in the paper can be written past the introduction
   without its tables. Disk: check `cindata` first; 88 checkpoints plus
   STG's 21.6 GB of COLMAP output land in a work area at 92.9 %.

## 3. Critical path to 2026-11-16 (nine weeks)

| week | dates | what | verify |
|---|---|---|---|
| 1 | Sep 14–20 | the four decisions; §13.21 amendments; L draws; stage-2 chain + STG submitted with a warden; SAM 3 access request and login-node install; Charge availability check + absence census (CPU); declare the two ablations of §4.3 | every job id in the ledger; freeze list fully valued; montage of the scored crop on every arm as cells complete |
| 2 | Sep 21–27 | reducer on stage 2 (`--paired`, v2 keys); verdicts recorded append-only; paper plan with a narrowed title; figure list; decide the v3 route (§4.2) by Sep 25 | verdict page written from frozen text and primary inputs |
| 3–4 | Sep 28–Oct 11 | v3 cells (Charge or insertion) under their own frozen spec + kill-argument; ablations (product gate, free boundaries); GESTMEM-S3 on sear | each has a precondition on the accepted set before any score |
| 5–6 | Oct 12–25 | full draft; result tables from the reducer output only; kill-argument on the draft; claim audit against raw result files | `paper-claim-audit`, `citation-audit` |
| 7–8 | Oct 26–Nov 8 | second draft; related work (PersistGS first, then SharpTimeGS/FreeTimeGS/RetimeGS/CubifyGS); supplementary; code-release manifest | fresh-context review |
| 9 | Nov 9–16 | freeze, submit; supplementary by Nov 23 | — |

Compute over the nine weeks ≈ 500–600 A100-h of the 127k available.

## 4. The step back that buys three steps forward

### 4.1 SAM 3 as the membership instrument (not for wave 2)

Wave 2's S2 is frozen at SAM 2.1 with one primary-placed click per
camera (§13.8–13.10). SAM 3 changes the instrument in the one way that
matters for this project: a text prompt ("wine bottle") segments every
instance of the concept in every camera without a placed click, with a
presence token that discriminates near concepts. Consequences:

* the two Claim-B admission failures are exactly the failure modes a
  click-free concept prompt removes (a click that landed on the
  countertop; DEVA segment granularity differing 10–30× across cameras,
  the binding cause of the negative plate census, §13.12);
* the P2 "construction-dependent oracle" objection weakens further,
  because no human or agent chooses pixels for the estimated arm;
* it is the natural instrument for the v3 spec, pinned the same way
  (checkpoint sha256, config, commit, prompt string, instance-selection
  rule).

Recommended use now: a §13.21 pre-registered descriptive arm
GESTMEM-S3 on the eight confirmatory prefixes (T1 gap, SAM 3
membership through the SAME vote and the SAME §11.3 preconditions),
declared before any stage-2 score is read, run after stage 2 (≈ 8 cells,
≈ 20 A100-h). It cannot rescue Claim B under v2.0.0, and the paper says
so; it can show that the fully estimated gate is admitted on a
confirmatory scene once the segmenter is click-free. Blockers: gated
Hugging Face access; compute nodes have no outbound network, so the
install and the checkpoint download happen on the login node.

### 4.2 A ghost-window ground truth that is not a teacher render (P1)

Three routes, ranked:

1. **Charge census (one CPU day, decisive either way).** Charge ships
   per-frame object segmentation on 25 + 16 cameras of rendered scenes;
   an object absent from every camera for a window and present before
   and after is a mask-empty interval per instance id, computable
   without any model. If one exists, it is an un-authored absence with
   exact ground truth in every window and on every camera, the strongest
   available answer to P1 and P5 short of a physical capture. Verify the
   public release and licence first; the HTML fetch today found no URL.
2. **Insertion fixture (v3 as recorded).** Note what it does and does
   not buy: inserting SA4D's render of the object outside a window
   makes P1 read real pixels but moves the teacher into P2/S1. As a
   COMPLEMENT to the removal fixture it shows that no single window's
   teacher dependence explains the effect; on its own it does not remove
   the teacher. Cost ≈ one fixture construction, a new spec with its own
   kill-argument, 4 prefixes and ~8 arms ≈ 100–150 A100-h and two weeks.
3. **A physical capture** (≥ 8 calibrated cameras, an object that leaves
   and returns). Only if a rig is available; not otherwise feasible
   before the deadline.

Recommendation: run 1 in week 1; choose between 1 and 2 by Sep 25; do
not run both.

### 4.3 Two cheap ablations that a reviewer will ask for

* **Product gate.** Presence currently REPLACES the temporal marginal
  for gated rows ([[elgs-m1-evidence-wiring-record]] §; renderer
  `_temporal_multiplier`). The gate's only measured harm, −0.16 to
  −0.28 dB in the untouched control window in every pair, is attributed
  to forcing full presence on ~7,000 rows for 270 frames
  ([[absence-fixture-lane-2026-09-10]] §7B reading 4). A gate that
  MULTIPLIES the learned marginal by the presence function keeps exact
  zeros in the gap and lets the rows keep their marginal elsewhere. Four
  cells on the calibration prefixes (≈ 10 A100-h). If it removes the C1
  cost, it is the formulation the paper should present.
* **Free boundaries.** `elgs_a_lr: 0.0` freezes the interval where T1
  put it. One arm with a small positive rate, initialised from T1,
  answers "why not learn the boundaries?" and measures whether the
  photometric gradient pulls them (four cells).

Both belong on the deferred-ablation list of the JSON before they run.

### 4.4 The external baseline

STG is the only admissible per-primitive temporal-opacity baseline
(FreeTimeGS/++ and RetimeGS have no code; SharpTimeGS's project page has
no repository as of today). Run it as pinned (2 seeds × 2 scenes, 6
models per scene per seed at 30k iterations). Its role in the paper is
descriptive: the ghost and the degraded return are a property of the
unimodal-support family, not of this substrate. Do not enlarge its
role.

## 5. Research structuring — the user's framework applied to this project

The user's five-step ladder (isolate → headroom → representative real
data → engineer against observed failures → competitive evaluation) and
the revision (implement → observe → diagnose → revise → establish the
mechanism → broaden) describe what this project has in fact done:

| step | this project | status |
|---|---|---|
| isolate | LRV3 synthetic, oracle membership, localized total gate vs matched temporal control: +1.05 dB return, zero error on 21/27 gap frames; 2-frame mistiming −2.39 dB | done 2026-08-20 |
| headroom | SA4D-edited absence on real footage: +4.2 / +2.9 dB, 4/4, survives four seeds and estimated timing (G-est = G-oracle to 0.04 dB); membership-specific share +3.8 / +2.3 | done 2026-09-10 |
| representative real data | the real occlusion: render-time −3.4 dB, training-time null. The hidden assumption it exposed: a 4D primitive with broad support is shared between an object and its occluder, so occlusion ≠ absence at the row level | done 2026-09-09/10 |
| engineer against failures | gate-off evaluation defect (montage); fixture cap (build vs preview); vote seeded from T1 (re-seeded from authored boxes); SAM2 click membership (precision ≥ 0.9996) | done 2026-09-10/13 |
| competitive evaluation | stage 2 on three scenes + STG | not run |

So the mechanism claim (Claim A) is at step 5, and the fully estimated
claim (Claim B) is at step 4. That is the honest position for the
paper: a mechanism result with quantified, click-level membership
estimation, not a discovery result.

Three refinements to the user's framework, each drawn from this
project's own record:

1. **Every frozen rule names the alternative explanation it excludes.**
   The user's diagnostic table is the right list. The arms already map
   onto it: G-mis excludes "timing"; A/B exclude "generic deletion"; L
   excludes "local deletion"; G-ones excludes "code path"; G-est and
   GESTMEM are the oracle-vs-learned rows; the preconditions exclude
   "the mechanism was never exercised". A rule that excludes nothing in
   the table is cost without information, and this block spent real
   effort on two such rules (the `w_in` mass match, the in-box clause on
   A/B) before catching them.
2. **Few claim rules, many diagnostic readings, and an explicit
   "unresolved" verdict.** Wave 1's NOT_MET was a miscalibrated rule (a
   sham was required silent when generic deletion has a real +0.65 dB
   effect), not a failed idea; the correct response was to re-specify
   the primary, which is what v2.0.0 did. DESIGN_WITHOUT_POWER is this
   project's "unresolved" and must never be reported as a negative.
3. **Loss curves were never the decisive diagnostic here.** What caught
   every material defect was one of: a montage of the scored region on
   every arm; an invariant that looked wrong (`points = 20157`); or a
   precondition stated about the accepted set before any score. Keep
   "one montage per arm before any table" as a standing rule.

One addition the framework lacks and the deadline forces: **time-box
the iterations within a stage.** Stage 2 is the last full iteration of
the mechanism stage; after it, only ablations and the P1 complement are
affordable in calendar time, whatever compute remains.

## 6. Code: what to do before and after the deadline

Inventory (today): the presence gate itself is compact — `elgs/presence.py`
(124 lines), `elgs/runtime.py`, the seeding/gate half of
`elgs/trainer_hooks.py`, ~15 lines of `gaussian_renderer/__init__.py`,
a few `scene/gaussian_model.py` accessors, `scripts/eval_n3v_gated.py`
and `scripts/gate_cell_precondition.py` — under 3,000 lines. It sits in
a 10,318-line `elgs/` package whose rounds/search/acceptance/evidence/
tracks/Determined machinery is dormant for the program-only lane, and
in a repository where `depth_visibility/` (CSVL-VPL, ~12,300 lines),
packet birth + appearance edit (~1,600), the Apollo/Determined stack
(~2,100), the hide-reveal/visibility-event PoC gates (~3,700, still
called from the renderer but a no-op unless `hide_reveal_runtime_events`
or `visibility_event_manifest` is set) and ~70 historical
`configs/n3v/*.yaml` are accretion the paper's path never imports.

Before the deadline: **no refactor.** The frozen cohorts depend on the
training path being byte-identical across waves, and a refactor buys no
result. Do three small things instead: (i) add `visibility_event_manifest:
""` and the hide-reveal fields to the must-stay-off checklist of
[[minimal-path-config-2026-09-11]] §2; (ii) write a `docs/paper-path.md`
manifest naming the ~25 load-bearing files, configs and scripts with the
commit they were frozen at; (iii) keep every historical surface in place
and mention it nowhere in the paper. After submission: a release branch
containing the manifest's files only, built with `git worktree` from the
submitted commit, never by deleting from the main branch.

## 7. Risks

* The CINECA certificate lasts 12 h and only the user renews it; a
  stage-2 chain must be submitted with Slurm dependencies so it runs
  through a lapse, and the warden must be a persistent session.
* Start-up stalls (0-byte log longer than siblings' runtime; node
  lrdn0136 twice): the warden cancels and resubmits identically.
* Disk at 92.9 %: stage 2 + STG could exhaust the work area mid-chain;
  clear or move `preview_*` and `build_*_v1/_v2` composites (preserve,
  do not delete) before submitting.
* SAM 3 access is gated; request it today so the §4.1 arm is not
  blocked by a licence queue.
* Charge's public availability is unverified; if it is not public, route
  2 of §4.2 is the only P1 complement.
* Nine weeks is enough for stage 2, one P1 complement and two ablations,
  and not for a second fixture family.
