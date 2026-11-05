# Review request: CER scoring for Multi4D, the CER proof reducer, and the two submission scripts (2026-10-01)

You are an independent reviewer of analysis and orchestration code (not substrate code). Your working directory contains
`agent/`, the local masters of files that run on the cluster. Read these files yourself:

- `agent/ars_score_m4dcer.diff` (and the full `agent/ars_score.py`): a new scoring method `multi4d_cer`. Already installed
  on the cluster after these checks passed there: the smoke run is refused; a doctored fake CER run (base-run outputs,
  smoke gate files, provenance edited to the reviewed commit) reaches the evaluator; negative variants are each refused:
  extra schedule set, worktree dirty after the run, 41 evaluations, wrong cer_scene, wrong branch, no cer_gates.json; the
  unchanged `multi4d` path still reaches the evaluator; a base run scored as `multi4d_cer` is refused.
- `agent/ars_score_worker_m4dcer.diff`: 9 confirmatory worker rows.
- `agent/cer_reduce.py` and `agent/test_cer_reduce.py`: the proof reducer and its tests (10 tests pass locally and on the
  cluster; three deliberate mutations of the reducer were each caught by a failing test).
- `agent/submit_cer_ex4dgs.sh` and `agent/submit_cer_multi4d.sh`: fail-closed preflight + submission (dry runs pass;
  nothing has been submitted). The launchers they call are `agent/train_ex4dgs_cer.sbatch`, `agent/train_multi4d_cer.sbatch`.
- The CER code the gates come from: `ex4dgs/scene/cer.py` (`gates`, commit 5b82ae7) and `multi4d/scene/cer.py` (`gates`,
  commit 00357aa).

Check, and score 1-10 each:
1. Does the reducer implement the frozen rules (below) and the amendment rev 2 (below) exactly? Look hard at: the OLS
   arm + layout model and its standard error; one-sided p-values (superiority on the gap window, non-inferiority at
   -0.25 dB on the return window); Holm over the family with p = 1 for design-without-power endpoints; clause (c) per
   layout; the overall verdict logic (PASS / FAIL / DESIGN_WITHOUT_POWER); rule 3 with arm + scene fixed effects; rule 2's
   denominator; the `pass` integrity check; refusal on missing or non-finite inputs. Is the "statistical reading" in the
   module docstring a faithful and defensible reading of the frozen rule, and is anything in it a choice that should be
   surfaced to the principal?
2. Do the tests actually exercise each verdict path, or could any pass vacuously?
3. Does the `multi4d_cer` scoring method accept exactly the runs it should and refuse the rest? Any hole?
4. Are the submission preflights complete (wrong commit, dirty tree, reused run id, missing inputs, stale build lock,
   unfrozen design)? Anything that could submit a wrong or duplicate run?
5. Anything that should be cut.

List findings as severity (fatal / major / minor), the claim, what the code says (file:line), and the concrete fix.

Context you cannot read from this directory:
================ PLAN §4 frozen rules (S0, as amended), verbatim ================
**Frozen rules (as amended; nothing below changes after this line without a new dated entry).**

1. **Causal proof, CER against Ex4DGS base.** Six endpoints: the gap window and the return window, for each of
   same-pose, moved and pass-through. Pooled over the three layouts, 9 runs against 9; two-way fixed effects (arm +
   layout); Holm correction over the 6 endpoints. An event type passes when (a) its gap improvement is
   Holm-significant, (b) the one-sided 95% lower bound of CER − base on its return window exceeds −0.25 dB, and
   (c) *(amendment 2026-09-24)* CER's mean gap improvement over base is positive in each of the three layouts
   separately. Causal proof: at least 2 of the 3 event types pass, one of them same-pose.
2. **Against the benchmarks (descriptive).** Per scene, CER (mean of 3 runs) Pareto-dominates a benchmark when it is
   ≥ that benchmark on both gap and return windows. Counts out of 9 are reported against STG, Multi4D and the Ex4DGS
   base mean. Benchmark proof: CER dominates both STG and Multi4D on ≥ 6 of 9 scenes, including at least one of each
   event type.
3. **Do-no-harm.** Whole-frame and ordinary-region PSNR, CER − base, one-sided 95% lower bound > −0.2 dB, pooled.
4. **Preconditions.** A method arm is scored on a scene only if it passes that scene's contribution gates; a failure
   is reported as DESIGN_WITHOUT_POWER for that scene, never dropped.
5. **Scope** *(amendment 2026-09-24)*: repeated trainings measure training noise on these scenes; every claim holds
   "on these nine synthetic scenes". Per-layout differences are reported for every endpoint. The real fixtures are
   always reported separately and never pooled with synthetic scenes; any other combination is written here before
   its runs start.
6. **Multi4D port census threshold (§4b gate 2):** persistent foreground + background carry ≥ 50% of the
   target-region contribution mass over the pre-gap frames, in every layout, on the wave-1 Multi4D same-pose runs.
7. **Independent identity check, strengthened criterion (2026-09-24):** a scene passes iff away-from-edge agreement
   ≥ 0.999 on every pair and pooled, and overall agreement ≥ 0.9999 pooled. Measured on wave 1: 9/9 at 100.0000%.
8. **Dense-initialisation variant (wave 2, not part of the proof):** precondition and robustness criterion exactly as
   written in the 2026-09-24 evening entry.
9. **COLMAP criterion A3:** unchanged; the wave-1 miss (3 of 2,700 STG frames, mug-layout gap frames) stands as the

================ PLAN §4b (Multi4D), verbatim ================
## 4b. Conditional wave-2 item: port CER to Multi4D's persistent bank (added 2026-09-23, principal)

**Why.** Ex4DGS was chosen for speed and engineering safety, not as the better model:

- a single Python opacity junction;
- ≈ 1.2 vs 2.7 GPU-h per run;
- an evaluation stack validated to 0.0016 dB.

The record's history of that choice is in
[multi-interval review round 2](ex4dgs-multi-interval-review-round2-2026-09-21.md)
§"substrate reversal" and [sol_round1.md:397](../assets/neurips-plan-sol-2026-09-21/sol_round1.md).

Multi4D reproduced its paper (3/3 inside the band), is quieter (0.3445 dB range at
n = 3) and newer (ECCV 2026). A port shows that the mechanism is
**substrate-independent**, and it answers the objection that we picked the weaker
substrate. Sol listed a successful Multi4D port as one of the moves from 7/10 to
8/10 (sol_round1.md:469).

**Entry gate.** All three conditions must hold. They are checked in order, and
they concern the setup, not a score:

1. **The Ex4DGS causal proof passes** at the Mon Sep 28 readout (§4 rule 1).
2. **Multi4D's persistent bank actually paints the target (census, non-vacuity).**
   On the wave-1 Multi4D base runs of A0, persistent FG + BG must carry
   **≥ 50% of the target-region contribution mass** over the pre-gap frames,
   **in every layout**.
   - If the transient bank paints the target, gating the persistent bank is
     vacuous. That outcome is reported as a finding about Multi4D's architecture,
     not as a port failure.
   - The 50% threshold is **proposed here and frozen at S0**.
3. **The port diff reaches the principal by Thu Oct 1.** If it does not, the port
   is deferred past the data freeze, to the ideal plan and the camera-ready.

**Design, reusing the Multi4D code map §5; verify first at the pinned commit `c483e82`:**

- **Gate all three render paths:** standalone `gaussian_renderer/__init__.py:198`,
  mask render `:314` and hybrid `:468`.
- **`:314` is the recorded silent bypass.** A test renders through each path and
  asserts identical gating.
- **The transient bank is not gated.** Its temporal cull is in CUDA
  (`forward.cu:281-285`).
- **The gap-only residual cohort goes in the persistent PyTorch path**, since the
  transient bank cannot host it.
- **Lifecycle:**
  - densify/prune and the FG→TR lift are disabled for the residual cohort during
    its stage, because the lift **copies rows and leaves the donor alive**
    (query pack, 2026-09-21);
  - every run is one uninterrupted job, because resume restores the FG branch only.

**Tests before any training:**

- `q = 0` gives byte-identical images to unmodified Multi4D, on all three render paths.
- The contribution non-vacuity gate passes.
- The residual is exactly zero outside the gap.
- A checkpoint round-trip reproduces the model.
- The diff is **review-gated**, the same control model as the Ex4DGS diff.

**Runs:**

- **Confirmatory:** CER-Multi4D vs Multi4D base on **A0 across three layouts**, 3 vs
  3 per layout. The base reuses the wave-1 run plus 2 new runs. The model is the
  same pooled blocked one as §4 rule 1, over P1 and P2 with Holm correction.
- **Descriptive, only if the queue allows:** CER-Multi4D ×1 on R2 and G1 per layout.
- **Checkpoint policy:** Multi4D **final iteration** is primary, for both arms.

**Cost:**

- ≈ 2–3 engineering days.
- **≈ 40 GPU-h** for the core: 6 new base runs and 9 CER runs at 2.7 GPU-h each.
- **≈ 57 GPU-h** with the descriptive R2 and G1 runs.

**Head start at zero GPU, taken regardless of the proof:**

- On Sun 27, while CER-Ex4DGS trains, verify the code-map attachment points on
  Leonardo at `c483e82`.
- Write the FG/TR census, and run it on the wave-1 Multi4D A0 models.

The census is informative on its own terms: it shows which bank paints an object
that leaves and returns.


================ AMENDMENT rev 2: Ex4DGS spec §13 ================
## 13. Gate amendment (2026-10-01, rev 2; the principal's decision (a); appended, before any real CER run)

Applies the plan-log entry "2026-10-01 (later): principal's decisions on the contribution gates and on Multi4D", item
(a). It supersedes, **as preconditions**:

- §6 row 4 ("residual active inside the gap ≥ 0.20, evidence");
- §11 item 4;
- amendment (a) §A5's "gate thresholds unchanged (residual 0.20 frozen)";
- the plan-log freeze of 2026-10-01, item 1.

Everything else in this spec and in amendment (a) stands. The Multi4D spec §14 is the companion.

**Revision history.** Rev 1 of this section is preserved verbatim in
[`assets/cer-gates-amendment-2026-10-01/review_prompt.md`](../assets/cer-gates-amendment-2026-10-01/review_prompt.md).
It was reviewed by Fable 5.1 (ready after fixes, 13 findings) and GPT-6.1 Sol (high; material defect, 4 defects
and 3 missing items); both reviews are in the same folder. Rev 2 absorbs every finding. 13.6 lists the changes.
**Two points are proposals for the principal, not yet frozen:** 13.4 (how a scene that fails the preconditions
enters rules 1–3) and the Multi4D reading in §14.2.

**13.1 Preconditions (plan §4 rule 4) on Ex4DGS, complete list.** A CER run meets the preconditions when all four hold,
read from its `cer_gates.json`:

| precondition | field | rule |
|---|---|---|
| memory active before the gap | `memory_share_pregap` | ≥ 0.50 |
| memory active after the return | `memory_share_return` | ≥ 0.50 |
| memory suppressed inside the gap | `memory_mass_gap` | exactly 0.0 |
| residual zero outside the gap | `residual_mass_outside_gap` | exactly 0.0 |

**Input rule.** The reducer refuses to run when any of these fields, or `residual_share_gap` or `pass`, is missing or
not a finite number.

**13.2 The residual share is descriptive.** `residual_share_gap` is the residual copies' share of the gap-region mass
over the gap window (training cameras, final model). The memory rows carry exactly zero there, so the share splits
the gap content between the residual copies and the ordinary rows. It decides nothing about scoring.

- **Narrower than the plan log's wording.** The plan log calls this "attribution between the memory gate and the
  residual copies". A mass share cannot split anything with the memory gate, whose mass in the gap is zero. It says
  which rows paint the gap. It does not attribute the gap improvement causally; that needs the memory-gate-only
  ablation (wave 2, §10).
- **Per run and per scene.** The reducer reports each run's share and, per scene, the mean of its three runs' shares
  with their minimum and maximum. The denominators are not stored, so a pooled ratio is not available.
- **Thresholds, descriptive.** At 0.10, 0.20 and 0.30 the reducer reports how many runs and how many scene means lie
  at or above each value.
- **Zero denominator.** Ex4DGS clamps a zero gap-region mass to a share of 0 (`scene/cer.py:158`), so a share of 0
  is reported as "0 (zero residual mass or zero gap-region mass)".
- **Stated with it.** The shortened smoke runs (integration tests, not evidence) already showed shares 0.10 and 0.09
  with the rev-2 snapshot birth, and 0.06 with amendment (a).

**13.3 No substrate code change; the plan-log item is discharged by the reducer.** The plan log and the hand-off
listed "the code that reports instead of gating" as the next step. This amendment decides not to change `gates()`.

- **The fields exist.** `cer_gates.json` written by commit `5b82ae7` already stores every field in 13.1 and 13.2
  (`scene/cer.py:159-167`).
- **The `pass` field.** It still applies the superseded rule, requiring `residual_share_gap ≥ 0.2`
  (`scene/cer.py:168-170`), so it will read `false` on most runs.
- **Who reads it.** Nothing acts on it. The scorer copies the whole file under `run_checks.cer_gates` and gates
  nothing on it. The scoring worker and the run reports never open it. The launchers only print it to the job log.
- **What the reducer does with it.** It recomputes the verdict from the four fields. It asserts, as an integrity
  check, that the stored `pass` equals the superseded formula recomputed from the file's own fields, which shows the
  file came from the pinned `gates()`. It never branches on that field and reports it as `pass_superseded_rule`.
- **Its tests.** They include a record with `pass: false` that meets every precondition and must be scored.
- **Why.** This keeps the reviewed commit `5b82ae7` and the scorer pin. The 27 CER runs are submitted on `5b82ae7`.

**13.4 How the reducer applies the preconditions (proposal for the principal's decision; not frozen until accepted).**
Rule 4 says only that a failure "is reported as DESIGN_WITHOUT_POWER for that scene, never dropped". The readings
below are new analysis policy. They follow one principle, proposed by both reviewers: **no gate changes a
denominator, the correction for multiple testing, or a threshold.** A result that needs a scene that failed the
preconditions is reported as not established, and every subset estimate stays descriptive.

1. **Per scene, all three runs.** A scene meets the preconditions when all three of its CER runs do. Otherwise it is
   DESIGN_WITHOUT_POWER, with the failing fields. A run whose gate never acted would otherwise enter the pooled model
   as a number from a mechanism that was not exercised. Majority and mean-share readings were considered and
   rejected.
2. **Rule 1, per event type.** An event type whose three scenes all meet the preconditions is evaluated exactly as
   frozen. If any of its scenes is DESIGN_WITHOUT_POWER:
   - the event type is DESIGN_WITHOUT_POWER, neither passed nor failed;
   - its estimates on the remaining layouts are reported descriptively.
3. **Rule 1, the multiple-testing correction.** The Holm family is always the six planned endpoints. An endpoint of
   a DESIGN_WITHOUT_POWER event type enters it as never rejected (p = 1), so no gate relaxes the correction. Its
   subset p-value is reported descriptively.
4. **Rule 1, the overall verdict.**
   - PASS when the event types that can be scored meet "at least 2 of 3, one of them same-pose" on their own.
   - FAIL when rule 1 could not be met even if every event type that cannot be scored passed.
   - DESIGN_WITHOUT_POWER otherwise, for example when the same-pose event type cannot be scored.
5. **Rule 2 (descriptive).** The denominator stays 9. A scene that fails the preconditions counts as no domination
   and is reported as "x of 9 (y DESIGN_WITHOUT_POWER)". The "≥ 6 of 9, one of each event type" clause is unchanged.
6. **Rule 3.** The pooled bound is certified only on all nine scenes. If any scene fails the preconditions, do-no-harm
   is DESIGN_WITHOUT_POWER and the bound on the remaining scenes is reported descriptively.
7. **Never dropped.** Every scene's numbers, both arms, appear in the output whatever its status.

**13.5 Inputs, failed runs, and the base-run choice.**

- **Manifest.** The reducer reads an explicit list:
  - the 27 planned CER run ids;
  - the base run ids the principal designates for the proof;
  - the STG and Multi4D benchmark runs.

  Every planned CER run must resolve to exactly one score, or to one declared replacement (below); otherwise the
  reducer refuses to run.
- **Infrastructure failure.** A run that ends without a training exit code 0 for a cause outside CER (node, file
  system, Slurm, walltime) is resubmitted under a new run id. The ledger records the new run as the planned run's
  replacement, the original stays on the record, and the manifest names the replacement.
- **CER failure.** A run that stops on a CER assertion (non-finite opacity, a per-bank length, a lifecycle
  precondition in `begin`) is not replaced. Its scene is DESIGN_WITHOUT_POWER with the reason "CER run failed".
- **Base-run choice, not decided here.** Which Ex4DGS base runs enter the proof is still open from 2026-09-29: the
  frozen originals, or the replacements for the runs trained on `lrdn0066`. The reducer takes that choice as an
  explicit argument with no default and records the run list in its output.

**13.6 Changes from rev 1**

| change | source |
|---|---|
| 13.2: "attribution" narrowed and the departure named; per-scene aggregation (mean, min, max); zero-denominator wording | Fable 8, 13; Sol (missing) |
| 13.3: the plan-log item discharged by the reducer, stated; the old `pass` field checked for integrity, renamed in the output, never branched on, and tested with a `pass: false` record | Fable 7, 12; Sol (suspected) |
| 13.4: marked as new policy for the principal; no denominator, Holm family or threshold changes; per-event-type and overall verdicts; rule 2 called descriptive; rule 3 not certified on a reduced population | Sol 2; Fable 2, 3, 4, 9 |
| 13.5: manifest; infrastructure failure versus CER failure; refusal on missing or non-finite fields | Fable 5; Sol (missing) |

================ AMENDMENT rev 2: Multi4D spec §14 ================
## 14. Gate amendment (2026-10-01, rev 2; the principal's decisions (a), (b), (c); appended, before any real CER run)

Applies the plan-log entry "2026-10-01 (later): principal's decisions on the contribution gates and on Multi4D". It
supersedes §6's "residual share of the gap region ≥ 0.20 in the gap" and the last bullet of §13 **as preconditions**.
Everything else in rev 2 and §13 stands. The companion is the Ex4DGS spec §13. Its 13.2 (descriptive residual share),
13.3 (no code change, the old `pass` field) and 13.4 (how a scene that fails the preconditions enters the rules)
apply here with the Multi4D fields and design below. Rev 1 is preserved verbatim in
[`assets/cer-gates-amendment-2026-10-01/review_prompt.md`](../assets/cer-gates-amendment-2026-10-01/review_prompt.md),
with both reviews beside it.

**14.1 What the Multi4D arm is.** CER-Multi4D is a **memory-gate arm**: the exact memory gate on the persistent
foreground and background banks. Whatever Multi4D itself learns paints the gap.

- **Residual copies.** The background residual copies are still born by the unchanged code (§4.2; forecast 6, 0 and
  1 donors on the three same-pose scenes).
- **What paints the gap is reported by bank:** residual copies, transient rows, ordinary persistent rows.
- **What may be said.** The measured contributions are reported as they are, residual copies included. What is
  restricted is a causal claim that the residual copies or the transient bank cause the gap improvement.
- **Name in reports:** "memory gate on Multi4D".

**14.2 Preconditions on Multi4D, complete list** (proposed reading of decision (b), for the principal's
confirmation), read from `cer_gates.json`:

| precondition | field | rule |
|---|---|---|
| memory active before the gap | `memory_share_pregap` | ≥ 0.50 |
| memory active after the return | `memory_share_return` | ≥ 0.50 |
| memory suppressed inside the gap | `memory_mass_gap` | exactly 0.0 |
| residual zero outside the gap | `residual_mass_outside_gap` | exactly 0.0 |

**The reading of (b), stated.** Decision (b) speaks of a Multi4D "gap-content gate that counts residual or transient
rows". Decision (a) lists the preconditions that stay with no Multi4D exception. This rev reads (b)'s gate as
**descriptive**: the share of the gap region painted by residual or transient rows, reported with the 0.10 / 0.20 /
0.30 table (14.3), never a precondition. Both reviewers recommend this reading.

- **What such a precondition would guard.** The four rows above already assert that the memory gate acted, exactly,
  on rows that carry the target. A fifth row would test only which rows paint the gap at the end of training: ordinary
  persistent rows painting it would fail it. That is a property of the result's makeup, not of whether the mechanism
  was exercised.
- **What it would cost.** The values seen disagree. The forecast on full-schedule base models, at birth, is 0.88–0.96
  transient. The shortened smoke run (`cer_t7_m4d_r4`, lift at 4,000, CER start 4,500, end 5,000; not evidence) shows
  0.025 at the end (residual 0.024, transient 0.0009), with foreground 0.45 and background 0.55 at birth. A
  precondition at 0.20 could therefore make all three same-pose scenes design-without-power although the memory gate
  acted exactly.
- **If the principal meant a precondition.** The row "residual + transient share of the gap region ≥ 0.20" goes back
  into this table with these values stated, and (a)'s list is read as Ex4DGS-only.

**14.3 Descriptive.** Per run, and per scene as the mean of the three runs with minimum and maximum (as Ex4DGS 13.2):

- **The residual share** (`residual_share_gap`): the residual copies' share of the gap content, with the 0.10 / 0.20
  / 0.30 table.
- **The gap-content share** (`residual_share_gap + transient_share_gap_region`), with the same table. Both terms have
  the same denominator, the gap-region mass of every row in all three banks (`scene/cer.py:204-206`, `:211`, `:213`).
- **The transient share and the ordinary-persistent share**, as plain values. The ordinary-persistent share is one
  minus the gap-content share, which holds only because `memory_mass_gap` is exactly 0 (a precondition).
- **A zero gap-region mass** has no share. Multi4D does not clamp: `gates()` raises on division by zero, so such a run
  stops (a CER failure, Ex4DGS 13.5).

**14.4 Confirmatory design and how the reducer reads it (plan §4b as amended 2026-09-30; the 13.4 policy applies once
accepted).**

- **Manifest.** Nine CER-Multi4D runs, three on each same-pose scene (bottle, carton, mug; run ids
  `ars_w1_<scene>_m4dcer_rep<r>`), against the nine wave-1 Multi4D base runs 1–3 on the same scenes. Moved-return and
  pass-through CER-Multi4D runs (§4b, "only if the queue allows") are descriptive and never block the reducer.
- **Rule 1.** Model pooled over the three layouts, 9 against 9, with arm and layout fixed effects. Endpoints: the gap
  window and the return window; Holm over these 2, always 2. The same-pose condition passes under plan §4 rule 1's
  three clauses:
  - the gap improvement is Holm-significant;
  - the one-sided 95% lower bound on the return window exceeds −0.25 dB;
  - the mean gap improvement is positive in each layout.

  The Multi4D verdict is that condition's PASS, FAIL or DESIGN_WITHOUT_POWER.
- **Rule 3.** Do-no-harm applies with the same −0.2 dB one-sided bound on the whole frame and the ordinary region,
  pooled over the three same-pose scenes.
- **Rule 2.** Not applied. The benchmark comparison is CER on Ex4DGS against STG and Multi4D; the Multi4D arm is a
  substrate replication.
- **Readings.** Multi4D's final iteration 20,000, scored as the base runs are.

**14.5 Exploratory work (decision (c)).** Transient donors, gating the transient bank, and any other Multi4D variant
start only after this confirmatory design is frozen, meaning after the principal accepts this amendment. Their run
ids, the ledger and every report label them exploratory. They are never pooled with the confirmatory runs.

**14.6 No substrate code change; the scoring path.**

- **The fields exist.** The Multi4D `cer_gates.json` at `00357aa` stores every field in 14.2 and 14.3.
- **The `pass` field** (`:216-218`) applies the superseded rule. It is handled as on Ex4DGS (13.3): integrity check,
  reported as `pass_superseded_rule`, never branched on.
- **The scorer** (outside the substrate) has a method `multi4d_cer` (installed 2026-10-01), which requires:
  - commit `00357aa` on branch `cer`;
  - no untracked or modified files before and after the job;
  - the full schedule (no `EXTRA`);
  - `cer_scene` equal to the scored scene;
  - exactly the 40 fine evaluations 500…20,000 in `debug_metrics.json`;
  - the existing Multi4D final-render checks.

  It copies `cer_begin.json` and `cer_gates.json` into the score. The scoring worker lists the nine confirmatory runs.
  The pin moves with any later commit.

**14.7 Changes from rev 1**

| change | source |
|---|---|
| 14.1: residual contributions may be reported; only causal claims are restricted | Sol 4 |
| 14.2: the gap-content row read as descriptive, the reading stated with what a precondition would guard and cost; "0.20 is the threshold of the gate it replaces" removed | Fable 1; Sol 1 |
| 14.3: thresholds only on the residual and gap-content shares; the ordinary share's condition; zero denominator | Fable 11; Sol (missing) |
| 14.4: own manifest of 9 + 9, optional runs never block; Holm over 2; rule 3 applied; rule 2 not applied | Sol 3; Fable 10 |
| 14.6: the scoring path stated | Fable 6 |

================ Facts about the inputs ================
- A score file (scores/wave1/<method>/<label>.json) has windows[W].psnr.psnr (pooled PSNR) for W in P1 (gap window), P2
  (return window), H2 (whole frame), ORD (ordinary region), and run_checks (provenance; for CER, run_checks.cer_gates is
  the run's cer_gates.json verbatim).
- Ex4DGS base runs trained on lrdn0066 have replacement runs labelled ..._rep<r>_repl for (L1-A0,1), (L2-A0,3), (L2-R2,1),
  (L2-G1,1), (L3-G1,2); which set enters the proof is the principal's open decision.
- Multi4D debug_metrics.json of a full base run: 40 JSON lines, iterations 500..20000 step 500, no stage key.
- Multi4D base walltime used: 3:40 (runs took 2:32-2:42); CER adds begin ~1.5 min + gates ~2 min (smoke run).
- Ex4DGS base runs took ~1:10; CER adds begin + gates 10-15 min.
