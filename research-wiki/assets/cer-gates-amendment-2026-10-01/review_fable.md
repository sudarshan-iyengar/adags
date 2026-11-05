# Review: gate amendment (Ex4DGS spec §13, Multi4D spec §14), 2026-10-01

Reviewer: Fable 5.1, read-only subagent, on `review_prompt.md` in this folder (rev 1 of the amendment). Saved verbatim.

Reviewed read-only against `D:\adags-cer\ex4dgs` @ 5b82ae7 and `D:\adags-cer\multi4d` @ 00357aa (both clean, 0 porcelain), the two specs, amendment (a), plan §4/§4b/§9 (2026-09-28 onward), the hand-off page, `ars_score.py`, `ars_score_worker.py`, `D:\adags\scripts\ars_run_report.py`, both launchers in `D:\adags-cer\agent\`, and both `test_cer.py` T7 functions.

## Verdict: READY AFTER FIXES

The code-facing claims are exact: every field the amended preconditions and descriptive tables need exists under the stated name, the Multi4D shares share one denominator, the cited line numbers are right, and no consumer acts on the stale `pass` field. The defects are in what the reducer will need and cannot infer (Holm family, rule-2 denominator, proof-level verdict, failed runs), in one unflagged reading of decision (b), and in a Multi4D scoring path that does not exist. None is fatal; none needs a substrate change.

## Scores

1. **Faithfulness: 7/10.** (a) and (c) are implemented as decided. (b) is read as a *precondition* at 0.20 on residual + transient without saying that (a)'s list of "the preconditions that stay" has no Multi4D exception and that (b)'s "gap-content gate" admits a descriptive reading (finding 1). 13.4's reducer rules (all-runs-per-scene, both arms excluded, event type DWP) are necessary new rules, presented as applications of rule 4 rather than as proposed readings (finding 9). 13.3/14.6's "no substrate code change" closes the plan-log item "Not yet done: the code that reports instead of gating" (plan :1206-1207, hand-off item 2) by a different route without saying so (finding 12). 13.2 narrows the principal's word "attribution" and says so; 14.3 then contradicts 13.2 (finding 8).
2. **No substrate code change / fields: 8/10.** Ex4DGS: `memory_share_pregap`, `memory_share_return`, `memory_mass_gap`, `residual_share_gap`, `residual_mass_outside_gap` at `ex4dgs/scene/cer.py:159-161`, per-bank rows and masses `:162-167`, `pass` `:168-170` — exactly as 13.3 says. Multi4D: the five fields plus `transient_share_gap_region` at `multi4d/scene/cer.py:208-213`; `residual_share_gap = cohort(gap_in, 2) / whole(gap_in)` (`:211`) and `transient_share_gap_region = gap_in[2].sum() / whole(gap_in)` (`:213`) share the denominator `whole` (`:204-206`, all rows of all three banks); with `memory_mass_gap == 0` the remainder is ordinary persistent rows, so 14.3's "one minus the sum" is right. `pass` is read by nothing that acts: scorer copies the whole file (`ars_score.py:95`), worker never opens it, `scripts/ars_run_report.py` has no `cer`/`gates` reference, both launchers only `cat` it into the log (`train_ex4dgs_cer.sbatch:81`, `train_multi4d_cer.sbatch:95`), Multi4D T7 only echoes it in `detail` (`multi4d/test_cer.py:491-498`). Two points off: the field will read `false` on every real run and sits under `run_checks`, a key whose other children are checks that passed (finding 7); and for Multi4D there is no scorer to "stay as it is" (finding 6).
3. **Reducer application of the preconditions: 6/10.** Per-scene DWP and "never dropped" are honoured (numbers kept in a separate table). Reporting an event type DWP rather than failed when clause (c) cannot be checked is consistent with rule 4 (DWP is a statement about the design, failure about the method) and is the more faithful reading. But the reducer cannot run on 13.4 as written: the Holm family after an exclusion, the rule-2 denominator and threshold, the causal-proof-level verdict when same-pose is DWP, and the treatment of a crashed run are all unspecified (findings 2-5). A simpler reading exists for two of them: no gate ever changes a denominator (Holm stays at 6, rule 2 stays out of 9).
4. **Multi4D gap-content precondition: 4/10.** A precondition is one of two readings of (b), and the weaker one. (a) states exhaustively which preconditions stay, with no Multi4D carve-out; (b) says "the gap-content gate counts residual or transient rows", which fits equally as the descriptive 0.10/0.20/0.30 gate with a Multi4D-specific numerator, and its parenthetical reads as the justification for calling the arm a memory-gate arm. On a memory-gate arm the precondition protects nothing about the mechanism's exercise: memory ≥ 0.50 before and after plus the exact zero in the gap already assert that the gate acted on rows that carry the target and acted exactly. What the fifth row asserts is the *composition* of the gap content at the end of training, which is outcome-adjacent (it reads training-camera mass, so it leaks no cam00 pixel, but it is what the gap PSNR is made of). Its only protective role is for 14.1's description "with the transient bank as the gap content", and a description can be made conditional in 14.3. Its cost is the one (a) was taken to avoid: smoke 0.025 vs forecast 0.88-0.96 at birth, nothing measured at the end of a full run; a value below 0.20 makes all three same-pose scenes DWP and kills the only Multi4D claim although the memory gate was exercised exactly.
5. **Missing / to cut: 6/10.** Missing: findings 2-6, 10, 13. Cut: 14.3's threshold table on the transient, sum and ordinary shares (finding 11); 14.2's "0.20 is the threshold of the gate it replaces" (the replaced gate is now descriptive, so this is not a rationale).

## Findings

**1. major — 14.2 row 5 reads decision (b) one way without saying so, and conflicts with (a).**
Claim: "gap content from residual or transient rows ≥ 0.20" is a precondition.
Record: plan log :1192-1199 (a) lists "the preconditions that stay" as memory ≥ 0.50 ×2 and the exact zeros, no Multi4D exception; :1200-1202 (b) says "the gap-content gate counts residual or transient rows" with the census/forecast as parenthetical. Smoke `cer_gates.json`: residual 0.0239 + transient 0.00087 = 0.025.
Fix (recommended): move the row to 14.3 as the descriptive "gap-content share (residual + transient)" with the 0.10/0.20/0.30 table, and make 14.1's description conditional ("the gap content is reported by bank"). If the principal meant a precondition: add to 14.2 "This reads (b)'s 'gap-content gate' as a precondition and (a)'s list as Ex4DGS-only; the principal confirms", and state what vacuity it guards (today: none beyond 14.1's wording). Delete "0.20 is the threshold of the gate it replaces".

**2. major — Holm family after an event-type exclusion is unspecified (13.4, inherited by 14.4).**
Record: rule 1 "Holm correction over the 6 endpoints" (plan :822-824); 13.4 bullet 3 removes an event type's endpoints from the pass/fail reading but says nothing about m.
Fix: "The Holm family is always the 6 planned endpoints (2 on Multi4D), whether or not an event type is DESIGN_WITHOUT_POWER; the DWP type's adjusted p-values are reported descriptively." (Conservative; no gate can relax the correction.) If a smaller family is intended, write m = 2 × (non-DWP types) explicitly.

**3. major — Rule-2 denominator and the benchmark-proof threshold under a DWP scene are unspecified.**
Record: 13.4 bullet 2 "leave ... the counts of rule 2"; rule 2 "≥ 6 of 9 scenes, including at least one of each event type" (plan :828-831).
Fix: "The denominator stays 9. A DWP scene is not a domination. Counts are reported as x of 9 (y DWP). The ≥ 6 of 9 and per-event-type clauses are unchanged." Otherwise "6 of 8" is a relaxation produced by a gate.

**4. major — Causal-proof-level verdict under DWP event types is unspecified.**
Record: rule 1 "at least 2 of the 3 event types pass, one of them same-pose" (plan :827); 13.4 stops at the event type.
Fix: "Causal proof PASS when the non-DWP event types alone satisfy rule 1; FAIL when rule 1 would not be satisfied even with every DWP type counted as a pass; otherwise DESIGN_WITHOUT_POWER." (Covers same-pose DWP → DWP, not fail.)

**5. major — Failed runs deadlock the reducer.**
Claim: 13.1 "a run that fails [check_finite] stops before `cer_gates.json` exists and is reported as a failed run, never scored"; 13.4 "the reducer refuses to run when any planned score is missing: 27 CER runs".
Record: a non-zero train exit makes the launcher exit non-zero (`train_ex4dgs_cer.sbatch:68`); the worker then writes `<label>.blocked` and never scores (`ars_score_worker.py:159-163`); a scorer failure writes `.failed`, "never retried" (`:9`, `:199-202`). The worker already anticipates resubmission under `NAME_r2` (`:76-83`), but the launchers refuse an existing run dir (hand-off §4), so a resubmission is a new run id the worker's fixed row does not point at.
Fix: write in 13.4 how a crashed, non-finite or Slurm-failed CER run enters the proof: either (i) it is resubmitted under a new run id recorded in the ledger as the planned run's replacement (a crash is not a gate failure; recommended, matching the worker's `_rN` convention, with the worker row updated), or (ii) the scene is DWP with reason "run failed". The reducer's input is then "27 planned run ids, each with exactly one of {score, declared replacement score}".

**6. major — Multi4D has no scoring path; 14.6's "no substrate code change" is silent on it.**
Record: `ars_score.py:69` choices are `ex4dgs, ex4dgs_cer, multi4d, stg`; the `multi4d` branch requires `substrate_commit` starting `c483e82` (`:111`), so a CER-Multi4D run (00357aa) fails provenance; `ars_score_worker.py:70-73` lists Ex4DGS CER rows only. Hand-off item 3 already names the gap.
Fix: 14.6 states: "the scorer gains a `multi4d_cer` method pinned to 00357aa (branch `cer`, porcelain 0 before and after, the existing Multi4D checks, `<run>/model/cer_gates.json` copied under `run_checks.cer_gates`) and the worker 9 confirmatory rows, before any Multi4D CER run; the pin moves with any later commit." Not a substrate change, but a precondition for 14.4.

**7. minor — the stale `pass` will read `false` on every real run and sits among passed checks.**
Record: forecast residual share 0.02-0.10 < 0.2 → `pass: false` in every `cer_gates.json`, every job log (`cat` at both launcher tails) and every score under `run_checks.cer_gates` (`ars_score.py:95`), whose siblings are checks the scorer passed. No reader acts on it today.
Fix: 13.3/14.6 require the reducer to (i) assert `pass == (superseded formula recomputed from the file's own fields)` as an integrity check that the file came from the pinned `gates()`, and never branch on it; (ii) carry a unit fixture with `pass: false` and all preconditions met that must be scored. Optionally the reducer output renames it `pass_superseded_rule`.

**8. minor — 13.2 and 14.3 disagree on what the residual share is a split of.**
Record: 13.2 "split of the gap content between the residual copies and the ordinary rows"; 14.3 "the split between memory gate and residual copies, as on Ex4DGS". The memory rows carry exactly zero mass in the gap (`memory_mass_gap == 0`), so a mass share cannot be split with the memory gate. 13.2's reading narrows the principal's word "attribution" (plan :1193) and says so, but does not name the departure.
Fix: in 14.3 use 13.2's wording; in 13.2 add "this is narrower than the plan-log's 'attribution between the memory gate and the residual copies'; the causal split needs the wave-2 ablation".

**9. minor — "a scene meets the preconditions when every one of its CER runs does" is one reading of rule 4, not stated as such.**
Record: rule 4 speaks of "a method arm ... on a scene" (plan :833-834); three repeated trainings per scene.
Fix: one sentence: "Read as all three runs (a run whose gate is inert would enter the pooled model as a non-mechanism number); alternatives (majority, mean share) rejected."

**10. minor — 14.4 omits do-no-harm (rule 3) for Multi4D.**
Record: 13.4 names "the bound of rule 3"; 14.4 lists runs, model, endpoints, pass, preconditions, readings, descriptive runs, no rule 3; §4b is also silent.
Fix: "Rule 3 applies to CER-Multi4D, pooled over the three same-pose scenes, same −0.2 dB bound" (the only guard against the memory gate harming the whole frame), or state that it does not and why.

**11. minor — 14.3 attaches the 0.10/0.20/0.30 table to four quantities.**
Record: the principal asked for the table on the residual share (plan :1198-1199); thresholds on "share of ordinary persistent rows" have no meaning.
Fix: table on the residual share and (if finding 1 is taken) the gap-content sum; the transient and ordinary shares as plain values.

**12. minor — "no substrate code change" closes a recorded next step by another route without saying so.**
Record: plan :1206-1207 "Not yet done: ... the code that reports instead of gating"; hand-off item 2 ("`gates()` `pass` no longer requires the residual share ... Multi4D also adds the transient share").
Fix: 13.3 adds "this amendment decides not to change `gates()`; the plan-log item is discharged by the reducer's recomputation (and on Multi4D by the reducer's sum), to keep 5b82ae7 / 00357aa and avoid a further review cycle before submit."

**13. minor — "per scene" in the descriptive tables is undefined.**
Record: 13.2 "per run and per scene, the share and the verdict ≥ 0.10 / 0.20 / 0.30".
Fix: define the per-scene value (mean of the three runs' shares, with min and max shown).

## Verified, no action
- Line references 13.3 (`:159-167`, `:168-170`) and 14.2/14.6 (`:204-206`, `:211`, `:213`, `:216-218`) are correct at the named commits.
- `gates()` runs after the last training iteration and before the final render in both trees (`ex4dgs/train.py:289-290`; `multi4d/train.py:1000-1006`), on the in-memory model; Multi4D restores SH leaves and the training kernel in `finally` (`cer.py:116-118`, `:85-86`), so the scored render is unaffected.
- Multi4D `memory_share_return` denominator includes the transient bank, as 14.2 states.
- The smoke values quoted in 13.2 and 14.2 match the `cer_gates.json` / `cer_begin.json` in the prompt (0.0604 → "0.06"; 0.0239 + 0.00087; fg 0.4501 / bg 0.5485 / tr 0.0014).

Ready to freeze once findings 1-6 are written in (1 needs the principal's word on the reading); 7-13 are wording and can go in the same revision.
