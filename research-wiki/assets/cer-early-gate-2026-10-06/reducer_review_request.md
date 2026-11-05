# Code review request: the CER reducer extended to the early-gate arm (one round, focused)

Context: early-gate CER on Multi4D (plan D:\adags\research-wiki\operations\cer-multi4d-early-gate-plan-2026-10-06.md;
you reviewed its code in five rounds, records in D:\adags\research-wiki\assets\cer-early-gate-2026-10-06\). The principal
froze the confirmatory design in the plan log D:\adags\research-wiki\operations\ars-fast-path-plan-2026-09-23.md, entry
"2026-10-07: principal's decisions on the early gate" (read it): 9 early runs vs the 9 Multi4D base runs; rules 1, 3, 4 as
on Multi4D (spec D:\adags\research-wiki\operations\cer-multi4d-spec-2026-09-30.md section 14.4); rule 4 adds frozen ==
true, mode early, begin 3000, 0 lifts before begin; descriptive: vs the 16k arm, transient share, drift. The 9 runs are
queued now; nothing has been scored.

Files (workspace D:\): D:\adags-cer\agent\cer_reduce.py (previous version cer_reduce.pre_m4dearly.py, diff
cer_reduce_m4dearly.diff) and test_cer_reduce.py (previous test_cer_reduce.pre_m4dearly.py, diff
test_cer_reduce_m4dearly.diff). The scorer that writes the inputs: D:\adags-cer\agent\ars_score.py, method
multi4d_cer_early (it carries run_checks.cer_begin, cer_gates and cer_drift into each score; diff ars_score_m4dearly.diff).
Run state: all 14 reducer tests pass locally (python test_cer_reduce.py). Three mutations were tried: ignoring the early
preconditions and weakening `frozen is True` were both caught; a mutation making rule 1 depend on the 16k comparison was
inert because that comparison is computed after rule 1.

Check: (1) the early arm is reduced exactly by the frozen design (manifest, Holm family of 2, rules 1/3/4 clauses, the
added preconditions, nothing else changed for ex4dgs and multi4d); (2) refusals vs design-without-power are right
(missing fields refuse; wrong values fail rule 4); (3) the descriptive 16k comparison and drift shares cannot influence a
verdict and are correctly computed; (4) the scorer's carried fields match what the reducer reads (names, types, where
cer_drift lives); (5) the tests would fail if the new logic broke; anything untested that matters; (6) JSON
serialisability of every new output value (main() json.dumps the result).
Output: verdict line (READY / READY WITH FIXES / MATERIAL DEFECT), findings with severity, file:line, fix. Terse.
