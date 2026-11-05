# Code review request: "CER on Multi4D, early gate" (code review, round 3)

Your rounds 1 and 2: D:\adags\research-wiki\assets\cer-early-gate-2026-10-06\code_review_r{1,2}_codex.md (requests beside
them). The implementer has added 9a9a747 (tests after round 2) and 6b75e9c (begin before the final iteration; freeze only
before it) on branch cer-early in D:\adags-cer\multi4d-early, and edited D:\adags-cer\agent\train_multi4d_cer_early.sbatch
and D:\adags-cer\agent\cer_tests_m4d_early.sbatch. Review `git -C D:\adags-cer\multi4d-early diff 2f3bd6c..cer-early`, the
two sbatch files in full, and any surrounding code. State which files you opened.

Dispositions of your round-2 findings sent to the implementer:
1. Newline bypass -> newlines/CR refused in EXTRA, RENDER_AT, T7_ARGS, SCENES, TESTS, REP; only the validated, quoted
   arrays are executed.
2. Git failures read as clean -> every git call's exit status checked; failure = refusal before / failure after; the
   training launcher now checks the pinned and cer trees before (refuse) and after (non-zero exit on change).
3. F1 recipe broken by the range rule -> rule is now cer_begin_iter < densify_until_iter and cer_begin_iter < iterations;
   the freeze happens iff densify_until_iter < iterations; cer_gates.json records "frozen"; the confirmatory scorer will
   require frozen == true and the full schedule (scorer change is made at submission time, not now).
4./5. E8 per-witness opacity and DC-colour gradients at q=0, exact zeros at q=1 for the same rows, pre-gap positives for
   the visible ones; non-finite values rejected before tolerances.
6. Unarmed lift_select vs direct select_fg_to_transient: bit-identical donors and RNG under a forced subsample (E4).
Inherited CUDA `clamped` sizing and the drift-pass/max_contrib suspicion: escalated to the principal, not changed here.

GPU results already in (on 2f3bd6c, before these two commits): existing T0, T1, T2, T2b, T3 PASS on the new tree; E1a,
E1b (legacy begin equals the cer tree: memory [fg 0, bg 2525], 6 donors), E1c (unlabelled lifecycle hashes equal to the
cer tree), E4 PASS. Tree pins held before and after.

Check that each disposition is implemented correctly; any new defect introduced; anything left from rounds 1-2 other
than the two escalated items. Verdict line READY / READY WITH FIXES / MATERIAL DEFECT; numbered findings (severity,
file:line, fix; confirmed vs suspected). Reserve MATERIAL DEFECT for defects that make the code or its tests wrong; do not
re-raise the two escalated items as defects of this change (list them once under "escalated").
