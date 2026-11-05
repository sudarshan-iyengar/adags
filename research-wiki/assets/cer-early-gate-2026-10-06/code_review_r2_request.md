# Code review request: "CER on Multi4D, early gate" (code review, round 2)

You reviewed this code in round 1 (your review: D:\adags\research-wiki\assets\cer-early-gate-2026-10-06\code_review_r1_codex.md;
the round-1 request with the implementer's deviations: code_review_r1_request.md in the same folder). The implementer has
added two commits on branch cer-early in D:\adags-cer\multi4d-early: 51cd832 (tests after review round 1) and 2f3bd6c
(range rule to iterations, lift_select helper), plus edits to D:\adags-cer\agent\train_multi4d_cer_early.sbatch and
D:\adags-cer\agent\cer_tests_m4d_early.sbatch. Review `git -C D:\adags-cer\multi4d-early diff 42e818c..cer-early`, the two
sbatch files, and re-read whatever surrounding code you need. State at the top which files you opened.

The orchestrator's dispositions of your round-1 findings, sent to the implementer:
1. E1 cache isolation -> legacy subprocess gets its own fresh extensions dir under the test OUT dir.
2. Ungated final model -> range rule adds cer_begin_iter < densify_until_iter < iterations; E9 tests the refusals.
3. EXTRA override -> reserved options refused, allowlist --iterations/--phase3_start_iter/--densify_until_iter; resolved
   cer_begin_iter recorded from the run's cfg_args.
4. Unpinned comparisons -> test sbatch refuses unless the early tree is at EXPECT_EARLY on cer-early and clean, and the cer
   tree is at 00357aa on cer and clean, before and after (read-only); training launcher records repo/multi4d-cer HEAD and
   porcelain before and after.
5. E8 witnesses -> both banks required; bottle has no foreground memory rows, so (T3 precedent) visible foreground rows are
   labelled memory for the test; one witness set for all controls; per-witness q=0 positive control.
6. E8 cross-process agreement -> losses and per-row gradients compared with explicit tolerances; disagreement fails.
7. Circular donor check -> one helper scene/cer.py lift_select used by train.py; E4 tests that helper with a forced
   subsample; E1a asserts statically that train.py calls only lift_select at the lift site; E9 requires >= 1 lift after begin.
8. Inherited CUDA `clamped` sizing -> NOT fixed here; goes to the principal as a substrate decision (affects base, 16k and
   early arms alike; already recorded in the spec §8).
Suspected drift-pass/max_contrib interaction -> not acted on (the 16k arm's begin and gates passes share it); noted.

Check: whether each disposition is implemented correctly and completely; any new defect introduced by the fixes
(especially legacy equivalence of the new lift_select path, the tolerance logic in E8, the launcher's EXTRA parsing and
refusals, and the pinning shell logic); and whether anything from round 1 remains. Same output format as round 1
(verdict line READY / READY WITH FIXES / MATERIAL DEFECT; numbered findings with severity, file:line, fix; confirmed vs
suspected). Reserve MATERIAL DEFECT for defects that make the code or its tests wrong.
