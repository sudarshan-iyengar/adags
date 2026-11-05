# Code review request: "CER on Multi4D, early gate" implementation (code review, round 1)

You are an independent adversarial code reviewer. The design was agreed with you in three plan-review rounds; the agreed
plan (rev 4) is D:\adags\research-wiki\operations\cer-multi4d-early-gate-plan-2026-10-06.md (read it first, all of it;
your plan reviews are in D:\adags\research-wiki\assets\cer-early-gate-2026-10-06\). Now review the CODE that implements it.
State at the top which files you opened (quote one changed line from scene\cer.py to prove it).

Where the code is (workspace D:\):
- Git worktree D:\adags-cer\multi4d-early, branch cer-early: base 00357aa (= branch cer), then
  b842f41 "tests E1-E9 (written before the code)" (new file test_cer_early.py) and
  42e818c "implementation" (arguments\__init__.py, scene\__init__.py, scene\cer.py, scene\gaussian_model.py, train.py).
  Use `git -C D:\adags-cer\multi4d-early diff 00357aa..42e818c` and read the full files around every hunk.
- New launcher and test job scripts: D:\adags-cer\agent\train_multi4d_cer_early.sbatch (compare with
  train_multi4d_cer.sbatch there) and D:\adags-cer\agent\cer_tests_m4d_early.sbatch (compare with cer_tests_m4d.sbatch).
- The unmodified base for comparison: D:\adags-cer\multi4d (branch cer @ 00357aa), including test_cer.py.
- The final-render wrapper lives only on the cluster; its relevant behaviour: it replaces utils.eval_utils.run_evaluation
  at runtime, calls the original, and at each target fine iteration (default [opt.iterations]) renders every test camera
  with render_rawall3pc (black background) and saves PNGs; it exits non-zero if nothing was rendered.

Check, and report every defect you can substantiate:
1. Scientific: is the gate exact at every render path from begin on; can any memory row, any descendant of one, or any
   copy of one paint a gap frame (lift, split, clone, residual birth, begin timing, coarse stage); are labels chosen once
   by the spec rules and inherited exactly; does the early arm differ from the 16k arm only as the plan states; is the
   drift report descriptive only.
2. Legacy equivalence: with cer_begin_iter = -1 (default), is training behaviour identical to cer@00357aa (same calls,
   same RNG consumption, same order; note any added work and whether it can change any tensor or RNG stream).
3. Engineering: label/row agreement after every row-changing call in both persistent banks (clone, split incl. parent
   removal and .repeat(N) ordering vs new_xyz, prune_points, prune, weight_prune, residual birth); optimizer state; O_EXCL
   files; NaN-restart behaviour; the -inf eligibility mask and anything else that reads dynamic_score; counters; error
   paths; the launcher's refusals and pins; isolation (worktree, extensions directory, never writing into
   repo/multi4d-cer or build/torch_extensions_multi4d_cer).
4. Tensor shapes / dtypes / devices: cer_labels [N] uint8 cuda vs factor [N,1] in utils\cer_gate.py; masks; cat dims.
5. CUDA: no CUDA source changed (verify); training-build backward and the instrument kernel usage; anything in the hybrid
   rasterizer that would let a zero-opacity persistent row contribute or receive gradient.
6. Tests (test_cer_early.py and the job script): does each of E1-E9 test what the plan's table says, fail when the thing
   it guards is broken, and avoid vacuous passes (witnesses, positive controls, non-empty selections, RNG states incl.
   numpy/python/all CUDA devices, fresh processes for E8, E9 refusal cases and ordering)? Will they actually run (imports,
   CLI, paths on the cluster: W=/leonardo_work/EUHPC_D36_068/sri/proj_adags, scenes under
   /leonardo_scratch/large/userexternal/siyengar/ars/wave1, wrapper $W/agent-control/ars/m4d_final_render.py)? Look for
   Python errors that only appear at runtime (wrong attribute names, wrong signatures, undefined names).

Output: a verdict line (READY / READY WITH FIXES / MATERIAL DEFECT), then numbered findings with severity
(critical/major/minor), evidence (file:line), and a concrete fix. Separate "confirmed" from "suspected". Terse; no restating.

## The implementer's own report of deviations from the plan (verify each; do not take them on trust)
1. cer_drift.json has extra fields (rows per bank, lifts, transient_target_share, per-label counts).
2. Lift counting is by SELECTION with >= 1 donor (the transfer is applied in maintain_transient whenever
   visibility_filter_transient is not None); cer_begin.json also gets lifts_before_begin_by_stage; scene.cer_lifts holds
   labelled_candidates_excluded.
3. begin() in early mode ASSERTS lifts_before_begin == 0 and 1 <= iteration <= transient_merging_iter (plan: record only).
4. The range rule adds cer_begin_iter < densify_until_iter.
5. mask_lift_scores also asserts no labelled row satisfies score > threshold after masking.
6. The train launcher refuses a non-positive-integer CER_BEGIN_ITER and a worktree not on branch cer-early.
7. Launcher comments changed; job name unchanged.
8. E1a passes on the unmodified tree by design (it checks guards exist and that CUDA sources, renderer and gate are
   unchanged vs 00357aa), not that new code is present.
9. E4 sets labels by hand (armed gate, random labels, overridden scores) rather than calling begin().
10. E8 checks the foreground bank only if it has witnesses (else skipped with a note); the background bank must have
    witnesses.
11. The implementer says --transient_merging_iter and --coarse_iterations CANNOT be overridden on the command line
    because merge_hparams applies the config (arguments\dynerf.py) afterwards, so E9's shortened schedule keeps coarse
    2000 / merging 3000 and uses EXTRA="--phase3_start_iter 3200 --densify_until_iter 3500 --iterations 4000" with
    CER_BEGIN_ITER=3000. VERIFY which of these flags survive merge_hparams (utils\params_utils.py) — if phase3_start_iter,
    densify_until_iter or iterations are also overwritten by the config, E9 would silently run the full schedule.
12. E9 reads a finished run's outputs (pattern of T7) and runs the refusal subprocesses itself; args --e9_run <run dir>
    [--e9_metrics <file>] passed through T7_ARGS.
Static only so far: py_compile on all six files; nothing has run on a GPU.
