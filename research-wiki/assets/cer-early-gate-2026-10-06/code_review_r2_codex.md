Opened: round-1 review/request, round-2 request, rev-4 plan and CER spec; `multi4d-early`’s arguments, scene, training, test, parameter-merge and renderer files; hybrid `rasterizer_impl.cu`, `forward.cu`, `backward.cu`; both early and legacy launcher pairs.

## Verdict

MATERIAL DEFECT

## Confirmed defects

1. **Major — both argument allowlists remain bypassable.** [Training launcher:40](/D:/adags-cer/agent/train_multi4d_cer_early.sbatch:40) validates `read -r -a EXTRA_WORDS`, but line 123 executes raw `$EXTRA`. I reproduced `EXTRA=$'--iterations 4000\n--cer_begin_iter -1'` passing validation; the actual argument parser resolves begin to `-1`. [Test launcher:34](/D:/adags-cer/agent/cer_tests_m4d_early.sbatch:34) similarly accepts a second-line `--cer_tree /unreviewed/tree`, bypassing comparison pinning. Bash reads one line during validation but splits unquoted expansions on newlines during execution. [Bash documentation](https://www.gnu.org/s/bash/manual/html_node/Bash-Builtins.html), [word splitting](https://www.gnu.org/s/bash/manual/html_node/Word-Splitting.html). **Fix:** reject multiline input and execute only the validated, quoted arrays.

2. **Major — E8 does not implement the claimed per-witness positive control.** [test_cer_early.py:920](/D:/adags-cer/multi4d-early/test_cer_early.py:920) requires only `2 * active >= w_gap`, counting any parameter’s gradient. Pre-gap requires merely one active row. An in-memory `_e8_flags` reproduction passed with half the witnesses inactive and every witness’s opacity/DC gradient zero. **Fix:** validate the required opacity/DC gradients for each tested witness; use those same identified rows for the subsequent zero checks.

3. **Major — E8’s comparator accepts non-finite results.** [test_cer_early.py:948](/D:/adags-cer/multi4d-early/test_cer_early.py:948) uses `abs(a_ - b_) > tolerance` without checking finiteness. The extracted comparator accepted both NaN versus `0.1` and infinity versus `0.1`. Gradient comparisons at lines 962–966 have the same omission. **Fix:** reject non-finite losses and gradient tensors before applying tolerances; [torch.isfinite](https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html) provides the tensor check.

4. **Major — Git inspection failures are interpreted as cleanliness.** [Test launcher:44](/D:/adags-cer/agent/cer_tests_m4d_early.sbatch:44) embeds `status --porcelain | wc -l` inside an `echo`; [training launcher:67](/D:/adags-cer/agent/train_multi4d_cer_early.sbatch:67) similarly compares its output. Fault injection making every `git status` return 128 still produced three “clean” states and `check_trees_rc=0`. **Fix:** check each Git command’s exit status explicitly before evaluating its output.

5. **Critical, inherited — the CUDA allocation defect remains.** [rasterizer_impl.cu:160](/D:/adags-cer/multi4d-early/diff_gaussian_rasterization_hybrid/cuda_rasterizer/rasterizer_impl.cu:160) allocates `clamped` with `P * 3` entries. Static preprocessing passes that buffer to [forward.cu:502](/D:/adags-cer/multi4d-early/diff_gaussian_rasterization_hybrid/cuda_rasterizer/forward.cu:502), which writes `clamped[3 * idx + …]` for static indices. Eligible static indices beyond the transient count exceed the allocation. **Fix:** resolve the substrate decision with separate, correctly sized buffers and corresponding validation. The deferral leaves this round-1 defect unresolved.

## Suspected issues

- **Drift-pass neutrality remains unestablished.** Training backward reads `max_contrib` at [backward.cu:1229](/D:/adags-cer/multi4d-early/diff_gaussian_rasterization_hybrid/cuda_rasterizer/backward.cu:1229); its forward write remains instrument-only. Sharing this defect with other arms does not establish that the additional drift pass leaves training unchanged. Settle with matched continuations with and without that pass.

## Unverified claims

- No GPU suite, numerical legacy-equivalence run or E9 integration ran here: local Python lacks PyTorch. Predicate probes and Bash reproductions establish the findings above; compilation and syntax checks passed.
- Cluster environment, wrapper behavior, live tree states and the bottle foreground-memory census were not independently verified.

## Missing

- A revised F1 recipe: the plan’s `--iterations 3500` command now fails [the new range assertion](/D:/adags-cer/multi4d-early/scene/cer.py:176) with default freeze iteration 16000.
- Direct donor/RNG equivalence testing of the unarmed `lift_select` helper under forced subsampling.
- Training-launcher checks for the protected pinned baseline, and enforcement of recorded before/after tree states.