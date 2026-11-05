## Verdict

MATERIAL DEFECT

Opened the full rev-4 plan, referenced spec and rev-3 review; all six changed files; renderer, gate, configuration, parameter merge and evaluation code; base `test_cer.py`; both launcher pairs; and relevant hybrid CUDA sources. Changed-line evidence, [scene/cer.py:194](D:/adags-cer/multi4d-early/scene/cer.py:194): `deformed_state["dynamic_score"] = score.masked_fill(labelled, float("-inf"))`.

## Confirmed defects

1. **Major — E1 violates extension-cache isolation.** [Test launcher:29](D:/adags-cer/agent/cer_tests_m4d_early.sbatch:29) exports `TORCH_EXTENSIONS_DIR_CER=$W/build/torch_extensions_multi4d_cer`. [E1:501](D:/adags-cer/multi4d-early/test_cer_early.py:501) passes it to the legacy subprocess, whose renderer calls JIT `load()`. This permits locks, build-file writes and rebuilds in the protected cache; JIT loading is a build operation. [PyTorch loader source](https://raw.githubusercontent.com/pytorch/pytorch/v2.1.2/torch/utils/cpp_extension.py). **Fix:** give legacy comparisons a separate cache under the early-test directory.

2. **Major — an accepted schedule can export an ungated final model.** [Range validation:160](D:/adags-cer/multi4d-early/scene/cer.py:160) never requires `iterations > cer_begin_iter`; I reproduced acceptance of both being 3000. [train.py:894](D:/adags-cer/multi4d-early/train.py:894) saves, then [1002](D:/adags-cer/multi4d-early/train.py:1002) evaluates, before [1013](D:/adags-cer/multi4d-early/train.py:1013) calls `begin()`. The extracted AST produces `save → evaluation/wrapper render → begin`. Under the supplied wrapper behavior, final PNGs are ungated and the final save lacks CER state, while subsequent gates inspect the armed model. **Fix:** reject `iterations <= cer_begin_iter` and test this refusal.

3. **Major — `EXTRA` bypasses the launcher’s early-arm refusal.** [Launcher:105](D:/adags-cer/agent/train_multi4d_cer_early.sbatch:105) appends `$EXTRA` after `--cer_begin_iter "$CER_BEGIN_ITER"`. `EXTRA="--cer_begin_iter -1"` overrides the validated value; I reproduced the parser returning `-1`. Provenance still records the positive environment value. **Fix:** reject reserved options in `EXTRA`, allow only intended schedule overrides, and record resolved arguments.

4. **Major — test comparisons are not pinned.** [Test launcher:34](D:/adags-cer/agent/cer_tests_m4d_early.sbatch:34) merely echoes HEAD and cleanliness; it never refuses incorrect or dirty trees. [E1:501](D:/adags-cer/multi4d-early/test_cer_early.py:501) compares against whatever currently occupies `CER_TREE_LEGACY`. The training launcher checks only its early worktree, contrary to the plan’s protected-worktree checks. **Fix:** enforce expected commits and cleanliness before tests, and verify protected HEADs and status afterward.

5. **Major — E8’s witness controls are incomplete.** [Test:828](D:/adags-cer/multi4d-early/test_cer_early.py:828) allows foreground checks to disappear with `"no foreground memory witnesses: skipped"`. [323–329](D:/adags-cer/multi4d-early/test_cer_early.py:323) also uses different gap and pre-gap witness sets. Positive controls require merely one nonzero row, leaving other witness buckets unchecked. **Fix:** require nonempty witnesses in both banks, carry identifiable witnesses through the controls, and establish nonzero control gradients for the rows whose subsequent zeros are being validated.

6. **Major — E8 accepts numerical disagreement between fresh processes.** [Test:865](D:/adags-cer/multi4d-early/test_cer_early.py:865) compares only Boolean flags and witness counts. `loss_diff` is reported but never tested. Direct execution of `_e8_flags` accepted processes with different opacity-gradient counts and losses differing by **0.8**. **Fix:** compare losses and per-row gradient results with explicit tolerances; propagate disagreement as failure.

7. **Major — E9’s donor-label check is circular.** [Test:939](D:/adags-cer/multi4d-early/test_cer_early.py:939) accepts exit zero plus `labelled_candidates_excluded >= 0`, which cannot demonstrate that selected donors were unlabelled. [E4:633](D:/adags-cer/multi4d-early/test_cer_early.py:633) invokes masking itself rather than exercising the training call site. Deleting that call site is not independently caught by these checks. **Fix:** inspect selected donor indices and source labels, or exercise the actual training selection path with labelled candidates and a positive control.

8. **Critical, inherited — the hybrid `clamped` buffer remains incorrectly sized.** [rasterizer_impl.cu:160](D:/adags-cer/multi4d-early/diff_gaussian_rasterization_hybrid/cuda_rasterizer/rasterizer_impl.cu:160) allocates `P * 3` entries for transient rows. Static preprocessing ranges over `P_static`, then calls the SH routine at [forward.cu:502](D:/adags-cer/multi4d-early/diff_gaussian_rasterization_hybrid/cuda_rasterizer/forward.cu:502), which writes `clamped[3 * idx + …]` at lines 62–64. Visible static indices beyond `P` exceed that logical allocation and can overwrite neighboring buffers. This predates the patch; no tracked CUDA source changed. **Fix:** resolve through a separate substrate decision with correctly separated buffers and corresponding baseline validation.

## Suspected issues

- **The drift pass may affect subsequent training through the inherited uninitialized-memory defect.** Training backward reads `max_contrib` at [backward.cu:1229](D:/adags-cer/multi4d-early/diff_gaussian_rasterization_hybrid/cuda_rasterizer/backward.cu:1229); its forward write is instrument-only. Restoring Python tensors does not establish numerical neutrality under this defect. **Settle:** compare identical training continuations with and without the drift pass, after correcting E8’s agreement checks.

## Unverified claims

- No GPU tests, real integration run or numerical legacy-equivalence comparison ran here. Six files compile without writes.
- Cluster paths, installed dependencies, protected remote trees, and the complete wrapper/environment scripts were unavailable. The boundary finding uses the supplied wrapper behavior.
- Historical PSNR and PNG-equality claims were not reproduced.
- E9’s proposed shortening flags **do survive** merging: the resolved schedule is coarse 2000, merging/begin 3000, phase3 3200, densify-until 3500, iterations 4000.

## Missing

- Gradient and memory-safety validation at actual early-stage bank sizes; E8 loads a 20,000-iteration stand-in.
- Negative tests for reserved `EXTRA` overrides, end-at-begin schedules, wrong comparison commits, and forbidden cache writes.