## Verdict

MATERIAL DEFECT

Independent judgment; no ARIS skill used.

## Confirmed defects

1. **Major — the deletion-sensitivity ratio changes its averaging population.** The probe averages ordinary gradients over `denom > 0`, but corrected gradients over `du > 0` ([densify_probe.py:140](/D:/adags-cer/agent/densify_probe.py:140), lines 147–149). The reader divides these means ([densify_read.py:68](/D:/adags-cer/agent/densify_read.py:68)). Example: accumulators `[2,0]`, denominators `[1,1]`, zero increments `[0,1]` produce unchanged gradient vectors `[2,0]`, yet the reported ratio is **2**. **Fix:** compare both vectors over the same rows, retaining the specified zeros; report corrected-zero counts separately.

2. **Moderate — incomplete accounting can be marked complete and accepted.** An unmatched visibility call merely increments `unmatched`, omitting its zero increments ([densify_probe.py:97](/D:/adags-cer/agent/densify_probe.py:97)); completion still writes `"complete": True` ([line 341](/D:/adags-cer/agent/densify_probe.py:341)). The reader checks only that boolean and silently skips absent snapshots ([densify_read.py:11](/D:/adags-cer/agent/densify_read.py:11), [line 60](/D:/adags-cer/agent/densify_read.py:60)). An in-memory execution accepted empty records, missing snapshots and nonzero unmatched counts. It also never checks the required job exit status. **Fix:** validate exit status, both populated snapshots and complete visibility accounting before producing ratios.

3. **Moderate — the probe test does not implement the promised mask and neutrality checks.** It checks only `added == clone.sum() + split.sum()` ([test_densify_probe.py:145](/D:/adags-cer/agent/test_densify_probe.py:145)); wrong row selections with equal counts pass. Its “unwrapped” control still calls densifiers through patched class postfix/prune methods ([line 173](/D:/adags-cer/agent/test_densify_probe.py:173)), and exercises only background banks. **Fix:** capture and compare actual clone/split masks for both banks; use a genuinely unwrapped control and compare optimizer state as well.

## Suspected issues

- **F5’s fixture remains checkpoint-dependent.** Copying a witness leaves its covariance and spatial configuration uncontrolled. Setting `t = timestamp` makes the temporal-marginal derivative zero; remaining time sensitivity depends on spatial–temporal covariance ([backward.cu:719](/D:/adags-cer/multi4d-early/diff_gaussian_rasterization_hybrid/cuda_rasterizer/backward.cu:719), line 728). Contribution alone does not establish nine nonzero parameter-group gradients. **Settle:** specify deterministic geometry/covariance and demonstrate all nine positive controls before using the fixture.

## Unverified claims

- Historical PSNR/alpha figures and runtime estimates were not independently recomputed from primary outputs.
- Snapshot and restart fixes are present in the probe; GPU execution remains unverified because this environment lacks PyTorch/CUDA.
- The test changed during inspection. Findings above use its final observed SHA256 `81852AF78DA28E29621402A31E70929C6C49A7C313515002197FB35FA56D1608`; the earlier same-camera issue was removed.

## Missing

- **D1 needs an explicit validation-call edit.** Currently, `check_begin_range` is called only when `cer_begin_iter != -1` ([train.py:1027](/D:/adags-cer/multi4d-early/train.py:1027)). Adding assertions inside that function alone cannot reject `cer_transient_gate=1, cer_begin_iter=-1`. Invoke flag validation unconditionally, before scene construction.