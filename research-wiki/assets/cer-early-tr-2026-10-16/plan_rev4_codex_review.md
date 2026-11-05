## Verdict

MATERIAL DEFECT

Independent judgment; no ARIS skill used.

## Confirmed defects

1. **Moderate — a failed probe job can still be accepted.** The launcher writes `"exit_code=$rc"` before its worktree check can change `rc` from 0 to 1 ([densify_probe.sbatch:78](/D:/adags-cer/agent/densify_probe.sbatch:78), line 79). The reader trusts that recorded value ([densify_read.py:14](/D:/adags-cer/agent/densify_read.py:14)). An in-memory reproduction accepted a completed payload recording `exit_code=0` and a changed worktree. **Fix:** finalize all checks before recording the exit code; require matching before/after substrate states in the reader.

2. **Moderate — the neutrality test still overstates its coverage.** Its digest lists nine tensors and reads only `gr["params"][0]` and the two Adam moment buffers ([test_densify_probe.py:182](/D:/adags-cer/agent/test_densify_probe.py:182), lines 185–189). It omits foreground buffers such as `mean_radii2D` and `_deformation_accum`, deformation parameters, optimizer step counters and group metadata. Foreground groups contain parameter lists ([gaussian_model.py:321](/D:/adags-cer/multi4d-early/scene/gaussian_model.py:321)). Changing omitted buffers and a step counter left the extracted digest unchanged. **Fix:** compare every relevant parameter/buffer and the complete optimizer state, including all parameters within each group. [PyTorch’s state format](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.state_dict.html) includes both per-parameter state and group metadata.

## Suspected issues

- **Whole-training neutrality remains unestablished.** The probe allocates CUDA tensors during accounting ([densify_probe.py:69](/D:/adags-cer/agent/densify_probe.py:69)); the training backward reads `max_contrib` ([backward.cu:1229](/D:/adags-cer/multi4d-early/diff_gaussian_rasterization_hybrid/cuda_rasterizer/backward.cu:1229)), whose initialization is already an acknowledged substrate problem. The neutrality sequence exercises lifecycle calls, without rendering/backward ([test_densify_probe.py:210](/D:/adags-cer/agent/test_densify_probe.py:210)). **Settle:** compare paired wrapped/unwrapped training segments including rendering and backward, or narrow the neutrality claim.

## Unverified claims

- F5’s nine positive controls and the proposed transient implementation remain unexecuted here: the available Python environment has no PyTorch.
- Historical PSNR/alpha values, donor counts and GPU-hour estimates were not independently recomputed from primary outputs.
- D1’s unconditional validation and the revised F5 fixture are specified fixes, not verified implementation.

## Missing

- A regression assertion for the **corrected** average: the test named “both averages over the same rows” checks only `grad_mean_seen`, never `grad_u_mean_seen` ([test_densify_probe.py:149](/D:/adags-cer/agent/test_densify_probe.py:149)). Add the round-3 `[2,0]` example and assert both means equal 1, with one zero corrected denominator.