## Verdict
MATERIAL DEFECT

Reviewed on my own judgment; no ARIS skill used.

## Confirmed defects

- **High — the nonzero-gradient oracle is invalid for L1.** [test_cer_early.py:304](D:/adags-cer/multi4d-early/test_cer_early.py:304) uses `(render - gt).abs().mean()`. [backward.cu:1395](D:/adags-cer/multi4d-early/diff_gaussian_rasterization_hybrid/cuda_rasterizer/backward.cu:1395) accumulates `weight * dL_dchannel_total`; PyTorch’s absolute-value derivative is `grad * self.sgn()`. Opposite residual signs can cancel exactly. A two-pixel calculation with opacity/colour `0.5`, underlying colours/targets `[0,1]`, gives **colour gradient 0, opacity gradient 0.5**, without clamping. Thus “spurious zero” is not established, even with repeatability. **Fix:** add a positive upstream probe such as `render.sum()` for DC connectivity; retain L1 as diagnostic. Establish opacity expectations separately. [PyTorch source](https://github.com/pytorch/pytorch/blob/v2.5.1/tools/autograd/derivatives.yaml#L220-L222).

- **Medium — E8 can accept incorrect clamp gradients.** [test_cer_early.py:361](D:/adags-cer/multi4d-early/test_cer_early.py:361) uses `o_nz & (d_nz | all_clamped)`, so a fully clamped row with **nonzero DC gradient passes**. Partial-channel mismatches are explicitly “reported, not failing.” Yet [backward.cu:35](D:/adags-cer/multi4d-early/diff_gaussian_rasterization_hybrid/cuda_rasterizer/backward.cu:35) zeros each clamped channel before computing DC gradients. **Fix:** require zero gradients on clamped channels; validate unclamped channels using the controlled probe above.

- **Medium — the proposed background aliasing explanation has the overwrite direction backwards.** [rasterizer_impl.cu:282](D:/adags-cer/multi4d-early/diff_gaussian_rasterization_hybrid/cuda_rasterizer/rasterizer_impl.cu:282) launches transient preprocessing first, then static preprocessing using the same buffer. Static processing writes its own `clamped[3*idx+c]` and static backward reads those entries. Consequently, ordinary overlap corrupts **transient** flags. The supplied example static indices are below the claimed ~85k allocation. **Fix:** separate transient/static clamp buffers and record `P_static`; investigate any overflow separately instead of attributing these background zeros to overlap. Same-stream launches execute in order. [NVIDIA documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html).

## Suspected issues

- **Near-zero clamp classification may differ.** The SH signs/constants, `pos - campos`, transient degree, first-16 layout, and foreground deformation inputs match the inspected source. However, [preclamp():413](D:/adags-cer/multi4d-early/test_cer_early.py:413) recomputes normalization/arithmetic through tensor operations; that does not prove numerical identity with CUDA. Fused arithmetic can change rounding. **Settle it:** compare actual static forward RGB/clamp flags against the helper, especially near zero. [NVIDIA floating-point documentation](https://docs.nvidia.com/cuda/floating-point/index.html#the-fused-multiply-add-fma).

## Unverified claims

- GPU pass/failure counts, the E5 exception, 881-witness measurements, exact repeatability, and successful post-commit behavior lack raw logs/checkpoints in the referenced checkout. Syntax checks passed; runtime tests were not run because local Python lacks PyTorch.

## Missing

- **The training build reads unwritten `max_contrib`.** Its writer is guarded by `CER_WRITE_MAX_CONTRIB` at [forward.cu:735](D:/adags-cer/multi4d-early/diff_gaussian_rasterization_hybrid/cuda_rasterizer/forward.cu:735); default build flags omit that macro, while [backward.cu:1229](D:/adags-cer/multi4d-early/diff_gaussian_rasterization_hybrid/cuda_rasterizer/backward.cu:1229) uses the buffer for an early return. Its effect on these failures remains unresolved.
- E8 checks only foreground/background gradients; it does not examine the transient gradients directly affected by clamp-buffer overlap.