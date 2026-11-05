## Verdict

MATERIAL DEFECT — not ready for implementation.

Code opened at `00357aa`. [train.py:748](D:/adags-cer/multi4d/train.py:748): `loss += l1_loss(image_tensor_hybrid_full, gt_image_tensor)`.

Rev-1 dispositions, at plan level. “P” refers to the [on-disk rev-2 plan](D:/adags/research-wiki/operations/cer-multi4d-early-gate-plan-2026-10-06.md).

| # | Disposition | Evidence |
|---|---|---|
| 1 | Rationale fixed; F1 wrapper claim unverified. | P:81–98 withdraws the claim; `train.py:748–752` confirms L1 precedes merging. |
| 2 | **Partially fixed.** Immutable membership fixes relabel leaks; begin validation still permits previous copies. | P:124–153; `train.py:916–936`; finding 1 below. |
| 3 | Fixed as specified. | P:234–242 requires populated moments, resets and complete RNG comparisons; NumPy resets exist at `gaussian_model.py:578,935`. |
| 4 | **Partially fixed.** E7 has witnesses and E9 carries §14.2; E8 retains a vacuity hole. | P:246–248; `test_cer.py:308–315`; finding 4 below. |
| 5 | Fixed as specified; implementation pending. | P:222–229 explicitly passes, records and validates the argument. Existing launcher passes additional arguments at `D:/adags-cer/agent/train_multi4d_cer.sbatch:93`. |
| 6 | Fixed. | P:173–177 distinguishes gradient contribution from statistics; `forward.cu:510` assigns radius and `gaussian_model.py:117` increments `denom`. |
| 7 | Wording fixed. | P:34–35 limits equality to checked PNG values; `cer_m4d_banks.py:143–146` compares quantized images. |
| 8 | Cache description fixed. | P:212–217 agrees with the official [PyTorch versioner](https://github.com/pytorch/pytorch/blob/v2.2.0/torch/utils/_cpp_extension_versioner.py) and [loader](https://raw.githubusercontent.com/pytorch/pytorch/v2.1.2/torch/utils/cpp_extension.py). |

## Confirmed defects

1. **Major — D4 still accepts starts after ungated copies exist.** P:153 rejects only `fg_to_tr_start_iter < cer_begin_iter < phase3_start_iter`, while P:77 accepts starts through 16,000. Begin **10,000 or later** therefore remains valid after earlier transfers. [train.py:916](D:/adags-cer/multi4d/train.py:916) permits transfers before phase 3; [gaussian_model.py:415](D:/adags-cer/multi4d/scene/gaussian_model.py:415) explicitly describes copying donors. Those transient copies remain ungated. **Fix:** restrict explicit early starts to `begin <= fg_to_tr_start_iter`, retain `-1` as legacy, and test rejection at 10,000 and later.

2. **Major — E9’s schedule ordering does not exclude coarse-stage copies.** Lifting has **no stage guard** at [train.py:916](D:/adags-cer/multi4d/train.py:916), and transient insertion occurs at `train.py:211–213`. Coarse and fine training reuse the same banks (`:1032–1052`). For example, coarse=1,000, begin=merging=100, phase3=300, densify_until=800, iterations=900 satisfies E9 but allows coarse transfers at 150/200/250 before fine CER begins. **Fix:** require `coarse_iterations <= transient_merging_iter` for this design and verify that no transfer occurred before fine begin.

3. **Major — E5’s feature edits cannot induce the intended drift.** P:244 proposes changing memory features so rows cease painting the target. The membership instrument [replaces every bank’s features](D:/adags-cer/multi4d/scene/cer.py:92) with white DC features (`:103–106`). Feature edits cannot change that membership measure. **Fix:** perturb known rows’ geometry or deformation to cross the majority threshold; independently establish expected nonzero drift, then check counts and masses.

4. **Major — E8 can still pass with vacuous memory zeros.** Nonzero ordinary-row gradients in a gap frame and memory gradients in another frame do not establish that the tested memory rows could receive **gap-frame** gradients. The training backward can skip their buckets using [unwritten `max_contrib`](D:/adags-cer/multi4d/diff_gaussian_rasterization_hybrid/cuda_rasterizer/backward.cu:1229); the write is instrument-only (`forward.cu:735–747`). Existing [T3](D:/adags-cer/multi4d/test_cer.py:308) supplies a same-gap-frame `q=0` memory-gradient control, but on the instrument build. **Fix:** add that control to E8 on the training build with identified memory witnesses, followed by `q=1` exact-zero checks.

## Suspected issues

- D4’s score multiplication excludes labelled donors only with a nonnegative threshold: `0 > negative_threshold` passes [the selector](D:/adags-cer/multi4d/scene/gaussian_model.py:423). The current default is positive. Settle this by asserting a nonnegative threshold or using an explicit eligibility mask.

## Unverified claims

- The native loop supports F1’s 3,500 stop and preserves the pre-begin schedule. However, “`iterations` is read only by end-of-training checks” is false: it supplies the fine-loop bound at `train.py:1050`. The final-render wrapper was not available locally, so its saving behavior remains unverified.
- Current Leonardo jobs, remote cleanliness, installed PyTorch version, runtime estimates and numerical results were not independently verified. No GPU tests ran.

## Missing

- Explicit plumbing from `ModelParams.cer_begin_iter` into training/CER. Parameter extraction keeps groups separate (`arguments/__init__.py:41–46`); merely declaring the argument does not put it in `optimization_params`.
- E8 must actually accumulate densification statistics and check denominator increments. Backward alone does not exercise [that update](D:/adags-cer/multi4d/scene/gaussian_model.py:114); initially finite buffers would make the finiteness check vacuous.