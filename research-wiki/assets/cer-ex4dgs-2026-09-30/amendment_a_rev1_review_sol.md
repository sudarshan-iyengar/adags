## Verdict

MATERIAL DEFECT

Independent judgment; no ARIS skill used. Not ready as written. Ready to implement after the specification/test fixes below; GPU exactness remains to be demonstrated.

| Area | Score | Reason |
|---|---:|---|
| Rule and exactness | 8/10 | Matching primitive parameters does not guarantee matching composite contributions. |
| A2 temporal opacity | 9/10 | Keeping `b_j` preserves donor opacity; temporal gradients can remain zero throughout the plateau. |
| A3 bookkeeping | 9/10 | The eight parameters and nine dynamic buffers match the code; concatenating rows preserves K. |
| Lifecycle | 7/10 | The claimed cap assertion is absent; NaN-gradient handling needs explicit treatment. |
| Surgical scope | 9/10 | Birth/report changes fit in `cer.begin`; existing renderer, pruning and persistence support both banks. |
| Tests | 4/10 | T2c has an invalid universal criterion and insufficient non-vacuity requirements. |

## Confirmed defects

1. **High — T2c’s whole-image byte-equality requirement is not guaranteed.**  
   [`rasterizer_impl.cu:90–108`](D:/adags-cer/ex4dgs/submodules/diff_gaussian_rasterization_df/cuda_rasterizer/rasterizer_impl.cu:90) emits entries in row order and constructs keys from tile and depth. NVIDIA documents that “DeviceRadixSort is stable.” [CUB source](https://raw.githubusercontent.com/NVIDIA/cub/1.15.0/cub/device/device_radix_sort.cuh)  
   An appended copy can move behind an ordinary row with equal depth. [`forward.cu:382–418`](D:/adags-cer/ex4dgs/submodules/diff_gaussian_rasterization_df/cuda_rasterizer/forward.cu:382) composites using `C[ch] += ... * alpha * T`. Two overlapping red/blue rows with alpha 0.5 produce different colours when their order reverses. Disabling the donor does not restore its original position in that ordering.  
   **Fix:** Define “starts as donor” as equality of primitive inputs. Compare evaluated getters across frames and render donor/copy in controlled isolation or equivalent ordering. Treat whole-scene swap equality as diagnostic. This objection does **not** invalidate q=0 RGB identity: existing active rows retain their relative order.

2. **High — A6 permits vacuous dynamic-copy coverage and invisible-copy equality.**  
   Requiring only one residual **in total** permits `D_d = 0`; T2c, T5 and T6 can then exercise no dynamic residual. Three gap frames also need not show any selected donor in the chosen camera. The renderer skips `alpha < 1/255` ([`forward.cu:379–381`](D:/adags-cer/ex4dgs/submodules/diff_gaussian_rasterization_df/cuda_rasterizer/forward.cu:379)). Equal images can therefore demonstrate nothing about active copies.  
   Existing T5/T6 merely call `begun` and compare the resulting state ([`test_cer.py:233`](D:/adags-cer/ex4dgs/test_cer.py:233), [`:278`](D:/adags-cer/ex4dgs/test_cer.py:278)).  
   **Fix:** Require or construct residuals in both banks, positive measured contribution for each rendered comparison, and explicit pruning/survival of dynamic residual rows.

3. **Medium — “the window is at its cap, asserted in begin” is false.**  
   [`scene/cer.py:78–79`](D:/adags-cer/ex4dgs/scene/cer.py:78) checks sampling coverage and `max(timestamp) < sample_len - interval`; neither asserts the duration cap. `expand_duration` remains called after CER begins ([`train.py:264–269`](D:/adags-cer/ex4dgs/train.py:264)); its no-op condition is `int(duration)+1 <= self.duration` ([`c_gaussian_model.py:1275–1277`](D:/adags-cer/ex4dgs/scene/c_gaussian_model.py:1275)).  
   **Fix:** State the configuration-dependent argument accurately, or assert the actual no-op precondition and matching position/rotation K.

## Suspected issues

- **Exact-zero guarantees during training need a finite-opacity qualification.** The finite check runs in final `gates`, not `begin` or each training render ([`scene/cer.py:124–134`](D:/adags-cer/ex4dgs/scene/cer.py:124)). Training sanitizes `_opacity_duration_var.grad`, not evaluated opacity ([`train.py:248–250`](D:/adags-cer/ex4dgs/train.py:248)). A non-finite temporal factor defeats multiplication by zero. No actual failing run was available; dynamic-copy lifecycle tests would settle this.

## Unverified claims

- Job 59077300’s donor counts, masses, motion statistics and scene extent: underlying measurements/plan log were unavailable.
- Principal approval, Multi4D’s implementation, handoff and frozen-threshold provenance: referenced sources were unavailable at the supplied paths.
- Actual amended byte identity and GPU test results: the amendment is unimplemented, and the required external checkpoints/scenes were unavailable.

## Missing

- A2 should distinguish **gradient support** from nonzero gradient: the plateau returns `ones_like`, so temporal parameters receive no rendering gradient strictly inside it ([`interpolations.py:61`](D:/adags-cer/ex4dgs/utils/interpolations.py:61)).
- Dynamic copies change `motion_reg`’s mean and receive regularization outside the gap ([`train.py:161–163`](D:/adags-cer/ex4dgs/train.py:161)); update rev 2’s static-only q=0 training caveat.
- Specify donor indices against the **original** bank sizes before either append, and test static-only, dynamic-only and mixed births. Otherwise the static append can invalidate conversion of concatenated donor indices to dynamic-local indices.