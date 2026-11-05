## Verdict

SOUND WITH QUALIFICATIONS

Code opened at `00357aa`. [train.py:918](D:/adags-cer/multi4d/train.py:918): `and iteration < fg_to_tr_end_iter)`.

Ready for implementation with the clarifications below. No confirmed new material defect; runtime correctness remains untested.

`P` refers to the [on-disk rev-3 plan](D:/adags/research-wiki/operations/cer-multi4d-early-gate-plan-2026-10-06.md). Other prefixes follow the request.

| Rev-2 item | Disposition | Evidence |
|---|---|---|
| 1: starts after earlier transfers | Fixed as specified | P:82–84 restricts begin to merging or earlier; T:917 requires `iteration > fg_to_tr_start_iter`, defined at T:360. |
| 2: coarse-stage transfers | Fixed as specified | P:85–87 restricts coarse duration to merging; the stage-free transfer predicate is T:916–918. |
| 3: ineffective feature doctoring | Fixed as specified | P:258 moves `_xyz` and requires demonstrated threshold crossing; C:105–106 replaces features, while R:479 reads background positions. |
| 4: vacuous training-build zeros | Fixed as specified | P:261 requires nonzero gradients on the same gap view at q=0 before q=1 zeros; G:32 supplies factors 1 and 0 respectively. |
| Suspected: negative threshold | Fixed | P:156 specifies `-inf`; GM:423 uses strict `dynamic_score > dynamic_score_threshold`. |
| Unverified: `iterations` usage | Corrected | P:98–99 includes the loop bound; T:1050 passes `optimization_params.iterations`. F1’s 3,500 evaluation is consistent with T:991 and the supplied wrapper excerpt. |
| Missing: argument plumbing | Fixed as specified | P:76–80 and E9 specify scene storage and verification; existing storage is at `scene/__init__.py:36`. |
| Missing: statistics exercise | Fixed as specified | P:261 calls accumulation and checks denominator increments; GM:115–117 updates accumulation and `denom`. |

## Confirmed defects

1. **Minor — “optimizer state only” is false.** P:126 describes three helpers this way. They also replace parameter tensors: [GM:140](D:/adags-cer/multi4d/scene/gaussian_model.py:140), GM:644 and GM:691 assign new `nn.Parameter` objects. **Fix:** describe parameter replacement and optimizer-state maintenance. Leaving these helpers unchanged remains consistent with the propagation design.

2. **Minor — the stated 3,000 boundary is imprecise.** P:91 says SSIM and lifting start there. T:393 defines `after_merge = iteration > optimization_params.transient_merging_iter`; transfers additionally require their cadence at T:916. **Fix:** say SSIM starts at 3,001 and lifting at the next eligible transfer iteration. Begin at the end of 3,000 still precedes both.

## Suspected issues

3. **Potential memory-copy exception before training.** [scene/__init__.py:97](D:/adags-cer/multi4d/scene/__init__.py:97) copies the first foreground point’s position into the transient seed; lines 99–103 also copy its colour. This precedes the organic-transfer window, and transient rows remain ungated. D4 therefore proves absence of earlier *organic lifts*, not every foreground-derived transient row. **Settle:** establish whether this seed or its descendants correspond to a later memory ancestor and contribute in the gap, or explicitly qualify P:271’s universal copy guarantee.

4. **Potential E8 instability from the existing substrate.** [forward.cu:735](D:/adags-cer/multi4d/diff_gaussian_rasterization_hybrid/cuda_rasterizer/forward.cu:735) guards the `max_contrib` write behind the instrument macro; [backward.cu:1229](D:/adags-cer/multi4d/diff_gaussian_rasterization_hybrid/cuda_rasterizer/backward.cu:1229) reads it in the training build. E8’s positive control prevents the previous vacuous success, but does not establish stable backward behaviour. **Settle:** run E8 in fresh processes; failures require a substrate decision before training.

## Unverified claims

- Numerical results, exact PNG equality, runtime estimates, Leonardo job status and remote cleanliness were not independently reproduced. Local narrative records repeat the results; primary outputs were unavailable locally.
- The complete remote final-render wrapper and installed PyTorch version remain unchecked. The general cache description matches the official [versioner](https://github.com/pytorch/pytorch/blob/v2.2.0/torch/utils/_cpp_extension_versioner.py) and [loader](https://raw.githubusercontent.com/pytorch/pytorch/v2.1.2/torch/utils/cpp_extension.py).
- No GPU tests ran.

## Missing

- Explicitly identify E8’s gap-frame instrument witnesses at **q=0**; at armed q=1 their contribution is zero by construction.
- Specify where validation receives optimization parameters: the scene constructor currently receives model parameters only (`scene/__init__.py:28–29`). Validate immediately before scene construction or explicitly extend its inputs.