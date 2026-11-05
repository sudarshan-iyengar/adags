## Verdict

MATERIAL DEFECT. Code opened at `00357aa`; [cer.py:147](D:/adags-cer/multi4d/scene/cer.py:147) reads: `assert iteration == opt.densify_until_iter and iteration >= getattr(opt, "phase3_start_iter", 10000)`.

## Confirmed defects

1. **Major — the rationale for fine 3,000 misreads transient training.** [train.py:748](D:/adags-cer/multi4d/train.py:748) adds `loss += l1_loss(image_tensor_hybrid_full, gt_image_tensor)` throughout the pre-phase-3 regime; diversity loss also runs at lines 784–785. `after_merge` gates hybrid SSIM, not all transient losses. **Fix:** describe 3,000 as the lift/SSIM boundary. Withdraw “the occluder has no reason to form” and inspect transient contribution at begin.

2. **Major — recomputation and lift exclusion do not satisfy the stated memory-copy guarantee.** The [majority rule](D:/adags-cer/multi4d/scene/cer.py:151) can demote a row with positive target mass when its target fraction falls below 0.5. That row becomes ungated and eligible for lifting. Conversely, an ordinary row can be [copied into transient](D:/adags-cer/multi4d/scene/gaussian_model.py:439), then become memory at a later relabel; its existing copy remains ungated. The permitted begin range also includes starts after lifting has begun. **Fix:** use immutable inherited membership for the minimal timing experiment, or specify ancestry and treatment of previously transferred copies. Monotone membership alone does not fix ordinary→memory after transfer. Test both transition sequences.

3. **Major — E3 can pass despite broken optimizer preservation or extra RNG consumption.** The [stand-in loader](D:/adags-cer/multi4d/test_cer.py:57) creates fresh optimizer state; existing T2b explicitly [populates moments](D:/adags-cer/multi4d/test_cer.py:170). E3 omits that requirement and checks no RNG states. Extra draws after the last stochastic operation can leave parameters identical; NumPy opacity resets are omitted entirely. **Fix:** initialize nonzero moments, exercise resets, and compare complete Torch CPU/CUDA, NumPy and Python RNG states.

4. **Major — E7/E8 allow vacuous zeros.** Pruning can remove the tested cohorts; zero mass then proves nothing. Existing [T4:354–355](D:/adags-cer/multi4d/test_cer.py:354) additionally requires positive active-cohort mass. **Fix:** preserve visible descendant witnesses and require positive ungated contribution before checking gated zeros. E8 must enforce the accepted [§14.2 preconditions](D:/adags/research-wiki/operations/cer-multi4d-spec-2026-09-30.md:307), including memory activity before and after the gap. Do not require the superseded `cer_gates["pass"]`.

5. **Major — the specified launcher changes leave early gating disabled.** [Plan:175–176](D:/adags/research-wiki/operations/cer-multi4d-early-gate-plan-2026-10-06.md:175) permits only worktree, extension-directory, branch and pin changes, while `cer_begin_iter=-1` retains legacy timing. **Fix:** explicitly pass `--cer_begin_iter 3000`, record the chosen relabel policy, and require the scorer to validate these settings.

6. **Minor — gated rows do not have entirely zero densification statistics.** [CUDA preprocessing](D:/adags-cer/multi4d/diff_gaussian_rasterization_hybrid/cuda_rasterizer/forward.cu:510) assigns geometric radii despite zero opacity; visibility uses radii, and [statistics accumulation](D:/adags-cer/multi4d/scene/gaussian_model.py:114) increments `denom`. **Fix:** specify finite zero *new rendering-gradient contribution*. Document inactive-frame denominator counts; changing that policy would introduce another intervention.

7. **Minor — “bit-exact to the saved model” overstates the evidence.** The [record](D:/adags/research-wiki/operations/ars-fast-path-plan-2026-09-23.md:1552) establishes equal 8-bit render values on five frames of eighteen runs. The [script](D:/adags/research-wiki/assets/cer-wave1-montages-2026-10-06/cer_m4d_banks.py:143) quantizes before comparing. **Fix:** state saved-PNG equality on the checked frames, without implying floating-point model equivalence.

8. **Minor — source-path cache keying is inaccurately described.** PyTorch’s checked versioner hashes source **contents**, build arguments and build directory; extension name indexes its entries. **Fix:** retain separate extension directories, but explain isolation through separate build artifacts rather than a name-plus-source-path cache key. [Primary source](https://raw.githubusercontent.com/pytorch/pytorch/main/torch/utils/_cpp_extension_versioner.py).

## Suspected issues

- **D3 adds a scientific factor.** Propagation is required bookkeeping; repeated membership reassignment is optional. The proposed comparison identifies the package’s effect, not timing alone. An immutable-cohort early arm would settle whether reassignment is necessary.
- **Early residual birth may provide little usable capacity.** Donors are selected once from the current background ([cer.py:158](D:/adags-cer/multi4d/scene/cer.py:158)). A mature saved-model test does not establish donor availability at fine 3,000. Measure birth counts, contribution and subsequent survival on an actual early-stage model.

## Unverified claims

- Current Leonardo job status, remote worktree cleanliness, installed PyTorch version and existing launcher contents were not checked live; the referenced launcher directory is absent locally.
- Numerical outcomes were checked against the supplied records, not independently recomputed from original run artifacts.
- No CUDA tests were executed. Per-pass runtime and total cost remain estimates.

## Missing

- **Actual training-backward validation.** Existing [T3](D:/adags-cer/multi4d/test_cer.py:294) and E7 use the instrument kernel. The ordinary build’s backward [reads `max_contrib`](D:/adags-cer/multi4d/diff_gaussian_rasterization_hybrid/cuda_rasterizer/backward.cu:1229), whose forward write is instrument-only. Add active-gradient positive controls and finite lifecycle-statistics checks on the actual training extension.
- **Complete confirmatory criteria.** Carry forward the [binding rules](D:/adags/research-wiki/operations/cer-multi4d-spec-2026-09-30.md:348): return lower bound above −0.25 dB, positive gap gain in every layout, and whole-frame/ordinary-region do-no-harm.
- **Production-boundary coverage.** E8 uses `begin < merging`; the proposed arm uses equality. Test `begin == merging`, nonaligned relabel intervals, and final-freeze ordering.