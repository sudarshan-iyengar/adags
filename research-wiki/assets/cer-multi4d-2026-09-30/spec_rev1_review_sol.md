## Verdict

MATERIAL DEFECT

## Confirmed defects

- **Major — the proposed persistence state is insufficient for T5.** Transient construction sets `active_sh_degree_t = 0` ([gaussian_model.py:1266](/D:/adags-cer/multi4d-read/scene/gaussian_model.py:1266)); training advances it through `oneupSHdegree`, but `load_ply` restores only the spatial degree at line 1825. Rendering consumes the temporal degree at [renderer:402](/D:/adags-cer/multi4d-read/gaussian_renderer/__init__.py:402). Thus fresh reloads disable learned temporal SH terms, independently of CER labels. **Fix:** persist and restore rendering metadata, including both active SH degrees and transient configuration. T5 must include nonzero temporal SH coefficients.

- **Major — missing-sidecar rejection contradicts the specified loader.** §5 restores CER state “when present”; T5 requires rejection when it is absent. The PLYs contain no CER marker, and the existing [scene load path:61](/D:/adags-cer/multi4d-read/scene/__init__.py:61) cannot distinguish a stripped CER save from a base save. **Fix:** retain a separate CER-required marker or require an explicit expected-CER load mode; missing state must then raise.

- **Major — T6 cannot establish its claimed invariant.** Equal endpoint row counts do not detect pruning followed by insertion, row permutation, opacity resets, or statistic changes. These are separate operations in [train.py:38–229](/D:/adags-cer/multi4d-read/train.py:38). **Fix:** in T7, snapshot labels and all persistent per-row statistics immediately after `begin`, compare them every subsequent iteration, and make densify/prune/reset/transfer methods raise if called. Allow parameters and optimizer moments to change.

- **Major — T3 does not require a foreground witness.** Its actual-scene deletion comparison can pass with no visible foreground memory rows; then bypassing `render_foreground` and `render_mask` changes nothing. Neither T2 nor T4 requires that witness. **Fix:** add a tiny synthetic fixture with visible labelled FG and BG rows. Capture each rasterizer’s opacity input and assert the exact factors, FG→BG concatenation order, and unchanged transient opacity. Test image removal equivalence and zero gradients on that fixture.

- **Minor — two audit statements overreach.** “Every … per-row statistic update lives inside” maintenance is false: displacement accumulators change at [train.py:524–536](/D:/adags-cer/multi4d-read/train.py:524). They stop through the phase-3 `continue`, so this does **not** refute freezing after 16,000. Likewise, post-training evaluation calls all four render functions in [eval_utils.py:235–253](/D:/adags-cer/multi4d-read/utils/eval_utils.py:235), contradicting “only `render_rawall3pc` runs … in evaluation.” **Fix:** document both exceptions and their actual guards.

## Suspected issues

- **“Every gap frame differs” is not guaranteed.** Changed rows may have no visible contribution in a particular held-out view, or differences may disappear in PNG quantization. Settle this with raw tensor comparisons and a demonstrated visible contribution witness.

- **T4’s aggregate conservation check may miss incorrect per-bank attribution.** Add a three-bank occlusion fixture and compare each bank’s masked and total DC gradient against its one-hot contribution in the **main composite image**. The auxiliary persistent render uses `T_3D`, whereas the main image uses shared `T` ([forward.cu:679–682](/D:/adags-cer/multi4d-read/diff_gaussian_rasterization_hybrid/cuda_rasterizer/forward.cu:679)); it is unsuitable as that oracle.

## Unverified claims

- The companion Ex4DGS spec, lifecycle map, census report/loader, wave-1 artifacts, wrapper, launcher and scoring/reducer files are absent from this checkout. Their numerical results, byte reproduction, provenance and gate propagation remain unverified.
- No GPU tests were run. Source inspection supports the main-image DC route: the launched backward computes `weight = alpha * T` and accumulates colour gradients with it ([backward.cu:1376–1395](/D:/adags-cer/multi4d-read/diff_gaussian_rasterization_hybrid/cuda_rasterizer/backward.cu:1376)); numerical tolerances remain untested.

## Missing

- Explicit background append rebinding: `cat_tensors_to_optimizer` returns replacement parameters; the caller must assign all six model attributes, as [densification_postfix:1157–1163](/D:/adags-cer/multi4d-read/scene/gaussian_model.py:1157) does. Test preserved original Adam moments and zero appended moments.
- A `mass` contract requiring `torch.enable_grad()` and restoration in `finally`. Evaluation uses `no_grad`, which disables graph construction even for gradient-requiring inputs. [PyTorch documentation](https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html)
- An enforceable exclusion for semantic scripts. `render_semantic_features.py` also reaches the ungated semantic renderer; “never run on CER models” is currently only a convention.
- Exact masks, cohort aggregation windows, strength policy and gate definitions from the unavailable companion spec.

Ready to implement after these specification fixes and recovery of the referenced contracts; acceptance still requires GPU validation. Reviewed independently, without an ARIS skill.