## Verdict

MATERIAL DEFECT

The diff is not ready for the principal’s review. Reviewed independently; no ARIS skill was used.

## Confirmed defects

- **High — semantic loaders bypass the CER reload refusal.** [cluster_semantic_features.py:118](/D:/adags-cer/multi4d/cluster_semantic_features.py:118) sets `scene_load_path = None` for directory checkpoints, then directly calls `load_ply` at lines 144–146. The same bypass exists in [render_cluster_by_text.py:131](/D:/adags-cer/multi4d/render_cluster_by_text.py:131) and [render_semantic_features.py:53](/D:/adags-cer/multi4d/render_semantic_features.py:53). The new check only runs inside `if self.loaded_iter`. Therefore §2.1’s claim that these scripts “cannot render” CER models is false. **Fix:** enforce the sidecar refusal at every geometry-loading entry point and test these three paths.

- **High — the default test suite cannot run sequentially.** [begin:144](/D:/adags-cer/multi4d/scene/cer.py:144) sets `cer_gate.ARMED = True`. [test_cer.py:106](/D:/adags-cer/multi4d/test_cer.py:106) subsequently loads fresh models without resetting that flag. After T2, T2b’s initial `mass()` reaches the armed gate without labels and raises `"CER is armed but this model carries no CER state"`. **Fix:** isolate tests in separate processes or explicitly reset test-owned global state before loading each fixture; preserve the production fail-closed behavior.

- **High — T4 necessarily crashes.** [test_cer.py:327](/D:/adags-cer/multi4d/test_cer.py:327) assigns the generator returned by `train_views`; line 336 evaluates `len(views)`. A read-only reproduction returned `TypeError: object of type 'generator' has no len()`. **Fix:** materialize the views or count them during iteration.

- **Medium — the gate report omits a required measurement.** [scene/cer.py:186](/D:/adags-cer/multi4d/scene/cer.py:186) measures the frozen gap region, and line 202 reports only `transient_share_gap_region`. It never measures the transient share of the actual target mask inside the gap, required by §6. **Fix:** add that separate measurement, with explicit handling when target mass is zero.

- **Medium — the documented forecast name is silently ignored.** The spec calls it `T4-forecast`, but [test_cer.py:481](/D:/adags-cer/multi4d/test_cer.py:481) dispatches only `"FORECAST"`. Unknown requested tests are not rejected, so a request containing `T0,T1,T4-forecast` can succeed without running the forecast. **Fix:** accept the documented name and reject unknown selectors.

## Suspected issues

- **Native extension isolation remains unproven.** The two builds use different Python module names but identical native function names. Python’s loader flags govern symbol sharing; naming alone does not settle native isolation. Verify the actual deployment’s loader flags, exported symbols, and kernel behavior with both modules loaded. [Python documentation](https://docs.python.org/3.12/library/sys.html#sys.setdlopenflags)

- **Keeping the defective training backward may confound the comparison.** The production kernel still reads unwritten `max_contrib` at `backward.cu:1229`. CER adds allocations and changes row counts; unchanged kernel source does not establish equivalent gradient corruption across arms. T3’s gradients are explicitly tested on the repaired instrument build. Production-backward validation in both arms would settle this.

## Unverified claims

- Claimed GPU test results, census percentages, byte-identical historical renders, principal decisions, and the in-progress run: supporting artifacts were unavailable.
- The referenced Ex4DGS spec, lifecycle map, wrapper, launcher, scorer, and worker were absent from this checkout.
- Successful CUDA compilation, deployed native isolation, and macro-free binary equivalence were not established. I performed source inspection and read-only Python checks, not the GPU suite.

## Missing

- **T4b non-vacuity:** it accepts `"FRESH_OK"` even when the printed view count is zero ([test_cer.py:354](/D:/adags-cer/multi4d/test_cer.py:354)).
- **Promised test coverage:** T6 invokes only four sentinel methods; T3 samples camera paths rather than checking every camera; T7 counts metric lines without validating unique fine-iteration records or the saved image inventory.
- A reproducible CUDA validation record covering bucket boundaries, empty tiles, partial image tiles, and loading both extensions. A separate extension name gives a separate build location, but that is narrower than the isolation guarantee being requested. [PyTorch documentation](https://docs.pytorch.org/docs/main/cpp_extension.html#torch.utils.cpp_extension.load)