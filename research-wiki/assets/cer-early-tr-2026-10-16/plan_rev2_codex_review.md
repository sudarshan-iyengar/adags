## Verdict

MATERIAL DEFECT

## Confirmed defects

1. **Major — F5 still drops an E8 opacity control.** F5 requires every witness’s **probe** opacity gradient to be nonzero. E8 instead accepts L1 **or** probe: `op_fail = ~o_nz & ~po_nz` ([test_cer_early.py:382](D:/adags-cer/multi4d-early/test_cer_early.py:382)). Positive mass establishes colour sensitivity; opacity sensitivity depends on compositing contrast and can cancel under `render.sum()` ([backward.cu:1395](D:/adags-cer/multi4d-early/diff_gaussian_rasterization_hybrid/cuda_rasterizer/backward.cu:1395), lines 1396–1397 and 1426). Correct code can fail this specification. **Fix:** retain both loss controls or construct and validate a fixture with guaranteed opacity sensitivity.

2. **Major — Task B cannot produce its requested shadow count at 16,000.** The probe records `rows_at_freeze` only inside its `cer.freeze` wrapper ([densify_probe.py:262](D:/adags-cer/agent/densify_probe.py:262)). Shadow restores `scene.cer_scene = ""` at line 244, while training calls freeze only under `if stage == "fine" and scene.cer_scene` ([train.py:1005](D:/adags-cer/multi4d-early/train.py:1005)). Moreover, the reader compares `rows_last`, a **pre-densification** count, rather than freeze snapshots ([densify_read.py:29](D:/adags-cer/agent/densify_read.py:29), line 52). **Fix:** independently capture both banks at fine 16,000 in both modes, consume those snapshots, and test their presence.

3. **Moderate — a NaN restart discards the probe.** The wrapper sets `sys.argv = ["train.py"] + train_args` ([densify_probe.py:299](D:/adags-cer/agent/densify_probe.py:299)); training restarts through that argv ([train.py:848](D:/adags-cer/multi4d-early/train.py:848)). Because [`execv` replaces the process](https://docs.python.org/3/library/os.html#os.execv), the replacement runs bare training without instrumentation. An eventual exit 0 could accompany partial diagnostics. **Fix:** fail diagnostic runs on restart, and require terminal completion plus both 16,000 snapshots before accepting outputs.

## Suspected issues

- **F5’s additional parameter-group positives remain checkpoint-dependent.** Requiring one positive witness per group does not supply round 1’s requested nondegenerate fixture. **Settle:** specify a controlled 4D fixture and establish positive controls for every group. Failure of the actual bottle checkpoint is unverified.

## Unverified claims

- Historical metrics match the cited research log, but I did not independently recompute them from primary diagnostic outputs. GPU test outcomes and proposed runtime estimates remain unverified.
- The probe changed during inspection. Findings 2–3 cite the subsequently stable version with SHA256 `8D5F95CBBB7DADB758073AA4F2AF648BAC1B362AC680240C230D6BC0A75DB47C`. The base checkout remained clean at `7b30ef8`.

## Missing

- **The comparison endpoint script needs the new arm.** `target_edge_interior.py` hard-codes only base, 16k and early ([lines 19–20](D:/adags-cer/agent/target_edge_interior.py:19)); its CLI accepts only `--out`. Section 8’s named script therefore cannot score the proposed arm as written. Specify an exploratory copy or parameterized run manifest and verify that all nine new runs enter its output.