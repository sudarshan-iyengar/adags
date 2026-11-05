## Verdict

MATERIAL DEFECT

PLAN: READY  
CODE: READY WITH FIXES  
PREMISE CHECK: QUALIFIED

Independent judgment; no ARIS skill used. Opened [cer.py:18](/D:/adags-cer/multi4d-early-tr/scene/cer.py:18): `import torch`. The round-4 probe fixes are present, including rejection of changed worktree states in an in-memory check.

## Confirmed defects

1. **Moderate — code: F9 can accept a failed training job.** The launcher records `exit_code=$rc` before changing `rc` when trees changed ([launcher:176](/D:/adags-cer/agent/train_multi4d_cer_early_tr.sbatch:176), line 187). F9 checks that recorded code but never checks `trees_changed` or the before/after states ([test:1755](/D:/adags-cer/multi4d-early-tr/test_cer_early_tr.py:1755)). **Fix:** finalize checks before recording the exit status; require unchanged tree states in F9.

2. **Moderate — code interpretation: “They can only add zeros” is false.** The shared clamp flags multiply colour derivatives by either zero or one ([backward.cu:160](/D:/adags-cer/multi4d-early-tr/diff_gaussian_rasterization_hybrid/cuda_rasterizer/backward.cu:160)). Overwriting a true clamp flag with false can enable an otherwise suppressed derivative; the test explicitly recognizes that anomaly ([test:1002](/D:/adags-cer/multi4d-early-tr/test_cer_early_tr.py:1002)). **Fix:** remove this argument. Forward exclusion has direct support from the alpha skip; failed positive controls limit the row-by-row gradient evidence.

3. **High — premise: candidate counts do not bound the intervention to “a few dozen rows.”** The counter correctly sums **labelled candidate occurrences above threshold**, before subsampling ([early cer.py:197](/D:/adags-cer/multi4d-early/scene/cer.py:197), line 209). It does not count unique donors, realized copies or descendants. Restoring candidates also changes the pool supplied to `randperm`, potentially changing unlabelled selections and subsequent training ([gaussian_model.py:430](/D:/adags-cer/multi4d-early-tr/scene/gaussian_model.py:430)). **Fix:** describe low measured exposure on the early trajectory, not a counterfactual effect bound. Track realized copies, descendants and contribution mass if that bound matters.

4. **High — premise: temporal overlap does not establish “occluder’s tails,” causation or impossibility of recovery.** “Also paints the gap” means merely `gap_mass > 0` ([diagnostic:73](/D:/adags-cer/agent/transient_time_interior.py:73)); the reported percentage weights those rows’ **pre-gap** mass ([line 84](/D:/adags-cer/agent/transient_time_interior.py:84)). It measures neither meaningful gap opacity nor occluder identity. These are composited contribution masses, whereas the earlier alpha diagnostic used the bank alone ([cer.py:93](/D:/adags-cer/multi4d-early-tr/scene/cer.py:93)). **Fix:** call this evidence consistent with temporal tails. A contribution/colour ablation is needed for the causal claim. Deferring 26 GPU-h is defensible prioritization; “the premise is false” and “cannot restore detail without the occluder” are not established.

5. **High — task B: the blanket conclusion contradicts the foreground result and excludes an unmeasured mechanism.** The supplied reader output gives foreground memory split selections **0 versus 54**, with all early foreground memory rows gone. Its early mean gradient is **1.077×10⁻⁷**, not exactly zero. Background totals alone cannot establish the blanket claim, and cohort size does not bound image impact. Moreover, the deletion sensitivity removes fully zero increments ([probe:103](/D:/adags-cer/agent/densify_probe.py:103)); it does not recover missing gradient contributions in mixed batches, whose gradients are summed before taking the norm ([train.py:855](/D:/adags-cer/multi4d-early-tr/train.py:855)). **Fix:** report similar background selections in this pair, suppressed foreground densification, and unresolved effects of fewer gradient views.

## Suspected issues

- **The remaining failures have not been isolated to the substrate.** I found no direct label-path suppression at `q=0`: the factor is one, donor labels use the final geometry mask, and split inheritance has matching repeat order. Settle attribution with paired gate/bypass backward checks and initialization diagnostics; NVIDIA documents [Initcheck](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) for uninitialized device-memory reads.

## Unverified claims

- GPU pass results, ten whole-row failures, historical PSNRs, per-run exclusion totals and costs were not independently reproduced. I did not reach the Leonardo checkpoints, raw logs and provenance; local Python has no PyTorch.

## Missing

- The time diagnostic lacks its own saved-render fidelity check. Importing the loader does not execute the bank tool’s comparison against saved renders ([banks:145](/D:/adags-cer/agent/cer_m4d_banks.py:145)).
- Explicit limits on the premise reading: two frames, cam00, and only one early replicate per scene. The time mapping, erosion convention and frame-59 gap mask agree with the inspected dataset and diagnostic code.