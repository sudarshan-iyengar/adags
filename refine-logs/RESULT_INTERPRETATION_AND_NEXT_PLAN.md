# Result Interpretation And Next Plan

Date: 2026-07-08

## Scope

This document interprets R017-R027 for the event-crop-fix objective and predeclares the next compact milestone. It separates evidence from speculation and treats the frozen R009 windows as evaluation-only unless a run is explicitly labeled diagnostic/oracle-support.

## Bottom Line

The current evidence does not support the claim that the tested non-oracle Gaussian mechanisms recover the frozen event-crop failures. It does support a narrower claim: the oracle crop intervention exposes a large real local error, and the tested non-oracle mechanisms have not yet realized that upper bound.

Do not declare the broad scientific idea impossible. R027 was a valid non-oracle checkpoint-backed run, but the posthoc support audit shows its boundary support barely touched the frozen event crops. Therefore M2 mainly falsifies the specific R026 support generator plus 400-iteration micro-densification recipe, not every possible event-local densification/reveal mechanism.

Update after R031-R033: Depth Anything 3 setup and inference are now operational on Leonardo, and the DA3 support artifacts are valid non-oracle artifacts. However, the current DA3 support formulations do not provide strong frozen-window crop coverage. R031 sparse support barely improves over R026, R032 high-recall sparse support improves temporal hit rate but covers only `0.002846` mean crop area, and R033 tile-fill support does not rescue coverage. This is evidence against the current DA3 support-fusion recipe as a training-loop support candidate, not against all possible depth/geometry use.

## 1. What The Evidence Supports

- PASS: The frozen R009 crops are meaningful error regions. R013/R015 oracle GT crop compositing improves route0 from PSNR `30.5021` / L1 `0.0148316` to PSNR `41.7149` / L1 `0.00266536`.
- PASS: R017, R025, and R027 produced comparable rendered/evaluated artifacts for their stated checks. The integrity audit found no evidence of fake GT, self-normalized scores, or phantom result files, but overall status is WARN because LPIPS is proxy L1 and scope is only five windows.
- PASS: R025 is a valid negative test for M1 under the R020 support pool and 200-iteration local refinement recipe. It worsened PSNR/L1 on every frozen window.
- PASS: R027 is a valid negative test for the R026 boundary-support artifact plus 400-iteration M2 recipe. It fails the predeclared recovery gate: `2/5` strict all-baseline PSNR+L1 wins, mean PSNR `+0.0569 dB`, mean L1 `-0.0000903`, and oracle recovery below 1%.
- PARTIAL: R027 has a tiny directional signal over route0 (`3/5` route0 PSNR+L1 wins, `3/5` static no-worse), but this is far below the claim threshold and may be ordinary continuation noise.
- PASS diagnostic: R028 support-overlap audit shows R020 candidate boxes covered most of 4/5 windows to some degree, but missed one window entirely. R026 boundary masks had mean crop coverage `0.000000` and support-frame fraction `0.0250`, so M2 did not substantially operate inside the frozen crop regions.

## 2. What The Evidence Rules Out

- FAIL: Broad runtime opacity attenuation over event crops is not a recovery mechanism. R017 worsened all five windows and all mean metrics.
- FAIL: The tested M1 recipe, `event_candidate_refine` from route0 `6000 -> 6200`, is not a recovery mechanism. It passed `0/5` route0 PSNR+L1 windows and worsened mean PSNR by `-1.5629 dB`.
- FAIL: The tested M2 recipe, R026 boundary masks plus `event_boundary_micro_densify` from `6000 -> 6400`, is not a recovery mechanism. It recovered less than 1% of the oracle upper bound.
- FAIL: The current evidence rules out using R025 or R027 as a positive paper claim. Both result-to-claim reviews judged `claim_supported: no`.
- SKIP for now: paper-scale validation, broad baselines, or positive ablations around M1/M2. The main gate is not met.

## 3. What Remains Plausible But Untested

- INCONCLUSIVE: Whether a good non-oracle support detector plus local Gaussian capacity could recover the crops. R026 did not cover the crops; R020 covered more but M1 still failed.
- INCONCLUSIVE: Whether 400 posthoc iterations are enough. R027 needs a matched route0 continuation control before its tiny gain can be interpreted as mechanism signal.
- INCONCLUSIVE: Whether hyperparameters dominate. We have not run a controlled LR/loss/crop/densification sweep; however, R025 is sufficiently negative that a blind sweep is not justified before stronger diagnostics.
- INCONCLUSIVE: Whether training-from-scratch or integration into the original training loop is required. Current runs are posthoc checkpoint refinements.
- INCONCLUSIVE: Whether five frozen windows are representative. They are enough for this diagnostic gate, not for broad generalization claims.
- INCONCLUSIVE: Whether a depth-informed support signal can localize occlusion/reveal better than masks/flow/render boundaries. No depth sidecar method has been tested yet.
- PARTIAL/WEAK after R031-R033: DA3 can produce valid depth sidecars and non-oracle support manifests, but the tested fusion variants do not localize enough frozen crop area to justify a positive support claim.

## 4. Most Likely Failure Modes

1. Support localization is the largest current bottleneck for M2. R026 boundary support missed the frozen crops almost completely.
2. Posthoc local optimization may be structurally weak. R020 had much better crop overlap than R026, yet R025 still degraded all windows.
3. The mechanism may need reveal/identity synthesis, not just opacity changes, ROI loss, or densification. Oracle crops add missing/revealed appearance; local loss reweighting may not create the right surface.
4. The tiny R027 improvement may be continuation noise. It must be compared to a matched route0 `6400` continuation.
5. Hyperparameters may matter, but a broad sweep before the continuation/support/oracle-support diagnostics would be uncontrolled.
6. The frozen set may emphasize hard occlusion/reveal cases. That is useful for a failure diagnostic but insufficient for broad method-level claims.

## 5. Next Compact Milestone

### R028 - Support-Overlap Diagnostic

Status: PASS diagnostic.

Question resolved: did existing support artifacts actually cover the frozen failures?

Results:
- R020 candidates: mean support-frame fraction `0.6375`, mean crop coverage `0.491371`, but `cut_roasted_beef_hand_tongs_meat_095_110` had zero support.
- R026 boundary support: mean support-frame fraction `0.0250`, mean crop coverage `0.000000`.

Interpretation: R028 showed M2 failure was partly confounded by support selection, but R030 now shows oracle crop support alone does not rescue the current posthoc micro-densification mechanism. M1 failure remains concerning for local refinement because R020 covered 4/5 windows substantially.

### R029 - Matched Route0 6400 Continuation Control

Status: COMPLETE.

Question resolved: is R027's tiny gain caused by event-boundary micro-densification or by 400 extra training iterations?

Design:
- Resume the same route0 `chkpnt6000.pth` checkpoints to `chkpnt6400.pth`.
- Use `configs/n3v/route0_continue_6400_control.yaml`.
- No event support manifest and no motion-aware densification.
- Eval and score on the frozen R009 windows as system `route0_continue_6400`.

Decision:
- Outcome: route0 continuation was worse than route0 and worse than R027. Mean PSNR `30.353246` versus route0 `30.502119`; mean L1 `0.015060306` versus route0 `0.014831561`; route0 PSNR+L1 wins `1/5`; static no-worse `1/5`.
- Interpretation: R027's tiny positive movement is not explained by generic 400-iteration continuation, but R027 still remains far below the predeclared recovery thresholds.

### R030 - Oracle-Support Gaussian-Only Diagnostic

Status: FAIL.

Question resolved: if support localization were perfect, could the current posthoc Gaussian refinement/densification machinery recover a meaningful fraction of the oracle crop upper bound without GT compositing?

Design:
- Explicitly diagnostic/oracle-support; not a non-oracle method.
- Use frozen R009 crop windows as event support via `configs/n3v/oracle_crop_support_micro_densify_6400.yaml`.
- Resume route0 `chkpnt6000.pth` to `chkpnt6400.pth`.
- Do not paste GT crop pixels; output must be checkpoint-backed Gaussian renders.
- Score as system `oracle_crop_support_micro_densify`.

Decision:
- Outcome: R030 failed despite oracle crop support. Mean PSNR `29.902108` versus route0 `30.502119`; mean L1 `0.015877018` versus route0 `0.014831561`; route0 PSNR+L1 wins `0/5`; strict all-baseline wins `0/5`; static no-worse `4/5`; oracle recovery negative on both PSNR and L1.
- Interpretation: support alignment alone is not enough for the current posthoc Gaussian micro-densification recipe. The bottleneck is likely the posthoc optimization/capacity mechanism, not only support discovery.

Execution update 2026-07-08T15:45+02:00:
- R029 train jobs submitted and running: `48935431`, `48935450`, `48935478`; manifest `refine-logs/route0_continue_6400_train_jobs_20260708_152429.tsv`.
- R030 train jobs submitted and running: `48935580`, `48935581`, `48935583`; manifest `refine-logs/oracle_crop_support_micro_densify_train_jobs_20260708_153750_manual.tsv`.
- Duplicate oracle cut attempts from SSH timeouts are excluded from the valid manifest. This is still a training wave only; no scientific verdict should be inferred until eval/scoring completes.

Execution update 2026-07-08T16:10+02:00:
- R029 and R030 train jobs all completed with `ExitCode=0:0` and wrote the expected `chkpnt6400.pth` checkpoints.
- Eval submission is BLOCKED by Slurm account/partition acceptance. Evidence: `boost_usr_prod/euhpc_d21_034/boost_qos_lprod` returned invalid account/partition for eval submission, an environment-loaded retry returned the same, `sbatch --test-only` probes rejected the available QoS/partitions or reported expired budget on non-boost partitions, and a `QOS=boost_usr_prod` retry hung before producing job IDs.
- `saldo -b` shows the project allocation itself is active through 2026-07-30 with 44.1% consumed, so this appears to be scheduler/account routing rather than experiment code failure.
- No R029/R030 metric verdict exists until eval renders and frozen-window scoring complete.

Execution update 2026-07-09T00:35+02:00:
- Eval retry succeeded. R029 eval jobs `48969017`, `48969019`, `48969021` and R030 eval jobs `48969090`, `48969092`, `48969093` completed with `ExitCode=0:0`.
- Frozen-window scoring job `48969825` wrote `refine-logs/hide_reveal_poc/r029_r030_disambiguation_real_eval/` and summary artifacts under `refine-logs/hide_reveal_poc/r029_r030_disambiguation_summary/`.
- R029 result: control complete, worse than route0 on mean PSNR/L1.
- R030 result: FAIL, worse than route0 on mean PSNR/L1 despite oracle crop support.
- Scientific consequence: further support-only variants of this posthoc micro-densification recipe are low priority. Meaningful next tests need a changed optimization mechanism, training-loop integration, or a tightly scoped hyperparameter/iteration sensitivity diagnosis.

### Depth-Informed Support Proposal

Motivation: monocular depth may expose occlusion boundaries and reveal candidates that mask/flow/render-boundary support missed.

Candidate depth sources:
- Preferred first: existing depth sidecars if present on HPC. None are known locally yet.
- If generating new sidecars: Depth Anything V2 small/base for fast robust relative depth; Depth Pro for sharper metric depth boundaries; VGGT only if multi-frame geometry/camera/depth/track outputs become worth the extra integration cost.

Concrete method:
1. Generate or load per-frame depth maps aligned to route0 frame resolution.
2. Compute depth discontinuity maps from local depth gradients.
3. Intersect depth discontinuities with dynamic-mask boundaries, flow invalid/large-motion boundaries, and route0 flicker/static-delta cues.
4. Add a temporal reveal cue: regions behind a foreground boundary whose relative depth or local texture becomes visible after occluder motion.
5. Produce a non-oracle support manifest with the same guardrails as R026: no GT residuals, no frozen crop labels, no GT crop pixels.
6. Use it either as event-boundary support for micro-densification or as candidate windows for local refinement.

Expected failure modes:
- Monocular depth may be unstable on flames, specular food, utensils, motion blur, and close hands.
- Relative depth scale may drift frame-to-frame, producing false discontinuities.
- Depth boundaries may mark object silhouettes but not newly revealed texture.
- If intersected too strictly with masks/flow, it may repeat R026's support miss.
- If too broad, it may repeat R017/R025 damage.

Test plan:
- First run only a depth-support diagnostic on the frozen source scenes, then posthoc overlap audit.
- Proceed to Gaussian training only if the depth support covers more frozen crop/frame mass than R026 while staying within support pixel caps and without obvious full-frame leakage.

Execution update 2026-07-09:
- PASS setup: cloned `ByteDance-Seed/depth-anything-3` to `$WORK/proj_adags/repo/depth-anything-3` at commit `41736238f5bced4debf3f2a12375d2466874866d`; cached `depth-anything/DA3NESTED-GIANT-LARGE-1.1` locally under `$WORK/proj_adags/models/depth-anything/DA3NESTED-GIANT-LARGE-1.1`.
- PASS environment after repair: DA3 venv imports DA3 with ADAGS `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`, local `numpy==1.26.4`, and `xformers==0.0.28.post3`. Initial smoke failed on undeclared `addict`; repair logs remain on HPC under `logs/da3_env_repair_*`.
- PASS R031 frame/depth wave: Slurm prepare job `49030039` wrote 900 `cam00` frames with `uses_gt_residual=false`, `uses_gt_crop_pixels=false`, `uses_frozen_window_labels=false`; inference job `49030185` wrote 900 DA3 depth sidecars in `00:05:09`.
- PASS artifact validity, WEAK support: R031 support job `49030546` produced 83 support frames, validation OK, mean support-frame fraction `0.0625`, mean crop coverage `0.000030`.
- PASS sensitivity artifact, WEAK support: R032 high-recall sparse job `49031389` produced 355 support frames, validation OK, mean support-frame fraction `0.4125`, mean crop coverage `0.002846`.
- PASS tile-fill artifact, WEAK support: R033 high-recall tile-fill job `49032424` produced 408 support frames, validation OK, mean support-frame fraction `0.3750`, mean crop coverage `0.001253`.
- Comparison: R026 boundary support had mean crop coverage `0.000000`; R020 high-recall candidate boxes had mean crop coverage `0.491371`. The DA3 variants improve over R026 only marginally and remain far below the earlier high-recall non-oracle box pool.
- Conclusion: do not promote current DA3 support into a rendered training-loop experiment as the main next milestone. Keep DA3 sidecars as reusable evidence, but the next scientifically meaningful rendered method remains training-loop integration with a stronger support/capacity mechanism.

## 6. Why Not Immediately Run A Large Sweep

The evidence now says support discovery alone is not the only axis responsible. A broad sweep over LR, loss weights, crop size, threshold, iterations, seeds, and training-from-scratch would still be hard to interpret. The compact wave resolved the highest-value uncertainties first:

- R028: support localization.
- R029: continuation/iteration confound.
- R030: support discovery versus local Gaussian optimization.

After R030, the strongest next direction is not another support detector by itself. Depth support remains plausible as an input cue, but it should be paired with a changed training/optimization mechanism or tested first only as an overlap diagnostic.
