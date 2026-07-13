# ADAGS Follow-Up Notes

Updated: 2026-05-21

This is a handoff for continuing the scaffold-motion-priors work without having to rediscover the repo state and the recent run evidence.

## Current Research Problem

The project is dynamic Gaussian rendering for N3V cooking scenes. The important visible failure is still fast-motion blur or smear in food, hands, heads, and utensils. Static/dynamic leakage also appears in diagnostic renders: the static branch can contain dark silhouette-like foreground remnants even when the final composite hides some of it.

The goal is not just a small PSNR bump. The useful next step should improve dynamic-region sharpness and reduce static-branch ghosting.

## Current Branch And Run Scope

The active direction is `scaffold-motion-priors`, based on reversible routing plus LoRA motion.

The current `runit.sh` schedules only two configs across the six N3V cooking scenes:

1. `configs/n3v/scaffold_lora_route0.yaml`
2. `configs/n3v/scaffold_lora_route0_dyn_densify.yaml`

The older `configs/n3v/lora_route0_dynmask.yaml` still exists, but it is not currently scheduled by `runit.sh`. As of this note, that older config still uses the longer 15k/blur-style settings, so update it before using it in a fair comparison.

## Current Config State

For both scaffold configs:

- `batch_size: 4`
- `iterations: 9_000`
- `densify_until_iter: 6_000`
- `lambda_track_flow: 0.0`
- `enable_rendered_flow: false`
- `blur_until_iter: 0`
- `histogram_log_interval: 100`

The base scaffold config has:

- `motion_scaffold_enable: true`
- `enable_motion_aware_densify: false`

The dynamic-densify config has:

- `motion_scaffold_enable: true`
- `enable_motion_aware_densify: true`
- `densify_until_num_points: 800000`
- `motion_aware_densify_boost: 1.5`

`runit.sh` now uses `CKPT_ITER=9000` for eval and a 15 hour training walltime.

## Important Code-Level Cost Model

Do not compare `iterations` naively to papers or other 3DGS repos. In this code, `batch_size: 4` means each optimizer iteration performs four full render/loss/backward passes. So:

- `9_000` iterations is about `36_000` render/backward passes.
- `6_000` iterations is about `24_000` render/backward passes.

Soft routing is also expensive: the renderer sends dynamic and static hypotheses to the rasterizer, so the effective point workload is larger than the raw Gaussian count suggests. Since current runs keep `static=0`, there is no hard static point-count saving.

Heavy histogram logging was previously a hidden cost. It is now throttled with `histogram_log_interval: 100`, while scalar logging remains every iteration.

## Recent Coffee Martini Evidence

Two Coffee Martini runs are especially informative:

- Base scaffold:
  - `runs/scaffold_lora_route0/20260512_160147_coffee_martini_scaffold_lora_route0`
- Dynamic-densify scaffold:
  - `runs/scaffold_lora_route0_dyn_densify/20260512_160148_coffee_martini_scaffold_lora_route0_dyn_densify`

### Base Scaffold Run

The base scaffold run completed:

- Final iteration: `9000/9000`
- Runtime: about `7:49:57`
- Initial points: `294,950`
- Final points: `584,228`
- Late speed: about `3.5 s/it`
- Final progress-bar train PSNR: about `36.86`
- Final progress-bar train loss: about `0.0832`
- W&B summary `best_test_psnr`: `28.57602`

Important checkpoint observation:

- Best checkpoint was saved at `3000`.
- Best checkpoint was saved again at `6000`.
- No best checkpoint was saved at `9000`.

Interpretation: for this scene, `9000` iterations likely are not needed for held-out quality. Training loss and train PSNR continue improving after `6000`, but test PSNR did not beat the `6000` checkpoint. Treat `6000` as the more promising default unless later scenes contradict this.

Point growth also stabilizes at `6000`, because densification ends there. The final 3000 iterations are a polish phase over a fixed point set, and for Coffee Martini that polish did not improve best test PSNR.

### Scaffold Regularizers

`Lscaffold_smooth` and `Lscaffold_reg` now log in scientific notation. They did increase, but only at tiny weighted magnitudes.

For the base Coffee Martini run:

- Early `Lscaffold_smooth`: around `1e-9` to `4e-8`
- Late `Lscaffold_smooth`: around `1.3e-7`
- Early `Lscaffold_reg`: around `3e-11` to `1e-9`
- Late `Lscaffold_reg`: around `9e-9`

This is not an explosion. The losses are weighted by:

- `lambda_scaffold_smooth: 0.0001`
- `lambda_scaffold_reg: 0.000001`

The scaffold coefficients initialize at zero and the basis is initialized at scale `0.01`, so these values are expected to start tiny and rise as the scaffold begins carrying motion.

## Dynamic-Densify Run Evidence

The dynamic-densify Coffee Martini run did not finish within the 15 hour walltime:

- Last logged iteration: `8120/9000`
- Runtime at last log: about `15:06:21`
- Initial points: `294,950`
- Points by `3000`: `1,021,368`
- Points by `6000`: `1,591,135`
- Points at last log: `1,591,135`
- Late speed: about `9.23 s/it`

At `8120`, about 880 iterations remained. At `9.23 s/it`, it needed roughly another 2 hours and 15 minutes, so the expected full runtime was around 17 hours and 20 minutes.

The cause is point explosion. Dynamic-aware densification with boost `2.0` allocates far more points than the base config:

| Iteration | Base Points | Dyn-Densify Points |
| ---: | ---: | ---: |
| 1000 | 359,380 | 474,517 |
| 3000 | 472,048 | 1,021,368 |
| 5000 | 556,372 | 1,427,842 |
| 6000 | 584,228 | 1,591,135 |
| 8000 | 584,228 | 1,591,135 |

The dynamic-densify run fits the training views better, but at a very high cost:

- At `8000`, dyn-densify train loss was about `0.0724`.
- At `8000`, base scaffold train loss was about `0.0859`.

This does not prove dyn-densify is better. It may simply be spending 2.7x more points to fit residual masks/noise. It did not finish, and the full eval summary is absent. The presence of `chkpnt_best.pth` and `chkpnt6000.pth` with matching timestamps suggests the best available dyn-densify checkpoint was likely at `6000`, but confirm with eval before making a final quality claim.

Current recommendation: do not run the old uncapped dyn-densify config as a main experiment. The config has now been constrained with an `800000` point cap and `motion_aware_densify_boost: 1.5`, but it should still be treated as an ablation until the capped run proves useful.

If dyn-densify is revisited further, keep it constrained:

- Keep `densify_until_num_points` around `700000` or `800000`.
- Keep `motion_aware_densify_boost` around `1.2` or `1.5`.
- Consider ending densification earlier, for example `densify_until_iter: 4000`.
- Compare at `6000`, not only at `9000`.

## Current Interpretation

The base scaffold configuration is the current practical candidate. For Coffee Martini, the most sensible operating point is probably `6000` iterations, not `9000`.

The dynamic-densify idea still has scientific motivation, but the current implementation is too aggressive. It creates many more Gaussians before proving better held-out quality. Because the residual dynamic mask is computed online from the model error, it can feed back into densification and allocate points to hard residuals, mask noise, or view-specific artifacts. With soft routing, every extra point is especially expensive.

Do not conclude that motion-aware densification is conceptually bad. Conclude that the old uncapped `boost=2.0` version was not a good default.

## Motion Prior Status

Track-flow supervision is currently disabled:

- `lambda_track_flow: 0.0`
- `enable_rendered_flow: false`

Reason: previous logs showed `Ltrack_flow=0.0000`, while local data had flow files under `<scene>/flow`, not under the default `MotionPriorCache` root `<scene>/motion_priors`. The renderer was also computing rendered flow even when no flow target was found, which was wasted work.

If track/flow priors are reintroduced, first fix the cache path or move/symlink files into:

```text
<scene>/motion_priors/flow
<scene>/motion_priors/flows
<scene>/motion_priors/track_flows
```

Then verify that `Ltrack_flow` is nonzero before launching long jobs.

## Current Motion Prior Cache Layout

`utils/motion_prior_utils.py` contains `MotionPriorCache`.

Default cache location:

```text
<scene>/motion_priors
```

Dynamic masks can be placed under:

```text
motion_priors/masks
motion_priors/dynamic_masks
motion_priors/foreground_masks
```

Accepted mask formats:

```text
png, jpg, jpeg, pt, pth, npy, npz
```

Dense track-flow caches can be placed under:

```text
motion_priors/track_flows
motion_priors/flows
motion_priors/flow
```

Expected dense flow shape:

```text
[H, W, 2] or [2, H, W]
```

Optional track-flow masks can be placed under:

```text
motion_priors/track_flow_masks
motion_priors/flow_masks
motion_priors/track_masks
```

If no masks exist and `dynamic_mask_from_residual: true`, training computes residual masks online. Evaluation metrics should not use residual fallback, because that would bias the metric.

## Relevant Prior Results

Representative best numbers supplied earlier by the user:

| Method | PSNR | SSIM | LPIPS | GS Count |
| --- | ---: | ---: | ---: | ---: |
| x1 Baseline | 31.57 | 0.9495 | 0.0582 | 363574 |
| gaussian_blur_15k | 31.26 | 0.9419 | 0.0607 | 339134 |
| reversible_routing_no_blur_15k | 32.51 | 0.9549 | 0.0495 | 462123 |
| lora_r8_a32_route0 | 32.73 | 0.9542 | 0.0497 | 403995 |
| part_basis_k8_r8_a32 | 32.39 | 0.9507 | 0.0536 | 413939 |

Key LoRA variants:

| Method | PSNR | SSIM | LPIPS | GS Count |
| --- | ---: | ---: | ---: | ---: |
| lora_r8_a32_route0 | 32.73 | 0.9542 | 0.0497 | 403995 |
| lora_r8_a32_coeff2x_basis025x | 32.62 | 0.9543 | 0.0498 | 433944 |
| lora_r8_a16 | 32.55 | 0.9538 | 0.0501 | 407396 |
| lora_r8_a32_route2 | 32.43 | 0.9539 | 0.0506 | 402368 |
| lora_r16_a32 | 32.36 | 0.9540 | 0.0501 | 387049 |

Part-basis variants did not clearly beat LoRA:

| Method | PSNR | SSIM | LPIPS | GS Count |
| --- | ---: | ---: | ---: | ---: |
| part_basis_k8_r8_a32 | 32.39 | 0.9507 | 0.0536 | 413939 |
| part_basis_k16_r8_a32 | 32.04 | 0.9504 | 0.0550 | 392412 |
| part_basis_k8_r16_a32 | 31.53 | 0.9446 | 0.0570 | 388014 |

Interpretation:

- `lora_r8_a32_route0` remains the strongest stable non-scaffold base.
- The scaffold work should be judged by dynamic-region quality and ghost reduction, not only global PSNR.
- Part-basis is not fully ruled out, but the tested variants did not beat LoRA.

## Literature Motivation

These papers motivated the scaffold-prior direction:

- MoSca, CVPR 2025: dynamic Gaussian fusion with 4D motion scaffolds and foundation priors.
  - https://openaccess.thecvf.com/content/CVPR2025/html/Lei_MoSca_Dynamic_Gaussian_Fusion_from_Casual_Videos_via_4D_Motion_CVPR_2025_paper.html
- Shape of Motion, ICCV 2025 / arXiv 2407.13764: motion bases with depth and long-range 2D tracks.
  - https://arxiv.org/abs/2407.13764
- Prior-Enhanced GS, arXiv 2512.11356: segmentation, epipolar masks, object-depth, tracks, virtual-view depth, and scaffold projection.
  - https://arxiv.org/abs/2512.11356
- MAPo, arXiv 2508.19786: motion-aware partitioning; unified motion models can blur highly dynamic regions.
  - https://arxiv.org/abs/2508.19786
- SharpTimeGS, arXiv 2602.02989: lifespan and velocity-aware densification for dynamic Gaussians.
  - https://arxiv.org/abs/2602.02989
- PaMoSplat: Part-Aware Motion-Guided Gaussian Splatting for Dynamic Scene Reconstruction, arXiv 2605.10307
  - https://arxiv.org/pdf/2605.10307
  
The key lesson is not simply "add more priors every iteration." The useful direction is targeted dynamic supervision, long-range correspondence, and controlled dynamic Gaussian allocation.

## Losses And Metrics

Active or intended training losses:

- `lambda_dynamic_roi`: extra photometric loss inside dynamic masks.
- `lambda_static_exclusion`: penalizes static probability for visible Gaussians projected inside dynamic masks.
- `lambda_track_flow`: rendered dense flow versus cached track-flow prior. Currently disabled.
- `lambda_scaffold_smooth`: locally smooth scaffold motion.
- `lambda_scaffold_reg`: scaffold coefficient and basis regularization.

Evaluation metrics implemented in `main.py`:

- `test/psnr`
- `test/dynamic_mask_psnr`
- `test/static_ghost_score`
- `test/dynamic_edge_magnitude`
- `test/track_flow_l1`

Important caveat: these eval metrics are logged to TensorBoard/W&B, not printed clearly in `train.log`. The train log only shows best-checkpoint messages and the W&B run summary when the run completes.

## Verification Already Done

Recent verification:

```bash
python3 -m py_compile main.py arguments/__init__.py
```

This passed after the histogram-throttling and scientific-notation logging edits.

Also verified:

- `histogram_log_interval` is accepted by the argument parser.
- The scaffold configs contain the current 9k/6k/flow-off/blur-off settings.

## Immediate Next Steps

1. Treat `scaffold_lora_route0` as the practical candidate.

2. Compare `6000` versus `9000` across more scenes. Coffee Martini suggests `6000` may be enough or better for test quality.

3. Do not launch the current dyn-densify config broadly without a point cap. It exceeded 15 hours on Coffee Martini and reached 1.59M points.

4. Run or inspect eval outputs for the `6000` checkpoints, especially visual dynamic regions:

```text
runs/scaffold_lora_route0/*/chkpnt6000.pth
runs/scaffold_lora_route0_dyn_densify/*/chkpnt6000.pth
```

5. If using long-range priors again, fix `MotionPriorCache` pathing first and confirm `Ltrack_flow` becomes nonzero in a short run.

6. Consider updating `runit.sh` and eval scripts to make `6000` the default checkpoint for the base scaffold sweep if other scenes agree with Coffee Martini.

## If Results Are Bad

Use targeted ablations, not a large sweep:

- If static branch still shows foreground ghosts, inspect masks and consider tuning `lambda_static_exclusion`.
- If dynamic masks select too much background, replace residual fallback masks with precomputed temporal-change or segmentation-refined masks.
- If scaffold hurts but dynamic ROI/static exclusion helps, keep the losses and revisit scaffold attachment or scaffold learning rates.
- If scaffold appears inactive, check scaffold coefficient and basis norms, not just the tiny weighted regularizer losses.
- If dynamic metrics are absent, confirm masks are present under `<scene>/motion_priors`.
- If track loss is inactive, confirm dense flow files are under the cache path and named to match image names.
- If dyn-densify looks promising visually, cap the point count before rerunning it.

## Scientific Priorities

The most promising next research path is not a bigger basis alone. It is supervision and allocation targeted at the failure:

1. Dynamic masks to prevent static leakage.
2. Long-range correspondence to sharpen fast motion.
3. Scaffold or node-based residual motion to avoid forcing one global temporal basis to explain all moving content.
4. Controlled motion-aware densification so new Gaussians are allocated where dynamic residual and velocity are high, without runaway point growth.
5. Dynamic-region metrics and visual diagnostics so fast-motion failures are measured directly.

Global PSNR gains below about 0.5 dB are not the main goal. The key question is whether fast-moving food/head regions become visibly sharper and whether static-branch ghost silhouettes decrease.
