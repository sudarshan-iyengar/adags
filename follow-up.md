# ADAGS Follow-Up Notes

This file is a handoff for the next agent. It summarizes the research direction, what has already been tried, what the latest implementation is meant to do, and where to continue without relearning the repo from scratch.

## Current Research Problem

The project is dynamic Gaussian rendering for N3V cooking scenes. The user is no longer looking for small hyperparameter gains. The visible failure mode is that fast moving regions, especially food, hands, and heads, remain blurry or smeared. Static/dynamic leakage is also visible in diagnostic renders: the static branch can contain black silhouette-like outlines of the cooking person. The final composite hides some of this leakage, but the blur remains.

The desired next step is a research-worthy improvement in photometric quality and motion representation, not another broad config sweep.

## Important Results So Far

Representative best numbers supplied by the user:

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

- `lora_r8_a32_route0` is the strongest stable base and should be treated as the current best branch/config.
- The improvements are incremental. They do not solve fast-motion smear.
- Part-basis is not necessarily a failed idea. The tested part-basis configs used a more dynamic-biased route init, while the best LoRA result used `route_logit_init: 0.0`.
- Gaussian blur was useful for stability in some earlier tests, but the strongest result here is no-blur LoRA route0.

## Ideas Already Tried

1. SEA-RAFT optical flow plus optical-flow loss.
2. Hard or thresholded per-Gaussian dynamic/static scores.
3. Logistic monotonic gate for static/dynamic regions.
4. Multi-resolution teacher-student style broad-to-detail training.
5. Gaussian blur curriculum.
6. Polynomial motion.
7. Low-rank learned temporal basis, referred to as LoRA motion.
8. Soft part-basis motion.

Important constraints from prior discussion:

- Avoid hard static conversion as the main path.
- Hard dynamic/static thresholding is unstable and too close to another recent paper.
- Keep reversible routing as the stable base.
- Use external priors as training supervision only, not final test-time inputs.

## Diagnosis

The current best methods still optimize mostly global photometric loss. That objective does not directly say: "make this fast-moving head or food region sharp."

The black silhouette issue is consistent with static/dynamic leakage. Reversible routing renders:

- static hypothesis from `x0`, opacity scaled by `1 - dynamic_prob`
- dynamic hypothesis from motion-deformed `x_dynamic`, opacity scaled by `dynamic_prob`

This preserves opacity budget, but without a dynamic-mask exclusion loss, the static branch can still partially explain moving foreground.

Flow/distillation attempts likely underperformed because they did not provide robust, long-range, occlusion-aware rendered-pixel correspondence. Previous distillation was closer to transferring temporal parameters or nearest Gaussian behavior than learning sharp trajectories.

## Literature Motivation

These papers motivated the latest implementation direction:

- MoSca, CVPR 2025: dynamic Gaussian fusion with 4D motion scaffolds and foundation priors.
  - https://openaccess.thecvf.com/content/CVPR2025/html/Lei_MoSca_Dynamic_Gaussian_Fusion_from_Casual_Videos_via_4D_Motion_CVPR_2025_paper.html
- Shape of Motion, ICCV 2025 / arXiv 2407.13764: SE(3) motion bases with depth and long-range 2D tracks.
  - https://arxiv.org/abs/2407.13764
- Prior-Enhanced GS, arXiv 2512.11356: segmentation and epipolar masks, object-depth, tracks, virtual-view depth, and scaffold projection.
  - https://arxiv.org/abs/2512.11356
- MAPo, arXiv 2508.19786: motion-aware partitioning; unified motion models can blur highly dynamic regions.
  - https://arxiv.org/abs/2508.19786
- SharpTimeGS, arXiv 2602.02989: lifespan and velocity-aware densification for dynamic Gaussians.
  - https://arxiv.org/abs/2602.02989
- PaMoSplat: Part-Aware Motion-Guided Gaussian Splatting for Dynamic Scene Reconstruction, arXiv 2605.10307
  - https://arxiv.org/pdf/2605.10307
  

The high-level conclusion was to stop relying only on global photometric fitting and add targeted dynamic-region supervision, long-range correspondence, and motion-aware Gaussian allocation.

## Latest Implementation Direction

The requested branch was `scaffold-motion-priors`, based on the LoRA motion code, with only three configs.

The implementation is intended to add:

1. Dynamic foreground masks.
2. Static-exclusion loss inside dynamic regions.
3. Optional long-range track or flow supervision.
4. Optional scaffold residual motion on top of LoRA motion.
5. Optional motion-aware densification for dynamic residual and high-motion regions.
6. Failure-targeted metrics beyond global PSNR, SSIM, and LPIPS.

The strongest base remains reversible routing plus LoRA route0.

## Expected Modified Files

These files were modified or added during the scaffold-prior implementation:

- `arguments/__init__.py`
- `gaussian_renderer/__init__.py`
- `main.py`
- `runit.sh`
- `scene/gaussian_model.py`
- `scripts/run_leonardo.sh`
- `scripts/build_motion_priors.py`
- `utils/motion_prior_utils.py`
- `configs/n3v/lora_route0_dynmask.yaml`
- `configs/n3v/scaffold_lora_route0.yaml`
- `configs/n3v/scaffold_lora_route0_dyn_densify.yaml`

There were unrelated pre-existing untracked files. Do not remove or overwrite them without checking:

- `configs/n3v/bootstrap.yaml`
- `configs/n3v/det_con.yaml`
- `requirements.txt`
- `verify_mask.jpg`
- `verify_masked_flow.jpg`
- `verify_raw_flow.jpg`

## New Configs

Only these three configs should be run first:

1. `configs/n3v/lora_route0_dynmask.yaml`
   - Current best LoRA route0 plus dynamic ROI and static-exclusion losses.
   - No scaffold residual motion.
   - Uses external masks if available, otherwise residual dynamic masks can be used for training.

2. `configs/n3v/scaffold_lora_route0.yaml`
   - LoRA route0 plus scaffold residual motion.
   - Adds track or flow reprojection loss if cached priors exist.
   - No motion-aware densification.

3. `configs/n3v/scaffold_lora_route0_dyn_densify.yaml`
   - LoRA route0 plus scaffold residual motion.
   - Dynamic ROI, static exclusion, optional track/flow loss.
   - Motion-aware densification enabled.

`runit.sh` should schedule exactly these three configs across the existing scenes, all with eval at `CKPT_ITER=15000`.

## Run Directory Behavior

`scripts/run_leonardo.sh` was updated to support `RUN_LABEL`.

Expected behavior:

- If `RUN_LABEL` is set, new runs go under:
  - `$WORK/proj_adags/runs/<RUN_LABEL>/<RUN_ID>`
- `runit.sh` exports `RUN_LABEL="$CFG_NAME"` for both train and eval jobs.
- This should produce paths like:
  - `runs/lora_route0_dynmask/<timestamp>_<scene>_lora_route0_dynmask`
  - `runs/scaffold_lora_route0/<timestamp>_<scene>_scaffold_lora_route0`
  - `runs/scaffold_lora_route0_dyn_densify/<timestamp>_<scene>_scaffold_lora_route0_dyn_densify`

## Motion Prior Cache

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

Accepted mask formats include:

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

If no masks exist and `dynamic_mask_from_residual: true`, training can compute residual masks online. Evaluation metrics should not use residual fallback, because that would bias the metric.

## Preprocessing

A simple temporal-change mask builder was added:

```bash
python scripts/build_motion_priors.py --scene data/n3v/cook_spinach
```

On Leonardo this would usually be:

```bash
python scripts/build_motion_priors.py --scene "$WORK/proj_adags/data/n3v/cook_spinach"
```

Repeat for all scenes if using this simple mask prior. For a stronger version, generate dynamic masks and long-range tracks externally with a foundation tracker or segmenter, then place the outputs in the cache structure above.

The most important missing research ingredient is true long-range correspondence. Dense track-flow caches from CoTracker, TAPIR, SEA-RAFT with occlusion handling, or a similar tracker should be exported as `motion_priors/track_flows/<image_name>.npy`.

## Scaffold Motion Details

The scaffold residual is intended to be neutral at initialization.

Added tensors in `scene/gaussian_model.py`:

- `_motion_scaffold_node_xyz`
- `_motion_scaffold_coeff`
- `_motion_scaffold_basis`
- `_motion_scaffold_attach_idx`
- `_motion_scaffold_attach_w`

Intended dynamic position:

```text
x_dynamic = x0 + lora_motion_offset + scaffold_motion_offset
```

Because scaffold coefficients initialize to zero, the model should initially reduce exactly to LoRA route0.

Scaffold nodes are initialized from sampled Gaussian positions. Gaussians attach to nearby scaffold nodes with fixed KNN-style soft weights.

Per-Gaussian routing remains reversible and opacity-conservative. The scaffold should not revive hard static conversion.

## Losses And Metrics

New or intended training losses:

- `lambda_dynamic_roi`: extra photometric loss inside dynamic masks.
- `lambda_static_exclusion`: penalizes static probability for visible Gaussians projected inside dynamic masks.
- `lambda_track_flow`: rendered dense flow versus cached track-flow prior.
- `lambda_scaffold_smooth`: locally smooth scaffold motion.
- `lambda_scaffold_reg`: scaffold coefficient or basis regularization.

New or intended evaluation metrics:

- `test/dynamic_mask_psnr`
- `test/static_ghost_score`
- `test/dynamic_edge_magnitude`
- `test/track_flow_l1`

If no external masks or flow priors exist, some dynamic metrics may be absent.

## Verification Already Done

Previous verification reported:

```bash
python -m py_compile main.py scene/gaussian_model.py gaussian_renderer/__init__.py utils/motion_prior_utils.py scripts/build_motion_priors.py
```

This passed after using command escalation because local Windows sandbox process creation was failing.

`git diff --check` passed except line-ending warnings.

`bash -n` was not run because `bash` was not available locally.

OmegaConf YAML validation was not completed because that command was rejected or unavailable. A future agent should validate the YAMLs before long cluster runs.

## Immediate Next Steps

1. Inspect the current branch and diff.

```bash
git status --short --branch
git diff --stat
```

2. Confirm the branch is `scaffold-motion-priors`, not `LoRa-motion`.

```bash
git branch --show-current
```

3. Validate that `runit.sh` schedules only the three scaffold-prior configs and exports `RUN_LABEL`.

4. Validate YAML loading with OmegaConf if available.

```bash
python -c "from omegaconf import OmegaConf; [OmegaConf.load(p) for p in ['configs/n3v/lora_route0_dynmask.yaml','configs/n3v/scaffold_lora_route0.yaml','configs/n3v/scaffold_lora_route0_dyn_densify.yaml']]; print('ok')"
```

5. Build simple dynamic masks for at least one scene, then do a short sanity run before overnight jobs.

6. For full research value, generate long-range track-flow caches and rerun `scaffold_lora_route0`.

## If Results Are Bad

Start with targeted ablations, not a large sweep:

- If static branch looks over-suppressed or background quality drops, reduce `lambda_static_exclusion`.
- If dynamic masks select too much background, replace residual fallback masks with precomputed temporal-change or segmentation-refined masks.
- If scaffold hurts but dynmask helps, keep dynamic ROI/static exclusion and revisit scaffold attachment.
- If scaffold has no effect, check that scaffold coeff and basis norms become nonzero and that `lambda_track_flow` is active.
- If no dynamic metrics appear, confirm masks are present in `<scene>/motion_priors`.
- If track loss is inactive, confirm dense flow files are named to match image names and have shape `[H,W,2]`.

Do not immediately increase config count. The user explicitly wants a maximum of three configs until the method shows clear promise.

## Scientific Priorities

The most promising next research path is not a bigger basis alone. It is supervision and allocation targeted at the failure:

1. Dynamic masks to prevent static leakage.
2. Long-range correspondence to sharpen fast motion.
3. Scaffold or node-based residual motion to avoid forcing one global temporal basis to explain all moving content.
4. Motion-aware densification so new Gaussians are allocated where dynamic residual and velocity are high.
5. Dynamic-region metrics so visual failures are measured directly.

Global PSNR gains below about 0.5 dB are not the main goal. The important question is whether fast-moving food/head regions become visibly sharper and whether static-branch ghost silhouettes decrease.
