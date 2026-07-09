# Depth Occlusion Event Support Plan

Date: 2026-07-09

## Problem Anchor

- Bottom-line problem: recover occlusion/reveal event support for N3V cooking scenes without using frozen R009 crop labels as method input.
- Must-solve bottleneck: R017/R025/R027/R030 show that the current posthoc Gaussian refinement/micro-densification mechanism fails; support discovery still matters, but support-only tweaks must be tested as support artifacts before any positive method claim.
- Non-goals: do not claim event-crop repair from a support-only artifact, do not tune thresholds on frozen crop overlap, and do not use GT crop residuals or frozen R009 crops as test-time support.
- Constraints: use Depth Anything 3 from `ByteDance-Seed/depth-anything-3`, store the checkout under `$WORK/proj_adags/repo/depth-anything-3`, download on the login node only, run heavy inference through Slurm, keep durable state in `refine-logs/`.
- Success condition: DA3-based support is compact, structurally valid, non-oracle, visually plausible, and improves posthoc frozen-window support overlap versus R026 without expanding into broad masks.

## Method Thesis

Use Depth Anything 3 as a frozen visual-geometry prior to propose occlusion/reveal support from depth discontinuities, temporal depth changes, confidence discontinuities, dynamic-mask boundaries, flow cues, and route0 render diagnostics. The output is the existing `support_frames` mask-manifest schema consumed by `event_boundary_support_manifest`.

## Depth Source

Primary model: `depth-anything/DA3NESTED-GIANT-LARGE-1.1`.

Rationale:
- It is the strongest released DA3 model family relevant to multiview visual geometry.
- It supports relative depth, pose estimation, pose-conditioned depth, metric depth, and 3D geometry outputs.
- The DA3 README says refreshed `-1.1` checkpoints are preferred over original giant/large checkpoints.

Fallbacks, only after documented setup/runtime failure:
- `depth-anything/DA3-LARGE-1.1` if the nested giant model is too slow or too memory-heavy for the first support screen.
- `depth-anything/DA3MONO-LARGE` only as a monocular diagnostic if any-view models cannot run.

## Concrete Pipeline

R031a setup:
- Clone DA3 under `$WORK/proj_adags/repo/depth-anything-3`.
- Create or reuse a DA3-capable Python environment.
- Cache the selected Hugging Face model under `$WORK/proj_adags/cache/huggingface` or `$WORK/proj_adags/models`.
- Record repo commit, model id, environment path, and download status.

R031b frame manifest:
- Build a frame list from `scene_sources` only: scenes, frame ranges, data paths, and image sizes.
- Default first wave: camera `cam00`, frames `0..299`, scenes `cut_roasted_beef`, `flame_steak`, and `sear_steak`.
- Do not read `windows[].crop_xyxy`, occluder text, or frozen-window labels for support generation.

R031c DA3 inference:
- Run DA3 through Slurm on the frame manifest.
- Save per-frame `.npz` sidecars with depth and confidence.
- Make inference resumable by skipping already-written sidecars unless `--overwrite` is passed.

R031d support build:
- Compute robust normalized depth edges from each depth map.
- Add temporal depth change for adjacent frames in the same scene/camera.
- Add confidence-edge/low-confidence cues when DA3 emits confidence.
- Combine with dynamic-mask boundary/interior, flow boundary/magnitude, route0 dynamic/static disagreement, and route0 flicker.
- Select compact tiled components under caps and write binary `support_masks/<scene>/<image_name>.png`.

R031e audit:
- Run `scripts/audit_event_support_overlap.py` only after support generation.
- Treat overlap as a posthoc diagnostic, not threshold tuning.
- Compare against R026 support overlap and record PASS/FAIL/SKIP.

## Predeclared Gates

Support artifact PASS:
- `depth_occlusion_support_validation.json` has `ok=true`, zero errors, and no scientific guardrail violation.
- Manifest records `uses_gt_residual=false`, `uses_gt_crop_pixels=false`, and `uses_frozen_window_labels=false`.
- Mean support-frame fraction on frozen windows improves over R026's `0.0250`.
- Mean crop coverage improves over R026's `0.000000`.
- Support remains compact: max selected support fraction per frame at or below the declared cap.

Support artifact FAIL:
- DA3 support is structurally valid but has essentially zero frozen-window support overlap, or produces broad masks that violate caps.
- DA3 depth is visibly dominated by flames/specularities/utensils and does not produce plausible occlusion/reveal zones.

Rendered-method PASS:
- Not claimed in R031. Requires a later training-loop-integrated or checkpoint-backed Gaussian-rendered experiment to pass the existing strict event-crop gate.

SKIP:
- Skip immediate posthoc micro-densification with DA3 support if R031 only improves support but R030 already showed oracle support does not rescue the current posthoc mechanism.

BLOCKED:
- Only after three serious setup/run attempts fail because of external access, permissions, incompatible CUDA/PyTorch/DA3 dependencies, or unavailable weights.

## Expected Failure Modes

- DA3 depth edges select all object silhouettes, not only occlusion/reveal zones.
- Flames, specular meat, utensils, and motion blur corrupt depth or confidence.
- Relative-depth scale flickers temporally.
- Monocular cam00 inference misses multiview-consistent geometry; a later multiview-frame mode may be needed.
- Support improves overlap but still cannot produce a positive rendered method without training-loop integration.

## First Wave

- R031a: setup DA3 repo/env/weights.
- R031b: prepare full-scene `cam00` frame manifest.
- R031c: run DA3 inference for the three R009 source scenes.
- R031d: build non-oracle depth support.
- R031e: posthoc support overlap audit and qualitative strips if support passes validation.

## Execution Result

Status as of 2026-07-09T22:40+02:00: pipeline PASS, support usefulness WEAK/FAIL for the tested formulations.

R031a setup:
- PASS. DA3 cloned to `$WORK/proj_adags/repo/depth-anything-3`.
- DA3 commit: `41736238f5bced4debf3f2a12375d2466874866d`.
- Model snapshot: `$WORK/proj_adags/models/depth-anything/DA3NESTED-GIANT-LARGE-1.1`.
- Environment repaired after an initial missing-`addict` smoke failure; final runtime uses DA3 venv packages plus ADAGS Torch/Torchvision.

R031b/R031c depth sidecars:
- PASS. Prepare job `49030039` wrote 900 non-oracle `cam00` frame records.
- PASS. Inference job `49030185` wrote 900/900 DA3 depth sidecars with `skipped_existing=0`.
- Evidence: `refine-logs/depth_occlusion_support/r031_da3_frame_manifest.json` and `refine-logs/depth_occlusion_support/r031_da3_depth_full/da3_depth_manifest.json`.

R031d/R031e sparse support:
- PASS structural validation, WEAK support diagnostic.
- Support job `49030546`: 83 support frames, validation OK, guardrail flags false.
- Posthoc frozen-window overlap: mean support-frame fraction `0.0625`; mean crop coverage `0.000030`.

R032 high-recall sparse sensitivity:
- PASS structural validation, WEAK support diagnostic.
- Support job `49031389`: 355 support frames, validation OK, guardrail flags false.
- Posthoc frozen-window overlap: mean support-frame fraction `0.4125`; mean crop coverage `0.002846`.

R033 high-recall tile-fill sensitivity:
- PASS structural validation, WEAK support diagnostic.
- Commit `dac940a` added `--fill-component-tiles`.
- Support job `49032424`: 408 support frames, validation OK, guardrail flags false.
- Posthoc frozen-window overlap: mean support-frame fraction `0.3750`; mean crop coverage `0.001253`.

Interpretation:
- The setup and sidecar pipeline are reusable.
- The current DA3 support-fusion recipe is not a strong event-crop support mechanism. It improves over R026's zero crop coverage only marginally and remains far below R020 high-recall box coverage (`0.491371` mean crop coverage).
- Do not run the current posthoc micro-densification recipe with R031/R032/R033 as a claimed method. R030 already showed oracle crop support fails under that recipe.
- The next meaningful rendered-method milestone should be training-loop integration or a changed capacity/optimization mechanism. DA3 sidecars may be reused there, but this support-selection formulation is not enough by itself.
