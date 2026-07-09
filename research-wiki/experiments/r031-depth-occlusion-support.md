---
type: experiment
node_id: exp:r031-depth-occlusion-support
status: complete
created: 2026-07-09
idea: idea:depth-occlusion-event-support
---

# R031 Depth Occlusion Event Support

## Question

Can Depth Anything 3 provide non-oracle occlusion/reveal support masks that cover the frozen R009 event zones more plausibly than R026 boundary support, without using the frozen crop labels as method input?

## Method

- Depth source: `ByteDance-Seed/depth-anything-3`.
- Primary model: `depth-anything/DA3NESTED-GIANT-LARGE-1.1`.
- Input selection: scene sources and frame ranges only; default first wave uses `cam00` frames `0..299` for `cut_roasted_beef`, `flame_steak`, and `sear_steak`.
- Support cues: depth-gradient edges, temporal depth change, confidence discontinuity, dynamic-mask boundary/interior, flow boundary/magnitude, route0 dynamic/static disagreement, and route0 flicker.
- Output schema: `support_frames` mask manifest compatible with `event_boundary_support_manifest`.

## Guardrails

- `uses_gt_residual=false`
- `uses_gt_crop_pixels=false`
- `uses_frozen_window_labels=false`
- Frozen R009 windows are used only after support generation for posthoc diagnostic overlap.

## Predeclared Gates

Support PASS requires a valid compact support artifact and better posthoc frozen-window support overlap than R026. Rendered-method PASS is not claimed by R031 and requires later checkpoint-backed/training-loop evidence.

## Status

COMPLETE as a support-only diagnostic as of 2026-07-09.

## Results

Setup and sidecar generation passed:
- DA3 repo: `$WORK/proj_adags/repo/depth-anything-3`.
- DA3 commit: `41736238f5bced4debf3f2a12375d2466874866d`.
- Model: `depth-anything/DA3NESTED-GIANT-LARGE-1.1`.
- Full frame manifest: 900 `cam00` frames across `cut_roasted_beef`, `flame_steak`, and `sear_steak`.
- Inference job `49030185`: wrote 900/900 depth sidecars.

Support diagnostics:

| Run | Variant | Support Frames | Mean Support-Frame Fraction | Mean Crop Coverage | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| R031 | sparse default | 83 | 0.0625 | 0.000030 | weak |
| R032 | high-recall sparse | 355 | 0.4125 | 0.002846 | weak |
| R033 | high-recall tile-fill | 408 | 0.3750 | 0.001253 | weak |

Comparison points:
- R026 boundary support: mean crop coverage `0.000000`.
- R020 high-recall non-oracle boxes: mean crop coverage `0.491371`.

## Interpretation

PASS: the DA3 setup, weight caching, Slurm inference, and non-oracle support-manifest pipeline are operational.

FAIL/WEAK: the current DA3 support-fusion formulations do not localize the frozen event crops strongly enough to justify a positive support claim. High-recall expansion improved temporal hit rate, but spatial crop coverage stayed extremely low. Tile-fill did not rescue the signal.

This does not rule out depth as a useful cue. It does rule out this specific DA3 depth-edge/confidence/flow/mask fusion as an immediately compelling support artifact for the next rendered method. Given R030, the next rendered milestone should prioritize training-loop integration or a changed capacity/optimization mechanism rather than another posthoc support-only run.

## Evidence

- `refine-logs/depth_occlusion_support/r031_da3_frame_manifest.json`
- `refine-logs/depth_occlusion_support/r031_da3_depth_full/da3_depth_manifest.json`
- `refine-logs/depth_occlusion_support/r031_depth_support_full/`
- `refine-logs/depth_occlusion_support/r031_support_overlap_full/support_overlap_summary.json`
- `refine-logs/depth_occlusion_support/r032_depth_support_highrecall/`
- `refine-logs/depth_occlusion_support/r032_support_overlap_highrecall/support_overlap_summary.json`
- `refine-logs/depth_occlusion_support/r033_depth_support_highrecall_tilefill/`
- `refine-logs/depth_occlusion_support/r033_support_overlap_tilefill/support_overlap_summary.json`
