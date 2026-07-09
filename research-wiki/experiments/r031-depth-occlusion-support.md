---
type: experiment
node_id: exp:r031-depth-occlusion-support
status: planned
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

PLANNED as of 2026-07-09. See `refine-logs/DEPTH_OCCLUSION_EVENT_SUPPORT_PLAN.md`.
