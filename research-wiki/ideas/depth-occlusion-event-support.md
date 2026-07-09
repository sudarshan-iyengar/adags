---
id: idea:depth-occlusion-event-support
status: active
created: 2026-07-08
related_gap: G13
---

# Depth Occlusion Event Support

Use Depth Anything 3 as a frozen visual-geometry prior for non-oracle occlusion/reveal support in the event-crop objective.

## Refined Thesis

Depth should not be treated as a direct Gaussian supervision target yet. Its first scientifically clean role is to propose compact visibility-event support masks from depth discontinuities, temporal depth changes, and confidence discontinuities, then test whether those masks identify the real frozen-window event zones better than R026 boundary support without using the frozen crop labels.

This is a support-discovery line in parallel with training-loop integration. R030 already showed oracle crop support does not rescue the current posthoc micro-densification recipe, so a DA3 support pass is not a rendered-method success by itself.

## Method Sketch

Generate per-frame depth maps with Depth Anything 3, then compute depth discontinuities and foreground/background ordering cues. Candidate support should be the calibrated union of:

- depth-gradient boundaries,
- temporal depth changes,
- confidence boundaries or low-confidence zones,
- dynamic-mask boundaries,
- flow invalidity or large flow-magnitude changes,
- route0 render flicker,
- route0 dynamic/static disagreement.

The output is a support manifest compatible with the existing event-local ROI/densification code. It must record `uses_gt_residual=false`, `uses_gt_crop_pixels=false`, and `uses_frozen_window_labels=false`.

## Possible Depth Sources

Primary source: `ByteDance-Seed/depth-anything-3`, stored on HPC under `$WORK/proj_adags/repo/depth-anything-3`.

Primary model: `depth-anything/DA3NESTED-GIANT-LARGE-1.1`.

Reason: DA3 is any-view and multiview-oriented, and the nested giant-large checkpoint combines any-view geometry with metric-scale depth. The refreshed `-1.1` checkpoint is preferred over the original nested giant-large checkpoint.

Fallbacks only after documented setup/runtime failure:
- `depth-anything/DA3-LARGE-1.1` for lower memory any-view support.
- `depth-anything/DA3MONO-LARGE` as a monocular diagnostic.

## How It Interacts With Gaussians

Depth support should not directly supervise Gaussian depth at first. It should first act as a support prior:

1. propose occlusion/reveal boundary masks or boxes;
2. audit support size and posthoc frozen-window coverage;
3. only then gate local ROI loss or micro-densification;
4. preserve static-exclusion safeguards and point caps.

## Failure Modes

- Depth may fail on flames, specular meat, utensils, hands, and motion blur.
- Relative depth can flicker across frames.
- Depth edges may identify silhouettes but not actual newly revealed texture.
- Strict intersection with masks/flow may miss the event, as R026 did.
- Broad depth support may repeat R017/R025 damage.

## Minimal Test

Before training, build a DA3 depth-support manifest on the three R009 source scenes and run `scripts/audit_event_support_overlap.py`. Training is justified only if the support improves over R026's near-zero crop coverage while staying compact and visually plausible.

Predeclared support gate lives in `refine-logs/DEPTH_OCCLUSION_EVENT_SUPPORT_PLAN.md`.

## Current Execution Plan

- R031a: clone/setup DA3 and cache `DA3NESTED-GIANT-LARGE-1.1` on the login node.
- R031b: build a non-oracle `cam00` full-scene frame manifest from `scene_sources` only.
- R031c: run DA3 inference through Slurm and write resumable depth sidecars.
- R031d: build `depth_occlusion_support_manifest.json` with the existing `support_frames` schema.
- R031e: run posthoc frozen-window overlap audit as a diagnostic only.

## Result Update 2026-07-09

Status: active but not validated as a strong support mechanism.

What passed:
- DA3 setup on Leonardo completed.
- `DA3NESTED-GIANT-LARGE-1.1` weights were cached locally.
- Full `cam00` inference wrote 900/900 depth sidecars.
- Support manifests for R031, R032, and R033 were structurally valid and recorded `uses_gt_residual=false`, `uses_gt_crop_pixels=false`, `uses_frozen_window_labels=false`.

What failed or remained weak:
- R031 default sparse support: mean crop coverage `0.000030`.
- R032 high-recall sparse support: mean crop coverage `0.002846`.
- R033 high-recall tile-fill support: mean crop coverage `0.001253`.
- These improve over R026 boundary support's zero crop coverage only weakly and remain far below R020 high-recall boxes (`0.491371`).

Conclusion: DA3 sidecars are reusable, but the current support-fusion recipe should not be used as the main support input for a positive rendered-method claim. Future depth use should either change the formulation substantially, use multiview/temporal DA3 outputs more directly, or be embedded into training-loop integration as a soft reliability cue rather than as a standalone crop-support detector.
