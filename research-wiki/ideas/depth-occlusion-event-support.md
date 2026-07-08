---
id: idea:depth-occlusion-event-support
status: proposed
created: 2026-07-08
related_gap: G13
---

# Depth Occlusion Event Support

Use monocular depth as a non-oracle cue for occlusion and reveal support in the event-crop objective.

## Method Sketch

Generate or load per-frame monocular depth maps, then compute depth discontinuities and foreground/background ordering cues. Candidate support should be the intersection or calibrated union of:

- depth-gradient boundaries,
- dynamic-mask boundaries,
- flow invalidity or large flow-magnitude changes,
- route0 render flicker,
- route0 dynamic/static disagreement.

The output is a support manifest compatible with the existing event-local ROI/densification code. It must record `uses_gt_residual=false`, `uses_gt_crop_pixels=false`, and `uses_frozen_window_labels=false`.

## Possible Depth Sources

- Existing depth sidecars, if found on HPC.
- Depth Anything V2 for fast robust relative depth and fine details.
- Depth Pro for sharp zero-shot metric depth boundaries.
- VGGT if multi-frame geometry, depth, cameras, and point tracks become worth the integration cost.

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

Before training, build a depth-support manifest on the three R009 source scenes and run `scripts/audit_event_support_overlap.py`. Training is justified only if the support improves over R026's zero crop coverage while staying compact and visually plausible.
