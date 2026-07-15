---
type: paper
node_id: paper:lin2025_depth_anything_3
title: "Depth Anything 3: Recovering the Visual Space from Any Views"
authors: ["Haotong Lin", "Sili Chen", "Junhao Liew", "Donny Y. Chen", "Zhenyu Li", "Guang Shi", "Jiashi Feng", "Bingyi Kang"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2511.10647"
  doi: null
  s2: null
tags: ["depth", "any-view-geometry", "pose-conditioning", "foundation-model"]
added: 2026-07-14T22:18:30Z
---

# Depth Anything 3: Recovering the Visual Space from Any Views

## One-line thesis

A single depth-ray transformer can recover spatially consistent geometry from arbitrary image collections, with or without supplied camera poses.

## Problem / Gap

General visual geometry models often separate monocular depth, multiview depth, and camera estimation into specialized pipelines.

## Method

DA3 predicts depth and rays from one or more inputs and exposes optional known-camera conditioning. Its API can accept intrinsics and extrinsics, align the prediction to supplied pose scale, and return depth, confidence, and camera estimates.

## Key Results

The paper reports strong any-view geometry, camera-pose, and monocular-depth performance on its visual geometry benchmark.

## Assumptions

Spatial consistency is learned from broad public academic data. Dynamic frames are not themselves an explicit occlusion-state model.

## Limitations / Failure Modes

Any-view consistency does not guarantee correct non-rigid temporal correspondence, foreground/background ordering, or calibrated event uncertainty. Processing disconnected image groups can leave them mutually unaligned.

## Reusable Ingredients

Known-camera conditioning, joint image groups, depth/confidence outputs, and alignment to input camera scale.

## Open Questions

How should dynamic synchronized cameras and adjacent times be grouped so moving surfaces are not forced into a static geometry?

## Claims

None yet.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

R031 used the checkpoint but omitted its camera-conditioning and common-geometry capabilities. DA3 remains a useful uncertainty-bearing cue, not an occlusion oracle.
