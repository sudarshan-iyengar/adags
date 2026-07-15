---
type: paper
node_id: paper:zhang2026_vad_gs
title: "Visibility-Aware Densification for 3D Gaussian Splatting in Dynamic Urban Scenes"
authors: ["Yikang Zhang", "Rui Fan"]
year: 2026
venue: "CVPR"
external_ids:
  arxiv: "2510.09364"
  doi: null
  s2: null
tags: ["visibility", "densification", "multi-view-stereo", "geometry-completion"]
added: 2026-07-14T22:18:30Z
---

# Visibility-Aware Densification for 3D Gaussian Splatting in Dynamic Urban Scenes

## One-line thesis

Voxel visibility reasoning and calibrated cross-frame MVS can identify missing structures and create new Gaussians where clone/split densification cannot.

## Problem / Gap

Partially initialized Gaussian scenes can send gradients to incorrect visible primitives because no existing point represents an occluded or missing surface.

## Method

VAD-GS identifies unreliable first-visible voxels, selects diverse supporting camera-time views, reconstructs geometry by patch-match MVS, and initializes new Gaussians.

## Key Results

It reports stronger geometry and rendering on Waymo and nuScenes, including dynamic objects.

## Assumptions

Urban LiDAR, boxes or instance structure, calibrated views, and locally rigid or planar surfaces support its reconstruction.

## Limitations / Failure Modes

Rigid urban assumptions do not transfer cleanly to deformable hands, food, flames, and utensils; the paper reports weaker behavior on non-rigid pedestrians.

## Reusable Ingredients

First-surface visibility, diverse supporting-view selection, MVS-guided new capacity, and geometry evaluation beyond PSNR.

## Open Questions

Can the same principle operate without LiDAR or boxes on non-rigid N3V content, with uncertainty and matched point budgets?

## Claims

None yet.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

This is the closest competing visibility-to-capacity work. ADAGS novelty must be non-oracle, non-rigid multiview-temporal visibility and hidden-surface memory rather than generic visibility-aware densification.
