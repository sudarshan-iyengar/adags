---
type: paper
node_id: paper:liu2025_occlugaussian
title: "OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering"
authors: ["Shiyong Liu", "Xiao Tang", "Zhihao Li", "Yingfan He", "Chongjie Ye", "Jianzhuang Liu", "Binxiao Huang", "Shunbo Zhou", "Xiaofei Wu"]
year: 2025
venue: "ICCV"
external_ids:
  arxiv: "2503.16177"
  doi: null
  s2: null
tags: ["occlusion", "camera-covisibility", "large-scenes", "scene-partitioning"]
added: 2026-07-14T23:36:29Z
---

# OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering

## One-line thesis

Camera co-visibility and position can partition an occluded large scene into better-supervised Gaussian regions and cull region-invisible primitives during rendering.

## Problem / Gap

Position- or grid-based large-scene partitioning groups cameras that share little visible content, wasting training capacity and producing weak regional reconstructions.

## Method

OccluGaussian builds an attributed camera graph whose edges encode co-visibility, clusters cameras into regions, selects base/extended/border views for each region, trains regions separately, merges them, and uses region-level visibility for rendering culls.

## Key Results

The ICCV 2025 paper reports stronger reconstruction and faster rendering on OccluScene3D, Zip-NeRF, and large-scene benchmarks.

## Assumptions

The scene is static, camera co-visibility is informative at regional scale, and independently reconstructed spatial partitions can be merged coherently.

## Limitations / Failure Modes

Region-level camera visibility is not per-ray temporal occlusion reasoning. The method does not model non-rigid reveal events, surface identity, or capacity lifecycle within a dynamic representation.

## Reusable Ingredients

Attributed co-visibility graphs, visibility-aware camera selection, and explicit accounting of which observations contribute to a reconstruction region.

## Open Questions

Can an analogous graph be defined over dynamic surface tracks rather than static camera regions?

## Claims

None yet.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

OccluGaussian is an occlusion-aware Gaussian precedent and prevents broad novelty wording. Its static scene-division mechanism is nevertheless distinct from Route 1's proposed dynamic surface-state and budget-reassignment hypothesis.
