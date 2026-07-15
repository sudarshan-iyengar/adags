---
type: paper
node_id: paper:yoon2020_globally_coherent_depths
title: "Novel View Synthesis of Dynamic Scenes with Globally Coherent Depths from a Monocular Camera"
authors: ["Jae Shin Yoon", "Kihwan Kim", "Orazio Gallo", "Hyun Soo Park", "Jan Kautz"]
year: 2020
venue: "CVPR"
external_ids:
  arxiv: "2004.01294"
  doi: null
  s2: null
tags: ["dynamic-depth", "scale-alignment", "multi-view-stereo", "scene-flow"]
added: 2026-07-14T22:18:30Z
---

# Novel View Synthesis of Dynamic Scenes with Globally Coherent Depths from a Monocular Camera

## One-line thesis

Complete but view-variant monocular depth becomes temporally useful only after alignment to incomplete but globally coherent multiview geometry.

## Problem / Gap

Monocular depth is dense but scale-inconsistent, while MVS depth is geometrically anchored but incomplete on moving content.

## Method

The method fuses monocular and multiview depth, corrects scale, and regularizes geometry with optical-flow and 3D scene-motion consistency before view synthesis.

## Key Results

It reports more coherent dynamic depth and stronger dynamic view synthesis than prior single-source depth pipelines.

## Assumptions

Camera poses, useful MVS structure, flow, and foreground handling are available.

## Limitations / Failure Modes

The pipeline is preprocessing-heavy, and disoccluded content can still be incomplete or artifact-prone.

## Reusable Ingredients

A scene-level geometric scale anchor, explicit scale correction, and temporal comparison after correspondence rather than at the same raw pixel.

## Open Questions

What is the smallest calibrated depth anchor needed for the N3V cooking scenes?

## Claims

None yet.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

It directly explains why R031's independent percentile normalization removed the temporal meaning needed for visibility inference.
