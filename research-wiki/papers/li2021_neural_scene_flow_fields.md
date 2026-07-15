---
type: paper
node_id: paper:li2021_neural_scene_flow_fields
title: "Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes"
authors: ["Zhengqi Li", "Simon Niklaus", "Noah Snavely", "Oliver Wang"]
year: 2021
venue: "CVPR"
external_ids:
  arxiv: "2011.13084"
  doi: null
  s2: null
tags: ["scene-flow", "dynamic-radiance-field", "disocclusion", "temporal-consistency"]
added: 2026-07-14T22:18:30Z
---

# Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes

## One-line thesis

A time-varying radiance field with forward/backward 3D scene flow can compare corresponding dynamic surfaces and downweight invalid disocclusion supervision.

## Problem / Gap

Monocular dynamic view synthesis must disentangle camera motion, geometry, and non-rigid scene motion despite missing observations.

## Method

NSFF learns dynamic geometry, appearance, forward/backward 3D scene flow, and disocclusion weights, with flow reprojection and cycle consistency.

## Key Results

The paper reports improved space-time view synthesis on complex real videos, including thin structures and view-dependent effects.

## Assumptions

Known camera poses, useful optical flow, and enough visible observations exist to constrain the learned scene flow.

## Limitations / Failure Modes

Disocclusion weights mostly suppress invalid loss. They do not guarantee capacity for never-observed or poorly represented surfaces, and long or fast motions can create local minima.

## Reusable Ingredients

Forward/backward 3D consistency, reprojection to observed flow, and explicit invalid-correspondence weighting.

## Open Questions

Can calibrated multiview N3V observations produce surface visibility evidence without learning a full scene-flow field first?

## Claims

None yet.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

It provides the temporal 3D comparison missing from R031's unwarped same-pixel depth subtraction.
