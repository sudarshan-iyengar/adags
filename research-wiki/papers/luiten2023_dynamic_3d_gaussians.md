---
type: paper
node_id: paper:luiten2023_dynamic_3d_gaussians
title: "Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis"
authors: ["Jonathon Luiten", "Georgios Kopanas", "Bastian Leibe", "Deva Ramanan"]
year: 2023
venue: "3DV 2024"
external_ids:
  arxiv: "2308.09713"
  doi: null
  s2: null
tags: ["dynamic-gaussians", "persistence", "tracking", "identity"]
added: 2026-07-14T22:18:30Z
---

# Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis

## One-line thesis

Keeping Gaussian appearance and size persistent while moving and rotating primitives can make dense long-term surface tracking emerge from rendering supervision.

## Problem / Gap

Dynamic reconstruction needs persistent scene-element identity rather than an unrelated representation at every time.

## Method

The representation optimizes per-frame Gaussian motion and rotation while preserving color, opacity, and scale, regularized by local rigidity and isometry.

## Key Results

It demonstrates dynamic novel-view synthesis, dense 6-DoF tracking, and downstream editing without input flow.

## Assumptions

Synchronized multiview observations and persistent physical elements make local rigidity a useful prior.

## Limitations / Failure Modes

Uniform persistence is poorly matched to birth, death, topology change, and transient content, and the method does not explicitly infer occlusion/reveal evidence.

## Reusable Ingredients

Persistent primitive identity and regularized motion across hidden intervals.

## Open Questions

Which Gaussian properties should persist under occlusion, and which should be allowed to branch or retire on verified reveals?

## Claims

None yet.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

It shows why surface identity matters, but Phase 8 must combine persistence with evidence-driven birth or reassignment rather than impose persistence everywhere.
