---
type: paper
node_id: paper:guo2026_usplat4d
title: "Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction"
authors: ["Fengzhi Guo", "Chih-Chuan Hsu", "Sihao Ding", "Cheng Zhang"]
year: 2026
venue: "ICLR"
external_ids:
  arxiv: "2510.12768"
  doi: null
  s2: null
tags: ["uncertainty", "dynamic-gaussians", "occlusion", "motion-propagation"]
added: 2026-07-14T22:18:30Z
---

# Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction

## One-line thesis

Time-varying per-Gaussian uncertainty can identify reliable anchors and propagate their motion to poorly observed dynamic primitives.

## Problem / Gap

Uniform optimization ignores that primitives seen repeatedly are better constrained than primitives observed sparsely or under occlusion.

## Method

USplat4D estimates per-Gaussian uncertainty, builds an uncertainty-aware spatiotemporal graph, and propagates motion from reliable anchors.

## Key Results

The paper reports more stable geometry under occlusion and stronger extreme-view synthesis across real and synthetic data.

## Assumptions

The base dynamic Gaussian model contains usable primitives and repeated observations reveal reliable anchors.

## Limitations / Failure Modes

It reorganizes optimization of existing primitives but does not geometrically infer occlusion/reveal states or create hidden-surface capacity.

## Reusable Ingredients

Time-varying primitive uncertainty, reliable anchors, and uncertainty-aware propagation.

## Open Questions

Can geometric visibility residuals calibrate per-surface uncertainty well enough to decide preservation, creation, or reassignment?

## Claims

None yet.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

It motivates uncertainty as a first-class Gate A output while showing that uncertainty-weighted optimization alone does not solve Gate B.
