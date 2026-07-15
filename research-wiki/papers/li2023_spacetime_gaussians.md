---
type: paper
node_id: paper:li2023_spacetime_gaussians
title: "Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis"
authors: ["Zhan Li", "Zhang Chen", "Zhong Li", "Yi Xu"]
year: 2023
venue: "CVPR 2024"
external_ids:
  arxiv: "2312.16812"
  doi: null
  s2: null
tags: ["dynamic-gaussians", "temporal-opacity", "capacity-allocation", "densification"]
added: 2026-07-14T22:18:30Z
---

# Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis

## One-line thesis

Temporal-opacity Gaussians plus error-and-depth-guided sampling can represent transient content and add genuinely new primitives beyond clone/split densification.

## Problem / Gap

Static Gaussian parameterizations and clone/split-only densification struggle with time-local content and missing geometry.

## Method

Spacetime Gaussians have temporal opacity, parametric motion/rotation, and neural features; high-error rays and coarse depth guide sampling of new primitives.

## Key Results

The paper reports high-quality real-time dynamic rendering with compact storage on established dynamic datasets.

## Assumptions

Photometric error and coarse rendered depth are informative proxies for missing representation capacity.

## Limitations / Failure Modes

Photometric error is not an occlusion label. Sampling can chase appearance or optimization error and does not establish foreground/background order.

## Reusable Ingredients

Temporal support, new-primitive sampling, and an explicit alternative to inheriting every new primitive from an existing one.

## Open Questions

Can calibrated visibility evidence replace photometric error as the capacity trigger while preserving a fixed total budget?

## Claims

None yet.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

It is a close capacity precedent for Phase 8, but the proposed contribution must be the visibility-to-capacity coupling rather than temporal opacity alone.
