---
type: paper
node_id: paper:liu2021_neuray
title: "Neural Rays for Occlusion-aware Image-based Rendering"
authors: ["Yuan Liu", "Sida Peng", "Lingjie Liu", "Qianqian Wang", "Peng Wang", "Christian Theobalt", "Xiaowei Zhou", "Wenping Wang"]
year: 2021
venue: "CVPR 2022"
external_ids:
  arxiv: "2107.13421"
  doi: null
  s2: null
tags: ["learned-visibility", "occlusion", "image-based-rendering", "ray-representation"]
added: 2026-07-14T22:18:30Z
---

# Neural Rays for Occlusion-aware Image-based Rendering

## One-line thesis

A depth-conditioned per-input-ray visibility function can prevent invisible source-view features from contaminating novel-view rendering.

## Problem / Gap

Image-based radiance fields aggregate inconsistent source features when a queried 3D point is hidden in some input views.

## Method

NeuRay predicts a ray visibility distribution and uses it to weight source-view features, then refines visibility with scene-specific consistency.

## Key Results

The paper reports stronger generalization and per-scene fine-tuning than prior feature-aggregation renderers.

## Assumptions

Source images contain enough visible evidence, and depth or cost-volume initialization is sufficiently informative.

## Limitations / Failure Modes

The model is primarily static and visibility controls observation weighting. It does not model temporal reveal identity or allocate Gaussian capacity.

## Reusable Ingredients

View-conditioned ray visibility, differentiable source-view trust, and consistency refinement.

## Open Questions

Should an ADAGS visibility field predict first-surface transmittance, a discrete surface state, or only evidence reliability?

## Claims

None yet.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

It is the closest clean precedent for a learned visibility field, while clarifying that visibility weighting alone is not hidden-surface representation.
