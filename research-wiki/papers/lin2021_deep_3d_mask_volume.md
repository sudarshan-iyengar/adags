---
type: paper
node_id: paper:lin2021_deep_3d_mask_volume
title: "View Synthesis of Dynamic Scenes based on Deep 3D Mask Volume"
authors: ["Kai-En Lin", "Guowei Yang", "Lei Xiao", "Feng Liu", "Ravi Ramamoorthi"]
year: 2021
venue: "ICCV"
external_ids:
  arxiv: "2108.13408"
  doi: null
  s2: null
tags: ["layered-representation", "disocclusion", "dynamic-view-synthesis", "background-memory"]
added: 2026-07-14T22:18:30Z
---

# View Synthesis of Dynamic Scenes based on Deep 3D Mask Volume

## One-line thesis

A 3D mask volume can separate instantaneous dynamic content from a background layer accumulated when visible and reuse that background after disocclusion.

## Problem / Gap

Framewise dynamic view synthesis flickers and invents content when foreground motion reveals previously covered background.

## Method

The method combines layered multiplane representations with a 3D mask volume that selects static background content aggregated across time.

## Key Results

The paper reports improved temporal stability and larger view extrapolation on binocular dynamic videos.

## Assumptions

Static cameras and a time-invariant background explain the relevant hidden content.

## Limitations / Failure Modes

The fixed foreground/background decomposition does not cover arbitrary deformable hidden surfaces or dynamic-behind-dynamic occlusion.

## Reusable Ingredients

Learning hidden content when visible, storing it separately from the current occluder, and reusing it on reveal.

## Open Questions

Can a Gaussian surface layer be created only where calibrated evidence supports a second surface, without assuming all hidden content is static?

## Claims

None yet.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

It is the clearest layered precedent for the desired hidden-surface memory, while its static-background assumption marks the ADAGS novelty boundary.
