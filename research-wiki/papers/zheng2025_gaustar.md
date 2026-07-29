---
type: paper
node_id: paper:zheng2025_gaustar
title: "GauSTAR: Gaussian Surface Tracking and Reconstruction"
authors: ["Chengwei Zheng", "et al. (ETH AIT)"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2501.10283"
tags: [dynamic-gs, surface-tracking, topology-change, multiview]
status: abstract-level
---

# GauSTAR: Gaussian Surface Tracking and Reconstruction

**Paper:** https://arxiv.org/abs/2501.10283
**Code:** stated released at eth-ait.github.io/GauSTAR/ (not yet read)
**Evidence tier:** abstract + project page only (2026-07-29 sweep). Deep-dive
required before any related-work section is written.

## One-line thesis

Binds 3D Gaussians to mesh faces for multiview dynamic capture; tracks
surfaces with a surface-based scene-flow initialization; handles topology
changes (appearing/disappearing/splitting surfaces) by adaptively unbinding
Gaussians from the mesh and generating new surfaces from the optimized
Gaussians.

## Relevance to ADAGS

The cleanest published foil for CSVL-VPL v2: GauSTAR's answer to surface
appearance/disappearance is unbind-and-re-create, while the ADAGS thesis is
hide-and-reveal-the-same-primitives (preservation, not recreation). It is
also a multiview persistent-surface-identity precedent (mesh-bound), so any
"persistent surface identity in multiview dynamic capture" claim must be
positioned against it. No occlusion-evidence-driven capacity lifecycle, no
uncertainty/abstention per the abstract.

## Connections

- Foil for [[operations/phase9-csvl-vpl-v2-direction]]
- Pressures [[ideas/event-causal-visibility-gaussians]]
