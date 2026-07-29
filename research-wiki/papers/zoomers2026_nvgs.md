---
type: paper
node_id: paper:zoomers2026_nvgs
title: "NVGS: Neural Visibility for Occlusion Culling in 3D Gaussian Splatting"
authors: ["Zoomers et al."]
year: 2026
venue: "CVPR"
external_ids:
  arxiv: "2511.19202"
tags: [static-gs, visibility, occlusion-culling, learned-visibility]
status: abstract-level
---

# NVGS: Neural Visibility for Occlusion Culling in 3D Gaussian Splatting

**Paper:** https://arxiv.org/abs/2511.19202 (CVPR 2026)
**Code:** availability not confirmed
**Evidence tier:** abstract only (2026-07-29 sweep).

## One-line thesis

Learns the viewpoint-dependent visibility function of all Gaussians in a
trained static model with a small shared MLP; queried before rasterization to
discard occluded primitives (render-time occlusion culling).

## Relevance to ADAGS

Occupies the phrase "learned per-Gaussian visibility function". Static
scenes, render-time culling only — no lifecycle, no temporal state, no
preservation. Constrains the wording of the deferred Route 2 (learned
visibility field) in [[objectives/depth-visibility-capacity-v1]]: if that
route is ever activated it must be positioned against NVGS.

## Connections

- Constrains Route 2 wording in [[objectives/depth-visibility-capacity-v1]]
