---
type: paper
node_id: paper:gao2026_proxy_gs
title: "Proxy-GS: Efficient 3D Gaussian Splatting via Proxy Mesh"
authors: ["Yuanyuan Gao", "Yuning Gong", "Yifei Liu", "Jingfeng Li", "Zhihang Zhong", "Dingwen Zhang", "Yanci Zhang", "Dan Xu", "Xiao Sun"]
year: 2026
venue: "CVPR"
external_ids:
  arxiv: "2509.24421"
  doi: null
  s2: null
tags: ["occlusion", "proxy-depth", "densification", "structured-gaussians"]
added: 2026-07-14T23:36:29Z
---

# Proxy-GS: Efficient 3D Gaussian Splatting via Proxy Mesh

## One-line thesis

A lightweight proxy mesh supplies occlusion-depth maps that cull invisible structured Gaussians at render time and guide anchor densification toward proxy surfaces during training.

## Problem / Gap

MLP-decoded structured Gaussian systems retain redundant anchors and can densify inconsistently behind occluding geometry because training and rendering lack a common occlusion prior.

## Method

Proxy-GS rasterizes a proxy mesh into a fast view-dependent depth map. It rejects anchors or decoded Gaussians behind that depth for rendering and biases new anchor growth toward proxy-consistent surfaces during training.

## Key Results

The CVPR 2026 paper reports more than 2.5x speedup over Octree-GS in heavily occluded MatrixCity streets while improving rendering quality.

## Assumptions

A useful proxy can be extracted from a pretrained static 3DGS and remains geometrically meaningful for the target structured-GS scene.

## Limitations / Failure Modes

The proxy can inherit missing or wrong geometry. The method targets static large scenes and efficient surface-aligned anchor growth; it does not infer non-rigid temporal occlusion/reveal state or preserve a surface identity through hidden intervals.

## Reusable Ingredients

Fast proxy-depth visibility, a shared training/inference occlusion prior, safety margins, and proxy-guided capacity placement.

## Open Questions

Can a dynamic, uncertainty-calibrated proxy be constructed without first solving the hidden surface, and can budget-neutral reassignment replace net densification?

## Claims

None yet.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

Proxy-GS is a direct precedent for using occlusion depth to change densification. ADAGS cannot claim visibility-guided capacity as new in general; its working distinction must concern calibrated non-rigid reveal state, hidden-surface learning, uncertainty, and budget-neutral preservation/reassignment.
