---
type: paper
node_id: paper:kang2025_cec_4dgs
title: "Clustered Error Correction with Grouped 4D Gaussian Splatting"
authors: ["Taeho Kang", "Jaeyeon Park", "Kyungjin Lee", "Youngki Lee"]
year: 2025
venue: "SIGGRAPH Asia 2025"
external_ids:
  arxiv: "2511.16112"
tags: [dynamic-gs, error-correction, densification, time-local-birth, occlusion]
status: mechanism-verified
---

# CEC-4DGS: Clustered Error Correction with Grouped 4D Gaussian Splatting

**Paper:** https://arxiv.org/abs/2511.16112 (SIGGRAPH Asia 2025; code public)
**Base method:** Ex4DGS (explicit keyframed 4DGS with temporal opacity)

Added 2026-08-08 during the STAR-GS novelty check; mechanism extracted
from the arXiv HTML full text (verified, not summary-level).

## One-line thesis

Every few hundred iterations, per-view rendered-vs-GT error regions
(dynamicity + RGB thresholds) are DBSCAN/K-means clustered into ellipses;
cross-view color consistency classifies each cluster as missing-color vs
occlusion error; missing-color errors are corrected by birthing a new
splat back-projected at SINGLE-VIEW RENDERED DEPTH (k depth samples
around the ellipse center; the depth minimizing cross-view color
discrepancy wins), with temporal opacity initialized at the error frame's
timestamp and motion inherited from the nearest splat group; occlusion
errors are corrected by splitting the occluding splat.

## Key mechanism facts

- Error detection is per-view; multiview enters only as a color-
  consistency CHECK and depth-sample selector — 3D localization trusts
  the model's rendered depth.
- Time-local birth EXISTS here (temporal opacity maximal at the error
  frame), attached to the nearest group's keyframed motion.
- NO budget neutrality: splats are added without donors (565K vs 593K
  baseline count on their config); no matched-capacity controls.
- Results: Technicolor +0.42 dB PSNR / LPIPS −8%; N3V +0.12 dB global.

## Relevance to ADAGS

THE closest prior work to STAR-GS v9 (deficit-driven time-local birth in
4DGS on N3V). Mandatory primary baseline for any STAR-GS experiment:
faithful reimplementation AND a budget-matched variant. The STAR-GS
deltas: (1) localization — CEC back-projects at single-view rendered
depth, which is structurally unreliable exactly at disocclusions/missing
surfaces; SRC uses visibility-gated multiview residual carving with no
depth at the deficit; (2) accounting — CEC grows unbudgeted; STAR-GS is
bank-matched budget-neutral with audited donors; (3) evaluation — CEC
reports global metrics; STAR-GS adds the causal-control matrix and
annotated event benchmark. Its small N3V global effect (+0.12 dB) both
warns (the axis under-delivers at global tier — consistent with our G7
evidence) and motivates event-level evaluation.

## Connections

- Pressures [[gap_map#G13 - Visibility Events Are Not Smooth Deformation]]
- Pressures [[gap_map#G5 - Capacity Allocation Must Be Matched And Dynamic-Aware]]
- Primary baseline for the STAR-GS direction (see operations/star-gs-v9-method)

## Sources

- https://arxiv.org/abs/2511.16112
- https://arxiv.org/html/2511.16112
