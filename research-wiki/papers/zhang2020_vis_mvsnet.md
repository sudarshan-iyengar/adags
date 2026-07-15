---
type: paper
node_id: paper:zhang2020_vis_mvsnet
title: "Visibility-aware Multi-view Stereo Network"
authors: ["Jingyang Zhang", "Yao Yao", "Shiwei Li", "Zixin Luo", "Tian Fang"]
year: 2020
venue: "BMVC"
external_ids:
  arxiv: "2008.07928"
  doi: null
  s2: null
tags: ["multi-view-stereo", "visibility", "uncertainty", "occlusion"]
added: 2026-07-14T22:18:30Z
---

# Visibility-aware Multi-view Stereo Network

## One-line thesis

Pairwise matching uncertainty can estimate pixel visibility and prevent occluded source views from corrupting multiview depth fusion.

## Problem / Gap

Naive multiview cost aggregation treats occluded and visible source evidence alike and can become less reliable as views are added.

## Method

Vis-MVSNet jointly predicts pairwise depth and uncertainty, then uses uncertainty to weight pairwise cost volumes before multiview fusion.

## Key Results

The paper reports improved depth accuracy on severe-occlusion scenes across standard MVS benchmarks.

## Assumptions

Calibrated views share a static surface state for the reconstruction instant.

## Limitations / Failure Modes

It suppresses occluded source evidence but does not track temporal reveals, infer persistent hidden surfaces, or allocate dynamic representation capacity.

## Reusable Ingredients

Visibility-weighted view selection, pairwise uncertainty, and an explicit distinction between missing correspondence and usable evidence.

## Open Questions

Can the visibility-weighted fusion principle be extended to dynamic, temporally corresponding N3V observations?

## Claims

None yet.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

It supplies the calibrated multiview visibility principle absent from R031-R033 and warns against indiscriminate camera aggregation.
