---
type: paper
node_id: paper:zhang2024_monst3r
title: "MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion"
authors: ["Junyi Zhang", "Charles Herrmann", "Junhwa Hur", "Varun Jampani", "Trevor Darrell", "Forrester Cole", "Deqing Sun", "Ming-Hsuan Yang"]
year: 2024
venue: "ICLR 2025"
external_ids:
  arxiv: "2410.03825"
  doi: null
  s2: null
tags: ["dynamic-geometry", "pointmaps", "video-depth", "camera-pose"]
added: 2026-07-14T22:18:30Z
---

# MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion

## One-line thesis

Per-timestep pointmaps provide a geometry-first representation for videos containing moving and deforming content.

## Problem / Gap

Static multiview geometry can break when image pairs contain scene motion, while independent monocular depth lacks common geometry.

## Method

MonST3R fine-tunes a DUSt3R-style model for per-timestep pointmaps and adds video-oriented global optimization to align dynamic geometry and camera motion.

## Key Results

The paper reports robust video depth and camera pose estimation and promising feed-forward 4D reconstruction.

## Assumptions

Dynamic geometry training data and sufficient static or correspondable content are available for alignment.

## Limitations / Failure Modes

It does not explicitly label foreground/background visibility order or decide how a Gaussian representation should preserve hidden content.

## Reusable Ingredients

Dynamic pointmaps, geometry-first temporal alignment, and separation of camera motion from scene motion.

## Open Questions

Can calibrated N3V cameras make the alignment simpler and produce reliable surface visibility states?

## Claims

None yet.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

It is a direct alternative to R031's independently normalized, same-pixel temporal depth comparison.
