---
type: paper
node_id: paper:rai2026_packuv
title: "PackUV: Packed Gaussian UV Maps for 4D Volumetric Video"
authors: ["Aashish Rai", "Angela Xing", "Anushka Agarwal", "Xiaoyan Cong", "Zekun Li", "Tao Lu", "Aayush Prakash", "Srinath Sridhar"]
year: 2026
venue: "CVPR"
external_ids:
  arxiv: "2602.23040"
  doi: null
  s2: null
tags: ["4d-gaussians", "disocclusion", "temporal-consistency", "uv-representation"]
added: 2026-07-14T23:36:29Z
---

# PackUV: Packed Gaussian UV Maps for 4D Volumetric Video

## One-line thesis

Structured multi-layer Gaussian UV atlases, flow-guided keyframes, and dynamic-Gaussian labeling provide temporally coherent long-sequence fitting through large motion and disocclusion.

## Problem / Gap

Independent or unstructured per-frame 3DGS fitting drifts over long volumetric videos, handles disocclusions poorly, and is difficult to encode with standard video infrastructure.

## Method

PackUV-GS directly optimizes Gaussian attributes in layered UV maps. It initializes each frame from its predecessor, promotes high-flow, occlusion/disocclusion, or appearance-break frames to keyframes, labels dynamic Gaussians through multiview projected flow masks, freezes static gradients, and applies UV-aware density control.

## Key Results

The CVPR 2026 paper reports temporally consistent reconstruction for sequences up to 30 minutes and introduces a synchronized 50-plus-camera dataset with frequent disocclusions.

## Assumptions

Dense synchronized multiview coverage, reliable flow-based motion masks, and a per-frame atlas sequence are available. Child Gaussians inherit their parent's dynamic/static label.

## Limitations / Failure Modes

Keyframing reacts to motion or breaks rather than inferring foreground/background order and hidden-surface identity. Sequential initialization can propagate errors, and the representation uses a different dense-capture/storage regime from ADAGS.

## Reusable Ingredients

Disocclusion-sensitive keyframes, multiview projection of flow support, static preservation, layered surface storage, and explicit long-horizon temporal diagnostics.

## Open Questions

Can sparse calibrated views identify which rear surface should receive reassigned capacity rather than merely starting a new keyframe?

## Claims

None yet.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

PackUV-GS prevents any broad claim that temporal consistency or disocclusion handling is absent from Gaussian representations. ADAGS must test a narrower surface-order-to-budget-reassignment hypothesis.
