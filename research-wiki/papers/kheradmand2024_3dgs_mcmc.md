---
type: paper
node_id: paper:kheradmand2024_3dgs_mcmc
title: "3D Gaussian Splatting as Markov Chain Monte Carlo"
authors: ["Shakiba Kheradmand", "et al."]
year: 2024
venue: "NeurIPS"
external_ids:
  arxiv: "2404.09591"
tags: [static-gs, densification, fixed-budget, relocation]
status: reference
---

# 3D Gaussian Splatting as Markov Chain Monte Carlo

**Paper:** https://arxiv.org/abs/2404.09591
**Code:** public (ubc-vision/3dgs-mcmc)
**Evidence tier:** well-known method; recorded here as provenance for the
budget-neutral relocation pattern, not deep-dived in this wiki.

## One-line thesis

Reformulates densification as MCMC sampling under a fixed Gaussian budget:
"dead" (low-opacity) Gaussians are relocated to high-opacity regions instead
of being pruned while new ones are cloned — a point-count-preserving
relocation operator with noise-driven exploration.

## Relevance to ADAGS

Static-scene precedent for budget-neutral reassignment/relocation. Together
with SharpTimeGS's fixed-count stage-2 densification
([[papers/liao2026_sharptimegs]]), this blocks any claim that point-neutral
reassignment is itself a contribution: the v13/B01 transaction substrate is
implementation infrastructure and a control, and papers must cite this
lineage. Recorded as part of the CSVL-VPL v2 borrowed-mechanism ledger
([[operations/phase9-csvl-vpl-v2-direction]]).

## Connections

- Provenance for the B01 transaction substrate role in
  [[operations/phase9-slice-b-v13-b01-decision]]
