---
type: idea
node_id: idea:event-causal-visibility-gaussians
stage: proposed
outcome: mixed
tags: [problem-first-redo, high-upside, visibility, occlusion, dynamic-gs, representation]
target_gaps: [G1, G2, G6, G9, G11, G13]
---

# Event-Causal Visibility Gaussians

## Thesis

Dynamic Gaussian Splatting should first solve the narrower hide/reveal problem: when a primitive is temporarily occluded, the model should decide whether to keep the same identity hidden and reveal it later, rather than forcing visible smooth deformation or deleting/recreating capacity.

## Method Shape

- Keep ADAGS route0 LoRA or another dynamic-GS backbone as the smooth-transport base.
- Propose hide/reveal candidates from fixed residual, boundary, visibility, flow-disagreement, and track-toggle cues.
- Compare `H_smooth` and `H_event` with a normalized counterfactual shadow score on the same crop/time support.
- Commit hide/reveal only when the event hypothesis beats smooth visible transport under a frozen synthetic-calibrated margin.
- Treat birth/retire, learned prioritization, geometry scoring, and event-local densification as future or appendix extensions, not the first paper's main method.

## Why It Matters

Occlusion/disocclusion is a representation-level failure mode. If handled as smooth visible deformation or generic lifespan modulation, methods blur, ghost, flicker, or fragment primitive identity.

## Minimum Decisive Experiment

First test whether the normalized event margin separates synthetic hide/reveal events from normal motion. Then compare real occlusion-heavy event windows against ADAGS route0, matched lifespan-only, residual/uncertainty gating, and the strongest available dynamic-GS comparator. Measure margin AUC, candidate recall, hide/reveal precision/recall, identity reconnection, event-window LPIPS/PSNR, flicker, static ghosts, and confident-track identity switches.

## Closest Prior Work

[[papers/wang2026_multi4d]], [[papers/wu2026_rigs]], [[papers/liao2026_sharptimegs]], [[papers/jiao2026_mapo]], [[papers/zhao2026_ground4d]], [[papers/sandu2026_temporally_aware_densification]], [[papers/ramlal2026_persistgs]], [[papers/zhang2026_vad_gs]]

## Kill Conditions

- Matched lifespan-only gets the same candidate support, hysteresis, local refinement, and budget and still matches identity reconnection plus event-window artifact metrics.
- The normalized margin does not separate true synthetic hide/reveal from normal motion.
- Deterministic candidates miss true hide/reveal events under the fixed candidate budget.
- Real improvements come only from residual/uncertainty gating rather than the counterfactual test.

## Failure / Risk Notes

- R017 actual checkpoint-backed opacity gating failed all five frozen R009 windows and degraded all mean event-window metrics.
- R025 M1 non-oracle residual-component local refinement produced checkpoint-backed Gaussian renders without oracle crop support, but failed the strict gate: 0/5 windows improved versus route0 on both PSNR and L1/proxy-LPIPS, mean PSNR dropped by 1.5629 dB, and mean L1 worsened by 0.004043.
- R027 M2 non-oracle occlusion-boundary gated micro-densification was a valid checkpoint-backed test and was less damaging than M1, but still failed: only 2/5 strict all-baseline PSNR+L1 windows, mean PSNR improved by only 0.0569 dB, mean L1 by only 0.0000903, and oracle recovery stayed below 1%.
- R036/R037 tested the first full training-loop persistent opacity-gate real pilot. The matched smooth control completed, but the event-gated method failed the strict gate: 0/5 route0 PSNR+L1 wins, 0/5 strict all-baseline wins, static no-worse only 1/5, and mean oracle PSNR-gap recovery -0.0391. This rejects the current fixed opacity-gate form, not all visibility-event modeling.
- The positive R013/R015 event-crop result should remain an oracle upper bound until a non-oracle Gaussian method recovers a positive fraction of it.

## ADAGS Feasibility

Prototype first on ADAGS route0 LoRA with no new trainable components. Add only a multiplicative opacity gate for committed hide/reveal events and use scheduler jobs for synthetic margin tests plus real event-window comparisons.
