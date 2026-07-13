# Selected Problem-First Direction

Date: 2026-06-30

## Primary High-Upside Direction

**Event-Causal Visibility Gaussians**

The strongest problem-first bet is to make visibility events first-class in dynamic Gaussian reconstruction:

> Occlusion, disocclusion, birth, split, merge, and retirement should not be forced through smooth deformation everywhere.

## Method Shape

- Keep persistent identity carriers for normal smooth transport.
- Add differentiable event states for visibility and primitive lifecycle changes.
- Use residual, rendered visibility, track/flow disagreement, and geometry-consistency cues to propose event windows.
- Allow transient detail or new geometry only when an event justifies it.
- Require every event-created detail carrier to reconcile, promote, or retire.

## Why This Beats The ADAGS-First Direction

Reliability-gated ADAGS priors are feasible, but they mostly answer: "how can this code trust priors better?"

Event-causal visibility answers a field-level question: "when is a dynamic scene change deformation, and when is it new visibility or new geometry?"

## Minimum Decisive Evidence

Compare against RiGS, Multi4D, SharpTimeGS, MAPo, and a strong ADAGS route0/reliability baseline on occlusion-heavy synthetic and real clips.

Required metrics:

- dynamic-boundary PSNR/LPIPS,
- flicker and ghosting around occlusion/disocclusion,
- static ghost score,
- track identity switches,
- event birth/death precision on synthetic cases,
- matched primitive/point budget,
- qualitative disocclusion crops.

## Kill Condition

Stop or demote this idea if SharpTimeGS, RiGS, Multi4D, or Ground4D already handles the same occlusion/disocclusion failures at matched budget without explicit event states.

## Secondary SOTA Seeds

- [[research-wiki/ideas/identity-conserving-detail-carriers]]
- [[research-wiki/ideas/frequency-adaptive-temporal-support]]
- [[research-wiki/ideas/counterfactual-prior-usefulness-routing]]

## Safe ADAGS Fallback

**Self-Calibrated Reliability-Gated Priors**

Keep the previous reliability direction as the conservative track and as prototype infrastructure. It becomes primary only if the high-upside representation ideas fail novelty or feasibility checks.

