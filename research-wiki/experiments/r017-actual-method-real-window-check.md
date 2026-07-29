---
id: experiment:r017-actual-method-real-window-check
date: 2026-07-07
status: failed
related_idea: idea:event-causal-visibility-gaussians
---

# R017 Actual-Method Real-Window Check

R017 tested whether a checkpoint-backed runtime hide/reveal opacity gate could improve the frozen R009 real windows without GT crop compositing.

Artifacts:
- [[../../refine-logs/hide_reveal_poc/r017_actual_method_report|R017 actual-method report]]
- `refine-logs/hide_reveal_poc/r017_actual_real_renders/actual_render_metadata.json`
- `refine-logs/hide_reveal_poc/r017_actual_real_eval/real_event_window_metrics.csv`

Result: FAIL. The actual Gaussian-rendered method passed 0/5 frozen windows. It was worse than route0, matched-lifespan, and residual/uncertainty on PSNR, L1/proxy-LPIPS, flicker, and static ghost no-degradation criteria.

Interpretation: the derived GT-crop-composite R013 result should remain an upper-bound sanity check only. The actual method needs a different implementation idea before paper-scale validation is justified.

## Correction - 2026-07-29

This page understated the strength of the negative. Per
`refine-logs/EVENT_CROP_FIX_EVIDENCE.md` (R017 section), the runtime gate
used the predeclared R009 event crop itself as support and selected tens of
thousands of Gaussians per frame near the event centers. R017 therefore had
oracle image-space localization and still lost 11.1 dB (19.3667 vs route0
30.5021). The correct generalization is: opacity attenuation of all
candidate-local primitives fails even given perfect region localization,
because it is subtractive-only (cannot supply revealed content) and
region-scoped (attenuates occluder and hidden surface together). "Better
support" is not a permitted explanation for this failure. This strengthens
the constraint recorded in [[operations/phase9-csvl-vpl-v2-direction]] (C-1,
C-2, C-3).
