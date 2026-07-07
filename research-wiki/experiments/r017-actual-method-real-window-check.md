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
