---
id: experiment:r025-event-candidate-refine-real-window-check
date: 2026-07-07
status: failed
related_idea: idea:event-causal-visibility-gaussians
---

# R025 Event-Candidate Refine Real-Window Check

R025 tested whether M1, a non-oracle residual-component local-refinement method, could produce the event-crop fix suggested by R001-R017 on the five frozen R009 real windows. The method used non-oracle candidate supports from route0/dynamic/mask/flicker diagnostics, resumed route0 checkpoints, locally refined, and rendered checkpoint-backed Gaussian output folders.

Artifacts:
- `refine-logs/hide_reveal_poc/r025_event_candidate_refine_manifest.json`
- `refine-logs/hide_reveal_poc/r025_event_candidate_refine_real_eval/real_event_window_summary.json`
- `refine-logs/hide_reveal_poc/r025_event_candidate_refine_real_eval/`
- `refine-logs/hide_reveal_poc/r025_event_candidate_refine_decision_memo.md`
- `refine-logs/hide_reveal_poc/r025_event_candidate_refine_summary/crop_strips/`

Result: FAIL. R025 passed 0/5 windows on the strict PSNR-and-L1 improvement gate versus route0, matched_lifespan, and residual_uncertainty. It also passed 0/5 versus route0 alone on both PSNR and L1/proxy-LPIPS.

Key means:

| System | PSNR up | L1/proxy-LPIPS down | Flicker down | Static ghost down |
| --- | ---: | ---: | ---: | ---: |
| R025 event_candidate_refine | 28.9393 | 0.0188750 | 0.00847709 | 0.125652 |
| route0 | 30.5021 | 0.0148316 | 0.00799083 | 0.127333 |
| matched_lifespan | 29.8181 | 0.0163546 | n/a | n/a |
| residual_uncertainty | 30.0734 | 0.0165723 | n/a | n/a |
| oracle hide_reveal upper bound | 41.7149 | 0.00266536 | 0.00168586 | 0.127333 |

Against route0, R025 changed mean PSNR by -1.5629 dB and mean L1/proxy-LPIPS by +0.004043. It recovered a negative fraction of the oracle upper bound on both PSNR and L1. Static ghost improved slightly on average, but only 2/5 windows were no worse than route0, below the 3/5 gate.

Interpretation: this is valid negative evidence for M1. The run appears to satisfy the checkpoint-backed/non-oracle render requirement, but the quantitative gate rejects the claim that this non-oracle Gaussian method can reproduce the event-crop fix on the frozen windows. The oracle crop result remains an upper bound, not a demonstrated model capability.
