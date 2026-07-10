# R036/R037 Visibility-Event Decision Memo

Date: 2026-07-10

## Decision

FAIL for the first training-loop persistent visibility-gate real pilot.

This is a valid scientific negative result, not an infrastructure failure. R034 passed the synthetic sanity fixture, R035 rejected the cheap proxy admission rule, and R036/R037 then ran a matched-budget full-training pilot on the frozen R020 non-oracle candidate field. All six train jobs completed, all six eval renders completed, the strict manifest validation passed, and the final five-window scorer ran once.

## What Was Tried

- `H_smooth` / R036: route0-style smooth transport with the same R020 high-recall non-oracle candidate support for local ROI/capacity pressure, but no visibility opacity gate.
- `H_event` / R037: the same backbone, scenes, candidate field, 6000-iteration budget, and point budget, plus time-dependent visibility opacity gates before rasterization.
- Frozen crop labels were not used to select the support or tune thresholds. They were used only after rendering for the predeclared frozen-window scoring.
- Final scoring started from `r029_r030_disambiguation_manifest.json`, so the comparison includes route0, residual/uncertainty, matched-lifespan, the oracle upper bound, and prior controls.

## Jobs And Artifacts

- R036 train jobs: `49042444`, `49042445`, `49042446`; all `COMPLETED`, exit `0:0`.
- R036 eval jobs: `49045923`, `49045924`, `49045925`; all `COMPLETED`, exit `0:0`.
- R037 train jobs: `49042510`, `49042512`, `49042514`; all `COMPLETED`, exit `0:0`.
- R037 eval jobs: `49051779`, `49051782`, `49051783`; all `COMPLETED`, exit `0:0`.
- Final manifest: `refine-logs/hide_reveal_poc/r036_r037_visibility_event_manifest.json`.
- Manifest validation: `refine-logs/hide_reveal_poc/r036_r037_visibility_event_manifest.validation.json`.
- Metrics: `refine-logs/hide_reveal_poc/r036_r037_visibility_event_real_eval/`.
- Gate summary: `refine-logs/hide_reveal_poc/r036_r037_visibility_event_summary/gate_decision.json`.
- Crop strips: `refine-logs/hide_reveal_poc/r036_r037_visibility_event_summary/crop_strips/`.

## Gate Result

| Metric | route0 | R036 smooth control | R037 visibility event |
| --- | ---: | ---: | ---: |
| Mean PSNR | 30.5021 | 30.2936 | 30.1089 |
| Mean L1 / proxy-LPIPS | 0.0148316 | 0.0144448 | 0.0157600 |
| Mean flicker | 0.00799083 | 0.00805326 | 0.00841330 |
| Mean static ghost | 0.127333 | 0.116426 | 0.165836 |

Strict gate counts:

- R037 route0 PSNR+L1 wins: `0/5`.
- R037 strict all-baseline PSNR+L1 wins: `0/5`.
- R037 static no-worse than route0: `1/5`.
- R037 mean oracle PSNR-gap recovery: `-0.0391`.
- R036 smooth control was mixed (`2/5` strict all-baseline wins, `3/5` static no-worse), but it is not the visibility-event hypothesis and it also has negative mean oracle PSNR-gap recovery.

## Interpretation

The event-gated training-loop mechanism did not recover the oracle crop gap. It worsened the route0 average on the main crop-fidelity metrics and introduced a large static-ghost regression. The smooth control's lower L1 and static ghost show that a matched full-training run can change the metric surface, but the visibility gate itself did not help.

This does not prove that visibility-event modeling is impossible. It does show that this first persistent-opacity-gate implementation, using the R020 non-oracle candidate field and current local-capacity pressure, is not a paper-worthy positive result. The failure is stronger than R035 alone because R037 performed actual checkpoint-backed Gaussian training and rendering, then failed the frozen-window gate.

## Next Scientific Move

Do not tune R037 thresholds or support selection on the five frozen windows. A defensible next iteration would need a new predeclared mechanism, for example:

- a soft visibility/reliability feature in the loss rather than direct opacity attenuation;
- explicit separation of occluder and revealed surface capacity instead of gating all candidate-local Gaussians together;
- a training-time event identity/reconnection objective with multi-view or temporal surface-state evidence;
- or a decision to pivot away from hide/reveal repair and use these results as negative evidence in the broader ADAGS reliability-gated direction.
