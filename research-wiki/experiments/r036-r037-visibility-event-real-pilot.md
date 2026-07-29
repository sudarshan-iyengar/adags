---
type: experiment
id: r036-r037-visibility-event-real-pilot
status: failed
date: 2026-07-10
related_idea: event-causal-visibility-gaussians
---

# R036/R037 Visibility-Event Real Pilot

Purpose: test whether a mechanism-changing training-loop visibility event method can recover the hide/reveal event-crop upper bound without using frozen crop labels as method input.

Setup:

- R034 synthetic fixture passed, so one real pilot was justified.
- R035 proxy admission rejected all R020 candidates, so the real pilot used the fixed R020 high-recall non-oracle candidate field directly rather than weakening the margin.
- R036 `H_smooth`: matched 6000-iteration smooth control with the same R020 candidate field for local ROI/capacity pressure but no visibility opacity gate.
- R037 `H_event`: same backbone, data, budget, and candidate field, plus time-dependent opacity visibility gates before rasterization.
- Final scoring used the strict `r029_r030_disambiguation_manifest.json` baseline carrier so the comparison included route0, residual/uncertainty, matched-lifespan, prior controls, and the oracle `hide_reveal` upper bound.

Artifacts:

- `refine-logs/visibility_event_smooth_train_jobs_20260710_024626.tsv`
- `refine-logs/visibility_event_smooth_eval_jobs_20260710_060900.tsv`
- `refine-logs/visibility_event_train_train_jobs_20260710_024651.tsv`
- `refine-logs/visibility_event_train_eval_jobs_20260710_073836.tsv`
- `refine-logs/hide_reveal_poc/r036_r037_visibility_event_manifest.json`
- `refine-logs/hide_reveal_poc/r036_r037_visibility_event_manifest.validation.json`
- `refine-logs/hide_reveal_poc/r036_r037_visibility_event_real_eval/`
- `refine-logs/hide_reveal_poc/r036_r037_visibility_event_summary/gate_decision.json`
- `refine-logs/hide_reveal_poc/r036_r037_visibility_event_decision_memo.md`
- `refine-logs/hide_reveal_poc/r036_r037_visibility_event_summary/experiment_audit.md`

Result: FAIL. R037 `visibility_event_train` completed checkpoint-backed Gaussian training and rendering but failed the predeclared frozen-window gate.

Key means:

| System | PSNR up | L1/proxy-LPIPS down | Flicker down | Static ghost down |
| --- | ---: | ---: | ---: | ---: |
| route0 | 30.5021 | 0.0148316 | 0.00799083 | 0.127333 |
| R036 smooth control | 30.2936 | 0.0144448 | 0.00805326 | 0.116426 |
| R037 visibility event | 30.1089 | 0.0157600 | 0.00841330 | 0.165836 |
| oracle hide_reveal upper bound | 41.7149 | 0.00266536 | 0.00168586 | 0.127333 |

Gate counts:

- R037 route0 PSNR+L1 wins: `0/5`.
- R037 strict all-baseline PSNR+L1 wins: `0/5`.
- R037 static no-worse than route0: `1/5`.
- R037 mean oracle PSNR-gap recovery: `-0.0391`.

Interpretation: this invalidates the current fixed opacity-gate training-loop form of the idea. It does not prove visibility-event modeling impossible, but the current non-oracle candidate-local opacity attenuation worsens crop fidelity and static ghosting. Future work should change the mechanism rather than retune R037 on the frozen windows.

## Correction - 2026-07-29

Two material facts were missing from this page:

1. R035 admission rejected 0 of 72 candidates with mean delta score
   `+0.200982` (`refine-logs/hide_reveal_poc/r035_visibility_event_admission/
   visibility_event_admission_report.md`) — on the normal-motion side of the
   R034 synthetic separation (true events `-0.566`, normal `+0.394`). The
   counterfactual event margin, the idea's core discriminator, had
   effectively zero separation on real data: Kill Condition 2 of
   [[ideas/event-causal-visibility-gaussians]] fired before the pilot ran.
   Running R037 on the candidate set the method's own admission had rejected
   was a process failure; a method with abstention that is overridden is no
   longer the method.
2. R034's perfect synthetic result (AUC 1.0, n=120) predicted nothing about
   real admission. Synthetic-fixture passage must never be a Go criterion —
   a trap repeated by the Stage 1 fixtures
   ([[operations/phase9-csvl-vpl-stage1-result]]).

The +30% static-ghost regression (0.127333 to 0.165836) is the mechanistic
fingerprint of attenuating dynamic primitives over an intact static branch.
