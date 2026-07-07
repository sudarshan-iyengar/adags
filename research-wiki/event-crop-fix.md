# Event-Crop Fix Working Memory

Updated: 2026-07-07

This page tracks the event-crop fix objective after R001-R025.

## Core Evidence

R013/R015 showed a strong upper bound: replacing the frozen event crops with GT pixels sharply improves the frozen real-window metrics. This proves the crop regions are meaningful but does not prove the Gaussian model can create the fix.

R017 tested an actual checkpoint-backed Gaussian renderer path with a runtime opacity gate. It did not use GT pixels, but it failed all five frozen R009 windows and worsened PSNR, L1/proxy-LPIPS, flicker, and static ghost.

The current state is therefore:

- Oracle event-crop upper bound: positive and large.
- Actual Gaussian opacity gate: negative.
- Non-oracle residual-component local refinement: negative.
- Non-oracle Gaussian method: unsolved.

## Frozen Evaluation Windows

Source manifest: `refine-logs/hide_reveal_real_windows.json`

- `cut_roasted_beef_hand_tongs_meat_095_110`
- `cut_roasted_beef_hand_knife_meat_140_155`
- `flame_steak_torch_pan_155_170`
- `flame_steak_torch_sweep_195_210`
- `sear_steak_spoon_pan_220_235`

These windows are evaluation-only for future non-oracle methods. The method must not consume these event crops as test-time support.

## Key Artifacts

- Evidence summary: `refine-logs/EVENT_CROP_FIX_EVIDENCE.md`
- Method tracker: `refine-logs/EVENT_CROP_METHOD_TRACKER.md`
- R015 decision bundle: `refine-logs/hide_reveal_poc/r015_poc_summary/poc_decision_inputs.json`
- R017 report: `refine-logs/hide_reveal_poc/r017_actual_method_report.md`
- R017 metrics: `refine-logs/hide_reveal_poc/r017_actual_real_eval/real_event_window_metrics.csv`
- R017 renderer metadata: `refine-logs/hide_reveal_poc/r017_actual_real_renders/actual_render_metadata.json`
- R025 manifest: `refine-logs/hide_reveal_poc/r025_event_candidate_refine_manifest.json`
- R025 metrics: `refine-logs/hide_reveal_poc/r025_event_candidate_refine_real_eval/real_event_window_summary.json`
- R025 decision memo: `refine-logs/hide_reveal_poc/r025_event_candidate_refine_decision_memo.md`
- R025 qualitative strips: `refine-logs/hide_reveal_poc/r025_event_candidate_refine_summary/crop_strips/`
- Wiki experiment page: `research-wiki/experiments/r017-actual-method-real-window-check.md`
- R025 wiki experiment page: `research-wiki/experiments/r025-event-candidate-refine-real-window-check.md`

## Baseline Means

| System | PSNR up | L1/proxy-LPIPS down | Flicker down | Static ghost down |
| --- | ---: | ---: | ---: | ---: |
| route0 | 30.5021 | 0.0148316 | 0.00799083 | 0.127333 |
| matched_lifespan | 29.8181 | 0.0163546 | 0.00795601 | 0.127333 |
| residual_uncertainty | 30.0734 | 0.0165723 | 0.00803902 | 0.145702 |
| derived oracle hide_reveal | 41.7149 | 0.00266536 | 0.00168586 | 0.127333 |
| R017 actual opacity gate | 19.3667 | 0.0761056 | 0.0162899 | 0.152789 |
| R025 event_candidate_refine | 28.9393 | 0.0188750 | 0.00847709 | 0.125652 |

## Method Constraint

Future attempts must use non-oracle event-support discovery. Acceptable cues include training residuals, dynamic masks, flow disagreement, visibility/occlusion boundaries, uncertainty, and learned or deterministic candidate maps computed without access to the frozen evaluation crop labels.

The method must produce Gaussian-rendered output folders. GT crop compositing remains an upper bound only.

## Wiki Links

- [[ideas/event-causal-visibility-gaussians]]
- [[experiments/r017-actual-method-real-window-check]]
- [[experiments/r025-event-candidate-refine-real-window-check]]
- [[gap_map#G13 - Visibility Events Are Not Smooth Deformation]]
- [[papers/sandu2026_temporally_aware_densification]]
- [[papers/ramlal2026_persistgs]]
- [[papers/zhang2026_vad_gs]]
