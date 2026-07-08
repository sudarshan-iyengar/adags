# Event-Crop Fix Working Memory

Updated: 2026-07-07

This page tracks the event-crop fix objective after R001-R027.

## Core Evidence

R013/R015 showed a strong upper bound: replacing the frozen event crops with GT pixels sharply improves the frozen real-window metrics. This proves the crop regions are meaningful but does not prove the Gaussian model can create the fix.

R017 tested an actual checkpoint-backed Gaussian renderer path with a runtime opacity gate. It did not use GT pixels, but it failed all five frozen R009 windows and worsened PSNR, L1/proxy-LPIPS, flicker, and static ghost.

The current state is therefore:

- Oracle event-crop upper bound: positive and large.
- Actual Gaussian opacity gate: negative.
- Non-oracle residual-component local refinement: negative.
- Non-oracle Gaussian method: unsolved; M2 occlusion-boundary gated micro-densification produced valid checkpoint-backed renders but failed the frozen-window recovery gate.
- Posthoc support diagnosis: R026 boundary support missed the frozen crop regions almost entirely, so R027 is a failure of the tested support+training recipe rather than a clean rejection of good-support event-local densification.

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
- R026 M2 support manifest: `refine-logs/hide_reveal_poc/r026_m2_boundary_support/event_boundary_support_manifest.json`
- R026 M2 support validation: `refine-logs/hide_reveal_poc/r026_m2_boundary_support/event_boundary_support_validation.json`
- R027 train manifest: `refine-logs/event_boundary_micro_densify_train_jobs_20260707_234937.tsv`
- R027 metrics: `refine-logs/hide_reveal_poc/r027_event_boundary_micro_densify_real_eval/real_event_window_summary.json`
- R027 gate summary: `refine-logs/hide_reveal_poc/r027_event_boundary_micro_densify_summary/gate_decision.json`
- R027 decision memo: `refine-logs/hide_reveal_poc/r027_event_boundary_micro_densify_decision_memo.md`
- R028 interpretation and next plan: `refine-logs/RESULT_INTERPRETATION_AND_NEXT_PLAN.md`
- R028 support-overlap diagnostics: `refine-logs/hide_reveal_poc/r028_support_overlap_diagnostics/`
- Wiki experiment page: `research-wiki/experiments/r017-actual-method-real-window-check.md`
- R025 wiki experiment page: `research-wiki/experiments/r025-event-candidate-refine-real-window-check.md`
- R027 wiki experiment page: `research-wiki/experiments/r027-event-boundary-micro-densify-real-window-check.md`
- R028 wiki experiment page: `research-wiki/experiments/r028-support-overlap-diagnostics.md`

## Baseline Means

| System | PSNR up | L1/proxy-LPIPS down | Flicker down | Static ghost down |
| --- | ---: | ---: | ---: | ---: |
| route0 | 30.5021 | 0.0148316 | 0.00799083 | 0.127333 |
| matched_lifespan | 29.8181 | 0.0163546 | 0.00795601 | 0.127333 |
| residual_uncertainty | 30.0734 | 0.0165723 | 0.00803902 | 0.145702 |
| derived oracle hide_reveal | 41.7149 | 0.00266536 | 0.00168586 | 0.127333 |
| R017 actual opacity gate | 19.3667 | 0.0761056 | 0.0162899 | 0.152789 |
| R025 event_candidate_refine | 28.9393 | 0.0188750 | 0.00847709 | 0.125652 |
| R027 event_boundary_micro_densify | 30.5591 | 0.0147412 | 0.00801033 | 0.125634 |

## Method Constraint

Future attempts must use non-oracle event-support discovery. Acceptable cues include training residuals, dynamic masks, flow disagreement, visibility/occlusion boundaries, uncertainty, and learned or deterministic candidate maps computed without access to the frozen evaluation crop labels.

The method must produce Gaussian-rendered output folders. GT crop compositing remains an upper bound only.

## M2 Outcome

R026b generated non-oracle support masks from dynamic-mask boundaries, flow sidecars, and route0 render diagnostics. Validation passed with `uses_gt_residual=false`, `uses_gt_crop_pixels=false`, `uses_frozen_window_labels=false`, 66 support frames, 108 selected components, and max support fraction `0.005205`.

R027 trained and evaluated checkpoint-backed `event_boundary_micro_densify` renders. It failed: strict all-baseline PSNR+L1 wins were `2/5`, mean PSNR improved over route0 by only `+0.0569 dB`, mean L1 improved by only `-0.0000903`, and oracle recovery fractions were PSNR `0.00508` and L1 `0.00743`. Independent result-to-claim review judged `claim_supported: no` with high confidence.

R028 posthoc support-overlap audit changed the interpretation: R026 boundary masks had mean support-frame fraction `0.0250` and mean crop coverage `0.000000` on the frozen windows. Thus R027 remains a FAIL for the tested M2 recipe, but is not decisive against an event-local densification method with genuinely aligned support. R020 M1 candidate boxes had much better support overlap (`0.6375` mean support-frame fraction, `0.491371` mean crop coverage), so R025 is stronger negative evidence against the current posthoc local-refinement machinery.

Next compact diagnostics are R029 route0 continuation control and R030 oracle-support Gaussian-only refinement. Training completed for both diagnostics on 2026-07-08, but eval/scoring is blocked by Slurm account/partition submission acceptance. R030 is explicitly diagnostic and cannot support a non-oracle claim because it uses the frozen crop windows as support.

## Wiki Links

- [[ideas/event-causal-visibility-gaussians]]
- [[experiments/r017-actual-method-real-window-check]]
- [[experiments/r025-event-candidate-refine-real-window-check]]
- [[experiments/r027-event-boundary-micro-densify-real-window-check]]
- [[experiments/r028-support-overlap-diagnostics]]
- [[experiments/r029-r030-disambiguation-wave]]
- [[ideas/depth-occlusion-event-support]]
- [[gap_map#G13 - Visibility Events Are Not Smooth Deformation]]
- [[papers/sandu2026_temporally_aware_densification]]
- [[papers/ramlal2026_persistgs]]
- [[papers/zhang2026_vad_gs]]
