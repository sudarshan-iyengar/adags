# R027 Event-Boundary Micro-Densify Decision Memo

Date: 2026-07-08

## Decision

FAIL for M2 non-oracle occlusion-boundary gated micro-densification.

R027 produced valid checkpoint-backed Gaussian renders from route0 `chkpnt6000.pth` resumes to `chkpnt6400.pth`, using the R026 non-oracle boundary-support artifact. It did not pass the predeclared event-crop-fix gate on the five frozen R009 real windows.

## What Was Tried

- R026b generated deterministic non-oracle support masks from dynamic-mask boundaries, flow sidecars, and route0 render diagnostics.
- R026b guardrails: `uses_gt_residual=false`, `uses_gt_crop_pixels=false`, `uses_frozen_window_labels=false`.
- Three scene train jobs resumed route0 checkpoints to `chkpnt6400.pth` with the M2 support mask and point cap `625000`.
- Three eval jobs rendered complete `test/ours_6400` folders for `cut_roasted_beef`, `flame_steak`, and `sear_steak`.
- Frozen-window scoring compared `event_boundary_micro_densify` against route0, matched_lifespan, residual_uncertainty, and the derived oracle hide_reveal upper bound.

## Source Artifacts

- R026 support manifest: `refine-logs/hide_reveal_poc/r026_m2_boundary_support/event_boundary_support_manifest.json`
- R027 scoring manifest: `refine-logs/hide_reveal_poc/r027_event_boundary_micro_densify_manifest.json`
- R027 manifest validation: `refine-logs/hide_reveal_poc/r027_event_boundary_micro_densify_manifest_validation.json`
- R027 scoring output: `refine-logs/hide_reveal_poc/r027_event_boundary_micro_densify_real_eval/`
- R027 gate summary: `refine-logs/hide_reveal_poc/r027_event_boundary_micro_densify_summary/gate_decision.json`
- R027 crop strips: `refine-logs/hide_reveal_poc/r027_event_boundary_micro_densify_summary/crop_strips/`
- Train logs: `logs/event_boundary_micro_densify_train_*_488732*.{out,err}`
- Eval logs: `logs/event_boundary_micro_densify_eval_*_488741*.{out,err}`
- Scoring logs: `logs/hide_reveal_real_48874592.{out,err}`

## Gate Result

| Metric | R027 event_boundary_micro_densify | route0 | Delta vs route0 |
| --- | ---: | ---: | ---: |
| Mean PSNR | 30.5591 | 30.5021 | +0.0569 |
| Mean L1/proxy-LPIPS | 0.0147412 | 0.0148316 | -0.0000903 |
| Mean flicker | 0.00801033 | 0.00799083 | +0.0000195 |
| Mean static ghost | 0.125634 | 0.127333 | -0.001698 |

Strict gate counts:
- 2/5 windows improved versus route0, matched_lifespan, and residual_uncertainty on both PSNR and L1/proxy-LPIPS.
- 3/5 windows improved versus route0 on both PSNR and L1/proxy-LPIPS.
- 3/5 windows were no worse than route0 on static ghost.
- Mean PSNR gain was `+0.0569 dB`, below the required `+0.5 dB`.
- Mean L1 improvement was `-0.0000903`, below the required `-0.001`.
- Oracle recovery fractions were PSNR `0.00508` and L1 `0.00743`, below the required `0.25`.

## Interpretation

M2 is a valid non-oracle method test and is much less destructive than M1, but the effect is too small and too inconsistent to support the event-crop-fix claim. The method barely improves mean PSNR/L1 over route0, slightly worsens mean flicker, and passes only 2/5 strict all-baseline PSNR+L1 windows.

The result should be recorded as negative evidence for boundary-gated micro-densification under this support and budget, not as a near-positive result. The clearest lesson is that local support quality and micro-densification alone are not sufficient to recover the oracle crop gap.

## Next Step

Do not run paper-scale validation or positive ablations for M2. A defensible next phase would need a different mechanism or a diagnostic decomposition, such as checking whether support aligns with the actual failure pixels without using that alignment for training, then separating support selection failure from renderer/optimization failure.
