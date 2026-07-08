---
id: experiment:r027-event-boundary-micro-densify-real-window-check
date: 2026-07-08
status: failed
related_idea: idea:event-causal-visibility-gaussians
---

# R027 Event-Boundary Micro-Densify Real-Window Check

R027 tested whether M2, a non-oracle occlusion-boundary gated micro-densification method, could recover the event-crop fix on the five frozen R009 real windows. R026b first generated support masks from dynamic-mask boundaries, flow sidecars, and route0 render diagnostics, with guardrails `uses_gt_residual=false`, `uses_gt_crop_pixels=false`, and `uses_frozen_window_labels=false`. R027 then resumed route0 checkpoints from `chkpnt6000.pth` to `chkpnt6400.pth` and rendered checkpoint-backed Gaussian output folders.

Artifacts:
- `refine-logs/hide_reveal_poc/r026_m2_boundary_support/event_boundary_support_manifest.json`
- `refine-logs/event_boundary_micro_densify_train_jobs_20260707_234937.tsv`
- `refine-logs/event_boundary_micro_densify_eval_jobs_20260708_001303.tsv`
- `refine-logs/hide_reveal_poc/r027_event_boundary_micro_densify_manifest.json`
- `refine-logs/hide_reveal_poc/r027_event_boundary_micro_densify_real_eval/real_event_window_summary.json`
- `refine-logs/hide_reveal_poc/r027_event_boundary_micro_densify_summary/gate_decision.json`
- `refine-logs/hide_reveal_poc/r027_event_boundary_micro_densify_decision_memo.md`
- `refine-logs/hide_reveal_poc/r027_event_boundary_micro_densify_summary/crop_strips/`

Result: FAIL. R027 passed 2/5 windows on the strict PSNR-and-L1 improvement gate versus route0, matched_lifespan, and residual_uncertainty. It passed 3/5 versus route0 alone on both PSNR and L1/proxy-LPIPS, and 3/5 windows were no worse on static ghost.

Key means:

| System | PSNR up | L1/proxy-LPIPS down | Flicker down | Static ghost down |
| --- | ---: | ---: | ---: | ---: |
| R027 event_boundary_micro_densify | 30.5591 | 0.0147412 | 0.00801033 | 0.125634 |
| route0 | 30.5021 | 0.0148316 | 0.00799083 | 0.127333 |
| matched_lifespan | 29.8181 | 0.0163546 | 0.00795601 | 0.127333 |
| residual_uncertainty | 30.0734 | 0.0165723 | 0.00803902 | 0.145702 |
| oracle hide_reveal upper bound | 41.7149 | 0.00266536 | 0.00168586 | 0.127333 |

Against route0, R027 changed mean PSNR by `+0.0569 dB`, mean L1/proxy-LPIPS by `-0.0000903`, mean flicker by `+0.0000195`, and mean static ghost by `-0.001698`. Oracle recovery fractions were PSNR `0.00508` and L1 `0.00743`, far below the required `0.25`.

Interpretation: this is valid negative evidence for M2. The run satisfies the checkpoint-backed and non-oracle method-form requirements, and it is much less damaging than M1, but the effect is too small and inconsistent to support event-crop recovery. Boundary support plus micro-densification under this cap is insufficient to recover the oracle gap.
