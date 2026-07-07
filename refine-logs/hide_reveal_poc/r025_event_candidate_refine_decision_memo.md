# R025 Event-Candidate Refine Decision Memo

Date: 2026-07-07

## Decision

FAIL for M1 non-oracle residual-component local refinement.

R025 produced valid checkpoint-backed Gaussian-rendered outputs, but it did not produce the event-crop fix on the five frozen R009 real windows.

## What Was Tried

- Non-oracle event candidates were generated from route0/dynamic/mask/flicker diagnostics rather than from the frozen event crops.
- Route0 checkpoints were resumed with candidate-local ROI/exclusion refinement to `chkpnt6200.pth`.
- Three scene eval jobs rendered complete `test/ours_6200` folders:
  - `cut_roasted_beef`
  - `flame_steak`
  - `sear_steak`
- Frozen-window scoring compared `event_candidate_refine` against route0, matched_lifespan, residual_uncertainty, and the derived oracle hide_reveal upper bound.

## Source Artifacts

- Candidate manifest: `refine-logs/hide_reveal_poc/r020_high_recall_motion_supported_nonoracle_candidates/nonoracle_candidate_manifest.json`
- R025 scoring manifest: `refine-logs/hide_reveal_poc/r025_event_candidate_refine_manifest.json`
- R025 scoring output: `refine-logs/hide_reveal_poc/r025_event_candidate_refine_real_eval/`
- R025 summary: `refine-logs/hide_reveal_poc/r025_event_candidate_refine_real_eval/real_event_window_summary.json`
- R025 crop strips: `refine-logs/hide_reveal_poc/r025_event_candidate_refine_summary/crop_strips/`
- Train logs: `logs/event_candidate_refine_train_*_487999*.{out,err}`
- Eval logs: `logs/event_candidate_refine_eval_*_488023*.{out,err}`
- Scoring logs: `logs/hide_reveal_real_48805053.{out,err}`

## Gate Result

| Metric | R025 event_candidate_refine | route0 | Delta vs route0 |
| --- | ---: | ---: | ---: |
| Mean PSNR | 28.9393 | 30.5021 | -1.5629 |
| Mean L1/proxy-LPIPS | 0.0188750 | 0.0148316 | +0.004043 |
| Mean flicker | 0.00847709 | 0.00799083 | +0.000486 |
| Mean static ghost | 0.125652 | 0.127333 | -0.001681 |

Strict gate counts:
- 0/5 windows improved versus route0, matched_lifespan, and residual_uncertainty on both PSNR and L1/proxy-LPIPS.
- 0/5 windows improved versus route0 on both PSNR and L1/proxy-LPIPS.
- 2/5 windows were no worse than route0 on static ghost, below the 3/5 requirement.
- PSNR oracle recovery fraction: -0.1394.
- L1 oracle recovery fraction: -0.3323.

## Interpretation

The method-form constraint was satisfied: R025 used checkpoint-backed Gaussian renders rather than GT crop compositing, and the method support was non-oracle. The scientific gate still fails decisively. M1 worsened the crop reconstruction metrics while slightly reducing mean static ghost, which is not enough to support the event-crop-fix claim.

Independent result-to-claim review returned `claim_supported: no` with high confidence.

## Next Step

Do not run paper-scale validation or positive ablations for M1. The next scientifically meaningful step is to change mechanism, not tune this method on the frozen windows. The most defensible follow-up is M2 occlusion-boundary gated micro-densification or a decomposition experiment that separates candidate support quality from local-refinement damage.
