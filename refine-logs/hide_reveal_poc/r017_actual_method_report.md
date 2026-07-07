# R017 Actual-Method Real-Window Check

Status: FAIL

R017 rendered `actual_hide_reveal` from existing route0 Gaussian checkpoints using the runtime opacity gate in the Gaussian renderer. It did not use GT crop compositing: `actual_render_metadata.json` records `is_checkpoint_backed_inference=true`, `uses_gaussian_renderer=true`, `uses_gt_pixels_in_render=false`, and `newly_trained_checkpoint=false`.

The frozen R009 windows, synthetic thresholds, route0 outputs, matched-lifespan outputs, and residual/uncertainty outputs were unchanged. The actual render manifest validates with 5/5 windows OK and no warnings.

## Jobs

- Render job: Slurm `48760029`; produced actual render manifest/metadata/validation, then failed only during optional LPIPS download because the compute node had no outbound network.
- Eval job: Slurm `48760448`; completed successfully with LPIPS disabled.

## Gate Result

Pass rule: a majority of the five frozen windows must improve against route0, matched-lifespan, and residual/uncertainty without static ghost degradation.

Result: 0/5 windows passed.

| Window | PSNR beats all | L1 beats all | Flicker beats all | Static ghost no worse | Gate |
| --- | --- | --- | --- | --- | --- |
| `cut_roasted_beef_hand_tongs_meat_095_110` | no | no | no | no | FAIL |
| `cut_roasted_beef_hand_knife_meat_140_155` | no | no | no | no | FAIL |
| `flame_steak_torch_pan_155_170` | no | no | no | no | FAIL |
| `flame_steak_torch_sweep_195_210` | no | no | no | no | FAIL |
| `sear_steak_spoon_pan_220_235` | no | no | no | no | FAIL |

## Mean Metrics

| System | PSNR up | L1 down | Flicker down | Static ghost down |
| --- | ---: | ---: | ---: | ---: |
| `actual_hide_reveal` | 19.3667 | 0.0761056 | 0.0162899 | 0.152789 |
| `route0` | 30.5021 | 0.0148316 | 0.00799083 | 0.127333 |
| `matched_lifespan` | 29.8181 | 0.0163546 | 0.00795601 | 0.127333 |
| `residual_uncertainty` | 30.0734 | 0.0165723 | 0.00803902 | 0.145702 |

LPIPS: not available. The optional LPIPS attempt tried to download AlexNet weights on the compute node and failed with `Network is unreachable`; the final evaluator records null LPIPS values. Confident-track ID-switch sidecars were not available from the R009 discovery surface.

## Decision

R017 does not justify paper-scale validation. The implementation check shows that the naive checkpoint-backed runtime opacity gate is much worse than route0 and the two baselines on all frozen windows, and it increases static ghost score. The earlier R013 positive result remains a GT-crop-composite upper-bound, not evidence for the actual Gaussian/checkpoint-backed method.
