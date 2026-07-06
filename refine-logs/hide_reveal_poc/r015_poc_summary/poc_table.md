# R015 PoC Summary Table and Crop Strips

Generated: 2026-07-06T00:53:01+00:00

## Synthetic Gate Summary

| Split | n | Candidate recall | Margin AUC | Accepted precision | Accepted recall | Identity reconnection | Matched-lifespan identity | False event rate | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| heldout | 40 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | PASS |

Frozen synthetic thresholds/weights: `C_min=0.55`, `m_event=0.02`, `lambda_id=1.0`, `lambda_static=0.5`, `lambda_budget=0.05`, `support_radius=5.5`.

## Real Window Summary

| System | n | PSNR up | Delta PSNR | L1/proxy-LPIPS down | Delta L1 | Flicker down | Delta flicker | Static ghost down | Delta ghost | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| route0 | 5 | 30.5021 | +0.0000 | 0.01483 | +0.00000 | 0.00799 | +0.00000 | 0.12733 | +0.00000 | trained route0 LoRA baseline |
| matched_lifespan | 5 | 29.8181 | -0.6840 | 0.01635 | +0.00152 | 0.00796 | -0.00003 | 0.12733 | +0.00000 | derived lifespan-only image composite |
| hide_reveal | 5 | 41.7149 | +11.2128 | 0.00267 | -0.01217 | 0.00169 | -0.00630 | 0.12733 | +0.00000 | derived GT-crop upper-bound; not a trained Gaussian output |
| residual_uncertainty | 5 | 30.0734 | -0.4287 | 0.01657 | +0.00174 | 0.00804 | +0.00005 | 0.14570 | +0.01837 | existing residual/filemask baseline eval from R011 |

Learned LPIPS and confident-track ID switches are unavailable in the current evaluator, so `L1/proxy-LPIPS` is the reported perceptual proxy and identity conclusions rely on the synthetic fixture plus the real qualitative strips.

## Qualitative Crop Strips

Rows are `route0`, `lifespan`, and `hide/reveal`; columns are four fixed frames from each predeclared event window. The hide/reveal row is an image-level GT-crop upper bound, not a trained Gaussian output.

- `cut_roasted_beef_hand_tongs_meat_095_110` frames [95, 100, 105, 110], crop `[245, 315, 455, 470]`: `refine-logs/hide_reveal_poc/r015_poc_summary/crop_strips/cut_roasted_beef_hand_tongs_meat_095_110.jpg`
- `cut_roasted_beef_hand_knife_meat_140_155` frames [140, 145, 150, 155], crop `[245, 315, 455, 470]`: `refine-logs/hide_reveal_poc/r015_poc_summary/crop_strips/cut_roasted_beef_hand_knife_meat_140_155.jpg`
- `flame_steak_torch_pan_155_170` frames [155, 160, 165, 170], crop `[250, 300, 445, 460]`: `refine-logs/hide_reveal_poc/r015_poc_summary/crop_strips/flame_steak_torch_pan_155_170.jpg`
- `flame_steak_torch_sweep_195_210` frames [195, 200, 205, 210], crop `[250, 300, 445, 460]`: `refine-logs/hide_reveal_poc/r015_poc_summary/crop_strips/flame_steak_torch_sweep_195_210.jpg`
- `sear_steak_spoon_pan_220_235` frames [220, 225, 230, 235], crop `[245, 320, 450, 470]`: `refine-logs/hide_reveal_poc/r015_poc_summary/crop_strips/sear_steak_spoon_pan_220_235.jpg`

## Source Files

- Synthetic summary: `refine-logs/hide_reveal_poc/synthetic/synthetic_summary.json`
- Route0 real eval: `refine-logs/hide_reveal_poc/r010_route0_real_eval/real_event_window_summary.json`
- Residual baseline eval: `refine-logs/hide_reveal_poc/r011_residual_uncertainty_real_eval/real_event_window_summary.json`
- Matched-lifespan/hide-reveal eval: `refine-logs/hide_reveal_poc/r012_r013_derived_real_eval/real_event_window_summary.json`
- Derived-render caveats and remote paths: `refine-logs/hide_reveal_poc/r012_r013_derived_real_renders/derived_poc_metadata.json`
