# Event Support Overlap Audit

Diagnostic only: this audit reads frozen R009 windows after support generation.
It must not be used as test-time support or for threshold tuning.

- Frozen manifest: `refine-logs/hide_reveal_real_windows.json`
- Support manifest: `refine-logs/hide_reveal_poc/r020_high_recall_motion_supported_nonoracle_candidates/nonoracle_candidate_manifest.json`
- Windows: `5`
- Mean support-frame fraction: `0.6375`
- Mean crop coverage: `0.491371`

| Window | Scene | Support frame frac | Mean crop coverage | Max crop coverage | Best crop IoU | Best temporal IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cut_roasted_beef_hand_tongs_meat_095_110 | cut_roasted_beef | 0.0000 | 0.000000 | 0.000000 | 0.0 | 0.0 |
| cut_roasted_beef_hand_knife_meat_140_155 | cut_roasted_beef | 0.5000 | 0.420507 | 0.967742 | 0.666189111747851 | 0.3333333333333333 |
| flame_steak_torch_pan_155_170 | flame_steak | 0.9375 | 0.820513 | 1.000000 | 0.5865921787709497 | 0.8823529411764706 |
| flame_steak_torch_sweep_195_210 | flame_steak | 0.7500 | 0.459736 | 0.913462 | 0.5865921787709497 | 0.28 |
| sear_steak_spoon_pan_220_235 | sear_steak | 1.0000 | 0.756098 | 0.756098 | 0.702416918429003 | 1.0 |
