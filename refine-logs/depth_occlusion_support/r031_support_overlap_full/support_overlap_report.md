# Event Support Overlap Audit

Diagnostic only: this audit reads frozen R009 windows after support generation.
It must not be used as test-time support or for threshold tuning.

- Frozen manifest: `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/refine-logs/hide_reveal_real_windows.json`
- Support manifest: `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/refine-logs/depth_occlusion_support/r031_depth_support_full/depth_occlusion_support_manifest.json`
- Windows: `5`
- Mean support-frame fraction: `0.0625`
- Mean crop coverage: `0.000030`

| Window | Scene | Support frame frac | Mean crop coverage | Max crop coverage | Best crop IoU | Best temporal IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cut_roasted_beef_hand_tongs_meat_095_110 | cut_roasted_beef | 0.0000 | 0.000000 | 0.000000 |  |  |
| cut_roasted_beef_hand_knife_meat_140_155 | cut_roasted_beef | 0.0000 | 0.000000 | 0.000000 |  |  |
| flame_steak_torch_pan_155_170 | flame_steak | 0.1250 | 0.000000 | 0.000000 |  |  |
| flame_steak_torch_sweep_195_210 | flame_steak | 0.1250 | 0.000152 | 0.002436 |  |  |
| sear_steak_spoon_pan_220_235 | sear_steak | 0.0625 | 0.000000 | 0.000000 |  |  |
