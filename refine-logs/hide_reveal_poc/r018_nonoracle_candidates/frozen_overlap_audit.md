# R018 Frozen-Window Overlap Audit

Generated: 2026-07-07T01:08:20+00:00

This is a posthoc audit only. Frozen R009 crop labels were not used by candidate discovery.

Coverage rule: `crop_iou >= 0.1 and temporal_iou >= 0.25`
Covered windows: `0/5`

| Frozen window | Best candidate | Crop IoU | Temporal IoU | Joint |
| --- | --- | ---: | ---: | ---: |
| `cut_roasted_beef_hand_tongs_meat_095_110` | `cut_roasted_beef_nonoracle_01_088_103_320_0_480_160` | 0.0000 | 0.3913 | 0.0000 |
| `cut_roasted_beef_hand_knife_meat_140_155` | `cut_roasted_beef_nonoracle_01_088_103_320_0_480_160` | 0.0000 | 0.0000 | 0.0000 |
| `flame_steak_torch_pan_155_170` | `flame_steak_nonoracle_01_164_179_240_0_400_160` | 0.0000 | 0.2800 | 0.0000 |
| `flame_steak_torch_sweep_195_210` | `flame_steak_nonoracle_01_164_179_240_0_400_160` | 0.0000 | 0.0000 | 0.0000 |
| `sear_steak_spoon_pan_220_235` | `sear_steak_nonoracle_01_128_143_320_0_480_160` | 0.0000 | 0.0000 | 0.0000 |

Interpretation: R018 is a detector failure/diagnostic, not a method pass.
