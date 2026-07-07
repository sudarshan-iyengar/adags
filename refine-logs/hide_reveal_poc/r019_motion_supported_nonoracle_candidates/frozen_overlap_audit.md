# R019 Frozen-Window Overlap Audit

Generated: 2026-07-07T01:16:39+00:00

This is a posthoc audit only. Frozen R009 crop labels were not used by candidate discovery.

Coverage rule: `crop_iou >= 0.1 and temporal_iou >= 0.25`
Covered windows: `2/5`

| Frozen window | Best candidate | Crop IoU | Temporal IoU | Joint |
| --- | --- | ---: | ---: | ---: |
| `cut_roasted_beef_hand_tongs_meat_095_110` | `cut_roasted_beef_nonoracle_01_248_263_320_240_480_400` | 0.2458 | 0.0000 | 0.0000 |
| `cut_roasted_beef_hand_knife_meat_140_155` | `cut_roasted_beef_nonoracle_01_248_263_320_240_480_400` | 0.2458 | 0.0000 | 0.0000 |
| `flame_steak_torch_pan_155_170` | `flame_steak_nonoracle_07_156_171_320_320_480_480` | 0.4453 | 0.8824 | 0.3929 |
| `flame_steak_torch_sweep_195_210` | `flame_steak_nonoracle_01_168_183_240_320_400_480` | 0.5866 | 0.0000 | 0.0000 |
| `sear_steak_spoon_pan_220_235` | `sear_steak_nonoracle_03_220_235_240_320_400_480` | 0.7024 | 1.0000 | 0.7024 |

Interpretation: R019 is partial detector recovery, not sufficient M1 event support.
