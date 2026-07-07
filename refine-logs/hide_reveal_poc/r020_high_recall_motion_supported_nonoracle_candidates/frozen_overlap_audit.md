# R020 Frozen-Window Overlap Audit

Generated: 2026-07-07T01:22:08+00:00

This is a posthoc audit only. Frozen R009 crop labels were not used by candidate discovery.

Coverage rule: `crop_iou >= 0.1 and temporal_iou >= 0.25`
Covered windows: `3/5`

| Frozen window | Best candidate | Crop IoU | Temporal IoU | Joint |
| --- | --- | ---: | ---: | ---: |
| `cut_roasted_beef_hand_tongs_meat_095_110` | `cut_roasted_beef_nonoracle_01_248_263_320_240_480_400` | 0.2458 | 0.0000 | 0.0000 |
| `cut_roasted_beef_hand_knife_meat_140_155` | `cut_roasted_beef_nonoracle_09_148_163_240_320_400_480` | 0.6662 | 0.3333 | 0.2221 |
| `flame_steak_torch_pan_155_170` | `flame_steak_nonoracle_07_156_171_320_320_480_480` | 0.4453 | 0.8824 | 0.3929 |
| `flame_steak_torch_sweep_195_210` | `flame_steak_nonoracle_10_184_199_240_320_400_480` | 0.5866 | 0.1852 | 0.1086 |
| `sear_steak_spoon_pan_220_235` | `sear_steak_nonoracle_03_220_235_240_320_400_480` | 0.7024 | 1.0000 | 0.7024 |

Interpretation: R020 is a high-recall diagnostic pool, not a method pass.
