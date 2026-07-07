# Non-Oracle Event Candidate Discovery

Generated: 2026-07-07T01:03:13+00:00

## Scientific Guardrails

- Uses GT residual: `False`
- Uses frozen event-crop labels: `False`
- Candidate scores use route0 dynamic output, route0-vs-static render deltas, motion-mask boundaries, and route0 render flicker.
- The generated candidate crops are method inputs; the frozen R009 crops remain evaluation-only.

## Parameters

- `window_length`: `16`
- `temporal_stride`: `4`
- `tile_size`: `160`
- `tile_stride`: `80`
- `top_k_per_scene`: `8`
- `crop_iou_threshold`: `0.5`
- `temporal_iou_threshold`: `0.5`
- `score_weights`: `{'dynamic_render': 0.35, 'static_render_delta': 0.25, 'motion_mask_boundary': 0.25, 'route0_render_flicker': 0.15}`

## Scene Summary

| Scene | Raw candidates | Selected | Scored frames | Indexed masks |
| --- | ---: | ---: | ---: | ---: |
| cut_roasted_beef | 3456 | 8 | 300 | 300 |
| flame_steak | 3456 | 8 | 300 | 300 |
| sear_steak | 3456 | 8 | 300 | 300 |

## Selected Candidates

| Candidate | Scene | Frames | Crop | Score |
| --- | --- | --- | --- | ---: |
| `cut_roasted_beef_nonoracle_01_088_103_320_0_480_160` | `cut_roasted_beef` | 88-103 | `[320, 0, 480, 160]` | 0.336002 |
| `cut_roasted_beef_nonoracle_02_080_095_320_0_480_160` | `cut_roasted_beef` | 80-95 | `[320, 0, 480, 160]` | 0.335843 |
| `cut_roasted_beef_nonoracle_03_096_111_320_0_480_160` | `cut_roasted_beef` | 96-111 | `[320, 0, 480, 160]` | 0.335764 |
| `cut_roasted_beef_nonoracle_04_104_119_320_0_480_160` | `cut_roasted_beef` | 104-119 | `[320, 0, 480, 160]` | 0.335146 |
| `cut_roasted_beef_nonoracle_05_072_087_320_0_480_160` | `cut_roasted_beef` | 72-87 | `[320, 0, 480, 160]` | 0.335071 |
| `cut_roasted_beef_nonoracle_06_112_127_320_0_480_160` | `cut_roasted_beef` | 112-127 | `[320, 0, 480, 160]` | 0.334510 |
| `cut_roasted_beef_nonoracle_07_064_079_320_0_480_160` | `cut_roasted_beef` | 64-79 | `[320, 0, 480, 160]` | 0.334079 |
| `cut_roasted_beef_nonoracle_08_120_135_320_0_480_160` | `cut_roasted_beef` | 120-135 | `[320, 0, 480, 160]` | 0.334029 |
| `flame_steak_nonoracle_01_164_179_240_0_400_160` | `flame_steak` | 164-179 | `[240, 0, 400, 160]` | 0.288907 |
| `flame_steak_nonoracle_02_172_187_240_0_400_160` | `flame_steak` | 172-187 | `[240, 0, 400, 160]` | 0.288806 |
| `flame_steak_nonoracle_03_156_171_240_0_400_160` | `flame_steak` | 156-171 | `[240, 0, 400, 160]` | 0.288666 |
| `flame_steak_nonoracle_04_180_195_240_0_400_160` | `flame_steak` | 180-195 | `[240, 0, 400, 160]` | 0.288446 |
| `flame_steak_nonoracle_05_148_163_240_0_400_160` | `flame_steak` | 148-163 | `[240, 0, 400, 160]` | 0.288422 |
| `flame_steak_nonoracle_06_140_155_240_0_400_160` | `flame_steak` | 140-155 | `[240, 0, 400, 160]` | 0.288257 |
| `flame_steak_nonoracle_07_188_203_240_0_400_160` | `flame_steak` | 188-203 | `[240, 0, 400, 160]` | 0.288248 |
| `flame_steak_nonoracle_08_196_211_240_0_400_160` | `flame_steak` | 196-211 | `[240, 0, 400, 160]` | 0.288171 |
| `sear_steak_nonoracle_01_128_143_320_0_480_160` | `sear_steak` | 128-143 | `[320, 0, 480, 160]` | 0.304130 |
| `sear_steak_nonoracle_02_136_151_320_0_480_160` | `sear_steak` | 136-151 | `[320, 0, 480, 160]` | 0.303957 |
| `sear_steak_nonoracle_03_128_143_240_0_400_160` | `sear_steak` | 128-143 | `[240, 0, 400, 160]` | 0.303931 |
| `sear_steak_nonoracle_04_120_135_320_0_480_160` | `sear_steak` | 120-135 | `[320, 0, 480, 160]` | 0.303882 |
| `sear_steak_nonoracle_05_136_151_240_0_400_160` | `sear_steak` | 136-151 | `[240, 0, 400, 160]` | 0.303864 |
| `sear_steak_nonoracle_06_120_135_240_0_400_160` | `sear_steak` | 120-135 | `[240, 0, 400, 160]` | 0.303673 |
| `sear_steak_nonoracle_07_144_159_240_0_400_160` | `sear_steak` | 144-159 | `[240, 0, 400, 160]` | 0.303539 |
| `sear_steak_nonoracle_08_144_159_320_0_480_160` | `sear_steak` | 144-159 | `[320, 0, 480, 160]` | 0.303361 |

## Validation

- validation_ok: `True`
- validation_errors: `0`
- validation_warnings: `1`

## Outputs

- `nonoracle_candidate_manifest.json`
- `nonoracle_candidate_metadata.json`
- `nonoracle_candidate_components.csv`
- `nonoracle_candidate_validation.json`
