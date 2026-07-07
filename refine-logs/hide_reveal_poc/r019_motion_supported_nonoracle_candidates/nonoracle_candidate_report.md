# Non-Oracle Event Candidate Discovery

Generated: 2026-07-07T01:13:19+00:00

## Scientific Guardrails

- Uses GT residual: `False`
- Uses frozen event-crop labels: `False`
- Candidate scores use motion-supported route0 dynamic output, route0-vs-static render deltas, motion masks, motion-mask boundaries, and route0 render flicker.
- The generated candidate crops are method inputs; the frozen R009 crops remain evaluation-only.

## Parameters

- `window_length`: `16`
- `temporal_stride`: `4`
- `tile_size`: `160`
- `tile_stride`: `80`
- `top_k_per_scene`: `8`
- `crop_iou_threshold`: `0.5`
- `temporal_iou_threshold`: `0.5`
- `score_weights`: `{'motion_supported_dynamic_render': 0.35, 'motion_supported_static_render_delta': 0.25, 'motion_mask_interior': 0.25, 'motion_mask_boundary': 0.1, 'motion_supported_route0_render_flicker': 0.05}`

## Scene Summary

| Scene | Raw candidates | Selected | Scored frames | Indexed masks |
| --- | ---: | ---: | ---: | ---: |
| cut_roasted_beef | 3456 | 8 | 300 | 300 |
| flame_steak | 3456 | 8 | 300 | 300 |
| sear_steak | 3456 | 8 | 300 | 300 |

## Selected Candidates

| Candidate | Scene | Frames | Crop | Score |
| --- | --- | --- | --- | ---: |
| `cut_roasted_beef_nonoracle_01_248_263_320_240_480_400` | `cut_roasted_beef` | 248-263 | `[320, 240, 480, 400]` | 0.059986 |
| `cut_roasted_beef_nonoracle_02_256_271_320_240_480_400` | `cut_roasted_beef` | 256-271 | `[320, 240, 480, 400]` | 0.051115 |
| `cut_roasted_beef_nonoracle_03_252_267_320_320_480_480` | `cut_roasted_beef` | 252-267 | `[320, 320, 480, 480]` | 0.048412 |
| `cut_roasted_beef_nonoracle_04_244_259_240_240_400_400` | `cut_roasted_beef` | 244-259 | `[240, 240, 400, 400]` | 0.046366 |
| `cut_roasted_beef_nonoracle_05_240_255_320_240_480_400` | `cut_roasted_beef` | 240-255 | `[320, 240, 480, 400]` | 0.045988 |
| `cut_roasted_beef_nonoracle_06_244_259_320_320_480_480` | `cut_roasted_beef` | 244-259 | `[320, 320, 480, 480]` | 0.045322 |
| `cut_roasted_beef_nonoracle_07_248_263_320_160_480_320` | `cut_roasted_beef` | 248-263 | `[320, 160, 480, 320]` | 0.041339 |
| `cut_roasted_beef_nonoracle_08_252_267_240_240_400_400` | `cut_roasted_beef` | 252-267 | `[240, 240, 400, 400]` | 0.040573 |
| `flame_steak_nonoracle_01_168_183_240_320_400_480` | `flame_steak` | 168-183 | `[240, 320, 400, 480]` | 0.056724 |
| `flame_steak_nonoracle_02_176_191_240_240_400_400` | `flame_steak` | 176-191 | `[240, 240, 400, 400]` | 0.054179 |
| `flame_steak_nonoracle_03_160_175_240_320_400_480` | `flame_steak` | 160-175 | `[240, 320, 400, 480]` | 0.052716 |
| `flame_steak_nonoracle_04_164_179_320_320_480_480` | `flame_steak` | 164-179 | `[320, 320, 480, 480]` | 0.051428 |
| `flame_steak_nonoracle_05_168_183_240_240_400_400` | `flame_steak` | 168-183 | `[240, 240, 400, 400]` | 0.049175 |
| `flame_steak_nonoracle_06_176_191_240_320_400_480` | `flame_steak` | 176-191 | `[240, 320, 400, 480]` | 0.047623 |
| `flame_steak_nonoracle_07_156_171_320_320_480_480` | `flame_steak` | 156-171 | `[320, 320, 480, 480]` | 0.044698 |
| `flame_steak_nonoracle_08_172_187_320_320_480_480` | `flame_steak` | 172-187 | `[320, 320, 480, 480]` | 0.044296 |
| `sear_steak_nonoracle_01_276_291_240_320_400_480` | `sear_steak` | 276-291 | `[240, 320, 400, 480]` | 0.032130 |
| `sear_steak_nonoracle_02_284_299_240_320_400_480` | `sear_steak` | 284-299 | `[240, 320, 400, 480]` | 0.029121 |
| `sear_steak_nonoracle_03_220_235_240_320_400_480` | `sear_steak` | 220-235 | `[240, 320, 400, 480]` | 0.025405 |
| `sear_steak_nonoracle_04_268_283_240_320_400_480` | `sear_steak` | 268-283 | `[240, 320, 400, 480]` | 0.024539 |
| `sear_steak_nonoracle_05_228_243_240_320_400_480` | `sear_steak` | 228-243 | `[240, 320, 400, 480]` | 0.024046 |
| `sear_steak_nonoracle_06_080_095_320_160_480_320` | `sear_steak` | 80-95 | `[320, 160, 480, 320]` | 0.018541 |
| `sear_steak_nonoracle_07_204_219_240_320_400_480` | `sear_steak` | 204-219 | `[240, 320, 400, 480]` | 0.018343 |
| `sear_steak_nonoracle_08_080_095_240_160_400_320` | `sear_steak` | 80-95 | `[240, 160, 400, 320]` | 0.018135 |

## Validation

- validation_ok: `True`
- validation_errors: `0`
- validation_warnings: `1`

## Outputs

- `nonoracle_candidate_manifest.json`
- `nonoracle_candidate_metadata.json`
- `nonoracle_candidate_components.csv`
- `nonoracle_candidate_validation.json`
