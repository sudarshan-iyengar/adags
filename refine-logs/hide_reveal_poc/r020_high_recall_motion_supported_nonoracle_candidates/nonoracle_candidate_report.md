# Non-Oracle Event Candidate Discovery

Generated: 2026-07-07T01:19:31+00:00

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
- `top_k_per_scene`: `24`
- `crop_iou_threshold`: `0.5`
- `temporal_iou_threshold`: `0.5`
- `score_weights`: `{'motion_supported_dynamic_render': 0.35, 'motion_supported_static_render_delta': 0.25, 'motion_mask_interior': 0.25, 'motion_mask_boundary': 0.1, 'motion_supported_route0_render_flicker': 0.05}`

## Scene Summary

| Scene | Raw candidates | Selected | Scored frames | Indexed masks |
| --- | ---: | ---: | ---: | ---: |
| cut_roasted_beef | 3456 | 24 | 300 | 300 |
| flame_steak | 3456 | 24 | 300 | 300 |
| sear_steak | 3456 | 24 | 300 | 300 |

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
| `cut_roasted_beef_nonoracle_09_148_163_240_320_400_480` | `cut_roasted_beef` | 148-163 | `[240, 320, 400, 480]` | 0.040328 |
| `cut_roasted_beef_nonoracle_10_260_275_320_320_480_480` | `cut_roasted_beef` | 260-275 | `[320, 320, 480, 480]` | 0.038562 |
| `cut_roasted_beef_nonoracle_11_184_199_240_320_400_480` | `cut_roasted_beef` | 184-199 | `[240, 320, 400, 480]` | 0.038338 |
| `cut_roasted_beef_nonoracle_12_156_171_240_320_400_480` | `cut_roasted_beef` | 156-171 | `[240, 320, 400, 480]` | 0.037364 |
| `cut_roasted_beef_nonoracle_13_184_199_320_320_480_480` | `cut_roasted_beef` | 184-199 | `[320, 320, 480, 480]` | 0.036032 |
| `cut_roasted_beef_nonoracle_14_176_191_240_320_400_480` | `cut_roasted_beef` | 176-191 | `[240, 320, 400, 480]` | 0.035781 |
| `cut_roasted_beef_nonoracle_15_256_271_320_160_480_320` | `cut_roasted_beef` | 256-271 | `[320, 160, 480, 320]` | 0.035676 |
| `cut_roasted_beef_nonoracle_16_244_259_240_320_400_480` | `cut_roasted_beef` | 244-259 | `[240, 320, 400, 480]` | 0.035540 |
| `cut_roasted_beef_nonoracle_17_152_167_320_320_480_480` | `cut_roasted_beef` | 152-167 | `[320, 320, 480, 480]` | 0.035479 |
| `cut_roasted_beef_nonoracle_18_120_135_320_320_480_480` | `cut_roasted_beef` | 120-135 | `[320, 320, 480, 480]` | 0.035425 |
| `cut_roasted_beef_nonoracle_19_236_251_240_240_400_400` | `cut_roasted_beef` | 236-251 | `[240, 240, 400, 400]` | 0.034906 |
| `cut_roasted_beef_nonoracle_20_176_191_320_320_480_480` | `cut_roasted_beef` | 176-191 | `[320, 320, 480, 480]` | 0.033335 |
| `cut_roasted_beef_nonoracle_21_248_263_240_160_400_320` | `cut_roasted_beef` | 248-263 | `[240, 160, 400, 320]` | 0.031104 |
| `cut_roasted_beef_nonoracle_22_240_255_320_160_480_320` | `cut_roasted_beef` | 240-255 | `[320, 160, 480, 320]` | 0.030959 |
| `cut_roasted_beef_nonoracle_23_264_279_320_240_480_400` | `cut_roasted_beef` | 264-279 | `[320, 240, 480, 400]` | 0.030831 |
| `cut_roasted_beef_nonoracle_24_208_223_240_320_400_480` | `cut_roasted_beef` | 208-223 | `[240, 320, 400, 480]` | 0.030789 |
| `flame_steak_nonoracle_01_168_183_240_320_400_480` | `flame_steak` | 168-183 | `[240, 320, 400, 480]` | 0.056724 |
| `flame_steak_nonoracle_02_176_191_240_240_400_400` | `flame_steak` | 176-191 | `[240, 240, 400, 400]` | 0.054179 |
| `flame_steak_nonoracle_03_160_175_240_320_400_480` | `flame_steak` | 160-175 | `[240, 320, 400, 480]` | 0.052716 |
| `flame_steak_nonoracle_04_164_179_320_320_480_480` | `flame_steak` | 164-179 | `[320, 320, 480, 480]` | 0.051428 |
| `flame_steak_nonoracle_05_168_183_240_240_400_400` | `flame_steak` | 168-183 | `[240, 240, 400, 400]` | 0.049175 |
| `flame_steak_nonoracle_06_176_191_240_320_400_480` | `flame_steak` | 176-191 | `[240, 320, 400, 480]` | 0.047623 |
| `flame_steak_nonoracle_07_156_171_320_320_480_480` | `flame_steak` | 156-171 | `[320, 320, 480, 480]` | 0.044698 |
| `flame_steak_nonoracle_08_172_187_320_320_480_480` | `flame_steak` | 172-187 | `[320, 320, 480, 480]` | 0.044296 |
| `flame_steak_nonoracle_09_172_187_320_240_480_400` | `flame_steak` | 172-187 | `[320, 240, 480, 400]` | 0.043566 |
| `flame_steak_nonoracle_10_184_199_240_320_400_480` | `flame_steak` | 184-199 | `[240, 320, 400, 480]` | 0.042127 |
| `flame_steak_nonoracle_11_248_263_240_320_400_480` | `flame_steak` | 248-263 | `[240, 320, 400, 480]` | 0.041830 |
| `flame_steak_nonoracle_12_256_271_240_320_400_480` | `flame_steak` | 256-271 | `[240, 320, 400, 480]` | 0.039905 |
| `flame_steak_nonoracle_13_184_199_240_240_400_400` | `flame_steak` | 184-199 | `[240, 240, 400, 400]` | 0.039511 |
| `flame_steak_nonoracle_14_180_195_320_240_480_400` | `flame_steak` | 180-195 | `[320, 240, 480, 400]` | 0.038510 |
| `flame_steak_nonoracle_15_164_179_320_240_480_400` | `flame_steak` | 164-179 | `[320, 240, 480, 400]` | 0.037770 |
| `flame_steak_nonoracle_16_240_255_240_320_400_480` | `flame_steak` | 240-255 | `[240, 320, 400, 480]` | 0.037434 |
| `flame_steak_nonoracle_17_076_091_240_240_400_400` | `flame_steak` | 76-91 | `[240, 240, 400, 400]` | 0.037013 |
| `flame_steak_nonoracle_18_176_191_320_160_480_320` | `flame_steak` | 176-191 | `[320, 160, 480, 320]` | 0.036248 |
| `flame_steak_nonoracle_19_080_095_240_320_400_480` | `flame_steak` | 80-95 | `[240, 320, 400, 480]` | 0.035514 |
| `flame_steak_nonoracle_20_176_191_240_160_400_320` | `flame_steak` | 176-191 | `[240, 160, 400, 320]` | 0.035331 |
| `flame_steak_nonoracle_21_204_219_240_240_400_400` | `flame_steak` | 204-219 | `[240, 240, 400, 400]` | 0.033579 |
| `flame_steak_nonoracle_22_168_183_320_160_480_320` | `flame_steak` | 168-183 | `[320, 160, 480, 320]` | 0.032798 |
| `flame_steak_nonoracle_23_160_175_240_240_400_400` | `flame_steak` | 160-175 | `[240, 240, 400, 400]` | 0.032672 |
| `flame_steak_nonoracle_24_264_279_240_320_400_480` | `flame_steak` | 264-279 | `[240, 320, 400, 480]` | 0.032559 |
| `sear_steak_nonoracle_01_276_291_240_320_400_480` | `sear_steak` | 276-291 | `[240, 320, 400, 480]` | 0.032130 |
| `sear_steak_nonoracle_02_284_299_240_320_400_480` | `sear_steak` | 284-299 | `[240, 320, 400, 480]` | 0.029121 |
| `sear_steak_nonoracle_03_220_235_240_320_400_480` | `sear_steak` | 220-235 | `[240, 320, 400, 480]` | 0.025405 |
| `sear_steak_nonoracle_04_268_283_240_320_400_480` | `sear_steak` | 268-283 | `[240, 320, 400, 480]` | 0.024539 |
| `sear_steak_nonoracle_05_228_243_240_320_400_480` | `sear_steak` | 228-243 | `[240, 320, 400, 480]` | 0.024046 |
| `sear_steak_nonoracle_06_080_095_320_160_480_320` | `sear_steak` | 80-95 | `[320, 160, 480, 320]` | 0.018541 |
| `sear_steak_nonoracle_07_204_219_240_320_400_480` | `sear_steak` | 204-219 | `[240, 320, 400, 480]` | 0.018343 |
| `sear_steak_nonoracle_08_080_095_240_160_400_320` | `sear_steak` | 80-95 | `[240, 160, 400, 320]` | 0.018135 |
| `sear_steak_nonoracle_09_276_291_320_320_480_480` | `sear_steak` | 276-291 | `[320, 320, 480, 480]` | 0.017911 |
| `sear_steak_nonoracle_10_212_227_240_320_400_480` | `sear_steak` | 212-227 | `[240, 320, 400, 480]` | 0.017723 |
| `sear_steak_nonoracle_11_164_179_240_320_400_480` | `sear_steak` | 164-179 | `[240, 320, 400, 480]` | 0.017281 |
| `sear_steak_nonoracle_12_280_295_240_240_400_400` | `sear_steak` | 280-295 | `[240, 240, 400, 400]` | 0.016676 |
| `sear_steak_nonoracle_13_284_299_320_320_480_480` | `sear_steak` | 284-299 | `[320, 320, 480, 480]` | 0.015921 |
| `sear_steak_nonoracle_14_228_243_160_320_320_480` | `sear_steak` | 228-243 | `[160, 320, 320, 480]` | 0.015804 |
| `sear_steak_nonoracle_15_280_295_160_240_320_400` | `sear_steak` | 280-295 | `[160, 240, 320, 400]` | 0.015776 |
| `sear_steak_nonoracle_16_172_187_240_320_400_480` | `sear_steak` | 172-187 | `[240, 320, 400, 480]` | 0.015688 |
| `sear_steak_nonoracle_17_088_103_240_160_400_320` | `sear_steak` | 88-103 | `[240, 160, 400, 320]` | 0.015621 |
| `sear_steak_nonoracle_18_008_023_320_160_480_320` | `sear_steak` | 8-23 | `[320, 160, 480, 320]` | 0.015456 |
| `sear_steak_nonoracle_19_088_103_320_160_480_320` | `sear_steak` | 88-103 | `[320, 160, 480, 320]` | 0.015401 |
| `sear_steak_nonoracle_20_220_235_160_320_320_480` | `sear_steak` | 220-235 | `[160, 320, 320, 480]` | 0.015225 |
| `sear_steak_nonoracle_21_080_095_320_240_480_400` | `sear_steak` | 80-95 | `[320, 240, 480, 400]` | 0.015224 |
| `sear_steak_nonoracle_22_284_299_160_320_320_480` | `sear_steak` | 284-299 | `[160, 320, 320, 480]` | 0.015150 |
| `sear_steak_nonoracle_23_084_099_240_240_400_400` | `sear_steak` | 84-99 | `[240, 240, 400, 400]` | 0.014982 |
| `sear_steak_nonoracle_24_008_023_240_160_400_320` | `sear_steak` | 8-23 | `[240, 160, 400, 320]` | 0.014954 |

## Validation

- validation_ok: `True`
- validation_errors: `0`
- validation_warnings: `1`

## Outputs

- `nonoracle_candidate_manifest.json`
- `nonoracle_candidate_metadata.json`
- `nonoracle_candidate_components.csv`
- `nonoracle_candidate_validation.json`
