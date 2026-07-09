# Depth Occlusion Support

Generated: 2026-07-09T20:14:54+00:00

## Scientific Guardrails

- Uses GT residual: `False`
- Uses GT crop pixels: `False`
- Uses frozen event-crop labels: `False`
- Source manifest usage: `scene_sources_only_for_paths_and_frame_ranges`
- Depth model: `/leonardo_work/EUHPC_D21_034/proj_adags/models/depth-anything/DA3NESTED-GIANT-LARGE-1.1`

## Scene Summary

| Scene | Depth frames | Flow matched | Raw comps | Selected comps | Support frames | Max support frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cut_roasted_beef | 300 | 299 | 1200 | 360 | 121 | 0.005237 |
| flame_steak | 300 | 299 | 1200 | 360 | 114 | 0.005981 |
| sear_steak | 300 | 299 | 1200 | 360 | 120 | 0.003469 |

## Validation

- validation_ok: `True`
- validation_errors: `0`
- validation_warnings: `0`

## Outputs

- `depth_occlusion_support_manifest.json`
- `depth_occlusion_support_metadata.json`
- `depth_occlusion_support_components.csv`
- `depth_occlusion_support_validation.json`
- `support_masks/`
