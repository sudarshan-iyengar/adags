# Depth Occlusion Support

Generated: 2026-07-09T20:35:20+00:00

## Scientific Guardrails

- Uses GT residual: `False`
- Uses GT crop pixels: `False`
- Uses frozen event-crop labels: `False`
- Source manifest usage: `scene_sources_only_for_paths_and_frame_ranges`
- Depth model: `/leonardo_work/EUHPC_D21_034/proj_adags/models/depth-anything/DA3NESTED-GIANT-LARGE-1.1`

## Scene Summary

| Scene | Depth frames | Flow matched | Raw comps | Selected comps | Support frames | Max support frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cut_roasted_beef | 300 | 299 | 1200 | 360 | 139 | 0.029997 |
| flame_steak | 300 | 299 | 1200 | 360 | 142 | 0.029997 |
| sear_steak | 300 | 299 | 1200 | 360 | 127 | 0.029997 |

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
