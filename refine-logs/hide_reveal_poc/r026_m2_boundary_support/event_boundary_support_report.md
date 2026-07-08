# M2 Event-Boundary Support

Generated: 2026-07-07T21:46:05+00:00

## Scientific Guardrails

- Uses GT residual: `False`
- Uses GT crop pixels: `False`
- Uses frozen event-crop labels: `False`
- Source manifest usage: `scene_sources_only_for_paths_and_frame_ranges`

## Parameters

- `max_components_per_scene`: `36`
- `max_pixel_fraction`: `0.03`
- `boundary_dilate`: `6`
- `min_component_area`: `16`
- `min_score`: `0.05`
- `use_flow`: `True`
- `score_weights`: `{'dynamic_mask_boundary': 0.5, 'flow_valid_boundary': 0.2, 'flow_magnitude_boundary': 0.1, 'route0_dynamic_render_boundary': 0.1, 'route0_static_delta_boundary': 0.05, 'route0_render_flicker_boundary': 0.05}`

## Scene Summary

| Scene | Masks | Flow matched | Raw comps | Selected comps | Support frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| cut_roasted_beef | 300 | 299 | 1200 | 36 | 23 |
| flame_steak | 300 | 299 | 1200 | 36 | 21 |
| sear_steak | 300 | 299 | 1200 | 36 | 22 |

## Selected Components

| Component | Scene | Image | Frame | Area | Score | BBox |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `cut_roasted_beef_boundary_001_cam00_0091` | `cut_roasted_beef` | `cam00_0091` | 91 | 47 | 0.994042 | `[288, 192, 352, 256]` |
| `cut_roasted_beef_boundary_002_cam00_0090` | `cut_roasted_beef` | `cam00_0090` | 90 | 41 | 0.993171 | `[288, 192, 352, 256]` |
| `cut_roasted_beef_boundary_003_cam00_0160` | `cut_roasted_beef` | `cam00_0160` | 160 | 177 | 0.992994 | `[416, 224, 480, 288]` |
| `cut_roasted_beef_boundary_004_cam00_0251` | `cut_roasted_beef` | `cam00_0251` | 251 | 35 | 0.990857 | `[256, 224, 320, 288]` |
| `cut_roasted_beef_boundary_005_cam00_0263` | `cut_roasted_beef` | `cam00_0263` | 263 | 126 | 0.988571 | `[288, 224, 352, 288]` |
| `cut_roasted_beef_boundary_006_cam00_0158` | `cut_roasted_beef` | `cam00_0158` | 158 | 131 | 0.988397 | `[416, 224, 480, 288]` |
| `cut_roasted_beef_boundary_007_cam00_0249` | `cut_roasted_beef` | `cam00_0249` | 249 | 26 | 0.987692 | `[416, 224, 480, 288]` |
| `cut_roasted_beef_boundary_008_cam00_0264` | `cut_roasted_beef` | `cam00_0264` | 264 | 106 | 0.986415 | `[288, 224, 352, 288]` |
| `cut_roasted_beef_boundary_009_cam00_0159` | `cut_roasted_beef` | `cam00_0159` | 159 | 46 | 0.986087 | `[224, 288, 288, 352]` |
| `cut_roasted_beef_boundary_010_cam00_0186` | `cut_roasted_beef` | `cam00_0186` | 186 | 408 | 0.986078 | `[416, 320, 480, 384]` |
| `cut_roasted_beef_boundary_011_cam00_0088` | `cut_roasted_beef` | `cam00_0088` | 88 | 137 | 0.985985 | `[384, 192, 448, 256]` |
| `cut_roasted_beef_boundary_012_cam00_0186` | `cut_roasted_beef` | `cam00_0186` | 186 | 119 | 0.985882 | `[224, 288, 288, 352]` |
| `cut_roasted_beef_boundary_013_cam00_0187` | `cut_roasted_beef` | `cam00_0187` | 187 | 474 | 0.985823 | `[416, 320, 480, 384]` |
| `cut_roasted_beef_boundary_014_cam00_0119` | `cut_roasted_beef` | `cam00_0119` | 119 | 114 | 0.985614 | `[224, 352, 288, 416]` |
| `cut_roasted_beef_boundary_015_cam00_0257` | `cut_roasted_beef` | `cam00_0257` | 257 | 261 | 0.984674 | `[288, 224, 352, 288]` |
| `cut_roasted_beef_boundary_016_cam00_0263` | `cut_roasted_beef` | `cam00_0263` | 263 | 134 | 0.984478 | `[416, 160, 480, 224]` |
| `cut_roasted_beef_boundary_017_cam00_0186` | `cut_roasted_beef` | `cam00_0186` | 186 | 419 | 0.984153 | `[256, 256, 320, 320]` |
| `cut_roasted_beef_boundary_018_cam00_0261` | `cut_roasted_beef` | `cam00_0261` | 261 | 374 | 0.983636 | `[384, 160, 448, 224]` |
| `cut_roasted_beef_boundary_019_cam00_0186` | `cut_roasted_beef` | `cam00_0186` | 186 | 19 | 0.983158 | `[256, 224, 320, 288]` |
| `cut_roasted_beef_boundary_020_cam00_0157` | `cut_roasted_beef` | `cam00_0157` | 157 | 146 | 0.983014 | `[224, 288, 288, 352]` |
| `cut_roasted_beef_boundary_021_cam00_0257` | `cut_roasted_beef` | `cam00_0257` | 257 | 182 | 0.982637 | `[32, 320, 96, 384]` |
| `cut_roasted_beef_boundary_022_cam00_0257` | `cut_roasted_beef` | `cam00_0257` | 257 | 182 | 0.982637 | `[64, 320, 128, 384]` |
| `cut_roasted_beef_boundary_023_cam00_0258` | `cut_roasted_beef` | `cam00_0258` | 258 | 239 | 0.981925 | `[288, 224, 352, 288]` |
| `cut_roasted_beef_boundary_024_cam00_0265` | `cut_roasted_beef` | `cam00_0265` | 265 | 87 | 0.981609 | `[288, 224, 352, 288]` |
| `cut_roasted_beef_boundary_025_cam00_0187` | `cut_roasted_beef` | `cam00_0187` | 187 | 449 | 0.981470 | `[256, 256, 320, 320]` |
| `cut_roasted_beef_boundary_026_cam00_0260` | `cut_roasted_beef` | `cam00_0260` | 260 | 366 | 0.981312 | `[384, 160, 448, 224]` |
| `cut_roasted_beef_boundary_027_cam00_0049` | `cut_roasted_beef` | `cam00_0049` | 49 | 80 | 0.981000 | `[96, 384, 160, 448]` |
| `cut_roasted_beef_boundary_028_cam00_0185` | `cut_roasted_beef` | `cam00_0185` | 185 | 242 | 0.980992 | `[288, 224, 352, 288]` |
| `cut_roasted_beef_boundary_029_cam00_0252` | `cut_roasted_beef` | `cam00_0252` | 252 | 71 | 0.980845 | `[416, 224, 480, 288]` |
| `cut_roasted_beef_boundary_030_cam00_0160` | `cut_roasted_beef` | `cam00_0160` | 160 | 236 | 0.980678 | `[416, 256, 480, 320]` |
| `cut_roasted_beef_boundary_031_cam00_0185` | `cut_roasted_beef` | `cam00_0185` | 185 | 150 | 0.980533 | `[416, 288, 480, 352]` |
| `cut_roasted_beef_boundary_032_cam00_0185` | `cut_roasted_beef` | `cam00_0185` | 185 | 412 | 0.980485 | `[256, 256, 320, 320]` |
| `cut_roasted_beef_boundary_033_cam00_0048` | `cut_roasted_beef` | `cam00_0048` | 48 | 65 | 0.980308 | `[96, 320, 160, 384]` |
| `cut_roasted_beef_boundary_034_cam00_0251` | `cut_roasted_beef` | `cam00_0251` | 251 | 430 | 0.979628 | `[256, 256, 320, 320]` |
| `cut_roasted_beef_boundary_035_cam00_0158` | `cut_roasted_beef` | `cam00_0158` | 158 | 182 | 0.979560 | `[416, 256, 480, 320]` |
| `cut_roasted_beef_boundary_036_cam00_0185` | `cut_roasted_beef` | `cam00_0185` | 185 | 703 | 0.979459 | `[288, 256, 352, 320]` |
| `flame_steak_boundary_001_cam00_0078` | `flame_steak` | `cam00_0078` | 78 | 49 | 0.973061 | `[96, 352, 160, 416]` |
| `flame_steak_boundary_002_cam00_0219` | `flame_steak` | `cam00_0219` | 219 | 464 | 0.969655 | `[352, 288, 416, 352]` |
| `flame_steak_boundary_003_cam00_0077` | `flame_steak` | `cam00_0077` | 77 | 396 | 0.968283 | `[352, 160, 416, 224]` |
| `flame_steak_boundary_004_cam00_0188` | `flame_steak` | `cam00_0188` | 188 | 309 | 0.964531 | `[352, 160, 416, 224]` |
| `flame_steak_boundary_005_cam00_0187` | `flame_steak` | `cam00_0187` | 187 | 331 | 0.964471 | `[352, 160, 416, 224]` |
| `flame_steak_boundary_006_cam00_0178` | `flame_steak` | `cam00_0178` | 178 | 381 | 0.963780 | `[320, 160, 384, 224]` |
| `flame_steak_boundary_007_cam00_0176` | `flame_steak` | `cam00_0176` | 176 | 323 | 0.961238 | `[320, 160, 384, 224]` |
| `flame_steak_boundary_008_cam00_0208` | `flame_steak` | `cam00_0208` | 208 | 265 | 0.958641 | `[352, 160, 416, 224]` |
| `flame_steak_boundary_009_cam00_0217` | `flame_steak` | `cam00_0217` | 217 | 219 | 0.958356 | `[416, 288, 480, 352]` |
| `flame_steak_boundary_010_cam00_0181` | `flame_steak` | `cam00_0181` | 181 | 1187 | 0.957304 | `[384, 256, 448, 320]` |
| `flame_steak_boundary_011_cam00_0078` | `flame_steak` | `cam00_0078` | 78 | 54 | 0.957037 | `[96, 320, 160, 384]` |
| `flame_steak_boundary_012_cam00_0193` | `flame_steak` | `cam00_0193` | 193 | 881 | 0.956776 | `[384, 256, 448, 320]` |
| `flame_steak_boundary_013_cam00_0085` | `flame_steak` | `cam00_0085` | 85 | 1063 | 0.956651 | `[320, 384, 384, 448]` |
| `flame_steak_boundary_014_cam00_0193` | `flame_steak` | `cam00_0193` | 193 | 365 | 0.956493 | `[384, 224, 448, 288]` |
| `flame_steak_boundary_015_cam00_0085` | `flame_steak` | `cam00_0085` | 85 | 67 | 0.956418 | `[96, 352, 160, 416]` |
| `flame_steak_boundary_016_cam00_0029` | `flame_steak` | `cam00_0029` | 29 | 80 | 0.956000 | `[0, 320, 64, 384]` |
| `flame_steak_boundary_017_cam00_0029` | `flame_steak` | `cam00_0029` | 29 | 80 | 0.956000 | `[0, 352, 64, 416]` |
| `flame_steak_boundary_018_cam00_0178` | `flame_steak` | `cam00_0178` | 178 | 177 | 0.955706 | `[352, 416, 416, 480]` |
| `flame_steak_boundary_019_cam00_0178` | `flame_steak` | `cam00_0178` | 178 | 177 | 0.955706 | `[384, 416, 448, 480]` |
| `flame_steak_boundary_020_cam00_0074` | `flame_steak` | `cam00_0074` | 74 | 466 | 0.955193 | `[320, 160, 384, 224]` |
| `flame_steak_boundary_021_cam00_0181` | `flame_steak` | `cam00_0181` | 181 | 512 | 0.954531 | `[416, 256, 480, 320]` |
| `flame_steak_boundary_022_cam00_0182` | `flame_steak` | `cam00_0182` | 182 | 526 | 0.954221 | `[320, 160, 384, 224]` |
| `flame_steak_boundary_023_cam00_0085` | `flame_steak` | `cam00_0085` | 85 | 68 | 0.954118 | `[96, 320, 160, 384]` |
| `flame_steak_boundary_024_cam00_0187` | `flame_steak` | `cam00_0187` | 187 | 648 | 0.953827 | `[320, 160, 384, 224]` |
| `flame_steak_boundary_025_cam00_0188` | `flame_steak` | `cam00_0188` | 188 | 626 | 0.953355 | `[320, 160, 384, 224]` |
| `flame_steak_boundary_026_cam00_0033` | `flame_steak` | `cam00_0033` | 33 | 208 | 0.953269 | `[32, 384, 96, 448]` |
| `flame_steak_boundary_027_cam00_0217` | `flame_steak` | `cam00_0217` | 217 | 722 | 0.952798 | `[384, 256, 448, 320]` |
| `flame_steak_boundary_028_cam00_0082` | `flame_steak` | `cam00_0082` | 82 | 860 | 0.952512 | `[320, 384, 384, 448]` |
| `flame_steak_boundary_029_cam00_0086` | `flame_steak` | `cam00_0086` | 86 | 102 | 0.952157 | `[96, 352, 160, 416]` |
| `flame_steak_boundary_030_cam00_0216` | `flame_steak` | `cam00_0216` | 216 | 778 | 0.951825 | `[320, 384, 384, 448]` |
| `flame_steak_boundary_031_cam00_0182` | `flame_steak` | `cam00_0182` | 182 | 529 | 0.951758 | `[416, 256, 480, 320]` |
| `flame_steak_boundary_032_cam00_0077` | `flame_steak` | `cam00_0077` | 77 | 513 | 0.951423 | `[384, 192, 448, 256]` |
| `flame_steak_boundary_033_cam00_0182` | `flame_steak` | `cam00_0182` | 182 | 1180 | 0.951390 | `[384, 256, 448, 320]` |
| `flame_steak_boundary_034_cam00_0185` | `flame_steak` | `cam00_0185` | 185 | 369 | 0.951328 | `[416, 352, 480, 416]` |
| `flame_steak_boundary_035_cam00_0182` | `flame_steak` | `cam00_0182` | 182 | 729 | 0.951276 | `[352, 256, 416, 320]` |
| `flame_steak_boundary_036_cam00_0207` | `flame_steak` | `cam00_0207` | 207 | 573 | 0.951204 | `[320, 160, 384, 224]` |
| `sear_steak_boundary_001_cam00_0091` | `sear_steak` | `cam00_0091` | 91 | 509 | 0.951906 | `[320, 160, 384, 224]` |
| `sear_steak_boundary_002_cam00_0086` | `sear_steak` | `cam00_0086` | 86 | 160 | 0.944250 | `[32, 384, 96, 448]` |
| `sear_steak_boundary_003_cam00_0091` | `sear_steak` | `cam00_0091` | 91 | 477 | 0.940210 | `[352, 160, 416, 224]` |
| `sear_steak_boundary_004_cam00_0171` | `sear_steak` | `cam00_0171` | 171 | 81 | 0.939136 | `[256, 443, 320, 507]` |
| `sear_steak_boundary_005_cam00_0171` | `sear_steak` | `cam00_0171` | 171 | 590 | 0.938780 | `[256, 416, 320, 480]` |
| `sear_steak_boundary_006_cam00_0092` | `sear_steak` | `cam00_0092` | 92 | 526 | 0.935970 | `[320, 160, 384, 224]` |
| `sear_steak_boundary_007_cam00_0102` | `sear_steak` | `cam00_0102` | 102 | 318 | 0.935849 | `[288, 443, 352, 507]` |
| `sear_steak_boundary_008_cam00_0102` | `sear_steak` | `cam00_0102` | 102 | 318 | 0.935849 | `[320, 443, 384, 507]` |
| `sear_steak_boundary_009_cam00_0088` | `sear_steak` | `cam00_0088` | 88 | 498 | 0.935422 | `[320, 160, 384, 224]` |
| `sear_steak_boundary_010_cam00_0095` | `sear_steak` | `cam00_0095` | 95 | 216 | 0.933889 | `[224, 288, 288, 352]` |
| `sear_steak_boundary_011_cam00_0099` | `sear_steak` | `cam00_0099` | 99 | 18 | 0.932222 | `[64, 288, 128, 352]` |
| `sear_steak_boundary_012_cam00_0096` | `sear_steak` | `cam00_0096` | 96 | 280 | 0.931286 | `[224, 288, 288, 352]` |
| `sear_steak_boundary_013_cam00_0171` | `sear_steak` | `cam00_0171` | 171 | 914 | 0.931028 | `[288, 320, 352, 384]` |
| `sear_steak_boundary_014_cam00_0090` | `sear_steak` | `cam00_0090` | 90 | 529 | 0.930737 | `[320, 160, 384, 224]` |
| `sear_steak_boundary_015_cam00_0093` | `sear_steak` | `cam00_0093` | 93 | 890 | 0.930472 | `[256, 384, 320, 448]` |
| `sear_steak_boundary_016_cam00_0095` | `sear_steak` | `cam00_0095` | 95 | 214 | 0.930467 | `[224, 320, 288, 384]` |
| `sear_steak_boundary_017_cam00_0086` | `sear_steak` | `cam00_0086` | 86 | 246 | 0.930406 | `[0, 384, 64, 448]` |
| `sear_steak_boundary_018_cam00_0088` | `sear_steak` | `cam00_0088` | 88 | 494 | 0.930364 | `[352, 160, 416, 224]` |
| `sear_steak_boundary_019_cam00_0102` | `sear_steak` | `cam00_0102` | 102 | 974 | 0.929733 | `[320, 416, 384, 480]` |
| `sear_steak_boundary_020_cam00_0138` | `sear_steak` | `cam00_0138` | 138 | 204 | 0.929412 | `[256, 416, 320, 480]` |
| `sear_steak_boundary_021_cam00_0093` | `sear_steak` | `cam00_0093` | 93 | 225 | 0.928178 | `[288, 160, 352, 224]` |
| `sear_steak_boundary_022_cam00_0092` | `sear_steak` | `cam00_0092` | 92 | 211 | 0.927773 | `[288, 160, 352, 224]` |
| `sear_steak_boundary_023_cam00_0173` | `sear_steak` | `cam00_0173` | 173 | 868 | 0.927650 | `[256, 416, 320, 480]` |
| `sear_steak_boundary_024_cam00_0172` | `sear_steak` | `cam00_0172` | 172 | 746 | 0.926917 | `[256, 416, 320, 480]` |
| `sear_steak_boundary_025_cam00_0203` | `sear_steak` | `cam00_0203` | 203 | 577 | 0.925685 | `[256, 416, 320, 480]` |
| `sear_steak_boundary_026_cam00_0035` | `sear_steak` | `cam00_0035` | 35 | 81 | 0.924938 | `[64, 416, 128, 480]` |
| `sear_steak_boundary_027_cam00_0204` | `sear_steak` | `cam00_0204` | 204 | 355 | 0.923718 | `[320, 443, 384, 507]` |
| `sear_steak_boundary_028_cam00_0093` | `sear_steak` | `cam00_0093` | 93 | 222 | 0.923694 | `[256, 416, 320, 480]` |
| `sear_steak_boundary_029_cam00_0031` | `sear_steak` | `cam00_0031` | 31 | 425 | 0.923388 | `[384, 443, 448, 507]` |
| `sear_steak_boundary_030_cam00_0040` | `sear_steak` | `cam00_0040` | 40 | 37 | 0.923243 | `[64, 416, 128, 480]` |
| `sear_steak_boundary_031_cam00_0202` | `sear_steak` | `cam00_0202` | 202 | 642 | 0.923115 | `[256, 416, 320, 480]` |
| `sear_steak_boundary_032_cam00_0007` | `sear_steak` | `cam00_0007` | 7 | 171 | 0.922573 | `[64, 416, 128, 480]` |
| `sear_steak_boundary_033_cam00_0173` | `sear_steak` | `cam00_0173` | 173 | 496 | 0.922339 | `[320, 443, 384, 507]` |
| `sear_steak_boundary_034_cam00_0011` | `sear_steak` | `cam00_0011` | 11 | 286 | 0.921818 | `[384, 384, 448, 448]` |
| `sear_steak_boundary_035_cam00_0099` | `sear_steak` | `cam00_0099` | 99 | 205 | 0.921756 | `[256, 416, 320, 480]` |
| `sear_steak_boundary_036_cam00_0204` | `sear_steak` | `cam00_0204` | 204 | 450 | 0.921511 | `[288, 443, 352, 507]` |

## Validation

- validation_ok: `True`
- validation_errors: `0`
- validation_warnings: `0`

## Outputs

- `event_boundary_support_manifest.json`
- `event_boundary_support_metadata.json`
- `event_boundary_support_components.csv`
- `event_boundary_support_validation.json`
- `support_masks/`
