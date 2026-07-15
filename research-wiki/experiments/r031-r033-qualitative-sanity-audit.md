---
type: experiment
node_id: exp:r031-r033-qualitative-sanity-audit
status: complete
created: 2026-07-15
idea: idea:depth-occlusion-event-support
---

# R031-R033 Qualitative Sanity Audit

## Purpose

Diagnose whether existing depth and support examples visibly contradict the Phase 8A forensic reading. This audit does not select thresholds, change a method, or score frozen windows.

## Procedure

Two temporary local contact sheets were inspected because submitting a Slurm job was explicitly forbidden and the normal image viewer failed under Leonardo namespace exhaustion.

1. Top-component examples selected deterministically from the first scene entry in the R032 component table:
   - `cut_roasted_beef/cam00_0260`
   - `flame_steak/cam00_0170`
   - `sear_steak/cam00_0096`
2. Midpoint frames from the five historical R009 windows:
   - `cut_roasted_beef/cam00_0102`
   - `cut_roasted_beef/cam00_0147`
   - `flame_steak/cam00_0162`
   - `flame_steak/cam00_0202`
   - `sear_steak/cam00_0227`
3. Panels compared source RGB, R031 DA3 visualization, R032 support, and R033 support. The historical-window sheet also placed `cam00` and `cam10` RGB side by side.

Temporary sheets were written only under `/tmp` and are intentionally not referenced as durable wiki assets. The underlying tracked evidence pointers and ignored generated images remain authoritative.

## Observations

- DA3 depth is qualitatively coherent at coarse scene scale: person, countertop, large foreground objects, and background depth layers are visible. This supports retaining depth as evidence.
- Fine interaction geometry is substantially less reliable. Hands, utensils, flame, food boundaries, and contact surfaces are blurred, merged, or weak relative to the dominant silhouette.
- R032 support is extremely sparse and contour-like. In the inspected examples it frequently selects fragments of person/object silhouettes or countertop structure rather than a complete occlusion/reveal region.
- R033 converts some sparse selections into conspicuous rectangular blocks. The blocks are spatially coarse and sometimes coexist with unrelated thin fragments; fill enlarges the selected tile but does not establish correct event localization.
- Several historical midpoint frames have no stored R032/R033 support at all, matching the low temporal presence in the overlap reports.
- `cam00` and `cam10` visibly differ in person position, foreshortening, counter layout, and object projection. A rectangle defined in `cam10` cannot be interpreted directly in `cam00` coordinates.

## Conclusion

The visual sanity check agrees with the forensic audit. R031 depth contains useful coarse geometry, but R032/R033 reduce it to brittle 2D contour/tile support without surface order or cross-view reprojection. The check does not validate Route 1 and was not used to tune its thresholds.

## Evidence

- `refine-logs/depth_occlusion_support/r031_da3_depth_full/depth_vis/`
- `refine-logs/depth_occlusion_support/r032_depth_support_highrecall/support_masks/`
- `refine-logs/depth_occlusion_support/r033_depth_support_highrecall_tilefill/support_masks/`
- `refine-logs/depth_occlusion_support/r032_support_overlap_highrecall/support_overlap_windows.csv`
- `refine-logs/depth_occlusion_support/r032_depth_support_highrecall/depth_occlusion_support_components.csv`
