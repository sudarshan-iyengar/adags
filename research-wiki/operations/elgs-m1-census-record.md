# EL-GS M1 — Census Record (IN PROGRESS)

Date started: 2026-08-11, immediately after
[[operations/elgs-m0-implementation-record]] (M0 PASSED). Governing
gate: `configs/elgs/prereg_m1_census_v1.json` (floors reviewer-signed
before any DiVa-360 statistic is computed; the failure/retry policy of
[[operations/elgs-m0-m1-implementation-plan]] §11.2 binds).

## Data acquisition — COMPLETE (2026-08-11)

- Inventory: `/apollo/users/sri/proj_adags/data/` held only `n3v`;
  DiVa-360 absent ⇒ acquisition executed.
- Source: the official public Dropbox release linked from
  github.com/brown-ivl/DiVa360 (MIT license; no registration —
  autonomous download authorized per plan §16b). Full release is
  ~8.4 TB; only the frozen five-sequence subset was pulled.
- Frozen subset (prereg amendment, committed pre-inspection at
  `42f94fe`): dev = {battery, flip_book, unlock}; calibration =
  {peel_apple, pour_salt}.
- Destination: `/apollo/users/sri/proj_adags/data/diva360/` (outside
  Git; 33 TB free at acquisition). Per-sequence `processed_data`
  zips downloaded via detached Determined CPU tasks (pilot battery
  task `8659e24b`; parallel tasks `91e9be0e`, `3e3b2a2d`,
  `8d73ea1f`, `4fc6bbee`; extraction+seal `f5a20a20`).
- Sizes (extracted): battery 29G, flip_book 11G, pour_salt 16G,
  peel_apple 8.7G, unlock 3.3G ≈ 68G total.
- Integrity: `MANIFEST.sha256` over the five zips —
  battery `1380a03a…`, flip_book `ef67321a…`, unlock `7b2a5bc5…`,
  peel_apple `aefd3e87…`, pour_salt `ba6ff2f5…`; zips retained;
  whole tree `chmod -R a-w` (read-only raw policy).
- Extraction note: the image's `unzip` fails on Dropbox's zip64
  archives ("stripped absolute path spec"); `python3 -m zipfile`
  extracts cleanly — recorded for reproducibility.

## Structure discovery (de-risks preprocessing)

Each sequence ships `transforms_train.json` / `transforms_test.json`
/ `transforms_val.json` (+ circle/spiral render paths) — the
NeRF/Blender convention the existing `scene/` loader already reads —
plus `image.tar`, `frames_1.tar.gz`, `segmented_gt.tar.gz`,
`segmented_ngp.tar.gz`. The planned converter likely reduces to
frame un-tarring into the layout the transforms reference plus the
camera-convention verification (reprojection checks) — no `scene/`
reader changes expected, as planned.

## Schema findings for the converter (2026-08-11)

- `transforms_train.json` (unlock inspected): NeRF-style `frames`
  entries with `file_path: "undist/camXX/NNNNNNNN.png"` (per-camera,
  per-frame undistorted images), `sharpness`, and a 4x4
  `transform_matrix`; the intrinsics-key block and frame count need
  the full schema dump (the raw JSON is single-line — line-grep
  counts are useless; parse it).
- `segmented_gt.tar.gz`: per-camera mask PNGs
  (`segmented_gt/camNN/NNNNNNNN.png`) — the fg/bg masks the
  model-free census statistics consume.
- Per-sequence structural variation: battery ships `image.tar` +
  `frames_1.tar.gz`; unlock has NO `image.tar` — the converter must
  discover each sequence's frame tarball(s) rather than assume one
  name.
- Converter scope confirmed: map DiVa transforms/intrinsics into the
  ADAGS Blender-reader convention, un-tar frames to the referenced
  `undist/` layout, verify by reprojection; no `scene/` reader
  changes expected.

## Remaining M1 steps (per the plan)

1. Un-tar + layout validation on the Apollo host AND inside the
   Determined runtime; reprojection camera-convention checks.
2. The one budgeted image revision (tracker stack), new tag pinned
   by digest, build-checked.
3. Tracker weights acquisition (same provenance treatment) + frozen
   tracks artifact + shift/shuffle controls (preprocessing GPU-h
   accounted for reproducibility only).
4. Census cells M1-A0 → A0b → A → B → C/D (≤25 GPU-h ceiling);
   independent recomputation; gate application; result recorded here
   either way.
