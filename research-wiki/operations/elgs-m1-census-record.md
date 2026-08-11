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
- `segmented_gt.tar.gz`: sparse per-camera segmented PNGs
  (`segmented_gt/camNN/NNNNNNNN.png`; 6 files for unlock) in
  ORIGINAL 1280×720 space — an audit reference, NOT a census input
  (see the corrected INPUT MAPPING below).
- Per-sequence structural variation: tarball names/compression vary
  (battery `image.tar`+`segmented_ngp.tar.gz`; pour_salt
  `image.tar.gz`+`segmented_ngp.tar`) — the landed converter
  discovers frame archives by CONTENT, never by name.
- TEMPORAL WIRING measured (unlock): the shipped transforms are
  single-instant snapshots, but `frames_1.tar.gz` holds the full
  per-frame set (12,751 PNGs; 41 cameras × 311 frames, indices
  0..310) and `segmented_ngp.tar.gz` holds 16,483 per-frame files
  (`segmented_ngp/camNN/00000000.png…`, all cameras incl. test/val).
- INPUT MAPPING — CORRECTED 2026-08-11 evening by pixel-level
  measurement (det tasks `405a7a8d`, `c662fadf`; supersedes this
  page's earlier "masks = segmented_ngp" note, which was wrong):
  `frames_1` members are RGBA at 1160×550 — EXACTLY the transforms'
  declared undistorted resolution — with photographic foreground and
  a continuous fg matte in the ALPHA channel (99.9% of alpha in
  [0,8]∪[247,255]). `segmented_ngp` members are RGBA at 1280×720 —
  the ORIGINAL pre-undistortion space — with binary alpha and zeroed
  background; same per-frame fg fractions as frames_1 alpha (e.g.
  0.1072 vs 0.1087 at index 150), i.e. the same segmentation in the
  WRONG pixel space for the shipped calibration. Census "fg/bg
  masks" therefore = frames_1 ALPHA binarized (>127),
  calibration-aligned; segmented_ngp and segmented_gt (sparse, 6
  files, also original-space) are NOT census inputs. The gate's
  eligibility definitions are unaffected (they reference "shipped
  fg/bg masks" abstractly; the mask SOURCE is an input-mapping fact
  recorded here and in the converter provenance).
- Required converter extension (owner decision D-M1-1): a --window
  mode crossing the STATIC rig calibration with frame indices
  (time = index/120 fps) to emit genuine temporal transforms for
  M1-B and the tracker; the landed single-instant mode remains for
  quick smokes.
- Converter scope confirmed: map DiVa transforms/intrinsics into the
  ADAGS Blender-reader convention, un-tar frames to the referenced
  `undist/` layout, verify by reprojection; no `scene/` reader
  changes expected.

## Image revision — COMPLETE (2026-08-11)

`apollo-h100-v2` = v1 + commit-pinned CoTracker3
(`co-tracker@82e02e80`) + imageio; built locally over the cached v1
layers with the in-Dockerfile build-check gate; pushed; manifest
digest `sha256:a2877f26cb8528…` recorded in
[[operations/apollo-determined-execution-authority]] — every
evidence-bearing M1 run pins it.

## Tracker weights — COMPLETE (2026-08-11)

Official CoTracker3 offline checkpoint from
`huggingface.co/facebook/cotracker3` (public, no registration):
`/apollo/users/sri/proj_adags/data/tracker_weights/cotracker3/scaled_offline.pth`
(101,890,938 bytes, sha256 `2670d4562ed69326dda775a26e54883925cd11b6…`,
MANIFEST.sha256 beside it, tree read-only; task `a5979f4c`).

## Converter + tracks pipeline landings (2026-08-11 evening)

- `--window` temporal-cross converter extension landed at `2ef4275`
  (Sonnet draft, owner-reviewed; 17 new fixture tests) and the frozen
  tracks artifact builder `scripts/build_elgs_tracks.py` at `943fc1f`
  (owner-implemented; model-free visual-hull seed constructor,
  CoTracker3/fake backend interface, miss-token conversion, IRLS
  consensus triangulation, r_u diagnostics frozen in the manifest,
  shift/shuffle controls; 14 CPU tests against a fully analytic
  four-camera ray-sphere oracle).
- `det_cfg_apollo_ctx.yaml` landed at `1596eaa`: context-based
  `det cmd run` config with NO work_dir and no worktree
  ADAGS_REPO_ROOT, so preprocessing tasks run only from git-archive
  contexts of pushed commits (closure rules).

## Real-data validation loop (fail-closed catches, 2026-08-11)

The first real-Apollo dry-runs did exactly what the fail-closed design
intends — each surfaced a real-schema property the synthetic fixtures
had idealized, was fixed at a pushed commit with regression tests, and
re-run:

1. det task `4a34184a` (ctx `943fc1f`): real transforms ship
   `file_path` WITH the image extension (`undist/cam01/00000000.png`,
   measured in det task `85424662`; ADAGS reader contract expects
   extension-less + append) → discovery hashed `.png.png` paths.
   FIX `fd047a9`: `normalize_source_file_path` at the schema layer;
   converted output now emits the extension-less canonical form.
2. det task `72bf9fd6` (ctx `fd047a9`): `segmented_ngp` MIRRORS the
   frame archive's exact member layout, so both archives fully cover
   the referenced paths — path matching cannot separate frames from
   masks. FIX `4a85fc5`: binary-content probe (decoded probe member
   with ≤2 grayscale values ⇒ mask archive, excluded from frame
   candidacy) + mask-source disambiguation by FULL window coverage
   (sparse `segmented_gt` can hit a window without covering it).
   Both deterministic, fail-closed on residual ambiguity.
3. det task `1c0951c0` (ctx `4a85fc5`): the binary-content probe
   FAILED on real data — segmented_ngp members are not binary
   grayscale but RGBA segmented IMAGERY. The pixel-level measurement
   that followed (tasks `405a7a8d`, `c662fadf`) produced the
   corrected input mapping above and the replacement design now in
   rework: frame-source disambiguation by RESOLUTION match against
   the transforms' declared (w,h) — exact and a priori — and census
   masks DERIVED from frames_1's alpha channel instead of extracting
   segmented_ngp at all (which would be geometrically wrong at
   1280×720 vs the 1160×550 calibration).
- unlock split facts (det task `85424662`): train=35 cams, test=val=
  {cam00,16,17,33,43,44}; all splits reference index 0 only;
  `frames_1` holds 41 cams × 311 frames (indices 0..310), all
  present at index 0; `image.tar.gz` is `.jpg` (never a candidate for
  `.png` references).

## Floor sign-off review (plan 7.2 item 6) — REJECTED then amended

- The preregistered fresh-context floor review ran 2026-08-11 BEFORE
  any census statistic: verdict REJECTED (pre-data, repairable).
  Floors 36/36/36 and sensitivity rows 47/55 were independently
  recomputed and verified CORRECT; four blocking definitional gaps
  made the gate non-mechanical (unfrozen r_site; undefined
  component/identity association rules; underived+undefined coverage
  floor statistic; M1-A gated:true contradicting plan 11.2's
  "M1-A0 ONLY").
- Repair: `prereg_m1_census_v1.json` revision 2 at `ef3252f` — all
  four findings repaired by pre-data amendment with the integrity
  statement recorded in the file (all repairs text-derived; no
  DiVa-360 measurement; floors unchanged). A fresh-context re-review
  of the amended text is REQUIRED to SIGN before any census statistic
  is computed (in progress).

## Remaining M1 steps (per the plan)

1. Converter dry-run PASS on unlock (in progress at ctx `4a85fc5`),
   then real conversion + load smoke in the Determined runtime +
   tracks-builder dry-run (the hull-seed construction doubles as the
   reprojection camera-convention check: a consistent non-empty hull
   requires correct conventions).
2. Floor re-review SIGN, then the M1-A0 evaluator implementation
   against the revision-2 frozen definitions.
3. Frozen tracks artifact + shift/shuffle controls via the v2 image
   (preprocessing GPU-h accounted for reproducibility only).
4. Census cells M1-A0 → A0b → A → B → C/D (≤25 GPU-h ceiling);
   independent recomputation; gate application; result recorded here
   either way.
