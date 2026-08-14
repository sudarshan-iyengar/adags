# EL-GS Image-Substrate Conversion Defect (VERIFIED; 2026-08-13)

Status: **VERIFIED DEFECT** under the frozen verified-defect rule
([[operations/elgs-m0-m1-implementation-plan]] §11.2), limb (b) satisfied
by measurement; limb (a) partially satisfied and explicitly labelled
unverified below. A hostile fresh-context review returned **SOUND WITH
REPAIRS** and its five repairs are applied in this page. Read-only
evidence bundle sealed outside Git before review (122 files,
`MANIFEST.sha256` = `adcb990a847a89af133c0eaa8b1c994516fe272449af4cfbd9a96a9699d99d67`).

This page supersedes NOTHING by deletion. Cycle-1, cycle-2 and cycle-3
records and all sealed artifacts are preserved unchanged; the affected
pages carry append-only supersession notes pointing here.

## 1. The defect

`scripts/diva360_to_blender.py::select_archive_for_split` selected
`segmented_ngp.tar.gz` (original pre-undistortion space, decoded
**1280x720**) as the FRAME source for three conversions whose calibration
declares **(w,h) = (1160,550)**. The intended source, `frames_1.tar.gz`
(calibration space, decoded 1160x550), was present and unused.

Two independent code facts combine:

**(a) Root cause — variant archive nesting.** `elgs/diva360_schema.py::
split_top_level_dir` strips exactly ONE path component. For writing_2 and
xylophone, `frames_1.tar.gz` ships every member one level deeper:

    writing_2  frames_1.tar.gz -> dynamic_data/frames_1/cam00/00000000.png
    pour_tea   frames_1.tar.gz -> frames_1/cam00/00000000.png      (control)

so `rest` becomes `frames_1/camNN/...` and matches NONE of the 35 wanted
paths. frames_1 covered **zero**, not "some". `image.tar.gz` is excluded
by extension (`.jpg`), `segmented_gt.tar.gz` by size (~1 MB). Exactly one
candidate covered.

**(b) The check that would have caught it is conditional.**
`select_archive_for_split` runs its resolution probe only inside
`if len(covering) > 1:` (line 377); line 404 is a bare `return covering[0]`
that decodes nothing and never consults `declared_sizes`. With one covering
candidate the wrong-space archive is accepted silently.

The function's own docstring states the invariant unconditionally: 1280x720
is "a resolution the declared intrinsics could never have been computed
against." The invariant is stated globally and enforced on one branch.

**No compensating check exists anywhere** (reviewer-verified across
`build_plan`, `_execute_windowed_plan`, `derive_mask_from_frame`,
`build_elgs_tracks.py::_camera_intrinsics`,
`build_m1_census.py::load_component_labels`, and the test suite).

## 2. Blast radius (independently enumerated twice)

`rclone lsjson --dirs-only` -> 55 dirs = 27 `*_tracks` + 28 others, of
which `unlock_w0_47_tracks_r2_defective_hexfloat` is a preserved defective
tracks artifact => **27 conversions**. Every one has its provenance read
and its `masks/cam01/00000000.png` (and, in review, `undist/...`) IHDR
decoded. Declared is (1160,550) for EVERY sequence.

| | conversions | archive | decoded | verdict |
|---|---|---|---|---|
| clean | 24 | `frames_1.tar.gz` | 1160x550 | match |
| **affected** | 3 | `segmented_ngp.tar.gz` | 1280x720 | **mismatch** |

Affected set, complete: `writing_2_full_w0_480`, `writing_2_screen_w0_239`,
`xylophone_screen_w0_306`. Both writing_2 conversions are affected — the
cycle-2 SCREENING conversion that produced its eligibility row and the
cycle-3 FULL conversion that produced the G-R statistics.

## 3. Magnitude (reviewer-measured)

- Best translational alignment between writing_2's calibration-space mask
  and its raw segmented mask: **(dx,dy) = (107,120), IoU 0.875** — the two
  spaces differ by ~160 px of translation plus undistortion nonlinearity.
- **IoU at the (0,0) offset the code actually uses: 0.082.**
- Fraction of foreground inside the addressable `[0,1159]x[0,549]` window:
  **writing_2 0.1444**, **xylophone 0.0609**; controls pour_tea and
  tambourine **1.0000**.

The clip in `build_elgs_tracks.py::_mask_positive` is **inert**, not
protective: `inside` already bounds uv to the declared box, strictly inside
the 1280x720 raster, so the code silently reads the top-left 1160x550
corner. Same for `build_m1_census.py::build_association`.

## 4. Why "internally consistent" is not a defense

The cycle-3 gate result recorded association hit-rate 0.997 and consensus
reprojection median 2.35 px as evidence the chain was internally
consistent. Precisely stated:

- **image<->image consistency HOLDS** — tracker and masks share raw space,
  so association is genuinely coherent.
- **geometry<->image consistency FAILS** — hull carving, query placement,
  in-domain/miss labelling, triangulation, frustum containment, and the
  coverage denominator all mix calibration-space geometry with raw-space
  rasters.

An approximately uniform per-camera pixel offset is an alternative (wrong)
camera model. It *predicts* both observed diagnostics: consistent shifted
`cx,cy` triangulates with small residuals to a DISPLACED 3-D point. The
2.35 px median is the signature of the defect, not evidence against it.

Additionally the coverage denominator counts components over the FULL
1280x720 mask while the numerator can only be reached inside the top-left
window, so writing_2's 0.8637 is not the quantity the prereg defines.

And: content in the lower ~22% of each frame is systematically labelled
`is_miss`, a mechanism that MANUFACTURES disappear/reappear transitions —
which is exactly the class G-R counts.

## 4b. REMEASUREMENT COMPLETE (2026-08-14) — outcomes are now KNOWN

The "CORRECTED OUTCOME UNKNOWN" entries in §5 below are RESOLVED by
[[operations/elgs-substrate-remeasurement-result]]. Under the unchanged
frozen predicates on the corrected substrate:

- **G-R FAILS** (unscreened-half union returns 0 < 36; the recorded PASS at
  union 64 was defect-produced).
- **writing_2 and xylophone are BOTH NOT ELIGIBLE** under the cycle-2
  predicate => **tranche 1 has ZERO eligible candidates**.
- Coverage IMPROVED in every case (0.845->0.924, 0.8637->0.9340,
  0.577->0.779), confirming the correction rather than a second defect.
- The magnitude prediction in §4 is CONFIRMED empirically: the
  misregistration MANUFACTURED disappear/reappear transitions
  (writing_2 true-absence 84 -> 2).
- G-OA is unchanged and remains FINAL, exactly as §5 requires.

Independent fresh-context recomputation agreed EXACTLY on all seventeen
gate-bearing numbers; integrity audit clean (experiments 62-67, commit
`79ae5b7`).

## 5. Status of every affected claim (ORIGINAL assessment; see §4b for resolved outcomes)

| Object | Classification |
|---|---|
| Wrong archive selected for 3/27 conversions | DIRECTLY VERIFIED DEFECT |
| Root cause = variant `dynamic_data/` nesting | DIRECTLY VERIFIED DEFECT |
| Conditional resolution probe; no check elsewhere | DIRECTLY VERIFIED DEFECT |
| Geometry<->image mapping wrong (~160 px; IoU 0.082) | DIRECTLY VERIFIED DEFECT |
| writing_2 / xylophone tracks + census artifacts | INVALIDATED MEASUREMENT |
| **G-R PASS** (union 64, coverage 0.8637) | **CORRECTED OUTCOME UNKNOWN** — not established, NOT refuted |
| writing_2 cycle-2 eligibility (union 50, cov 0.845) | CORRECTED OUTCOME UNKNOWN |
| Cycle-2 "exactly one eligible => DRY" outcome | CORRECTED OUTCOME UNKNOWN |
| xylophone screening row and its NO | CORRECTED OUTCOME UNKNOWN — may flip the frozen checkpoint autonomy condition (>= 2 eligible) |
| **G-OA valid FAIL** (pour_tea per-seq coverage 0.3748) | **VERDICT ROBUST INDEPENDENTLY — remains FINAL** |
| 24 clean conversions and all statistics from them | unaffected (verified by measurement) |

**G-OA is not reopened.** Its sole violation is a per-sequence floor
computed entirely within pour_tea, which is clean. writing_2 contributes
only to POOLED numbers, and correcting it cannot convert the FAIL into a
PASS: pooled coverage would if anything fall (writing_2's 0.8637 is the
highest of the three), and pooled absence survives striking writing_2
entirely (205 - 63 = 142 >= 36). The frozen policy's finality applies.

## 6. Verified-defect qualification

- **Limb (b) — result-affecting: SATISFIED by measurement** (IoU 0.082;
  14.4% / 6.1% addressable foreground). Deliberately distinguished from
  "verdict-changing", which is UNKNOWN.
- **Limb (a) — discovery independent of outcome: PARTIALLY SATISFIED,
  UNVERIFIED.** The archive-selection fact was recorded contemporaneously
  in [[operations/elgs-cycle3-gate-result]] §"Material provenance finding",
  not conjured after the fact; what came later is the re-assessment of its
  severity. Discovery provenance is not determinable from artifacts and is
  labelled unverified.
- **Direction argues against motivated reasoning:** the defect removes a
  PASS. Outcome-shopping would not target the one verdict that went the
  project's way.
- **The real abuse risk is the opposite one and is refused here:** this
  defect must NOT be used to reopen G-OA's FAIL (see §5).
- **Same-standard check:** no earlier PASS was held to a laxer substrate
  audit; the 24 clean conversions are clean by measurement.

## 7. Corrections to the first characterisation (review repairs)

1. **Novelty was overstated.** [[operations/elgs-cycle3-gate-result]]
   already recorded the archive selection AND the single-covering-archive
   mechanism. This page's contribution is overturning the
   "verdict-robust / internally consistent" assessment, establishing the
   root cause and magnitude, and widening the exposure to cycle 2.
2. "Internally consistent" was treated too loosely by BOTH the gate page
   (as proof of correctness) and the first write-up (as proof of error).
   §4 states it precisely.
3. Cycle-2 exposure was understated: writing_2 was the SOLE eligible
   candidate, so the entire formal selection outcome depends on one
   defective row.
4. xylophone was under-weighted: it may flip a recorded decision, not just
   a table row.
5. Root cause was left open ("missing at least one path"); it is now
   closed — frames_1 covered zero, due to variant nesting.

## 8. NOT established

- Whether corrected re-measurement changes G-R, writing_2 eligibility, the
  cycle-2 selection outcome, or xylophone's row — in EITHER direction.
- The zero-query-camera figures (writing_2 2/26, xylophone 7/26):
  **unverified** by the evidence bundle and by the reviewer. Consistent
  with the defect; not recomputed.
- Byte-identity of the derived arrays against `segmented_ngp` members
  (space-identity IS verified: frames_1 is 1160x550, image is `.jpg`,
  segmented_gt is ~1 MB, so only segmented_ngp ships 1280x720 RGBA).

## 9. Remediation

Landed with this record:
1. Unconditional resolution postcondition in `select_archive_for_split`
   (applies on the single-covering path too), fail-closed.
2. Variant-nesting support so `frames_1` is discoverable, fail-closed on
   ambiguous prefixes.
3. Provenance reports the ACTUAL selected archive, its resolved prefix and
   its DECODED dimensions instead of unconditional boilerplate.
4. Regression tests for both the single- and multi-covering paths and for
   the nested layout.

Pending (not run at this record): re-convert, re-track and re-census
writing_2 and xylophone on the corrected substrate, then reapply the
UNCHANGED frozen cycle-2 eligibility predicate and cycle-3 gate logic and
record the outcome either way. No threshold, prereg, floor or gate
definition is altered.
