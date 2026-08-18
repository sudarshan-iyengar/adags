# FROZEN — coverage bounding pair (A1): design, before any outcome

Date: 2026-08-18. Status: **FROZEN DESIGN. DIAGNOSTIC ONLY.** Written and
committed BEFORE the reducer ran. No output of this measurement changes an
eligibility verdict, a floor, a census figure, a gate, or the standing of
G-OA's FAIL. It produces a *bounding pair* on one quantity and a
per-sequence class label used only to define the candidate population of a
future preregistration.

Reads [[elgs-absence-diagnostic-result]],
[[elgs-m2-oncomponent-split-design]], [[elgs-m2-oncomponent-split-result]],
[[elgs-cycle2-screening-record]], [[elgs-exhaustive-screen-scope]].

## 1. Why this exists

`track_coverage_upper_bound` and the true-absence limb share ONE constant.
The census statistic counts a foreground component as covered only where a
report with `v >= 0.5` lands in it (`prereg_m1_census_v1.json`
`track_coverage_statistic` plus `association_rules.identity_association`),
and the absence limb scores a window only where reports fail that same
test. So coverage and absence are mechanically coupled through `v >= 0.5`,
and a correction to the absence instrument moves coverage too.

M-2 then established that **`v` is binary**: over 10.8M reports it takes
only `0.0` and `1.0`, verified three independent ways
([[elgs-m2-oncomponent-split-result]] sections 3 and 8). Two consequences
bind here:

* the "lower the visibility threshold" correction is **vacuous** — every
  threshold in `(0, 1]` partitions the reports identically, so it cannot
  move coverage by a single component;
* the repair the evidence does support is a **component-membership** gate,
  and its effect on coverage has never been measured.

This measurement brackets that effect. It answers: if the instrument's
report-admission rule were corrected, how far could each sequence's
coverage move, and which sequences could cross the 0.5 floor?

## 2. Population and inputs

The **screened half of tranche 1** — the same 20 sequences and the same
sealed screened-half conversions that produced the cycle-2 screening table,
and therefore the same population as the 597 candidate windows. Artifacts
are named, never matched by timestamp, per
[[elgs-m2-oncomponent-split-design]] Appendix A: `writing_2` takes
`writing_2_screen_w0_239_fix79ae5b7` with census
`m1c3fix_census_writing_2_screen_r0`, and `xylophone` takes its
`_fix79ae5b7` conversion with census `m1c3fix_census_xylophone_screen_r0`.
The defective `writing_2_screen_w0_239` and the defective xylophone census
are excluded by name.

Nothing is re-tracked. No mask, track, or calibration input is modified.

## 3. The three readings — one denominator, three numerators

The denominator is FIXED for all three readings and is the census
statistic's own denominator: the number of eligible foreground components
over (tracking cameras x conversion frames), eligible per
`association_rules.component_definition` (8-connected, strictly greater
than 127, at least 64 px). Changing the denominator is prohibited; only the
report-admission rule varies.

A component is *covered* under a reading iff at least one report ADMITTED
by that reading has its rounded pixel inside that component.

| reading | admission rule | bound |
|---|---|---|
| **(i) frozen** | not a miss, in-domain, and `v >= 0.5` (i.e. `v == 1`) | **LOWER** |
| **(ii) any-report** | not a miss, in-domain, ANY `v` | **UPPER** |
| **(iii) anchor-agreeing** | reading (i), PLUS any `v == 0` report whose pixel's label equals the label of its identity's ANCHOR pixel in the same (camera, frame), with that label greater than 0 and eligible | **MIDDLE** |

Reading (i) reproduces `index_tracks`'s filter exactly and must therefore
equal the sealed census artifact's `components_covered / components_total`
to the integer. That equality is a hard contract check (section 7).

**Monotonicity, and it is load-bearing.** Reading (ii) admits a superset of
reading (iii)'s admitted reports, which admits a superset of reading (i)'s,
over an identical denominator. Therefore

```
coverage(i)  <=  coverage(iii)  <=  coverage(ii)
```

for every sequence, by construction rather than by observation.

## 4. Conventions — stated, because M-2 left one unstated

M-2 section 4.5 commissioned `anchor_agreement` and the independent reducer
DECLINED to compute it, on the ground that the frozen text never specifies
a camera-convention axis order ([[elgs-m2-oncomponent-split-result]]
section 5). That abstention was correct for M-2. It is resolved here by
derivation, not by preference, and the alternative is measured rather than
assumed away.

* **Anchor.** The anchor of identity `j` at (camera `c`, frame `t`) is the
  projection of `j`'s consensus point at frame `t` into camera `c`, through
  the same `w2c` and `K` the census uses (`frustum_containment`).
* **Projection axis order is FORCED by the census prereg, not chosen
  here.** `frustum_containment` bounds the first projected coordinate by
  `W - 1` and the second by `H - 1`. So the first is the column and the
  second is the row. `pixel_rounding` then fixes both to `floor(x + 0.5)`.
* **Label lookup.** `labels` is `(H, W)`; both the report pixel and the
  anchor pixel are read as `labels[row, col]`, row from the y/second
  coordinate, col from the x/first. This is the indexing
  `build_association` already performs for report pixels.
* **Out-of-raster anchors** are not admitted and are counted as
  `anchor_out_of_domain`.
* **Undefined consensus at frame `t`**: the report is NOT admitted, and is
  counted as `anchor_undefined`. The conservative disposition is chosen so
  that reading (iii) cannot be inflated by missing data.
* **Sensitivity readings, both reported:** (a) the transposed anchor
  indexing `labels[col, row]`, and (b) the anchor taken from the LAST
  DEFINED consensus point at or before `t` (the census's own "last
  triangulated position" language) instead of exactly at `t`. If either
  sensitivity moves a sequence across 0.5, reading (iii) is declared
  CONVENTION-DEPENDENT for that sequence and the sequence is classed
  `indeterminate` regardless of its primary value.

No new component machinery, no lineage, no IoU chaining, no new association
rule. The classification is per (camera, frame) and memoryless, exactly as
M-2 required.

## 5. Classification rule — fixed now

| class | rule |
|---|---|
| `eligible` | reading (iii) at least 0.5 |
| `ineligible` | reading (ii) below 0.5 |
| `indeterminate` | otherwise, or convention-dependent per section 4 |

The classes are inputs to a FUTURE preregistration's population definition.
They are not admissions. Admitting any sequence to evidence use requires a
fresh preregistration under a corrected instrument, per
[[elgs-absence-diagnostic-result]].

## 6. Scope, and the omission is disclosed rather than silent

Reading (i) is available for all 20 sequences from the sealed census
artifacts. Only two sequences fall below the 0.5 floor there. By section 3's
monotonicity, every sequence with reading (i) at least 0.5 has reading (iii)
at least 0.5 and is therefore `eligible` under section 5 WITHOUT computing
readings (ii) or (iii) at all. That is a proof, not an approximation.

Readings (ii) and (iii) are therefore computed for **four** sequences:

| sequence | reading (i) | why it is computed |
|---|---:|---|
| `poker` | 0.38244 | below the floor; decisive |
| `scissor` | 0.44090 | below the floor; decisive |
| `put_candy` | 0.50673 | nearest ABOVE the floor — a positive control that the (ii)/(iii) machinery behaves as monotonicity requires |
| `pour_tea` | 0.59082 | named in [[elgs-absence-diagnostic-result]] as the sequence whose coverage a corrected instrument would move upward |

The other 16 are reported with readings (ii)/(iii) as `NOT COMPUTED` and a
class of `eligible` carrying the tag `by_monotonicity`. `music_box`
(2,867 frames) is the single largest omission and is named here so the
omission is visible rather than silent.

## 7. Contract checks — any failure VOIDS the run

1. Reading (i) equals the sealed census artifact's `components_covered` and
   `components_total` **as integers**, per sequence.
2. The denominator `components_total` is IDENTICAL across all three
   readings, per sequence.
3. `coverage(i) <= coverage(iii) <= coverage(ii)`, per sequence.
4. The unreadable-mask count is 0.
5. The conversion directory names used are exactly those in section 2.

A void run is preserved and reported as void; it is not reconciled.

## 8. Independent recomputation

Reading (iii) is recomputed by a fresh-context worker from THIS frozen text
and the primary inputs only, forbidden from reading the reducer or its
output, and sealed before the primary result is disclosed to it. Agreement
confirms the reduction, not the inputs — the section 7 contract checks are
the guard on the inputs, as in M-2 section 8.

## 9. What this cannot establish

* **Physical presence or absence.** Nothing here observes the scene. A
  `v == 0` report agreeing with its anchor means the tracker's reported
  position and its own consensus point fall in the same mask component, not
  that the object was there. The frozen M1-A0b audit sample (73 windows)
  remains emitted and unrun, and no physical-absence claim is permitted
  before it returns.
* **That any sequence should be admitted.** Section 5 classes a candidate
  population; it admits nothing.
* **Anything about the occlusion supply**, which is untouched.
* **G-OA's FAIL**, which is not reopened.

## 10. Termination

The measurement ends when section 5 classes exist for all 20 sequences,
section 7's checks have been evaluated, and the independent recomputation of
reading (iii) has agreed or its disagreement has been reported. No
re-tracking, no tranche-2 work, and no further screening is authorised by
this page under any outcome.
