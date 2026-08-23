# SPEC (FROZEN) — LRV5-NCX, the non-convex falsifier for axis-aligned
# hull completion (2026-08-24)

EXPLORATORY, `evidence_bearing: false`. **Design document only. Nothing
in `D:\adags` was modified to produce it, no experiment was submitted,
and no GPU work was run.** Every number below is either quoted from a
tracked research-wiki page or produced by the CPU preflight
`nonconvex_preflight.py`, whose full output is reproduced in §14.

## 0. What is under test, and why a new fixture is required

[[lrv3-membership-candidates-result-2026-08-23]] §7 records an axis-aligned
**hull-completion** step that would have taken candidate A2+B from
0.9375 / 0.8088 to **0.9400 / 1.0000** on LRV3 by filling the two holes in
a contiguous 2 × 2 × 2 block. It was **not adopted**, for two reasons that
this spec inherits verbatim:

1. it was identified *after* seeing A2+B fall 0.09 short of the recall
   floor, so adopting it there would have been post-hoc selection;
2. *"hull completion assumes the object is cell-convex. That is true of a
   sphere and need not be true of anything else, so it is a
   fixture-shaped rule until tested on a non-convex object."*

**LRV3 cannot test (2) at all.** Its event object is a sphere of radius
0.20, and a sphere is cell-convex at every grid resolution and phase: the
set of cells it meets has no interior hole, so hull completion on LRV3
*cannot* gate a cell the object does not occupy. The recorded 0.9400 is
therefore not evidence that the rule is sound; it is evidence that the
rule is untested.

**The claim under test, stated so it can fail:** *axis-aligned cell-hull
completion of the accepted component adds only cells the event object
actually occupies.* LRV5-NCX is built to make that claim false if it is
false — a connected, non-convex event object whose axis-aligned cell hull
provably contains cells with **zero** object volume, and whose concavity
is **occupied by a persistent static object** so that gating it is
measurably and physically wrong.

## 1. Analytic geometry — FROZEN

Scene id **`LRV5`**, directory `data/synthetic/lrv5_ncx/`. All lengths are
in the same world units as LRV1/LRV2/LRV3.

### 1.1 The event object: an L formed by two axis-aligned boxes

Two axis-aligned boxes joined along a corner block, lying in the x–z
plane and extruded in y.

| | x | y | z |
|---|---|---|---|
| **Arm A** (long in z) | `[-0.64, -0.24]` | `[-0.30, +0.30]` | `[-0.64, +0.64]` |
| **Arm B** (long in x) | `[-0.64, +0.64]` | `[-0.30, +0.30]` | `[-0.64, -0.24]` |

Membership predicate, exact and closed:

```
is_event(x, y, z) =
      (-0.64 <= x <= -0.24 and -0.30 <= y <= 0.30 and -0.64 <= z <= 0.64)
   or (-0.64 <= x <=  0.64 and -0.30 <= y <= 0.30 and -0.64 <= z <= -0.24)
```

* **Connected**: the arms share the solid corner block
  `x∈[-0.64,-0.24] × y∈[-0.30,0.30] × z∈[-0.64,-0.24]`, of volume
  0.4·0.6·0.4 = 0.096. Connectivity is verified at the *voxel* level in
  §14 (P1), not merely asserted in the continuum.
* **Exact volume** = 1.28·0.60·1.28 − 0.88·0.60·0.88 = **0.518400**.
* **Notch (the concavity)** = `x ∈ (-0.24, 0.64] × y ∈ [-0.30, 0.30] ×
  z ∈ (-0.24, 0.64]`, i.e. **0.88 × 0.60 × 0.88**.

**Why these numbers, and each is load-bearing.**

* **Arm thickness 0.40, hull half-extent 0.64, notch 0.88.** The binding
  constraint is that the notch must contain a *fully* object-free cell at
  the grid the estimator actually builds. That grid is the **trained**
  cloud's own bounding box, whose extent is not knowable in advance. On
  LRV3 it is recoverable from the record: the 8 hull cells hold 6.6% of
  the sphere's volume, so cell size ≈ 0.3989 against the fresh cloud's
  0.325 — a 22.7% inflation. An interval of width `W` contains a complete
  cell of size `c` at **every** phase iff `W ≥ 2c`. At 0.88 that is
  guaranteed up to `c = 0.44`, i.e. a 35% inflation, and the swept
  measurement in §14 (P3) shows ≥ 1 empty in-hull cell in **600 of 600**
  swept configurations covering `c ∈ [0.30, 0.50]` and all phases.
* **Extrusion 0.60 in y.** The notch is bounded by object only in x and
  z, so y never has to satisfy the `W ≥ 2c` condition; 0.60 is chosen as
  the smallest thickness that still spans two cells at the fresh grid
  (0.60 / 0.325 = 1.85) so the object is not a degenerate plate.
* **An L rather than a tri-lobed object.** Its notch is a *through-slot*
  open on +x, +z and both ±y, so the static content standing in it is
  never enclosed. The preflight measures the consequence: at most **2 of
  the 4 sampled training cameras** ever see the notch contents change
  (§14, P5b).

### 1.2 The notch filler: a STATIC vertical cross

Present at **every** frame, never absent, never part of the event object.

| | x | y | z |
|---|---|---|---|
| **Wall F1** (spans x) | `[-0.12, +0.52]` | `[-0.30, +0.30]` | `[+0.14, +0.26]` |
| **Wall F2** (spans z) | `[+0.14, +0.26]` | `[-0.30, +0.30]` | `[-0.12, +0.52]` |

* Clearance to both inner faces of the L is exactly **0.12** on each side
  (`-0.12` vs arm A's `-0.24`; `-0.12` vs arm B's `-0.24`), and 0.12 to
  the hull edge at `±0.64`. Verified disjoint from the event object in
  §14 ("event object and notch filler are disjoint").
* **This object is the falsifier's teeth.** If hull completion gates the
  concavity, the filler's rows enter a gated family and are driven to
  presence ≈ 0 over the whole 27-frame absence gap — i.e. a persistent,
  always-visible static object is **deleted from the render for 27
  frames**. That is not a bookkeeping deficit; it is a rendering error a
  held-out metric will see.
* **Why a cross of thin walls and not a solid block.** Trained rows sit on
  *surfaces*, not in volumes. A solid block in the notch would leave its
  own interior cells empty of rows, so gating them would cost nothing and
  the falsifier would be toothless in exactly the way the design brief
  warns about. A full-height cross spreads surface across the notch's
  cells in both x and z and in both hull y-cells. Measured effect: the
  filler supplies **0.2738** (O1) and **0.3639** (O2) of the event
  object's own in-cell surface (§14, P5).

### 1.3 Static context

Relocated from LRV3 so that **neither** predeclared orientation
intersects them (verified in §14). Radii, colours and textures are
unchanged from `build_synthetic_reveal_scene.STATIC_SPHERES`.

| sphere | centre | radius |
|---|---|---|
| S1 (`stripe_y`, teal) | `(-0.95, -0.20, 0.85)` | 0.28 |
| S2 (`stripe_x`, magenta) | `(0.95, 0.15, -0.95)` | 0.26 |
| S3 (`checker`, olive) | `(-0.30, -0.40, 1.05)` | 0.22 |

Ground plane: `y = -0.62`, half extent **1.3**, unchanged. It equals the
initialization half-width, so LRV2's uncovered-surface defect cannot
recur: every ground point with `|x|,|z| ≤ 1.3` has `|y| = 0.62 ≤ 1.3` and
therefore lies inside the init cube.

The generator must gain one new primitive, `box_hit` (slab test), used by
the event object and the filler. Nothing else about the ray tracer,
shading, or emission changes.

## 2. Placement and grid phase — FROZEN

The object's cell hull is **centred on the world origin in x and z**
(`±0.64` about 0) and on `y = 0` (`±0.30`).

* **Why centred.** The fresh grid runs from the cloud minimum
  `-1.299888` in steps of `0.324985`, giving x edges
  `… -0.649918, -0.324933, 0.000052, 0.325037, 0.650022 …`; the object's
  hull edges at `±0.64` therefore sit **~0.010 inside** the cell
  boundaries at `-0.649918` and `+0.650022`, so no face is coincident
  with a cell boundary and no row lands on a degenerate edge. Centring
  also minimises the object's angular extent
  from the surround ring, which is what makes the enlarged object fit the
  frame (§2.1).
* **Why the concavity lands in a cell rather than straddling one.** At
  the fresh grid the notch spans world `(-0.24, 0.64]` in x and z, and
  cells 4 (`[0.000, 0.325]`) and 5 (`[0.325, 0.650]`) lie strictly inside
  it in both axes. That is a full **2 × 2** block of object-free (x,z)
  cells crossed with the hull's 2 y-cells → **8 fully empty cells**, not a
  single lucky one (§14, P2).
* **Grid phase is not fully controllable, and that is declared.** The
  estimator's grid is the *trained* cloud's bounding box
  (`build_voxel_groups(gaussians._xyz)`), which cannot be predicted before
  the run. The design response is (a) a notch wide enough that at least
  one empty cell exists at **every** swept phase and cell size (§14, P3),
  and (b) the fixture-validity preconditions of §8, which are checked on
  the *realized* grid before any score is read.

### 2.1 One changed rig constant, declared

`RIG_RADIUS` moves **2.4 → 3.3**. `FOCAL` (420), resolution (400×300),
`N_CAMERAS` (20), elevations (12°, 28°) are unchanged.

The enlarged object does not fit the 400×300 frame at 2.4: the preflight
measures clipping in 11 of 20 cameras, worst normalised frame occupancy
1.428. At 3.3 the worst occupancy is **0.723** (O1) and **0.904** (O2),
against **0.925** for LRV3's own event sphere at its 2.4 — i.e. the new
fixture's framing margin is *at least as generous* as LRV3's. Measured
over both orientations jointly: 3.0 clips 1 camera (occupancy 1.030),
3.1 clips none at 0.984, 3.2 at 0.942, **3.3 at 0.904**. 3.1 is the
smallest 0.1 step with zero clipping; **3.3** is taken so the worst-case
occupancy sits below LRV3's own 0.925 rather than merely inside the
frame.

The object still covers **8.0%–19.1%** of the frame from the four sampled
training cameras (9,556–22,942 front-most pixels), against LRV3's sphere
at 8,570 pixels per held-out view-frame, so observability is not weakened
by the move.

## 3. Voxel resolution — FROZEN

**`cells_per_axis = 8`, `min_group_rows = 4`. Unchanged, and not
negotiable.**

`VOXEL_CELLS_PER_AXIS = 8` is a frozen constant of
`scripts/estimate_episodes.py` (line 157), matching the preregistered
family-seeding grid `configs/elgs/prereg_structural_v1.json`; the module
docstring states that changing it is "a NEW specification, not a flag",
and `build_voxel_groups` is called with the default from
`estimate_episode_program`. `MIN_GROUP_ROWS = 4` is mirrored from
`scene.packet_birth.MIN_PACKET_ROWS` and is checked by
`assert_min_group_rows_single_sourced()`.

**Consequence, stated as a cost.** At 8 cells per axis over a
`[-1.3, 1.3]³` cloud the cell is 0.325, so an object whose notch must be
≥ 2 cells wide is necessarily ~1.28 units across — 3.2× LRV3's sphere
diameter. The enlarged object is a *forced* consequence of holding the
estimator fixed, not a free choice. §12 records what it costs.

A 2× grid (`cells_per_axis = 16`) is evaluated in the preflight **only**
as evidence that the concavity is not a resolution artefact. It is not
used in the experiment.

## 4. The deliberately empty cell — FROZEN

At the fresh seeding grid
(`lo = [-1.299888, -1.299991, -1.299972]`,
`span = [2.599880, 2.599955, 2.599938]`, `cells_per_axis = 8`):

* the event object occupies **24** cells;
* their axis-aligned index bbox is `i[2,5] × j[3,4] × k[2,5]` = **32
  cells**;
* **8** of those 32 have *exactly* zero object volume (verified
  analytically, not by sampling) and *zero* object rows.

Empty-cell keys (`key = i·64 + j·8 + k`):
**284, 285, 292, 293, 348, 349, 356, 357**.

**The designated concavity cell is `(i,j,k) = (4,4,4)`, key `292`.** It is
the empty cell holding the most filler volume.

| quantity | value |
|---|---|
| event-object rows in cell 292 (fresh cloud) | **0** |
| all rows in cell 292 (fresh cloud) | **95** |
| of which filler rows | **57** |
| hull cells spanned by the object | **32** |
| occupied / empty | 24 / 8 |

For comparison, LRV3's sphere spans **8** hull cells with **0** empty
ones, at every resolution.

## 5. Initialization and cloud density — FROZEN

* Initialization cloud: **50,000** points, `numpy.random.default_rng(0)`,
  uniform in `[-1.3, 1.3]³`. Byte-identical construction to LRV1/2/3, so
  the fresh grid is the one quoted in §4.
* Event-object rows in the fresh cloud, measured: **1,589** (O1) / 1,565
  (O2), against LRV3's sphere at ~84. Notch-filler rows: **244** (O1) /
  241 (O2). Rows in the 24 occupied cells: **2,510**; rows in the 8 empty
  cells: **802**, ratio **0.3195**.
* Trained cloud: `densify_until_num_points: 150_000` as in
  `configs/lrv3/a0.yaml`; LRV3's trained cloud was 149,794 rows.
  **The trained row distribution is not predictable and is NOT assumed.**
  §8's preconditions measure it on the realized cloud.
* **Background density inside the concavity — the critical quantity.**
  Trained rows lie on surfaces, so the meaningful proxy is surface supply.
  Measured on a 256³ boundary-shell lattice:

  | | O1 | O2 |
  |---|---:|---:|
  | event-object shell in **occupied** cells | 40,672 (~4.560 area units) | 41,370 (~4.638) |
  | filler shell in **empty (hull)** cells | 11,136 (~1.248) | 15,056 (~1.688) |
  | **bite ratio r = added / base** | **0.2738** | **0.3639** |

  At LRV3's measured areal row density (10,650 object rows over a sphere
  of area 0.5027 → 21,188 rows/unit²), hull completion is projected to add
  ~26,000 (O1) / ~36,000 (O2) wrong rows. Carrying LRV3's own base
  precision of 0.94 through `P' = P/(1+r)` gives **0.7379** (O1) and
  **0.6892** (O2) — both below the 0.80 floor. The falsifier bites.

## 6. Absence and return frames — FROZEN, identical to LRV3

`N_FRAMES = 60`, `FPS = 6.0`, `TIME_DURATION = [0.0, 10.0]`,
`first_return_frame = 57` (the admissible maximum;
`gated_presence_admissible(57)` is True).

* episode 1: frames **0–29**
* gap (event object removed from the world, not occluded): frames **30–56**
* episode 2 (return at the same pose, colour and texture): frames **57–59**

So the authored boundaries the T1 scorer compares against are
`authored_offset = 30`, `authored_onset = 57` — bit-identical to LRV3's,
which is what keeps the boundary estimator's behaviour comparable and
makes any boundary regression attributable to the object's shape rather
than to the schedule.

## 7. Camera split, and the estimator configuration — FROZEN

* `N_CAMERAS = 20`, `TEST_CAMERAS = (2, 7, 12, 17)`, 16 training cameras
  — the LRV3 convention, unchanged.
* Held-out cameras are forbidden as estimator input and the existing
  `LeakageGuard` enforces it (`getTestCameras` raiser, forbidden-path open
  guard, `assert_train_only` identity check, empty event manifests).

**The T1 estimator is reused UNCHANGED.** No constant, threshold or rule
in `scripts/estimate_episodes.py` may be edited for this fixture.

```
python scripts/estimate_episodes.py \
    --config configs/lrv5/a0_ncx.yaml \
    --start_checkpoint <LRV5 A0' substrate checkpoint> \
    --out_report      <run>/episode_estimate_t1_lrv5.json \
    --emit_program    <run>/estimated_program_v2_lrv5.json \
    --membership_mode row_ids \
    --cameras 4 \
    --coarse_stride 4
```

Frozen estimator constants that apply unchanged: `VOXEL_CELLS_PER_AXIS 8`,
`MIN_GROUP_ROWS 4`, `CONTRAST_MAD_MULTIPLE 4.0`, `MIN_MODE_SAMPLES 3`,
`HYSTERESIS_FRACTION 0.25`, `MIN_AGREEING_CAMERAS 3`,
`AGREEMENT_TOLERANCE_FRAMES 1`, `FOOTPRINT_DILATE_PX 4`.

`select_cameras(train_ids, 4)` resolves to sampled cameras
**[0, 5, 10, 15]** (verified in §14), i.e. azimuths 0°, 90°, 180°, 270°.

`configs/lrv5/a0_ncx.yaml` is `configs/lrv3/a0.yaml` with `source_path`
repointed and nothing else changed; `configs/lrv5/a1_est_ncx.yaml` is
`configs/lrv3/a1_est.yaml` likewise. `membership_mode: row_ids` is
required because the scored binding is candidate B's same-cloud binding
(§9).

**Pre-identified: the estimator must NOT accept the concavity on its own.**
The filler is static, so ablating it can only make the render *worse when
the L is absent* — a low→high→low **bump**, never an interior gap.
`detect_gap` requires high→low then low→high and rejects a bump with
`no_interior_gap`. The preflight measures the stronger structural
statement: only **2** of the 4 sampled cameras see the filler change at
all (O1: cam10 85.7%, cam15 64.0% occluded; cam00 and cam05 0.0%), and
`MIN_AGREEING_CAMERAS = 3`, so no boundary can be accepted for a filler
cell even if the shape test were passed. Any acceptance of a concavity
cell by the base rule is therefore a **reportable anomaly**, and §9's
separate base/base+hull scoring keeps it attributable.

## 8. Fixture-validity preconditions — FROZEN, checked BEFORE any score

Declared here, in advance, because LRV4 taught this project that *"a ratio
without its n is not a measurement"*. Each is evaluated on the **realized
trained cloud and its realized grid**, before precision or recall is read.

* **V1 — the concavity exists at the realized grid.** The axis-aligned
  cell-index bbox of the event object's occupied cells must contain
  ≥ 1 cell with **zero** rows satisfying `is_event`.
* **V2 — the concavity is populated.** The union of V1's cells must
  contain ≥ **200** rows, and ≥ 1 of them must individually hold
  ≥ `MIN_GROUP_ROWS = 4` rows (so it is a real group, not substrate).
* **V3 — the operator is not vacuous.** The accepted component must
  contain at least one cell whose object volume comes **only from arm A**
  and at least one whose object volume comes **only from arm B**.
* **V4 — the bite is real.** Rows added by H1 must be ≥ **0.25 ×** rows
  already gated by the base rule.

**If V1, V2, V3 or V4 fails, the fixture is INVALID for this question and
NO verdict on hull completion may be read from the run.** The run is
reported as an invalid instrument, exactly as
[[lrv4-starved-fixture-result-2026-08-23]] was.

**V3 exists because the preflight found a real vacuity mode and it is not
hidden.** H1 is defined per connected component; if T1's accepted
component lies entirely inside one arm, its index bbox never covers the
notch and H1 adds nothing. Measured over every pair of occupied cells, the
minimum number of zero-object cells H1 fills is **0**; measured over
arm-A × arm-B pairs it is **≥ 1** in both orientations (§14). V3 is
the precondition that separates those two regimes. It is a statement
about the *accepted set*, never about the score, so checking it cannot
leak the outcome.

## 9. The exact hull operator under test — FROZEN

Two operators. **H1 is the operator under test**; H2 is a predeclared
secondary, scored and reported in the same table so that the reading is
not ambiguous about which "hull completion" was refuted.

Let `A ⊂ Z³` be the set of accepted cell indices produced by the base rule.

```
# H1 -- AXIS-ALIGNED BOUNDING-BOX FILL OF THE ACCEPTED COMPONENT.
#      This is the operator described in
#      lrv3-membership-candidates-result-2026-08-23 §7 ("A2's COMPONENT
#      occupies 6 of the 8 cells of a contiguous 2x2x2 block, and filling
#      the block's two holes"). Per-component, then unioned.
def H1(A):
    out = empty set
    for C in six_connected_components(A):            # 6-connectivity
        i0, i1 = min/max over C of index 0
        j0, j1 = min/max over C of index 1
        k0, k1 = min/max over C of index 2
        out |= { (i,j,k) : i0<=i<=i1, j0<=j<=j1, k0<=k<=k1 }
    return out

# H2 -- 3x3x3 MORPHOLOGICAL CLOSING, CLIPPED TO THE ACCEPTED BBOX.
#      Predeclared secondary. Dilation and erosion use the full 26-neighbour
#      structuring element (Chebyshev radius 1).
def H2(A):
    D = { c : exists a in A with chebyshev(c, a) <= 1 }        # dilate
    E = { c in D : all n with chebyshev(n, c) <= 1 are in D }  # erode
    return (E | A) & bbox(A)      # extensive, and never grows outward
```

**Base rule, unchanged from the frozen LRV3 spec:** candidate **A2**
(transitive adjacency flood fill under face adjacency + exact onset/offset
agreement + `agreeing_cameras >= 2`) combined with candidate **B**
(same-cloud `row_ids` binding on the trained substrate). A2's transitive
construction guarantees the accepted set is a 6-connected component, so
H1's per-component definition is well posed.

**Attribution rule, frozen.** Precision, recall and false activations are
reported **three times** — for the base rule alone, for base+H1, and for
base+H2 — and the verdict on hull completion is read **only from the
delta**. This is required because rows modelling the L's inner faces may
drift into concavity cells and depress base precision on their own; the
delta is immune to that, an absolute post-hull number is not.

**Metric definitions, frozen.** For gated row set `R` and
`E = { rows p : is_event(p) }` on the same cloud:
`precision = |R ∩ E| / |R|`; `recall = |R ∩ E| / |E|`; a **false
activation** is any gated group containing **zero** rows of `E`. `is_event`
is the closed-box predicate of §1.1 applied to row positions — the exact
analogue of LRV3's `(xyz - centre).norm <= radius`.

## 10. THE GATE — FROZEN BEFORE ANY OUTPUT

Identical floors to [[lrv3-membership-gate-spec-2026-08-23]] §2, plus the
false-activation condition made explicit:

1. **precision ≥ 0.80**
2. **recall ≥ 0.90**
3. **ZERO false activations**

**No floor here may be moved after any score is read.** The gate is not
weakened if the fixture turns out to be hard; if a fair test cannot be
built, the fixture is declared invalid under §8 and the gate stands
untested.

### 10.1 The rejection rule, verbatim

> **If hull completion fills the concavity and violates precision ≥ 0.80,
> OR produces any false activation, hull completion is REJECTED as
> fixture-shaped.**

Expanded so it is unambiguous at reading time, and binding:

* **REJECTED** — base+H1 gates ≥ 1 cell with zero event-object rows AND
  ( base+H1 precision < 0.80 OR base+H1 false activations > 0 ). The rule
  is fixture-shaped: it worked on LRV3 because the sphere is cell-convex,
  and it fails as soon as the object is not. It may not be adopted for
  any membership instrument, on any fixture, and the LRV3 0.9400 / 1.0000
  reading may not be cited as support for it.
* **SURVIVES ROUND 1** — base+H1 meets all three gate conditions **and**
  V1–V4 all held. Not admission: §11 must then run.
* **INVALID** — any of V1–V4 failed. No verdict. The run is recorded as
  an invalid instrument.

H2 is scored under the identical rule and reported separately. A result
in which H1 is REJECTED while H2 SURVIVES is an expected and informative
outcome — the preflight predicts it, since a 3×3×3 closing cannot fill a
notch two cells wide (§14: H1 adds 8 zero-object cells, H2 adds 0).

## 11. Second orientation — PREDECLARED NOW

Declared here, before any result, so it cannot be chosen after seeing the
first one. **If and only if H1 SURVIVES ROUND 1, orientation O2 must be
built, trained and scored under this same spec before hull completion may
be admitted.** If O2 rejects, hull completion is REJECTED.

**O2 is defined by an exact map, not by a description.** Apply to every
event box and every filler box of §1.1–§1.2:

```
(x, y, z)  ->  (-x + 0.1625,  y,  z + 0.1625)
```

i.e. a **mirror through the plane x = 0** followed by a translation of
**half a fresh cell** (0.325 / 2 = 0.1625) in x and z. Static spheres,
ground, cameras, frames and the estimator configuration are unchanged.
The mirror changes the object's chirality relative to the camera ring; the
half-cell translation changes the concavity's phase within the grid. The
resulting boxes, already evaluated by the preflight:

| | x | y | z |
|---|---|---|---|
| Arm A′ | `[+0.4025, +0.8025]` | `[-0.30, +0.30]` | `[-0.4775, +0.8025]` |
| Arm B′ | `[-0.4775, +0.8025]` | `[-0.30, +0.30]` | `[-0.4775, -0.0775]` |
| Wall F1′ | `[-0.3575, +0.2825]` | `[-0.30, +0.30]` | `[+0.3025, +0.4225]` |
| Wall F2′ | `[-0.0975, +0.0225]` | `[-0.30, +0.30]` | `[+0.0425, +0.6825]` |

O2's designated concavity cell is `(3,4,5)`, key **229**: 0 object rows,
107 total rows, 49 filler rows. Hull 50 cells, 32 occupied, **18** empty.

## 12. Ways this fixture is weaker than intended — declared

1. **The object is much larger and much more cell-filling than LRV3's.**
   Volume fraction within its hull cells is **47.20%** (O1) / **30.21%**
   (O2) against LRV3's **12.20%** at the fresh grid and **6.60%** at the
   trained grid. So LRV5 tests **cell-convexity specifically**; it does
   *not* reproduce LRV3's regime in which hull completion over-reaches by
   15× in volume. A rule that survives LRV5 has been shown non-fixture-
   shaped *with respect to shape*, not with respect to over-reach
   magnitude. This is forced by `cells_per_axis = 8` (§3) and cannot be
   fixed without changing the estimator.
2. **The trained-cloud grid is not controllable.** Everything in §4 is
   computed at the *fresh* grid. The realized grid is checked only by the
   §8 preconditions, so a run can end INVALID after paying for training.
   The sweep in §14 (P3) bounds the risk but does not eliminate it.
3. **H1 has a genuine vacuity mode** (an accepted component confined to
   one arm). V3 catches it, but V3 can only be evaluated after training.
4. **The filler makes the scene busier than LRV3's.** The concavity is
   occupied by design, so the L's "hole" is visually a slot rather than an
   open corner, and the notch contents are partially occluded from ~2 of
   the 4 sampled cameras. That occlusion is what guarantees the filler
   abstains (§7), but it also means the filler is reconstructed from fewer
   views than the rest of the scene.
5. **The bite ratio is a projection, not a measurement.** `r = 0.2738`
   (O1) is a surface-shell proxy; the trained-row translation uses LRV3's
   areal density as a constant. V4 replaces the projection with a
   measurement before any verdict is read.
6. **`RIG_RADIUS` changed** 2.4 → 3.3 (§2.1). Every rendered quantity is
   therefore not directly comparable to a recorded LRV3 number; only
   membership precision/recall, which are geometric, transfer.
7. **The result will be about the OPERATOR, not about membership.** A
   rejection tells us hull completion is fixture-shaped; it does not
   supply a membership instrument that clears the gate. The structural
   recall cap measured on LRV3 is untouched by this experiment.

## 13. Permitted and forbidden readings

**Permitted after the run.** Whether axis-aligned cell-hull completion of
the accepted component gates cells the event object does not occupy, on a
connected non-convex object; the precision and false-activation
consequence of doing so; whether a 3×3×3 closing behaves differently.

**Forbidden.** That LRV5 establishes anything about membership *recall*
(it does not test the estimator's sensitivity cap). That a SURVIVES
verdict admits hull completion (O2 must run first, §11). That any LRV3
number is retracted. That this fixture's absence transfers to real data —
LRV5's absence, like LRV3's, is a clean ray-trace removal.

## 14. Preflight evidence

Script: `nonconvex_preflight.py` (numpy 1.26.4, no torch, no GPU).
Exit code **0**; every check PASS for both orientations. Verbatim output
is in `preflight_output.txt` alongside this document. Headline lines:

```
fresh seeding cloud: 50000 rows, seed 0
  grid lo   = [-1.299888 -1.299991 -1.299972]
  grid span = [2.599880 2.599955 2.599938]
  cell size = [0.324985 0.324994 0.324992]  (cells_per_axis 8)

[P1] O1 8^3 : 24 occupied cells, 1 six-connected component        PASS
     O1 16^3: 156 occupied cells, 1 six-connected component       PASS
[P2] cell hull i[2,5] j[3,4] k[2,5] = 32 cells | occupied 24 | EMPTY 8
     max object VOLUME over empty cells = 0.000e+00               PASS
     max object ROWS   over empty cells = 0                       PASS
     H1 bbox-fill   adds 8 zero-object cells (keys 284,285,292,293,
                                              348,349,356,357)
     H1 over ARM-A x ARM-B pairs: MIN zero-object cells filled = 1 PASS
     H1 over ALL occupied pairs : MIN = 0   <-- declared vacuity mode (V3)
     H2 3x3x3 close adds 0 zero-object cells
[P3] 2x resolution (16^3): hull 256, occupied 156, EMPTY 100      PASS
     sweep cell 0.30/0.325/0.35/0.40/0.45/0.50 x 100 phases each:
        100/100 configs with >=1 empty in-hull cell at EVERY size PASS
[P4] object volume 0.518400 | 32 hull cells x 0.034325 = 1.098404
     LRV5 volume fraction 0.4720 (47.20%)
     LRV3 sphere, fresh grid 0.1220 | LRV3 recorded, trained grid 0.0660
[P5] designated concavity cell (4,4,4) key 292
     object rows 0 | all rows 95 | filler rows 57                 PASS
     fresh-cloud rows inside the event object 1589 | inside the filler 244
     rows in 8 empty cells / 24 occupied cells = 802 / 2510 = 0.3195
     surface proxy: base 40672 shell sites, added 11136, r = 0.2738
     projected precision after H1 from base 0.94 -> 0.7379        PASS
[P5b] sampled train cameras = [0, 5, 10, 15]
     filler occluded by the event object: cam00 0.0%, cam05 0.0%,
                                         cam10 85.7%, cam15 64.0%
     -> 2 changing cameras < MIN_AGREEING_CAMERAS = 3             PASS
     event object front-most px: 9556 / 13626 / 20492 / 22942 (8.0-19.1%)
     filler px cam10: 679 present -> 5993 during the gap
framing at RIG_RADIUS 3.30: worst occupancy O1 0.723, O2 0.904    PASS
     reference LRV3 sphere at 2.40: 0.925, 0 cameras clipped
O2: 32 occupied, hull 50, EMPTY 18 | designated (3,4,5) key 229
     object rows 0 | all rows 107 | filler rows 49 | r = 0.3639
     projected precision after H1 from base 0.94 -> 0.6892        PASS
```

**Iterations.** The geometry of §1 was not revised: it passed P1–P5 on the
first execution. Two revisions were made to the *preflight's own
diagnostics*: (a) the minimal-accepted-set probe initially used an
orientation-specific heuristic for the arm tips and mis-selected them
under O2, and was replaced by an exhaustive minimum over all occupied-cell
pairs and over all arm-A × arm-B pairs — which is what surfaced the
vacuity mode now frozen as precondition V3; (b) the framing check
initially bundled the static context spheres with the event object, which
is the wrong requirement (LRV3's ground plane also leaves frame) and was
split, after which `RIG_RADIUS` was set from the measured minimum.
