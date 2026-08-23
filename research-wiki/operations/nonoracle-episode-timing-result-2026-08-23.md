# RESULT — non-oracle episode boundaries recovered EXACTLY on LRV3;
# phase T1 PASSES (2026-08-23)

EXPLORATORY, `evidence_bearing: false`. Design and gate frozen before
output in [[nonoracle-episode-timing-spec-2026-08-23]]; nothing there
moved. Cell: Determined experiment **235** (`lrv3_episode_estimate_t1`
r0, commit `2ba6a62`, pool `dgx`, admitted V100 image
`sha256:70a28e3d…`, seed 0), COMPLETED. Report
`episode_estimate_t1.json` in experiment 184's run dir, sha256
`4dc6f085a1c999726c885890db56ed732ead5144f8d319e44a564452adc2ae12`,
frozen program sha256 `3f85c5c5d4693161…`.

## 1. The headline

**A training-view-only estimator placed both episode boundaries EXACTLY
right, and abstained everywhere else.**

| quantity | measured |
|---|---:|
| candidate groups (8³ voxel grid over the cloud's own bounds) | 417 |
| groups GATED | **2** |
| groups abstained | 415 (**99.52%**) |
| **onset error** | **0 frames on both** (authored 57, estimated 57) |
| **offset error** | **0 frames on both** (authored 30, estimated 30) |
| **false activations** | **0** |
| groups overlapping the event object | 8 |
| estimation cost | 677.9 s = **0.188 slot-h**, 240 base + 63,120 ablated renders at 10.7 ms |

Both gated groups sit on the event object and carry 4-of-4 camera
agreement against a requirement of 3:

| group | rows | rows inside the event sphere | offset | onset | agreeing cameras |
|---:|---:|---:|---:|---:|---:|
| 291 | 1,213 | 1,167 | 30 | 57 | 4 |
| 292 | 2,412 | 2,345 | 30 | 57 | 4 |

All 60 frames were evaluated (16 coarse at stride 4, plus 44 fine), so
**no boundary was interpolated** — the requirement that frame accuracy
never be inferred between samples was met by construction.

## 2. The frozen gate, applied

1. For the groups corresponding to the event object,
   `|onset error| ≤ 1` AND `|offset error| ≤ 1` — **PASS**, both are
   exactly 0.
2. No ACCEPTED group has a boundary error ≥ 2 frames — **PASS**,
   `max_abs = 0` across all accepted boundaries.
3. Anti-leakage assertions all held — **PASS** (§3).

**Phase T1 PASSES. Phase T2 — schema v2, a computed-program branch in
seeding, and A0 / A-oracle / A-est training cells — is now justified.**

## 3. Anti-leakage, verified in the artifact rather than asserted

The report records every check as passed: `get_test_cameras_disabled`,
`cameras_are_train_split_objects`, `forbidden_path_open_guard`,
`event_manifests_empty`, `audit_hook_installed`, plus
`estimation_saw_ground_truth: false` and
`program_frozen_before_scoring: true`.

Independently confirmed from the sampling block: the cameras used were
**0, 5, 10, 15**, all members of the training split
`[0,1,3,4,5,6,8,9,10,11,13,14,15,16,18,19]`. The held-out cameras
**2, 7, 12, 17** — the only ones with identity buffers on disk — were
never touched.

The hazard that would have faked this result was inheriting the oracle
`region` for membership, since that sphere IS the event object's true
geometry. It was not inherited: membership came from a voxel grid over
the trained cloud's own bounding box, and the report records
`method: "voxel_grid_over_cloud_bounding_box"`.

## 4. Where the specificity actually comes from

Abstention reasons across the 415 abstaining groups: **contrast 224,
no interior gap 125, empty footprint 39, camera disagreement 27**.

This matches the synthetic characterization measured before the run and
recorded in the commit that froze the amended rule: the contrast test
admits ~30% of pure noise at threshold 4 and is NOT the component
providing specificity; the shape test (requiring a high→low→high
interior gap) and the 3-of-4 camera agreement are. Here the shape test
removed 125 groups and camera disagreement a further 27 — and **zero
false activations survived**, against a pre-run prediction of under
0.015 expected across the whole group set.

The rule was left at multiple 4 rather than tuned after that
measurement. The outcome vindicates leaving it: raising it would have
traded real sensitivity for redundancy that was never binding.

## 5. What this establishes, and what it does NOT

**Established.** On LRV3, episode onset and offset are recoverable to
EXACT frame accuracy from training views alone, with membership derived
without any oracle, and with an abstention mechanism that produced zero
false activations. This is the first evidence against the standing
concern that inferred timing must be too imprecise to be usable — a
concern that was well-founded, because the measured mistiming control
puts a 2-frame error at −2.39 dB, *below* not gating at all.

**NOT established, and the gap is the important part.**

* **Recall is 2 of 8.** Six groups that overlap the event object
  abstained. The two that gated hold 3,512 of the in-sphere rows between
  them, so a substantial share of the object is covered — but this is
  not a complete segmentation of the event, and the estimator is
  currently a high-precision, low-recall instrument. That is the right
  direction for a mechanism whose errors are known to be worse than
  inaction, but it is a limitation, not a feature to be claimed.
* **No reconstruction claim follows.** T1 measured BOUNDARY ACCURACY
  against authored ground truth. It did NOT retrain anything, so nothing
  here shows that gating on the estimated program reproduces the
  +1.05 dB the oracle program achieved. That is exactly what phase T2
  is for, and until T2 runs, the localized-presence positive still rests
  on authored boundaries.
* **One fixture, one seed, one substrate.** LRV3's absence is genuine —
  the object is removed from the ray-trace, not occluded — so the
  ablation signal is a clean step. On real data the equivalent signal is
  an occlusion, which is not the same measurement. Nothing here
  transfers to N3V.
* The estimator ran on the A0′ ordinary-temporal substrate
  (experiment 184), which is the state that would exist at inference
  time. Whether the result survives on a differently-trained substrate
  is untested.

## 6. Phase T2 comparator validity — VERIFIED, not assumed

T2 will compare an A-est arm against the RECORDED A0′ (experiment 184)
and A1-LOCAL (experiment 185) numbers, both trained at commit `b7952b0`.
Since then the training path has gained **1,959 insertions across 8
files**, so reuse cannot be assumed. Re-verified line by line:

* **`scripts/eval_lrv1_event.py` — the evaluator that produced the
  +1.0496 dB figure — changed by 10 insertions and ZERO deletions.** The
  addition is a new reported quantity (`ghost_gap_psnr_by_frame`) built
  on its own fresh `Region()` instance; no existing accumulator is
  touched. **Every existing region is bit-identical.**
* **`main.py` has only 3 removed lines**, all inside the `validation()`
  signature and its training-time `psnr` call — the `--val` path, which
  Lane T does not use. LRV3 training is untouched.
* `utils/mesh_utils.py` changed the **`--val`** PSNR from channel-split
  to pooled. That repair does NOT reach this comparison:
  `eval_lrv1_event.py` pools squared error over channels inside
  `Region.add` and never had the channel-split bias.
* `utils/image_utils.py` changed `.view` to `.reshape`, a
  non-contiguity fix that is numerically identical where `.view` was
  legal.
* Everything else added — `scene/packet_birth.py`,
  `scene/packet_birth_flow.py`, `scene/appearance_edit.py`, and the
  `get_features`/`get_opacity` redirects — is flag-gated. Confirmed from
  the configs: `configs/lrv3/a1_local.yaml` and
  `configs/lrv3/a0_local_control.yaml` set no `packet_birth_*` key and no
  appearance-edit key, and the redirects require an
  `_appearance_source_idx` column that training never installs.

**Conclusion: A-est trained at the current commit is comparable to the
recorded A0′ and A1-LOCAL figures**, provided it is scored with
`scripts/eval_lrv1_event.py` and its existing regions — not with
`main.py --val`, whose convention did move.

## 7. Phase T2 seeding — observed BEFORE any A-est result, and it inverts the expected risk

From experiment 245's seeding log at iteration 0:

```
program_schema  adags-episode-program-v2   membership_mode  spatial_voxel
families 2   gated_rows 336   rows 50000   unassigned_rows 49664
v2_group_rows {291: 155, 292: 181}   v2_group_K {291: 2, 292: 2}
```

**The estimated program gates 336 rows in 2 families. The ORACLE gated
~84 rows in 8 families** ([[lrv3-local-presence-corrected-cell-2026-08-20]]).
So A-est gates roughly **four times MORE** rows than the oracle, not
fewer.

The arithmetic explains it and was predictable: the voxel grid is 8 cells
per axis over a span of ~3.23, so each cell is a ~0.40 cube — volume
~0.065 — and two cells give ~0.13, against the oracle sphere's
`4/3·π·0.2³ ≈ 0.034`. **The two cells enclose ~3.8× the sphere's volume,
matching the observed 336/84 ≈ 4× row ratio.**

**This inverts the risk this page flagged in §5.** The concern recorded
there was low recall making A-est UNDERPOWERED. The seeding shows the
opposite: coarse voxel membership OVER-gates, capturing background and
ground rows that sit inside the cells but are not the event object.
Those rows will be driven to exact absence during the gap, suppressing
content that should be present.

T1 measured **zero false activations at the GROUP level**; this is the
distinct, finer question of false membership at the ROW level within an
accepted group, which T1 did not measure and could not.

**Consequence for reading A-est, recorded in advance:** with timing
numerically identical to the oracle's, a shortfall against A1-LOCAL is
attributable to MEMBERSHIP — and specifically to over-gating rather than
under-gating. A per-row membership precision figure against the authored
sphere is the diagnostic that would quantify it, and it is not part of
this cell.

## 8. Bookkeeping

Claim consumed: `lrv3_episode_estimate_t1` r0 (experiment 235). Cost
**0.188 slot-h** measured. Grouping: 417 groups over 149,794 rows,
`min_group_rows` 4, 8³ cells. Decision rule as frozen:
within-mode separation ≥ 4 × within-mode MAD, `min_mode_samples` 3,
hysteresis fraction 0.25, ≥ 3 agreeing cameras within ±1 frame,
footprint dilation 4 px.
