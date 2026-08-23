# RESULT — the T2 membership failure is measured: 4.5% precision at the
# acting moment, and non-oracle recall is HARD-CAPPED at 0.18
# (2026-08-23, block 2)

EXPLORATORY, `evidence_bearing: false`. **Zero GPU-hours** — every number
below is recomputed on the workstation from primary artifacts. This is
the per-row membership scoring that
[[nonoracle-timing-t2-result-2026-08-23]] §6 pre-identified as the next
step, executed in the ordering it demanded: **score the instrument
against the authored sphere BEFORE any retraining.**

Ground truth: `EVENT_SPHERE_CENTRE = (0.70, 0.10, 0.35)`,
`EVENT_SPHERE_RADIUS = 0.20`, from `data/synthetic/lrv3/event_spec.json`.

## 1. There are TWO membership measurements and only one of them acted

This is the distinction the T2 page did not separate, and it explains an
apparent contradiction.

* The **substrate** cloud is experiment 184's trained A0′ checkpoint,
  149,794 rows. The T1 estimator ran on it.
* The **fresh seeding** cloud is `data/synthetic/lrv3/points3d.ply`,
  50,000 rows. Experiments 245/246 trained from scratch, so
  `resolve_v2_membership` re-applied the program's **absolute
  world-space voxel grid** to this cloud
  (`elgs/trainer_hooks.py`, `membership_mode: spatial_voxel`).

**The fresh cloud is the one that bound membership at seeding, so it is
the one that acted.**

Provenance verified: the substrate ply's bounding box reproduces the
program's `spatial.lo`/`span` exactly, and re-running the grouping
reproduces the recorded T1 per-group numbers bit-for-bit — group 291 =
1,213 rows / 1,167 in sphere, group 292 = 2,412 / 2,345.

## 2. The numbers

| | substrate (149,794 rows) | **fresh seeding cloud (50,000 rows)** |
|---|---:|---:|
| rows inside the authored sphere | 10,650 | **84** |
| gated rows | 3,625 | **336** (291: 155, 292: 181) |
| gated ∩ sphere | 3,512 | **15** |
| **precision** | **0.9688** | **0.0446** |
| **recall** | **0.3298** | **0.1786** |

The fresh cloud's 84 in-sphere rows are exactly the "~84 rows" the
oracle A1-LOCAL cell gated — **the oracle membership IS the sphere test**,
which confirms the two instruments are being compared on the same
quantity.

**A correction to a reading made earlier in this block.** The substrate's
96.9% precision looks like it contradicts the T2 page's over-gating
attribution. It does not: it is measured on a different cloud and
describes nothing that happened during training. On the trained substrate
densification has already concentrated 10,650 rows onto the object, so
cells 364/365 are 96.9% object; on the fresh cloud the same two cells are
**95.5% background**. **The T2 page's attribution was right.**

## 3. The 8 object-overlapping groups, and why the 6 were missed

| group | cell | rows | in sphere | gated | cams | abstain reason |
|---:|---:|---:|---:|---|---:|---|
| 292 | 365 | 2,412 | 2,345 | **YES** | 4 | — |
| 285 | 357 | 2,040 | 1,815 | no | 2 | `camera_disagreement` |
| 340 | 421 | 1,547 | 1,469 | no | 2 | `camera_disagreement` |
| 347 | 429 | 1,487 | 1,453 | no | 0 | `no_interior_gap` |
| 291 | 364 | 1,213 | 1,167 | **YES** | 4 | — |
| 284 | 356 | 1,154 | 1,049 | no | 2 | `camera_disagreement` |
| 346 | 428 | 822 | 769 | no | 2 | `camera_disagreement` |
| 339 | 420 | 655 | 583 | no | 0 | `no_interior_gap` |

Four of the six missed groups abstained at **exactly 2 agreeing cameras
against a required 3**; two saw no interior gap at all. The eight
in-sphere counts sum to 10,650 exactly, so no in-sphere row lands in a
sub-threshold cell.

## 4. Fixing recall makes over-gating WORSE — the two limbs are one defect

Gating all 8 overlapping cells on the fresh cloud gives perfect recall
and precision **84 / 1,471 = 0.0571**, i.e. 17.5× the oracle's row count.
The ceiling is geometric: the sphere's volume is **0.03351** against the
8 cells' **0.50724** — **6.6%**. An 8³ cell over this cloud's bounds is
0.404 × 0.383 × 0.410 against an object of diameter 0.40.

At the binding moment the gate held **321 wrongly-gated background rows
against 69 missed object rows** — the over-gating error is 4.7× the
under-gating error in rows. Under-recall is therefore **not a separately
repairable limb**.

## 5. Better membership GEOMETRY works, and it does not help

Since the two gated cells are 96.9% object *on the substrate*, the
obvious repair is to carry membership as the substrate's own row
geometry rather than as voxel cells. Measured on the fresh cloud, scored
against the authored sphere:

| instrument (source = substrate rows in the 2 GATED cells) | gated | precision | recall |
|---|---:|---:|---:|
| the 2 voxel cells — **what actually ran** | 336 | 0.0446 | 0.1786 |
| bounding box of the substrate rows | 300 | 0.0500 | 0.1786 |
| occupancy grid 32³ | 154 | 0.0974 | 0.1786 |
| **occupancy grid 64³** | **56** | **0.2679** | **0.1786** |
| occupancy grid 128³ | 33 | 0.3636 | 0.1429 |
| occupancy grid 256³ | 12 | 0.5833 | 0.0833 |

Precision improves **6×**. **Recall does not move, because it cannot.**

> **RECALL CEILING: only 15 of the fresh cloud's 84 in-sphere rows lie
> inside the two gated cells at all — 0.1786.** The other 69 object rows
> are in the six cells the estimator abstained on. No geometric
> refinement of the two gated cells can reach them; they are spatially
> elsewhere.

So every repair available downstream of the current estimator lands in
**partial membership** — the regime this project has already measured as
worse than both alternatives: **fully gated 28.19 > NOT gated 27.14 >>
partially gated 24.67**.

**Recorded so it is not mistaken for an available option:** an occupancy
grid sourced from all 8 overlapping cells reaches precision 0.2968 at
recall 1.000. That source is **oracle-supervised** — only the sphere test
identifies which 8 cells overlap the object — so it is a ceiling, not an
instrument.

Two further bounds worth carrying. Even the sphere's **own** bounding
cube, handed over by the oracle, reaches only **0.4746** precision on the
fresh cloud, so axis-aligned-box membership is structurally weak here.
And membership binds once at seeding: `_elgs_family_ids` are inherited by
clone/split (`scene/gaussian_model.py:1967-1976`) and decremented on
prune, so a 4.5%-precision seed is **amplified** by densification — the
final gated sets are 4,317 (exp 245) and 4,576 (exp 246) rows.

## 6. Verdict, and the decision it forces

**The 8³ voxel-cell membership is refuted, and the binding constraint is
upstream in the ESTIMATOR's recall, not in the membership geometry.**

**No clear pre-output repair exists, so no retraining was run.** The only
route to recall is relaxing the camera-agreement bar from 3 to 2, which
would admit the 4 missed object groups **and** whatever else among the
415 abstainers sits at 2 cameras — destroying the T1 result's
zero-false-activation property, while still leaving precision at the 6.6%
geometric ceiling. That is a threshold change made after seeing a null,
it is not clearly a repair, and it was not made.

**What survives intact:** T1's boundary output. The estimated gaps match
the oracle's to ~1e-14, and nothing here touches them. **Boundary
inference is solved on this fixture; membership inference by spatial
partition is refuted.**

## 7. Provenance

Substrate `point_cloud.ply` pulled from experiment 184's run dir
(37,150,443 bytes) and byte-verified against the program's recorded
`cloud.xyz_sha256`. Program `configs/lrv3/estimated_program_v2.json`
(local, byte-identical to the copy in exp 184's run dir). Fresh cloud
`data/synthetic/lrv3/points3d.ply` (local). Abstention reasons from
`episode_estimate_t1.json`, sha256 `4dc6f085…`. The voxel binding is a
line-for-line numpy transcription of
`elgs.trainer_hooks.resolve_v2_membership`'s `spatial_voxel` branch,
validated by reproducing the recorded T1 per-group counts exactly.
