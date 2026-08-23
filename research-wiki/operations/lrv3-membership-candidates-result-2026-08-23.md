# RESULT — no membership instrument clears the frozen gate; the shortfall
# is STRUCTURAL, and the false-positive control is CLEAN
# (2026-08-23, block 3)

EXPLORATORY, `evidence_bearing: false`. **Zero GPU-hours** — CPU scoring
on primary artifacts. Gate, both Candidate A variants, Candidate B and
the required false-positive control were frozen and committed at
`2789fef` **before any score below was computed**:
[[lrv3-membership-gate-spec-2026-08-23]].

## 1. The gate, as frozen

recall ≥ **0.90** · precision ≥ **0.80** · non-degeneracy (must not
reduce to the oracle sphere test, must not consume held-out cameras).
The recall floor is a declared judgment, labelled as one in the spec.

## 2. Scores — each on the exact cloud it would bind to

| instrument | cloud | cells | rows gated | precision | recall | gate |
|---|---|---:|---:|---:|---:|---|
| 2 cells — *the configuration that failed* | fresh | 2 | 336 | 0.0446 | 0.1786 | — |
| **A1** single-pass | fresh | 5 | 899 | 0.0667 | 0.7143 | **FAIL both** |
| **A2** transitive | fresh | 6 | 1,109 | 0.0667 | 0.8810 | **FAIL both** |
| **B alone** (`row_ids`) | trained | 2 | 3,625 | **0.9688** | 0.3298 | **FAIL recall** |
| **A2 + B** | trained | 6 | 9,188 | **0.9375** | **0.8088** | **FAIL recall** |
| *(ceiling: all 8 cells + B)* | trained | 8 | 11,330 | 0.9400 | 1.0000 | *would pass, but oracle-derived* |

**Nothing clears the gate.** The pre-authorized A2+B combination fails on
recall at **0.8088 against 0.90**.

## 3. The false-positive control — CLEAN, and this is a genuine positive

Run over **all 417 groups** as required, not the 8.

| | A1 | A2 |
|---|---:|---:|
| newly accepted groups | 284, 285, 346 | 284, 285, 340, 346 |
| **newly accepted NON-object groups** | **0** | **0** |
| **false activations** | **0** | **0** |
| abstention | 98.80% | 98.56% (T1: 99.52%) |

**Relaxing camera support from 3 to 2 created zero false activations.**
That is not luck and the reason is measurable: **exactly 4 of the 417
groups satisfy the boundary-agreement conjunction (`onset == 57` AND
`offset == 30` AND `agreeing_cameras >= 2`), and all 4 lie inside the
object's 2 × 2 × 2 block.** The adjacency condition is therefore
*redundant here* — the boundary agreement alone is already fully
specific. T1's zero-false-activation property survives the relaxation
intact, and only 0.96 percentage points of abstention are given up.

**So the contestable choice recorded in the spec — setting camera support
to 2 knowing the missed cells sit at 2 — was adjudicated by its own
control and passed.** The rule is not a threshold relaxation that buys
recall with false positives.

## 4. Why A2+B fails — the cap is STRUCTURAL and was predicted before scoring

A2 admits **every cell it is possible to admit**. The measured recall
0.8088 equals the *ceiling* exactly:

| unreachable cell | in-sphere rows | share of object |
|---|---:|---:|
| group 339, cell 420 (6,4,4) | 583 | 5.47% |
| group 347, cell 429 (6,5,5) | 1,453 | 13.64% |
| **total** | **2,036** | **19.12%** → ceiling 0.8088 |

Both carry `abstain_reason: no_interior_gap` with **0 agreeing cameras
and no onset/offset estimate at all**. No boundary-agreement rule can
ever admit them — the estimator found no absence signal in them. The
spec recorded this cap *before* scoring; the measurement confirmed it to
four decimals.

## 5. What the separate evaluation of B alone bought

The T2 failure confounded low precision with low recall. **B alone
isolates them**: precision **0.9688** — a 21.7× improvement over the
0.0446 that failed — at recall 0.3298. It clears the precision floor
comfortably and fails recall outright.

**So recall is decisively the binding variable.** That is the finding the
separate evaluation existed to produce, and it is now measured rather
than inferred. Precision is a solved problem the moment membership stops
crossing the densification boundary; coverage is not.

## 6. The two limbs, now cleanly separated

* **Precision is a CLOUD-BINDING problem.** Same cells, same rule:
  0.0446 on the fresh cloud, 0.9688 on the trained cloud. Binding
  membership to the cloud it was inferred on fixes precision entirely.
* **Recall is an ESTIMATOR-SENSITIVITY problem.** Two of eight object
  cells produce no absence signal whatsoever, and nothing downstream can
  manufacture one.

These are independent, and no instrument tested closes both.

## 7. Recorded but NOT adopted — and why

An axis-aligned **hull-completion** step would admit 420 and 429: A2's
component occupies 6 of the 8 cells of a contiguous 2 × 2 × 2 block, and
filling the block's two holes is a purely spatial, non-oracle rule that
would reach recall 1.0000 at precision 0.9400.

**It is not adopted, and the reason is the ordering rule this project
runs on.** It was identified *after* seeing that A2+B fell 0.09 short of
the recall floor. Adopting it now would be selecting a rule because it
clears a gate the previous rule missed — the same post-hoc adjustment
refused for the LRV4 floor. It is recorded here as a candidate for a
**future frozen spec**, with its post-hoc origin disclosed, and it must
be frozen with its own false-positive control before any score is read.

Its known weakness, stated now so a future spec inherits it: hull
completion assumes the object is cell-convex. That is true of a sphere
and need not be true of anything else, so it is a fixture-shaped rule
until tested on a non-convex object.

## 8. Permitted and forbidden

**Permitted.** No membership instrument tested clears the frozen gate.
Precision is fixed by same-cloud binding (0.0446 → 0.9688). Recall is
capped at 0.8088 by two cells with no absence signal. The boundary
agreement conjunction is fully specific on this fixture — zero false
activations among 417 groups.

**Forbidden.** That non-oracle membership is impossible; that A2+B is
"close enough" (the gate was frozen first and it fails); that hull
completion works (unscored under a frozen spec, post-hoc in origin); that
any of this transfers off LRV3, whose absence is a clean ray-trace
removal.

## 9. Provenance

Substrate `point_cloud.ply` from experiment 184 (37,150,443 bytes),
verified against the program's recorded `cloud.xyz_sha256`. The 417-group
construction was reproduced from it exactly (`build_voxel_groups`,
`cells_per_axis 8`, `min_group_rows 4`), and the program's
`spatial.lo`/`span` verified equal to that cloud's bounding box. Fresh
cloud `data/synthetic/lrv3/points3d.ply` (local). Per-group `gated`,
`onset_frame`, `offset_frame`, `agreeing_cameras` and `abstain_reason`
read from `episode_estimate_t1.json`, sha256 `4dc6f085…`. **No
retraining was run.**
