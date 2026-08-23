# HANDOVER — 2026-08-23 block 3

Self-contained. EXPLORATORY throughout, `evidence_bearing: false` on
every cell. **Nothing is running at handover.**

## 1. State

| item | value |
|---|---|
| branch | `apollo/csvl-vpl-v2-exploratory` |
| local == origin | `3f6ad1e` at the time of writing; final SHA in the block's last commit |
| block start | `877c4f0` |
| protected files | `research-wiki/deep-dive-prompt.txt`, `run-deep-dive.ps1` — untracked, untouched |
| other outstanding | `overnight-handover-23aug.md` (user-owned, not touched) |

## 2. The two decisive results

**(A) The N3V 50-frame protocol cannot resolve its own deltas**
([[same-code-replicate-floor-spec-2026-08-23]] RESULT). Three
byte-identical cells — commit, image digest, pool, seed,
`config_canonical_hash` and `archive_sha256` all verified equal *before*
any metric was read — give an event-union spread of **R = 0.4945 dB**,
**1.45× the 0.341 dB seed spread**, and a pooled+clamped PSNR spread of
0.4017 dB. The frozen top row fires.

**Every recorded event-union ladder delta is inside that floor** — B1
+0.077/+0.345, B1-D −0.145/−0.313, B1-F, B1-X, all of them. No number is
retracted; what changes is what they resolve, which is nothing. Region A
alone spreads **1.41 dB**. The floor *exceeds* the seed spread, so
fixed-seed run-to-run variation is the dominant term.

**N3V utility scaling is HALTED** pending protocol redesign. The
LRV3/LRV4 fixture lanes are explicitly **not** gated by this.

**(B) No membership instrument clears the frozen gate**
([[lrv3-membership-candidates-result-2026-08-23]]; gate frozen at
`2789fef` *before* scoring).

| instrument | cloud | precision | recall | gate |
|---|---|---:|---:|---|
| 2 cells (what failed) | fresh | 0.0446 | 0.1786 | — |
| A1 single-pass | fresh | 0.0667 | 0.7143 | FAIL both |
| A2 transitive | fresh | 0.0667 | 0.8810 | FAIL both |
| B alone (`row_ids`) | trained | **0.9688** | 0.3298 | FAIL recall |
| **A2 + B** | trained | **0.9375** | **0.8088** | **FAIL recall** |

The A2+B shortfall is **structural and was predicted before scoring**:
groups 339 and 347 carry no onset/offset estimate and 0 agreeing
cameras, hold **19.12%** of the object, and cap recall at exactly
0.8088 — which A2 attains to four decimals, so it admits every
admissible cell.

**The false-positive control is CLEAN**: zero false activations for both
variants across all 417 groups, abstention 99.52% → 98.56%. Exactly 4
groups satisfy the boundary-agreement conjunction and all 4 lie in the
object block, so adjacency is redundant and the contestable `cams >= 2`
choice was adjudicated by its own control and passed.

**B alone isolates what T2 confounded:** precision is a **cloud-binding**
problem (0.0446 fresh → 0.9688 trained, same cells); recall is an
**estimator-sensitivity** problem. **Recall is decisively the binding
variable.**

## 3. Verified from primary artifacts

The prompt's `2 × 2 × 2` decode is **confirmed**: the 8 sphere-overlapping
cells are exactly `x ∈ {5,6}`, `y ∈ {4,5}`, `z ∈ {4,5}`. The 417-group
construction was reproduced from experiment 184's `point_cloud.ply`
exactly, and the program's `spatial.lo`/`span` verified equal to that
cloud's bounding box.

**New load-bearing fact:** four of the six rejected cells report *exactly*
the accepted onset 57 and offset 30 — they were rejected on camera
support (2 vs 3), not on disagreeing boundaries. The other two have no
boundary estimate at all.

## 4. Recorded but NOT adopted

**Hull completion** would reach recall 1.0000 at precision 0.9400 by
filling the two holes in A2's 2 × 2 × 2 block — a purely spatial,
non-oracle rule. **Not adopted**: it was identified *after* seeing A2+B
miss by 0.09, so adopting it now is the post-hoc adjustment refused for
the LRV4 floor. Recorded for a future frozen spec with its origin
disclosed, and with its known weakness stated: it assumes the object is
cell-convex, which is true of a sphere and need not be true otherwise.

## 5. Cost

| cells | slot-h |
|---|---:|
| 261/262/263 replicate training | 7.50 |
| 264/265/266 `--val` evaluation | 0.09 |
| 259/260 LRV4 `lo` diagnostics | 0.02 |
| all membership scoring (CPU, local) | **0** |
| **total** | **≈7.61** against a 24 slot-h ceiling |

## 6. Open blockers

* **N3V protocol redesign** is the binding blocker for the paper's
  real-data claim. Options not yet costed: more replicates per arm, a
  longer schedule, a lower-variance endpoint, or paired evaluation on
  identical densification trajectories.
* **A question this block raises but does not answer:** the LRV3 family's
  recorded 0.09–0.17 dB same-arm spread predates this measurement, and
  the A-est arm already showed **0.703 dB**. Given that the N3V floor is
  driven by densification-amplified nondeterminism and LRV3 also
  densifies, **the LRV3 spread should be treated as unverified**. Every
  new LRV3 arm needs ≥2 replicates reporting its own measured spread —
  this is now a requirement, not a precaution.
* Groups 339/347 produce no absence signal; no downstream rule can
  manufacture one.
* The observation-supply mechanism claim remains UNTESTED and LRV4 cannot
  test it.
