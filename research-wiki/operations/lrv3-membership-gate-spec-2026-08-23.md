# SPEC (FROZEN) — the membership gate, and the Candidate A / B rules
# (2026-08-23, block 3)

EXPLORATORY, `evidence_bearing: false`. **Written and committed BEFORE
any repaired instrument was scored.** That ordering is what made the T1
and LRV4 diagnostics decisive and it is not optional here.

Prior measurement this spec builds on:
[[lrv3-membership-diagnostic-2026-08-23]].

## 1. Facts established BEFORE this spec (re-derived from primary artifacts)

The 417-group construction was reproduced exactly from experiment 184's
`point_cloud.ply` (`build_voxel_groups`, `cells_per_axis 8`,
`min_group_rows 4`), and the program's `spatial.lo`/`span` were verified
equal to that cloud's own bounding box.

The 8 groups overlapping the authored sphere decode to a **contiguous
2 × 2 × 2 block**, `x ∈ {5,6}`, `y ∈ {4,5}`, `z ∈ {4,5}`:

| group | cell | (x,y,z) | sub rows | sub in-sph | fresh rows | fresh in-sph | gated | onset | offset | cams | reason |
|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| 292 | 365 | (5,5,5) | 2412 | 2345 | 181 | 12 | **YES** | 57 | 30 | 4 | — |
| 285 | 357 | (5,4,5) | 2040 | 1815 | 190 | 25 | no | 57 | 30 | 2 | `camera_disagreement` |
| 340 | 421 | (6,4,5) | 1547 | 1469 | 210 | 14 | no | 57 | 30 | 2 | `camera_disagreement` |
| 347 | 429 | (6,5,5) | 1487 | 1453 | 181 | 4 | no | — | — | 0 | `no_interior_gap` |
| 291 | 364 | (5,5,4) | 1213 | 1167 | 155 | 3 | **YES** | 57 | 30 | 4 | — |
| 284 | 356 | (5,4,4) | 1154 | 1049 | 189 | 18 | no | 57 | 30 | 2 | `camera_disagreement` |
| 346 | 428 | (6,5,4) | 822 | 769 | 184 | 2 | no | 57 | 30 | 2 | `camera_disagreement` |
| 339 | 420 | (6,4,4) | 655 | 583 | 181 | 6 | no | — | — | 0 | `no_interior_gap` |

**The load-bearing new fact: four of the six rejected cells report
EXACTLY the accepted onset (57) and offset (30).** They were rejected on
camera support (2 against a required 3), not on disagreeing boundaries.
Two cells — 420 and 429 — carry **no boundary estimate at all**, so no
boundary-agreement rule can ever admit them. That caps Candidate A at 6
of 8 cells by construction, before any measurement.

Face adjacency to an accepted cell: 356 ✓, 357 ✓, 428 ✓, 429 ✓;
**420 ✗, 421 ✗**.

## 2. THE GATE — frozen

An instrument is ADMITTED for retraining only if, measured **per row on
the exact cloud to which it will bind**, it satisfies all three:

1. **recall ≥ 0.90**
2. **precision ≥ 0.80**
3. **non-degeneracy**: it must not reduce to the oracle sphere test and
   must not consume held-out cameras (2, 7, 12, 17).

**Justification, and the recall floor is explicitly a JUDGMENT.** The
measured ordering on `event_return` is fully gated 28.19 > ungated 27.14
>> partially gated 24.67. Partial gating is the known failure mode and it
is worse than doing nothing. **There is no measurement at intermediate
recall**, so any interpolated floor would be invented. 0.90 is therefore
declared as a judgment: it is the level at which "partial" is no longer a
fair description. It is not derived from data and is labelled as such.

Precision 0.80 is anchored to the oracle's 1.00 and to the measured fact
that the failing configuration ran at 0.0446.

**No floor here may be moved after any score is read.**

## 3. Candidate A — boundary-conditioned spatial coherence, frozen

A rejected cell may join an accepted component only if ALL hold:

* **(i) adjacency** — face adjacency (6-connectivity) in the 8³ grid;
* **(ii) boundary agreement** — its inferred onset AND offset equal the
  component's **exactly** (no tolerance; a cell with no boundary
  estimate can never qualify);
* **(iii) camera support** — `agreeing_cameras >= 2`.

**On (iii), stated plainly because it is the contestable choice.** T1's
bar is 3 cameras for accepting a cell on its OWN evidence with no prior.
Condition (iii) governs *joining* a cell to an already-accepted component
whose boundaries it reproduces exactly — a joint-evidence condition, so a
lower solo bar is defensible on principle. But I set it to 2 **knowing**
the missed cells sit at 2, and that must be recorded rather than
presented as neutral. **The false-positive control in §5, not this
judgment, adjudicates whether it was legitimate.**

**Two variants, BOTH pre-registered and both reported, so neither is
selected for its outcome:**

* **A1 single-pass** — adjacency measured against the ORIGINALLY accepted
  cells only.
* **A2 transitive** — flood fill, adjacency measured against the growing
  component, propagating only through cells that themselves satisfy
  (ii) and (iii).

## 4. Candidate B — same-cloud `row_ids` binding, frozen

Membership inferred on the trained substrate and applied to **that same
cloud**, via the existing `membership_mode: "row_ids"` path and its
cloud-fingerprint guard — no transfer across the densification boundary.

Scored on the substrate cloud, which is the cloud it binds to.

**B is a scientific mechanism change, not an engineering repair.** It
moves the method from *infer regions on one model, retrain a fresh model*
to *train a model, detect the episode, apply episodic structure to that
same model*. Consequences, pre-identified:

1. **The recorded comparator set becomes invalid.** A1-LOCAL's +1.0496 dB
   trained from scratch and cannot be compared to a
   continue-from-checkpoint protocol. No-gate and oracle arms must be
   regenerated on the same starting checkpoint: ~3 arms × 2 replicates ≈
   6 cells × ~0.87 slot-h ≈ **5.2 slot-h**.
2. **Pre-identified scientific risk.** A model already trained through
   the gap has learned to explain it with a smeared temporal marginal.
   **Failure indicator, declared in advance:** if the gated
   continuation's `ghost_gap` does not fall toward the oracle's 22.83
   within the continuation budget while `event_return` fails to exceed
   the ungated continuation, the model has not re-optimized cleanly and
   B is refuted as a mechanism.
3. **Continuation schedule** must be specified and counts against the
   6,000-iteration cap.

**A2 + B (the combination) is PRE-AUTHORIZED** on the stated ground that
it is the only configuration reaching oracle-comparable membership. It
does not require a "neither alone passed" demonstration.

**B alone is evaluated separately even though it is expected to fail
recall**, because the T2 failure confounded low precision with low
recall, and B alone isolates them: high precision at partial recall. **If
B alone fails, recall is decisively the binding variable** — a real
finding worth having.

## 5. REQUIRED false-positive control — frozen

T1's headline properties are **zero false activations** and **99.52%
abstention**. Candidate A relaxes acceptance and puts both at risk.

The coherence rule is run over **all 417 groups**, not the 8, and the
**full confusion** is reported: newly accepted object cells AND newly
accepted non-object cells, with the resulting abstention rate.

**If false activations rise above zero that is a headline regression and
is reported as one**, not absorbed into a summary. A rule that clears the
§2 gate while creating false activations has traded T1's strongest
property and must be reported as such.

## 6. Scoring discipline

All scoring is **CPU-only and precedes any training**. An instrument is
scored on the exact cloud to which it would bind — Candidate A on the
fresh seeding cloud, Candidate B on the trained substrate. **No
retraining is authorized for any instrument that fails §2.**
