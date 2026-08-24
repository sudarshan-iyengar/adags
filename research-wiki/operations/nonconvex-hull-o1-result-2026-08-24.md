# RESULT — LRV5-NCX orientation O1 is an INVALID instrument; no hull
# verdict may be read from it, and the near-miss is the finding
# (2026-08-24)

EXPLORATORY, `evidence_bearing: false`. Gate, operators and preconditions
were frozen and committed at `dad8fd6` **before the fixture existed**.

## 1. The verdict

**INVALID.** Precondition **V3 (the operator is not vacuous) FAILED**, so
per the frozen §8 no verdict on hull completion may be read from this run.
It is recorded as an invalid instrument, exactly as
[[lrv4-starved-fixture-result-2026-08-23]] was.

## 2. What ran

| cell | exp | result |
|---|---:|---|
| fixture generation, O1 | 273 | COMPLETED |
| ungated substrate (`configs/lrv5/a0_ncx.yaml`) | 274 | COMPLETED — 26.77 dB, **149,965 rows** (LRV3's was 149,794) |
| T1 boundary estimation, `--skip-scoring` | 275 | COMPLETED — **452 groups, 2 gated** (LRV3: 417 groups, 2 gated) |
| membership scoring | — | CPU, local, zero GPU |

Trained cloud: 149,965 rows, **33,893** inside the event object, **7,628**
inside the notch filler.

## 3. The scores, and why they must NOT be read as a result

| rule | cells | rows | precision | recall | false act. | 0-object cells |
|---|---:|---:|---:|---:|---:|---:|
| base (A2 + B) | 4 | 10,374 | 0.5868 | 0.1796 | 0 | 0 |
| base + **H1** | 6 | 16,071 | 0.6317 | 0.2995 | 0 | **0** |
| base + **H2** | 4 | 10,374 | 0.5868 | 0.1796 | 0 | 0 |

```
DELTA base -> base+H1   cells +2 | rows +5,697 | precision +0.0449 | recall +0.1199 | false +0 | 0-obj +0
DELTA base -> base+H2   cells +0 | rows     +0 | precision +0.0000 | recall +0.0000 | false +0 | 0-obj +0
```

## 4. THE NEAR-MISS — this is the part worth carrying

**Read without V3, that table says hull completion SURVIVED**: H1 improved
precision by +0.0449 **and** recall by +0.1199, added **zero** false
activations, and filled **zero** zero-object cells. On its face, a clean
pass of an operator this project had refused as post-hoc.

**It is vacuous.** V3 records exactly why:

```
cells whose object volume comes ONLY from arm_a_z : 0   []
cells whose object volume comes ONLY from arm_b_x : 3   [[4,5,2], [4,5,3], [5,5,3]]
accepted cells: [3,5,2] [4,5,2] [4,5,3] [5,5,3]
```

**T1's accepted component lies entirely along arm B and never reaches arm
A.** H1 is defined per connected component, so its index bounding box
spans only arm B's region — and the notch sits between the arms. **H1 was
never in a position to fill the concavity, so it could not fail.** The
+0.0449 precision is H1 absorbing more of arm B, not H1 declining to
over-gate.

This is the vacuity mode the CPU preflight measured in advance: over all
occupied-cell pairs the minimum number of zero-object cells H1 fills is
**0**, while over arm-A x arm-B pairs it is **>= 1**. V3 is the
precondition written to separate those regimes, and it is a statement
about the **accepted set**, never about the score, so checking it cannot
leak the outcome.

**The generalization, and it is the same lesson LRV4 taught in a different
costume:** *a favourable delta from an operator that never had the
opportunity to fail is not evidence.* LRV4's version was "a ratio without
its n is not a measurement". This block's version is **an operator's pass
is meaningless until you check it was exposed to the case that would
refute it.** Both were caught only because a precondition was frozen
before the numbers.

## 5. What IS established, and what is NOT

**Established.**
* The falsifier's machinery works end to end: fixture generation, ungated
  substrate, boundary estimation and scoring all ran, and the frozen
  precondition fired correctly on the one regime that makes the test
  meaningless.
* T1's behaviour on a non-convex object is *shaped like* its behaviour on
  the sphere — 2 gated of 452 groups against 2 of 417 — so the estimator's
  extreme selectivity is not a sphere artefact.

**NOT established — and none of this may be claimed.**
* **Nothing about hull completion.** It is neither refuted nor supported by
  this run. The question stands exactly where
  [[lrv3-membership-candidates-result-2026-08-23]] §7 left it.
* **Nothing comparative about base-rule precision.** The base rule reads
  0.5868 here against 0.9375 on LRV3, and it is tempting to call that a
  non-convexity penalty. **It is not a controlled comparison** — different
  object, different grid realization, different substrate, different rig
  radius. It is recorded as an observation and explicitly not as a result.
* That the base rule gating **1,152 notch-filler rows** is a defect of the
  base rule rather than of this fixture's geometry. Undetermined.

## 6. What happens next, and the disclosure that must travel with it

Orientation **O2** was **predeclared in the frozen spec** —
`(x,y,z) -> (-x + 0.1625, y, z + 0.1625)` — before any fixture existed, so
running it is not selection of a favourable arm. Whether V3 passes there
is genuinely unknown: it depends on which cells T1 happens to gate, and O2
is a re-roll of that, not a guaranteed repair.

**Disclosure that must accompany any O2 reading:** O1 ran first and was
INVALID on V3, and O2 was run *because of that*. If O2 is valid, its
verdict stands on its own, with this ordering disclosed. **If O2 also
fails V3, the correct conclusion is that this fixture design cannot expose
H1 to the concavity at 8³ with T1's selectivity — a statement about the
INSTRUMENT, not about hull completion** — and a future spec must force a
spanning accepted component by construction rather than hope for one.

**No threshold, operator or precondition was changed after seeing these
numbers, and none may be.**
