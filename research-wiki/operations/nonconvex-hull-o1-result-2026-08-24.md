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

---

## O2 RESULT (2026-08-24, append-only) — also INVALID, and the conclusion the O1 record PRE-DECLARED now fires

Experiments **276** (generation), **278** (substrate, 26.647 dB, 149,966
rows), **285** (T1, `--skip-scoring`), all `STATE_COMPLETED`; scoring on
CPU, zero GPU.

```
scene / orientation    LRV5 / O2
trained cloud          149,966 rows | 34,408 in the event object | 7,224 in the filler
groups                 511 | T1 gated cells 2

rule                  cells    rows   precision   recall   false   0-obj
base (A2 + B)             4    8,130     0.6768   0.1599       0       0
base + H1                 4    8,130     0.6768   0.1599       0       0
base + H2                 5   10,857     0.6837   0.2157       0       0

DELTA base -> base+H1   cells +0 | rows     +0 | precision +0.0000 | 0-obj +0
DELTA base -> base+H2   cells +1 | rows +2,727 | precision +0.0070 | 0-obj +0

  [PASS] V1   [PASS] V2   [FAIL] V3   [FAIL] V4
VERDICT: INVALID
```

### The failure is COMPLEMENTARY to O1, which strengthens the reading

| | cells only from `arm_a_z` | cells only from `arm_b_x` |
|---|---:|---:|
| **O1** | **0** | 3 |
| **O2** | 3 | **0** |

**The mirror flipped which arm T1 latches onto, and in both orientations
the accepted component lies entirely inside ONE arm.** It never spans. So
H1's per-component bounding box never reaches the notch — in O2 it adds
**zero** cells, so V4 also fails (`measured_fraction 0.0` against the
required 0.25).

That the two predeclared orientations fail the same precondition *by
opposite arms* makes this a property of the method, not a quirk of one
geometry.

### The conclusion, and it was FROZEN BEFORE O2 RAN

The O1 record states verbatim: *"If O2 also fails V3, the correct
conclusion is that this fixture design cannot expose H1 to the concavity
at 8³ with T1's selectivity — a statement about the INSTRUMENT, not about
hull completion."*

**That is now the conclusion, and it is reached by a rule frozen before
the data existed.**

**Root cause, measured:** T1 gates **2 of 452** groups on O1 and **2 of
511** on O2. That extreme selectivity — the same property that made T1's
boundary inference precise and gave it zero false activations — produces
an accepted component far too small to span an object whose two arms are
separated by a notch. **The instrument that makes T1 good at boundaries
makes it unable to exercise a hull operator on a non-convex object.**

### Permitted and forbidden

**Permitted.** Both predeclared orientations of LRV5-NCX are invalid
instruments for the hull question, by the same precondition, via opposite
arms. The cause is T1's selectivity, not the fixture's geometry — V1 and
V2 pass in both, so the concavity exists and is populated exactly as
designed.

**Forbidden.** **Any verdict on hull completion.** It is neither refuted
nor supported, and stands exactly where
[[lrv3-membership-candidates-result-2026-08-23]] §7 left it. Also
forbidden: reading O1's +0.0449 precision or O2's H2 +0.0070 as evidence
of anything, and treating the base-rule precision (0.5868 O1, 0.6768 O2)
as a controlled comparison against LRV3's 0.9375.

### What a future spec must do differently

**Force a spanning accepted component by construction rather than hope for
one.** Three routes, none adopted here and each needing its own frozen
spec:

1. **seed the base rule** with a component known to span both arms, making
   the hull operator the only variable;
2. **shrink the cells** so that one arm alone spans several, raising the
   chance a 2-cell component reaches across — at the cost of changing the
   grid the LRV3 result was measured on;
3. **relax T1's camera bar** for this fixture only, accepting more groups
   — which trades away the zero-false-activation property that makes T1
   worth testing at all, and would need its own control.

**Cost of the closed lane:** generation 0.086, substrates 1.135, T1 passes
~0.6 slot-h; ~1.8 slot-h total for both orientations, all scoring on CPU.
