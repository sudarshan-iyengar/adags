---
title: The shell-supply projection cannot decide hull completion — route retired
date: 2026-08-24
evidence_bearing: false
---

# The projection route is retired, in absolutes AND in deltas

**ZERO GPU.** Reproducible via the tracked
`scripts/nonconvex_hull_projection_limits.py`.

Successor to [[nonconvex-hull-o1-result-2026-08-24]], which closed the lane
INVALID in both orientations because T1's accepted component never spanned the
L's two arms. That page's open question — *what would the hull operator do to a
component that DOES span?* — **is not answered here, and this page records why
it cannot be answered by projection.**

**No verdict on hull completion is issued. It remains neither refuted nor
supported**, exactly where
[[lrv3-membership-candidates-result-2026-08-23]] §7 left it.

## 1. ABSOLUTES are unusable — calibrated, not asserted

The preflight's proxy scores `object_shell / (object_shell + filler_shell)`,
assuming every gated row is object or filler. On this fixture that is
measurably false, from data already on the record.
[[nonconvex-hull-o1-result-2026-08-24]] records O1's base rule gating **10,374
rows at precision 0.5868**, of which **1,152 are notch filler**:

| | rows | share |
|---|---:|---:|
| object | 6,087 | 58.7% |
| filler | 1,152 | 11.1% |
| **neither — invisible to the proxy** | **3,135** | **30.2%** |

Fed those exact counts, the proxy's accounting returns **0.8409** against a
measured **0.5868**.

> **BIAS = +0.2541, about 25x the ~0.01 discretization error** that earlier
> drafts quoted as their uncertainty. Because the denominator structurally
> cannot contain background, **the bias is one-sided — always high, never
> low.**

O2 fails in kind rather than degree: `filler_shell[occupied] == 0` **exactly**,
so the proxy assigns O2's base set literally zero false positives where the
substrate measured 0.6768.

## 2. DELTAS are unusable too, and seeing that took three attempts

Spec §9 reads the verdict only from the delta, precisely because absolutes are
contaminated. So the delta was the obvious fallback. It does not survive.

**(a) The natural enumeration is VACUOUS.** On a connected mask H1's output is
exactly `bbox(mask)`, which invites enumerating index bboxes and evaluating
each at its maximal in-bbox mask `box & occ`. But then **every added cell lies
outside `occ`**, and object shell on non-occupied cells is **exactly zero**
(asserted in the script, both orientations). So `delta <= 0` is a **theorem of
the construction**, not a finding. **The favourable branch is unreachable.**

An earlier draft of this page enumerated exactly that family, reported
**156/156 negative deltas**, and read it as a result — on a page whose method
section recites *"every frozen rule needs a precondition that the mechanism was
exercised."* **The rule was written in one section of the document and violated
in another.**

**(b) Without that restriction the two accountings DIVERGE.** Over all
connected spanning components of size 2–7 — surface-area-weighted shell counts
against volume-weighted row counts that **do** include background:

| | size 5 | size 6 | size 7 | full set |
|---|---|---|---|---|
| **O1** shell % positive | 0.0% | 2.7% | 0.6% | −0.182 |
| **O1** row % positive | **33.1%** | **30.3%** | **27.8%** | −0.153 |
| **O2** shell % positive | 0.0% | 0.0% | 0.0% | −0.267 |
| **O2** row % positive | 1.1% | 2.5% | 3.5% | −0.177 |

The two disagree by **an order of magnitude** on how often H1 helps. They do
not corroborate one another; they diverge. **No sign statement is available.**

This also refutes a claim an earlier draft made outright: *"H1 never improves
precision on this fixture"* is **false** — positive deltas exist in both
orientations, best **+0.0350** (shell, O1) and **+0.1608** (row, O1).

**And the excluded regime was the realistic one.** T1 accepted **4 cells of 452
groups** on O1. Real accepted components are small and sparse — the very
regime where the sign is contested — while `box & occ` (24 and 32 cells) is the
least realistic case and was the only one scored.

## 3. One reusable result: BFS tie-breaking cannot matter here

A shortest 6-connected path is chosen by neighbour-iteration order, an
undeclared selection rule. Here it is provably irrelevant: every shortest path
between an arm-A-only and an arm-B-only cell has length exactly `L1 + 1`, hence
is monotone, hence cannot leave its endpoints' bounding box. Verified **per
pair across all 720 orderings** — 64 pairs (O1), 144 (O2), **at most 1 distinct
bbox per pair, 0 non-monotone paths**. Checking aggregates would not have
established this.

## 4. Scope — the fresh seeding grid, not what the runs realized

Every decomposition here is the **fresh 50k seeding cloud's** grid. Spec §12.2
flags that the trained grid is not controllable, and the divergence is measured
rather than hypothetical: the fresh grid places the object at `j` in {3,4},
while experiment 274's realized accepted cells all sit at **`j = 5`**. Cell
counts do not transfer to a trained substrate.

## 5. What this does NOT establish

* **Not** a rejection or an admission of hull completion.
* **Not** that an inferred pipeline could reach any spanning component. It
  cannot — the measured finding of [[nonconvex-hull-o1-result-2026-08-24]]
  stands.
* **Not** a comparison against LRV3's 0.9375; that page explicitly forbids it.
* **Not** recall, and **not** false activations. On this fixture the gate's
  `SURVIVES` branch was already **unreachable** (recall 0.2995 / 0.1599 against
  a 0.90 floor).
* **Not** a change to the structural recall cap of **0.8088** on LRV3.

## 6. Corrections to four drafts of this work

Recorded in full because the sequence is the finding.

* **"100% of spanning components violate the floor"** — quantified over
  shortest paths while claiming all spanning components. WITHDRAWN.
* **"O1's conclusion is unconditional"** — rested on a critical `p0` above 1.0
  from a mis-based ratio. WITHDRAWN.
* **"O2 reaches 0.874, comfortably above the floor"** — absolute reading of a
  +0.2541-biased proxy. WITHDRAWN.
* **"A minimal spanning component is the WORST case"** — artifact of a
  mis-specified base. WITHDRAWN.
* **"156/156 negative deltas"** and **"H1 never improves precision"** — vacuous
  and false respectively. WITHDRAWN (§2).
* **"Counting background pushes the delta MORE negative"** — backwards; a
  population common to both sides compresses the difference. WITHDRAWN.
* **"The preflight's base omits the occupied cells H1 also gates"** —
  misattributed. The preflight computes its base over the full occupied set,
  and object shell on empty cells is zero, so nothing is omitted there. That
  defect was real in an intermediate version of *this* script only. **One**
  defect was inherited — `r` mis-based by a factor `p0` — and in isolation it
  flattered "the falsifier bites" on O1 (0.7379 → 0.7524) and was neutral on
  O2. The later "mixed direction" claim came from also substituting a different
  base precision, not from the correction. WITHDRAWN, including the
  method note built on it.
* **A seed rule was frozen and abandoned**; a **REJECTION was issued** without
  licence; **O2's precision was attached to O1's cell count**.

## 7. Method notes — the durable product of this lane

**(1) A proxy needs a CALIBRATION, not a stability estimate.** The ±0.01
carried through two drafts is the *discretization* error and is defensible as
such — shell counting recovers ~95% of object and ~93% of filler surface area.
It says nothing about whether the proxy scores the right **population**, and
the population error was **25x larger and one-sided**. **A tight stability
figure attached to a badly biased instrument is worse than no figure**, because
it invites precisely the comparison that had to be withdrawn.

**(2) A bias direction must be computed, not intuited.** "The missing rows are
false positives, so counting them makes it worse" is the natural argument, was
written into a draft, and is **backwards**.

**(3) An enumeration can be vacuous in exactly the way a reading rule can.**
The block's standing rule — *a frozen rule needs a precondition that the
mechanism was exercised* — was recited in §9 of the very draft whose §2
enumerated a family in which the favourable answer was unreachable by
construction. **Writing the rule into the same document did not prevent
violating it.** The discipline transfers only when the precondition is
*computed*, here: `object_shell[~occ] == 0`, one line, which settles vacuity
immediately.

**(4) When several instruments agree, check whether they CAN disagree.** Two
accountings agreeing 156/156 looked like corroboration and was the same theorem
twice. Where they genuinely could disagree, they did — by an order of
magnitude.

These join *"a ratio without its n is not a measurement"* (LRV4), *"every edit
experiment needs a control separating MAGNITUDE from CORRECTNESS"*
(2026-08-20), and *"every frozen reading rule needs a frozen precondition"*
(this block).

## 8. What it motivates

A successor spec that **measures rather than projects**: supply a spanning
component by construction, run V4 on the trained substrate, compute precision,
recall and false activations over `row_ids` — and **state the gate's reachable
range before freezing**, since on this fixture the recall limb was already
unreachable. Cost is not estimated here.
