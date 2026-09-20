# Multi-interval presence on Ex4DGS — plan v0.2 (2026-09-21)

**Status: DRAFT. Nothing implemented.** Supersedes
[`v0.1`](ex4dgs-multi-interval-plan-2026-09-20.md), which was **rejected 3/10**
by cross-model adversarial review; v0.1 is left standing and is not edited.
Review and the independent re-verification of its objections:
[`ex4dgs-multi-interval-review-round1-2026-09-21`](ex4dgs-multi-interval-review-round1-2026-09-21.md).

**What changed:** the mechanism, the identity metric, the capacity control,
three of four preconditions, the headline reproduction number, and the novelty
position. What survived: the two-part claim structure (photometric ≠ identity),
and the decision to build on Ex4DGS rather than Multi4D.

---

## 0. Corrections to the record carried into this version

* **The reproduction gap is 1.06 dB, not 0.59 dB.** 0.59 was best-of-six
  against the published number — selection. Mean 32.6731 vs published 33.7329
  = **1.0598 dB**. Every future statement uses the mean.
* **v0.1's "nesting" property was false.** K=2 noisy-OR does not contain K=1
  (duplication gives `2a - a²`). The licence to call A2 underperformance an
  optimisation failure is withdrawn.
* **v0.1's noisy-OR gradient table was wrong.** `dA/da_k = Π_{j≠k}(1-a_j)`
  vanishes when any other interval is on its plateau.

---

## 1. The claim (unchanged in structure, narrowed in scope)

**C1 (photometric).** On a disappear-and-return event, disconnected temporal
support improves the ghost window **and** the return window over corrected
single-support, at matched capacity, beyond the measured noise floor.

**C2 (operational identity).** Object state survives the absence in a way that
is *usable*: contribution mass returns to the **same instances**, and an edit
applied before the gap reappears on the correct object after it.

**The licensed claim, stated at the scope the evidence can reach:**

> On controlled counterfactual-removal fixtures, disconnected temporal support
> improves event rendering and/or contribution-weighted identity retention
> relative to this Ex4DGS reproduction.

Physical disappearance is **not** licensed by authored fixtures. That requires
the capture described in §7.

---

## 2. The mechanism — ordered lifespan with a learned absence notch

v0.1's K exchangeable intervals under noisy-OR are **withdrawn**. Three defects
killed it, each verified numerically: the plateau is a gradient dead zone so
absence frames cannot carve a gap; noisy-OR starves an interval whenever
another is saturated; and it does not nest K=1.

### 2.1 The form

Boundaries from cumulative positive widths, so ordering holds **by
construction** — no sorting, no permutation symmetry, no optimiser-state
permutation:

```
b0 < b1 <= b2 < b3,     b = b0 + cumsum(softplus(raw_widths))
B_tau(t; l, r) = sigmoid((t-l)/tau) * sigmoid((r-t)/tau)
p(t) = B_tau(t; b0, b3) * (1 - q * B_tau(t; b1, b2))
```

`q ∈ [0,1]` gates the notch and **has an exact-zero state**.

### 2.2 Why this form, verified

* **Interior absence frames produce gradient on the notch boundaries.**
  Measured `dp/db1 = +3.5e-02` at t=45 inside a [40,60] notch, against
  **exactly 0** for v0.1's plateau. This is the defect that killed v0.1.
* **`q = 0` reproduces single support exactly** (equality at every probe), so
  the capacity control is a **one-scalar change on identical tensors** — not a
  constraint projection whose cleanliness has to be argued.
* **Ordering is structural**, so the identifiability machinery of v0.1 §2.4 is
  deleted rather than fixed.
* `tau` may be annealed after the notch is found, if sharp edges are needed.

### 2.3 Capacity accounting

Ex4DGS already stores **four temporal scalars** per dynamic primitive
(`_opacity_duration_center` `[N,2,1]` and `_opacity_duration_var` `[N,2,1]`,
`opacity_degree = 2` at `c_gaussian_model.py:86`). The notch needs four
boundaries plus `q`. A **capacity-neutral variant** reinterprets the existing
four scalars as `b0,b1,b2,b3` with global edge softness — the closest available
analogue to Multi4D reusing SH channels rather than adding parameters. It
sacrifices learned per-side tail widths, so it is an **ablation, not
automatically the better model**, and is reported as such.

---

## 3. Identity instrumentation — replacing `dominent_idxs`

`dominent_idxs` is the single Gaussian maximising `alpha_i * T_i` per pixel. It
discards all non-winning contribution, weights a one-pixel lineage equally with
a thousand-pixel one, and flips on opacity changes too small to alter the
image. **It is demoted to a visualisation.**

### 3.1 Soft contribution-mass attribution (no new parameters, no CUDA change)

The renderer already passes a dummy three-channel per-Gaussian `flow` tensor,
and training injects `[acc, L1, SSIM]` through its VJP so the CUDA backward
accumulates `alpha_i*T_i`-weighted quantities per primitive. **At evaluation,
inject `object_mask × acc` instead** and read

```
W(instance, window) = sum over cameras, frames, masked pixels of  alpha * T
```

This is blend-aware, multiview, and costs nothing. It is the one genuinely
undersold item from v0.1 — and it exists only because the code map noticed an
existing-but-unused channel.

### 3.2 Mutation-aware identity

A single `_lineage` column cannot express "the same primitive": split destroys
the parent, clone creates descendants, promotion deletes the donor row. Track
instead **`instance_id`, `root_lineage_id`, `parent_id`, birth iteration + op,
death iteration + op**, and report three quantities separately:

* **exact-instance** mass retention across the gap;
* **descendant** mass retention;
* **new-root** mass appearing after the gap.

"Created during the absence" is **ill-defined** and is dropped: cameras and
timestamps are shuffled per epoch, so optimizer iteration is not scene
chronology.

### 3.3 The operational test that actually carries C2

Apply a colour edit (or semantic label) to the pre-gap cohort, then check it
reappears **on the correct object** after the return. This is the test that
distinguishes preserved *object state* from preserved *bookkeeping* — and
bookkeeping is precisely what a copied id can fake.

---

## 4. Arms

| arm | what | purpose |
|---|---|---|
| **A0** | shipped discontinuous single support | quantifies the continuity repair only |
| **A1** | corrected single support (`q = 0`) | **the real baseline** |
| **A2** | notch enabled | primary treatment |
| **A4** | `q` frozen at 0, identical tensors | capacity control, exact by construction |
| **A6** | corrected single support, **matched final primitive / densification budget** | tests the respawn-with-more-primitives alternative |
| A3 / A5 | operator and multi-notch ablations | **only if A2 beats A1, A4 and A6** |

Plus an **oracle-initialised diagnostic**: a notch placed at the known event.
This separates *representation failure* from *gap-discovery failure*, which the
arms alone cannot.

**A0 must be re-run on the fixture.** The six existing A0 runs were trained on
the ordinary reproduction scene, not the edited fixture — they cannot satisfy
parity nor estimate fixture-trained event variance. This was a contradiction in
v0.1.

Report for every arm: final primitive counts, and contribution-weighted
births / splits / prunes / promotions.

---

## 5. Preconditions (rewritten)

* **P1 — causal, event-specific non-vacuity.** At fixed trained parameters,
  disable the notch and require the registered event-window render (or object
  mask contribution) to change by more than both the 99th percentile of a
  repeat-render null band **and** the frozen smallest claim-relevant effect.
  Also require the exercised instances to carry measurable object-mask
  contribution before *and* after across multiple cameras, and to fall below
  the renderer's effective visibility threshold during the gap.
  *(v0.1's global 5% rule is withdrawn: it could pass entirely on off-event
  primitives.)*
* **P2 — deleted as circular.** It required an interval-straddling primitive to
  dominate before and after, which is already a weak C2 success. Absence of
  overlap means **C2 = 0**, not "untestable".
* **P3 — A4 constrained by construction**, `q ≡ 0` for **100%** of primitives,
  asserted numerically. 1% leakage is fatal if that 1% explains the object.
* **P4 — common manifest plus declared arm overlay**: edited-data and mask
  hashes, camera split, point cloud, **paired seeds**, environment and
  CUDA-extension hash, evaluator flags, integrated commit.
* **P5 (new) — validate the instrument** on a synthetic overlapping-splat scene,
  plus mutation unit tests through clone / split / prune / promotion.

**A failed precondition is a method failure**, not automatically
DESIGN_WITHOUT_POWER. DWP applies only when the *data* lacks a verified event.

---

## 6. Staged execution — 1.2 GPU-h, then 18, then decide

**Stage 0 (zero GPU).** Validate soft attribution on synthetic overlapping
splats; validate birth/parent/death logging through every mutation; freeze
masks and the ghost / early-return / late-return / control windows; freeze the
smallest relevant effect **before** seeing any arm; measure event-window
variance from existing runs. Treat independently trained seed×scene runs as the
experimental units — never frames or pixels.

**Stage 1 — the 1.2 GPU-h kill test.** One A1 run plus an **oracle render-time
notch / cohort reactivation**. If the known pre-gap cohort returns at the wrong
geometry *even under oracle support*, the bottleneck is trajectory, not
interval learning, and the direction stops here. This is the cheapest decisive
experiment available and it runs first.

**Stage 2 — 18 GPU-h pilot.** Paired n=3 on A0/A1/A2/A4/A6, one fixture.
Continue only if A2 passes causal non-vacuity **and** beats A1, A4 **and** A6
on C1 *and* operational C2, with no material control-window harm.

**Stage 3 — only then**, the ~119 GPU-h package across three authored fixtures
and two physical captures, n=6 paired seeds, cap at n=12. If a 0.5 dB effect
needs dozens of runs, the method has poor value per GPU-hour and that is itself
the answer.

Power, paired, two-sided 5%, 90%: **σ=0.356 → 6 pairs; σ=0.52 → 12;
σ=1.00 → 43.** Event-window σ is **not yet measured** and must not be assumed
equal to whole-scene σ.

---

## 7. Scope limits that are now explicit

* **Authored fixtures license only fixture claims.** Physical disappearance
  needs: calibrated synchronised multiview capture, the object physically
  leaving the shared volume and returning, all-view evidence it is *absent*
  rather than untracked, independently annotated identity, **changed-pose
  return**, frozen masks, held-out cameras. One event is a proof-of-principle;
  a venue claim needs multiple objects, scenes and gap lengths.
* **The trajectory through the gap may matter more than the gate.** Nothing
  supervises pose during an unsupervised absence, and `motion_reg`
  (`train.py:159-161`) compares the first keyframe against every later one
  rather than enforcing adjacent-keyframe smoothness. A re-entry constraint may
  be the more load-bearing contribution.
* **"Render the empty gap better" cannot be the contribution.** This project
  already measured STG rendering the authored gap 3.4–7.5 dB better than the
  gated arm while being worse on the return.

---

## 8. Novelty position (materially worse than v0.1 assumed)

Verified by fetching abstracts, not by accepting citations:

* **Flow4DGS-SLAM** (arXiv 2604.22339) — models *"the temporal opacity and
  rotation using a Gaussian Mixture Model"*. Multi-modal temporal opacity is
  therefore **occupied**.
* **TRiGS** (arXiv 2604.00538) — frames *"primitives repeatedly eliminated and
  regenerated"* and *"eliminates the long-term temporal identity of objects"*
  as its problem. This is **very close to C2**.
* **PersistGS** (2606.03479) — already on this project's record as owning
  object permanence under all-camera occlusion.
* **SharpTimeGS** — learnable flat-top lifespan modulation.

**Consequence:** a union of two temporal lobes is not a contribution by itself.
What is not obviously occupied is the conjunction of *disconnected support* with
*contribution-weighted, mutation-aware, operational* identity verification —
i.e. proving the object state is **usable** after the gap, not merely that
pixels match. That is a measurement contribution as much as a representation
one, and it is unproven.

---

## 9. Honest position

The reviewer's strongest argument against the whole direction stands unrefuted:
Gaussian identity may be unidentifiable and of no user-visible value, since the
renderer can split, clone and respawn exchangeable basis functions while
producing identical images — and a copied id can manufacture "continuity" for a
row that was destroyed. **The operational edit-transfer test in §3.3 is the
only thing in this plan that answers it**, which is why it is promoted from a
nice-to-have to the load-bearing C2 endpoint.

Stage 1 costs 1.2 GPU-h and can end the direction. That is the right next
action, and it should be taken before any substrate code is written.
