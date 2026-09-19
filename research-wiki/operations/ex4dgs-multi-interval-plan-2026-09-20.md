# Multi-interval presence on Ex4DGS — implementation and experiment plan (v0.1, 2026-09-20)

**Status: DRAFT, not approved, nothing implemented.** Written to be attacked.
Submitted for external adversarial review before any substrate code is touched.

Substrate: Ex4DGS @ `7fac649`. Code map:
[`ex4dgs-code-map-2026-09-19`](ex4dgs-code-map-2026-09-19.md). Reproduction and
gates: [`ex4dgs-n3v-leonardo-reproduction-2026-09-19`](ex4dgs-n3v-leonardo-reproduction-2026-09-19.md).

Control model (user, 2026-09-18): every substrate change is a **review-gated
small diff** — one diff, user reads it, user says submit, and the **user does
the evaluation and analysis**. This plan is therefore a sequence of small
reviewable steps, not a single change.

---

## 1. The claim, stated so it can fail

> A dynamic primitive whose presence is a **union of K intervals** can be
> absent across a full-multiview occlusion/disappearance and then **resume as
> the same primitive**, where the single-interval (K=1) parameterisation must
> either keep painting through the gap (ghost) or hand the content to different
> primitives after the gap (respawn).

Two things must be shown, and **the second is the one that is actually novel**:

* **C1 (photometric).** On an authored absence event, K>1 improves the ghost
  window and the return window over K=1 at matched capacity, by more than the
  measured noise floor.
* **C2 (identity).** The primitives painting the object *after* the return are
  **the same primitives** that painted it before. This is not implied by C1: a
  respawn can be photometrically perfect. C2 is what distinguishes "multi-interval
  presence" from "more capacity".

**If C1 holds and C2 fails, the contribution is a capacity trick, not a presence
representation, and should be reported as such.**

### Why the substrate permits this at all

From the code map, verified: Ex4DGS computes its temporal factor in Python
(`utils/interpolations.py:55-61`), CUDA has no time concept, and α decays
smoothly rather than being hard-culled — so an absent primitive still receives
gradient and can learn to return. Multi4D cannot host this without a CUDA edit
and a change to its `marginal_t > 0.05` early-return in `backward.cu`.

---

## 2. The mechanism

### 2.1 Current (K=1), verbatim

```python
def time_bigaussian(mean, var, t, training=False, var_min=0.1):
    m =  (t - mean).min(dim=1)[0]
    v = torch.where((t > mean).any(dim=1), var[:, 1], var[:, 0])
    opacity_ = m.pow(2) / (v.exp() + var_min/2.36).pow(2)
    opacity_ = torch.exp(-1*opacity_)
    return torch.where( (mean[:, 0] - t) * (mean[:, 1] - t) < 0, torch.ones_like(opacity_), opacity_)
```

Plateau of exactly 1.0 on `(c0, c1)`, Gaussian tails outside, single interval by
construction.

### 2.2 Proposed: a continuous per-interval distance

For interval *k* with endpoints `(c0_k, c1_k)` define the **signed distance to
the interval**

```
d_k(t) = max(c0_k - t,  t - c1_k)          # < 0 inside, 0 at the edge, > 0 outside
a_k(t) = exp( - relu(d_k(t))^2 / sigma_k(t)^2 )
sigma_k(t) = sigma_left_k  if t < c0_k  else  sigma_right_k
```

`a_k` is **exactly 1.0 inside** (relu clamps the negative distance to zero) and
Gaussian outside, and it is **continuous at both edges**. Then

```
alpha_temporal(t) = UNION_k a_k(t)
```

Three properties worth stating because they are what make the comparison fair:

* **Nesting.** At K=1 this reduces to the intended K=1 semantics, so the K>1
  model **strictly contains** the K=1 model. If K>1 underperforms, that is an
  optimisation failure, not a capacity failure, and must be reported that way.
* **It repairs a real defect.** The shipped `.min(dim=1)` measures the *left*
  tail from `c1`, the far edge, so α jumps discontinuously from
  `exp(-(c1-t)²/σ²)` to `1.0` at `t = c0`. The proposed form has no
  discontinuity. **This means the bug fix and the K>1 mechanism are confounded
  unless separated — see the arms in §4.**
* **Plateau semantics are preserved**, so `sigmoid(_opacity_motion)` still sets
  the in-interval opacity and nothing downstream changes meaning.

### 2.3 The union operator is a real design choice, not a detail

| | `max_k a_k` | noisy-OR `1 - Π_k (1 - a_k)` |
|---|---|---|
| value on a plateau | exactly 1.0 | exactly 1.0 (one term is 1) |
| gradient | only to the argmax interval | to **all** intervals |
| overlapping intervals | idempotent | saturates toward 1 |
| failure mode | unused intervals get no gradient and stay frozen at init | intervals can co-adapt into a smeared union |

**Prediction, to be registered before running:** `max` will collapse to K=1 more
often, because an interval that is never the argmax never receives gradient —
the same dead-gradient pathology that rules out Multi4D, reappearing at the
operator level. **Noisy-OR is the primary; `max` is an ablation.** If the
prediction is wrong, that is a finding and is recorded as such.

### 2.4 Identifiability

Intervals are exchangeable: permuting *k* leaves α unchanged. Consequences:

* any statement like "interval 0 is the first presence" is **unwarranted** unless
  the intervals are sorted;
* a permutation-variant regulariser or prune rule is ill-defined;
* clone/split must not scramble the assignment.

**Disposition:** project `c` to sorted order by `c0_k` after each optimizer step
(a cheap sort, no gradient issue since it is a relabelling). This also makes the
degenerate-interval prune well-defined.

---

## 3. What must change, and in what order

Each step is one reviewable diff. Steps 0–2 are instrumentation and must land
before any behavioural change, so that the behavioural change is measurable.

| # | step | substrate touched? | why it is separate |
|---|---|---|---|
| **0** | **Seed control.** Add `--seed`, applied inside `safe_state` (not the visible `setup_seed`). | yes, 3 lines | Without it there is no seed spread, only CUDA nondeterminism. Multi4D's lesson: patching the visible seed line is silently overwritten 28 lines later. |
| **1** | **Lineage column** `_lineage` on both banks, threaded through every site in code map §7, plus a test asserting all per-primitive tensors have equal length after every mutation. | yes | C2 is unmeasurable without it. Landing it alone means a failure here cannot be blamed on the mechanism. |
| **2** | **Identity readout.** Record `dominent_idxs` (already returned by the rasterizer, `gaussian_renderer/__init__.py:121`) per test frame inside the ROI, map through `_lineage`. | no — new ADAGS script | Turns an existing, unused rasterizer output into the C2 measurement. No substrate change at all. |
| **3** | **Continuity fix at K=1** — replace `.min(dim=1)` with the `relu(d_k)` form, still K=1. | yes, ~7 lines | **The confound control.** Any K>1 gain is otherwise contaminated by the bug fix. |
| **4** | **K>1 parameterisation** — `[N,2,1]` → `[N,K,2,1]`, `opacity_degree`, noisy-OR union, sorted projection. | yes | the mechanism |
| **5** | **Make the four bypasses K-aware** — clone/split `var←2` reset, `adjust_temp_opa` `.any(dim=1)`, `extract_from_static` 2-element stack, `load_ply` allocation. | yes | Without this, multi-interval structure is destroyed every 300 iterations and the treatment is vacuous. |
| **6** | **Assign the dead prune** at `c_gaussian_model.py:1078` and generalise it to per-interval degeneracy. | yes, 1 line + generalisation | It currently computes a `logical_or` and throws it away. At K>1 collapsed intervals must actually be removed or the precondition in §5 cannot be trusted. |

**Step 5 is the one most likely to silently eat the mechanism.** `densify_and_split:945`
and `densify_and_clone:1007` reset `var` to the constant 2 and jitter `center`;
a naive `repeat` at K>1 duplicates all intervals identically every 300
iterations.

---

## 4. Experimental design

### 4.1 Arms

| arm | K | union | continuity fix | purpose |
|---|---|---|---|---|
| **A0** | 1 | — | no | published baseline, already run (n=6, mean 32.673, sd 0.3564) |
| **A1** | 1 | — | **yes** | isolates the bug fix from the mechanism |
| **A2** | 2 | noisy-OR | yes | **primary treatment** |
| **A3** | 2 | max | yes | union-operator ablation (§2.3) |
| **A4** | 2 | noisy-OR | yes, **intervals constrained contiguous** | **capacity control**: identical parameter count, multi-interval capability removed |
| **A5** | 3 | noisy-OR | yes | does K beyond 2 buy anything, or is 2 the natural ceiling for one absence |

**A4 is the load-bearing control.** The parameter delta from K=1→2 is 4 floats
per dynamic primitive against ~300 already there (37×3 xyz keyframes + 37×4
rotation + 48 SH + …), i.e. **~1%**, so a generic capacity confound is
implausible — but "implausible" is not "excluded". A4 has *exactly* the same
parameters and optimiser state and differs only in whether the intervals may
separate. A gain in A2 over A4 is attributable to multi-interval presence and
to nothing else.

### 4.2 Scenes and endpoints

**Primary: the authored absence fixtures**, not whole-scene N3V. The project has
already measured that the event union is ~0.34% of pixel-times, so even a large
event-region effect moves whole-frame PSNR by ~0.02 dB — whole-scene PSNR cannot
resolve this and must not be the endpoint.

* ghost window (object absent): PSNR in the ROI over the absence frames
* return window (object back): PSNR in the ROI over the first N frames after
* control window (object never absent): do-no-harm
* whole-scene PSNR: do-no-harm only, never the headline

Reuse the existing frozen machinery: `scripts/event_region_frame_profile.py`
with `absfix/evaluation_rois` and `absfix/absence_event_masks.json`, the same
masks STG and EL-GS were scored with, so the numbers are comparable to the
existing record.

### 4.3 C2, the identity endpoint

For each test frame in the return window, take the ROI pixels, read
`dominent_idxs`, map to `_lineage`, and compute

* **lineage recall** — fraction of the lineages that dominated ROI pixels
  *before* the absence that dominate ROI pixels *after* it;
* **lineage precision** — fraction of post-return dominating lineages that are
  pre-absence lineages (a respawn drives this toward 0);
* **respawn count** — dominating lineages created *during or after* the absence.

Prediction: K=1 shows high respawn / low recall; K>1 shows the reverse. **If
both show the same, C2 fails and the mechanism is not doing what it claims,
whatever C1 says.**

### 4.4 Preconditions, frozen before any run

Following the standing rule that a frozen reading rule needs a frozen
precondition that the mechanism was exercised (2026-08-24), and stated about the
*setup*, never the score:

* **P1 (non-vacuity).** At least 5% of dynamic primitives must end training with
  ≥2 intervals that are each non-degenerate (width ≥ `0.5/interval`, the
  existing degeneracy scale) **and** separated by a gap ≥ one keyframe spacing.
  If P1 fails, A2 has collapsed to K=1 and the comparison is
  **DESIGN_WITHOUT_POWER**, not a negative result.
* **P2 (the event is covered).** At least one primitive whose intervals straddle
  the authored absence must dominate ≥1 ROI pixel both before and after. If no
  primitive spans the event, C2 is untestable on that scene.
* **P3 (A4 really is constrained).** In the capacity control, the measured
  inter-interval gap must be ≤ 0 (contiguous) for ≥99% of primitives. If A4
  leaks multi-interval structure it is not a control.
* **P4 (parity).** A0–A5 must share commit, config hash, data hash and iteration
  count; only the declared diff differs.

### 4.5 Power

Whole-scene sd is **0.3564 dB** at n=6 (measured). **The event-window sd is not
yet measured and must not be assumed equal** — event ROIs are small, so it is
plausibly larger. **Step: measure it from the existing A0 runs before sizing
anything**, by scoring the six completed baseline runs through the event
profiler. That costs zero GPU.

Provisional sizing at the whole-scene sd, unpaired, 80% power, α=0.05
(~1.2 GPU-h per run):

| effect | n/arm | cost for A2 vs A4 |
|---|---|---|
| 1.0 dB | 3 | ~7 GPU-h |
| 0.5 dB | 9 | ~22 GPU-h |
| 0.3 dB | 23 | ~55 GPU-h |

With six arms the full matrix at n=9 is ~65 GPU-h. Affordable. **But arms should
be added in order A0/A1/A2/A4 first** — A3 and A5 only if the primary contrast
survives.

---

## 5. What would falsify this, and what we would report

| outcome | reported as |
|---|---|
| P1 fails | DESIGN_WITHOUT_POWER. The mechanism did not engage; no claim either way. |
| C1 yes, C2 yes | the full claim |
| C1 yes, C2 no | **a capacity/optimisation result, not a presence representation.** Report as such. |
| C1 no, C2 yes | identity is preserved but buys no photometric gain — an honest negative with a mechanism story |
| A1 ≈ A2 | the gain is the continuity bug fix, not multi-interval presence |
| A2 ≈ A4 | the gain is parameters, not the union structure |

---

## 6. Candidate small technical contributions

Recorded because the review is asked to judge whether any of these is real, and
because this project's convention is to name them explicitly rather than let
them accrete into a vague novelty claim.

1. **The continuous per-interval distance** replacing a discontinuous published
   formulation. Small, verifiable, and it is a defect in a NeurIPS 2024 method.
   Weak alone; meaningful as a component.
2. **Noisy-OR over `max` for the presence union**, with the dead-gradient
   argument for why, plus the ablation.
3. **Repurposing `dominent_idxs`** — an existing rasterizer output currently used
   only for per-primitive error accounting — as an **identity attribution
   channel**. This is what makes C2 measurable at all, and it costs nothing.
4. **The sorted projection** for interval identifiability.
5. **The observation that "static" is a misnomer** in Ex4DGS: the static bank has
   no temporal *opacity* but its positions do drift (`_xyz + _xyz_disp·t/duration`).
   Probably a remark, not a contribution.

**Honest assessment to put to the reviewer: items 1, 2, 4 are implementation
craft, not novelty. Item 3 is the only one that changes what can be
*measured*. The novelty, if any, is the C2 claim itself — that presence as a
union of intervals preserves primitive identity across absence — and that
stands or falls on the C2 numbers, not on the code.**

---

## 7. Known weaknesses of this plan

Stated so the reviewer does not have to find them first.

* **The gap is unexplained.** The substrate reproduces at 32.67 against a
  published 33.73, with E0 proving the eval stack exact and the point-cloud
  swap refuted. We build on a baseline 0.59 dB below the paper and do not know
  why.
* **Fixtures, not physical absence.** The absence events are authored
  (SA4D-edited). A prior cross-model review already returned FAIL on this point
  for an earlier thesis version. Whether authored absence licenses the claim is
  a live question, not settled here.
* **Nothing optimises the trajectory across a gap.** No gradient reaches the
  keyframes during an absence, yet `expand_duration` and `motion_reg` will
  extrapolate a smooth path straight through it. The pose at return is therefore
  whatever extrapolation produced, and may be wrong in a way PSNR in a small ROI
  will not catch.
* **Single-scene risk.** The fixtures exist for three scenes; the reproduction
  is validated on one.
* **n=1 intuitions are worthless here.** The measured spread across six
  identical runs is 0.91 dB.
* **No result yet distinguishes this from simply giving K=1 more primitives.**
  A4 is designed to, but A4 is a control we have designed ourselves and the
  reviewer should attack it.
