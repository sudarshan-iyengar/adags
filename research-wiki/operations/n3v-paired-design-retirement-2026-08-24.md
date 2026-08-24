---
title: The paired design cannot support a B1-vs-B0 comparison at the block ceiling
date: 2026-08-24
evidence_bearing: false
---

# Option 1 is retired for the mechanism comparison, at ZERO GPU cost

EXPLORATORY. Successor to [[n3v-paired-design-packet-2026-08-24]], which
designed three options, selected none, and listed *"an identical-arm placebo
and a localized positive control are specified"* as **NOT SPECIFIED**.

**Outcome: no screen is frozen and none is recommended.** Specifying it
carefully is what showed it cannot work. The products are a decision, three
durable code facts, and corrected arithmetic — all without compute.

## 1. THE DECISIVE INTERLOCK — cost and mechanism are in direct opposition

A paired B1-vs-B0 comparison needs **births after the branch** (packet birth is
a point-neutral in-place relocation, so a birth inside the shared prefix is
identical in both arms) and **a large shared prefix** (or pairing buys no
correlation). Those are not independent, and the cost model makes the conflict
exact — **with no threshold chosen anywhere**:

```
cost per pair = 2.4443 x (2 - k/6000) slot-h        [k shared + 2 x (6000-k) divergent]
7 pairs within the 24 slot-h ceiling  =>  cost/pair <= 3.4286  =>  k >= 3584
```

Packet birth fires at `{1000, 1500, 2000, 2500, 3000, 3500, 4000}`
(`scene/packet_birth.py` `should_fire`, `from 1000 / until 4000 / interval
500`, both bounds inclusive). **After `k = 3584` only the 4000 birth remains.**

> **Any branch point affordable at the block ceiling leaves at most 1 of 7
> births post-branch.**

And a smaller prefix is **strictly more expensive**, so there is no escape by
accepting less correlation:

| `k` | births post-branch | cost/pair | 7 pairs |
|---:|---:|---:|---:|
| 900 | 7 / 7 | 4.52 | **31.7** |
| 1500 | 5 / 7 | 4.28 | **29.9** |
| 2000 | 4 / 7 | 4.07 | **28.5** |
| 3250 | 2 / 7 | 3.56 | **25.0** |
| **3584** | **1 / 7** | 3.43 | **24.0** |

**Option 1 cannot support a B1-vs-B0 comparison within the ceiling.** This is
arithmetic on the frozen schedule and the measured per-cell cost — not an
experiment — and it did not need the ~22 slot-h a screen would have cost.

**Scope, stated because the premise is a config field, not a constant.**
`packet_birth_from_iter` / `until_iter` / `interval` are `OptimizationParams`.
A different schedule — say births confined to 3500–4000 — would change this
conclusion. **The retirement is of option 1 under the frozen
`ladder_b1_crb.yaml` schedule**; a schedule change is a different mechanism and
needs its own spec, including why the new schedule is not chosen to rescue the
design.

**This supersedes the screen.** An earlier version specified a 6-pair placebo
at `k = 3250` and then conceded, in its own final section, that `k = 3250`
cannot support a mechanism comparison — a ~22 slot-h purchase that could not
license anything at any outcome. That contradiction is why this packet exists.

## 2. Three durable code facts, newly verified

Reusable regardless of the decision above.

**(a) A localized, time-windowed opacity control ALREADY EXISTS, and is
reachable by configuration alone.** `gaussian_renderer/__init__.py`
`_apply_visibility_event_gate` selects rows by a **2D `crop_xyxy` screen box**
via `project_points_to_screen`, restricts them to an inclusive
**`[frame_start, frame_end]` window**, and multiplies **activated** opacity by
a tunable `opacity_attenuation`. It is wired unconditionally from `opt`
(`main.py:1058`, `:1147`) and self-disables when no manifest is set
(`main.py:606-612`).

**It needs no code change:** a **tracked** manifest exists under
`refine-logs/hide_reveal_poc/`, and **tracked** configs already drive the gate
with all seven keys (`configs/n3v/visibility_event_train_6000.yaml`).
`ladder_b1_crb.yaml` simply does not set them. A second multiplier —
`visibility_event_training_scale`, a warmup ramp — also applies and must be
accounted for in any calibration.

**This corrects an error in the previous version**, which declared a localized
control impossible because region A is a 2D bbox with no 3D referent, because
`_opacity` is a pre-activation logit whose halving *brightens* most rows, and
because opacity is time-invariant. **All three are true of `_opacity` and all
three are irrelevant** — the gate acts at render time on activated opacity and
never touches the parameter, so it also carries **no pruning confound and no
`inverse_sigmoid` hazard**, two problems the previous version spent a section
mitigating.

**(b) Branching at `k` silently discards iteration `k`'s densification round,
optimizer step, and packet birth.** Within one iteration the order is
`scene.save` (`main.py:1655`) → densification gate (`:1658`) →
`densify_and_prune` (`:1699`) → `optimizer.step` (`:1724`) →
`maybe_packet_birth` (`:1768`). Resume sets `iteration = first_iter` (`:1240`)
and increments before the body, so a run resumed from a checkpoint stamped `k`
begins at `k + 1`. **The checkpoint captures state BEFORE iteration `k`'s work,
and that work never happens in either arm** — branching at a birth iteration
destroys one of seven mechanism firings with no error and no log line. Each
branched arm also runs one fewer optimizer step than a continuous run.

Mechanically, branching at an arbitrary `k` **is** available: `scene.save`
writes `chkpnt<k>.pth` (`scene/__init__.py:104`) whenever
`k in saving_iterations`, so `k` must be added to `--save_iterations` (default
`[3000, 6000, ...]`). The separate `chkpnt_best.pth` (`main.py:1651`) is
written only inside `if test_psnr >= best_psnr:` and is not a periodic path.

**(c) The 600k HARD cap provably never fired.** `main.py:1658` gates the
densification block on a **conjunction** — `iteration < densify_until_iter`
**and** (`densify_until_num_points < 0` **or** `count < densify_until_num_points`).
There are five point-removing call sites in `scene/gaussian_model.py`, but
every one is reachable only through `densify_and_prune`, called only at `:1699`
inside that gate, and packet birth is point-neutral by assertion. So once the
count limb closes, the count can never fall again, and the recorded finals of
**599,396 / 599,448 / 599,406 / 599,470** are proof it never closed.

A previous version listed this as an open precondition and proposed reading a
checkpoint's row count. That was wrong twice: the question was already answered
by data on the record, and a row count at one iteration could not have ruled
out an earlier crossing. Note also that **`max_total_points` is not a separate
config field** — it is the parameter name of `densify_and_prune`, bound to the
same `opt.densify_until_num_points`. There is **one knob read twice**, not a
hard cap plus a soft cap; within the gate it truncates new points to
`600000 - count`, which binds every round.

## 3. Corrected sizing arithmetic

A paired comparison at `delta* = 0.30`, `alpha = 0.05` two-sided, power 0.80
needs

```
n_pairs = k * sigma_d^2 / delta*^2 + z^2_{a/2}/4,   k = (z_{a/2} + z_b)^2 = 7.8489
```

— one-sample on the differences, so **no factor of 2**, with the **Guenther
correction (`+0.9604`)** the variance spec binds itself to. **Verified against
that spec:** the two-sample form returns
`2 x 7.8489 x (0.4530/0.30)^2 + 0.9604 = 36.75 -> 37`, reproducing the recorded
37 replicates/arm exactly. An earlier version omitted Guenther, making every
threshold ~8.4% **too lenient** — the anti-conservative direction.

**Affordability at ZERO variance reduction** (`sd_d = sqrt(2) * sigma`), at
`k = 3250`. **Both rows are n=6 figures available only under the variance
spec's labelled secondary relaxation** — under the primary frozen reading the
study is INCONCLUSIVE at n=6 and the fresh cohort is n=3 with `sd = 0.125114`:

| sigma (n=6, RELAXED reading) | value | paired | independent | saving |
|---|---:|---:|---:|---:|
| point estimate | 0.1847 | 7 pairs = **24.95** | 7/arm = **34.22** | 1.37x |
| upper limit (the spec's binding rule) | 0.4530 | 37 pairs = **131.87** | 37/arm = **180.86** | 1.37x |

**Neither is affordable at a 24 slot-h ceiling under either convention**, so
prefix-sharing alone does not rescue the lane and a genuine variance reduction
would have been required — which, per section 1, cannot be bought where the
mechanism can act.

**A further comparability limit.** The variance cohort's `sigma` was produced
at a commit whose training-path diff against HEAD is **not** empty — the
`capture()`/`restore()` repair and the fail-closed guard add ~45 lines across
`main.py` and `scene/gaussian_model.py`. Those lines are inert for a fresh run
from scratch, but the spec's exclusion rule names `commit` and
`archive_sha256`, so pooling would be a **third** labelled relaxation and is
not assumed here.

## 4. Corrections to earlier versions of this packet

* **"The localized control is impossible" — WITHDRAWN.** See 2(a).
* **"Does the 600k cap bind before `k`?" — WITHDRAWN as already closed.** 2(c).
* **"A 3-pair screen CANNOT return an ADOPT verdict" — OVERSTATED.** ADOPT was
  well-defined; the required reduction was merely implausible. Calling it the
  block's vacuity class was wrong.
* **"6.83x / 2.66x required reduction" — WITHDRAWN.** They mixed conventions,
  inflating the screen's sd to its upper limit while comparing against a
  point-estimate `0.2612`, and were **standard-deviation** ratios described as
  *variance* reductions.
* **The saturation premise — WITHDRAWN.** It quoted the s0 row of the b1f
  post-mortem as flat while the s1 row rises, and those rows are **B1-F vs
  B1-X** — different configs, executing identical code only to iteration 1000 —
  so the later windows cannot speak to same-code divergence at all. The
  same-code evidence is two points per seed, and on both seeds the second
  exceeds the first. The prediction built on it goes with it.
* **The "≥5 of 7 births / >50% prefix" bars — WITHDRAWN as undeclared
  judgment calls**, replaced by section 1's threshold-free interlock.
* **`k = 3000`, then `k = 3250`** — both moot; 3000 additionally destroys a
  birth per 2(b).
* **A pass condition requiring "the 6-pair design detects the effect" was
  priced as "one further pair"** — a 6x cost error, with no test ever named.

## 5. What remains open

**Option 2** (freeze topology after the branch) is untouched by section 1,
because it changes *what* diverges rather than *when*. It remains the declared
fallback and would need its substrate change disclosed in every number it
produces. **Not specified here; no cost claimed.**

**Option 3** (shared densification decisions) likewise: per the predecessor
packet it needs new machinery and must pin random offsets, not merely replay
decisions.

**N3V utility scaling remains HALTED** regardless.

## 6. Method notes

**Specifying an experiment carefully is a cheap way to discover it cannot
work.** Section 1 is arithmetic on a schedule frozen long before; it surfaced
only when the design was written precisely enough to state what the branch
point had to do simultaneously.

**Prefer an interlock to a threshold.** The first version of section 1 argued
from two chosen bars — "≥5 of 7 births", ">50% prefix" — which invited exactly
the objection that the bars were picked. The cost model supplies the same
conclusion with **no chosen number at all**, because affordability and
mechanism coverage are functions of the same `k` pulling in opposite
directions. **Where a conclusion can be derived from an interlock rather than a
threshold, it should be.**

**A negative existence claim needs a SEARCH, not checks on the first
candidate.** An earlier draft declared a localized control impossible on three
findings that were each independently **verified and true** — and each true of
`_opacity`, the wrong object. **Three true facts composed into a false
conclusion because the question was scoped to one mechanism.** The capability
was in the render path the whole time, with tracked configs already using it.
