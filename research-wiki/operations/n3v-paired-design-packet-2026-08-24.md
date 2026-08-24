# PACKET (DESIGN ONLY, NOT SUBMITTED) — paired fixed-topology protocols
# for the N3V variance problem (2026-08-24)

EXPLORATORY. **Nothing here is proposed for submission and no option is
preferred.** The 2026-08-24 block directive requires this be designed but
NOT automatically submitted, and lists five conditions that must all hold
before any paired mechanism experiment runs (§5 below). At least one does
not hold today.

These address lever 4 of
[[membership-occupancy-and-decision-2026-08-23c]] §2 — the only lever that
attacks the measured *cause* (densification-amplified divergence) rather
than the symptom. Levers 1 and 3 are effectively closed; see
[[n3v-variance-study-spec-2026-08-24]] §9 and §2.2.

## 1. What the densification path actually does — verified in code

* The densification block is gated on `iteration < opt.densify_until_iter`
  and the point cap (`main.py:1648`); `densify_and_prune` fires when
  `iteration > densify_from_iter` and `iteration % densification_interval
  == 0`.
* With `configs/n3v/ladder_b1_crb.yaml` (`densification_interval: 100`,
  `densify_from_iter: 500`, `densify_until_iter: 6_000`) that is iterations
  **600, 700, …, 5900 — 54 densification rounds.**
* **SELECTION is deterministic given state, with no RNG.**
  `densify_and_clone` selects `grad_norm >= grad_threshold` AND
  `max(get_scaling) <= percent_dense * extent`
  (`scene/gaussian_model.py:2233-2239`); the split path uses the analogous
  padded-gradient test.
* **The new-row OFFSETS are random**: `torch.normal(mean=means, std=stds)`
  at `:2115` (spatial) and `:2135` (temporal). `rot_4d: False` in this
  config, so those two draws are the live ones.
* **Pruning lives inside the same call**, so anything that disables
  densification also disables opacity/size pruning.
* **There is NO persistent per-row identity column.** `_elgs_family_ids`
  and `_packet_ids` are group-level and are *inherited* through
  clone/split; row indices shift at every prune.
* **Checkpoint round-trip exists and carries the Adam state**
  (`gaussian_model.py:550-556`), so branching from a common checkpoint
  needs no new machinery.
* **B1's own mechanism is OUTSIDE the densification gate.**
  `maybe_packet_birth` is called at `main.py:1758`, past the `:1648` gate,
  and is a **point-neutral relocation** — it rewrites rows in place and
  does not change the row count. **So a fixed-topology design does not
  disable B1's mechanism**, which is what makes options 2 and 3 coherent
  at all.

**The mechanistic consequence, and it matters for option 3:** divergence
has two sources, not one — the random offsets, *and* selection flipping
once accumulated float differences cross a threshold. Replaying recorded
*decisions* pins only the second. **A faithful option 3 must pin the
offsets too**, or it will reproduce the decisions and still diverge.

## 2. The three options

### Option 1 — common checkpoint, independent later densification
Both arms resume from one checkpoint at iteration `k`; densification then
proceeds independently.

* **Estimand:** the *total* effect conditional on one particular shared
  state `S_k`. Generalization beyond that `S_k` is not licensed.
* **Suppresses:** pre-`k` initialization and trajectory variance only.
* **Cost:** cheaper than independent runs — the two arms share `k` of the
  6,000 iterations.
* **New code:** **none.** Checkpointing and `--start_checkpoint` exist.

### Option 2 — common checkpoint, densification DISABLED after branching
As option 1, then `densify_until_iter: k` so topology is frozen at the
branch point.

* **Estimand:** a **DIRECT effect conditional on fixed topology.** This is
  a different scientific question from the one the ladder asked.
* **Suppresses:** every topology-mediated effect — including any part of
  the mechanism's real benefit that operates *through* densification. It
  also disables pruning, because pruning shares the call.
* **CHANGES THE SUBSTRATE**, and must never be reported as if it measured
  the same thing as an ordinary arm.
* **Cost:** cheapest of the three.
* **New code:** **none** — it is `densify_until_iter: k` alone.

### Option 3 — recorded and replayed identical densification decisions
Record the per-round decisions (and, per §1, the offsets) from a control
run and replay them in the intervention arm.

* **Estimand:** the effect with **intervention-mediated topology changes
  suppressed**, the baseline's topology pinned throughout.
* **Suppresses:** exactly the intervention's own influence on topology —
  the narrowest of the three, and the closest to "shared trajectories".
* **Cost:** ~2x a single run; nothing is shared.
* **New code:** **substantial.** No persistent per-row id exists, row
  indices shift at every prune, and the offsets are random draws that must
  also be captured and replayed. This is audited machinery, not a config
  change.

## 3. Comparison — NO SELECTION IS MADE HERE

| | option 1 | option 2 | option 3 |
|---|---|---|---|
| estimand | total effect \| shared `S_k` | **direct** effect \| fixed topology | controlled direct effect, topology pinned to control |
| suppresses | pre-`k` variance only | all topology-mediated effect | intervention-mediated topology change |
| substrate changed? | no | **yes** | no |
| new machinery | none | none | substantial |

## 4. A BLOCKING PREREQUISITE for options 1 and 2 on a B1 arm

**`_packet_ids` is absent from `capture()`/`restore()`**
(`scene/gaussian_model.py`; it appears only at `:241` init, `:1731-1732`
prune slice, and `:1984-1995` append/inherit). **Any
branch-from-checkpoint on a packet-birth arm therefore silently loses the
packet-id column across the branch point** — and `scripts/consolidate_packets.py`
and the whole B2 lane consume it.

**This must be repaired before options 1 or 2 are run on a B1 arm.** It
was deliberately NOT repaired this block, because a fix changes
training-path bytes and the current variance cohort's entire value rests
on being byte-identical to the code that produced the 0.4945 floor.

## 5. The directive's five conditions — evaluated

No paired mechanism experiment may be submitted unless ALL hold:

| condition | status |
|---|---|
| the fresh variance result is terminal | **NOT YET** — the n=6 cohort is still running |
| the paired design and estimand are frozen | not frozen; this packet is design-only by instruction |
| an identical-arm placebo and a localized positive control are specified | **NOT SPECIFIED** |
| a focused implementation review passes | not run |
| expected cost inside the block ceiling | option 3 unpriced; options 1/2 affordable |

**So no paired experiment is authorized, and none was submitted.** This
packet is the "implementation-complete or frozen launch packet" the
directive asks be left behind instead.

## 6. What a future spec must add

An **identical-arm placebo** (two arms that differ in nothing, run through
the same paired machinery) is the check that the pairing itself does not
manufacture a difference — and it is not optional here, because this
project has already been caught once by a control that was matched on
everything enumerated and differed in something unenumerated. A
**localized positive control** — an intervention with a known, large,
localized effect — is what distinguishes "the paired design has power"
from "the paired design returns null on everything".
