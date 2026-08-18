# User decision memo — 2026-08-18

Eleven decisions. Each states what is actually blocked, what evidence
exists, and a recommendation. **None of these is taken here.** They are
scientific-scope choices that belong to the project owner.

The single most consequential fact: **the standing event-supply decision
has been open since 2026-08-12 and blocks nine of the other ten.** It is
decision 2. Everything else is downstream or independent.

---

## 1. Census-level correction versus re-tracking — the M-2 branch

**Blocked on:** experiment 149, submitted 2026-08-18, **result pending at
the time of writing.**

The absence instrument's dominant limb — 87.60% of report-pairs stamped
`LOW_VISIBILITY` because the tracker's confidence `v < 0.5` — has never
been split into "the tracker was right but unconfident" (`ON_ELIGIBLE`)
and "the tracker drifted off the object" (`OFF_ELIGIBLE` / `BACKGROUND`).
The design and its decision rule were frozen before any outcome
([[elgs-m2-oncomponent-split-design]]):

| `p_on` pooled | verdict | consequence |
|---|---|---|
| `>= 0.70` | drift not dominant | census-level threshold correction admissible: ~1 CPU-hour, ZERO GPU-hours, no re-tracking |
| `0.30 – 0.70` | mixed | a threshold change alone is inadmissible; the corrected instrument must gate on component membership too — still census-level, but a NEW instrument needing its own prereg and adversarial round |
| `<= 0.30` | drift dominates | lowering the threshold makes the instrument WORSE; the census branch is REFUTED and re-tracking or a different presence instrument is required |

**Recommendation:** none — this is the one decision that should follow
mechanically from a number, and the number is not in yet. Read the
verdict, then act on the table.

## 2. Event-supply route — THE standing decision, open since 2026-08-12

**Blocks:** the calibration boundary, hence all five fitted evidence
heads, `r_u`, and `rho`; the sequence for the matched triple; every M2
precondition.

M1 failed and its continuation cycles found **zero eligible DiVa-360
sequences** after the substrate defect was corrected. Options:

| route | what it costs | what it buys |
|---|---|---|
| **tranche-2 short screening (20 seqs)** | GPU-hours + storage; only sensible under a corrected instrument | more chances at return supply |
| **long-sequence link** | needs a user-supplied link, storage and GPU ceilings | longer sequences plausibly carry more returns |
| **R4 dataset extension** | see decision 8 | supply from outside DiVa-360 |
| **R3 descope** — drop reactivation; EL-GS becomes fission/truncation/birth on the occlusion supply, which STANDS at 239,545 | zero new compute | a defensible smaller claim; novelty falls to the recorded 6.5–7.0 fall-back |
| **synthetic controlled scene** — see decision 3 | modest | exercises C1/C2 mechanics with ground-truth absence while real supply is unresolved |

**Recommendation: resolve decision 1 first, then choose between R3 and a
synthetic testbed** — and treat tranche-2 screening as conditional on
M-2 returning an admissible census-level correction. Screening 20 more
sequences with a known-defective instrument would repeat the cycle-2
spend for the same reason it failed. The occlusion supply is the one
limb that has never been in doubt, and R3 is the only route that needs no
new measurement at all.

## 3. Synthetic controlled event scene — admit or not

The record contains no dataset where reactivation can be exercised
against **ground-truth absence**. Every real candidate measures absence
through a tracker, which is precisely the instrument now known to be
defective.

**Recommendation: ADMIT, as a mechanism-validation testbed only.** It is
the only place C1 and C2's operation-level metrics — return timing, false
reactivation rate, identity retention — can be measured against truth
rather than against a tracker's opinion. It can never carry C3, which
needs real data, and it must be labelled that way from the start so it is
never quietly promoted.

## 4. The three unhoused structural constants

`anchor_report_floor` (8), `plateau_seed_fraction` (0.5) and the
report-population bound have **no prereg home** — verified as zero
occurrences in both prereg files. (Contra the strategic audit, `rho` IS
housed, in `prereg_observability_v1.json`.)

**Recommendation: house all three as STRUCTURAL constants frozen by
disclosure**, in a new `prereg_structural_search_v1.json`. They are not
estimates — there is no likelihood to maximise — and fitting them to data
would select the structural search on its own outcome. `plateau_seed_fraction`
was already user-directed on 2026-08-14 after the previous predicate was
measured inert. Full reasoning: [[elgs-evidence-head-freeze-packet-2026-08-18]].

## 5. Evidence-head freeze

**Cannot be executed** until decision 2 names the admitted sequence set,
because every head's authorized data is "disjoint from
dev/locked/held-out" and there is currently nothing to be disjoint from.
Containment is already fail-closed in code, so nothing is at risk in the
meantime. No decision needed beyond decision 2.

## 6. C/F/X rendered-flow supervision — keep shelved?

The renderer work of 2026-08-17/18 **did** make the flow gradient live
(experiment 138: `flows` 2.1435 under a flow loss, 0.0 under a colour
loss). So the blocker that stopped C/F/X is gone.

That does not make it worth running:

* no EL-GS claim has a flow term, and DiVa-360 — the primary dataset —
  has no flow at all;
* the chosen N3V scene's flow has median magnitude 0.015–0.035 px, too
  small to establish even the sign;
* the flow-mediated opacity gradient disagrees with finite differences by
  ~35% with a sign flip, so F cells would train on an unverified coupling;
* "broad flow supervision without reliability" is a preserved weak idea
  in this project's own negative-results inventory.

**Recommendation: keep shelved.** If it is wanted anyway, the
preconditions are: pick a scene and stride with pixel-scale flow, verify
the sign on high-magnitude pixels, verify the flow→opacity coupling
oracle-independently, then run S-C / S-C(seed 2) / S-F / S-X for ~8
GPU-hours — and it still would not be a claim.

## 7. The dead depth and alpha gradients — repair or leave?

**New finding, 2026-08-18.** `dL_depths` and `dL_masks` reach the
launched backward kernel and are never read; their only reads are in the
unlaunched `renderCUDA`. **A loss on rendered depth or rendered alpha
produces no Gaussian gradient at all** — silently, the same failure mode
as the `max_contrib` guard. Pre-existing; deliberately not bundled into
the integrity repair.

**Recommendation: repair it, as its own bounded change, before any depth
or alpha supervision is ever attempted** — and note that this retires
"depth supervision" as a cheap option, since it would have silently done
nothing.

## 8. Dataset roles and external intake

Deferred to the dataset-admission matrix produced in this block. The
binding criterion for any new dataset is not resolution or scene count:
it is whether the dataset can supply, **and let us measure**, genuine
full-applicable-camera disappearance and return — measured independently
of tracker failure. A frontal or planar rig cannot, for the same
geometric reason that retired the P03 representation.

ImViD and MPEG-GSC intake procedures are prepared, not performed; neither
access control was touched.

## 9. Retire the v1 N3V Gate A/B protocol?

`query_pack.md` still calls the 2026-07-29 CSVL-VPL v2 section "the last
user-approved direction record" while ALSO recording the 2026-08-09 EL-GS
approval. Two direction records are simultaneously authoritative.

**Recommendation: formally record v1 as SUPERSEDED by EL-GS**, preserving
its text per the append-only rule. This is bookkeeping with real
consequences — the two protocols imply different gates and different
datasets.

## 10. Sequence for the matched presence-substrate triple

Follows from decisions 1 and 2. **Note the blocker first:** a bare
comparison is currently invalid because an EL-GS cell trains on ~75% of
units while a temporal-substrate control trains on 100% — a 33%
training-data advantage to the control, on a benchmark whose open
question is held-out generalization. The minimal repair is specified in
[[elgs-matched-triple-readiness-2026-08-18]] and is NOT implemented.

**Recommendation: do not launch the triple until that filter exists**,
whatever sequence is chosen.

## 11. Does scissor remain a legitimate development scene?

Scissor was excluded in cycle 2 on the **coverage** limb (0.441 < 0.5) —
and coverage shares the very `v >= 0.5` constant the instrument
diagnostic found defective. It is also the only tranche-1 sequence with
≥12 union returns (75). So its exclusion is contingent on decision 1.

Meanwhile roughly 50 projected GPU-hours of photometric development ran
on scissor with `elgs_enable: False` throughout.

**Recommendation: wait for M-2.** If the census-level correction is
admissible, scissor's eligibility must be recomputed under it before any
further development spend lands there — and if it is not admissible, the
development substrate should move to whatever sequence decision 2 names.

---

## What was NOT decided here, deliberately

Every item above. This memo's function is to put each choice in front of
the owner with its evidence and cost, not to pre-empt it. The two that
should be taken FIRST, because they unblock the rest, are **decision 1**
(mechanical, once experiment 149 lands) and **decision 2** (genuinely
scientific, and open for six days).
