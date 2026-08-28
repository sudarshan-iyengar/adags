# LoRA phase-alignment ablation — frozen spec (2026-08-27)

Operational/engineering record. **EXPLORATORY**, `evidence_bearing: false`.
Append-only. **Written and committed BEFORE any cell was submitted.**

## 1. Why this exists

A novelty check on the LoRA motion model returned **2.5/10, ABANDON the
current framing** (reviewer `gpt-5.6-sol` at xhigh, Codex thread
`01a0438b-ff03-7900-8e4b-edc04ca69b61`; full prompt and response saved locally
under the untracked `.aris/traces/novelty-check/2026-08-27_run01/`, which is
gitignored and therefore NOT recoverable from this repository). The core equation — a globally shared low-rank trajectory
dictionary with per-primitive coefficients — is **DynMF** (Kratimenos, Lei,
Daniilidis, ECCV 2024, arXiv:2312.00112), which uses `B = 32` bases on N3DV,
and behind DynMF sits classical trajectory-basis NRSfM (Akhter et al. 2008).
Shape of Motion (arXiv:2407.13764) and Motion Trajectory Field
(arXiv:2508.07182) crowd the same cell.

Exactly **one** mechanism survived that review as possibly-not-occupied: this
implementation indexes the shared dictionary at time **relative to each
primitive's own temporal centre**, `B_r(t - t_i)`, not at absolute time.
Unless the basis family is closed under translation, that is **not** a rank-R
factorization in absolute time — it is a shifted/time-warped dictionary.

This ablation tests whether that difference is load-bearing. It is the
cheapest thing that can either rescue the direction or retire it.

## 2. The structural fact the design rests on (measured, not assumed)

The anchor coordinate is `u = ((t - t_i)/d + 1)/2`, clamped to `[0,1]`, over a
32-anchor table. For `t` spanning the sequence, the window is **always exactly
half the table wide** and its centre is set by `t_i`:

| `t_i` | reachable `u` | anchors reachable |
|---|---|---:|
| 0.0 | [0.500, 1.000] | 15.5 / 31 |
| 2.5 | [0.375, 0.875] | 15.5 / 31 |
| 5.0 | [0.250, 0.750] | 15.5 / 31 |
| 7.5 | [0.125, 0.625] | 15.5 / 31 |
| 10.0 | [0.000, 0.500] | 15.5 / 31 |

**No primitive ever addresses more than half the anchor table.**
`motion_lora_anchors: 32` therefore buys ~16 effective anchors per primitive
and uses the table as a phase-indexed atlas. This was not previously recorded
and it is why the naive control is confounded (§3).

## 3. The arms

One variable: `motion_lora_time_reference`.

| arm | `u` | role |
|---|---|---|
| **P** `primitive` | `((t - t_i)/d + 1)/2` | incumbent — the current design |
| **G-M** `global_matched` | `0.25 + 0.5(t - t0)/d` | **PRIMARY control**: same half-width window, same temporal resolution, same parameter count; ONLY the per-primitive phase removed |
| **G-F** `global` | `(t - t0)/d` | SECONDARY: the DynMF convention. Removes phase AND doubles resolution — **confounded by construction**, never to be cited as evidence about phase |

`global_matched` exists because `global` alone would repeat the error this
project already recorded: **a control must separate MAGNITUDE from
CORRECTNESS** ([[block-2026-08-23-schedule-amendment]] §2b). Verified
algebraically before submission: at identical `t_i = midpoint`, P and G-M agree
to **2.4e-08 in u (float32 exact)**, while P and G-F differ by **0.25 in u**.

Everything else is the 181 protocol
([[stg-n3v-protocol-parity-2026-08-19]] Appendix C): `cut_roasted_beef`,
frames 0-49, 1352x1014, `cam00` held out, batch 2, 6,000 iterations, 600k cap.
`route_logit_init: 4.0` is pinned explicitly in all arms — experiment 181
trained at an effective 4.0 despite its config saying 0.0 (the value did not
bind until the 2026-08-20 repair), and 4.0 is worth ~+0.50 dB, which is larger
than the contrast under test. Shared by all arms, so it cannot separate them.

## 4. PRECONDITION — evaluated on the SETUP, never on the score

Per [[block-2026-08-24-handover]]: *a frozen reading rule is not enough; every
frozen rule needs a frozen precondition asserting the mechanism it reads was
actually exercised.*

**V1 (non-vacuity of the mechanism).** The arms differ only insofar as `t_i`
are dispersed. If the temporal centres collapse, P and G-M coincide **by
construction** and no score from those cells says anything about phase.

> **V1 passes iff the middle-90% span of `get_t` at the END of training is
> >= 0.5 of the sequence duration, in every cell of both arms.**

`create_from_pcd` initializes `t_i ~ U(t0 - 0.1d, t0 + 1.1d)`, so V1 holds at
initialization by construction; it is trainable, so it must be re-checked at
the end. `tests/test_lora_time_reference.py::temporal_centre_dispersion`
computes it.

**V2 (the arms actually diverged).** Held-out PSNR of P and G-M must differ by
more than 0 in at least one cell, and the run summaries must record different
`motion_lora_time_reference` values. A silent revert to the default arm — e.g.
via a branch-from-checkpoint that dropped the field — makes the comparison
vacuous; `capture()` carries the field and a test pins that.

**If V1 or V2 fails, the ablation is INVALID, not negative.** Record it as
untested, exactly as [[lrv4-starved-fixture-result-2026-08-23]] was.

## 5. READING RULE — frozen before submission

Primary contrast: **P minus G-M**, pooled+clamped held-out PSNR (`cam00`,
frames 0-49), paired on shared seeds `{0, 1, 2}`.

The measured same-code replicate floor at this protocol is **0.4945 dB**
([[block-2026-08-23-schedule-amendment]], corroborated at 0.4913 by the n=6
variance study on a disjoint cohort). The threshold is set from that measured
floor, not chosen — the `delta* = 0.30` category error is on the record and is
not repeated here.

> **Phase alignment is SUPPORTED iff the paired mean (P − G-M) exceeds
> +0.50 dB AND all three per-seed differences carry the same sign.**
>
> Otherwise: **NO RESOLVABLE EFFECT at n=3.** The honest conclusion is that
> per-primitive phase is not the load-bearing inductive bias, the last
> unoccupied mechanism in the motion model is not doing work, and the
> reframing the novelty check proposed is **retired**. This is a real outcome,
> not a failure — it is the cheapest available exit.

A negative does **not** license re-running at a lower threshold or a larger n:
n=9 would buy a better estimate of a refuted endpoint
([[block-2026-08-24-handover]] §6 stopping rule).

Secondary, reported but never load-bearing: **P minus G-F** — "is the standard
convention simply better here?" Confounded with temporal resolution by
construction (§3).

## 6. Cost and what is NOT being run

9 cells would be ~22 slot-h against a 24 slot-h block ceiling. **Only the
decisive 6 are submitted** (P and G-M, 3 seeds each, ~15 slot-h); the G-F arm
is written, committed and held. Apollo is under load and the primary question
does not need G-F.

Not being run, deliberately: any hyperparameter sweep over rank/anchors. Under
the variance study's own binding rule a two-arm comparison needs 37
replicates/arm (181 slot-h); a 12-arm grid is ~1,090 slot-h. That was settled
in [[block-2026-08-24-handover]] §6 and is not reopened here.

## 7. Status

Spec frozen 2026-08-27, before submission. Results append below, never above.

---

## RESULT (2026-08-28, append-only) — NOT RESOLVED at n=3

Run on **Leonardo** (`boost_usr_prod`, A100-SXM-64GB, account `euhpc_d36_068`),
not Apollo. The Apollo attempt of 2026-08-27 never produced a cohort: the three
`global_matched` cells stalled short of 6,000 iterations and `det` became
unresponsive. Deviations from §3, recorded rather than absorbed:

* **pool**: Leonardo A100 replaces Apollo `dgx`/`hopper`. All six cells share
  it, so it cannot separate the arms — but it is not the hardware the 33.5050
  reference was measured on.
* **the seed genuinely varies now.** `scripts/run_leonardo.sh` never passed
  `--seed`, so every Leonardo run had silently taken main.py's default 6666.
  The Apollo cohort could not have varied it either. Repaired in `ec6075d`,
  verified per cell in `meta/run_info.txt` before reading any score.

### Gates, evaluated before the contrast

| gate | result |
|---|---|
| **V1** non-vacuity: worst middle-90% `get_t` span | **1.139** of the sequence against a 0.5 floor — **PASS** |
| **V2** arms diverged: bit-identical P/G-M pairs | **none** — **PASS** |

V1 passes with room to spare, so the mechanism was genuinely exercised: the
temporal centres stayed dispersed across the whole sequence and the two arms
were reading the shared dictionary at materially different phases. This is not
a null from a dead instrument.

### Cells

| arm | seed | PSNR | t-dispersion |
|---|---:|---:|---:|
| primitive | 0 | 33.4373 | 1.178 |
| primitive | 1 | 33.4738 | 1.177 |
| primitive | 2 | 33.8420 | 1.176 |
| global_matched | 0 | 33.3140 | 1.146 |
| global_matched | 1 | 33.2965 | 1.146 |
| global_matched | 2 | 33.4295 | 1.139 |

P mean **33.5844** (spread 0.4047), G-M mean **33.3467** (spread 0.1330).

### The frozen rule, applied

| seed | P − G-M |
|---|---:|
| 0 | +0.1233 |
| 1 | +0.1773 |
| 2 | +0.4125 |
| **paired mean** | **+0.2377 dB** |

* sign consistency: **3 of 3 positive — PASSES**
* magnitude: **+0.2377 against the +0.50 floor — FAILS**

**VERDICT: NOT RESOLVED at n=3. Phase alignment is not established as the
load-bearing inductive bias, and the reframing
[[../.aris novelty check 2026-08-27]] proposed is retired.**

### What this does and does not say

It does **not** say the effect is zero. Every seed favours the primitive arm,
and 3-of-3 in the predicted direction is p = 0.125 one-sided — suggestive, and
nothing more.

What it says is that **the effect, if real, is about +0.24 dB, which is below
this protocol's own resolution.** The measured same-code replicate floor is
0.4945 dB, and the primitive arm's own three-seed spread here is **0.4047 dB —
larger than the contrast being measured.** A difference smaller than the noise
of one arm is not a finding at this n, which is exactly why the floor was fixed
before the cells ran rather than after.

**The stopping rule fires and n=9 is deliberately NOT run.** Per §5 a negative
does not license a lower threshold or a larger sample; that would buy a better
estimate of an endpoint the spec already declared. The honest close is that the
one mechanism the novelty check identified as possibly unoccupied does not
carry measurable weight at the protocol this project can afford.

### Two things worth carrying

**A cross-platform corroboration nobody asked for.** The primitive arm — the
incumbent design under the 181 protocol — reads **33.5844** here against the
**33.5050** recorded on Apollo `dgx` (experiment 194), a difference of
**+0.079 dB** across different hardware, a different Slurm site, a different
allocator setting and a genuinely different seed. That is well inside the
replicate floor and is the strongest evidence to date that the Leonardo tree
reproduces the Apollo protocol.

**The arms differ in cost, and it was not predicted.** `global_matched` cells
ran **2h39m** against `primitive`'s **1h44m** — roughly 53% slower for
arithmetic that should be near-identical. Not diagnosed here; the plausible
cause is a different densification trajectory rather than the mapping itself,
and it is recorded because a per-arm cost asymmetry can bias any future
comparison run under a wall-clock budget rather than an iteration budget.
