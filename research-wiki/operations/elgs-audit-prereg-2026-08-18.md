# AMENDED two-stage M1-A0b audit — PREREGISTRATION, not executed

Date: 2026-08-18. Status: **WRITTEN AND FROZEN, NOT SIGNED, NOT EXECUTED.**
Machine-readable frozen text: `configs/elgs/prereg_m1_a0b_audit_v1.json`
(sha256 recorded in the commit that adds it; revision 1,
`status: frozen_pending_signoff`).

**No candidate frame has been rendered, displayed, or transmitted to any
auditor, agent or panel, and none may be until the user authorizes stage 1.**
This page and its JSON are a design; nothing here is a result.

Reads [[elgs-absence-diagnostic-result]],
[[elgs-m2-oncomponent-split-result]],
[[elgs-coverage-bounding-pair-result]], [[elgs-cycle2-screening-record]],
[[elgs-exhaustive-screen-scope]], `configs/elgs/prereg_m1_census_v1.json`.

## 1. What this audit is for, and the one thing it cannot do

The tranche-1 absence instrument has one unanswered question: of the 597
screened-half true-absence candidate windows, how many correspond to
something a person looking at the footage would call *not visible in the
applicable views, then returning*? The frozen instrument's own answer is
**zero of 597 corroborated**, and M-2 then showed the visibility flag that
produced that answer carries no confidence information at all.

**The estimand is deliberately NOT physical absence.** Three quantities,
frozen in the JSON:

| symbol | definition | grade |
|---|---|---|
| `A3` | both auditors return NOT VISIBLE at every audited frame in all 3 audit cameras (any UNSURE prevents it) | diagnostic; an **UPPER BOUND** on absence |
| `A_S` | an A3-positive window where every camera in the applicable set returns NOT VISIBLE at >= 3 frames including the midpoint and the last `v == 0` frame | claim-grade ONLY with per-positive human confirmation |
| `R_S` | an A_S-positive window whose identity reappears within `r_site_census` under the sealed census return rule | claim-grade ONLY with per-positive human confirmation |

`A_S` is **full-applicable-camera unobservability, not literal physical
absence**, and the JSON forbids that phrasing outright. Human confirmation
validates the *operational visual label*; it does not observe physical
ground truth. The words "route dead" are prohibited in every outcome.

## 2. The eight mandatory corrections, and where each landed

| correction | disposition |
|---|---|
| estimand triple `A3` / `A_S` / `R_S`, with `A_S` never called physical absence | **RESOLVED** — frozen in the JSON with the prohibited phrasings listed explicitly |
| applicable-camera set frozen from calibration, never from verdicts / track visibility / mask occupancy | **BLOCKING, UNRESOLVED** — see section 3. The requirement is measurably unsatisfiable on this rig without destroying the instrument |
| simultaneous bounds at overall `alpha = 0.10`, Bonferroni `1 - alpha/m` | **RESOLVED**, and the direction is stated: raising per-sequence confidence RAISES `UB_s` and so makes the kill HARDER. Bonferroni is conservative with respect to killing |
| complete kill rule over `U_s = N_s * UB_s` with the 3-subset inequality | **RESOLVED**, and power-analysed before execution — section 4 |
| `E_eligible` / `E_indeterminate` / `E_select`, with `E_select = E_eligible` initially | **RESOLVED** — taken from A1's frozen classes; `E_select` has 10 sequences, `scissor` and `poker` are `E_indeterminate` |
| anchor association uses the sealed M-2 rule and artifact, no new rule | **RESOLVED** — the candidates and their anchors are `audit_sample_B8`'s `ltp` / `ltp_frame`, and the recomputation of that rule was checked against the sealed record on **73 of 73** windows |
| recall probe is diagnostic only | **RESOLVED** — the JSON lists its five prohibited uses |
| model output is diagnostic-grade; claim-grade needs per-positive human confirmation; decoys do NOT validate sensitivity | **RESOLVED** — and the JSON states what stays unmeasured: the instrument's miss rate on genuine full-view absence, which this design cannot estimate because no window is known to be genuinely absent |

## 3. THE BLOCKING FINDING — the applicable-camera set

The correction asks for an applicable-camera set fixed from sealed
calibration, camera validity and a predeclared capture volume, with no
dependence on candidate-specific track visibility. Three definitions were
implemented and **measured** against the sealed calibration and the frozen
73-window sample. The measurement, not an argument, is what settles it.

| definition | candidate-independent? | measured outcome | verdict |
|---|---|---|---|
| **D2** cameras containing a predeclared ball of radius `k * rig_radius` | **yes** | `\|S_seq\|` = 13 / 7 / 6 / 2 / 0 at `k` = 0.10 / 0.15 / 0.20 / 0.25 / 0.30, identically on all 12 sequences (one shared rig, `rig_radius` 5.028). The frozen anchors sit at 0.26–0.69 `rig_radius` from the centroid. Containing the anchors needs `k >= 0.69`; keeping 3 cameras needs `k <= 0.20`. | **REFUTED — the feasible radius set is empty** |
| **D1** all 26 tracking cameras, audit triple = the max-separated triple `(7, 37, 46)` at 116.2 deg | **yes** | that triple's frusta contain all three of a candidate's anchor in only **14 of E_select's 49** frozen windows (28.6%), 32 of 73 (43.8%) overall | **available but destructive** — the geometric-admissibility exclusion would cut the stage-1 sample from 49 to 14 and collapse the kill rule's decidability |
| **D3** the census generator's own per-candidate frustum rule | **no** — candidate-specific | `\|S\|` per window 3–26, pooled median 23, and **100% of the 73 windows have >= 3 applicable cameras**; best in-set triple separation 104–116 deg everywhere except `soda` | **usable; recommended** |

D2 was this page's own first proposal. It is recorded as refuted because it
was measured before it was believed, which is the only reason the audit is
not now specified around an empty camera set.

**Why D3 is nevertheless defensible, stated precisely.** D3 is
candidate-*specific* but strictly **outcome-independent**: it is a purely
geometric predicate, frozen in a signed prereg, evaluated on the sealed
candidate record, and structurally incapable of responding to an auditor
verdict. Recomputing it from calibration reproduces the sealed
`containing_cameras` field on 73 of 73 windows, so it is also verified. The
cost is a real narrowing of what `A_S` may be said to mean: *unobservability
across the cameras the candidate generator considered applicable*, not
across an independently defined set. Every `A_S` claim must carry that
clause.

**This is a scientific-semantic decision and it is NOT taken here.** Stage 1
may not execute until a reviewer or the user selects a definition in a signed
revision.

**Data-quality flag, reported rather than pooled away.**
`soda_screen_w0_171`'s single frozen window has its anchor at **1.955 ×
`rig_radius`** from the rig centroid — outside the camera sphere — with
`|S| = 3` exactly and a best triple separation of 28.7 deg. Under any
definition it is near-degenerate.

## 4. Stage-1 power, computed BEFORE execution — and it is the most consequential thing on this page

`U_s = N_s * UB_s`, where `UB_s` is the one-sided Clopper-Pearson upper bound
on the A3 proportion at confidence `1 - alpha/m`. The kill fires iff **no**
3-subset `T` of `E_select` has `U_s >= 12` for all `s` in `T` and
`sum U_s >= 72`.

`N_s` comes from the 20 sealed screening censuses and sums to 597; `n_s` from
`audit_sample_B8` and sums to 73. Both are artifact-traceable.

### With `E_select` = the 10 sequences A1 leaves eligible: DECIDABLE

Only **4 of the 10** have `N_s >= 12` and can therefore ever reach the
per-sequence floor — `pour_tea` 73, `tambourine` 18, `put_candy` 18, `tea`
13. Enumerating every attainable outcome vector over those four:

| | |
|---|---|
| outcomes where the kill FIRES | **2,580 of 7,000** |
| outcomes where the route survives | 4,420 of 7,000 |
| minimal surviving outcome | `pour_tea >= 2`, `tambourine >= 1`, `put_candy >= 3` A3-positives |

So the stage-1 verdict genuinely depends on what the audit finds. Minimum
A3-positives for each sequence to reach `U_s >= 12`: `pour_tea` 0 of 9,
`tambourine` 1 of 6, `put_candy` 3 of 9, `tea` 6 of 9.

### THE DEGENERACY TRIPWIRE — if `scissor`, `poker` and `pour_tea` are all admitted, stage 1 CANNOT kill

At **zero** observed A3-positives everywhere:

| sequence | `N_s` | `n_s` | `UB_s` | `U_s` |
|---|---:|---:|---:|---:|
| `scissor` | 343 | 15 | 0.2732 | **93.72** |
| `poker` | 109 | 9 | 0.4125 | **44.97** |
| `pour_tea` | 73 | 9 | 0.4125 | **30.12** |

Those three alone satisfy the inequality (each `>= 12`, sum 168.80 `>= 72`),
so **no possible stage-1 data can fire the kill.** Bonferroni is not the
cause — the same holds at an uncorrected 90%.

The mechanism is worth stating because it is counter-intuitive: `U_s` scales
with the candidate count `N_s` while `UB_s` is floored by the sample size
`n_s`, so a **larger** candidate population makes the kill **harder**. To
remove the tripwire the sample would have to grow to roughly 37–39% of each
sequence's own candidate windows:

| sequence | `N_s` | frozen `n_s` | `n_s` needed |
|---|---:|---:|---:|
| `scissor` | 343 | 15 | **132** (38.5%) |
| `poker` | 109 | 9 | 41 (37.6%) |
| `pour_tea` | 73 | 9 | 27 (37.0%) |

Admitting `scissor` therefore converts stage 1 from a cheap screen into a
near-census of that sequence — a factor of ~9 more windows than the frozen
sample allocates it. **Any revision that admits `scissor` or `poker` to
`E_select` MUST re-specify stage 1 before execution.** That is now a
preregistered tripwire rather than a discovery waiting to happen.

### The uncomfortable coupling, stated plainly

A1's coverage classification — which demoted `scissor` and `poker` to
`indeterminate` on a sensitivity reading that sits 0.003 and 0.015 below a
threshold — is the reason stage 1 is decidable at all. Nobody designed that
dependency, and it means the audit's usefulness currently rests on a
near-knife-edge outcome in a different measurement. It is recorded because it
is true, not because it is comfortable.

## 5. The instrument

**Stage 1.** The frozen 73-window `audit_sample_B8` sample UNCHANGED (seed
20260814, 3 rounds, fresh per-stratum RNG), plus >= 20 presence decoys
(associated at `v == 1` in >= 3 cameras, consensus reprojection RMS <= 2 px)
plus 20 recall-probe windows, all shuffled into one indistinguishable stream
with no class labels or per-class counts disclosed before verdicts return.
Views: the 3 audit cameras × 5 uniformly strided frames, endpoints included.
The auditor's only context is the identity's pre-window reference crop. Two
blinded auditors from **different model families**, per-frame per-camera
verdicts from `{VISIBLE_with_box, NOT_VISIBLE, UNSURE}`.

Validity gates — any failure **VOIDS** stage 1 and licenses no route
inference in either direction:

| gate | threshold |
|---|---|
| decoy false-A3 rate | <= 1 of 20 |
| inter-auditor window agreement (decoys held out) | >= 0.8 |
| human spot-check of 20 `VISIBLE` verdicts | >= 18 confirmed (user, ~30 min) |

**Stage 2**, only if the kill does not fire: all candidate windows of every
`E_select` sequence, through 3-view → `A_S` → return, with **human
confirmation of every `A_S`- and `R_S`-positive**, then the frozen cycle-2
selection inequality on the human-confirmed `R_s`.

## 6. What the design cannot establish, however it comes out

* **Physical presence or absence.** Not once, not for `A_S`, not with human
  confirmation.
* **The instrument's sensitivity.** Presence decoys bound the false-positive
  side only. Nothing here estimates the miss rate on genuine full-view
  absence, enclosure, reveal, or identity-ambiguous events, because no window
  is known to be genuinely absent. This is the design's largest hole and it
  is structural, not an oversight: constructing a known-absent window on
  DiVa-360 is exactly what the synthetic testbed exists for.
* **Anything about sequences outside `E_select`**, including `scissor` and
  `poker`, which hold 452 of the 597 candidates.
* **G-OA's FAIL**, not reopened.

## 7. Sign-off status — NOT OBTAINED

The block required one fresh-context statistical sign-off before this
preregistration is executable. **It was not obtained.** Five attempts to
launch fresh-context reviewer and reducer agents failed immediately on
transient API `529 Overloaded` errors. What exists is this document, its
frozen JSON, and the pre-execution power analysis; what does not exist is an
independent reviewer's verdict.

Consequences: `status` stays `frozen_pending_signoff`; stage 1 is not
executable; and the sign-off must cover, at minimum, the section 3 blocking
finding, the Bonferroni construction, and the section 4 power analysis
including the tripwire.

## 8. Termination

This preregistration is complete when section 3's blocking finding is
resolved in a signed revision, a fresh-context statistical reviewer has
signed, and the user has authorized stage 1. Until all three hold, no
candidate frame may be displayed to anyone.

---

## REVIEW ROUND 1 (2026-08-18) — VERDICT: **BLOCKED**, repaired to revision 2, re-review NOT obtained

A fresh-context adversarial statistical reviewer with no prior project context
read this page and the JSON and returned **BLOCKED**. Recorded append-only.

### What the reviewer independently REPRODUCED

Every number in section 4, recomputed in Python without `scipy` (Clopper-Pearson
verified two ways — the closed form at `k = 0` and bisection on the exact
binomial CDF, agreeing to < 1e-13):

* kill fires in **2,580 of 7,000** attainable outcomes, survives in 4,420 —
  exact match;
* per-sequence minimum A3-positives to reach `U_s >= 12`: pour_tea 0 of 9,
  tambourine 1 of 6, put_candy 3 of 9, tea 6 of 9 — exact match;
* the minimal surviving outcome `{pour_tea 2, tambourine 1, put_candy 3,
  tea 0}` — exact match;
* the tripwire's `U_s` = 93.72 / 44.97 / 30.12 at m = 12, sum 168.805 — exact
  match;
* the "Bonferroni is not the cause" check at an uncorrected 90%:
  48.81 / 24.61 / 16.48, sum 89.89, still unkillable — exact match.

Found **SOUND with no defect**: the Bonferroni construction, including the
question of whether the 3-subset maximisation introduces a multiplicity the
per-sequence correction fails to cover (it does not); the estimand definitions;
and section 3's applicable-camera analysis, with nothing that overturns D3 as
the least-bad choice.

### BLOCKING finding — the human spot-check could never touch a real candidate

Revision 1 specified the spot-check sample only as "20 windows the auditors
labelled VISIBLE", without naming the population. The stream is 73 candidates
plus >= 20 decoys plus 20 recall probes, and **decoys are engineered to be
unambiguous** — so the entire spot-check could be satisfied from decoys,
confirming only that an auditor can recognise an obviously present object.

Why that specific gap matters: it is the **only** gate that can catch the
failure direction producing a **wrongful kill**. An auditor calling a
genuinely-visible-but-occluded candidate NOT VISIBLE deflates A3, deflates
every `U_s`, and pushes the decision toward firing the kill. And that is not a
hypothetical failure mode — it is exactly what
[[elgs-absence-diagnostic-result]] found in the TRACKER instrument this audit
exists to check: 96.6% of the 597 candidates have a multi-view-consistent
foreground component the tracker simply failed to associate.

**Repaired in revision 2:** the sample is drawn ONLY from the 73 real candidate
windows labelled VISIBLE, decoys and recall probes explicitly excluded, and a
shortfall is recorded as a diagnostic fact about auditor behaviour rather than
backfilled with easy decoys.

### MATERIAL findings, all repaired in revision 2

1. **UNSURE verdicts were unchecked by any gate.** An UNSURE anywhere prevents
   A3, so it has the same suppressive effect as a wrong NOT VISIBLE call, and an
   auditor that defaults to UNSURE under uncertainty would deflate A3 invisibly.
   A 10-window UNSURE spot-check is added, **deliberately without a numeric
   threshold** — none can be justified before the instrument has run once — and
   a high rate is a VOID candidate adjudicated by the user, not an automatic
   failure.
2. **An m = 11 / m = 12 inconsistency**, and it was real rather than a rounding
   artifact: section 4's `U_s` values are at m = 12 (correctly — the tripwire
   scenario ADDS scissor and poker to a 10-member `E_select`) while the adjacent
   "required `n_s`" table was computed at m = 11. Corrected to **135 / 42 / 27**
   at m = 12; the superseded m = 11 values (132 / 41 / 27) are recorded in the
   JSON rather than deleted.
3. **No finite-population correction.** Clopper-Pearson assumes i.i.d. binomial
   sampling, but stage 1 samples WITHOUT replacement at fractions up to 100%
   (`put_fruit` 4/4, `slice_apple` 4/4, `writing_2` 2/2, `maracas` 1/1, `soda`
   1/1) and 69% (`tea` 9/13). The direction is now disclosed: an FPC would
   TIGHTEN `UB_s` and make the kill EASIER, so omitting it is conservative with
   respect to killing and anti-conservative with respect to passing. Applying
   one needs a signed revision.
4. **The amendment policy's protected list omitted** the frozen sample and its
   stratification, the decoy construction, the recall-probe construction, and
   the spot-check protocol — all now protected.

### Status after the repair

`revision: 2`, `status: frozen_pending_signoff`, **unchanged**. The directive's
rule is repair once and resubmit; **the re-review was NOT obtained** because the
block ended first. So this preregistration now carries one reviewer's BLOCKED
verdict plus repairs that reviewer has not seen, which is weaker than a
sign-off and must not be read as one.

**Two things still gate execution**: the section 3 applicable-camera-set choice
(a scientific-semantic decision for the user or a reviewer) and a second review
round on revision 2.

---

## REVISION 3 (2026-08-18) — the sequence universe is reconciled and the camera set is DECIDED

Append-only. Nothing above is rewritten. Revision 3 resolves the two items that
gated execution after round 1: the section 3 applicable-camera-set choice, and
an apparent three-way conflict in how the sequence universe was being counted.
Machine-readable text: `configs/elgs/prereg_m1_a0b_audit_v1.json` at
`revision: 3`. Status remains `frozen_pending_signoff` until round 2 returns.

### R3.1 The sequence-universe reconciliation — there was no contradiction

Three different sets were being named by three different numbers. Reconciled
once, exactly, before any audit outcome is viewed.

| sequence | `N_s` | `n_s` | A1 reading (i) | A1 class | basis | `E_eligible` | `E_indet` | `E_select` | exclusion reason |
|---|---:|---:|---:|---|---|:-:|:-:|:-:|---|
| `music_box` | 0 | 0 | 1.000 | eligible | monotonicity | ✓ | | | zero candidates — not in the audit population |
| `piano` | 0 | 0 | 0.998 | eligible | monotonicity | ✓ | | | zero candidates |
| `chess` | 0 | 0 | 0.936 | eligible | monotonicity | ✓ | | | zero candidates |
| `writing_2` | 2 | 2 | 0.924 | eligible | monotonicity | ✓ | | ✓ | — |
| `writing_1` | 0 | 0 | 0.918 | eligible | monotonicity | ✓ | | | zero candidates |
| `maracas` | 1 | 1 | 0.917 | eligible | monotonicity | ✓ | | ✓ | — |
| `jenga` | 0 | 0 | 0.896 | eligible | monotonicity | ✓ | | | zero candidates |
| `put_fruit` | 4 | 4 | 0.887 | eligible | monotonicity | ✓ | | ✓ | — |
| `pan` | 11 | 4 | 0.853 | eligible | monotonicity | ✓ | | ✓ | — |
| `keyboard_mouse` | 0 | 0 | 0.853 | eligible | monotonicity | ✓ | | | zero candidates |
| `tambourine` | 18 | 6 | 0.815 | eligible | monotonicity | ✓ | | ✓ | — |
| `xylophone` | 0 | 0 | 0.779 | eligible | monotonicity | ✓ | | | zero candidates (corrected conversion; the defective-era row read 0.577) |
| `kindle` | 0 | 0 | 0.779 | eligible | monotonicity | ✓ | | | zero candidates |
| `soda` | 1 | 1 | 0.774 | eligible | monotonicity | ✓ | | ✓ | — |
| `tea` | 13 | 9 | 0.743 | eligible | monotonicity | ✓ | | ✓ | — |
| `slice_apple` | 4 | 4 | 0.731 | eligible | monotonicity | ✓ | | ✓ | — |
| `pour_tea` | 73 | 9 | 0.591 | eligible | **measured** | ✓ | | ✓ | — |
| `put_candy` | 18 | 9 | 0.507 | eligible | **measured** | ✓ | | ✓ | — |
| `scissor` | 343 | 15 | 0.441 | **indeterminate** | **measured** | | ✓ | | transposed-anchor sensitivity 0.497 crosses the floor; the frozen `indeterminate_rule` bars it from a winning 3-subset |
| `poker` | 109 | 9 | 0.382 | **indeterminate** | **measured** | | ✓ | | transposed-anchor sensitivity 0.485 crosses the floor; same rule |

**The identities, all verified:**

* 20 = 4 measured + 16 by monotonicity;
* 20 = 18 `eligible` + 2 `indeterminate` + 0 `ineligible`;
* 20 = 12 with `N_s > 0` + **8 with `N_s == 0`**;
* **10 = |`E_select`| = |{eligible} ∩ {`N_s > 0`}| = 12 − 2**;
* Σ`N_s` over all 20 = **597** exactly; Σ`n_s` = **73** exactly;
* Σ`N_s` over `E_select` = 145, Σ`n_s` over `E_select` = **49**;
* Σ`N_s` over `E_indeterminate` = **452** = 75.7%.

**Where each disputed number came from:**

* *"16 more sequences are eligible"* counts A1's **monotonicity rows**, not
  audit-population members. A1 computed readings (ii)/(iii) for four sequences
  and proved the other sixteen eligible without computing them. 4 + 16 = 20.
* *`E_select` = 10* is `E_eligible` **restricted to `N_s > 0`**. Eight eligible
  sequences carry zero true-absence candidates and are absent from the audit
  population entirely. That restriction, not any disagreement, is the whole gap
  between 18 and 10.
* *m = 11 vs m = 12* — **neither is this audit's divisor.** The audit's divisor
  is **m = 10**. The 12 appears only inside the hypothetical degeneracy
  tripwire, where `scissor` and `poker` are *added* to the ten. The 11 was a
  revision-1 arithmetic error inside that same tripwire table, corrected in
  revision 2. No construction in this design uses m = 11.

**Frozen here:** `E_select` is exactly the ten sequences listed above, and the
simultaneous-bound divisor is **m = 10**. `scissor` and `poker` remain
`indeterminate` and excluded for this audit; **their status may not change
after power or audit results are seen, under any circumstance.**

### R3.2 The applicable-camera set — D3 adopted, and what that costs

**The user has adopted D3**, the sealed census per-candidate frustum rule that
reproduced `containing_cameras` on 73 of 73 frozen windows. Section 3's
blocking finding is resolved by decision, not by discovery: D2 stays refuted
and D1 stays destructive.

`S_w` = the tracking cameras whose frustum contains the frozen candidate
anchor, under the frozen `prereg_m1_census_v1` predicate — positive
camera-frame depth and a pixel inside `[0, W−1] × [0, H−1]`. Its **only**
inputs are the sealed anchor (`ltp` / `ltp_frame`), the frozen calibration, and
the existing deterministic frustum computation. Auditor verdicts, post-result
visibility, tracker visibility flags and mask occupancy are all prohibited
inputs, and the candidate generator is not modified.

**The permitted estimand, verbatim and binding:**

> unobservability across the cameras that the frozen candidate generator
> geometrically considered applicable, followed by same-identity reappearance.

**This does not establish** literal physical absence; unobservability across an
independently fixed rig-wide camera set; or candidate-generator-independent
event supply. Two phrasings are added to the prohibited list accordingly.

**The audit triple, frozen before it is computed.** The three audit cameras of
a window are the 3-subset of `S_w` maximising the **minimum pairwise
optical-axis angular separation**, ties broken by the lexicographically
smallest sorted camera-id tuple. Optical axis of camera `c` is
`w2c[:3,:3].T @ [0,0,1]` — the census's own construction. The order is total:
(min separation DESC, id-tuple ASC). No RNG, no free constant, no unresolved
tie. A window with `|S_w| < 3` is excluded from **both** the A3 numerator and
its denominator and is named in the report.

The mapping is computed once, written to a run artifact, and its sha256
recorded in the JSON **before any auditor sees any frame**; thereafter it is
immutable.

### R3.3 Power, recomputed at the reconciled m = 10 — and unchanged

Recomputed independently by the primary after the reconciliation and the D3
adoption, with Clopper-Pearson evaluated two ways (closed form
`1 − (α/m)^(1/n)` at `k = 0`, and bisection on the exact binomial CDF; they
agree to `< 1e-12`) and without `scipy`. **Every published figure reproduces
exactly:** 2,580 of 7,000; minimum A3-positives 0/1/3/6; minimal surviving
outcome `{pour_tea 2, tambourine 1, put_candy 3, tea 0}`; tripwire
93.72 / 44.97 / 30.12 summing to 168.805 at m = 12; the uncorrected-90% check
48.81 / 24.61 / 16.48 summing to 89.894; required `n_s` 135 / 42 / 27 at m = 12
with the superseded 132 / 41 / 27 at m = 11.

**The camera rule does not move the power arithmetic.** D3 leaves 100% of the
73 windows with at least three applicable cameras, so no window is excluded on
geometric admissibility and every `N_s` and `n_s` is unchanged. Under D1 the
`E_select` sample would have fallen from 49 windows to 14 and the analysis
would have had to be redone — which is the concrete form of "destructive".

One structural fact worth stating: **`pour_tea` reaches `U_s` = 29.24 ≥ 12 at
zero observed A3-positives**, so it occupies a slot in a surviving 3-subset
unconditionally. The stage-1 decision therefore turns entirely on `tambourine`,
`put_candy` and `tea`.

### R3.4 Decidability is CONDITIONAL — state this wherever stage 1 is described

**Stage 1 is not generally decisive.** Its kill power exists *only* because
`scissor` and `poker` are excluded from `E_select`. With `scissor`, `poker` and
`pour_tea` all admitted, the kill cannot fire under any outcome. Stage 1's
decidability is therefore conditional on A1's coverage classification, which
demoted those two sequences on a frozen sensitivity reading sitting 0.003 and
0.015 below a threshold. **Any statement that stage 1 "can kill the route"
without that clause is a misdescription of this design.**

### R3.5 What revision 3 preserves unchanged

Simultaneous one-sided coverage across the frozen `E_select`; the complete
three-sequence kill inequality; the per-sequence floor of 12 and the pooled
floor of 72; candidate-set-relative conclusions; the separate eligible /
indeterminate / selectable sets; the diagnostic-only recall probe; the
model-auditor vs human-confirmed evidence grades; the corrected human
spot-check population (the 73 real candidates only, decoys and probes
excluded); the explicit `UNSURE` handling; the finite-population disclosure;
and the frozen amendment policy.

### R3.6 Round-2 status

Revision 3 is submitted for **one** fresh-context statistical and semantic
re-review. A PASS requires explicit confirmation of the audit population and
`E_select`, the simultaneous-bound divisor, the candidate-specific camera
mapping, the narrowed estimand, the human spot-check population, the power
table, and the outcome-to-action rules. A second substantive rejection stops
audit execution and returns the issue to the user.

---

## REVIEW ROUND 2 (2026-08-18) — VERDICT: **BLOCKED**. Substantive, so audit execution STOPS and returns to the user

Append-only. A fresh-context adversarial statistical and semantic reviewer with
no prior project context read revision 3 and its JSON and returned **BLOCKED**
with four blocking findings. Revision 4 repairs **mechanical and wording
defects only**. `status` is now `BLOCKED_AT_ROUND_2_RETURNED_TO_USER`.

**The governing rule is the directive's own:** a substantive second rejection
stops audit execution and returns the issue to the user. The user's
authorization of stage 1 was conditional on a *clean* PASS. It did not arrive.
**Stage 1 was not executed, no panel was built, and no candidate frame was
rendered, displayed or transmitted to anyone.**

### What the reviewer reproduced exactly

Every published figure, with Clopper-Pearson implemented twice (closed form and
bisection, agreeing to 1.11e-16) and no `scipy`: 2,580 of 7,000; minimum
A3-positives 0/1/3/6; the tripwire 93.72 / 44.97 / 30.12 summing to 168.805;
the uncorrected-90% check summing to 89.894; required `n_s` 135 / 42 / 27 at
m = 12 and 132 / 41 / 27 at m = 11; and every reconciliation identity —
including the **set** identity `{N_s > 0} \ E_indeterminate == E_select`, not
merely its cardinality. It also went beyond the document on the tripwire,
enumerating **all 1,120,000** attainable vectors in the 12-member set and
finding the kill fires in **zero** — stronger than the `k = 0` corner claimed.

It independently verified in git that the scissor/poker exclusion is **not
post-hoc**: freeze commit `dad3360` carries the sensitivity rule verbatim and
contains no coverage figure for either sequence, and the A1 result commits are
strict descendants.

### The four BLOCKING findings

**B1 — the decision-bearing bound assumes a sampling design the frozen sample
does not have.** Per-sequence Clopper-Pearson on `(k_s, n_s)` needs equal
within-sequence inclusion probabilities. The frozen B8 sample is a
**round-robin** over (sequence × terminal class × pooled staleness tercile)
strata, 3 rounds, without replacement — so stratum `h` contributes
`min(3, size_h)`, and equal inclusion requires equal stratum sizes, i.e.
`3·N_s/n_s` integral.

**Primary-verified.** It is not integral for **scissor 68.600, poker 36.333,
pour_tea 24.333, tea 4.333, pan 8.250** — and `pour_tea` and `tea` are two of
the four decisive sequences. So `k_s/n_s` is not design-unbiased for `p_s`,
`U_s = N_s·UB_s` is not a `1 − α/m` upper confidence bound on the population
count, and the "at least 90% simultaneous coverage" claim is unsupported. The
design-consistent Horvitz-Thompson estimator is already in the file and is
deliberately given **no** decision weight, while the decision rests on the
design-inconsistent one with no argument that it is conservative. Repairing it
either proves per-sequence CP conservative under the actual design, or moves to
per-stratum bounds with a larger Bonferroni divisor — **which moves the power
analysis.**

**B2 — a known, quantified, kill-direction bias is absent from this
preregistration.** The same frozen B8 block discloses that audit frames are
drawn uniformly over the census window, which *includes bridged frames at which
the identity is associated by construction of the flicker rule*. A bridged
window therefore **can never be A3-positive whatever the scene contains** (85
of 109 `poker` windows carry at least one). Deflated A3 → deflated `U_s` →
**kill easier** — the wrongful-kill direction the entire round-1 spot-check
repair exists to guard, and one the spot-check cannot catch because it is
instrument geometry rather than auditor behaviour. **Primary-verified by grep:
`bridg` occurred zero times in the JSON and zero times in this page**, and B8's
mandate to report `bridged_sampled_frame` was not carried forward.

**B3 — `A_S` was defined over a symbol the adopted rule does not define.**
Three binding operational lines said `S_seq`, which only the **rejected** D1 and
D2 define; D3 constructs `S_w` per candidate. **Mechanically repaired in
revision 4.** The residue is not: "the last `v == 0` frame" is undefined for a
camera that never queried the identity (the documented 12.2% never-queried
limb) and for one in which `v` is never 0 — and it is driven by the very
tracker flag M-2 showed carries no confidence information.

**B4 — the recall probe has no construction rule, and it sits inside a validity
gate.** `"20 windows, DIAGNOSTIC ONLY"` is the complete specification; no
construction rule exists in the JSON, this page, or the census prereg. Yet 20 of
the 93 stream windows — **21.5% of the ≥ 0.8 inter-auditor agreement gate's
denominator** — are of undefined provenance and difficulty, and the amendment
policy protects a rule that does not exist. This is structurally the same defect
round 1 blocked on, repaired for the spot-check and left standing here. **The
preregistration is not executable as written: the stream cannot be built.**

### The material finding that moves a published number

**M3 — the finite-population omission is not a rounding matter.**
Primary-verified: `U_s` routinely exceeds the logically attainable maximum
`k + (N_s − n_s)` — for `tea` at **every** `k` (5.207 against a hard maximum of
4 at `k = 0`), for `tambourine` and `put_candy` from `k = 2`, for `pour_tea`
from `k = 7`. Capping `U_s` at that maximum moves the kill from **2,580 to
3,141 of 7,000 — a 21.7% shift in the decision region.** The disclosed
*direction* is correct and the omission is conservative with respect to killing;
the *magnitude* was undisclosed while the power table was presented as exact.
And `tea`'s published "6 of 9" threshold is **unattainable in fact**: 6 of 9
drawn from 13 permits at most 10 true positives, below the floor of 12.

Further material findings — D3's undisclosed statistical direction (M4), the
undisclosed supersession of another frozen prereg's still-binding camera and
estimator rules (M5), the missing comparison tolerance and separation floor in
the triple rule (M7), the 16 monotonicity rows never subjected to the
sensitivity test that demoted scissor and poker (M8, and 8 of the 10 `E_select`
members are monotonicity rows), the decoy count that is not a count (M11), and
stage 2's absent feasibility statement (M12) — are recorded in full in the JSON
`review_history` round-2 entry.

### Repaired in revision 4 — mechanical and wording only

`S_seq` → `S_w` in the three binding operational lines (B3); the bound's
misnamed parameter, which said "the proportion among the sequence's audited
windows" when `U_s` requires a **population** proportion (M9); the provenance
pointer that resolved to a mutable script rather than frozen text (M6); the
conditional-decidability clause added to `stage_1.purpose`,
`kill_rule.fires_iff` and `kill_rule.on_fire`, which the document had instructed
and then not done (M10); and two false claims withdrawn — "**the** minimal
surviving outcome", which is one of **six** minimal surviving vectors and is not
necessary (M1), and "the decision turns entirely on tambourine, put_candy and
tea", which is false because `pour_tea`'s own count enters the ≥ 72 sum: (2,1,3,0)
survives at 74.113 while (1,1,3,0) fires the kill at 65.916 (M2).

### What is NOT repaired, and why

B1, B2, B4 and M3–M5, M7, M8, M11, M12 are **substantive**. B1 would move the
power analysis; B2 needs a disclosure and possibly an exclusion rule; B4 needs a
construction rule that does not exist. None is inside the frozen semantics, so
none may be improvised here.

### The reviewer's own judgement, recorded because it bears on what to do next

None of the four blocking findings is an arithmetic error, and none is evidence
of motivated reasoning. **B1 and B2 are defects of inheritance** — this
preregistration inherits a sample and an instrument from another frozen
document, carries the sample's identity forward, and drops the design facts and
the disclosed bias that came with it. B3 and B4 are unfinished text. All four
are repairable without restructuring the power analysis, though B1's repair will
move its numbers.

### Status

**BLOCKED at round 2. Stage 1 is not executed and is not executable.** The
camera mapping remains unbuilt pending the decision below, and building it
changes nothing about this verdict — the mapping was never the binding
constraint.

### M8's residual risk, BOUNDED by the primary from the existing record

M8 — that the 16 monotonicity rows were never subjected to the sensitivity rule
that demoted `scissor` and `poker`, and that **8 of the 10 `E_select` members
are monotonicity rows** — stands as a documentation gap. Its residual risk is
now bounded, from numbers already on the A1 result page:

* on all **four** measured sequences **both** sensitivity readings sit **above**
  reading (i), not below — transposed 0.4852 / 0.4972 / 0.5321 / 0.6158 against
  reading (i) 0.3824 / 0.4409 / 0.5067 / 0.5908, and last-defined 0.8250 /
  0.8717 / 0.7696 / 0.7403;
* the weakest `E_select` monotonicity row is `slice_apple` at reading (i)
  **0.731 — a margin of 0.231 above the floor**. A sensitivity reading would
  have to fall 0.231 *below* reading (i) to demote it, which is the opposite
  direction to every measurement taken;
* `scissor` and `poker` were demoted from reading (i) = 0.441 and 0.382, i.e.
  **from below the floor**. No `E_select` monotonicity row is below the floor on
  reading (i) at all, so the two situations are not analogous.

Computing the two sensitivities for the 16 monotonicity rows is cheap and would
close this outright. **This bounds the finding; it does not discharge it.**

---

## THE SEALED CAMERA MAPPING (experiment 163) — computed, hashed, and it corrects one of this page's own numbers

Determined experiment **163**, cell `a0b_camera_mapping` r0, commit `c4ff0d4`,
admitted image, pool `dgx`, `evidence_bearing: false`, COMPLETED. Artifact
`runs/elgs/a0b_camera_mapping_r0/camera_mapping.json`, 43,478 bytes, sha256
`aa6d93d200aa1a3ce74945427b903ea99019fda5ee4413a532511c07b0e506ad`.

**The load-bearing contract passed exactly: the recomputed `S_w` reproduces the
sealed `audit_sample_B8` `containing_cameras` on 73 of 73 windows.** `void`
is false. This was previously an unverified claim — the round-2 reviewer listed
it among the things it had to take on the document's word. It is now measured.

| quantity | measured |
|---|---|
| `\|S_w\|` min / q1 / median / q3 / max | **3 / 20 / 23 / 26 / 26** |
| windows with `\|S_w\| < 3` | **0** — none excluded on geometric admissibility |
| windows where a tie was broken | **0** |
| data-quality flag | `soda_screen_w0_171` — anchor at **1.955 × rig_radius**, `\|S_w\| = 3` exactly, best triple separation **28.7°** |

**A correction to revision 3's own D3 measurement.** Revision 3 recorded that
the best-separated triple has minimum axis separation "104–116 degrees for
every sequence except `soda`". **Measured, `slice_apple`'s worst window is
98.5° — below the stated band** (and `writing_2` is 104.8°). The correct range
excluding `soda` is **98.5–116.2°**. Nothing depends on it, because
admissibility tests only `|S_w| ≥ 3` — which is precisely review finding M7's
complaint that no separation floor exists. Recorded as a correction rather than
quietly restated.

**This bounds review finding M7.** M7 warned that the tie set is formed with
exact float equality, so a 1e-16 `arccos` difference could decide the triple
instead of the frozen id-tuple rule. **Measured: zero of the 73 windows had
more than one triple at the maximum**, so the tie-break never fires on this
data and the concern is empirically inert here. It remains a genuine defect of
the frozen text for any future candidate set and is **not discharged**.

**The mapping changes nothing about the round-2 verdict.** It was never the
binding constraint — B1, B2 and B4 are — and stage 1 remains blocked.
