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
