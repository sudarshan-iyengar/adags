# EL-GS Cycle-2 Continuation Design (post-M1-negative)

Status: REFINED — the hostile fresh-context review returned SOUND
WITH REPAIRS (blocking findings {1,2,3,5,6,7,8,9,10,14} + notes
{4,11,12,13,15}; full record in the session artifacts, verdict
quoted below). All blocking repairs are applied in this revision;
the notes are absorbed here and in the experiment plan. The
subsequent prereg SIGN-OFF review (2026-08-12) added four further
one-clause repairs (B1–B4: R2′ z>0 participation; nearest-rank
angular percentile; exactly-3 selection cardinality; tranche-2 /
continuation semantics) — embodied in prereg revision 2, which is
the operative frozen text wherever this page and the prereg differ.
N8 prose correction: tol_c = r_site·fl_x/z_c is the exact image of
the 3-D ball only under isotropic undistorted intrinsics; the
operational pin to fl_x is the frozen deterministic rule. Authorized by the user
2026-08-12: "form a full plan and implement this given the M1
preserved negative."

Review verdict (verbatim): "SOUND WITH REPAIRS. The route choice
(R1+R2 with R4 contingency and R3 as user decision), the reuse of
the validated census stack, the screen-then-select ordering with
pre-download freezing, and the preservation of the cycle-1 negative
are all correct and policy-conformant. […] No finding invalidates
the continuation direction itself; every blocking defect has a
bounded, frozen-text repair."

Inputs: [[operations/elgs-m1-census-result]],
[[operations/elgs-m1-census-record]], [[gap_map]] (G13/G14),
`prereg_m1_census_v1.json` revision 3, the independent recomputation
record (twelve named definitional readings; return-distance
bimodality).

## 1. What the negative does and does not say

DOES say (confirmed by two independent computations):
- The three dev sequences richly supply occlusion opportunity
  (93,841), genuine multi-view absence (600 candidates), and tracked
  foreground coverage (0.72) — the evidence classes feeding BIRTH,
  FISSION, TRUNCATE, and the censored-evidence ontology (G13).
- They under-supply SAME-OBJECT RETURNS (23–30 pooled vs floor 36):
  flip_book 0, battery ≤7, unlock 17–23. REACTIVATE — the mechanism
  occupying G14's reactivation-with-own-content cell — cannot be
  powered on this dev subset.
- A measurement limitation is characterized: 70–109 candidates ended
  in re-appearance runs with NO defined multi-camera consensus
  (single-camera tracker re-acquisition), counted as never-returns
  by the frozen anchor. Outcome-relevant, frozen, binding for
  cycle 1.

Does NOT say: nothing about EL-GS's method, math, or implementation
was refuted; nothing about DiVa-360 beyond the three measured
sequences.

## 2. Route enumeration (ideation record)

R1. SCREENED RE-SELECTION within DiVa-360 (cycle 2). Selected.
R2. MEASUREMENT REPAIR of the return statistic. Selected, in the
    strengthened R2' form of §4.3 (the single-camera form was ruled
    not measurement-valid by the refine round and is diagnostic-only).
R3. DESCOPE REACTIVATION (claim change): user decision only;
    surfaced at the frozen checkpoint (§4.6) or on DRY.
R4. DATASET EXTENSION (Ego-Exo4D/HOT3D): preregistered contingency,
    triggered by DRY or at the checkpoint.
R5. ABANDON: rejected — no refutation exists.

## 3. Candidate pool

54 official sequences minus the 5 assigned (cycle-1 dev battery,
flip_book, unlock; calibration peel_apple, pour_salt) = 49
candidates (41 short + 8 long; names in the prereg). The
name-prior ordering (chess/jenga/put_*/poker/writing/legos/puzzle
first) affects only QUEUE COMPOSITION under the frozen full-tranche
halt rule (§4.5) — never selection within the screened pool; the
residual pool-composition effect is disclosed.

## 4. Cycle-2 integrity design (all refine repairs applied)

### 4.1 Gate on the UNSCREENED half (repair of finding 1)
Screening computes statistics on the FIRST HALF of each candidate's
common frame range. The cycle-2 gate binds on the SECOND
(unscreened) half ONLY, floors 36/36/36/0.5 pooled, with frozen
boundary conventions: a gate-window return is one whose terminating
re-appearance run STARTS in the second half; LTP lookup may reach
into the first half (full-range tracks exist at gate time);
occlusion runs and true-absence windows are attributed to the half
containing their FIRST frame. What the gate tests, stated plainly:
winner's-curse-robust within-sequence generalization of event
abundance from the screened half to the unscreened half. Full-range
counts are additionally reported (never gated) for B2/B3 power
accounting.

### 4.2 Screening thresholds (repair of finding 2)
Screened-half pooled thresholds: >= 72 on ALL THREE event statistics
(2x floor each) AND coverage >= 0.5. The 2x is a DISCLOSED DESIGN
MARGIN, not a derivation (note 11): under half-to-half stationarity,
screened 72 gives P(unscreened >= 36) ≈ 0.99999 (Poisson), 0.97 at
true per-half rate 48, 0.52 at 36 — it buffers winner's-curse
shrinkage and mild nonstationarity and cannot protect against a
task whose event phase lives entirely in one half (that is exactly
what the gate exists to catch).

### 4.3 Return statistic (repair of finding 3; note 4; note 15iii)
- PRIMARY (unchanged cycle-1 rule): multi-camera consensus return
  within r_site.
- R2' (strengthened repair): a return ALSO counts if >= 2 TRAINING
  cameras (held-out mod-4 excluded), with pairwise angular
  separation above a frozen calibration-derived floor, each have
  the identity's report pixel within a per-camera tolerance equal
  to the projected radius of the r_site ball at the LTP
  (tol_c = r_site * f_c / z_c — the exact image of the same 3D
  predicate; calibration + frozen tracks only, no free pixel
  constant).
- GATE-BEARING statistic = primary OR R2' as ONE frozen union
  predicate. The single-camera >= 1 form is DIAGNOSTIC-ONLY. The
  SELECTION statistic is IDENTICAL to the gate-bearing statistic.
- Standing disclosure on every cycle-2 result page (note 4): the
  union predicate is a post-outcome one-sided loosening motivated
  by the characterized censoring gap; the strict cycle-1 primary is
  always reported alongside so readers see how much any pass owes
  to the loosening.

### 4.4 Selection rule as a total function (repairs of findings 5, 6)
Frozen deterministic algorithm over the FULLY SCREENED pool:
1. Eligibility: per-sequence screened-half union-returns >= 12
   (= floor/3; rationale: no dev sequence may free-ride — cycle 1's
   flip_book contributed zero returns and zero absences and pooled
   floors hid it) AND per-sequence screened-half coverage >= 0.5
   (the coverage floor's own rationale is per-scene: evidence-driven
   selection cannot reach most content in an under-covered scene).
2. Order eligible candidates by screened-half union-return count
   descending; ties by ascending alphabetical name.
3. Take the top 3; test pooled thresholds (§4.2; coverage pooled as
   ratio-of-sums per revision-3 pooling).
4. On failure, advance through 3-subsets of eligible candidates in
   descending order of pooled union-returns, ties lexicographic by
   ordered member names; first subset passing all pooled thresholds
   is selected.
5. DRY: the enumeration exhausts without a passing subset. DRY is
   always relative to the screened tranche and the budget ("dry
   within the frozen budget"); the record never claims "DiVa-360 is
   exhausted" while unscreened candidates remain.

### 4.5 Budget, halt rule, checkpoint (repairs of findings 8, 13)
- Per-candidate screening cost is DURATION-SCALED: est. GPU-h ≈
  0.06 + 0.10 * (n_frames / 1000) (from measured runs: 311 frames
  0.02 GPU-h; 1,519 frames ~0.2 chunked), halved by half-window
  tracking. Long sequences use the disclosed chunked path.
- TRANCHE 1 = the first 20 queue candidates. The tranche is screened
  IN FULL regardless of interim qualification (queue order can
  therefore never select within the pool). Ceiling: 6 GPU-h
  screening + 1 GPU-h gate; storage cap 400 GB extracted at any
  time (deletion per §4.7).
- MANDATORY CHECKPOINT after tranche 1, with frozen decision
  content (note 13): the full interim screening table, projected
  cost to continue, and the explicit options — continue to tranche
  2, R4 (dataset extension), or R3 (descope, a claim change) — is
  put to the USER. Only "continue" is autonomous-eligible, and only
  if a qualifying subset already exists or >= 2 candidates cleared
  per-sequence eligibility (evidence the pool is live).
- A cycle-2 gate run happens only after selection from a fully
  screened tranche.

### 4.6 Frozen definitional readings (repair of finding 7)
The cycle-2 prereg enumerates and freezes ALL TWELVE residual
readings documented by the cycle-1 independent recomputation
(re-appearance camera set = the containing set S; disappearance is a
TRANSITION: associated in >= 1 camera of S at t−1 and none at t, so
no window opens at frame 0; P and S fixed at event start; occlusion
per-frame position must be defined, undefined breaks the run; one
event per maximal run per (identity, fixed camera); duration floor
on window SPAN incl. bridged flicker; end-truncated re-appearance
runs < 4 are non-terminating; frustum containment on continuous
coordinates, inclusive bounds, z > 0; in-domain visible report =
non-miss, v >= 0.5, round-half-up pixel in bounds, identical for
association and coverage; position_undefined tallied at global
disappearance transitions with no prior consensus; consensus defined
iff point non-null, no n_cam threshold; return distance Euclidean
3-D, inclusive <= r_site; |S| >= 2 suffices under the verified
static rig). The census implementation is ALIGNED to these frozen
readings (with tests) before any screening statistic is computed;
the reviewer sign-off covers them explicitly.

### 4.7 Verifiability seal set (repair of finding 9)
Retained indefinitely per screened candidate: the half-window tracks
artifact (+ controls), the screening census JSON, all manifests, and
the realized-window record — everything the independent-verification
protocol needs to recompute the RANKING without GPU re-runs. Only
extracted frame images may be deleted after the seal is verified;
raw zips are always retained; every deletion is ledgered.

### 4.8 Selection-bias disclosure (repair of finding 10; note 15)
Standing disclosure on every cycle-2 page: selection maximizes
returns AS MEASURED BY THE TRACKER-BASED CENSUS and therefore
selects tracker-legible scenes; EL-GS's evidence machinery consumes
exactly those tracks while ported baselines consume none, so arm
comparisons on the selected subset estimate EL-GS's advantage in
its cleanest-evidence, maximal-opportunity regime. All downstream
claims are scoped to "event-rich, tracker-legible sequences"
(consistent with the conditional 8.0 novelty rating). The FULL
screening table (every screened candidate's statistics) is
published with the selection. Within-scene shuffle/shift controls
address evidence informativeness, not scene-selection external
validity. The revision-3 necessary-condition note (upper bounds;
floors are necessary, never sufficient) is restated on every
cycle-2 result page. M1-A0b (audited true absence) runs post-gate
on the selected subset, diagnostic-only as frozen.

### 4.9 Authority chain (repair of finding 14)
Cycle 1's negative is FINAL and never overwritten. Authority for
cycle 2: the user's 2026-08-12 directive -> this refined design ->
the cycle-2 screening prereg (fresh-context signed) -> the cycle-2
gate. A cycle-2 gate PASS restores the scientific precondition that
the M1 gate tested (adequate event supply on a dev subset) —
and starting M2 REMAINS a separate explicit user approval, exactly
as the original plan required after an M1 pass. A cycle-2 valid
FAIL on the selected subset is final for that subset under the same
frozen failure policy; remaining options return to the user.

### 4.10 Concordance diagnostic (note 12)
Screening tracks (half-window) and gate tracks (full-window) are
different artifacts. At gate time, the screened-half statistics are
recomputed from the FULL-range tracks of the selected sequences
(CPU-only) and the deltas reported next to the gate.

## 5. What follows

(a) `prereg_m1_cycle2_screen_v1.json` embodying §4 verbatim,
fresh-context signed BEFORE any candidate download; (b) the cycle-2
experiment plan page; (c) evaluator alignment (twelve readings +
union predicate + per-sequence eligibility + half-window gate
attribution) with tests; (d) tranche-1 screening; (e) selection;
(f) checkpoint or gate per §4.5.
