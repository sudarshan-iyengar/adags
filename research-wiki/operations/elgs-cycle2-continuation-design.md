# EL-GS Cycle-2 Continuation Design (post-M1-negative)

Status: DRAFT under adversarial refine (this page is the ideation +
route-selection record; the experiment plan follows after the refine
round passes). Authorized by the user 2026-08-12: "form a full plan
and implement this given the M1 preserved negative."

Inputs: [[operations/elgs-m1-census-result]] (the preserved
negative), [[operations/elgs-m1-census-record]] (execution trail),
[[gap_map]] (G13/G14 boundaries), `prereg_m1_census_v1.json`
revision 3 (the signed cycle-1 gate), the independent recomputation
record (12 named definitional readings; return-distance bimodality).

## 1. What the negative does and does not say

DOES say (confirmed by two independent computations):
- The three dev sequences richly supply occlusion opportunity
  (93,841), genuine multi-view absence (600 candidates), and tracked
  foreground coverage (0.72) — the evidence classes feeding BIRTH,
  FISSION, TRUNCATE, and the censored-evidence ontology (G13).
- They under-supply SAME-OBJECT RETURNS (23–30 pooled vs floor 36):
  flip_book 0, battery ≤7, unlock 17–23. REACTIVATE — the mechanism
  occupying G14's reactivation-with-own-content cell — cannot be
  powered on this dev subset. The B1 census did exactly the cheap
  kill it was designed for.
- A measurement limitation is now characterized: 70–109 candidates
  ended in re-appearance runs with NO defined multi-camera consensus
  (single-camera tracker re-acquisition after occlusion), which the
  frozen return anchor counts as never-returns. Outcome-relevant
  (~6 resolved returns would have flipped the floor), frozen, and
  binding for cycle 1.

Does NOT say:
- Nothing about EL-GS's method, math, or implementation was refuted;
  the whole M0 stack and the census machinery are validated assets.
- Nothing about DiVa-360 as a dataset: 49 of 54 sequences are
  unmeasured, and cycle 1 measured only 3.

## 2. Route enumeration (ideation record)

R1. SCREENED RE-SELECTION (cycle 2 within DiVa-360). Freeze a
    screening protocol; measure return abundance model-free over
    candidate sequences with the validated census pipeline; select a
    new dev subset by a frozen rule; run a cycle-2 gate with genuine
    content (see §4 temporal split). Preserves every approved claim.
    Cheap (~0.1–0.2 GPU-h + ~5–15 GB per candidate).

R2. MEASUREMENT REPAIR (cycle-2 return statistic). Add a frozen
    SECONDARY return statistic that resolves single-camera
    re-acquisition: a return also counts if, within the terminating
    re-appearance run, >= 1 containing camera's report pixel lies
    within a frozen pixel tolerance of the projection of the
    disappearance position (LTP) into that camera. Still model-free
    (tracks + calibration only); repairs the characterized
    measurement gap rather than the outcome (justification: the
    cycle-1 recomputation documented single-camera-first
    re-acquisition as the dominant undefined-consensus mode). BOTH
    statistics are reported in cycle 2; the gate binds to the
    primary unless the refine round rules the repaired statistic the
    better-grounded primary — either way frozen BEFORE any cycle-2
    screening statistic is computed.

R3. DESCOPE REACTIVATION (drop REACTIVATE/MERGE claims; proceed to
    M2 on occlusion/absence/truncation claims). Scientifically
    coherent given the passing floors, but it ALTERS the approved
    claim set and weakens the G14 novelty cell that the 8.0
    conditional rating priced in. NOT taken autonomously: surfaced
    to the user only if R1+R2 fail (screening finds no adequate
    subset).

R4. DATASET EXTENSION (Ego-Exo4D cooking / HOT3D): put-down/pick-up
    events are abundant in cooking; but a new calibration/converter
    stack is a large lift and DiVa-360 has 49 unmeasured sequences.
    CONTINGENCY, triggered only by an R1 dry result.

R5. ABANDON EL-GS: unjustified — no mechanism, math, or
    implementation refutation exists.

SELECTED: R1 + R2 as Cycle 2; R4 as the preregistered contingency;
R3 as a user-decision fallback. R5 rejected.

## 3. Candidate pool (from the official README, the same sanctioned
source cycle 1 used for calibration selection)

54 sequences total; minus the 5 already-assigned (dev cycle-1:
battery, flip_book, unlock; calibration: peel_apple, pour_salt) =
49 candidates: 45 short (blue_car, bunny, chess, clock, dog, drum,
horse, hour_glass, jenga, k1_double_punch, k1_hand_stand,
k1_push_up, keyboard_mouse, kindle, maracas, music_box, pan,
penguin, piano, plasma_ball, plasma_ball_clip, poker, pour_tea,
put_candy, put_fruit, red_car, scissor, slice_apple, soda, stirling,
tambourine, tea, tornado, trex, truck, wall_e, wolf, world_globe,
writing_1, writing_2, xylophone) + 8 long (chess_long, crochet,
jenga_long, legos, origami, painting, puzzle, rubiks_cube).
Name-level priors (chess/jenga/put_*/poker/writing/legos/puzzle
suggest placement-and-return semantics) are used ONLY to order the
screening queue, never to select — cycle 1's lesson is that names
and paper prose do not predict event content; measurement does.

## 4. Cycle-2 integrity design (the load-bearing part)

The screen-then-select structure must not collapse the gate into a
tautology, and dev selection is now data-informed. Frozen-by-design
answers, for the refine round to attack:

- SCREENING PROTOCOL FROZEN FIRST: candidate order, per-sequence
  budget, the screening statistic set (identical frozen definitions
  as revision 3 + the R2 secondary), the selection rule, and the
  stop rule are all committed BEFORE the first candidate download.
- TEMPORAL SPLIT gives the gate content: screening computes
  statistics on the FIRST HALF of each candidate's frame range
  only. The cycle-2 gate then computes the frozen statistics on the
  FULL range of the SELECTED subset. The gate therefore measures
  generalization of event abundance from the screened half to the
  whole sequence — satisfiable-by-construction is broken.
- SELECTION RULE (frozen): rank candidates by screened-half pooled
  same-object-return count (primary statistic); select the minimal
  set of 3 sequences whose screened-half totals reach >= 2x floor
  (72) on returns AND >= floors on the other three statistics;
  ties by ascending alphabetical order. If no 3-subset reaches 2x
  on returns after the full pool is screened, R1 is DRY -> R4
  contingency + user decision on R3.
- SELECTION-BIAS DISCLOSURE: dev selection maximizes event
  opportunity, which inflates FEASIBILITY statistics by
  construction; it does not touch the downstream B2/B3 ARM
  comparisons (all arms share the selected data; kill rules compare
  arms, not opportunity counts). Recorded as a standing disclosure
  in every cycle-2 result page.
- CYCLE-1 ASSETS UNCHANGED: the cycle-1 negative stands as final for
  its subset; calibration sequences (peel_apple, pour_salt) remain
  calibration-only; held-out camera rule (mod 4) unchanged; floors
  36/36/36/0.5 carry over UNCHANGED (the power analysis is
  data-independent). The R2 secondary statistic and the split
  design are the only definitional additions, both frozen pre-data.
- REVIEW GATES: the screening prereg is reviewed by a fresh-context
  reviewer BEFORE the first candidate download (same three-round
  discipline as cycle 1 if needed); the cycle-2 gate application
  requires the same independent-recomputation + integrity-audit
  protocol as cycle 1.

## 5. Budget envelope (proposal; hard numbers to the experiment plan)

- Screening: <= 20 candidates in queue order before a mandatory
  checkpoint (storage ~150–300 GB; tracker+census ~0.2 GPU-h per
  candidate on the measured timing; ceiling proposal 6 GPU-h
  total). Long-duration sequences use the disclosed chunked tracker
  path. Half-window screening halves tracker cost.
- Cycle-2 gate: ~0.5 GPU-h (full-window tracks for up to 3 selected
  sequences + CPU census).
- Storage policy: screened-out candidates' extracted frames may be
  deleted AFTER their screening census artifact is sealed (zips are
  retained; re-extraction is cheap) — deletion of derived data only,
  never raw zips, recorded per candidate.

## 6. What follows this page

(a) hostile fresh-context refine of THIS design (the /research-refine
step); (b) repairs; (c) the cycle-2 experiment plan + frozen
screening prereg (`prereg_m1_cycle2_screen_v1.json`) with reviewer
sign-off; (d) implementation: screening pipeline (the validated
converter/tracks/census stack + a half-window mode already supported
by --window), submissions, selection, cycle-2 gate.
