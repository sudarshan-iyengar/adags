# EL-GS Cycle-3 Design: writing_2-Anchored Reactivation Rescope

Status: DRAFT under adversarial refine. USER DECISION 2026-08-12:
after the cycle-2 tranche-1 DRY ([[operations/elgs-cycle2-screening-record]]),
the user selected the narrow variant of option R3 — "proceed with
option 3 narrow variant ... with writing_2's anchor reactivation" —
an APPROVED CLAIM CHANGE. This page is the durable record of what
changes and what does not.

## 1. The claim rescope (user-approved)

UNCHANGED (broad, G13): every occlusion/absence/censored-evidence
claim — tracker visibility states as representation-level
presence/identity evidence; measurement-model existence inference in
differentiable rendering; BIRTH/FISSION/TRUNCATE under the censored
energy; conditional claims only. The screening measured this event
supply as massive across DiVa-360 (~240k occlusion opportunities,
~700 absence candidates over 20 half-windows).

RESCOPED (narrow, G14): REACTIVATE — reactivation of dormant
lineages with their OWN trained content — is claimed and tested ONLY
on hand-held-object leave-and-return content as exemplified by
writing_2 (a hand-held rigid object repeatedly removed from and
returned to a tracked workspace: pen/eraser during writing and
erasing). No claim of general-scene reactivation benefit is made.
The G14 novelty-cell occupancy (mechanism + measurement) stands; its
EVIDENCE BASE is disclosed as one activity type. The novelty record
([[operations/elgs-novelty-record]]) receives a scoping addendum,
not a re-rating; a fresh novelty round is OPTIONAL and deferred to
the user.

Rationale from measurement (not names): tranche-1 screening found
returns concentrated in exactly one well-tracked sequence (writing_2:
84 absences / 50 union returns in HALF a sequence) and two
tracker-illegible ones (scissor 75 union at coverage 0.441, poker at
0.382); every other candidate measured ≤ 9. DiVa-360's return events
genuinely concentrate in writing-like activity.

## 2. Dev subset (frozen selection rule over SEALED screening data)

- ANCHOR: writing_2 (the only eligible candidate under the cycle-2
  rule; union 50, coverage 0.845).
- COMPANIONS (deterministic rule over the fully screened tranche-1
  pool): the two sequences with screened-half coverage >= 0.5,
  ranked by screened-half true_absence_candidate_count descending,
  ties by union returns descending then ascending alphabetical.
  Applying it: pour_tea (73 absences) first; tambourine and
  put_candy tie at 18 absences → tambourine (union 9 > 4).
- DEV = {writing_2, pour_tea, tambourine}. Selection inputs are the
  sealed, published screening table — data-informed selection is the
  disclosed methodology of this program; the tracker-legibility and
  selection-bias disclosures of cycle 2 carry forward verbatim.

## 3. The cycle-3 gate (what it tests, and that it can fail)

Computed on the UNSCREENED (second) halves from full-range tracks,
with cycle-2's frozen boundary conventions, definitions (revision-3
base + twelve readings + R2′ union statistic with its loosening
disclosure), and verification protocol (independent fresh-context
recomputation from primary inputs + integrity audit before any
recorded verdict).

- RETURN floor (the anchor claim's power): union returns >= 36 on
  writing_2's unscreened half ALONE. Honest winner's-curse
  disclosure: writing_2 was selected as the top-1 of 20 by its
  screened half (50), so shrinkage applies; under stationarity
  P(second half >= 36 | rate 50) ≈ 0.98 but the true rate may be
  lower — the margin is 1.39x, BELOW cycle-2's 2x screening margin,
  and the gate is genuinely failable. A valid FAIL is final for
  this subset under the standing policy.
- OCCLUSION and TRUE-ABSENCE floors: >= 36 each, pooled over the
  three dev sequences' unscreened halves (screened halves measured
  175 absences and ~30k occlusions pooled — these floors are
  necessary-condition checks, not the risk carriers).
- COVERAGE: pooled >= 0.5 AND per-sequence >= 0.5 on the unscreened
  halves.
- Concordance diagnostic (cycle-2 §4.10) reported next to the gate.
- M1-A0b audited true absence runs post-gate on the selected subset,
  diagnostic-only, per the frozen protocol.

## 4. Consequence structure

- Gate PASS ⇒ the scientific precondition the original M1 gate
  tested is restored FOR THE RESCOPED CLAIM SET; starting M2 remains
  a separate explicit user approval, now with the disclosed claim
  scope. The M2 experiment matrix inherits the scope: reactivation
  arms/ablations run on writing_2(-like) content; occlusion/absence
  arms run on the full dev subset.
- Gate FAIL ⇒ preserved negative, final for this subset; options
  return to the user (R4 dataset extension or full R3 descope).
- Cycle-1 and cycle-2 records remain final and untouched.

## 5. Budget

Full-window conversions for the three dev sequences (737 + 452 +
256 frames — writing_2/pour_tea/tambourine), full tracks (≤ 0.3
GPU-h), gate census + recomputation (≤ 0.2 GPU-h). Ceiling: 1.5
GPU-h. All sequences already acquired and manifested; no downloads.

## 6. Process

(a) hostile fresh-context refine of THIS design; (b) repairs;
(c) `prereg_m1_cycle3_gate_v1.json` frozen + fresh-context signed
BEFORE any full-window statistic is computed (conversions/tracks may
proceed in parallel — they compute no census statistic);
(d) execution; (e) independent recomputation + integrity audit;
(f) verdict recorded either way.
