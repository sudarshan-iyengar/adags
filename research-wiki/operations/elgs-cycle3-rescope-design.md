# EL-GS Cycle-3 Design: writing_2-Anchored Reactivation Rescope

Status: SIGNED — the frozen gate prereg
(`prereg_m1_cycle3_gate_v1.json` at `6de4d60`) passed its
fresh-context sign-off with ZERO blocking findings and six notes;
three BINDING READINGS form part of the signature record: (NOTE-2)
the G-R subset_note reads as "≥ 36 true-absence candidates whose
terminating re-appearance run starts in the unscreened half", with
the straddle count reported; (NOTE-3) pooled coverage is
ratio-of-sums; (NOTE-4) an undefined gate statistic fails its floor.
The reviewer verified every cited number against the sealed table,
recomputed the shrinkage/overdispersion arithmetic, confirmed the
dev subset is forced with no free choice, and confirmed both
verdicts mechanical and genuinely failable. Verdict (verbatim):
"SIGNED. […] Tracks may be unquarantined; the gate may run."
Design history: hostile refine SOUND WITH REPAIRS (8 blocking + 8
notes; all applied). USER DECISION 2026-08-12: after the cycle-2
tranche-1 DRY, the user selected the narrow R3 variant — an APPROVED
CLAIM CHANGE. This page is the durable record.

Review verdict (verbatim): "SOUND WITH REPAIRS. The rescope's
skeleton is correct: the user-approved claim change is authorized
under the frozen checkpoint; the anchor selection is forced by the
sealed data, not chosen; the companion rule is deterministic and
correctly applied; the gate is genuinely failable and its floors are
the right magnitude; the consequence structure preserves both prior
negatives. No structural break found. […] The single most important
repair is 5: without the split verdict, this design can reproduce
the exact failure mode it was created to escape."

## 1. The claim rescope (user-approved; repairs 1, 2, 13)

> **APPEND-ONLY CORRECTION (2026-08-14).** The "~700 absence" figure below
> is the defective-era total; corrected it is **597**. More importantly,
> the ABSENCE half of "measured this supply as massive" does not survive:
> the frozen absence instrument is diagnosed at **status_2 (material
> defect), UNANIMOUS across 144 readings** — zero of 597 windows
> corroborated, 96.6% explained by a per-point visibility flag below 0.5
> (87.6% of pairs) or by never-queried cameras (12.2%). The OCCLUSION
> supply claim is untouched. See
> [[operations/elgs-absence-diagnostic-result]]. Original text preserved.

UNCHANGED (broad, G13): every occlusion/absence/censored-evidence
claim — tracker visibility states as representation-level
presence/identity evidence; measurement-model existence inference;
BIRTH/FISSION/TRUNCATE under the censored energy; conditional claims
only. Screening measured this supply as massive (~240k occlusion,
~700 absence over 20 half-windows).

RESCOPED (narrow, G14): REACTIVATE is claimed and tested ONLY on
sequences satisfying the OPERATIONAL scope predicate: **the frozen
tracker census measures >= 12 union returns at >= 0.5 coverage (the
cycle-2 eligibility predicate). writing_2 is the sole current
member.** The phrase "hand-held object removed from and returned to
the workspace" is informal characterization only — the scope is
defined by measured event supply, not activity semantics: the SAME
activity type (writing_1) measured ZERO returns, and "names do not
predict event content" is this program's own validated lesson.
"writing_2-like" means exactly: passing the same census predicate.
Favorable fact (claimed): writing_2's screened half shows primary 49
vs union 50 — the anchor owes essentially nothing to the R2′
loosening; it passes eligibility on the strict cycle-1 statistic
alone.

NOVELTY SCOPING ADDENDUM (prescribed text, to be appended verbatim
to [[operations/elgs-novelty-record]] at gate time):
"Scoping addendum (2026-08-12, cycle-3 rescope): (i) CC4/G14
(reactivation-with-own-content) empirical support = one sequence,
one activity type, one rig (DiVa-360 surround; writing_2); (ii) the
same activity type (writing_1) measured zero returns under the
frozen census — the claim scope is the operational census predicate,
not the activity type; (iii) CC5 (conjunction) inherits the same
scope wherever reactivation contributes to it; (iv) an optional
fresh novelty round was offered to the user and deferred. If any
future claim statement asserts G14 occupancy as a general
dynamic-scene capability, this addendum position collapses and a
re-rating becomes mandatory." The addendum-not-re-rating position:
the 8.0-conditional priced mechanism-cell occupancy, whose
conditionality concerns q as an observation model; evidence breadth
was not what it priced.

## 2. Dev subset (frozen rule over SEALED screening data; repairs 8, 9)

- ANCHOR: writing_2 — the UNIQUE ELIGIBLE candidate and the union
  maximum among the 18 coverage-passing candidates (50 vs runner-up
  9; scissor's 75 exceeded 50 but failed coverage at 0.441).
  Realized common range: 481 frames (0–480; screened half 0–239).
- COMPANIONS (deterministic rule, EXCLUDING THE ANCHOR): the two
  sequences with screened-half coverage >= 0.5, ranked by
  screened-half true_absence_candidate_count descending, ties by
  union returns descending then ascending alphabetical → pour_tea
  (73 absences; 452 frames), then tambourine over put_candy on the
  18-absence tie (union 9 > 4; 256 frames).
- DEV = {writing_2, pour_tea, tambourine}. DISCLOSURE: the companion
  rule was authored after the sealed table was visible; it is
  deterministic given its statement, but its parameterization
  (absence-primary, union tie-break) is post-data — consistent with
  the disclosed data-informed-selection methodology of this program,
  and said plainly here. Selection-bias and tracker-legibility
  disclosures of cycle 2 carry forward verbatim.

## 3. The cycle-3 gate — TWO SEPARATELY-VERDICTED PRECONDITIONS
(repairs 3, 4, 5, 6; notes 11, 12, 16)

Both computed on the UNSCREENED (second) halves from full-range
tracks, under cycle-2's frozen boundary conventions, definitions
(revision-3 base + twelve readings + R2′ union with its loosening
disclosure and primary co-reporting), the evaluator at `1d8f3b0`+
including its disclosed greedy per-identity enumeration convention
(R2′ verification record), and the standing verification protocol
(independent fresh-context recomputation from primary inputs +
integrity audit before any recorded verdict).

**G-R (reactivation precondition; writing_2 alone):**
union returns >= 36 AND coverage >= 0.5 on writing_2's unscreened
half. Because union returns are a subset of true-absence candidates
by frozen definition, a G-R pass automatically establishes >= 36
absence windows on writing_2's unscreened half — REACTIVATE's
absence-window precondition needs no separate floor.
SHRINKAGE DISCLOSURE (corrected mechanism): selection shrinkage is
NEGLIGIBLE by order statistics — the runner-up eligible rate is 9,
and P(rate-9 >= 36) ~ 1e-11, so top-of-pool selection cannot have
inflated writing_2's 50 (this strengthens the gate). The honest risk
is the VARIANCE MODEL: returns are burst-clustered events from one
or two identities, and under overdispersion var = φ·mean,
P(unscreened >= 36 | rate 50) falls from the Poisson idealization
0.98 to ~0.94 (φ=2) and ~0.85 (φ=4); no half-to-half model is
trusted, and the gate exists to catch phase confinement. Genuinely
failable; a valid G-R FAIL is final for the reactivation claim
family on this subset.
ONE-SCENE INFERENCE MACHINERY (chosen repair 4a): the scene-level
cluster bootstrap is degenerate for a one-scene claim; the cycle-3
prereg therefore specifies the reactivation-arm resampling unit as
a TEMPORAL-BLOCK bootstrap within writing_2 (blocks of consecutive
events; exchangeability caveat disclosed), and the gate artifact
must publish the per-identity decomposition of writing_2's returns.

**G-OA (occlusion/absence precondition; the three dev sequences):**
pooled unscreened-half occlusion >= 36 AND pooled true-absence
>= 36 AND per-sequence true-absence >= 12 (the frozen per-sequence
construction — companions were selected for absence supply and must
not free-ride; screened halves measured pour_tea 73 / tambourine 18)
AND pooled + per-sequence coverage >= 0.5. Companions carry NO
return floor BECAUSE no reactivation claim is made on them.

Each precondition receives its OWN verdict; each PASS restores the
M1 precondition for ITS claim family; each valid FAIL is final for
its family on this subset. Concordance diagnostic reported next to
both. M1-A0b runs post-gate on the subset, diagnostic-only.

## 4. Consequence structure (note 14, 15)

- G-R PASS + G-OA PASS ⇒ both preconditions restored; starting M2
  remains a separate explicit user approval, with the M2 matrix
  inheriting the scope (reactivation arms on scope-predicate
  content; occlusion/absence arms on the full dev subset).
- Partial or full FAIL ⇒ preserved negatives per family; remaining
  options return to the user: R4 dataset extension, full R3
  descope, OR tranche-2 screening (still live under the frozen
  cycle-2 prereg: 21 short + 8 long unscreened, the deferred long
  list including placement names legos/puzzle/origami).
- Cycle-1 and cycle-2 records remain final and untouched. The
  cycle-2 DRY is not retroactively modified: this dev subset is
  constructed by a NEW cycle-3 rule under a user-approved claim
  change (the checkpoint autonomy condition was unmet and the
  decision was the user's to make).

## 5. Track-metadata quarantine (repair 7)

Full-window CONVERSIONS are content-neutral (frame/mask counts
already public from screening) and proceed pre-SIGN. TRACKER
outputs are informative about unscreened-half statistics and are
QUARANTINED: no human or agent reads, summarizes, or inspects track
artifacts or track-bearing logs before the prereg is SIGNED; only
exit status, wall time, and mechanically computed sha256 manifests
are observable; the execution ledger records conformance; a
pre-SIGN tracker failure is a disclosed infra note authorizing no
design edit. (Execution choice: tracks are simply HELD until SIGN.)

## 6. Budget

Conversions (481 + 452 + 256 frames), full tracks ≈ 0.3 GPU-h, gate
census + recomputation ≈ 0.2 GPU-h. Ceiling: 1.5 GPU-h. No
downloads (all sequences acquired and manifested).

## 7. Process

(a) this refined design; (b) `prereg_m1_cycle3_gate_v1.json`
embodying §1–§6 verbatim, fresh-context SIGNED before any
full-window census statistic and before any track artifact is read;
(c) tracks + gate census; (d) independent recomputation + integrity
audit; (e) per-family verdicts recorded either way; (f) novelty
addendum appended at gate time.
