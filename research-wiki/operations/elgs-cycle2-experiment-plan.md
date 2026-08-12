# EL-GS Cycle-2 Experiment Plan (screen → select → gate)

Status: ACTIVE once `prereg_m1_cycle2_screen_v1.json` is
fresh-context SIGNED (review in progress). Authority chain:
user directive 2026-08-12 → [[operations/elgs-cycle2-continuation-design]]
(REFINED) → the frozen prereg → this plan. Cycle 1's negative
([[operations/elgs-m1-census-result]]) is FINAL and never
overwritten. All frozen content (pool, queue, windows, statistics,
twelve readings, thresholds, selection algorithm, budget, seal set,
disclosures) lives in the prereg — this page adds only execution
structure.

## Phases

P0. PREREG SIGN-OFF (blocking): fresh-context review of the frozen
    screening prereg. No download before SIGN. On REJECT: repair by
    pre-data amendment + scoped re-review (the cycle-1 pattern).

P1. EVALUATOR ALIGNMENT (blocking, parallel with P0): align
    `scripts/build_m1_census.py` to the twelve frozen readings; add
    the union return statistic (primary OR R2′), the single-camera
    diagnostic, per-sequence eligibility outputs, half-window
    attribution (screen/gate modes with the frozen boundary
    conventions), and the screening-table emitter. CPU tests with
    hand-derived oracles for every changed predicate, incl. reading
    R3's transition rule and the R2′ tolerance geometry. The
    alignment commit hash is recorded in the prereg's
    implementation_alignment clause at sign-off. Full suite green.

P2. TRANCHE-1 SCREENING (per candidate, queue order, ALL 20 before
    selection): acquire zip (public Dropbox; provenance + sha256
    manifest as cycle 1) → convert `--window 0 floor(n/2)-1`
    (probe n first) → half-window tracks (wrapper submission,
    preprocessing category) → screening census (screen mode) →
    seal (tracks + census + manifests + realized window) → optional
    frame deletion (ledgered). Ceilings: 6 GPU-h screening,
    400 GB extracted. Every run through `submit_apollo` with the
    digest-pinned image and the screening prereg as named config.

P3. SELECTION: the frozen total-function algorithm over the fully
    screened tranche; publish the full screening table; record the
    selected subset (or DRY-within-budget).

P4. CHECKPOINT (mandatory, user-facing): interim table + projected
    continuation cost + options (continue / R4 / R3-descope). Only
    "continue" is autonomous-eligible, and only under the prereg's
    liveness condition.

P5. CYCLE-2 GATE (after selection): full-window conversions +
    tracks for the selected subset (≤1 GPU-h) → gate census on the
    UNSCREENED halves (floors 36/36/36/0.5 pooled + per-sequence
    gate coverage) → concordance diagnostic → independent
    fresh-context recomputation from primary inputs → integrity
    audit → verdict recorded either way. PASS restores the M1
    precondition; starting M2 remains a separate user approval.
    A valid FAIL is final for the selected subset; options return
    to the user.

P6. POST-GATE DIAGNOSTICS (only after a PASS): M1-A0b audited true
    absence on the selected subset (diagnostic-only, frozen
    protocol), M1-A / M1-D equivalents as budget allows within the
    ceiling.

## Ownership and verification

Single owner (Fable) for evaluator alignment and the screening
driver; fresh-context workers for the prereg sign-off (P0), the
gate recomputation (P5), and any post-outcome defect confirmation
(frozen policy, max two). Every claim in the result pages traces to
a sealed artifact or ledger line; ledger + claims via the
established `submit_apollo` path.

## Failure policy

Identical to cycle 1 (frozen): infra failures retryable (max 2)
with ledgered defects; scientific results never retried; a valid
gate FAIL on the selected subset is final for that subset.
