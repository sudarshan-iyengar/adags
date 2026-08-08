# STAR-GS v9 — Preserved Training-Side Candidate Method

Date: 2026-08-08 (status finalized same day after user review)
Status: **preserved candidate, not the lead direction.** The complete
method, review history, novelty assessment, and preregistered experiment
plan are durable and sufficient to implement and test STAR-GS later, but
STAR-GS is NOT currently approved as the lead research direction; the
project's next research phase is representation-level method discovery
(not yet begun, and not part of this record). Nothing from this run was
implemented, trained, or submitted to any scheduler.
Pipeline provenance: 5-round research-refine (6.1 RETHINK → 9.1 READY on a
predecessor formulation), then 5 fresh-context adversarial rounds (4 SINKS
→ redesigns → SURVIVES-WITH-RISKS), then novelty check (5.5/10, PROCEED
WITH CAUTION). Durable record: [[operations/star-gs-v9-review-history]]
(review substance), [[operations/star-gs-v9-experiment-plan]] (test plan),
[[operations/rejected-approaches-2026-08]] (rejected predecessors),
[[operations/sota-sweep-2026-08]] (literature/code findings). Raw
generated transcripts remain in transient `refine-logs/` but contain
nothing essential beyond these pages.
Relationship to prior direction: [[operations/phase9-csvl-vpl-v2-direction]]
remains the last user-approved direction record; its controls, discipline,
and negative record carry over unchanged and bind any future STAR-GS
implementation.

## Positioning (novelty-referee-approved wording)

**Budget-neutral correction of depth-deficient dynamic Gaussian models
through multiview residual-space carving.** Existing error-driven
correction for 4DGS — closest: [[papers/kang2025_cec_4dgs]] (CEC-4DGS,
SIGGRAPH Asia 2025) — identifies residual regions but places corrective
geometry by back-projecting at the model's own rendered depth, which is
structurally unreliable exactly at disocclusions and missing surfaces.
STAR-GS removes that dependency and places correction under explicit
capacity accounting. Never claim "the first"; the claim is the exact
conjunction, results-carried.

## Method

Backbone: existing ADAGS hybrid 3D/4D trainer, unchanged; zero new
trainable components; model-internal only (no external evidence — the
round-1 L5 ≥ L3 refutation is respected by construction); no rendered-
opacity manipulation; annotations evaluation-only.

**SRC (occlusion-aware soft residual carving), the proposer.** At
preregistered checkpoints: synchronous low-res residual sweeps over a
preregistered construction camera fold (disjoint audit fold reserved;
cam00 never used anywhere). Per sampled scene time, a coarse voxel grid is
scored by the MEAN of rank-normalized residuals at each voxel's projection
over cameras whose ray is not blocked by confident model geometry
(accumulated-opacity gate; minimum eligible-view count). Selection is
conservative: top-K per event, min-score floor, non-maximum suppression,
single birth per component — preregistered constants, no statistical
calibration claims (an FDR/permutation formulation was reviewed and
rejected as statistically invalid; the time-shifted-input construction
survives only as a sanity diagnostic that must collapse candidate yield).
Ancestry acknowledged: continuous, occlusion-gated residual analogue of
space carving, applied to capacity allocation during training.

**Birth.** Each selected candidate births ONE parent-free short-support
Gaussian at (x, t): temporal center t, temporal scale = constructor time-
grid spacing; no lifespan inference, no motion fitting (extensions, not
claims); init: backbone-default opacity, isotropic rotation, scale from
consensus support extent, SH DC = robust median of contributing cameras'
pixels, motion zero (disclosed).

**Budget neutrality.** Dynamic-bank births retire dynamic-bank donors only
(static allocation untouched), donors = lowest rendered utility
(Multi4D-style transmittance-weighted contribution; cited substrate) with
protections (min age, min lifetime-observation floor, hysteresis,
per-event caps), through the existing crash-safe point-neutral B01
transaction with per-row optimizer-state reset. Resource ledger reports
constructor compute — parameter neutrality is not claimed as compute
neutrality (an iso-compute control lane exists).

**Substrate (not a claimed contribution).** Presence/information-weighted
growth statistics extending the validated round-1 L2 effect
(TAD-GS/4D-Scaffold-GS cited); ablated, demotable.

## Claims

1. (Dominant, results-carried) The conjunction: depth-free multiview
   residual-carving proposer + parent-free time-local birth + bank-matched
   budget-neutral retirement, validated by a matched-capacity AND
   matched-compute causal-control matrix. Viability conditions from the
   novelty check are binding: beat CEC-4DGS faithful AND budget-matched
   reimplementations; event-level gains materially larger than global;
   direct localization evidence; component ablations failing as predicted.
2. (Secondary) The annotated disocclusion-event benchmark + causal-control
   evaluation matrix (output-blind, evaluation-only; supplementary to
   standard metrics per the run mandate).
3. (Measurement) One-step target-set exclusion (definitional) + measured
   multi-step reachability of audit-defined deficit sites under baseline
   training.

## Review-imposed obligations (binding)

- Wording: "we introduce", never "the first" (FreeTimeGS relocation and
  SharpTimeGS stage-2 occupy budgeted reallocation; CEC-4DGS occupies
  error-driven time-local birth).
- Count-match controls to REALIZED accepted births with identical resets/
  init/retirement; WHEN-control strata from permitted signals only;
  within-run attribution labeled "local direct effect under the mixed
  deployment policy"; shifted-input collapse criterion preregistered as
  diagnostic; occlusion abstention not called "conservative" unqualified.
- Top risks on record: (1) "space carving + churn" incrementality
  perception; (2) residual consensus finding non-capacity-fixable errors;
  (3) six-scene statistical power.

## Experiment plan

Durable version: [[operations/star-gs-v9-experiment-plan]] — M0 sanity →
M1 Phase-A constructor gate (~10-15 GPU-h; kills cheaply) → M2 CEC port +
smokes → M3 main result + decisive contrasts (3 seeds) → M4 remaining
controls + localization/attribution → M5 freeze, locked transfer,
six-scene full-length finals, stress tier. Total ~405-460 GPU-h within
the 400-600 envelope. All preregistration, activation-diagnostic, and
scene-discipline rules of the objective apply unchanged.

## Honest assessment

Novelty 5.5/10 (PROCEED WITH CAUTION): a results-carried systems-mechanism
paper. Existence evidence for the effect: round-1 L3>L4 (+0.29 dB at
matched counts from ≤0.4% of rows re-targeted, crude external targets,
single seed) and CEC-4DGS's published gains (+0.42 dB Technicolor; +0.12
dB N3V — small at global tier, consistent with this project's G7
evidence, hence the event benchmark). If Phase A or the decisive contrasts
fail, the preregistered outcome is a cheap, recorded negative.
