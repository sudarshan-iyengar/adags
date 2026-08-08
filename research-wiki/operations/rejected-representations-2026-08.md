# Rejected Representation-Level Candidates — 2026-08-08 run

Ledger of serious candidates examined and rejected during the
representation-first discovery run that produced [[operations/lgs-method]].
Reviewer: GPT-5.6-Sol via Codex MCP at xhigh reasoning, FRESH context per
adversarial round (5 rounds; plus a 4-round same-thread refine loop).
Full round records: [[operations/lgs-review-history]]; transcripts in
transient `refine-logs/` (adversarial-round-*-review/redesign.md).

## 1. Occlusion-ordered layered dynamic GS (not pursued past triage)

- Mechanism: multiple ranked occlusion layers with persistent
  hidden-layer state.
- Why it looked promising: the conjunction (order + memory) is verified
  unoccupied (sweep B); the project's core failure is hide/reveal.
- Rejection: structural rig constraint — P03 showed 93.4% of multilayer
  bins lack multi-camera co-support on a frontal rig; all-camera
  occlusion is rare, so per-view visibility is already handled by
  compositing; a global order stack answers the wrong question here.
- Status: FUNDAMENTAL for this rig class. Reconsider only for capture
  geometries with genuine surround coverage.

## 2. Reactivation-by-appearance-matching lifecycle (dossier v1)

- Mechanism: densification-time matching of new-capacity sites against
  existing elements' content; match ⇒ add instance instead of birth.
- Why it looked promising: direct route to reappearance identity.
- Rejection (internal adversarial pass, pre-refine): the matcher is a
  training-side heuristic doing the load-bearing work; gains would be
  attributed to it, not the representation; collides in spirit with
  data-association machinery. Replaced by persistent-by-default +
  fission (v2), later by retrieval-at-predicted-pose (v9) where the
  arbitration is cost-based, not appearance-matching.
- Status: fundamental as a CORE mechanism; appearance similarity may
  return only as a diagnostic.

## 3. Description-cost ratio economics as a unifying principle (v6)

- Mechanism: all structural ops scored by loss-reduction per added
  stored scalar; "reuse wins by construction."
- Why it looked promising: single principle unifying fission/extension/
  birth; codec ancestry.
- Rejection (fresh-context round 2, SINKS): the ratio is undefined for
  zero-cost ops (edge extension, truncation) and negative-cost ops
  (pruning); not an MDL objective (counting float slots ≠ description
  length; greedy per-scalar ≠ knapsack-valid); gameable (quarter-fit
  reuse wins at quarter cost); clone/split/prune/routing were never
  governed, falsifying "one principle."
- Status: FUNDAMENTAL as a claimed principle. Survives only as a
  Lagrangian ACCEPTANCE TEST (ΔL̂ + λ·ΔS < 0) with disclosed
  three-regime scope and no optimality claims ([[operations/lgs-method]]).

## 4. Soft content assignment / rate-annealed consolidation (v7)

- Mechanism: episode instances hold softmax weights over ≤4 candidate
  library contents; rendered content = mixture; temperature annealed to
  hard; usage-entropy rate penalty; association learned by gradient
  ("migration" repairs wrong births).
- Why it looked promising: makes association differentiable — answers
  the unreachable-topology fatal without heuristic matching.
- Rejection (fresh-context round 3, SINKS): NO COHERENT GEOMETRY under
  migration — instance pose offsets learned against one library entry's
  canonical gauge are wrong under another (quaternion sign/gauge
  transport undefined); mixing rotations/scales ill-defined; opacity
  mixture semantics unspecified (σ of mix vs mix of σ); soft phase can
  exploit synthetic average content no hard assignment reproduces;
  CAP CONTRADICTION — keeping the born-content alternative trainable
  consumes the scalar budget, dropping it kills correctability;
  entropy sharpening fights migration (recoverability conditional on
  candidate recall, live temperature, gauge alignment); structural
  lineage vs content-sharing lineage conflated; post-hardening
  correction impossible.
- Status: FUNDAMENTAL as formulated. Reconsider only with (a) an
  appearance-only library (no geometry in shared content), (b)
  candidate-specific gauge transport defined, and (c) a staging policy
  reconciling correctability with the scalar cap.

## 5. SMS/optimizer/training-side families (prior run)

See [[operations/rejected-approaches-2026-08]] — the 2026-08-08 STAR-GS
run's ledger (residual routing, support-matched momentum, observation
clocking, calibrated-consensus claims) remains binding; this run did not
re-open training-side axes.

## Cross-cutting lessons (this run)

- The kernel survived all five hostile rounds; every DISCOVERY mechanism
  bolted onto it (statistics-driven carving v5, ratio economics +
  prediction-gated extension v6, soft assignment v7) sank UNTIL the
  search was demoted to honest supporting machinery with trial-render
  confirmation (v9).
- Hostile reviewers consistently rated the unoccupied kernel
  "compositionally obvious" (4-7/10) while conceding no occupant —
  unoccupied ≠ deep; the burden shifts to decisive empirical validation.
- Exactness language ("disjoint by construction", "guaranteed
  gradient floor", "confidence bound") was attacked every time it
  exceeded what the math delivered; the surviving spec states invariants
  precisely (count/order/floors) and demotes the rest to disclosed
  monitoring.
- Same-thread refine approval (9.2 READY) again failed to predict
  fresh-context survival (immediate SINKS) — reconfirming the project's
  standing rule that fresh-context adversarial review is the binding
  standard.
