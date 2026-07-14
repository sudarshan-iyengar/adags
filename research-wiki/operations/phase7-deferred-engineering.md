---
type: operations
status: deferred
date: 2026-07-14
phase: 7
---

# Phase 7 deferred engineering

Phase 7 recovers evidenced artifacts and establishes a reproducible baseline. It
does not redesign analysis, export, or submission tooling. The entries below are
durable reconsideration gates, not authorization to implement or run them.

## Broad W&B analysis

- Rescue branch: `origin/codex/rescue-hpc-20260713`
- Exact source commit: `71814ed1caec351fafd350c49e68250353fb85bf`
- Original artifact path: `scripts/analyze_wandb_adags.py`
- Disposition: defer as a separately scoped engineering task.
- Why deferred: the script performs broad run discovery, silent duplicate
  canonicalization, and optional writeback while newer checkpoint-scoped
  analyzers already exist. Importing it unchanged would not be scientifically
  safe, and redesign is outside Phase 7.
- Useful concepts worth preserving: paired deltas, full-curve history,
  decomposition and factor views, fixed-budget/capacity comparisons, routing
  diagnostics, and explicit metric-availability reporting.
- Acceptance criteria for reconsideration:
  1. an approved engineering task defines its consumers and scientific question;
  2. input is an immutable manifest of exact run IDs, with no silent deduping;
  3. the default is read-only and writeback is separately authorized;
  4. missing diagnostics and run-budget semantics are explicit, including the
     current 6000-iteration contract; and
  5. offline fixtures pass before any later Slurm-backed runtime validation.

## Manifest-scoped W&B synchronization

- Rescue branch: `origin/codex/rescue-hpc-20260713`
- Exact source commit: `71814ed1caec351fafd350c49e68250353fb85bf`
- Original artifact path: `scripts/sync_wandb_runs.sh`
- Disposition: defer as a separately scoped engineering task.
- Why deferred: its broad offline-payload discovery default can sync unintended
  runs. Phase 7 prohibits W&B writes.
- Useful concepts worth preserving: offline-run detection, append behavior,
  explicit entity/project routing, disabling TensorBoard sync, and dry-run
  reporting.
- Acceptance criteria for reconsideration:
  1. accept only an explicit manifest or exact payload-directory list;
  2. provide a deterministic dry run and idempotency checks;
  3. keep credentials and raw payload details out of logs and Git; and
  4. obtain separate approval for network writes.

## File-mask-only 600k control

- Rescue branch: `origin/codex/rescue-hpc-20260713`
- Exact source commit: `71814ed1caec351fafd350c49e68250353fb85bf`
- Original artifact path:
  `configs/n3v/fixed_budget_lora_route0_filemask_600k.yaml`
- Disposition: defer.
- Why deferred: no documented result currently depends on this exact
  configuration, tracked metadata does not select it, and it is not in the
  immediately approved experiment family.
- Useful concepts worth preserving: file-backed motion masks without residual
  fallback, a 6000-iteration/600k-point route0 control, and disabled scaffold
  and motion-aware densification.
- Acceptance criteria for reconsideration:
  1. an approved experiment plan selects this exact control;
  2. the motion-prior layout and mask provenance are fixed;
  3. comparator, compute budget, and dynamic/static gates are predeclared; and
  4. runtime validation is scheduled in the later experiment-validation phase.

## Full PanopticSports experiment configuration

- Rescue branch: `origin/codex/rescue-hpc-20260713`
- Exact source commit: `71814ed1caec351fafd350c49e68250353fb85bf`
- Original artifact path:
  `configs/panopticsports/scaffold_lora_route0_dyn_densify_ptbudget.yaml`
- Disposition: defer.
- Why deferred: it is neither required by the current approved research
  direction nor a maintained regression/smoke test. A historical launcher
  reference is not evidence for importing a full experiment.
- Useful concepts worth preserving: PanopticSports metadata/image conventions,
  cross-dataset intent, and a 6000-iteration scaffold-family setup.
- Acceptance criteria for reconsideration:
  1. an approved research plan selects PanopticSports, or the configuration
     becomes a maintained regression for functionality already in the base;
  2. current loader support and the baseline implementation are reconciled;
  3. preprocessing, data shape, and priors are documented; and
  4. runtime validation occurs through Slurm in a later phase.

## Explicit rejection

`scripts/extract_results.py` from
`origin/codex/rescue-hpc-20260713` at
`71814ed1caec351fafd350c49e68250353fb85bf` is rejected for Phase 7. It is a
legacy, layout-coupled extractor whose assumptions are not evidenced as safe for
the current result surfaces. No replacement or rewrite is authorized here.
