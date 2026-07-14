---
type: operations
status: active
date: 2026-07-14
phase: 7
validation: static-only
---

# Phase 7 next-phase baseline

This page fixes the repository and evidence baseline for the next research
phase. It does not claim a new runtime result.

## Provenance

- Current integration base before Phase 7 edits:
  `codex/hpc-orchestrator-bootstrap` at
  `0f019f55ef76b4e3e2a9437ab391285ccc63110b`.
- HPC rescue snapshot:
  `origin/codex/rescue-hpc-20260713` at
  `71814ed1caec351fafd350c49e68250353fb85bf`.
- Windows rescue snapshot:
  `origin/codex/rescue-windows-20260713` at
  `46a4013b84bff193848e47af3cf4621af1f0a07f`.
- Leonardo environment entry point:
  `$WORK/proj_adags/exp_index/leonardo_env.sh`, SHA-256
  `186e67f276683fd3f91b0107a391ff522b1c45d25ff19004d9d321fe1e21b472`.
- Phase 7 performed no Slurm submission and no W&B write.

## Canonical N3V baseline

- Scenes: `cut_roasted_beef`, `flame_steak`, and `sear_steak`.
- Budget: 6000 iterations, capped at 600000 points.
- Configuration:
  `configs/n3v/fixed_budget_lora_route0_600k.yaml`, SHA-256
  `b7372f3ffe5e1a4916eaf991c99669f241fc305d8890e00223ee4c161b46eb3f`.
- Motion family: LoRA route0; scaffold and motion-aware densification disabled.
- An empty `motion_prior_root` follows the current dataset-relative
  `motion_priors` convention.
- Raw data remains read-only; outputs belong under
  `$WORK/proj_adags/runs`.

The manifest-grounded route0 run IDs are:

- `20260701_012706_cut_roasted_beef_fixed_budget_lora_route0_600k`
- `20260701_012711_flame_steak_fixed_budget_lora_route0_600k`
- `20260701_012714_sear_steak_fixed_budget_lora_route0_600k`

Their authoritative local linkage is
`refine-logs/hide_reveal_poc/r011_residual_uncertainty_manifest.json`.
Generated refinement artifacts may remain ignored; durable conclusions are
recorded here and in the research wiki.

## Recovered documented comparator

- Configuration:
  `configs/n3v/fixed_budget_lora_route0_filemask_residual_600k.yaml`.
- Source: HPC rescue commit
  `71814ed1caec351fafd350c49e68250353fb85bf`.
- Imported SHA-256:
  `3b7416c975c2f12e4ade2d9250dd5f6d1030e6362f449a1b5d3aed925ddaaa16`.
- Role: exact configuration for the documented R011 residual-uncertainty
  comparator, not a recommendation to rerun it.

Documented run IDs:

- `20260619_184247_cut_roasted_beef_fixed_budget_lora_route0_filemask_residual_600k`
- `20260619_184247_flame_steak_fixed_budget_lora_route0_filemask_residual_600k`
- `20260619_184247_sear_steak_fixed_budget_lora_route0_filemask_residual_600k`

R011 records a negative result: mean PSNR 30.0734 versus route0 30.5021, and
mean L1/proxy 0.0165723 versus 0.0148316, with worse static ghosting. This
configuration is preserved for reproducibility, not promoted as the next method.

## Current research endpoint

`research-wiki/experiments/r036-r037-visibility-event-real-pilot.md` remains
the current tracked endpoint. It records a failed gate: R037 mean PSNR 30.1089
versus route0 30.5021, 0/5 route0 wins on both PSNR and L1, static-no-worse in
1/5 cases, and a negative oracle gap. Phase 7 does not retune or redesign that
method family.

## Maintained static regression surface

- PanopticSports smoke configuration:
  `configs/panopticsports/smoke.yaml`.
- Source: HPC rescue commit
  `71814ed1caec351fafd350c49e68250353fb85bf`.
- Imported SHA-256:
  `1aa1c1603c5c937abf8cc2d5dbbadc2dfdfb189410ce93169094684717e1283b`.
- Launcher contract: `SMOKE=1` selects this 20-iteration configuration;
  non-smoke execution requires an explicit `CONFIG`.
- Phase 7 validation is syntax and consistency only. No smoke job is submitted.

Deferred rescue engineering is governed by
[Phase 7 deferred engineering](../operations/phase7-deferred-engineering.md).

## Static acceptance boundary

Before runtime experiment validation, the repository must satisfy:

1. imported YAML parses;
2. the two imported files match their rescue-source bytes;
3. launcher shell syntax passes;
4. non-smoke execution cannot silently select an unmaintained full config;
5. scheduler log paths resolve under repository `logs/`;
6. research-wiki pages are not globally ignored; and
7. generated logs, checkpoints, arrays, caches, and refinement artifacts remain
   ignored.
