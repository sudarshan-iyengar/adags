# Overnight Runbook

## Objective

Scientifically complete `R009` through `R016` in `refine-logs/EXPERIMENT_TRACKER.md`.
Completion means each row is marked `PASS`, `FAIL`, or `SKIP` with evidence.

## Required State Files

- `refine-logs/OVERNIGHT_STATUS.md`: live recovery state, job IDs, commits, results, blockers, and next command.
- `refine-logs/EXPERIMENT_TRACKER.md`: milestone status and concise evidence.
- `refine-logs/OVERNIGHT_RUNBOOK.md`: operational instructions and recovery notes.

## Operating Loop

1. Read `OVERNIGHT_STATUS.md` and `EXPERIMENT_TRACKER.md`.
2. Identify the next incomplete run ID.
3. Inspect local code and HPC state needed for that run.
4. Patch only the smallest necessary scripts/docs/configs.
5. Run local smoke/unit checks.
6. Commit with a clear message, push, and record the commit hash.
7. On HPC, fetch/switch/pull the branch under `$WORK/proj_adags/repo/adags`.
8. Submit scheduler jobs through Slurm. Do not run heavy CPU/GPU work on the login node.
9. Poll with `squeue` and `sacct`; avoid fragile interactive tail loops.
10. On failure, read `.out` and `.err` logs under `$WORK/proj_adags/repo/adags/logs`, diagnose, patch, commit, push, pull, and rerun.
11. On success, collect outputs/metrics, update tracker/status, and advance to the next run.

## HPC Facts

- SSH: `ssh siyengar@login.leonardo.cineca.it`
- `$WORK`: `/leonardo_work/EUHPC_D21_034`
- HPC repo: `$WORK/proj_adags/repo/adags`
- HPC logs: `$WORK/proj_adags/repo/adags/logs`
- HPC data: `$WORK/proj_adags/data/n3v`
- HPC runs: `$WORK/proj_adags/runs`
- Existing environment setup: `exp_index/leonardo_env.sh` and existing run scripts.

## Scientific Rules

- R009 must freeze the real-window manifest before any scoring.
- R010-R013 must produce comparable outputs or explicitly document why a baseline cannot be produced.
- R014 runs only if no-refinement/full method is noisy or unstable.
- R015 must generate the PoC table and qualitative crop-strip artifacts.
- R016 must write a concise go/no-go memo grounded in produced outputs.
- Do not tune real-window thresholds to make the method look good unless the tracker explicitly calls for calibration.
- Preserve the predeclared/frozen nature of R009-R013.
