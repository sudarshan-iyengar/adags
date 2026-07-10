# Overnight Runbook

## Objective

Develop and verify a non-oracle Gaussian-model method that can produce the event-crop fix suggested by R001-R017.

Completion means one of:

- PASS: a non-oracle Gaussian-rendered method improves the frozen R009 windows and recovers a meaningful fraction of the R013 oracle event-crop upper bound.
- FAIL: reasonable non-oracle method attempts do not improve over route0, matched-lifespan, and residual/uncertainty baselines, with evidence.
- BLOCKED: missing permissions/data/compute or the same unresolved implementation blocker persists after three serious diagnosis-and-patch attempts.

## Required State Files

- `refine-logs/OVERNIGHT_STATUS.md`: live recovery state, job IDs, commits, results, blockers, and next command.
- `refine-logs/EXPERIMENT_TRACKER.md`: milestone status and concise evidence.
- `refine-logs/EVENT_CROP_FIX_EVIDENCE.md`: compact source-of-truth evidence summary for R001-R017 and the event-crop objective.
- `refine-logs/EVENT_CROP_METHOD_TRACKER.md`: method candidates, predeclared metrics, thresholds, attempts, job IDs, and verdicts.
- `refine-logs/OVERNIGHT_RUNBOOK.md`: operational instructions and recovery notes.
- `research-wiki/event-crop-fix.md`: human-facing wiki memory for the event-crop fix objective.

## Operating Loop

1. Read `OVERNIGHT_STATUS.md` and `EXPERIMENT_TRACKER.md`.
2. Read `EVENT_CROP_FIX_EVIDENCE.md` and `EVENT_CROP_METHOD_TRACKER.md`.
3. Identify the current method candidate and the next reproducible command.
4. Inspect local code and HPC state needed for that method.
5. Patch only the smallest necessary scripts/docs/configs.
6. Run local smoke/unit checks, including dry-run path validation when available.
7. Commit with a clear message, push, and record the commit hash.
8. On HPC, fetch/switch/pull the branch under `$WORK/proj_adags/repo/adags`.
9. Submit scheduler jobs through Slurm. Do not run heavy CPU/GPU work on the login node.
10. Poll with `squeue` and `sacct`; avoid fragile interactive tail loops.
11. On failure, read `.out` and `.err` logs under `$WORK/proj_adags/repo/adags/logs`, diagnose, patch, commit, push, pull, and rerun.
12. On success, collect outputs/metrics, update tracker/status/wiki, and decide PASS/FAIL/next candidate.

## R034+ Mechanism-Changing Loop

The next hide/reveal milestone must change the mechanism rather than only changing support masks.

1. Keep route0 as the smooth-motion backbone and matched control.
2. Generate event candidates only from training-time, non-oracle cues. The frozen R009 crops may be used only after a run for scoring/audit.
3. Compare two matched-budget hypotheses on the same support:
   - `H_smooth`: ordinary visible smooth transport / matched route0 continuation.
   - `H_event`: persistent Gaussian state with time-dependent opacity gate `v_i(t)` applied before rasterization.
4. Freeze the local photometric/temporal admission margin before real-window scoring.
5. Use hysteresis for accepted event states to avoid frame-by-frame flicker.
6. Allocate or reinitialize event-local capacity during the original training loop, under the same strict point/capacity budget as the matched smooth control.
7. Validate first on a synthetic hide/reveal fixture where identity/reconnection is observable.
8. Only then run one predeclared real pilot against route0 and matched `H_smooth`.

R034+ kill rule: if synthetic identity/reconnection or admission separation fails, do not launch the real pilot. If the real pilot is valid but fails the frozen-window gate, record a scientific FAIL and iterate to the next predeclared mechanism candidate rather than tuning on the five crops.

## HPC Facts

- SSH: `ssh siyengar@login.leonardo.cineca.it`
- `$WORK`: `/leonardo_work/EUHPC_D21_034`
- HPC repo: `$WORK/proj_adags/repo/adags`
- HPC logs: `$WORK/proj_adags/repo/adags/logs`
- HPC data: `$WORK/proj_adags/data/n3v`
- HPC runs: `$WORK/proj_adags/runs`
- Existing environment setup: `exp_index/leonardo_env.sh` and existing run scripts.

## Scientific Rules

- Preserve the frozen R009 evaluation protocol.
- Do not use R009 event crops as test-time method support for future non-oracle methods.
- Keep oracle/perfect-crop results clearly separated from Gaussian-rendered method outputs.
- Do not tune thresholds on the frozen real windows to manufacture success.
- Record candidate support generation, accepted event counts, render folders, metrics, logs, manifests, and commit hashes.
- If a method cannot produce comparable Gaussian-rendered folders, explicitly document why.
- Learned LPIPS and confident-track ID switches are unavailable unless their sidecars/weights are produced without retuning or network downloads on compute nodes.
