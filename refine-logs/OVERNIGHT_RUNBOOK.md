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

## 2026-07-10 R035 Failure Branch

R035 proxy admission rejected all R020 candidates: `0/72` accepted, validation `ok=true`, mean delta score `+0.200982`. Do not change the frozen R035 admission margin to create accepted events.

Proceed with the revised full-training branch:

1. Use configs `configs/n3v/visibility_event_smooth_control_6000.yaml` and `configs/n3v/visibility_event_train_6000.yaml`.
2. Both configs use the fixed R020 high-recall candidate manifest: `refine-logs/hide_reveal_poc/r020_high_recall_motion_supported_nonoracle_candidates/nonoracle_candidate_manifest.json`.
3. Submit R036 smooth-control full trainings.
4. Submit R037 event-gated full trainings.
5. After both eval render folders exist, build a combined frozen-window manifest and score once.
6. Treat the result as PASS/FAIL by the predeclared gate in `EVENT_CROP_METHOD_TRACKER.md`; do not tune candidate thresholds on the frozen windows.

## 2026-07-10 R036/R037 Resume State

R036/R037 full trainings completed successfully with six `chkpnt6000.pth` checkpoints:

- R036 smooth jobs: `49042444`, `49042445`, `49042446`.
- R037 event jobs: `49042510`, `49042512`, `49042514`.

R036 smooth eval was submitted:

- Eval jobs: `49045923`, `49045924`, `49045925`.
- Eval manifest: `refine-logs/visibility_event_smooth_eval_jobs_20260710_060900.tsv`.

R037 event eval is not submitted yet because the Leonardo SSH certificate expired at `2026-07-10T06:08:38`. This is an infrastructure blocker, not a method verdict.

After SSH renewal, resume with:

```bash
ssh siyengar@login.leonardo.cineca.it 'cd /leonardo_work/EUHPC_D21_034/proj_adags/repo/adags && sacct -j 49045923,49045924,49045925 --format=JobID,State,Elapsed,Timelimit,ExitCode -P'

ssh siyengar@login.leonardo.cineca.it 'cd /leonardo_work/EUHPC_D21_034/proj_adags/repo/adags && EVAL_TIME=01:30:00 scripts/submit_visibility_event_pilot.sh --variant event --mode eval --run-manifest refine-logs/visibility_event_train_train_jobs_20260710_024651.tsv'
```

When both eval variants complete:

1. Verify each eval folder has `test/ours_6000/renders`, `gt`, `static`, and `dynamic`.
2. Build one frozen-window manifest that includes route0, residual/uncertainty, matched-lifespan, R036 smooth, and R037 event outputs.
3. Run `scripts/run_hide_reveal_poc.py real-eval` once on that manifest.
4. Run the result-to-claim / experiment-audit review before making any positive claim.
5. Commit, push, and record all job IDs, output folders, metrics, and commit hashes.

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
