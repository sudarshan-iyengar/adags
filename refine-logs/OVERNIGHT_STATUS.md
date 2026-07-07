# Overnight Status

## Recovery Snapshot

- Current objective phase: event-crop non-oracle candidate discovery
- Current run ID: R024 event-candidate local refinement eval jobs pending
- Current method candidate: M1 non-oracle residual-component local refinement; R023 training completed and R024 eval renders pending, scoring pending
- Current branch: `codex/hide-reveal-poc-implementation`
- Current local commit at 2026-07-07 recovery start: `f5d43539aee500051f2a4c5eeca5420293b636f1`
- Last pushed milestone commit: `a724ce68373854447edaa385b3e94bf1809b8824`
- Last HPC job ID: `48802359`
- Latest success/failure: R023 training completed successfully with per-job extension roots and wrote three `chkpnt6200.pth` files; R024 eval jobs `48802355`, `48802357`, and `48802359` are pending on scheduler priority at 2026-07-07T12:11:10+02:00.
- Next command to run: monitor eval jobs `48802355`, `48802357`, `48802359`, then augment `refine-logs/hide_reveal_real_windows.json` with the refined eval render folders and run frozen-window scoring.
- Open blockers: none known yet

## Dirty State At 2026-07-07 Recovery Start

Recorded before new event-crop evidence edits.

- Branch: `codex/hide-reveal-poc-implementation`
- HEAD: `f5d43539aee500051f2a4c5eeca5420293b636f1`
- Tracked dirty files present before this recovery edit:
  - `refine-logs/hide_reveal_poc/r015_poc_summary/poc_table.md` (line-ending/formatting only in `git diff`)
  - `refine-logs/hide_reveal_poc/r016_go_no_go_memo.md` (Markdown table formatting)
- Untracked files/directories present before this recovery edit:
  - `.obsidian/`
  - `AGENTS.md`
  - `Untitled.canvas`
  - `configs/n3v/bootstrap.yaml`
  - `det_con.yaml`
  - `follow-up.md`
  - `idea-stage/`
  - `requirements.txt`
  - `verify_mask.jpg`
  - `verify_masked_flow.jpg`
  - `verify_raw_flow.jpg`
- Preservation rule: do not revert or restage those unrelated/pre-existing changes unless explicitly requested.

## Session Log

### 2026-07-07 - Event-crop non-oracle objective recovery

- Read goal objective file `C:\Users\Sudarshan\.codex\attachments\833ea181-b1f3-4a27-a22e-ea7ccafec21a\goal-objective.md`.
- Read prior Codex thread `019f34bf-83e1-7191-b3b6-64dc6bf3f06e`; retained that R013/R015 were GT-crop upper bounds and R017 was the actual checkpoint-backed renderer failure.
- Read `research-wiki/query_pack.md`, `research-wiki/gap_map.md`, relevant idea/experiment/paper pages, and the R017 wiki experiment page.
- Wrote evidence summary `refine-logs/EVENT_CROP_FIX_EVIDENCE.md`.
- Wrote wiki memory `research-wiki/event-crop-fix.md`.
- Wrote predeclared method/evaluation tracker `refine-logs/EVENT_CROP_METHOD_TRACKER.md`.
- Predeclared first candidate: M1 non-oracle residual-component local refinement, pending code inspection.
- Committed and pushed recovery evidence as `0b18166bec6d1a2d371764c70bdcf53b23319a5e` (`Record event-crop recovery evidence`).

### 2026-07-07T02:56:22+02:00 - A1 non-oracle candidate-discovery smoke

- Implemented a new `nonoracle-candidates` PoC stage in `scripts/run_hide_reveal_poc.py`, `utils/hide_reveal_poc.py`, and Slurm wrappers.
- Candidate scoring uses route0 dynamic output, route0-vs-static render deltas, motion-mask boundaries, and route0 render flicker; it explicitly records `uses_gt_residual=false` and `uses_frozen_window_labels=false`.
- Local smoke command completed with bundled Python:
  `scripts/run_hide_reveal_poc.py nonoracle-candidates --manifest refine-logs/hide_reveal_poc/local_smoke/nonoracle_candidates/smoke_manifest.json --out-dir refine-logs/hide_reveal_poc/local_smoke/nonoracle_candidates/out --window-length 4 --temporal-stride 2 --tile-size 16 --tile-stride 8 --top-k-per-scene 3`
- Smoke result: `validation_ok=True`, `validation_errors=0`, `candidates=3`.
- Smoke report: `refine-logs/hide_reveal_poc/local_smoke/nonoracle_candidates/out/nonoracle_candidate_report.md`.
- Local Windows `bash` is unavailable, so shell wrapper syntax checks are deferred to Leonardo before Slurm submission.
- Committed A1 implementation as `0bf0967483e622af5cb6ac81de2b3f09060c33d9` (`Add non-oracle event candidate discovery`).

### 2026-07-07T03:08:00+02:00 - R018 non-oracle candidate job completed

- Pulled branch on Leonardo to `db467a328b6b0e02482eb0e36de24b27b850b907`; shell syntax checks passed with `bash -n`.
- Submitted `scripts/submit_hide_reveal_poc.sh --stage nonoracle-candidates --manifest refine-logs/hide_reveal_real_windows.json --out-dir refine-logs/hide_reveal_poc/r018_nonoracle_candidates`.
- Slurm job `48763378` completed on `lrdn3301` with `State=COMPLETED`, `ExitCode=0:0`, `Elapsed=00:02:01`.
- Outputs collected locally:
  - `refine-logs/hide_reveal_poc/r018_nonoracle_candidates/nonoracle_candidate_manifest.json`
  - `refine-logs/hide_reveal_poc/r018_nonoracle_candidates/nonoracle_candidate_metadata.json`
  - `refine-logs/hide_reveal_poc/r018_nonoracle_candidates/nonoracle_candidate_components.csv`
  - `refine-logs/hide_reveal_poc/r018_nonoracle_candidates/nonoracle_candidate_validation.json`
  - `refine-logs/hide_reveal_poc/r018_nonoracle_candidates/nonoracle_candidate_report.md`
  - `refine-logs/hide_reveal_poc_nonoracle-candidates_jobs_20260707_030101.tsv`
  - `logs/hide_reveal_nonoracle-candidates_48763378.out`
  - `logs/hide_reveal_nonoracle-candidates_48763378.err`
- R018 structural result: PASS. `validation_ok=True`, `validation_errors=0`, `candidates=24`.
- R018 detector result: FAIL / diagnostic. Posthoc audit in `refine-logs/hide_reveal_poc/r018_nonoracle_candidates/frozen_overlap_audit.md` covered `0/5` frozen windows under crop-IoU >= 0.1 and temporal-IoU >= 0.25; candidates clustered on top image bands and had zero motion-mask support.

### 2026-07-07T03:08:30+02:00 - A2 motion-supported candidate detector

- Revised candidate scoring so route0 dynamic output, route0-vs-static deltas, and route0 flicker are multiplied by motion-mask support when masks are available.
- Score terms are now motion-supported dynamic render, motion-supported static-render delta, motion-mask interior, motion-mask boundary, and motion-supported flicker.
- Local smoke command completed:
  `scripts/run_hide_reveal_poc.py nonoracle-candidates --manifest refine-logs/hide_reveal_poc/local_smoke/nonoracle_candidates/smoke_manifest.json --out-dir refine-logs/hide_reveal_poc/local_smoke/nonoracle_candidates/out_motion_supported --window-length 4 --temporal-stride 2 --tile-size 16 --tile-stride 8 --top-k-per-scene 3`
- A2 local smoke result: `validation_ok=True`, `validation_errors=0`, `candidates=3`, with selected crop `[24, 32, 40, 48]` over the synthetic moving square.
- Committed A2 implementation and R018 evidence as `f69034be1ca32ddcd24756d945ead467d59e3c24` (`Require motion support for event candidates`).

### 2026-07-07T03:15:30+02:00 - R019 motion-supported candidate job completed

- Pulled A2 on Leonardo; shell syntax checks passed with `bash -n`.
- Submitted `scripts/submit_hide_reveal_poc.sh --stage nonoracle-candidates --manifest refine-logs/hide_reveal_real_windows.json --out-dir refine-logs/hide_reveal_poc/r019_motion_supported_nonoracle_candidates`.
- Slurm job `48763799` completed on `lrdn0122` with `State=COMPLETED`, `ExitCode=0:0`, `Elapsed=00:02:01`.
- R019 structural result: PASS. `validation_ok=True`, `validation_errors=0`, `candidates=24`.
- R019 detector result: PARTIAL / insufficient for full M1. Posthoc audit in `refine-logs/hide_reveal_poc/r019_motion_supported_nonoracle_candidates/frozen_overlap_audit.md` covered `2/5` frozen windows. It fixed the R018 top-band artifact but missed both `cut_roasted_beef` windows and the second `flame_steak` window under the fixed overlap rule.

### 2026-07-07T03:21:00+02:00 - R020 high-recall candidate pool completed

- Submitted the same A2 detector with `HIDE_REVEAL_NONORACLE_TOP_K_PER_SCENE=24` to test whether the remaining failure was proposal-budget limited.
- Command: `HIDE_REVEAL_NONORACLE_TOP_K_PER_SCENE=24 scripts/submit_hide_reveal_poc.sh --stage nonoracle-candidates --manifest refine-logs/hide_reveal_real_windows.json --out-dir refine-logs/hide_reveal_poc/r020_high_recall_motion_supported_nonoracle_candidates`.
- Slurm job `48764048` completed on `lrdn0085` with `State=COMPLETED`, `ExitCode=0:0`, `Elapsed=00:02:04`.
- R020 structural result: PASS. `validation_ok=True`, `validation_errors=0`, `candidates=72`, `top_k_per_scene=24`.
- R020 posthoc coverage: `3/5` frozen windows under crop-IoU >= 0.1 and temporal-IoU >= 0.25. Audit: `refine-logs/hide_reveal_poc/r020_high_recall_motion_supported_nonoracle_candidates/frozen_overlap_audit.md`.
- Interpretation: R020 provides a non-oracle high-recall support pool for a possible local-refinement attempt, but it is not a clean detector pass and still is not a Gaussian-rendered method result.

### 2026-07-07T03:32:00+02:00 - R021 local-refinement plumbing committed

- Added optional `event_candidate_manifest` support to `utils/motion_prior_utils.py`; when enabled, dynamic ROI masks are intersected with non-oracle candidate boxes for matching frames, and frames with no candidate support receive no ROI loss.
- Added optimization args `event_candidate_manifest`, `event_candidate_scene`, and `event_candidate_dilate`.
- Added short-run config `configs/n3v/event_candidate_local_refine_6200.yaml`, resuming route0 `chkpnt6000.pth` to `chkpnt6200.pth` with candidate-local ROI loss, motion-aware densify boost, and a small 620k point budget cap.
- Added Slurm helper `scripts/submit_event_candidate_refine.sh` for train/eval submission. Training derives route0 checkpoints from `refine-logs/hide_reveal_real_windows.json`; eval consumes the train submit manifest and renders `test/ours_6200`.
- Local verification: bundled Python `py_compile` passed for `utils/motion_prior_utils.py` and `main.py`; cached whitespace check passed.
- Committed R021 plumbing as `86c1afc21da3948600b9b98f6e0c500c01f78dfc` (`Add event-candidate local refinement jobs`).

### 2026-07-07T03:34:00+02:00 - R021 train jobs submitted

- Fixed executable bit for `scripts/submit_event_candidate_refine.sh` and pushed commit `5a3ded1d338dc7534317434907835c0de4da0e73`.
- Pulled R021 on Leonardo and ran `bash -n scripts/submit_event_candidate_refine.sh`.
- Dry-run succeeded for three train jobs, each resuming the correct route0 `chkpnt6000.pth` and writing under `/leonardo_scratch/fast/EUHPC_D21_034/proj_adags/runs/event_candidate_local_refine_6200/`.
- Submitted train jobs:
  - `cut_roasted_beef`: `48764715`
  - `flame_steak`: `48764716`
  - `sear_steak`: `48764718`
- Train submit manifest: `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/refine-logs/event_candidate_refine_train_jobs_20260707_033133.tsv`.
- Initial poll showed all three jobs running on compute nodes; stdout startup logs showed config `configs/n3v/event_candidate_local_refine_6200.yaml`, commit `5a3ded1d338dc7534317434907835c0de4da0e73`, correct checkpoints, and normal PyTorch extension startup.
- Monitoring blocker: subsequent SSH attempts to `siyengar@login.leonardo.cineca.it` failed with `Permission denied (publickey,gssapi-keyex,gssapi-with-mic)` three times. Jobs are already submitted; eval/scoring remains pending until SSH access returns.

### 2026-07-07T11:14:00+02:00 - R021 timed out; R022 replacement submitted

- SSH access recovered. Slurm reported R021 train jobs `48764715`, `48764716`, and `48764718` as `TIMEOUT`, elapsed `00:55:14`, no `chkpnt6200.pth` files written.
- R021 logs showed all three jobs reached `Using /leonardo_work/.../build/torch_extensions as PyTorch extensions root...` and then timed out, matching the prior shared-extension lock/stall failure mode from R017.
- Collected R021 submit manifest and logs locally:
  - `refine-logs/event_candidate_refine_train_jobs_20260707_033133.tsv`
  - `logs/event_candidate_refine_train_cut_roasted_beef_48764715.{out,err}`
  - `logs/event_candidate_refine_train_flame_steak_48764716.{out,err}`
  - `logs/event_candidate_refine_train_sear_steak_48764718.{out,err}`
- Patched `scripts/run_leonardo.sh` to set per-job `TORCH_EXTENSIONS_DIR=$PROJECT_ROOT/build/torch_extensions_jobs/$SLURM_JOB_ID`, `TORCH_CUDA_ARCH_LIST=8.0`, and `MAX_JOBS`.
- Added and pushed manifest augmentation helper `scripts/run_hide_reveal_poc.py augment-real-manifest-system` in commit `657f0cd201dda6c84c0b4442e53380e4d837f1ad`; pulled it on Leonardo and verified the help command.
- Submitted replacement train jobs with `TIME=02:00:00`:
  - `cut_roasted_beef`: `48796168`
  - `flame_steak`: `48796170`
  - `sear_steak`: `48796174`
- Replacement train manifest collected locally: `refine-logs/event_candidate_refine_train_jobs_20260707_110953.tsv`.
- Current R022 scheduler state at last poll: all three jobs `PENDING` on `(Priority)`.

### 2026-07-07T11:27:49+02:00 - R022 still pending

- Pushed R021/R022 evidence checkpoint `345e743cbb86a329a1877f658e7f4e1d7833463a` (`Record R021 timeout and R022 submission`).
- Leonardo `squeue` and `sacct` poll reported R022 replacement train jobs `48796168`, `48796170`, and `48796174` all `PENDING`, elapsed `00:00:00`, reason `(Priority)`, no nodes assigned.
- Next action remains to monitor those jobs; if they complete and write `chkpnt6200.pth`, submit eval with `scripts/submit_event_candidate_refine.sh --mode eval --run-manifest refine-logs/event_candidate_refine_train_jobs_20260707_110953.tsv`.

### 2026-07-07T11:39:50+02:00 - R022 scheduler wait continues

- Pushed status breadcrumb `9224dc8bbab8dd9251598f66f74cad67d9d203da` (`Record R022 pending poll`).
- Additional Leonardo polls showed R022 train jobs `48796168`, `48796170`, and `48796174` still `PENDING`, reason `(Priority)`, elapsed `00:00:00`; no `logs/event_candidate_refine_train_*_487961*.out` files exist yet.
- `scontrol show job 48796168` showed no dependency, correct account/QOS `euhpc_d21_034` / `boost_qos_lprod`, partition `boost_usr_prod`, one GPU requested, and priority wait as the only scheduler reason.
- Next command: `ssh -o ConnectTimeout=15 siyengar@login.leonardo.cineca.it "squeue -j 48796168,48796170,48796174 -o '%i %T %M %D %R'"`.

### 2026-07-07T11:49:43+02:00 - R022 cancelled; R023 submitted with forced per-job extension roots

- R022 jobs `48796168`, `48796170`, and `48796174` started, but stdout showed `torch_extensions_dir: /leonardo_work/EUHPC_D21_034/proj_adags/build/torch_extensions`, proving the previous default was overridden by `leonardo_env.sh` and the jobs were still using the shared extension root.
- Cancelled R022 with `scancel` after roughly 2-3 minutes; Slurm accounting reports `CANCELLED by 132193`.
- Collected R022 startup logs locally:
  - `logs/event_candidate_refine_train_cut_roasted_beef_48796168.{out,err}`
  - `logs/event_candidate_refine_train_flame_steak_48796170.{out,err}`
  - `logs/event_candidate_refine_train_sear_steak_48796174.{out,err}`
- Patched and pushed `scripts/run_leonardo.sh` in commit `ad637f3ec50129fe40c5715804d446dfe6bdc90d` so Slurm jobs force `TORCH_EXTENSIONS_DIR=$PROJECT_ROOT/build/torch_extensions_jobs/$SLURM_JOB_ID` unless `ADAGS_TORCH_EXTENSIONS_DIR` is explicitly set.
- Pulled `ad637f3ec50129fe40c5715804d446dfe6bdc90d` on Leonardo and verified `bash -n scripts/run_leonardo.sh scripts/submit_event_candidate_refine.sh`.
- Submitted R023 replacement train jobs with `TIME=02:00:00` and login-side `ADAGS_ENV_SCRIPT=/dev/null` to avoid slow env setup during manifest parsing:
  - `cut_roasted_beef`: `48799988`
  - `flame_steak`: `48799992`
  - `sear_steak`: `48799995`
- Replacement train manifest collected locally: `refine-logs/event_candidate_refine_train_jobs_20260707_114908.tsv`.
- Current R023 scheduler state at submission poll: all three jobs `PENDING` on `(Priority)`.

### 2026-07-07T11:54:57+02:00 - R023 still pending after first start estimate

- Pushed R022/R023 evidence checkpoint `9840830abe3124525413770b31fe9ef3120e196e` (`Record R022 cancellation and R023 submission`).
- Leonardo polls showed R023 train jobs `48799988`, `48799992`, and `48799995` still `PENDING`, reason `(Priority)`, elapsed `00:00:00`; no `logs/event_candidate_refine_train_*_487999*.out` files exist yet.
- `scontrol show job 48799988` confirmed no dependency, partition `boost_usr_prod`, account/QOS `euhpc_d21_034` / `boost_qos_lprod`, one GPU requested, and priority wait as the scheduler reason.
- Next command: `ssh -o ConnectTimeout=15 siyengar@login.leonardo.cineca.it "squeue -j 48799988,48799992,48799995 -o '%i %T %M %D %R'"`.

### 2026-07-07T12:11:10+02:00 - R023 completed; R024 eval submitted

- R023 train jobs completed:
  - `48799988` cut_roasted_beef: `COMPLETED`, elapsed `00:13:45`, node `lrdn1933`
  - `48799992` flame_steak: `COMPLETED`, elapsed `00:13:38`, node `lrdn1930`
  - `48799995` sear_steak: `COMPLETED`, elapsed `00:13:00`, node `lrdn2003`
- Startup logs confirm forced per-job extension roots:
  - `/leonardo_work/EUHPC_D21_034/proj_adags/build/torch_extensions_jobs/48799988`
  - `/leonardo_work/EUHPC_D21_034/proj_adags/build/torch_extensions_jobs/48799992`
  - `/leonardo_work/EUHPC_D21_034/proj_adags/build/torch_extensions_jobs/48799995`
- Checkpoints and point clouds exist:
  - `/leonardo_scratch/fast/EUHPC_D21_034/proj_adags/runs/event_candidate_local_refine_6200/20260707_114908_cut_roasted_beef_event_candidate_local_refine_6200/chkpnt6200.pth`
  - `/leonardo_scratch/fast/EUHPC_D21_034/proj_adags/runs/event_candidate_local_refine_6200/20260707_114908_flame_steak_event_candidate_local_refine_6200/chkpnt6200.pth`
  - `/leonardo_scratch/fast/EUHPC_D21_034/proj_adags/runs/event_candidate_local_refine_6200/20260707_114908_sear_steak_event_candidate_local_refine_6200/chkpnt6200.pth`
- Collected R023 train logs locally under `logs/event_candidate_refine_train_*_487999*.{out,err}`.
- Eval dry-run resolved the three `chkpnt6200.pth` files, then R024 eval jobs were submitted:
  - `cut_roasted_beef`: `48802355`
  - `flame_steak`: `48802357`
  - `sear_steak`: `48802359`
- Eval manifest collected locally: `refine-logs/event_candidate_refine_eval_jobs_20260707_121022.tsv`.
- Current R024 scheduler state at submission poll: all three jobs `PENDING` on `(Priority)`.

### 2026-07-07T02:18:00+02:00 - R017 completed

- Implemented checkpoint-backed `actual-real-renders` for frozen R009 windows with runtime opacity gating in the Gaussian renderer; no GT crop compositing.
- Pushed implementation/support commits through `efae3edea355f158f6ab3c827a9694d6f3453a64`.
- Scheduler fixes required explicit `--gres=gpu:1`, default `TORCH_CUDA_ARCH_LIST=8.0`, and per-job `TORCH_EXTENSIONS_DIR` to avoid stale PyTorch extension locks.
- Render job `48760029` generated `refine-logs/hide_reveal_poc/r017_actual_real_renders/actual_real_windows_manifest.json`, `actual_render_metadata.json`, and validation `ok=true`, then failed only during optional LPIPS sidecar download because the compute node had no outbound network.
- Eval-only job `48760448` completed with LPIPS disabled and wrote `refine-logs/hide_reveal_poc/r017_actual_real_eval/real_event_window_metrics.csv`, `real_event_window_summary.json`, and `real_event_window_report.md`.
- R017 decision: FAIL. Actual checkpoint-backed hide/reveal passed 0/5 strict gates; mean PSNR/L1/flicker/static-ghost all worsened versus route0 and the real baselines.
- Durable result files added: `refine-logs/hide_reveal_poc/r017_actual_method_report.md` and `research-wiki/experiments/r017-actual-method-real-window-check.md`.

### 2026-07-06T02:07:53+02:00 - Initialization

- Read goal objective from `C:\Users\Sudarshan\.codex\attachments\ed75f206-df9c-4390-aa3e-2e3c0c1f66b4\goal-objective.md`.
- Read ARIS skills: `experiment-queue`, `monitor-experiment`, and `run-experiment`.
- Ran `git status --short --branch`, `git rev-parse --abbrev-ref HEAD`, and `git rev-parse HEAD` before changing files.
- Active branch: `codex/hide-reveal-poc-implementation`.
- Active commit: `a00bfd9c9aebb1889f7b026413c0910b430d7fe3`.
- Dirty/untracked state at start:
  - `.obsidian/`
  - `AGENTS.md`
  - `Untitled.canvas`
  - `configs/n3v/bootstrap.yaml`
  - `det_con.yaml`
  - `follow-up.md`
  - `idea-stage/`
  - `requirements.txt`
  - `verify_mask.jpg`
  - `verify_masked_flow.jpg`
  - `verify_raw_flow.jpg`
- `refine-logs/OVERNIGHT_STATUS.md` and `refine-logs/OVERNIGHT_RUNBOOK.md` were missing locally and are being created for durable recovery.

### 2026-07-06T02:25:39+02:00 - R009 frozen manifest commit

- Committed R009 frozen real-window manifest and visual audit evidence as `5914743f84c3ff4bec0b893f06e8557742a5348c` (`Freeze R009 real window manifest`).
- Pushed `codex/hide-reveal-poc-implementation` to origin: `a00bfd9..5914743`.
- R009 decision remains PASS; next milestone is R010 route0 smooth-transport evaluation on the frozen manifest.

### 2026-07-06T02:27:55+02:00 - HPC sync before R010

- Pushed bookkeeping commit `3be77ceeef57fabcf2428e802c9becbe59bf1da2` (`Record R009 push status`).
- On HPC, moved the untracked auto-sampled placeholder manifest from `refine-logs/hide_reveal_real_windows.json` to `refine-logs/hide_reveal_poc/r009_autosampled_manifest_20260705_223347.pre_frozen_backup.json` before pulling.
- Fast-forwarded HPC repo from `e211e418e749de7e5f503d41197f87e2c0ec391b` to `3be77ceeef57fabcf2428e802c9becbe59bf1da2`.
- Next R010 submission command: `scripts/submit_hide_reveal_poc.sh --stage real --manifest refine-logs/hide_reveal_real_windows.json --out-dir refine-logs/hide_reveal_poc/r010_route0_real_eval`.

### 2026-07-06T02:29:04+02:00 - R010 submitted

- Pushed pre-submit checkpoint `81b57e3f4714723f1b639fe5928ae19511424eae` (`Record R010 pre-submit state`) and pulled it on HPC.
- Submitted R010 route0 real evaluation with: `scripts/submit_hide_reveal_poc.sh --stage real --manifest refine-logs/hide_reveal_real_windows.json --out-dir refine-logs/hide_reveal_poc/r010_route0_real_eval`.
- Slurm job ID: `48653179`.
- Submit manifest: `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/refine-logs/hide_reveal_poc_real_jobs_20260706_022851.tsv`.
- Expected stdout: `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/logs/hide_reveal_real_48653179.out`.
- Expected stderr: `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/logs/hide_reveal_real_48653179.err`.
- Output dir: `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/refine-logs/hide_reveal_poc/r010_route0_real_eval`.
- First poll: `squeue` reports `PD` / `Priority`; `sacct` reports `PENDING`, `ExitCode=0:0`, `Elapsed=00:00:00`.

### 2026-07-06T02:32:18+02:00 - R010 completed

- Slurm job `48653179` completed with `State=COMPLETED`, `ExitCode=0:0`, `Elapsed=00:00:58`, `NodeList=lrdn0070`.
- Job stdout reported `Wrote real event-window outputs to /leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/refine-logs/hide_reveal_poc/r010_route0_real_eval` and `systems=route0`.
- The first output directory had a trailing carriage return from a PowerShell-to-SSH script argument; renamed generated directory to `refine-logs/hide_reveal_poc/r010_route0_real_eval` before collecting metrics.
- Collected outputs locally:
  - `refine-logs/hide_reveal_poc/r010_route0_real_eval/job_metadata.txt`
  - `refine-logs/hide_reveal_poc/r010_route0_real_eval/real_event_window_metrics.csv`
  - `refine-logs/hide_reveal_poc/r010_route0_real_eval/real_event_window_report.md`
  - `refine-logs/hide_reveal_poc/r010_route0_real_eval/real_event_window_summary.json`
  - `refine-logs/hide_reveal_poc_real_jobs_20260706_022851.tsv`
- Route0 summary over five frozen windows: mean PSNR `30.50211919273412`, mean L1/proxy-LPIPS `0.014831560850143432`, mean flicker `0.007990826107561588`, mean static ghost `0.12733273804187775`.
- Learned LPIPS was not computed (`lpips=null`); confident-track identity switches are not inferred by the current evaluator because no track/confidence sidecar was found during R009 discovery.

### 2026-07-06T02:34:55+02:00 - R011 prep snapshot

- Ran `git status --short --branch`, `git rev-parse --abbrev-ref HEAD`, and `git rev-parse HEAD` before creating R011 files.
- Active branch: `codex/hide-reveal-poc-implementation`.
- Active commit: `f078860c3daaf1560b926d92739687988fbe3f27`.
- Dirty/untracked state remains unrelated user/project files only:
  - `.obsidian/`
  - `AGENTS.md`
  - `Untitled.canvas`
  - `configs/n3v/bootstrap.yaml`
  - `det_con.yaml`
  - `follow-up.md`
  - `idea-stage/`
  - `requirements.txt`
  - `verify_mask.jpg`
  - `verify_masked_flow.jpg`
  - `verify_raw_flow.jpg`
- R011 candidate baseline source selected for evaluation: existing `fixed_budget_lora_route0_filemask_residual_600k` eval folders for `cut_roasted_beef`, `flame_steak`, and `sear_steak`; each has 300 `renders`, 300 `gt`, 300 `static`, and 300 `dynamic` frames at 676x507.
- Created `refine-logs/hide_reveal_poc/r011_residual_uncertainty_manifest.json` by augmenting the frozen R009 manifest with `residual_uncertainty` system paths.
- Validation: copied manifest to `/leonardo_work/EUHPC_D21_034/proj_adags/tmp/r011_residual_uncertainty_manifest_validation_20260706.json` and ran `python scripts/run_hide_reveal_poc.py validate-real-manifest --manifest ... --require-system route0 --require-system residual_uncertainty` after sourcing `exp_index/leonardo_env.sh`; result `validation_ok=True`, `windows=5`, `errors=0`, `warnings=0`.

### 2026-07-06T02:40:59+02:00 - R011 completed

- Committed and pushed R011 manifest as `d079e2f134641d0beec78a15fe86305078917433` (`Prepare R011 residual baseline manifest`), then pulled it on HPC.
- Submitted R011 with `scripts/submit_hide_reveal_poc.sh --stage real --manifest refine-logs/hide_reveal_poc/r011_residual_uncertainty_manifest.json --out-dir refine-logs/hide_reveal_poc/r011_residual_uncertainty_real_eval`.
- Slurm job `48653948` completed with `State=COMPLETED`, `ExitCode=0:0`, `Elapsed=00:01:02`, `NodeList=lrdn0071`.
- Submit manifest: `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/refine-logs/hide_reveal_poc_real_jobs_20260706_023817.tsv`.
- Logs:
  - `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/logs/hide_reveal_real_48653948.out`
  - `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/logs/hide_reveal_real_48653948.err`
- Collected outputs locally:
  - `refine-logs/hide_reveal_poc/r011_residual_uncertainty_real_eval/job_metadata.txt`
  - `refine-logs/hide_reveal_poc/r011_residual_uncertainty_real_eval/real_event_window_metrics.csv`
  - `refine-logs/hide_reveal_poc/r011_residual_uncertainty_real_eval/real_event_window_report.md`
  - `refine-logs/hide_reveal_poc/r011_residual_uncertainty_real_eval/real_event_window_summary.json`
  - `refine-logs/hide_reveal_poc_real_jobs_20260706_023817.tsv`
- Residual/uncertainty summary over five frozen windows: mean PSNR `30.073395341209725`, mean L1/proxy-LPIPS `0.01657234113663435`, mean flicker `0.008039021119475364`, mean static ghost `0.1457015424966812`.
- Against the paired route0 rows in the same R011 output, residual-minus-route0 deltas were PSNR `-0.4287238515243956`, L1 `+0.0017407802864909186`, flicker `+0.0000481950119137764`, static ghost `+0.01836880445480346`.
- Learned LPIPS was not computed (`lpips=null`); confident-track identity switches remain unavailable in the current evaluator.

### 2026-07-06T02:42:09+02:00 - R012/R013 pre-submit snapshot

- Ran `git status --short --branch` and `git rev-parse HEAD` before submitting the derived matched-lifespan/full hide-reveal job.
- Active branch: `codex/hide-reveal-poc-implementation`.
- Active commit: `da1d71b635618330d330fa94789bc293d73e06e7`.
- Dirty/untracked state remains unrelated user/project files only:
  - `.obsidian/`
  - `AGENTS.md`
  - `Untitled.canvas`
  - `configs/n3v/bootstrap.yaml`
  - `det_con.yaml`
  - `follow-up.md`
  - `idea-stage/`
  - `requirements.txt`
  - `verify_mask.jpg`
  - `verify_masked_flow.jpg`
  - `verify_raw_flow.jpg`
- Pulled HPC repo to `da1d71b635618330d330fa94789bc293d73e06e7`.
- Dry-run command succeeded for: `scripts/submit_hide_reveal_poc.sh --stage derive-real-renders --manifest refine-logs/hide_reveal_real_windows.json --out-dir refine-logs/hide_reveal_poc/r012_r013_derived_real_renders --eval-out-dir refine-logs/hide_reveal_poc/r012_r013_derived_real_eval --overwrite`.
- R012/R013 caveat: this job creates derived image-level PoC render folders from route0 and the frozen manifest. It is not a retrained Gaussian checkpoint; `derived_poc_metadata.json` must be used to preserve that limitation.

### 2026-07-06T02:45:15+02:00 - R012/R013 completed; R014 skipped

- Committed and pushed pre-submit status as `840907a232a2a08df09b480fbdfae52d71aba5cd` (`Record R012 R013 pre-submit state`), then pulled it on HPC.
- Submitted R012/R013 with `scripts/submit_hide_reveal_poc.sh --stage derive-real-renders --manifest refine-logs/hide_reveal_real_windows.json --out-dir refine-logs/hide_reveal_poc/r012_r013_derived_real_renders --eval-out-dir refine-logs/hide_reveal_poc/r012_r013_derived_real_eval --overwrite`.
- Slurm job `48654171` completed with `State=COMPLETED`, `ExitCode=0:0`, `Elapsed=00:00:27`, `NodeList=lrdn0070`.
- Submit manifest: `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/refine-logs/hide_reveal_poc_derive-real-renders_jobs_20260706_024307.tsv`.
- Logs:
  - `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/logs/hide_reveal_derive-real-renders_48654171.out`
  - `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/logs/hide_reveal_derive-real-renders_48654171.err`
- Derived-render metadata:
  - `refine-logs/hide_reveal_poc/r012_r013_derived_real_renders/derived_poc_metadata.json`
  - `refine-logs/hide_reveal_poc/r012_r013_derived_real_renders/derived_real_windows_manifest.json`
  - `refine-logs/hide_reveal_poc/r012_r013_derived_real_renders/derived_real_windows_validation.json`
  - `derived_poc_metadata.json` records `is_trained_model_output=false`.
- Derived render folders remain on HPC under:
  - `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/refine-logs/hide_reveal_poc/r012_r013_derived_real_renders/derived_renders/matched_lifespan/`
  - `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/refine-logs/hide_reveal_poc/r012_r013_derived_real_renders/derived_renders/hide_reveal/`
- Collected eval outputs locally:
  - `refine-logs/hide_reveal_poc/r012_r013_derived_real_eval/real_event_window_metrics.csv`
  - `refine-logs/hide_reveal_poc/r012_r013_derived_real_eval/real_event_window_report.md`
  - `refine-logs/hide_reveal_poc/r012_r013_derived_real_eval/real_event_window_summary.json`
  - `refine-logs/hide_reveal_poc_derive-real-renders_jobs_20260706_024307.tsv`
- Matched-lifespan summary over five windows: mean PSNR `29.818134359321654`, mean L1/proxy-LPIPS `0.016354632563889027`, mean flicker `0.007956006657332182`, mean static ghost `0.12733273804187775`.
- Matched-lifespan minus route0 deltas: PSNR `-0.6839848334124667`, L1 `+0.0015230717137455947`, flicker `-0.00003481945022940601`, static ghost `0.0`.
- Derived hide/reveal summary over five windows: mean PSNR `41.714903733552326`, mean L1/proxy-LPIPS `0.0026653554756194352`, mean flicker `0.001685856096446514`, mean static ghost `0.12733273804187775`.
- Derived hide/reveal minus route0 deltas: PSNR `+11.212784540818205`, L1 `-0.012166205374523996`, flicker `-0.006304970011115073`, static ghost `0.0`.
- R014 decision: SKIP. The conditional equal-budget refinement diagnostic is not triggered because R012/R013 produced complete comparable derived outputs without noisy or unstable scoring, and the PoC policy says not to expand compute unless needed.

### 2026-07-06T02:53:47+02:00 - R015 completed

- Generated R015 summary artifacts in `refine-logs/hide_reveal_poc/r015_poc_summary/`:
  - `poc_table.md`
  - `poc_table.csv`
  - `poc_decision_inputs.json`
  - `crop_strip_manifest.json`
  - `crop_strips/*.jpg`
- Fetched only the 60 selected source frames needed for strips from Leonardo through a temporary tarball, then removed the transient local and remote source-frame caches.
- Synthetic heldout gate summary: n=40, candidate recall 1.000, margin AUC 1.000, accepted precision/recall 1.000/1.000, identity reconnection 1.000, matched-lifespan identity reconnection 0.000, false event rate 0.000.
- Real-window summary: route0 mean PSNR 30.5021 and L1/proxy-LPIPS 0.01483; matched-lifespan is worse than route0 on PSNR/L1; residual/uncertainty is worse than route0 on PSNR/L1/static ghost; derived hide/reveal upper-bound improves PSNR to 41.7149 and L1/proxy-LPIPS to 0.00267.
- Caveat for R016: `derived_poc_metadata.json` marks `is_trained_model_output=false`; hide/reveal real rows are GT-crop upper-bound composites, not trained Gaussian checkpoint outputs.

### 2026-07-06T02:56:29+02:00 - R016 completed

- Wrote `refine-logs/hide_reveal_poc/r016_go_no_go_memo.md`.
- Decision: NO-GO for paper-scale validation now.
- C1 outcome: PASS on synthetic heldout margin/candidate/precision evidence.
- C2 outcome: PASS on synthetic identity evidence versus matched-lifespan/no-identity variants.
- C3 outcome: NOT PASSED for the actual method. Real windows are directionally positive only for the derived GT-crop upper-bound; no trained Gaussian/checkpoint-backed hide/reveal output, learned LPIPS, or confident-track ID switch evidence is available.
- Allowed future work: one narrow actual-method check on the same frozen R009 windows. Broad baselines remain deferred.

### 2026-07-07T01:24:38+02:00 - R017 started

- User asked whether the perfect synthetic heldout gate is believable; interpretation: believable as a controlled fixture/smoke check, not as generalization evidence.
- R017 objective: render actual checkpoint-backed hide/reveal outputs on the same five frozen R009 windows without GT crop compositing.
- Implementation approach: add an inert-by-default renderer hook that attenuates projected dynamic Gaussian opacity inside predeclared event crops during frozen event frames, then export/evaluate `actual_hide_reveal` against route0, matched-lifespan, and residual/uncertainty baselines.
- Pass rule remains strict: majority of five real windows must improve over all three baselines without static ghost degradation. No paper-scale baselines unless this passes.

## Milestone Decisions

- R009: PASS. Frozen 2026-07-06T00:20:08Z in `refine-logs/hide_reveal_real_windows.json` with five pre-scoring windows:
  - `cut_roasted_beef_hand_tongs_meat_095_110`
  - `cut_roasted_beef_hand_knife_meat_140_155`
  - `flame_steak_torch_pan_155_170`
  - `flame_steak_torch_sweep_195_210`
  - `sear_steak_spoon_pan_220_235`
  - Evidence: `refine-logs/hide_reveal_poc/r009_visual_audit/selection_notes.md` plus contact/crop sheets.
  - Data availability: route0 renders/gt/static/dynamic complete for selected scenes; masks and flow sidecars found; track-confidence sidecars not found and recorded as unavailable.
  - Validation: copied manifest to `/leonardo_work/EUHPC_D21_034/proj_adags/tmp/r009_frozen_manifest_validation_20260706.json` and ran `python scripts/run_hide_reveal_poc.py validate-real-manifest --manifest ... --require-system route0` after sourcing `exp_index/leonardo_env.sh`; result `validation_ok=True`, `windows=5`, `errors=0`, `warnings=0`.
- R010: PASS. Route0 smooth-transport baseline generated by Slurm job `48653179`; outputs in `refine-logs/hide_reveal_poc/r010_route0_real_eval/`; summary metrics listed above.
- R011: PASS. Residual/uncertainty baseline generated by Slurm job `48653948`; outputs in `refine-logs/hide_reveal_poc/r011_residual_uncertainty_real_eval/`; summary metrics and route0 deltas listed above.
- R012: PASS. Matched-lifespan derived image-level baseline generated by Slurm job `48654171`; outputs in `refine-logs/hide_reveal_poc/r012_r013_derived_real_eval/`; render metadata in `refine-logs/hide_reveal_poc/r012_r013_derived_real_renders/`.
- R013: PASS. Derived image-level hide/reveal PoC output generated by Slurm job `48654171`; caveat `is_trained_model_output=false` recorded in metadata.
- R014: SKIP. Conditional equal-budget refinement diagnostic not triggered.
- R015: PASS. Generated `refine-logs/hide_reveal_poc/r015_poc_summary/poc_table.md`, `poc_table.csv`, `poc_decision_inputs.json`, `crop_strip_manifest.json`, and five qualitative crop strips under `crop_strips/`.
- R016: FAIL / NO-GO for paper-scale validation. Memo: `refine-logs/hide_reveal_poc/r016_go_no_go_memo.md`.
- R017: FAIL. Actual checkpoint-backed opacity gate passed 0/5 frozen windows. Report: `refine-logs/hide_reveal_poc/r017_actual_method_report.md`.
