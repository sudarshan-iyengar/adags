# Overnight Status

## Fresh Snapshot - 2026-07-10T02:07:43+02:00

- Current objective phase: R034+ training-loop visibility-state method planning and implementation.
- Current run ID: R034 synthetic matched-hypothesis visibility fixture, then R035/R036 real pilot only if R034 passes.
- Current method candidate: non-oracle event-candidate field plus matched-budget `H_smooth` versus `H_event`, where `H_event` keeps persistent Gaussian state and learns/applies a time-dependent opacity visibility gate before rasterization.
- Current branch: `codex/hide-reveal-poc-implementation`
- Current local commit: `f10071f4f676cdf60b9989d7ea0bfe7af7df6ae7`
- Current local status command: `git status --short --branch`
- Current dirty/untracked state before this 2026-07-10 orchestration edit:
  - tracked dirty:
    - `refine-logs/hide_reveal_poc/r015_poc_summary/poc_table.md`
    - `refine-logs/hide_reveal_poc/r016_go_no_go_memo.md`
  - untracked:
    - `.codex/`
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
- Preservation rule: these pre-existing tracked/untracked changes are not part of the R034+ method unless explicitly staged later after review.
- Latest inherited evidence: R031-R033 DA3 support pipeline is operational but weak as a hard support detector; R030 showed oracle crop support does not rescue the current posthoc micro-densification mechanism.
- Next command: inspect route0 training/rasterization code paths and implement the smallest synthetic `H_smooth`/`H_event` matched-budget visibility-gate fixture before touching real-scene training.
- Open blockers: none yet.

## Recovery Snapshot

- Current objective phase: depth-occlusion support setup in parallel with training-loop integration planning
- Current run ID: R031 depth-occlusion event support
- Current method candidate: DA3-based non-oracle depth support artifact; not a rendered-method success unless later training-loop/checkpoint-backed evidence passes
- Current branch: `codex/hide-reveal-poc-implementation`
- Current local commit at 2026-07-07 recovery start: `f5d43539aee500051f2a4c5eeca5420293b636f1`
- Current local commit at 2026-07-07T12:25:00+02:00: `1a747fae7079f7352c3103f51d735912fcedf10a`
- Last pushed milestone commit: `2a27ef0`
- Last HPC job ID: scoring job `48969825`
- Latest success/failure: R029/R030 scoring completed. R029 route0 continuation worsened route0, so generic continuation is not the source of R027's tiny gain. R030 oracle-support micro-densification failed with mean PSNR `29.9021`, mean L1 `0.0158770`, route0 PSNR+L1 wins `0/5`, and negative oracle recovery.
- Next command to run: implement resumable R031 DA3 frame/depth/support tooling, commit, push, pull on HPC, clone/setup DA3 under `$WORK/proj_adags/repo/depth-anything-3`, then launch the first Slurm inference/support wave.
- Open blockers: none.

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

## Dirty State At 2026-07-07T12:25:00+02:00 Pre-R025 Edit

Recorded before collecting R024 logs and creating the R025 scoring manifest.

- Branch: `codex/hide-reveal-poc-implementation`
- HEAD: `1a747fae7079f7352c3103f51d735912fcedf10a`
- Tracked dirty files present before this edit:
  - `refine-logs/hide_reveal_poc/r015_poc_summary/poc_table.md`
  - `refine-logs/hide_reveal_poc/r016_go_no_go_memo.md`
- Untracked files/directories present before this edit:
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

### 2026-07-07T12:25:00+02:00 - R024 eval completed

- Slurm reported all three R024 eval jobs completed with `ExitCode=0:0`:
  - `48802355` cut_roasted_beef: `COMPLETED`, elapsed `00:08:27`, node `lrdn1670`
  - `48802357` flame_steak: `COMPLETED`, elapsed `00:08:52`, node `lrdn0873`
  - `48802359` sear_steak: `COMPLETED`, elapsed `00:08:36`, node `lrdn2343`
- Eval logs show checkpoint loading from the R023 `chkpnt6200.pth` files and export of `test/ours_6200` outputs.
- Next command: collect `logs/event_candidate_refine_eval_*_488023*.{out,err}`, build `refine-logs/hide_reveal_poc/r025_event_candidate_refine_manifest.json`, validate it, and submit frozen-window scoring.

### 2026-07-07T12:29:00+02:00 - R025 scoring pre-submit

- Collected R024 eval logs locally under `logs/event_candidate_refine_eval_*_488023*.{out,err}`.
- Verified each R024 eval folder contains 300 `renders`, 300 `gt`, 300 `static`, and 300 `dynamic` frames under `test/ours_6200`.
- Built remote manifest `refine-logs/hide_reveal_poc/r025_event_candidate_refine_manifest.json` with `system_name=event_candidate_refine` from eval root `/leonardo_scratch/fast/EUHPC_D21_034/proj_adags/runs/event_candidate_local_refine_6200`.
- Strict validation passed with `validation_ok=True`, `windows=5`, `errors=0`, `warnings=0` for required systems `route0`, `event_candidate_refine`, `residual_uncertainty`, and `matched_lifespan`.
- Dry-run command succeeded:
  `scripts/submit_hide_reveal_poc.sh --stage real --manifest refine-logs/hide_reveal_poc/r025_event_candidate_refine_manifest.json --out-dir refine-logs/hide_reveal_poc/r025_event_candidate_refine_real_eval --dry-run`
- Submitted R025 scoring job `48805053`.
- Submit manifest on Leonardo: `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/refine-logs/hide_reveal_poc_real_jobs_20260707_122508.tsv`.
- Next command: monitor Slurm job `48805053`, then collect logs and scoring outputs.

### 2026-07-07T12:37:00+02:00 - R025 completed; M1 claim rejected

- Slurm job `48805053` completed with `State=COMPLETED`, `ExitCode=0:0`, elapsed `00:01:02`, node `lrdn1214`.
- Collected locally:
  - `logs/hide_reveal_real_48805053.{out,err}`
  - `refine-logs/hide_reveal_poc_real_jobs_20260707_122508.tsv`
  - `refine-logs/hide_reveal_poc/r025_event_candidate_refine_manifest.json`
  - `refine-logs/hide_reveal_poc/r025_event_candidate_refine_real_eval/real_event_window_summary.json`
  - `refine-logs/hide_reveal_poc/r025_event_candidate_refine_real_eval/real_event_window_metrics.csv`
  - `refine-logs/hide_reveal_poc/r025_event_candidate_refine_real_eval/real_event_window_report.md`
  - `refine-logs/hide_reveal_poc/r025_event_candidate_refine_summary/crop_strips/*.jpg`
- Raw R025 means: event_candidate_refine PSNR `28.9393`, L1/proxy-LPIPS `0.0188750`, flicker `0.00847709`, static ghost `0.125652`.
- Route0 means: PSNR `30.5021`, L1/proxy-LPIPS `0.0148316`, flicker `0.00799083`, static ghost `0.127333`.
- Gate counts: 0/5 windows improved over all three baselines on both PSNR and L1; 0/5 improved over route0 on both PSNR and L1; 2/5 were no worse on static ghost.
- Oracle recovery fractions: PSNR `-0.1394`, L1 `-0.3323`.
- Result-to-claim reviewer verdict: `claim_supported: no`, confidence `high`.
- Decision: FAIL for M1. The method produced valid checkpoint-backed Gaussian renders without GT crop compositing, but did not recover the oracle event-crop fix.
- Pushed evidence commit: `69a877a897eb239dcb236be67542c641e5ae38aa` (`Record R025 event candidate refine failure`).
- Next scientific step, if continuing: change mechanism rather than tuning M1 on the frozen windows. The most defensible next candidate is M2 occlusion-boundary gated micro-densification or a detector/optimization decomposition that separates candidate support quality from refinement damage.

### 2026-07-07T23:24:44+02:00 - M2 predeclared; R026a support builder cancelled for performance

- Recovered from `refine-logs/M2_HANDOFF.md`, `EVENT_CROP_METHOD_TRACKER.md`, `OVERNIGHT_STATUS.md`, `EVENT_CROP_FIX_EVIDENCE.md`, `research-wiki/event-crop-fix.md`, `query_pack.md`, `gap_map.md`, and relevant idea/experiment/paper/claim pages.
- Confirmed local branch `codex/hide-reveal-poc-implementation` at `f74ff87c9fad01a4d39d9574f9f9d1b48c0c41d2` before M2 edits, with the known unrelated dirty/untracked files preserved.
- Predeclared M2 in `refine-logs/EVENT_CROP_METHOD_TRACKER.md`: non-oracle dynamic-mask/flow/render-boundary support, no frozen crop support, R026 support artifact, target `chkpnt6400.pth`, point cap `625000`, and strict unchanged PASS/FAIL gates.
- Implemented and pushed commit `13297e290a8511512f4f3be08f7ceb425fcc3ca7` (`Add M2 boundary micro-densification plumbing`): added `event-boundary-support`, M2 support-mask loading in `MotionPriorCache`, residual-free motion-aware densify option, generic submit labels, and `configs/n3v/event_boundary_micro_densify_6400.yaml`.
- Verification before remote submission: bundled local Python `py_compile` passed, CLI help passed, synthetic support smoke passed with `validation_ok=True`; Leonardo `bash -n` and `py_compile` passed after fast-forwarding the remote repo to `13297e2`.
- Remote pull required moving conflicting untracked files listed by Git into `logs/pre_pull_untracked_backup_20260707_232249_m2/`; no tracked remote/user files were reverted.
- Submitted R026a support job `48872013` with output `refine-logs/hide_reveal_poc/r026_m2_boundary_support` and logs `logs/hide_reveal_event-boundary-support_48872013.{out,err}`.
- R026a was cancelled after about 9 minutes on `lrdn0033` because it had produced only `job_metadata.txt`. Diagnosis: the first pure-Python connected-component pass was too slow for full-scene sidecars.
- Current next action: push the tile-capped support-box fix, pull it on Leonardo, rerun `bash -n`/`py_compile`, then submit R026b.

### 2026-07-07T23:52:00+02:00 - R026b support passed; R027 train submitted

- Implemented and pushed commit `efca178e7fd5fef49aa84c762350dfa8cbdd36d8` (`Speed up M2 boundary support selection`): replaced the slow connected-component scan with tile-capped compact support boxes while preserving the non-oracle support contract and 3 percent pixel cap.
- Reran Leonardo `bash -n` and `py_compile` after fast-forwarding the HPC repo to `efca178e7fd5fef49aa84c762350dfa8cbdd36d8`.
- Submitted R026b support job `48872653`; it completed in `00:08:53` on `lrdn1861` with `ExitCode=0:0`.
- R026b outputs are local and remote at `refine-logs/hide_reveal_poc/r026_m2_boundary_support/`: `event_boundary_support_manifest.json`, `event_boundary_support_validation.json`, `event_boundary_support_report.md`, `event_boundary_support_components.csv`, support-mask PNGs, and job metadata.
- R026b validation: `ok=true`, errors `0`, warnings `0`, support frames `66`, selected components `108`, max support pixel fraction `0.005205`.
- R026b guardrails: `uses_gt_residual=false`, `uses_gt_crop_pixels=false`, `uses_frozen_window_labels=false`, and `source_manifest_usage=scene_sources_only_for_paths_and_frame_ranges`.
- Submitted R027 train jobs from route0 `chkpnt6000.pth` to target `chkpnt6400.pth` using `configs/n3v/event_boundary_micro_densify_6400.yaml`: `48873219` cut_roasted_beef, `48873220` flame_steak, `48873221` sear_steak.
- R027 train manifest: `refine-logs/event_boundary_micro_densify_train_jobs_20260707_234937.tsv`.
- Current next action: monitor R027 train completion, then submit eval and frozen-window scoring. Do not declare M2 PASS/FAIL until checkpoint-backed `test/ours_6400` renders have been scored on the frozen R009 windows.

### 2026-07-08T00:30:00+02:00 - R027 completed; M2 claim rejected

- R027 train jobs `48873219`, `48873220`, and `48873221` completed with `ExitCode=0:0`; all three wrote `chkpnt6400.pth` and `point_cloud/iteration_6400/point_cloud.ply`.
- R027 eval jobs `48874148`, `48874152`, and `48874155` completed with `ExitCode=0:0`; all three eval folders contain 300 `renders`, 300 `gt`, 300 `static`, and 300 `dynamic` frames under `test/ours_6400`.
- Built and validated `refine-logs/hide_reveal_poc/r027_event_boundary_micro_densify_manifest.json` with required systems `route0`, `event_boundary_micro_densify`, `residual_uncertainty`, and `matched_lifespan`; validation had `windows=5`, `errors=0`, `warnings=0`.
- R027 frozen-window scoring job `48874592` completed with `ExitCode=0:0`, elapsed `00:01:06`, and wrote `refine-logs/hide_reveal_poc/r027_event_boundary_micro_densify_real_eval/`.
- R027 result: FAIL. Mean PSNR `30.5591` vs route0 `30.5021` (`+0.0569 dB`), mean L1/proxy-LPIPS `0.0147412` vs route0 `0.0148316` (`-0.0000903`), mean flicker slightly worsened by `+0.0000195`, and mean static ghost improved by `-0.001698`.
- Gate counts: strict all-baseline PSNR+L1 wins `2/5`, route0 PSNR+L1 wins `3/5`, static no-worse `3/5`.
- Oracle recovery fractions: PSNR `0.00508`, L1 `0.00743`, far below the required `0.25`.
- Independent result-to-claim reviewer verdict: `claim_supported: no`, confidence `high`.
- Durable outputs: `refine-logs/hide_reveal_poc/r027_event_boundary_micro_densify_decision_memo.md`, `refine-logs/hide_reveal_poc/r027_event_boundary_micro_densify_summary/gate_decision.json`, crop strips under `r027_event_boundary_micro_densify_summary/crop_strips/`, and trace `.aris/traces/result-to-claim/2026-07-08_run01/`.
- Current next action: preserve the negative result as durable knowledge. Do not run paper-scale validation or claim-supporting ablations for M2.

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
- R018: FAIL. First non-oracle candidate detector validated structurally but covered 0/5 frozen windows in posthoc overlap audit.
- R019: PARTIAL. Motion-supported detector removed the top-band artifact but covered only 2/5 frozen windows.
- R020: SUPPORT POOL. High-recall detector produced 72 candidates and posthoc coverage of 3/5 windows, but it was not a rendered Gaussian method result.
- R021: FAIL. First candidate-local refinement training attempt timed out at shared PyTorch extension startup.
- R022: CANCELLED. Replacement jobs were stopped after logs showed the shared extension-root bug persisted.
- R023: PASS. Forced per-job extension roots allowed all three candidate-local refinement train jobs to complete and write `chkpnt6200.pth`.
- R024: PASS. Eval jobs rendered complete `test/ours_6200` folders for all three scenes.
- R025: FAIL. Checkpoint-backed non-oracle `event_candidate_refine` scoring passed 0/5 PSNR+L1 gates and worsened mean PSNR/L1 versus route0. Decision memo: `refine-logs/hide_reveal_poc/r025_event_candidate_refine_decision_memo.md`.

### 2026-07-08 - Interpretation and R028 support-overlap diagnostic

- Used result-to-claim and experiment-audit style bounded reviews.
- Result-to-claim verdict: current artifacts do not support the non-oracle recovery claim; this does not prove every future event/reveal method impossible.
- Integrity audit verdict: WARN. No fake-GT, self-normalization, or phantom results found; caveats are proxy L1 instead of learned LPIPS, five-window scope, R017 oracle support, and oracle comparison rows in R025/R027.
- Added `scripts/audit_event_support_overlap.py`.
- Ran R028 locally:
  - R020 candidates: mean support-frame fraction `0.6375`, mean crop coverage `0.491371`; one frozen window had zero support.
  - R026 boundary support: mean support-frame fraction `0.0250`, mean crop coverage `0.000000`.
- Interpretation update: R027 fails the R026/R027 recipe, but is not a clean rejection of good-support event-local micro-densification because R026 did not meaningfully cover the evaluated crops.
- Predeclared next compact wave:
  - R029 route0 `6000 -> 6400` continuation control, config `configs/n3v/route0_continue_6400_control.yaml`.
  - R030 oracle-support Gaussian-only diagnostic, config `configs/n3v/oracle_crop_support_micro_densify_6400.yaml`.

### 2026-07-08T15:45:00+02:00 - R029/R030 train wave submitted

- Pushed interpretation/diagnostic milestone commit `8ebb53d`.
- HPC checkout fast-forwarded to `8ebb53d`.
- Dry-run validation passed for both train configs using the route0 `chkpnt6000.pth` starts.
- R029 route0 continuation train jobs submitted:
  - `48935431` cut_roasted_beef
  - `48935450` flame_steak
  - `48935478` sear_steak
  - Manifest: `refine-logs/route0_continue_6400_train_jobs_20260708_152429.tsv`
- R030 oracle-support diagnostic train jobs submitted:
  - `48935580` cut_roasted_beef
  - `48935581` flame_steak
  - `48935583` sear_steak
  - Manifest: `refine-logs/oracle_crop_support_micro_densify_train_jobs_20260708_153750_manual.tsv`
- Duplicate cut submissions from timed-out submission attempts:
  - `48935560` excluded from manifest; `scancel 48935560` returned success.
  - `48935682` excluded from manifest; observed `JobState=CANCELLED`.
- Early logs showed both configs launching and compiling per-job CUDA extensions. No R029/R030 checkpoint/eval/scoring verdict yet.

### 2026-07-08T16:10:00+02:00 - R029/R030 train complete; eval blocked

- Valid train jobs completed with `ExitCode=0:0`:
  - R029 `48935431`, `48935450`, `48935478`
  - R030 `48935580`, `48935581`, `48935583`
- Checkpoints observed:
  - `/leonardo_scratch/fast/EUHPC_D21_034/proj_adags/runs/route0_continue_6400_control/20260708_152429_*_route0_continue_6400_control/chkpnt6400.pth`
  - `/leonardo_scratch/fast/EUHPC_D21_034/proj_adags/runs/oracle_crop_support_micro_densify_6400/20260708_153750_*_oracle_crop_support_micro_densify_6400/chkpnt6400.pth`
- Eval submission attempts:
  - R029 eval with default `boost_usr_prod`, account `euhpc_d21_034`, QoS `boost_qos_lprod`: BLOCKED, Slurm returned invalid account/partition.
  - R029 eval after loading project environment: BLOCKED, same error.
  - `sbatch --test-only` probes over `boost_usr_prod`, `dcgp_usr_prod`, and `dcgp_cmcc_prod` with available QoS values: BLOCKED, invalid account/partition or expired budget on non-boost partitions.
  - R029 eval with `QOS=boost_usr_prod`: BLOCKED, command timed out before writing job IDs; header-only manifest remained.
- Budget check: `saldo -b` reports account `EUHPC_D21_034` active `20260130` to `20260730`, total `144000` local h, consumed `63571` local h (`44.1%`). This does not look like project exhaustion.
- Current scientific status: no R029/R030 metric verdict. The next scientific step remains eval rendering and frozen-window scoring once Slurm accepts new jobs.

### 2026-07-09T00:35:00+02:00 - R029/R030 eval and scoring complete

- Retried eval with a lightweight login wrapper (`python` shim for manifest parsing; compute jobs still source `exp_index/leonardo_env.sh`).
- R029 eval jobs completed with `ExitCode=0:0`: `48969017`, `48969019`, `48969021`.
- R030 eval jobs completed with `ExitCode=0:0`: `48969090`, `48969092`, `48969093`.
- Validated all six eval folders: each has 300 `renders`, 300 `gt`, 300 `static`, and 300 `dynamic` frames under `test/ours_6400`.
- Built and validated combined manifest `refine-logs/hide_reveal_poc/r029_r030_disambiguation_manifest.json`.
- Scoring job `48969825` completed and wrote:
  - `refine-logs/hide_reveal_poc/r029_r030_disambiguation_real_eval/`
  - `refine-logs/hide_reveal_poc/r029_r030_disambiguation_summary/`
  - `refine-logs/hide_reveal_poc/r029_r030_disambiguation_decision_memo.md`
- R029 route0 continuation result: mean PSNR `30.3532`, mean L1 `0.0150603`, route0 PSNR+L1 wins `1/5`, static no-worse `1/5`, oracle recovery negative. Verdict: CONTROL COMPLETE; generic continuation does not explain R027's tiny positive movement.
- R030 oracle-support diagnostic result: mean PSNR `29.9021`, mean L1 `0.0158770`, route0 PSNR+L1 wins `0/5`, strict all-baseline wins `0/5`, static no-worse `4/5`, oracle recovery negative. Verdict: FAIL.
- Scientific conclusion: oracle-aligned support alone does not rescue the current posthoc Gaussian micro-densification recipe; the likely bottleneck is the posthoc optimization/capacity mechanism.

### 2026-07-09 - R031 depth-occlusion support predeclaration

- User selected training-loop integration as the next scientifically meaningful rendered-method direction and asked to add/refine depth occlusion support in parallel.
- Refined `research-wiki/ideas/depth-occlusion-event-support.md` to use Depth Anything 3.
- Predeclared R031 in `refine-logs/DEPTH_OCCLUSION_EVENT_SUPPORT_PLAN.md`.
- Primary depth model: `depth-anything/DA3NESTED-GIANT-LARGE-1.1`, with DA3 repo to live under `$WORK/proj_adags/repo/depth-anything-3`.
- R031 is a support-only diagnostic. It may PASS as a support artifact if valid compact masks improve posthoc frozen-window overlap versus R026, but it cannot claim event-crop repair without a later checkpoint-backed/training-loop rendered method.
- Guardrails: no GT residual, no GT crop pixels, no frozen R009 crop labels as support; frozen windows are posthoc diagnostic only.

### 2026-07-09T22:40:00+02:00 - R031-R033 DA3 depth support wave complete

- Commit state:
  - `14f3c83` added DA3 support tooling and predeclaration docs.
  - `dac940a` added `--fill-component-tiles` for the R033 sensitivity test.
- DA3 setup PASS:
  - Repo: `$WORK/proj_adags/repo/depth-anything-3`.
  - DA3 repo commit: `41736238f5bced4debf3f2a12375d2466874866d`.
  - Model snapshot: `$WORK/proj_adags/models/depth-anything/DA3NESTED-GIANT-LARGE-1.1`.
  - Environment repaired after first smoke failure: use local DA3 venv with local `numpy==1.26.4`, ADAGS `torch==2.5.1+cu121` / `torchvision==0.20.1+cu121`, and `xformers==0.0.28.post3`.
- Smoke PASS after repair:
  - Prepare job `49029452`: 6 frames, non-oracle flags false.
  - First infer job `49029467`: FAIL due missing undeclared `addict`.
  - Retry infer job `49029847`: PASS, wrote 6 depth sidecars.
  - Support job `49029955`: PASS, wrote 6 support masks, validation OK.
- R031 full PASS for depth sidecars, WEAK for support:
  - Prepare job `49030039`: 900 frames.
  - Inference job `49030185`: 900/900 depth sidecars, elapsed `00:05:09`, MaxRSS about `7047772K`.
  - Support job `49030546`: validation OK, 83 support frames.
  - Posthoc overlap: mean support-frame fraction `0.0625`, mean crop coverage `0.000030`.
- R032 high-recall sparse sensitivity:
  - Support job `49031389`: validation OK, 355 support frames.
  - Posthoc overlap: mean support-frame fraction `0.4125`, mean crop coverage `0.002846`.
- R033 high-recall tile-fill sensitivity:
  - Support job `49032424`: validation OK, 408 support frames, `fill_component_tiles=true`.
  - Posthoc overlap: mean support-frame fraction `0.3750`, mean crop coverage `0.001253`.
- Scientific interpretation:
  - PASS: DA3 setup/inference/support-generation pipeline is operational and durable.
  - FAIL/WEAK: current DA3 support-fusion variants do not produce strong frozen-window event-crop coverage. They are much weaker than R020 high-recall boxes (`0.491371` mean crop coverage) and should not be promoted into a positive rendered-method claim.
- Next meaningful rendered-method direction remains training-loop integration or a changed capacity/optimization mechanism. Do not spend more runs on this exact posthoc support-only family unless there is a new, predeclared depth formulation.

### 2026-07-10T02:32:14+02:00 - R034/R035 visibility-gate state

- Local branch at start of this wave: `codex/hide-reveal-poc-implementation`.
- Pre-edit HEAD: `f10071f4f676cdf60b9989d7ea0bfe7af7df6ae7`.
- Pre-edit dirty/untracked state was recorded before file changes; unrelated dirty files were preserved:
  - modified: `refine-logs/hide_reveal_poc/r015_poc_summary/poc_table.md`
  - modified: `refine-logs/hide_reveal_poc/r016_go_no_go_memo.md`
  - untracked workspace/personal files including `.codex/`, `.obsidian/`, `AGENTS.md`, `Untitled.canvas`, `configs/n3v/bootstrap.yaml`, `det_con.yaml`, `follow-up.md`, `idea-stage/`, `requirements.txt`, and verification images.
- Protocol/predeclaration commit pushed: `1a34b4d528f59d1bd887753dd9e5c736c02b3ef4`.
- Visibility-event implementation commit pushed: `e2c8cc08966a1b88f54bad75dfc999ab8b62b452`.
- Leonardo checkout fast-forwarded to `e2c8cc08966a1b88f54bad75dfc999ab8b62b452`; remote bash syntax and Python compile checks passed.
- R034 synthetic fixture PASS:
  - output: `refine-logs/hide_reveal_poc/r034_visibility_gate_synthetic/`
  - held-out candidate recall `1.0`, accepted precision/recall `1.0/1.0`, false event rate `0.0`, margin AUC `1.0`, identity reconnection `1.0`, `proceed_to_real_windows=true`.
- R035 proxy admission FAIL:
  - output: `refine-logs/hide_reveal_poc/r035_visibility_event_admission/`
  - candidates scored `72`, accepted `0`, validation `ok=true`.
  - mean delta score `+0.200982`; minimum delta score `+0.118458`, so no near-threshold acceptance case exists.
  - scientific interpretation: the simple image-space dynamic-attenuation proxy is worse than smooth on all scored training-observation candidates. Do not weaken the margin to create accepted events.
- Revised next step before frozen real scoring:
  - run R036/R037 as a full matched-budget training comparison on the fixed R020 high-recall candidate field directly.
  - `H_smooth` receives the same R020 field for ROI/capacity pressure but no visibility gate.
  - `H_event` receives the same R020 field plus the opacity visibility gate during the original 6000-iteration training loop.
  - no thresholds are tuned on frozen R009 crop overlap or frozen-window metrics.

### 2026-07-10T02:49:00+02:00 - R036/R037 train submission state

- Submitter executable-bit commit pushed: `0fa08ddf70e48ec93baf821a8a54e06f61f27280`.
- First train wave used the default `TIME=02:30:00` and was cancelled early because observed progress projected too close to the walltime:
  - R036 smooth jobs: `49042345`, `49042346`, `49042347`, cancelled after about 8 minutes.
  - R037 event jobs: `49042385`, `49042386`, `49042389`, cancelled after about 7 minutes.
  - Engineering interpretation: config/method launch was valid; event logs loaded 24 R020 visibility events per scene. The cancellation was only to avoid a predictable timeout.
- Replacement train wave submitted with `TIME=04:30:00`:
  - R036 smooth manifest: `refine-logs/visibility_event_smooth_train_jobs_20260710_024626.tsv`
  - R036 jobs: `49042444` cut_roasted_beef, `49042445` flame_steak, `49042446` sear_steak
  - R037 event manifest: `refine-logs/visibility_event_train_train_jobs_20260710_024651.tsv`
  - R037 jobs: `49042510` cut_roasted_beef, `49042512` flame_steak, `49042514` sear_steak
- Next command:

```bash
ssh siyengar@login.leonardo.cineca.it 'sacct -j 49042444,49042445,49042446,49042510,49042512,49042514 --format=JobID,State,Elapsed,Timelimit,ExitCode -P'
```

### 2026-07-10T06:12:18+02:00 - R036/R037 train complete; eval partially submitted, SSH blocked

- Local branch: `codex/hide-reveal-poc-implementation`.
- Local/remote branch state before this status update: `codex/hide-reveal-poc-implementation...origin/codex/hide-reveal-poc-implementation`.
- Dirty/untracked state remains unrelated and preserved:
  - modified: `refine-logs/hide_reveal_poc/r015_poc_summary/poc_table.md`
  - modified: `refine-logs/hide_reveal_poc/r016_go_no_go_memo.md`
  - untracked: `.codex/`, `.obsidian/`, `AGENTS.md`, `Untitled.canvas`, `configs/n3v/bootstrap.yaml`, `det_con.yaml`, `follow-up.md`, `idea-stage/`, `requirements.txt`, and verification images.
- R036 smooth-control training completed successfully:
  - `49042444` cut_roasted_beef: `COMPLETED`, elapsed `03:20:43`, exit `0:0`
  - `49042445` flame_steak: `COMPLETED`, elapsed `03:17:53`, exit `0:0`
  - `49042446` sear_steak: `COMPLETED`, elapsed `03:01:59`, exit `0:0`
- R037 event-gated training completed successfully:
  - `49042510` cut_roasted_beef: `COMPLETED`, elapsed `03:14:24`, exit `0:0`
  - `49042512` flame_steak: `COMPLETED`, elapsed `03:15:55`, exit `0:0`
  - `49042514` sear_steak: `COMPLETED`, elapsed `03:00:49`, exit `0:0`
- All six `chkpnt6000.pth` checkpoints were observed under:
  - `$WORK/proj_adags/runs/visibility_event_smooth_control_6000/20260710_024626_*_visibility_event_smooth_control_6000/`
  - `$WORK/proj_adags/runs/visibility_event_train_6000/20260710_024651_*_visibility_event_train_6000/`
- R036 smooth-control eval was submitted with `EVAL_TIME=01:30:00`:
  - manifest: `refine-logs/visibility_event_smooth_eval_jobs_20260710_060900.tsv`
  - jobs: `49045923` cut_roasted_beef, `49045924` flame_steak, `49045925` sear_steak
- R037 event-gated eval has not been submitted. Two submit attempts failed with SSH permission denial.
- Blocking diagnosis:
  - `ssh-add -L | ssh-keygen -Lf -` shows the loaded Leonardo SSH certificate was valid only from `2026-07-09T18:08:38` to `2026-07-10T06:08:38`.
  - A fresh `ssh -o BatchMode=yes -o ConnectTimeout=15 siyengar@login.leonardo.cineca.it "hostname"` at `2026-07-10T06:12:18+02:00` failed with `Permission denied (publickey,gssapi-keyex,gssapi-with-mic)`.
  - Alternate host/proxy attempts also failed after the certificate expiry. No obvious non-interactive certificate-renewal command was found in local context.
- Current status: infrastructure BLOCKED on Leonardo SSH credential renewal, not a scientific FAIL.

Resume after SSH certificate renewal:

```bash
ssh siyengar@login.leonardo.cineca.it 'cd /leonardo_work/EUHPC_D21_034/proj_adags/repo/adags && sacct -j 49045923,49045924,49045925 --format=JobID,State,Elapsed,Timelimit,ExitCode -P'

ssh siyengar@login.leonardo.cineca.it 'cd /leonardo_work/EUHPC_D21_034/proj_adags/repo/adags && EVAL_TIME=01:30:00 scripts/submit_visibility_event_pilot.sh --variant event --mode eval --run-manifest refine-logs/visibility_event_train_train_jobs_20260710_024651.tsv'
```

After both R036 and R037 eval renders complete, build the combined frozen-window scoring manifest, score once, then run R038 result-to-claim / experiment-audit before declaring PASS or FAIL.

### 2026-07-10T06:12+02:00 continuation - SSH still blocked; scoring path pinned

- Fresh access check again failed:
  - command: `ssh -o BatchMode=yes -o ConnectTimeout=15 siyengar@login.leonardo.cineca.it "hostname"`
  - result: `Permission denied (publickey,gssapi-keyex,gssapi-with-mic)`.
- The only loaded cert is still the expired Leonardo user cert:
  - principal: `siyengar`
  - valid: `2026-07-09T18:08:38` to `2026-07-10T06:08:38`.
- Local SSH config contains only host/proxy definitions for `login.leonardo.cineca.it`, `leonardo-gw`, and `leonardo-login02`; no non-interactive renewal hook was found in local config/scripts.
- Best strict baseline manifest for R036/R037 scoring is:
  - `refine-logs/hide_reveal_poc/r029_r030_disambiguation_manifest.json`
  - It already contains the five frozen windows with `route0`, `residual_uncertainty`, `matched_lifespan`, `hide_reveal` oracle upper bound, `event_boundary_micro_densify`, `oracle_crop_support_micro_densify`, and `route0_continue_6400`.

Exact post-renewal scoring sequence after R036/R037 eval folders exist:

```bash
cd /leonardo_work/EUHPC_D21_034/proj_adags/repo/adags

python scripts/run_hide_reveal_poc.py augment-real-manifest-system \
  --manifest refine-logs/hide_reveal_poc/r029_r030_disambiguation_manifest.json \
  --eval-root /leonardo_work/EUHPC_D21_034/proj_adags/runs/visibility_event_smooth_control_6000 \
  --system-name visibility_event_smooth_control \
  --out refine-logs/hide_reveal_poc/r036_r037_visibility_event_manifest_smooth.json

python scripts/run_hide_reveal_poc.py augment-real-manifest-system \
  --manifest refine-logs/hide_reveal_poc/r029_r030_disambiguation_manifest.json \
  --merge-manifest refine-logs/hide_reveal_poc/r036_r037_visibility_event_manifest_smooth.json \
  --eval-root /leonardo_work/EUHPC_D21_034/proj_adags/runs/visibility_event_train_6000 \
  --system-name visibility_event_train \
  --out refine-logs/hide_reveal_poc/r036_r037_visibility_event_manifest.json

python scripts/run_hide_reveal_poc.py real-eval \
  --manifest refine-logs/hide_reveal_poc/r036_r037_visibility_event_manifest.json \
  --out-dir refine-logs/hide_reveal_poc/r036_r037_visibility_event_real_eval
```

Do not score R036/R037 against `hide_reveal_real_windows.json` alone; that would omit the residual and matched-lifespan baselines needed for the strict gate.

### 2026-07-10T06:18:12+02:00 - blocked audit threshold reached

- Third consecutive goal-turn access check still fails:
  - command: `ssh -o BatchMode=yes -o ConnectTimeout=15 siyengar@login.leonardo.cineca.it "hostname"`
  - result: `Permission denied (publickey,gssapi-keyex,gssapi-with-mic)`.
- Loaded certificate remains unchanged and expired:
  - key ID: `sudarshan.iyengar@kuleuven.be`
  - principal: `siyengar`
  - valid: `2026-07-09T18:08:38` to `2026-07-10T06:08:38`.
- This blocks:
  - checking R036 smooth eval jobs `49045923`, `49045924`, `49045925`;
  - submitting R037 event eval;
  - fetching eval logs/renders;
  - running final frozen-window scoring and R038 audit.
- Scientific state remains unresolved, not failed:
  - R034 synthetic PASS.
  - R035 proxy admission FAIL.
  - R036/R037 full trainings COMPLETE.
  - R036 eval SUBMITTED.
  - R037 eval NOT SUBMITTED due external credential expiry.
- Resume condition: renew the Leonardo SSH certificate, then run the post-renewal commands in the previous status section.

### 2026-07-10T07:55+02:00 - resumed after SSH renewal; R036/R037 final FAIL

- SSH certificate was renewed and access recovered at `2026-07-10T07:36:50+02:00`:
  - `ssh siyengar@login.leonardo.cineca.it "hostname"` returned `login02.leonardo.local`.
  - renewed cert valid `2026-07-10T07:34:27` to `2026-07-10T19:34:27`.
- Remote repo fast-forwarded to pushed branch state before completing eval/scoring.
- R036 smooth eval jobs completed:
  - `49045923`, `49045924`, `49045925`: all `COMPLETED`, exit `0:0`.
- R037 event eval submitted and completed:
  - manifest: `refine-logs/visibility_event_train_eval_jobs_20260710_073836.tsv`
  - jobs: `49051779`, `49051782`, `49051783`: all `COMPLETED`, exit `0:0`.
- Verified R036 and R037 eval folders:
  - each of six scene/variant eval folders contains 300 `renders`, 300 `gt`, 300 `static`, and 300 `dynamic` PNG frames under `test/ours_6000`.
- Built final strict manifest:
  - `refine-logs/hide_reveal_poc/r036_r037_visibility_event_manifest.json`
  - validation `ok=true`, zero errors, zero warnings.
- Ran frozen-window scoring:
  - output: `refine-logs/hide_reveal_poc/r036_r037_visibility_event_real_eval/`
  - systems: `event_boundary_micro_densify`, `hide_reveal`, `matched_lifespan`, `oracle_crop_support_micro_densify`, `residual_uncertainty`, `route0`, `route0_continue_6400`, `visibility_event_smooth_control`, `visibility_event_train`.
- Fetched train/eval Slurm logs into:
  - `refine-logs/hide_reveal_poc/r036_r037_visibility_event_logs/`
  - scan for hard error markers (`Traceback`, `ERROR`, `FAILED`, `CANCELLED`, etc.) returned no matches.
- Final R037 gate result:
  - mean PSNR `30.1089` versus route0 `30.5021`
  - mean L1/proxy-LPIPS `0.0157600` versus route0 `0.0148316`
  - route0 PSNR+L1 wins `0/5`
  - strict all-baseline PSNR+L1 wins `0/5`
  - static no-worse versus route0 `1/5`
  - mean oracle PSNR-gap recovery `-0.0391`
- R038 independent review:
  - claim_supported `no`
  - confidence `high`
  - integrity_status `warn`
  - no evidence of fake GT, self-normalized scores, or phantom metrics; warnings are limited LPIPS/identity/gate-stat evidence and five-window scope.
- Completion state: FAIL. This is a valid negative method result, not blocked.
