# Overnight Status

## Recovery Snapshot

- Current run ID: R016
- Current branch: `codex/hide-reveal-poc-implementation`
- Last local commit before R015 artifacts: `7580cd4eae3f48215bab1a48468bcce6abe8ac39`
- Last pushed milestone commit before R015 artifacts: `7580cd4eae3f48215bab1a48468bcce6abe8ac39`
- Last HPC job ID: `48654171`
- Latest success/failure: R015 PASS; PoC table, decision inputs, and five qualitative crop strips generated locally from R010-R013 outputs
- Next command to run: write R016 go/no-go memo from `refine-logs/hide_reveal_poc/r015_poc_summary/poc_decision_inputs.json`
- Open blockers: none known yet

## Session Log

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
- R016: pending.
