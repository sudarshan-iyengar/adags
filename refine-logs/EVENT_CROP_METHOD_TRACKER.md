# Event-Crop Method Tracker

Generated: 2026-07-07

## Current Phase

Phase 4: R025 scoring completed for the first non-oracle method candidate.

The R017 runtime opacity gate is closed as a failed actual-method check. M1 non-oracle residual-component local refinement is also closed as a failed checkpoint-backed Gaussian method check. Future methods must not use the frozen R009 event crops as test-time method inputs.

## Source Metrics

| Role | System | Mean PSNR | Mean L1/proxy-LPIPS | Mean flicker | Mean static ghost | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | route0 | 30.5021 | 0.0148316 | 0.00799083 | 0.127333 | `refine-logs/hide_reveal_poc/r010_route0_real_eval/real_event_window_summary.json` |
| baseline | matched_lifespan | 29.8181 | 0.0163546 | 0.00795601 | 0.127333 | `refine-logs/hide_reveal_poc/r012_r013_derived_real_eval/real_event_window_summary.json` |
| baseline | residual_uncertainty | 30.0734 | 0.0165723 | 0.00803902 | 0.145702 | `refine-logs/hide_reveal_poc/r011_residual_uncertainty_real_eval/real_event_window_summary.json` |
| upper_bound | derived oracle hide_reveal | 41.7149 | 0.00266536 | 0.00168586 | 0.127333 | `refine-logs/hide_reveal_poc/r012_r013_derived_real_eval/real_event_window_summary.json` |
| failed actual | R017 actual_hide_reveal | 19.3667 | 0.0761056 | 0.0162899 | 0.152789 | `refine-logs/hide_reveal_poc/r017_actual_real_eval/real_event_window_summary.json` |
| failed non-oracle | R025 event_candidate_refine | 28.9393 | 0.0188750 | 0.00847709 | 0.125652 | `refine-logs/hide_reveal_poc/r025_event_candidate_refine_real_eval/real_event_window_summary.json` |

Oracle upper-bound deltas versus route0:
- PSNR: `+11.2128`
- L1/proxy-LPIPS: `-0.0121662`
- Flicker: `-0.0063050`
- Static ghost: `0.0`

## Predeclared Evaluation Protocol

Evaluation split:
- Frozen R009 windows in `refine-logs/hide_reveal_real_windows.json`.
- Crops are evaluation-only and must not be consumed by the method as test-time event support.

Primary metrics:
- Per-window crop PSNR, higher is better.
- Per-window crop L1/proxy-LPIPS, lower is better.
- Per-window crop flicker, lower is better.
- Per-window static ghost score, lower is better.
- Accepted event count and false/broad event count if the method produces event candidates.
- Qualitative crop strips using the same five frozen windows.

Currently unavailable metrics:
- Learned LPIPS: unavailable unless sidecar weights are made available without compute-node network downloads.
- Confident-track identity switches: unavailable because R009 discovery found no track-confidence sidecars.

Strict PASS gate for a method:
- The method is checkpoint-backed or newly trained Gaussian-rendered output, not GT crop compositing.
- The method does not use R009 frozen event crops as test-time support.
- At least 3/5 frozen windows improve versus route0, matched_lifespan, and residual_uncertainty on both PSNR and L1/proxy-LPIPS.
- At least 3/5 frozen windows do not worsen static ghost versus route0.
- Mean PSNR improves over route0 by at least `+0.5 dB` and mean L1/proxy-LPIPS improves over route0 by at least `-0.001`.
- The method recovers at least 25% of the oracle upper bound on either mean PSNR improvement or mean L1/proxy-LPIPS reduction:
  - PSNR fraction: `(method_psnr - 30.5021) / 11.2128`
  - L1 fraction: `(0.0148316 - method_l1) / 0.0121662`

FAIL gate:
- A complete method run worsens mean PSNR and L1/proxy-LPIPS versus route0, or passes fewer than 3/5 windows, after logs and outputs are valid.
- A method requires oracle event-crop labels at test time to obtain its gain.
- A method cannot produce comparable Gaussian-rendered folders and only produces image composites.

## Candidate Methods

| Candidate | Mechanism | Why It Might Recover Oracle-Like Fix | Cost | Failure Modes | Required Data | Status |
| --- | --- | --- | --- | --- | --- | --- |
| M1 non-oracle residual-component local refinement | Detect candidate event supports from high residual, dynamic masks, flow validity/disagreement, and local flicker in training/eval render diagnostics; select connected components without R009 crop labels; locally optimize a small set of Gaussian color/opacity/visibility parameters under a fixed budget; render normally. | R013 says the error is local and large. R017 failed by subtracting content; M1 can add/refine local appearance while preserving route0 elsewhere. | Medium. Reuses route0 checkpoints and renderer; needs candidate-map generation, local optimization CLI, Slurm wrapper, metadata. | Detector selects easy residuals or wrong regions; local fitting overfits observed view; static ghost rises; output becomes crop-like if support leaks from evaluation labels. | Route0 renders/GT/static/dynamic, masks, flow, checkpoints. | R025 FAIL: checkpoint-backed Gaussian renders completed, but 0/5 frozen windows improved and mean PSNR/L1 worsened versus route0. |
| M2 occlusion-boundary gated micro-densification | Use dynamic-mask boundaries and flow occlusion/disocclusion cues to seed or enable a small event-local densification budget, with strict point caps and no use of frozen crop labels. | The oracle fix may require new or sharper local capacity, not just opacity changes or local loss reweighting. | Medium-high. Requires support-map generation, densification/update path, budget accounting, and checkpoint-backed renders. | Novelty pressure from visibility-aware densification; may become known densification rather than identity event; may fragment identity; may repeat M1 damage if support is broad or wrong. | Masks, flow, route0 checkpoints/renders, training images. | Selected next candidate after R025 FAIL; predeclared below. |
| M3 temporal inconsistency event proposal plus conservative visibility gate | Build non-oracle event candidates from route0 render flicker/residual over time; apply a much narrower, component-local gate than R017 and log selected Gaussian counts. | R017 selected too many Gaussians. A component-local proposal may avoid broad content deletion. | Low-medium. Reuses R017 renderer hook with non-oracle support maps. | Still only removes content; likely cannot synthesize revealed texture; may repeat R017 failure with smaller damage. | Route0 renders/GT for candidate construction, masks. | Backup or diagnostic. |
| M4 identity-aware reveal matching with tracks/features | If reliable track/feature sidecars can be generated, commit hide/reveal only when hidden identity evidence reconnects across the event; train/refine selected carriers. | Aligns with original novelty boundary against lifespan-only gating. | High. Track sidecars were absent in R009; generation may be a separate project. | Blocked by unavailable tracks; noisy tracks around occlusions; high implementation burden. | Confident tracks/features, checkpoints, masks/flow. | Deferred. |

## Selected First Candidate

Selected first candidate: M1 non-oracle residual-component local refinement.

Reason:
- It directly addresses the R017 failure mode: opacity attenuation removed/dimmed content but did not synthesize the hidden/revealed surface.
- It is non-oracle if candidate components are generated from residual/mask/flow cues without reading R009 crop labels.
- It can produce actual Gaussian-rendered output folders if implemented as checkpoint-backed local refinement.
- It has a clear fail interpretation: if local Gaussian refinement cannot beat route0 on the frozen windows, the oracle crop fix may require stronger geometry/identity machinery rather than simple local updates.

Before full local-refinement implementation:
- Inspect existing train/resume/render code for the smallest local-refinement entry point.
- Verify candidate-map inputs exist on HPC for the three scenes with the R018 dry-run output.
- Use the A1 dry-run output to choose whether M1 has plausible event support before writing any checkpoint-updating code.
- Record the exact command and output directory before submitting any Slurm job.

Outcome:
- R023 trained three resumed checkpoints to `chkpnt6200.pth`.
- R024 rendered complete `test/ours_6200` folders for `cut_roasted_beef`, `flame_steak`, and `sear_steak`.
- R025 evaluated those folders on the frozen windows and failed the predeclared gate:
  - 0/5 windows improved over all three baselines on both PSNR and L1/proxy-LPIPS.
  - 0/5 windows improved over route0 on both PSNR and L1/proxy-LPIPS.
  - 2/5 windows were no worse than route0 on static ghost, below the 3/5 requirement.
  - Mean PSNR delta versus route0: `-1.5629 dB`.
  - Mean L1/proxy-LPIPS delta versus route0: `+0.004043`.
  - Oracle recovery fractions were negative for both PSNR and L1.
  - Independent result-to-claim review returned `claim_supported: no`, confidence `high`.

## Selected Second Candidate

Selected second candidate: M2 occlusion-boundary gated micro-densification.

Base state:
- Predeclared after R025 FAIL at local HEAD `f74ff87c9fad01a4d39d9574f9f9d1b48c0c41d2` on branch `codex/hide-reveal-poc-implementation`.
- Existing unrelated dirty/untracked files must remain preserved and must not be folded into M2 commits.

Scientific purpose:
- Test a different mechanism from M1. M1 refined existing route0 capacity inside non-oracle candidate supports and worsened the frozen-window metrics. M2 tests whether a small amount of event-local capacity injection at visibility boundaries can move checkpoint-backed Gaussian renders toward the oracle crop upper bound.
- This is a bounded diagnostic, not a paper claim. Do not force a positive result or tune on the frozen windows.

Allowed non-oracle support cues:
- Dynamic-mask interior and boundary maps from the scene sidecars/training data.
- Flow valid/invalid boundaries, flow magnitude discontinuities, and flow disagreement/occlusion-like cues if the sidecars are available.
- Route0 render diagnostics only if computed without reading R009 crop labels: route0 dynamic/static render disagreement and temporal flicker over full scene frames are allowed; GT crop residuals and threshold tuning against the frozen crop windows are not allowed.
- The frozen R009 manifest may be used only to locate scene/system paths and for final evaluation. M2 support generation must record `uses_frozen_window_labels=false`, `uses_gt_crop_pixels=false`, and whether any manifest fields were used only as path sources.

Support artifact:
- R026 will build a deterministic boundary-support artifact under `refine-logs/hide_reveal_poc/r026_m2_boundary_support/`.
- The artifact must include per-scene/per-frame support maps or compact component boxes, a JSON metadata file, validation report, and enough statistics to audit support size.
- Support caps before training: at most 36 retained components per scene, at most 3 percent supported pixels per frame after dilation, dilation radius no greater than 12 pixels, and no component selection using frozen-window overlap.

Training/method cap:
- Resume each route0 scene from `chkpnt6000.pth`; write new outputs under a new M2 run name, not under R023/R024.
- Target checkpoint: `chkpnt6400.pth` unless code inspection shows a narrower existing iteration convention is safer.
- Point budget cap: target no more than `625000` total Gaussians per scene and no more than `25000` net new/split Gaussians above the resumed checkpoint count. If the resumed checkpoint already exceeds the cap, record the actual count and cap net growth at zero rather than silently increasing capacity.
- Densification boost must be gated by the R026 boundary support. Losses may remain globally comparable, but the extra micro-densification mechanism must be local to the support artifact.
- Static-background safeguards from route0/M1 should not be weakened. Any intentional change to static loss, static exclusion, or pruning must be logged before submission.

Outputs:
- R026: support artifact, local/dry-run validation, py_compile checks for edited Python files, and shell syntax checks on Leonardo for edited shell wrappers.
- R027: Slurm train/eval manifests, checkpoint paths, complete `test/ours_6400` render folders for all three scenes, fetched stdout/stderr logs, and a frozen-window scoring manifest that adds system `event_boundary_micro_densify`.
- R028 if needed: final scoring/strips/review if R027 is split into train/eval and scoring jobs.

PASS gate:
- Same strict gate as above, with system name `event_boundary_micro_densify`.
- The method must be checkpoint-backed or newly trained Gaussian-rendered output and must not use frozen R009 event crops as support.
- At least 3/5 frozen windows must improve versus route0, matched_lifespan, and residual_uncertainty on both PSNR and L1/proxy-LPIPS.
- At least 3/5 frozen windows must not worsen static ghost versus route0.
- Mean PSNR must improve over route0 by at least `+0.5 dB`, mean L1/proxy-LPIPS must improve over route0 by at least `-0.001`, and the method must recover at least 25 percent of the oracle upper bound on either mean PSNR or mean L1.

FAIL/SKIP/BLOCKED gates:
- FAIL if a complete valid M2 method run worsens mean PSNR and L1/proxy-LPIPS versus route0, or passes fewer than 3/5 windows after valid logs and outputs.
- FAIL if M2 requires frozen crop labels or GT crop compositing to obtain its gain.
- SKIP if code/data inspection shows M2 would only be generic visibility-aware densification without event-local support, budget accounting, or a plausible mechanism distinct from M1.
- BLOCKED only for missing permissions/data/compute or the same unresolved implementation blocker after three serious diagnosis-and-patch attempts.

## Attempt Log

| Attempt | Candidate | Commit | Job ID | Output | Verdict | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Recovery | Evidence/state recovery | `0b18166bec6d1a2d371764c70bdcf53b23319a5e` | n/a | `refine-logs/EVENT_CROP_FIX_EVIDENCE.md`, `research-wiki/event-crop-fix.md` | COMPLETE | Preserved R001-R017 evidence, oracle upper bound, R017 failure, frozen-window paths, and predeclared non-oracle evaluation gate. |
| A0 | R017 runtime opacity gate | `f5d43539aee500051f2a4c5eeca5420293b636f1` | `48760029`, `48760448` | `refine-logs/hide_reveal_poc/r017_actual_real_eval/` | FAIL | Checkpoint-backed renderer output, no GT pixels, but 0/5 windows passed and all mean metrics worsened. |
| A1 | Non-oracle event candidate discovery dry-run | `0bf0967483e622af5cb6ac81de2b3f09060c33d9` | local smoke | `refine-logs/hide_reveal_poc/local_smoke/nonoracle_candidates/out/` | SMOKE PASS | New `nonoracle-candidates` stage produced 3 valid smoke candidates with `validation_ok=True`, `uses_gt_residual=false`, and `uses_frozen_window_labels=false`; not yet a Gaussian-rendered method result. |
| R018 | A1 candidate discovery on real scene sources | `db467a328b6b0e02482eb0e36de24b27b850b907` | `48763378` | `refine-logs/hide_reveal_poc/r018_nonoracle_candidates/` | DETECTOR FAIL | Slurm job completed and validation passed with 24 candidates, but posthoc frozen-overlap audit covered 0/5 windows; selected boxes clustered on top image bands with zero motion-mask support. |
| A2 | Motion-supported candidate discovery dry-run | `f69034be1ca32ddcd24756d945ead467d59e3c24` | local smoke | `refine-logs/hide_reveal_poc/local_smoke/nonoracle_candidates/out_motion_supported/` | SMOKE PASS | Revised detector multiplies dynamic/static-delta/flicker evidence by motion-mask support when masks are available; local smoke still selects the synthetic moving-square support. |
| R019 | A2 motion-supported candidate discovery | `a46c79678150881f6c3fc50e074e7a4de100a9bc` | `48763799` | `refine-logs/hide_reveal_poc/r019_motion_supported_nonoracle_candidates/` | PARTIAL | Validation passed with 24 candidates and top-band artifact removed; posthoc frozen-overlap audit covered 2/5 windows, so M1 support remains insufficient. |
| R020 | A2 high-recall motion-supported candidate pool | `a46c79678150881f6c3fc50e074e7a4de100a9bc` | `48764048` | `refine-logs/hide_reveal_poc/r020_high_recall_motion_supported_nonoracle_candidates/` | SUPPORT POOL | Validation passed with 72 candidates using `HIDE_REVEAL_NONORACLE_TOP_K_PER_SCENE=24`; posthoc frozen-overlap audit covered 3/5 windows. This is candidate support only, not a rendered-method pass. |
| R021 | Candidate-local refinement train jobs | `5a3ded1d338dc7534317434907835c0de4da0e73` | `48764715`, `48764716`, `48764718` | `refine-logs/event_candidate_refine_train_jobs_20260707_033133.tsv` | RUNNING / MONITOR BLOCKED | Train jobs submitted for cut_roasted_beef, flame_steak, and sear_steak; startup logs looked normal, but later SSH auth rejection blocked monitoring and collection. Eval/scoring not yet run. |
| R021b | Leonardo runner extension-dir fix | `7d848fc` | n/a | `scripts/run_leonardo.sh` | COMPLETE | Added per-job `TORCH_EXTENSIONS_DIR`, `TORCH_CUDA_ARCH_LIST=8.0`, and `MAX_JOBS` after R021 timed out at shared PyTorch extension startup without checkpoints. |
| R021c | Real-manifest augmentation helper | `657f0cd201dda6c84c0b4442e53380e4d837f1ad` | n/a | `scripts/run_hide_reveal_poc.py augment-real-manifest-system` | COMPLETE | Adds reusable command to attach future refined eval folders to frozen windows and merge baseline manifests before scoring. |
| R022 | Candidate-local refinement replacement train jobs | `657f0cd201dda6c84c0b4442e53380e4d837f1ad` | `48796168`, `48796170`, `48796174` | `refine-logs/event_candidate_refine_train_jobs_20260707_110953.tsv` | CANCELLED / RUNNER ENV BUG | Jobs started but logs showed inherited shared `TORCH_EXTENSIONS_DIR=/leonardo_work/.../build/torch_extensions`; cancelled before repeating the full R021 timeout. |
| R022b | Forced per-job extension-root fix | `ad637f3ec50129fe40c5715804d446dfe6bdc90d` | n/a | `scripts/run_leonardo.sh` | COMPLETE | Slurm jobs now force `$PROJECT_ROOT/build/torch_extensions_jobs/$SLURM_JOB_ID` unless `ADAGS_TORCH_EXTENSIONS_DIR` is explicitly set, avoiding `leonardo_env.sh` shared-root override. |
| R023 | Candidate-local refinement train jobs after forced extension-root fix | `ad637f3ec50129fe40c5715804d446dfe6bdc90d` | `48799988`, `48799992`, `48799995` | `refine-logs/event_candidate_refine_train_jobs_20260707_114908.tsv`, `logs/event_candidate_refine_train_*_487999*.{out,err}` | TRAIN COMPLETE | Jobs completed in 13-14 minutes, used per-job extension roots, and wrote three `chkpnt6200.pth` checkpoints plus point clouds under `/leonardo_scratch/fast/EUHPC_D21_034/proj_adags/runs/event_candidate_local_refine_6200/20260707_114908_*`. |
| R024 | Candidate-local refinement eval renders | `ad637f3ec50129fe40c5715804d446dfe6bdc90d` | `48802355`, `48802357`, `48802359` | `refine-logs/event_candidate_refine_eval_jobs_20260707_121022.tsv`, `logs/event_candidate_refine_eval_*_488023*.{out,err}` | EVAL COMPLETE | All three eval jobs completed with `ExitCode=0:0`; each output folder has 300 `renders`, `gt`, `static`, and `dynamic` frames under `test/ours_6200`. |
| R025 | Candidate-local refinement frozen-window scoring | `1a747fae7079f7352c3103f51d735912fcedf10a` | `48805053` | `refine-logs/hide_reveal_poc/r025_event_candidate_refine_real_eval/`, `refine-logs/hide_reveal_poc/r025_event_candidate_refine_summary/` | FAIL | Valid scoring job completed with `ExitCode=0:0`. R025 mean PSNR `28.9393` vs route0 `30.5021`; mean L1 `0.0188750` vs route0 `0.0148316`; 0/5 windows improved on PSNR+L1; static no-worse was 2/5. |
| R025 review | Result-to-claim audit | n/a | n/a | `.aris/traces/result-to-claim/2026-07-07_run01/`, `research-wiki/experiments/r025-event-candidate-refine-real-window-check.md`, `findings.md` | CLAIM NOT SUPPORTED | Independent reviewer judged `claim_supported: no` with high confidence. The method-form constraint was satisfied, but the quantitative gate failed decisively. |
| M2 predeclare/plumbing | Occlusion-boundary gated micro-densification | `13297e290a8511512f4f3be08f7ceb425fcc3ca7` | n/a | `configs/n3v/event_boundary_micro_densify_6400.yaml`, `scripts/run_hide_reveal_poc.py event-boundary-support` | COMPLETE | Predeclared M2 after R025 FAIL, added non-oracle boundary-support artifact builder, support-mask loader, residual-free densify option, and M2 config capped at 625k points / 25k net growth. Local `py_compile`, CLI help, synthetic support smoke, and remote `bash -n`/`py_compile` passed. |
| R026a | M2 boundary-support artifact, connected-component implementation | `13297e290a8511512f4f3be08f7ceb425fcc3ca7` | `48872013` | `refine-logs/hide_reveal_poc/r026_m2_boundary_support/job_metadata.txt`, `logs/hide_reveal_event-boundary-support_48872013.{out,err}` | CANCELLED / PERF BUG | Job ran on `lrdn0033` for about 9 minutes and produced only metadata; cancelled before timeout. Diagnosis: pure Python per-pixel connected-component pass was too slow for 3 scenes x 300 masks. Patch next: tile-capped compact support boxes preserving the same non-oracle support contract. |
| R026b | M2 boundary-support artifact, tile-capped support boxes | `efca178e7fd5fef49aa84c762350dfa8cbdd36d8` | `48872653` | `refine-logs/hide_reveal_poc/r026_m2_boundary_support/` | SUPPORT PASS | Job completed in `00:08:53` with `ExitCode=0:0`; validation `ok=true`, errors `0`, warnings `0`. Generated 108 selected components across 66 support frames. Guardrails record `uses_gt_residual=false`, `uses_gt_crop_pixels=false`, `uses_frozen_window_labels=false`, source usage `scene_sources_only_for_paths_and_frame_ranges`, and max support fraction `0.005205` below the predeclared 3 percent cap. This is support only, not a rendered-method result. |
| R027 train | M2 event-boundary micro-densification train resumes | `efca178e7fd5fef49aa84c762350dfa8cbdd36d8` | `48873219`, `48873220`, `48873221` | `refine-logs/event_boundary_micro_densify_train_jobs_20260707_234937.tsv` | SUBMITTED | Submitted three Slurm jobs from route0 `chkpnt6000.pth` to target `chkpnt6400.pth` using `configs/n3v/event_boundary_micro_densify_6400.yaml` and R026 support manifest. Await train completion before eval/scoring; no PASS/FAIL decision until frozen-window metrics are produced. |
