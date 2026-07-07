# M2 Handoff Packet

Created: 2026-07-07

## Current Goal

Continue the non-oracle event-crop-fix objective with M2: occlusion-boundary gated micro-densification.

The goal is not to rescue M1 or force a positive result. The new session should act as orchestrator for M2 and conclude `PASS`, `FAIL`, `SKIP`, or `BLOCKED` with evidence.

Completion standard:

- `PASS`: M2 is a non-oracle checkpoint-backed/newly trained Gaussian method that improves the frozen R009 real windows and recovers a meaningful fraction of the oracle event-crop upper bound.
- `FAIL`: M2 is implemented/evaluated enough to be scientifically tested and does not improve over the baselines.
- `SKIP`: M2 is rejected before implementation by a documented scientific/engineering reason that makes it clearly unpromising or redundant.
- `BLOCKED`: progress is blocked by missing permissions/data/compute or the same unresolved implementation blocker after three serious diagnosis+patch attempts.

## Scientific State Through R025

R001-R017 established a strong upper-bound but not a working method:

- R009 froze five real event windows before scoring in `refine-logs/hide_reveal_real_windows.json`.
- R010 route0 baseline over those windows: PSNR `30.5021`, L1/proxy-LPIPS `0.0148316`, flicker `0.00799083`, static ghost `0.127333`.
- R011 residual/uncertainty baseline was worse than route0 on mean PSNR/L1/flicker/static ghost.
- R012 matched-lifespan was worse than route0 on mean PSNR/L1.
- R013/R015 derived oracle hide/reveal upper bound strongly improved the crops: PSNR `41.7149`, L1/proxy-LPIPS `0.00266536`, flicker `0.00168586`, static ghost `0.127333`.
- R017 actual checkpoint-backed opacity gate failed: PSNR `19.3667`, L1 `0.0761056`, flicker `0.0162899`, static ghost `0.152789`, and `0/5` windows passed.

R018-R025 tested the first non-oracle line, M1 residual-component local refinement:

- R018 first non-oracle detector produced structurally valid candidates but posthoc overlap covered `0/5` frozen windows.
- R019 motion-supported detector fixed the top-band artifact but covered only `2/5`.
- R020 high-recall candidate pool covered `3/5`; this was only support discovery, not a rendered method.
- R021/R022 exposed and fixed HPC shared PyTorch extension-root failures.
- R023 trained M1 checkpoints to `chkpnt6200.pth`.
- R024 rendered complete `test/ours_6200` folders for all three scenes.
- R025 scored M1 on the frozen windows and failed the predeclared gate.

R025 result:

| System | Mean PSNR | Mean L1/proxy-LPIPS | Mean flicker | Mean static ghost |
| --- | ---: | ---: | ---: | ---: |
| route0 | 30.5021 | 0.0148316 | 0.00799083 | 0.127333 |
| oracle derived hide_reveal | 41.7149 | 0.00266536 | 0.00168586 | 0.127333 |
| R025 event_candidate_refine | 28.9393 | 0.0188750 | 0.00847709 | 0.125652 |

R025 gate counts:

- `0/5` windows improved versus all three baselines on both PSNR and L1/proxy-LPIPS.
- `0/5` windows improved versus route0 on both PSNR and L1/proxy-LPIPS.
- `2/5` windows were no worse than route0 on static ghost, below the `3/5` requirement.
- PSNR oracle recovery fraction: `-0.1394`.
- L1 oracle recovery fraction: `-0.3323`.
- Independent result-to-claim review: `claim_supported: no`, confidence `high`.

Conclusion through R025: the oracle crop repair is real as an upper bound, but the tested actual Gaussian methods have not produced it. M1 is closed as `FAIL`.

## Why M1 Failed

M1 tried to discover candidate event supports non-oracle, then locally refine route0 Gaussian checkpoints inside those candidate regions. It satisfied the method-form constraint by producing checkpoint-backed Gaussian renders, but the quantitative result worsened crop reconstruction.

Likely failure modes to preserve:

- Candidate support was only partial. The high-recall R020 pool reached posthoc overlap `3/5`, so it was good enough to test but not a clean detector.
- Local ROI/exclusion refinement did not add the missing/revealed surface; it mainly tracked or damaged route0-like render behavior.
- The method had no explicit mechanism to allocate new local geometry/capacity at occlusion/disocclusion boundaries.
- It may have over-optimized observed views or wrong regions while not improving the frozen event crops.

Do not tune M1 on the frozen windows to manufacture success. Treat it as negative knowledge.

## Why M2 Is Still Worth Trying

M2 is different enough from M1 to be scientifically meaningful:

- M1 mostly reweighted/refined existing Gaussian state within candidate ROIs. R017 showed hiding content is insufficient; R025 showed local refinement of existing capacity is insufficient.
- M2 explicitly targets capacity/geometry: dynamic-mask boundaries and flow occlusion/disocclusion cues seed or enable a small event-local micro-densification budget.
- The oracle upper bound suggests the missing information is local and large. A method that can add or sharpen local Gaussian capacity at the visibility boundary is a plausible next mechanism.
- M2 can be predeclared and budget-limited without using frozen evaluation crops as support.
- A clean M2 failure would answer a different scientific question: whether event-local capacity injection, not just support selection/local loss, can move the actual Gaussian renderer toward the oracle crop fix.

Keep novelty risk in mind: M2 may look like generic visibility-aware densification. It should be framed as an event-crop/frozen-window diagnostic method unless it passes and survives literature/novelty review.

## Git State At Handoff Creation

Workspace: `D:\adags`

Branch:

```text
codex/hide-reveal-poc-implementation
```

Commit before creating this handoff file:

```text
3a279198727a5e88d3a44059c62d7580cf0117a8
```

Last pushed evidence commits:

- `69a877a897eb239dcb236be67542c641e5ae38aa` - `Record R025 event candidate refine failure`
- `3a279198727a5e88d3a44059c62d7580cf0117a8` - `Record R025 evidence commit hash`

Git status before creating this file:

```text
## codex/hide-reveal-poc-implementation...origin/codex/hide-reveal-poc-implementation
 M refine-logs/hide_reveal_poc/r015_poc_summary/poc_table.md
 M refine-logs/hide_reveal_poc/r016_go_no_go_memo.md
?? .obsidian/
?? AGENTS.md
?? Untitled.canvas
?? configs/n3v/bootstrap.yaml
?? det_con.yaml
?? follow-up.md
?? idea-stage/
?? requirements.txt
?? verify_mask.jpg
?? verify_masked_flow.jpg
?? verify_raw_flow.jpg
```

Preservation rule: those dirty/untracked files predate this handoff and should not be reverted or folded into M2 commits unless directly needed and explicitly understood.

## Relevant Local Paths

Core state:

- `refine-logs/M2_HANDOFF.md`
- `refine-logs/EVENT_CROP_METHOD_TRACKER.md`
- `refine-logs/OVERNIGHT_STATUS.md`
- `refine-logs/EVENT_CROP_FIX_EVIDENCE.md`
- `refine-logs/EXPERIMENT_TRACKER.md`
- `research-wiki/event-crop-fix.md`
- `research-wiki/experiments/r025-event-candidate-refine-real-window-check.md`
- `findings.md`

Frozen evaluation and baselines:

- Frozen windows: `refine-logs/hide_reveal_real_windows.json`
- R010 route0 eval: `refine-logs/hide_reveal_poc/r010_route0_real_eval/real_event_window_summary.json`
- R011 residual baseline eval: `refine-logs/hide_reveal_poc/r011_residual_uncertainty_real_eval/real_event_window_summary.json`
- R012/R013 matched-lifespan/oracle eval: `refine-logs/hide_reveal_poc/r012_r013_derived_real_eval/real_event_window_summary.json`
- R017 actual-method failure: `refine-logs/hide_reveal_poc/r017_actual_method_report.md`
- R025 decision memo: `refine-logs/hide_reveal_poc/r025_event_candidate_refine_decision_memo.md`
- R025 summary: `refine-logs/hide_reveal_poc/r025_event_candidate_refine_real_eval/real_event_window_summary.json`
- R025 qualitative strips: `refine-logs/hide_reveal_poc/r025_event_candidate_refine_summary/crop_strips/`

M1 candidate/refinement artifacts that may be useful for M2 comparisons:

- R020 high-recall non-oracle support: `refine-logs/hide_reveal_poc/r020_high_recall_motion_supported_nonoracle_candidates/nonoracle_candidate_manifest.json`
- R023 train manifest: `refine-logs/event_candidate_refine_train_jobs_20260707_114908.tsv`
- R024 eval manifest: `refine-logs/event_candidate_refine_eval_jobs_20260707_121022.tsv`
- R025 scoring manifest: `refine-logs/hide_reveal_poc/r025_event_candidate_refine_manifest.json`
- R025 scoring logs: `logs/hide_reveal_real_48805053.{out,err}`
- R024 eval logs: `logs/event_candidate_refine_eval_*_488023*.{out,err}`

Scripts/configs likely relevant:

- `scripts/run_hide_reveal_poc.py`
- `utils/hide_reveal_poc.py`
- `scripts/submit_hide_reveal_poc.sh`
- `scripts/submit_event_candidate_refine.sh`
- `scripts/run_leonardo.sh`
- `scripts/make_hide_reveal_crop_strips.py`
- `configs/n3v/event_candidate_local_refine_6200.yaml`
- `utils/motion_prior_utils.py`
- `main.py`

## HPC Access Info

SSH:

```text
ssh siyengar@login.leonardo.cineca.it
```

Roots:

- `$WORK` resolves to `/leonardo_work/EUHPC_D21_034`
- repo: `$WORK/proj_adags/repo/adags`
- logs: `$WORK/proj_adags/repo/adags/logs`
- data: `$WORK/proj_adags/data/n3v`
- runs: `$WORK/proj_adags/runs`

Concrete expanded paths:

- repo: `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags`
- logs: `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/logs`
- data: `/leonardo_work/EUHPC_D21_034/proj_adags/data/n3v`
- runs: `/leonardo_work/EUHPC_D21_034/proj_adags/runs`

Route0 render folders:

- `/leonardo_scratch/fast/EUHPC_D21_034/proj_adags/runs/fixed_budget_lora_route0_600k/20260701_012706_cut_roasted_beef_fixed_budget_lora_route0_600k/test/ours_6000`
- `/leonardo_scratch/fast/EUHPC_D21_034/proj_adags/runs/fixed_budget_lora_route0_600k/20260701_012711_flame_steak_fixed_budget_lora_route0_600k/test/ours_6000`
- `/leonardo_scratch/fast/EUHPC_D21_034/proj_adags/runs/fixed_budget_lora_route0_600k/20260701_012714_sear_steak_fixed_budget_lora_route0_600k/test/ours_6000`

M1 R023/R024 run root:

- `/leonardo_scratch/fast/EUHPC_D21_034/proj_adags/runs/event_candidate_local_refine_6200/20260707_114908_cut_roasted_beef_event_candidate_local_refine_6200`
- `/leonardo_scratch/fast/EUHPC_D21_034/proj_adags/runs/event_candidate_local_refine_6200/20260707_114908_flame_steak_event_candidate_local_refine_6200`
- `/leonardo_scratch/fast/EUHPC_D21_034/proj_adags/runs/event_candidate_local_refine_6200/20260707_114908_sear_steak_event_candidate_local_refine_6200`

HPC rules:

- Do not run heavy CPU/GPU work on the login node.
- Create scheduler scripts under `scripts/` or `jobs/`.
- Submit long work through Slurm (`sbatch`).
- Store logs under `logs/` with job IDs.
- Fetch logs directly over SSH; do not ask the user to paste logs.
- Keep per-job `TORCH_EXTENSIONS_DIR`; `scripts/run_leonardo.sh` already forces `$PROJECT_ROOT/build/torch_extensions_jobs/$SLURM_JOB_ID` unless `ADAGS_TORCH_EXTENSIONS_DIR` is explicitly set.

## Next Concrete M2 Steps

1. Recover state:
   - Read this file.
   - Read `refine-logs/EVENT_CROP_METHOD_TRACKER.md`.
   - Read `refine-logs/OVERNIGHT_STATUS.md`.
   - Read `refine-logs/EVENT_CROP_FIX_EVIDENCE.md`.
   - Read `research-wiki/event-crop-fix.md`, `research-wiki/query_pack.md`, `research-wiki/gap_map.md`, and relevant idea pages such as `research-wiki/ideas/event-causal-visibility-gaussians.md` and `research-wiki/ideas/motion-aware-densification-budget.md`.

2. Predeclare M2 before implementation:
   - Add an M2 section to `refine-logs/EVENT_CROP_METHOD_TRACKER.md`.
   - Define exact non-oracle cues: dynamic-mask boundary, flow valid/invalid or disagreement boundary, route0 residual/flicker only if computed outside frozen crop labels.
   - Define budget caps: max new/split Gaussians per scene or per candidate, max points, max iterations, and static ghost guard.
   - Define M2 outputs, logs, and PASS/FAIL gates before scoring.

3. Inspect implementation points:
   - Search for existing densification/splitting/pruning code in `main.py`, Gaussian model classes, and optimizer setup.
   - Inspect `utils/motion_prior_utils.py` and the M1 candidate-local ROI loss path.
   - Decide whether M2 should use:
     - a new training config that boosts densification near event boundaries;
     - a candidate-boundary mask used inside existing densification logic;
     - a small post-resume micro-densification phase with strict point budget.

4. Build a small deterministic boundary-support artifact:
   - Input: route0 eval folders, masks, flow sidecars, dynamic/static frames, and possibly R020 candidates.
   - Output: a manifest of event-boundary support maps or boxes that explicitly records `uses_frozen_window_labels=false`.
   - Smoke-test on a tiny/local or dry-run path if possible.

5. Implement M2 narrowly:
   - Prefer extending existing config/script patterns instead of a large refactor.
   - Add metadata logging: method name, commit hash, config, support manifest, checkpoint paths, point budget, and output folders.
   - Do not overwrite R023/R024 outputs; use a new run/config name.

6. Verify locally/remotely before full runs:
   - `python -m py_compile` for edited Python files.
   - `bash -n` for Slurm wrappers on Leonardo if local bash is unavailable.
   - Run dry-runs/manifest validation before submitting.

7. Submit M2 on HPC:
   - Pull the pushed branch on Leonardo.
   - Submit train jobs through Slurm.
   - Poll with `squeue`/`sacct`, not fragile interactive loops.
   - Fetch `.out/.err` logs locally.

8. Evaluate scientifically:
   - Render complete Gaussian eval folders.
   - Augment `refine-logs/hide_reveal_real_windows.json` with the M2 system using `augment-real-manifest-system` or a new equivalent helper.
   - Validate required systems: `route0`, M2 system, `matched_lifespan`, `residual_uncertainty`.
   - Run frozen-window scoring into a new R026/R027 directory.
   - Generate crop strips.
   - Run result-to-claim or experiment-audit style review before declaring PASS/FAIL.

## Known Blockers And Failed Attempts

- M1 is a scientific failure, not just an engineering failure.
- R018 detector selected top-band false candidates; R019 fixed this with motion support.
- R020 high recall reached only posthoc `3/5` overlap and is not clean enough to claim support quality.
- R021 timed out because concurrent jobs shared a PyTorch extension build root.
- R022 showed `leonardo_env.sh` could override the first extension-root fix; `scripts/run_leonardo.sh` now forces per-job extension roots.
- Learned LPIPS remains unavailable unless sidecar weights are locally available without compute-node network downloads.
- Confident-track identity switch metrics remain unavailable because R009 discovery found no track-confidence sidecars.
- Remote repo/worktree may contain many untracked logs and configs. Preserve user/project files and do not clean/reset.
- Local worktree has unrelated dirty/untracked files listed in the git status above; preserve them.

## Scientific Verification Rules

Use the same frozen R009 protocol unless a change is explicitly predeclared and justified.

Strict PASS gate from `EVENT_CROP_METHOD_TRACKER.md`:

- Method is checkpoint-backed or newly trained Gaussian-rendered output, not GT crop compositing.
- Method does not use R009 frozen event crops as test-time support.
- At least `3/5` frozen windows improve versus route0, matched_lifespan, and residual_uncertainty on both PSNR and L1/proxy-LPIPS.
- At least `3/5` frozen windows do not worsen static ghost versus route0.
- Mean PSNR improves over route0 by at least `+0.5 dB`.
- Mean L1/proxy-LPIPS improves over route0 by at least `-0.001`.
- Method recovers at least 25% of the oracle upper bound on either mean PSNR or mean L1 reduction:
  - PSNR fraction: `(method_psnr - 30.5021) / 11.2128`
  - L1 fraction: `(0.0148316 - method_l1) / 0.0121662`

FAIL gate:

- Complete method run worsens mean PSNR and L1/proxy-LPIPS versus route0, or passes fewer than `3/5` windows, after logs and outputs are valid.
- Method requires oracle event-crop labels at test time.
- Method cannot produce comparable Gaussian-rendered folders and only produces image composites.

Result hygiene:

- Do not massage metrics.
- Do not weaken claims to declare success.
- Do not tune thresholds on the frozen real windows unless explicitly creating a separately labeled calibration split.
- Keep oracle/perfect-crop results clearly separated from non-oracle method results.
- Mark exploratory diagnostics as exploratory.
- Preserve failed ideas as negative knowledge in `research-wiki/` and `refine-logs/`.

## Handoff Instruction For New Session

The new Codex session should continue as orchestrator for M2. It must recover from:

- `D:/adags/refine-logs/M2_HANDOFF.md`
- `D:/adags/refine-logs/EVENT_CROP_METHOD_TRACKER.md`
- `D:/adags/refine-logs/OVERNIGHT_STATUS.md`
- `D:/adags/refine-logs/EVENT_CROP_FIX_EVIDENCE.md`
- relevant `research-wiki/` files
- the previous chat only if needed

Continue until M2 is scientifically completed or blocked after three serious fix attempts. Make small commits and push after meaningful milestones.
