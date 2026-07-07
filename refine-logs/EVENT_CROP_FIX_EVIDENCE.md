# Event-Crop Fix Evidence Summary

Generated: 2026-07-07

## Objective

Find a non-oracle Gaussian-model method that produces the event-crop fix suggested by the R001-R017 results, without using oracle event-crop information at test time and without tuning on the frozen real windows.

Completion standard:
- PASS: a non-oracle method improves the frozen windows and recovers a meaningful fraction of the oracle event-crop upper bound.
- FAIL: reasonable non-oracle attempts do not improve over baselines, with evidence.
- BLOCKED: missing permissions/data/compute or the same unresolved implementation blocker persists after three serious fix attempts.

## Prior Thread

Read Codex thread `019f34bf-83e1-7191-b3b6-64dc6bf3f06e`.

Key retained point: the previous "worked" result was the R013/R015 derived upper-bound composite. It pasted GT pixels into the predeclared event crops and therefore answered only "would a perfect crop fix help?" R017 then tested a real checkpoint-backed Gaussian-rendered opacity gate and failed on all five frozen windows.

## What R001-R017 Established

- The early R001-R008 tracker rows remain TODO in `refine-logs/EXPERIMENT_TRACKER.md`, but the synthetic PoC artifacts exist under `refine-logs/hide_reveal_poc/synthetic/` and are the actual source for the synthetic gate claims.
- Synthetic heldout fixture: `n=40`, candidate recall `1.000`, margin AUC `1.000`, accepted precision/recall `1.000/1.000`, identity reconnection `1.000`, matched-lifespan identity reconnection `0.000`, false event rate `0.000`.
- Frozen synthetic params: `C_min=0.55`, `m_event=0.02`, `lambda_id=1.0`, `lambda_static=0.5`, `lambda_budget=0.05`, `support_radius=5.5`.
- R009 froze five real event windows before scoring in `refine-logs/hide_reveal_real_windows.json`.
- R010 established the route0 smooth-transport baseline on those windows.
- R011 established the residual/uncertainty baseline; it was worse than route0 on mean PSNR, L1/proxy-LPIPS, flicker, and static ghost.
- R012 established a matched-lifespan image-level baseline; it was worse than route0 on PSNR/L1, had nearly tied flicker, and did not improve static ghost.
- R013 established a derived full hide/reveal upper bound; it used GT only inside the frozen event crops and strongly improved event-window metrics.
- R015 packaged the evidence into tables and crop strips. The synthetic fixture passed perfectly, but that is a controlled fixture/smoke result, not proof of real generalization.
- R016 correctly rejected paper-scale validation because C3 was not proven by an actual Gaussian/checkpoint-backed method.
- R017 implemented and ran an actual checkpoint-backed runtime opacity gate through the Gaussian renderer. It did not use GT crop compositing, but it damaged all five frozen windows and failed the strict gate.

## Oracle / Perfect Event-Crop Result

The oracle-style result is R013 derived hide/reveal:

| System | n | Mean PSNR | Mean L1/proxy-LPIPS | Mean flicker | Mean static ghost |
| --- | ---: | ---: | ---: | ---: | ---: |
| route0 | 5 | 30.5021 | 0.0148316 | 0.00799083 | 0.127333 |
| derived hide_reveal | 5 | 41.7149 | 0.00266536 | 0.00168586 | 0.127333 |

Oracle upper-bound deltas versus route0:
- PSNR: `+11.2128`
- L1/proxy-LPIPS: `-0.0121662`
- Flicker: `-0.0063050`
- Static ghost: `0.0`

Interpretation: if the event crop were perfectly repaired, the frozen windows would improve dramatically. This is an upper bound and not a deployable method because `derived_poc_metadata.json` records `is_trained_model_output=false` and the hide/reveal row uses GT inside the event crops.

## Actual Gaussian Method Result So Far

R017 actual checkpoint-backed opacity gate:

| System | n | Mean PSNR | Mean L1/proxy-LPIPS | Mean flicker | Mean static ghost |
| --- | ---: | ---: | ---: | ---: | ---: |
| actual_hide_reveal | 5 | 19.3667 | 0.0761056 | 0.0162899 | 0.152789 |
| route0 | 5 | 30.5021 | 0.0148316 | 0.00799083 | 0.127333 |
| matched_lifespan | 5 | 29.8181 | 0.0163546 | 0.00795601 | 0.127333 |
| residual_uncertainty | 5 | 30.0734 | 0.0165723 | 0.00803902 | 0.145702 |

Strict gate result: `0/5` windows passed. R017 worsened PSNR, L1/proxy-LPIPS, flicker, and static ghost. The metadata confirms `is_checkpoint_backed_inference=true`, `uses_gaussian_renderer=true`, `uses_gt_pixels_in_render=false`, and `newly_trained_checkpoint=false`.

Important diagnosis: R017 used the predeclared R009 event crop as support for the runtime gate and selected tens of thousands of Gaussians per frame with opacity scales near `0.05` at event centers. It tested that a broad crop-level opacity attenuation is not enough. It is also not a valid final non-oracle method for the new objective because the new objective requires no oracle event-crop information at test time.

## What Remains Unsolved

- A non-oracle method must discover event support from model/data cues rather than consuming the R009 crop as an input.
- The Gaussian model must produce comparable render folders; image-level GT crop compositing is not acceptable.
- The method must improve the frozen windows against route0, matched-lifespan, and residual/uncertainty baselines without static ghost degradation.
- R017 shows that simply hiding visible dynamic Gaussians can remove content but does not synthesize or reconnect the revealed surface.
- Learned LPIPS and confident-track ID switches remain unavailable in the current real-window evaluator. L1/proxy-LPIPS, PSNR, flicker, static ghost, gate stats, and qualitative strips are the current source metrics.

## Frozen Windows: Source of Truth

Frozen manifest: `D:\adags\refine-logs\hide_reveal_real_windows.json`

The five frozen R009 windows are:

| Window ID | Scene | Frames | Crop xyxy | Occluder |
| --- | --- | --- | --- | --- |
| `cut_roasted_beef_hand_tongs_meat_095_110` | `cut_roasted_beef` | 95-110 | `[245,315,455,470]` | hands, tongs, and knife over sliced meat |
| `cut_roasted_beef_hand_knife_meat_140_155` | `cut_roasted_beef` | 140-155 | `[245,315,455,470]` | hand and knife over sliced meat |
| `flame_steak_torch_pan_155_170` | `flame_steak` | 155-170 | `[250,300,445,460]` | torch flame over steak and pan |
| `flame_steak_torch_sweep_195_210` | `flame_steak` | 195-210 | `[250,300,445,460]` | torch flame sweep over steak |
| `sear_steak_spoon_pan_220_235` | `sear_steak` | 220-235 | `[245,320,450,470]` | spoon and hand over steak |

The crops are evaluation windows only for future non-oracle methods. They must not be used as test-time method inputs.

## Source-of-Truth Artifacts

Local metrics and reports:
- Synthetic summary: `D:\adags\refine-logs\hide_reveal_poc\synthetic\synthetic_summary.json`
- Synthetic report: `D:\adags\refine-logs\hide_reveal_poc\synthetic\synthetic_report.md`
- Synthetic frozen params: `D:\adags\refine-logs\hide_reveal_poc\synthetic\frozen_params.json`
- R010 route0 metrics: `D:\adags\refine-logs\hide_reveal_poc\r010_route0_real_eval\real_event_window_metrics.csv`
- R010 route0 summary: `D:\adags\refine-logs\hide_reveal_poc\r010_route0_real_eval\real_event_window_summary.json`
- R011 residual metrics: `D:\adags\refine-logs\hide_reveal_poc\r011_residual_uncertainty_real_eval\real_event_window_metrics.csv`
- R011 residual summary: `D:\adags\refine-logs\hide_reveal_poc\r011_residual_uncertainty_real_eval\real_event_window_summary.json`
- R012/R013 derived metrics: `D:\adags\refine-logs\hide_reveal_poc\r012_r013_derived_real_eval\real_event_window_metrics.csv`
- R012/R013 derived summary: `D:\adags\refine-logs\hide_reveal_poc\r012_r013_derived_real_eval\real_event_window_summary.json`
- R012/R013 derived metadata: `D:\adags\refine-logs\hide_reveal_poc\r012_r013_derived_real_renders\derived_poc_metadata.json`
- R015 decision bundle: `D:\adags\refine-logs\hide_reveal_poc\r015_poc_summary\poc_decision_inputs.json`
- R015 crop strips: `D:\adags\refine-logs\hide_reveal_poc\r015_poc_summary\crop_strips\`
- R016 go/no-go memo: `D:\adags\refine-logs\hide_reveal_poc\r016_go_no_go_memo.md`
- R017 actual-method report: `D:\adags\refine-logs\hide_reveal_poc\r017_actual_method_report.md`
- R017 actual metrics: `D:\adags\refine-logs\hide_reveal_poc\r017_actual_real_eval\real_event_window_metrics.csv`
- R017 actual summary: `D:\adags\refine-logs\hide_reveal_poc\r017_actual_real_eval\real_event_window_summary.json`
- R017 actual metadata: `D:\adags\refine-logs\hide_reveal_poc\r017_actual_real_renders\actual_render_metadata.json`
- R017 actual manifest: `D:\adags\refine-logs\hide_reveal_poc\r017_actual_real_renders\actual_real_windows_manifest.json`

Local Slurm submission manifests:
- R010: `D:\adags\refine-logs\hide_reveal_poc_real_jobs_20260706_022851.tsv`
- R011: `D:\adags\refine-logs\hide_reveal_poc_real_jobs_20260706_023817.tsv`
- R012/R013: `D:\adags\refine-logs\hide_reveal_poc_derive-real-renders_jobs_20260706_024307.tsv`
- R017 render: `D:\adags\refine-logs\hide_reveal_poc_actual-real-renders_jobs_20260707_015205.tsv`
- R017 eval: `D:\adags\refine-logs\hide_reveal_poc_real_jobs_20260707_020138.tsv`

Remote logs:
- R010 stdout/stderr: `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/logs/hide_reveal_real_48653179.out`, `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/logs/hide_reveal_real_48653179.err`
- R011 stdout/stderr: `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/logs/hide_reveal_real_48653948.out`, `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/logs/hide_reveal_real_48653948.err`
- R012/R013 stdout/stderr: `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/logs/hide_reveal_derive-real-renders_48654171.out`, `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/logs/hide_reveal_derive-real-renders_48654171.err`
- R017 render stdout/stderr: `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/logs/hide_reveal_actual-real-renders_48760029.out`, `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/logs/hide_reveal_actual-real-renders_48760029.err`
- R017 eval stdout/stderr: `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/logs/hide_reveal_real_48760448.out`, `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/logs/hide_reveal_real_48760448.err`

Remote render folders:
- Route0 source renders for `cut_roasted_beef`: `/leonardo_scratch/fast/EUHPC_D21_034/proj_adags/runs/fixed_budget_lora_route0_600k/20260701_012706_cut_roasted_beef_fixed_budget_lora_route0_600k/test/ours_6000/renders`
- Route0 source renders for `flame_steak`: `/leonardo_scratch/fast/EUHPC_D21_034/proj_adags/runs/fixed_budget_lora_route0_600k/20260701_012711_flame_steak_fixed_budget_lora_route0_600k/test/ours_6000/renders`
- Route0 source renders for `sear_steak`: `/leonardo_scratch/fast/EUHPC_D21_034/proj_adags/runs/fixed_budget_lora_route0_600k/20260701_012714_sear_steak_fixed_budget_lora_route0_600k/test/ours_6000/renders`
- Derived hide/reveal renders: `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/refine-logs/hide_reveal_poc/r012_r013_derived_real_renders/derived_renders/hide_reveal/`
- Derived matched-lifespan renders: `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/refine-logs/hide_reveal_poc/r012_r013_derived_real_renders/derived_renders/matched_lifespan/`
- R017 actual renders: `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/refine-logs/hide_reveal_poc/r017_actual_real_renders/actual_renders/actual_hide_reveal/`

HPC roots:
- Repo: `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags`
- Logs: `/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/logs`
- Data: `/leonardo_work/EUHPC_D21_034/proj_adags/data/n3v`
- Runs: `/leonardo_work/EUHPC_D21_034/proj_adags/runs`

## Next Decision

Do not rerun R017 as-is. The next method must be non-oracle: detect candidate event support from residual, mask, flow, visibility, or training-loss cues, then update or refine Gaussian state so the renderer produces the corrected crop. The frozen R009 crops remain evaluation-only.
