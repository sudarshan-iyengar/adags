---
type: experiment
id: r038-lora-route0-600k-9000-six-scene-launch
status: failed
date: 2026-07-22
related_idea: lora-route0-fixed-budget
---

# R038 LoRA Route0 600k 9000 Six-Scene Launch

Purpose: run a six-scene N3V fixed-budget LoRA route0 600k training sweep to 9000 iterations, with checkpoint-backed validation at 6000 and 9000.

Setup:

- Config: `configs/n3v/fixed_budget_lora_route0_600k_9000.yaml`.
- Scenes: `coffee_martini`, `cook_spinach`, `cut_roasted_beef`, `flame_salmon_1`, `flame_steak`, `sear_steak`.
- Run label: `fixed_budget_lora_route0_600k_9000`.
- Slurm jobs: `50050777` through `50050782`.
- Manifest: `refine-logs/n3v_lora_route0_600k_9000_train_jobs_20260722_003456.tsv`.

Result: FAIL. All six jobs exited with code `1:0` after about 2 minutes 42 seconds, before training started.

Failure signature:

```text
ModuleNotFoundError: No module named 'simple_knn'
```

Evidence:

- `sacct` reports all six top-level jobs and batch steps as `FAILED`.
- Per-scene output directories contain only `meta/`; no checkpoints, renders, or W&B metric histories were produced.
- The configured ADAGS Python environment cannot import `simple_knn` without additional package/path setup.

Interpretation: this launch produced no scientific metric result. It is an environment/launcher dependency failure, not evidence about LoRA 600k behavior at 9000 iterations.

Recommended recovery: fix the `simple_knn` import path or install/build the extension in the Leonardo ADAGS environment, then relaunch the same six-scene sweep with new run IDs after checking `squeue` and `sacct` for the failed job IDs.

## Relaunch After `simple_knn` Fix

Date: 2026-07-22.

After cleaning the repo-local `simple-knn` package path and validating `from simple_knn import distCUDA2` in Slurm job `50071552`, the six-scene training sweep was relaunched with fresh run IDs.

Relaunch manifest: `refine-logs/n3v_lora_route0_600k_9000_relaunch_train_jobs_20260722_102349.tsv`.

Relaunch jobs:

- `50073059`: `coffee_martini`
- `50073063`: `cook_spinach`
- `50073065`: `cut_roasted_beef`
- `50073067`: `flame_salmon_1`
- `50073071`: `flame_steak`
- `50073074`: `sear_steak`

Startup verification: all six jobs reached the training loop with no traceback and no `ModuleNotFoundError`. Observed progress shortly after launch was 70-90 / 9000 iterations depending on scene, so the original dependency failure was cleared.
