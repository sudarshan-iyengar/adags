# ADAGS project operating rules

These instructions are repository-specific and intended to work across
platforms. Keep tracked paths repository-relative wherever possible. Never add
personal credentials, authentication material, private machine configuration,
or model preferences to this file.

## Repository boundaries

- This repository contains source code, maintained configurations, launchers,
  documentation, and durable research state.
- Treat raw datasets as read-only.
- Treat the separate `depth-anything-3` utility checkout as read-only.
- Store checkpoints and experiment outputs outside this repository.
- Do not commit credentials, raw scheduler logs, checkpoints, binary arrays,
  compiled extensions, caches, installed skills, or temporary verification
  artifacts.

## Leonardo layout

On Leonardo, `$WORK/proj_adags` is the authoritative project root:

- repository: `$WORK/proj_adags/repo/adags`
- environment setup: `$WORK/proj_adags/exp_index/leonardo_env.sh`
- raw N3V data: `$WORK/proj_adags/data/n3v`
- experiment runs: `$WORK/proj_adags/runs`
- scheduler logs: `$WORK/proj_adags/repo/adags/logs`
- durable research memory: `$WORK/proj_adags/repo/adags/research-wiki`
- generated refinement work: `$WORK/proj_adags/repo/adags/refine-logs`
- transient agent state: `$WORK/proj_adags/agent-control`
- read-only depth utility: `$WORK/proj_adags/repo/depth-anything-3`

On other systems, use repository-relative paths and explicit environment
variables instead of embedding private home directories or machine-specific
settings in tracked files.

## Execution and Slurm

- Login nodes are for lightweight inspection, orchestration, Git, scheduler
  commands, and static validation only.
- Submit training, evaluation, rendering, dataset generation, and substantial
  analysis through Slurm; never run them on a login node.
- Use the existing Leonardo environment setup unless a replacement is
  explicitly justified.
- Write experiment outputs under `$WORK/proj_adags/runs`.
- Write Slurm stdout and stderr under `logs/`, with `%j` in filenames.
- Capture every submitted job ID immediately. Before resubmitting an experiment
  ID, check both `squeue` and `sacct`.
- Prefer static parsing and consistency checks before runtime validation.

## Durable research protocol

- Read `research-wiki/query_pack.md`, `research-wiki/gap_map.md`, and the
  relevant current research pages before planning experiments or integration.
- Track durable scientific conclusions, negative results, engineering
  decisions, baselines, and deferred work in `research-wiki/`.
- Use `research-wiki/operations/` for durable operational and engineering
  decisions. Keep `agent-control/` for transient execution state, current
  checkpoints, job ledgers, and resumability only.
- `refine-logs/` may hold generated refinement artifacts and can remain
  ignored. Promote every durable conclusion or decision from it into the
  research wiki.
- Keep unmanaged Obsidian attachments under `research-wiki/attachments/`
  ignored. Store every figure referenced by a durable wiki page intentionally
  under tracked `research-wiki/assets/` or another tracked repository path, and
  commit the figure with the page that references it.
- Preserve failed and negative experiments. Distinguish exploratory runs from
  claim-grade experiments and record the applicable evidence boundary.

## Git and validation safety

- Integrate rescue or historical branches selectively; never merge them
  wholesale or rewrite them as snapshots.
- Never force-push, perform destructive cleanup, or discard unknown working-tree
  changes.
- Keep commits narrow and intentional. Verify `git status --short`,
  `git diff --stat`, and the relevant static checks before committing.
- New research-wiki pages must be tracked normally and must not require
  `git add -f`.
- Validate configuration syntax, launcher syntax, referenced paths, import
  surfaces, and artifact ignore behavior before requesting runtime experiments.
- Run shell probes and Git inspections sequentially on Leonardo to avoid
  exhausting sandbox namespace capacity.
