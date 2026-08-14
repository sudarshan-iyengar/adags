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

## Apollo layout

Apollo is a separate Determined-AI-scheduled cluster. There is no SSH access
to any Apollo node; all access goes through the `det` CLI, run from the
workstation.

- Master: `determined.intern.denayer.be:8080` (user `sri`). Master/CLI
  version skew is expected, not a fault (see below).
- Pools: `hopper` (H100 PCIe) and `dgx` (V100 SXM2 32GB). Workspace `adags`,
  project `elgs`. Census/diagnostic cells run on `hopper`; a pool switch is a
  new ledger entry, never silent.
- Storage root inside containers: `/apollo/users/sri/proj_adags`, with
  `data/` (read-only raw), `runs/elgs/<run_id>/`, and `logs/`. `/apollo`
  automounts on both pools.
- Evidence-bearing runs reference the image by digest
  (`sudarshaniyengar/adags@sha256:...`), never by mutable tag.
- Code reaches the container only via the `git archive <commit>` context that
  `det e create` uploads. The shared worktree at
  `/apollo/users/sri/proj_adags/repo/adags` is never the execution source and
  is refused in-container.
- Full operational detail:
  `research-wiki/operations/apollo-determined-execution-authority.md`.

## Execution and Determined (Apollo)

- Submit and monitor from PowerShell, never Git Bash: Git Bash rewrites
  absolute `/apollo/...` arguments (e.g. into
  `C:/Program Files/Git/apollo/...`). Inspect the rendered entrypoint for a
  `Program Files` substring before submitting.
- `det` writes version-skew warnings to stderr on every successful call.
  Non-empty stderr, and PowerShell's `NativeCommandError`/`$?` wrapping of
  native stderr, are not failure signals on their own.
- Route all `det` invocations through `elgs/determined.py`; the `det e create`
  call in `scripts/submit_apollo.py` is the one accepted direct-subprocess
  exception. Do not add a second ad-hoc subprocess path.
- `scripts/submit_apollo.py` writes its O_EXCL cell claim before `--dry-run`
  returns, so every dry run consumes a retry index; expect real submissions
  to land at a later retry, and never delete or reuse a consumed claim.
- New entrypoints must be added to `ALLOWED_ENTRYPOINT_SCRIPTS`. Non-`main.py`
  entrypoints take their whole CLI from `--extra-arg` (values starting with
  `-` need the `--extra-arg=--flag` form) and do not receive a generated run
  dir.
- `slots_per_trial: 1` in `det_exp_apollo.yaml` means every cell, including
  CPU-bound ones, occupies a GPU slot; report slot-hours accordingly.
- Evidence-bearing runs need the exact pushed commit, digest-pinned image,
  content-hashed config, an O_EXCL claim, and an O_APPEND ledger line; code
  never executes from the shared worktree.
- Monitor only through the tracked `scripts/det_monitor.py`; never write an
  ad-hoc monitor with a broad `except` — that has previously produced silent
  no-state observations. Local `det` log streams cap at roughly 10 minutes
  and exit 255 while the remote task continues; that is benign.
- Scheduler completion is not scientific completion: verify the output
  artifact, hash it, and check the recorded commit/config/image before
  treating a run as done.
- Read-only inventory and small-artifact pulls from Windows use the `apollo:`
  rclone remote (`rclone lsd apollo:/apollo/users/sri/proj_adags`); pull
  multi-hundred-MB track artifacts deliberately, not routinely.

## Durable research protocol

- Read `research-wiki/query_pack.md`, `research-wiki/gap_map.md`, and the
  relevant current research pages before planning experiments or integration.
- Track durable scientific conclusions, negative results, engineering
  decisions, baselines, and deferred work in `research-wiki/`.
- Use `research-wiki/operations/` for durable operational and engineering
  decisions. Keep `agent-control/` for transient execution state, current
  checkpoints, job ledgers, and resumability only; never commit or delete it.
- `refine-logs/` may hold generated refinement artifacts and can remain
  ignored. Promote every durable conclusion or decision from it into the
  research wiki.
- Keep unmanaged Obsidian attachments under `research-wiki/attachments/`
  ignored. Store every figure referenced by a durable wiki page intentionally
  under tracked `research-wiki/assets/` or another tracked repository path, and
  commit the figure with the page that references it.
- Preserve failed, superseded, and defective experiments and artifacts.
  Distinguish exploratory runs from claim-grade experiments and record the
  applicable evidence boundary. Corrections to durable records are
  append-only; never rewrite historical prose.
- Independent recomputation means the reducer works from frozen text and
  primary inputs, never reading the implementation or its outputs first.

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
- `research-wiki/deep-dive-prompt.txt` and `run-deep-dive.ps1` are user-owned
  and untracked; never modify or stage them.
- Stage only explicit, reviewed paths; never `git add` a directory.

## ARIS skills

Use whichever ARIS skills apply to the task at hand. Before using a skill:

- read its complete `SKILL.md`;
- follow its instructions;
- do not modify or reinstall it;
- do not invoke it merely to demonstrate skill use.

## Operating rules

Act as soon as there is enough information to act. Do not re-derive facts
already established, re-litigate a decision without new evidence, survey
options that will not be pursued, or return a menu when a recommendation is
possible.

Delegate independent subtasks to subagents and keep working while they run.
Intervene when a subagent lacks relevant context or drifts from scope. Review
every subagent result before it influences the method or the durable research
record.

Prefer simplicity:

- Add no component without a concrete role in the method's central mechanism.
- Avoid complexity justified only by hypothetical future needs.
- Do not combine several weak ideas and present their bundle as novelty.
- If a simpler representation creates the same capability, prefer it.
- Distinguish essential representation from replaceable implementation
  machinery.

Before reporting progress, audit every claim against a tool result or
artifact from the current run. If something is unverified, say so. If a
search, deep-dive, review, or check failed or was skipped, report that
plainly; do not report a phase as complete merely because a document was
produced.

Operate largely unattended. Pause only when the work genuinely requires a
destructive or irreversible action, a real scope change, or information only
the user can provide. Otherwise proceed autonomously, and do not end a turn
with a promise, plan, question, or statement of intent when authorized work
remains.

## Surgical changes

Touch only what the task requires; clean up only what the current change
creates.

When editing existing code:

- Do not "improve" adjacent code, comments, or formatting.
- Do not refactor things that are not broken.
- Match existing style, even where a different choice would be preferred.
- If unrelated dead code is noticed, mention it — do not delete it.

When changes create orphans:

- Remove imports, variables, or functions that the change made unused.
- Do not remove pre-existing dead code unless asked.

Every changed line should trace directly to the request being fulfilled.

## Goal-driven execution

Define success criteria and loop until verified. Turn tasks into verifiable
goals:

- "Add validation" -> write tests for invalid inputs, then make them pass.
- "Fix the bug" -> write a test that reproduces it, then make it pass.
- "Refactor X" -> ensure tests pass before and after.

For multi-step tasks, state a brief plan of `[step] -> verify: [check]`
before starting. Strong success criteria allow independent looping; weak
criteria ("make it work") require constant clarification.

## Working style

- Give brief progress updates; lead with the outcome.
- Reuse existing modules; avoid speculative abstractions or redundant review
  layers.

<!-- ARIS-CODEX:BEGIN -->
## ARIS Codex Skill Scope
ARIS Codex packages installed in this project: skills-codex
Managed entries: 82
Manifest: `.aris/installed-skills-codex.txt`
ARIS repo root: `/leonardo/home/userexternal/siyengar/aris_repo`
Project skill path: `.agents/skills/<skill-name>`
For ARIS Codex workflows, prefer the project-local skills under `.agents/skills/`.
When a skill needs ARIS helper scripts, resolve the repo root from the manifest or set it explicitly:
`ARIS_REPO=$(awk -F'\t' '$1=="repo_root"{print $2; exit}' "/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags/.aris/installed-skills-codex.txt")`
Do not edit or delete symlinked skills in place; update upstream or rerun:
`bash /leonardo/home/userexternal/siyengar/aris_repo/tools/install_aris_codex.sh "/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags" --reconcile`
For copied Codex installs, use:
`bash /leonardo/home/userexternal/siyengar/aris_repo/tools/smart_update_codex.sh --project "/leonardo_work/EUHPC_D21_034/proj_adags/repo/adags"`
<!-- ARIS-CODEX:END -->
