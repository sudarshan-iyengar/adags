# ADAGS Claude Code entrypoint

@AGENTS.md
@research-wiki/query_pack.md

Before research planning, implementation, job submission, or result analysis:

- read `research-wiki/gap_map.md`;
- read the relevant current objective, experiment, baseline, review, and prior-result pages;
- read `$WORK/proj_adags/agent-control/Objective.md` when it exists.

`AGENTS.md` is the authoritative project operating policy. Do not duplicate or override it here.
<!-- ARIS:BEGIN -->
## ARIS Skill Scope
ARIS skills installed in this project: 82 entries.
Manifest: `.aris/installed-skills.txt` (lists every skill ARIS installed and its upstream target).
For ARIS workflows, prefer the project-local skills under `.claude/skills/` over global skills.
Do not modify or delete files inside any skill that is a symlink (symlinks point into `/leonardo/home/userexternal/siyengar/aris_repo`).
Update with: `bash /leonardo/home/userexternal/siyengar/aris_repo/tools/install_aris.sh`  (re-runnable; reconciles new/removed skills).
<!-- ARIS:END -->
