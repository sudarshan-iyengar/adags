## ARIS Codex Research Workflow

This project has ARIS Codex skills installed under .agents/skills.

Useful invocations:
- /idea-discovery "research direction"
- /experiment-bridge
- /research-pipeline "research direction"
- /auto-review-loop "paper and experiment results"

HPC rules:
- Do not run heavy CPU/GPU jobs on the login node.
- Create scheduler scripts under scripts/ or jobs/.
- Submit long-running work through the cluster scheduler, for example sbatch.
- Store logs under logs/ with job ids in the filename.
- Keep run metadata, configs, and outputs organized enough for Codex to inspect and iterate.

## Research Wiki Memory

Use `research-wiki/` as the project LLM wiki and durable research memory.

Before proposing new research directions, read `research-wiki/query_pack.md`, `research-wiki/gap_map.md`, and relevant pages under `papers/`, `ideas/`, `experiments/`, and `claims/`.

When literature is reviewed, ingest important papers into `research-wiki/papers/` and record relationships in `research-wiki/graph/edges.jsonl`.

When experiments finish, summarize the result as an `experiments/` page and connect it to the idea or claim it supports, partially supports, or invalidates.

Failed or weak ideas must be preserved in the wiki; they are useful negative knowledge for future ideation.

## Obsidian Wiki Usage

The project root `D:\adags` can be opened as an Obsidian vault. Use `research-wiki/Home.md` as the human-facing home page.

Maintain both layers:
- Obsidian wikilinks for human navigation and graph view.
- ARIS structured files for machine-readable research memory.

When adding notes manually, prefer pages under `research-wiki/papers/`, `research-wiki/ideas/`, `research-wiki/experiments/`, and `research-wiki/claims/`. Preserve ARIS frontmatter fields where present.
