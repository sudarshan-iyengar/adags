# CER hand-off (2026-10-01): both substrates built and tested, nothing submitted

For a fresh session continuing the CER (Complementary Episode Residuals) work of the ARS fast path. Read this page, then
the plan log [ars-fast-path-plan-2026-09-23](ars-fast-path-plan-2026-09-23.md) §4, §4b and the §9 entries from
2026-09-30 on, then the specs listed below. The principal's control rules stand: substrate code changes go
spec → two-model review → tests → code → two-model review → principal's diff review; **runs wait for "submit"**; every
job excludes `lrdn0066`; reports use plain words (codes only in a glossary).

**Glossary.** Memory cohort = existing rows that paint the target before it leaves, switched off inside the gap
(frames 60–89). Residual copies = gap-only copies of the donor rows. Donors = rows painting where the target was during
the gap once the memory rows are off. Same-pose / moved / pass-through = the three ARS event types (A0 / R2 / G1); bottle /
carton / mug = layouts L1 / L2 / L3. Smoke run (T7) = a shortened integration training, never evidence.

## 1. State

### Ex4DGS — ready for the principal's diff review and "submit"

- **Specs:** [cer-ex4dgs-spec-2026-09-30](cer-ex4dgs-spec-2026-09-30.md) (rev 2) and
  [cer-ex4dgs-spec-amendment-a-2026-10-01](cer-ex4dgs-spec-amendment-a-2026-10-01.md) (rev 2: residual copies are
  verbatim copies in the donor's own bank, dynamic copies keep their temporal opacity). Reviews of every round:
  `research-wiki/assets/cer-ex4dgs-2026-09-30/`.
- **Branch `cer`** (worktree `$W/repo/ex4dgs-cer`, at **5b82ae7**; pinned `$W/repo/ex4dgs` untouched at 7fac649):
  f37694c tests → cd5ef1e code → 73e9b65 tests → f71c1b0 code fixes → 7e3ccf7 tests (amendment) → 4a92369 code
  (amendment) → 319b1e4 tests → 5b82ae7 code nits. No CUDA change.
- **Diffs to review:** `assets/cer-ex4dgs-2026-09-30/cer_ex4dgs_7fac649..5b82ae7.diff` (whole change) and
  `cer_ex4dgs_f71c1b0..5b82ae7_amendment_a.diff` (amendment only).
- **Tests:** all pass on 5b82ae7 (job 59078863): q = 0 renders byte-identical to unmodified Ex4DGS on 300 frames with
  residual rows in both banks (12 static, 287 dynamic); exact factors, zero rendering gradient to the switched-off
  cohort; copies equal their donors in parameters and evaluated inputs (also for `begin()` driven to static-only,
  dynamic-only, mixed donors); birth state, checkpoint round trip, ply and missing-state refusals, pruning alignment,
  finite-opacity guard. Tests-only commits failed before their code (59050085, 59078802). T7 smoke passes (59078864/65).
- **Scoring is deployed:** `agent-control/ars/ars_score.py` method `ex4dgs_cer` accepts only commit **5b82ae7** on branch
  `cer` with a clean worktree at training and at render and render iteration 40,000, and copies `cer_gates.json` into the
  score; `ars_score_worker.py` lists the 27 CER runs (`runs/ex4dgs_cer/ars_w1_<scene>_cer_rep<r>`, job names
  `ars_w1_cer_<scene>_rep<r>`). Pre-CER copies: `ars_score.pre_cer.py`, `ars_score_worker.pre_cer.py`. **If the code
  changes again, move `EX4_CER_COMMIT`.**
- **Launcher:** `agent-control/ars/sbatch/train_ex4dgs_cer.sbatch` (export `TREE`, `CER_SCENE` (canonical
  `.../ars/wave1/<scene>`), `REP`, `RUN_ID`; render iteration read from the config; refuses an existing run dir;
  provenance at training and at render). Measured: a base run takes ≈ 1:10; CER adds `begin` + `gates` ≈ 10–15 min; use
  `--time=02:00:00`.

### Multi4D — port built and tested; the gate decision (below) still has to be written into spec and code

- **Spec:** [cer-multi4d-spec-2026-09-30](cer-multi4d-spec-2026-09-30.md) (rev 2 + rev-3 note §13). Census, lifecycle
  study and reviews: `research-wiki/assets/cer-multi4d-2026-09-30/`.
- **Branch `cer`** (worktree `$W/repo/multi4d-cer`, at **00357aa**; pinned `$W/repo/multi4d` untouched at c483e82):
  bf838c6 tests → c5ac11e code → 9f7f669 tests → **33215f1 the CUDA change** (a block under `-DCER_WRITE_MAX_CONTRIB`
  in the hybrid forward that writes `max_contrib`, compiled only into the instrument build) → c1e6045 instrument wiring →
  531c27b tests → 7d8c5e2 code fixes → 4261b1a tests → 00357aa tests. Diff: `assets/cer-multi4d-2026-09-30/
  cer_multi4d_c483e82..00357aa.diff`. The training kernel is unmodified in both arms.
- **Tests:** all pass — T0–T2 (59078492), T2b–T3 (59078553), T4–T4b (59078592), T5–T6 (59079481); T7 smoke passes
  (59078493/94). T4b confirms descriptively that the unmodified training build breaks conservation in a fresh process
  (upstream `max_contrib` defect) and the instrument build conserves.
- **Finding behind the principal's decision (b):** with the memory rows off, Multi4D's transient bank paints 0.88–0.96
  of the gap region; the background donors the spec copies are 6 / 0 / 1 rows on the three same-pose scenes
  (jobs 59078553, 59078606).
- **Launcher (exists, unused for evidence):** `agent-control/ars/sbatch/train_multi4d_cer.sbatch` (CER-only extensions
  directory `build/torch_extensions_multi4d_cer`, both builds checked at t = 0, provenance before and after).
- **Extension builds:** run `sbatch/cer_m4d_build.sbatch` once before any Multi4D CER test or training after a code
  change (it refuses if a build lock exists).

### Descriptive results already on the record (plan log, 2026-10-01 entries)

Montage of the smoke run against the base at frames 59 / 74 / 90:
`assets/cer-ex4dgs-2026-09-30/cer_t7_smoke_vs_base_montage.png`. Donors by bank on all nine scenes (mostly dynamic,
moving during the gap). Upstream pruning removes ≥ 44–77% of the dynamic memory rows over 30k–40k in the base arm too.
Membership forecast on the nine scenes: memory share 0.88–0.95.

## 2. Open items, in order

1. **Principal:** review the Ex4DGS diff (`7fac649..5b82ae7`), then "submit".
2. **Write the principal's 2026-10-01 gate decisions into both CER specs** (plan log, "principal's decisions on the
   contribution gates and on Multi4D"): (a) the residual share becomes descriptive (attribution between memory gate and
   residual copies), preconditions = memory ≥ 0.50 before and after + exact zeros; (b) Multi4D is a memory-gate arm whose
   gap-content gate counts residual or transient rows; (c) exploratory Multi4D work only after the confirmatory design is
   frozen, labelled exploratory. Then the small code changes: `gates()` `pass` no longer requires the residual share
   (Ex4DGS `scene/cer.py`, Multi4D `scene/cer.py`, which also adds the transient share to gap content). Each is a
   substrate change: spec amendment → both reviewers → tests first → code → both reviewers → tests → principal. **On
   Ex4DGS this moves the commit after 5b82ae7, so the scorer pin moves too** (decide with the principal whether to submit
   on 5b82ae7 first: `pass` is carried into the score but does not gate scoring, so the change only affects the
   reducer's reading).
3. **Multi4D scoring:** a method `multi4d_cer` in `ars_score.py` (provenance: commit, branch `cer`, porcelain before and
   after; the existing Multi4D checks; 40 evaluations in `debug_metrics.json`) and worker rows for the confirmatory runs
   (§4b: CER ×3 on the three same-pose scenes); wrapper `m4d_final_render.py` is reused with `--repo
   $W/repo/multi4d-cer`.
4. **The reducer:** the proof rule (§4 rule 1, Holm over six endpoints, the per-layout condition), the benchmark counts
   (rule 2), do-no-harm (rule 3), rule 4 with the amended preconditions, and the descriptive residual-share table at
   0.10 / 0.20 / 0.30. Must refuse to run on partial inputs, like `g_floor.py`.
5. **Wave-2 ablations, on hold until the wave-1 readout:** notch only (memory gate without residual copies); residual
   initialisation variants; start iteration 20k vs 30k; the 0.5 majority rule.
6. **On the principal's yes only:** the `max_contrib` diagnostic on the baselines (fraction of tiles skipped in a real
   training process at a few iterations; no reruns).

## 3. Where things live

- `W = /leonardo_work/EUHPC_D36_068/sri/proj_adags`. Worktrees `$W/repo/ex4dgs-cer`, `$W/repo/multi4d-cer`; pinned
  checkouts `$W/repo/ex4dgs`, `$W/repo/multi4d` (never modified; baseline jobs run from them).
- **Local clones for editing:** `D:/adags-cer/ex4dgs`, `D:/adags-cer/multi4d` (branch `cer`, `core.autocrlf false`).
  Read-only Multi4D clone for reference: `D:/adags-cer/multi4d-read`. Local masters of the launchers and helpers:
  `D:/adags-cer/agent/`.
- **Bundle-and-pull workflow (the worktrees have no push remote):** commit locally; `git bundle create ../<name>.bundle
  <pin>..cer`; `scp` it to `$W/agent-control/ars/cer/`; on Leonardo `git -C $W/repo/<x>-cer pull --ff-only <bundle> cer`.
  Never pull while a job runs from that worktree (provenance records the commit before and after). Backup bundles of the
  final branches: `D:/adags-cer/cer_ex4dgs_7fac649..5b82ae7.bundle`, `D:/adags-cer/cer_multi4d_c483e82..00357aa.bundle`.
- **Submission:** `source $W/agent-control/ars/ars_sub.sh; GO=--go sub NAME STEP DEP REASON <sbatch args> <file>`;
  writes `agent-control/ars/ledger.txt` and `jids/NAME.jid`, refuses a reused name. Ledger corrections are appended,
  never edited.
- **Launchers** in `$W/agent-control/ars/sbatch/`: `train_ex4dgs_cer.sbatch`, `cer_tests.sbatch` (Ex4DGS tests),
  `train_multi4d_cer.sbatch`, `cer_tests_m4d.sbatch`, `cer_m4d_build.sbatch`, `cer_diag.sbatch`,
  `cer_diag_forecast6.sbatch`; smoke config `configs/ex4dgs_cer_smoke.json`; helpers and outputs in
  `$W/agent-control/ars/cer/` (`cer_diag.py`, `chunk_sub.sh`, `ex4_steps.sh`, test outputs, diagnostics `diag/`).
- **Tests run from the worktree:** `python test_cer.py --run <model dir> --tree <tree> --cer_scene <scene> --pinned
  <pinned checkout> --out <dir> --tests T0,...` (via the sbatch files; `TESTS` is colon-separated there).
- **Reviewers:** `codex-review.ps1 -File <prompt> -Model gpt-6.1-sol -Effort high -Cwd <clone> -Skills none` (prompt
  and context inlined; Codex reads files only inside `-Cwd`); Fable 5.1 as a read-only subagent.

## 4. Traps learned on 2026-09-30 / 10-01

- **Stale JIT build locks.** A job killed while torch builds an extension leaves `<ext dir>/<name>/lock`; every later
  build waits on it forever (a test chunk timed out, a T7 sat stuck for 35 min). Build once in a dedicated job before
  any user; if a lock is found, confirm no live job holds it before removing it.
- **Never cancel a started job** to re-queue it faster; re-check the state in the same command as any `scancel`. A
  cancelled training may already have created its run directory, and the launchers refuse an existing one (new run id).
- **Debug queue:** 30-minute cap, at most 2 **submitted** jobs per user (pending ones count). Same partition and nodes
  as the normal queue, so results are comparable. Long queue waits on the normal queue are usually the saturated
  booster plus reservations (`sinfo`, `sprio`, `scontrol show reservation`), not a job defect.
- **`sbatch --export` splits on commas:** pass test lists colon-separated (`TESTS=T0:T1`), converted inside the sbatch.
- **Python file writes on Windows** convert `\n` to `\r\n` unless `newline=''`; check `git diff --stat` after any
  scripted edit. Git Bash heredocs containing `'''` can break; write patch scripts with the file tool. A shell command
  that reads stdin with no input hangs the tool.
- **Chained ssh submissions:** a failure mid-chain can leave a stale draft to run; submit one script per step and read
  back `squeue`/`scontrol` after.
- **Provenance of in-place edits:** `capture()` returns live references (deep-copy before comparing); `sbatch`
  snapshots its script at submission.
- **Certificates:** the CINECA user certificate lasts 12 h; check `ssh-add -L | ssh-keygen -L -f -` before long chains.

## 5. Uncommitted, and where

See the final message of the 2026-10-01 session; the wiki work of 2026-09-30 / 10-01 was committed in three commits on
`apollo/csvl-vpl-v2-exploratory` and is to be synced to `$W/repo/adags` by bundle once the certificate is renewed.
