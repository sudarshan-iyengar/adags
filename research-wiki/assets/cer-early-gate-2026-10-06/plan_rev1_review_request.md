# Review request: implementation PLAN for "CER on Multi4D, early gate" (plan review, round 1)

You are an independent adversarial reviewer (scientific + engineering). Review the PLAN below before any code is written.
State at the very top whether you could open the code files (quote one line from D:\adags-cer\multi4d\scene\cer.py to prove it).

Code you can read (workspace D:\):
- The CER Multi4D code at the base commit: D:\adags-cer\multi4d (branch cer @ 00357aa): train.py, scene\cer.py,
  utils\cer_gate.py, scene\gaussian_model.py, gaussian_renderer\__init__.py, arguments\__init__.py, arguments\dynerf.py,
  test_cer.py, scene\__init__.py, diff_gaussian_rasterization_hybrid\ (CUDA).
- The existing spec: D:\adags\research-wiki\operations\cer-multi4d-spec-2026-09-30.md (sections 1-8 and 13-14).
- The evidence that motivates the plan: D:\adags\research-wiki\operations\ars-fast-path-plan-2026-09-23.md (log entries
  dated 2026-10-06), D:\adags\research-wiki\assets\cer-wave1-montages-2026-10-06\substrate_debate_2026-10-06.md and the
  m4d_banks_*.jpg images there.
- Project rules: D:\adags\AGENTS.md.

Your job:
1. Check every code reference and factual claim in the plan against the code (line numbers, call order, schedule
   windows, which functions change row counts, RNG use, what the lift does, what the gate does in each render path).
2. Find anything the plan misses that would make labels and rows disagree, let a memory row (or a copy of one) paint a
   gap frame, break legacy-mode equivalence, consume extra RNG, interfere with the running jobs (worktree repo/multi4d-cer
   and the extensions directory build/torch_extensions_multi4d_cer on Leonardo), or make a test vacuous.
3. Judge the scientific design: is "begin at fine 3000 + propagate + periodic recompute relabel + lift exclusion" the
   simplest change that tests the stated hypothesis (the gap deficit is a schedule artefact: the revealed background gets
   too little training/capacity and the transient bank learns an occluder)? Are D3 (recompute vs monotone) and D1
   (3000 vs fine start) right? Is anything here an unjustified extra component?
4. Judge the tests: does each one fail if the thing it guards is broken? Is anything important untested?

Output format: a verdict line (READY / READY WITH FIXES / MATERIAL DEFECT), then numbered findings, each with severity
(critical / major / minor), the evidence (file:line), and the concrete fix. Be specific and terse. Do not restate the plan.

=============================== THE PLAN ===============================
# CER on Multi4D, early gate: implementation plan (rev 1, 2026-10-06)

Status: **plan for review**. Nothing is written or run.

- **Approved by the principal:** writing and reviewing the code (plan log, "2026-10-06 evening").
- **Not approved:** any training submission. The principal reviews the code and approves submission.
- **Pipeline:**
  1. Opus (orchestrator) writes this plan.
  2. Codex (gpt-6.1-sol) reviews it until there is consensus.
  3. A Sonnet 5.5 agent writes the code and tests.
  4. Codex reviews the code (scientific, engineering, tensor-shape, CUDA) until the orchestrator is satisfied.

Base: branch `cer` at `00357aa` (spec [cer-multi4d-spec-2026-09-30](cer-multi4d-spec-2026-09-30.md), sections 1–14).
References below are `T:` = `train.py`, `GM:` = `scene/gaussian_model.py`, `R:` = `gaussian_renderer/__init__.py`, `C:` = `scene/cer.py`,
`G:` = `utils/cer_gate.py`, all at `00357aa` (local clone `D:/adags-cer/multi4d`).

## 1. Why

On the three same-pose ARS scenes, Multi4D+CER passes its causal test (gap +4.25 dB, return −0.01 dB) but renders the gap
at 32.9 / 34.5 / 35.3 dB.

- **The benchmark.** STG renders the gap at 44.9 / 47.6 / 43.2 dB.
- **Bank renders (2026-10-06).** The renders are bit-exact to the saved model, and they explain the shortfall:
  - Base Multi4D keeps the object in its static background bank and paints a transient occluder over it during the gap.
  - CER removes the object from the persistent banks, but the occluder fragments survive.
  - Even with the transient bank removed (at render time only), the gap is 31–36.5 dB, against 44.4 dB for the same
    model's ordinary static background.
- **Working hypothesis.** The gap deficit is a schedule artefact of the port:
  - The gate starts at fine 16,000 (`C:147`, `T:996`), when every densify, prune, lift and reset has stopped. CER then
    freezes all lifecycle calls (`C:171-174`).
  - So the background behind the object gets its first unoccluded gradient with 4,000 steps left and no row can be added.
  - Meanwhile the transient bank has had 13,000 iterations (merging at 3,000) to learn an occluder.
- **An open alternative.** A schedule probe (3 runs, fine iterations 30,000, gate still at 16,000) is running now. It can
  separate "too few iterations" from "too few rows or the learning-rate floor" only in one direction (plan log
  2026-10-06 evening).

**The change, one factor: when the gate starts.** The memory gate starts early in the fine stage (default proposal: fine
3,000 = `transient_merging_iter`, before the transient bank enters the loss `T:393` and before the lift starts `T:916-918`).
Labels are carried through every row-changing operation until densification ends. The memory set is re-evaluated
periodically with the unchanged rule. Everything else stays as on `cer`:

- the gate function, the five render sites;
- the instrument, the residual birth, the gates;
- persistence, and the 20,000-iteration budget.

**Predictions, stated before any code exists (descriptive, for the later design freeze):**

1. The transient share of the gap region at the end falls well below the 16k arm's 0.18–0.32.
2. Persistent-alone gap PSNR on cam00 rises toward the ordinary-background level (44 dB).
3. The return window is unchanged within the noise floor (≈ 0.35 dB).

## 2. Design decisions (each one for the reviewer to accept or reject)

**D1. Begin iteration.**

- New argument `cer_begin_iter` in `ModelParams` beside `cer_scene` (`arguments/__init__.py`). Default `-1` means
  `densify_until_iter`, the legacy behaviour.
- Early arm: `3000`.
- **Valid range, asserted:** `1 ≤ cer_begin_iter ≤ densify_until_iter`. With the reference config it may be ≤
  `phase3_start_iter`; the legacy assertion `iteration ≥ phase3_start_iter` (`C:147`) applies only in legacy mode.
- **Why 3,000:**
  - the transient bank is not in the loss before 3,000 (`T:393`, `after_merge`);
  - the lift runs 3,000–10,000 (`T:916-918`);
  - background densification runs to 16,000 (`T:113`, `:133`).

  So the occluder has no reason to form, and the revealed background gets 13,000 iterations of densification.
- **Alternative: fine iteration 1.** More densification time, but labels are taken from a model with only coarse 2,000
  behind it. Not proposed: the labelling quality is unknown, and 3,000 is a schedule boundary of the substrate itself,
  not a tuned value.

**D2. Label propagation through every row-changing operation, by a minimal edit of the two persistent bank classes.**

The legacy spec avoided editing model classes; that is no longer possible, because propagation must know the
clone/split selection. The edits:

- **`GaussianModel_dynamic` (foreground) and `GaussianModel` (background):**
  - `densify_and_clone` (`GM:766`, `GM:1202`): new rows' labels = `cer_labels[selected_pts_mask]`.
  - `densify_and_split` (`GM:733`, `GM:1174`): `cer_labels[selected_pts_mask].repeat(N)`. That is the same ordering as
    every other `.repeat(N)` in the function, so child `k·S + j` inherits from selected parent `j`. The parent rows are
    removed by the following `prune_points(prune_filter)`, which filters labels.
  - `prune_points` (`GM:653`, `GM:1110`): `cer_labels = cer_labels[valid_points_mask]`.
  - All other row-count changes (`prune`, `weight_prune`, `densify`) go through these three.
  - The label tensor is passed to `densification_postfix` through a keyword argument defaulting to `None`. Each edit is
    a no-op when the model has no `cer_labels` attribute (`hasattr` check), which is every base run and every legacy
    run before `begin`.
- **Not edited:**
  - `cat_tensors_to_optimizer`, `_prune_optimizer` and `replace_tensor_to_optimizer`: they change optimizer state, not
    labels.
  - The opacity resets (`GM:570`, `GM:927`): they change no row count, so labels are unaffected. They keep running in the
    early arm as in the base run.
- **Invariant:** after every `maintain_*` call and inside every relabel, `cer_labels.shape[0] == get_xyz.shape[0]`,
  dtype `uint8`, on cuda. This is a cheap assertion in the CER path only. The gate (`G:24-41`) already asserts the shape
  at every render.
- **No RNG consumed by propagation:** indexing only, so the random draws of split (`torch.normal`) and of the opacity
  resets (`np.random.choice`) are unchanged.

**D3. Periodic relabel of labels {0, 1} with the unchanged memory rule (spec §4.1).**

- **When:** every `cer_relabel_interval` fine iterations after begin (new argument; default 1000; `0` = never), plus a
  final pass at `densify_until_iter`. Then labels freeze and the legacy sentinels are installed.
- **What a pass does:**
  - It recomputes `memory = (total > 0) & (inside ≥ 0.5·total)` over pre-gap frames × training cameras × target masks
    (the `C:150-151` computation) on the current gated model. Memory rows are on outside the gap, so the pre-gap
    masses do not depend on the gate.
  - It sets labels 0/1 accordingly. Label 2 (residual) is never changed, and a residual row is never relabelled memory:
    it has zero pre-gap mass because it is off outside the gap.
- **Recompute, not monotone growth.** A row that stopped painting the target becomes ordinary, so the labels describe the
  current model, and the 16k arm is exactly "one pass at 16,000". Monotone growth (once memory, always memory) is the
  alternative for the reviewer.
- **Logging:** each pass appends a line to `cer_relabel.jsonl` with iteration, rows per bank, memory rows per bank,
  flips 0→1 and 1→0 per bank, and memory share pre-gap.
- **Cost:** one pre-gap mass pass (≈ 1,140 renders with backward), estimated at ≈ 1 min from the legacy run's combined
  begin and gates time (UNMEASURED per pass). So ≈ 14 passes ≈ 15 min per run; E8 measures it. Frame subsampling is not
  proposed.

**D4. Lift exclusion.**

- The foreground-to-transient lift copies foreground rows into the ungated transient bank and leaves the donor alive
  (`T:923-936`, `GM:412-481`; query pack 2026-09-21 point 5).
- A labelled foreground row (memory or residual) must never be a donor: its copy would carry the object into the gap
  ungated.
- **Edit at the call site (`T:925-936`):** when CER is armed, multiply `deformed_state['dynamic_score']` by
  `(fg.cer_labels == 0)` before `select_fg_to_transient`. The threshold test (`GM:422-423`) then excludes them, and the
  random subsample (`GM:428-434`) draws only from eligible rows.
- **RNG caveat:** the subsample draws `randperm(current_count)`. If excluding rows changes `current_count`, the RNG stream
  changes. This is acceptable because it is part of the intervention, and it does not occur in legacy mode (begin after
  the lift window).
- **Unchanged:** transient rows are never labelled or gated (spec §1, principal decision (c)).

**D5. Residual cohort.**

- Born once, at begin, by the unchanged rule (spec §4.2; `C:155-170`).
- In the early arm the residual rows take part in densify, prune, weight-prune and reset like any background row. Their
  children inherit label 2.
- No other change, so the early arm differs from the 16k arm only in timing and in what timing forces (propagation,
  relabel, lift exclusion).

**D6. Sentinels.**

- **Legacy mode:** installed at begin (unchanged).
- **Early mode:** installed after the final relabel at `densify_until_iter`. Between begin and then, lifecycle methods run
  normally with propagation.
- `begin` still writes `cer_begin.json` with `O_EXCL` (NaN-restart protection), now with `begin_iter` and `mode`.

**D7. Evaluation, gates and persistence: unchanged.**

- `run_evaluation` at 3,500, 4,000, … scores the gated model, as at 16,500+ on the 16k arm.
- `cer_gates.json` keeps every field, plus `begin_iter`, `mode` and the last relabel's counts.
- `cer_state.pt` and the reload refusal are unchanged.
- The 20,000-iteration budget is unchanged.

**D8. No CUDA change.** The gate is a python multiply on the opacities (spec §8). The instrument kernel
(`-DCER_WRITE_MAX_CONTRIB`, `C:69-86`) is reused unchanged.

## 3. Isolation, git and builds

- **Local:**
  - `git -C D:/adags-cer/multi4d worktree add D:/adags-cer/multi4d-early -b cer-early cer`. That is a new branch from
    `cer@00357aa` in a separate local worktree.
  - `D:/adags-cer/multi4d` stays on `cer`, untouched.
  - Commits are narrow: code, then tests.
- **Leonardo:**
  - A bundle `cer..cer-early` is fetched into `repo/multi4d` (`git fetch <bundle> cer-early:cer-early`).
  - `git worktree add repo/multi4d-cer-early cer-early`.
  - `repo/multi4d` (pinned `c483e82`, detached) and `repo/multi4d-cer` (branch `cer`, used by the running probe jobs
    59534383–85) are never checked out, edited or pulled. A new branch and worktree in the shared `.git` change neither
    worktree's HEAD nor its porcelain, which the launchers check.
- **Extensions:**
  - A new `TORCH_EXTENSIONS_DIR=$W/build/torch_extensions_multi4d_cer_early` holds the hybrid rasterizer and the instrument
    build for this tree.
  - **Why mandatory, not optional:** torch's JIT `load` keys builds by name and source path. A same-name build from a
    different source path in the existing directory would rebuild into it and race the running jobs' lock.
  - No CUDA source changes, but the separate directory is required anyway, by that path-keying.
- **New launcher and test scripts:**
  - Copies of the existing ones with only the worktree, extensions directory, branch and commit pins changed:
    `agent-control/ars/sbatch/train_multi4d_cer_early.sbatch` and `cer_tests_m4d_early.sbatch`.
  - The existing scripts are not edited.
  - The scorer gets a new method `multi4d_cer_early` only at submission time, pinned to the reviewed commit.

## 4. Tests, written before the code (debug QOS, one GPU, the bottle same-pose base run 1 loaded as a stand-in model, as in spec §7)

| id | test | pass |
|---|---|---|
| E0 | existing suite T0–T6 on the new tree, default arguments (legacy mode) | all pass unchanged |
| E1 | legacy path unchanged | static: every new code path is reached only when `cer_begin_iter` ≠ default or `cer_labels` exists before the sentinels; with the default, `begin` at `densify_until_iter` produces the same `cer_begin.json` row counts, memory set and residual donors as `cer@00357aa` on the same loaded model (both trees in one job) |
| E2 | propagation exactness, both persistent banks | tag every row with a unique id in a tensor that clone/split copy exactly (e.g. `_rotation[:,0]`, on a deep copy); assign labels at random over {0,1,2}; run `densify_and_clone` and `densify_and_split` with forced gradients (known selection; both non-empty), then `prune_points` with a random mask, `prune`, `weight_prune`; for every surviving row, label == the label of the row its id came from; lengths equal |
| E3 | propagation neutral to the model | the same operations on two deep copies with the same RNG seed, one armed with labels and one without, give bit-identical parameters, optimizer moments and per-row buffers |
| E4 | lift exclusion | with labelled foreground rows given dynamic scores above threshold, the masked selection returns none of them; positive control: unlabelled rows above threshold are returned; legacy (unarmed) selection unchanged |
| E5 | relabel | on an unchanged model, a pass reproduces `begin`'s memory set exactly (idempotent); residual labels unchanged; flips counted correctly on a doctored model; log line written |
| E6 | sentinels | early mode: lifecycle methods run before the freeze and raise after it; legacy mode: raise right after begin (T6 as before) |
| E7 | exactness after lifecycle | after E2's operations at q = 1, memory mass in the gap and residual mass outside the gap are exactly 0 (instrument, training cameras) |
| E8 | integration (not evidence) | the real entry point through the final-render wrapper with a shortened schedule (orderings kept: begin < merging < phase3 < densify_until < iterations; values fixed in the test script) and early mode; exit 0; `cer_begin.json`, `cer_relabel.jsonl` (≥ 2 passes, flips reported) and `cer_gates.json` written; background rows changed after begin (non-vacuity); exact zeros in the gates; label lengths consistent at every pass |

E0–E1 pass on the unmodified tree; E2–E8 fail on it.

## 5. What the code reviewer must check (Codex, after the code exists)

- **Scientific:**
  - The gate is exact at every render path, from begin on.
  - No path lets a memory row paint a gap frame: the lift, splitting a memory row, or a relabel that drops a memory row
    while it still paints the target.
  - The relabel rule is the spec's rule, applied to the current model.
  - The early arm differs from the 16k arm only as stated.
- **Engineering:**
  - Labels and rows agree after every operation.
  - Label propagation uses no RNG (§2 D2).
  - Legacy mode is untouched.
  - The extensions directory and worktree are isolated.
  - O_EXCL is kept.
  - No change in the shared worktrees.
- **Tensor shapes:** `cer_labels` is `[N]` uint8 against the factor `[N,1]` (`G:28-34`); `.repeat(N)` ordering against
  `new_xyz`'s; masks are bool with the right lengths; dtype and device.
- **CUDA:**
  - Zero-opacity rows are skipped in forward and in the launched backward, so gated rows get no rendering gradient.
  - The densification statistics of gated rows in gap frames are zero, not NaN.
  - The instrument kernel swap is restored in a `finally` on every relabel.

## 6. After review (for the principal, not now)

**Proposed confirmatory design** (to freeze before any run):

- **Runs:** the early arm ×3 on each same-pose scene, against the same nine Multi4D base runs.
- **Endpoints:** the gap and return windows, Holm over 2.
- **Descriptive:** comparison with the 16k arm, the probe, the transient share, and persistent-alone renders.
- **Cost:** ≈ 9 × 3 h ≈ 27 GPU-h.

The smoke run E8 and the tests are not evidence.

**Out of scope:**

- gating the transient bank (decision (c), exploratory, later);
- pass-through and moved-return runs;
- any change to schedule constants.
