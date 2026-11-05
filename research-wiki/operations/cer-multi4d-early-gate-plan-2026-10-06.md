# CER on Multi4D, early gate: implementation plan (rev 4, final, 2026-10-06)

Status: **plan agreed (rev 4).** Codex round 3: "SOUND WITH QUALIFICATIONS, ready for implementation"; its clarifications are folded in (section 9). Nothing is written or run.

- **Reviews so far:** Codex gpt-6.1-sol returned MATERIAL DEFECT on rev 1 (8 findings) and on rev 2 (4 major findings).
  Each text and review is in [assets/cer-early-gate-2026-10-06/](../assets/cer-early-gate-2026-10-06/); the dispositions
  are in sections 7 and 8.
- **Approved by the principal:** writing and reviewing the code (plan log, "2026-10-06 evening").
- **Not approved:** any training submission. The principal reviews the code and approves submission.
- **Pipeline:**
  1. Opus (orchestrator) writes this plan.
  2. Codex (gpt-6.1-sol) reviews it until there is consensus.
  3. A Sonnet 5.5 agent writes the code and tests.
  4. Codex reviews the code (scientific, engineering, tensor-shape, CUDA) until the orchestrator is satisfied.

Base: branch `cer` at `00357aa` (spec [cer-multi4d-spec-2026-09-30](cer-multi4d-spec-2026-09-30.md), sections 1–14).

References below use these prefixes, all at `00357aa` (local clone `D:/adags-cer/multi4d`):

| prefix | file |
|---|---|
| `T:` | `train.py` |
| `GM:` | `scene/gaussian_model.py` |
| `R:` | `gaussian_renderer/__init__.py` |
| `C:` | `scene/cer.py` |
| `G:` | `utils/cer_gate.py` |
| `TC:` | `test_cer.py` |

## 1. Why

On the three same-pose ARS scenes, Multi4D+CER passes its causal test (gap +4.25 dB, return −0.01 dB). But it renders the
gap at 32.9 / 34.5 / 35.3 dB.

- **The benchmark.** STG renders the gap at 44.9 / 47.6 / 43.2 dB.
- **Bank renders (2026-10-06).** The full re-renders equal the saved final-render PNGs (0 differing 8-bit values on the 5
  checked frames of 18 runs). They show:
  - Base Multi4D keeps the object in its static background bank and paints a transient occluder over it during the gap.
  - CER removes the object from the persistent banks, but occluder fragments survive.
  - With the transient bank removed at render time only, the gap is 31–36.5 dB. The same model renders ordinary static
    background at 44.4 dB.
- **Working hypothesis.** The gap deficit is a schedule artefact of the port:
  - The gate starts at fine 16,000 (`C:147`, `T:996`), after every densify, prune, lift and reset has stopped. CER then
    freezes all lifecycle calls (`C:171-174`).
  - So the background behind the object first trains unoccluded with 4,000 steps left, and no row can be added.
  - Meanwhile the transient bank is in the loss from fine iteration 1: hybrid L1 at `T:747-748` and diversity at
    `T:784-785` before phase 3; hybrid SSIM joins after merging (`T:749-752`). So it has the whole fine stage to learn
    an occluder.
- **An open alternative.** A schedule probe is running now: 3 runs, fine iterations 30,000, gate still at 16,000. Its
  reading is one-directional (plan log 2026-10-06 evening).

**The change, one factor: when the gate starts.**

- The memory gate starts early in the fine stage.
- The memory and residual cohorts are chosen once, at begin, by the unchanged rules (spec §4.1, §4.2).
- Every row created afterwards inherits its parent's label through clone and split.
- Labels are filtered through every prune.
- Labelled foreground rows are never lift donors.

Everything else stays as on `cer`:

- the gate function, the five render sites;
- the instrument, the gates;
- persistence;
- the 20,000-iteration budget.

**Predictions, stated before any code exists (descriptive, for the later design freeze):**

1. The transient share of the gap region at the end is below the 16k arm's 0.18–0.32.
2. Persistent-alone gap PSNR on cam00 rises toward the ordinary-background level (≈ 44 dB).
3. The return window is unchanged within the noise floor (≈ 0.35 dB).

## 2. Design decisions

**D1. Begin iteration: a new argument, a forecast before it is fixed.**

- **The argument.** `cer_begin_iter` in `ModelParams` beside `cer_scene` (`arguments/__init__.py`).
  - **Plumbing.** It is stored on the scene by the scene constructor, exactly as `cer_scene` is (`scene/__init__.py:36`),
    because `scene_reconstruction` receives no model arguments. `begin`, the train-loop call site, the lift exclusion and
    the freeze read `scene.cer_begin_iter`. A test asserts that the value given on the command line reaches
    `scene.cer_begin_iter`.
  - **Legacy.** Default `-1` means `densify_until_iter`: legacy behaviour, the same code path as `cer@00357aa`.
  - **Early mode valid range, asserted in `training()` (`T:1009`) before the scene is built.** That function holds both
    the model and the optimisation arguments; the scene constructor receives the model arguments only
    (`scene/__init__.py:28-29`). The range: `1 ≤ cer_begin_iter ≤ transient_merging_iter`. That is
    no later than the start of the lift, so no lift transfer can precede begin; it rejects every value inside or after the
    lift window, including 10,000–16,000.
  - **Coarse-stage condition.** Early mode also asserts `coarse_iterations ≤ transient_merging_iter`. The lift has no stage
    guard (`T:916-918`) and runs in the coarse loop too, so a coarse stage longer than the merging iteration could lift
    rows before fine begin. This holds for the reference config (coarse 2,000, merging 3,000).
  - **The legacy assertion** `iteration ≥ phase3_start_iter` (`C:147`) applies in legacy mode only.
- **Proposed early value: 3,000** (`transient_merging_iter`).
  - It is a schedule boundary of the substrate, not a tuned value.
  - Hybrid SSIM joins the loss at 3,001 (`T:393`, strict `>`), and the lift at its first eligible cadence iteration after
    3,000 (`T:916-918`). A begin at the end of iteration 3,000 precedes both.
  - It leaves 13,000 iterations of background densification (`T:113`, `:133`) with the gate on.
  - Rev 1's claim that "the occluder has no reason to form" before 3,000 was wrong (`T:747-748`) and is withdrawn.
- **Forecast F1, before the value is frozen (debug QOS, not evidence).**
  - **Run:** bottle same-pose, the real config, CER in early mode with begin 3,000. Training stops shortly after begin: a
    launcher run with `--iterations 3500`. The value 3,500 is the first evaluation after begin; the final-render wrapper
    saves only at evaluation iterations (`T:991`; the wrapper renders at `opt.iterations` inside `run_evaluation`). It keeps
    every schedule window below 3,000 identical. `iterations` sets the fine loop bound (`T:1050`) and the end-of-training
    checks (`T:889`, `:952`, `:962`, `:975`); no schedule window reads it.
  - **Read from its `cer_begin.json`:**
    - the gap-region share by bank once memory is off;
    - the memory row count and memory share before the gap;
    - the residual donor count.
  - **Rule, written now:**
    - If the transient share of the gap region at begin is ≥ 0.50, an occluder has already formed by 3,000. The
      principal then chooses between begin 3,000 (as planned) and begin at fine iteration 1, with this number in hand.
    - Otherwise 3,000 stands.
    - If memory share before the gap is < 0.50 at 3,000, the labelling is too early to be meaningful, and the plan returns
      to review.

**D2. Label propagation, by a minimal edit of the two persistent bank classes.** Propagation must know the clone/split
selection, so the legacy spec's "no model class is edited" is given up.

- **Foreground `GaussianModel_dynamic` and background `GaussianModel`:**
  - `densify_and_clone` (`GM:766`, `GM:1202`): new labels = `cer_labels[selected_pts_mask]`.
  - `densify_and_split` (`GM:733`, `GM:1174`): `cer_labels[selected_pts_mask].repeat(N)`. That is the same ordering as
    every other `.repeat(N)` there, so child `k·S + j` inherits from selected parent `j`. The parents are removed by the
    following `prune_points(prune_filter)`, which filters labels.
  - `prune_points` (`GM:653`, `GM:1110`): `cer_labels = cer_labels[valid_points_mask]`.
  - `prune`, `weight_prune` and `densify` change rows only through these three.
- **How labels are passed and when the edit acts.**
  - Labels pass to `densification_postfix` through a keyword argument defaulting to `None`.
  - The edit is a no-op when the model has no `cer_labels` attribute: every base run, and every legacy run before begin.
    After a legacy begin, the sentinels make these methods unreachable.
- **Not changed:**
  - `cat_tensors_to_optimizer`, `_prune_optimizer` and `replace_tensor_to_optimizer`. They replace the parameter tensors
    with new `nn.Parameter` objects and maintain the optimizer state (`GM:140`, `:644`, `:691`). Labels are not parameters,
    so propagation happens in their callers.
  - The opacity resets (`GM:570`, `GM:927`). They change no row count and keep running in the early arm as in the base.
- **Invariant, asserted in the CER path:** after every `maintain_*` call, each persistent bank's `cer_labels` is a 1-D
  `uint8` cuda tensor with `shape[0] == get_xyz.shape[0]`. The gate (`G:24-41`) additionally asserts the shape at every
  render.
- **No RNG consumed by propagation:** indexing only.

**D3. Membership is immutable and inherited. No relabel (rev 1's periodic recompute is dropped).**

- **Why dropped.** Recomputing membership is an extra scientific factor, and it opens two leaks:
  - A demoted row becomes ungated and lift-eligible.
  - A row copied into the transient bank while ordinary, then promoted, leaves an ungated copy (Codex rev-1 finding 2).
- **What remains.** Immutable inherited membership is the minimal timing experiment. The early arm is then "the 16k arm's
  labelling, done at 3,000 and carried forward".
- **The known cost, measured rather than fixed.** A row born after begin from an ordinary parent that comes to paint the
  target is ungated in the gap.
- **A drift report at `densify_until_iter`, descriptive only, never acted on.**
  - It computes the memory rule on the current model and writes `cer_drift.json`. This is one pre-gap mass pass, the
    same computation as `C:150-151`, but no label is changed.
  - **Contents:**
    - the rows that now meet the rule but are labelled 0, per bank, and the pre-gap target mass they carry, as a share
      of all target mass;
    - the rows labelled 1 that no longer meet it, and their mass.
  - If drift turns out to matter, a reassignment arm is a separate, later design with its own ancestry rules.

**D4. Lift exclusion.**

- **The risk.** The foreground-to-transient lift copies foreground rows into the ungated transient bank and leaves the donor
  alive (`T:923-936`, `GM:412-481`; query pack 2026-09-21 point 5). A labelled foreground row must never be a donor.
- **At the call site (`T:925-936`), when CER is armed:**
  - Set `deformed_state['dynamic_score']` to `-inf` for rows with `fg.cer_labels != 0` (`masked_fill`) before
    `select_fg_to_transient`. This is an explicit eligibility mask: `-inf > threshold` is false for any finite threshold,
    including a negative one.
  - The threshold test (`GM:422-423`) then excludes labelled rows, and the subsample (`GM:428-434`) draws only from
    eligible rows.
- **RNG:** the `randperm(current_count)` stream may change, as part of the intervention. This cannot happen in legacy
  mode: the lift ends at 10,000 and legacy begin is 16,000.
- **One pre-existing exception, qualified rather than fixed.** At scene construction the transient bank is seeded with one
  row copied from the first foreground point (position and colour, `scene/__init__.py:95-103`). The seed is identical in
  the base, 16k and early arms. The guarantee is therefore "no labelled row is ever lifted", not "no transient row derives
  from the foreground".
- **Transfers before begin: none, by construction.**
  - The lift fires only at `iteration > transient_merging_iter` (`T:918`, strict `>`).
  - D1 restricts early begin to `≤ transient_merging_iter` and the coarse stage to `≤ transient_merging_iter`.
  - So no transfer, in either stage, precedes an early begin.
  - E9 also checks this at run time: it counts transfers before begin, which must be 0.
- **Unchanged:** transient rows are never labelled or gated (spec §1, principal decision (c)).

**D5. Residual cohort.**

- **Birth.** Born once, at begin, by the unchanged rule (`C:155-170`). Its children inherit label 2.
- **Then.** It takes part in densify, prune, weight-prune and reset like any background row.
- **Donor supply at 3,000 is unknown.** F1 reports it; the 16k arm had 2–7 donors.

**D6. Sentinels.**

- **Legacy mode:** installed at begin (unchanged).
- **Early mode:** installed at the end of fine iteration `densify_until_iter`, after the drift report.
- **`cer_begin.json`:**
  - is still written with O_EXCL;
  - gains `begin_iter` and `mode`;
  - keeps the existing `gap_region_share_by_bank` field, which is F1's reading.

**D7. Densification statistics of gated rows: documented, not changed.**

- In a gap frame a gated row still gets a geometric radius from preprocessing (`forward.cu:510`), so it counts as
  visible. `add_densification_stats` (`GM:114-117`) then increments its `denom` with zero rendering gradient.
- So gated rows' average gradient is diluted by the gap frames, about 10% of frames. That is a consequence of the gate
  and is not altered, since altering it would be a second intervention.
- Gated rows receive zero new **rendering-gradient** contribution in gap frames. Their statistics stay finite.

**D8. Evaluation, gates, persistence: unchanged.**

- `run_evaluation` scores the gated model from begin on.
- `cer_gates.json` keeps every field, adds `begin_iter` and `mode`, and its reading follows spec §14.2 (the stale `pass`
  field is never read).
- `cer_state.pt` and the reload refusal are unchanged.
- The 20,000-iteration budget is unchanged.

**D9. No CUDA change.**

- The gate is a python multiply (spec §8).
- The instrument kernel (`C:69-86`) is reused unchanged, and the training build is used for training as in the base.

## 3. Isolation, git and builds

**Local:**

- Branch `cer-early` from `cer@00357aa`, in the separate worktree `D:/adags-cer/multi4d-early`. It was created
  2026-10-06 and holds no edits.
- `D:/adags-cer/multi4d` stays on `cer`.
- Commits are narrow: tests first, then code.

**Leonardo:**

- A bundle `cer..cer-early` is fetched into `repo/multi4d` as branch `cer-early`, then
  `git worktree add repo/multi4d-cer-early cer-early`.
- These are never checked out, edited or pulled:
  - `repo/multi4d` (pinned `c483e82`, detached);
  - `repo/multi4d-cer` (branch `cer`, used by the running probe jobs 59534383–85).
- A new branch and worktree change neither worktree's HEAD nor its porcelain. Both launchers check those.

**Extensions:**

- A new `TORCH_EXTENSIONS_DIR=$W/build/torch_extensions_multi4d_cer_early` holds this tree's hybrid rasterizer and
  instrument builds.
- **Why separate.** Torch versions a JIT build by hashing source contents, build arguments and build directory. A
  rebuild in the shared directory would take that directory's lock and replace the artefact a running job loads.
- A separate directory keeps every build artefact of this tree apart from those used by running jobs, whatever the
  sources.

**Launcher and tests:**

- `agent-control/ars/sbatch/train_multi4d_cer_early.sbatch` is a copy of the `cer` launcher. It changes the worktree,
  the extensions directory and the branch and commit pin, and **passes `--cer_begin_iter $CER_BEGIN_ITER` (required, no
  default)**. It records `cer_begin_iter` in `provenance.txt`.
- `cer_tests_m4d_early.sbatch` is the equivalent test script.
- Existing scripts are not edited.
- **Scorer method `multi4d_cer_early`,** added at submission time. It pins the reviewed commit and requires:
  - `cer_begin.json` `mode == "early"`;
  - `begin_iter == 3000`, or the frozen value;
  - the full schedule.

## 4. Tests, written before the code

Every test runs on the debug QOS, on one GPU. The stand-in model is the bottle same-pose base run 1, loaded as in spec §7.
Every test with a stochastic operation populates nonzero Adam moments first (one optimizer step with small gradients, as
`TC:170`).

| id | test | pass |
|---|---|---|
| E0 | existing suite T0–T6 on the new tree, default arguments (legacy mode) | all pass |
| E1 | legacy path unchanged | (a) static: each new code path is reached only if `cer_begin_iter` is set or a model carries `cer_labels` before the sentinels; (b) on the same loaded model, `begin` with default arguments gives the same memory set, residual donors and `cer_begin.json` counts on the new tree as on `cer@00357aa` (both trees in one job) |
| E2 | propagation exactness, both persistent banks | on a deep copy, tag every row with a unique id in a tensor that clone and split copy exactly; labels random over {0,1,2}; `densify_and_clone` and `densify_and_split` with forced gradients so both selections are non-empty and known, then `prune_points` with a random mask, then `prune` and `weight_prune` with thresholds chosen to remove ≥ 1 row each; every surviving row's label equals its source row's label; lengths equal; ≥ 1 row of each label survives each step |
| E3 | propagation neutral to the model | the E2 sequence plus one `reset_opacity_partially` (BG) and one `reset_opacity_partially_small` (FG) on two deep copies from the same RNG states (torch CPU, every CUDA device, NumPy, Python `random`): one carries labels, one does not. Afterwards parameters, Adam moments and every per-row buffer are bit-identical, and so are all four RNG states |
| E4 | lift exclusion | labelled FG rows given dynamic scores above threshold are never selected; positive control: unlabelled rows above threshold are selected; unarmed selection is unchanged; the begin-range rule rejects 3,001–9,999 |
| E5 | drift report | on an unchanged model the report finds zero drift (the rule reproduces `begin`'s set); on a doctored model the counts and masses are correct and no label changes. **Doctoring:** move the positions of k known memory rows (background bank: `_xyz`) off the target silhouette, so they fall below the majority rule. Features cannot be used: the instrument replaces every bank's features with white (`C:103-106`). The expected drift is established independently first: the doctored rows' inside/total ratios are computed by the same mass pass and are asserted < 0.5, with k > 0 |
| E6 | sentinels | early mode: lifecycle methods run before the freeze and raise after it; legacy mode: they raise right after begin (T6) |
| E7 | exactness after lifecycle, non-vacuous | after E2's operations at q = 1, on the training cameras: memory mass before the gap > 0 and residual mass in the gap > 0 (witnesses survive); memory mass in the gap and residual mass outside it exactly 0 |
| E8 | training-build gradients | on the training (non-instrument) build, real backwards of an L1 render loss through `render_rawall3pc`. The memory witnesses are identified by the instrument **at q = 0** (gate factor 1) as memory rows with positive contribution in that gap view; at q = 1 their contribution is zero by construction. E8 runs in two fresh processes. Any failure or run-to-run disagreement goes to the principal as a substrate decision (the training build's backward reads the unwritten `max_contrib`, `backward.cu:1229`), not a code fix. **Same gap frame and view, q = 0 (gate on, factor 1):** the witnesses' rendering gradients (opacity, features) are nonzero; this is the positive control that the training build's backward reaches their buckets. **q = 1:** those gradients are exactly 0, and non-gated visible rows' gradients are nonzero. **Pre-gap frame, q = 1:** the witnesses' gradients are nonzero. **Densification statistics:** after each backward, `add_densification_stats` is called as `maintain_background` does; `denom` increments for visible rows, gated rows included (D7), and `xyz_gradient_accum` and `denom` stay finite |
| E9 | integration (not evidence) | real entry point through the final-render wrapper, early mode, shortened schedule with coarse ≤ merging, begin == `transient_merging_iter`, and the order begin = merging < phase3 < densify_until < iterations, so densification and the lift run after begin. Also: the command-line `cer_begin_iter` reaches the scene; 0 transfers before begin; begin values 0, merging + 1, 10,000 and densify_until, and coarse > merging, are each refused before training; exit 0; `cer_begin.json`, `cer_drift.json` and `cer_gates.json` written; background rows changed after begin; ≥ 1 lift after begin and no labelled donor; label lengths consistent; spec §14.2 preconditions (memory share ≥ 0.50 before the gap and after the return; exact zeros) |

- E0–E1 pass on the unmodified tree; E2–E9 fail on it.
- F1 (§2, D1) runs after the tests pass.

## 5. What the code reviewer must check (Codex, after the code exists)

- **Scientific:**
  - The gate is exact at every render path from begin on.
  - No path lets a memory row or a copy of one paint a gap frame: the lift, a split, or a begin inside the lift window.
  - Labels are chosen once by the spec's rules.
  - The early arm differs from the 16k arm only as stated.
- **Engineering:**
  - Labels and rows agree after every operation.
  - Propagation uses no RNG.
  - Legacy mode is untouched.
  - The extensions directory and worktree are isolated.
  - O_EXCL is kept.
  - The launcher passes `cer_begin_iter`.
- **Tensor shapes:**
  - `cer_labels [N]` uint8 against the factor `[N,1]` (`G:28-34`).
  - `.repeat(N)` ordering against `new_xyz`.
  - Bool masks of the right length.
  - Dtype and device.
- **CUDA:**
  - Gated rows get exactly zero rendering gradient on the training build (E8).
  - The instrument kernel swap is restored in a `finally`.
  - Densification statistics stay finite.

## 6. After review (for the principal, not now)

**Proposed confirmatory design, to freeze before any run:**

- **Runs:** the early arm ×3 on each same-pose scene, against the same nine Multi4D base runs.
- **Rule 1 as bound in spec §14.4:**
  - the gap improvement is Holm-significant over 2 endpoints;
  - the return window's one-sided 95% lower bound is > −0.25 dB;
  - the mean gap gain is positive in every layout.
- **Rule 3:** do-no-harm on the whole frame and the ordinary region (lower bound > −0.2 dB).
- **Descriptive:**
  - against the 16k arm and the probe;
  - the transient share;
  - drift;
  - persistent-alone renders.
- **Cost:** ≈ 9 × 3 h ≈ 27 GPU-h, plus F1 ≈ 0.3 GPU-h.

Tests, F1 and E9 are not evidence.

**Out of scope:**

- gating the transient bank (decision (c));
- membership reassignment;
- pass-through and moved-return runs;
- any change to schedule constants.

## 7. Rev 1 → rev 2 (Codex rev-1 findings)

| # | finding | disposition |
|---|---|---|
| 1 | the transient bank is in the loss before 3,000 (`T:748`) | accepted: the rationale is rewritten; 3,000 is the lift/SSIM boundary; forecast F1 with a pre-stated rule |
| 2 | recompute relabel and lift exclusion leak (demotion; promotion after transfer; begin inside the lift window) | accepted: immutable inherited membership (D3); the begin-range rule (D4); drift reported, not acted on |
| 3 | E3 could pass with broken moments or extra RNG | accepted: nonzero moments; resets exercised; four RNG states compared |
| 4 | E7/E8 vacuous zeros; §14.2 preconditions | accepted: witnesses (E7); §14.2 in E9 |
| 5 | the launcher would leave early gating off | accepted: `--cer_begin_iter` required, recorded, and validated by the scorer |
| 6 | gated rows' densification statistics are not zero | accepted: D7 documents it, no change |
| 7 | "bit-exact to the saved model" overstated | accepted: §1 wording |
| 8 | extension cache keying misdescribed | accepted: §3 wording |
| suspected | the relabel is an extra factor | accepted (D3) |
| suspected | early residual donor supply unknown | accepted: F1 reports it |
| missing | training-build gradients | accepted: E8 |
| missing | complete confirmatory criteria | accepted: §6 |
| missing | boundary coverage | accepted: E9 begin == merging; no relabel intervals remain |

## 8. Rev 2 → rev 3 (Codex rev-2 findings)

| # | finding | disposition |
|---|---|---|
| 1 | begin ≥ 10,000 still allowed after earlier transfers | accepted: early begin restricted to `1 … transient_merging_iter`; refusals tested in E9 |
| 2 | the lift has no stage guard; coarse could lift before fine begin | accepted: `coarse_iterations ≤ transient_merging_iter` asserted in early mode; E9 counts transfers before begin |
| 3 | E5's feature doctoring cannot change the white-feature mass | accepted: positions are moved; expected drift established first |
| 4 | E8 lacks a same-frame positive control on the training build | accepted: q = 0 control on identified witnesses at the same gap frame and view, then q = 1 zeros |
| suspected | a score multiplication fails for a negative threshold | accepted: explicit `-inf` eligibility mask |
| unverified | "`iterations` is read only by end-of-training checks" | corrected: it also bounds the fine loop (`T:1050`); no schedule window reads it |
| missing | plumbing of `cer_begin_iter` | accepted: stored on the scene as `cer_scene` is; tested in E9 |
| missing | E8 must exercise the densification statistics | accepted: `add_densification_stats` called; `denom` increments checked |

## 9. Rev 3 → rev 4 (Codex round 3: SOUND WITH QUALIFICATIONS, ready for implementation)

| # | item | disposition |
|---|---|---|
| 1 | "optimizer state only" is false for the three helpers | corrected (D2) |
| 2 | the 3,000 boundary imprecise | corrected: SSIM from 3,001; lift at the next eligible cadence (D1) |
| 3 | the transient seed copies one foreground point at construction | qualified (D4): the guarantee is "no labelled row is ever lifted"; the seed is common to every arm |
| 4 | E8 stability on the training build | E8 runs in two fresh processes; failure goes to the principal as a substrate decision |
| missing | E8 witnesses at q = 0 | stated (E8) |
| missing | where the range validation gets the optimisation arguments | in `training()` before scene construction (D1) |

Consensus reached at round 3. No further plan round. The code review follows the implementation.

## 10. Implementation, code review and tests (2026-10-06 evening; appended)

**Code.**

- **Branch.** `cer-early` from `cer@00357aa`.
  - Local worktree: `D:/adags-cer/multi4d-early`.
  - Leonardo worktree: `repo/multi4d-cer-early`, made by bundle fetch and `git worktree add`. `repo/multi4d` and
    `repo/multi4d-cer` were never checked out or edited; their states were recorded before and after every job.
- **Extensions:** `build/torch_extensions_multi4d_cer_early`. The E1 legacy subprocess builds into a fresh directory
  under its test output.

**Commits** (written by a Sonnet 5.5 agent unless marked):

| commit | content |
|---|---|
| `b842f41` | tests E1–E9, written before the code |
| `42e818c` | implementation |
| `51cd832`, `2f3bd6c` | fixes after code review 1 |
| `9a9a747`, `6b75e9c` | fixes after code review 2 |
| `96e92ee` | fix after code review 3 |
| `486b78a`, `15596d8` | E8 test fixes after the GPU run and code review 5 |
| `7b30ef8` (orchestrator) | E8 probe-agreement tolerance |

**The code has not changed since `96e92ee`;** every later commit touches `test_cer_early.py` only.

**Launchers** (local masters in `D:/adags-cer/agent/`, on Leonardo in `agent-control/ars/sbatch/`):

- `train_multi4d_cer_early.sbatch` requires `CER_BEGIN_ITER`. It accepts only an allowlist of options in `EXTRA`; newlines
  and reserved options are refused. It pins all three trees and records the resolved begin from `cfg_args`.
- `cer_tests_m4d_early.sbatch` pins all three trees before and after every test run.

**Code review (Codex gpt-6.1-sol, xhigh), records in `assets/cer-early-gate-2026-10-06/`.**

| round | verdict | findings |
|---|---|---|
| 1 | MATERIAL DEFECT | 7 major findings |
| 2 | MATERIAL DEFECT | 4 major findings |
| 3 | MATERIAL DEFECT | 1 major finding (E8 witness coverage) |
| 4 | SOUND WITH QUALIFICATIONS | no confirmed defect |
| 5 (focused, on the E8 test change) | MATERIAL DEFECT | the L1 colour control can cancel; fixed by a `render.sum()` probe |

Every finding was fixed, except the two items escalated below.

**Changes made during review, against rev 4:**

- **Range rule.** It is `1 ≤ cer_begin_iter ≤ transient_merging_iter`, `cer_begin_iter < densify_until_iter`,
  `cer_begin_iter < iterations` and `coarse_iterations ≤ transient_merging_iter`.
  - The freeze happens only if `densify_until_iter < iterations`.
  - `cer_gates.json` records `frozen`.
- **Lift helper.** The training lift site calls one helper, `cer.lift_select`. With the gate unarmed it is bit-identical
  to the original selection (E4).
- **Not yet built:** the scorer method `multi4d_cer_early`, which requires `mode == early`, `frozen == true`, begin 3,000
  and the full schedule. It is to be written at submission.

**GPU tests on Leonardo (debug QOS; stand-in model: bottle same-pose base run 1).**

| test | result | job, commit | what it showed |
|---|---|---|---|
| E0 | PASS | 59542900 and 59543384, @96e92ee | T0–T6 on the new tree |
| E1 | PASS | 59542905 | legacy begin equals the `cer` tree (memory fg 0 / bg 2,525, 6 donors); the unlabelled lifecycle is hash-identical |
| E2 | PASS | 59543387 | propagation exact in both banks |
| E3 | PASS | 59543387 | 95 tensors bit-identical; RNG states equal |
| E4 | PASS | 59542905 | lift exclusion, the unarmed helper equivalence, and the range refusals |
| E5–E7 | PASS | 59546982, @15596d8 | E5 drift report; E6 freeze; E7 exact zeros after the lifecycle with live witnesses |
| E8 | PASS | 59548520, @7b30ef8 | details below |
| E9 | PASS | 59548335, on run `e9_smoke96`, made by @96e92ee | details below |

**E8.**

- **Witnesses:** 881 background memory witnesses and 200 foreground rows labelled as witnesses. Every witness has nonzero
  gradients with the gate open and exactly zero under both losses with the gate shut. The two fresh processes agree.
- **Colour clamping.**
  - 71 background witnesses are clamped in all three channels.
  - All 822 clamped channels have exactly zero probe gradient.
  - All 1,821 unclamped channels have positive probe gradient.
- **The earlier E8 failure** (job 59543940) was this clamping, read through an L1 control that can cancel. It was not a
  gradient defect.
- **Tolerance change, made after the fact.** The probe's agreement tolerance was raised from 1e-6 to 1e-5 of the largest
  gradient after job 59546984 measured a 1.4e-6 relative difference. That is float32 atomic-add order over a sum of
  whole images.

**E9 and the integration runs.**

- **E9:** every refusal case and plumbing check passes, and every check on the run passes.
- **Integration runs** (not evidence): `runs/multi4d_cer_early/e9_smoke_ARS-L1-A0` (@2f3bd6c) and `e9_smoke96_ARS-L1-A0`
  (@96e92ee). Shortened schedule: begin 3,000 = merging, phase 3 at 3,200, densify-until 3,500, end at 4,000.
  - Exit 0; 0 lifts before begin.
  - Memory share 0.94 before the gap and after the return; exact zeros in the gap and outside it.
  - Frozen at 3,500; the drift report is small (0.03% of target mass in the foreground bank).
  - 99 residual donors.

**F1 answered without a run.** At fine 3,000 the transient bank has exactly 1 row under the real schedule: the 16k
run's log shows `point_transient=1` until the first lift at 3,050, which adds 2,000 rows. It paints 0.0 of the gap region
in the integration run's `cer_begin.json`. The F1 rule (transient share ≥ 0.50) therefore cannot fire, and 3,000 stands.
Memory share at 3,000 under the exact real schedule is not measured; the integration run's 0.94 used a shortened densify
schedule, whose thresholds differ before 3,000.

**Escalated to the principal, not changed here:**

1. **The inherited hybrid `clamped` buffer** (`rasterizer_impl.cu:160`) is sized for the transient count but written by
   both preprocess passes.
   - The static pass runs second, so static rows read their own flags. E8 confirms their colour gradients behave exactly
     as clamping predicts.
   - Transient rows' clamp flags at indices below the static count are overwritten by static flags. Static indices at or
     above the transient count would write past the buffer.
   - This affects the transient bank's colour gradients in every Multi4D run (base, 16k and early). Upstream defect,
     unquantified.
2. **`max_contrib` (`backward.cu:1229`)** is read by the training build but written only by the instrument build.
   - E8's positive controls found no witness whose gradient was dropped. The two processes agreed.
   - Whether the extra mass passes (begin at 3,000, drift at 16,000) perturb later training through this uninitialised
     buffer is not established.

**Before any training submission (principal):**

- review the code;
- freeze the confirmatory design (section 6);
- approve the scorer method;
- say "submit".
