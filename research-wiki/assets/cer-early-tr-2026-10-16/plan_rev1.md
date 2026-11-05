# CER on Multi4D, early gate with a gated transient copy path: plan (rev 1, 2026-10-16, exploratory)

Status: **rev 1, for review.** Nothing is written or run. Exploratory (spec §14.5, principal decision (c)): every run id,
ledger line and report says so, and nothing here is pooled with a confirmatory run.

- **Pipeline (the principal's requirement, as for the early gate):** Opus writes this plan; Codex (gpt-6.1-sol, xhigh)
  reviews it until there is consensus; a Sonnet 5.5 agent writes the tests first, then the code; Codex reviews the code
  (scientific, engineering, tensor shapes, CUDA) until the orchestrator is satisfied; GPU tests on Leonardo (debug QOS).
- **Not approved:** any training submission. The comparison design (section 8) is prepared for the principal.
- **Base:** branch `cer-early` at `7b30ef8` (code unchanged since `96e92ee`), plan
  [cer-multi4d-early-gate-plan-2026-10-06](cer-multi4d-early-gate-plan-2026-10-06.md) rev 4 and its section 10.

References use these prefixes, all at `7b30ef8` (local worktree `D:/adags-cer/multi4d-early`):

| prefix | file |
|---|---|
| `T:` | `train.py` |
| `GM:` | `scene/gaussian_model.py` |
| `R:` | `gaussian_renderer/__init__.py` |
| `C:` | `scene/cer.py` |
| `G:` | `utils/cer_gate.py` |
| `SI:` | `scene/__init__.py` |
| `F:` / `B:` | `diff_gaussian_rasterization_hybrid/cuda_rasterizer/forward.cu` / `backward.cu` |

## 1. Why

The early gate's confirmatory reading (plan log 2026-10-07 morning) failed only the return non-inferiority clause
(−0.10 dB, lower bound −0.32 against −0.25). The 2026-10-16 diagnostic (plan log of that date) showed it is not a return
problem: the early arm renders the object 0.1–0.3 dB worse at every time the object is present, in its interior. One of
the two sources it found is that **transient detail is missing**:

- In base Multi4D the transient bank covers the object interior with alpha 0.05–0.52 and adds detail in some runs
  (carton +0.31, mug +0.60 / +0.27 dB; on the bottle it hurts, −1.68 / −0.11 / −0.28).
- In the early arm it covers 0.000–0.053 and adds nothing.
- The transient bank is filled by the foreground-to-transient lift (`T:915-938`, `GM:412-481`, every 50 iterations in
  (3,000, 10,000)), and the early gate makes every labelled foreground row ineligible as a donor (`C:189-200`, plan D4),
  because the lift copies a row and leaves the donor alive, and an ungated copy would carry the object into the gap.
- So the most likely mechanism is that the lift exclusion cuts off the transient detail path. **Not proven:** the lift
  lineage is not tracked in the early arm.

**The change, one factor: memory rows may be lift donors again, and their transient copies are memory rows.**

- A transient copy of a labelled row inherits label 1 and is gated exactly as persistent memory rows are: opacity factor
  `1 − q·n(t)`, so 0 inside `[A, B]` and 1 outside.
- Transient labels follow their rows through every transient row-changing operation, as persistent labels do.
- Everything else stays as on `cer-early`: begin at 3,000, the persistent banks' labelling and inheritance, the residual
  cohort, the freeze at 16,000, the 20,000 budget.

**Predictions, stated before any code exists (descriptive):**

1. The transient bank covers the object interior pre-gap again (alpha toward base's 0.05–0.52), with label-1 transient
   rows carrying most of the transient target mass.
2. Object-interior PSNR, fix minus early, is positive pre-gap and late on carton and mug; on the bottle it is near 0 or
   negative (base's transient hurt the bottle).
3. The gap window is unchanged within the early arm's spread (≈ 0.6 dB): transient memory rows are exactly 0 in the gap.
4. The return window is at or above the early arm's.

## 2. Design decisions

**D1. One new argument, `cer_transient_gate`, default 0.**

- In `ModelParams` beside `cer_begin_iter` (`arguments/__init__.py`), stored on the scene by the scene constructor as
  `cer_scene` and `cer_begin_iter` are (`SI:36-37`).
- **0:** the `cer-early@7b30ef8` behaviour in both modes, including the lift exclusion. Every new path is a no-op.
- **1:** requires early mode. `check_begin_range` (`C:160-179`, called in `training()` before the scene exists, `T:1027`)
  also asserts `cer_transient_gate ∈ {0, 1}` and `cer_transient_gate == 0 or cer_begin_iter != -1`.
- **Why an argument and not a new mode value:** flag 0 must reproduce the early arm bit for bit (test F0), so the
  comparison (section 8) differs by this factor only.

**D2. Lift: labelled rows are eligible again; the copy carries the donor's label.**

- With flag 1, `cer.lift_select` (`C:218-230`) no longer calls `mask_lift_scores`. The donor set is therefore exactly
  base Multi4D's: `dynamic_score > threshold`, then the same `randperm` subsample (`GM:430-437`). With flag 0 the
  exclusion stays.
- `select_fg_to_transient` (`GM:412-481`) gains one line: when the foreground carries `cer_labels`, the returned dict
  holds `"cer_labels": self.cer_labels[prune_mask]`, computed with the same final boolean mask (after the subsample) as
  `xyz` (`GM:439`), so copy `k` inherits from the `k`-th selected donor in row order. No RNG is consumed.
- `add_points_from_dict` (`GM:1456`) passes `data_dict.get("cer_labels")` to `add_points` (`GM:1376`), which passes it
  to `densification_postfix_nodenom` (`GM:1414`) as a keyword argument defaulting to `None`.
- **Lineage is now recorded.** `count_lift` (`C:203-215`) also counts, after begin, the copies made and the labelled
  copies made, so the share of the transient bank descended from memory rows is known for the first time.

**D3. Transient labels: values {0, 1}; propagation through every transient row-changing operation.**

| operation | rule |
|---|---|
| `densification_postfix_nodenom` (`GM:1414`; lift) | `cat(labels, new_cer_labels)`; asserts the new labels are 1-D, length = new rows, before any row is added |
| `densification_postfix` (`GM:1590`; clone and split children) | `cat(labels, new_cer_labels)`, same assertion |
| `densify_and_clone` (`GM:1682`) | `labels[selected_pts_mask]` |
| `densify_and_split` (`GM:1624`) | `labels[selected_pts_mask].repeat(N)`; the same order as every `.repeat(N, 1)` there, so child `k·S + j` inherits from selected parent `j`; the parents go by `prune_points(prune_filter)` (`GM:1677-1679`) |
| `prune_points` (`GM:1527`) | `labels[~mask]` |
| `weight_prune` (`GM:1551`), `densify_and_prune` (`GM:1710`) | through `prune_points` and the two densifiers only |

- **The edit acts only if the transient bank has a `cer_labels` attribute** (flag 1, after begin). Otherwise
  `new_cer_labels` is ignored, and an assertion refuses non-`None` labels with any nonzero value (with flag 0 the lift
  exclusion makes every lifted label 0).
- **Invariant, asserted after every `maintain_transient` call while armed with flag 1** (`T:974-979`, as `T:953-954`
  and `:965-966` do for the persistent banks): `cer_labels` is a 1-D `uint8` cuda tensor of length
  `get_xyz.shape[0]` with values in {0, 1}. The gate asserts the length at every render (`G:42`).
- **Not changed:** the transient `cat_tensors_to_optimizer` / `_prune_optimizer` (`GM:1509`, `:1564`; labels are not
  parameters), the transient densification statistics (`GM:1728`), and the transient `max_radii2D` / `mean_radii2D`
  bookkeeping.
- **No RNG consumed by propagation:** indexing and concatenation only.

**D4. Membership stays immutable and inherited (early plan D3).**

- A transient copy of a label-0 row is label 0 forever, even if it later comes to paint the target. Labels change only
  by inheritance at a row's birth.
- **The cost is reported, not fixed.** The drift report (`C:291-327`) gains the transient bank with the same fields
  (rows meeting the memory rule while labelled 0, and label-1 rows no longer meeting it, with their pre-gap target mass
  shares).

**D5. The transient bank at begin: labelled by the same memory rule as the persistent banks.**

- At begin (`C:249-252`) the pre-gap mass pass already returns per-row transient `inside` and `total`. With flag 1, the
  transient rows with `total > 0` and `inside ≥ 0.5·total` get label 1; the others 0. Never label 2: the residual
  cohort stays background only (spec §4.2).
- **Why:** one definition of memory in every bank ("a row that paints the target before it leaves"). It also closes
  the only route by which a transient row that already paints the target at begin would stay ungated.
- **What it decides under the real schedule:** at fine 3,000 the transient bank has exactly 1 row (the seed copied
  from the first foreground point, `SI:95-103`; early plan section 10, "F1 answered"). So this rule decides at most one
  row. It is reported as `memory_rows.tr` in `cer_begin.json`.

**D6. The gate at the transient opacity site.**

- `render_rawall3pc` (`R:355`) is the only function that renders the transient bank. `render_background`,
  `render_foreground` and `render_mask` take no transient bank; `run_evaluation` and `final_test_dump` reach the
  transient bank only through `render_rawall3pc` (`utils/eval_utils.py:68`, `:177`, `:251`).
- **One line after `R:423`** (`opacity = pc_transient.get_opacity`, already activated, `[N, 1]`), before the dropout
  (`R:498-508`): `opacity = cer_gate.apply_transient(pc_transient, viewpoint_camera, opacity)`.
- **`apply_transient`** (new, in `G:`): returns `opacity` unchanged unless a module flag `TRANSIENT` is set; `begin`
  sets it with flag 1. When set, it is `apply` (`G:37-43`): it raises on a bank without CER state, asserts the factor
  shape equals the opacity shape, and multiplies by `factor` (`G:27-34`). Transient labels are never 2, so the factor is
  `1 − q·n(t)` on memory rows and 1 elsewhere.
- **Exactness with no CUDA change**, transient path under `rot_4d = True` (the reference config):
  - preprocess multiplies the opacity by the temporal marginal (`F:285`) and stores it in `conic_opacity.w` (`F:427`);
    a gated row's stored opacity is `0 · marginal = 0`;
  - the forward skips any splat with `alpha < 1/255` before `T` is touched (`F:651-654`), and so does the backward
    kernel actually launched (`PerGaussianRenderCUDA`, `B:1373-1375`). So a gated transient row adds nothing to colour,
    depth, the 4D/3D split outputs or alpha;
  - every transient parameter's gradient (opacity, features, `xyz`, `t`, `scaling`, `scaling_t`, `rotation`,
    `rotation_r`) flows only from those per-pixel terms, so it is exactly 0 in a gap frame. The transient bank has no
    non-rendering loss term (`T:814-828` regularise the foreground and background scales only).
- **What still acts on a gated transient row in a gap frame (documented, not changed, as early plan D7):**
  - the hard temporal cull `marginal_t > 0.05` (`F:281-284`) does not read opacity, so the row keeps a radius > 0. It
    counts as visible: `add_densification_stats_grad` (`GM:1728-1732`) increments its `denom` with zero gradient;
  - its `max_weights_t` in that frame is 0, so its transient weight-prune score (`max_radii2D`, `T:187-194`) comes only
    from non-gap frames;
  - its Adam moments keep moving it.

**D7. Instrument and gates count transient memory rows.**

- `gates()` (`C:330-362`): the cohort sum `cohort(x, c)` (`C:339-341`) adds the transient bank's label-`c` rows when
  the transient bank carries labels. So:
  - `memory_share_pregap` and `memory_share_return` count transient memory mass;
  - `memory_mass_gap` must be exactly 0.0 including transient memory rows (spec §14.2);
  - `residual_*` are unchanged (no transient row is label 2).
- **New descriptive fields:** `cer_transient_gate`; `memory_rows.tr`; the memory share before the gap and after the
  return by bank; the label-1 transient share of the gap region (must be 0); the lift counts of D2.
- `transient_share_gap_region` stays as it is (all transient rows, now necessarily label 0 inside the gap).
- `cer_begin.json` gains `cer_transient_gate` and `memory_rows.tr`.

**D8. Persistence.**

- `Scene2gs_mixed.save` (`SI:122-126`) writes `tr_labels` into `cer_state.pt` when the transient bank carries labels.
  The reload refusal is unchanged.
- The bank-alone tool `cer_m4d_banks.py` (outside the substrate) gets an exploratory copy `cer_m4d_banks_tr.py` that
  re-attaches `tr_labels` and sets `cer_gate.TRANSIENT`, so its `full` render still reproduces the saved final render.

**D9. Sentinels.** Unchanged: `freeze()` (`C:326`) already replaces the transient bank's lifecycle methods (the
`LIFECYCLE` list includes `add_points`, `add_points_from_dict`, `densification_postfix_nodenom`, `densify_and_prune`,
`weight_prune`, `prune_points`). The lift ends at 10,000 and the freeze is at 16,000, so no lift can follow the freeze.

**D10. Unchanged.** The persistent-bank gate and its five sites; the persistent labelling, inheritance and lift-time
accounting; the residual cohort; the begin-range rule; the freeze; `run_evaluation`; the 20,000 budget; no CUDA change
(`F:`, `B:` untouched; the instrument kernel is reused).

**Why this is one factor and not two.** The transient copy path is restored to base's (same eligibility, same `randperm`
stream given the same state), and the only thing added is that copies of memory rows are themselves memory. The
labelling of the one transient row at begin (D5) is a definitional consequence and is reported.

## 3. Isolation, git and builds

- **Local:** new branch `cer-early-tr` from `cer-early@7b30ef8`, in a new worktree
  (`git -C D:/adags-cer/multi4d worktree add D:/adags-cer/multi4d-early-tr -b cer-early-tr 7b30ef8`).
  `D:/adags-cer/multi4d` (`cer`) and `D:/adags-cer/multi4d-early` (`cer-early`) are not edited.
- **Leonardo:** a bundle `cer-early..cer-early-tr` fetched into `repo/multi4d` as branch `cer-early-tr`, then
  `git worktree add $W/repo/multi4d-cer-early-tr cer-early-tr`. Never checked out, edited or pulled: `repo/multi4d`
  (`c483e82`, detached), `repo/multi4d-cer` (`cer`), `repo/multi4d-cer-early` (`cer-early`).
- **Extensions:** `TORCH_EXTENSIONS_DIR=$W/build/torch_extensions_multi4d_cer_early_tr`, for both the training build and
  the instrument build. No CUDA source changes.
- **Launchers** (local masters in `D:/adags-cer/agent/`, installed in `agent-control/ars/sbatch/`; existing scripts not
  edited): `train_multi4d_cer_early_tr.sbatch` (a copy of the early launcher: worktree, extensions directory, branch and
  commit pin; it pins all four trees; it requires `CER_TRANSIENT_GATE` and passes `--cer_transient_gate`; it writes under
  `runs/multi4d_cer_early_tr`) and `cer_tests_m4d_early_tr.sbatch`.
- **Scorer:** a method `multi4d_cer_early_tr` in `ars_score.py`, written only if the principal approves the comparison:
  it pins the reviewed commit and requires `mode == early`, begin 3,000, `frozen`, `cer_transient_gate == 1`, and the
  full schedule.

## 4. Tests, written before the code

Stand-in model: the bottle same-pose base run 1 at iteration 20,000 (its transient bank is full), loaded as in spec §7.
Every stochastic test populates nonzero Adam moments first. Where a test needs labelled transient rows that are visible
in a gap frame, it labels them by an instrument pass at `q = 0` (as E8 does), not by hand-picking indices.

| id | test | pass |
|---|---|---|
| F0 | flag 0 is the early arm | `test_cer.py` T0–T6 and `test_cer_early.py` E1–E8 pass on the new tree with flag 0; static: every new path is reached only with flag 1 or a transient bank carrying `cer_labels`; dynamic (two subprocesses, this tree and `cer-early@7b30ef8`): `begin` in early mode gives the same labels, donors and `cer_begin.json` counts, and the E2/E3 lifecycle sequence plus a lift sequence give hash-identical models |
| F1 | transient propagation exactness | on a deep copy, every row tagged with a unique id in a tensor that clone, split and the lift copy exactly; random labels in {0, 1}; `add_points_from_dict` with a labelled dict, `densify_and_clone` and `densify_and_split` (rot_4d) with forced gradients so both selections are non-empty, `densify_and_prune` with thresholds that remove ≥ 1 row, `prune_points` with a random mask, `weight_prune`: every surviving row's label equals its source's; lengths equal; ≥ 1 row of each label survives each step |
| F2 | propagation neutral to the model | the F1 sequence on two deep copies from the same four RNG states, one with transient labels, one without: parameters, Adam moments, every per-row buffer and all four RNG states bit-identical afterwards |
| F3 | lift with labels | (a) flag 1, armed: labelled foreground rows above threshold are selected; the selection (donor indices, `randperm` draw) is identical to the unarmed selection from the same RNG state, in both the no-subsample and the subsample (pool > `max_transfer`) cases; the returned `cer_labels` equal `fg.cer_labels` at the selected rows, in `xyz` order; (b) flag 0, armed: the E4 exclusion is unchanged; (c) the copied rows land in the transient bank with those labels |
| F4 | gate identity at the transient site | at a gap frame and a non-gap frame, the flag-1 render equals, byte for byte, the same call with the transient gate off after setting the transient memory rows' opacity logits to −inf; the reference differs from the plain render at the gap frame (non-vacuity); the rasterizer's `forward` is wrapped to capture the transient opacity it receives, which must equal `factor · ungated`; flag 0 or an unarmed gate passes the transient opacity through unchanged; an armed transient gate raises on a transient bank without labels |
| F5 | training-build gradients, transient | E8's protocol on the training build, in two fresh processes, for transient witnesses (label-1 transient rows with positive contribution in a gap view at `q = 0`): at `q = 0` every witness has nonzero rendering gradient in opacity, features and `xyz`; at `q = 1` every transient parameter group's gradient (`xyz`, `f_dc`, `f_rest`, `opacity`, `scaling`, `rotation`, `t`, `scaling_t`, `rotation_r`) is exactly 0 for every witness, while ungated visible transient rows have nonzero gradient; pre-gap frame at `q = 1`: witnesses nonzero; `add_densification_stats_grad` keeps `denom` and the accumulators finite |
| F6 | instrument and gates | after `begin` with flag 1 and F1's operations, at `q = 1` on the training cameras: transient memory mass before the gap > 0 (witnesses), transient memory mass in the gap exactly 0, `memory_mass_gap` (now with transient rows) exactly 0; the memory share before the gap equals an independent recomputation from the per-bank masses; the drift report's transient fields are correct on a doctored model (E5's position-move doctoring applied to transient rows, expected drift computed first) |
| F7 | begin labelling of the transient bank | flag 1: transient labels equal the memory rule computed independently from the same mass pass; never 2; with the real-schedule one-row bank, at most 1 label; flag 0: the transient bank has no `cer_labels` |
| F8 | persistence and sentinels | `save` writes `tr_labels` equal to the in-memory labels; the reload refusal stands; after `freeze()` every transient lifecycle method raises |
| F9 | integration (not evidence) | the real entry point through the final-render wrapper, flag 1, early mode, the E9 shortened schedule (begin 3,000 = merging < phase 3 < densify-until < iterations), on the **carton** same-pose scene (229–288 foreground memory rows at fine 3,000 on the real schedule, so labelled donors exist; the bottle had 0–20): exit 0; ≥ 1 lift after begin with ≥ 1 labelled copy; transient labels consistent in length; `cer_begin.json`, `cer_drift.json`, `cer_gates.json` written with the new fields; spec §14.2 preconditions with transient rows counted; the label-1 transient share of the gap region exactly 0; the refusal `cer_transient_gate 1` with `cer_begin_iter -1` before training |

- F0 passes on the unmodified tree; F1–F9 fail on it.
- Tests and F9 are not evidence.

## 5. What the code reviewer must check

- **Scientific:** no path lets a memory row, a copy of one, or a descendant of a copy paint a gap frame (lift, clone,
  split; persistent and transient); the flag-1 lift is base's lift; labels are chosen once, at begin, and only
  inherited afterwards; flag 0 is the early arm.
- **Engineering:** labels and rows agree after every transient operation; no RNG in propagation; flag 0 untouched;
  isolation of worktree and extensions; O_EXCL kept; the launcher passes and records the flag.
- **Tensor shapes:** transient `cer_labels [N]` uint8 against the factor `[N, 1]` and the opacity `[N, 1]`;
  `.repeat(N)` against `.repeat(N, 1)`; the lift's labels against `xyz` after the `randperm` subsample (`GM:430-437`);
  bool masks of the right length; dtype and device.
- **CUDA:** a zero transient opacity is skipped in the forward and in the launched backward (`F:651-654`,
  `B:1373-1375`); the gated row's other per-splat outputs (depth, 4D/3D colour, alpha) are untouched by it; the
  instrument swap is restored in a `finally`; densification statistics stay finite.

## 6. What this does not test or change

- The second source of the deficit (persistent banks slightly worse, section 9).
- Membership reassignment; the pass-through and moved-return scenes; any schedule constant; the escalated substrate
  defects (`clamped`, `max_contrib`).

## 7. Risks, stated before the code

- **Transient memory copies may displace persistent memory mass before the gap.** If they then sit outside the gap only
  partly (they are time-local), the persistent memory rows that remain may be under-trained where the copies painted.
  The return window and the bank-alone renders read this.
- **Memory share may change.** It now includes transient memory mass; transient label-0 copies of label-0 rows that come
  to paint the target lower it. The precondition ≥ 0.50 still applies, as a reported precondition of the exploratory arm.
- **Gap ghosts from label-0 transient rows.** Unchanged in kind from the early arm; read through the gap window and
  `transient_share_gap_region`.

## 8. Exploratory comparison design (for the principal; nothing submitted)

- **Runs:** the fix (flag 1, begin 3,000, published schedule, freeze 16,000, 20,000 iterations) ×3 on each same-pose
  scene (bottle, carton, mug): 9 runs, `runs/multi4d_cer_early_tr/ars_w1_<scene>_m4dearlytr_rep<r>`.
- **Comparison:** against the 9 existing early-arm runs (`runs/multi4d_cer_early/ars_w1_<scene>_m4dearly_rep<r>`);
  base Multi4D's 9 runs and the 16k arm as descriptive anchors.
- **Endpoints (all descriptive; exploratory, no verdict):**
  1. object-interior PSNR (mask eroded 4 px; `target_edge_interior.py`) pre-gap (frames 30–57) and late (230–259);
  2. the gap window; 3. the return window and the transition frames 90–91;
  4. bank-alone interior PSNR and transient interior alpha at frames 40 and 57 (`interior_by_bank.py`, with
     `cer_m4d_banks_tr.py`).
- **Model:** arm and layout fixed effects, 9 v 9, as the confirmatory reading; per-layout means with min and max; no
  Holm family, no rule. The fix's gate files must meet spec §14.2's preconditions with transient rows counted.
- **Cost:** 9 × ≈ 2.6 GPU-h ≈ 23 GPU-h, plus scoring (≈ 11 CPU-min each) and the diagnostics (two debug jobs, as on
  2026-10-16), plus the tests: F0–F8 ≈ 1 GPU-h on the debug QOS and F9 ≈ 1 GPU-h. **≈ 26 GPU-h in total.**

## 9. Task B: are memory rows densified less in the early arm? (a measurement, no code change to densification)

**Question.** The persistent banks alone are slightly worse in the early arm on bottle and carton (−0.28 / −0.16 dB
interior). Two candidate causes: gated rows' densification statistics are diluted by the gap views (early plan D7), and
memory rows get ~10% fewer gradient views over 17,000 iterations.

**Measurement: a runtime wrapper, no substrate edit.** `agent-control/ars/densify_probe.py` runs the reviewed
`train.py` by `runpy` (as `m4d_final_render.py` and `ex4_collapse_probe.py` do) from `repo/multi4d-cer-early`
(`7b30ef8`, clean), and records, without changing any computation:

- **At every call of `densify(max_grad, extent)`** on the background and the foreground bank (`GM:119-124`), before the
  original runs, per label group (memory 1, residual 2, other 0): row count; sum and mean of `denom`; mean and quantiles
  of `xyz_gradient_accum / denom` (NaN as 0, as `GM:120-121` does); the number of rows the clone and the split would
  select (the same masks as `GM:776-791` and `GM:741-766`, recomputed read-only from the same tensors); the threshold.
- **The statistics window.** Both persistent `densification_postfix` methods reset `xyz_gradient_accum` and `denom` to
  zero for every row (`GM:1188-1189`, and the foreground's), so the average a densify step reads covers the iterations
  since the previous densify step (100 iterations, about 200 views at batch 2).
- **Per row, the gap-only increments in that window:** at every `add_densification_stats` call the wrapper also counts,
  per row, the increments made in iterations whose training batch held only gap frames (frames parsed from the training
  renders of that iteration; cam00 evaluation renders excluded). The counter is filtered by every `prune_points`, reset
  when `densification_postfix` resets the accumulators, and zero-padded for rows appended at the end (the residual rows at
  begin). For a gated row such an increment carries exactly zero gradient, so `accum / (denom − gap_only)` is the
  undiluted average; the clone and split counts are also computed under it. Increments from mixed batches (one gap view,
  one non-gap view) are counted separately and left in.
- **Row counts by label group** at every densification step and at the freeze.

**Two runs, both bottle same-pose (ARS-L1-A0), full published schedule, through the probe wrapper:**

1. **Early arm:** `--cer_scene <bottle> --cer_begin_iter 3000`, as the confirmatory runs. Labels come from `begin`.
2. **Shadow base:** no `cer_scene`, so CER is off and the run is base Multi4D on the `cer-early` tree (T1 and E4
   showed that the CER-off tree renders and lifts as the pinned tree). At the end of fine iteration 3,000, right where
   `begin` would run (after `run_evaluation(3000)`, `T:997-1000`), the wrapper computes the same memory rule with
   `cer.mass` on the ungated model and attaches the labels as `cer_labels` to both persistent banks **without arming the
   gate** (`cer_gate.ARMED` stays False). The `cer-early` propagation (`GM:653-795`, `:1110-1240`) then carries them
   through clone, split and prune; E3 showed that carrying labels changes nothing in the model. The gate is not armed,
   so no render and no lift is affected; `check_labels` and the lift masking run only when armed.
   - **What differs from a plain base run:** the two mass passes at 3,000 (the early arm runs the same passes in
     `begin`). They consume no random numbers (dropout off), but they swap in the instrument kernel; whether they
     perturb later training through the uninitialised `max_contrib` buffer is not established (early plan section 10,
     escalation 2). Both runs carry the same perturbation.

**Readout (descriptive):** per bank, over the densification steps 3,100–15,900, for memory rows in the early arm against
memory rows in the shadow base: the mean average gradient (ratio early / shadow), the clone and split selections summed
over steps (ratio), the memory row count at 16,000 (ratio); and, within the early arm, the undiluted average gradient
and the selections it would give. "Lower in the early arm" is reported with the ratio; no threshold decides anything.
Only then is a densification-statistics change proposed (not implemented without the principal).

**Cost:** 2 × ≈ 2.7 GPU-h ≈ 5.4 GPU-h (approved budget ≤ 6 GPU-h), lprod QOS, walltime 3:50, one debug-QOS smoke run of the
wrapper on a shortened schedule first (≈ 0.3 GPU-h). Recorded in the ledger as exploratory diagnostics.
