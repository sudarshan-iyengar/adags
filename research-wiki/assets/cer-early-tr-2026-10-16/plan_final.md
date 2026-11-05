# CER on Multi4D, early gate with a gated transient copy path: plan (rev 5, final, 2026-10-16, exploratory)

Status: **rev 5, final.** Codex rounds 1–4 returned MATERIAL DEFECT (6, 3, 3 and 2 confirmed findings; round 4's were
all in the task-B probe); round 5 returned PLAN READY. Every text and review is in
[assets/cer-early-tr-2026-10-16/](../assets/cer-early-tr-2026-10-16/), the dispositions in sections 10–13. **Section 14 (appended)
is the implementation, test and premise-check record; its recommendation is to defer the comparison of section 8.**

Exploratory (spec §14.5, principal decision (c)): every run id,
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
- **1:** requires early mode. A new check, `check_transient_gate(dataset)`, is called in `training()` **unconditionally**,
  before the scene exists (beside `T:1027`, whose `check_begin_range` call runs only when `cer_begin_iter != -1`). It
  asserts `cer_transient_gate ∈ {0, 1}`, and `cer_transient_gate == 0` or (`cer_begin_iter != -1` and `cer_scene` set).
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
- **Lineage is now recorded, two ways.** (a) Births: `count_lift` (`C:203-215`) also counts, after begin, the copies made,
  the labelled copies made, and the labelled rows that were eligible (`dynamic_score > threshold`) at each selection.
  (b) Survivors: the label-1 transient row count at the freeze (`cer_drift.json`) and at the end (`cer_gates.json`),
  after clone, split and prune. With D5 every label-1 transient row descends from a labelled donor.

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

**D5. The transient bank at begin: all label 0 (rev 2; rev 1 labelled it by the memory rule).**

- With flag 1, `begin` gives the transient bank `cer_labels` of zeros (one per row), and `cer_clock` and `cer_q` equal to
  the persistent banks' (`C:252`); `G:30-33` read all three. Transient labels thereafter arise **only** by inheritance from
  a labelled lift donor. Never label 2: the residual cohort stays background only (spec §4.2).
- **Why (Codex round 1):** labelling the pre-existing transient rows at begin would gate rows independently of the lift,
  an intervention separate from restoring the copy path. At begin the early arm's transient bank holds the one seed row
  (`SI:95-103`; early plan section 10, "F1 answered"); it stays ungated, exactly as in the early arm.
- **Reported:** the transient rows' pre-gap target mass and all-pixel gap mass at begin (from the begin mass passes), in
  `cer_begin.json`, so their size is known.

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
- **NaN guard (Codex round 1).** `check_finite` (`C:134-137`) exists because `min(0.99, NaN)` renders at alpha 0.99
  (`F:651`), so a NaN opacity times a zero factor is not excluded. With flag 1 it also checks the transient bank's
  activated opacity, at `begin`, `freeze` and `gates`, as it does the persistent banks.
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

- `Scene2gs_mixed.save` (`SI:122-126`) writes `tr_labels` into `cer_state.pt` when the transient bank carries labels
  (the clock and strength are the persistent banks', already saved).
  The reload refusal is unchanged.
- The bank-alone tool `cer_m4d_banks.py` (outside the substrate) gets an exploratory copy `cer_m4d_banks_tr.py` that
  re-attaches `tr_labels` with the clock and strength, and sets `cer_gate.TRANSIENT` per run (reset before every run, since
  one job may hold runs of different arms, `cer_m4d_banks.py:119-127`), so its `full` render still reproduces the saved
  final render.

**D9. Sentinels.** Unchanged: `freeze()` (`C:326`) already replaces the transient bank's lifecycle methods (the
`LIFECYCLE` list includes `add_points`, `add_points_from_dict`, `densification_postfix_nodenom`, `densify_and_prune`,
`weight_prune`, `prune_points`). The lift ends at 10,000 and the freeze is at 16,000, so no lift can follow the freeze.

**D10. Unchanged.** The persistent-bank gate and its five sites; the persistent labelling, inheritance and lift-time
accounting; the residual cohort; the begin-range rule; the freeze; `run_evaluation`; the 20,000 budget; no CUDA change
(`F:`, `B:` untouched; the instrument kernel is reused).

**Why this is one factor.** The transient copy path is restored to base's (same eligibility, same `randperm` stream given
the same state), and the only thing added is that copies of memory rows, and their descendants, are themselves memory.
No transient row present at begin is gated (D5).

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
| F2 | propagation neutral to the model (a regression test; it can pass on the unmodified tree) | the F1 sequence on two deep copies from the same four RNG states, one with transient labels and labelled lift payloads, one without labels and with the labels stripped from the same payloads (model inputs unchanged): parameters, Adam moments, every per-row buffer and all four RNG states bit-identical afterwards |
| F3 | lift with labels | (a) flag 1, armed: labelled foreground rows above threshold are selected; the selection (donor indices, `randperm` draw) is identical to the unarmed selection from the same RNG state, in both the no-subsample and the subsample (pool > `max_transfer`) cases; the returned `cer_labels` equal `fg.cer_labels` at the selected rows, in `xyz` order; (b) flag 0, armed: the E4 exclusion is unchanged; (c) the copied rows land in the transient bank with those labels |
| F4 | gate identity at the transient site | at a gap frame, the flag-1 render equals, byte for byte, the same call with the transient gate off after setting the logits of the transient rows whose factor is 0 at that frame to −inf (the cohort chosen per frame, as T3 does, `test_cer.py:257-265`), and that reference differs from the plain render (non-vacuity); at a non-gap frame, the flag-1 render equals the plain render byte for byte; the rasterizer's `forward` is wrapped to capture the transient opacity it receives, which must equal `factor · ungated`; flag 0 or an unarmed gate passes the transient opacity through unchanged; an armed transient gate raises on a transient bank without labels |
| F5 | training-build gradients, transient | E8's protocol on the training build, in two fresh processes, with **every** E8 control kept: both losses (L1 and the non-cancelling `render.sum()` probe) with E8's either-loss opacity rule (`op_fail = ~o_nz & ~po_nz`, `test_cer_early.py:382`), the colour-clamping rule (`:380-399`), and for the pre-gap control only witnesses visible in that view (`:474-480`; the temporal cull `F:281-285` makes this necessary). Witnesses: label-1 transient rows with positive contribution in a gap view at `q = 0`. At `q = 0`: every witness passes E8's opacity and colour controls. At `q = 1`, same frame and view: every transient parameter group (`xyz`, `f_dc`, `f_rest`, `opacity`, `scaling`, `rotation`, `t`, `scaling_t`, `rotation_r`) is exactly 0 for every witness, while ungated visible transient rows pass the same controls. Pre-gap frame at `q = 1`: the pre-visible witnesses pass. **Per-group positive control on a deterministic fixture (rev 4):** one transient row appended to the loaded bank, label 1, with its geometry set, not copied: its mean on the gap view's optical axis at the median depth of that view's visible transient rows; anisotropic spatial scales (1.0 : 0.7 : 0.5 of a scale that covers about 20 px); a non-identity spatial rotation and a non-identity `rotation_r` (a 10° space–time mixing), so the space–time covariance term (`B:719-728`) is nonzero; `t` offset from the view's timestamp by half a temporal standard deviation, so the marginal's time derivative is nonzero; opacity 0.5; DC 0.5 per channel and higher SH 0.1 (unclamped). Loss: L1 against a constant target that differs from the render on the fixture's footprint. The test first establishes all nine positive controls at `q = 0` (each group's gradient nonzero for the fixture row) and fails, as a fixture defect, if any is zero; then at `q = 1` all nine are exactly 0. `add_densification_stats_grad` keeps `denom` and the accumulators finite. A NaN transient opacity on a memory row makes `check_finite` refuse |
| F6 | instrument and gates | after `begin` with flag 1 and F1's operations, at `q = 1` on the training cameras: transient memory mass before the gap > 0 (witnesses), transient memory mass in the gap exactly 0, `memory_mass_gap` (now with transient rows) exactly 0; the memory share before the gap equals an independent recomputation from the per-bank masses; the drift report's transient fields are correct on a doctored model (E5's position-move doctoring applied to transient rows, expected drift computed first) |
| F7 | the transient bank at begin | flag 1: after `begin` the transient bank carries `cer_labels` all 0, of its row count, with the persistent banks' clock and strength, and `cer_begin.json` reports its pre-gap target and gap masses; flag 0: the transient bank has no `cer_labels` |
| F8 | persistence and sentinels | `save` writes `tr_labels` equal to the in-memory labels; the reload refusal stands; after `freeze()` every transient lifecycle method raises |
| F9 | integration (not evidence) | the real entry point through the final-render wrapper, flag 1, early mode, the E9 shortened schedule (begin 3,000 = merging < phase 3 < densify-until < iterations), on the **carton** same-pose scene (229–288 foreground memory rows at fine 3,000 on the real schedule; the bottle had 0–20): exit 0; the lift counters record, at each selection, the labelled rows eligible (`dynamic_score > threshold`); ≥ 1 lift after begin with ≥ 1 labelled copy (if no labelled row is ever eligible, F9 is reported vacuous for that clause, not passed); transient labels consistent in length; `cer_begin.json`, `cer_drift.json`, `cer_gates.json` written with the new fields; spec §14.2 preconditions with transient rows counted; the label-1 transient share of the gap region exactly 0; the refusal `cer_transient_gate 1` with `cer_begin_iter -1` before training |

- F0 and F2 pass on the unmodified tree; F1 and F3–F9 fail on it.
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
- **Endpoint scripts.** `target_edge_interior.py` and `interior_by_bank.py` hard-code the base, 16k and early arms
  (`target_edge_interior.py:19-20`). Exploratory copies (`*_tr.py`) take the arms from a run manifest; each verifies that
  all 9 fix runs and all 9 early runs enter its output and refuses otherwise.
- **Model:** arm and layout fixed effects, 9 v 9, as the confirmatory reading; per-layout means with min and max; no
  Holm family, no rule. The fix's gate files must meet spec §14.2's preconditions with transient rows counted.
- **Cost:** 9 × ≈ 2.6 GPU-h ≈ 23 GPU-h, plus scoring (≈ 11 CPU-min each) and the diagnostics (two debug jobs, as on
  2026-10-16), plus the tests: F0–F8 ≈ 1 GPU-h on the debug QOS and F9 ≈ 1 GPU-h. **≈ 26 GPU-h in total.**

## 9. Task B: are memory rows densified less in the early arm? (a measurement, no code change to densification)

**Question.** The persistent banks alone are slightly worse in the early arm on bottle and carton (−0.28 / −0.16 dB
interior). Two candidate causes: gated rows' densification statistics are diluted by the gap views (early plan D7), and
memory rows get ~10% fewer gradient views over 17,000 iterations.

**Measurement: a runtime wrapper, no substrate edit.** What it measures is the statistics as they were recorded in each run,
and one deletion sensitivity; it cannot reconstruct training without gating. `agent-control/ars/densify_probe.py` runs the reviewed
`train.py` by `runpy` (as `m4d_final_render.py` and `ex4_collapse_probe.py` do) from `repo/multi4d-cer-early`
(`7b30ef8`, clean), and records, without changing any computation:

- **At every call of `densify(max_grad, extent)`** on the background and the foreground bank (`GM:119-124`), before the
  original runs, per label group (memory 1, residual 2, other 0): row count; sum and mean of `denom`; mean and quantiles
  of `xyz_gradient_accum / denom` (NaN as 0, as `GM:120-121` does); the number of rows the clone and the split would
  select (the same masks as `GM:776-791` and `GM:741-766`, recomputed read-only from the same tensors); the threshold.
- **The statistics window.** Both persistent `densification_postfix` methods reset `xyz_gradient_accum` and `denom` to
  zero for every row (`GM:1188-1189`, and the foreground's), so the average a densify step reads covers the iterations
  since the previous densify step (100 iterations, about 200 views at batch 2).
- **Per row, the zero-gradient increments in that window (rev 2, Codex round 1).** `denom` counts batches in which a row is
  visible in any view (`visibility_filter_second = cat(...).any(dim=0)`, `T:593`; the numerator is the norm of the
  gradient summed over the batch's views, `T:855-860`, `GM:114-117`). For a gated memory row, an increment contributes
  zero gradient exactly when **every view of the batch in which the row is visible is a gap view**. The wrapper records,
  for every training render of the iteration (cam00 evaluation renders excluded), the frame and the bank's per-row
  visibility (`visibility_filter` of `render_background` for the background bank in phase 2; `visibility_filter_fg` /
  `visibility_filter_bg` of `render_rawall3pc`; `visibility_filter` of `render_foreground`), and at each
  `add_densification_stats` call it counts per row `zero_inc = update_filter & ~(visible in some non-gap view)`. Two
  coarser counts are kept beside it: increments in all-gap batches, and increments in mixed batches.
- **The deletion sensitivity.** `accum / (denom − zero_inc)` is the average with those zero increments deleted (rows with
  `denom − zero_inc = 0` get 0 and are counted apart); the clone and split counts are also computed under it. The
  ordinary and the corrected averages are compared **over the same rows** (`denom > 0`, the corrected zeros kept as 0;
  Codex round 3). It is
  exact for a gated row and is not a model of an ungated run, whose numerator would differ.
- **Alignment.** The counters are filtered by every `prune_points`, reset when `densification_postfix` resets the
  accumulators, and zero-padded for rows appended at the end without it (the residual rows at begin).
- **Row counts by label group** at every densification step (the count before that step's densification), and as
  snapshots in **both** modes: after the labelling at 3,000 (early: after `begin`; shadow: after the shadow labels) and at
  the end of fine iteration 16,000 (the probe hooks `run_evaluation(16000)`, which precedes the early arm's `freeze` in the
  same iteration, so no row changes between them; the shadow run has no `freeze`, `T:1005`). The reader uses these
  snapshots for the row-count ratio.
- **Validity of a probe run (Codex round 2).** A NaN loss restarts training by `os.execv` (`T:846-848`), which would
  replace the instrumented process with a bare one. The probe replaces `os.execv` with a function that raises, so a
  restart fails the job. A probe output is accepted only if the job exits 0 (`provenance.txt`, which records the exit
  code after every check of the job) with the worktree state unchanged, the probe writes its
  completion record after `train.py` returns, both snapshots exist and are populated, and every fine-stage
  `add_densification_stats` call was matched to its visibility source (an unmatched call would omit its zero
  increments); the reader refuses otherwise. Unmatched coarse-stage calls are reported (no labels exist then).

**Two runs, both bottle same-pose (ARS-L1-A0), full published schedule, through the probe wrapper:**

1. **Early arm:** `--cer_scene <bottle> --cer_begin_iter 3000`, as the confirmatory runs. Labels come from `begin`.
2. **Shadow base:** no `cer_scene`, so CER is off and the run is base Multi4D on the `cer-early` tree (T1 and E4
   showed that the CER-off tree renders and lifts as the pinned tree). At the end of fine iteration 3,000, right where
   `begin` would run (after `run_evaluation(3000)`, `T:997-1000`), the wrapper computes the same memory rule with
   `cer.mass` on the ungated model and attaches the labels as `cer_labels` to both persistent banks **without arming the
   gate** (`cer_gate.ARMED` stays False). The `cer-early` propagation (`GM:653-795`, `:1110-1240`) then carries them
   through clone, split and prune; E3 showed that carrying labels changes nothing in the model. The gate is not armed,
   so no render and no lift is affected; `check_labels` and the lift masking run only when armed.
   - **What differs from a plain base run:** one ungated pre-gap mass pass at 3,000 (the early arm's `begin` runs that
     pass and a second, gated, gap pass). The passes consume no random numbers (dropout off), but they swap in the
     instrument kernel; whether they perturb later training through the uninitialised `max_contrib` buffer is not
     established (early plan section 10, escalation 2), and the two runs' perturbations are not claimed to be equal.

**Readout (descriptive):** per bank and over that bank's own densification window (background 3,100–15,900; foreground
3,100–9,900, since foreground density control stops at phase 3, `T:948`), with the actual number of records, for memory rows in the early arm against
memory rows in the shadow base: the mean average gradient (ratio early / shadow), the clone and split selections summed
over steps (ratio), the memory row count at 16,000 (ratio); and, within the early arm, the undiluted average gradient
and the selections it would give. "Lower in the early arm" is reported with the ratio; no threshold decides anything.
Only then is a densification-statistics change proposed (not implemented without the principal).

**Test of the probe first (debug QOS, ≈ 0.2 GPU-h):** on a loaded model (spec §7 loader) with the wrapper installed, and
with hand-built batches: (a) an all-gap batch, a mixed batch where a row is visible only in the gap view, and a mixed
batch where it is visible in both, give `zero_inc` 1, 1, 0 for that row; (b) the counters stay aligned through
`prune_points`, `densification_postfix` and an end append; (c) on both banks, the recomputed clone and split masks equal
the rows `densify` actually copies and splits (rows tagged by a unique id in `_opacity`, which clone and split copy exactly
and which the selection does not read: the final tags must be `[orig[~split], orig[clone], orig[split] ×2]`); (d) on both
banks, the wrapped calls leave every tensor attribute, every module's `state_dict` (the foreground's deformation network)
and the optimizer's complete `state_dict` (per-parameter state, step counters, group metadata) and the RNG bit-identical
to a twin run with the original class methods restored (no wrapper active); (e) a zero corrected denominator gives 0 and is
counted, and the corrected and ordinary averages are taken over the same rows (the example accumulators `[2, 0]`,
denominators `[1, 1]`, zero increments `[0, 1]` gives 1 and 1, with one zero corrected denominator). The phase-2
standalone background rasterizer (`render_background`) is exercised in (a).

**What the neutrality test establishes, and what it does not.** It shows the probe's bookkeeping leaves the lifecycle
calls bit-identical. It does not establish whole-training neutrality: the probe allocates CUDA memory and records
visibility tensors, which can change allocation patterns, and the training backward reads the uninitialised `max_contrib`
buffer (`B:1229`, early plan section 10, escalation 2). So each probe run is read as one run of its arm (one early-arm
run, one base-like run), with that qualification, not as a replicate of the confirmatory runs.

**Cost:** 2 × ≈ 2.7 GPU-h ≈ 5.4 GPU-h, plus the probe test ≈ 0.2 GPU-h (approved budget ≤ 6 GPU-h), lprod QOS, walltime 3:50.
The shortened-schedule smoke run is dropped: the config fixes `coarse_iterations` 2,000 and `transient_merging_iter` 3,000
(`arguments/dynerf.py:50`, `:58`; `merge_hparams` overrides the command line), so it would cost ≈ 1 GPU-h. Recorded in
the ledger as exploratory diagnostics.

## 10. Rev 1 → rev 2 (Codex round 1: MATERIAL DEFECT)

| # | finding | disposition |
|---|---|---|
| 1 | the transient gate lacks the NaN guard (`C:134-137`) | accepted: `check_finite` covers the transient bank with flag 1 (D6); F5 tests the refusal |
| 2 | F4's non-gap reference wrongly removes memory rows | accepted: the cohort is chosen per frame; at a non-gap frame the render must equal the plain render |
| 3 | F5 drops E8's controls (clamping, probe, pre-visible subset) | accepted: kept; per-witness positives only for opacity and unclamped colour, the other groups on at least one witness |
| 4 | task B's counterfactual overstated (denom counts OR-visible batches) | accepted: row-level zero-increment count from per-view visibility; called a deletion sensitivity; zero denominators handled |
| 5 | F2 conflicts with the refusal rule and passes on the unmodified tree | accepted: labels stripped from the control payloads; F2 classed as a regression test |
| 6 | foreground window ends at 9,900 | accepted: bank-specific windows and record counts |
| suspected 1 | D5 is a separate intervention | accepted: D5 now labels nothing at begin; the transient masses at begin are reported |
| suspected 2 | F9's donor guarantee unsupported | accepted: eligible labelled donors recorded; the clause is reported vacuous if none |
| unverified | "same perturbation" in both task-B runs | withdrawn |
| missing | transient clock and strength; `TRANSIENT` reset in the bank tool | accepted (D5, D8) |
| missing | tests of the probe itself | accepted: probe test (a)-(e) before the runs |
| missing | births vs surviving descendants | accepted: both recorded (D2) |

## 11. Rev 2 → rev 3 (Codex round 2: MATERIAL DEFECT)

| # | finding | disposition |
|---|---|---|
| 1 | F5 still drops E8's either-loss opacity rule | accepted: both losses and every E8 control kept |
| suspected | F5's per-group positives depend on the checkpoint | accepted: a controlled transient fixture row gives the per-group positive and zero controls |
| 2 | task B has no 16,000 snapshot in the shadow run; the reader used a pre-densification count | accepted: snapshots after labelling and at 16,000 in both modes, from `run_evaluation(16000)` |
| 3 | a NaN restart would drop the instrumentation silently | accepted: `os.execv` raises inside the probe; completion record and snapshots required |
| missing | endpoint scripts cannot read the new arm | accepted: exploratory copies with a run manifest that refuse a missing run |

## 12. Rev 3 → rev 4 (Codex round 3: MATERIAL DEFECT)

| # | finding | disposition |
|---|---|---|
| 1 | the corrected and ordinary average gradients were averaged over different rows | accepted: both over `denom > 0`, corrected zeros kept; zero corrected denominators reported apart |
| 2 | an incomplete probe run could be accepted | accepted: the reader requires exit 0, the completion record, populated snapshots and no unmatched fine-stage call |
| 3 | the probe test checks counts, not masks; its control was still wrapped; background only | accepted: tagged-row mask check and a genuinely unwrapped twin (optimizer state included), both banks |
| suspected | F5's fixture copied an uncontrolled witness; `t = timestamp` zeroes the time derivative | accepted: deterministic geometry, a space–time rotation, a half-std time offset; positive controls established first |
| missing | D1's validation is called only in early mode | accepted: `check_transient_gate` runs unconditionally before the scene |

## 13. Rev 4 → rev 5 (Codex round 4: MATERIAL DEFECT, task-B probe only)

| # | finding | disposition |
|---|---|---|
| 1 | the launcher recorded the exit code before its worktree check | accepted: recorded after every check; the reader also requires an unchanged worktree state |
| 2 | the neutrality digest omits buffers, modules and optimizer step counters | accepted: every tensor attribute, every module `state_dict`, the full optimizer `state_dict` |
| suspected | whole-training neutrality not established | accepted: the claim is narrowed; each probe run is read as one run of its arm |
| missing | no regression check of the corrected average | accepted: the `[2, 0]` example |

## 14. Implementation, tests, and the premise check (2026-10-16 evening; appended)

**Review state, stated first.** Codex reviewed the plan in four rounds (MATERIAL DEFECT each: 6, 3, 3 and 2 confirmed
findings; round 4's were all in the task-B probe; every finding was fixed, sections 10–13). Round 5 could not run at once:
Codex hit its usage limit at 17:00, resetting at 21:28. Round 4 raised nothing against the task-A design, so the
implementation was started during the wait and the GPU tests below ran before the first code review. **Round 5 (21:31, one
combined review): PLAN READY (consensus); CODE READY WITH FIXES; PREMISE CHECK QUALIFIED** (record
`review_r5_plan_r1_code_premise_codex.md`; dispositions at the end of this section).

**Code** (branch `cer-early-tr` from `cer-early@7b30ef8`; local worktree `D:/adags-cer/multi4d-early-tr`; Leonardo worktree
`repo/multi4d-cer-early-tr`, made by bundle fetch into `repo/multi4d` and `git worktree add`; extensions
`build/torch_extensions_multi4d_cer_early_tr`). `repo/multi4d`, `repo/multi4d-cer` and `repo/multi4d-cer-early` were never
checked out or edited; their HEAD and porcelain were recorded before and after the deployment and by every job.

| commit | content (Sonnet 5.5 unless marked) |
|---|---|
| `5edd293` | tests F0–F9, written before the code |
| `5afa119` | the code: D1–D9 (8 files, no CUDA file) |
| `e857935` (orchestrator) | tests: loaded transient weight buffers sized in the F0/F5 subprocesses; F0 split into parts for the 30-min debug QOS |
| `36024cf` | tests: F5 attributes positive-control anomalies to the escalated defects |
| `65e0eb9` (orchestrator) | tests: an "unwritten" static entry counts as unattributed |
| `c7866b6` | tests: F5c, the controls with the static set out of view |
| `b5254ee` (orchestrator) | tests: F5c requires zero witness contribution in the gated gap view |
| `c087731` (orchestrator) | tests: F9 requires unchanged trees from the launcher record (Codex code round 1) |
| `96f6fe0` (orchestrator) | tests: F9 reads spec §14.2 (the residual share and the stale `pass` field are descriptive) |
| `8e3d2d0` (orchestrator) | tests: F9 requires the carton scene (Codex code round 2) |

The code has not changed since `5afa119`. Decisions the implementer took that the plan does not state (for the code review):
`begin` sets `cer_gate.TRANSIENT = bool(flag)`; with flag 0 the armed lift payload carries an all-zero `cer_labels` that the
unlabelled transient bank accepts (no numerical effect, one extra device sync); the new JSON fields appear only with flag 1
except `cer_transient_gate`; E1a of the early suite fails on this tree by construction (it forbids a labelled transient class
and an edited renderer) and F0 tolerates exactly those two entries, checked separately by F0's static part.

**Launchers** (local masters in `D:/adags-cer/agent/`, installed in `agent-control/ars/sbatch/`):
`train_multi4d_cer_early_tr.sbatch` (requires `CER_TRANSIENT_GATE`, pins all four trees) and `cer_tests_m4d_early_tr.sbatch`.

**GPU tests (debug QOS unless marked; stand-in model: bottle same-pose base run 1; not evidence).**

| test | result | job | what it showed |
|---|---|---|---|
| F0 static | PASS | 60733454 | every new path is flag-guarded; renderer gains one line; gate only gains; no CUDA change |
| F0 dynamic | PASS | 60733454 | flag 0 equals `cer-early@7b30ef8`: begin labels, donors and `cer_begin.json` counts, lifecycle and lift sequences hash-identical (6,059 lifted rows, the `randperm` exercised) |
| F0 suites | PASS | 60733456, 60734698, 60734699 | T0–T6 and E1–E8 on the new tree with flag 0 (E1a tolerated by construction, as above) |
| F1 | PASS | 60730613 | transient propagation exact through lift, clone, split, prune, densify-and-prune, weight-prune |
| F2 | PASS | 60730613 | labels neutral to the model: 50 tensors bit-identical, RNG equal |
| F3 | PASS | 60730613 | with flag 1 the lift selection equals the unarmed one (7,096 eligible, 3,571 labelled, subsample case exercised) |
| F4 | PASS | 60730613 | the transient gate's render identity at gap and non-gap frames (200 labelled rows) |
| F5 | gate clauses PASS; positive controls FAIL | 60733454, 60736116 | see below |
| F5c | gate clauses PASS; positive controls FAIL | 60743389 | see below |
| F6 | PASS | 60730613 | memory mass in the gap exactly 0 in all three banks (transient memory gap mass at q = 0: 69,535); gates and drift correct |
| F7 | PASS | 60730613 | flag 1: transient labels all 0 at begin, the masses reported and recomputed |
| F8 | PASS | 60730613 | `tr_labels` saved, reload refused, every transient lifecycle method raises after the freeze |
| F9 | PASS | run 60736117 (lprod), test 60750535 and 60750999 (60750297 failed on the test's superseded 0.20 residual rule, fixed in `96f6fe0`) | carton, shortened schedule; the three refusals and the plumbing pass: exit 0, trees unchanged, flag resolved 1; memory mass in the gap 0.0, residual outside 0.0, label-1 transient share of the gap region 0.0, frozen; **8 labelled rows eligible over 3 lift selections, 2 labelled copies of 6,000** |

**F5 and F5c.** In every run (two fresh processes each, four jobs) every witness (200 label-1 transient rows that paint the
gap view when the gate is open) has **all nine parameter groups exactly 0 with the gate shut**, under both losses, and the
deterministic fixture row has all nine nonzero with the gate open and all nine exactly 0 with it shut (F5c, and F5 in job
60733454). The **positive controls** fail:

- **Colour channels:** with the static set in view, 31–36 witnesses have a zero colour gradient in an unclamped channel; all
  36 are explained by the static row at the same index writing its clamp flag into the transient bank's buffer (the escalated
  `clamped` defect, `rasterizer_impl.cu:160`). With the static set moved out of view (F5c) these anomalies vanish (594 of
  597 unclamped channels positive).
- **Whole-row zeros:** 10 witnesses (11 at the gap view), the same rows in both processes, have exactly zero rendering
  gradient in every group on the training build although they contribute (contribution at least 11.5). This persists with
  the static set out of view and is not resolved on the build that writes `max_contrib`. Cause not established.
- **Run-to-run:** transient probe gradients differ between fresh processes (up to 319 absolute at the gap view), also with
  the static set out of view.
- **Reading (corrected after Codex round 5):** the exact zeros with the gate shut, and the forward skip of a zero opacity
  (`F:651-654`), support the gate's exactness. The failed positive controls limit the row-by-row gradient evidence: a
  clamp flag overwritten from true to false could also *enable* a suppressed derivative, so "the defect can only add zeros"
  (rev of this draft) is withdrawn. The failures are not isolated to the substrate by a paired gate-off control; Codex
  found no label-path suppression at `q = 0` by reading the code (factor 1; donor labels on the final mask; split order
  matches). **Escalated to the principal** as an observation whose attribution is unresolved (beside the `clamped` and
  `max_contrib` defects): some transient rows get no training-build gradient although they render, and transient gradients
  differ between fresh processes. Settling it would take paired gate-on / gate-off backward checks, compute-sanitizer
  (memcheck, then initcheck), and showing that fixing the identified fault removes the failures.

**The premise check (descriptive; this changes the recommendation).**

1. **Low measured exposure of memory rows to the lift on the early trajectory.** The counter
   `labelled_candidates_excluded` sums, over all 139 lift selections of each confirmatory early run, the labelled rows above
   the threshold at each selection, before the subsample (`cer_drift.json`): bottle 0 / 0 / 0, carton 45 / 12 / 16, mug
   1 / 0 / 6 candidate occurrences. Each selection draws 2,000 of about 15,000 eligible rows; there were 64,000–85,000
   transient rows at the freeze. F9 agrees (8 eligible, 2 labelled copies). **Qualification (Codex round 5):** this is
   exposure on the early trajectory, not a bound on the fix's effect; a restored pool also changes the `randperm` draw and
   so later unlabelled selections and training; realised copies, descendants and their contribution are not counted.
2. **The transient rows that paint the object interior in base runs are mostly centred in time inside the gap.** Job 60742905
   (`agent-control/ars/cer/return_diag/transient_time/transient_time_interior.json`), cam00, interior mass at frames 40 and 57:
   the share of transient interior mass from rows centred in time inside the gap is 0.83–1.00 on the mug, 0.52–0.99 on the
   bottle at frame 57, and 0.10–1.00 on the carton (runs 1 and 3 partly centred before the gap); 0.43–1.00 of it comes from
   rows with any contribution (mass > 0) in the gap region at frame 74. In the early arm (run 1 per scene) the transient share
   of interior mass is 0.000–0.001. Limits: two frames, cam00, composited contribution mass (not the bank-alone alpha of
   the 2026-10-16 diagnostic), one early run per scene, no re-check of the diagnostic's renders against the saved renders.
3. **Reading (qualified after Codex round 5).** The data are consistent with base Multi4D's transient "detail" on the object
   being mostly the time tails of rows centred in the gap, and with the lift exclusion having had little to exclude. They do
   not establish that these rows are an occluder, that the detail causes the interior PSNR difference, or that the fix could
   not recover detail by another route (a contribution or colour ablation would be needed). The 2026-10-16 entry's "most
   likely mechanism" (the lift exclusion) is unestablished on this evidence (append-only: the entry stands as written).

**Recommendation (for the principal).** Defer the section 8 comparison (≈ 26 GPU-h): the measured exposure of memory rows to
the lift is low, so the fix is unlikely to change much, although this is a prioritisation, not a proof. The code stays on
its branch, tested as above; no scorer method was written.

**Codex round 5 dispositions.**

| # | finding | disposition |
|---|---|---|
| 1 | the new launcher records the exit code before its tree check; F9 does not check the trees | accepted: exit code recorded last (`train_multi4d_cer_early_tr.sbatch`); F9 requires `trees_changed 0` and equal before/after states (`c087731`). The same order exists in the pre-existing early launcher (not edited) |
| 2 | "the defect can only add zeros" is false | accepted: withdrawn (F5 reading above) |
| 3 | candidate counts do not bound the fix's effect | accepted: "low measured exposure", qualified |
| 4 | temporal overlap does not establish occluder tails or causation | accepted: "consistent with"; the causal claim needs an ablation |
| 5 | the task-B blanket conclusion contradicts the foreground result | accepted: the plan-log reading is rewritten (the early arm's foreground memory rows vanish by 3,300; mixed-batch loss unmeasured) |
| suspected | the F5 failures not isolated to the substrate | stated as open; a paired gate-off control or initcheck would settle it |
| missing | the time diagnostic lacks a saved-render fidelity check; its limits | the limits are stated; the fidelity check is not run |


**Codex code review round 2 (focused): CODE READY WITH FIXES, READINGS QUALIFIED** (`review_code_r2_codex.md`).

| # | finding | disposition |
|---|---|---|
| 1 | F9 does not require the carton scene | accepted: `8e3d2d0`; F9 rerun PASS (job 60750999) |
| 2 | the premise heading still said "occluder's tails"; "no longer most likely" overstated | accepted: "centred in time inside the gap"; the earlier mechanism "unestablished" |
| 3 | a bullet said the foreground gradient is 0 | accepted: ≈ 1.1e-7, with the unequal windows (2 vs 69) stated |
| suspected | "substrate finding" outruns the missing paired control | accepted: "observation, attribution unresolved"; the settling steps are listed |
| missing | unequal foreground observation windows | accepted: stated beside the comparison |

The orchestrator is satisfied with the code (`5afa119`) and its tests at `8e3d2d0`: every code finding of both rounds is fixed,
and what remains open is attribution in the substrate, not the gate.
