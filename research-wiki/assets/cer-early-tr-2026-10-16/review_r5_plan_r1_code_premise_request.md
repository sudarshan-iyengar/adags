# Review request: (1) PLAN rev 5 round 5, (2) CODE review round 1, (3) independent check of a premise refutation and a diagnostic reading
# Project: CER on Multi4D, early gate with a gated transient copy path (exploratory)

You are the same independent adversarial reviewer (scientific + engineering + tensor shapes + CUDA). Three parts, in order.
Workspace D:\ . State at the top that you could open D:\adags-cer\multi4d-early-tr\scene\cer.py (quote one line).

PART 1 (plan, round 5). The plan rev 5 is inlined below (section 13 answers your round-4 review, which is also inlined). Check the
round-4 fixes in D:\adags-cer\agent\densify_probe.sbatch, densify_read.py, test_densify_probe.py. Verdict for the plan.

PART 2 (code, round 1). The implementation is on branch cer-early-tr in D:\adags-cer\multi4d-early-tr: code commit 5afa119 on top of
7b30ef8 (the code diff is inlined below; read the files themselves too), tests in test_cer_early_tr.py (commits 5edd293, e857935,
36024cf, 65e0eb9, c7866b6, b5254ee). Check per plan section 5: scientific (no path lets a memory row, a copy or a descendant paint a
gap frame; flag-1 lift = base lift; flag 0 = early arm), engineering (labels/rows agree after every transient op; no RNG; flag 0
untouched; launchers D:\adags-cer\agent\train_multi4d_cer_early_tr.sbatch and cer_tests_m4d_early_tr.sbatch isolate the worktree and
extensions), tensor shapes ([N] uint8 labels vs [N,1] factor and opacity; .repeat(N) vs .repeat(N,1); the lift labels vs xyz after
randperm), CUDA (zero transient opacity skipped in forward and the launched backward). The implementer's stated decisions are in
the "implementation record" below; judge each. The GPU test results are below too: the gate clauses pass; F5/F5c positive controls
fail in ways attributed in part to the escalated `clamped` defect and in part unexplained (10 rows with all-zero training-build
gradient). Judge whether the tests support "the gate is exact" and whether anything in the CODE (not the substrate) could cause the
positive-control failures (e.g. could apply_transient or the label path zero a row that should not be gated?).

PART 3 (independent check, the most important for the decision). The orchestrator concludes the plan's premise is false (draft
section 14 below, "The premise check"): (i) the early arm's lift exclusion removed only 0-45 labelled candidates per run over 139
lift selections (cer_drift.json "labelled_candidates_excluded"; check the counting semantics in D:\adags-cer\multi4d-early\scene\cer.py
count_lift / mask_lift_scores: is it a per-selection count of labelled rows above threshold, summed?), so restoring the lift changes
a few dozen rows; (ii) in base runs the transient rows painting the object interior are mostly centred in time in the gap and also
paint the gap region (script D:\adags-cer\agent\transient_time_interior.py; results inlined). Is the refutation sound? Is anything
in the measurement wrong (time-centre mapping t -> frame; the interior mask; mass under occlusion vs bank-alone alpha; the gap-region
mask at frame 59; the cam00 test camera; loading base models)? Is the recommendation "do not run the 26 GPU-h comparison" justified?
Also check the task-B reading (draft plan-log entry below; reader output inlined): does the data support "memory rows are not
densified less in the early arm"?

Output: three verdict lines (PLAN: READY/READY WITH FIXES/MATERIAL DEFECT; CODE: same; PREMISE CHECK: SOUND/QUALIFIED/UNSOUND), then
numbered findings per part with severity, evidence (file:line) and the concrete fix. Terse.

=============================== PART 1: THE PLAN (rev 5) ===============================
# CER on Multi4D, early gate with a gated transient copy path: plan (rev 5, 2026-10-16, exploratory)

Status: **rev 5, for review.** Codex rounds 1–4 returned MATERIAL DEFECT (6, 3, 3 and 2 confirmed findings; round 4's were
all in the task-B probe); every text and review is in [assets/cer-early-tr-2026-10-16/](../assets/cer-early-tr-2026-10-16/),
the dispositions in sections 10–13. Nothing is written or run. Exploratory (spec §14.5, principal decision (c)): every run id,
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

=============================== YOUR ROUND-4 PLAN REVIEW ===============================
## Verdict

MATERIAL DEFECT

Independent judgment; no ARIS skill used.

## Confirmed defects

1. **Moderate — a failed probe job can still be accepted.** The launcher writes `"exit_code=$rc"` before its worktree check can change `rc` from 0 to 1 ([densify_probe.sbatch:78](/D:/adags-cer/agent/densify_probe.sbatch:78), line 79). The reader trusts that recorded value ([densify_read.py:14](/D:/adags-cer/agent/densify_read.py:14)). An in-memory reproduction accepted a completed payload recording `exit_code=0` and a changed worktree. **Fix:** finalize all checks before recording the exit code; require matching before/after substrate states in the reader.

2. **Moderate — the neutrality test still overstates its coverage.** Its digest lists nine tensors and reads only `gr["params"][0]` and the two Adam moment buffers ([test_densify_probe.py:182](/D:/adags-cer/agent/test_densify_probe.py:182), lines 185–189). It omits foreground buffers such as `mean_radii2D` and `_deformation_accum`, deformation parameters, optimizer step counters and group metadata. Foreground groups contain parameter lists ([gaussian_model.py:321](/D:/adags-cer/multi4d-early/scene/gaussian_model.py:321)). Changing omitted buffers and a step counter left the extracted digest unchanged. **Fix:** compare every relevant parameter/buffer and the complete optimizer state, including all parameters within each group. [PyTorch’s state format](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.state_dict.html) includes both per-parameter state and group metadata.

## Suspected issues

- **Whole-training neutrality remains unestablished.** The probe allocates CUDA tensors during accounting ([densify_probe.py:69](/D:/adags-cer/agent/densify_probe.py:69)); the training backward reads `max_contrib` ([backward.cu:1229](/D:/adags-cer/multi4d-early/diff_gaussian_rasterization_hybrid/cuda_rasterizer/backward.cu:1229)), whose initialization is already an acknowledged substrate problem. The neutrality sequence exercises lifecycle calls, without rendering/backward ([test_densify_probe.py:210](/D:/adags-cer/agent/test_densify_probe.py:210)). **Settle:** compare paired wrapped/unwrapped training segments including rendering and backward, or narrow the neutrality claim.

## Unverified claims

- F5’s nine positive controls and the proposed transient implementation remain unexecuted here: the available Python environment has no PyTorch.
- Historical PSNR/alpha values, donor counts and GPU-hour estimates were not independently recomputed from primary outputs.
- D1’s unconditional validation and the revised F5 fixture are specified fixes, not verified implementation.

## Missing

- A regression assertion for the **corrected** average: the test named “both averages over the same rows” checks only `grad_mean_seen`, never `grad_u_mean_seen` ([test_densify_probe.py:149](/D:/adags-cer/agent/test_densify_probe.py:149)). Add the round-3 `[2,0]` example and assert both means equal 1, with one zero corrected denominator.
=============================== PART 2: CODE DIFF 7b30ef8..5afa119 (tests excluded) ===============================
diff --git a/arguments/__init__.py b/arguments/__init__.py
index ce5f77f..778e124 100644
--- a/arguments/__init__.py
+++ b/arguments/__init__.py
@@ -60,6 +60,9 @@ class ModelParams(ParamGroup):
         self.cer_scene = ""
         # -1 = CER begins at densify_until_iter (the legacy arm); otherwise the fine iteration at which it begins.
         self.cer_begin_iter = -1
+        # 1 (early mode only) = memory rows may be foreground-to-transient lift donors, their transient copies are memory
+        # rows, and the transient bank is gated like the persistent ones; 0 = the early gate unchanged.
+        self.cer_transient_gate = 0
         super().__init__(parser, "Loading Parameters", sentinel)
 
     def extract(self, args):
diff --git a/gaussian_renderer/__init__.py b/gaussian_renderer/__init__.py
index 6baec64..cce6de4 100644
--- a/gaussian_renderer/__init__.py
+++ b/gaussian_renderer/__init__.py
@@ -421,6 +421,7 @@ def render_rawall3pc(viewpoint_camera, pc_foreground: GaussianModel, pc_backgrou
     means3D = pc_transient.get_xyz
     means2D = screenspace_points_transient
     opacity = pc_transient.get_opacity
+    opacity = cer_gate.apply_transient(pc_transient, viewpoint_camera, opacity)
 
     scales_t = None
     rotations_r = None
diff --git a/scene/__init__.py b/scene/__init__.py
index 17286bc..1ca1fb5 100644
--- a/scene/__init__.py
+++ b/scene/__init__.py
@@ -35,6 +35,7 @@ class Scene2gs_mixed:
         self.gaussians_transient = gaussians_transient
         self.cer_scene = getattr(args, "cer_scene", "")
         self.cer_begin_iter = getattr(args, "cer_begin_iter", -1)
+        self.cer_transient_gate = getattr(args, "cer_transient_gate", 0)
 
         if load_iteration:
             self.loaded_iter = searchForMaxIteration(os.path.join(load_iteration, "point_cloud"))
@@ -120,10 +121,12 @@ class Scene2gs_mixed:
         if self.gaussians_transient is not None:
             self.gaussians_transient.save_ply(os.path.join(point_cloud_path, "point_cloud_transient.ply"))
         if hasattr(self.gaussians_second, "cer_labels"):
-            torch.save({"fg_labels": self.gaussians.cer_labels, "bg_labels": self.gaussians_second.cer_labels,
-                        "clock": self.gaussians_second.cer_clock, "q": self.gaussians_second.cer_q,
-                        "active_sh_degree_t": self.gaussians_transient.active_sh_degree_t},
-                       os.path.join(point_cloud_path, "cer_state.pt"))
+            state = {"fg_labels": self.gaussians.cer_labels, "bg_labels": self.gaussians_second.cer_labels,
+                     "clock": self.gaussians_second.cer_clock, "q": self.gaussians_second.cer_q,
+                     "active_sh_degree_t": self.gaussians_transient.active_sh_degree_t}
+            if hasattr(self.gaussians_transient, "cer_labels"):  # the transient gate (cer_transient_gate 1)
+                state["tr_labels"] = self.gaussians_transient.cer_labels
+            torch.save(state, os.path.join(point_cloud_path, "cer_state.pt"))
 
         try:
             self.gaussians_second.save_deformation(point_cloud_path)
diff --git a/scene/cer.py b/scene/cer.py
index 300457b..a203142 100644
--- a/scene/cer.py
+++ b/scene/cer.py
@@ -131,9 +131,10 @@ def mass(fg, bg, tr, pipe, views):
 mass.__wrapped__ = _mass  # the same computation on the training build (T4b's descriptive comparison)
 
 
-def check_finite(fg, bg):
-    """A NaN opacity times a 0 factor renders at alpha 0.99; Multi4D has no NaN prune."""
-    for m in (fg, bg):
+def check_finite(fg, bg, tr=None):
+    """A NaN opacity times a 0 factor renders at alpha 0.99; Multi4D has no NaN prune. The transient bank is covered too when
+    it is gated (tr given, cer_transient_gate 1)."""
+    for m in (fg, bg) if tr is None else (fg, bg, tr):
         assert torch.isfinite(torch.sigmoid(m._opacity)).all(), "non-finite opacity"
 
 
@@ -179,6 +180,16 @@ def check_begin_range(cer_scene, cer_begin_iter, opt):
         "is also before it)" % (cer_begin_iter, opt.iterations, opt.densify_until_iter)
 
 
+def check_transient_gate(dataset):
+    """cer_transient_gate must be 0 or 1, and 1 needs early mode (cer_begin_iter set, with cer_scene). Asserted in training()
+    before the scene is built, whatever the mode."""
+    flag = dataset.cer_transient_gate
+    assert flag in (0, 1), "cer_transient_gate must be 0 or 1, not %r" % (flag,)
+    assert flag == 0 or (dataset.cer_begin_iter != -1 and dataset.cer_scene), \
+        "cer_transient_gate 1 needs early mode: cer_begin_iter set (it is %d) and cer_scene (it is %r)" \
+        % (dataset.cer_begin_iter, dataset.cer_scene)
+
+
 def check_labels(m):
     """Label invariant of a persistent bank: a 1-D uint8 cuda tensor with one entry per row."""
     lab = m.cer_labels
@@ -186,6 +197,13 @@ def check_labels(m):
         "CER labels %s %s (cuda %s) against %d rows" % (tuple(lab.shape), lab.dtype, lab.is_cuda, m.get_xyz.shape[0])
 
 
+def check_transient_labels(m):
+    """Label invariant of the transient bank under the transient gate: that of a persistent bank, with values in {0, 1}
+    (the residual cohort, 2, is background only)."""
+    check_labels(m)
+    assert not bool((m.cer_labels > 1).any()), "a transient row carries a label above 1"
+
+
 def mask_lift_scores(fg, deformed_state, threshold):
     """Make labelled foreground rows ineligible as donors of the foreground-to-transient lift: their dynamic score is set
     to -inf (never above any finite threshold, a negative one included) on a clone of the score, which only the donor
@@ -200,10 +218,12 @@ def mask_lift_scores(fg, deformed_state, threshold):
     return excluded
 
 
-def count_lift(scene, stage, prune_data, excluded=0, masked=False):
+def count_lift(scene, stage, prune_data, excluded=0, masked=False, eligible=None):
     """Count one lift selection (a non-empty donor set) by stage, and the labelled candidates a selection excluded.
     `fine_after_begin` counts the selections made while the gate is armed; `masked_selections` those among them made
-    with the labelled rows masked out."""
+    with the labelled rows masked out. With the transient gate (`eligible` given: the labelled rows above the threshold at
+    this selection) the labelled rows stay donors, and after begin the copies made, the labelled copies made and the
+    labelled rows eligible are counted too."""
     lifts = scene.__dict__.setdefault("cer_lifts", {"coarse": 0, "fine": 0, "fine_after_begin": 0, "masked_selections": 0,
                                                     "labelled_candidates_excluded": 0})
     lifts["labelled_candidates_excluded"] += excluded
@@ -213,6 +233,14 @@ def count_lift(scene, stage, prune_data, excluded=0, masked=False):
             lifts["fine_after_begin"] += 1
         if masked:
             lifts["masked_selections"] += 1
+    if eligible is not None and cer_gate.ARMED:
+        for k in ("copies", "labelled_copies", "labelled_eligible"):
+            lifts.setdefault(k, 0)
+        lifts["labelled_eligible"] += eligible
+        if prune_data is not None and prune_data["xyz"].shape[0] > 0:
+            lifts["copies"] += int(prune_data["xyz"].shape[0])
+            if "cer_labels" in prune_data:
+                lifts["labelled_copies"] += int((prune_data["cer_labels"] != 0).sum())
 
 
 def lift_select(fg, deformed_state, threshold, select_kwargs, scene, stage):
@@ -220,13 +248,18 @@ def lift_select(fg, deformed_state, threshold, select_kwargs, scene, stage):
     run and every legacy run before begin) it is exactly fg.select_fg_to_transient(deformed_state,
     dynamic_score_threshold=threshold, **select_kwargs). Armed with a labelled foreground bank, labelled rows are first made
     ineligible (mask_lift_scores asserts, before the selection, that the eligibility set holds no labelled row) and the
-    selection is counted when the scene runs CER. Returns the donor dict or None."""
-    masked, excluded = False, 0
+    selection is counted when the scene runs CER. With the transient gate (scene.cer_transient_gate 1) nothing is masked: the
+    donor set is base Multi4D's, each copy carries its donor's label (select_fg_to_transient), and the labelled rows above the
+    threshold are counted. Returns the donor dict or None."""
+    masked, excluded, eligible = False, 0, None
     if cer_gate.ARMED and hasattr(fg, "cer_labels"):
-        excluded, masked = mask_lift_scores(fg, deformed_state, threshold), True
+        if getattr(scene, "cer_transient_gate", 0) == 1:
+            eligible = int(((deformed_state["dynamic_score"] > threshold) & (fg.cer_labels != 0)).sum())
+        else:
+            excluded, masked = mask_lift_scores(fg, deformed_state, threshold), True
     prune_data = fg.select_fg_to_transient(deformed_state, dynamic_score_threshold=threshold, **select_kwargs)
     if scene.cer_scene:
-        count_lift(scene, stage, prune_data, excluded, masked)
+        count_lift(scene, stage, prune_data, excluded, masked, eligible)
     return prune_data
 
 
@@ -235,6 +268,8 @@ def begin(fg, bg, tr, scene, pipe, opt, iteration):
     Legacy mode installs the sentinels here; early mode leaves them to freeze()."""
     A, B, ids = event(scene.cer_scene)
     mode = begin_mode(scene)
+    flag = getattr(scene, "cer_transient_gate", 0)
+    assert flag == 0 or mode == "early", "cer_transient_gate %d needs early mode" % flag
     if mode == "legacy":
         assert iteration == opt.densify_until_iter and iteration >= getattr(opt, "phase3_start_iter", 10000)
     else:
@@ -244,13 +279,16 @@ def begin(fg, bg, tr, scene, pipe, opt, iteration):
     lifts_before = lifts.get("coarse", 0) + lifts.get("fine", 0)
     assert mode == "legacy" or lifts_before == 0, \
         "early begin after %d lift selections: a labelled donor could have been lifted" % lifts_before
-    check_finite(fg, bg)
+    check_finite(fg, bg, tr if flag else None)
     report = open(os.path.join(scene.model_path, "cer_begin.json"), "x")  # a NaN restart must not begin twice
     inside, total = mass(fg, bg, tr, pipe, _region(scene, range(0, A)))
     memory = [(t > 0) & (i >= 0.5 * t) for i, t in zip(inside[:2], total[:2])]
     for m, mem in zip((fg, bg), memory):
         m.cer_labels, m.cer_clock, m.cer_q = mem.to(torch.uint8), (A, B), 1.0
+    if flag:  # the transient bank: every row label 0 (labels arise only by inheritance from a labelled lift donor), never 2
+        tr.cer_labels, tr.cer_clock, tr.cer_q = torch.zeros(tr.get_xyz.shape[0], dtype=torch.uint8, device="cuda"), (A, B), 1.0
     cer_gate.ARMED = True
+    cer_gate.TRANSIENT = bool(flag)
     # donors: background rows that paint the gap region during the gap once the memory cohort is off
     g_in, g_tot = mass(fg, bg, tr, pipe, _region(scene, range(A, B + 1), mask_frame=A - 1))
     majority = [(t > 0) & (i >= 0.5 * t) for i, t in zip(g_in, g_tot)]
@@ -273,18 +311,24 @@ def begin(fg, bg, tr, scene, pipe, opt, iteration):
 
     gap = sum(x.sum() for x in g_in).item()
     share = lambda x: x / gap if gap > 0 else None  # a gap region with no mass is reported, not divided by
-    json.dump({"A": A, "B": B, "target_ids": ids, "iteration": iteration, "begin_iter": iteration, "mode": mode,
-               "lifts_before_begin": lifts_before,
-               "lifts_before_begin_by_stage": {"coarse": lifts.get("coarse", 0), "fine": lifts.get("fine", 0)},
-               "memory_rows": {"fg": int(memory[0].sum()), "bg": int(memory[1].sum())},
-               "memory_share_pregap": ((inside[0][memory[0]].sum() + inside[1][memory[1]].sum())
-                                       / sum(x.sum() for x in inside)).item(),
-               "residual_rows": k,
-               "gap_region_share_by_bank": {b: share(g_in[i].sum().item()) for i, b in enumerate(("fg", "bg", "tr"))},
-               "majority_donor_share_by_bank": {b: share(g_in[i][majority[i]].sum().item())
-                                                for i, b in enumerate(("fg", "bg", "tr"))},
-               "rows_after_birth": {"fg": fg.get_xyz.shape[0], "bg": bg.get_xyz.shape[0], "tr": tr.get_xyz.shape[0]}},
-              report, indent=1)
+    memory_rows = {"fg": int(memory[0].sum()), "bg": int(memory[1].sum())}
+    if flag:
+        memory_rows["tr"] = int((tr.cer_labels == 1).sum())
+    out = {"A": A, "B": B, "target_ids": ids, "iteration": iteration, "begin_iter": iteration, "mode": mode,
+           "cer_transient_gate": flag, "lifts_before_begin": lifts_before,
+           "lifts_before_begin_by_stage": {"coarse": lifts.get("coarse", 0), "fine": lifts.get("fine", 0)},
+           "memory_rows": memory_rows,
+           "memory_share_pregap": ((inside[0][memory[0]].sum() + inside[1][memory[1]].sum())
+                                   / sum(x.sum() for x in inside)).item(),
+           "residual_rows": k,
+           "gap_region_share_by_bank": {b: share(g_in[i].sum().item()) for i, b in enumerate(("fg", "bg", "tr"))},
+           "majority_donor_share_by_bank": {b: share(g_in[i][majority[i]].sum().item())
+                                            for i, b in enumerate(("fg", "bg", "tr"))},
+           "rows_after_birth": {"fg": fg.get_xyz.shape[0], "bg": bg.get_xyz.shape[0], "tr": tr.get_xyz.shape[0]}}
+    if flag:  # the transient rows (all ungated, label 0) at begin: their pre-gap target mass and all-pixel gap mass
+        out["transient_at_begin"] = {"rows": tr.get_xyz.shape[0], "pregap_target_mass": inside[2].sum().item(),
+                                     "gap_all_pixel_mass": g_tot[2].sum().item()}
+    json.dump(out, report, indent=1)
     report.close()
 
 
@@ -295,21 +339,26 @@ def freeze(fg, bg, tr, scene, pipe, opt, iteration):
     The drift report is one pre-gap mass pass on the current (gated) model: the memory rule of begin (total > 0 and
     inside >= 0.5 total) is evaluated on every row, and per persistent bank it reports the rows that now meet the rule
     but are labelled 0 and the rows labelled 1 that no longer meet it, each with its share of the total pre-gap
-    target mass (inside mass, all three banks). Residual rows are exactly 0 before the gap, so they never meet the rule."""
+    target mass (inside mass, all three banks); with the transient gate (labelled transient bank) the transient bank is
+    reported too, with the same fields. Residual rows are exactly 0 before the gap, so they never meet the rule."""
     assert begin_mode(scene) == "early", "freeze is for early mode"
     assert cer_gate.ARMED and hasattr(bg, "cer_labels"), "CER never began"
     assert iteration == opt.densify_until_iter and iteration < opt.iterations, \
         "freeze at fine iteration %d, densify_until_iter is %d, iterations %d" % (iteration, opt.densify_until_iter, opt.iterations)
     A, B = bg.cer_clock
-    check_finite(fg, bg)
+    tl = hasattr(tr, "cer_labels")  # the transient gate (cer_transient_gate 1)
+    check_finite(fg, bg, tr if tl else None)
     check_labels(fg)
     check_labels(bg)
+    if tl:
+        check_transient_labels(tr)
     report = open(os.path.join(scene.model_path, "cer_drift.json"), "x")
     inside, total = mass(fg, bg, tr, pipe, _region(scene, range(0, A)))
     denom = sum(x.sum() for x in inside).item()
     share = lambda x: x / denom if denom > 0 else None  # a pre-gap target with no mass has no shares
     per_bank = {}
-    for name, m, i, t in (("fg", fg, inside[0], total[0]), ("bg", bg, inside[1], total[1])):
+    for name, m, i, t in (("fg", fg, inside[0], total[0]), ("bg", bg, inside[1], total[1])) + (
+            (("tr", tr, inside[2], total[2]),) if tl else ()):
         meets = (t > 0) & (i >= 0.5 * t)
         lab = m.cer_labels
         up, down = meets & (lab == 0), (lab == 1) & ~meets
@@ -331,14 +380,18 @@ def gates(fg, bg, tr, scene, pipe):
     """Contribution gates of spec section 6 on the final in-memory model; writes cer_gates.json."""
     assert cer_gate.ARMED, "CER never began"
     A, B = bg.cer_clock
-    check_finite(fg, bg)
+    tl = hasattr(tr, "cer_labels")  # the transient gate (cer_transient_gate 1): transient memory rows count in the cohorts
+    check_finite(fg, bg, tr if tl else None)
     pre_in, pre_tot = mass(fg, bg, tr, pipe, _region(scene, range(0, A)))
     gap_in, gap_tot = mass(fg, bg, tr, pipe, _region(scene, range(A, B + 1), mask_frame=A - 1))
     ret_in, ret_tot = mass(fg, bg, tr, pipe, _region(scene, range(B + 1, B + 31)))
 
     def cohort(x, c):
-        """Mass of cohort c over the foreground and background banks."""
-        return (x[0][fg.cer_labels == c].sum() + x[1][bg.cer_labels == c].sum()).item()
+        """Mass of cohort c over the foreground and background banks (and the transient bank when it carries labels)."""
+        s = x[0][fg.cer_labels == c].sum() + x[1][bg.cer_labels == c].sum()
+        if tl:
+            s = s + x[2][tr.cer_labels == c].sum()
+        return s.item()
 
     def whole(x):
         """Mass of every row of every bank."""
@@ -355,7 +408,16 @@ def gates(fg, bg, tr, scene, pipe):
            "residual_mass_outside_gap": cohort(pre_tot, 2) + cohort(ret_tot, 2),
            "transient_share_gap_region": gap_in[2].sum().item() / whole(gap_in),
            "memory_rows": {"fg": int((fg.cer_labels == 1).sum()), "bg": int((bg.cer_labels == 1).sum())},
-           "rows_at_end": {"fg": fg.get_xyz.shape[0], "bg": bg.get_xyz.shape[0], "tr": tr.get_xyz.shape[0]}}
+           "rows_at_end": {"fg": fg.get_xyz.shape[0], "bg": bg.get_xyz.shape[0], "tr": tr.get_xyz.shape[0]},
+           "cer_transient_gate": getattr(scene, "cer_transient_gate", 0)}
+    if tl:  # descriptive: memory by bank (label 1 inside the target), the transient memory in the gap region, the lift counts
+        by_bank = lambda x, den: {b: (x[i][m.cer_labels == 1].sum() / den).item()
+                                  for i, (b, m) in enumerate((("fg", fg), ("bg", bg), ("tr", tr)))}
+        out["memory_rows"]["tr"] = int((tr.cer_labels == 1).sum())
+        out["memory_share_pregap_by_bank"] = by_bank(pre_in, whole(pre_in))
+        out["memory_share_return_by_bank"] = by_bank(ret_in, whole(ret_in))
+        out["memory_transient_share_gap_region"] = (gap_in[2][tr.cer_labels == 1].sum() / whole(gap_in)).item()
+        out["lifts"] = dict(getattr(scene, "cer_lifts", {}))
     out["pass"] = (out["memory_share_pregap"] >= 0.5 and out["memory_share_return"] >= 0.5
                    and out["memory_mass_gap"] == 0.0 and out["residual_share_gap"] >= 0.2
                    and out["residual_mass_outside_gap"] == 0.0)
diff --git a/scene/gaussian_model.py b/scene/gaussian_model.py
index 25e5d30..0727bd1 100644
--- a/scene/gaussian_model.py
+++ b/scene/gaussian_model.py
@@ -469,7 +469,7 @@ class GaussianModel_dynamic(PersistentBranchMixin, GaussianBranchBase):
         current_prob = torch.sigmoid(opacity)
         opacity = inverse_sigmoid(torch.clamp(current_prob + 0.05, max=0.999))
 
-        return {
+        out = {
             "xyz": xyz,
             "features_dc": features_dc,
             "features_rest": features_rest,
@@ -479,6 +479,9 @@ class GaussianModel_dynamic(PersistentBranchMixin, GaussianBranchBase):
             "t_ref": t_ref,
             "velocity": velocity,
         }
+        if hasattr(self, "cer_labels"):  # CER: copy k carries the label of the k-th selected donor (same mask as xyz)
+            out["cer_labels"] = self.cer_labels[prune_mask]
+        return out
 
     def weight_prune(self, weight_threshold: float, mean_ratio):
         """
@@ -1373,7 +1376,8 @@ class GaussianModelTransient(GaussianBranchBase):
                 self._rotation_r = nn.Parameter(rots_r.requires_grad_(True))
 
 
-    def add_points(self, xyz, features_dc, features_rest, scaling, rotation, opacity, t_ref, velocity=None, scaling_t_div=10, velocity_scale=1.0):
+    def add_points(self, xyz, features_dc, features_rest, scaling, rotation, opacity, t_ref, velocity=None, scaling_t_div=10, velocity_scale=1.0,
+                   new_cer_labels=None):
         num_points = xyz.shape[0]
         new_t = None
         new_scaling_t = None
@@ -1410,9 +1414,15 @@ class GaussianModelTransient(GaussianBranchBase):
                 features_rest = torch.cat([features_rest, padding], dim=1)
 
         self.densification_postfix_nodenom(xyz, features_dc, features_rest, opacity, scaling, rotation,
-                                   new_t, new_scaling_t, new_rotation_r)
+                                   new_t, new_scaling_t, new_rotation_r, new_cer_labels=new_cer_labels)
     def densification_postfix_nodenom(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
-                              new_rotation, new_t, new_scaling_t, new_rotation_r):
+                              new_rotation, new_t, new_scaling_t, new_rotation_r, new_cer_labels=None):
+        if hasattr(self, "cer_labels"):  # checked before any row is added
+            assert new_cer_labels is not None and new_cer_labels.shape == (new_xyz.shape[0],) \
+                and self.cer_labels.shape[0] == self.get_xyz.shape[0], "CER labels do not match the rows being added"
+        else:  # an unlabelled bank takes unlabelled rows only (cer_transient_gate 0: every lifted label is 0)
+            assert new_cer_labels is None or not bool((new_cer_labels != 0).any()), \
+                "labelled rows lifted into a transient bank without CER labels"
         d = {"xyz": new_xyz,
              "f_dc": new_features_dc,
              "f_rest": new_features_rest,
@@ -1452,6 +1462,8 @@ class GaussianModelTransient(GaussianBranchBase):
         self.max_radii2D = torch.cat((self.max_radii2D, torch.full((num_new,), 0.1, device="cuda")), dim=0)
         self.mean_radii2D = torch.cat((self.mean_radii2D, torch.zeros((num_new), device="cuda")), dim=0)
         self.t_gradient_accum = torch.cat((self.t_gradient_accum, torch.zeros((num_new, 1), device="cuda")), dim=0)
+        if hasattr(self, "cer_labels"):  # new rows carry their donors' (lift) CER labels
+            self.cer_labels = torch.cat([self.cer_labels, new_cer_labels])
 
     def add_points_from_dict(self, data_dict, velocity_scale=1.0):
         if data_dict is None:
@@ -1466,6 +1478,7 @@ class GaussianModelTransient(GaussianBranchBase):
             data_dict["t_ref"],
             velocity=data_dict.get("velocity", None),
             velocity_scale=velocity_scale,
+            new_cer_labels=data_dict.get("cer_labels"),
         )
 
     def training_setup(self, training_args):
@@ -1547,6 +1560,8 @@ class GaussianModelTransient(GaussianBranchBase):
             if self.rot_4d:
                 self._rotation_r = optimizable_tensors['rotation_r']
             self.t_gradient_accum = self.t_gradient_accum[valid_points_mask]
+        if hasattr(self, "cer_labels"):  # transient CER membership follows its rows through every prune
+            self.cer_labels = self.cer_labels[valid_points_mask]
 
     def weight_prune(self, weight_threshold: float):
         """
@@ -1588,7 +1603,11 @@ class GaussianModelTransient(GaussianBranchBase):
         return optimizable_tensors
 
     def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
-                              new_rotation, new_t, new_scaling_t, new_rotation_r, new_mean_radii2D, new_max_radii2D):
+                              new_rotation, new_t, new_scaling_t, new_rotation_r, new_mean_radii2D, new_max_radii2D,
+                              new_cer_labels=None):
+        if hasattr(self, "cer_labels"):  # checked before any row is added
+            assert new_cer_labels is not None and new_cer_labels.shape == (new_xyz.shape[0],) \
+                and self.cer_labels.shape[0] == self.get_xyz.shape[0], "CER labels do not match the rows being added"
         d = {"xyz": new_xyz,
              "f_dc": new_features_dc,
              "f_rest": new_features_rest,
@@ -1620,6 +1639,8 @@ class GaussianModelTransient(GaussianBranchBase):
         self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
         self.max_radii2D = torch.cat([self.max_radii2D, new_max_radii2D], -1)
         self.mean_radii2D = torch.cat([self.mean_radii2D, new_mean_radii2D], -1)
+        if hasattr(self, "cer_labels"):  # new rows carry their parents' CER labels
+            self.cer_labels = torch.cat([self.cer_labels, new_cer_labels])
 
     def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
         n_init_points = self.get_xyz.shape[0]
@@ -1670,9 +1691,11 @@ class GaussianModelTransient(GaussianBranchBase):
 
         new_max_radii2D = self.max_radii2D[selected_pts_mask].repeat(N)
         new_mean_radii2D = self.mean_radii2D[selected_pts_mask].repeat(N)
+        new_cer_labels = self.cer_labels[selected_pts_mask].repeat(N) if hasattr(self, "cer_labels") else None
 
         self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
-                                   new_t, new_scaling_t, new_rotation_r, new_mean_radii2D, new_max_radii2D)
+                                   new_t, new_scaling_t, new_rotation_r, new_mean_radii2D, new_max_radii2D,
+                                   new_cer_labels=new_cer_labels)
 
         prune_filter = torch.cat(
             (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
@@ -1703,9 +1726,11 @@ class GaussianModelTransient(GaussianBranchBase):
 
         new_max_radii2D = self.max_radii2D[selected_pts_mask]
         new_mean_radii2D = self.mean_radii2D[selected_pts_mask]
+        new_cer_labels = self.cer_labels[selected_pts_mask] if hasattr(self, "cer_labels") else None
 
         self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
-                                   new_rotation, new_t, new_scaling_t, new_rotation_r, new_mean_radii2D, new_max_radii2D)
+                                   new_rotation, new_t, new_scaling_t, new_rotation_r, new_mean_radii2D, new_max_radii2D,
+                                   new_cer_labels=new_cer_labels)
 
     def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_grad_t=None):
         grads = self.xyz_gradient_accum / self.denom
diff --git a/train.py b/train.py
index 5267537..302fc57 100644
--- a/train.py
+++ b/train.py
@@ -928,7 +928,8 @@ def scene_reconstruction(optimization_params, hypernetwork_config, pipeline_conf
                 # Dynamic-score threshold for FG→TR donor pre-filter (default 0.05).
                 _ds_thresh = float(getattr(optimization_params, 'fg_to_tr_dynamic_score_threshold', 0.05))
                 # CER: lift_select is select_fg_to_transient, except that while the gate is armed a labelled foreground
-                # row is never a donor (the lift copies rows and leaves the donor alive)
+                # row is never a donor (the lift copies rows and leaves the donor alive); with cer_transient_gate 1 it is a
+                # donor again and its copy carries its label
                 prune_data = cer.lift_select(
                     foreground_gaussians, deformed_state, _ds_thresh,
                     dict(max_transfer=_max_transfer,
@@ -977,6 +978,8 @@ def scene_reconstruction(optimization_params, hypernetwork_config, pipeline_conf
                                    viewspace_point_tensor_list_transient,
                                    fg_to_tr_active, prune_data, weight_prune_fre_transient,
                                    phase3_start_iter)
+                if cer_gate.ARMED and scene.cer_begin_iter != -1 and scene.cer_transient_gate:
+                    cer.check_transient_labels(transient_gaussian)
             # ---- Transient Model Optimization ----
             if iteration < optimization_params.iterations:
                 transient_gaussian.optimizer.step()
@@ -1024,6 +1027,7 @@ def scene_reconstruction(optimization_params, hypernetwork_config, pipeline_conf
 def training(dataset, hypernetwork_config, optimization_params, pipeline_config, saving_iterations,
              checkpoint_iterations, checkpoint, expname):
     """Entry-point that prepares data structures and launches the two-stage training."""
+    cer.check_transient_gate(dataset)  # whatever the mode: a transient gate without the early gate is refused here
     if dataset.cer_begin_iter != -1:  # early CER: refuse an invalid start before anything is created
         cer.check_begin_range(dataset.cer_scene, dataset.cer_begin_iter, optimization_params)
     prepare_output_folder(expname)
diff --git a/utils/cer_gate.py b/utils/cer_gate.py
index 1d1e8bd..149d3e1 100644
--- a/utils/cer_gate.py
+++ b/utils/cer_gate.py
@@ -7,6 +7,7 @@ import re
 import torch
 
 ARMED = False  # set by scene/cer.py begin(); once armed, a model without CER state may not be rendered
+TRANSIENT = False  # set by begin() with cer_transient_gate 1: the transient bank is gated too
 _PATH = re.compile(r"cam(\d\d)/images/(\d{4})\.png$")
 
 
@@ -41,3 +42,10 @@ def apply(pc, cam, opacity):
     f = factor(pc, cam)
     assert f.shape == opacity.shape, "CER labels %s vs opacity %s" % (tuple(f.shape), tuple(opacity.shape))
     return opacity * f
+
+
+def apply_transient(pc, cam, opacity):
+    """The transient bank's activated opacity: unchanged unless the transient gate is set, else gated as apply gates it."""
+    if not TRANSIENT:
+        return opacity
+    return apply(pc, cam, opacity)

=============================== IMPLEMENTATION RECORD, GPU TESTS AND PREMISE CHECK (draft plan section 14) ===============================

## 14. Implementation, tests, and the premise check (2026-10-16 evening; appended)

**Review state, stated first.** Codex reviewed the plan in four rounds (MATERIAL DEFECT each: 6, 3, 3 and 2 confirmed
findings; round 4's were all in the task-B probe; every finding was fixed, sections 10–13). Round 5 could not run: Codex hit
its usage limit at 17:00, resetting at 21:28. Round 4 raised nothing against the task-A design, so the implementation was
started during the wait. **The code has not yet been reviewed by Codex.** The GPU tests below ran on the unreviewed code.

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
| F9 | PASS (integration, lprod) | 60736117 | carton, shortened schedule: exit 0, trees unchanged, flag resolved 1; memory mass in the gap 0.0, residual outside 0.0, label-1 transient share of the gap region 0.0, frozen; **8 labelled rows eligible over 3 lift selections, 2 labelled copies of 6,000** |

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
- **Reading:** these are properties of the hybrid rasterizer's transient backward, present in every Multi4D arm. They can
  only add zeros, so the gate's exactness (the zeros) stands; what is not shown is that the training build trains every
  transient row. **Escalated to the principal** as a third substrate finding beside `clamped` and `max_contrib`.

**The premise check (descriptive; this changes the recommendation).**

1. **The lift exclusion removed almost nothing in the early arm.** Its counter `labelled_candidates_excluded`, summed over all
   139 lift selections of each confirmatory early run (`cer_drift.json`): bottle 0 / 0 / 0, carton 45 / 12 / 16, mug 1 / 0 /
   6. Each selection draws 2,000 rows from about 15,000 eligible ones, so without the exclusion these would have yielded at
   most a few dozen copies over the run, against 64,000–85,000 transient rows at the freeze. F9 agrees (8 eligible, 2 copies).
2. **The transient rows that paint the object interior in base runs are mostly the gap occluder's tails.** Job 60742905
   (`agent-control/ars/cer/return_diag/transient_time/transient_time_interior.json`), cam00, interior mass at frames 40 and 57:
   the share of transient interior mass from rows centred in time inside the gap is 0.83–1.00 on the mug, 0.52–0.99 on the
   bottle at frame 57, and 0.10–1.00 on the carton (runs 1 and 3 partly centred before the gap); 0.43–1.00 of it comes from
   rows that also paint the gap region at frame 74. In the early arm the transient share of interior mass is 0.000–0.001.
3. **Reading.** Source (1) of the 2026-10-16 diagnostic is not a lost lift path. Base Multi4D's transient "detail" on the
   object is mostly the time tail of the occluder it learns for the gap; the memory gate removes the object from the
   persistent banks in the gap, so no occluder forms. The planned fix cannot restore that detail without restoring the
   occluder. The 2026-10-16 entry's "most likely mechanism" is withdrawn on this evidence (append-only: the entry stands as
   written).

**Recommendation (for the principal).** Do not run the section 8 comparison (≈ 26 GPU-h): by point 1 the fix changes a few
dozen rows per run. The code stays on its branch, tested as above, unreviewed by Codex; no scorer method was written.

=============================== DRAFT PLAN-LOG ENTRY (task A and task B readings) ===============================

### 2026-10-16 evening: the two sources of the early arm's object deficit, tested (exploratory; no training of an arm)

Brief from the principal (2026-10-16): (A) plan, review, implement and test a fix for the missing transient detail (memory rows
lift donors again, their transient copies gated as memory); prepare, not submit, a comparison. (B) Measure whether memory rows
are densified less in the early arm before any densification change. The record is
[cer-multi4d-early-tr-plan-2026-10-16](cer-multi4d-early-tr-plan-2026-10-16.md) (plan rev 5, sections 1–13; implementation,
tests and premise check, section 14). Reviews: `assets/cer-early-tr-2026-10-16/`. Every job id below is in the ledger.

**Task A, the fix (transient copies of memory rows gated as memory).**

- **Plan:** five revisions; Codex rounds 1–4 each MATERIAL DEFECT (6, 3, 3, 2 findings), all fixed. Round 4 found nothing in
  task A. Round 5 and the code review wait for the Codex usage limit (reset 21:28); the implementation went ahead during the wait.
- **Code:** branch `cer-early-tr`, code at `5afa119` (tests to `b5254ee`), flag `cer_transient_gate` (0 = the early arm, shown
  bit-identical by F0). GPU tests: F0–F4, F6–F8 PASS; F9 integration PASS; F5/F5c gate clauses PASS (exact zeros for every
  witness in all nine parameter groups) with positive-control failures traced to the transient backward of the substrate
  (one part to the escalated `clamped` buffer; 10 rows with entirely zero gradient, cause not established).
- **The premise does not hold.** The early arm's lift exclusion removed 0–45 labelled candidates per run over 139 selections
  (bottle 0/0/0, carton 45/12/16, mug 1/0/6), so restoring the lift changes a few dozen rows. In base runs the transient rows that
  paint the object interior before the gap are mostly centred in the gap (mug 0.83–1.00, bottle 0.52–0.99 at frame 57, carton
  0.10–1.00 of their mass) and 0.43–1.00 of their mass is from rows that also paint the gap region: the occluder's tails. The
  memory gate removes the need for the occluder, so this "detail" cannot be restored without it. The 2026-10-16 entry's "most
  likely mechanism" is withdrawn on this evidence.
- **Recommendation:** do not run the comparison (≈ 26 GPU-h). Nothing was submitted for training.

**Task B, densification of memory rows (bottle same-pose, one run each, exploratory diagnostics).**

- **Probe:** `agent-control/ars/densify_probe.py` (runpy around `cer-early@7b30ef8` `train.py`), its test (17 checks PASS, jobs
  60728104/60728741/60728898/60729394), reader `densify_read.py`. Runs: early arm 60729395 (2:49), shadow base 60729396 (2:32;
  CER off, the memory rule applied at fine 3,000 without arming the gate). Output
  `agent-control/ars/cer/densify_probe/densify_read_bottle_r1.json`.
- **Background memory rows (3,349 of the early arm's 3,388 memory rows at 3,000), steps 3,100–15,900, early / shadow:** mean average gradient 1.011×
  (median 1.026×); clone selections 5 / 5; split 0 / 0; rows at 16,000 2,652 / 2,667 (0.994×); visibility count per window
  99.6 / 99.6. Increments that carry no gradient for a gated row (every view of the batch that saw the row was a gap view) are
  1.8% of the count, not ≈ 10%; deleting them raises the average by 1.9% and changes no selection.
- **Other background rows:** gradient 1.02×; clones 213 / 208; splits 668 / 850 (0.79×).
- **Foreground memory rows:** 39 (early) / 93 (shadow) at 3,000; in the early arm their average gradient is 0 and all are
  gone by 16,000; the shadow base keeps 18 after 54 splits. A small cohort (39 of 3,388 memory rows in the early arm).
- **Reading:** memory rows are not densified less in the early arm; the diluted-statistics and fewer-views candidates do not
  explain the persistent banks' −0.28 dB (bottle). No densification-statistics change is proposed. Unverified: one run per arm,
  one scene; the two runs already differ at fine 3,000 by GPU nondeterminism.

**Cost.** Task B 5.9 GPU-h (probe tests 0.5, runs 5.4). Task A tests 3.3 GPU-h (F9 1.1 on lprod). Premise check 0.05 GPU-h.

**For the principal.** (1) Whether to drop the fix (recommended) or still run the comparison. (2) The new substrate finding:
some transient rows get no training-build gradient although they render, and transient gradients differ between fresh
processes. (3) Where the object deficit comes from remains open: the occluder's tails explain the transient part; the
persistent part (bottle −0.28, carton −0.16, mug +0.20 dB) is not explained by densification statistics.

=============================== transient_time_interior OUTPUT (job 60742905, per run and frame) ===============================
ars_w1_ARS-L1-A0_multi4d_rep1 {"scene": "ARS-L1-A0", "cer": false, "rows_tr": 85824, "frames": {"40": {"transient_share_of_interior_mass": 0.00016473136693868147, "centre_before_gap": 0.0, "centre_in_gap": 0.8354347432336982, "centre_after_gap": 0.16456525676630201, "also_paints_gap_region_at_74": 1.0, "mean_abs_centre_offset_frames": 66.61883814328301, "rows_with_mass": 4}, "57": {"transient_share_of_interior_mass": 0.05431678933253757, "centre_before_gap": 0.013343455306140433, "centre_in_gap": 0.9862365213911067, "centre_after_gap": 0.000420023302753006, "also_paints_gap_region_at_74": 0.4381921928160588, "mean_abs_centre_offset_frames": 9.523330776340591, "rows_with_mass": 139}}}
ars_w1_ARS-L1-A0_multi4d_rep2 {"scene": "ARS-L1-A0", "cer": false, "rows_tr": 97372, "frames": {"40": {"transient_share_of_interior_mass": 0.00023919809726479922, "centre_before_gap": 0.0, "centre_in_gap": 0.9858320890857363, "centre_after_gap": 0.014167910914263691, "also_paints_gap_region_at_74": 1.0, "mean_abs_centre_offset_frames": 40.2979733581599, "rows_with_mass": 8}, "57": {"transient_share_of_interior_mass": 0.006864585839123759, "centre_before_gap": 0.14906868583532168, "centre_in_gap": 0.8504864836601601, "centre_after_gap": 0.0004448305045183296, "also_paints_gap_region_at_74": 0.8504124352947502, "mean_abs_centre_offset_frames": 11.3325431063755, "rows_with_mass": 52}}}
ars_w1_ARS-L1-A0_multi4d_rep3 {"scene": "ARS-L1-A0", "cer": false, "rows_tr": 85027, "frames": {"40": {"transient_share_of_interior_mass": 1.759158642423969e-06, "centre_before_gap": 1.0, "centre_in_gap": 0.0, "centre_after_gap": 0.0, "also_paints_gap_region_at_74": 1.0, "mean_abs_centre_offset_frames": 1.1826210021972656, "rows_with_mass": 1}, "57": {"transient_share_of_interior_mass": 0.010996891138774688, "centre_before_gap": 0.4754183656085873, "centre_in_gap": 0.5245816343914127, "centre_after_gap": 0.0, "also_paints_gap_region_at_74": 0.5149775552683189, "mean_abs_centre_offset_frames": 7.392964167969227, "rows_with_mass": 73}}}
ars_w1_ARS-L1-A0_m4dearly_rep1 {"scene": "ARS-L1-A0", "cer": true, "rows_tr": 71749, "frames": {"40": {"transient_share_of_interior_mass": 9.080785564476115e-05, "centre_before_gap": 0.004440597204174813, "centre_in_gap": 0.9697600519158008, "centre_after_gap": 0.025799350880024288, "also_paints_gap_region_at_74": 0.9999999999999999, "mean_abs_centre_offset_frames": 36.144217991753834, "rows_with_mass": 6}, "57": {"transient_share_of_interior_mass": 0.0006807055887538074, "centre_before_gap": 0.09192517920817711, "centre_in_gap": 0.9043479525144194, "centre_after_gap": 0.0037268682774035736, "also_paints_gap_region_at_74": 0.7940901015982142, "mean_abs_centre_offset_frames": 13.972823203576477, "rows_with_mass": 20}}}
ars_w1_ARS-L2-A0_multi4d_rep1 {"scene": "ARS-L2-A0", "cer": false, "rows_tr": 92741, "frames": {"40": {"transient_share_of_interior_mass": 0.005845029609982033, "centre_before_gap": 0.6830301009747412, "centre_in_gap": 0.2237804549710335, "centre_after_gap": 0.09318944405422531, "also_paints_gap_region_at_74": 0.9172162146365106, "mean_abs_centre_offset_frames": 19.3094462975527, "rows_with_mass": 38}, "57": {"transient_share_of_interior_mass": 0.011448785938804563, "centre_before_gap": 0.4536541630105353, "centre_in_gap": 0.47109334977821, "centre_after_gap": 0.07525248721125469, "also_paints_gap_region_at_74": 0.6916282081790684, "mean_abs_centre_offset_frames": 16.483206716773307, "rows_with_mass": 59}}}
ars_w1_ARS-L2-A0_multi4d_rep2 {"scene": "ARS-L2-A0", "cer": false, "rows_tr": 73157, "frames": {"40": {"transient_share_of_interior_mass": 0.002197221646187656, "centre_before_gap": 0.0, "centre_in_gap": 0.9995433956715807, "centre_after_gap": 0.0004566043284193633, "also_paints_gap_region_at_74": 1.0, "mean_abs_centre_offset_frames": 28.432600076288054, "rows_with_mass": 6}, "57": {"transient_share_of_interior_mass": 0.0070380573243491356, "centre_before_gap": 0.05460238546733029, "centre_in_gap": 0.9452620677295156, "centre_after_gap": 0.00013554680315411877, "also_paints_gap_region_at_74": 0.9282337314734735, "mean_abs_centre_offset_frames": 12.510710893461173, "rows_with_mass": 22}}}
ars_w1_ARS-L2-A0_multi4d_rep3 {"scene": "ARS-L2-A0", "cer": false, "rows_tr": 77433, "frames": {"40": {"transient_share_of_interior_mass": 0.0014671796898978998, "centre_before_gap": 0.756318050839746, "centre_in_gap": 0.09805864952204248, "centre_after_gap": 0.14562329963821138, "also_paints_gap_region_at_74": 0.9703949784790679, "mean_abs_centre_offset_frames": 20.443601793850146, "rows_with_mass": 17}, "57": {"transient_share_of_interior_mass": 0.005065432162124071, "centre_before_gap": 0.5805557723266423, "centre_in_gap": 0.3408669476044576, "centre_after_gap": 0.07857728006889997, "also_paints_gap_region_at_74": 0.46821117780258964, "mean_abs_centre_offset_frames": 12.595342966905083, "rows_with_mass": 46}}}
ars_w1_ARS-L2-A0_m4dearly_rep1 {"scene": "ARS-L2-A0", "cer": true, "rows_tr": 77172, "frames": {"40": {"transient_share_of_interior_mass": 4.2297175441255414e-05, "centre_before_gap": 0.0, "centre_in_gap": 0.0, "centre_after_gap": 1.0, "also_paints_gap_region_at_74": 1.0, "mean_abs_centre_offset_frames": 172.31993103027344, "rows_with_mass": 1}, "57": {"transient_share_of_interior_mass": 4.174643619445184e-05, "centre_before_gap": 0.0, "centre_in_gap": 0.0, "centre_after_gap": 1.0, "also_paints_gap_region_at_74": 1.0, "mean_abs_centre_offset_frames": 155.31993103027344, "rows_with_mass": 1}}}
ars_w1_ARS-L3-A0_multi4d_rep1 {"scene": "ARS-L3-A0", "cer": false, "rows_tr": 73695, "frames": {"40": {"transient_share_of_interior_mass": 0.016145653166955373, "centre_before_gap": 0.0, "centre_in_gap": 0.9990986320523784, "centre_after_gap": 0.0009013679476215297, "also_paints_gap_region_at_74": 0.9999999999999998, "mean_abs_centre_offset_frames": 42.79009191899304, "rows_with_mass": 12}, "57": {"transient_share_of_interior_mass": 0.02027131468964075, "centre_before_gap": 0.02539295067477478, "centre_in_gap": 0.9740688331674408, "centre_after_gap": 0.0005382161577845835, "also_paints_gap_region_at_74": 0.9589401758534497, "mean_abs_centre_offset_frames": 22.724971291357125, "rows_with_mass": 38}}}
ars_w1_ARS-L3-A0_multi4d_rep2 {"scene": "ARS-L3-A0", "cer": false, "rows_tr": 76489, "frames": {"40": {"transient_share_of_interior_mass": 0.01808622934724622, "centre_before_gap": 0.0, "centre_in_gap": 0.9983102306412934, "centre_after_gap": 0.001689769358706581, "also_paints_gap_region_at_74": 0.9983131648065864, "mean_abs_centre_offset_frames": 44.79512545947115, "rows_with_mass": 20}, "57": {"transient_share_of_interior_mass": 0.024349298467791006, "centre_before_gap": 0.005407670597828478, "centre_in_gap": 0.9939160204780201, "centre_after_gap": 0.0006763089241513661, "also_paints_gap_region_at_74": 0.9962468465196173, "mean_abs_centre_offset_frames": 23.482030254275255, "rows_with_mass": 52}}}
ars_w1_ARS-L3-A0_multi4d_rep3 {"scene": "ARS-L3-A0", "cer": false, "rows_tr": 70201, "frames": {"40": {"transient_share_of_interior_mass": 6.141642539432595e-05, "centre_before_gap": 0.0, "centre_in_gap": 0.8962280544303329, "centre_after_gap": 0.10377194556966698, "also_paints_gap_region_at_74": 0.9677473744745065, "mean_abs_centre_offset_frames": 42.90777802698501, "rows_with_mass": 7}, "57": {"transient_share_of_interior_mass": 0.005902652426247724, "centre_before_gap": 0.16892543125668383, "centre_in_gap": 0.8305629741178566, "centre_after_gap": 0.0005115946254595191, "also_paints_gap_region_at_74": 0.43236430021425376, "mean_abs_centre_offset_frames": 7.9108171028483145, "rows_with_mass": 30}}}
ars_w1_ARS-L3-A0_m4dearly_rep1 {"scene": "ARS-L3-A0", "cer": true, "rows_tr": 84800, "frames": {"40": {"transient_share_of_interior_mass": 2.915928928131368e-06, "centre_before_gap": 0.04846313049725658, "centre_in_gap": 0.0, "centre_after_gap": 0.9515368695027434, "also_paints_gap_region_at_74": 0.04620966127591235, "mean_abs_centre_offset_frames": 170.566821694545, "rows_with_mass": 3}, "57": {"transient_share_of_interior_mass": 1.436377749338285e-07, "centre_before_gap": 1.0, "centre_in_gap": 0.0, "centre_after_gap": 0.0, "also_paints_gap_region_at_74": 1.0, "mean_abs_centre_offset_frames": 37.85073471069336, "rows_with_mass": 1}}}

=============================== densify_read OUTPUT (bottle, early vs shadow) ===============================
{
 "window": {
  "bg": [
   3100,
   15900
  ],
  "fg": [
   3100,
   9900
  ]
 },
 "snapshots": {
  "early": {
   "snapshot_labelling": {
    "iteration": 3000,
    "fg": {
     "0": 34340,
     "1": 39,
     "2": 0
    },
    "bg": {
     "0": 135798,
     "1": 3349,
     "2": 226
    },
    "rows": {
     "fg": 34379,
     "bg": 139373
    }
   },
   "snapshot_16000": {
    "iteration": 16000,
    "fg": {
     "0": 10426,
     "1": 0,
     "2": 0
    },
    "bg": {
     "0": 83754,
     "1": 2652,
     "2": 96
    },
    "rows": {
     "fg": 10426,
     "bg": 86502
    }
   }
  },
  "shadow": {
   "snapshot_labelling": {
    "iteration": 3000,
    "fg": {
     "0": 34337,
     "1": 93,
     "2": 0
    },
    "bg": {
     "0": 137726,
     "1": 3271,
     "2": 0
    },
    "rows": {
     "fg": 34430,
     "bg": 140997
    }
   },
   "snapshot_16000": {
    "iteration": 16000,
    "fg": {
     "0": 10749,
     "1": 18,
     "2": 0
    },
    "bg": {
     "0": 83860,
     "1": 2667,
     "2": 0
    },
    "rows": {
     "fg": 10767,
     "bg": 86527
    }
   }
  }
 },
 "bg": {
  "early": {
   "steps": 129,
   "0": {
    "steps_with_rows": 129,
    "grad_mean_seen": 5.6783378673825676e-05,
    "grad_u_mean_seen": 5.933304142785374e-05,
    "grad_q50_seen": 3.820633119457923e-05,
    "frac_ge_thr_seen": 6.08850169210341e-05,
    "denom_mean_seen": 81.82203331289365,
    "zero_inc_per_row": 3.196833219345975,
    "all_gap_per_row": 0.889524839156776,
    "zero_inc_share_of_denom": 0.03905939639722322,
    "mixed_per_row": 14.69966401469529,
    "clone": 213,
    "split": 668,
    "clone_undiluted": 424,
    "split_undiluted": 985,
    "rows_first": 135798,
    "rows_last": 83754,
    "iter_last": 15900
   },
   "1": {
    "steps_with_rows": 129,
    "grad_mean_seen": 9.68445192836586e-05,
    "grad_u_mean_seen": 9.866178927689523e-05,
    "grad_q50_seen": 8.833587268126457e-05,
    "frac_ge_thr_seen": 1.2552529679928986e-05,
    "denom_mean_seen": 99.61456145057383,
    "zero_inc_per_row": 1.7960600425079702,
    "all_gap_per_row": 1.0721865037194473,
    "zero_inc_share_of_denom": 0.01803017948167991,
    "mixed_per_row": 17.871886291179596,
    "clone": 5,
    "split": 0,
    "clone_undiluted": 5,
    "split_undiluted": 0,
    "rows_first": 3349,
    "rows_last": 2652,
    "iter_last": 15900
   },
   "2": {
    "steps_with_rows": 129,
    "grad_mean_seen": 5.132358364437462e-06,
    "grad_u_mean_seen": 5.262296923767506e-06,
    "grad_q50_seen": 3.1117026383459556e-06,
    "frac_ge_thr_seen": 0.0,
    "denom_mean_seen": 99.53030442821887,
    "zero_inc_per_row": 2.1169283437610673,
    "all_gap_per_row": 1.079978751032936,
    "zero_inc_share_of_denom": 0.021269340481428988,
    "mixed_per_row": 17.90792114272223,
    "clone": 0,
    "split": 0,
    "clone_undiluted": 0,
    "split_undiluted": 0,
    "rows_first": 226,
    "rows_last": 96,
    "iter_last": 15900
   }
  },
  "shadow": {
   "steps": 129,
   "0": {
    "steps_with_rows": 129,
    "grad_mean_seen": 5.554261935135718e-05,
    "grad_u_mean_seen": 5.8035593695443145e-05,
    "grad_q50_seen": 3.719963056627343e-05,
    "frac_ge_thr_seen": 7.252615455928567e-05,
    "denom_mean_seen": 81.553688108459,
    "zero_inc_per_row": 3.201972557512025,
    "all_gap_per_row": 0.8866204712104253,
    "zero_inc_share_of_denom": 0.03925612464765663,
    "mixed_per_row": 14.65033428400225,
    "clone": 208,
    "split": 850,
    "clone_undiluted": 503,
    "split_undiluted": 1228,
    "rows_first": 137726,
    "rows_last": 83860,
    "iter_last": 15900
   },
   "1": {
    "steps_with_rows": 129,
    "grad_mean_seen": 9.576643702008693e-05,
    "grad_u_mean_seen": 9.758528259656367e-05,
    "grad_q50_seen": 8.613390602914133e-05,
    "frac_ge_thr_seen": 1.2992084154483768e-05,
    "denom_mean_seen": 99.60030122505601,
    "zero_inc_per_row": 1.8065746960292492,
    "all_gap_per_row": 1.0710558200833262,
    "zero_inc_share_of_denom": 0.018138371873118474,
    "mixed_per_row": 17.870522808434657,
    "clone": 5,
    "split": 0,
    "clone_undiluted": 8,
    "split_undiluted": 0,
    "rows_first": 3271,
    "rows_last": 2667,
    "iter_last": 15900
   }
  },
  "rows_at_16000_early_over_shadow": {
   "0": 0.9987359885523491,
   "1": 0.9943757030371203
  },
  "memory_early_over_shadow": {
   "grad_mean_seen": 1.011257412274256,
   "grad_q50_seen": 1.0255644583374433,
   "frac_ge_thr_seen": 0.966167516363948,
   "denom_mean_seen": 1.0001431745219886,
   "clone": 1.0,
   "split": null
  },
  "memory_undiluted_over_diluted_early": {
   "grad": 1.0187648202157298,
   "clone": 1.0,
   "split": null
  },
  "other_early_over_shadow": {
   "grad_mean_seen": 1.022338869447614,
   "clone": 1.0240384615384615,
   "split": 0.7858823529411765
  }
 },
 "fg": {
  "early": {
   "steps": 69,
   "0": {
    "steps_with_rows": 69,
    "grad_mean_seen": 0.00014182676913797098,
    "grad_u_mean_seen": 0.00014406375840485123,
    "grad_q50_seen": 0.00012761266196426678,
    "frac_ge_thr_seen": 0.010611833100193653,
    "denom_mean_seen": 99.14636086726534,
    "zero_inc_per_row": 1.4507681640017485,
    "all_gap_per_row": 1.100851816219333,
    "zero_inc_share_of_denom": 0.014684467271031692,
    "mixed_per_row": 17.648700988015875,
    "clone": 7357,
    "split": 2972,
    "clone_undiluted": 8247,
    "split_undiluted": 3490,
    "rows_first": 34340,
    "rows_last": 10385,
    "iter_last": 9900
   },
   "1": {
    "steps_with_rows": 2,
    "grad_mean_seen": 1.0772971137384957e-07,
    "grad_u_mean_seen": 1.1113788289662807e-07,
    "grad_q50_seen": 4.173727496947777e-08,
    "frac_ge_thr_seen": 0.0,
    "denom_mean_seen": 99.51282119750977,
    "zero_inc_per_row": 2.6153846153846154,
    "all_gap_per_row": 1.5,
    "zero_inc_share_of_denom": 0.026281886111826847,
    "mixed_per_row": 20.0,
    "clone": 0,
    "split": 0,
    "clone_undiluted": 0,
    "split_undiluted": 0,
    "rows_first": 39,
    "rows_last": 39,
    "iter_last": 9900
   }
  },
  "shadow": {
   "steps": 69,
   "0": {
    "steps_with_rows": 69,
    "grad_mean_seen": 0.00014005911931314546,
    "grad_u_mean_seen": 0.00014224219447224522,
    "grad_q50_seen": 0.00012510998380278488,
    "frac_ge_thr_seen": 0.010992434948844753,
    "denom_mean_seen": 99.00766101781873,
    "zero_inc_per_row": 1.443397110244097,
    "all_gap_per_row": 1.0964833546393145,
    "zero_inc_share_of_denom": 0.014628375822916968,
    "mixed_per_row": 17.611734508335523,
    "clone": 7687,
    "split": 3222,
    "clone_undiluted": 8607,
    "split_undiluted": 3705,
    "rows_first": 34337,
    "rows_last": 10773,
    "iter_last": 9900
   },
   "1": {
    "steps_with_rows": 69,
    "grad_mean_seen": 0.00013655696459639896,
    "grad_u_mean_seen": 0.00013872994307256295,
    "grad_q50_seen": 0.0001222739981733906,
    "frac_ge_thr_seen": 0.05934328839614771,
    "denom_mean_seen": 99.88974872533826,
    "zero_inc_per_row": 1.5100436681222706,
    "all_gap_per_row": 1.1930131004366813,
    "zero_inc_share_of_denom": 0.015116146912511693,
    "mixed_per_row": 17.92838427947598,
    "clone": 0,
    "split": 54,
    "clone_undiluted": 0,
    "split_undiluted": 60,
    "rows_first": 93,
    "rows_last": 18,
    "iter_last": 9900
   }
  },
  "rows_at_16000_early_over_shadow": {
   "0": 0.9699506930877291,
   "1": 0.0
  },
  "memory_early_over_shadow": {
   "grad_mean_seen": 0.0007888994288372637,
   "grad_q50_seen": 0.00034134219533978304,
   "frac_ge_thr_seen": 0.0,
   "denom_mean_seen": 0.9962265644609346,
   "clone": null,
   "split": 0.0
  },
  "memory_undiluted_over_diluted_early": {
   "grad": 1.0316363190740507,
   "clone": null,
   "split": null
  },
  "other_early_over_shadow": {
   "grad_mean_seen": 1.0126207406807506,
   "clone": 0.9570703785612072,
   "split": 0.9224084419615146
  }
 }
}