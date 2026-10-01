# CER on Multi4D: implementation spec (2026-09-30, rev 1, before any Multi4D CER code exists)

The port of [CER on Ex4DGS (spec rev 2)](cer-ex4dgs-spec-2026-09-30.md) to Multi4D at the pinned commit `c483e82`,
for plan §4b. The Ex4DGS spec is the reference for the mathematics, the contribution-mass instrument, the cohort rules
and the gates; this page states only what differs on Multi4D and where it attaches. Code goes on branch `cer` in the
worktree `$W/repo/multi4d-cer`; the pinned checkout `$W/repo/multi4d` is never modified.

**Entry.** §4b gate 2 (census) passed in every layout on 2026-09-30 (plan log): persistent foreground + static
background carry 0.983–0.991 of the pre-gap target mass. **The static background bank carries 0.983–0.990 of it and
the persistent foreground bank ≤ 0.006**, so the memory cohort is, in practice, background rows.

**Sources.** Every `file:line` below was read at `c483e82` in a clean clone. The row-lifecycle study is
[`assets/cer-multi4d-2026-09-30/lifecycle_map.md`](../assets/cer-multi4d-2026-09-30/lifecycle_map.md). `R` =
`gaussian_renderer/__init__.py`, `GM` = `scene/gaussian_model.py`, `T` = `train.py`.

**Glossary.** As in the Ex4DGS spec. Banks: persistent foreground (`GaussianModel_dynamic`, deformed by a HexPlane
network), static background (`GaussianModel`, no time), transient (`GaussianModelTransient`, 4D Gaussians whose temporal
marginal is applied in CUDA). Fine iteration = the iteration counter of the fine `scene_reconstruction` call (20,000).

---

## 1. Mathematics: unchanged, applied to the foreground and background banks

Labels `c ∈ {0, 1, 2}` per row of the foreground and background banks, clock `[A, B]`, strength `q`; factor
`f = 1 − q·n(t)` for memory rows, `q·n(t)` for residual rows, 1 otherwise, `n(t) = 1` on the gap frames; the opacity
handed to the rasterizer is multiplied by `f` (float32 cuda `[N, 1]`). The same three exactness consequences hold for a
fixed model (Ex4DGS spec §1), with the same caveats: zero **rendering-loss** gradient to a switched-off row; not a
frozen row (Adam moments, and Multi4D's scale/aspect regularisers `T:814-828` and soft displacement-variance term
`T:723` still reach it).

**The transient bank is not gated** (plan §4b). It carries 0.9–1.7% of the pre-gap target mass (census); those rows keep
their own temporal marginal. Its rows are neither memory nor residual; its mass enters every denominator.

**Time.** A Multi4D camera has `time = f/300` with `f` the frame's position in the sorted `camXX/images` listing
(`scene/neural_3D_dataset_NDC.py:93-105`) and `image_path = .../camXX/images/<frame>.png` (`scene/dataset.py:11-41`). The
gap test uses the integer frame parsed from `image_path`, never the float time; the tests assert that it equals
`round(time·300)` for every camera used.

## 2. Where CER attaches, and when

### 2.1 Four render functions, one factor

Opacity is activated and then, in training only, multiplied by a random dropout vector in every render function. The
gate goes **after activation and before dropout** (for a factor in {0, 1} the two commute exactly, and the gate draws no
random numbers):

| render | site | bank(s) gated | used in |
|---|---|---|---|
| `render_background` `R:18` | after `R:86` (outside the bare `try/except` at `R:66-82`, which would swallow an error) | background | coarse and fine phase 2 (`T:542`), debug panel (`utils/eval_utils.py:242`) |
| `render_foreground` `R:121` | after `R:198` | foreground | phase 2 (`T:514`), debug panel (`eval_utils.py:235`) |
| `render_mask` `R:248` | after `R:314` | foreground | phase 2 (`T:505`; the recorded silent bypass: it sets the masks and region weights of the other two) |
| `render_rawall3pc` `R:351` | on `opacity_static` after the concatenation at `R:486`, before dropout at `R:492` | foreground then background, in the order of the concatenation | phase 3's only render (`T:450`), `run_evaluation` (`eval_utils.py:68`, `:177`), the project's final-render wrapper, debug panel (`:251`) |

From the start iteration (§2.2) onward only `render_rawall3pc` runs in training and in evaluation; the three standalone
sites are gated identically anyway, as the principal required, so that no render of a CER model anywhere shows
ungated rows, and the test renders every path (§7).

Each insertion is two lines of the form

```python
    if getattr(pc, "cer_clock", None) is not None:
        opacity = opacity * cer_factor(pc, viewpoint_camera)
```

(and for the hybrid, the concatenation of the foreground and background factors). `cer_factor` lives in a new file
`utils/cer_gate.py` (no import of the renderer, so no import cycle) and asserts that the label vector has the bank's row
count. The CER state is set on the model objects themselves (`cer_labels`, `cer_clock`, `cer_q`), not in `train.py`,
because the final render is made after `train()` returns, by a wrapper that holds only the models. No model class is
edited: a model without the attribute is a base model.

Not gated, and outside the training/evaluation path: the semantic scripts (`cluster_semantic_features.py`,
`render_cluster_by_text.py`, `train_semantic.py`) build opacity themselves. They are never run on CER models; the spec
records them as bypasses.

### 2.2 When CER starts: at the end of fine iteration `densify_until_iter` (16,000 of 20,000)

`maintain_foreground` (`T:944`), `maintain_background` (`T:955`) and `maintain_transient` (`T:965`) all require
`iteration < densify_until_iter` (16,000), and every row-count change, opacity reset and per-row statistic update lives
inside them (lifecycle map §2). The foreground → transient lift requires `iteration < fg_to_tr_end_iter` and is off after
9,950. So **from 16,000 on no bank's row count, reset or per-row statistic changes**; only parameters and Adam moments
move. This is stronger than Ex4DGS (where pruning continues), and no label has to be threaded through any densify or
prune path. There is no optimizer step at 20,000 (`T:950`, `:960`, `:973`), so CER trains for 3,999 steps
(16,001–19,999) and the evaluated model is the one after step 19,999. Base and CER arms are identical up to 16,000.

`begin` is called at the end of the loop body of fine iteration 16,000, after `run_evaluation` (`T:989-992`, which at
16,000 still scores the ungated model), outside the `torch.no_grad()` block, guarded by `stage == "fine"`. Preconditions
asserted in `begin`: `iteration == densify_until_iter`; `iteration ≥ phase3_start_iter` (only the hybrid render trains);
`fg_to_tr_end_iter ≤ iteration` (the lift is off).

**Resume** restores the foreground bank only and silently re-initialises the others (record); it is refused when CER is
on (`T:286-292`).

### 2.3 Configuration

One argument, `cer_scene` (the canonical ARS scene directory), registered where Multi4D registers its arguments
(`arguments/__init__.py`), so that `utils/params_utils.py` does not drop it silently (it applies a key only if the
argument exists, code map §8). Default `""` = CER off.

## 3. The contribution-mass instrument on Multi4D

The hybrid rasterizer takes the persistent set's colours only as SH coefficients (census report). `mass` therefore
replaces, for the duration of one render and on every bank, `_features_dc` by a leaf tensor whose DC term decodes to 1.0
in every channel (`0.5/SH_C0`, unclamped) and `_features_rest` by zeros, renders through `render_rawall3pc` (stage
`fine`, black background, dropout off, the current CER factor), takes `torch.autograd.grad` of
`(image[0]·mask).sum() + image[1].sum()` with respect to the three DC leaves, divides by `SH_C0`, and restores the
original tensors. The colour gradient of the static and transient sets in the hybrid backward is `α·T·dL/dpixel` summed
over the pixels a row reaches (per-pixel skip at `forward.cu:651-654` and in the launched backward at
`backward.cu:1373-1375`); the foreground deformation only adds to SH coefficient 16, which the render slices off
(`R:471`). No parameter `.grad` is touched. The census used the same colour route with a one-hot per bank and matched
the rendered alpha within 1e-6.

## 4. Cohorts

### 4.1 Memory (label 1), at the start

On the ungated model: pre-gap frames × training cameras × target masks; a foreground or background row is memory when
`total > 0` and `inside ≥ 0.5·total` (majority-region membership, as on Ex4DGS). Transient rows are never labelled.

### 4.2 Residual (label 2), at the start, from the memory-gated model, **background bank only**

On the memory-gated model: gap frames × training cameras × gap regions `G_c = M_c(A − 1)`; donors are the rows with
`total > 0` and `inside ≥ 0.5·total`, **restricted to the background bank**. A background row has no time, so its copy
is exact: the six parameter groups (`xyz`, `f_dc`, `f_rest`, `opacity`, `scaling`, `rotation`) are appended as they are,
through the background `cat_tensors_to_optimizer` (`GM:1122-1146`, which raises on a missing group rather than skipping
it); `max_radii2D` (`[D]`), `xyz_gradient_accum` and `denom` (`[D, 1]`) are extended by zeros for the new rows only (not
`densification_postfix`, `GM:1148-1168`, which resets the accumulators of every row; harmless after 16,000 but not
needed). Why background only: the census puts ≥ 98% of the target region on the background bank; a foreground donor
would need its deformed state at a chosen time and a conversion into a timeless row, and a transient donor a
conditioning of its 4D covariance. The birth report gives the donor mass by bank, so the exclusion is visible.

## 5. Persistence

The score is rendered from the in-memory models after `train()` returns (the wrapper), as for the base runs. The saved
point clouds (`Scene2gs_mixed.save`, `scene/__init__.py:104-128`, at 19,999, 20,000 and `best/`) do not carry the
labels. `save` therefore also writes `cer_state.pt` (labels per bank, clock, q) beside the plys when CER is on, and the
load path (`scene/__init__.py:61-89`) restores it when present and asserts that the label lengths equal the loaded row
counts. Multi4D writes no checkpoints in these runs (`checkpoint_iterations` empty; its resume path is broken), so the
save/load pair of point clouds is the round trip that exists; test T5 exercises it.

## 6. Gates

The Ex4DGS gates (spec §6), computed at the end of training by `train.py` on the in-memory model, training cameras only,
written to `<model_path>/cer_gates.json`: memory share of target mass ≥ 0.50 before the gap and after the return;
residual share of the gap region ≥ 0.20 in the gap; memory mass in the gap and residual mass outside it exactly 0
(guards). The transient share is reported. Every run is scored; plan §4 rule 4 acts in the reducer.

## 7. Tests, written before the code

One GPU job, one node, on the wave-1 Multi4D run of the bottle same-pose scene. Multi4D saves no checkpoints, so the
test model is loaded from that run's `iteration_20000` point clouds and deformation files with the census's loader
(which reproduced the saved final renders byte for byte), in the CER tree.

| id | test | pass |
|---|---|---|
| T0 | the unmodified checkout renders cam00 frames 0–299 twice through `render_rawall3pc` | byte-identical |
| T1 | the `cer` tree, CER off | byte-identical to T0 |
| T2 | `begin` (memory + background residual rows appended), `q = 0` | byte-identical to T0 on all frames; ≥ 1 residual row |
| T3 | `q = 1` | exact factors per bank and label for every frame; frames outside the gap byte-identical to T0, every gap frame differs; **the four render functions gate identically**: for a probe gap frame and a probe non-gap frame, each of `render_background`, `render_foreground`, `render_mask` and `render_rawall3pc` gives the same image with CER on as with the memory rows removed and the residual rows removed or kept as the factor dictates (and, for `render_mask`, the same mask channels); zero rendering gradient to the switched-off cohort |
| T4 | instrument and non-vacuity | frame mapping (`round(time·300)` equals the parsed frame); conservation of the mass against the ones-colour render within 1e-4; split within 1e-5; memory share ≥ 0.50 (ungated); exact zeros at `q = 1` |
| T4-forecast | memory share and donor count on the nine wave-1 Multi4D run-1 models | reported per scene |
| T5 | persistence | `save` then reload through the scene's load path into fresh models: labels, clock, q equal; renders at the probe frames byte-identical; a reload without `cer_state.pt` of a model saved with CER on fails |
| T6 | the lifecycle is frozen | after `begin`, a short training from the loaded state is not possible (no resume), so T6 is T7's assertion that the row counts at 20,000 equal those at 16,000 plus the residual count |
| T7 | integration (not evidence) | the real entry point with CER on a shortened schedule (fine iterations shortened so that 16,000 → a smaller `densify_until_iter` with `phase3_start_iter` and `fg_to_tr_end_iter` below it), `begin` and gates written, no NaN, row counts frozen after `begin`, the wrapper's final render equal to a direct gated render |

## 8. CUDA: no change

The gate is a python multiply before the autograd `Function` on the persistent set's `opacities_static`, which gets a
gradient (input slot 16, `gaussian_renderer/diff_gaussian_rasterization_hybrid.py:299`) and on the standalone
rasterizers' `opacities`; zero-opacity rows are skipped per pixel in forward and in the backward kernel actually launched
(`PerGaussianRenderCUDA`); the transient CUDA gate is untouched. The factor must be float32 cuda (`data_ptr<float>()`).

## 9. Outside the substrate

A launcher `train_multi4d_cer.sbatch` (the wave-1 launcher from the `cer` worktree, `--cer_scene`, provenance with the
`cer` commit and worktree cleanliness, lrdn0066 excluded); the wrapper `m4d_final_render.py` is reused unchanged (it
renders through `render_rawall3pc`, which is gated). Scoring: a method `multi4d_cer` in `ars_score.py` with the same
provenance rule as `ex4dgs_cer`, the gate verdict carried; the runs added to `ars_score_worker.py`. Runs: §4b —
confirmatory CER-Multi4D ×3 on the three same-pose scenes against the three wave-1 Multi4D runs per scene (runs 2–3 are
already submitted); nothing is submitted until the principal says "submit".

## 10. Decisions this spec takes

1. CER starts at fine 16,000 (the frozen-lifecycle point) inside the unchanged 20,000-iteration budget.
2. The transient bank is not gated (plan §4b as written); its 1–2% of the target is left to its own marginal.
3. Residual donors from the background bank only (§4.2).
4. Persistence by a sidecar beside the saved point clouds (§5); the score does not depend on it.
