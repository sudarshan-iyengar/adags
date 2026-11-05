# CER on Multi4D: implementation spec (2026-09-30, rev 2, before any Multi4D CER code exists)

The port of [CER on Ex4DGS](cer-ex4dgs-spec-2026-09-30.md) to Multi4D at the pinned commit `c483e82`, for plan §4b.
The Ex4DGS spec is the reference for the mathematics, the cohort rules and the gates; this page states what differs on
Multi4D and where it attaches. Code goes on branch `cer` in the worktree `$W/repo/multi4d-cer`; the pinned checkout
`$W/repo/multi4d` is never modified.

**Revision history.** Rev 1 (archived as
[`assets/cer-multi4d-2026-09-30/spec_rev1.md`](../assets/cer-multi4d-2026-09-30/spec_rev1.md)) was reviewed by GPT-6.1 Sol
(high; MATERIAL DEFECT) and Fable 5.1 (ready after eight major fixes); both reviews are in the same folder. Rev 2 absorbs
them; §12 lists the changes.

**Entry.** §4b gate 2 (census) passed in every layout on 2026-09-30 (plan log): persistent foreground + static
background carry 0.983–0.991 of the pre-gap target mass, **the static background alone 0.983–0.990**, the persistent
foreground ≤ 0.006.

**Sources.** Every `file:line` was read at `c483e82`. Lifecycle study:
[`assets/cer-multi4d-2026-09-30/lifecycle_map.md`](../assets/cer-multi4d-2026-09-30/lifecycle_map.md), whose headline
(no row, reset or statistic change after fine 16,000) Fable re-traced call by call. `R` = `gaussian_renderer/__init__.py`,
`GM` = `scene/gaussian_model.py`, `T` = `train.py`.

**Glossary.** As in the Ex4DGS spec. Banks: persistent foreground (`GaussianModel_dynamic`, HexPlane-deformed), static
background (`GaussianModel`, no time), transient (`GaussianModelTransient`, 4D Gaussians with a CUDA temporal marginal).
Fine iteration = the counter of the fine `scene_reconstruction` call (20,000).

---

## 1. Mathematics: unchanged, on the foreground and background banks

Labels `c ∈ {0, 1, 2}` per row of the foreground and background banks, clock `[A, B]`, strength `q`; factor
`1 − q·n(t)` for memory rows, `q·n(t)` for residual rows, 1 otherwise, `n(t) = 1` on the gap frames; the activated
opacity is multiplied by it (float32 cuda `[N, 1]`) before dropout. The exactness consequences hold for a fixed model
(Ex4DGS spec §1): the static set is never culled on opacity in preprocessing (`diff_gaussian_rasterization_hybrid/
cuda_rasterizer/forward.cu:432-516`), zero-opacity rows are skipped per pixel before `T` is touched (`forward.cu:651-654`)
and in the backward kernel actually launched (`PerGaussianRenderCUDA`, `backward.cu:1373-1375`, launched at `:1542`),
keys are emitted in row order and sorted stably (`rasterizer_impl.cu:67-104`, `:340-345`), and residual rows are appended
at the end of the background block. A switched-off row receives zero **rendering-loss** gradient; it is not frozen
(Adam moments; the scale/aspect regularisers `T:814-828` and the soft displacement-variance term `T:713-730`, which reads
foreground `get_opacity` at `T:723`). **There is no q = 0 training identity**: residual rows change the length of the
dropout vector (`R:492-502`) and so every later random draw; q = 0 identity is a statement about a fixed model's render.

**The transient bank is not gated** (plan §4b as written). It carries 0.9–1.7% of the pre-gap target mass. On Multi4D,
exact-zero presence inside the gap is therefore a claim about the persistent banks; the transient bank's mass inside the
gap is reported as a residual ghost (§6). The live transient cull is the hard-coded `0.05` at `forward.cu:282`
(`opa_threshold` at `R:410` is dead under `rot_4d = True`, code map §2).

**Time and frame.** A Multi4D camera has `time = f/300`, `f` the frame's position in the sorted `camXX/images` listing
(`scene/neural_3D_dataset_NDC.py:93-105`), and `image_path = .../camXX/images/<frame>.png` (`scene/dataset.py:11-41`).
The gate reads the frame from `image_path` with a strict pattern (`cam\d\d/images/\d{4}\.png`) and **raises** if the
path is missing or does not match, never defaulting to "outside the gap"; the parsed frame is cached on the camera
object. Tests assert that it equals `round(time·300)`.

## 2. Where CER attaches, and when

### 2.1 Five opacity sites in four render functions, one gate function

| render | site (after activation, before dropout) | bank | used in |
|---|---|---|---|
| `render_background` `R:18` | after `R:86`, outside the bare `try/except` at `R:66-82` | background | coarse and fine phase 2 (`T:542`); `final_test_dump` (`utils/eval_utils.py:242`) |
| `render_foreground` `R:121` | after `R:198` | foreground | phase 2 (`T:514`); `final_test_dump` (`eval_utils.py:235`) |
| `render_mask` `R:248` | after `R:314` | foreground | phase 2 (`T:505`, the recorded silent bypass: it sets the masks and region weights of the other two); `final_test_dump` (`eval_utils.py:246`) |
| `render_rawall3pc` `R:351` | foreground after `R:468`, background after `R:475` (both before the concatenation at `R:486` and dropout at `R:492`) | both | phase 3's only render (`T:450`); `run_evaluation` (`eval_utils.py:68`, `:177`); `final_test_dump` (`:251`); the final-render wrapper |

Each site is one line, `opacity = cer_gate.apply(pc, viewpoint_camera, opacity)`. `utils/cer_gate.py` (new; it does not
import the renderer, so there is no import cycle) holds a module flag `ARMED`, set by `begin`:

- not armed → returns `opacity` unchanged (base runs and every render before `begin`);
- armed and the model carries CER state → returns `opacity · factor`, after asserting the label length equals the bank's
  row count;
- armed and the model carries no CER state → **raises** (a fresh or reloaded model reaching a render inside a CER
  process).

The CER state (`cer_labels`, `cer_clock`, `cer_q`) lives on the model objects, because the final render is made after
`train()` returns by a wrapper that holds only the models. No model class is edited.

From 16,000 on only `render_rawall3pc` runs in training and in `run_evaluation`; `final_test_dump` after training calls
all four functions for its debug panel. All five sites are gated identically, as the principal required; test T3
checks each function (§7).

Outside the pipeline and **not gated**: the semantic scripts (`train_semantic.py:399/405`,
`cluster_semantic_features.py:33/39`, `render_cluster_by_text.py:47/52`, `render_semantic_features.py`) build opacity
themselves. They load models from disk, and the load path refuses a CER model (§5), so they cannot render one.

### 2.2 When CER starts: at the end of fine iteration `densify_until_iter` (16,000 of 20,000)

Traced call by call with the config (Fable): from 16,000 on, `maintain_foreground` (`T:944`, also gated to phase 2),
`maintain_background` (`T:955`) and `maintain_transient` (`T:965`) do not run (`iteration < 16000`), the lift is off
(`iteration < phase3_start_iter` = 10,000, `T:914-916`), and no other row-count, reset or per-row-statistic call exists
in the loop; the displacement-variance accumulators are updated only in the phase-2 branch (`T:524-536`). Last possible
changes: background and transient 15,900, foreground 9,900. There is no optimizer step at 20,000 (`T:950`, `:960`,
`:973`): CER trains for 3,999 steps (16,001–19,999) and the evaluated model is the one after step 19,999. Base and CER
runs are identical up to 16,000.

`begin` is called at the end of the loop body of fine iteration 16,000, after `run_evaluation` (`T:989-992`, which at
16,000 still scores the ungated model), outside the `torch.no_grad()` block, guarded by `stage == "fine"` and a set
`cer_scene`. Asserted preconditions: `iteration == densify_until_iter`; `iteration ≥ phase3_start_iter`. `begin` also
installs a **runtime sentinel**: on the three model instances it replaces every row-count-changing or resetting method
(`prune_points`, `densify*`, `densification_postfix`, `add_points_from_dict`, `weight_prune`, `prune`,
`reset_opacity*`, `cat_tensors_to_optimizer`, `replace_tensor_to_optimizer`, `_prune_optimizer`) with one that raises, so
any lifecycle call after `begin` stops the run.

**Dose, pre-registered.** Ex4DGS's CER stage is 10,000 single-view steps (≈ 1,000 gap views). Multi4D's is 3,999 steps at
batch 2 (≈ 8,000 views, ≈ 800 gap views, ≈ 760 steps with at least one gap view), each residual row additionally dropped
in 10% of its views. That is ≈ 0.4× the steps and ≈ 0.7–0.8× the gap exposures. Starting earlier would require threading
labels through the background lifecycle (`GM:1088-1228`) and the opacity resets (`GM:925-936`); running past 20,000
would break the equal-budget rule. A smaller Multi4D effect is to be read against this asymmetry, not as a substrate
difference. Position learning rate at 16,000: ≈ 4.0e-6 × scale (`position_lr_max_steps` 20,000), 2.5× Ex4DGS's floor.

**Resume** restores the foreground only (`T:285-292`); refused when CER is on. **NaN restart**: a NaN loss `os.execv`s
the whole program (`T:843-845`); `begin` writes `cer_begin.json` exclusively (`O_EXCL`) and refuses if it exists, so a
restarted CER run fails loudly instead of re-running silently; the scorer checks that `debug_metrics.json` holds exactly
the 40 fine evaluations.

### 2.3 Configuration and plumbing

One argument, `cer_scene`, registered in `ModelParams` (`arguments/__init__.py:49`) so that `merge_hparams` and the parser
accept it; default `""` = CER off. `scene_reconstruction` (`T:233`) does not receive `dataset`, so the scene
constructor (which receives the model arguments, `scene/__init__.py:28`) stores `scene.cer_scene`, and `begin`, `gates`
and the resume refusal read it from there.

## 3. The contribution-mass instrument

The hybrid rasterizer takes the persistent set's colours only as SH coefficients. `mass` (under `torch.enable_grad()`,
restoring in a `finally`) replaces, on all three banks and for one render at a time, `_features_dc` by a leaf tensor
whose DC decodes to 1.0 in every channel (`0.5/SH_C0`, unclamped) and `_features_rest` by zeros (so every spatial and
temporal SH term vanishes), renders `render_rawall3pc` (stage `fine`, black background, dropout off, current CER factor),
takes `torch.autograd.grad` of `(image[0]·mask).sum() + image[1].sum()` with respect to the three DC leaves, divides by
`SH_C0`, and restores the originals. Verified (Fable): the colour gradient is `Σ_p α·T·dL/dpix` for the static and
transient sets (`backward.cu:1376-1395`), `dL/dsh[0] = SH_C0·dL/dRGB` for both (`:50-51`, `:175-176`), the foreground
deformation adds only to SH coefficient 16, which the render slices off (`R:471`, `scene/deformation.py:83-86`), and
rebinding the attributes leaves the optimizer's parameters and the deformation network untouched.

**Upstream hazard, guarded per render.** `max_contrib` is allocated (`rasterizer_impl.cu:189`) and never written, and the
launched backward returns early when `bucket_idx·32 ≥ max_contrib[tile]` (`backward.cu:1229`): a small uninitialised
value would silently drop a whole tile's gradient. `mass` therefore asserts, **on every render**, that
`Σ_i total_i` equals `Σ_p image[1]` within 1e-4 relative. (This defect affects Multi4D training itself too; it is
recorded as a substrate fact, not fixed.)

## 4. Cohorts

### 4.1 Memory (label 1)

On the ungated model: pre-gap frames × training cameras × target masks; a foreground or background row is memory when
`total > 0` and `inside ≥ 0.5·total`. Transient rows are never labelled.

### 4.2 Residual (label 2), background bank only

On the memory-gated model: gap frames × training cameras × gap regions `G_c = M_c(A − 1)`; donors are rows with
`total > 0` and `inside ≥ 0.5·total`, **restricted to the background bank**, for exactness: a background row has no
time, so its copy renders exactly as it does (six groups `xyz, f_dc, f_rest, opacity, scaling, rotation` appended through
the background `cat_tensors_to_optimizer`, `GM:1122-1146`, which raises on a missing group, with the returned parameters
rebound to the model's six attributes as `densification_postfix` does at `GM:1157-1163`; `max_radii2D [D]`,
`xyz_gradient_accum [D, 1]`, `denom [D, 1]` extended by zeros for the new rows only). A foreground donor would need a
deformed-state conversion that is not exact, and a transient donor a conditioning of its 4D covariance. **The census does
not bear on this choice**: it measured the pre-gap target, not the revealed surface in the gap, which may sit on
transient rows. The birth report gives the donor mass by bank on the memory-gated model, and T4-forecast reports it for
all nine scenes before any CER training; a scene with little background donor mass is expected to fail the residual
gate (§6) and is flagged as a DESIGN_WITHOUT_POWER forecast.

## 5. Persistence

The score is rendered from the in-memory models after `train()` returns, as for the base runs. The wave-1 launcher passes
`--checkpoint_iterations 20000`, so Multi4D writes a foreground and a background checkpoint at 20,000 (the transient bank
has no `capture` and is never checkpointed); its resume restores the foreground only, and its point-cloud reload loses the transient bank's temporal SH degree (`GM:1825` restores `active_sh_degree` only; `active_sh_degree_t`
stays 0 from `GM:1266`, used at `R:402`) — **so upstream Multi4D has no exact round trip even for a base model.**
CER therefore does not add one:

- `Scene2gs_mixed.save` writes `cer_state.pt` (labels per bank, clock, q, `active_sh_degree_t`) beside the point clouds
  whenever CER is on, before the two bare `try/except` blocks, including the `best/` saves;
- the load path (`scene/__init__.py:61-89`) **refuses** a directory that holds `cer_state.pt` ("a CER Multi4D model is
  evaluated in memory; reload is unsupported"). This also keeps the semantic scripts away from CER models.

This replaces the "checkpoint round trip" test the plan listed for Multi4D (§4b) — a decision for the principal (§11).

## 6. Gates

As on Ex4DGS (spec §6), computed at the end of training on the in-memory model, training cameras only, written to
`<model_path>/cer_gates.json`: memory share of target mass ≥ 0.50 before the gap and after the return; residual share of
the gap region ≥ 0.20 in the gap; memory mass in the gap and residual mass outside it exactly 0 (guards); the transient
bank's share of the gap region and of the target mass in the gap reported. `begin` and `gates` assert that every
foreground and background opacity is finite (`min(0.99f, NaN)` renders a NaN at alpha 0.99, `forward.cu:651`; Multi4D
has no NaN prune). Every run is scored; plan §4 rule 4 acts in the reducer.

## 7. Tests, written before the code

One GPU job, one node, on the wave-1 Multi4D run 1 of the bottle same-pose scene, loaded from its `iteration_20000`
point clouds and deformation files with the census's loader (which sets `active_sh_degree_t` explicitly and reproduced
the saved final renders byte for byte), in the CER tree, as a stand-in for the model at 16,000.

| id | test | pass |
|---|---|---|
| T0 | the unmodified checkout renders cam00 frames 0–299 twice through `render_rawall3pc` with the census loader | byte-identical |
| T1 | the `cer` tree, CER off | byte-identical to T0 |
| T2 | `begin` (memory + background residual rows), `q = 0` | byte-identical to T0 on all frames; ≥ 1 residual row; at `q = 1` a gap frame differs |
| T2b | birth state | every existing row's parameters, per-row buffers and Adam moments unchanged in all three banks (one optimizer step with tiny gradients first, so every group has moments); new background rows have zero moments; the six background attributes are the optimizer's parameters |
| T3 | gate identity per render function | for `render_background`, `render_foreground`, `render_mask` and `render_rawall3pc`, at a gap frame and a non-gap frame: the CER-on render equals, byte for byte, the same call with CER off after setting the switched-off rows' opacity logits to −inf (σ(−inf) = 0 exactly; opacity passes through the deformation unchanged, `scene/deformation.py:79-80`), including `render_mask`'s channels, and that reference differs from the plain ungated render (non-vacuity; if the real model has no visible foreground memory row, visible foreground rows are labelled memory for this test). The rasterizers' `forward` is wrapped to capture the opacity each receives, which must equal `factor · ungated` (so a fifth path added later is caught). Zero rendering gradient to the switched-off cohort, with a positive control at `q = 0`. Frame parse: every camera's parsed frame equals `round(time·300)`; a malformed path raises |
| T4 | instrument and non-vacuity | per-render conservation (the assertion inside `mass`) on the views used; split within 1e-5; memory share ≥ 0.50 ungated; exact zeros at `q = 1` with memory mass before the gap > 0 |
| T4-forecast | on the nine wave-1 Multi4D run-1 models | memory share, residual count and donor mass by bank, reported |
| T5 | persistence | `save` writes `cer_state.pt` equal to the in-memory state; loading that directory raises; an armed gate raises on a model without CER state |
| T6 | frozen lifecycle | (a) static: every row-changing call in `train.py` lies inside a `maintain_*` function; (c) after `begin`, each sentinel method raises. The proposed unit call of the three `maintain_*` functions at iterations ≥ 16,000 (b) is not built: their callers are gated `iteration < densify_until_iter`, (a) checks that no other call site exists, and the sentinels make any call after `begin` fatal, which T7 exercises over a real training |
| T7 | integration (not evidence) | the real entry point through the final-render wrapper, CER on, with a shortened schedule that keeps the ordering (`phase3_start_iter` < `densify_until_iter` < `iterations`): exit 0 (so no sentinel fired), `cer_begin.json` and `cer_gates.json` written, row counts at the end equal those after birth, 300 final renders saved, exactly one evaluation series in `debug_metrics.json`. A CER model cannot be reloaded after the process ends (§5), so "the saved render equals a direct gated render" is not re-checked here; T3 checks the same call on a loaded model |

T0 and T1 pass on the unmodified tree; T2–T7 import the CER code and fail on it.

## 8. CUDA: no change

The gate is a python multiply before the autograd `Function` on the standalone rasterizers' `opacities` and on the
persistent set's `opacities_static` (the 16th argument of `_RasterizeGaussians`, whose gradient is returned at
`gaussian_renderer/diff_gaussian_rasterization_hybrid.py:299`); zero-opacity rows are skipped per pixel in forward and in
the launched backward; the transient CUDA gate is untouched; the factor is float32 cuda (`data_ptr<float>()`,
`rasterize_points.cu:154`). Upstream CUDA defects recorded, not fixed: `max_contrib` uninitialised (§3); `clamped` sized
for the transient set but written by the static preprocess (`rasterizer_impl.cu:160`, `forward.cu:502-503`); transient
degree-1 SH gradient uses `l0m0` (`backward.cu:191`).

## 9. Outside the substrate

A launcher `train_multi4d_cer.sbatch` (the wave-1 launcher from the `cer` worktree, `--cer_scene`, provenance with the
`cer` commit and worktree cleanliness at training and at the wrapper's render, lrdn0066 excluded); the wrapper
`m4d_final_render.py` reused unchanged (it renders through the gated `render_rawall3pc`). Scoring: a method `multi4d_cer`
in `ars_score.py` with the same provenance rule as `ex4dgs_cer`, the gate verdict carried, the 40-evaluation check; the
runs added to `ars_score_worker.py`. Runs (§4b): CER-Multi4D ×3 on the three same-pose scenes against the three wave-1
Multi4D runs per scene. Nothing is submitted until the principal says "submit".

## 10. Deliberately not built

The doubled layer at birth (donor and copy both render in the gap) is a known design point, not a bug; the wave-2
ablations and the frozen 0.20 residual gate recorded for Ex4DGS (its spec §10, plan log 2026-10-01) apply here too.

Transient gating, foreground or transient donors, a reload path for CER models, label threading through the lifecycle
(nothing changes rows after 16,000), a q = 0 training identity.

## 11. Decisions this spec takes

1. CER starts at fine 16,000 (the frozen-lifecycle point) inside the 20,000-iteration budget; the dose asymmetry with
   Ex4DGS is pre-registered (§2.2).
2. The transient bank is not gated; on Multi4D exact-zero presence is a persistent-bank claim (§1).
3. Residual donors from the background bank only (§4.2), with a per-scene forecast before training.
4. **For the principal:** the Multi4D "checkpoint round trip" test is replaced by "the saved CER state equals the
   in-memory state, and a saved CER model refuses to reload" (§5), because upstream Multi4D has no exact reload even for
   a base model.

## 12. Changes from rev 1

| change | source |
|---|---|
| fail-closed gate (`ARMED`), strict frame parse, one `apply` call per site, per-bank gating before the hybrid concatenation | Fable M3 |
| persistence: sidecar incl. `best/`, reload refused; round-trip test replaced (decision for the principal) | Sol (persistence, marker), Fable M1, M4 |
| `cer_scene` plumbed through `scene.cer_scene`; lift precondition via `phase3_start_iter` | Fable M4, m2 |
| per-render conservation guard against the uninitialised `max_contrib` | Fable M2 |
| donor justification corrected; donor mass by bank forecast | Fable M5 |
| NaN restart: exclusive `cer_begin.json`, 40-evaluation check | Fable M6 |
| finite-opacity guard | Fable M7 |
| dose asymmetry pre-registered | Fable M8 |
| no q = 0 training identity stated; "every per-row statistic" and "only rawall3pc in evaluation" corrected | Fable, Sol |
| T3 by −inf logits + captured rasterizer inputs + foreground witness; T6 static/unit/sentinel; T2b; vacuity guards | Sol, Fable |
| `mass` under `enable_grad`, restore in `finally`; background append rebinding stated | Sol |
| upstream CUDA defects recorded | Fable |

## 13. Rev 3 (2026-10-01): after the code reviews and the principal's decisions (appended; the sections above are rev 2)

- **§8 is superseded on one point: there is now one CUDA change, and it never touches training.** The upstream hybrid
  forward never writes `max_contrib`, which the launched backward's early exit reads (§3), and in a fresh process the
  instrument measured exactly 0 mass against 1.37e6. Principal's option (a): a block in `forward.cu` compiled **only**
  under `-DCER_WRITE_MAX_CONTRIB` writes the per-tile maximum of `last_contributor` (commit 33215f1, its own commit);
  `scene/cer.py` builds a second extension from the same sources with that macro, under the name
  `diff_gaussian_rasterization_hybrid_cer`, and `instrument_kernel()` swaps it into the hybrid wrapper for the
  duration of `mass()` only (c1e6045). The training and evaluation renders use the unmodified kernel in both arms. The
  per-render conservation assertion stays. Both code reviewers verified the CUDA change (right quantity for the early
  exit, index-consistent, sync-safe; without the macro the training build is the upstream kernel). The upstream
  defect's effect on the baselines is not measured now (principal: later, diagnostic only).
- **The reload refusal lives in the foreground and background `load_ply`** (7d8c5e2), not in the scene's load path: the
  semantic scripts load directory checkpoints by calling `load_ply` directly. This edits two model classes by two lines
  each, contrary to "no model class is edited" in §2.1.
- **§6:** the transient bank's share of the gap region is reported; "of the target mass in the gap" is withdrawn (the
  target is absent from the world in the gap, so its mask is empty there).
- **§7:** the forecast is selected as `FORECAST`; unknown test names are rejected; T3's gradient checks run on the
  instrument build; T4b (new) runs `mass()` in a fresh process and reports, descriptively, whether the training build
  breaks conservation there; T5 also requires the refusal at the direct `load_ply`; T6's regex and sentinel check use
  the full `LIFECYCLE` list (three names added); T7 requires one fine-evaluation series and 300 saved renders.
- **Launcher:** builds both extensions before training starts (a build problem fails at t = 0) and records the
  `forward.cu` hash.
- Residual gate primary threshold 0.20 frozen (plan log 2026-10-01); 0.10 / 0.20 / 0.30 reported descriptively.

## 14. Gate amendment (2026-10-01, rev 2; the principal's decisions (a), (b), (c); appended, before any real CER run)

Applies the plan-log entry "2026-10-01 (later): principal's decisions on the contribution gates and on Multi4D". It
supersedes §6's "residual share of the gap region ≥ 0.20 in the gap" and the last bullet of §13 **as preconditions**.
Everything else in rev 2 and §13 stands. The companion is the Ex4DGS spec §13. Its 13.2 (descriptive residual share),
13.3 (no code change, the old `pass` field) and 13.4 (how a scene that fails the preconditions enters the rules)
apply here with the Multi4D fields and design below. Rev 1 is preserved verbatim in
[`assets/cer-gates-amendment-2026-10-01/review_prompt.md`](../assets/cer-gates-amendment-2026-10-01/review_prompt.md),
with both reviews beside it.

**14.1 What the Multi4D arm is.** CER-Multi4D is a **memory-gate arm**: the exact memory gate on the persistent
foreground and background banks. Whatever Multi4D itself learns paints the gap.

- **Residual copies.** The background residual copies are still born by the unchanged code (§4.2; forecast 6, 0 and
  1 donors on the three same-pose scenes).
- **What paints the gap is reported by bank:** residual copies, transient rows, ordinary persistent rows.
- **What may be said.** The measured contributions are reported as they are, residual copies included. What is
  restricted is a causal claim that the residual copies or the transient bank cause the gap improvement.
- **Name in reports:** "memory gate on Multi4D".

**14.2 Preconditions on Multi4D, complete list** (proposed reading of decision (b), for the principal's
confirmation), read from `cer_gates.json`:

| precondition | field | rule |
|---|---|---|
| memory active before the gap | `memory_share_pregap` | ≥ 0.50 |
| memory active after the return | `memory_share_return` | ≥ 0.50 |
| memory suppressed inside the gap | `memory_mass_gap` | exactly 0.0 |
| residual zero outside the gap | `residual_mass_outside_gap` | exactly 0.0 |

**The reading of (b), stated.** Decision (b) speaks of a Multi4D "gap-content gate that counts residual or transient
rows". Decision (a) lists the preconditions that stay with no Multi4D exception. This rev reads (b)'s gate as
**descriptive**: the share of the gap region painted by residual or transient rows, reported with the 0.10 / 0.20 /
0.30 table (14.3), never a precondition. Both reviewers recommend this reading.

- **What such a precondition would guard.** The four rows above already assert that the memory gate acted, exactly,
  on rows that carry the target. A fifth row would test only which rows paint the gap at the end of training: ordinary
  persistent rows painting it would fail it. That is a property of the result's makeup, not of whether the mechanism
  was exercised.
- **What it would cost.** The values seen disagree. The forecast on full-schedule base models, at birth, is 0.88–0.96
  transient. The shortened smoke run (`cer_t7_m4d_r4`, lift at 4,000, CER start 4,500, end 5,000; not evidence) shows
  0.025 at the end (residual 0.024, transient 0.0009), with foreground 0.45 and background 0.55 at birth. A
  precondition at 0.20 could therefore make all three same-pose scenes design-without-power although the memory gate
  acted exactly.
- **If the principal meant a precondition.** The row "residual + transient share of the gap region ≥ 0.20" goes back
  into this table with these values stated, and (a)'s list is read as Ex4DGS-only.

**14.3 Descriptive.** Per run, and per scene as the mean of the three runs with minimum and maximum (as Ex4DGS 13.2):

- **The residual share** (`residual_share_gap`): the residual copies' share of the gap content, with the 0.10 / 0.20
  / 0.30 table.
- **The gap-content share** (`residual_share_gap + transient_share_gap_region`), with the same table. Both terms have
  the same denominator, the gap-region mass of every row in all three banks (`scene/cer.py:204-206`, `:211`, `:213`).
- **The transient share and the ordinary-persistent share**, as plain values. The ordinary-persistent share is one
  minus the gap-content share, which holds only because `memory_mass_gap` is exactly 0 (a precondition).
- **A zero gap-region mass** has no share. Multi4D does not clamp: `gates()` raises on division by zero, so such a run
  stops (a CER failure, Ex4DGS 13.5).

**14.4 Confirmatory design and how the reducer reads it (plan §4b as amended 2026-09-30; the 13.4 policy applies once
accepted).**

- **Manifest.** Nine CER-Multi4D runs, three on each same-pose scene (bottle, carton, mug; run ids
  `ars_w1_<scene>_m4dcer_rep<r>`), against the nine wave-1 Multi4D base runs 1–3 on the same scenes. Moved-return and
  pass-through CER-Multi4D runs (§4b, "only if the queue allows") are descriptive and never block the reducer.
- **Rule 1.** Model pooled over the three layouts, 9 against 9, with arm and layout fixed effects. Endpoints: the gap
  window and the return window; Holm over these 2, always 2. The same-pose condition passes under plan §4 rule 1's
  three clauses:
  - the gap improvement is Holm-significant;
  - the one-sided 95% lower bound on the return window exceeds −0.25 dB;
  - the mean gap improvement is positive in each layout.

  The Multi4D verdict is that condition's PASS, FAIL or DESIGN_WITHOUT_POWER.
- **Rule 3.** Do-no-harm applies with the same −0.2 dB one-sided bound on the whole frame and the ordinary region,
  pooled over the three same-pose scenes.
- **Rule 2.** Not applied. The benchmark comparison is CER on Ex4DGS against STG and Multi4D; the Multi4D arm is a
  substrate replication.
- **Readings.** Multi4D's final iteration 20,000, scored as the base runs are.

**14.5 Exploratory work (decision (c)).** Transient donors, gating the transient bank, and any other Multi4D variant
start only after this confirmatory design is frozen, meaning after the principal accepts this amendment. Their run
ids, the ledger and every report label them exploratory. They are never pooled with the confirmatory runs.

**14.6 No substrate code change; the scoring path.**

- **The fields exist.** The Multi4D `cer_gates.json` at `00357aa` stores every field in 14.2 and 14.3.
- **The `pass` field** (`:216-218`) applies the superseded rule. It is handled as on Ex4DGS (13.3): integrity check,
  reported as `pass_superseded_rule`, never branched on.
- **The scorer** (outside the substrate) has a method `multi4d_cer` (installed 2026-10-01), which requires:
  - commit `00357aa` on branch `cer`;
  - no untracked or modified files before and after the job;
  - the full schedule (no `EXTRA`);
  - `cer_scene` equal to the scored scene;
  - exactly the 40 fine evaluations 500…20,000 in `debug_metrics.json`;
  - the existing Multi4D final-render checks.

  It copies `cer_begin.json` and `cer_gates.json` into the score. The scoring worker lists the nine confirmatory runs.
  The pin moves with any later commit.

**14.7 Changes from rev 1**

| change | source |
|---|---|
| 14.1: residual contributions may be reported; only causal claims are restricted | Sol 4 |
| 14.2: the gap-content row read as descriptive, the reading stated with what a precondition would guard and cost; "0.20 is the threshold of the gate it replaces" removed | Fable 1; Sol 1 |
| 14.3: thresholds only on the residual and gap-content shares; the ordinary share's condition; zero denominator | Fable 11; Sol (missing) |
| 14.4: own manifest of 9 + 9, optional runs never block; Holm over 2; rule 3 applied; rule 2 not applied | Sol 3; Fable 10 |
| 14.6: the scoring path stated | Fable 6 |

**Accepted by the principal (plan log, "2026-10-01 evening: principal's decisions on the reducer, the statistics, the base
runs and Multi4D").** The proposals marked above as not frozen are frozen as written.
