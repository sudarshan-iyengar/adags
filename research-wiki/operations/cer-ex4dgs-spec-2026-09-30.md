# CER on Ex4DGS: implementation spec (2026-09-30, rev 2, before any CER run)

Complementary Episode Residuals (CER), as proposed in
[handoff rev 3 §3(b)](neurips-workshop-paper-handoff-2026-09-23.md) and
[sol_round1.md §1](../assets/neurips-plan-sol-2026-09-21/sol_round1.md), specified against
Ex4DGS at the pinned commit `7fac649`. Code goes on branch `cer` in the worktree
`$W/repo/ex4dgs-cer`; the pinned checkout `$W/repo/ex4dgs` is never modified.

**Revision history.** Rev 1 (archived as
[`assets/cer-ex4dgs-2026-09-30/spec_rev1.md`](../assets/cer-ex4dgs-2026-09-30/spec_rev1.md)) was reviewed by GPT-6.1 Sol
(high; MATERIAL DEFECT, 7 confirmed defects) and by Fable 5.1 (ready after fixes; one decision for the principal). Both
reviews are in the same folder. Rev 2 absorbs every confirmed finding; the changes are listed in §12.

**Glossary.** Target = the object that leaves and returns. Gap = the frames `[A, B]` in which
the target is removed from the world (the designed clock, `event_spec.json` `clock`; A = 60,
B = 89 on all nine wave-1 scenes). Memory cohort = the existing rows that paint the target.
Residual cohort = new rows that exist only inside the gap. Static bank / dynamic bank =
Ex4DGS's two disjoint row sets (static rows have no temporal opacity; dynamic rows have
keyframed motion and a plateau temporal opacity). Training cameras = every camera except cam00.
Contribution weight of row *i* at pixel *p* = `α_i(p)·T_i(p)`, the weight with which its
colour enters the composite.

---

## 1. The mathematics

Ex4DGS hands the rasterizer one opacity per row, already evaluated at time *t*:

```
o(t) = [ σ(ω_s) ; σ(ω_d) · b(t) ]        float32 cuda [N_s + N_d, 1]
```

static first, dynamic second (`c_gaussian_model.py:368-375`), with `b(t)` the plateau
temporal factor (`utils/interpolations.py:55-61`). CER adds one per-row label
`c_i ∈ {0 ordinary, 1 memory, 2 residual}`, the clock `[A, B]` and a strength `q ∈ [0, 1]`,
and multiplies:

```
n(t)      = 1 if A ≤ t ≤ B else 0                      (absence indicator)
f_i(t)    = 1 − q·n(t)    if c_i = 1                    (memory: off in the gap)
          = q·n(t)        if c_i = 2                    (residual: on only in the gap)
          = 1             if c_i = 0
o'_i(t)   = o_i(t) · f_i(t)          f is float32 cuda [N_s + N_d, 1] (a [N] vector would broadcast to [N, N])
```

Residual rows are created in the **static bank**, so `o_j(t) = σ(ω_j)` and
`o'_j(t) = q·n(t)·σ(ω_j)`, exactly the handoff's `α_j(t) = q·n(t)·σ(o_j)`. Memory rows can be
in either bank; the factor multiplies the concatenated vector, so both banks are covered
(handoff rev 3 §2.4: the fixture targets are static, and Ex4DGS's static bank has no temporal
opacity at all).

**Why a hard indicator and not the smooth flat-top.** The clock is a declared input (plan §4:
"the designed clock (A\*, B\*)"), not learned. Ex4DGS trains and renders only at integer frame
timestamps (§3.2), so on every time the model ever sees, the hard indicator equals any flat-top
whose ramps lie strictly between frames. The smooth form exists to give gradient to learned
boundaries; estimated windows are deferred (plan §4), so there is no boundary parameter to
receive that gradient. The hard form makes every factor exactly 0.0 or 1.0 in float32 when
`q ∈ {0, 1}`, which is what the exactness tests (§7) rely on.

**Exactness consequences (these are what the tests check), for a fixed model.**

1. `q = 0`: `f_i = 1` for memory and ordinary rows and `f_j = 0` for residual rows, at every *t*.
   `x·1.0 = x` in IEEE float; a row with opacity exactly 0 is skipped per pixel
   (`forward.cu:379-381`, `alpha < 1/255 → continue`) without touching the transmittance, and
   `last_contributor` (the index of the last entry that contributed, `forward.cu:364-422`) is unchanged by skipped
   entries after it. The residual rows add entries to the tile
   lists, but `duplicateWithKeys` emits keys in row order and the radix sort is stable, and the
   residual rows sit between the static and dynamic blocks, so every existing row keeps its
   relative order, including among equal depth keys. The image therefore equals the image of the
   same model without residual rows, byte for byte.
2. `q = 1`, `t ∉ [A, B]`: identical to case 1, so the residual is **exactly zero outside the gap**
   and the render equals that of the model without CER.
3. `q = 1`, `t ∈ [A, B]`: every memory row has opacity exactly 0 (exact-zero presence) and
   receives **zero rendering-loss gradient** from gap frames (skipped per pixel in forward and in
   `backward.cu:589-590`; the multiply gives `∂o'/∂o = 0`). It is not frozen: `static_reg` and
   `motion_reg` (`train.py:155-161`) reach every row, and RAdam applies its moments, exactly as for
   any row out of view in the base.

These are statements about rendering a given model. CER training is not equal to base training
even at `q = 0`: the residual rows change the static regulariser's mean and receive optimizer
steps. CER runs use `q = 1`. `q` is not a config key; it exists for the `q = 0` identity test.

## 2. Where CER attaches, and when

### 2.1 The single opacity junction

`gaussian_renderer/__init__.py:64`
`opacity = pc.get_opacity_at_t(timestamp, mode=mode, training=training)` is the **only** caller
of `get_opacity_at_t` in the repository (grep over every `.py` outside `submodules/`), and the
opacity reaches CUDA only through the rasterizer call at `:100-109` (`opacities = opacity`).
Every render (training `train.py:138`, evaluation `render.py:68` and its timing loop `:91-96`,
`training_report` `:341`) goes through `render()`. CER adds, after `:64`:

```python
    if pc.cer_clock is not None:
        opacity = opacity * pc.get_cer_factor(timestamp)
```

So training and evaluation are gated by the same line (the 2026-09-10 gate-off evaluation defect
cannot recur through a second render path: there is none). No caller passes `mode ≠ 0` (grep).
`get_cer_factor` asserts that each bank's label vector has that bank's row count, so a
count-preserving move of rows between banks cannot pass silently.

### 2.2 When CER starts: at the end of iteration `densify_until_iter` (30,000 of 40,000)

The wave-1 config (`agent-control/ars/configs/ex4dgs_n3v_base_res1.json`) has
`iterations 40000`, `densify_until_iter 30000`, `time_pad_type 0`, `interp_type cube`. After
iteration 30,000 the upstream lifecycle can only **remove** rows:

| operation | site | runs after 30,000? |
|---|---|---|
| clone / split / opacity-error prune | `train.py:203-227` (`if iteration < opt.densify_until_iter`) | no |
| static→dynamic promotion, in-window | `train.py:228-232` (the `elif` of the same block) | no |
| static→dynamic promotion, window-end | `train.py:127-130` → `:277-282` | no, **but only because of the config**: `sample_len` reaches its cap 312 (`duration + time_shift`, `time_shift` = 12 with cube interpolation) at 12,800; `mark_last` is set at 12,500 and stays set, and its consumer needs a camera with timestamp ≥ `sample_len − interval` = 302, which does not exist (last frame 299). With linear interpolation it would fire. **Asserted in `begin`:** `max timestamp < sample_len − interval`. |
| `adjust_temp_opa` | `train.py:233-234` (`< densify_until_iter − 3000`) | no |
| `prune_invisible`, `prune_small` | `train.py:237-241`, once per epoch (≈ 5,700 iterations; at 34,201 and 39,901) | **yes** |
| `prune_nan_points` | `train.py:253`, every iteration | **yes** (removes rows with NaN positions) |

Starting CER there means no growth path runs, and both pruning paths go through
`prune_points` (`c_gaussian_model.py:715-763`), which CER extends so the labels follow.
**Deviation from Sol's round-1 text, deliberate:** Sol wrote "disable densification, pruning and
promotion during the residual stage". Densification and promotion are already off after 30,000;
pruning is **kept** because the base arm prunes in the same iterations, and disabling it would
make the arms differ by more than CER. The base and CER arms train identically for 30,000
iterations and differ only in the last 10,000; total iterations, schedule and evaluator are
unchanged.

The call goes at the end of the loop body, outside the `torch.no_grad()` block that ends at
`train.py:282`, because building the cohorts needs autograd (§4):

```python
        if dataset.cer_scene and iteration == opt.densify_until_iter:
            cer.begin(gaussians, scene, pipe, dataset)
```

`chkpnt30000.pth` (`train.py:195-197`) is written **before** that iteration's optimizer step and
NaN prune, so `begin` sees the model one RAdam step later than the checkpoint. Tests may use the
checkpoint as an input model; a real run's `cer_begin.json` is not exactly reproducible from it.

**Preconditions asserted in `begin`** (about the setup, never a score): every training camera
with timestamp in `[0, 299]` is in the sampling window (`scene.samplelist`), i.e. the gap frames
are trained; the window-end promotion cannot fire (above).

**Resume is refused.** `train.py:63-65` restores a checkpoint and continues from its iteration;
a CER run resumed from `chkpnt30000` would skip `begin` and train a base model under a CER run
id. `train.py` asserts that `--start_checkpoint` is not combined with `cer_scene` (the record
already treats Ex4DGS resume as unusable: the progressive window resets).

### 2.3 New configuration

One key, `cer_scene` (string, default `""` = CER off), registered in `ModelParams`
(`arguments/__init__.py:47-`), because `train.py:398-408` silently drops unregistered keys. It
points to the canonical ARS scene directory (`.../ars/wave1/<scene>`) holding `event_spec.json`
(clock A, B; the target's part ids = the parts of the object whose `role` is `target`) and
`gt/identity/<camera>.npz` (front-most identity per pixel, `f<frame:04d>`, int16 1014×1352,
`ARS_SPEC_FORMAT.md` §5, §7). Only training cameras' identities are read, never cam00 (plan §4,
declared input). Nothing else is configurable: `q = 1`, the start iteration is
`opt.densify_until_iter`, and the thresholds in §5–§6 are constants in the code.

## 3. Facts about the substrate that the design depends on

1. **One opacity junction, python-side** (§2.1). CUDA receives an evaluated scalar per row
   (`opacities[idx]·cov.w`, `forward.cu:267`) and has no notion of time (code map §4).
2. **Time is the integer frame index.** The neural3dvideo reader sets
   `timestamp = frame − start_timestamp` (`scene/dataset_readers.py:308`), `start_timestamp = 0`;
   `time_pad_type 0` adds no mirrored padding cameras (`scene/__init__.py:124-162` only act for
   types 1 and 2). cam00 frame *f* renders at timestamp *f* (`render.py:66-68`).
3. **Evaluation restores the checkpoint, not the ply.** `render.py:128-129` builds the model and
   `Scene(..., load_iteration=40000, opt=opt)`; `scene/__init__.py:166-170` calls
   `restore(chkpnt40000)` when the checkpoint exists (always: `train.py:55`). There is no optimizer
   step at 40,000 (`:244`), so the final model is the checkpoint. The ply path (`:171-175`,
   `load_ply`) is a fallback only; **CER refuses it** (§5.4).
4. **Row removal has one path**, `prune_points` (`:715-763`), masking the static and dynamic
   banks separately; every non-parameter per-row tensor is masked there by hand.
5. **Row addition** goes through `cat_tensors_to_optimizer` (`:765-787`), which silently skips
   any optimizer group not in the dict (`:770-771`). The clone path shows the pattern for adding
   static rows: extend `xyz_error_min(_timestamp)` by hand (`:981-984`), then
   `densification_postfix_onlystatic` (`:846-872`).
   **CER does not call `densification_postfix_onlystatic`**: it resets the accumulators and
   `min_radii2D` of *every* static row (`:866-872`), which would change `prune_small` behaviour
   for the ordinary rows after 30,000 relative to the base arm. CER calls
   `cat_tensors_to_optimizer` with the seven static groups and extends each non-parameter static
   tensor by the new rows only (§5.2), so the existing rows' state is untouched.
6. **The colour gradient is the exact contribution weight.** For precomputed colours the
   backward pass accumulates `dL/dc_{i,ch} += α_i·T_i · dL/dpixel_ch` over every pixel the row
   reaches (`backward.cu:590-636`, `dchannel_dcolor = alpha * T`), with the same `1/255` skip
   and early termination as the forward pass, `T` reconstructed by division and summed with
   float32 atomics (so exact up to float rounding). `render()` already accepts precomputed colours
   (`override_color`, `gaussian_renderer/__init__.py:96-97`).
7. **Checkpoint tuple.** `capture()` returns a 40-element tuple (`:217-260`); `restore()`
   unpacks exactly 40 (`:262-321`).
8. **Static `_xyz` is nearly frozen after 30,000.** The static `xyz` learning rate follows
   `get_expon_lr_func` (`utils/general_utils.py:74-88`) with `position_lr_max_steps` 30,000, so from
   30,000 on it is `position_lr_final × scale` = 1.6e-6 × scale, 100× below its start. A static row can still
   move through `_xyz_disp` (its own optimizer group), scaled by `t / duration` (≈ 0.25 in the gap). (The dynamic
   position rate is never scheduled: `update_learning_rate` returns at the first group, code map §1.) This is why
   the residual donors should already sit at the right depth (§5.2); how far residual rows actually drift is not
   measured.

## 4. The contribution-mass instrument (shared by membership, birth and the gates)

```
mass(pc, pipe, dataset, views) -> inside[N], total[N]      (float64, cuda)
for each (camera, mask) in views:
    colours = ones([N, 3], requires_grad)                         N = N_s + N_d
    image   = render(camera, pc, pipe, bg = zeros(3), override_color = colours,
                     near = dataset.near, far = dataset.far)["render"]                      [3, H, W]
    g       = autograd.grad( (image[0] * mask).sum() + image[1].sum(), colours )[0]           [N, 3]
    inside += g[:, 0];  total += g[:, 1]
```

The clipping planes are passed explicitly: `render()`'s defaults are `near=0.2, far=100.0`
(`gaussian_renderer/__init__.py:19`), while training passes the config's 4.0 / 300.0
(`train.py:138`); the instrument must see the same rows the training render sees.
`inside_i = Σ_{frames, cameras} Σ_{p ∈ mask} α_i T_i`, `total_i` the same over every pixel.
`torch.autograd.grad` is used, not `.backward()`, so **no parameter `.grad` is touched** (the
call happens between two optimizer steps); `render()` consumes no random numbers. The render uses
the current CER factor, so the same function measures the ungated model (before labels exist) and
the gated one. `mask` is `[H, W]` float {0, 1}; `H, W` must equal the render's (asserted; both
1014×1352 on the 1352 route). Camera number and frame come from the camera object:
`cam.timestamp` and the `camNN` directory of `cam.image_path`; the tests assert that
`realpath(image_path)` ends in `frames/camNN/<frame:04d>.png` of the canonical scene for every
camera.

Masks, training cameras only:

- **target mask** `M_c(t)` = identity `∈` target part ids, camera *c*, frame *t*;
- **gap region** `G_c` = `M_c(A − 1)`: where the target was, in camera *c*, on the last frame
  before it left. (The canonical `ghost` region, the target-alone projection, exists only for
  cam00; the front-most target mask at `A − 1` is its training-camera proxy.)

Windows: pre-gap `[0, A − 1]`; gap `[A, B]`; return `[B + 1, B + 30]`.

## 5. The cohort lifecycle

### 5.1 Memory cohort (label 1), set once at CER start

On the model at the end of iteration 30,000, ungated: `(inside, total) =
mass(pre-gap frames × training cameras, target masks)`.

```
memory = (total > 0) & (inside >= 0.5 * total)
```

A row joins the memory cohort when most of its pre-gap contribution lies on the target
(majority-region membership). The labels are stored per bank, `_cer_static [N_s]` and
`_cer_dynamic [N_d]` (uint8, cuda), and the clock is set, so from here on the model is gated.

### 5.2 Residual cohort (label 2), born once at CER start, from the memory-gated model

**Decision (Fable F1, taken here, for the principal's confirmation).** With the memory cohort
already switched off (`q = 1`): `(inside, total) = mass(gap frames × training cameras, gap
regions G_c)`.

```
donors = (total > 0) & (inside >= 0.5 * total)          |D| rows, either bank; D ∩ memory = ∅
t_m    = (A + B) // 2 = 74                              (a trained frame)
```

Donors are the rows that paint where the target was, during the gap, once the target's rows are
off: the revealed surfaces, at their own depth. Measuring donors on the ungated model (rev 1)
would select the target's ghost rows themselves (Ex4DGS has no per-row temporal appearance and a
single presence interval, so on this substrate the rows that paint the gap region before gating
are the target's), and their copies could not move to the revealed surface (§3.8). This is a
departure from the literal "largest gap-region contributors": it is majority-region membership,
not a ranking by mass, and it is measured after gating.

Each donor is copied into the **static bank** as its rendered state at `t_m`, taken through the
same getters the renderer uses, so a residual row renders at `t_m` exactly as its donor did
(opacity up to the logit round trip):

| new static tensor | value | shape |
|---|---|---|
| `_xyz` | `get_xyz_at_t(t_m)[D]` | `[D, 3]` |
| `_xyz_disp` | zeros | `[D, 3]` |
| `_features_dc`, `_features_rest` | `get_features()[D]` split `[:, :1]`, `[:, 1:]` | `[D, 1, 3]`, `[D, 15, 3]` |
| `_scaling` | `cat(_scaling, _scaling_motion)[D]` (pre-activation) | `[D, 3]` |
| `_rotation` | `get_rotation_at_t(t_m)[D]`, raw as passed to CUDA (which does not normalise, `forward.cu:137`) | `[D, 4]` |
| `_opacity` | `inverse_sigmoid(get_opacity_at_t(t_m)[D].clamp(1e-6, 1 − 1e-6))` (ungated opacity; donors are not memory rows) | `[D, 1]` |

Added through `cat_tensors_to_optimizer` with exactly the seven static groups (zero RAdam
moments for the new rows). Non-parameter static tensors are extended by `D` rows only:
`xyz_gradient_accum, denom, xyz_error_accum, xyz_ssim_error_accum, error_denom` zeros `[D, 1]`;
`xyz_error_min` 1000 `[D, 1]`; `xyz_error_min_timestamp` = A `[D, 1]` (non-negative, so
`prune_invisible` does not delete an unobserved newborn: it prunes rows whose timestamp is
negative, `:1074-1085`, and after 30,000 nothing updates that tensor); `max_radii2D` 0 `[D]`,
`min_radii2D` 1000 `[D]`. `_cer_static` gets `D` twos. After birth `N_s' = N_s + D`.

**What the residual rows then do.** They receive rendering-loss gradient only from gap frames
(≈ 1,000 of the last 10,000 iterations: 30 of 300 frames). Their donors keep painting the same
surface outside the gap; the residual copies add gap-only capacity where the gap shows something
the donors do not already explain. `prune_small` (`:1087-1093`) deletes a residual row whose
minimum on-screen radius in any gap view it contributes to falls below 5 px, at 34,201 and
39,901, like any other row; this can thin the cohort, and gate 4 is where it shows.

### 5.3 From 30,001 to 40,000

Every render is gated by §1. Upstream `prune_invisible`, `prune_small` and `prune_nan_points`
keep running for all rows; `prune_points` masks `_cer_static` and `_cer_dynamic` with the same
keep masks it applies to the other per-row tensors. No other code path changes row counts in
this interval (§2.2), and `get_cer_factor`'s per-bank assertion would stop the run if one did.

### 5.4 Persistence

- **Checkpoint.** When CER is active, `capture()` appends one element, the dict
  `{static, dynamic, clock, q}`; when inactive it returns the upstream 40-tuple unchanged.
  `restore()` accepts 41 (splits the dict off) or 40 (upstream checkpoints, CER off). `restore()`
  is only ever called on a freshly constructed model (`train.py:65`, `scene/__init__.py:169`).
- **Ply fallback refused.** `scene/__init__.py` asserts, on the fallback branch, that the model's
  `cfg_args` has no `cer_scene`: a ply-loaded CER model would render its residual rows as
  ordinary always-on static rows. The CER code writes no ply sidecar.

## 6. The contribution gates (plan §4 rule 4; rev 2 §6), computed at the end of training

Written by `train.py` after the last iteration to `<model_path>/cer_gates.json`, on the final
model with `q = 1`, training cameras only (the gates read no cam00 pixel, so they cannot leak the
scored outcome). Thresholds declared here, before any CER output exists:

| gate (rev 2 §6 wording) | measurement | pass | kind |
|---|---|---|---|
| memory active before the gap | memory share of target-mask mass over the pre-gap window | ≥ 0.50 | evidence |
| memory active after the return | memory share of target-mask mass over the return window | ≥ 0.50 | evidence |
| memory suppressed inside the gap | memory mass over the gap window, every pixel | exactly 0 | guard (implied by §1 once T3 passes) |
| residual active inside the gap | residual share of gap-region mass over the gap window | ≥ 0.20 | evidence |
| residual zero outside the gap | residual mass over the pre-gap and return windows, every pixel | exactly 0 | guard (implied by §1) |
| both banks covered | memory rows and pre-gap memory mass per bank | reported | the factor acts on the concatenated vector by construction; T3 checks a memory row in each bank |

`gates` also asserts that every evaluated opacity is finite on every frame (`check_finite`): a NaN opacity times a 0 factor is NaN,
and `min(0.99f, NaN)` renders it at alpha 0.99 (`forward.cu:379`), while `prune_nan_points`
checks positions only (`:1229-1241`).

The birth-time numbers (memory share on the ungated model, memory rows per bank, donor count,
donor share of the gap region on the gated model) go to `cer_begin.json` at 30,000. The 0.50
memory threshold is the one frozen for the Multi4D census (S0 rule 6); 0.20 is Sol's round-1
figure. **A failure of the after-return gate on a same-pose scene** can mean the base already
represented the return with a second lineage (a row whose plateau starts at B + 1 has no pre-gap
mass and is not memory); it is recorded as that, not as a CER defect. On moved-return scenes the
memory rows may not follow the target to its new pose, in which case that gate fails there.

**How the gates reach the proof.** Every planned CER run is scored in full; the scorer copies the
gate verdict into the score file. Plan §4 rule 4 is applied by the reducer: a scene whose CER runs
fail a gate is reported DESIGN_WITHOUT_POWER in the proof, and its numbers stay on the record. No
run is withheld from scoring and none is dropped.

## 7. Tests, written before the code

In one GPU job on one node (forward rendering is compared **on one GPU only**; the record shows up
to 36/255 differences across GPUs). Inputs: the wave-1 base checkpoint `chkpnt30000.pth`
(upstream 40-tuple) of a healthy base run of the bottle same-pose scene, copied to a scratch
directory, and the canonical scenes. Images are compared by the sha256 of the float32 render
tensor. The file is `test_cer.py` at the root of the `cer` worktree, committed before the code.

| id | test | pass criterion |
|---|---|---|
| T0 | control: the unmodified checkout renders cam00 frames 0–299 twice (subprocess, render.py's loading path) | byte-identical (establishes that the comparisons below are meaningful on this GPU) |
| T1 | the `cer` tree with CER off (upstream checkpoint): same renders | byte-identical to T0 |
| T2 | `begin` on that model, then `q = 0`: same renders | byte-identical to T0 on all 300 frames, **and at least one residual row exists** |
| T3 | `q = 1` | factor: residual exactly 0.0 and memory exactly 1.0 for `t ∉ [A,B]`; memory exactly 0.0 and residual exactly 1.0 for `t ∈ [A,B]`; ordinary exactly 1.0; a memory row in each bank (one is forced if a bank has none). Renders: `t ∉ [A,B]` byte-identical to T0, every `t ∈ [A,B]` differs. Gradients: after `render().sum().backward()` at a gap frame, every parameter gradient of every memory row is exactly 0; at the frames A − 1 and B + 1, every parameter gradient of every residual row is exactly 0 (`_opacity_duration_var.grad` passed through `nan_to_num` first, as `train.py:246-248` does) |
| T4 | instrument and non-vacuity | view mapping for every training camera; conservation `Σ_i total_i` vs `Σ_p` of the ones-colour render within 1e-4 relative; aggregate split `Σ|inside + inside_complement − total| / Σ total` below 1e-5; memory share of pre-gap target mass ≥ 0.50 on the ungated model (`cer_begin.json`); with `q = 1`, memory mass over the gap window exactly 0, residual mass over the pre-gap and return windows exactly 0, residual mass over the gap window > 0 |
| T4-forecast | memory share (ungated) and donor count for each of the nine wave-1 run-1 base models | reported per scene; a share below 0.50 is a DESIGN_WITHOUT_POWER forecast before any CER training |
| T5 | checkpoint round trip and fallback | `capture` → `torch.save` → load → fresh model `restore`: the first 40 tensors, labels, clock and q equal; cam00 renders at frames {0, 59, 60, 74, 89, 90, 299} byte-identical before and after. A model directory with a CER `cfg_args` and only a ply must **fail to load** |
| T6 | pruning keeps labels aligned | prune a fixed random 10% of static and of dynamic rows through `prune_points`: surviving labels equal the pre-prune labels under the keep masks, label lengths equal row counts, and the RAdam `exp_avg` of `xyz` and `motion_xyz` equal their pre-prune values under the keep masks |
| T2b | birth state | every existing row's parameters, per-row buffers and both RAdam moments (all 15 groups) unchanged by `begin`; the new static rows have zero moments; ≥ 1 residual row |
| T7 | integration (not evidence) | `train.py` with CER on the bottle same-pose scene at 17,200 iterations, `densify_until_iter` 13,000 (the window covers the sequence by 12,800; the epoch refill at 17,101 runs `prune_invisible`/`prune_small` through the labelled `prune_points`), then `render.py`: `cer_begin.json` and `cer_gates.json` written, no NaN row, render.py's cam00 PNG at the mid-gap frame equals a direct gated render within 1/255, and the same frame with `q = 0` differs |

T0 and T1 pass on the unmodified tree; T2–T7 import the CER code and fail on it. Code review round (Sol, Fable)
added: T3 requires every listed gradient to exist plus a positive control (memory rows do get gradient at `q = 0`);
T4 requires each training image to be the canonical scene's frame for its camera, never cam00; T5 compares the whole
checkpoint recursively including optimizer state, and requires a CER directory whose checkpoint holds no CER state to
refuse; T6 requires a non-finite evaluated opacity (finite logits) to be caught. **Not tested, one line each:** the
resume refusal and `densify_until_iter < iterations` assertions in `train.py`, and the per-bank length assertion in
`get_cer_factor`. T7 runs 17,200 iterations, below `error_base_prune_steps` (20,000), so the error-based prune is not
exercised there; the real runs exercise it before CER starts.

## 8. CUDA: no change

- The factor is an elementwise python multiply on the opacity tensor before the autograd
  `Function`; `grad_opacities` flows back through it (`∂o'/∂o = f`).
- Residual rows are ordinary static rows with no new attribute the kernels would need.
- The contribution mass uses the existing colour gradient (§3.6).
- CUDA has no time argument (code map §4), so nothing temporal exists there to gate.

Both reviewers confirmed this (Sol 10/10, Fable 9/10 with the NaN caveat now guarded in §6).

## 9. Outside the substrate (not part of the Ex4DGS diff)

- **Launcher**: `agent-control/ars/sbatch/train_ex4dgs_cer.sbatch`, the wave-1 sbatch with
  `cd $W/repo/ex4dgs-cer` and `--cer_scene <canonical scene dir>`, provenance recording the
  branch, the `cer` commit, the worktree's porcelain and tracked-diff counts and the sha256 of
  `scene/cer.py`; `--exclude=lrdn0066`. Run ids `ars_w1_<scene>_cer_rep<r>` under
  `runs/ex4dgs_cer/`.
- **Scoring** (`agent-control/ars/ars_score.py`): a method `ex4dgs_cer` whose provenance rule
  accepts exactly the reviewed `cer` commit with porcelain 0 and tracked diff 0, otherwise the
  Ex4DGS checks unchanged; the gate verdict from `cer_gates.json` copied into the score (§6). The
  CER runs are added to `ars_score_worker.py`'s run list so each one is scored and gets a visual
  report.
- **Cost**: `begin` ≈ 1,710 render+backward passes and `gates` ≈ 2,280 (≈ 5–12 min together) on
  top of the base run's 1.2 GPU-h.
- **Nothing is submitted** until the principal says "submit".

## 10. Deliberately not built

**Known design point, not a bug:** at birth each donor and its residual copy both render inside the gap (a doubled
layer); the optimisation must thin one of them. **Wave-2 ablations recorded by the principal (2026-10-01), not built
now:** residual initialisation (low opacity at birth; a donor → copy swap inside the gap; dynamic copies for moving
donors, since a static snapshot at frame 74 of a donor that moves in the gap leaves a stationary ghost); the start
iteration (20k vs 30k, with the label plumbing through densification an earlier start needs); the 0.5 majority rule.
The residual gate's primary threshold 0.20 is frozen in the plan log (2026-10-01) with the two smoke values already
seen (0.10, 0.09); the reducer also reports 0.10 / 0.20 / 0.30 descriptively.

The plain notch (memory gating without a residual cohort) and a residual-only arm are wave-2
ablations, on hold. No smooth indicator, no learned or estimated clock, no learned `q`, no
lifecycle changes, no ply persistence of CER state, no new CLI beyond `cer_scene`.

## 11. Decisions this spec takes (for the principal's review of the diff)

1. **CER runs inside the same 40,000-iteration budget, starting at 30,000** (§2.2), not as extra
   iterations after a base run; upstream pruning is kept in both arms.
2. **Hard gap indicator on frame times** instead of the smooth flat-top (§1).
3. **Membership and birth by majority contribution** (≥ half of a row's own mass inside the
   region), memory on the ungated model, **residual donors on the memory-gated model** (§5.2,
   Fable F1): the one method decision the reviews raised.
4. **Gate thresholds**: memory ≥ 0.50 before the gap and after the return; residual ≥ 0.20 in the
   gap (§6). Every run is scored; rule 4 acts in the reducer.

## 12. Changes from rev 1

| change | source |
|---|---|
| resume refused (`train.py` assertion) | Sol 1, Fable F3 |
| instrument passes `near`/`far` explicitly | Sol 2 |
| "zero gradient" narrowed to the rendering loss; byte identity restricted to a fixed model | Sol 3, Fable F6 |
| donors measured on the memory-gated model; rule named as majority membership | Sol 4, Fable F1 |
| ply sidecar removed; ply fallback refused for a CER model | Sol 5–6, Fable F8 |
| every run scored; gate verdict carried to the reducer | Sol 7 |
| factor shape `[N, 1]` stated; per-bank length assertion | Sol (missing), Fable |
| promotion precondition corrected and asserted | Fable F2 |
| chkpnt30000 is one step before `begin` | Fable F4 |
| stable-sort step in the q = 0 argument | Fable F5 |
| `t_m = (A + B) // 2` | Fable F7 |
| T2 non-vacuity; T3 gradient test; T6 optimizer values; T7 at 17,200 iterations | Sol, Fable F9, F10 |
| aggregate split tolerance | Sol (suspected), Fable F11 |
| finite-opacity guard | Fable F12 |
| gates 3 and 5 labelled guards; after-return failure on same-pose explained | Fable F13 |
| pruning retained, deviation from Sol's round-1 text stated | Sol (missing) |
| **after the code reviews:** `begin` asserts `time_pad_type == 0` and that training images are the canonical scene's; no cam00 view; finite evaluated opacity on every frame (`check_finite`); a CER directory whose checkpoint lacks CER state refuses to load; `densify_until_iter < iterations` asserted; the launcher renders the config's final iteration and re-records provenance at render time; the scorer checks both | Sol code review, Fable code review (M1, m1–m9) |
