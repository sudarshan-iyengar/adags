# CER on Ex4DGS: implementation spec (2026-09-30, rev 1, before any CER code exists)

Complementary Episode Residuals (CER), as proposed in
[handoff rev 3 §3(b)](neurips-workshop-paper-handoff-2026-09-23.md) and
[sol_round1.md §1](../assets/neurips-plan-sol-2026-09-21/sol_round1.md), specified against
Ex4DGS at the pinned commit `7fac649`. Every `file:line` below was read in a clean clone of
that commit (`git status` empty). Code goes on branch `cer` in the worktree
`$W/repo/ex4dgs-cer`; the pinned checkout `$W/repo/ex4dgs` is never modified.

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
o(t) = [ σ(ω_s) ; σ(ω_d) · b(t) ]        shape [N_s + N_d, 1]
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
o'_i(t)   = o_i(t) · f_i(t)
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

**Exactness consequences (these are what the tests check).**

1. `q = 0`: `f_i = 1` for memory and ordinary rows and `f_j = 0` for residual rows, at every *t*.
   `x·1.0 = x` in IEEE float, and a row with opacity exactly 0 is skipped per pixel
   (`forward.cu:379-381`, `alpha < 1/255 → continue`) without touching the transmittance, so the
   image equals the image of the model without residual rows.
2. `q = 1`, `t ∉ [A, B]`: identical to case 1, so the residual is **exactly zero outside the gap**
   and the render equals unmodified Ex4DGS.
3. `q = 1`, `t ∈ [A, B]`: every memory row has opacity exactly 0: exact-zero presence, and
   **zero gradient** to its parameters from gap frames (the per-pixel skip also skips it in
   `backward.cu:586-588`; the multiply gives `∂o'/∂o = 0`).

CER runs use `q = 1`. `q` is not a config key; it exists for the `q = 0` identity test.

## 2. Where CER attaches, and when

### 2.1 The single opacity junction

`gaussian_renderer/__init__.py:64`
`opacity = pc.get_opacity_at_t(timestamp, mode=mode, training=training)` is the **only** caller
of `get_opacity_at_t` in the repository (grep over every `.py` outside `submodules/`), and the
opacity reaches CUDA only through the rasterizer call at `:100-109` (`opacities = opacity`).
Every render (training `train.py:138`, evaluation `render.py:68` and its timing loop `:94`, the
network GUI `train.py:100`, `training_report` `:341`) goes through `render()`. CER adds, after
`:64`:

```python
    if pc.cer_clock is not None:
        opacity = opacity * pc.get_cer_factor(timestamp)
```

So training and evaluation are gated by the same line (the 2026-09-10 gate-off evaluation defect
cannot recur through a second render path: there is none). No caller passes `mode ≠ 0`
(grep); if one did, the factor length would not match and the multiply would raise.

### 2.2 When CER starts: at the end of iteration `densify_until_iter` (30,000 of 40,000)

The wave-1 config (`agent-control/ars/configs/ex4dgs_n3v_base_res1.json`) has
`iterations 40000`, `densify_until_iter 30000`, `time_pad_type 0`. After iteration 30,000 the
upstream lifecycle can **only remove** rows, never add them:

| operation | site | runs after 30,000? |
|---|---|---|
| clone / split / opacity-error prune | `train.py:203-227` (`if iteration < opt.densify_until_iter`) | no |
| static→dynamic promotion, in-window | `train.py:228-232` (the `elif` of the same block) | no |
| static→dynamic promotion, window-end | `train.py:257-259`, `:262-274`, `:277-282` | no: `mark_last` needs `need_extract`, which is set only when `expand_duration` returns True (`:269-273`), and the window stops growing once it covers the sequence (`expand_duration`, `c_gaussian_model.py:1243-1262`) around iteration 12,400. **Asserted at CER start**, see below. |
| `adjust_temp_opa` | `train.py:233-234` (`< densify_until_iter − 3000`) | no |
| `prune_invisible`, `prune_small` | `train.py:237-241`, once per epoch | **yes** |
| `prune_nan_points` | `train.py:253`, every iteration | **yes** (removes NaN rows) |

Starting CER there gives the handoff's "short residual stage with the lifecycle frozen" without
disabling anything: no growth path runs, and the two pruning paths both go through
`prune_points` (`c_gaussian_model.py:715-763`), which CER extends so the labels follow. The base
and CER arms train identically for 30,000 iterations and differ only in the last 10,000; total
iterations, schedule and evaluator are unchanged (the plan's per-run cost line, 1.2 GPU-h for
each CER run, assumes exactly this).

The call goes at the end of the loop body, outside the `torch.no_grad()` block that ends at
`train.py:282`, because building the cohorts needs autograd (§4):

```python
        if dataset.cer_scene and iteration == opt.densify_until_iter:
            cer.begin(gaussians, scene, pipe, dataset)
```

The checkpoint at 30,000 (`train.py:195-197`, default `checkpoint_iterations` includes 30,000)
is written earlier in the same iteration, so `chkpnt30000.pth` is the model just before CER
starts. **Preconditions asserted inside `begin`** (they describe the setup, never a score):
every training camera with timestamp in `[0, 299]` is in the sampling window
(`scene.samplelist`, `scene/__init__.py:270-274`), i.e. the gap frames are being trained;
the model duration covers the sequence.

### 2.3 New configuration

One key, `cer_scene` (string, default `""` = CER off), registered in `ModelParams`
(`arguments/__init__.py:47-`), because `train.py:398-408` silently drops unregistered keys. It
points to the canonical ARS scene directory (`.../ars/wave1/<scene>`) holding `event_spec.json`
(clock A, B; the target's part ids = the parts of the object whose `role` is `target`) and
`gt/identity/<camera>.npz` (front-most identity per pixel, `f<frame:04d>`, int16 1014×1352,
`ARS_SPEC_FORMAT.md` §5, §7). Only training cameras' identities are read, never cam00 (plan §4,
declared input). Nothing else is configurable: `q = 1`, the start iteration is
`opt.densify_until_iter`, and the thresholds in §4–§6 are constants in the code.

## 3. Facts about the substrate that the design depends on

1. **One opacity junction, python-side** (§2.1). CUDA receives an evaluated scalar per row and
   has no notion of time (code map §4, verified).
2. **Time is the integer frame index.** The neural3dvideo reader sets
   `timestamp = frame − start_timestamp` (`scene/dataset_readers.py:308`), `start_timestamp = 0`;
   `time_pad_type 0` adds no mirrored padding cameras (`scene/__init__.py:124-162` only act for
   types 1 and 2). cam00 frame *f* renders at timestamp *f* (`render.py:66-68`).
3. **Evaluation restores the checkpoint, not the ply.** `render.py:128` builds `Scene(...,
   load_iteration=40000, opt=opt)`; `scene/__init__.py:166-170` calls `restore(chkpnt40000)`
   when the checkpoint exists (it always does: `train.py:55`). The ply path
   (`:171-175`, `load_ply`) is only the fallback.
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
   and early termination as the forward pass. `render()` already accepts precomputed colours
   (`override_color`, `gaussian_renderer/__init__.py:96-97`).
7. **Checkpoint tuple.** `capture()` returns a 40-element tuple (`:217-260`); `restore()`
   unpacks exactly 40 (`:262-321`).

## 4. The contribution-mass instrument (shared by membership, birth and the gates)

```
mass(pc, cams, masks) -> inside[N], total[N]      (float64, cuda)
for each (camera, mask) pair:
    colours = ones([N, 3], requires_grad)                         N = N_s + N_d
    image   = render(camera, pc, pipe, bg = zeros(3), override_color = colours)["render"]   [3, H, W]
    g       = autograd.grad( (image[0] * mask).sum() + image[1].sum(), colours )[0]           [N, 3]
    inside += g[:, 0];  total += g[:, 1]
```

`inside_i = Σ_{frames, cameras} Σ_{p ∈ mask} α_i T_i`, `total_i` the same over every pixel.
`torch.autograd.grad` is used, not `.backward()`, so **no parameter `.grad` is touched** (the
call happens between two optimizer steps). The render uses the current CER factor, so the same
function measures the ungated model (before labels exist, or `q = 0`) and the gated one.
`mask` is `[H, W]` float {0, 1}; `H, W` must equal the render's (asserted; both 1014×1352 on the
1352 route). Camera number and frame come from the camera object: `cam.timestamp` and the
`camNN` directory of `cam.image_path`; the test suite asserts that `realpath(image_path)` ends in
`frames/camNN/<frame:04d>.png` of the canonical scene for every camera used.

Masks, training cameras only:

- **target mask** `M_c(t)` = identity `∈` target part ids, camera *c*, frame *t*;
- **gap region** `G_c` = `M_c(A − 1)`: where the target was, in camera *c*, on the last frame
  before it left. (The canonical `ghost` region, the target-alone projection, exists only for
  cam00; the front-most target mask at `A − 1` is its training-camera proxy.)

Windows: pre-gap `[0, A − 1]`; gap `[A, B]`; return `[B + 1, B + 30]`.

## 5. The cohort lifecycle

### 5.1 Memory cohort (label 1), set once at CER start

On the model at the end of iteration 30,000, ungated (no labels yet): `(inside, total) =
mass(pre-gap frames × training cameras, target masks)`.

```
memory = (total > 0) & (inside >= 0.5 * total)
```

A row joins the memory cohort when most of its pre-gap contribution lies on the target. The
labels are split into `_cer_static [N_s]` and `_cer_dynamic [N_d]` (uint8, cuda).

### 5.2 Residual cohort (label 2), born once at CER start

Same ungated model: `(inside, total) = mass(gap frames × training cameras, gap regions G_c)`.

```
donors = (total > 0) & (inside >= 0.5 * total)          |D| rows, either bank
t_m    = (A + B) / 2
```

Each donor is copied into the **static bank** as its rendered state at `t_m`, taken through the
same getters the renderer uses, so a residual row renders at `t_m` exactly as its donor did
(opacity up to the logit round trip):

| new static tensor | value | shape |
|---|---|---|
| `_xyz` | `get_xyz_at_t(t_m)[D]` | `[D, 3]` |
| `_xyz_disp` | zeros | `[D, 3]` |
| `_features_dc`, `_features_rest` | `get_features()[D]` split `[:, :1]`, `[:, 1:]` | `[D, 1, 3]`, `[D, 15, 3]` |
| `_scaling` | `cat(_scaling, _scaling_motion)[D]` (pre-activation) | `[D, 3]` |
| `_rotation` | `get_rotation_at_t(t_m)[D]` (as passed to CUDA) | `[D, 4]` |
| `_opacity` | `inverse_sigmoid(get_opacity_at_t(t_m)[D].clamp(1e-6, 1 − 1e-6))` | `[D, 1]` |

Added through `cat_tensors_to_optimizer` with exactly the seven static groups (zero Adam
moments for the new rows). Non-parameter static tensors are extended by `D` rows only:
`xyz_gradient_accum, denom, xyz_error_accum, xyz_ssim_error_accum, error_denom` zeros `[D, 1]`;
`xyz_error_min` 1000 `[D, 1]`; `xyz_error_min_timestamp` = A `[D, 1]` (non-negative, so
`prune_invisible` does not delete an unobserved newborn: it prunes rows whose timestamp is
negative, `:1074-1085`, and after 30,000 nothing updates that tensor); `max_radii2D` 0 `[D]`,
`min_radii2D` 1000 `[D]` (`prune_small` then treats a residual row like any other once it has
been seen in a gap frame). `_cer_static` gets `D` twos. After birth `N_s' = N_s + D`; the
dynamic rows keep their labels because labels are stored per bank.

**Why the largest gap-region contributors.** On the real fixture the same rows paint the target
outside the gap and the counterfactual background inside it
([fixture lane](absence-fixture-lane-2026-09-10.md) §5); the residual cohort starts as a copy of
exactly those rows, positioned where they rendered mid-gap, and from then on receives gradient
only from gap frames.

### 5.3 From 30,001 to 40,000

Every render is gated by §1. Upstream `prune_invisible`, `prune_small` and `prune_nan_points`
keep running for all rows; `prune_points` masks `_cer_static` and `_cer_dynamic` with the same
keep masks it applies to the other per-row tensors. No other code path changes row counts in
this interval (§2.2); if one did, the factor length would mismatch at the next render and raise.

### 5.4 Persistence

- **Checkpoint.** When CER is active, `capture()` appends one element, the dict
  `{static, dynamic, A, B, q}`; when inactive it returns the upstream 40-tuple unchanged.
  `restore()` accepts 41 (splits the dict off) or 40 (upstream checkpoints, CER off).
- **Ply fallback.** When CER is active, `save_ply` also writes `cer.pt` (the same dict) beside
  `point_cloud.ply`; `load_ply` loads it if present. Without this a ply-loaded CER model would
  render its residual rows as ordinary always-on static rows, silently.

## 6. The contribution gates (plan §4 rule 4; rev 2 §6), computed at the end of training

Written by `train.py` after the last iteration to `<model_path>/cer_gates.json`, on the final
model with `q = 1`, training cameras only (the gates read no cam00 pixel, so they cannot leak the
scored outcome). Thresholds declared here, before any CER output exists:

| gate (rev 2 §6 wording) | measurement | pass |
|---|---|---|
| memory active before the gap | memory share of target-mask mass over the pre-gap window | ≥ 0.50 |
| memory active after the return | memory share of target-mask mass over the return window | ≥ 0.50 |
| memory suppressed inside the gap | memory mass over the gap window, every pixel | exactly 0 |
| residual active inside the gap | residual share of gap-region mass over the gap window | ≥ 0.20 |
| residual zero outside the gap | residual mass over the pre-gap and return windows, every pixel | exactly 0 |
| both banks covered | memory rows and memory mass per bank, reported | reported (the factor acts on the concatenated vector by construction; test T4 checks one memory row per bank) |

The birth-time numbers (memory share on the ungated model, donor count, donor share of the gap
region) are written to `cer_begin.json` at 30,000. The 0.50 memory threshold is the one frozen
for the Multi4D census (S0 rule 6); 0.20 is Sol's round-1 figure. A scene whose CER run fails a
gate is DESIGN_WITHOUT_POWER for that run (plan §4 rule 4), never dropped; the scorer checks
`cer_gates.json` before scoring (§9).

## 7. Tests, written before the code (they must fail on the unmodified tree)

All on Leonardo in one GPU job on one node (forward rendering is compared **on one GPU only**;
the record shows up to 36/255 differences across GPUs). Inputs: the wave-1 base checkpoints
`runs/ex4dgs/ars_w1_<scene>_ex4dgs_rep1/model/chkpnt{30000,40000}.pth` (upstream 40-tuples) and
the canonical scenes. Images are compared as float32 tensors with `torch.equal`.

| id | test | pass criterion |
|---|---|---|
| T0 | control: unmodified checkout renders cam00 frames 0–299 of the bottle same-pose base model twice | byte-identical (establishes that the comparison below is meaningful on this GPU) |
| T1 | CER tree, CER off (no labels): same renders | byte-identical to T0 |
| T2 | CER tree, `begin` on that checkpoint (memory and residual cohorts built, residual rows appended), `q = 0`: same renders | byte-identical to T0, all 300 frames |
| T3 | `q = 1`: opacity factor on every frame 0–299; renders | residual entries exactly 0.0 and memory entries exactly 1.0 for `t ∉ [A,B]`; memory entries exactly 0.0 and residual exactly 1.0 for `t ∈ [A,B]`; ordinary exactly 1.0 always; renders for `t ∉ [A,B]` byte-identical to T0; renders for `t ∈ [A,B]` differ from T0 |
| T4 | instrument and non-vacuity | per render, `Σ_i total_i` equals `Σ_p image_1(p)` of the ones-colour render within 1e-4 relative; `inside + (complement-mask inside) = total` within 1e-5 relative; memory share of pre-gap target mass ≥ 0.50 on the ungated model **for each of the nine wave-1 run-1 base models** (reported per scene: a failure here is a DESIGN_WITHOUT_POWER forecast before any training); with `q = 1`, memory mass over the gap window is exactly 0 and residual mass over pre-gap and return windows is exactly 0, residual mass over the gap window > 0; at least one memory row in each bank is forced (by construction in the test) and zeroed in the gap |
| T5 | checkpoint round trip | `capture` → `torch.save` → `torch.load` → fresh model `restore`: every tensor, the labels, A, B and q equal; cam00 renders at frames {0, 59, 60, 74, 89, 90, 299} byte-identical before and after; the same through `save_ply`/`load_ply` + `cer.pt`; an upstream 40-tuple still restores with CER off |
| T6 | pruning keeps labels aligned | prune a fixed random 10% of static and of dynamic rows through `prune_points`: surviving labels equal the pre-prune labels under the keep masks; label lengths equal row counts; optimizer state lengths equal row counts |
| T7 | integration smoke (not evidence) | `train.py` with CER on the bottle same-pose scene at a shortened schedule on the debug queue (iterations 14,000, `densify_until_iter` 13,000, so the window already covers the sequence); `begin` runs, no NaN row appears, `cer_begin.json` and `cer_gates.json` are written, `render.py` on the result gives cam00 images equal to a direct gated render, and a render at a gap frame with the CER state stripped differs |

T0–T6 use `render()` and the model directly (a small driver under `agent-control/ars/cer/`,
not in the substrate); T7 uses the real entry points. Tests T1–T6 import CER code, so they fail
on the unmodified tree; T0 must pass on it.

## 8. CUDA: no change

- The factor is an elementwise python multiply on the opacity tensor before the autograd
  `Function`; `grad_opacities` flows back through it (`∂o'/∂o = f`).
- Residual rows are ordinary static rows with no new attribute the kernels would need.
- The contribution mass uses the existing colour gradient (§3.6).
- CUDA has no time argument (code map §4, verified), so nothing temporal exists there to gate.

## 9. Outside the substrate (not part of the Ex4DGS diff)

- **Launcher**: `agent-control/ars/sbatch/train_ex4dgs_cer.sbatch`, the wave-1 sbatch with
  `cd $W/repo/ex4dgs-cer` and `--cer_scene <canonical scene dir>`, provenance recording the
  `cer` commit, the worktree's porcelain count and its tracked-diff count; `--exclude=lrdn0066`.
  Run ids `ars_w1_<scene>_cer_rep<r>` under `runs/ex4dgs_cer/`.
- **Scoring** (`agent-control/ars/ars_score.py`): a provenance rule that accepts exactly the
  `cer` branch head commit with a clean worktree for the method `ex4dgs_cer`; a precondition that
  reads `cer_gates.json` and marks a failing run DESIGN_WITHOUT_POWER instead of scoring it; the
  CER runs added to `ars_score_worker.py`'s run list so each is scored and gets a visual report.
- **Nothing is submitted** until the principal says "submit".

## 10. Deliberately not built

The plain notch (memory gating without a residual cohort) and a residual-only arm are wave-2
ablations, on hold; either is a one-line change to the birth rule and is not written now. No
smooth indicator, no learned or estimated clock, no learned `q`, no lifecycle changes, no new
CLI beyond `cer_scene`.

## 11. Decisions this spec takes (for the principal's review of the diff)

1. **CER runs inside the same 40,000-iteration budget, starting at 30,000** (§2.2), not as extra
   iterations after a base run, so base and CER differ only in the mechanism.
2. **Hard gap indicator on frame times** instead of the smooth flat-top (§1).
3. **Membership and birth by majority contribution** (≥ half of a row's own mass inside the
   region), both measured on the ungated model at 30,000 (§5).
4. **Gate thresholds**: memory ≥ 0.50 before the gap and after the return; residual ≥ 0.20 in
   the gap (§6). On the moved-return scenes the memory rows may not follow the target to its new
   pose; if so, the after-return gate fails there and those runs are DESIGN_WITHOUT_POWER, which
   would make the moved-return event type untestable in the causal proof.
