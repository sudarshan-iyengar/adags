# BLOCKING: rendered-flow supervision has no gradient (2026-08-17)

Status: **F and X cells CANNOT be run. The flow lane is stopped at a
scope boundary.** EXPLORATORY diagnostic; no training result.

The bounded N3V plumbing smoke (`scripts/flow_plumbing_smoke.py`,
experiments 128-131) was built to establish that the rendered-flow path
works BEFORE any supervision cell could be interpreted. It found that it
does not.

## What passes

| check | result |
|---|---|
| target units | PIXELS of their own raster |
| resize rescales magnitudes | YES — ratio 0.4998/0.4999 at half size, expected 0.5 |
| rendered vs target shapes at the loss | MATCH, so `compute_flow_loss`'s value-preserving interpolate is a no-op |
| mask coverage | 97.7-99.4% of pixels supervised |
| `training_setup` propagates `enable_rendered_flow` | YES (asserted; the smoke refuses otherwise) |
| rendered flow output is in the autograd graph | `requires_grad=True`, `grad_fn` present |
| projection chain is differentiable | YES — coefficients 1.675e-07, shared basis 5.845e-03 |

## What fails

```
rendered flow (untrained model)      EXACTLY zero
gradient on LoRA coefficients        0.000e+00
gradient on shared basis             0.000e+00

after perturbing coefficients to definitely-nonzero motion:
rendered flow absmax                 8.033e-04   (field DOES respond)
gradient on LoRA coefficients        0.000e+00
gradient on shared basis             0.000e+00
```

**VERDICT: severed at the rasterizer boundary.** The perturbation probe
exists precisely to separate a degenerate zero-init (harmless) from a
dead path (fatal), and it returns the fatal one: the field responds to
motion, so the renderer branch IS taken, and yet no gradient survives.

## Where the defect is — the Python is correct

`gaussian_renderer/diff_gaussian_rasterization.py` was read end to end.
The autograd Function saves `flow_2d`, `backward` accepts `grad_flow`,
passes it into `_C.rasterize_gaussians_backward`, receives `grad_flows`,
and returns it in the 5th slot, which matches `flow_2d`'s position in
the forward argument list. **The wrapper is wired correctly.**

`grad_flows` therefore comes back from the COMPILED CUDA EXTENSION as
zero. The defect is in the rasterizer's CUDA backward, which does not
populate the flow gradient. That code is not in this repository.

This is consistent with the historical record: `enable_rendered_flow`
and `lambda_track_flow` have NEVER been set together in any tracked
config or any recorded run ([[gap_map]] lists the hook as an open gap),
so a non-functional backward would never have been noticed.

## Why this stops the lane rather than being repaired here

Repairing it means modifying and recompiling the differentiable
rasterizer's CUDA sources. That is:

* NOT "target/loss semantics" — the targets are correct (pixels, correct
  resize, 99% mask coverage) and the loss is correct (shapes match,
  `compute_flow_loss` behaves);
* a change to the core rendering machinery, which is
  representation-critical and would need its own review;
* subject to the directive's own precondition that the disabled-flow
  path be proven NUMERICALLY UNCHANGED before experiments 104 and 123
  may still serve as controls — a proof that is much harder after
  touching the rasterizer than before.

**Running F and X in this state would compare two identical no-ops.**
Both cells would train exactly the model the control trains, and any
difference between them would be scheduler noise. The attribution gate
("if X matches F, treat as unattributable") would fire for a reason that
has nothing to do with flow semantics, and would waste the gate.

## A second finding: the direction test is not reliable on this scene

The RGB-warp direction check returned `minus` on two runs and `plus` on a
third, over different probe cameras. The cause is visible in the data:
N3V `cut_roasted_beef` flow has a median magnitude of **0.015-0.035
pixels**. At sub-0.1-pixel motion, warping by +F and by -F produce
almost the same image and the test is decided by noise, not by
convention.

So the sign convention of the SEA-RAFT sidecars relative to the
renderer's `screen(t+dt) - screen(t)` is **NOT established**, and this
page does not claim it. Any future direction test must restrict itself
to pixels whose flow magnitude is well above the noise floor. Recorded
because an unestablished sign is exactly the error that presents as
"flow supervision hurts".

## CORRECTION and exact diagnosis (2026-08-17, same day)

Two statements above are WRONG and are corrected here rather than
rewritten. A dedicated investigation, independently re-verified against
source by this author, settled the class of the defect.

**CORRECTION 1.** This page says the rasterizer's CUDA backward "is not
in this repository". **It is.** `diff-gaussian-rasterization/` is TRACKED
(`git ls-files` returns `cuda_rasterizer/backward.cu`, `forward.cu`,
`CMakeLists.txt`), `Dockerfile.apollo-{v100,h100}` COPY it into the image
and `pip install` it, and `backward.cu` is unmodified since the initial
commit. The claim was load-bearing for the "out of scope" conclusion and
was false.

**CORRECTION 2.** This page implies the repair means writing new kernel
arithmetic. It does not. **The flow gradient is already implemented and
correct — it lives in a kernel that is never launched.**

### The defect, verified line by line

`backward.cu` contains exactly ONE `<<<...>>>` launch, at `:1599`:

```cuda
PerGaussianRenderCUDA<NUM_CHANNELS> <<<((B*32) + THREADS - 1) / THREADS, THREADS>>>(
```

Every flow computation in the file — `:995` through `:1164`, including
the only accumulation

```cuda
:1164   atomicAdd(&(dL_dflows[global_id * 2 + ch]), dchannel_dcolor * dL_dchannelflow);
```

— lives inside `renderCUDA`, defined at `:984` and **NEVER LAUNCHED**.
The only `renderCUDA<...><<<>>>` in the codebase is `forward.cu:825`,
which instantiates forward.cu's OWN `renderCUDA` at `forward.cu:602`.
backward.cu's is an uninstantiated template: it emits no code, no linker
error and NO COMPILER WARNING.

The kernel that does run declares the flow parameters at `:1244`,
`:1254`, `:1259` and then contains **zero** flow references in its whole
body (`:1260`-`:1575`). It computes colour only. So `dL_dflows` returns
exactly as allocated at `rasterize_points.cu:241`,
`torch::zeros({P, 2})`.

That is why the failure was silent: no exception, no shape mismatch, no
illegal access — just a zero tensor.

### Wrapper and binding are BOTH correct

Ruled out with counting, not impression. Forward returns 10 values and
`backward` takes 10 grads, with `flow` as output #5 and `grad_flow` as
arg #5. Forward has 19 inputs and the returned `grads` tuple has 19
entries with `grad_flows` at position 5, matching `flow_2d` at position
5. `flow_2d` IS passed to C++ and IS in `save_for_backward`; there is no
`mark_non_differentiable`. On the binding side `rasterize_points.cu`
allocates `dL_dflows` (`:241`), passes `dL_dout_flow` (`:316`) and
`dL_dflows` (`:324`), and returns it in the right tuple slot (`:343`),
matching `backward.h:52-53`. Pointer threading is intact all the way to
the dead kernel.

The forward path is also genuinely differentiable in principle —
`forward.cu:756-757` composites flow with standard `alpha * T` weighting
and writes it at `:794`. No argmax, no nearest-Gaussian, no detach.

### Which extension runs — settled statically

`gaussian_renderer/diff_gaussian_rasterization.py:18` imports
`_adags_diff_gaussian_rasterization`, a PROJECT-UNIQUE module name that
no PyPI package can shadow, and the JIT fallback compiles the SAME
in-repo sources. So the in-repo `.cu` files ARE what ran.

### All three rasterizer trees are the same

| tree | verdict |
|---|---|
| `D:\adags\diff-gaussian-rasterization` | reference |
| `apollo:.../proj_adags/repo/adags/diff-gaussian-rasterization` | identical (size deltas are exactly line counts, i.e. CRLF vs LF; `diff` returns 0 after stripping `\r`) |
| `apollo:.../project_adags/experiments/budget_match/diff-gaussian-rasterization` | byte-identical to local |

**No tree implements a flow gradient another lacks** — all three carry
the same 30 flow identifiers at the same lines, and the same defect. The
one real difference is tree C's `setup.py`, which builds a stock-named
extension with `nvcc -g -G` (device debug, optimization off); relevant
only if C was ever used for timing, irrelevant to flow.

### The minimal repair, and its one disclosed approximation

Purely additive inside `PerGaussianRenderCUDA`, about six lines, no
forward change and no new buffers: a `Register_dL_dflows[2]`
accumulator beside `:1324`, reading `dL_dpix_flow[ch * H * W + pix_id]`
and accumulating `weight * dL_dchannelflow` after `:1394` (mirroring the
colour path at `:1402`), then an `atomicAdd` into `dL_dflows` in the
`gaussian_idx < P` epilogue (mirroring `:1449`). `weight = alpha * T` is
already in hand.

**Disclosed approximation:** the dead kernel also fed flow into
`dL_dalpha` (`:1160`). Reproducing that in the per-bucket kernel needs a
running `ar_flow[2]` reconstruction, which requires a flow analogue of
`sampled_ar` in the forward — `SampleState` (`rasterizer_impl.h:81-88`)
carries only `bucket_to_tile, T, ar, ard`. Omitting it means the flow
loss shapes flow VALUES (hence motion) but does not push opacity, conic
or mean2D. That is a defensible scope for flow supervision and must be
recorded rather than glossed.

### What still stands

The lane remains STOPPED pending authorization, but for a smaller and
better-understood reason than this page originally gave: the change is a
`.cu` edit requiring an image REBUILD and a NEW DIGEST, plus the
directive's own precondition that the disabled-flow path be proven
numerically unchanged before experiments 104 and 123 may still serve as
controls. The verification gate is `scripts/flow_plumbing_smoke.py`
probes F and G, which already fail today and must pass after the repair.

## What is NOT concluded

That flow supervision would or would not help. Nothing about the idea
has been tested — only that the machinery cannot currently express it.
The N3V targets themselves are sound on every axis measured.

## Focused adversarial review of the repair (2026-08-17)

**Reviewer substitution, recorded rather than glossed.** The directive
asked for Codex `gpt-5.6-sol-xhigh`. That identifier is REJECTED by the
API on this account: `HTTP 400, "The 'gpt-5.6-sol-xhigh' model is not
supported when using Codex with a ChatGPT account."` The review was run
instead on Codex at default model with `model_reasoning_effort: high`,
`sandbox: read-only`, `approval-policy: never`, carrying the identical
12-item prompt. **The requested reviewer did not run.** The review that
did run is a real independent read of the diff, but it is not the one
specified, and no claim here rests on it having been the specified one.

Scope given: the actual CUDA/Python diff at 6550772, the reference
oracle, the gradient tests, the active build/import path and
disabled-flow equivalence. No file modification, no unrelated renderer
refactoring.

### Result: 9 of 12 OK, 3 defects, verdict REJECT

OK, and these are the load-bearing ones: the implemented derivative IS
the VJP of the ACTIVE forward compositor (item 1); `T` is `T_{i-1}` and
`weight` is `w_i` (2); the `arflow` checkpoint reconstruction has NO
off-by-one-batch error and its index arithmetic matches `ar` (3); both
`arflow` and `dL_dpixflow` are shuffled (4); static coupling is correct
and `dL_dflows` is correctly restricted to `gaussian_idx < P` (5); the
background omission is correct (6); buffer sizing and aliasing are sound
(7); positional threading agrees at every declaration, definition, launch
and call site (8); disabled-flow equivalence HOLDS, so experiments 104
and 123 remain valid controls (11); `resize_flow` semantics and its
autograd safety are correct (12).

The reviewer independently confirmed the point that decides the patch:
the dead `renderCUDA` is NOT a better oracle. It does not assign the
flow-mediated alpha derivative to static contributors, so relative to the
active forward the patch is right and the dead kernel is wrong. Not
reviving it was correct.

### Finding 1, ranked HIGH by the reviewer — verified real, NOT fixable here

`alpha = min(0.99f, con_o.w * G)` (`backward.cu:1410`), but `dL_dG =
con_o.w * dL_dalpha` and `Register_dL_dopacity += G * dL_dalpha` both use
`d(alpha)/d(con_o.w*G) = 1`. In the saturated regime the true derivative
is zero, so opacity and geometry gradients are nonzero where they should
vanish.

Verified in source, and the reviewer's own text concedes it "predates the
patch for other channels". The structural fact that settles the
disposition: `dL_dalpha` is a **single shared accumulator**. Colour, flow,
depth and the background term all add into it, and ONE `dL_dG` is formed
from the total. Therefore:

* gating the clamp changes **colour and depth** gradients, which
  numerically invalidates experiments 104 and 123 as disabled-flow
  controls — a precondition this lane is explicitly bound by;
* gating only the flow share would be incoherent, because the clamp is a
  property of alpha, not of a channel.

So the fix is forbidden by a binding constraint, not merely declined. It
is also out of scope under the surgical-change rule: pre-existing,
shared, and inherited identically by every channel. It was already
disclosed in 6550772's own commit body before the review ran.

**The claim is narrowed rather than defended.** The patch delivers an
exact VJP **with respect to alpha** — which is what the reviewer's item 1
confirms. It does NOT deliver an exact gradient to opacity and geometry
in the saturated-alpha regime, and the word "complete" in the commit
subject overstates that by exactly this much. Recorded here because the
commit message cannot be rewritten.

### Finding 2, B == 0 — verified real, pre-existing, untouched

`rasterize_points.cu:273` guards `P != 0` only, while the launch grid is
`((B*32) + 31) / 32 = B`, so `B == 0` with `P > 0` is a zero-grid launch
(`cudaErrorInvalidConfiguration`). Reachable in principle when every
primitive touches zero tiles. Shared with colour, untouched by this
patch, mentioned and not fixed.

### Finding 3, checkpoint boundary untested — ACCEPTED and FIXED

The only finding inside this patch's own scope, and it targets precisely
its novel part. Every existing scene ran entirely inside bucket 0, where
`sampled_arflow` is zero, so a wrong bucket stride, a wrong channel
stride or a checkpoint taken one batch late would all have PASSED.

Fixed by `CheckpointBoundaryTests` in `tests/test_flow_backward_vjp.py`
(5 tests): 72 primitives at strictly increasing depth in one tile,
crossing the stride-32 checkpoint twice.

* `dL_dflows` alone would NOT have caught this — it is `weight *
  upstream` and reads no `arflow`. Only the alpha path consumes the
  reconstructed suffix, so the opacity tests are the ones that bite.
* One test is a guard on the test itself: it asserts the scene really
  does exceed 32 contributors, AND that no pixel hits the kernel's
  `T < 1e-4` early-out, which the oracle does not model. Per-primitive
  opacity is held at 0.08 for that reason.
* The strongest check is oracle-independent: central differences on the
  opacity of primitive 50, which lies past the first checkpoint and whose
  gradient therefore comes from a RECONSTRUCTED `arflow`, not a
  zero-initialised bucket.

Also noted by the reviewer and accepted as a stated limitation: the
saturated-alpha test asserts only finiteness. That is deliberate and now
consistent with finding 1 — asserting correctness there would assert
behaviour the patch does not and may not provide.

### Status

Tests: 36 in this file, collection verified. All CUDA tests SKIP on the
workstation, which has no GPU and no working torch. **A skip is not a
pass.** Nothing here is validated until it runs on a V100 against the
rebuilt image.

## V100 runtime verification, experiments 132-136 (2026-08-17)

Image `sudarshaniyengar/adags@sha256:aaddebe080c11ec02b489af7fde88e565f47d518170dcff2a10ef19d8c286bfa`
(tag `apollo-v100-flow-vjp-c15ef19`), built from the patched commit,
`TORCH_CUDA_ARCH_LIST="7.0 8.9"`, Dockerfile UNCHANGED (the patch adds no
build dependency). `apollo-v100-v1` was NOT overwritten and remains at
`sha256:51f8a852...`. All cells DGX/V100, exploratory,
`evidence_bearing: false`. Total cost ~0.5 GPU-hours.

### The headline: the flow VJP is still UNVERIFIED

Not confirmed, not refuted. Five runs produced no scientific result about
flow supervision. Recorded as a negative outcome rather than dressed up.

### What each run established

| exp | question | result |
|---|---|---|
| 132 | does the repaired gradient run? | FAILED, and the failure message was WRONG (see below) |
| 133 | is the zero a confounder? | scene renders (radii 7, 6), forward flow absmax 1.339, upstream 0.996, flow in the graph — and gradient still 0.0 |
| 134 | where does the gradient go? | a COLOUR loss also gives 0.0 on opacities, means3D, means2D |
| 135 | is the backward grid empty? | NO — `num_buckets` 4 / 4 / 12 and `num_rendered` 8 / 32 / 288 at 2 / 8 / 72 primitives; all gradients still 0.0 |
| 136 | is this the patch's fault? | **NO** — the OLD pre-patch image behaves IDENTICALLY |

### The patch is exonerated, and one thing is established in passing

Experiment 136 ran the identical probe against the pre-patch image. It
returned the same zeros, the same `num_buckets: 4`, the same
`num_rendered: 8`, and a **bit-identical** colour forward
(`0.519281804561615` on both images). So:

* the CUDA patch did NOT break the colour, alpha or depth path;
* forward equivalence between the two images is directly MEASURED on this
  scene, which is a small independent addition to the disabled-flow
  equivalence argument (it is one scene, not a proof over all inputs).

### The real defect: the test harness, and it is PRE-EXISTING

`TinyScene` in `tests/test_flow_backward_vjp.py` produces **zero
gradients for every loss** — colour included — on BOTH images, at every
scene size, with a valid bucket count and a correct forward. The real
training path does not have this problem: experiments 84, 104 and 123 all
trained through the colour gradient, and the earlier N3V smoke measured
nonzero gradients through the projection chain.

So the harness constructs a call the real renderer never makes. The cause
is NOT yet identified, and this page does not guess at it. Every argument
position from Python to the kernel WAS read end to end and matches:
autograd input 4 = `flow_2d` ↔ grads 4 = `grad_flows`; C++ return 6 =
`dL_dflows` ↔ unpack 6 = `grad_flows`; `grad_flow` → `dL_dout_flow` →
`dL_dpix_flow`; launch and kernel parameter lists agree positionally.

### CORRECTION: the 36 tests are not evidence

The previous entry on this page reported "36 tests, collection verified"
after adding `CheckpointBoundaryTests`. That count stands, and its
evidential value does not. **Every CUDA test in that file is built on
`TinyScene`, and `TinyScene` cannot currently produce a gradient at all.**
Had they run, they would have failed for a harness reason and said
nothing about the VJP. They must not be cited as verification of the flow
repair until the harness is fixed and the colour-gradient control passes.

The adversarial review's item 10 asked whether these tests would catch a
wrong sign, a wrong suffix, a missing shuffle or dropped static coupling,
and concluded they would. That assessment was STATIC and could not have
known the harness returns zero at runtime. The honest answer to item 10
today is: they would catch none of it, because they cannot distinguish
any gradient from any other.

### CORRECTION: experiment 132's failure message asserted what it had not shown

132 died with "dL_dflows is EXACTLY zero: this is the pre-repair binary.
The image does not contain the flow VJP patch." Both sentences were
unfounded and the second is now known false. The image was pulled by
digest; its sources carry the patch (`arflow` x10 in `backward.cu`); the
wheel rebuilt (55 s of nvcc, 383 KB, installed over the old one); and the
resulting `.so` differs from the old image's (1,301,952 vs 1,301,920
bytes; `db6c3025...` vs `95819edf...`).

The script now measures radii, rasterized count, forward flow, autograd
connectivity and upstream magnitude BEFORE concluding anything, and
reports every measurement even when a stage raises. Recorded because a
verifier that misattributes its own failure is worse than no verifier: it
manufactures false confidence in a wrong diagnosis.

### What is NOT concluded

That the flow VJP is correct, or incorrect. That flow supervision helps
or hurts. F and X remain BLOCKED — not because a gradient term is known
missing, but because no instrument has yet demonstrated the gradient is
live on the real path.

## CAUSE IDENTIFIED (2026-08-17, later the same day)

The zero-gradient harness defect recorded above — "`TinyScene` produces
zero gradients for every loss, colour included, on BOTH images, at every
scene size, with a valid bucket count and a correct forward. The cause is
NOT yet identified" — **has been identified and repaired.** Full record
and measurements: [[rasterizer-backward-two-defects-2026-08-17]].

It was `backward.cu`'s per-tile early-out

```cuda
if (bucket_idx_in_tile * 32 >= max_contrib[tile_id]) { return; }
```

`max_contrib` is READ there and written NOWHERE — that is its only
appearance in the whole rasterizer. Upstream taming-3dgs fills it with a
block-wide max-reduction at the end of its forward render kernel; this
fork's `forward.cu` takes the pointer and never touches it, and
`ImageState` is carved from a `torch::empty` buffer with only `ranges`
memset. So the guard compared against uninitialised device memory, and
when that memory was zero the condition held for every bucket and the
kernel returned before doing any work.

Measured on a V100, same scene and forward (radii `[7, 6, 6]`): with the
guard, `dL_dopacity`, `dL_dmeans3D` and `dL_dmeans2D` all exactly 0.0;
with it removed, 17.90, 3.67 and 6.03.

Consequences for what this page concluded:

* **The diagnosis "severed at the rasterizer boundary" was right, and the
  location was one level earlier than anyone looked** — the kernel's first
  twenty lines, before any flow code.
* **The flow VJP patch stays exonerated.** Experiment 136 attributed the
  zeros to something shared with the pre-patch image, which was correct.
* **The CORRECTION that "the 36 tests are not evidence" is now spent.**
  With the guard removed, `tests/test_flow_backward_vjp.py` runs 36 and
  **33 pass**, including the checkpoint-boundary suite, the static
  coupling tests and both LoRA reachability tests. The residue is one
  `P == 0` host-pointer error and finite-difference comparisons whose
  black-background baseline is itself unreliable (see the open items on
  the new page); none of it is a flow-VJP defect.

This does NOT by itself unblock F and X. What it establishes is that the
instrument works, which is the precondition the page asked for.
