# Two defects in the ACTIVE backward render kernel (2026-08-17)

Status: **both CONFIRMED on a V100 and REPAIRED.** EXPLORATORY
diagnostic; no training result, no scientific claim about any method.

`PerGaussianRenderCUDA` is the only launched backward render kernel
(`backward.cu`; the tile-based `renderCUDA` in the same file is an
uninstantiated template — see
[[rendered-flow-supervision-is-nonfunctional]]). Two independent defects
were found in it. They are recorded together because the second one hid
the first: while it was present, every gradient the kernel produced was
exactly zero, so no test could distinguish any kernel from any other.

## Defect 1 — the colour opacity gradient counted the background twice

### What the kernel computed

The compositing forward is, per pixel over contributors ordered front to
back,

```
out[ch] = sum_i c_i[ch] * w_i  +  T_final * bg[ch]
```

so

```
d out[ch] / d alpha_i = c_i[ch] * T_{i-1}
                        - (1 / (1 - alpha_i))
                          * ( sum_{k>i} c_k[ch] * w_k  +  T_final * bg[ch] )
```

The bracket is the whole remainder of the pixel behind primitive `i`, and
the background is behind everything, so it belongs INSIDE the bracket
rather than forming a term of its own.

The kernel reconstructs that bracket as `-ar[ch]`, initialised at the
32-stride checkpoint to `-pixel_colors + sampled_ar`. In this fork
`pixel_colors` is a straight `cudaMemcpy` of `out_color`
(`rasterizer_impl.cu`), and `out_color` already has `T_final * bg` folded
into it (`forward.cu`). So `-ar[ch]` was ALREADY the complete bracket —
and the kernel then added

```cuda
dL_dalpha += (-T_final / (1.0f - alpha)) * bg_dot_dpixel;
```

on top of it.

### Why the upstream comparison is the thing that settles it

That line is CORRECT upstream and wrong here, for a reason that is easy
to miss when diffing:

| tree | who writes `pixel_colors` | contents | separate bg term |
|---|---|---|---|
| taming-3dgs | its forward render kernel | `C[ch]` — **no background** | REQUIRED |
| this fork | a memcpy of `out_color` | `C[ch] + T*bg[ch]` | **double count** |

Upstream `forward.cu` writes both, on adjacent lines:

```cuda
pixel_colors[ch * H * W + pix_id] = C[ch];
out_color[ch * H * W + pix_id]    = C[ch] + T * bg_color[ch];
```

This fork's `forward.cu` writes only the second and never takes
`pixel_colors` at all. speedy-splat does not have the per-Gaussian kernel
and uses the original 3DGS `accum_rec` recursion, which is also
background-free and also needs the separate term. **Both references count
the background exactly once.** The repair therefore deletes the extra
term rather than changing what `pixel_colors` holds — one file, three
statements, no forward change.

A comment now states the divergence at the site, because the next person
to diff against upstream will otherwise see a missing line and "restore"
it.

### Measured, V100, one three-Gaussian scene, colour-only loss

Central differences of the CUDA forward against the analytic
`dL_dopacity`, white background:

| primitive | finite difference | background counted once | counted twice |
|---|---|---|---|
| 0 | −18.188 | within tolerance | −48.884 (**169% off**) |
| 1 | −13.660 | −12.992 (4.9%) | −36.419 (**167% off**) |
| 2 | −7.434 | −7.166 (3.6%) | −24.858 (**234% off**) |

`dL_dopacity` absmax over the scene: **17.900** correct vs **48.884**
double-counted; `dL_dmeans3D` 3.668 vs 10.017; `dL_dmeans2D` 6.026 vs
18.344. The defect entered the shared `dL_dalpha` accumulator, so it
reached opacity, the 2D mean, the conic and hence the densification
signal — not opacity alone.

### Blast radius on the project: NONE, and this is checkable

`bg . dL_dpixel` is identically zero for `bg == 0`. All 72 occurrences of
`white_background` under `configs/` are `False`, N3V and DiVa-360 are
black-background, and `gaussian_renderer/__init__.py` passes zeros
whenever an env map is used. The black-background finite differences are
unchanged between the two kernels (5.35% and 5.02% under both, to the
digit). **No recorded ADAGS result is affected by defect 1.**

## Defect 2 — the kernel gated itself on uninitialised device memory

### What was there

```cuda
// if first gaussian in bucket is useless, then others are also useless
if (bucket_idx_in_tile * 32 >= max_contrib[tile_id]) {
    return;
}
```

`max_contrib[` appears **exactly once in the entire rasterizer** — that
read. Nothing writes it. Upstream taming-3dgs fills it at the end of its
forward render kernel with a `cub::BlockReduce` max of `last_contributor`
stored by thread 0; that reduction does not exist in this fork's
`forward.cu`, which takes the pointer and threads it through unused.
`ImageState` is carved out of a `torch::empty` buffer and only
`imgState.ranges` is memset, so the guard compared against whatever was
in that memory.

When it was zero, `0 >= 0` held for **every** bucket, the kernel returned
before doing any work, and the ENTIRE backward produced exactly zero —
for every channel, every loss and every background, with no exception, no
shape error and no warning. When it was large the guard was a no-op. When
it was small and nonzero the gradient was silently TRUNCATED to the first
few buckets of each tile. Which of the three happened depended on
allocator history, not on the scene.

### Measured, V100, same scene and forward (radii `[7, 6, 6]` throughout)

| build | `max_contrib` guard | dL_dopacity | dL_dmeans3D | dL_dmeans2D |
|---|---|---|---|---|
| image's baked extension | present | **0.0** | **0.0** | **0.0** |
| background repaired | present | **0.0** | **0.0** | **0.0** |
| background repaired | removed | 17.900 | 3.668 | 6.026 |
| background double-counted | removed | 48.884 | 10.017 | 18.344 |

### This is the cause of the standing flow-lane blocker

[[rendered-flow-supervision-is-nonfunctional]] records, from experiments
132–136, that `TinyScene` "produces zero gradients for every loss —
colour included — on BOTH images, at every scene size, with a valid
bucket count and a correct forward", and states plainly that "the cause
is NOT yet identified". **It is identified here, and it is this guard.**
The page's other conclusions stand: the flow VJP patch was exonerated
correctly, and the harness really was returning zero — but for a reason
in the kernel's first twenty lines rather than anywhere in the flow path.

With the guard removed, `tests/test_flow_backward_vjp.py` runs 36 tests
and 33 pass. The remainder are NOT flow-VJP defects and are recorded as
open items below.

### The repair, and what it costs

The guard is deleted. Correctness never depended on it: it is a per-tile
early-out over a bound that the per-pixel test
`splat_idx_in_tile >= last_contributor` — taken from `n_contrib`, which
`forward.cu` DOES write — already enforces exactly. What is lost is the
optimisation: warps whose splats all sit behind every pixel's last
contributor now run their loop and do nothing.

**That cost is UNMEASURED.** No timing run was done. Restoring the
optimisation means computing `max_contrib` in the forward the way
upstream does, which is a separate change to a kernel with early-exit
control flow and was not attempted here.

## Consequence for prior runs — stated, not resolved

Every training run in this repository executed a backward whose per-tile
early-out read uninitialised memory. Its behaviour was not determined by
the scene, the config or the commit. It follows that **no prior run's
gradients are known to be reproducible**, and that a run could have
trained on silently truncated gradients without any diagnostic firing.

What is NOT concluded: that any specific recorded result is wrong.
Nothing here measures how often the garbage was benign, and training
evidently did make progress, which is consistent with the guard usually
being a no-op in a long-lived process that reuses the buffer. Establishing
more than that would need re-running, not reasoning.

## Open items, NOT repaired here

1. **Finite-difference residuals, all background-independent.** Measured
   on a BLACK background, where both defects above are provably inert:

   | quantity | analytic vs central difference |
   |---|---|
   | colour `dL_dopacity` | 2.2% / 5.4% / 5.0% |
   | projected geometry `dL_dmeans3D` | **46%** (4.167 against 2.238) |
   | flow-mediated `dL_dopacity` (pre-existing suite) | **35%** (1.175 against 1.587) |
   | flow-mediated geometry (pre-existing suite) | up to a **sign flip** (0.356 against −0.657) |

   The likely cause of the opacity residual is the `alpha < 1/255`
   contribution floor making the forward non-smooth in the perturbed
   opacity; for geometry, moving a mean also re-quantises the radius and
   the tile touch list. **Both are hypotheses and neither was tested.**
   Until one of them is settled, central differences cannot arbitrate a
   geometry gradient in this renderer at better than a factor of two.

   `tests/test_colour_background_vjp.py` therefore pins the white-vs-black
   COMPARISON rather than an absolute tolerance, so it cannot be passed by
   loosening a number, and it refuses outright when its black baseline is
   unreliable — which is exactly how the 46% figure above surfaced. It
   makes no finite-difference claim about geometry at all.
2. **`test_empty_scene_renders_zero_flow` errors** with "The specified
   pointer resides on host memory" at `P == 0`. Pre-existing, unrelated,
   untouched.
3. **The `min(0.99f, ...)` alpha clamp is still ignored** when forming
   `dL_dG`. Pre-existing, shared by every channel, and already disclosed
   in 6550772 and in [[rendered-flow-supervision-is-nonfunctional]].
4. **`bg_color`, `final_Ts` and `max_contrib` are now unused parameters**
   of `PerGaussianRenderCUDA`. They stay: all three are still consumed by
   the unlaunched `renderCUDA`, they are part of the shared
   `BACKWARD::render` signature, and removing them would ripple through
   `backward.h` and `rasterizer_impl.cu` for no functional gain.

## Provenance

Diagnosed statically against the upstream sources, decided numerically in
pure NumPy before any GPU was involved (the double count reproduces the
closed form `-T_final / (1 - alpha_i) * (bg . dL_dpixel)` to 1e-10), and
confirmed on Apollo `dgx` / Tesla V100-SXM2-32GB by compiling the kernel
four ways in a single job and importing each build ahead of the image's
baked extension via `PYTHONPATH`. Exploratory `det cmd run` cells, one
GPU slot, well under 0.5 slot-hours total; no run directory, no ledger
entry, `evidence_bearing: false` throughout. The image was NOT rebuilt or
republished: every variant was compiled inside the container from the
uploaded context, which is also why none of this required a new digest.
