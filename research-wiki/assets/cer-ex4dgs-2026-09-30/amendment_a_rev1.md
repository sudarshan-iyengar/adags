# CER on Ex4DGS, spec amendment (a): residual copies are verbatim copies in their donor's bank (2026-10-01, draft for review)

Amends [cer-ex4dgs-spec-2026-09-30](cer-ex4dgs-spec-2026-09-30.md) rev 2, §5.2 (residual birth) and the parts of §1, §3,
§7 that depend on it. Everything else in rev 2 stands. Motivation, measured (plan log 2026-10-01, job 59077300): on the
three pass-through scenes 239–369 of 243–371 donors are dynamic-bank rows carrying 0.36–0.75 of the gap-region mass,
and their largest move across the gap has median 0.27–0.34 and maximum 3.4–4.9 scene units (extent ≈ 6.5). Rev 2 copies
every donor into the static bank as a snapshot of its rendered state at frame 74, so a donor that moves during the gap
leaves a stationary copy. The principal chose option (a): copy each donor so the copy renders like the donor at every gap
frame.

## A1. The rule

Each donor (selected exactly as in rev 2 §5.2: majority of its own gap-frame mass inside the gap region, measured on the
memory-gated model) is copied **verbatim into its own bank**, labelled residual (2):

- **static donor → static bank:** the seven static parameters copied as they are (`_xyz`, `_xyz_disp`, `_features_dc`,
  `_features_rest`, `_scaling`, `_rotation`, `_opacity`), so the copy is the donor, displacement included, at every time
  (rev 2 copied a snapshot at frame 74 with zero displacement);
- **dynamic donor → dynamic bank:** the eight dynamic parameters copied as they are (`_xyz_motion` `[D_d, K, 3]`,
  `_features_dc_motion` `[D_d, 1, 3]`, `_features_rest_motion` `[D_d, 15, 3]`, `_scaling_motion` `[D_d, 3]`,
  `_opacity_motion` `[D_d, 1]`, `_opacity_duration_center` `[D_d, 2, 1]`, `_opacity_duration_var` `[D_d, 2, 1]`,
  `_rotation_motion` `[D_d, K, 4]`), so the copy follows the donor's trajectory, rotation and temporal opacity.

At birth, therefore, every copy renders exactly as its donor at every frame; with the factor, exactly as its donor at
every gap frame and not at all outside. This is the literal form of "a residual starts as its donor".

## A2. The one spec decision: a dynamic copy keeps its own temporal opacity

The gated opacity of a dynamic residual row is

```
o'_j(t) = q · n(t) · σ(ω_j) · b_j(t)          (b_j the copied plateau factor, utils/interpolations.py:55-61)
```

i.e. the general rule `o'_i(t) = o_i(t) · f_i(t)` of rev 2 §1 applied unchanged; nothing in the renderer changes. The
alternative, `q · n(t) · σ(ω_j)` (dropping `b_j` inside the gap), would make a copy render at the donor's peak opacity on
gap frames where the donor itself is fading in or out, so the copy would not start as its donor, and it would need a
second code path in `get_opacity_at_t`. **Recommended and specified: keep `b_j`.** Consequences, stated: a dynamic copy
can be partly transparent on some gap frames if its donor was; its temporal parameters (`_opacity_duration_center/var`)
receive rendering gradient only from gap frames; rev 2 §1's sentence "`o'_j(t) = q·n(t)·σ(ω_j)`, exactly the handoff's
form" now holds for static residual rows only.

## A3. Tensors and bookkeeping

Appended through `cat_tensors_to_optimizer` in two calls, the seven static groups for static donors and the eight
`motion_*` groups for dynamic donors (zero RAdam moments for the new rows). Non-parameter per-row tensors are extended by
the new rows only:

| bank | tensor | new value |
|---|---|---|
| static (as rev 2) | `xyz_gradient_accum`, `denom`, `xyz_error_accum`, `xyz_ssim_error_accum`, `error_denom` | 0 `[D_s, 1]` |
| | `xyz_error_min` / `xyz_error_min_timestamp` | 1000 / A `[D_s, 1]` |
| | `max_radii2D` / `min_radii2D` | 0 / 1000 `[D_s]` |
| dynamic (new) | `motion_xyz_gradient_accum`, `motion_denom`, `motion_xyz_error_mean`, `motion_xyz_ssim_error_accum`, `motion_error_denom` | 0 `[D_d, 1]` |
| | `motion_xyz_error_min` / `motion_xyz_error_min_timestamp` | 1000 / A `[D_d, 1]` (A ≥ 0 protects newborns from `prune_invisible`, which reads the motion timestamps, `c_gaussian_model.py:1074-1085`) |
| | `motion_max_radii2D` / `motion_min_radii2D` | 0 / 1000 `[D_d]` |

`_cer_static` gets `D_s` twos, `_cer_dynamic` gets `D_d` twos. `densification_postfix` is still not used (it resets the
accumulators of every row). The dynamic tensor list is the one `prune_points` masks (`:745-763`) and
`extract_dynamic_points_from_static` extends (`:1191-1225`); the test checks that each is extended by exactly `D_d`.

## A4. What changes in the exactness argument

Static residual rows sit between the static and dynamic blocks (as in rev 2); dynamic residual rows are appended at the
end of the dynamic block, i.e. at the end of the concatenated vector. In both cases every existing row keeps its
relative order, so the `q = 0` byte-identity argument of rev 2 §1 (stable sort, zero-opacity skip) is unchanged. The
residual is still exactly zero outside the gap (`n(t) = 0`), and the memory rows still exactly zero inside it.

After 30,000 the dynamic bank's lifecycle is what rev 2 §2.2 states for both banks: only `prune_invisible`,
`prune_small` and `prune_nan_points` run, through `prune_points`, which already masks `_cer_dynamic`. `expand_duration`
cannot change `K` (the window is at its cap, asserted in `begin`), so the copied keyframe tensors keep the bank's `K`.
The dynamic position learning rate is never scheduled (code map §1), so dynamic copies can still move; `motion_reg`
(`train.py:159-161`) acts on them like on any dynamic row.

## A5. Report and gates

`cer_begin.json` reports residual rows per bank (`residual_rows_static`, `residual_rows_dynamic`) and the donors' gap
mass per bank. Gates unchanged (the residual share counts both banks' residual rows).

## A6. Tests that change

- **T2** (`q = 0` identity): unchanged criterion; it now has residual rows in both banks (≥ 1 in total required, the per-
  bank counts reported).
- **T2b** (birth state): existing rows' parameters, per-row buffers and RAdam moments unchanged in **both** banks; the
  new rows' moments zero in both banks; every per-row tensor of each bank extended by exactly its `D`.
- **T2c (new): copies are verbatim.** For every residual row, each parameter equals its donor's parameter exactly
  (`torch.equal`), static and dynamic; and on a gap frame, the render with the residual rows on and their donors' logits
  set to −inf equals, byte for byte, the render with the residual rows off and the donors on (each copy paints exactly
  where its donor did) — reported, with the per-frame equality as the criterion on three gap frames.
- **T3, T4, T5, T6, T7**: unchanged; T6's prune alignment and T5's checkpoint round trip now cover dynamic residual rows.

## A7. Not changed

Donor selection, membership, the start iteration, the gate thresholds (residual 0.20 frozen), CUDA (no change),
persistence, the launcher, the scorer. **Multi4D is unaffected**: its donors are background rows, which have no time, so
its copies were already verbatim.
