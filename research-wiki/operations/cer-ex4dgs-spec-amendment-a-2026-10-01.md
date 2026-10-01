# CER on Ex4DGS, spec amendment (a): residual copies are verbatim copies in their donor's bank (2026-10-01, rev 2)

Amends [cer-ex4dgs-spec-2026-09-30](cer-ex4dgs-spec-2026-09-30.md) rev 2, §5.2 (residual birth) and the parts of §1, §3,
§7 that depend on it. Everything else in rev 2 stands. **Rev 1 of this amendment** (archived as
[`assets/cer-ex4dgs-2026-09-30/amendment_a_rev1.md`](../assets/cer-ex4dgs-2026-09-30/amendment_a_rev1.md)) was reviewed by
GPT-6.1 Sol (material defect, in the tests) and Fable 5.1 (ready after fixes); both reviews are beside it. Rev 2 absorbs
them (A8). Line numbers are the `cer` worktree's at f71c1b0.

**Motivation, measured** (plan log 2026-10-01, jobs 59077300 and 59078436, wave-1 base run 1 at 30k): on all nine scenes
almost every donor is a dynamic-bank row (189–398 dynamic against 2–36 static), and the dynamic donors move during the
gap: median of each donor's largest move 0.26–0.52 scene units, maximum 1.8–4.9 (scene extent ≈ 6.5); largest on the
pass-through scenes (3.4–4.9) but present on same-pose (1.8–2.7) and moved-return (2.2–3.1) too. Rev 2 copies every donor
into the static bank as a snapshot at frame 74, so a moving donor leaves a stationary copy. The principal's choice,
option (a): copy each donor so the copy is the donor at every gap frame.

## A1. The rule

Donors are selected exactly as in rev 2 §5.2 (majority of their own gap-frame mass inside the gap region, measured on the
memory-gated model). The donor indices, which index the concatenated static-then-dynamic vector, are split by bank
**with the static row count `n_s` taken before any row is appended**: `ds = donors[donors < n_s]`,
`dd = donors[donors >= n_s] − n_s`. Each donor is copied **verbatim into its own bank** and labelled residual (2):

- **static donor → static bank:** the seven static parameters as they are (`_xyz`, `_xyz_disp`, `_features_dc`,
  `_features_rest`, `_scaling`, `_rotation`, `_opacity`);
- **dynamic donor → dynamic bank:** the eight dynamic parameters as they are (`_xyz_motion` `[D_d, K, 3]`,
  `_features_dc_motion` `[D_d, 1, 3]`, `_features_rest_motion` `[D_d, 15, 3]`, `_scaling_motion` `[D_d, 3]`,
  `_opacity_motion` `[D_d, 1]`, `_opacity_duration_center` `[D_d, 2, 1]`, `_opacity_duration_var` `[D_d, 2, 1]`,
  `_rotation_motion` `[D_d, K, 4]`); `K` is inherited by slicing.

So each copy's **per-row** rendering inputs (position, rotation, scale, colour and evaluated opacity at every time) equal
its donor's bit for bit; gated, the copy contributes only on gap frames. The composite is not the donor's alone: at
birth donor and copy both render in the gap (the doubled layer, rev 2 §10). The indices `ds` and `dd` are recorded in
`cer_begin.json` (a few hundred integers), because the tests cannot recompute them after birth (the copies change the
gated masses).

## A2. The one spec decision: a dynamic copy keeps its own temporal opacity

The gated opacity of a dynamic residual row is the general rule of rev 2 §1, `o'_j(t) = o_j(t) · f_j(t)`, with
`o_j(t) = σ(ω_j) · b_j(t)` the copied plateau factor (`utils/interpolations.py:55-61`):

```
o'_j(t) = q · n(t) · σ(ω_j) · b_j(t)
```

Nothing in the renderer changes. **Recommended and specified: keep `b_j`.** Dropping it inside the gap would make a copy
render at the donor's peak opacity on gap frames where the donor itself is fading in or out, so the copy would not start
as its donor. (It would not need a second code path: a choice of `_opacity_duration_center/var` could make `b ≈ 1`; the
decision rests on "starts as its donor".) Consequences:

- a dynamic copy is partly transparent on the gap frames where its donor was;
- its temporal parameters receive rendering gradient only from gap frames, and **none at all** while the gap lies
  inside its plateau, where `time_bigaussian` returns exactly 1 (`interpolations.py:61`), so such a copy's window stays
  the donor's; `var` gradients that are NaN (for `var` large enough that `exp` overflows) are zeroed by `train.py:248-250`
  for copies as for donors;
- rev 2 §1's sentence "`o'_j(t) = q·n(t)·σ(ω_j)`" now holds for static residual rows only.

## A3. Tensors and bookkeeping

One `cat_tensors_to_optimizer` call (`c_gaussian_model.py:796-818`) with the seven static and eight `motion_*` group
names, each the donors' rows (`ds` for static groups, `dd` for dynamic ones; an empty index in a bank appends zero rows),
as `densification_postfix` does (`:824-840`); the fifteen returned parameters are rebound. Zero RAdam moments for the new
rows. Non-parameter per-row tensors are extended by the new rows only:

| bank | tensor | new value |
|---|---|---|
| static | `xyz_gradient_accum`, `denom`, `xyz_error_accum`, `xyz_ssim_error_accum`, `error_denom` | 0 `[D_s, 1]` |
| | `xyz_error_min` / `xyz_error_min_timestamp` | 1000 / A `[D_s, 1]` |
| | `max_radii2D` / `min_radii2D` | 0 / 1000 `[D_s]` |
| dynamic | `motion_xyz_gradient_accum`, `motion_denom`, `motion_xyz_error_mean`, `motion_xyz_ssim_error_accum`, `motion_error_denom` | 0 `[D_d, 1]` |
| | `motion_xyz_error_min` / `motion_xyz_error_min_timestamp` | 1000 / A `[D_d, 1]` (A ≥ 0 protects newborns from `prune_invisible`, which reads the motion timestamps) |
| | `motion_max_radii2D` / `motion_min_radii2D` | 0 / 1000 `[D_d]` |

`_cer_static` gets `D_s` twos and `_cer_dynamic` `D_d` twos. `densification_postfix` itself is still not used (it resets
the accumulators of every row). The rev-2 snapshot machinery (`t_m`, the getters at `t_m`, `inverse_sigmoid`) is removed.

## A4. Exactness, lifecycle and preconditions

- **q = 0 identity.** Static copies sit between the static and dynamic blocks, dynamic copies at the end of the dynamic
  block, i.e. the end of the concatenated vector; every existing row keeps its relative order, so rev 2 §1's argument
  (stable sort, zero-opacity skip, `last_contributor`) is unchanged. The residual is exactly zero outside the gap and the
  memory rows exactly zero inside it.
- **Lifecycle after 30,000** (traced by both reviewers with the config): only `prune_invisible`, `prune_small` and
  `prune_nan_points` change rows, through `prune_points`, which masks both label vectors. `adjust_temp_opa` stops before
  27,000; promotion cannot fire (asserted); `expand_duration` is a no-op once the duration is at its cap. **New assertion
  in `begin`:** `g.duration == dataset.duration + g.time_shift + 1` (313 on these scenes), the condition under which
  `expand_duration` returns without changing `K`. The dynamic position learning rate is never scheduled, so dynamic
  copies move at full rate; `motion_reg` (`train.py:161-163`) and `static_reg` reach copies, so their means change too:
  rev 2's caveat that CER training is not base training even at `q = 0` now covers both regularisers.
- **Finite opacity at birth.** Verbatim copies inherit a donor's evaluated opacity, so `begin` calls
  `check_finite(g, range(dataset.duration))` before selecting donors (as `gates` does at the end).

## A5. Report and gates

`cer_begin.json` adds `residual_rows_static`, `residual_rows_dynamic`, the donors' gap mass per bank, and the indices
`ds`, `dd`. `cer_gates.json` adds the residual rows and residual mass per bank. Gate thresholds unchanged (residual 0.20
frozen; both banks' residual rows count).

## A6. Tests that change

- **T2** (`q = 0` identity, unchanged criterion) on the bottle same-pose model, which has donors in both banks
  (19 static, 351 dynamic on run 1); the per-bank residual counts must both be ≥ 1.
- **T2b** (birth state): banks identified by capture-tuple position (static per-row entries 1–15 and 18 (`_xyz_disp`),
  dynamic 23–39 of the 40-tuple, `c_gaussian_model.py:222-264`), each extended by exactly its bank's `D` with the existing prefix unchanged;
  both banks' existing RAdam moments unchanged and new-row moments zero.
- **T2c (new): copies are their donors.** With the recorded `ds`, `dd`: (i) every parameter of every copy equals its
  donor's (`torch.equal`), both banks; (ii) on three gap frames, the evaluated per-row inputs of copy and donor are equal
  (`get_xyz_at_t`, `get_rotation_at_t`, the activated scaling, `get_features`, and the ungated `get_opacity_at_t`);
  (iii) the birth step run on its own with donor sets chosen to be static-only, dynamic-only and mixed, checking (i) and
  the per-bank counts each time (so the bank split is exercised in all three cases, not only the one a scene happens to
  produce). **Diagnostic, not a criterion:** on a gap frame, the render with the copies on and their donors' logits at
  −inf against the render with the copies off (relabelled memory temporarily, factor 0) and donors on, reported as
  differing pixels and max |diff|: a copy sorts at the end of its equal-depth group where its donor sorted at the donor's
  index, so exact ties can change the compositing order.
- **T3, T4, T5, T6, T7**: unchanged criteria. T6's random prune and T5's round trip now include dynamic residual rows (the
  per-bank residual counts are reported, both ≥ 1 required).

## A7. Not changed

Donor selection, membership, the start iteration, the gate thresholds, CUDA (no change), persistence, the launcher, the
scorer. Multi4D is unaffected by this amendment: its donors are background rows, which have no time, so its copies were
already verbatim. (Separately, the Multi4D forecast shows the transient bank paints 96% of the gap region there; that is
reported to the principal on its own.)

## A8. Changes from amendment rev 1

| change | source |
|---|---|
| "starts as its donor" tested as equality of parameters and evaluated per-row inputs; the swap render is a diagnostic (sort-order ties) | Sol 1, Fable F1 |
| per-bank residual counts ≥ 1 required; synthetic static-only / dynamic-only / mixed births | Sol 2, Sol (missing), Fable F2 |
| donor indices split with the pre-birth `n_s`; recorded in `cer_begin.json` | Fable F2, Sol (missing) |
| the duration-cap claim replaced by a real assertion | Sol 3, Fable F3 |
| `check_finite` at birth | Sol (suspected), Fable F4 |
| A2: rationale reworded; zero gradient inside the plateau stated; `motion_reg` caveat | Fable F5, F6, Sol (missing) |
| one `cat` call; snapshot machinery removed | Fable F7 |
| T2b by capture-tuple position | Fable F8 |
