# Fable 5.1 review of the CER Multi4D spec, rev 1 (2026-09-30)

Read-only subagent review against a clean clone of Multi4D at c483e82; condensed transcription.

| # | dimension | score | short reason |
|---|---|---|---|
| 1 | exactness | 8 | holds for a fixed model (no static opacity cull, per-pixel skip in forward and in the launched `PerGaussianRenderCUDA`, stable sort, residual rows appended at the end of the background block); missing: finite-opacity guard; no q = 0 training identity (dropout draw count changes) |
| 2 | file:line / facts | 8 | cited lines right; frozen-from-16,000 claim verified by a full trace; errors: "input slot 16" is the 16th argument; `fg_to_tr_end_iter` is a local; `cer_scene` does not reach `scene_reconstruction`; NaN `os.execv` restart unmentioned; ply reload drops the transient's temporal SH degree |
| 3 | shapes | 9 | background carries six groups plus `max_radii2D [N]`, `xyz_gradient_accum [N,1]`, `denom [N,1]`; base classes add none; concatenation foreground-then-background |
| 4 | bypasses | 7 | `getattr` fails open; frame parse must fail closed; load marker unspecified; `best/` sidecar; NaN restart |
| 5 | instrument | 8 | colour gradient is Σ αT dL/dpix; DC gradient is SH_C0·dL/dRGB for static and 4D sets; deformation touches only coefficient 16; two upstream hazards: `max_contrib` never written (`rasterizer_impl.cu:189`, read `backward.cu:1229`), `clamped` aliasing |
| 6 | minimality | 8 | `cer_gate.py` + state on the models is the smallest design; background-only donors right for exactness, wrong justification |
| 7 | tests | 6 | T3 heavy; T5 vacuous or failing (M1); T6 not a test; T7 equality can be vacuous; aggregate conservation blind to a dropped tile |
| 8 | CUDA | 9 | no change needed |

**Major.** M1 ply reload renders the transient bank without its temporal SH (`GM:1825` restores only `active_sh_degree`;
`active_sh_degree_t` stays 0, `GM:1266`; used at `R:402`). M2 `max_contrib` uninitialised: assert conservation on every
render inside `mass`. M3 fail-closed gating: a module-level "armed" flag; a strict frame parse that raises. M4 the
persistence marker and the plumbing of `cer_scene` into `scene_reconstruction`; write the sidecar outside the bare
`try/except` and for `best/` too. M5 background-only donors: the census does not bear on the donor question; forecast the
donor mass by bank on the memory-gated model. M6 NaN restart (`T:843-845`) re-runs the whole program: `begin` must refuse
if its report exists; the scorer should check `debug_metrics.json` has 40 lines. M7 carry the finite-opacity guard. M8
flag the dose difference.

**Minor.** 16th argument; `phase3_start_iter` for the lift precondition; upstream defects (transient degree-1 SH
gradient `backward.cu:191`, `clamped` aliasing, `T_3D` vs `T`) recorded as substrate facts; the reset oddity last fires at
14,000; position lr at 16,000 ≈ 4.0e-6·scale (max steps 20,000); `render_mask` also at `eval_utils.py:246`; T7 cannot
reach `best/`; cache the parsed frame; the live transient cull constant is the 0.05 at `forward.cu:282`.

**Test replacements proposed.** T3: reference = the same call with CER off after setting the switched-off rows' logits
to −inf (σ(−inf) = 0 exactly; opacity passes through the deformation unchanged), for all four functions and both probe
frames, plus capturing the rasterizer's opacity input to check `factor · ungated`. T6: static grep of row-changing calls
in `train.py`; a unit call of the three `maintain_*` at iterations ≥ 16,000 leaving state unchanged, with a negative
control at 15,900; a runtime sentinel installed by `begin`. Vacuity guards for T2, T4, T7.

**Dose.** CER stage ≈ 0.4× Ex4DGS's optimizer steps and ≈ 0.7–0.8× its gap-view exposures; keep 16,000 and pre-register
the asymmetry.

**Verdict: ready to implement after M1–M8.**
