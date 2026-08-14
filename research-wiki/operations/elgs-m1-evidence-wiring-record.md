# EL-GS M1 — Evidence-Stack Wiring Record

Date: 2026-08-14. Status: **IN REPAIR after two REJECTED integration
reviews.** Authority: user directive 2026-08-14 (wire the evidence stack;
repair once and re-review; a second rejection is a stop condition) ->
second rejection reached -> user authorized a THIRD repair cycle,
overriding the stop condition, with these corrections recorded first.

M0 disclosed the evidence stack as "implemented + unit-tested but unwired
pending the M1 track artifacts"
([[operations/elgs-m0-implementation-record]]). This page records the
wiring attempt, both review verdicts, and — separately — the claims this
author made that turned out to be FALSE. Corrections are append-only;
nothing earlier on this page is rewritten.

**NO GPU SMOKE HAS RUN. NO EVIDENCE-BEARING RUN HAS CONSUMED THIS PATH.**
The smoke was gated behind a passing integration review, which has not
been obtained. Nothing here is claim-grade and no result depends on it.

## Where execution became photometric-only (measured, not inferred)

`elgs_beta`, `elgs_tau_b`, `elgs_c_cap`, `elgs_binding_threshold`,
`elgs_r_site`, `elgs_kappa`, `elgs_chi`, `elgs_mu` were declared on the
argparse surface (`arguments/__init__.py:236-248`) and consumed NOWHERE.
`elgs_tracks_dir` appeared only inside a comment
(`elgs/trainer_hooks.py:296`). `run_pass` was called with
`exact_deltas_fn` defaulted to `None`, so `elgs.acceptance.decide`
received `exact_deltas = 0.0` on every candidate. That is the whole
boundary: the machinery existed, nothing called it.

The correct entry point is `exact_deltas`, fixed by `elgs/acceptance.py`'s
own contract — "'Exact' survives only for the non-sampled closed-form
tracker/prior deltas added outside the sampled render estimate". Phi is
such a term. Both reviews confirmed this placement is right; it is not
among the defects.

## Review 1 — REJECTED (four must-fix findings, all real)

1. **PIXEL DOMAIN.** Reports live in the converted scene's FULL raster
   (`scripts/build_elgs_tracks.py` bounds them by the transforms'
   declared `w`/`h`, 1160x550 for DiVa-360). The trainer may load that
   scene downscaled: `utils/camera_utils.py::loadCam` divides every
   intrinsic by `scale`, and `configs/elgs/smoke_elgs.yaml` sets
   `resolution: 4`. Projected bridge centres were therefore 4x smaller
   than the report positions compared against them, `g_pos` underflowed
   to zero for EVERY report, and the evidence term acquired a **constant
   sign favouring truncation**. It would have read as "the tracker
   evidence says carve these families" and been an axis-scale artifact.
   Secondary: `math.log(l1)` raises `ValueError`, which is not a
   `ContractError`, so it escaped `elgs/round_driver.py`'s rejection
   handling and would have killed the job mid-round.
2. **PROBE CACHE.** Keyed `(camera, frame, present_family)` — a key space
   of |cameras| x |frames| x |families|, hundreds of GB at the S1 scale
   M0 already ran at (111 families / 50k rows / 26 cameras).
3. **HEADS GATE DID NOT GUARD WHAT IT CLAIMED.** `elgs_smoke_schedule` is
   a SCHEDULE switch. `scripts/submit_apollo.py:313` derives
   `evidence_bearing` purely from working-tree cleanliness and never
   reads it. A clean-tree run stamped `evidence_bearing: true` could
   therefore have loaded unfrozen, hand-picked head constants.
4. **WINDOW ENDPOINTS.** `windows_between_anchors` builds
   `Window(earlier.end_frame, later.start_frame)`, so both endpoints ARE
   anchor frames and the window's bridges are fitted at exactly those
   frames. Inclusive endpoints scored anchor reports against a bridge
   fitted to them, and double-counted every frame shared by two windows
   when an anchor is a single frame.

Repaired at `15b5cee`. Findings 3 and 4 were confirmed correctly fixed by
review 2 and are closed.

## Review 2 — REJECTED (the repair introduced a defect and rested on a false invariant)

Both decisive findings were INDEPENDENTLY VERIFIED against source by this
author before being recorded here.

- **F1 — a NEW defect, introduced by the repair.** The repair added
  `get_marginal_t` to the probe's opacity, justified by the prereg's
  "exact w.r.t. the rasterizer's compositing model". **That justification
  is factually wrong.** `gaussian_renderer/__init__.py:232` and `:249`:
  under `elgs_active`, `marginal_t = pc.get_elgs_presence(timestamp)` —
  EL-GS presence **REPLACES** the temporal marginal and is never
  multiplied by it. The renderer's alpha is
  `get_opacity * dynamic_probability * elgs_presence`
  (`gaussian_renderer/__init__.py:209` supplies the
  `dynamic_probability` factor). The probe computed
  `get_opacity * get_marginal_t * presence`: a factor the renderer does
  not use under EL-GS, and missing one it does.
  **Direction of bias: q systematically TOO HIGH.** `get_marginal_t`
  decays to ~0 for temporally distant Gaussians, so occluders vanish, T
  collapses toward 1, and the evidence is treated as fully informative
  exactly where the model says the query point is occluded — the failure
  censored evidence exists to prevent. No test reached the branch: the
  test stub defines no `get_marginal_t`, so `getattr` returned `None`.
- **F2 — the cache repair rests on a FALSE invariant.** Its docstring
  argues "`get_xyz` is the canonical position and the rig is static, so
  screen geometry does not vary with frame". `scene/gaussian_model.py:816`
  `get_dynamic_xyz` returns
  `_xyz + get_lora_motion_offset(t) + get_scaffold_motion_offset(t)` for
  `gaussian_dim == 4` and `motion_model == "lora"` and `not rot_4d` —
  exactly the smoke configuration — and
  `gaussian_renderer/__init__.py:200` calls THAT, not `get_xyz`. Screen
  geometry IS frame-dependent. This was equally wrong at `cededf5`, so it
  is not a numerical regression, but the repair wrote the falsehood into
  the docstring as the cache's justification.
- **F3 — persisted binding is WRITE-ONLY.** `EvidenceContext.to_state`
  emits the binding, but `setup_elgs` never reads `loaded["evidence"]`
  and `attach_evidence` re-derives bindings against current, drifted
  positions on every resume. The reproducibility problem review 1 raised
  is NOT fixed.
- **F4 — the pixel-domain guard is FAIL-OPEN.** When the report raster
  cannot be determined, `evidence_pixel_scale` defaults to `1.0` AND the
  D_img check is skipped — silently restoring the exact condition that
  caused review 1's rejection.
- **F6** — the non-positive-likelihood guard converts a REACHABLE and
  MEANINGFUL state (report far from bridge => presence strongly
  disfavoured) into a fatal job kill. A likelihood floor is the
  principled handling; the model already declares `h_floor`, `pi_floor`,
  `g_cap`, `pos_cap`.
- **F7** — `elgs/probe_model.py` `__all__` still exports the renamed
  `ProjectedSplats`.

## CORRECTIONS to this author's own claims (recorded, not silently fixed)

The commit messages of `cededf5` and `15b5cee` are preserved unchanged as
history. These statements in them are WRONG:

1. `15b5cee`: *"gaussian_dim=4 composites get_marginal_t"* — presented as
   a faithfulness repair. It is a DEFECT (F1 above); the renderer does
   not do this under EL-GS.
2. `15b5cee`: *"the evidence binding table is persisted ... so a resume
   restores the bindings the run actually used"* — the write exists; the
   RESTORE does not (F3).
3. `15b5cee`: *"the previous parity test built its oracle by calling the
   implementation under test"* implies replacement. The self-referential
   test was NOT replaced; an independent one was added ALONGSIDE it and
   the original survives.
4. `cededf5`: the disclosed query-source-exclusion reading claimed the
   strict-front test is the CONSERVATIVE direction. Review 1 showed the
   claim is at best half true — `FOOTPRINT_CUTOFF` drops occluders and so
   pushes q the OTHER way, with a per-splat bound whose aggregate grows
   with the kept-set size. **The net direction of the q bias is NOT
   established and must not be asserted before the q-tilde distribution
   is measured.**

## What IS established

- **Apollo runs CPU cells at ZERO GPU slots.** `det cmd run --config
  resources.slots=0` verified live (task `a65f6001`, ~6 s). `AGENTS.md`'s
  "`slots_per_trial: 1` means every cell occupies a GPU slot" describes
  the EXPERIMENT template `det_exp_apollo.yaml` only; command configs
  (`det_cfg_apollo_ctx.yaml`) use `resources.slots` and accept 0. The
  full CPU unit suite now runs there in ~30 s at zero GPU cost.
- **Test state at `15b5cee`:** 851 tests, 2 errors, identical to the
  801-test baseline's 2 pre-existing errors (`$WORK` unset, absent
  refine-logs history). Measured on Apollo, not asserted.
- **Reviews 1 and 2 both confirm Phi's placement in `exact_deltas` is
  correct**, and that no census `v >= 0.5` existence semantics were
  imported into the trainer: the threshold appears only in the d_u
  plateau and the anchor plateau, while `_reports_in_window` passes raw
  `v` into the likelihood as a continuous value.
- **The heads gate is real.** `prereg_evidence_heads_v1.json` still marks
  `g_v`, `h_c`, `h_o`, `pi_miss`, `g_pos_sigma` and `reliability.r_u`
  unfrozen, so an evidence-bearing run cannot reach the evidence path.
  `anchor_report_floor` has NO prereg home at all and needs one before B2.

## Correction to the absence-diagnostic sequence set

The 12 sequences the completed absence diagnostic scored are those with
>= 1 true-absence window in
`configs/elgs/prereg_m1_absence_diagnostic_v1.json`
`disclosed_prior_knowledge.known_per_sequence_screened_half`: scissor 343,
poker 109, pour_tea 73, tambourine 18, put_candy 18, tea 13, pan 11,
put_fruit 4, slice_apple 4, **writing_2 2**, maracas 1, soda 1 = **597**.

**`xylophone` is NOT one of the 12** — it carries ZERO true-absence
windows. Its `_fix79ae5b7` artifact corrected its CENSUS COVERAGE
(0.577 -> 0.7787); it was never a diagnostic input. Only writing_2's
corrected artifact is. Any follow-up diagnostic reusing "the same 12
sequences, 597 windows" must use this list.

## chess_long acquisition — BLOCKED, and why

There is **no reproducible acquisition path in the repository**: no
download script, no URL, no share token, tracked or untracked. The
recorded tranche-1 procedure was MANUAL link collection through the
DiVa-360 Dropbox browser UI ("direct folder-path URL guesses serve the JS
app shell", [[operations/elgs-cycle2-screening-record]]), after which
detached Determined CPU tasks fetched the per-file links. Apollo holds 25
zips and `MANIFEST.sha256` has exactly 25 lines; `chess_long` is absent.
`scripts/diva360_to_blender.py` hard-requires a MANIFEST entry, so a new
zip must be hashed and appended before conversion will run.

The pilot cannot proceed without a user-supplied link. The remaining-seven
estimate is therefore UNCHANGED and still derived, not measured:
~50-70 GPU-h and ~6-8 TB for the 8 long sequences, explicitly an
order-of-magnitude bound ([[operations/elgs-exhaustive-screen-scope]] §4).

## Open at the time of writing

Third repair cycle authorized by the user. Targets: point the probe at
`get_dynamic_xyz(t)` and at `get_elgs_presence * dynamic_probability`
(dropping `get_marginal_t`), re-key the geometry cache per
`(camera, frame)`, make the raster determination fail CLOSED, add a
likelihood floor instead of a fatal raise, implement the binding restore,
fix `__all__`, and add tests that bind the probe to the renderer's actual
composition rather than to a stub that cannot exercise it.
