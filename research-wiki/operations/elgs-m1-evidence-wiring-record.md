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

## Exploratory training authorization — 2026-08-14 (recorded BEFORE execution)

User directive 2026-08-14, after three rejected reviews: the patch/review
loop has become the wrong instrument for the question that matters, and
an end-to-end exploratory run is more informative now than another broad
static review. **Authorized: ~15 GPU-hours, staged, for EXPLORATORY
training only.** Separate from, and not additive to, the chess_long
pilot's 12 GPU-h.

**The decisive fact behind this decision: the evidence path has NEVER
executed.** Defect B1 was a `NameError` on an unbound local — it would
have died in seconds of runtime, yet it survived three review cycles and
56 targeted tests, because every test drives the probe through stubs and
`configs/elgs/smoke_elgs.yaml` sets no `elgs_tracks_dir`. Review 3 stated
plainly that a smoke on that config "would exercise none of this commit".

Division of labour recorded for future phases:
- **Runtime catches plumbing** (unbound names, contract arity, config
  refusals, unit mismatches that raise). B1, B2, the K=0 interval
  contract and two lambda-arity bugs are all of this class and all die
  instantly at runtime.
- **Static review catches silent bias** (M1 pixel domain, F1
  `get_marginal_t`, F-A the static routing branch). These produce
  PLAUSIBLE NUMBERS, so no run flags them. This is where review earned
  its keep and where it must be retained.

**Evidence boundary for everything under this authorization: EXPLORATORY,
NOT claim-grade.** Runs are stamped `evidence_bearing: false` via
`--dirty-smoke`, which is also what permits the smoke-tier heads (the
frozen heads remain unfrozen, so no evidence-bearing run can reach this
path). No output may be cited as a result, a baseline, or a supply claim.
`chess` is deliberately the substrate: it is already screened (0
true-absence, coverage 0.9362) and therefore non-gate-bearing.

**Sequencing (user-directed): photometric baseline FIRST.** DiVa-360 has
never been trained in this codebase — every M0 smoke ran on N3V
`cut_roasted_beef` (best-val PSNR 28.24). Its `points3d.ply` is
SYNTHESIZED by the converter from a coarse frustum box explicitly
disclosed as "a coarse smoke-test volume, NOT a claim-grade
initialization", and its scale against this rig is unverified. If the
scene does not train sanely with `elgs_enable` off, every evidence
number is noise and any hyperparameter tuned on it is tuned on noise.

**No public hyperparameters exist to look up.** DiVa-360 was selected
precisely because it is "the event-dense benchmark with no GS baselines"
([[gap_map]] Loop-2 update). The only anchor is the N3V smoke.

Still outstanding and NOT waived by this authorization — they block any
evidence-bearing use, not exploratory runs: the prereg-mandated 1e-6
parity fixture is still self-referential; no test exercises
`transmittance` at `pixel_scale != 1`; `p_floor` has no fire-counter; the
geometry-cache eviction is uninstrumented.

## Exploratory Apollo results — 2026-08-15 (EXPLORATORY, never claim-grade)

All runs `evidence_bearing: false` via `--exploratory`, dgx/V100, smoke
heads unchanged and NOT tuned toward any outcome. Substrate throughout:
`scissor_screen_w0_561` + its sealed cotracker3 tracks.

**Exp 71 — DiVa-360 photometric baseline: PASSED.** The Blender loader
accepts the conversion unchanged ("Found transforms_train.json file").
3000 iterations in 8:48, best val PSNR 19.65 / SSIM 0.855, still
improving at the final iteration. Points 20,000 -> pruned to ~6,300 by
iter 1500 -> 10,806: the synthesized `points3d.ply` spans +/-6.5 against
scissor content at +/-1.2, so most of the initial cloud is empty space.
Routing collapsed near-fully dynamic (mean_dynamic_prob 0.976).

**Exp 73 — evidence path ACTIVATED.** Tracks loaded and bound: 512
seeds / 10,995 tracks, 512 clusters, **334 bound to families**, tier
smoke, `frame_dt` 1/120. All three rounds (200/350/500) ran; every
candidate rejected; rollback clean; `committed_decisions` 0 so the §8
post-refit pass correctly skipped. **But q_values 0 and windows 0 in
every round**: the M0 smoke proposer ranked families by interval span,
which is IDENTICAL for every K=1 spanning family, so selection was
evidence-blind and landed on a family with no bound cluster. The real
q/likelihood/Phi path was never entered.

**Exp 74 — checkpoint/resume: PASSED.** `rounds_run [200,350,500]` not
repeated, slot-grid `consumed [0,2,4]` not redrawn, 334/512 bindings
RESTORED (`elgs_evidence_binding_restored`) rather than recomputed,
a-logit moments reset as disclosed, training continued to 700 without
divergence.

**Exp 75 — crashed at the FIRST real q evaluation.** `setup_elgs`'s
restore branch built `ElgsRuntime` on the model device while the fresh
branch used the default (cpu), so presence and projected geometry lived
on different devices and `alpha[front]` raised "indices should be either
on cpu or on the same device as the indexed tensor". The asymmetry was
present from the start; exp 73 never reached `transmittance` because it
had no windows, so only the evidence-aware proposer could expose it.
Fixed with device-agnostic regression tests on both paths.

**Exp 76 — the evidence path reached q, and the COST is the result.**
Cancelled deliberately, not failed. Measured exactly on CPU for the
family the proposer selects (219):

| window | interior frames | reports |
|---|---:|---:|
| (45, 47) | 1 | 292 |
| (47, 512) | **464** | **135,488** |

135,780 reports x 3 bridges = **407,340 q values**, each with 7 sigma
points = **2,851,380 full-model transmittance passes for ONE round**.
That is hours of GPU time per round. The round was progressing, not
hung.

**NEW MEASURED KNOWLEDGE.** The §4 cap operator caps CAMERAS per
(bridge, track, frame); nothing caps FRAMES per window or reports per
family. A family whose two anchors sit at opposite ends of the sequence
produces a 464-frame "window" that is not a gap in any useful sense, and
its evidence cost is quadratic in the wrong thing. `q_values` per round
must be bounded and logged before any evidence-bearing use, and the
bound must be a preregistered constant, not an implementation detail.

**Still NOT demonstrated:** that q, the likelihood terms and the
evidence delta are finite and data-dependent. Exp 73 could not show it
(no windows); exp 76 could not finish. This remains the open claim.

Selection change recorded: `_propose_smoke_candidates` now prefers a
family bound to a cluster AND carrying >= 1 evidence window, using
window availability and family ids ONLY — never a likelihood, a q value,
an evidence-delta sign, or an acceptance outcome. SMOKE TIER ONLY;
production candidate generation is untouched.

## The real q/likelihood/Phi path HAS EXECUTED — exp 78, 2026-08-15

Determined experiment **78**, commit `e87e841`, dgx/V100, run manifest
`evidence_bearing: false`, config hash `503a8d97`, image
`apollo-v100-v1`, `STATE_COMPLETED`. EXPLORATORY; nothing below is
scientific.

```
elgs_evidence_round: windows 2, families_with_windows 1, q_values 576,
  reports {total 135780, retained 192, dropped 135588,
           frames_covered 97, cameras_covered 23,
           max_reports_per_window 96, selection_rule "smoke-only: ..."}
elgs_round: iteration 200, proposals 1,
  committed ["FISSION:219:a5395116"], rejected []
```

Checkpoint at 220: `rounds_run [200]`, `committed_decisions 1`
(FISSION on family 219, n_samples 8, **se 0.0**), evidence tier smoke,
334/512 bindings, 512 families.

**What this establishes.** q was evaluated 576 times and every value
passed `QSnapshot.put`'s `0 <= q <= 1` check, so all 576 are finite and
in range — a NaN raises there. The likelihood terms were evaluated
under the declared `p_floor`; a non-positive likelihood raises
`ContractError` in `stream_log_likelihoods`. `elgs.acceptance.decide`
refuses a non-finite `exact_deltas`, so the COMMIT proves the evidence
delta was finite. With `se = 0.0` the acceptance rule reduces to
`delta_total < 0`, so the total is strictly negative and finite. Phi was
computed from **192 real retained reports spanning 97 frames and 23
cameras**, not from a fallback: `families_with_windows` is 1, not 0, and
`q_values` is 576, not 0. Acceptance completed and committed without
corrupting state (registry intact at 512 families, peak_scalars grew
1400898 -> 1400900 as the fission's a-logit dimension requires).

**NOT established, and not claimed.** That Phi is nonzero. `se = 0.0`
means the eight confirmation units gave an identical paired render
delta, so the photometric arm carried no discriminating signal for this
candidate; the decision therefore rested on a strictly negative finite
`delta_render + exact_deltas`, but the two are not separable from the
artifacts. No smoke head was tuned toward any outcome, and Phi was not
forced.

**MEASURED q THROUGHPUT.** The round occupied 05:04 -> 06:08 of the
tqdm clock = **64 s**, of which ~6 s was acceptance (paired renders +
Phi). So ~58 s for 576 q values ~= **10 q/s**, i.e. ~70 full-model
transmittance passes per second over 20,000 Gaussians (each q costs 7
sigma points). Extrapolated to the unbounded window that experiment 76
attempted (407,340 q values): **~11.3 hours for ONE round**, which is
consistent with exp 76 still computing after 30 minutes.

Two device defects were found and fixed on the way, each reachable only
because the previous fix let execution get further: the fresh-run
runtime sat on CPU while the model was on CUDA (exp 75), and
planner-built candidate intervals stayed on CPU once the runtime moved
to CUDA (exp 77, in the paired candidate render). Both carry
device-agnostic regression tests.

## Open at the time of writing

Third repair cycle authorized by the user. Targets: point the probe at
`get_dynamic_xyz(t)` and at `get_elgs_presence * dynamic_probability`
(dropping `get_marginal_t`), re-key the geometry cache per
`(camera, frame)`, make the raster determination fail CLOSED, add a
likelihood floor instead of a fatal raise, implement the binding restore,
fix `__all__`, and add tests that bind the probe to the renderer's actual
composition rather than to a stub that cannot exercise it.

## Photometric continuation to 10,000 iterations — exp 79, 2026-08-16

EXPLORATORY (`evidence_bearing: false`), dgx/V100, commit `d47754e`,
cell `diva360_scissor_photo_c10k`, run dir
`runs/elgs/20260815T233850Z_diva360_scissor_photo_c10k_0_d47754e`,
audited (`terminal.json` written).

Exp 71's `best_val_iter == 3000` was cited here as "still improving at
the final iteration". **CORRECTION: that inference was not available
from exp 71.** Its `test_iterations` normalized to a SINGLE validation,
at iteration 3000, so `best_val_iter == 3000` was true by having nothing
to compare against. The continuation validates every 1000 iterations —
measurement cadence, not tuning — and answers the question properly.

Resumed exp 71's `chkpnt3000.pth`; every photometric setting identical
(scene, resolution 4, initialization, `elgs_enable: False`, learning
rates, `densify_until_iter: 2000`, `opacity_reset_interval: 30000`,
lora routing). `position_lr_max_steps` is 30,000 and independent of
`iterations`, so the run continues along exp 71's own decay rather than
rescaling it. Densification is behind the resumed `first_iter`, so no
densification ran and the point count held at **10,806 for all 7,000
iterations** — the capacity is exactly exp 71's.

| iteration | val PSNR | val SSIM | delta PSNR |
|---:|---:|---:|---:|
| 3000 (exp 71) | 19.6523 | 0.85525 | — |
| 4000 | 19.9689 | 0.86027 | +0.3166 |
| 5000 | 20.1636 | 0.86310 | +0.1946 |
| 6000 | 20.3671 | 0.86399 | +0.2035 |
| 7000 | 20.4459 | 0.86527 | +0.0788 |
| 8000 | 20.5758 | 0.86586 | +0.1299 |
| 9000 | 20.6567 | 0.86591 | +0.0809 |
| **10000** | **20.6983** | **0.86714** | +0.0416 |

**Still improving, and decelerating.** `best_val_iter == 10000` is now a
real statement: validation rose monotonically at every one of the seven
checks. But the per-1000-iteration gain fell from +0.32 dB to +0.04 dB,
and 3000->6000 bought +0.715 dB against 7000->10000's +0.252 dB. A
further extension buys progressively less; the curve has not turned
over, so nothing here says the model has converged.

Total +1.046 dB PSNR / +0.0119 SSIM over exp 71 at IDENTICAL capacity
(10,806 points), which is the useful part: the gain is optimization, not
capacity. Routing drifted slightly more dynamic (mean_dynamic_prob
0.9762 -> 0.9817, percent_uncertain 4.65% -> 3.24%). Runtime ~35 min
wall for 7,000 iterations plus seven validations.

The `points3d.ply` scale problem recorded above is INHERITED, not
retried: this lane starts from exp 71's trained state. A better
initialization remains a separate question and is untouched.

Operationally, `main.py`'s `DEFAULT_MAX_TRAIN_ITERATIONS` (6000) guard
had no reachable override: the template hard-coded the container
environment and the wrapper had no env passthrough. The ceiling is now a
template placeholder defaulting to main.py's own 6000, so every existing
lane renders the identical value it already got implicitly, and a
deliberate long run states its ceiling in its own rendered config.
## Experiment 78's decision decomposed — the photometric arm could not contribute (2026-08-16)

DIAGNOSTIC. This decomposes an EXPLORATORY run's already-recorded
decision; it establishes nothing scientific about EL-GS and does not
reopen exp 78's reading.

The open question this page left was "`delta_render` and `exact_deltas`
are not separable from the artifacts". That is TRUE as stated and is now
measured rather than asserted, and one further fact turned up that the
`se = 0.0` line alone did not show.

### What the artifact holds, and what it discards

`elgs.acceptance.decide` returns nine quantities: `delta_render`,
`exact_deltas`, `transaction_increment`, `se`, `k`, `n_samples`, `ess`,
`n_units`, `accepted`. `elgs.round_driver.run_pass` kept the record only
in the in-memory `RoundOutcome`, and `elgs.trainer_hooks` persisted
**`n_samples`, `se` and the drawn `units`** — nothing else. Nothing
EL-GS-side calls `tb_writer.add_scalar`, so the tfevents file has none
of it either, and the trial log carries only the `elgs_round` /
`elgs_evidence_round` summaries. Recovered from
`chkpnt220.pth` by `scripts/decompose_elgs_decision.py`:

```
candidate FISSION:219:a5395116   op FISSION   family 219
round_index 0   iteration 200
n_samples 8   n_units_drawn 8   se 0.0 (repr "0.0", exactly zero)
incumbent interval 219: K=1, a=[0.0], latch_pre/post true
```

The eight paired renders are deterministic given the model state AT
iteration 200. That state was never checkpointed — `chkpnt220.pth` is
after the fission was applied and after twenty further training
iterations — so exp 78's own per-unit deltas can be neither recovered
nor recomputed. That is a property of the run, not of the analysis.

### What `se = 0.0` does and does not pin down

`delta_render = sum_i w_i d_i / sum_i w_i` over the SHARED CRN weights,
so it IS the weight-normalized mean of the eight per-unit paired
photometric deltas `d_i`. `se` is the standard deviation of 200 paired
cluster-bootstrap replicates, each a weighted mean of a resampled
multiset of those same `d_i`. A common NONZERO `d` therefore also
collapses the spread — which is exactly why `se = 0` may not be read as
"the photometric contribution was zero".

Measured over 8 units with UNEQUAL SNIS weights and four bootstrap seeds:

| common per-unit delta | resulting `se` |
|---|---|
| `0.0` | `0.0` exactly, all seeds |
| `1e-12` … `1e-1` | `~4e-17` to `~6e-17`, never exactly zero |
| genuine spread (control) | `8.17e-3` |

So `se == 0.0` bit-exactly is the signature of an identically zero
photometric arm, and a nonzero common delta would have serialized as
`~5e-17`, not `0.0`. This is corroboration, not proof: it rests on the
two arms' SNIS ratios being accumulated separately, so only identical
inputs give a bit-identical difference.

### The mechanism: every confirmation unit was at one instant

The persisted `units` settle it independently. All eight are
`(index, 0.0)` — `distinct_unit_timestamps 1` — while the sequence spans
`time_span 4.675` at `frame_dt 1/120` (both from exp 78's own
`elgs_setup` log line, so the cameras do carry distinct timestamps).

The chain, read off source:

1. `setup_elgs` builds `reserved_pool` by iterating `sorted(by_time)` —
   ASCENDING TIMESTAMP. Frame 0 contributes 9 units under the
   `(f + c) % 4 == 0` diagonal with 35 cameras.
2. `SlotGrid.draw` returns a CONTIGUOUS slice,
   `reserved_pool[start : start + units_per_slot]`, and slot
   `(round 0, pass 0, rank 0)` has `start = 0`. So the first confirmation
   slot is eight of frame 0's nine reserved units.
3. The candidate is a mid-plateau FISSION: `_propose_smoke_candidates`
   opens a gap at `mid +/- half_gap` with `mid` the interval midpoint
   (~2.34 s for a spanning K=1 family) and
   `half_gap = 0.5 * 1.5 * floor_gap`, where
   `floor_gap = 2*w + 1.0 * frame_dt` is a few frame intervals.
4. `t = 0.0` is therefore ~2.3 s outside the only region where the
   candidate changes presence. Both arms render the identical image at
   every confirmation unit, so `d_i = 0` EXACTLY for all eight,
   `delta_render = 0.0` exactly, and `se = 0.0` exactly — which is what
   the artifact records.

The stratification guard in `setup_elgs` is not violated: it checks that
the reserved POOL spans at least half the sequence, which it does. It
does not check that a drawn SLOT does, and a contiguous slice of a
time-ordered pool is the one draw that systematically cannot.

### Consequence for the decision

With `delta_render = 0` and `k*se = 0`, acceptance reduces to

```
exact_deltas + transaction_increment < 0
```

so the commit was carried ENTIRELY by the non-photometric terms. It does
not follow that Phi is nonzero: `transaction_increment` is the other
term and is equally unrecorded. What is now excluded is the reading that
the photometric arm supported the fission — it was structurally
incapable of expressing an opinion.

**NOT a reinterpretation of exp 78.** Exp 78's recorded finding stands:
the real q/likelihood/Phi path executed on 192 real reports. This says
only which term could have moved its acceptance, and why the artifact
could not say so before.

### Recorded and repaired

- `elgs/trainer_hooks.py` now persists every `AcceptanceRecord` term in
  `committed_decisions` and emits an `elgs_acceptance` log line for
  commits AND rejections. Observability only: every field is an output
  of `decide`, written after the fact, and no decision changes.
- The confirmation-slot time collapse is RECORDED, NOT PATCHED. Changing
  which units confirm a decision changes what the §7 confirmation
  measure means and is preregistration-adjacent; it is not an
  implementation defect to be quietly fixed. It is a deferred item.
