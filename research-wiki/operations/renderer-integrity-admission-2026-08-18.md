# Renderer integrity closure and the admitted image (2026-08-18)

Operational/engineering record. EXPLORATORY throughout: no training
result, no scientific claim about any method, `evidence_bearing: false`
on every cell. Append-only; nothing on
[[rasterizer-backward-two-defects-2026-08-17]] or
[[rendered-flow-supervision-is-nonfunctional]] is rewritten.

## 1. What was admitted

Two defects in the ACTIVE backward render kernel, diagnosed and repaired
on 2026-08-17 and recorded in
[[rasterizer-backward-two-defects-2026-08-17]], are now committed,
reviewed, pushed, built, and verified on a V100.

| item | value |
|---|---|
| Branch | `apollo/csvl-vpl-v2-exploratory` |
| Admitted commit | `d21f1e9a34d2141c70bb5e86a6c4c376ab24b96e` |
| Image tag | `sudarshaniyengar/adags:apollo-v100-495ae16` |
| **Image digest** | `sha256:70a28e3d46a4768d595bda67328ddaacd18ae3af98ae863fc3f7303c159bb683` |
| Extension path | `/opt/conda/lib/python3.10/site-packages/_adags_diff_gaussian_rasterization.cpython-310-x86_64-linux-gnu.so` |
| CUDA source hash | `99c9fbd7f5eafec3b2d87a53f07fdc2d4faf688e2eac83c8f2fa1fb2c8c82d52` |
| Build | `Dockerfile.apollo-v100`, unedited; torch 2.0.1+cu118, nvcc 11.8, `TORCH_CUDA_ARCH_LIST="7.0 8.9"` |
| Superseded image | `apollo-v100-v1`, `sha256:51f8a852398ed0ca78ef7e9b0e41ddc7debc2d7475f95fae89239a64c4ceb2f1` — retained, never overwritten |

**The image tag names commit `495ae16`, and that is deliberate and
correct.** The extension was compiled from that commit's rasterizer
sources. The three commits after it (`58b2c95`, `13966fb`, `d21f1e9`)
touch only Python and Markdown, so the combined CUDA source hash at
`d21f1e9` is byte-identical to the one built into the image —
`99c9fbd7…`, verified by re-deriving it at HEAD — and the container
executes the code from the uploaded `git archive`, not from the image.
The image was therefore NOT rebuilt for a Python-only change. Experiment
139's own `cuda_sources.matches: true` is the check that this reasoning
holds rather than merely sounds right.

## 2. Commits

| commit | contents |
|---|---|
| `05e22be` | the two `backward.cu` repairs |
| `a63da6b` | independent colour compositing oracle + `tests/test_colour_background_vjp.py` |
| `495ae16` | the new durable defects page + append to the flow page |
| `58b2c95` | review corrections: single-seed comparative pin, docstring table, three wiki corrections, the dead depth/alpha gradient |
| `13966fb` | run the colour pins inside the V100 verifier |
| `d21f1e9` | `KNOWN_RESIDUALS`: named residuals vs new failures |

## 3. Independent review

A focused adversarial 12-item read-only review was obtained before the
image was admitted. **The requested reviewer `gpt-5.6-sol-high` was again
refused by the account** (HTTP 400, "not supported when using Codex with
a ChatGPT account"), as it was on 2026-08-17; the review ran on Codex's
default model at high reasoning effort. **The requested reviewer did not
run.**

It confirmed the two repairs and returned four actionable findings, each
re-verified against the source before being accepted. Three were
corrections to the record and one was a NEW pre-existing defect; all are
recorded in the 2026-08-18 append on
[[rasterizer-backward-two-defects-2026-08-17]]. The load-bearing one:

**The comparative finite-difference pin was unsound as first written.**
It differenced black with seed 67 and white with seed 61, so the two
measurements used different upstream gradients and therefore compared two
different scalar losses. The entire justification for that pin is that
the renderer's secant bias is common-mode and cancels, which requires the
two runs to differ in the background and in nothing else. Fixed in
`58b2c95`; it passes on the V100 with one seed, so the fix did not buy
the pass by loosening anything.

## 4. Verification — experiments 138 and 139

Both `dgx` / Tesla V100-SXM2-32GB, exploratory, image digest
`70a28e3d…`, ≈0.02 GPU-hours together.

| exp | commit | state | meaning |
|---|---|---|---|
| 138 | `13966fb` | ERROR | fail-closed on 5 test failures; produced the decisive gradient measurements below |
| 139 | `d21f1e9` | **COMPLETED** | `"verified": true` |

### 4.1 The gradient is live — the decisive measurement

Experiment 138's gradient-routing stage, on the repaired kernel:

| scene / loss | `flows` | `means2D` | `means3D` | `opacities` |
|---|---:|---:|---:|---:|
| n2, loss on render | 0.0 | 3.9586 | 9.2375 | 24.9197 |
| n8, loss on render | 0.0 | 7.15e-07 | 1.1753 | 27.5222 |
| n8, loss on flow | **2.1435** | 1.43e-06 | 2.4275 | 68.3836 |
| n72, loss on render | 0.0 | 5.96e-07 | 0.5977 | 7.7896 |
| n72, loss on flow | **2.1435** | 1.43e-06 | 2.7759 | 72.7209 |

Every one of these was **exactly 0.0** across experiments 132–137 on both
the pre-patch and post-patch images. Two things follow:

* **the colour-gradient control is nonzero** — the precondition
  [[rendered-flow-supervision-is-nonfunctional]] set before its 36 flow
  tests could be cited at all;
* **the rendered-flow gradient is live on the real path.** `flows` is
  0.0 under a colour loss and 2.1435 under a flow loss, at both scene
  sizes, which is the correct routing rather than a uniformly nonzero
  smear.

This closes the standing flow-lane blocker: the instrument works. It does
**not** establish that the flow VJP is numerically correct, and it does
not unblock F or X. See §7.

### 4.2 Tests — 57 of 62 pass, 5 residuals, 0 unexpected

Experiment 139, on the V100, with `tests.test_colour_background_vjp` now
inside the verifier (`13966fb`) because every pin in it needs a GPU and a
workstation run reports the whole module as skips:

```
"run": 62, "passed": 57, "failures": 4, "errors": 1,
"unexpected_failures": [], "known_residuals_no_longer_failing": [],
"cuda_sources": {"matches": true}, "verified": true
```

**All 11 colour-background pins pass**, including the independent-oracle
pins at 5e-3 on both backgrounds, the anti-vacuity sensitivity test
(reinstating the deleted term would move the gradient past the
tolerances), the black-background exact-zero test, and the corrected
single-seed comparative finite-difference pin.

The 5 failures are exactly the three classes already measured and
recorded, and no others:

| residual | measured | class |
|---|---|---|
| `test_empty_scene_renders_zero_flow` | host pointer at `P == 0` | pre-existing, no gradient path |
| `test_opacity_gradient_matches_finite_differences` (index=1) | 1.1749 vs 1.5871 | flow-mediated opacity secant, ~35% |
| `test_projected_geometry_gradient_matches_finite_differences` ×3 | −0.7328 vs −0.5355; 3.9212 vs 3.5875; **0.3557 vs −0.6572** | projected-geometry secant, up to 46%, one sign flip |

`d21f1e9` teaches the verifier to separate these NAMED residuals from any
new failure. It is an allowlist of named defects, not a tolerance: each
entry carries its recorded reason, a failure matching none of them still
fails the run, and the report separately lists any allowlisted residual
that did NOT fire — because a residual that stops failing means the
allowlist is claiming cover for a defect that may no longer exist. On
experiment 139 that list is empty, so the allowlist is exactly as wide as
the defects that still exist and no wider.

**These residuals were NOT chased.** All three are background-independent
and measured on a black background where both repaired defects are
provably inert, so none is a colour-VJP or flow-VJP defect. Per the
standing scope limit, forcing non-smooth finite differences to agree is
not renderer-integrity work.

## 5. Forward equivalence

Experiment 136 (2026-08-17) measured a **bit-identical** colour forward,
`0.519281804561615`, between the old and new images on the TinyScene
harness with identical bucket counts. The repairs are backward-only and
touch no forward kernel, which is consistent with that.

Real-batch forward equivalence is addressed by the §6 reproducibility
bound rather than by a separate cell: a run whose first evaluation
matches on both images is stronger evidence than a single forward, and it
was already being paid for.

## 6. Reproducibility bound — the question prior controls turn on

**Question.** [[rasterizer-backward-two-defects-2026-08-17]] states that
every training run in this repository executed a backward whose per-tile
early-out read uninitialised memory, so no prior run's gradients are
known reproducible. That is a statement about what is *known*, not a
finding that any result is wrong. The measurement that converts it into
something actionable is: **was the old image self-consistent, and does
the repaired kernel diverge from it?**

**Design.** `configs/elgs/smoke_elgs.yaml` — the M0 S1 smoke: 600
iterations, from scratch, N3V via `ADAGS_DATA_ROOT`, `elgs_enable: True`
— seed 0, pool `dgx`, commit `d21f1e9`, everything matched except the
image:

| exp | cell | image |
|---|---|---|
| 143 | `repro_bound_old_a` | old `51f8a852…` |
| 144 | `repro_bound_old_b` | old `51f8a852…` — identical to 143 |
| 145 | `repro_bound_new` | new `70a28e3d…` |

143 versus 144 measures the OLD image's own run-to-run spread. That is
the comparison the 2026-08-18 audit's independent challenge insisted on:
without it, any old-versus-new difference could be attributed to the
repair when it was really the old image's own nondeterminism. This smoke
is the right instrument because it is cheap, trains from scratch, and was
previously reported bit-identical across two old-image runs, so there is
a prior expectation to violate.

**A FIRST ATTEMPT FAILED and is preserved, not deleted.** Experiments
140/141/142 were submitted against experiment 123's hull configuration
capped by `--max-train-iterations 2000`. All three errored in ~15
seconds with

```
RuntimeError: Refusing to train for 15000 iterations;
the guarded maximum is 2000.
```

`--max-train-iterations` sets `ADAGS_MAX_ITERATIONS`, which is a REFUSAL
CEILING, not a truncation: `main.py:1028` aborts when the config asks for
more than the guard allows rather than training fewer iterations. That is
correct behaviour and the guard did its job. Two further facts made that
configuration the wrong instrument anyway, and both were missed before
submitting: `diva360_scissor_bench30_hull_c15k.yaml` RESUMES from
experiment 104's checkpoint at iteration 6000, so it would not have
exercised from-scratch gradients; and its 15,000-iteration schedule is
far past the bounded scale this lane is authorised for. Claim indices
`repro_bound_{old_a,old_b,new}__r0` are consumed and must never be reused
or deleted; the corrected triple runs at `r1`.

**Projected cost** ≈0.45 GPU-hours total (0.15 each), down from the 2.1
the first attempt would have cost.

**Termination.** The measurement ends when all three are terminal and the
three-way comparison exists. No further renderer runs are authorised by
this page under any outcome.

**Interpretation, fixed before the numbers:**

| reading | conclusion |
|---|---|
| 143 == 144 bit-identical, 145 differs materially | prior runs were self-consistent WITHIN the old image; controls remain internally comparable but are NOT comparable to anything trained on the new image, which must be re-run before any cross-image comparison |
| 143 != 144 | the old image was nondeterministic run to run; prior gradients are not reproducible even within themselves, and every cross-run comparison in the old image needs its own spread reported |
| 143 == 144 == 145 | the guard was a no-op in this configuration; prior controls survive unqualified for configurations resembling this one, and ONLY for those — the smoke is 600 iterations of N3V and generalises to a 15k DiVa-360 run only by assumption |

### RESULT (2026-08-18) — the old image is NOT run-to-run reproducible

All three COMPLETED. Final iteration, 600/600, identical config and seed:

| exp | image | Loss | PSNR | Ll1 | Lssim |
|---|---|---|---|---|---|
| 146 | old `51f8a852…` | 0.1021910 | 26.02 | 0.0321 | 0.1211 |
| 147 | old `51f8a852…` (identical to 146) | 0.1017597 | 26.38 | 0.0317 | 0.1197 |
| 148 | new `70a28e3d…` | 0.0974231 | 26.63 | 0.0303 | 0.1142 |

**146 != 147.** Two runs of the SAME image, SAME config, SAME seed
disagree: loss by `4.3e-4`, **PSNR by 0.36 dB**. The frozen
interpretation table's middle row applies.

### What this does and does not establish

**Established: the old image was not bit-reproducible at fixed seed.** The
earlier expectation that this smoke was bit-identical across runs is
refuted for this configuration.

**NOT established: that the repair changed training.** The new image's
run sits 0.25 dB above the higher of the two old runs, while the old
image's own spread is 0.36 dB. **148 lies within roughly one old-image
spread**, and with n=1 on the new image and n=2 on the old, this
measurement cannot separate "the repaired kernel trains differently" from
"this configuration is simply nondeterministic at this magnitude". It
would be an over-reading of exactly the kind this lane exists to prevent
to report 148's higher PSNR as an improvement.

**The nondeterminism is not attributed to the guard.** The backward
accumulates through `atomicAdd`, whose float summation order is not
deterministic, and that mechanism is present in BOTH images. So run-to-run
variation is expected even from a wholly correct kernel, and nothing here
isolates `max_contrib` as its source. Attributing the spread to the guard
would be the same unproven-attribution error that experiment 132's
failure message already made once on this lane.

### Consequence for prior controls — the actionable part

Prior photometric results are **not invalidated**. What changes is how
they may be READ:

* any comparison in this repository resting on a SINGLE run per arm
  carries an unreported run-to-run uncertainty, measured here at
  **0.36 dB PSNR** for a 600-iteration N3V smoke at 50k points;
* differences smaller than that spread are **not resolvable** from one
  run per arm in this configuration;
* the standing "no prior run's gradients are known reproducible" is now
  sharpened: they are not reproducible, and the reason need not be the
  guard.

**Scope limit, stated rather than assumed.** This is 600 iterations of
N3V `cut_roasted_beef` at 50k points with `elgs_enable: True`. It does
NOT license transferring the 0.36 dB figure to a 15k-iteration DiVa-360
run at 400k+ points, where both the spread and its drivers may differ. Any
lane that needs a resolvable difference should measure its OWN spread
rather than importing this one.

**What would settle the open question**, and is NOT authorised here: two
or three runs of the NEW image at the same seed, giving it its own spread
to compare against the old image's. That is ~0.3–0.45 GPU-hours and is
the natural next measurement if any future comparison needs to span the
two images.

## 7. What is NOT admitted

* **That the flow VJP is numerically correct.** The instrument now works;
  the VJP itself is neither confirmed nor refuted. The 36-test suite runs
  and 33 pass, but per §4.2 the flow-mediated finite differences are 35%
  off with a sign flip, so central differences cannot arbitrate it here.
* **F and X.** Still blocked, and per the 2026-08-18 strategic audit they
  are OFF the EL-GS critical path entirely: no EL-GS claim has a flow
  term, DiVa-360 has no flow, and the chosen N3V scene's flow has median
  magnitude 0.015–0.035 px, too small to establish sign. Their disposition
  is a user decision.
* **The dead depth and alpha gradients.** `dL_depths` and `dL_masks`
  reach the launched kernel and are never read (`backward.cu:1253-1254`;
  their only reads are at `:1061-1062` in the unlaunched `renderCUDA`). A
  loss on rendered depth or rendered alpha produces no Gaussian gradient
  — silently, the same failure mode as the guard. Pre-existing, NOT
  repaired here, deliberately not bundled. Recorded as a candidate for a
  separate bounded repair.
* **The guard's performance cost.** Still unmeasured. Restoring the
  optimisation means computing `max_contrib` in the forward, which is a
  separate change and was not attempted.
* **The alpha clamp, `P == 0`, and the geometry secant.** Unchanged, and
  explicitly out of scope.

## 8. Consumed claim indices

```
flow_vjp_v100_verify   r5 (exp 138), r6 (exp 139)   -> next free r7
repro_bound_old_a      r0 (exp 140)                 -> next free r1
repro_bound_old_b      r0 (exp 141)                 -> next free r1
repro_bound_new        r0 (exp 142)                 -> next free r1
```

Never reuse or delete a consumed claim.
