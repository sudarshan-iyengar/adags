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

---

## APPENDIX C (2026-08-18, append-only) — the new image's own spread, and it settles section 6's open question

Section 6 named the settling measurement and declined to run it: "two or
three runs of the NEW image at the same seed, giving it its own spread".
That was authorised separately and has now run. Nothing above is rewritten;
this appendix reports what the added runs show and where they change how
section 6's result may be READ.

**Setup, identical to 148 in every respect the wrapper records** except the
retry index: `configs/elgs/smoke_elgs.yaml`, 600 iterations, seed 0, from
scratch, N3V `cut_roasted_beef`, pool `dgx`, admitted image
`sha256:70a28e3d…`, `--max-train-iterations 600`, same `--source_path`.
Commit is `a70f14f` rather than 148's `392ba4a`; the only code difference
between them is `scripts/build_absence_diagnostic.py` and its tests
(`git diff --stat 392ba4a..a70f14f`), neither of which is in the training
path, and the CUDA source hash is byte-identical. So the arms differ by
nothing that executes.

| exp | image | retry | held-out `best_val/psnr` | `final/ssim` | training-log Loss | training-log PSNR |
|---|---|---|---:|---:|---:|---:|
| 146 | old `51f8a852…` | r2 | 28.54022950 | 0.92084739 | 0.1021910 | 26.02 |
| 147 | old `51f8a852…` | r2 | 28.43576755 | 0.92022758 | 0.1017597 | 26.38 |
| 148 | new `70a28e3d…` | r2 | 28.77180493 | 0.92390805 | 0.0974231 | 26.63 |
| **152** | new `70a28e3d…` | **r3** | **28.77185474** | 0.92390745 | 0.0974552 | 26.63 |
| **153** | new `70a28e3d…` | **r4** | **28.77213518** | 0.92392631 | 0.0974608 | 26.63 |

`best_val/psnr` and `final/ssim` are read from each run's `summary.json` on
Apollo; the two training-log columns are the final progress line, the same
quantity section 6's table used.

### The measured spreads

| quantity | old image (n=2) | new image (n=3) | ratio |
|---|---:|---:|---:|
| held-out `best_val/psnr` | **0.10446 dB** | **0.00033 dB** | ~317x |
| training-log Loss | 4.313e-4 | 3.77e-5 | ~11x |
| training-log PSNR | 0.36 dB | 0.00 dB (26.63 in all three) | — |

**The new image is reproducible run-to-run in this configuration; the old
one was not.** Three runs of the repaired kernel agree to 3.3e-4 dB of
held-out PSNR — five significant figures — while the two old-image runs
disagree by 0.10 dB on the same metric and by 0.36 dB on the training-log
metric section 6 reported.

### This RESOLVES section 6's "not established", in the direction of the repair

Section 6 concluded that 148 "lies within roughly one old-image spread" so
the measurement "cannot separate 'the repaired kernel trains differently'
from 'this configuration is simply nondeterministic at this magnitude'".
With the new image's own spread in hand, it separates:

* the new image sits **0.2316 dB** above the BETTER of the two old runs on
  held-out PSNR;
* that gap is **~700x the new image's own spread** and **~2.2x the old
  image's**;
* every one of the three new runs is above every old run, on all four
  reported quantities, with no overlap.

So the repaired kernel does train measurably differently on this
configuration, and section 6's cautious refusal to say so was a consequence
of n=1, not of the data.

**The weak side of this is stated too.** The old-image spread rests on
n=2, so it is a poor estimate of that image's variability, and an unlucky
pair cannot be excluded on the numbers alone. What does not depend on that
estimate is the new image's tightness at n=3, and the qualitative contrast
between three runs agreeing to five significant figures and two runs
disagreeing in the second.

### The atomicAdd attribution, revisited honestly

Section 6 declined to attribute the old spread to the guard, on the correct
ground that `atomicAdd` float-summation order is nondeterministic and exists
in BOTH images. That reasoning is unchanged and remains right as far as it
goes. What the new runs add is an empirical bound on it: **if `atomicAdd`
ordering were the dominant source of run-to-run variation here, the
repaired image would vary by a similar amount. It varies by 3.3e-4 dB.**
Therefore, in this configuration, `atomicAdd` ordering contributes at most
that, and the old image's 0.10–0.36 dB came from somewhere else. The
identified candidate — the only relevant difference between the two images
— is the per-tile early-out reading uninitialised memory
([[rasterizer-backward-two-defects-2026-08-17]]).

**Still NOT established, and deliberately not claimed:** that `max_contrib`
specifically is the mechanism. No experiment isolated it; the inference is
by elimination over a two-element difference set, which is weaker than a
direct measurement and is labelled as such.

**Also NOT established:** that the new image's higher PSNR is "better
training" in any scientific sense. It is a different, now-reproducible
trajectory. Whether the repaired gradient is the *correct* gradient is the
question section 4.2's finite-difference residuals could not arbitrate, and
this appendix does not touch it.

### What changes for prior results, and what does not

* Section 6's actionable consequence for OLD-image single-run comparisons is
  unchanged: they carry ~0.36 dB of unreported run-to-run uncertainty in
  this configuration and smaller differences are not resolvable.
* **NEW, and this is the useful part:** single-run-per-arm comparisons on
  the ADMITTED image do NOT inherit that penalty in this configuration.
  Three runs agreeing to 3.3e-4 dB means a future matched comparison on the
  new image can resolve differences far below 0.36 dB — which materially
  improves the power of the matched presence triple, whose decision rule
  uses `max(0.5 dB, |T-1 - T-1'|)`. The `|T-1 - T-1'|` term is measured
  per-experiment and is NOT replaced by this figure.
* **The scope limit stands unchanged.** This is 600 iterations of N3V at
  50k points with `elgs_enable: True`. It does NOT license transferring
  either spread to a 15k-iteration DiVa-360 run at 400k+ points. A lane
  needing a resolvable difference still measures its own spread; what has
  changed is the prior expectation about how large that will be.

### Cost and provenance

Two runs, ~13.6 min of training each, wall 21.5 and 21.3 min including
container start and evaluation — **≈0.72 slot-hours**, against a projected
0.30 (0.15 each, inherited from 146–148's projection). The projection
understated the actual by ~2.4x because it counted training time only.
Recorded rather than smoothed over: the block's stated ceiling for this item
was ≈0.5 GPU-h and the measured consumption exceeded it.

Both `evidence_bearing: false`, exploratory, `dgx`. Consumed claim indices:

```
repro_bound_new   r3 (exp 152), r4 (exp 153)   -> next free r5
```

No further renderer runs are authorised by this page under any outcome, and
this appendix does not authorise any.

## APPENDIX D (2026-08-20, append-only) — H100 image built to parity, not yet verified on-cluster

Section 1's admission covered `dgx`/V100 only. The `hopper`/H100 image had
not been rebuilt against the repaired kernel; `apollo-h100-v2`
(`sha256:a2877f26…`) still carries the pre-repair `backward.cu`. This
appendix records a workstation build/push closing that gap, on the
workstation, not on Apollo — no `det` submission, no GPU-hours, no ledger
entry.

**Precondition checked before building.** `git log --oneline
05e22be..HEAD -- diff-gaussian-rasterization/ simple-knn/ pointops2/`
returns nothing: no CUDA-affecting commit has landed since the repair.
`scripts/verify_flow_vjp_runtime.py --print-cuda-sha256` at HEAD (`88ee245`)
returns combined hash `99c9fbd7f5eafec3b2d87a53f07fdc2d4faf688e2eac83c8f2fa1fb2c8c82d52`
— byte-identical to the hash recorded in §1 for the admitted V100 image.
So the H100 image could be built at current HEAD without re-deriving
anything about the repair itself.

| item | value |
|---|---|
| Branch | `apollo/csvl-vpl-v2-exploratory` |
| Built at commit | `88ee245` (CUDA sources unchanged since `05e22be`, hash verified equal) |
| Image tag | `sudarshaniyengar/adags:apollo-h100-88ee245` |
| **Image digest** | `sha256:0d5771688c9b6580f70133f813b7a4110bd5c967920afe3c5fd1856bb098800e` |
| CUDA source hash | `99c9fbd7f5eafec3b2d87a53f07fdc2d4faf688e2eac83c8f2fa1fb2c8c82d52` (matches §1) |
| Build | `Dockerfile.apollo-h100`, unedited; torch 2.0.1+cu118, nvcc 11.8, `TORCH_CUDA_ARCH_LIST="8.9 9.0+PTX"` |
| Superseded image | `apollo-h100-v2`, `sha256:a2877f26cb8528454fe45e701ce638a6042dd68155fb5359cb7edc608a4a7816` — retained, never overwritten |

The in-Dockerfile `validate_apollo_runtime.py --build-check` gate ran during
the build and passed: all three extensions
(`_adags_diff_gaussian_rasterization`, `pointops2_cuda`, `simple_knn._C`)
compiled and imported, `nvcc` resolved to CUDA 11.8, `cuda_available: false`
as expected for a build with no GPU device attached. This is a build-time
smoke check, not a functional verification.

**What this does NOT establish, as first written.** No `det cmd run` or
`det e create` cell had yet executed against this digest on `hopper` at
build time. Section 4's `cuda_sources.matches: true` in-container check and
the gradient-liveness measurement were both run on `dgx` against the V100
digest; neither had been repeated here. The verification below closes that
gap.

### RESULT (2026-08-20) — verified on `hopper`, identical residuals to V100

Determined experiment **207**, cell `flow_vjp_h100_verify` retry `r1`
(`r0` consumed by a `--dry-run`), pool `hopper`, image digest
`sha256:0d5771688c9b6580f70133f813b7a4110bd5c967920afe3c5fd1856bb098800e`,
entrypoint `scripts/verify_flow_vjp_runtime.py --expect-commit
88ee245bf3c813ed3d752d1d0d50aef722de1f07 --expect-cuda-sha256
99c9fbd7f5eafec3b2d87a53f07fdc2d4faf688e2eac83c8f2fa1fb2c8c82d52`,
`evidence_bearing: false` (`--exploratory`). `STATE_COMPLETED`, container
exited zero.

```
device_name: NVIDIA H100 PCIe, device_capability: 9.0, cuda_available: true
cuda_sources.matches: true
tests: run 62, passed 57, failures 4, errors 1, unexpected_failures: []
verified: true
```

The 5 residuals (4 failures + 1 error) are the SAME named, tolerated
defects as V100 experiment 139's §4.2 table, with the SAME finite-difference
numbers to the last printed digit (e.g. opacity `1.1748790740966797 !=
1.5871202945709229`; geometry `0.35566091537475586 != -0.657229483127594`).
None is new. The gradient-routing block also matches V100's routing
pattern: `flows` is `0.0` under a colour loss and `2.1435470581054688`
under a flow loss at both scene sizes tested, and
`max_relative_error_vs_oracle: 1.0052688149127206e-07` on the colour
forward. This is the H100 counterpart of §4's decisive measurement.

No `terminal.json` was sealed locally: this workstation has no `/apollo`
mount (workstation reads route through the `apollo:` rclone remote per
[[apollo-determined-execution-authority]]), and this is an exploratory,
non-evidence-bearing cell, so a sealed audit was not required to establish
the result. Consumed claim indices:

```
flow_vjp_h100_verify   r0 (dry-run, no .experiment), r1 (exp 207)   -> next free r2
```

`apollo-h100-88ee245` is now VERIFIED on `hopper`, not merely built to
parity — the same evidentiary standard §4 applied to the V100 image.
