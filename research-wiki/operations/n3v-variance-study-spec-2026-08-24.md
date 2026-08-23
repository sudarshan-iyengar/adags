# SPEC (FROZEN) — N3V run-level variance study and endpoint validation
# at the 50-frame 6k protocol (2026-08-24)

EXPLORATORY, `evidence_bearing: false`. **Frozen and committed BEFORE any
cell of this study is submitted.** Every threshold, endpoint, exclusion
rule and stopping rule below is fixed as of this commit and none may move
after a number is read.

## 0. Relationship to the replicate-floor spec — an AMENDMENT, not a reopening

[[same-code-replicate-floor-spec-2026-08-23]] §5 states: *"No further
replicate cells are authorized by this spec under any outcome"*, and its
RESULT section adds that the result *"does not license re-running at a
different protocol to obtain a smaller floor."*

**Both prohibitions stand, historically and prospectively, and this study
does not violate either.** That spec asked *"is the replicate floor
non-negligible?"* and answered it: **R = 0.4945 dB**, top row, terminal.
Nothing here re-asks it, and no outcome below can retract or soften it.

This study asks two **different** questions that the earlier spec could
not answer at n=3 and did not attempt:

1. **What IS the run-level standard deviation?** `max − min` at n=3 is a
   demonstration that the floor is non-negligible, not an estimate of
   σ. The 95% CI for σ from those three runs spans **[0.1365, 1.6478]** —
   a **12.07× ratio**. Every downstream cost calculation depends on where
   in that range σ actually sits, and at 12× uncertainty none of them can
   be made.
2. **Do alternative endpoints have lower run-level variance?** The
   earlier spec fixed its two endpoints in advance and was right to; it
   therefore has nothing to say about a third.

Authorized by the 2026-08-24 block directive, which explicitly permits a
*"new, separately frozen variance-study protocol"* and requires this
amendment be recorded append-only. **Crucially, this is not a re-run at a
different protocol to obtain a smaller floor: the protocol is IDENTICAL
to the one that produced 0.4945.** Changing it would forfeit the very
continuity that makes the measurement useful.

## 1. The estimand

> **σ = sd(Y | C)**, where the experimental unit is the **RUN** and `C` is
> the fixed conditioning set: commit, `config_canonical_hash`, image
> digest, pool, seed, entrypoint, dataset and schedule.

Host assignment, allocation, and CUDA/atomics nondeterminism are **inside**
the estimand, not controlled away — they are the mechanism under study.
The three earlier cells ran on three different hosts, and that is
representative of how every future comparison will actually run.

**The unit is the run.** Not the pixel, not the frame, not the ray. This
is stated first because §5's forbidden estimators all violate exactly it.

## 2. Endpoints

### 2.1 Primary — `all_events_union` pooled PSNR

`scripts/event_ray_metrics.py` on
`configs/n3v/ladder_event_masks_crb0_49.json`, 8-bit saved-render basis —
**the same convention, masks and evaluator that produced every recorded
ladder delta and the 0.4945 floor.** Chosen for continuity: an endpoint
whose variance is measured on a different scale than the deltas it must
bound is not usable for bounding them.

### 2.2 Co-primary, PRE-REGISTERED AS A PREDICTION TO BE TESTED — the within-run contrast

> **`Δ = PSNR(all_events_union) − PSNR(complement)`**, per run.

**Rationale, and the finding that motivates it.** The three recorded
replicates say something sharp about *where* the variance lives.
Recomputed here from the recorded values:

| endpoint | pixel-times scored | s (n=3) | spread |
|---|---:|---:|---:|
| `all_events_union` | 231,480 | 0.262198 | 0.4945 |
| `complement` | 68,314,920 | 0.255700 | 0.4808 |
| `whole_frame` | 68,546,400 | 0.255482 | 0.4809 |
| **`union − complement`** | — | **0.174523** | **0.3089** |

**Scoring 296× more pixel-times reduces the spread by 2.8%.** If the
run-to-run variation were sampling noise over pixels it would fall by
roughly √296 ≈ 17×. It does not fall at all. **So the variance is a
global, run-level shift of the whole image, not spatial sampling noise**
— and a shift common to union and complement cancels in their difference.
The contrast's s is **33.4% lower**, which would be **2.26× fewer
replicates** for the same power.

**This is an n=3 hypothesis and it is registered as such.** The six fresh
runs TEST it; they do not assume it. The directive's stated purpose for
this study is *"correct estimation of run-level variance and validation of
alternative endpoints"*, and this is the endpoint being validated.

**Two costs of the contrast, disclosed now so no later reader discovers
them as surprises.** (i) It **changes the estimand** of any comparison
that uses it: it scores *event-region gain relative to the rest of the
frame*, not event-region quality. (ii) It scores **positive if an arm
damages the complement**. Therefore, if it is ever adopted as a decision
endpoint, it must be paired with a **complement-harm guard**: the
complement PSNR may not fall by more than δ* relative to the comparator.
That guard is frozen here, with the endpoint, and not invented later.

### 2.3 Controls, reported descriptively
`whole_frame` and `complement` pooled PSNR; pooled+clamped held-out PSNR
from `main.py --val`; SSIM; LPIPS; realized point count; rows relocated.

### 2.4 No dynamic-mask endpoint — and a code-verified refinement of the record

The query pack records that `motion_priors/masks/` holds only 300 `cam00`
files (the HELD-OUT camera), so with `dynamic_mask_from_residual: true`
every TRAINING camera's "dynamic mask" is a top-15% photometric-residual
quantile of the current render.

**Refinement, verified in code and recorded rather than quietly used:**
both evaluation call sites pass `allow_residual=False`, so that statement
describes the **training** path, not the evaluator. The endpoint is
nevertheless **excluded**, for a reason that survives the refinement: any
residual-derived mask makes the scored support a function of *the arm's
own render*, so two arms would be compared on different pixel sets. An
endpoint whose support depends on the thing being measured is not an
endpoint.

### 2.5 Linear-MSE endpoint — considered and NOT adopted

PSNR is a log transform of a mean, so its variance need not behave like
the error's. Checked rather than assumed: `event_ray_metrics.py` already
takes **one** log of **one** pooled mean (not a mean of per-frame logs),
and the delta-method prediction for sd on the dB scale is **0.2659**
against the measured **0.2622** — agreement to 1.4%. **dB is already the
variance-stabilizing scale here**, so a linear-MSE endpoint would buy
nothing and would forfeit continuity. Recorded as checked-and-rejected.

## 3. δ* — the minimum scientifically relevant effect

> **δ* = 0.30 dB** on the N3V event-union endpoint. This is a **JUDGMENT**
> and is labelled as one.

**Grounded externally, deliberately NOT derived from the ladder deltas** —
calibrating δ* against +0.077/+0.345 would be the circularity this project
has refused elsewhere. The published N3V field spans **FreeTimeGS 33.19 →
SharpTimeGS 33.57 = 0.38 dB** across the peer-reviewed and
self-reported state of the art. An effect materially smaller than that
whole span cannot carry a method claim on this dataset.

**An uncomfortable corollary, stated now rather than discovered later:
the mean recorded B1 event-union effect is +0.211 dB, which is BELOW δ*.**
Even a perfect, zero-variance instrument would not clear this bar for B1.
That is a fact about the effect, not about the instrument, and it is
frozen here before any new number exists.

## 4. Design

**Six fresh cells, byte-identical, in two waves of three** (`hopper` has
exactly 3 H100 slots, so a wave fills the pool):

* wave 1: `varstudy_b1_a`, `varstudy_b1_b`, `varstudy_b1_c`
* wave 2: `varstudy_b1_d`, `varstudy_b1_e`, `varstudy_b1_f`

`configs/n3v/ladder_b1_crb.yaml` (plain B1), `--seed 0`, pool `hopper`,
image `sudarshaniyengar/adags@sha256:0d577168…`, 6,000 iterations, frames
0-49, `cam00` sealed, one commit. They differ **only** in cell name and
retry index.

**These six are the primary cohort.** Experiments 261/262/263 are
historical development data and are **not** in the primary analysis.

### 4.1 Pooling with 261/262/263 — permitted ONLY as a secondary analysis, on a VERIFIED basis

The directive requires execution-relevant equivalence be established from
actual diffs and provenance, not belief. **Verified this block:**

```
git diff --stat eb293a2 HEAD -- elgs scene gaussian_renderer utils \
    arguments depth_visibility main.py det_exp_apollo.yaml \
    configs/n3v/ladder_b1_crb.yaml
  -> EMPTY
```

Every commit between `eb293a2` and this one is research-wiki prose plus
one standalone acquisition script that no training path imports. So the
training code is **byte-identical, verified by diff**.

**But the archives are NOT identical and must not be called so.**
`git archive` includes the wiki files and the new script, so
`archive_sha256` will differ between the two cohorts while
`config_canonical_hash` is equal. The honest statement is: **same
training code, verified empty diff; different archive.** Any pooled
analysis states both.

## 5. Variance and interval arithmetic — fixed in advance

Sample mean `ȳ = Σyᵢ/n`. Sample sd `s = sqrt(Σ(yᵢ−ȳ)²/(n−1))` — the
**n−1** denominator, always.

Confidence interval for σ, **assuming normality of the run-level
endpoint** (stated as an assumption; n=6 cannot verify it, and this is
disclosed rather than tested-and-passed):

> `[ s·sqrt((n−1)/χ²_{1−α/2, n−1}) , s·sqrt((n−1)/χ²_{α/2, n−1}) ]`

Computed widths, as a multiple of `s`, α = 0.05:

| n | df | lower factor | upper factor | width | upper/lower |
|---|---:|---:|---:|---:|---:|
| 3 | 2 | 0.5207 | 6.2870 | 5.77·s | **12.07×** |
| **6** | 5 | 0.6242 | 2.4526 | **1.83·s** | **3.93×** |
| 9 | 8 | 0.6755 | 1.9156 | 1.24·s | 2.84× |

Applied to the recorded n=3 union `s = 0.262198`: **95% CI for σ =
[0.1365, 1.6478]**. That is not a usable estimate, which is precisely why
this study exists.

**Median and MAD are reported descriptively alongside**, never as the
inferential quantity.

### 5.1 FORBIDDEN estimators — with the reason each is wrong

* **Bootstrapping pixels or frames and presenting the result as run-level
  uncertainty.** A frame bootstrap on these data yields SEM ≈ 0.0436 dB
  against a run-level σ of 0.2622 — **6× too small**. It would have
  licensed exactly the claims the replicate floor retired. Frames within a
  run share the run; they are not independent trained models.
* **`range / sqrt(n)`.** Not an estimator of anything. The unbiased
  range-based estimator is `range / d₂(n)`; that `d₂(3) = 1.693 ≈ √3` is a
  numerical coincidence at n=3 and does not survive to n=6 (`d₂(6) =
  2.534` vs `√6 = 2.449`).
* **Treating a 3-run range as a standard deviation.**
* **Attaching a p-value to anything in this study.** See §6.

## 6. Stopping rule — frozen before outputs

Run **n = 6**. Extend to **n = 9** if and only if the n=6 95% CI for σ on
the **contrast** endpoint straddles `σ_dec = 0.1672 dB` (the σ at which
the contrast endpoint's required replicate count crosses what a 24
slot-hour block can afford) — i.e. iff

> `0.0682 ≤ s₆ ≤ 0.2679`

**This is ESTIMATION, not hypothesis testing.** No test statistic is
computed, no null is posited, and therefore **no α-spending adjustment is
required and no p-value may ever be attached to this study's output.**
That is the reason the sequential rule is admissible at all, and it is
also a binding restriction on how the result may be reported.

**Continuation bias is disclosed and mitigated**: optional stopping on a
variance estimate biases σ̂. The mitigation, frozen here, is that **every
downstream power and cost calculation uses the UPPER confidence limit for
σ, not the point estimate.** A bias-free fixed-n=9 alternative was priced
at ≈14.6 slot-h extra and is available if a later block wants it.

## 7. Integrity and exclusion rules — frozen, exhaustive, checkable before any metric is read

A run is excluded if and ONLY if:

1. its Determined state is not `STATE_COMPLETED`;
2. any provenance field disagrees with the cohort: `commit`,
   `config_canonical_hash`, `archive_sha256`, `image_ref`, `pool`, `seed`,
   `entrypoint_script`;
3. a required artifact (`chkpnt6000`, the saved renders, the metrics JSON)
   is absent;
4. an endpoint evaluates to NaN or Inf;
5. an artifact hash does not match its own recorded manifest.

Every one of these is decidable **before** any endpoint value is looked
at, and the check is performed in that order.

> **A run may NOT be excluded for being an outlier, extreme, or
> "obviously wrong". Run-to-run variation IS the estimand.** Removing the
> tail of the distribution being measured would make the measurement
> report its own exclusion rule.

If fewer than 5 of 6 survive the integrity rules, the study is reported as
**inconclusive at n=6** and the shortfall is reported, not backfilled by
reaching into 261/262/263.

## 8. Cost

6 training cells × ≈2.23 slot-h (measured: 6.7 slot-h for three) ≈
**13.4 slot-h**, plus 6 `--val` cells × ≈0.2 ≈ **1.2 slot-h**. Total
≈ **14.6 slot-h** against the 24 slot-hour block ceiling. Wave 2 is
submitted only after wave 1 is terminal, so the cost is incurred in two
inspectable halves.

## 9. Power — computed in advance, and the answer is uncomfortable

Two-arm comparison, α = 0.05 two-sided, 80% power, `n ≈ 2(z_{α/2} +
z_β)²σ²/δ²` with a small-sample t correction:

| endpoint | σ (n=3) | δ* | replicates/arm | training slot-h for ONE comparison |
|---|---:|---:|---:|---:|
| `all_events_union` | 0.2622 | 0.30 | **14** | **68.1** |
| `union − complement` | 0.1745 | 0.30 | **7** | **34.1** |

**Lever 1 ("more replicates per arm") is UNAFFORDABLE on the union
endpoint**: 68.1 slot-h is **2.8× the entire 24 slot-hour block ceiling**
for a single two-arm comparison. At the upper confidence limit for σ it is
475/arm ≈ 2,312 slot-h. The earlier decision page's "roughly 25 replicates
per arm to resolve 0.1 dB" was optimistic by **4.4×**; the correct figure
is 109/arm ≈ 530 slot-h.

**Lever 3 ("a lower-variance endpoint by pooling more pixels or frames")
is NEAR-FALSIFIED by data already on the record** — see §2.2: 296× the
pixel-times buys 2.8%.

So of the four levers named in
[[membership-occupancy-and-decision-2026-08-23c]] §2, two are effectively
closed before this study runs, and this study is the measurement that
decides whether the remaining route — a lower-variance **contrast**
endpoint, at 7 replicates/arm and 34.1 slot-h — is real or an n=3 artifact.

## 10. What this study does NOT do

It does not compare mechanisms; it does not reopen B1-D, B1-F or the
deferred 300-frame comparison; it does not bear on the LRV3/LRV4 fixture
lanes; and **it does not license any mechanism comparison at the current
protocol.** N3V utility scaling remains HALTED per
[[membership-occupancy-and-decision-2026-08-23c]] §1 regardless of the
outcome here.

## 11. A defect found while designing this, recorded and NOT fixed

`_packet_ids` is manipulated by the prune and densify paths
(`scene/gaussian_model.py:1731`, `:1984`) but **does not appear in
`capture()`**, so a checkpoint does not carry it and any
branch-from-checkpoint on a B1 arm would silently lose the packet-id
column.

**This directly blocks paired-design options 1 and 2** (§G of the design
work), both of which branch from a common checkpoint. It is recorded here
rather than repaired because a fix would change training-path bytes and
this cohort's whole value rests on being byte-identical to the code that
produced the 0.4945 floor. **Any future paired design must repair this
first, and must not be submitted on a B1 arm until it is.**
