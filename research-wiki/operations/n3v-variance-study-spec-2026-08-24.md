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

---

## AMENDMENT (2026-08-24, append-only) — the contrast endpoint needs 8 replicates/arm, not 7

Section 9's table gives **7** replicates per arm for the contrast endpoint
at delta* = 0.30. **That is one too few**, and it is corrected here rather
than left to propagate.

The raw power figure is `2 (z_a/2 + z_b)^2 (sigma/delta)^2 = 5.3125`. A
**sample size must round UP** — `round(5.3125) = 5` would plan for fewer
replicates than the calculation requires — so with the same +2 small-sample
correction the answer is `ceil(5.3125) + 2 = 8`. The union endpoint is
unaffected: `ceil(11.9910) + 2 = round(11.9910) + 2 = 14`, which is why the
error showed up in only one row.

Corrected cost for a two-arm comparison on the contrast endpoint:
**~39 slot-h**, not 34.1.

**No decision changes.** Both figures sit far below the union endpoint's
68.1 slot-h, and both readings of the contrast lane were already "the only
affordable route on the record". The correction makes the requirement more
conservative, never less, and it was found by
`scripts/n3v_variance_analysis.py`'s self-test failing against the
tabulated value — which is the reason that tool exists.

---

## DEVIATION FROM THIS SPEC, RECORDED BEFORE WAVE 2 RUNS (2026-08-24, append-only)

**Section 4 states the six cells are "byte-identical" and "differ only in
their cell name and retry index". Wave 2 will NOT satisfy that literally,
and the deviation is recorded here BEFORE wave 2 is submitted and before
any wave-2 number exists.**

Wave 1 (experiments 267/268/269) ran at commit `ebe9972`. HEAD has since
moved, and `scripts/submit_apollo.py` is a member of the declared
execution set, so wave 2's `archive_sha256` **will differ from wave 1's**.

**Exactly what differs — the complete diff of the execution set between
the two commits:**

```
scripts/submit_apollo.py | 4 ++++
+    "scripts/imvid_verify_pinhole.py",
+    "scripts/imvid_event_proxy.py",
+    "scripts/build_nonconvex_reveal_scene.py",
+    "scripts/imvid_to_blender.py",
```

Four entries appended to `ALLOWED_ENTRYPOINT_SCRIPTS`, a tuple of strings.

**The numerical training path is byte-identical, verified by diff, not
believed:**

```
git diff --stat ebe9972 HEAD -- elgs scene gaussian_renderer utils     arguments depth_visibility main.py det_exp_apollo.yaml     configs/n3v/ladder_b1_crb.yaml
  -> EMPTY   (0 files changed)
```

`submit_apollo` is imported at runtime only for `runtime_assertions()`,
which runs before training and performs assertions; the changed constant
is a tuple of allowed entrypoint names and enters no computation.

### Why wave 2 proceeds anyway, and the alternative that was rejected

**Rejected: reporting n=3 fresh only.** The whole purpose of this study is
that n=3 gives a 95% CI for sigma spanning a **12.07x ratio**, which is
unusable for any cost or power calculation. n=6 brings that to **3.93x**.
Abandoning wave 2 would leave the study unable to answer its own question.

**Also rejected: reverting the allowlist to force an archive match.** That
would be editing the repository to make the record fit the protocol, which
is the wrong direction of accommodation.

**Adopted:** run wave 2 and disclose precisely. This applies **the same
standard this spec already set in section 4.1** for pooling with the
historical cells — *"same training code, verified empty diff; different
archive"* — consistently, within the cohort rather than only across
cohorts.

### Binding consequence for how the result is reported

Every report of this cohort **must state that the six cells span two
archives**, must state that the numerical diff between them is verifiably
empty, and **must not describe the six as byte-identical**. The phrase
"byte-identical" in section 4 applies within each wave, not across the
two.

**No threshold, endpoint, exclusion rule or stopping rule is changed by
this deviation**, and none may be.

---

## ADVERSARIAL REVIEW RESPONSE (2026-08-24, append-only) — verdict MATERIAL DEFECT, accepted

A fresh-context adversarial review of this spec returned **MATERIAL
DEFECT**. The primary agent independently verified every load-bearing
finding below before accepting it. **This response is written while wave 2
is still TRAINING and before any wave-2 endpoint exists.**

The review's own summary of what it checked and found sound is on the
record too: the empty-diff claims, the pixel accounting, the chi-square
constants and the pre-registration timestamps all verified exactly, and it
credits the decision to *register* the contrast rather than adopt it as
the reason this study caught its own error. The defects are in the
reasoning around the measurement, not in the measurement.

### 1. BLOCKING — §7, applied as frozen, EXCLUDES wave 2. The deviation amendment's claim that no exclusion rule changed is FALSE.

**Verified.** §7 rule 2 excludes a run if *"any provenance field disagrees
with the cohort"* and explicitly names **`commit`** and
**`archive_sha256`**. §4 defines the cohort as the six cells. Wave 1 ran at
`ebe9972`, wave 2 at `69a7795`; archives `8aac8b96…` and `cef8a008…`. Rule
2 fires. §7 then states that if fewer than 5 of 6 survive, *"the study is
reported as **inconclusive at n=6**."*

The DEVIATION amendment above says *"No threshold, endpoint, exclusion rule
or stopping rule is changed by this deviation, and none may be."*
**That is false, and it is corrected here.** The rule was changed — by not
being applied. §7's entire design premise is mechanical application
("frozen, exhaustive", "if and ONLY if"), and its own warning is about not
excluding on outlier grounds; this is the mirror image, a rule that *would*
exclude quietly not firing.

**Disposition — BOTH readings are reported, and the STRICT one is
primary:**

* **PRIMARY, the frozen protocol applied mechanically:** wave 2 is
  excluded by rule 2. The fresh cohort is **n=3**, and **this study is
  INCONCLUSIVE AT n=6 by its own frozen rule.**
* **SECONDARY, explicitly labelled:** relaxing rule 2's `commit` and
  `archive_sha256` clauses to *"training-path diff verifiably empty"*
  admits all six. Any n=6 number reported under this reading **must carry
  that label**, and must not be described as the frozen protocol's result.

This is deliberately more conservative than the review's suggested fix,
which was to amend §7 and proceed. Amending an exclusion rule so that it
admits data is the direction of concern this project guards against, so
the frozen reading keeps primacy and the relaxation is offered beside it
rather than in place of it.

### 2. MY OWN "CORRECTION" WAS WRONG — 7 replicates/arm is right, 8 is not

The amendment above states *"the contrast endpoint needs 8 replicates/arm,
not 7"* and credits `n3v_variance_analysis.py`'s self-test with catching
an error.

**The tool caught an error it introduced.** `ceil(n) + 2` is not the
customary small-sample correction; the customary one (Guenther) is
`+ z²_{α/2}/4 ≈ 0.960`. Verified:

| endpoint | raw | `ceil+2` (published) | **`ceil(raw + z²/4)`** |
|---|---:|---:|---:|
| union | 11.9910 | 14 | **13** |
| contrast | 5.3125 | 8 | **7** |

The review confirmed by exact noncentral-t power simulation that 13 gives
0.7997 and 7 gives 0.8403, while 14 and 8 overshoot. **So the original 7
was right, reached by a wrong route (`round`), and my amendment fixed the
route and broke the answer.**

**Corrected figures**, at the measured 2.444 slot-h per training cell:

| endpoint | replicates/arm | two-arm training cost |
|---|---:|---:|
| `all_events_union` | **13** | **63.5 slot-h** |
| `union − complement` | **7** | **34.2 slot-h** |

All movement is in the conservative direction and **no decision flips**:
the union endpoint remains unaffordable at 2.6x the block ceiling.

§9's prose figures (109/arm, 475/arm) use yet a third convention
(`ceil+1`), which happens to be correct. **Three conventions appeared in
one section; only `ceil+1`/Guenther is right.**

### 3. THE CORRELATION — never computed, and it is the quantity the whole co-primary rests on

**Verified and now recorded:**

| cohort | sd(union) | sd(complement) | **r(union, complement)** | sd(contrast) | break-even r |
|---|---:|---:|---:|---:|---:|
| historical | 0.262198 | 0.255700 | **+0.7732** | 0.174523 | 0.4876 |
| fresh wave 1 | 0.125143 | 0.173645 | **−0.6610** | 0.273024 | 0.6938 |

The contrast beats the union **iff ρ > s_c/(2·s_u)**. The reversal is
**entirely** the correlation changing sign. Two consequences:

**(a) §2.2's stated mechanism is quantitatively wrong on its own data.**
*"A shift common to union and complement cancels in their difference"*
predicts ρ ≈ +1 and sd(contrast) ≈ 0. The measured 0.1745 implies
ρ = 0.773 — only ~60% of the variance common. The spec reported that
number without noting what it implied about the premise that produced it.

**(b) The argument conflated equal magnitude with common source.** Three
nearly-equal sds are equally consistent with ρ = 0. **Only the correlation
separates them, and it was never computed.**

**And the estimate could never have supported the weight placed on it:
the Fisher-z standard error for a correlation at n = 3 is 1/√(n−3) =
1/0 — UNDEFINED.** The co-primary endpoint was selected on a quantity
estimated at exactly the sample size where it has zero degrees of freedom.

**The fresh ρ = −0.661 is itself the most informative thing wave 1
produced, and §15 did not interpret it:** runs better on events are worse
elsewhere. That is a **capacity trade-off**, not a global shift — a
different mechanism with different consequences.

### 4. THE ARCHIVE DEVIATION WAS A FALSE DILEMMA — option (iv) exists

The deviation section presents an exhaustive-looking option set (drop to
n=3, revert the allowlist, disclose) that is **not exhaustive**.

**Verified:** `scripts/submit_apollo.py:1196` exposes `--repo-root`, and
`:972` derives the commit from it. So

```
git worktree add <tmp> ebe9972
python scripts/submit_apollo.py submit --repo-root <tmp> ...
```

would have produced a clean tree at exactly `ebe9972`, a **byte-identical
`git archive`**, and n=6 — with **no repository edit and no protocol
deviation at all**. It is neither of the two things the section rejected.

**This is the finding I most regret.** The stated principle (do not edit
the repository so the record fits the protocol) is correct; it simply did
not force the dilemma I built on it. **Not re-run**, because a second wave
2 costs ~7.33 slot-h and the block is at ~19 of a 24 ceiling — recorded so
a future block uses the worktree route from the start.

### 5. OTHER MATERIAL FINDINGS, accepted and recorded

* **δ\*'s "external grounding" is a category error.** The event union is
  **0.3377%** of pixel-times, so even an INFINITE union PSNR moves
  whole-frame PSNR by only **0.0202 dB** — 19x less than the 0.38 dB
  "field span" the anchor is measured in. The anchor cannot be transported
  to this endpoint in either direction. **§3 should be relabelled a bare
  judgment**; the "uncomfortable corollary" does not follow, and flips at
  δ* = 0.20. Credited: δ* was *not* reverse-engineered from the ladder
  deltas — it is ungrounded, not circular.
* **The complement-harm guard admits catastrophic arms.** Reusing δ* as a
  maximum tolerable harm on a different endpoint permits an arm at
  union +0.30 / complement −0.29 that passes both gates while being
  **0.287 dB worse whole-frame**. The guard must be set in the
  complement's own units.
* **§6 and §9 contradict each other.** §6 binds every cost calculation to
  the UPPER confidence limit for σ; §9's "only affordable route" uses
  point estimates. Under §6's own rule there is **no affordable route at
  all** (212/arm on the contrast at the n=3 upper limit).
* **σ_dec = 0.1672 is unaudited** and could not be reconstructed from the
  spec or the tool; it is the one frozen constant the self-test does not
  check.
* **The stopping rule is perverse and was not flagged.** A *large*
  contrast sd terminates the study — so the worse the co-primary performs,
  the less data is collected about the primary endpoint. Wave 1's contrast
  sd (0.2730) already sits 1.9% outside the continuation band. The rule
  stands as frozen; **its rationale is void** and that is disclosed here.
* **Lever 3 was overclaimed.** The 296x argument holds camera and frames
  fixed and enlarges only spatial support, while lever 3 as written names
  *"more held-out views or more frames"*. What is falsified is "pool more
  pixels inside the same 50 frames of the same camera". Also:
  `whole_frame` is a superset of `complement` (r = 0.999996), so the
  three-row table is two data points. The directional conclusion survives
  (paired, 0.9721 vs the 0.0581 √296 predicts); the label does not.
* **§2.5's delta-method "check" is a tautology** — a first-order Taylor
  identity that cannot fail. The conclusion stands on other grounds.

### 6. What is NOT retracted

No measured value changes. Every endpoint number, every empty-diff claim,
the pixel accounting, the chi-square constants and the pre-registration
timestamps were independently verified and hold. What changes is what may
be **concluded** from them.

---

## RESULT (2026-08-24, append-only) — the co-primary is REFUTED; sigma is finally estimable; and under §6's own rule there is still no affordable route

All six fresh cells and all six evaluations `STATE_COMPLETED`
(267/268/269 + 286/287/288; 289/290/291 + 292/293/294). Analysis by
`scripts/n3v_variance_analysis.py`, whose self-test reproduces the
recorded historical values.

### Reading 1 — PRIMARY: the frozen protocol, applied mechanically

Per §7 rule 2 and the review response above, wave 2 disagrees with wave 1
on `commit` and `archive_sha256`, both explicitly named exclusion fields.
**Applied as frozen, rule 2 excludes wave 2, fewer than 5 of 6 survive,
and this study is INCONCLUSIVE AT n=6 by its own rule.**

The surviving fresh cohort is **n = 3** (wave 1):

```
all_events_union   sd 0.125114        union - complement   sd 0.272992
```

**At n=3 the 95% CI for sigma spans 12.07x and is not a usable estimate**
— which is precisely the deficiency this study was created to remove, and
under the strict reading it is not removed.

### Reading 2 — SECONDARY, EXPLICITLY LABELLED: rule 2 relaxed to "training-path diff verifiably empty"

**Every number below carries that label and is NOT the frozen protocol's
result.** The training-path diff between the two commits is verifiably
empty; the archives differ by four strings in a tuple.

| endpoint | mean | **sd (n=6)** | spread | 95% CI for sigma | ratio |
|---|---:|---:|---:|---|---:|
| **`all_events_union`** (primary) | 31.67045 | **0.184681** | 0.4913 | [0.1153, 0.4530] | 3.93x |
| `complement` | 33.10017 | 0.158518 | 0.4682 | [0.0989, 0.3888] | 3.93x |
| `whole_frame` | 33.09444 | 0.158097 | 0.4670 | [0.0987, 0.3878] | 3.93x |
| pooled+clamped PSNR | 33.39009 | 0.149021 | 0.4331 | [0.0930, 0.3655] | 3.93x |
| **`union − complement`** (co-primary) | −1.42972 | **0.191296** | 0.5260 | [0.1194, 0.4692] | 3.93x |

**r(union, complement) over the six = +0.3867.**

### THE CO-PRIMARY IS REFUTED, at n=6 and not only at n=3

**sd(contrast) = 0.191296 > sd(union) = 0.184681.** The pre-registered
prediction was that the contrast REDUCES variance. It does not, at either
sample size — 2.18x worse at n=3, and still worse at n=6.

**The mechanism is now fully explained and it is the correlation.** The
contrast beats the union iff `rho > s_c / (2 s_u)`; here that threshold is
`0.158518 / (2 x 0.184681) = 0.4292`, and the measured **rho = 0.3867
falls below it.** The endpoint fails for exactly the reason the spec never
checked: it assumed a common run-level shift (which would give rho near 1)
and never computed rho.

**Registering it as a prediction rather than adopting it remains the
single best decision in this study.** Adopting it on the historical three
would have moved every future N3V comparison onto an endpoint that is
worse at both sample sizes.

### WHAT THE STUDY DID ACHIEVE — sigma is estimable

Question 1 of §0 was *"what IS the run-level standard deviation?"*, since
n=3 gives a 12.07x CI. **At n=6 the CI ratio is 3.93x**, and the union
endpoint's sigma is **0.1847 dB, 95% CI [0.1153, 0.4530]**.

**An independent convergence worth recording:** the n=6 union spread is
**0.4913 dB** against the historical replicate floor of **0.4945 dB** —
two disjoint cohorts, different commits, different archives, agreeing to
**0.7%**. The 0.4945 floor is corroborated rather than merely repeated.

### AND YET — under §6's own binding rule there is STILL no affordable route

§6 states that *"every downstream power and cost calculation uses the
UPPER confidence limit for sigma, not the point estimate."* Applying that
to the n=6 result, at delta\* = 0.30 dB and the corrected Guenther
correction, with the measured 2.444 slot-h per training cell:

| endpoint | point sigma | /arm | cost | **upper-limit sigma** | **/arm** | **cost** |
|---|---:|---:|---:|---:|---:|---:|
| `all_events_union` | 0.1847 | 7 | 34.2 slot-h | 0.4530 | **37** | **180.9 slot-h** |
| `union − complement` | 0.1913 | 8 | 39.1 slot-h | 0.4692 | **40** | **195.5 slot-h** |

**The point estimate makes a two-arm comparison look affordable at ~34
slot-h; §6's binding mitigation makes it 181 slot-h, 7.5x the block
ceiling.** The adversarial review flagged this contradiction before the
data existed, and the data confirms it. **The honest statement is that
this study has estimated sigma well enough to show that a two-arm N3V
comparison at delta\* = 0.30 is not affordable under the spec's own
uncertainty rule.**

### The stopping rule fires, its rationale is void, and n=9 is NOT run

The contrast CI [0.1194, 0.4692] straddles `sigma_dec = 0.1672`, so §6's
rule says **extend to n=9**. Three things are true at once and all are
disclosed:

* the rule is honoured as frozen — it says extend;
* **its rationale is void**, because it keys continuation to the contrast
  endpoint, which is now refuted at both sample sizes;
* **n=9 is not run.** Three more cells cost ~7.3 slot-h against a block
  already at ~19 of 24, and — decisively — extending would buy a better
  estimate of a **refuted** endpoint. Not running it is recorded as a
  deliberate departure from the frozen rule, with the reason, rather than
  as an oversight.

### Permitted and forbidden

**Permitted.** Under the strict frozen reading the study is inconclusive
at n=6 and the fresh cohort is n=3. Under the labelled relaxed reading,
sigma(union) = 0.1847 dB with a 3.93x CI, the co-primary contrast is
refuted at n=6, rho = +0.3867 explains why, the n=6 spread corroborates
the 0.4945 floor to 0.7%, and no two-arm comparison at delta\* = 0.30 is
affordable under §6's upper-limit rule.

**Forbidden.** Describing the n=6 numbers as the frozen protocol's result.
Calling the six cells byte-identical. Attaching a p-value to anything here
(§5.1). Treating delta\* = 0.30 as externally grounded — the review
established it is a bare judgment. And concluding anything about
mechanisms: **N3V utility scaling remains HALTED.**
