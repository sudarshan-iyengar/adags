# RESULT — B1-F flow screen: the flow BIRTH prior is REJECTED, and
# correct flow did not beat wrong flow (2026-08-23)

EXPLORATORY, `evidence_bearing: false`. Design, arms and gates frozen
before any output in [[b1f-flow-birth-prior-spec-2026-08-23]]; launch
rule and preflight in [[b1f-preflight-result-2026-08-23]]. All six arms
and their evaluations are TERMINAL. Sections 2-5 were written while the
screen was still running, so the machinery findings are on the record
independently of the metrics in §6.

## 1. Cells

| arm | seed | exp | val exp | training state |
|---|---:|---:|---:|---|
| plain B1 (fresh on-pool comparator) | 0 | 237 | 251 | COMPLETED |
| plain B1 | 1 | 238 | 252 | COMPLETED |
| **B1-F** correct flow | 0 | 239 | 253 | COMPLETED |
| **B1-F** correct flow | 1 | 240 | 256 | COMPLETED |
| **B1-X** camera-swapped wrong flow | 0 | 241 r0 ERROR → **254 r1** | 257 | COMPLETED |
| **B1-X** camera-swapped wrong flow | 1 | 242 r0 ERROR → **255 r1** | 258 | COMPLETED |

Plain-B1 and B1-F at commit `789595a`, the B1-X reruns at `b4146d5`
(see §4 for why, and for the inertness argument). Pool `hopper`, admitted H100 image
`sha256:0d577168…`, 6,000 iterations, frames 0-49, cam00 sealed,
`route_logit_init: 4.0`, `elgs_reserved_parity: true`, 600k cap.

**Seeds are genuinely different in this family** — `--seed` is passed
through to the trainer as `--extra-arg=--seed`, verified in the rendered
entrypoint (`main.py … --seed 0 --test_iterations 6000`). That is a
departure from the historical ladder, where the wrapper's `--seed` never
reached `main.py` ([[seed-threading-defect-2026-08-23]]), and it is why
this screen trains its OWN plain-B1 comparators rather than reusing
experiments 197/200.

## 2. GATE CONDITION 6 (machinery health) — PASSES

The frozen gate requires flow to have actually applied, with no
missing-flow fallback, no invalid-unit pathology, and
`flow_sites_valid > 0` at every event. Read from `packet_state.pt`,
experiment 239, all seven birth events:

| iter | sites | valid | %valid | mask | bounds | non-finite | outlier | mean speed | coeff norm | **realized ratio** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 2174 | 2103 | 96.7 | 14 | 0 | 0 | 57 | 0.02051 | 0.0755 | **1.0000** |
| 1500 | 2540 | 2466 | 97.1 | 42 | 0 | 0 | 32 | 0.02108 | 0.0662 | **1.0000** |
| 2000 | 2873 | 2777 | 96.7 | 54 | 0 | 0 | 42 | 0.01335 | 0.0322 | **1.0000** |
| 2500 | 2998 | 2930 | 97.7 | 24 | 0 | 0 | 44 | 0.00969 | 0.0194 | **1.0000** |
| 3000 | 2998 | 2877 | 96.0 | 71 | 0 | 0 | 50 | 0.01672 | 0.0366 | **1.0000** |
| 3500 | 2997 | 2892 | 96.5 | 49 | 0 | 0 | 56 | 0.01408 | 0.0231 | **1.0000** |
| 4000 | 2998 | 2828 | 94.3 | 28 | 0 | 0 | 142 | 0.02669 | 0.0366 | **1.0000** |

* **94.3-97.7% of birth sites received valid flow at every event** —
  far above the 50% floor, and never zero.
* **`flow_realized_ratio_mean` is exactly 1.0000 at every event**: the
  LoRA probe basis reproduces the requested displacement in full, so the
  cell is measuring flow rather than basis rank-deficiency. That was the
  specific failure mode Appendix B's funnel fields were added to expose.
* **Zero out-of-bounds and zero non-finite failures** across all seven
  events. The only per-row rejections are the validity mask (14-71) and
  the 99th-percentile outlier rule (32-142, i.e. 1.5-4.7%).
* Coefficient norms are **0.019-0.076** — small, as the measured flow
  field predicts (p50 0.06 px, p99 3.1 px), and two orders of magnitude
  below the ~1.6 population median that the withdrawn magnitude guard
  would have clamped against. This confirms on real data what
  [[b1f-preflight-result-2026-08-23]] §3 recorded: **the original guard
  would not have bound here, so the amendment's urgency was overstated
  even though its reasoning was right.**

## 3. Capacity parity and flag-off behaviour — both verified

| arm | rows relocated over 7 events |
|---|---:|
| plain B1 (exp 237) | 19,538 |
| B1-F (exp 239) | 19,578 |

**Within 40 rows**, so the arms are capacity-matched and the only
difference is what the relocated rows' motion coefficients start at.
Recorded ladder B1 relocated 19,555 / 19,568 rows, so the fresh on-pool
comparator **reproduces the recorded operator's behaviour** despite the
pool and seed changes.

Flag-off behaviour verified from the artifact rather than assumed: the
plain-B1 record carries `flow_init: false` with every flow counter at
zero — `flow_sites_total 0`, `flow_mean_speed 0.0`,
`flow_realized_ratio_mean 0.0`. The funnel is written on EVERY record,
including arms where flow is off, which is what makes "a run whose flow
never applied" detectable from the record alone.

## 4. B1-X FAILED CLOSED, exposing two defects — repaired, rerun at retry 1

Experiments **241** and **242** — BOTH camera-swapped arms — reached
`STATE_ERROR` at the first birth event with

```
ContractError: the camera_swapped flow control needs a training-camera
roster; pass train_cameras to build_flow_assets
```

**Diagnosis.** `Scene.getTrainCameras()` returns a `CameraDataset` whose
`__getitem__` yields `(image, camera)` **TUPLES**
(`utils/data_utils.py:17-35`), so iterating it handed the roster builder
objects with no `image_name`. The trainer wiring passed the dataset
rather than its `viewpoint_stack`.

**The second defect is the serious one, and the crash is what exposed
it.** `_camera_ids` SKIPPED unnamed entries silently, so the wiring
mistake produced an **EMPTY roster** rather than an error. An empty
roster fails closed for the camera-swapped control — which is why B1-X
crashed — but leaves `assert_not_holdout` **INERT** for the correct arm.
**The held-out guard this page's §1 describes as structural was, in the
B1-F cells that already ran, silently protecting nothing.**

**No leakage occurred, and that is verified rather than argued.** The
birth record for experiment 239 lists all seven birth cameras:
`cam13_0035, cam20_0044, cam02_0004, cam07_0001, cam10_0023, cam08_0000,
cam16_0025` — **never `cam00`**. The trainer only ever presents training
cameras as the birth view, so the guard was redundant, not violated.

**Repairs (technical; no scientific semantics changed):** `main.py` now
passes `.viewpoint_stack`; `_camera_ids` fails closed on an unnamed entry
AND on any non-empty roster that yields no ids, so a guard can never
again degrade silently to "protects nothing". Four regression tests
added; 142 pass.

**A third defect, in the test suite itself.** The flag-off bit-identity
test pinned its baseline to `git show HEAD:scene/packet_birth.py`, so it
**self-invalidated the moment the flow work was committed** — the
"previous implementation" became the post-change code and the
non-vacuity assertion (that the baseline REJECTS the new keyword) began
failing. A bit-identity test whose reference moves with the branch cannot
prove anything. The baseline is now the fixed pre-flow commit `f666dd7`.

**Cross-commit comparability of the reruns.** B1-F (239/240) ran at
`789595a`; the B1-X reruns (**254**/**255**, retry 1) run at `b4146d5`.
The repair is inert for the CORRECT arm by construction: with an empty
roster `assert_not_holdout` never raised, and with a populated roster it
checks and passes because a training camera is not held out — same
outcome, no state change. The removed side effect (iterating the dataset
loaded every image) touches no RNG and mutates no camera. Recorded so the
cross-commit comparison is explicit rather than assumed.

Claims consumed and preserved: `ladder_b1x_crb_s{0,1}` r0 (241/242,
ERROR, never reused) → r1 (254/255).

## 5. The first genuinely-different-seed spread this project has measured

The plain-B1 comparators are the first cells in this repository to train
at genuinely different seeds, because `--seed` is now threaded to the
trainer ([[seed-threading-defect-2026-08-23]]).

| arm | pooled+clamped PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| plain B1 seed 0 (exp 237) | 33.3941 | 0.95804 | 0.08184 |
| plain B1 seed 1 (exp 238) | 34.0294 | 0.96065 | 0.08095 |
| B1-F seed 0 (exp 239) | 33.2568 | 0.95830 | 0.08240 |

**The seed-to-seed spread is 0.635 dB** — more than double the ~0.27 dB
run-to-run spread measured between fixed-seed replicates, and 2.3× the
±0.28 figure the ladder era cited as its "seed spread".

**This materially raises the bar for every 50-frame comparison in this
family.** An effect smaller than ~0.64 dB is not resolvable by two seeds
at this protocol. For reference, B1-F seed 0 sits **−0.137 dB** from
plain B1 seed 0 — comfortably inside that band, i.e. indistinguishable
from noise on the evidence of one seed pair.

Recorded now because it is independent of the screen's outcome and
because it bounds what any future two-seed 50-frame cell can claim.

## 6. RESULT — the flow birth prior is REJECTED, and wrong flow scored HIGHER than correct flow

All six arms terminal and evaluated. Event regions scored with
`scripts/event_ray_metrics.py` on the FROZEN
`configs/n3v/ladder_event_masks_crb0_49.json`, 8-bit saved-render basis;
`--val` at `chkpnt6000` for every arm, one convention throughout.

| arm | event union | A | B | C | complement | whole |
|---|---:|---:|---:|---:|---:|---:|
| plain B1 s0 | 31.7734 | 37.4977 | 36.2967 | 27.2202 | 33.1114 | 33.1061 |
| plain B1 s1 | 31.4320 | 36.5831 | 35.6523 | 27.0166 | 33.7827 | 33.7722 |
| B1-F s0 | 31.6853 | 37.4036 | 36.1172 | 27.1416 | 32.9718 | 32.9667 |
| B1-F s1 | 31.5569 | 37.1409 | 36.0359 | 27.0324 | 33.5731 | 33.5644 |
| B1-X s0 | 31.5871 | 37.1788 | 35.7321 | 27.0870 | 33.0233 | 33.0176 |
| B1-X s1 | 31.8456 | 36.8502 | 35.7691 | 27.4959 | 33.2892 | 33.2834 |

Paired deltas on the event union:

```
B1-F − B1    s0 −0.0881   s1 +0.1249   mean +0.0184
B1-X − B1    s0 −0.1863   s1 +0.4136   mean +0.1136
B1-F − B1-X  s0 +0.0982   s1 −0.2887   mean −0.0952
```

### The frozen gate, applied

| condition | outcome |
|---|---|
| 1. event union `B1-F − B1` ≥ 0 on BOTH seeds | **FAIL** (s0 −0.0881) |
| 2. paired mean event effect > 0 | pass (+0.0184) |
| 3. **B1-F > B1-X on the event union** | **FAIL** (mean −0.0952) |
| 4. complement paired mean ≥ −0.10 dB | **FAIL** (−0.1746) |
| 5. global non-catastrophe ≥ −0.30 dB per seed | pass (−0.139, −0.208) |
| 6. machinery health | pass (§2) |

**VERDICT: the flow BIRTH prior is REJECTED.** The frozen attribution
rule is decisive on its own: *if correct flow does not beat wrong flow on
the event endpoint, the result is UNATTRIBUTABLE and the prior is
rejected regardless of how B1-F compares to plain B1.* **It does not.
Camera-swapped flow scored HIGHER than correct flow** (paired mean
−0.0952 for B1-F − B1-X).

### The honest reading: everything here is noise

Every delta flips sign across seeds, and every magnitude sits far inside
the measured spread. The plain-B1 event-union seed spread is
**31.7734 vs 31.4320 = 0.341 dB**, and the global spread is 0.635 dB
(§5). Against that:

```
|B1-F − B1|    = 0.018   (5% of the union seed spread)
|B1-X − B1|    = 0.114   (33%)
|B1-F − B1-X|  = 0.095   (28%)
```

**No arm is distinguishable from any other.** The correct statement is
not "wrong flow is better" — it is that **the flow-derived velocity
initialization has no measurable effect at this protocol**, and the
wrong-flow control demonstrates that by scoring indistinguishably.

### This REPLICATES a prior project finding, and that is the durable part

CSVL-VPL Stage 1 (2026-07-26) measured camera-swapped flow scoring
**above** valid flow (0.922 vs 0.889) and treated it as a no-go. **This
is the second independent occasion in this project on which
camera-swapped flow has matched or beaten correct flow.** The control was
mandatory here precisely because of that precedent, and the precedent
held.

### What this closes

The BIRTH-prior role was the **last live zero-acquisition prior
experiment** ([[prior-asset-inventory-2026-08-20]] decision 4): the
LINEAGE-prior role is blocked behind an N3V tracks adapter, and this was
the one prior use that needed no new acquisition. **It is now closed
with a terminal negative rather than left open**, which is what justified
spending the cells on a pre-registered likely null.

**Forbidden by this result:** any claim that flow improves packet birth,
and any claim that wrong flow is *better* — the arms are not separable.
**Not established:** that flow is useless as a prior in general. What is
refuted is flow-derived velocity initialization for relocated packets at
the 50-frame protocol, on masks now known to score mostly static pixels
(Appendix C), with a measured flow field small enough (p50 0.06 px) that
a null was pre-registered as the likely outcome.

## 7. What is NOT established

**Everything the screen exists to decide.** No metric has been read. The
decisive comparison is **B1-F vs B1-X**: if correct flow does not beat
camera-swapped flow on the event endpoint, the result is UNATTRIBUTABLE
and the flow birth prior is rejected regardless of how B1-F compares to
plain B1 — a rule that exists because CSVL-VPL Stage 1 measured
camera-swapped flow scoring ABOVE valid flow.

Two pre-registrations bear on how a null must be read, both recorded
before any output: the measured flow field is small, so a null is
**likely and legitimate** ([[b1f-preflight-result-2026-08-23]] §4); and
the 0-49 masks this screen scores on are **not confirmed by ground
truth** and score mostly static pixels, raising the false-negative risk
([[b1f-flow-birth-prior-spec-2026-08-23]] Appendix C).

**Metrics and the frozen gate application are appended to this page when
all six arms and their evaluations are terminal.**
