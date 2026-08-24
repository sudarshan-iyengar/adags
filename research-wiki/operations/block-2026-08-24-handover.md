# HANDOVER — 2026-08-24 block

Self-contained. EXPLORATORY throughout, `evidence_bearing: false` on every
cell. **This page is written progressively during the block; sections
marked IN FLIGHT were still running when it was last updated.**

## 1. State

| item | value |
|---|---|
| branch | `apollo/csvl-vpl-v2-exploratory` |
| block start | `bcc7cf0` |
| protected, untouched | `research-wiki/deep-dive-prompt.txt`, `run-deep-dive.ps1`, `agent-control/`, all `overnight-handover-*.md`, the pasted images, `sync 21-08-2026.md`, `supervisor-brief-2026-08-20.md` |
| pools | `hopper` 3x H100, `dgx` 6x V100 |

## 2. The decisive results

### 2.1 ImViD calibration is ADMITTED — the reprojection gate passed on all three frozen frames

Experiments **270 / 271 / 272**, `dgx`, V100 image, all `STATE_COMPLETED`.

| frame | pairs | cameras | **mean px @ NATIVE** | recorded COLMAP |
|---:|---:|---|---:|---:|
| 0 | 20,366 | 35/35 | **1.215442** | 1.1953 |
| 150 | 31,331 | 35/35 | **1.162289** | 1.1361 |
| 299 | 28,986 | 35/35 | **1.213650** | 1.1808 |

Gate is **2.0 px AT NATIVE**; all three pass with every training camera
contributing. Decisive rather than merely green because the measured
detection margins are enormous — transposed rotation **1400.52 px**,
camera-to-world pose **7850.60 px**, mis-ordered `distCoeffs` **56.77 px**,
dropped distortion **24.90 px**, against **3.03e-13 px** for the correct
pipeline. And the cross-check limb reproduces COLMAP's own residual
through an independent code path (1.198136 vs recorded 1.1953), sitting
slightly above it exactly as the undistortion Jacobian predicts.

**Consequence: the existing 20,157-point sparse union is reusable in the
undistorted PINHOLE frame WITHOUT re-triangulation** — previously an
argument, now a measurement.

**Documented limitation, in the instrument itself so it cannot be
mis-cited:** the gate is **exactly blind to the principal point**, because
that term cancels between the two sides. A separate equality check against
experiment 156's intrinsics covers it (measured delta `0.0`).

### 2.2 Spacetime Gaussian Grouping DOES NOT occupy the inferred-window cell

Read in full from the CC-BY journal version (the paywall was an IdP
redirect); the 6-page EUVIP version is genuinely unreachable and is
recorded as unread, not read. Identity is explicitly **time-independent**,
**no suppression is implemented in any form**, and **nothing is inferred**
— temporal parameters are free photometric parameters and labels come from
SAM + DEVA. The string "PSNR" occurs **zero** times in the full text.

**The standing forbidden-claim entry is DISCHARGED.**

**One recorded premise corrected, append-only:** "none of the seven
operates on a 4D representation with per-primitive temporal support" was
true of those seven and false in general — this method is per-primitive
identity on exactly our `(mu^tau, s^tau)` substrate family. Every
consequence drawn from that section survives, and the per-primitive-metrics
finding is *strengthened* (n+1 methods reporting only rendered-2D mIoU).

**One genuinely new mechanism:** its supervision is absolute-label
cross-entropy, not a within-frame contrastive term, so unlike the
contrastive family it **does** have a cross-episode supervisory path —
bounded by the authors naming disappearance-reappearance as exactly where
their pipeline fails.

**SA4D now replaces it as the highest-value remaining read.**

### 2.3 The ImViD bulk transfer is rate-limited PER-IP at ~62 GiB

Full record: [[imvid-acquisition-quota-2026-08-24]]. **21 files verified
complete, 62.149 GiB (5.7%), zero partials, zero stale locks, zero corrupt
files.** Diagnosed by a single 1-byte probe from a different host returning
`206 / video/mp4` for an untouched file while Apollo was still refused — so
the release is intact and the limit is tied to the requesting host.

**Whether it is a RATE limit or a DAILY VOLUME cap is UNDETERMINED**, and
it is the difference between ~1 day and ~18 host-days to finish. Read the
resume log; do not assume.

### 2.4 The full Opera take is 15,215 frames — the exposure gap is 560x, not 11x

Measured from the first completed full-take file: h264 5312x2988,
`60000/1001`, 253.84 s, **nb_frames 15,215**, SHA-256 recomputed on Apollo
matching the manifest exactly. Full split = 532,525 training units against
N3V-50f's 950. **The full take cannot be trained at any authorized
schedule**; a frozen event-selected tranche is now binding, not advisable.

## 3. Frozen specs written this block (all BEFORE their outputs existed)

* [[n3v-variance-study-spec-2026-08-24]] — run-level variance and endpoint
  validation. Two of the four levers were closed **by arithmetic on
  existing data before spending anything**: pooling more pixels is
  near-falsified (296x the pixel-times reduces the spread by **2.8%**, not
  ~17x, so the variance is a global run-level shift and not spatial
  sampling noise), and more replicates is unaffordable (**14/arm =
  68.1 slot-h**, 2.8x the block ceiling). δ\* = **0.30 dB**, grounded in the
  published N3V field span, with the uncomfortable corollary frozen
  alongside it that the recorded B1 effect (+0.211 dB) is **below** it. A
  within-run contrast endpoint is pre-registered as an n=3 **prediction to
  be tested**, with its complement-harm guard frozen at the same time.
* [[imvid-event-definition-2026-08-24]] — six conditions, a three-way
  A/B/C classification requiring **positive** evidence for genuine absence,
  and a synchronization rule. Amended append-only: the proxy census
  **cannot** satisfy that rule at scouting rates (2 fps ⇒ 500.5 ms steps,
  25x coarser than the 20 ms bound), which splits it into scouting then
  narrow-window measurement.
* [[nonconvex-hull-falsifier-spec-2026-08-24]] — gate frozen unweakened at
  precision ≥ 0.80, recall ≥ 0.90, zero false activations, with the second
  orientation predeclared so it cannot be chosen after the first result.

## 4. Instruments built and self-validated

| script | validation |
|---|---|
| `fetch_imvid_release.py` | live enumeration matches the recorded inventory **exactly** (325 files, 1,181,076,959,285 bytes), twice from different hosts |
| `imvid_verify_pinhole.py` | 13/13 on Apollo incl. the cv2 production path; corruption margins 1400 / 7850 / 57 / 25 px |
| `imvid_event_proxy.py` | 91/91, with byte-exact frame-index mapping verified against an independent ffmpeg decode |
| `build_nonconvex_reveal_scene.py` | **zero** predicate disagreements over 16,740,729 samples; reproduces the preflight's pixel counts it was never fitted to |
| `score_nonconvex_membership.py` | 55/55, pinning **both** directions (H1 fills the notch, H2 does not) |
| `nonconvex_hull_preflight.py` | 42/42, independently re-run |

## 5. Corrections to my own records, all append-only

1. The `/30` frame-period factor is **1.998** (`2000/1001`), not the 2.002 I
   wrote. Direction and conclusion unchanged.
2. The union rebuild **HAS** been run (experiment 164, 20,157 points,
   sha256 verified on Apollo); `imvid-baseline-freeze.md` A4 saying
   otherwise was stale.
3. N3V has **19** training cameras, not 20.
4. Adding a second concurrent transfer worker bought ~13% and the two arms
   tripped the quota four seconds apart — the acquisition rules' warning
   was right and my judgement was the weaker one.

## 6. Defects found and fixed

* **Stale per-file locks** in the downloader: released in a `finally`,
  which `SIGKILL` skips, so a killed worker would have blocked one file
  **permanently and silently** and the transfer would have reported success
  with a hole in it. Found by reasoning, not failure. Fixed with a 2 h
  staleness steal, tested four ways.
* **Non-resuming quota handling**: a refusal exited cleanly but did not
  recover. Now backs off 15/30/60/60/120 min (**4.75 h of patience**) and
  resumes from the recorded offset.

## 7. Defects found and NOT fixed, recorded

* **`_packet_ids` is absent from `capture()`** (`scene/gaussian_model.py`),
  so a branch-from-checkpoint on a B1 arm silently loses the packet-id
  column. **This blocks two of the three paired variance designs.** Not
  fixed because a fix changes training-path bytes that the current cohort's
  whole value rests on being identical to.
* **The Blender loader route fails OPEN** on distortion: it reads
  `fl_x/fl_y/cx/cy` with no camera-model field and no distortion field, so
  distorted images plus pinhole intrinsics train silently wrong. The
  `OPENCV` check everyone was watching fails *closed* and is harmless.
* **A mis-named point cloud is silently replaced by a random uniform fill**
  with no error raised.

## 8. Cost

All cells `evidence_bearing: false`. Transfers ran at **zero GPU slots**
throughout.

## 9. Exact restart commands

**Resume the transfer** (it self-heals, but if the task is gone):
```
python <scratchpad>/det_cmd.py dgx 0 <h100-digest> <scratchpad>/cmd_resume.txt --context=<scratchpad>/ctx --detach
```
The command inside is idempotent: completed files are skipped by byte
count, partials resume by `Range`, and locks are stolen after 2 h.

**Transfer status at any time:**
```
python scripts/fetch_imvid_release.py status --inventory /apollo/users/sri/proj_adags/data/imvid/imvid_drive_inventory.json --dest-root /apollo/users/sri/proj_adags/data/imvid/raw
```

## 10. Open blockers

* **The ImViD quota** blocks every full-take lane: the fixed-rig test on a
  complete take (which the frozen event definition says metadata cannot
  substitute for), the event census, and all ImViD training. Opera at
  15/39 cameras is deliberately **not** used for a partial census — a
  census needs multi-camera support and a 15-camera subset biases which
  candidates reach `C_min = 3` by download order.
* **N3V utility scaling remains HALTED** per the block-3 decision,
  regardless of this block's variance result.

---

# APPENDED (2026-08-24, later in the block)

## 11. The ImViD loader gap is CLOSED — converted, verified, and training

`scripts/imvid_to_blender.py` (experiment **277**, `dgx`, COMPLETED) turned
the COLMAP/OPENCV Opera sample into the Blender-convention layout this
trainer reads. **Its cv2 production path finally executed on Apollo — 16/16
self-test checks with no skip**, where the workstation could only manage
15/16.

Verified from the emitted manifest, not asserted:

| check | value |
|---|---|
| derived PINHOLE vs frozen experiment 156 | `matches = true`, **max abs delta 0.0** |
| `new_camera_matrix` | `"scaled_k"` — stated explicitly, never implicit |
| invalid border | `invalid_fraction 0.0`, 0 fully-outside pixels |
| frame rate | `60000/1001`, period `1001/60000`, ratio to the repo's hard-coded 30 fps **1.998001998** |
| point cloud | source sha256 == destination sha256 == `d5b10be0…`, 20,157 points, basename **exactly** `points3d.ply` |
| reader replay | train 105 frames / test 12 frames, **both take `per_frame_intrinsics(:433)`**, every referenced image exists, `max_abs_time_delta 0.0` |
| split | verified against the **WRITTEN json bytes**, 35 train / 4 held out |

**A correction found by measuring rather than inherited.** The working
audit reasoned that barrel distortion (`k1 < 0`) "pushes the periphery
outward" and so creates an invalid border. That describes the **forward**
map, not the inverse map `initUndistortRectifyMap` builds: an output pixel
at normalized radius `r` is fetched from `r(1 + k1 r²)`, which for `k1 < 0`
is **inward**. So the expected invalid fraction on Opera is ~0 — measured
`0.0` — and the real cost is the opposite one, that peripheral source
content is **discarded**. The converter reports both quantities and assumes
neither. **That reasoning never reached a durable record**, so there is
nothing to retract.

### 11.1 The smoke — the first ImViD training in this repository

Experiment **279**, `configs/imvid/opera_smoke500.yaml`, 500 iterations,
`dgx`. **The loader accepted the converted data and the trainer ran.**

**The load-bearing detail is `points = 20157`.** The Blender reader
silently substitutes a random uniform fill for a mis-named point cloud
(`scene/dataset_readers.py:481-491`) — and that fill would have read
**50,000** (the config's `num_pts`). It read 20,157, so the union cloud was
genuinely loaded. The silent-initialization failure this project already
paid for once on DiVa-360 did **not** recur.

`Reading Training Transforms` and `Reading Test Transforms` are logged
separately, so the split path is exercised rather than merged.

**This is a PLUMBING result and nothing more.** 500 iterations over 105
training units is 4.76 presentations/unit. No ImViD number from this cell
may be compared with anything.

Two config fields are held fixed and load-bearing, each justified against a
code line: **`eval: True`**, because `:475-477` MERGES the test split into
training when it is False — flipping it would silently violate the frozen
split; and **`resolution: 1`**, because `utils/camera_utils.py:43-46`
rescales the principal point by a naive divide rather than the frozen
`(c+0.5)*s-0.5`, so undistorting offline to the final raster keeps the two
conventions from ever meeting.

## 12. The hull falsifier: O1 is INVALID; O2 is running

Full record: [[nonconvex-hull-o1-result-2026-08-24]].

**O1 returned INVALID on precondition V3 — no verdict on hull completion
may be read from it.** T1's accepted component lay entirely along arm B (3
cells only-arm-B, **0** only-arm-A), so H1's per-component bounding box
never reached the notch and the operator could not fail.

**Read without V3 the table says hull completion SURVIVED** — precision
+0.0449, recall +0.1199, zero false activations, zero zero-object cells
filled. That near-miss is the finding, and it is LRV4's "a ratio without
its n is not a measurement" in a new costume: **an operator's pass is
meaningless until you check it was exposed to the case that would refute
it.**

Orientation **O2** was predeclared in the frozen spec before any fixture
existed, so running it is not selection of a favourable arm — but the
disclosure that O1 ran first and was invalid must travel with any O2
reading. Fixture generated (**276**); substrate training (**278**), under a
**distinct `config_canonical_hash`** so the two orientations can never be
confused in the ledger.

**If O2 also fails V3**, the conclusion is about the INSTRUMENT — that this
fixture design cannot expose H1 to the concavity at 8³ given T1's
selectivity — and **not** about hull completion. A future spec must then
force a spanning accepted component by construction rather than hope for
one.

---

# APPENDED (2026-08-24, late block) — lanes closed, and the block's methodological finding

## 13. THE HULL LANE IS CLOSED — both orientations INVALID, complementarily

Full record: [[nonconvex-hull-o1-result-2026-08-24]] and its O2 append.

| | cells only from arm A | cells only from arm B | verdict |
|---|---:|---:|---|
| **O1** (exps 273/274/275) | **0** | 3 | INVALID (V3) |
| **O2** (exps 276/278/285) | 3 | **0** | INVALID (V3, V4) |

The mirror flipped which arm T1 latches onto and **neither accepted
component ever spans both arms**, so H1's per-component bounding box never
reaches the notch. Two predeclared orientations failing the same
precondition *by opposite arms* makes this a property of the method, not a
quirk of one geometry.

**The conclusion was frozen before O2 ran** — the O1 record states
verbatim that a second V3 failure means the fixture cannot expose H1 at 8³
given T1's selectivity, *"a statement about the INSTRUMENT, not about hull
completion"*.

**Root cause, measured:** T1 gates **2 of 452** (O1) and **2 of 511** (O2).
**The extreme selectivity that makes T1's boundary inference exact, with
zero false activations, is what makes it unable to exercise a hull
operator on a non-convex object.**

**Hull completion is neither refuted nor supported.** V1 and V2 pass in
both, so the fixture is sound and the estimator is the limiting factor.

## 14. THE BLOCK'S METHODOLOGICAL FINDING — three vacuity catches, and the asymmetry between them

Three separate instruments this block produced a favourable-looking result
that could not have produced an unfavourable one:

| # | instrument | how it was caught |
|---|---|---|
| 1 | hull falsifier O1 | **pre-declared precondition V3** |
| 2 | hull falsifier O2 | **pre-declared precondition V3/V4** |
| 3 | densification-amplifier probe | **an anomalous invariant I happened to notice** (`points` unchanged) |

**The asymmetry is the finding.** The two hull catches were automatic: V3
is a statement about the *accepted set*, evaluated before any score, so it
cannot leak the outcome and it fires whether or not anyone is paying
attention. The third was caught only because the point count looked wrong
— and had it not, the probe's frozen rule would have delivered a clean,
publishable-sounding null from an instrument that never engaged its own
mechanism.

**I wrote the hull spec with a precondition and the probe without one, an
hour apart, in the same block.** The discipline did not transfer.

**The rule to carry:** *freezing a reading rule is not enough. Every
frozen rule needs a frozen precondition asserting that the mechanism it
reads was actually exercised — stated about the setup, never about the
score, so that checking it cannot leak the outcome.*

## 15. WAVE 1 ENDPOINTS — and the pre-registered prediction reverses

Wave 1 (exps 267/268/269, evals 286/287/288), n=3 of the planned 6:

| endpoint | a | b | c | spread | **sd** |
|---|---:|---:|---:|---:|---:|
| `all_events_union` | 31.9414 | 31.6975 | 31.7708 | 0.2439 | **0.125114** |
| `complement` | 33.1152 | 33.3974 | 33.0810 | 0.3163 | 0.173622 |
| `whole_frame` | 33.1106 | 33.3904 | 33.0759 | 0.3145 | 0.172409 |
| pooled+clamped PSNR | 33.4011 | 33.6731 | 33.3722 | 0.3008 | 0.166001 |
| **`union − complement`** | −1.1738 | −1.6999 | −1.3102 | 0.5260 | **0.272992** |

**The co-primary contrast endpoint's prediction REVERSES.**

| | union sd | contrast sd | contrast vs union |
|---|---:|---:|---|
| historical n=3 (261-263) | 0.262198 | 0.174523 | **−33.4%** (better) |
| **fresh wave 1** | 0.125114 | 0.272992 | **+118%** (2.18x WORSE) |

The spec registered the contrast as *"an n=3 hypothesis... The six fresh
runs TEST it; they do not assume it."* On the first three fresh runs it
does not reduce variance — it more than doubles it. **Registering it as a
prediction rather than adopting it was load-bearing**: adopting it on the
historical three would have built the next comparison on an endpoint that
is worse, not better.

**Equally telling:** the *primary* endpoint's sd differs by **2.1x**
between two n=3 cohorts of the identical protocol (0.262 vs 0.125). That
is the study's premise made visible — n=3 gives a sigma CI spanning
12.07x, so two n=3 estimates disagreeing by 2x is unremarkable and
**neither is usable.**

**No conclusion is drawn at n=3.** The spec makes n=6 primary and
evaluates the stopping rule there. Wave 2 = exps 289/290/291.

## 16. A DEVIATION FROM MY OWN SPEC, recorded BEFORE wave 2 ran

Wave 2 runs at a later commit than wave 1, and `scripts/submit_apollo.py`
— a member of the declared execution set — gained **4 allowlist strings**.
So the six cells **span two archives and may NOT be called
byte-identical**, contrary to the spec's §4. The numerical training path
is verifiably empty-diff.

Rejected: dropping to n=3 (leaves the study unable to answer its own
question), and reverting the allowlist to force an archive match (*editing
the repository so the record fits the protocol*). Recorded append-only at
`69a7795`, before submission and before any wave-2 number existed.

## 17. COST — actual, not projected

**Actual is 9.88 slot-h against 16.2 projected** for the same cells — my
projections ran ~40% conservative, which is the safe direction for a
ceiling guard. The one that ran OVER: wave 1 training cost **7.33 actual
vs 6.9 projected**. Eval cells are ~0.03 each, not the 0.4 projected.

Block total with wave 2 and its evals: **~17.4 slot-h actual against the
24 ceiling.** Transfers ran at **zero GPU slots** throughout.

## 18. Directive items NOT completed, and why

* **ImViD event census** — blocked by the Drive rate limit. Opera is 15/39
  cameras; a census needs multi-camera support and a 15-camera subset
  would bias which candidates reach `C_min = 3` by download order.
* **ImViD fixed-rig test on a full take** — needs a complete take.
  Metadata cannot substitute, per the frozen event definition §4.
* **A meaningful ImViD pilot** — a 50-frame tranche at N3V-equivalent
  exposure needs ~11k iterations at ~6.2x N3V's pixel count, i.e. 10-18
  slot-h. Priced and not affordable inside this block's remainder.
* **A paired mechanism experiment** — designed and deliberately not
  submitted; the directive's five preconditions do not all hold.
