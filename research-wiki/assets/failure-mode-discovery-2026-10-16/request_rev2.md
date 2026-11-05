# Review request, round 2: failure-mode discovery plan rev 2 (ADAGS, 2026-10-16)

You reviewed rev 1 of this plan (your review: D:\adags\research-wiki\assets\failure-mode-discovery-2026-10-16\codex-review-rev1.md;
the rev 1 text: request_rev1.md in the same folder). Rev 2 is below. Section 10 maps each of your points to a
disposition. The census was re-run with corrected labels (tool D:\adags-ars\fmd\obs_census.py, outputs
D:\adags-ars\fmd\census_v2\ and the CSV D:\adags\research-wiki\assets\failure-mode-discovery-2026-10-16\census_v2_summary.csv).

Please answer briefly (under 700 words), marking each CRITICAL / MAJOR / MINOR:
1. Are your three CRITICAL points (narrow vs wide arc; label conflation; fallback) actually resolved? Check the v2
   label code (lines around "# 3. labels") and spot-check one census_v2 row against the table in section 1.2.
2. Is the new held-render counterfactual sound as a forgetting-vs-never-learned test for static background? What
   would break it?
3. Any new defect introduced by rev 2 (thresholds, dates, schedule, the evidence checkpoint, the novelty framing)?
4. Verdict: ACCEPT / ACCEPT WITH CHANGES / MATERIAL DEFECT, and the changes, if any, in order.

---- PLAN (rev 2) ----
# Failure-mode discovery plan (2026-10-16)

Status: **rev 2** (rev 1 reviewed by Codex gpt-6.1-sol, high effort, verdict "material defect"; every point is folded
in, see section 10). Append-only from here; revisions go in new sections.

Principal's brief (2026-10-16, in short): stop choosing by elimination; find a narrow failure ("in setting Y, X
breaks because Z") with the SOTA methods and data we have today; the failure probably lives where hidden regions
must be remembered or inferred (monocular, few cameras), not in dense capture; pick the setting, because averaged
PSNR hides the effect; stop rules; fight for the idea; benchmark and fix sketch; two-week schedule against CVPR 2027.

## 0. The plan in seven sentences

1. **Setting: two or three training cameras, placed so the held-out cameras sit inside their spread**; every other
   camera is held out for scoring. Not 4-6 cameras and not 20: today's exact census finds essentially nothing to
   remember there (section 1).
2. **Failure we hunt: what a method renders for a surface while no training camera sees it but a held-out camera
   does, and how that depends on how long it has been hidden.**
3. **Instrument: an observation-history label on every scored pixel** (seen now; seen before only; seen before and
   after; seen after only; never seen), with frames since last seen. DyCheck's co-visibility mask merges "seen now"
   with "seen at another time", and whole-frame PSNR averages over pixels that are mostly "seen now" (about 90-99 % of static background at 3 cameras), so today's
   benchmarks cannot show this failure.
4. **The first decisive measurement needs no new model code:** for static background, the model's own render at the
   frame the pixel was last seen is a free counterfactual ("did it know this surface and then lose it, or never
   learn it?").
5. Three hypotheses in the two weeks (a fourth, monocular, is deferred), each with a kill rule, a promote rule and an
   absolute deadline; an **evidence checkpoint on 24 October** and a **hard stop on 28 October**, with "no admissible
   CVPR claim" as an allowed outcome.
6. Cost: about 4 GPU-h of smoke tests, about 35 GPU-h for the first decisive runs, about 45 GPU-h of replication; all
   GPU work waits for the principal's approval.
7. CVPR (registration 10 Nov, paper 16 Nov, verified by Codex on the official dates page): realistic only for a
   measurement paper (benchmark + diagnosis) or for the failure whose fix already exists (the memory gate).

## 1. Which setting, and why

### 1.1 Evidence from the record

- **Dense real capture has almost nothing to remember.** On real N3V the hand occlusion offered the presence gate
  nothing to remove (contrasts within ~0.1 dB; at most ~0.8 dB available on the clean return frames). On DiVa-360,
  0 of 597 candidate windows were genuine absences.
- **The mechanism to watch is "paint, don't remove".** On the synthetic suite with 20 cameras, Multi4D keeps a
  departed object in its static-background bank (0.98-0.99 of its mass before the gap) and paints a transient
  occluder over it during the gap (transient alpha 0.88-0.99 on the ghost). With 20 cameras the painted patch must
  agree with 20 views; with 3, only with 3.
- **A dense-capture anomaly already exists in all three methods: the first frame after the object comes back**
  (section 3, H3). Its relevance to real footage is not established (the fixture switches the object's existence
  on in one frame).

### 1.2 New today: an exact census of what there is to remember (CPU, 0 GPU)

Tool: `D:/adags-ars/fmd/obs_census.py` (version 2, sha256 eda1a2f8..., tracked copy
`research-wiki/assets/failure-mode-discovery-2026-10-16/obs_census_v2.py`). For every 8th pixel (both directions)
of a held-out camera that shows the **static background** at a frame, it decides by ray casting from each training
camera to that same surface point - with every moving part present at that frame as a possible blocker - whether
a training camera sees it now, only earlier, earlier and later, only later, or never. 153 runs: the 9 wave-1 scenes
on the suite's N3V camera line (held-out cam00; plus the two end cameras for one 3-camera subset), and 6 scenes on a
probe arc (cameras on a circle of radius 10 around the table, -80 to +80 degrees, looking down; held-out cameras at
+44, -36, +04 degrees). Per-run summary: `assets/failure-mode-discovery-2026-10-16/census_v2_summary.csv`.

Checks: a training camera used as its own test camera gives 5,902,998 of 5,902,998 pixel-frames "seen now"; at the
busiest frame of one scene, an independent check from the renderer's stored identity buffers finds 244 pixels that
no training camera sees now, against the census's 71 memory + 168 never-seen + a few seen-after-only.

| Rig | Training cameras | Remembered (seen earlier), share of static-background pixel-frames: median (max) | of which seen before only / before and after | Remembered, as a share of the moving-object area | Never seen | Memory episodes of 10+ frames (all runs) |
|---|---|---|---|---|---|---|
| line | 19 | 0.000 % (0.000) | - | 0.00 % | 0.03 % | 0 |
| line | 6 | 0.002 % (0.034) | 0.000 / 0.002 | 0.02 % | 0.21 % | 213 |
| line | 3, subset a (cam00) | 0.100 % (0.214) | 0.048 / 0.067 | 1.2 % (max 2.4) | 0.83 % | 1,519 |
| line | 3, subset b (cam00) | 0.048 % (0.246) | 0.017 / 0.040 | 0.6 % (max 2.7) | 2.1 % | 1,165 |
| line | 3, subset a, end cameras held out | 0.25-0.26 % | 0.02-0.14 / 0.15-0.22 | 2.8-3.1 % | 1.8-11.5 % | 2,761-4,189 |
| line | 2 | 0.357 % (0.646) | 0.218 / 0.141 | 4.4 % (max 7.1) | 1.9 % | 3,541 |
| arc | 21 | 0.000 % (0.001) | - | 0.00 % | 0.22 % | 2 |
| arc | 6 | 0.004 % (0.035) | 0.000 / 0.004 | 0.04 % | 2.0 % | 425 |
| arc | 3 narrow (-16, 0, +16) | 1.04 % (1.59) | 0.12 / 0.73 | 8.9 % (max 11.8) | 19.1 % | 32,724 |
| arc | 3 wide (-80, 0, +80) | 0.85 % (1.68) | 0.10 / 0.73 | 7.0 % (max 13.1) | 8.5 % | 21,784 |
| arc | 2 wide (-80, +80) | 3.43 % (5.78) | 0.84 / 2.91 | 32 % (max 56) | 21.0 % | 62,498 |

What it says, within its limits (static background of these synthetic scenes, sampled pixels):

1. **Camera count is the main dial, and the threshold is low.** At 19-21 training cameras the remembered class is
   empty or near-empty on both rigs; at 6 it is still about 0.002-0.004 %. It opens at **3 or fewer**. "Sparse view"
   in the usual 4-6-camera sense is not where memory matters.
2. **Spread is not the second dial; rig geometry is.** The narrow and wide 3-camera arcs give similar remembered
   shares (1.04 vs 0.85 %); both give about 10 times the camera line, whose cameras are farther from the table and
   level with it. What spread does is cut **never seen** (19.1 % narrow vs 8.5 % wide), which is why held-out cameras
   must sit inside the training spread. Geometry has to be chosen on remembered share, never-seen share and whether
   the methods can still reconstruct, jointly.
3. **Most remembered pixels are seen again later.** "Seen before and after" dominates (0.73 of 0.85 % on the wide
   arc), so an offline method that trains on all frames can, in principle, interpolate them; "seen before only" is
   the strict memory case. Every claim must say which.
4. **Never seen grows as fast or faster.** At 3 cameras, an averaged metric is dominated by hallucination error, which
   generative-prior work occupies; the split is what makes a memory claim possible at all.
5. **Hidden stretches are long enough to matter:** 37-55 % of remembered pixel-frames are 31+ frames after the last
   sighting; thousands of episodes last 10+ frames.

Not covered yet: moving surfaces (the cook, rotator and target are unlabelled), real footage.

### 1.3 The recommendation

- **Primary: 3 training cameras, held-out cameras inside their spread.**
  - Synthetic (exact labels): the existing suite on its camera line, cut to 3 cameras - no re-render, smallest
    remembered class (about 1 % of the moving area) but enough episodes (1,500 of 10+ frames over 9 scenes).
  - Real: **N3V cut to 3 cameras** (the loaders and a released few-camera baseline, GC-4DGS, run on N3V as is), and
    **Panoptic Sports** (31-camera dome, 6 sequences of 150 frames, 716 MB on Leonardo; juggling and softball: balls
    and hands hidden and re-seen) if its adapter works within a one-day time box.
- **Deferred: one moving camera.** Synthetic track first, DyCheck iPhone only after approval; revisited after the
  commit. HyperNeRF vrig (on hand) scores on the stereo partner of the training camera, a small baseline: a loader
  smoke test, not a benchmark.
- **Inside the primary setting, not separate studies:** long horizons (frames since last seen) and viewpoint change
  (held-out camera's distance to the nearest training camera).
- **Second synthetic rig: cut** (Codex). The arc stays a census probe; a designed wide rig is built only if the
  camera line's remembered class proves too small to measure.

### 1.4 What would change my mind

- **Too small on real footage:** the real census (static background, proxy labels) at 3 cameras below 0.5 % of the
  moving area, or fewer than 100 episodes of 10+ frames per scene: the setting is synthetic-only; switch the primary
  to one moving camera (DyCheck download).
- **Generic, not memory:** remembered pixels degrade no more than matched pixels that stay seen (section 2): it is
  "few cameras are hard" (occupied by 4C4D, GC-4DGS, D2-4DGS). Drop the claim, keep the instrument.
- **Geometry, not forgetting:** the model's render at the last-seen frame is as wrong as at the hidden frame on
  held-out views: the surface was never reconstructed well, and the problem is few-view geometry, not memory.
- **No growth:** the 3-camera gap and return deficits equal the 20-camera ones: the dense failure is the paper.

## 2. The instrument and the scoring contract

| Label | Meaning | What it tests |
|---|---|---|
| seen now | a training camera sees this surface point at this frame | ordinary fitting (control) |
| seen before only | not seen now, seen earlier, never seen later | strict memory |
| seen before and after | not seen now, seen earlier and later | memory or interpolation (offline methods may use both sides) |
| seen after only | not seen now or earlier, seen later | non-causal |
| never seen | never seen by a training camera | inference; excluded from every claim |

Every remembered pixel also carries frames since last seen, and the event type: **occluded** (still in the world)
or **absent** (removed, as in the suite's gap). Episode durations count only frames in a remembered class.

- **Synthetic labels, exact:** census v3 adds moving surfaces (object-local hit coordinates with the part pose at
  each frame), full resolution, and per-pixel masks.
- **Real labels, proxy, static background only.** Depth from a dense model of the same sequence (all cameras),
  visibility by depth test with a tolerance band ("uncertain" pixels excluded). Valid only for surfaces that do not
  move; moving-surface history on real footage is out of scope (it needs independently checked tracks). Before use,
  the proxy is run on the suite and compared with the exact labels: precision and recall per class, coverage kept,
  agreement by gap length; plus one real event checked by eye. Below 90 % precision on remembered pixels, the real
  census is "unresolved".
- **Primary metric: masked PSNR (and masked L1) on exactly the labelled pixels.** The common evaluator's LPIPS is
  computed on a padded bounding box of the mask, not the mask (`eval_event_windows.py`, lines 150-190, 255-257), so
  it is reported only as "context-crop LPIPS"; a properly masked perceptual metric is optional.
- **Degradation score (positive = worse).** For remembered pixels, quality on the same pixels in the last 5 frames
  before they were hidden minus quality while hidden; minus the same quantity for control pixels (nearby background
  that stays seen) over the same frames. Requirements: the same surface point (exact on synthetic); the control's
  pre-trend checked (the score computed between two pre-hiding windows must be within noise); uncertainty by
  bootstrap over episodes within a run and over runs, not borrowed from whole-frame noise floors.
- **Held-render counterfactual (static background only).** Static background looks the same at every frame, so the
  model's render at the frame the pixel was last seen, scored against the hidden frame's ground truth, says what the
  model knew. Forgetting = good at the last-seen frame, bad while hidden. Never learned = bad at both.
- **Look at the scored pixels on every arm before reading any table** (2026-09-10 rule).

## 3. Failure hypotheses, ranked

Noise floors (for orientation only; each endpoint gets its own bootstrap): Multi4D whole-frame fixed-seed spread
0.34 dB; Ex4DGS sd 0.36 dB (n = 6); event-window paired sd 0.5-1.6 dB in earlier lanes.

### H1 (rank 1) - Hidden surfaces are forgotten

- **Claim.** With 3 training cameras, a static surface hidden from all of them is rendered progressively worse in
  held-out views the longer it stays hidden, because time-local primitives (Multi4D's transient bank, Ex4DGS's
  onset/offset opacity, STG's temporal opacity) are unconstrained while unseen and the occluder's own primitives,
  poorly placed in depth with 3 views, are painted onto it.
- **Measurement.** Degradation score by frames-since-seen bin (1-5, 6-15, 16-30, 31+), separately for "seen before
  only" and "seen before and after"; held-render counterfactual; bank-alone renders to name the bank; flicker on
  remembered pixels.
- **Mechanism controls.** Held render (forgetting vs never learned). Initialisation control: the same 3-camera run
  started from the training-camera cloud vs a cloud from all cameras on one scene (geometry vs memory; the
  all-camera cloud is a diagnostic only, never a result).
- **Data.** Suite same-pose scenes on the camera line, 3 cameras; N3V `cut_roasted_beef` and `flame_steak` at 3
  cameras; Panoptic juggle and softball if the adapter passes.
- **Promote.** Degradation of at least 3 dB in the 16+ bins, increasing with age (positive slope across bins,
  bootstrap interval excluding zero), same sign on 3 of 3 synthetic scenes and at least 1 real scene, in at least 2
  methods; held render says "forgetting"; artefact visible in a montage.
- **Kill.** Degradation interval within 1 dB of zero in every method; or the held render says "never learned" (then
  it is few-view geometry, occupied).
- **Fix it invites (sketch).** Hold unobserved primitives: freeze appearance and opacity of primitives with no
  training view instead of letting them be explained away, or route hidden surfaces to the persistent bank.

### H2 (rank 2) - Absence is read as occlusion, and 3 cameras make it worse

- **Claim.** Multi4D paints a transient occluder over a departed object rather than removing it (measured at 20
  cameras); with 3 cameras the patch need only match 3 views, so held-out views see the object around it.
- **Measurement.** Gap window (frames 63-87, ghost region) and return window on all held-out cameras at 20 and 3
  cameras; transient alpha over the ghost; static background as control; bootstrap over frames and runs.
- **Data.** Suite same-pose scenes (20-camera runs exist: gap 22.9-31.3 dB, base Multi4D, cam00).
- **Promote.** Gap PSNR falls at least 3 dB more than static-background PSNR from 20 to 3 cameras, 3 of 3 scenes, patch
  visible in the bank render. **Kill:** within 1 dB of the static fall.
- **Caveat (Codex).** Like H1, it can be geometry rather than misclassification; the bank render is descriptive. The
  decisive control is the memory gate itself at 3 cameras (an intervention), which needs estimated membership.
- **Fix it invites.** The memory gate (early gate: +9.2 dB gap at 20 cameras; return -0.10 dB with a known interior
  cost) with membership estimated from 3 views - **the only hypothesis whose fix exists**, but estimated membership
  is unbuilt and must be budgeted.

### H3 (rank 3) - The first frame after a reappearance

- **Measured today at 20 cameras** (45 local score files; held-out cam00; target region; deficit = steady return
  frames 92-99 minus frame 90; Codex recomputed the Multi4D row exactly):

  | Method (runs) | min | median | max | runs at 5 dB or more |
  |---|---|---|---|---|
  | STG, six 50-frame models (9) | 7.5 | 9.4 | 11.1 | 9 / 9 |
  | Multi4D run 1 (9) | 3.2 | 8.0 | 10.7 | 7 / 9 |
  | Ex4DGS (27) | 2.9 | 4.9 | 8.5 | 13 / 27 |

  Bottle same-pose: Multi4D 24.0 dB at frame 90, 31.7 at 91, 34.7 steady; STG 22.1 / 29.7 / 33.3. The early memory gate
  lifts frame 90 by +4.9 to +6.1 dB on Multi4D.
- **Why rank 3 (Codex).** The fixture removes the object from the world and switches it back on at frame 90: an
  existence step that real footage rarely has. And "smooth temporal support cannot switch in one frame" is
  contradicted by RetimeGS's report of very narrow Ex4DGS temporal fall-offs; training-view error would also fit an
  optimisation failure.
- **Test.** (a) Genuine re-reveals: background uncovered when the cook's arm or the occupant moves away, in the
  3-camera runs (labels from census v3), scored on the frame of re-reveal vs steady. (b) Training-camera renders of
  frames 88-92 for one run per method (under 1 GPU-h).
- **Promote** only if genuine re-reveals show at least 3 dB first-frame deficit in at least 2 methods. Otherwise H3 is
  recorded as a fixture property, descriptive.
- **Fix it invites.** A temporal onset that can switch at a learned instant; an intervention on temporal support is
  needed before any claim about representation.

### Deferred: H4 - one moving camera

Claim: moving-object surfaces seen only at other times are lost by time-local primitives; Multi4D lifts exactly those
surfaces into its transient bank. Deferred (Codex: cut first). When it comes back, it needs an actual canonical
model (SA4D or Shape of Motion) and a routing intervention, not a comparison of two Multi4D banks: bank-alone renders
drop complementary geometry and occlusion, so their difference is not an architecture comparison.

### Parked

- Never-seen regions: generative-prior territory (CAT4D, Vivid4D, CogNVS); sized and excluded.
- Ex4DGS late static-background collapse (21 of 65 runs) and ~9 dB object-texture loss: baseline caveats.

## 4. Stop rules

### 4.1 Per hypothesis (absolute dates)

| | Kill | Promote | Between kill and promote | Deadline |
|---|---|---|---|---|
| H1 | section 3 | section 3 | "unresolved": expires from the CVPR claim on the deadline | 24 Oct |
| H2 | section 3 | section 3 | same | 24 Oct |
| H3 | no genuine re-reveal deficit of 3 dB in 2 methods | section 3 | descriptive only | 23 Oct |

"Unresolved" is a valid record, never "failed" (project rule), but it cannot carry a CVPR claim.

### 4.2 When to stop exploring

- **No new hypothesis after 23 October**, unless a running test shows it directly (logged with the test).
- **Evidence checkpoint, 24 October.** Commit to one failure only if all hold: labels validated (synthetic exact;
  real proxy at 90 % precision or better), the failure promoted on synthetic **and** seen on at least one real scene,
  and a relevant few-camera baseline (GC-4DGS) run on the same cut.
- **Hard stop, 28 October.** One of exactly two outcomes: a committed failure that passed 4.3, or **"no admissible
  CVPR claim from this lane"**. A dense-capture fallback (H2 or H3 at 20 cameras) is allowed only if it passes 4.3
  on its own; it is not an automatic consolation.

### 4.3 Fight for it before committing

1. **Kill-argument** (`kill-argument` skill). Pass: no unresolved critical issue, or each one turned into a control
   that is run before the commit.
2. **Novelty check on the five-category rubric.** Expected: empirical/analytical discovery (3) plus
   benchmark/evaluation (5); with a fix, formulation (4) or synthesis (2). Codex's current rating: plausible 3/5,
   **novelty not established**. Must answer: non-trivial? generalisable insight (why it fails)? fair SOTA baselines
   at their own schedules? Nearest prior art to beat, with the point of difference to argue:
   - **PersistGS** (CVPRW 2026): hidden objects scored from withheld cameras with **five** training cameras; physics
     for a rigid tracked object. Difference to show: general surfaces, the observation-history split, the camera-count
     threshold, three SOTA 4DGS families.
   - **4D Primitive-Mache** (CVPR 2026, monocular permanence, qualitative); **DyCheck** (co-visibility mask merges the
     classes); **4C4D, GC-4DGS, D2-4DGS** (few-camera floaters/depth); **RetimeGS, Instant4D** (temporal opacity);
     **Temporally Aware Densification** (arXiv 2606.23212, multi-view densification of short-lived Gaussians, no
     history split; checked today); **USplat4D, GP-4DGS** (uncertainty for occluded regions; full texts unread).
   - The claim "no paper stratifies by visibility history" is from one scan and must be re-searched in the novelty
     check, not asserted.
3. **Real footage:** at least one real scene shows the failure.
4. **The sentence:** "Everyone does X; with 3 cameras, a surface hidden for k frames is forgotten at Z because W; the
   observation-history split shows it and whole-frame PSNR cannot."

## 5. From a surviving failure to a benchmark and a fix

| Failure | Benchmark | Baselines (own published schedules, one evaluator) | Fix it invites (sketch) |
|---|---|---|---|
| H1 | Three-camera memory benchmark: suite scenes on the camera line (exact labels) + N3V and Panoptic at 3 cameras (proxy labels, static background); windows = remembered pixels by age bin and by before-only / before-and-after, re-reveal frames, matched control | Multi4D, Ex4DGS, STG (one model over the sequence), GC-4DGS (released few-camera code), SA4D as a canonical-deformation reference | hold unobserved primitives; visibility-conditioned temporal prior |
| H2 | Suite gap and return windows on all held-out cameras at 20 and 3 cameras | same | memory gate with estimated membership |
| H3 | Re-reveal transitions, training and held-out cameras | same | learned-instant temporal onset |

Few-camera rules: published schedules; only the camera list changes; the initial cloud is rebuilt from the training
cameras only. Engineering facts checked today: Multi4D's N3V loader holds out the first camera of the folder and
needs **exactly 300 frames** with one shared focal length (`scene/neural_3D_dataset_NDC.py:45-112` in the pinned
tree), so the suite and N3V cut cleanly by symlinked folders, while 150-frame Panoptic needs another route (its
COLMAP-style "technicolor" reader, unchecked) - hence the one-day time box. Scoring more held-out cameras needs a
render wrapper (the pattern of `agent-control/ars/v9_run.py`).

## 6. First two weeks (17-28 October)

| Days | Work | GPU-h | Approval |
|---|---|---|---|
| 17 Oct | Census v3 (moving surfaces, full-resolution per-pixel masks); scoring contract coded (degradation score, held render, pre-trend, bootstrap, masked PSNR/L1); dry run of the scorer on existing 20-camera renders | 0 | no |
| 17-18 Oct | 3-camera trees on the camera line (Multi4D, Ex4DGS) and N3V; training-camera COLMAP init; held-out render wrapper; GC-4DGS install; 2 debug-queue smoke runs; H3 training-camera renders | ~4 | **yes** |
| 18-19 Oct | Real proxy labels for N3V (dense-model depth), validated on the suite; Panoptic adapter, one-day time box | ~2 | **yes** |
| 19-21 Oct | **Decisive runs:** Multi4D + Ex4DGS + GC-4DGS on the 3 same-pose suite scenes at 3 cameras (about 20 GPU-h); Multi4D + GC-4DGS on N3V `cut_roasted_beef` and `flame_steak` at 3 cameras (about 11 GPU-h); Panoptic if the adapter passed (about 4) | ~35 | **yes** |
| 22-23 Oct | Readouts, montages, held renders, kill/promote; H3 re-reveal check; hypothesis list closes 23 Oct | 0 | no |
| 24 Oct | Evidence checkpoint | 0 | - |
| 24-28 Oct | Replicates of the committed failure (3 runs x 3 scenes x 2-3 methods), kill-argument, novelty check; hard stop 28 Oct | ~45 | **yes** |

Total about 85 GPU-h. The critical path is the instrument and the adapters, not GPU hours. Submissions go through
`agent-control/ars/ars_sub.sh` with the ledger, under an `fmd_` prefix; no file or job of the early-gate memory-gate
lane is touched; debug queue allows 2 jobs.

**CVPR.** Registration 10 Nov, paper 16 Nov. If H1 is committed on 24 Oct, CVPR can carry the measurement
(three-camera memory benchmark + diagnosis with held renders) - a category 3/5 paper - and the fix is the ICCV 2027
paper. If H2 is committed, the paper needs estimated membership for the memory gate, which is unbuilt: budget at
least a week of code and review on top, so it is CVPR-feasible only if that work starts on 24 Oct.

## 7. Verified today vs from the record vs assumed

- **Verified in this session (tool output):** the census (153 runs v1, 153 runs v2, two self-checks); the H3 table
  (45 local score files; Codex recomputed the Multi4D row and the 23.991 dB bottle frame-90 value); the camera-line
  geometry (about 11.4 units wide, nearest depth about 8.7); renderer options and cost (about 23,000 CPU-seconds per
  20-camera scene); Multi4D's loader rules; the data inventory (read-only listing by a subagent on Leonardo, Apollo
  and D:); GC-4DGS has released training code for N3V/Technicolor (repository page); Temporally Aware Densification's
  scope (abstract).
- **From the record, not re-run:** bank masses, memory-gate effect sizes, Ex4DGS collapse counts, the real-occlusion
  null, GPU timings, the remaining allocation (about 15,500 GPU-h, measured 2026-09-21).
- **Assumed, checked first:** Ex4DGS and GC-4DGS train on a 3-camera N3V-style tree; a 3-camera COLMAP init succeeds;
  the real proxy labels pass validation; Panoptic converts within a day; dataset sizes for DyCheck/NeRF-DS.

## 8. Data inventory (read-only, 2026-10-16)

| Data | Where | Use here | Loaders |
|---|---|---|---|
| Suite wave 1 (9 scenes, 20 cameras, 300 frames, exact identity buffers) | `D:/adags-ars/wave1`; Leonardo scratch | exact labels; any camera subset | Multi4D, Ex4DGS, STG |
| N3V, 6 scenes, 20-21 cameras, 300 frames | Leonardo D21/D36, Apollo, `D:/adags/data` | real 3-camera cuts | all; GC-4DGS |
| Panoptic Sports, 6 sequences, 31 cameras x 150 frames, 640x360, masks for 27 cameras | Leonardo D21 `data/panopticsports` (716 MB) | real wide 3-camera cuts | SA4D only; others need an adapter |
| HyperNeRF vrig `broom2` (masks), `3dprinter` | `D:/4DGaussians/data/hypernerf/virg` | monocular loader smoke test | Multi4D (`nerfies`), SA4D |
| ImViD, DiVa-360 (Apollo only), Technicolor (D: only), Google Immersive | various | not in the two weeks | various |
| NeRF-DS, DyCheck iPhone, DAVIS, Kubric | absent everywhere | DyCheck only if the moving-camera lane returns, after approval | - |

## 9. Literature position (scan of 2026-10-16; venues unverified where marked in the scan)

Few-camera 4D papers target floaters, depth ambiguity and flicker; monocular papers report full-frame or
co-visibility-masked metrics, with disocclusion failures described qualitatively; PersistGS is the quantitative
persistence paper (five training cameras, withheld-camera scoring, rigid tracked object). Nothing found yet splits
quality by when a pixel was last seen, or compares opacity decay while hidden across Ex4DGS, STG and Multi4D; to be
re-searched in the novelty check.

## 10. Review record (Codex gpt-6.1-sol, high effort, 2026-10-16) and what changed

Request and review: `assets/failure-mode-discovery-2026-10-16/request_rev1.md`, `codex-review-rev1.md`.
Verdict on rev 1: MATERIAL DEFECT. Codex spot-checked two numbers (the wide-arc census row and the Multi4D H3 row);
both matched.

| # | Codex point | Severity | Disposition |
|---|---|---|---|
| 1 | Narrow 3-camera arc (1.04 %) beats the wide one (0.85 %); "spread multiplies it" is wrong | critical | accepted: 1.2 rewritten; spread mainly cuts never-seen; geometry chosen on three criteria |
| 2 | Sampled static-only census cannot support "dense has none"; dense arc has 29 pixel-frames | major | accepted: claims restricted to sampled static background; "near-empty" |
| 3 | "Seen only earlier" included points seen again later; hidden runs counted never-seen | critical | accepted and fixed: census v2 (five labels; episodes over remembered frames only); all numbers re-run |
| 4 | Deficit sign contradicts thresholds; cancellation overstated | major | accepted: positive degradation score; same surface; pre-trend test; bootstrap |
| 5 | Evaluator LPIPS uses the mask's bounding box | major | accepted (verified in the code): masked PSNR/L1 primary; LPIPS labelled context-crop |
| 6 | H3 fixture is an existence step; RetimeGS shows narrow fall-offs | major | accepted: H3 demoted; genuine re-reveal test; no representation claim without intervention |
| 7 | H1 promotion lacked an age test; H4 compared banks, not architectures | major | accepted: slope test; H4 deferred and needs a canonical model + routing intervention |
| 8 | Fallback defeated the stop rules | critical | accepted: absolute deadlines; "no admissible CVPR claim" allowed; fallback must pass 4.3 |
| s1 | Geometry/optimisation vs forgetting | major | accepted: held-render counterfactual + initialisation control |
| s2 | Real moving-surface history unspecified | major | accepted: real labels limited to static background; proxy precision/recall gate |
| s3 | Multi4D loader needs exactly 300 frames, one focal | major | accepted (verified locally, pinned tree): Panoptic time-boxed; N3V added as the guaranteed real lane |
| u1 | Remote inventories, timings and record numbers not independently checked | major | noted in section 7 |
| u2 | "Only quantitative one" is an exhaustive claim | major | accepted: to re-search in the novelty check |
| m1-m3 | Validate the instrument first; executable novelty (GC-4DGS, Temporally Aware Densification); evidence checkpoint around 23 Oct; cut H4 and the second rig; registration 10 Nov | - | accepted: schedule reordered; GC-4DGS budgeted; checkpoint 24 Oct; both cut |

## Glossary

Suite = synthetic Absence-Return Suite (ARS). Camera line = the N3V camera layout the suite was rendered with. Same-pose
return (A0). Gap window (P1, frames 63-87, ghost region); return window (P2, frames 92-99); first return frames (T,
90-91). Memory gate = CER; early gate = CER from iteration 3,000. Remembered = seen before only, or seen before and
after.
