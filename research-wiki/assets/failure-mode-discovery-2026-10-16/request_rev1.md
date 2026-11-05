# Review request: failure-mode discovery plan (ADAGS, 2026-10-16)

You are an independent senior reviewer (CVPR area chair level, dynamic Gaussian splatting). Review the PLAN below,
written by Claude for the project's principal. Be adversarial and concrete. You may read files under D:\ (the repo
is D:\adags; the synthetic suite data and the new census outputs are under D:\adags-ars, in particular
D:\adags-ars\fmd\obs_census.py, D:\adags-ars\fmd\census\ and D:\adags-ars\wave1\scores\). Please spot-check at least
two numeric claims against those files (e.g. one census row and one H3 table row) and say whether they match.

The principal's brief, in short: "Choose a direction by elimination is why work stays half-finished. Novelty comes
from a narrow claim in a crowded area: in setting Y, X breaks because Z, and here is a fix. Find, with the SOTA
methods and datasets available today, what exactly fails. The problem likely lives in monocular or sparse-view
video, where hidden regions must be remembered or inferred; dense multi-view has little true occlusion. Pick the
setting, because PSNR averaged over a standard dataset will hide the effect. Fix on finding failure modes."
Deliverable requirements: deliberate setting choice with reasons and what would change the mind; falsifiable
failure hypotheses with exact measurements, data on hand, effect sizes, cheap first tests; stop rules (kill,
promote, time box, when to commit) and a fight-for-it step (kill-argument + novelty check on the five-category
rubric); mapping to benchmark and fix sketch; first-two-weeks schedule with GPU cost, mindful of CVPR 2027
(16 Nov 2026).

Novelty rubric to apply (binding for this project): five legitimate categories - (1) context/domain transfer,
(2) architectural/methodological synthesis with non-obvious emergent properties, (3) empirical/analytical
discovery (mechanistic insight, disproving an assumption), (4) formulation/problem redefinition, (5)
resource/benchmark/evaluation. Combinatorial novelty is normal; judge: non-trivial? generalisable insight? fair SOTA
baselines at their own regimes? Rate novelty HIGH when clearly novel within its category under these tests.

Please answer, numbered:
1. Is the setting choice (2-3 widely spread cameras; monocular as extension) justified by the evidence, and is the
   census method sound (exactness, what it misses, any bug you can see in obs_census.py)? What alternative setting
   would you pick, if any, and why?
2. For each hypothesis H1-H4: is it falsifiable as written; is the measurement (matched difference-of-differences)
   right; are the thresholds sensible given the noise floors; is the first test the cheapest decisive one; what
   confound is missing?
3. Stop rules and the commit rule: will they actually stop the work? Any loophole that lets an unpromising lane run
   on? Is the fallback (commit to a dense failure) right?
4. Novelty and prior art: for the most likely surviving failure, which category, novelty rating, and the nearest
   paper that could sink it (check the web for 2025-2026 work on few-camera/monocular dynamic GS that stratifies
   by visibility history or studies objects hidden and re-seen). Anything the plan missed?
5. Schedule and GPU cost: realistic? What is the critical path, and what would you cut?
6. The CVPR question: given 31 days to the deadline, what is the best use of this plan? Be specific.
7. The three most important changes to make, in order.
Keep the review under 1,500 words. Mark each point CRITICAL / MAJOR / MINOR.

---- PLAN (rev 1) ----
# Failure-mode discovery plan (2026-10-16)

Status: rev 1, written for review. Principal's brief (2026-10-16): pick the setting deliberately; find, with the SOTA
methods and the data we have today, what exactly fails; stop rules; fight for the surviving failure; map it to a
benchmark and a fix; two-week schedule against CVPR 2027 (16 Nov 2026).

## 0. The plan in six sentences

1. **Setting: two or three training cameras spread widely around the action** (every other camera held out for
   scoring), with **one moving camera** as the extension. Not 4-6 cameras, and not a forward-facing line of
   cameras: today's census shows there is almost nothing to remember there (section 1).
2. **Failure we hunt: what a method renders for a surface while it is hidden from every training camera but seen
   by a held-out camera, and in the first frames after it is seen again.**
3. **Instrument: an observation-history label on every scored pixel** (seen now / seen only earlier / seen only
   later / never seen, plus frames since last seen). The standard monocular metric (DyCheck's co-visibility mask)
   merges "seen now" with "seen only at another time", and whole-frame PSNR averages over pixels that are 99 %
   "seen now", so today's benchmarks cannot show this failure by construction.
4. Four ranked failure hypotheses, each with a cheap first test, an effect size that counts, a kill rule and a
   time box; the hypothesis list closes on **23 October**; one failure is committed on **30 October** after a
   kill-argument and a novelty check.
5. Cost: about 32 GPU-h for the first tests and about 60 GPU-h for replication, all waiting for the principal's
   approval; everything before that is CPU.
6. CVPR honesty: committing on 30 October leaves 17 days. That is enough to write up a failure whose fix we
   already have (the memory gate) or a measurement paper; it is not enough to invent and validate a new fix.

## 1. Which setting, and why

### 1.1 Evidence from the record

- **Dense real capture has almost nothing to remember.** On real N3V the hand occlusion offered the presence gate
  nothing to remove (every contrast within ~0.1 dB; at most ~0.8 dB to gain on the clean return frames). On
  DiVa-360, 0 of 597 candidate windows were genuine absences.
- **The mechanism to watch is "paint, don't remove".** On the synthetic suite with 20 cameras, Multi4D keeps a
  departed object in its static-background bank (0.98-0.99 of the object's mass) and paints a transient occluder
  over it during the gap (transient alpha 0.88-0.99 on the ghost region). With 20 cameras the painted patch has to
  agree with 20 views; with 3 it only has to agree with 3.
- **A failure already exists at 20 cameras, in all three methods: the first frame after a return.** Tabulated
  today from the 45 local score files (section 3, H3): 3-11 dB below the steady return.

### 1.2 New evidence today: how much is there to remember? (exact census, CPU, 0 GPU)

Tool: `D:/adags-ars/fmd/obs_census.py` (sha256 d2c35164...; imports the suite renderer). For every 8th pixel of a
held-out camera that shows the static background at a frame, it decides exactly - by ray casting from each
training camera to the same surface point, with every moving part present at that frame as a potential blocker -
whether any training camera sees that point now, only earlier, only later, or never. Wave 1, all 9 scenes on the
suite's N3V rig (held-out cam00, plus two end cameras for one subset); 6 scenes on a probe "arc" rig (cameras on a
circle of radius 10 around the table, -80 to +80 degrees; held-out cameras at +44, -36, +04 degrees). Outputs in
`D:/adags-ars/fmd/census/`.

| Rig | Training cameras | "Seen only earlier", share of static-background pixel-frames: median (max) | Same, as a share of the moving-object area | "Never seen", median |
|---|---|---|---|---|
| N3V line | 19 | 0.000 % (0.000) | 0.00 % | 0.03 % |
| N3V line | 6 | 0.002 % (0.034) | 0.02 % | 0.2 % |
| N3V line | 3 (two subsets) | 0.05-0.10 % (0.25) | 0.6-1.2 % | 0.8-2.5 % |
| N3V line | 2 | 0.36 % (0.65) | 4.4 % | 1.9 % |
| Arc probe | 21 | 0.000 % (0.001) | 0.00 % | 0.2 % |
| Arc probe | 6 | 0.004 % (0.035) | 0.04 % | 2.0 % |
| Arc probe | 3 wide (-80, 0, +80) | 0.85 % (1.68) | 7.0 % (max 13.1) | 8.5 % |
| Arc probe | 2 wide (-80, +80) | 3.4 % (5.8) | 32 % (max 56) | 21 % |

What this says:

1. **The principal's intuition holds exactly.** With 19-21 training cameras the "remember" class is empty on both
   rigs. No dense benchmark can show a memory failure.
2. **It is not "sparse view" in the usual 4-6-camera sense.** At 6 cameras the class is still about zero. It opens
   up only at **2-3 cameras**, and the rig's spread multiplies it about 8-fold (3 cameras: 0.1 % on the line rig,
   0.85 % on the arc). So the setting is "two or three widely spread cameras, or one moving camera".
3. **"Never seen" grows faster than "seen only earlier".** Any averaged metric at 3 cameras is dominated by
   hallucination error, which is generative-prior territory. The split is what lets a memory claim be made at all.
4. **Hidden stretches are long.** On the arc with 3 cameras, 41 % of memory pixel-frames are 31+ frames after the
   last sighting; 10 % of sampled pixels have a hidden run of 10 frames or more.

Limits of the census: static background only (moving surfaces - the cook, the rotator, the target - are not
labelled yet); the arc rig is a probe (cameras about 10 units from the table, looking down; views checked by eye
today in `D:/adags-ars/fmd/arc_views.png`), not a designed rig; nothing yet on real footage.

### 1.3 The recommendation

- **Primary: few-camera, wide spread, 3 training cameras.** Synthetic: the suite re-rendered on a designed wide rig
  (CPU only) next to the existing line rig cut to 3 cameras (no re-render). Real: **Panoptic Sports** (31-camera
  dome, 6 sequences, 716 MB, already on Leonardo; juggling, softball, tennis: balls and hands that are hidden and
  return) cut to 3 training cameras with 28 held out.
- **Extension: one moving camera.** First a synthetic moving-camera track from the suite renderer (exact labels and
  poses), then DyCheck iPhone (static test cameras far from the moving training camera) if the synthetic test
  shows the failure. The only real monocular data on hand, HyperNeRF vrig, scores on the stereo partner of the
  training camera (small baseline, small memory class): a loader smoke test, not the benchmark.
- **Not separate studies:** long horizons and large viewpoint change become axes inside the primary setting
  (frames since last seen; held-out camera's distance from the nearest training camera).

Why few-camera before monocular: every held-out camera is real ground truth at every frame (28 on Panoptic, 17 on
the suite), poses are exact, the three methods run with only a camera-list change, and camera count and spread
are dials. Monocular gives one or two test cameras and the teleporting-camera confound DyCheck documented.

### 1.4 What would change my mind

- **Too small on real footage.** If the Panoptic proxy census at 3 cameras gives "seen only earlier" below 1 % of
  the moving-object area, or no hidden stretch of 10+ frames, the few-camera setting is synthetic-only; move the
  primary to monocular (DyCheck download).
- **The failure is generic.** If memory pixels are no worse than matched pixels that stay seen (section 2), the
  effect is "few cameras are hard", which 4C4D, GC-4DGS and D2-4DGS occupy. Drop it; keep the instrument.
- **No growth.** If the 3-camera gap and return deficits equal the 20-camera ones, the dense synthetic failure is
  the paper and the few-camera setting adds nothing.
- **Pose error dominates on real monocular data** (checked against the exact synthetic track): monocular stays
  synthetic-only.

## 2. The instrument: observation-history labels and a matched deficit

| Label | Meaning | Tests |
|---|---|---|
| seen now | some training camera sees this surface point at this frame | ordinary fitting (control) |
| seen only earlier | no training camera sees it now; one did earlier | remembering |
| seen only later | only at later frames | non-causal memory (offline methods may use it) |
| never seen | no training camera ever sees it | inference; excluded from every claim |

Each memory pixel also carries frames since last seen and the event type: **occluded** (still in the world) or
**absent** (removed, as in the suite's gap).

- **Synthetic, exact:** the census tool, extended in three ways before scoring: moving surfaces (object-local hit
  coordinates and the part pose at each frame), full-resolution masks, and per-pixel output as mask directories.
- **Real, proxy:** depth from a dense model of the same sequence (all cameras), visibility by depth test with a
  tolerance band; pixels inside the band are "uncertain" and excluded. The proxy is validated on the suite against
  the exact labels first (agreement of at least 95 % on memory pixels, otherwise the real census is "unresolved").
- **Scoring:** labels become `mask_dir` regions of the common evaluator `scripts/eval_event_windows.py` (one
  evaluator, common LPIPS convention; each method's native LPIPS is half-contrast).
- **The quantity is a difference of differences, not a level.** For memory pixels: quality while hidden minus
  quality on the same pixels in the last 5 frames before they were hidden. For control pixels (nearby background
  that stays seen, same frames): the same difference. **Memory deficit = the first minus the second.** That cancels
  "this surface is hard", "this run is worse" and "quality drifts in this stretch". Reported by frames since last
  seen (1-5, 6-15, 16-30, 31+) and by held-out camera distance from the nearest training camera.
- **Look at the pixels** of every scored region on every arm before reading a table (2026-09-10 rule).

## 3. Failure hypotheses, ranked

Noise floors: Multi4D whole-frame fixed-seed spread 0.34 dB; Ex4DGS sd 0.36 dB (n = 6); event-window paired sd
0.5-1.6 dB in earlier lanes. A real failure is several dB, specific, same sign on every scene.

### H1 (rank 1) - Hidden surfaces are not held

- **Claim.** With 2-3 widely spread training cameras, a surface hidden from all of them is rendered worse in
  held-out views the longer it stays hidden, because time-local primitives (Multi4D's transient bank, Ex4DGS's
  onset/offset opacity, STG's temporal opacity) are unconstrained while unseen, and the occluder's own primitives
  - poorly placed in depth with 3 views - get painted onto it.
- **Measurement.** Memory deficit (section 2) by frames since last seen; bank-alone renders (`cer_m4d_banks.py`) to
  name the bank; flicker on static memory pixels (render change where ground truth does not change).
- **Data.** Suite same-pose scenes at 3 cameras on the line rig (exists) and on a designed wide rig (CPU render);
  Panoptic juggle and softball at 3 cameras.
- **Counts as real.** Memory deficit of at least 3 dB at 16+ frames hidden, same sign on 3 of 3 synthetic scenes
  and on 1 of 2 real sequences, in at least 2 of the 3 methods, with the artefact visible in a montage.
- **First test.** Done today on CPU: the class exists on the suite (table 1.2). Next: G1 runs (section 6).
- **Fix it invites (sketch).** Hold unobserved primitives: freeze appearance and opacity of primitives that have no
  training view, instead of letting them be explained away; or route hidden surfaces to the persistent bank.

### H2 (rank 2) - Absence is read as occlusion, and few cameras make it worse

- **Claim.** Multi4D does not remove a departed object; it paints a transient occluder over it (measured, 20
  cameras). With 3 cameras the patch need only match 3 views, so held-out views see the object around it.
- **Measurement.** Gap window (frames 63-87, ghost region) on all held-out cameras at 20 and 3 cameras; transient
  alpha over the ghost; static background as the control.
- **Data.** Suite same-pose scenes, line rig (20-camera runs exist: gap 22.9-31.3 dB, base Multi4D, cam00).
- **Counts as real.** Gap PSNR falls at least 3 dB more than static-background PSNR from 20 to 3 cameras, 3 of 3
  scenes, patch visible in the bank render.
- **First test.** The same G1 runs; no extra GPU.
- **Fix it invites.** The memory gate already built (early gate: +9.2 dB gap at 20 cameras; return -0.10 dB with a
  known interior-detail cost). What 3 cameras add: the gate used construction-derived membership, which must now
  be estimated. **The only hypothesis whose fix exists today.**

### H3 (rank 3) - The first frame after a return is wrong in every method

- **Claim.** When something reappears, the first frame is several dB below the steady state in all three methods,
  because their temporal support is smooth and cannot switch on in one frame.
- **Already measured at 20 cameras** (held-out cam00, target region; steady return = frames 92-99; 0 GPU):

  | Method (runs) | deficit at frame 90: min | median | max | runs at >= 5 dB |
  |---|---|---|---|---|
  | STG, six 50-frame models (9) | 7.5 | 9.4 | 11.1 | 9 / 9 |
  | Multi4D run 1 (9) | 3.2 | 8.0 | 10.7 | 7 / 9 |
  | Ex4DGS (27) | 2.9 | 4.9 | 8.5 | 13 / 27 |

  Bottle same-pose: Multi4D 24.0 dB at frame 90, 31.7 at 91, 34.7 steady; STG 22.1 / 29.7 / 33.3. Ex4DGS's smaller
  deficit sits on a much lower steady level (its ~9 dB texture loss). The early memory gate lifts frame 90 by
  +4.9 to +6.1 dB (plan log 2026-10-16). Window T was never a frozen endpoint.
- **Open question that decides its nature.** Is frame 90 also bad on **training** cameras (the representation cannot
  switch on: a real failure) or only on the held-out camera (a generalisation effect, folded into H1)?
- **Counts as real.** Deficit of at least 5 dB on 8 of 9 scenes for at least two methods, present on training
  cameras, and present for background re-revealed when an occluder leaves in the 3-camera runs.
- **First test.** One debug-queue render of training cameras at frames 88-92 for one run per method (under 1 GPU-h).
- **Fix it invites.** A temporal support that can switch on at a learned instant, or carry-over from the memory
  gate. Novelty risk: RetimeGS (CVPR 2026) and Instant4D document under-constrained temporal opacity; the narrow
  claim must be the step at a reappearance, measured.

### H4 (rank 4) - One moving camera: what was seen at another time is lost by time-local primitives

- **Claim.** In a moving-camera video, moving-object surfaces seen only at other times (the far side of the spinning
  rotator, part of the cook) are rendered much worse by time-local 4D primitives than by a canonical deformation
  model; Multi4D lifts exactly those fast surfaces into its time-local transient bank.
- **Measurement.** Memory deficit on moving-object pixels; persistent-bank-alone vs transient-bank-alone renders.
- **Data.** A synthetic moving-camera track (renderer reuse with per-frame poses, exact labels); DyCheck iPhone only
  after approval.
- **Counts as real.** Memory deficit of at least 3 dB on moving-object pixels, and the persistent bank alone at least
  2 dB better than the transient bank alone on those pixels.
- **First test.** CPU render of the track and labels; one Multi4D run through its monocular path (`arguments/nerfds.py`
  regime, `nerfies` loader; about 3-4 GPU-h) after a loader smoke test on HyperNeRF `broom2`.
- **Starts only if** H1's G1 readout shows a memory deficit (it extends the claim) or H1 is killed only for class
  size (monocular has the larger class).
- **Fix it invites.** A lift veto or memory routing that keeps soon-to-be-unseen surfaces persistent.

### Parked, with reasons

- **Never-seen regions:** every per-scene optimiser must hallucinate; generative-prior work (CAT4D, Vivid4D,
  CogNVS) occupies it. Measured only to size and exclude.
- **Ex4DGS late static-background collapse** (21 of 65 runs, 3-20 dB, between 30k and 40k iterations; pruning
  refuted) and **Ex4DGS's ~9 dB object-texture loss:** defects of one baseline, not of the setting; recorded as
  baseline caveats.
- **Long horizons as their own study:** folded into H1's frames-since-last-seen axis; longer tracks are cheap to
  render if H1 shows age dependence.

## 4. Stop rules

### 4.1 Per hypothesis

| | Kill | Promote | Time box |
|---|---|---|---|
| H1 | memory deficit within 1 dB of zero at 16+ frames hidden in every method; or the montage shows nothing; or real proxy census below 1 % of moving-object area (then synthetic-only, rank drops) | criteria in H1 met and a bank render names the mechanism | 3 working days after the G1 runs finish |
| H2 | gap PSNR falls no more than static PSNR (within 1 dB) from 20 to 3 cameras | at least 3 dB extra fall, 3/3 scenes, patch visible | same G1 runs, 2 days |
| H3 | training-camera renders at frame 90 are clean (folds into H1); or deficit under 3 dB in all but one method | criteria in H3 met | 2 days, nearly all CPU |
| H4 | not started by 22 October under its start rule; or loader not working by 24 October; or deficit under 1 dB | criteria in H4 met | ends 28 October |

A hypothesis the instrument cannot decide is recorded **unresolved**, never "failed" (project rule).

### 4.2 When to stop exploring

- No new hypothesis after **23 October**, unless a running first test shows it directly (logged with the test).
- On **30 October** (earlier if one hypothesis is promoted and passes 4.3): commit to **one** failure, the
  highest-ranked promoted one; ties go to the one whose fix exists.
- If nothing is promoted by 30 October: commit to the best failure already measured in dense capture (H3 at 20
  cameras, or H2 at 20 cameras) and stop discovery. Do not open a new setting.

### 4.3 Fight for it before committing

1. **Kill-argument** (`kill-argument` skill: a fresh reviewer writes the strongest 200-word rejection, a second
   defends and lists unresolved critical issues). Pass: no unresolved critical issue, or each one turned into a
   control that is then run before the commit.
2. **Novelty check on the five-category rubric.** Expected category: empirical/analytical discovery plus
   benchmark/evaluation (with a fix: formulation or synthesis). It must answer the three reviewer questions:
   non-trivial; generalisable insight (why it fails, not only that it fails); fair SOTA baselines at their own
   regimes. Nearest prior art to beat: PersistGS (CVPRW 2026, multi-camera synthetic persistence, no Ex4DGS/STG),
   4D Primitive-Mache (CVPR 2026, monocular permanence, qualitative), DyCheck (co-visibility mask), 4C4D, GC-4DGS,
   D2-4DGS (few-camera floaters/depth), RetimeGS and Instant4D (temporal opacity), USplat4D, GP-4DGS.
3. **Real footage.** At least one real sequence must show the failure (synthetic-only was the decisive attack in
   the 2026-09-11 kill-argument).
4. **The sentence.** It must fit: "Everyone does X; with 3 widely spread cameras (or one moving camera), when a
   surface is hidden for k frames, X fails at Z because W, and the observation-history split is what shows it."

## 5. From a surviving failure to a benchmark and a fix

| Failure | Benchmark | Baselines (own published schedules, one evaluator) | Fix it invites (sketch) |
|---|---|---|---|
| H1 | Few-camera memory benchmark: suite scenes at 3 cameras on the line and wide rigs (exact labels) + Panoptic sequences at 3 cameras (proxy labels); windows = memory pixels by frames-since-seen bins, first 3 re-reveal frames, matched control | Multi4D, Ex4DGS, STG (one model over the sequence); a few-camera 4D method with released code (4C4D or GC-4DGS, to check); SA4D as a canonical-deformation reference | hold unobserved primitives; visibility-conditioned temporal prior |
| H2 | Suite gap and return windows on all held-out cameras at 20 and 3 cameras | same | memory gate with estimated membership |
| H3 | Per-frame return transition (frames 88-95), training and held-out cameras, every event; re-reveals in the 3-camera runs | same | step-capable temporal onset |
| H4 | Synthetic moving-camera track + DyCheck iPhone with the observation-history split | Multi4D (its NeRF-DS regime), Shape of Motion or MoSca | lift veto / memory routing |

Rules for the few-camera cut: methods keep their published schedules; only the camera list changes. The initial
point cloud is rebuilt **from the training cameras only** (the dense cloud would leak held-out views). Multi4D
reads the cameras present in the scene folder and holds out the first (`neural_3D_dataset_NDC.py`, checked today),
so a folder of symlinks gives a 3-camera tree without a code change; scoring the other held-out cameras needs a
render wrapper (from `agent-control/ars/v9_run.py`, which already wraps each method's camera lists).

## 6. First two weeks (17-30 October)

| Days | Work | GPU-h | Approval |
|---|---|---|---|
| Oct 17 | Census tool v2 (moving surfaces, full-resolution masks); H3 training-camera question set up | 0 | no |
| Oct 17-18 | Design the wide rig (3 training + 9 held-out cameras, spread about 150 degrees, depth comparable to the line rig); render the 3 same-pose scenes on it (CPU, about 20 min per scene in parallel); labels | 0 | no |
| Oct 18 | 3-camera trees for both rigs (Multi4D, Ex4DGS), 3-camera COLMAP init, held-out render wrapper; 2 debug-queue smoke runs; H3 training-camera renders | ~3 | **yes** |
| Oct 18-20 | Panoptic conversion to a COLMAP-style layout (per-camera intrinsics), loader check; dense SA4D or Multi4D model of juggle and softball for proxy labels; proxy validated on the suite | ~6 | **yes** |
| Oct 19-21 | **G1:** Multi4D + Ex4DGS on 3 same-pose scenes x 2 rigs at 3 cameras (about 29 GPU-h); Panoptic juggle and softball at 3 cameras, Multi4D (about 4 GPU-h) | ~33 | **yes** |
| Oct 22-23 | Readouts, montages, kill/promote; H4 starts or not; hypothesis list closes Oct 23 | 0 | no |
| Oct 24-28 | Replicate the promoted failure: 3 replicates x 3 scenes x 2 methods on the rig that showed it; 2 more Panoptic sequences; STG one-model where H3 needs it; H4 runs if started | ~60 | **yes** |
| Oct 29-30 | Kill-argument, novelty check, commit | 0 | no |

Total about 100 GPU-h (the allocation has about 15,500). Submissions go through `agent-control/ars/ars_sub.sh` with
the ledger under an `fmd_` name prefix; no file or job of the early-gate CER lane is touched. Debug queue: 2 jobs.

**Deadline.** If H2 is committed, its fix exists and the CVPR paper is "absence read as occlusion, made worse by
few cameras, fixed by the memory gate with estimated membership". If H1 or H3 is committed, CVPR can carry the
measurement (benchmark plus diagnosis) at most; its fix is the ICCV 2027 paper (deadline not yet checked).

## 7. Verified today vs assumed

**Verified in this session (tool output):** the census table (section 1.2, 153 census runs, exact); the H3 table (45
local score files); frame-90 and gap numbers for bottle same-pose (Multi4D 24.0 / 22.9 dB; STG 22.1 / 44.8 dB); the
N3V rig geometry (camera line about 11.4 units wide, nearest depth about 8.7); the suite renderer's options and
render cost (about 23,000 CPU-seconds per 20-camera scene); Multi4D's camera split rule; the data inventory
(section 8); the literature scan (section 9).

**From the record, not re-run:** every other number quoted (census of Multi4D banks, memory-gate effect sizes,
Ex4DGS collapse counts, real-occlusion null).

**Assumed, checked first:** that Ex4DGS and STG train on a 3-camera tree without code change; that a 3-camera COLMAP
init succeeds on the suite and Panoptic; that Panoptic converts cleanly; that the real proxy labels agree with exact
labels on the suite.

## 8. Data inventory (read-only, 2026-10-16)

| Data | Where | Use here | Loaders |
|---|---|---|---|
| Suite wave 1 (9 scenes, 20 cameras, 300 frames, exact identity buffers) | `D:/adags-ars/wave1`; Leonardo scratch | exact labels; any camera subset; new rigs by re-render | Multi4D, Ex4DGS, STG |
| N3V, 6 scenes | Leonardo, Apollo, `D:/adags/data` | real narrow few-camera cuts (small memory class, by the census) | all |
| Panoptic Sports, 6 sequences, 31 cameras x 150 frames, 640x360, masks for 27 cameras | Leonardo D21 `data/panopticsports` (716 MB) | real wide few-camera cuts | SA4D only; others need conversion |
| HyperNeRF vrig `broom2` (masks) and `3dprinter` | `D:/4DGaussians/data/hypernerf/virg` | monocular loader smoke test | Multi4D (`nerfies`), SA4D |
| ImViD, DiVa-360 (Apollo only), Technicolor (D: only), Google Immersive | various | not needed in two weeks | various |
| NeRF-DS, DyCheck iPhone, DAVIS, Kubric | **absent everywhere** | DyCheck iPhone (about 15-20 GB, size unverified) only if H4 survives its synthetic test | - |

## 9. Literature position (scan of 2026-10-16; venues marked unverified where so)

- Few-camera 4D (4C4D CVPR 2026, GC-4DGS, D2-4DGS, Splatography, SplatFields ECCV 2024): floaters, depth ambiguity,
  flicker. None isolates surfaces hidden from all training views and seen again.
- Monocular (DyCheck NeurIPS 2022, MonoDyGauBench TMLR 2025, 4DGS360 ECCV 2026, Prior-Enhanced GS SIGGRAPH Asia 2025):
  full-frame or co-visibility-masked metrics; disocclusion failures described qualitatively.
- Persistence: PersistGS (CVPRW 2026) is the only quantitative one (multi-camera synthetic, physics for an
  occluded rigid object); 4D Primitive-Mache (CVPR 2026) shows monocular permanence qualitatively.
- Open: a metric split by when a pixel was last seen; opacity decay while hidden and quality at return compared
  across Ex4DGS, STG and Multi4D.

## Glossary

Suite = synthetic Absence-Return Suite (ARS). Same-pose return (A0), moved return (R2), pass-through (G1). Gap window
(P1, frames 63-87, ghost region); return window (P2, frames 92-99); first return frames (T, 90-91). Memory gate = CER;
early gate = CER from iteration 3,000. Line rig = the N3V camera layout the suite was rendered with.
