# Kill-argument dossier: "observed absence is not represented" (2026-10-19)

## The claim as a CVPR 2027 submission would state it

Headline: **State-of-the-art dynamic Gaussian-splatting methods do not represent an OBSERVED absence.** When a
mostly-static object leaves the scene and the training views show the empty surface, they keep rendering the object
in held-out views. They do this across method families, camera counts and real footage. Whole-frame PSNR stays high
and hides it. The cause is a time-invariant static component (or temporal-opacity tails) that has no notion of when a
primitive exists in scene time.

Intended contributions:
1. The measured failure, with a matched local metric: the vacated region vs a static ring 6-30 px around it, same
   frames.
2. A leave-and-return evaluation split (a synthetic suite with exact observation-history labels, plus real authored
   absence fixtures).
3. A mechanism analysis. Component renders; two configuration interventions fail (transient densification on;
   object removed from the seed cloud).
4. A fix: per-primitive presence in scene time on the static component (a "presence gate"). NOT YET BUILT.

Venue: CVPR 2027 (deadline 16 Nov 2026). Commit decision due 24 Oct 2026, hard stop 28 Oct.

## Evidence (verbatim from the lab's plan page, sections 17-24)
Full page: research-wiki/operations/failure-mode-discovery-plan-2026-10-16.md (read it for setting, census and
earlier sections). Scorers: research-wiki/assets/failure-mode-discovery-2026-10-16/ghost_ring_real.py and
obs_census_v2.py. Real-footage montage: research-wiki/assets/failure-mode-discovery-2026-10-16/montage_real_mono.png.

## 17. First readout, 2026-10-17 ~04:30 (synthetic suite, 4-camera cut; exploratory, single seed per cell)

Scorer: `D:/adags-ars/fmd/fmd_score.py` (masked PSNR on exact census labels; per-event degradation = (reference - hidden)
minus the same for a control ring 6-20 px from the event and 8 px from any moving pixel, mixed edge pixels excluded;
paired gap by frames since seen; event bootstrap). Windows: `fmd_windows.py` with the suite's own masks. Scores in
`D:/adags-ars/fmd/scores/`. Instrument null (Multi4D trained on all 20 cameras, scored on the 4-camera labels):
degradation +0.42 dB (95 % interval -0.03 to +0.78), median 0.00; this is the instrument's noise level.

**Hypothesis 1 (hidden surfaces are forgotten): not supported on this rig.** Static background hidden from all four
training cameras is rendered about as well while hidden as at its reference frame, in every method:

| Method (setting) | Bottle | Carton | Mug | Largest paired gap by time hidden |
|---|---|---|---|---|
| 4C4D (published cut), cam00 / 4 cameras | +0.49 (-0.51, 1.25); cams 05/08/17: +0.42, -1.93, +0.34 | (pending) | +0.22 (-0.23, 0.53); cams 05/08/17: +0.40, -0.29, +0.28 | 2.1 dB (short ages), no growth |
| Multi4D low-view (nerfds.py) | +0.02 (-0.06, 0.06) | +0.06 (-0.07, 1.72) | -0.26 (-0.49, -0.04) | 0.12 dB |
| Multi4D dense setting | -0.01 (-0.13, 0.01) | +1.13 (0.54, 1.60) | +0.10 (0.00, 0.19) | 3.0 dB at 16-30 frames (carton only) |
| Ex4DGS | -0.47 (-0.93, 0.22) | +1.42 (0.98, 4.85) | +0.24 (-0.48, 0.88) | 1.4 dB |

(degradation score in dB, 95 % event-bootstrap interval; cam00 unless stated.) Mechanism, where visible: Multi4D's static
bank is time-invariant, so a hidden static point renders the same colour seen or not (paired gap 0.0-0.1 dB); Ex4DGS
also keeps a static set. Where hidden regions are bad (Ex4DGS bottle: 14-21 dB), they are equally bad when seen ("never
learned": few-view geometry, which few-camera papers already target). Under section 4, hypothesis 1 sits between kill
and promote on this rig (no cell reaches 3 dB; two cells exceed 1 dB) and is recorded **unresolved**; it expires from
the CVPR claim on 24 Oct unless 4C4D carton or the real scenes change the picture.

**Hypothesis 2 (absence read as occlusion, worse with few cameras): killed for Multi4D's dense setting.** Same config,
20 -> 4 cameras: gap ghost 22.9 -> 24.4 / 30.3 -> 22.5 / 31.7 -> 24.3 dB while static background falls 44.5 -> 27.9 /
44.7 -> 25.1 / 45.1 -> 28.7 dB. The gap falls far less than everything else.

**Hypothesis 3 (first frame after a return): not amplified at 4 cameras.** Frame-90 deficit vs frames 92-99: Multi4D
dense 0.6 / 2.3 / 1.1 dB (vs 8-11 at 20 cameras); 4C4D bottle 1.9 dB; Ex4DGS 2.8 / 8.3 / 3.7 dB.

**Found by a running test: in Multi4D's own low-view setting a mostly-static object that leaves is never removed.**
Gap window (frames 63-87, where the object was): 12.2 / 12.2 / 12.2 dB at 4 cameras, against 22.9 / 30.3 / 31.7 for the
dense setting at 20 cameras and 23.9 / 24.5 / 26.1 for Ex4DGS at 4. The target-region PSNR is identical in every frame
the object is present (bottle 26.64 dB), and a montage of frame 75 shows the bottle rendered fully and crisply while
every camera sees the empty table (`D:/adags-ars/fmd/montage_gap_L1A0.png`). Reading: the object is in the time-invariant
static bank, and `nerfds.py` turns transient densification off, so nothing can represent its absence. Open question
that decides what it is: config or camera count? Control submitted: the same low-view config with all 20 cameras
(jobs 60781043-45). If the ghost stays total, this is a failure of Multi4D's published monocular regime, to be tested
in that regime (one moving camera, synthetic track being built CPU-side, `D:/adags-ars/fmd/mono/`). Candidate
statement (not yet earned): "methods that factor a scene into a time-invariant static part and dynamic parts assign a
mostly-present object to the static part and, without a transient path, cannot represent its absence; in a monocular
setting the absence is ignored entirely." Nearest prior art to check: the project's own absence/return record, PersistGS,
static/dynamic splits in monocular methods (Shape of Motion, MoSca).

**Real-footage labels:** the proxy (clean plate + difference masks) validated on the suite reaches remembered-class
precision 0.535 as planned and 0.87 pooled with two fixes and an idealised person segmenter (0.93 / 0.89 / 0.75 per
scene): below the 0.90 bar, so real-scene memory labels are **unresolved**; whole-frame and event windows on the real
scenes are still reported. Details: `D:/adags-ars/fmd/proxy/validation.json`.

**Engineering notes:** 4C4D renders write test cameras in name order, 300 frames each; `train.py` appends the iteration
count to its save list before reading the config (a shortened run saves no final checkpoint; the launcher now passes
`--save_iterations`); real-scene 4C4D runs (1.9 M Gaussians, 2.7 it/s) overrun 3 h, rendered by a follow-up render-only
job. Five MASt3R jobs shared one `./tmp` cache folder; its keys include each scene's full path, so no cross-scene reuse,
but later jobs failed only at their final cache deletion (outputs intact); future jobs get separate working copies.


## 18. Move to the work area (2026-10-17, 04:30-05:40)

The storage-cleanup session reported 161.5 GiB freed on the work area; per the principal's instruction the lane moved
from scratch to `/leonardo_work/EUHPC_D36_068/sri/proj_adags/fmd/` (`repo/`, `weights/`, `data/`, `runs/`, `envs/`).
- Copy: Slurm serial job 60782297 (rsync with symlinks preserved): repo 0.8 GB, weights 4.1 GB, data 2.5 GB, 24
  finished runs 35 GB. Not copied: `envs/` (path-bound; rebuilt), the pip cache, and 4C4D's `chkpnt7000.pth` and
  `chkpnt_best.pth` (unused; we score the final checkpoint); these stay on scratch. Verification: checksum dry run,
  0 differing files for weights, data and every run; the repo's 9 differences are the extension builds the
  environment rebuilds recompiled in place plus 3 relinked checkpoint links. 8 symlinks repointed (5 Multi4D starting
  clouds, 3 MASt3R checkpoints); manifest cloud paths updated (hashes unchanged). The scratch copy is kept.
- Environments rebuilt on work with one consolidated script (`agent-control/fmd/build_envs_work.sh`): torch
  2.1.2+cu121 in all three, all imports OK, MASt3R script imports OK. A first attempt hit a pip network error
  (IncompleteRead) and one of my own `pkill -f` commands matched its own shell; partial environments were set aside
  (`envs/*.partial_0438`), builds rerun with pip retries.
- Launchers repointed to work (`*.pre_work_move` copies kept). A 4C4D debug-queue smoke on work (job 60787107):
  manifest gate OK, 2,000 iterations trained, render running.
- Still on scratch until they finish, then copied: the two real-scene 4C4D runs and their render-only jobs, the three
  20-camera low-view controls. Remaining scratch dependency: the suite's own source frames
  (`/leonardo_scratch/.../ars/wave1`, another lane's data) that every suite tree links to; flagged to the principal.
- Open engineering item: MASt3R writes its cache to `./tmp` inside the MAtCha repo; future MASt3R jobs need separate
  working copies.


## 19. The control decides it: Multi4D's low-view setting never removes a departed object, even with 20 cameras (2026-10-17 ~06:40)

Control (jobs 60781043-45; same `nerfds.py` config, all 20 cameras, the suite's all-camera dense cloud), event windows
on held-out cam00:

| Scene | Gap ghost (63-87) | Object present (30-57 / 92-99) | Static background | Whole frame |
|---|---|---|---|---|
| Bottle | 12.57 dB | 34.06 / 34.06 | 42.91 | 38.88 |
| Carton | 14.34 | 35.80 / 35.10 | 43.53 | 39.10 |
| Mug | 12.07 | 33.42 / 33.32 | 43.76 | 39.80 |

Against the dense config at 20 cameras: gap 22.92 / 30.29 / 31.67 dB. Montage of frame 75 for all three scenes at 20
and 4 cameras and the dense config at 4 (`D:/adags-ars/fmd/montage_lowview_control.png`): the absent object is
rendered fully and crisply by the low-view config in every case.

**Reading.** The failure belongs to the configuration, not to camera count: Multi4D's own low-view setting (its
published regime for monocular NeRF-DS) has no way to represent the disappearance of an object it placed in the
time-invariant static bank (transient densification is off; the bank has no temporal extent), so the departure is
ignored and the training views that show the empty table are simply not fitted. Whole-frame PSNR stays near 39 dB:
the averaged metric does not see a 30-frame absence at all, exactly the principal's point about averaged PSNR. In the
bottle scene the object-present PSNR is identical at every present frame (34.06 dB), consistent with a purely
time-invariant representation of the object.

**Status.** Found by a running test before 23 Oct, so admissible. Not yet: (1) in its published regime (one moving
camera; synthetic track being built, then Multi4D through its `nerfies` path), (2) a mechanism intervention (turning
transient densification on in the low-view config, or moving the object out of the static bank), (3) generality
beyond Multi4D (other monocular static/dynamic splits: Shape of Motion, MoSca), (4) the kill-argument and novelty
check (prior art: the project's own absence/return record and memory gate, PersistGS, monocular static/dynamic
decompositions). The fix it invites is the project's existing presence/memory gate, which makes this the one failure
with a CVPR-feasible fix path.

**Regime check for 4C4D (real N3V, 4 training cameras):** our port's own metric on held-out cams 0/5/8/17: 25.31 dB
(`cut_roasted_beef`), 25.17 (`flame_steak`) vs the paper's 23.22 / 23.72 averaged over all remaining cameras
(arXiv 2604.04063, Tables 6-7). Renders of all remaining cameras submitted (jobs 60793462-63) for a like-for-like
comparison.


## 20. Intervention: transient densification alone is not the cause (2026-10-17 ~09:00)

Same low-view config with only `enable_transient_densify = True` (config `agent-control/fmd/configs/nerfds_transient_on.py`,
one-line diff; jobs 60795223-25), 4-camera cut, cam00 windows: gap ghost **12.64 / 23.27 / 12.18 dB** (bottle / carton
/ mug); object-present PSNR again identical at every present frame on bottle (27.43) and mug (22.09). So the ghost
stays total on 2 of 3 scenes; only the carton scene reaches the partial removal the dense config shows (22.5 at 4
cameras). The mechanism statement narrows to what every Multi4D configuration shares: the departed object lives in the
time-invariant static bank, which cannot represent absence; whether anything paints over it depends on several config
differences between `dynerf.py` and `nerfds.py` (main-loss weight 4 vs 1, deformation grid size, transfer cap, batch
size, pruning ratios), not on transient densification alone. A factorial over those would name the switch; it is not
needed for the failure statement.

Also recorded: 4C4D on the carton scene, degradation -0.19 to +0.26 dB on cams 0/5/8/17 (hypothesis 1 null for 4C4D on
all three scenes); 4C4D first-return-frame deficit 1.9 / 7.1 / 4.1 dB (bottle / carton / mug) at its published cut,
descriptive (existence-step fixture).


## 21. In its published regime: Multi4D's monocular configuration ignores an observed absence (2026-10-17 ~12:30)

**Setting.** Synthetic one-moving-camera track per suite scene (built and verified CPU-side, `D:/adags-ars/fmd/mono/`;
uploaded to `$W/fmd/data/mono/`): a hand-held-like pan from the rig's left end to its right end over 300 frames;
held-out static cameras cam00/05/17 (DyCheck-style); starting cloud back-projected from the moving camera only;
Multi4D through its own `nerfies` reader with its published NeRF-DS config `arguments/nerfds.py`, code unmodified.
Checks: renderer reproduces a rig frame bit-exactly through the per-frame path; Multi4D's reader reprojects to within
1e-4 px; exact census for a per-frame training camera, self-check 100 % "seen now". The moving camera directly sees
83-100 % of the vacated surface during the absence (cam00: bottle 83 %, carton 100 %, mug 96 %), i.e. the absence is
observed in the training video. Runs 60799600-05 (one intervention run, carton, timed out before its final render and
was resubmitted as 60830718); the final renders of the other five exist (they hit the 4 h walltime after rendering).

**Result (held-out cam00; matched comparison = the vacated region vs a ring of static surface 6-30 px around it, same
frames 63-87; `D:/adags-ars/fmd/ghost_ring.py`):**

| Method / setting | Bottle | Carton | Mug |
|---|---|---|---|
| Multi4D, published monocular config | ghost 12.64 vs ring 32.90 (**-20.3 dB**) | 12.71 vs 33.06 (**-20.4**) | 12.43 vs 31.12 (**-18.7**) |
| Multi4D monocular, transient densification on | -20.1 | (rerunning) | -17.1 |
| Multi4D low-view config, 20 cameras | -31.7 | -25.6 | -34.1 |
| Multi4D dense config, 4 cameras | -5.3 | -1.5 | -5.8 |
| 4C4D (published cut), 4 cameras | -7.4 | -6.7 | -6.9 |
| Ex4DGS, 4 cameras | -7.1 | -7.3 | -8.6 |

The object itself, when present, renders at 32.4 / 32.9 / 28.7 dB; the frame-75 render (object absent in every camera,
and seen absent by the moving camera) is indistinguishable from the frame-40 render (`D:/adags-ars/fmd/montage_mono.png`).
Whole-frame PSNR on the static held-out camera is low (15.5-19.9 dB: monocular extrapolation, 20-25 % of its static
pixels are never seen), which is why the ring comparison, not the frame average, is the measurement.

**Status against section 4.** Hypothesis 5 (found by a running test; admissible): "Multi4D, in its own monocular /
low-view configuration, never removes a mostly-static object that leaves, even when the absence is observed": effect
18.7-34.1 dB below the matched ring in 3/3 scenes under three conditions (4 cameras, 20 cameras, one moving camera);
pixels inspected; mechanism located in the time-invariant static bank (transient densification alone does not fix
it). Broader pattern: every method leaves a 5-9 dB ghost of a departed object at 4 cameras. Still required before
commitment (section 4.3): (1) real footage: the project's authored real absence fixture (`cut_roasted_beef`, bottle
removed frames 60-89 on all 20 cameras, gap ground truth = SA4D teacher render) turned into a monocular sequence;
(2) one more monocular method with a static/dynamic split (generality); (3) kill-argument; (4) novelty check (prior art:
this project's own absence/return record and memory gate; PersistGS; monocular static/dynamic decompositions;
transient-object handling in in-the-wild NeRF/GS). The fix it invites (a presence gate on the static bank) is the
project's existing memory gate, adapted.

**4C4D regime check:** the all-camera renders were submitted with a comma list that `sbatch --export` split, so only
cam00 rendered (25.00 / 24.79 dB); resubmitted with colon-separated lists (60830716-17).


## 22. Novelty check on the surviving failure (2026-10-17 ~13:30; web scan by a subagent, venues marked unverified where so)

**Bottom line.** No 2023-2026 paper documents or measures this failure (a mostly-static object leaves, the camera sees
the empty surface, the time-invariant static part keeps rendering it). The general mechanism is folklore: canonical or
static representations "struggle" with appearing/disappearing objects (4D-Rotor GS / 4DGS, Duan et al. 2402.03307;
NeRFPlayer, TVCG 2023), which motivates time-varying opacity (Spacetime Gaussians, FreeTimeGS CVPR 2025, RetimeGS
2603.13783 with left/right temporal boundaries, SharpTimeGS). Multi4D's own appendix (0.F.2, Fig. 13) discusses newly
APPEARING content on monocular data and never disappearance; must be cited. Nearest named concept: "semi-transient"
distractors (T-3DGS 2412.00155, static scenes). PersistGS and 4D Primitive-Mache concern the opposite case (the object is
still there but unseen); this failure is an OBSERVED absence. No benchmark scores leave-and-return in monocular dynamic
video; standard sets (DyCheck, NeRF-DS, HyperNeRF, Nvidia) are not built around such events (not verified frame by frame).

**Rating on the five-category rubric (as stated today):** empirical discovery moderate-low (one method, synthetic, three
scenes; the config invites a "self-inflicted" objection although it is the authors' released NeRF-DS setting);
formulation low-moderate (observed absence vs unobserved occlusion is a clean distinction); benchmark/evaluation
moderate potential (a scripted leave-and-return split plus the vacated-region ring metric); a presence-gate fix alone is
close to RetimeGS/SharpTimeGS-style temporal boundaries (low-moderate).

**What raises it most, in order:** (1) the same full ghost in a second method family with a time-invariant static set
(Shape of Motion: static background + motion masks; or Hybrid 3D-4DGS 2505.13215, which converts long-lived 4D Gaussians
into time-invariant 3D ones); (2) a mechanism intervention in Multi4D that forces the object out of the static bank (or
gates the static bank) and removes the ghost with no other change; (3) real footage (even 2-3 clips of an object removed
and returned, ring metric), and a temporal-opacity method (Spacetime Gaussians / RetimeGS) under the same monocular
protocol, to show whether per-primitive temporal opacity already solves it.

**Strongest rejection to prepare for:** (a) "known limitation" -> answer with the measurement, the observed-vs-unobserved
distinction and averaged-PSNR blindness (19-34 dB local failure at ~39 dB whole frame); (b) "degraded config of one
method" -> it is the released NeRF-DS config, transient-on still fails 2/3, plus the second family; (c) "three synthetic
scenes" -> real clips, prevalence count in existing datasets.


## 23. Second method family, regime check, and what the failure now is (2026-10-18 ~22:30)

**Hybrid 3D-4DGS (arXiv 2505.13215, repo/hybrid3d4dgs, code unmodified, published `configs/n3v/default.yaml`) in its
published regime (all 20 cameras, cam00 held out), suite same-pose scenes, one switch: the dynamic-to-static
conversion threshold 3.0 (published) vs 1e9 (conversion off).** Adapters, all recorded: the authors' converter with its
in-place image-halving loop removed (it would overwrite symlinked source frames; ours are already 1352 wide); its own
COLMAP cloud in its own normalised frame; `envs/h3d4d` (Pillow<10 because the loader calls `Image.fromarray` on int8
arrays; `lpips` for its validation renderer; `ninja` for its just-in-time rasterizer build). One evaluation point
(iteration 6,000), so the saved "best" checkpoint is the final one. Render ground truth equals the suite frames exactly.

| Arm | Bottle | Carton | Mug |
|---|---|---|---|
| 3D-4DGS published (conversion on) | ghost 26.32 vs ring 41.75: **-15.4 dB** | 29.11 vs 42.11: **-13.0** | 30.44 vs 42.24: **-11.8** |
| 3D-4DGS conversion off | -13.3 | -11.0 | -11.7 |
| Multi4D dense config, 20 cameras (reference) | 22.92 vs 41.21: -18.3 | | |

Component renders at frame 75 (`D:/adags-ars/fmd/montage_h3d4d_L1.png`): with the published conversion, the
STATIC-only render still contains the full bottle while the dynamic part paints a background-coloured patch over it,
the same mechanism measured in Multi4D's dense setting; with the conversion off there are no static Gaussians and the
ghost comes from the 4D Gaussians' temporal tails.

**4C4D regime check closed:** all remaining cameras (4,800 / 5,100 images): 24.94 / 24.89 dB vs the paper's 23.22 /
23.72 (LPIPS and DSSIM also better), consistent with the README's full-resolution `--res` default; the port is at or
above the published level.

**What the failure now is (to be fought in section 4.3):** *observed absence is not represented.* When a mostly-static
object leaves and the training views show the empty surface, every SOTA dynamic method tested keeps rendering it in
held-out views: 11-18 dB below the matched static surround at 20 cameras in their published regimes (Multi4D dense,
3D-4DGS both arms), 5-9 dB at 4 cameras (4C4D, Ex4DGS, Multi4D dense), and totally (18.7-34 dB) in Multi4D's own
monocular/low-view configuration even with 20 cameras or one moving camera - while whole-frame PSNR reads about 39-44 dB.
Two mechanisms are visible in component renders: a time-invariant static component that keeps the object (Multi4D
static bank; 3D-4DGS converted Gaussians), patched over by dynamic primitives; and temporal-opacity tails that do not
switch off (3D-4DGS without conversion). Pending: real footage (three real fixtures, monocular sweeps, running), the
static-bank seed intervention (running), kill-argument and novelty re-check with this broader statement.

## 24. Real footage and the static-bank seed intervention (2026-10-19 ~01:15; single seed per cell)

**Real footage (requirement 1 of section 21).** The project's three real authored-absence fixtures (wine bottle removed
over frames 60-89 on every camera: `cut_roasted_beef`, `flame_steak`, `sear_steak`), each turned into a monocular
stop-and-go sweep over the 17 non-held-out cameras ordered left to right along the rig (~17.6 frames per camera;
`agent-control/fmd/build_real_sweep.py`). Held-out cameras cam00/05/17 are never in the sweep. The sweep shows the
empty surface throughout the gap. Multi4D, published monocular config `nerfds.py`, code unmodified, 20,000 iterations.
Vacated-region masks: fixture vs original footage (exact for the roasted beef; threshold 24 for the steaks, above
the measured route noise of 13-16). Scorer: `assets/failure-mode-discovery-2026-10-16/ghost_ring_real.py`. It reports
the ghost (render vs fixture inside the mask) and the static ring 6-30 px around the mask (pixels that move in the
original footage are excluded). It also scores the render against the ORIGINAL footage inside the mask, i.e. how much
the render looks like the departed bottle. All are pooled over gap frames 60-89.

| Scene | Camera | Ghost vs ring | Render vs the departed bottle (gap) | Same, before the gap (bottle present) | Vacated pixels closer to the bottle |
|---|---|---|---|---|---|
| cut_roasted_beef | cam00 / 05 / 17 | **-16.2 / -14.2 / -16.5 dB** | 28.4 / 27.0 / 29.1 dB | 29.0 / 27.9 / 29.6 | 70 / 73 / 72 % |
| flame_steak | cam00 / 05 / 17 | **-17.4 / -17.1 / -19.7** | 29.7 / 29.0 / 28.0 | 29.3 / 28.7 / 27.2 | 98 / 99 / 98 % |
| sear_steak | cam00 / 05 / 17 | **-19.2 / -17.7 / -20.4** | 26.8 / 26.1 / 28.2 | 23.4 / 22.2 / 26.8 | 95 / 95 / 99 % |

During the absence the render matches the departed bottle as closely as it matched the bottle while it was there.
Montage (`assets/.../montage_real_mono.png`, cam00 frame 75: original | fixture | render | render with mask): the bottle
is fully present in all three renders. **The failure reproduces on real footage, 3/3 scenes x 3 held-out cameras,
14-20 dB, the same size as on the synthetic suite (18.7-20.4 dB).** Caveat: the absence is authored (gap ground truth
is a teacher render / composite), not filmed; the fixture is only claimed to be consistent with the training sweep.

**Static-bank seed intervention (mechanism test).** The same synthetic monocular runs as section 21. One change: the
departed object's points are deleted from the starting cloud (`multi4d_notarget` trees; 168-204 points removed by
exact part id). Runs `m4d_monoNT_<scene>_rep1`, all other inputs identical.

| Scene | Published monocular (sec. 21) | Object points removed from the seed |
|---|---|---|
| Bottle | -20.3 dB | -19.2 |
| Carton | -20.4 | -21.6 |
| Mug | -18.7 | -14.7 |

**Removing the object from the initialisation does not remove the ghost.** The time-invariant static bank re-acquires
the object by densification from the ~90 % of training frames in which it is present. Nothing in the representation can
then switch it off for the 30 frames in which it is seen absent. Together with section 20 (turning transient
densification on does not fix it either), the cause is not where the object starts or which densifier runs. It is that
the component holding it has no notion of when it exists. This rules out the two cheapest "configuration" explanations
and points at the fix family: per-primitive presence in scene time (a presence gate) on the static component.

**Where the failure stands.** "Observed absence is not represented":
- synthetic suite: 3 scenes; two method families in their published 20-camera regimes; three further methods at 4 cameras;
- real footage: 3 scenes x 3 cameras;
- two configuration interventions fail to remove it.
Next: kill-argument and novelty re-check against this statement. Then the commit decision by the 24 Oct checkpoint
(hard stop 28 Oct). GC-4DGS still waits on the MVSFormer weights.
