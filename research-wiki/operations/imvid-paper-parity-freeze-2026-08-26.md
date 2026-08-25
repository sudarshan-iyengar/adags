# FROZEN — ImViD Opera/Puppy paper-protocol parity: windows, split, metrics, arms, endpoints (2026-08-26)

Status: **FROZEN BEFORE ANY CAM 00 RESULT EXISTS.** EXPLORATORY,
`evidence_bearing: false` until the strengthening in §10 closes.

Authority: the 2026-08-26 user directive establishing paper-protocol-aligned
ADAGS starting points on ImViD Scene 1 Opera and Scene 6 Puppy, before any
episodic-gating experiment. That directive is explicit on several points that
supersede earlier ImViD records; every supersession is named in §9 and none
is silent.

Branch `apollo/imvid-paper-parity`, forked from `22daf58`.

## 1. What this lane may and may not claim

Three kinds of parity are distinguished throughout, and they are never
conflated:

| | meaning | status here |
|---|---|---|
| **protocol parity** | 300 frames, 2x raster, Cam 00 held out, PSNR/SSIM/LPIPS-AlexNet | **CLAIMED** |
| **window parity** | the exact take and frame indices the paper evaluated | **Opera: constrained by measurement (§3). Puppy: NOT AVAILABLE.** |
| **method parity** | the authors' implementation | **NOT AVAILABLE** — no training or evaluation code is released |

No run from this lane may be called an exact reproduction of the ImViD
paper. The published table is an external reference, never a tuning target,
and **no Cam 00 number may be looked at before §7's freeze is in force.**

## 2. Published reference (external anchor, NOT a target)

Journal extension, arXiv **2604.09473**, **Table III**, whose caption states
that metrics are calculated on the test views across all frames, 300 frames
for each scene. Protocol sentences, quoted short: "we select 300 frames for
evaluation"; "we hold out a single viewpoint for testing", with "Camera 0
for ImViD, N3V, and MeetRoom"; ImViD is "evaluated using 2x downsampled
inputs"; "We report PSNR, SSIM, and LPIPS (AlexNet)."

| method | Opera P/S/L | Puppy P/S/L |
|---|---|---|
| Gaussian4D | 25.61 / 0.873 / 0.206 | 20.15 / 0.408 / 0.553 |
| STG | 26.30 / 0.899 / 0.169 | 21.25 / 0.597 / 0.231 |
| IVV (Ours) | 33.51 / 0.916 / 0.070 | 21.53 / 0.607 / 0.135 |

STG is the closest representation-family reference. IVV is a stretch
external anchor optimized "for 30 epochs" where an epoch is one full random
permutation of the training frames; **this lane trains 12,000 iterations and
does not claim to reproduce that schedule.**

**A protocol change between paper versions is recorded so the two are never
mixed:** the CVPR 2025 version (arXiv 2503.14359) held out **cam10** and
labelled Opera's take "Opera_girl"; the journal holds out **Camera 0** and
names no take. Its Opera numbers differ from CVPR's (STG 26.30 vs 28.482),
so the CVPR take label **cannot be transferred**. This lane follows the
journal.

**Underreported by the paper, and therefore not assumed:** which 300 frames;
which take (Opera has 2, Puppy has 3); the baselines' training schedules;
the units of the flow threshold `epsilon_f = 0.1`; every densification,
pruning and point-budget setting; SH degree.

## 3. WINDOWS — Opera is measured, Puppy is declared

### 3.1 Opera: the public sample IS the take's first 300 frames

The paper never says which 300 frames. The public release ships a 300-frame
`scene1_opera.zip`, and we hold **both** it and the full take, so the
question is empirically decidable rather than merely arguable.

`scripts/imvid_align_sample.py` decoded every frame of both to a 32x18 luma
signature and slid the sample's 300-frame block over the full take's 15,215,
scoring by mean absolute difference on centred signatures.

```
cam00   offset 0   score 0.18676   median-offset score 4.10483   (22.0x)
cam20   offset 0   score 0.12339   median-offset score 0.25066   ( 2.0x)
cameras_agree true    offset_delta 0    verdict ALIGNED
```

**Two independent cameras return offset EXACTLY 0 out of 14,916 candidate
offsets.** That agreement is the decisive statistic; the per-camera
separation ratios (2.26x and 1.51x against the best offset outside a
±30-frame guard) are weak precisely because the neighbouring offsets are the
shoulder of the same minimum, not competing alignments. `cam20`'s low median
score reflects a nearly static view, which is why its margin over the median
is small and why a second camera was required rather than trusted alone.

**Opera window: source frames 0-299 of `raw/scene1_opera`.** This is the
only 300-frame window ImViD has ever published for any scene.

**What this does NOT establish:** that the preview is the paper's benchmark
clip. The README offers it "for a quick look" and names no take. A preview
being a prefix is exactly what one would expect of a preview. Opera is
therefore labelled **`paper-protocol-aligned,
window-constrained-by-measurement`** — stronger than Puppy, weaker than
parity.

### 3.2 Puppy: unrecoverable, so the rule is declared instead

No sample exists for Scene 6, the journal names no take, and the download
layout is behind an application gate. Window parity for Puppy is
**unobtainable by any route short of asking the authors.**

**Puppy window: source frames 0-299 of `raw/scene6_puppy`.** Label:
**`paper-protocol-aligned, frame-window-unmatched`.**

**Selection rule, declared before any Puppy training and outcome-blind:**
index 0, for symmetry with the only window ImViD has published (Opera's
sample, measured above to be frames 0-299), and because index 0 is the one
offset that requires no choice. Model performance played no part and may not.

**Disclosed observation, recorded so it cannot later look like a hidden
criterion:** [[imvid-event-census-result-and-closure-2026-08-25]] places a
human-annotated departure at Puppy source frame ~210 with absence from ~240.
Frames 0-299 therefore contain real dynamic content and part of an
occlusion. **This was known before the rule was applied and did not
influence it** — the rule is index 0, not "the window with the event". This
lane creates no episode masks, no boundaries and no gated arms.

### 3.3 Frames are renumbered at extraction

The extracted window is written as frame indices `0..299` and the source
offset lives only in the manifest. The converter derives timestamps as
`frame_index / fps` and the trainer's `time_duration` is `[0, 299/fps]`, so
renumbering keeps every downstream consumer identical to the
300-frame-sample case. With both windows starting at 0 this is currently the
identity, and it is stated anyway because it is the property a future
non-zero window depends on.

## 4. INPUTS — takes, calibration, raster

Both takes verified directly this block, read-only, at zero GPU slots:

| | Opera | Puppy |
|---|---|---|
| frames | **15,215** | **5,936** |
| duration | 253.836917 s | 99.032267 s |
| rate | `60000/1001` | `60000/1001` |
| raster | 5312x2988 `yuv420p` h264 | 5312x2988 `yuv420p` h264 |
| cameras | 39 mp4 + `cameras.txt` (39) + `images.txt` (39) | same |
| camera model | **OPENCV** | **OPENCV** |

**Puppy's camera model was previously unread and is now read from the data**
(the rig-supersession page's check 1 requires exactly this). It is `OPENCV`,
so undistortion is required — but **its intrinsics are its own and Opera's
may not be reused**:

```
Opera  fx 2603.33268646004   fy 2602.2436600602796  cx 2656 cy 1494
       k1 -0.024546867645992888  k2 0.0035148158874614976
       p1 -0.0004507998572363207 p2 -0.00023832152424359775
Puppy  fx 2683.9114386217134  fy 2691.3045861717437  cx 2656 cy 1494
       k1 -0.012648661752274708  k2 0.0018349707972245913
       p1  0.0012571448602828959 p2  0.00011185122103600642
```

Note `p1` differs in SIGN between the two scenes. Applying Opera's
distortion to Puppy would displace every feature while leaving poses and
intrinsics superficially correct — the exact silent failure the admission
matrix warns about. Each scene derives its own maps.

**`images.txt` order is NOT camera order** in either take (image 1 is
`cam28.png` in both). Camera-to-MP4 binding is **by NAME**, never by index.

**Final raster: 2656x1494**, undistorted to PINHOLE offline, at scale 0.5,
principal point by COLMAP's `(c + 0.5) * scale - 0.5` convention. Training
then runs at `resolution: 1` so `utils/camera_utils.py:42-46`'s naive
`c / scale` is the identity and the two conventions never meet.

## 5. SPLIT — Cam 00 only, and a development split that never sees it

The user directive is explicit: hold out **Cam 00 only** for final testing,
train on every other valid camera. This **supersedes** the 4-camera split
frozen on 2026-08-17 (§9).

| split | test | train | excluded entirely |
|---|---|---|---|
| **final** (`paper_cam00`) | `cam00` | the other **38** | — |
| **development** (`dev_cam10`) | `cam10` | the other **37** | **`cam00`** |

`cam00` is excluded from the development scene *entirely* — not merely
untested — so no Cam 00 pixel can reach a development model through any
path. The split is implemented as a frozen named profile table in
`scripts/imvid_to_blender.py`, never as a free-form camera list, preserving
that file's standing rule that a CLI knob is how a frozen split gets changed
by accident.

**Prohibited held-out information:** no `cam00` image, feature, COLMAP
observation, triangulated point, flow field, depth, mask, metric, or derived
training asset may enter initialization, training, stopping or selection for
any arm. Framewise sparse geometry and SEA-RAFT flow are generated for
**training cameras only**.

## 6. ARMS — NF and FG

Both arms share, byte-for-byte: scene, window, processed pixels, split,
trainer, seed, schedule, losses, topology, point-ceiling policy, evaluator,
and container digest. **Their only intended difference is the construction
of the initial population, and the population that construction legitimately
produces.**

- **NF** — no flow. The 300 framewise training-view candidate clouds, each
  point carrying its own frame's timestamp. No flow classification, no
  static deduplication. Static duplication is a *recorded property*, not a
  defect.
- **FG** — SEA-RAFT-guided. Candidate geometry is classified
  static / dynamic / **abstain**; static geometry is initialized once from
  the reference frame; dynamic geometry is initialized at its observation
  timestamp with compact temporal support; abstention is preserved rather
  than forced.

Per-point time reaches the trainer through the PLY `time` vertex property,
which `scene/dataset_readers.py:137-140` already reads into
`BasicPointCloud.time` and `scene/gaussian_model.py:1098` already consumes.
No trainer change is required for temporal centres.

**SEA-RAFT is an explicit SUBSTITUTION for the paper's VideoFlow and is
labelled as such.** The paper's `epsilon_f = 0.1` is **not** transplanted:
its units are never stated, and 0.1 px at 2656x1494 versus 0.1 normalized by
image width differ by a factor of ~2,656. The threshold used here is
determined from measured SEA-RAFT pixel units and recorded with its
derivation.

### 6.1 Anti-vacuity preconditions — FROZEN, and evaluated BEFORE any score

Per the standing rule that a frozen reading rule without a frozen
precondition is how a vacuous instrument returns a clean result, an FG run
is **INVALID** unless every one of these holds. Each is a statement about
the SETUP, never about an outcome:

1. the intended SEA-RAFT assets were opened, and their count equals the
   expected (cameras x adjacent frame pairs);
2. a nonzero number of candidate observations were evaluated;
3. the classification is **not** degenerate — not all-static, not
   all-dynamic, not all-abstain;
4. static points were materially deduplicated (final static count strictly
   less than the per-frame static sum);
5. dynamic points received frame-local temporal support, i.e. their `time`
   column has more than one distinct value;
6. **no held-out asset was read** — the opened-file list contains no `cam00`
   path;
7. the serialized initializer reopens to an identical point count and hash;
8. the trainer's loaded point count equals the initializer's, and is **not**
   the `num_pts` fallback — a mis-named cloud is silently replaced by a
   uniform random fill (`scene/dataset_readers.py:481-491`), and that fill
   would read exactly `num_pts`.

## 7. METRICS AND ENDPOINTS — frozen

**Primary convention: the `--val` path**
(`main.py:2164` -> `validation` -> `utils/mesh_utils.py`). It is the
paper-aligned one and the only one used for any reported table:

- PSNR **pooled** over channels and pixels (`mesh_utils.py:116-119`), not
  channel-split;
- renders **clamped** to [0,1] (`:95`), GT unclamped;
- SSIM `utils/loss_utils.py:34-64`;
- LPIPS **AlexNet** via `torchmetrics`
  `LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)`
  (`mesh_utils.py:75-77`).

The training-time `training_report` path is **unclamped and has no LPIPS**;
it is a progress signal and **may never appear in a reported table**, nor be
averaged with the above.

**Endpoints: iterations 6,000 and 12,000 of ONE continuous trajectory.** A
checkpoint is saved and evaluated at each. **The 6k Cam 00 result may not
influence the 12k trajectory, the configuration, or anything else** — the
run is continuous and its trajectory is fixed before either number is read.

Cam 00 is scored over all 300 frames. Per-frame metrics are preserved; runs,
not frames, are the experimental unit for any uncertainty statement.

**Reported alongside every ImViD number, without exception:**
presentations/unit as well as iterations, because "6k" does not mean on
ImViD what it means on N3V — 300 frames x 38 cameras is 11,400 training
units.

## 8. DEVELOPMENT — bounded, and never on Cam 00

Development uses the `dev_cam10` split only. The search is bounded and its
promotion rule is declared here, before any candidate runs:

**Promotion rule (frozen):** screen candidates at 6,000 iterations on
**both** scenes using Cam 10 only. Promote the candidate with the highest
mean Cam 10 PSNR across the two scenes, subject to a hard guard: **a
candidate is ineligible if it is worse than H0 on EITHER scene by more than
0.10 dB.** At most two candidates are promoted to a 12,000-iteration
development run. Exactly one configuration is then frozen and used for all
four final arms.

Candidate dimensions are restricted to parameters that source inspection
shows actually control the active representation. Recorded from that
inspection:

- `position_lr_max_steps` and `densify_until_iter` are schedule-coupled and
  must be scaled to a 12k run; the 30,000 default is a 30k-schedule value.
- `position_t_lr_init` drives `_t`, the per-primitive temporal centre, and
  is a **genuine** analogue of the paper's temporal-centre learning rate —
  with the caveats that its `-1.0` default silently inherits
  `position_lr_init`, and that it is multiplied by `spatial_lr_scale`.
- **The paper's velocity `2e-3` has NO genuine equivalent under the
  configuration in use.** `_motion_v` is a real per-primitive velocity but
  enters the optimizer only when `motion_model == "poly"`
  (`scene/gaussian_model.py:1580-1582`); every ImViD config uses `"lora"`,
  under which the trainable motion parameter is a coefficient in a learned
  basis, not a velocity. Mapping `2e-3` onto it by name is exactly the error
  the directive forbids, and it is not done.
- **The paper's temporal-extent `3e-2` has NO independent knob at all.**
  `_scaling_t` is driven by `scaling_lr`, shared with the spatial
  `_scaling` (`:1563`, `:1571`). Giving it its own rate would require a new
  `OptimizationParams` field and a new optimizer group — a representation
  change, out of scope for a baseline lane, and recorded as deferred.

Hyperparameters, search tooling, preprocessing chunk sizes and runtime
dependencies may change freely during development. **The window, the split,
the metric, the evaluator, the promotion rule and the final endpoint may
not**, and any post-freeze amendment is append-only and reruns every
affected arm.

## 9. SUPERSESSIONS AND DEVIATIONS — all explicit

1. **Split.** [[imvid-baseline-freeze]]'s 4-camera held-out set
   (`cam00, cam13, cam25, cam38`, 35 training) is **superseded for this lane
   only** by the directive's Cam-00-only split. The earlier page's reasoning
   — that one held-out view is thin for a photometric claim — **stands and
   is not retracted**; it is a reason to weight this lane's conclusions
   accordingly, not a reason to depart from the published protocol this lane
   exists to align with. The 4-camera profile remains the code default.
2. **Hardware.** [[imvid-baseline-freeze]] reserved Hopper/H100 and excluded
   DGX/V100; the 2026-08-25 record flags this as an **open contradiction**,
   because every ImViD cell actually executed (155, 156, 158-160, 164,
   270-272, 277, 279-284) ran on `dgx`. **RESOLVED HERE, user-directed:
   `dgx`/V100 for all ImViD GPU work.** The binding requirement that the
   whole comparison stay on ONE hardware class is preserved, and `dgx` is
   also the class every prior ImViD cell used.
3. **Source data.** Prior verified ImViD artifacts derive from the 300-frame
   SAMPLE. This lane extracts from the **full takes** for both scenes, so
   Opera and Puppy are symmetric and neither is a lower-bitrate re-encode.
   The sample is used only to LOCATE Opera's window (§3.1). The sample-based
   calibration evidence remains valid for what it established.
4. **Initialization.** The 20,157-point 3-frame union
   (`init35/points3d_colmap_union.ply`, sha256 `d5b10be0...fdf71`) is a
   loader proof and is **not** used here; both arms build 300-frame
   framewise geometry, per the directive.
5. **Rig status** remains **supplier-declared**, and Puppy remains
   **uncorroborated** — see §10.

## 10. EVIDENCE BOUNDARY — carried verbatim with every result

> Fixed-rig status for Opera and Puppy is SUPPLIER-DECLARED, not
> independently measured by this project.

Opera has fixed-pose triangulation evidence on frames 0-299 only (1.97% of
its take); **Puppy has none of any kind**, and its transfer carries **no
publisher checksums** — its completeness is byte-count structural only. Per
[[imvid-rig-classification-supersession-2026-08-25]] §7, **no ImViD result
may be promoted beyond exploratory** until the fixed-pose residual is
extended across the take. Because both windows here are frames 0-299,
Opera's existing evidence covers this lane's window exactly and **Puppy's
window gets its first such measurement in this lane**; take-spanning
strengthening remains open and is why this page is
`evidence_bearing: false`.

Nothing here may be described as independently verified rig status, and
Puppy's rig class may not be inferred from the paper's Table II.

---

## AMENDMENT 1 (append-only, 2026-08-26, BEFORE any Cam 00 result exists)

Nothing above is rewritten. These are decisions the freeze left open and
which had to be fixed before the first training cell, recorded here rather
than left implicit in a config.

### A1.1 Capacity policy — shared, and both arms must exercise densification

```
densify_until_num_points  600_000    the SHARED point ceiling, identical in every arm
initial-population cap    300_000    half the ceiling
num_pts                 1_000_000    above any initial cloud
```

The cap exists because of a specific trap rather than for tidiness.
`main.py:1658` gates the ENTIRE densification-and-pruning block on
`get_xyz.shape[0] < densify_until_num_points`, so an arm whose initial
population already meets the ceiling would never densify **and never prune**,
while an arm below it would do both. That is not a difference in
initialization; it is a difference in training regime, and it would be
invisible in the metrics. Capping the initial population at half the ceiling
guarantees both arms enter the same regime.

NF is expected to exceed the cap (300 frames of un-deduplicated geometry) and
FG is not. When the cap binds, the subsample is uniform, **without
replacement**, at a fixed seed, and BOTH the pre-cap and post-cap counts are
recorded. Without-replacement matters: the reader's own subsample at
`scene/dataset_readers.py:498` uses `np.random.randint` and therefore returns
duplicates. `num_pts` is set above any initial cloud so that path never fires
at all.

### A1.2 The development initializer is the FINAL one, and this is a declared limitation

Framewise geometry and flow are built excluding `cam00` only. The development
split additionally holds out `cam10`, so a development arm trains on an
initializer that contains `cam10` observations.

This is stated rather than fixed. It is defensible **only** because the
development phase ranks schedule hyperparameters with the initializer held
FIXED across every candidate: a leak that is identical in all arms cannot
favour one candidate over another, so it cannot bias the ranking the
promotion rule reads. It would NOT be defensible for a reported metric, and
no development number is reported as a result. **`cam00` is excluded from
both, so the final protocol is untouched.** Building a second 37-camera
initializer would double the triangulation cost for no effect on the only
decision development makes.

### A1.3 No intermediate Cam 00 evaluation

`test_iterations` is the final iteration only. `main.py:1646-1651` writes
`chkpnt_best.pth` whenever the TEST psnr improves, and under `paper_cam00`
the test camera IS Cam 00 — an intermediate test evaluation would have the
trainer select a checkpoint on held-out data. `chkpnt_best.pth` and
`best_test_psnr` are **not** reported for any arm. The 6k and 12k numbers
come from a separate `--val` pass over `chkpnt6000.pth` / `chkpnt12000.pth`.

### A1.4 Exposure, stated once and carried everywhere

300 frames x 38 training cameras = **11,400 units**. At `batch_size 2` and
12,000 iterations that is **2.105 presentations/unit**, against **12.63** for
the canonical N3V 300-frame cell — **6.0x less exposure per unit**. This is
the concrete form of the standing warning that "12k" does not mean on ImViD
what it means on N3V, and it must appear beside every number this lane
produces.

### A1.5 Preprocessing changes that are NOT freeze amendments

Recorded for completeness, not because they bear on a claim. PNG encoder
effort in the window extractor was lowered after measurement (7 of 39
cameras in 40 min at the default put one window at ~3.7 h; the whole
extraction now takes ~10 min per scene). **PNG is lossless at every level, so
the decoded pixels are bit-identical** — only encode time and file size
moved. Under §8 this is search/preprocessing machinery and was free to
change.
