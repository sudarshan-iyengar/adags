# OVERNIGHT HANDOVER — ImViD Opera/Puppy paper-protocol parity (2026-08-26)

EXPLORATORY, `evidence_bearing: false`. Protocol frozen in
[[imvid-paper-parity-freeze-2026-08-26]]; results accumulate in
[[imvid-paper-parity-result-2026-08-26]].

**This handover leads with outcomes. Where a phase did not run, it says so
plainly rather than describing the document that was produced instead.**

---

## 1. THE HEADLINE — a published gap closed by measurement

**ImViD's paper says it evaluates "300 frames" per scene and never says
which 300.** We hold both the public 300-frame `scene1_opera.zip` and the
full 15,215-frame Opera take, which makes the question decidable rather than
arguable.

`scripts/imvid_align_sample.py` reduced every frame of both to a 32x18 luma
signature and slid the sample's block over the take:

```
cam00   offset 0   score 0.18676   median-offset score 4.10483   (22.0x)
cam20   offset 0   score 0.12339   median-offset score 0.25066   ( 2.0x)
cameras_agree true    offset_delta 0    verdict ALIGNED
```

**Two independent cameras return offset EXACTLY 0 out of 14,916 candidate
offsets.** Opera's public sample is the take's first 300 frames.

**What this does NOT establish**, and the distinction is the point: that the
preview IS the paper's benchmark clip. A preview being a prefix is exactly
what one expects of a preview. Opera is labelled
`window-constrained-by-measurement`; **Puppy has no sample, no named take,
and its window is declared by an outcome-blind rule and labelled
`frame-window-unmatched`.**

## 2. THE SECOND HEADLINE — a defect found in code nobody was reviewing

**The temporal marginal is applied TWICE.**
`gaussian_renderer/__init__.py` multiplies opacity by
`exp(-0.5 dt^2/sigma)`; `diff-gaussian-rasterization/.../forward.cu`
computes and applies the same marginal again. What renders is the square, so
**every 4D temporal support width renders at 1/sqrt(2) — 70.7% — of its
stored value.**

Proposed by a fresh-context adversarial reviewer with no stake in the code,
then **verified by reading both call sites directly** rather than accepted on
report. Full record and the reasoning for leaving it unrepaired:
[[temporal-marginal-applied-twice-2026-08-26]].

Ratios and same-trainer comparisons are untouched (one global constant, every
primitive, every arm), so the NF/FG comparison is unaffected. Every ABSOLUTE
statement of a support duration in this project is wrong by 41%.

## 3. WHAT COMPLETED, WITH EVIDENCE

| phase | state | evidence |
|---|---|---|
| Branch + push | DONE | `apollo/imvid-paper-parity`, remote verified to resolve to the exact local SHA at every push |
| Dependency closure | DONE | §5 |
| Runtime image | DONE | digest `sha256:02ad9cb41d0b613db73c0cee3777e547899c42dd2b93220edd30317d7f04b1e6` |
| Container validation | DONE | `OVERALL_FAIL=0`; 5 self-test suites, 6 pytest tests, RAFT constructed offline |
| Opera window extraction | **DONE + VERIFIED** | 11,700/11,700 PNGs, 300/camera x 39, IHDR read from every file, 137,142,068,651 B, `suppressed_offset 0.0`, 14m27s |
| Puppy window extraction | **DONE + VERIFIED** | 11,700/11,700, 300/camera x 39, 351,471,430,904 B, `suppressed_offset 0.0`, 59m27s |
| Flow wiring + direction | **DONE, MEASURED** | §6 |
| Arm assembly + splits | **DONE** | `paper_cam00` 38 train / test `['cam00']`, no overlap; `dev_cam10` 37 train / test `['cam10']` / `cam00` excluded; images symlinked, PLY hash preserved |
| **Initialization seam (CUDA)** | **DONE + PROVEN** | all three support bands land, 6,719 rows each = 20,157/3 exactly; 20,157 points at init, NOT the `num_pts` fallback |
| Opera conversion | **DONE + VERIFIED** | 11,700/11,700, 53m11s; derived PINHOLE matches frozen exp-156 (`cx 1327.75`); `invalid_fraction 0.0`; `motion_track_dt 1001/60000`; split profile `paper_cam00`, 38 train cameras |
| Opera flow | RUNNING | 4-way GPU shard, 3.82 pairs/s (0.96 single) |
| Opera triangulation | RUNNING | **1024 features, stride 3, 100 frames** (freeze A5, superseding A4); first frames 2,428 / 2,734 / 2,857 points |
| Puppy conversion onward | NOT RUN | §9 |
| Opera flow | **DONE** | 11,362/11,362 pairs, all 38 training cameras x 299 |
| Opera triangulation | **DONE** | 100/100 frames, 282,672 points, 1024 features, stride 3 |
| Opera NF initializer | **DONE + VERIFIED** | 282,672 pts; support 2.4942 / 0.9994 / 0.1335 s; window span == config `time_duration` exactly; upstream leak check `verified: true` |
| Opera FG initializer | **DONE + PRECONDITIONS PASS** | static 93.71% / abstain 5.24% / dynamic 1.05%; 0 points unseen; 19,825 written |
| **Opera NF training** | **RUNNING** | **Determined experiment 295** |
| **Opera FG training** | **RUNNING** | **Determined experiment 296** |
| Puppy conversion onward | see §9 | |

## 3A. LIVE RUNS AT HANDOVER

```
295  imvid_opera_nf   RUNNING  dgx  commit b9e89bf  seed 0
     /apollo/users/sri/proj_adags/runs/elgs/20260826T020643Z_imvid_opera_nf_0_b9e89bf
296  imvid_opera_fg   RUNNING  dgx  commit b9e89bf  seed 0
     /apollo/users/sri/proj_adags/runs/elgs/20260826T020708Z_imvid_opera_fg_0_b9e89bf
```

Both: image `sha256:02ad9cb4…b1e6`, 12,000 iterations, checkpoints at 6,000
and 12,000, `test_iterations` at the final iteration only so no checkpoint can
be selected on Cam 00.

**Expect very different wall clocks, on an ESTIMATE not a measurement.** NF
carries 282,672 initial points against FG's 19,825. The only measured
per-iteration figure on this trainer at this raster is the earlier
500-iteration plumbing smoke at 20,157 points, which ran ~1 s/iteration
including startup. A dedicated NF preflight was launched and **killed before
it printed**, so no per-iteration figure for a 282k population has been
measured; the expectation that NF is several times slower is inference from
the point count, and is labelled as such. The true rates are recorded when
`chkpnt6000.pth` lands in each run dir.

The asymmetry itself is the legitimate consequence of the two constructions,
not a defect.

Monitor: `python scripts/det_monitor.py`, or
`det e describe 295 --json` / `det task logs -f <task-id>`.

Evaluate each saved checkpoint through the FROZEN `--val` path once it exists:

```bash
.\submit_val.ps1 -Cell imvid_opera_nf_val6k -Config configs/imvid/opera_paper12k_nf.yaml -Ckpt <run_dir>/chkpnt6000.pth
```

## 4. PROVENANCE

```
branch  apollo/imvid-paper-parity   (forked from 22daf58)
pool    Apollo dgx / V100 (Tesla V100-SXM2-32GB, compute 7.0, 32768 MiB)
image   sudarshaniyengar/adags@sha256:02ad9cb41d0b613db73c0cee3777e547899c42dd2b93220edd30317d7f04b1e6
        tag apollo-v100-searaft-afc4200
SEA-RAFT princeton-vl/SEA-RAFT @ 886fb094fe21d4fa5ff675da18362b27b023ccc3 (cloned INTO the image)
ckpt    Tartan-C-T-TSKH-spring540x960-M.pth
        sha256 adcc169244e99d4e6fe645b60aa8eaf3e4263698a3e870b8fbae618e3d2acc28 (hash-verified per run)
```

Two superseded image digests are recorded in the result page; one of them
(`sha256:44a373ec…`) produced the window extractions and differs from the
final image only by `pytest`, a `chmod`, and a newer helper script — none of
which touch pixel output.

## 5. ENGINEERING FAILURES ENCOUNTERED AND REPAIRED

None of these is reported as a blocker; all were diagnosed, repaired, tested
and passed.

1. **SEA-RAFT's `requirements.txt` omits two of its own hard imports** —
   `huggingface_hub` (imported at `core/raft.py` module top level; `RAFT`
   inherits from `PyTorchModelHubMixin`, so it fails before a checkpoint is
   touched) and `h5py`. Conversely `einops`, `matplotlib` and `tensorboard`
   ARE listed and imported nowhere; deliberately not installed.
2. **Implicit relative imports** — `core/raft.py:7` is `from update import
   BasicUpdateBlock`, so `core/` must be on `sys.path` in its own right.
   Caught by the build-time gate.
3. **An unconditional ImageNet download at model construction** —
   `RAFT.__init__` fetches ResNet-34 weights the checkpoint then completely
   overwrites. Seeded into the image.
4. **A stale baked helper.** The build passed because the build step set
   `TORCH_HOME` on its own command line; the runtime does not.
   **Method point: a build-time gate proves the build environment, not the
   runtime one, whenever it supplies a variable the runtime will not.**
5. **Root-owned weights.** Build runs as root, tasks do not; the file was
   present, `TORCH_HOME` pointed at it, and the task still died on a
   `PermissionError` surfacing deep inside torchvision's loader.
6. **`pytest` absent from the image** — so the unit tests could not run in
   the only environment that has torch, cv2 and plyfile.
7. **Undistortion at 0.35 images/s** — 9.3 h per window, over eighteen for
   two scenes. Parallelised with threads (cv2 releases the GIL; the maps are
   ~32 MB so processes would cost more than the work) to **2.12 images/s**.
   The bytes do not depend on worker count; only the wall clock does.
8. **PNG encoding dominating extraction** — 7 of 39 cameras in 40 min at
   ffmpeg's default effort. PNG is lossless at every level, so the decoded
   pixels are bit-identical; only time and size moved. Extraction fell to
   ~13 min per scene.
9. **Per-frame COLMAP scratch accumulating** — tens of GB in `/tmp` over 300
   frames; the whole workdir is now removed once its points are on shared
   storage.
10. **The triangulation cost model was WRONG, and the correction improved the
    plan.** Planning assumed SIFT feature count scales with pixel area, so
    halving the raster would cut matching ~16x. COLMAP caps detections at
    8192 and a detailed scene saturates that cap at BOTH rasters, so the
    dominant term never moved: **>= 14 min per frame on all 80 cores against
    a projected 90 s**. The cap is the lever the raster is not, and unlike
    `--max-image-size` it leaves keypoint localisation — and therefore the
    2.0 px gate — untouched. At 1024 features a frame costs **33 s** with all
    38 cameras still contributing. The replacement is MORE temporal coverage
    than the stride it superseded (100 frames vs 50), paid for in points per
    frame, which is the right way round for an arm whose mechanism IS
    per-frame temporal assignment.
11. **The point collector read the EMPTY INPUT model.** It globbed for
    `points3D.txt` and matched `model_in/`, which sorts before `model_txt/`,
    reporting **0 points from a reconstruction with 38 contributing cameras
    and mean track length 4.11**. Unrepaired it produces one cloud file per
    frame, an initializer with no geometry, and success codes throughout.
    Caught only because a cost probe happened to print the point count beside
    the COLMAP statistics.

## 6. FLOW — direction is MEASURED, and the first test was wrong

Adjacent-frame motion on Opera, measured at the 2656x1494 training raster in
full-raster pixels: **mean 0.12-0.23 px, p99 0.93-3.69 px, max 7-12 px.**
Fields verified `(747, 1328, 2)` float16, finite, `stored_raster
[1328, 747]`, `source_raster [2656, 1494]`.

**The direction test failed twice before it was trustworthy, and the failure
was in the test.** Averaging warp error over the whole frame gave ratios of
1.083 (gap 1) and 1.132 (gap 8) against a 1.2 floor — for a field that is in
fact correctly oriented. The cause: an opera stage is ~95% static, and static
pixels warp to the same place under `+flow` and `-flow`, so they dominate the
mean. **A direction test has to be evaluated where direction is observable.**
Restricted to pixels carrying >= 1.0 px of displacement, with a minimum
evidence count below which the test ABSTAINS loudly rather than returning a
ratio from a handful of pixels:

```
cam01 frame 1  gap 8  evidence_pixels 83742  forward 9.772  reversed 13.275  ratio 1.358
cam01 frame 2  gap 8  evidence_pixels 89263  forward 9.151  reversed  ...    ratio 1.399
```

**Flow is FORWARD, confirmed.** This matters because this project has twice
been caught by flow orientation and nothing downstream could detect a
reversed field — it would simply classify the wrong points as dynamic.

Also recorded: the paper's `epsilon_f = 0.1` is **not** transplanted; its
units are never stated and the two readings differ by ~2,656x. The measured
magnitudes above are consistent with a PIXEL reading being the plausible one
at 60 fps, but that remains inference and is labelled as such.

## 7. INDEPENDENT REVIEW — findings accepted and repaired

A fresh-context adversarial review of the representation-critical change
returned findings that were **verified before acceptance**, several of which
would have produced a wrong number that looked right:

- **The reader dropped the new per-point column** in its subsample branch, so
  a cloud larger than `num_pts` lost per-point temporal support entirely
  while every manifest still reported three populated bands. The only defence
  was an unenforced assertion in a config comment.
- The same branch filtered times with **strict** inequalities, deleting
  everything at exactly `t = 0` — the whole reference-frame static set.
- **The anti-vacuity guards were themselves vacuous**: exact-equality
  degeneracy over incommensurable counts (five static points among six
  million passed); a fully collapsed static set passed and reported a
  duplication reduction of 1.0; the held-out leak check could not fire for
  the NF arm at all.
- A strided-view / `torch.from_numpy` incompatibility on a path no cloud in
  the repository had ever exercised.
- `rot_4d` inverts the extent with a different exponent — now refused rather
  than silently mis-scaled.
- The tests were tautological — `(x**0.25)**4 == x`, importing neither
  module, discriminating the exponents at the one value where they differ by
  0.1%. Rewritten to round-trip the real writer and reader and to pin the
  exponent against the trainer's source.

**The reviewer also found §2**, in code entirely outside the change.

## 8. RESOURCE SPEND

All preprocessing ran at **ZERO GPU slots**. GPU use so far is limited to
four short SEA-RAFT smokes at 1 slot (~1 minute of compute each). **No
training cell has been submitted, so no training slot-hours have been
spent.** Experiments 295+ are unused; the last ledger entry remains 294.

## 9. WHAT IS INCOMPLETE, AND THE EXACT RESUME

Live state at handover time is recorded in the result page. The pipeline is
implemented, tested and validated end to end; what remains is wall clock.

Resume, in order (contexts are `git archive` materialisations of the branch
HEAD; the image digest is in §4):

```bash
# 1. per scene, once its window is extracted: convert + framewise triangulate
bash stage_a.sh opera scene1_opera
bash stage_a.sh puppy scene6_puppy

# 2. as soon as a scene's conversion finishes (GPU; runs alongside 1's COLMAP)
bash stage_flow.sh opera

# 3. once BOTH clouds/ and flow/ exist for a scene
bash stage_pop.sh opera scene1_opera

# 4. cost preflight BEFORE committing four 12k trajectories
bash preflight_cost.sh configs/imvid/opera_paper12k_nf.yaml <run_dir>

# 5. the final arms
.\submit_arm.ps1 -Cell imvid_opera_nf -Config configs/imvid/opera_paper12k_nf.yaml -Hours <measured>
.\submit_arm.ps1 -Cell imvid_opera_fg -Config configs/imvid/opera_paper12k_fg.yaml -Hours <measured>

# 6. evaluate each saved checkpoint through the frozen --val path
.\submit_val.ps1 -Cell imvid_opera_nf_val6k  -Config ... -Ckpt <run_dir>/chkpnt6000.pth
.\submit_val.ps1 -Cell imvid_opera_nf_val12k -Config ... -Ckpt <run_dir>/chkpnt12000.pth
```

Monitoring: `python scripts/det_monitor.py`, or
`det task logs -f <task-id>`.

**The single most important thing to check first** when the FG populations
build: the manifest's `classified_shares`. If one class holds >= 99% of
classified observations the run REFUSES as INVALID — and the most likely such
outcome is *all-dynamic*, because the classifier takes a maximum over 38
views with no occlusion test, so a static point occluded in any one view
samples the occluder's flow. That is the paper's own stated rule ("dynamic if
flow exceeds the threshold in ANY view") and is therefore parity, but it is
disclosed and measured rather than assumed.

## 10. RECOMMENDED STARTING POINT FOR THE LATER GATING PAIR

**Deferred — and deliberately not guessed.** The recommendation between NF
and FG is exactly what the 6k/12k Cam 00 table is for, and no Cam 00 number
exists yet. What can be said now:

- The **schedule** is settled and outcome-blind: 12,000 iterations,
  `position_lr_max_steps = 12_000`, `densify_until_iter = 10_000`, derived by
  scaling the canonical N3V 300-frame config by its own two stated couplings,
  consulting no Cam 00 result.
- **Exposure must travel with every number**: 300 frames x 38 cameras =
  11,400 units, 2.105 presentations/unit at batch 2 — **6.0x less than the
  canonical N3V 300-frame cell**. "12k" does not mean here what it means
  there.
- The **split** for any later matched pair is `paper_cam00`, and the
  development profile `dev_cam10` excludes Cam 00 from the scene entirely.
