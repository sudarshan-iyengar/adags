# Leonardo move to EUHPC_D36_068, and the first Google Immersive pilot (2026-08-28)

Operational/engineering record. **EXPLORATORY**, `evidence_bearing: false`.
Append-only.

## 1. Why the move

`saldo -b` reports **EUHPC_D21_034 ended 2026-07-30**. Every Slurm submitter in
the repository still named it, so the whole Leonardo path was charging a dead
budget. **EUHPC_D36_068** runs to 2027-05-19 with 7.9% of 144,000 local h used.

New root: `/leonardo_work/EUHPC_D36_068/sri/proj_adags`. Note it is **one level
deeper than `$WORK`**, which is `/leonardo_work/EUHPC_D36_068` — so every
`${WORK}/proj_adags` in the repo resolves to the wrong place under the new
account. That is why `exp_index/leonardo_env.sh` now derives
`ADAGS_PROJECT_ROOT` from its own location.

## 2. What was migrated, and what it cost

| item | route | verification |
|---|---|---|
| `CLAUDE.md`, `exp_index/leonardo_env.sh` | `cp -a` | — |
| `repo/adags` | fresh `git clone` at `apollo/csvl-vpl-v2-exploratory` | HEAD matches origin |
| `repo/depth-anything-3`, `repo/SEA-RAFT` | `cp -a` | **file count AND byte total identical** (214 files / 49,265,142 B; 101 / 96,008,998 B) |
| `envs/` 14 GB, `models/` 6.3 GB | `rsync` in a `lrd_all_serial` job (21 min) | rsync stats matched |
| CUDA extensions | copied `.so` from the old tree | sha256 identical; `.so` sets across the two repos now diff-clean |

`runs/` is a real directory, not the symlink into
`/leonardo_scratch/fast/EUHPC_D21_034` that the old tree used.

## 3. Three breakages, each of which fails SILENTLY

**(a) A copied venv points back at the source tree.** `pyvenv.cfg`, every
`activate*` and every console-script shebang hardcode the creation prefix.
Observed directly: after copying, sourcing the *new* `leonardo_env.sh` yielded
`/leonardo_work/EUHPC_D21_034/.../bin/python`. It does not error — it just runs
the old interpreter. 132 files repaired by
`logs/repair_venv_prefix.py`; `sys.prefix` now resolves inside the new tree.

**(b) The account was unoverridable in four places.** Six submitters already
read `${ACCOUNT:-...}`, but `runit.sh` (x2), `run_panopticsports.sh` (x2),
`submit_exploratory_lane.sh` and `submit_phase0_census.sh` hardcoded
`euhpc_d21_034` with no environment override. Fixed in commit `813ceec`; no
`euhpc_d21_034` remains in any shell script and all ten pass `bash -n`.
Historical references under `refine-logs/` and `research-wiki/` are provenance
and were deliberately left alone.

**(c) The extensions are EDITABLE installs whose `.so` live in the repo tree.**
A fresh `git clone` has none, and `.so` files are not tracked. The failure
surfaces as `ModuleNotFoundError: No module named 'simple_knn._C'`.

**A test that was itself wrong, recorded because it cost a job.** The first
verification asserted `import diff_gaussian_rasterization` at top level and
"failed". The code never does that — `gaussian_renderer/__init__.py:16` does
`from .diff_gaussian_rasterization import ...`, a relative import of the
wrapper module, which JIT-builds the extension at
`-gencode=arch=compute_80,code=sm_80`. The **old** environment fails the
top-level import too, which is what proved the test rather than the environment
was at fault.

## 4. The A100 gate passed, and it settles an Apollo question

Job 54664419/54665148 on `boost_usr_prod`: A100-SXM-64GB, capability (8,0),
`simple_knn._C`, `pointops2_cuda`, the rasterizer, `gaussian_renderer.render`,
`scene.GaussianModel` and `main` all import. **`distCUDA2` on 366,366 points
returns finite values with 62.8 GiB of 63.4 GiB free.**

That is the exact call that fails on Apollo's `dgx` V100 pool with
`cudaErrorMemoryAllocation` **on a completely idle node** (Apollo experiments
319-325, 335). Running identically here confirms the Apollo failure is
environmental — a node or image property — and not intrinsic to the workload.

An N3V integration cell (job 54666321, 200 iterations) then trained on the new
tree, PSNR 13.9 -> 25.8 on 366,366 points. That separates "the tree is broken"
from "the new dataset is broken" before Immersive was introduced.

## 5. Google Immersive: acquisition and preprocessing

`02_Flames.zip` re-acquired on Leonardo: 5,474,948,990 B, publisher MD5 match,
**sha256 `0209febf…` identical to the Apollo acquisition of 2026-08-18** —
independent corroboration across two machines and two downloads. The read-only
inventory reported **zero drift** against the recorded 15-object table.

Two new scripts (commits `f38571e`, `0493dc4`):

* `scripts/immersive_decode_frames.py` — 45 calibrated cameras of 46 shipped;
  **`camera_0046` identified and dropped by NAME**, matching the recorded
  per-scene pattern. 50 frames each, native 2560x1920 validated against
  `models.json`, 13 GB on scratch, 6m41s.
* `scripts/immersive_to_blender.py` — fisheye -> pinhole at 1280x960
  (the ImViD paper's 2x downsample), `transforms_train/test.json`,
  `points3d.ply`, held-out `camera_0001`.

**A defect the point-count floor caught.** The first conversion refused at
4,213 points against a 5,000 floor. The shortfall was not tuning: taking each
camera's k nearest neighbours and then skipping `name_b <= name_a` drops every
pair whose neighbour sorts earlier, so a camera whose k nearest all sort below
it contributes nothing, unevenly across the dome. Pairs are now unordered and
deduped. **Lowering the floor would have hidden the defect instead of finding
it.**

## 6. The validation that actually decides it

Everything above could hold with the calibration silently wrong. Two checks
that could not:

**The publisher's own worked example.** The dataset README ships a projection
snippet and its expected output. `--self-test` reproduces
`[1377.855, 1017.614]` to 2e-2 px, and asserts **`cv2.fisheye` agrees with the
publisher's model to <1e-6 px** — so the OpenCV path is validated against their
code rather than assumed equivalent to it.

**Epipolar consistency, which never touches the triangulated cloud.** Using
`models.json` poses only, SIFT matches between adjacent cameras on the
undistorted images give median Sampson error:

| pair | matches | baseline | median | p90 |
|---|---:|---:|---:|---:|
| 0002/0003 | 73 | 0.178 m | **0.21 px** | 2.30 |
| 0010/0011 | 27 | 0.185 m | **0.15 px** | 50.69 |
| 0020/0021 | 29 | 0.177 m | **0.30 px** | 105.49 |
| 0030/0031 | 146 | 0.157 m | **0.15 px** | 0.62 |

Sub-pixel medians validate the axis-angle -> R conversion, `t = -R.C`, the
fisheye undistortion and the new intrinsics *together*. The p90 tails are
outlier matches in a dark scene. **This is the check that made the pinhole port
believable; nothing before it was decisive.**

**Two alarms that turned out not to be defects, recorded so they are not
re-raised.** The undistorted frames are ~54% "black" at threshold 4/255 — but
the SOURCE fisheye frames are already 32% black with mean brightness 21/255,
roughly uniform in radius. `02_Flames` is simply a very dark scene and the
metric was measuring content, not invalid pixels. Likewise the point cloud
centroid sits ~6 m from the origin while the rig is a sub-metre dome; the rig
looks OUTWARD, so distant content is expected, and the epipolar check rules out
the calibration explanation.

## 7. Standing limitation

**Numbers from this path are NOT comparable to published STG or ImViD
Immersive figures.** STG trains Immersive in fisheye space, warping the render
into fisheye at loss time through per-camera inverse flow maps;
`scene/cameras.py`'s `fisheyemapper` is a dead `None` and ADAGS has no such
path. This is a pinhole port — a different method. It is stated in the
converter docstring, in the config header, and in every emitted manifest, and
it must survive into any table.

## 8. Status

Training job 54674975 (6,000 iterations, 50 frames, 45 cameras, held-out
`camera_0001`), eval 54675528 chained `afterok`. Confirmed at launch:
`Found transforms_train.json` (the Blender branch, as intended) and **7,248
points at initialisation — the real cloud, not the reader's silent random
substitute**. Results append below, never above.

---

## APPENDIX (2026-08-28, append-only) — the pilot OOM'd, and it was fragmentation not capacity

**Job 54674975 FAILED at iteration 4,640 of 6,000** after 36m44s, at 426,916
points, having reached training PSNR 28.68. `sacct` gives exit `1:0` with no
Python traceback in stderr; the diagnosis is in stdout:

```
55.05 GiB in use.  Of the allocated memory 16.54 GiB is allocated by PyTorch,
and 37.94 GiB is reserved by PyTorch but unallocated.
```

**Only 16.5 GiB of a 64 GiB A100 held real tensors.** The other 38 GiB was
reserved and fragmented. Densification churns allocations of steadily changing
size, which is the pattern that fragments the caching allocator, and the point
count was still climbing toward the 600,000 cap.

Resubmitted (job **54687155**) with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
the setting PyTorch's own message names. **Confirmed effective**: the retry
passed 4,780 iterations at **457,222 points** — beyond the previous run in both
iteration and point count.

**Set via the job export, deliberately NOT in `scripts/run_leonardo.sh`.**
Changing the allocator repo-wide would alter memory behaviour for every
Leonardo lane, including the flow and variance work whose comparisons rest on
runs being bit-identical outside the variable under test. A memory-layout
change is not obviously neutral there, and this lane does not need it to be.

The failed run is preserved rather than overwritten, per the standing rule on
failed and superseded experiments.

**What this does NOT indicate.** Nothing upstream of training was implicated:
poses, undistortion, the split and the seed cloud were all validated before
this run and none of them changed. It is a scaling property of the trainer on
this raster and point budget, and the same fragmentation would be expected on
any lane that densifies to several hundred thousand primitives at 1280x960.

---

## APPENDIX 2 (2026-08-28, append-only) — the pilot number was wrong for a findable reason, and the corrected result

The first completed pilot (job 54687155, `focal_scale 0.5`) scored **17.90 dB
held-out, SSIM 0.535**, peaking at iteration 3,000 and degrading after. It
would have been easy to file that as "a pinhole port on a hard dark scene".

**The tell was the TRAINING number, not the held-out one.** 23.99 dB on
training views after 6,000 iterations, when the N3V integration smoke reached
25.8 dB in 200. A model that cannot fit the views it is optimising against is
not being limited by generalisation.

### What it was

Measured exactly — the fraction of output pixels whose remap source falls
outside the native fisheye raster, not inferred from brightness:

| `focal_scale` | invalid fraction, worst of 45 cameras |
|---|---:|
| 0.50 | **33.2%** |
| 0.75 | 2.94% |
| 0.80 | 0.67% |
| **0.85** | **0.000%** |

A third of every training image was `BORDER_CONSTANT` black with no source
pixel behind it. The trainer cannot distinguish fabricated black from scene
black, so it spent capacity fitting a constant, and the loss it minimised was
one third fiction.

**The error was carrying a constant across a method boundary.** 0.5 is STG's
undistorted-path focal scale, and it is safe *there* only because STG never
trains on the undistorted images — it trains in fisheye space and those frames
are an intermediate. Copying the number without its precondition is the whole
mistake.

**It was nearly missed twice.** The earlier "54% black" reading was correct and
was explained away as the dark scene — the scene *is* dark (source frames 32%
black at threshold 4/255, mean brightness 21/255), which made the wrong
explanation fit. Only the exact remap measurement separated the two.

### The corrected pilot

`02_Flames`, `focal_scale 0.85`, jobs 54698858 / 54698863 / 54698864,
convert 6m27s, train 1h08m, eval 4m39s:

| quantity | fs 0.50 | **fs 0.85** |
|---|---:|---:|
| invalid fraction (max over cameras) | 33.2% | **0.000%** |
| seed points | 7,248 | **13,328** |
| held-out PSNR | 17.90 | **26.72** |
| held-out SSIM | 0.535 | **0.852** |
| LPIPS-alex (reference, `normalize=True`) | — | **0.2606** |
| LPIPS-alex (3DGS-inherited convention) | — | **0.1881** |
| best iteration | 3,000 (then degrading) | **6,000 (still improving)** |
| final primitives | 479,852 | 599,744 (at the 600k cap) |

**+8.83 dB and +0.317 SSIM from one preprocessing constant.**

Two things worth carrying:

**The two LPIPS conventions differ by 28% relative on real renders**
(0.2606 vs 0.1881), which is larger than the 18.4% measured on DiVa-360 and
far larger than the spacing between published methods. Emitting both was not
bookkeeping; a table naming only one would be uninterpretable.

**The run is compute-limited, not over-densified.** It hits the 600k primitive
cap and `best_val_iter` is 6,000, i.e. still improving at the schedule's end —
unlike the fs 0.50 run, which peaked at 3,000 because the extra capacity was
going into fabricated black. Any future comparison at this protocol should
state that 6,000 iterations is a truncation, not a converged endpoint.

### Guard

`immersive_to_blender.py` now measures the invalid fraction per camera and
**refuses** above `--max-invalid-frac` (default 0.5%), with `focal_scale`
defaulting to 0.85 (commit `345af35`). Same shape as the `points3d.ply` floor:
the failure produced a plausible-looking scene, a completed run, and a quietly
wrong number.

Two fixes from this block that stand independently of the focal scale:
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for allocator fragmentation
under densification, and LPIPS weights pre-cached into `TORCH_HOME` from a
login node because **compute nodes have no egress** — the eval died on
`URLError [Errno 101] Network is unreachable`, the same failure this wiki
already recorded for job 48760029.

### Fan-out

The other six STG scenes are acquired (32.2 GiB, on scratch) and chained
decode -> convert -> train -> eval. Note the Slurm **submit cap is 20 jobs per
user**: five scenes filled it exactly and `12_Cave` was refused with
`QOSMaxSubmitJobPerUserLimit`, with no orphan jobs created. It is submitted
once the queue drains.
