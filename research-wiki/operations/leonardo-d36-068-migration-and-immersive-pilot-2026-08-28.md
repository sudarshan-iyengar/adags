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
