# Multi4D on N3V `cut_roasted_beef` — Leonardo reproduction (2026-09-18)

Status: **COMPLETE. Reproduction PASSES the rule frozen before it ran —
3 of 3 repetitions inside the 33.4-34.0 band.** Results in section 9; nothing
above that section was rewritten after the fact.

Substrate pin and every environment/licence fact:
[`research-wiki/assets/multi4d-pin.json`](../assets/multi4d-pin.json).

## 1. Why this run exists

The 2026-09-17/18 reading pass moved the project off its own substrate. Exact-zero
absence turned out to be free for every temporal-opacity method (the rasterizer
culls at `alpha < 1/255`; Multi4D additionally hard-culls `p(t) < 0.05` in CUDA),
so the claim no longer lives there. It lives on the **return**: whether the same
primitive lineages resume after a full-multiview absence, where single-interval
temporal opacity ghosts or respawns.

Multi4D (ECCV 2026, GPL-3.0) is one of two candidate substrates; Ex4DGS is the
other and comes second. Before any novelty test on either means anything, each
must reproduce its published `cut_roasted_beef` number **with its code
unmodified**. That is the whole content of this phase. Nothing here is a claim
about absence or return.

**Control model, set by the user 2026-09-18:** reproduction (Phases 0–2) is
autonomous; everything that changes substrate code (Phase 4 on) is a
review-gated small step, and the user does the evaluation and analysis.

## 2. The acceptance rule, frozen before anything ran

Declared 2026-09-18T21:30Z, before the first training job was submitted:

* **Band:** peak PSNR **33.4–34.0**, centre ≈ 33.85 — the authors' own band from
  the `validate_repro.sh` they dropped from the released tree. The paper reports
  **34.02**.
* **PASS** = at least **2 of 3** repetitions inside the band.
* **Reported quantity:** peak PSNR over the 500-iteration evaluations on cam00.
  This is *selection on the test camera*, and is reported as such; the
  final-iteration value is reported beside it, as is the `best/` value (which the
  code gates to `iteration > 16000`, i.e. the peak over the last 8 of 40 evals).
* Alongside: SSIM in **both** conventions the code computes, LPIPS-alex as the
  repo computes it (no `normalize=True`), Gaussian count per subset, model size,
  wall time.

**Declared limit of the three repetitions.** `train.py` hardcodes
`setup_seed(6666)` and exposes no `--seed`. The three reps therefore differ only
by CUDA kernel nondeterminism. Their spread is a **lower bound** on run-to-run
variance and must never be cited as a seed spread. This project has been bitten
before by reading a fixed-seed replicate spread as if it were a seed spread
(2026-08-23: same-code same-seed runs separated by 0.089–0.236 relative RMS
training loss before any intervention).

## 3. What the static audit found before any GPU time

Full evidence with `file:line` sits in the audit; the load-bearing items were
independently re-read against the checkout before being acted on. The ones that
changed the plan:

1. **Frame extraction is `cv2.VideoCapture`, not `ffmpeg`.** Leonardo has no
   ffmpeg module, so this was the first thing that could have blocked the lane.
   It writes `<scene>/camNN/images/%04d.png` at **1352×1014** (native 2704×2028
   halved, LANCZOS), exactly 300 per camera, guarded by a fail-closed count
   check, and skips on directory existence alone.
2. **Extraction writes next to the mp4 it decoded** (`video_path.split('.')[0]`).
   Run against the raw N3V tree it would deposit ~12 GB of PNGs into a
   read-only-by-policy dataset directory. The staging dir of symlinks prevents
   this: `glob` returns the staging path, so the PNGs land in `data_staging/`.
3. **`fused.ply` is read by that exact filename with no fallback and no
   try/except.** `cut_roasted_beef` ships `points3d.ply` (366,366 pts) with a
   byte-compatible header, which would have been the tempting substitute and
   would have been wrong — the paper initialises the static branch from the
   DeGauss **dense** cloud. The real `fused.ply` (386,905 pts) was supplied by
   the user and uploaded unchanged, sha256 verified on both ends.
4. **LPIPS first fires at fine iteration 500** and downloads two torchvision
   backbones. Compute nodes have no network, so an uncached run trains ~2,500
   steps and then dies. `$W/cache/torch` already held both files from earlier
   work; `TORCH_HOME` points there.
5. **The hybrid rasterizer passes no `-gencode` flags.** Torch derives them from
   the *visible* device, so a build with no GPU visible raises
   `IndexError: list index out of range`, and a build warmed on a non-A100 would
   omit sm_80 SASS. `TORCH_CUDA_ARCH_LIST=8.0` is exported explicitly to make the
   JIT build deterministic across nodes.
6. **Resume is unusable.** `--start_checkpoint` restores the **foreground branch
   only**; the static/background and transient branches are silently
   re-initialised, and the `_gs2_` background checkpoint that the same code
   *writes* is never read back. Worse than restarting, because the foreground has
   adapted to a background that no longer exists. Every repetition is therefore a
   single job sized to complete all 22,000 iterations in one allocation.
7. **`debug_metrics.json` is opened in append mode** and `expname` also names the
   results directory, so two runs sharing an `expname` interleave their records
   in one file with no run marker and overwrite each other's `best/`. Each
   repetition gets a unique `expname` and its own `--saving_folder`.
8. **Host RAM, not GPU RAM, is the binding constraint.** `train.py` materialises
   all 19×300 training `Camera` objects into a list, each retaining its decoded
   uint8 image ≈ **21.8 GiB**, and every evaluation rebuilds the 300 test cameras
   on top (≈1.15 GiB, 40 times). Training images are *not* held on the GPU
   (batch_size 2 ⇒ ~33 MB of GT per step). Jobs request **120 GB**, not the 64 GB
   of the standard ADAGS header.
9. **`--ip`/`--port` are parsed and never consumed** — no GUI, no socket. The
   README's `--port 6019` is inert and cannot clash on a shared node.

Two facts that are *deliberate upstream behaviour* and must not be "fixed":

* The background init colours are divided by 255 **twice**, with an in-source
  comment saying the released numbers depend on it.
* **LPIPS is computed on `[0,1]` inputs without `normalize=True`**, while the
  `lpips` default (`normalize=False`) assumes `[-1,1]`. The reported LPIPS is
  therefore *not* standard LPIPS. Flagged, not corrected — it may be exactly what
  the paper tabulates, and correcting it would break the reproduction. Any later
  cross-method LPIPS comparison must recompute on a common convention.

Also: **two SSIM conventions are computed and both reported** (`ssim1` =
GraphDECO 11×11 gaussian window σ=1.5; `ssim2` = skimage `win_size=7`,
`gaussian_weights=False`). Which one the paper tabulates is a repo-external
question and is **not yet resolved**; both are carried.

## 4. Environment decision

`envs/adags` already has the authors' exact torch pair (2.5.1+cu121 /
0.20.1+cu121), but carries **numpy 2.3.5** and **setuptools 82**, while Multi4D
needs `numpy<2` (mmcv 1.6.0 is built against the numpy 1.x ABI) and
`setuptools<66` (mmcv 1.6.0 has a legacy `setup.py` importing `pkg_resources`).
Downgrading either in `envs/adags` would mutate the environment every historical
ADAGS experiment was run against. **A separate `envs/multi4d` was built instead.**

The result matches the authors' verified stack exactly except for python 3.11.7
vs 3.11.6 and CUDA 12.2 vs 12.1 (torch is the cu121 build in both cases):
torch 2.5.1+cu121, torchvision 0.20.1+cu121, numpy 1.26.4, mmcv 1.6.0,
setuptools 65.5.0, opencv-python 4.11.0.86, scikit-image 0.26.0, lpips 0.1.4.

**The `libstdc++` trap recurred and is worth recording again.** Both compiled
extensions build fine and then fail to import with
``version `GLIBCXX_3.4.29' not found``, because the spack module chain puts
`gcc-runtime-8.5.0`'s `libstdc++` first on the loader path even after
`module load gcc/12.2.0`. Prepending to `LD_LIBRARY_PATH` is **not** enough;
`LD_PRELOAD` of the gcc-12.2.0 `libstdc++.so.6` is, which is what
`exp_index/leonardo_env.sh` has always done and what
`agent-control/substrates/multi4d_env.sh` now mirrors. With it, all of
`diff_gaussian_rasterization`, `simple_knn._C`, `mmcv` and `lpips` import
cleanly.

`multi4d_env.sh` deliberately does **not** `set -euo pipefail`. Sourcing
`leonardo_env.sh` leaks `set -e` into the caller, which once marked 74 complete
cells FAILED on a non-matching glob (spec §13.25); callers set their own shell
options after sourcing.

## 5. Logging: a sidecar, because the substrate has none

A repo-wide grep for `wandb|SummaryWriter|tensorboard` returns **zero hits**.
Adding logging would be a substrate modification, which Phase 1 forbids. So
`agent-control/substrates/wandb_replay_multi4d.py` reads only files the run
writes anyway — `debug_metrics.json` (JSONL), `best/per_frame_metrics.json`, the
rendered `best/{render,gt,static_persistent,transient}` frames, `provenance.txt`,
and the Gaussian counts that exist only in the tqdm postfix on stderr — and ships
them to W&B. It is idempotent and replayable (it records how many JSONL records
it has already sent), and runs on the **login node**, since compute nodes have no
network. The substrate checkout stays byte-identical to the pin.

## 6. Layout and provenance

```
$W = /leonardo_work/EUHPC_D36_068/sri/proj_adags
$W/repo/multi4d                      pinned clone, c483e82, porcelain-clean
$W/envs/multi4d                      the venv above
$W/data_staging/multi4d/<scene>/     symlinked mp4s + poses_bounds.npy, real fused.ply,
                                     and the extracted camNN/images/
$W/runs/multi4d/<run_id>/            model/, results/, provenance.txt
$W/agent-control/substrates/         multi4d_env.sh, prep_scene.py,
                                     wandb_replay_multi4d.py, sbatch/, jids/, ledger.txt
$W/repo/adags/logs/m4d_*_%j.{out,err}
```

Every job writes a `provenance.txt` carrying the substrate commit, the porcelain
count, sha256 of `train.py`, `arguments/dynerf.py` and `fused.ply`, the node, the
python/torch versions and the exit code. Every submitted job id goes into
`agent-control/substrates/ledger.txt` with its reason at submission time.

**Declared deviation from the bare README command:** `--saving_folder` is set so
results land under `runs/` instead of polluting the pristine checkout, and
`--checkpoint_iterations 20000` is passed so a final checkpoint exists for later
phases. Neither touches the optimisation; `--checkpoint_iterations` defaults to
`[]` upstream, so without it no `.pth` is ever written.

## 7. Results

*(see section 9 -- appended after the runs returned; nothing above this line is
retracted by what follows)*

## 8. Scheduling: Leonardo structurally disfavours single-GPU jobs (2026-09-19)

Recorded because it cost a cancel/resubmit cycle and will recur on every
substrate lane.

The three repetitions were submitted on `boost_qos_lprod`, the QOS the existing
`scripts/submit_exploratory_lane.sh` header uses. They sat `PENDING` with reason
`(Priority)` and never started. The request was not the problem: `gres/gpu:1`,
8 CPU, 120 G and 1 node are all satisfiable on a 32-CPU / 514 GB / 4-GPU boost
node, and `scontrol` confirmed `Dependency=(null)`.

**The queue was the problem, and the cause is the priority weighting:**

```
PriorityWeightJobSize   = 10000000      PriorityFavorSmall = no
PriorityWeightQOS       = 300000
PriorityWeightFairShare = 25000
PriorityWeightAge       = 20000
```

`JobSize` outweighs everything else by 30x and **favours large jobs**, so a
one-node GPU cell gets essentially nothing from the dominant term. Measured
state at 00:23 on 2026-09-19: zero `idle` nodes, 2,677 of 3,254 `alloc`,
**10,295 pending jobs in `boost_usr_prod`, with 9,424 ahead of ours**. Our jobs
were at rank ~9,425 with priority 133,220; the top of the queue was 274,300.

**QOS options actually available to a 1-GPU job, measured:**

| QOS | Priority | MaxWall | MinTRES | usable here |
|---|---|---|---|---|
| `normal` | 40 | partition 24 h | — | yes |
| `boost_qos_lprod` | 40 | 4 d | — | yes |
| `boost_qos_bprod` | 60 | 1 d | **node=65, gres/gpu=260** | **NO** |
| `boost_qos_dbg` | 80 | **00:30:00** | — | smoke only (max 2 jobs) |

**`boost_qos_bprod` is a big-production lane and refuses a small job outright**
with `QOSMinCpuNotSatisfied`. An attempt to move the reps there failed on all
three. And `normal` and `lprod` are *both* priority 40 — **there is no QOS
upgrade available to a single-GPU job on this cluster.**

The QOS factor itself is real and large, which the debug queue demonstrates: the
same sbatch submitted to `boost_qos_dbg` scored priority **253,262** against the
reps' **133,199** (+120,063), moving from rank ~9,448 to ~2,453, and it started
**within two minutes**. But dbg caps at 30 minutes, so it can only ever host a
smoke test.

**CARRY AS OPERATIONS:** on Leonardo a single-GPU cell has exactly two levers —
**walltime** (shorter jobs backfill into gaps; longer ones do not) and **age**.
Request the shortest walltime the run genuinely needs, which means *measuring*
the iteration rate on `boost_qos_dbg` before committing the real cells, rather
than padding a guess. A generous walltime is not free insurance here: it is
itself the thing that keeps the job from being scheduled. Note the opposing
constraint that makes this a real trade rather than a preference — Multi4D
resume is unusable (§3.6), so a walltime kill loses the entire run.

## 9. Results — PASS, 3 of 3 (2026-09-19)

Three repetitions, published schedule (coarse 2000 + fine 20000), code
unmodified, all `COMPLETED` with exit 0 and the full 40/40 evaluations.
Provenance is identical across all three: commit `c483e82`, `substrate_dirty=0`,
`train.py` sha `569816dc8cf9`, `arguments/dynerf.py` sha `c7b1093b4f60`,
`fused.ply` sha `df3572b2b6e7`.

| rep | jid | node | peak PSNR | @iter | peak>16k | final@20000 | SSIM gw | SSIM sk | LPIPS-a | LPIPS-v | FG / BG / TR | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 58156889 | lrdn1886 | **33.4226** | 18000 | 33.4226 | 32.7848 | 0.9367 | 0.9520 | 0.0429 | 0.1427 | 22,895 / 153,379 / 251,867 | 2:42:50 |
| 2 | 58156890 | lrdn2599 | **33.7672** | 19000 | 33.7672 | 33.5403 | 0.9367 | 0.9524 | 0.0408 | 0.1406 | 25,166 / 150,034 / 198,256 | 2:45:28 |
| 3 | 58156891 | lrdn3127 | **33.5383** | 13500 | 33.5152 | 32.8575 | 0.9366 | 0.9520 | 0.0426 | 0.1416 | 19,933 / 153,447 / 223,045 | 2:34:36 |

**Verdict under the frozen rule: PASS.** All three peaks lie in 33.4-34.0;
the rule required 2 of 3. Mean peak **33.576**, spread **0.3445 dB**.
Paper value 34.02; band centre 33.85.

### What the verdict does not say

* **Every repetition is below the paper and below the band centre.** The mean
  sits 0.44 dB under 34.02 and 0.27 dB under the band's own centre. The rule is
  met, and the reproduction is nonetheless consistently on the low side of it.
* **Rep 1 passes by 0.023 dB** (33.4226 against a floor of 33.4). One rep is
  effectively on the boundary; a slightly unluckier draw would have made the
  verdict 2 of 3 rather than 3 of 3. It would still have passed, but the margin
  is not comfortable and should not be reported as though it were.
* **`peak` and `best/` are different quantities for rep 3.** Its true peak,
  33.5383, occurred at iteration 13500, *below* the `iteration > 16000` gate the
  code applies to `best/`. Its `best/` artefacts therefore come from 16500 at
  33.5152. The gap is only 0.023 dB here, but the two quantities are not
  interchangeable in general, and anything reading `best/per_frame_metrics.json`
  as "the peak" is reading the wrong number. Carrying `peak_after_16000`
  separately in the collector is what made this visible.

### The result that matters most to the project

**The fixed-seed spread is 0.3445 dB.** These three repetitions share the
hardcoded `setup_seed(6666)`; they differ only by CUDA kernel nondeterminism.
So 0.3445 dB is a **lower bound** on run-to-run variance on this substrate, not
a seed spread.

That number is large, and it is the operative constraint on every comparison
built on Multi4D: it is the same order as this project's own historical N3V
noise floor (0.341 dB on the event union at the 50-frame protocol,
2026-08-23), and it exceeds the 0.20 dB Gate-B event-PSNR target inherited from
the approved objective. **Any future Multi4D-based claim must clear a
0.35 dB floor at minimum, established here before any intervention was
attempted** -- and must establish its own floor with seeds actually varied,
which requires a code change, since no `--seed` flag exists.

### Secondary observations

* **All three lose PSNR between their peak and iteration 20000** (-0.23, -0.64,
  -0.68 dB). The published schedule trains past its own optimum on this scene.
  This reproduces, on a different substrate, the 2026-08-20 finding that both
  ADAGS capacity regimes peak near 12k iterations and then lose PSNR.
* **The background branch is pruned to under half its initialisation**:
  386,905 points in `fused.ply` against 150-153k surviving static Gaussians,
  while the transient branch grows to 198-252k. Totals 373-428k.
* **Runtime was 2:34-2:45, well under the 8 h limit** and well under the
  4.5-5.6 h the smoke run extrapolated. The smoke over-estimated because it
  measured the early fine-stage rate, when eval cost per call is at its highest
  relative to a small model.

### Artefacts

Per run under `$W/runs/multi4d/20260918_232132_cut_roasted_beef_rep<N>/`:
`provenance.txt`, `results/<expname>/debug_metrics.json` (40 JSONL records),
`results/<expname>/best/per_frame_metrics.json` (300 frames),
`best/{render,gt,static_persistent,transient}/` (300 PNGs each, all four
subsets complete on all three runs), and point clouds (612-718 MB per run,
204-239 MB for `best/`). Collector output:
`$W/agent-control/substrates/multi4d_report.{json,md}`.

W&B, `models-ku-leuven/adags-multi4d`, group `multi4d-crb-repro`, one run per
repetition with all 40 evaluations, the 300-frame per-frame table, and five
sampled frames from each of the four rendered subsets:
[rep1](https://wandb.ai/models-ku-leuven/adags-multi4d/runs/a72b8062e3c409fe) ·
[rep2](https://wandb.ai/models-ku-leuven/adags-multi4d/runs/e5d59297d3672fc2) ·
[rep3](https://wandb.ai/models-ku-leuven/adags-multi4d/runs/fd7a8fbced7dd000).

### Still unresolved, and not resolvable from this run

* **Which SSIM convention the paper tabulates.** Both are carried above
  (gaussian-window 0.9366-0.9367, skimage 0.9520-0.9524). They differ by
  ~0.015, which is far larger than the spread between repetitions.
* **Whether the paper's LPIPS used the same non-normalized convention the code
  does.** The values above are as the repo computes them, on `[0,1]` inputs
  without `normalize=True`. They are not standard LPIPS and are not comparable
  to externally reported LPIPS without recomputation.

Both are repo-external questions. Neither affects the PSNR-based verdict.
