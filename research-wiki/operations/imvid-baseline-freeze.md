# FROZEN — ImViD Opera exploratory baseline: split, evaluation, budget

Date: 2026-08-17. Status: **FROZEN, NOT LAUNCHED.** Recorded before any
ImViD training. Three of the remaining gates close here (split,
evaluation conventions, compute/storage estimate); the visual
distortion check and the focused review remain open, and the baseline
may not launch until they close.

EXPLORATORY, `evidence_bearing: false`. Hopper/H100 is reserved for this
lane and has not been used.

## What is already closed

| gate | status |
|---|---|
| acquisition + integrity | CLOSED — sha256 verified, 41 files, read-only |
| decoding | CLOSED — 117 frames at 5312x2988, PNG IHDR-checked fail-closed |
| calibration, numerically | CLOSED — fixed-pose triangulation at 1.17-1.21 px over three frames, 39/39 cameras each |
| sparse initialization | CLOSED — 23,999-point union, hashed and manifested |
| initialization vs fallback | CLOSED — COLMAP cloud 12.8x more concentrated than a uniform fill of its own p01-p99 box |
| **split** | **CLOSED HERE** |
| **evaluation conventions** | **CLOSED HERE** |
| **compute / storage estimate** | **CLOSED HERE** |
| visual distortion check | OPEN |
| focused review | OPEN |

## Split — declared, and it is NOT ImViD's own

ImViD's paper holds out **one** camera (Camera 0 for ImViD/N3V/MeetRoom).
A single held-out view is too thin to carry a photometric claim, and this
project has already been burned once by ranking on a thin split
([[operations/diva360-scissor-sweep-matrix-v1]]: a +0.909 dB development
win that reversed on the official split).

**Frozen split for this lane:**

* **held-out (test): 4 cameras**, chosen by the same outcome-blind rule
  used throughout this project — evenly spaced through the sorted camera
  id list, reading ids only, never a metric. With 39 cameras
  (`cam00`-`cam38`) that is `np.linspace(0, 38, 4)` rounded =
  **`cam00, cam13, cam25, cam38`**.
* **training: the remaining 35 cameras.**
* `cam00` is deliberately INCLUDED in the held-out set so that a
  single-camera number comparable to ImViD's own protocol can be reported
  as a SUBSET of ours, without ever having trained on it.

**Prohibited held-out information**, explicitly: imagery, COLMAP
observations, triangulated points, or any metric from `cam00`, `cam13`,
`cam25`, `cam38` may not influence training, initialization, weights,
stopping or selection.

**This invalidates the current initialization and it must be rebuilt.**
The frozen sparse clouds were triangulated using ALL 39 cameras
([[operations/imvid-sample-ingestion]]), so they carry observations from
the four held-out views. That is initialization-time leakage. Before the
baseline runs, `scripts/imvid_sparse_init.py` must be re-run on the
35 training cameras only, and the union rebuilt. The 39-camera artifacts
are PRESERVED as the calibration-validation evidence they were built for
— they are sound for that purpose and unsound as an initializer.

Recorded rather than quietly fixed, because the 39-camera clouds already
exist and it would have been easy to reuse them.

## Evaluation conventions — frozen

* Metrics: PSNR, SSIM, LPIPS, reported under **both** convention sets and
  never averaged across them, per
  [[operations/diva360-protocol-parity-audit]]. Convention deltas are
  MEASURED per model, never applied as constants — that page shows the
  SSIM gap varies with quality regime and the LPIPS backbone delta
  changes sign.
* Resolution: the training resolution, declared per run. ImViD's own
  baseline evaluates at 2x downsample; whatever this lane uses is stated
  in its config header and not changed afterwards.
* Background: ImViD ships no masks, so there is NO alpha compositing and
  NO black-background convention here. Full-frame metrics only. This is
  a real difference from the DiVa-360 lane and any cross-dataset
  comparison must respect it.
* Temporal: all 300 frames of the sample, or a declared contiguous
  subset stated in the config.
* The held-out four are scored **once**, at the end, on the final model.

## Compute and storage — estimated from measurements, not guesses

Storage, measured:

```
archive (read-only, preserved)                     0.93 GiB
extracted mp4s                                     0.93 GiB
decoded frames, 3 frozen frames x 39 cameras       0.856 GiB  (7.5 MiB/frame measured)
sparse reconstructions + manifests                 < 0.1 GiB
```

Full-decode projection, from the MEASURED 7.5 MiB per frame:

```
300 frames x 39 cameras x 7.5 MiB = 85.7 GiB
```

That supersedes the preflight's ~557 GB raw-RGB arithmetic, which
described uncompressed RGB rather than PNG. Apollo had **31.165 TiB**
free, so storage is not a constraint. A decode of all 300 frames is
therefore affordable but is NOT authorized here; the baseline's frame
coverage is declared in its own config.

Compute: NOT estimated from ImViD's reported numbers, deliberately. Their
baseline trains 30 epochs over full permutations of the training frames
on an A100 with unreleased code, which is not this trainer. The honest
estimate comes from this project's own measurements: the DiVa-360
benchmark baseline ran 6000 iterations at 1160x550 over 4,935 training
units in ~1.9 h on a V100. ImViD at 5312x2988 is **~23x the pixels per
image**. Any first ImViD run therefore starts from a declared downscale
and a bounded iteration count, and its cost is MEASURED on a short
preflight before a full run is submitted — the same measure-then-commit
discipline used for the oracle and the acceptance path.

## Hardware

DGX/V100 is NOT used for this lane. Hopper/H100 is reserved for it, and
the whole ImViD comparison stays on one hardware class. H100 runtime is
never compared against V100 runtime as though hardware were controlled.
To record when a run happens: exact H100 model, resource pool, container
image digest, CUDA and dependency versions, numerical settings, seed,
runtime, peak memory, experiment and task ids.

## What this baseline cannot establish

Full-dataset performance; 1-5 minute scalability (the sample is 300
frames = 5 seconds); moving-rig validity (Opera is fixed-point only);
disappearance/reactivation supply. And ImViD's reported flow/depth gains
are NOT an expected gain for this lane, for scissor, or for N3V — they
were measured on a different implementation, a different representation,
and a deliberately peripheral held-out camera.
