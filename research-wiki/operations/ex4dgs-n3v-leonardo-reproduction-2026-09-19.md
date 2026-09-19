# Ex4DGS on N3V `cut_roasted_beef` — Leonardo reproduction (2026-09-19)

Status: **E0 evaluation-stack gate PASSED essentially exactly. Trained
reproduction is PARTIAL — best 33.14 against a published 33.73, inside the
32.5–33.3 "consistent with community" band and below the 33.3 PASS threshold.**

Substrate pin, audited run facts, environment and frozen gates:
[`research-wiki/assets/ex4dgs-pin.json`](../assets/ex4dgs-pin.json).
Companion lane: [`multi4d-n3v-leonardo-reproduction-2026-09-18`](multi4d-n3v-leonardo-reproduction-2026-09-18.md).

## 1. Why this run exists

Ex4DGS (NeurIPS 2024, CC BY-NC 4.0) is the second candidate substrate. Its
per-primitive onset/offset opacity is the **K=1 case of multi-interval
presence**, and its static Gaussians carry no temporal opacity at all, which
makes it the sharpest available foil for the return claim: a single interval
cannot resume after an absence. As with Multi4D, nothing about that claim can be
tested until the substrate reproduces its own published number with code
unmodified.

## 2. The two gates, frozen before anything ran

Declared 2026-09-19T09:30Z:

* **E0 (evaluation stack, no training).** Render the authors' released 40k model
  against our staged scene and compare with their own shipped metrics.
  **PASS** = PSNR within ±0.05 of 33.73290 **and** LPIPS-alex within 0.002 of
  0.04044. *Added clause, ours not the plan's:* per-frame PSNR max absolute
  deviation ≤ 0.20 dB. The release ships `all_metrics.json`, so a matching mean
  with disagreeing frames would be two errors cancelling, not a pass.
* **Training.** **PASS** = E0 passes **and** best ≥ 33.3. **32.5–33.3** =
  partial, consistent with community reports of 31.9–33.3. **< 32.5** =
  investigate before use.

## 3. E0 — PASS, essentially exactly

| metric | authors | ours | delta |
|---|---|---|---|
| PSNR | 33.7329025 | 33.7313461 | **−0.0015564** |
| SSIM (gaussian window) | 0.95765483 | 0.95764679 | −8.05e−06 |
| LPIPS-alex | 0.04044107 | 0.04043768 | −3.39e−06 |
| LPIPS-VGG | 0.13221921 | 0.13221747 | −1.74e−06 |

Per-frame over all 300 held-out frames: **max absolute deviation 0.0130 dB**,
mean 0.0025 dB. The mean is not matching by cancellation — every frame agrees.

Two things made this trustworthy rather than merely green:

* **`render.py` writes `mean_metrics.json` and `all_metrics.json` into
  `--model_path`.** Pointed at the pristine unzip it would have overwritten the
  very reference being checked. E0 renders from a copy, and the authors' files
  are byte-identical before and after (sha256 recorded in the run dir).
* Our frames are named `000000.png` where theirs are `0.png`, so no per-frame
  keys are shared. The comparator falls back to **positional** matching and says
  so in its output, rather than silently comparing zero frames and reporting
  nothing.

**What E0 validates: the staged images, the native-resolution extraction, the
`resolution: 2` halving, the COLMAP camera intrinsics and extrinsics, and both
metric conventions. What it does NOT validate: our point cloud** — E0 renders
the authors' already-trained Gaussians, so the point cloud plays no part. That
only becomes live in training.

## 4. Trained reproduction — PARTIAL

Three repetitions, 40,000 iterations, code unmodified, all `COMPLETED` exit 0.

| rep | jid | PSNR | SSIM gw | SKSSIM | SKSSIM2 | LPIPS-a | LPIPS-v | wall |
|---|---|---|---|---|---|---|---|---|
| 1 | 58200326 | 32.5202 | 0.95699 | 0.95251 | 0.97667 | 0.04562 | 0.13663 | 1:12:02 |
| 2 | 58200328 | **33.1426** | 0.95721 | 0.95307 | 0.97692 | 0.04537 | 0.13461 | 1:10:54 |
| 3 | 58200329 | 32.6767 | 0.95697 | 0.95287 | 0.97665 | 0.04534 | 0.13585 | 1:09:40 |

Mean **32.7798**, best **33.1426**, published **33.7329**.
**Verdict: PARTIAL, consistent with community** (best is 0.59 dB under the
paper and 0.16 dB under the PASS threshold).

### The finding that constrains everything built on this substrate

**The fixed-seed spread is 0.6224 dB.** `utils/general_utils.py:158-160` hardcodes
`random.seed(0)`, `np.random.seed(0)`, `torch.manual_seed(0)` and there is no
`--seed` flag, so these three runs differ only by CUDA kernel nondeterminism.
0.6224 dB is therefore a **lower bound** on run-to-run variance.

That number is larger than the shortfall it is being used to interpret:

* spread between repetitions: **0.6224 dB**
* gap from best repetition to the paper: **0.5903 dB**

**A single Ex4DGS run cannot distinguish "reproduced" from "0.6 dB short".**
rep2 alone would read as a near-reproduction; rep1 alone (32.5202) sits within
0.02 dB of the *investigate* threshold. Any future comparison on this substrate
must clear ~0.62 dB, which is **3× the 0.20 dB Gate-B event-PSNR target** and
nearly double Multi4D's 0.3445 dB floor. This was established before any
intervention was attempted.

### The point-count diagnostic refutes the convenient explanation

The plan anticipated recording our `input.ply` count against the authors' as
"the maintainer's own explanation of the community shortfall". Measured:

* authors' `input.ply`: **6,141** points
* our `colmap_0/sparse/0/points3D.bin`: **8,206** points (ratio **1.336**)

**We have 34% more points, not fewer.** A triangulation deficit therefore cannot
explain our shortfall. Combined with E0 passing, the usual escape routes are
closed: it is not the data, not the resolution, not the metric convention, and
not a sparser initialisation. Whatever accounts for the remaining 0.59 dB is in
training itself, and is not diagnosed here.

## 5. Two defects in the substrate that would have silently corrupted the work

**(a) The planned A/B arm was vacuous.** `opacity_reset_interval` is declared at
`arguments/__init__.py:115` and set in every config JSON, but **no Python reads
it**, and `reset_opacity` (defined at `scene/gaussian_model.py:210` and
`scene/c_gaussian_model.py:549`) is **never called** — against
`densification_interval`, which has seven `train.py` call sites. The planned
comparison of the shipped 50000 against the released `cfg_args` 10000 would have
produced two bit-identical arms reported as a hyperparameter ablation. Dropped,
and replaced with 3 repetitions of the one real configuration. It also explains
the 50000-vs-10000 discrepancy itself as cosmetic drift in a dead field.

This is the fourth time this project has met a treatment that could not have
produced a difference (cf. the 2026-08-24 block's three vacuity catches). It was
caught by reading the code before spending GPU on it, not by a precondition.

**(b) `scripts/pre_n3d_colmap.py` extracts nothing as shipped.** Its guard is

```python
if all(os.path.join(output_dir, "{:06d}.png".format(i)) for i in range(startframe, endframe)):
    print(f"Already extracted all the frames in {output_dir}")
    return
```

`os.path.join` returns a non-empty string, which is always truthy, so `all(...)`
over 300 of them is unconditionally `True` and `extractframes` returns having
written nothing. The author meant `os.path.exists`. Run unmodified against a
clean directory it prints "Already extracted all the frames" twenty times and
COLMAP then finds no images. **The substrate was not patched**: an external
script mirroring the upstream body line-for-line writes the frames, after which
the broken guard correctly skips and steps 2–4 build `colmap_0` from real data.

## 6. Operational facts worth carrying

* **The scene must be a writable copy.** `scene/dataset_readers.py` writes
  `points3D.ply` into `--source_path` on first open — at **render** time as well
  as train time. Pointing at the read-only N3V tree would fail.
* **`--configpath`, not the README's `--config`** (the latter works only via
  argparse prefix matching).
* **There is no in-training evaluation.** `test_iterations` defaults to
  `[7000000]` and the two lines that would append `args.iterations` are
  commented out, so each run yields exactly **one** held-out measurement, from
  `render.py`. There is no curve to inspect and no early-peak to detect —
  unlike Multi4D, where all three runs lost 0.23–0.68 dB after their peak.
* **Resume is unusable**, confirmed in code: `--start_checkpoint` restores the
  model but `train.py` then re-runs the progressive-window setup, resetting
  sampling to `start_duration=10` frames while `gaussians.duration` stays 300,
  so a resumed run never gets its window back. Each repetition is one
  indivisible 40k job.
* **`render.py` runs a 20×500 = 10,000-render timing loop** after the metrics.
  Slow (minutes), not hung; nothing reported depends on it.
* **Host RAM is ~2–4 GiB**, not Multi4D's ~22 GiB: `lazy_loader` streams images
  through two joblib workers rather than materialising them. Jobs requested 64 G.
* **Measured rate 11.9 it/s** → 40k in ~56 min, ~1h10 with render. Measured on
  `boost_qos_dbg` before committing the real cells, which is the practice the
  Multi4D lane arrived at.

## 7. Three environment defects, each caught before a run

Recorded because two of them were self-inflicted and the pattern generalises.

1. **A sequence of pins is not a pin.** `numpy<2` was installed in one step and
   silently undone by the next: `opencv-python 5.0.0.93` and `kornia 0.8.3` both
   pull `numpy>=2`, and the CUDA extension build then refused. Replaced with a
   pip **constraints file** applied to every install, plus a hard precondition
   that asserts numpy 1.x, scikit-image < 0.23 and torch 2.1.2 **before** any
   extension is built.
2. **`pip install --upgrade setuptools`** broke the build with
   `No module named 'pkg_resources'` — setuptools ≥ 81 no longer ships it and
   these legacy `setup.py` files import it. Pinned `<66`, the same pin Multi4D
   needed for the same reason.
3. **kornia 0.8.3 is unscriptable under torch 2.1.2** — it calls
   `.any(dim=(-2,-1))` with a tuple dim. Pinned `<0.8` (0.7.4). The repo uses
   kornia for exactly one symbol, `create_meshgrid`.

Also: `scikit-image` is pinned to 0.22.0 deliberately, because `render.py` passes
`multichannel=` to `structural_similarity`, which 0.22.x absorbs into `**kwargs`
and ignores but 0.23 removed.

The four LPIPS weight files (`alexnet-owt-7be5be79.pth`, `vgg16-397923af.pth`,
and the two richzhang linear-weight files `alex.pth`/`vgg.pth`) were already
staged under `$TORCH_HOME/hub/checkpoints`; the two small ones were
**hash-verified against fresh downloads** rather than assumed.

## 8. Comparison with the Multi4D lane

| | Multi4D | Ex4DGS |
|---|---|---|
| published | 34.02 | 33.73 |
| our best | 33.7672 | 33.1426 |
| verdict | **PASS** (3/3 in band) | **PARTIAL** |
| fixed-seed spread | 0.3445 dB | **0.6224 dB** |
| eval stack validated against authors? | no released model to check | **yes, to 0.0016 dB** |
| LPIPS convention | non-standard (no `normalize`) | standard |
| in-training eval | 40 evaluations | none |

Ex4DGS is the weaker reproduction but the better-instrumented one: because the
authors released a model *and* their own metrics, its evaluation stack is
validated to 0.0016 dB, which no amount of training could establish. The
Multi4D lane has no equivalent check available.

## 9. Artefacts

Per run under `$W/runs/ex4dgs/20260919_121607_cut_roasted_beef_rep<N>/`:
`provenance.txt` (commit, porcelain count, sha256 of `train.py`, `render.py`,
the config and `points3D.bin`), `model/mean_metrics.json`,
`model/all_metrics.json` (300 frames), point clouds and `chkpnt{30000,40000}.pth`.
E0 under `$W/runs/ex4dgs/e0_20260919_113909_cut_roasted_beef/` with
`e0_report.json` and `authors_reference.sha256`. Collector output:
`$W/agent-control/substrates/ex4dgs_report.json`.

## 10. Not established here

* **What accounts for the remaining 0.59 dB.** E0 passing and the point surplus
  rule out data, resolution, metrics and a sparse initialisation. The cause is
  in training and is undiagnosed.
* **A real seed spread.** Establishing one requires a code change, since
  `safe_state` hardcodes seed 0 and no flag exists.
* **Technicolor.** `D:\adags\data\Train` and `D:\adags\data\Birthday` turned out
  to be unzipped Ex4DGS **pretrained models**, not raw footage — their
  `cfg_args` point at the authors' own `/data/datasets/nerf/Technicolor/`. They
  do supply the main-table windows directly (Train `start_timestamp=151,
  end_timestamp=201, duration=50, resolution=1`, PSNR 31.774; Birthday 32.381),
  and note that the main-table Train window is `colmap_151`, not the
  `colmap_170` named for Appendix C. **The InterDigital footage itself is on
  neither this workstation nor Leonardo, so Technicolor is blocked on data
  acquisition.**

## 11. The point-cloud swap (2026-09-19, appended): the hypothesis is refuted, and the point estimate runs the other way

§4 left the 0.59 dB shortfall undiagnosed, with the first-frame triangulation
the one variable E0 could not exclude. It has now been tested directly.

**Design.** One variable. A second staging tree with **symlinked camera
directories and hash-identical `cameras.bin`/`images.bin`**, differing only in
`colmap_0/sparse/0/points3D.ply`, which was replaced by the authors' own
6,141-point `input.ply` from the release (sha256 verified identical to the
release copy). The loader prefers an existing `points3D.ply` over the `.bin`
(`scene/dataset_readers.py:562-568`), so this is the whole intervention.

Frame compatibility was checked before relying on it: the two clouds share a
coordinate frame (centroids 1.45 apart, near-identical minimum bounds), and E0
had already proved our cameras sit in the authors' frame by rendering their
trained Gaussians through them. Our cloud differs by a **far-field tail** —
max z 284.7 against 129.4, extents ~2× in every axis.

**Baseline was simultaneously extended from n=3 to n=6** so that σ was estimated
well enough for the contrast to mean anything.

| arm | n | mean | sd | min | max | range |
|---|---|---|---|---|---|---|
| ours, 8,206 pts | 6 | **32.6731** | 0.3564 | 32.2298 | 33.1426 | 0.9128 |
| authors, 6,141 pts | 3 | **32.1337** | 0.3699 | 31.7655 | 32.5053 | 0.7398 |

**Difference (ours − authors) = +0.5394 dB — our cloud is the BETTER one.**
Welch t = 2.087, p = 0.106; pooled t = 2.117, p = 0.072, df 7;
95% CI on the difference **[−0.063, +1.142]**.

### Reading

* **The point-cloud hypothesis is not supported, and the point estimate has the
  opposite sign to the one it predicts.** Swapping in the authors' own
  initialisation did not recover the shortfall; it produced runs 0.54 dB
  *lower*. The confidence interval includes zero, so the claim "their cloud is
  worse" is **not** established at n=6 vs 3 — but the claim the experiment was
  run to test, that our triangulation explains the gap, has no support at all.
* Combined with E0 passing and with our having *more* points rather than fewer,
  every cheap explanation is now closed: not the data, not the resolution, not
  the metric convention, not a sparse initialisation, and not the specific point
  cloud. **The remaining 0.59 dB is in training and stays undiagnosed.** No
  further targeted experiment is queued, because none of comparable cost remains.
* **Do not read this as "the authors' cloud is harmful."** At p = 0.07 with a CI
  spanning zero, the responsible statement is that the swap changed nothing we
  can resolve, in a direction opposite to the hypothesis.

### The noise floor is confirmed and slightly worse

σ at n=6 is **0.3564 dB**, against 0.3238 at n=3 — stable, so the estimate can
now be used. The observed **range across six identical runs is 0.9128 dB**,
larger than the 0.6224 dB seen at n=3, which is what a growing sample does to a
range and is worth stating so the earlier number is not read as a ceiling.

Sample sizes for an unpaired two-arm comparison at 80% power, α = 0.05, using
the pooled sd 0.3603 (~1.2 GPU-h per run, so two arms cost):

| detectable effect | n per arm | two-arm cost |
|---|---|---|
| 1.0 dB | 3 | ~7 GPU-h |
| 0.5 dB | 9 | ~22 GPU-h |
| 0.3 dB | 23 | ~55 GPU-h |

All affordable against the remaining allocation. The binding constraint on any
Ex4DGS-based claim is design and replication, not budget.

### Verdict unchanged

Best over all nine runs is **33.1426** (baseline rep2), still below the 33.3
PASS threshold. The reproduction remains **PARTIAL, consistent with community**.

## 12. Provenance blemish on the six later runs (appended 2026-09-20)

The six runs submitted from 20:04 on 2026-09-19 — baseline reps 4–6 and the
three `authorsply` reps — record **`substrate_dirty=1`** in their
`provenance.txt`, where reps 1–3 record `0`. Recorded rather than quietly
cleaned, because an unexplained dirty flag on an evidence-bearing run is exactly
the kind of thing that should not be discovered later.

**Cause, identified.** A single **0-byte untracked file named `=460`** in
`$W/repo/ex4dgs/`, created 2026-09-19 20:04 — a stray shell redirect from the
environment work earlier that evening (a `>=` in an unquoted pip argument). It
is the only entry in `git status --porcelain`, which reads exactly `?? =460`.

**Why the runs are unaffected.** `substrate_dirty` is computed as
`git status --porcelain | wc -l`, which counts untracked files. The stronger
check, `git diff --stat HEAD`, is **empty** — no tracked file differs from
commit `7fac649`. The `train.py` and `render.py` sha256 values recorded in all
nine provenance files are identical across every run, including across the
dirty/clean boundary, which independently confirms the executed source was the
same. The flag reflects a junk file in the working directory, not a modified
substrate.

**Disposition.** The file was removed after this note was written. The
`substrate_dirty=1` values stand in the affected `provenance.txt` files and are
not edited — the runs recorded what was true at submission.

**Carry:** `git status --porcelain | wc -l` is the wrong dirtiness probe for a
provenance record, because it conflates "someone edited the substrate" with
"a stray file exists in the directory". A future provenance line should record
both, e.g. `substrate_tracked_diff=$(git diff --stat HEAD | wc -l)` alongside
the porcelain count, so the two failure modes are distinguishable without a
forensic pass.
