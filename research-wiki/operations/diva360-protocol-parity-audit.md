# DiVa-360 protocol parity audit — the 20.698 dB number is not comparable

Date: 2026-08-16. Status: **AUDIT COMPLETE, PARITY NOT ESTABLISHED.**
Authority: user directive 2026-08-16 ("audit the current evaluation
against the official DiVa-360 protocol ... do not compare it directly
with published results until protocol parity is established").

Subject: experiment 79's `best_val/psnr 20.69827909741113` /
`best_val/ssim 0.8671356191944821` on `scissor`, and experiment 71's
19.652 / 0.855 before it. Both EXPLORATORY.

**Verdict: three mismatches, two of them decisive. Neither number may
be placed beside a published DiVa-360 figure.**

## The audit

| protocol element | official | current | verdict |
|---|---|---|---|
| temporal sampling | 30 FPS | **120 FPS**, stride 1, 562 frames | **MISMATCH** |
| training cameras | 35 | 35 | match |
| held-out cameras | 6 | 6 (ids 0, 16, 17, 33, 43, 44) | match |
| undistorted | yes | `scene_top_dir: undist` | match |
| cropped | 1160x550 calibration space | source frames are 1160x550 | match |
| background | black | black, GT composited | match |
| evaluation resolution | 1160x550 | **290x138** | **MISMATCH** |
| PSNR convention | — | `20*log10(1/sqrt(MSE))`, [0,1], per-image then mean | 3DGS-standard |
| SSIM convention | — | 3DGS 11x11 Gaussian, per-channel, global mean | 3DGS-standard, NOT verified against DiVa-360's own script |
| LPIPS | reported | **never computed** | **MISSING** |

### M1 — temporal sampling is 4x too dense

`diva360_conversion_provenance.json` for
`scissor_screen_w0_561` records `"fps": 120.0`, `"window": {"start": 0,
"end": 561, "stride": 1}`. `elgs.diva360_schema.DEFAULT_FPS` is `120.0`
and `frame_index_to_time` stamps `time = frame_index / fps`, so the
scene carries all 562 captured frames and `time_duration: [0, 4.7]`.
The official protocol samples at 30 FPS — every fourth frame, ~141
frames over the same interval.

This is not a rescaling of the same task. It changes the number of
training units by 4x, the temporal density the motion model must fit,
and the number of evaluation units. It is the reason the run reads
19,670 training units (35 x 562) rather than ~4,900.

### M2 — evaluation happens at 1/16 of the pixels

`configs/elgs/diva360_scissor_photometric.yaml` and its `_c10k`
continuation set `ModelParams.resolution: 4`.
`utils/camera_utils.py::loadCam` then computes
`round(1160/4) x round(550/4) = 290 x 138` and divides every intrinsic
by the same factor. Every PSNR and SSIM in experiments 71 and 79 was
computed at **290x138**, not 1160x550 — 1/4 in each linear dimension,
**1/16 the pixels**.

Corroborated independently: the evidence path derived
`evidence_pixel_scale = 1160 / 290 = 4.0` at runtime, which is the same
fact arriving from the other direction.

**Direction of the bias is not neutral.** Averaging 4x4 blocks removes
exactly the high-frequency residual an under-fit model gets wrong, so
downsampled PSNR is normally HIGHER than full-resolution PSNR on the
same model. The full-resolution number is therefore expected to be
BELOW 20.698, not above it.

### M3 — LPIPS is never computed

`main.py:1061` maps a `"lpips" -> "test/lpips"` summary key, but nothing
writes it: the evaluation block builds `eval_metrics` from `psnrs`,
`ssims`, `dyn_psnrs` and `static_region_psnrs` only. Experiment 79's
`summary.json` contains no `lpips` key, confirming it at runtime. A
protocol that reports LPIPS cannot be matched without adding it.

### What DID match

The camera split is the shipped one and is correct: the provenance
records 35 training cameras and 6 test cameras (`0, 16, 17, 33, 43, 44`),
each with 562 frames — 19,670 and 3,372 units. Evaluation iterates the
FULL test set, so there is no held-out subsampling on top of the FPS
issue.

Background handling is correct and worth stating precisely because it is
easy to get wrong: `_white_background` defaults `False`, so
`bg = [0,0,0]`, and `scene/dataset_readers.py` composites the ground
truth as `rgb * alpha + bg * (1 - alpha)` before it is ever compared.
The GT is genuinely black-composited, not an RGBA image with its alpha
silently dropped.

## Two things this audit cannot settle from the repository

1. **The official protocol is not recorded anywhere in this repository.**
   There is no DiVa-360 paper page under `research-wiki/papers/`. The
   elements audited above are the ones the directive stated. The exact
   LPIPS backbone (AlexNet vs VGG), the exact SSIM implementation, and
   any per-sequence evaluation frame range are NOT verifiable here and
   are not asserted.
2. **`transforms_val.json` is byte-identical to `transforms_test.json`**
   (both `sha256 f6c372c2a400331caf75be75ffb446d412e71e56a6b6192987fd97d994040f9a`).
   There is NO development split. Any model selection that ranks
   configurations must construct one from the 35 training cameras, or it
   selects on the held-out set.

## Consequence for the 20.698 dB result

It is a valid EXPLORATORY measurement of its own lane and nothing else.
It was produced at 290x138 on a 120-FPS scene by a model holding 10,806
points with densification stopped at iteration 2000, and it carries no
LPIPS. Placing it beside a published scissor figure would compare four
different things at once.

Recorded separately: [[gap_map]] states "DiVa-360 is the event-dense
benchmark with no GS baselines". If a published scissor PSNR/SSIM/LPIPS
triple is to be the parity target, the source of that triple needs its
own paper page, because the durable record currently says no GS baseline
exists to compare against.

## What parity requires

1. A NEW immutable 30-FPS materialization (stride 4). The existing
   120-FPS conversion and every artifact derived from it — the sealed
   cotracker3 tracks, the census, experiments 71-83 — stay exactly as
   they are.
2. `resolution: 1`, so evaluation happens at 1160x550.
3. An LPIPS implementation in the evaluation path, with its backbone
   declared.
4. A development split drawn from the 35 training cameras, leaving the
   6 official held-out cameras sealed.

## Addendum — M4: the eager path applies alpha TWICE

Found while pre-flighting `dataloader: True` for the benchmark-aligned
baseline. This is a fourth mismatch and it is not in the table above
because it was not visible until the two image paths were compared
directly.

**Eager path** (`dataloader: False` — what experiments 71 and 79 used):

1. `scene/dataset_readers.py:409-414` composites
   `arr = rgb*alpha + bg*(1-alpha)`, which with `bg = 0` is `rgb*alpha`,
   then — because `alpha.min() < 1` — CONCATENATES alpha back on, so the
   `CameraInfo` image is RGBA `[rgb*alpha, alpha]`.
2. `utils/camera_utils.py:62-66` splits that into
   `gt_image = resized[:3]` (already `rgb*alpha`) and
   `loaded_mask = resized[3:4]` (alpha).
3. `scene/cameras.py:55-56`, with `meta_only=False`, then applies
   `self.image *= gt_alpha_mask`.

Net ground truth: **`rgb * alpha^2`**.

**Lazy path** (`dataloader: True`): `utils/data_utils.py:24-25`
composites the same `rgb*alpha` but converts to `"RGB"`, dropping alpha.
Its `if resized_image_rgb.shape[1] == 4` guard tests the HEIGHT axis
rather than the channel axis, so for any image taller than 4 pixels it
takes the `else` branch and multiplies by ones. `Camera` skips the mask
entirely under `meta_only=True`.

Net ground truth: **`rgb * alpha`** — the standard black composite.

### What this means

* Experiments 71 and 79 were scored against a **doubly-matted** ground
  truth. DiVa-360's alpha is "continuous ... 99.9% concentrated at the
  extremes", so `alpha^2 == alpha` on ~99.9% of pixels and the effect is
  confined to matte boundaries — but it is a real deviation from the
  official black-composite convention, and boundary pixels are exactly
  where a dynamic-object benchmark is decided.
* The benchmark-aligned baseline runs `dataloader: True` (required at
  1160x550 for memory) and therefore uses `rgb * alpha`. It is the
  CORRECT convention, and it is **not GT-identical to experiments 71 and
  79**. Those runs were already non-comparable for the three reasons
  above; this is a fourth, and it means the new baseline is not a
  like-for-like continuation of them either.
* The `shape[1] == 4` channel/height confusion in `utils/data_utils.py`
  is a latent defect. It is INERT today — the image is RGB by
  construction two lines earlier, so the branch can never be taken and
  the `else` multiply is a no-op — so it is RECORDED, NOT PATCHED, per
  the standing rule on unrelated code.

## Addendum — M3 CLOSED for the evaluation path (2026-08-16)

LPIPS has been computed. Experiment 99, `scripts/eval_diva360_heldout.py`,
commit `cf3b34d`, on the benchmark baseline's iteration-5000 checkpoint,
over the six official held-out cameras at 1160x550 on black — 846 units:

```
PSNR   21.3567
SSIM   0.90701
LPIPS  0.14685   (AlexNet v0.1, inputs in [-1,1] — reference convention)
LPIPS  0.12398   (AlexNet v0.1, inputs in [0,1]  — the convention 3DGS
                  metrics.py ships, which many published GS numbers inherit)
```

The container DOES have network egress: both
`alexnet-owt-7be5be79.pth` and richzhang's `v0.1/alex.pth` downloaded
successfully once the hub cache was pointed at a directory the run owns.
The earlier `PermissionError` on `/tmp/adags_cache/torch` (experiment 85)
was solely the shared `XDG_CACHE_HOME` being owned by another container,
not an egress restriction.

**M3 is closed for the evaluation path only.** `main.py`'s training loop
still computes PSNR and SSIM and no LPIPS; the metric is available
post-hoc from a checkpoint, which is what a protocol comparison needs,
but an in-training LPIPS curve remains unavailable.

Both conventions are reported because this repository still cannot
establish which one DiVa-360's own script uses — there is still no
DiVa-360 paper page. The distinction matters here: 0.147 versus 0.124 is
larger than the gap between them and the 0.08-0.10 target, so which
convention is meant changes how far short the model is, but not WHETHER
it is short.

## Addendum — the official conventions are now READ, not inferred (2026-08-16)

The statement above that "this repository still cannot establish which
one DiVa-360's own script uses" is CLOSED. The evaluator was located and
read: **`utils/benchmark.py` in `brown-ivl/DiVa360`, branch `main`** —
the only branch. The paper is arXiv **2307.16897**, "DiVa-360: The
Dynamic Visual Dataset for Immersive Neural Fields", CVPR 2024
(Highlight).

```python
psnr  = cv.PSNR(gt, pred)                                # L52, uint8
ssim  = structural_similarity(gt, pred, channel_axis=2)  # L53, uint8
lpips_net = LearnedPerceptualImagePatchSimilarity(net_type='vgg')  # L30
image_pred = image_pred * 2 - 1;  image_gt = image_gt * 2 - 1      # L18-19
avg_* / count                                            # L65-67
```

| element | official | this repository (exp 99/100) | verdict |
|---|---|---|---|
| PSNR | `cv.PSNR` on **uint8**, R=255, per-image mean | float domain | MISMATCH (minor) |
| SSIM | skimage **defaults**: uniform **7x7**, `data_range` 255, `K1` 0.01, `K2` 0.03, sample covariance, `channel_axis=2` | 3DGS **11x11 Gaussian** | **MISMATCH (material)** |
| LPIPS backbone | **VGG** (torchmetrics) | AlexNet | **MISMATCH** |
| LPIPS input range | **[-1,1]** (`normalize=False` default) | [-1,1] | **MATCH** |
| background | black (`--wh_bg` omitted in the scissor scripts) | black | match |
| test cameras | `[0, 16, 17, 33, 43, 44]` (`utils/splitJson.py` L25) | same | match |
| train cameras | 35 = 53 − 12 discarded − 6 test | 35 | match |
| resolution | 1160x550 (paper §5.1/§5.2, "original resolution") | 1160x550 | match |
| frame rate | **30 FPS**, downsampled from 120 (`utils/moveVideo.py` L34, 33333 µs) | 30 FPS | match |

**The SSIM finding changes how the earlier gap should be read.** skimage's
own docstring states that the Wang-et-al (3DGS-style) form requires
`gaussian_weights=True, use_sample_covariance=False` — which DiVa-360
does **not** set. So 0.90701 against a published 0.937-0.944 was partly a
CONVENTION gap, not purely a quality gap. Likewise the `[0,1]` LPIPS
variant (0.12398) corresponds to nothing official and should be
discarded; the `[-1,1]` convention was right and only the backbone was
wrong.

**Published scissor rows** — CVPR 2024 supplementary **Table 5**
(= arXiv v2 Table 12), the only three methods the paper reports:

| scissor | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| PF I-NGP (black bg) | 25.346 | 0.944 | 0.076 |
| MixVoxels (black bg) | 25.090 | 0.937 | 0.086 |
| K-Planes (**white** bg) | 25.883 | 0.936 | 0.168 |

The K-Planes row is **white background** and must not be used as a target
for a black-background model. **There is still no Gaussian-splatting
baseline**: the official README's TODO `add Gaussian Splatting to the
benchmark` is unchecked, which corroborates [[gap_map]]'s "DiVa-360 is
the event-dense benchmark with no GS baselines".

### M1 is now QUANTIFIED, and it is larger than recorded

The audit above recorded M1 as "temporal sampling is 4x too dense" and
the parity requirement as a 30-FPS materialization. That materialization
exists (`scissor_screen_w0_561_s4_30fps`) and experiments 84-100 use it.
**It does not close M1.** DiVa-360's scissor configuration covers
**1125 frames** at 30 FPS — 37.5 s — verified three independent ways:
`objects_scripts/scissor/eval_mixvoxels.sh` passes `--num_frames 1125`;
`undistortion.sh` produces 625 + 500 = 1125 undistorted frames; and the
MixVoxels chunk configs sum to 7x150 + 75 = 1125. The `FRAMES=1200` in
`move.sh` / `test_ingp.sh` is **stale** — only 1125 frames have
undistorted ground truth on disk.

The scene evaluated here is **141 frames**, the first **4.7 s**. So the
published rows are a pooled mean over 1125 frames x 6 cameras and every
ADAGS scissor number is over 141 x 6 = 846 units covering an eighth of
the sequence. **Metric parity does not repair this**, and no scissor
number produced on this materialization may be placed beside a published
row as like-for-like, whichever convention is used.

Version pinning is UNAVAILABLE: DiVa-360's `environment.yml` pins only
`python=3.9`, `colmap=3.8` and `awscli`; scikit-image, torchmetrics,
opencv and torch are all unpinned.

### Implementation, and one environment constraint

`utils/diva360_official_metrics.py` + `scripts/eval_diva360_heldout.py
--official-metrics` (commit `b94a83e`) compute all three official
metrics alongside the existing ones, on the SAME images.

**scikit-image is NOT installed in the Apollo image.** Probed directly
at zero GPU slots: `skimage MISSING ModuleNotFoundError`, while
`cv2 4.10.0`, `torchmetrics 0.11.4`, `torch 2.0.1+cu118`,
`torchvision 0.15.2+cu118` and `numpy 1.24.4` are present. Rather than
let a score depend on whether an optional package happens to be
installed, skimage's defaults are reproduced exactly and pinned against
the real library by `tests/test_diva360_official_metrics.py` — matching
to 10 decimal places on random pairs, identical images, flat and
structured content, and at the official 1160x550 aspect. Those tests
SKIP where skimage is absent and RUN where it is (the workstation base
environment, scikit-image 0.24.0); one test reports the skip explicitly
so an all-green Apollo suite is never misread as having verified the
reimplementation there.

No scipy is needed: skimage crops the SSIM map by `(win_size-1)//2`
before averaging, which for a 7-tap filter is exactly the region whose
window lies wholly inside the image, so boundary padding never reaches a
surviving pixel and a valid-region box mean is identical.

The render is quantized as `torchvision.utils.save_image` would write it
(`x*255 + 0.5`, truncated, clamped) because the official evaluator reads
PNG FILES for both images; all three official metrics derive from the
quantized data, LPIPS included.

**Compositing, stated precisely.** The official evaluator applies
`rgb*alpha + (1-alpha)*bg` to gt AND pred using each image's OWN alpha.
Here the ground truth is already black-composited once by the lazy
dataset path and the render is produced against a black background by
the rasterizer, so it is composited by construction. The GT's alpha is
NEVER applied to the prediction — that would leak ground-truth geometry
into the score.

## Addendum — the rescore RAN, and it corrects the section above (2026-08-16)

Experiment **103**, commit `a824ae3`, dgx/V100, on experiment 84's
iteration-5000 checkpoint (338,528 points), the six official held-out
cameras, 846 units over 141 frames at 1160x550 on black. Both metric
sets computed on the SAME images in the same pass.

| metric | ADAGS internal convention | DiVa-360 OFFICIAL convention | delta |
|---|---:|---:|---:|
| PSNR | 21.3567 (float domain) | **21.3565** (`cv.PSNR`, uint8) | −0.0002 |
| SSIM | 0.90701 (3DGS 11x11 Gaussian) | **0.90034** (skimage default 7x7 uniform) | **−0.00667** |
| LPIPS | 0.14685 (AlexNet, [-1,1]) | **0.12284** (VGG, [-1,1]) | **−0.02401** |

Zero units hit the infinite-PSNR branch, as expected.

### CORRECTION to this page's own prediction

The addendum above states: *"So 0.90701 against a published 0.937-0.944
was partly a CONVENTION gap, not purely a quality gap."* **That is WRONG,
and the measurement refutes it.** The official SSIM convention gives
**0.90034**, which is LOWER than the 3DGS convention's 0.90701, so
switching to DiVa-360's own SSIM does not close any of the distance to
the published numbers — it widens it by 0.0067. The reasoning was
plausible (a uniform 7x7 window is less forgiving than an 11x11 Gaussian
one) but the direction was assumed rather than measured, and it went the
other way. The sentence stands as written above, as history; this is the
correction of record.

The two conventions that DID matter behaved as follows:

* **PSNR convention is immaterial here.** 0.0002 dB. Quantizing the
  render to uint8 changes essentially nothing, which retires the "minor
  mismatch" as a genuine non-issue rather than an unquantified one.
* **The LPIPS backbone was the real one.** VGG gives 0.12284 against
  AlexNet's 0.14685 — a 0.024 improvement purely from using the backbone
  the official evaluator uses. The `[0,1]` variant (0.12398) is now
  formally retired: it corresponds to nothing official, and its
  numerical closeness to the correct VGG figure is a coincidence, not a
  justification.

### Distance to published, under the OFFICIAL conventions

| | ADAGS (exp 103) | PF I-NGP | MixVoxels |
|---|---:|---:|---:|
| PSNR | 21.357 | 25.346 | 25.090 |
| SSIM | 0.9003 | 0.944 | 0.937 |
| LPIPS | 0.1228 | 0.076 | 0.086 |

Short on every metric: about **3.7-4.0 dB** of PSNR, **0.037-0.044** of
SSIM, and **0.037-0.047** of LPIPS. The metric-definition question is now
CLOSED and it was not the explanation — the gap is real and it is large.

**Still NOT parity, and this does not become a like-for-like comparison
by fixing metrics.** M1 stands: the published rows pool 1125 frames at
30 FPS; this scores 141 frames, the first eighth of the sequence. Both
numbers are recorded so the metric axis can be reasoned about
separately, but no ADAGS scissor figure may be placed beside a published
row as a comparison until the temporal extent matches too.

Two measurement notes recorded rather than glossed:

* The evaluator reports PSNR 21.3567 where the training loop's summary
  says 21.3705, a 0.014 dB difference. The evaluator clamps the render
  to [0,1] before scoring and the training loop does not. Neither is
  wrong; they are different conventions and the smaller one is the
  clamped one.
* Experiment 99's `per_camera` block grouped on `Camera.image_name`,
  which for this converter is the eight-digit FRAME index, so it
  reported 141 "cameras". The aggregates are means over all 846 units
  and are unaffected. Fixed at `4dac984` to group on the camera
  directory in `image_path`.
