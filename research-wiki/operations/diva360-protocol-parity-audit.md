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
