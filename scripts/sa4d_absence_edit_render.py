#!/usr/bin/env python
"""Render an SA4D object-removal ("absence") edit of an N3V scene.

WHAT THIS IS FOR
----------------
SA4D ("Segment Any 4D Gaussians", https://github.com/Marine318/sa4d, commit
``9b46359``) trains a per-Gaussian identity MLP + classifier on top of a
4DGaussians/HexPlane deformation model. ``delete.ipynb`` in that repo turns
that classifier into an object DELETION by rendering only the Gaussians it did
*not* select. This script turns the same machinery into a REPRODUCIBLE,
AUDITED asset: an edited N3V sequence in which one object is present, then
absent for a declared window ``[A, B]``, then present again -- together with
every diagnostic needed to decide whether the absence is a real absence or a
rendering artefact.

Two modes:

``--mode preview``
    Cheap reconnaissance on a few cameras and a few frames. Answers "is id 95
    the object I think it is, does it persist across time, and does the
    background behind it actually exist in the model?" before paying for a
    full-rig build.

``--mode build``
    The deliverable. One CANONICAL, time-independent keep table, then every
    rig camera x every frame in ``[A-margin, B+margin]``, plus the composited
    edited images, the removed-Gaussian ground truth, and provenance.

THE ALPHA TRICK, AND WHY IT IS EXACT
------------------------------------
``diff_gaussian_rasterization`` as used by ``render_segmentation``
(gaussian_renderer/__init__.py:154-266) returns only the composited RGB; the
alpha channel is not exposed. But the compositing model is
``I = C + (1 - a) * bg`` for a CONSTANT background ``bg``, so rendering the
same Gaussians twice, once on white and once on black, gives per channel

    I_white - I_black = (1 - a) * (1 - 0) = 1 - a

exactly. Hence ``alpha = 1 - mean_c |I_white - I_black|``. This is exact only
on UNCLIPPED renders, so alpha is always derived from the raw float output and
clipping to [0, 1] happens only when a PNG is written. (Where a render
saturates above 1 the difference collapses and alpha reads 1; that is a
saturated, i.e. opaque, pixel, so the bias is in the harmless direction.)

``alpha_obj`` (from the object-only render) is the object's screen footprint.
``alpha_bg`` (from the background-without-object render) is the SUPPORT map:
where ``alpha_bg`` is low behind the object, the model has NOTHING to show once
the object is removed -- a hole. ``hole_fraction`` is the share of object-mask
pixels that are holes, and it is the single number that decides whether an
absence edit on this scene is honest.

READ THIS BEFORE TRUSTING THE OUTPUT -- API FACTS AND ASSUMPTIONS
-----------------------------------------------------------------
Everything below was read out of commit ``9b46359``; file:line references are
to that commit. Nothing in this file was executed against a GPU while it was
written (the authoring workstation has no torch), so every runtime claim is a
STATIC reading of the SA4D source, not an observed behaviour.

VERIFIED BY READING THE SOURCE
    * ``render_segmentation(viewpoint_camera, pc, pipe, bg_color, mask, t=None,
      scaling_modifier=1.0, override_color=None)`` -- gaussian_renderer/
      __init__.py:154. It takes NO ``cam_type`` argument (unlike ``render()``
      at :20, which does). ``mask`` is a boolean KEEP mask over rows of
      ``pc._xyz``; the function gathers ``pc._xyz[mask]``, ``pc._opacity[mask]``,
      ``pc.get_features[mask]``, ``pc._scaling[mask]``, ``pc._rotation[mask]``
      (:168, :198-210), deforms only that subset (:213-215) and rasterizes it
      over ``bg_color``. Returned dict has "render", "viewspace_points",
      "visibility_filter", "radii", "points2d" -- and ``radii`` has length
      ``mask.sum()``, not N.
    * The identity read is ``gaussians._mlp(means3D, ts)`` ->
      ``gaussians._classifier(enc.unsqueeze(1).permute(2, 0, 1))`` ->
      ``softmax(dim=0)`` -> ``argmax(dim=0)``, with the soft limb
      ``(prob[ids, :, :] > thr).any(dim=0)`` -- delete.ipynb cells 13/15,
      recycler/ie_cut_roasted_beef.ipynb cells 12/18. ``SegNet.forward(point,
      time)`` takes ``point`` [N,3] and ``time`` [N,1] (scene/segnet.py:33) and
      ``_classifier`` is ``Conv2d(feature_dim, 256, 1)``
      (scene/feature_gaussian_model.py:89), so there are exactly 256 classes and
      they are the reference camera's DEVA short ids. Identity is a function of
      (position, time): there is NO per-Gaussian identity array.
    * ``Scene(dataset, gaussians, load_iteration=N, mode="feature",
      cam_view=CV)`` loads ``point_cloud/iteration_N/scene_point_cloud.ply``,
      ``deformation*.pth`` and ``iteration_N/<CV>/{mlp.pt,classifier.pt}``
      (scene/__init__.py:96-107, :399-411 of feature_gaussian_model.py).
      ``cam_view`` is MANDATORY for a dynerf feature model: with ``cam_view=None``
      line 101 does ``os.path.join(..., None)`` and raises.
    * ``cam_view`` ALSO filters the training split down to that ONE camera
      (scene/neural_3D_dataset_NDC.py:369). So ``scene.getTrainCameras()`` is
      300 frames of ``--cam_view`` only, NOT the 19-camera rig. To reach the
      whole rig this script builds a SECOND ``Neural3D_NDC_Dataset`` with
      ``cam_view=None``, constructed with exactly the arguments
      scene/dataset_readers.py:463-465 uses, and wraps it in
      ``FourDGSdataset`` (scene/dataset.py:11). ``scene.getTestCameras()``
      supplies cam00 (``eval_index=0``).
    * ``camera.image_name`` is USELESS for identifying a camera or a frame: it
      is ``f"{index}"``, the flat dataset index (scene/dataset.py:44). Camera
      and frame are therefore recovered from ``dataset.image_paths[i]``, which
      is ``<source>/camXX/images/FFFF.png`` (neural_3D_dataset_NDC.py:379).
      Ordering is sorted-camera-major, frame-minor (:333, :377).
    * ``camera.time == frame / 300`` exactly: ``image_times.append(idx/countss)``
      with ``countss = 300`` hard-coded (neural_3D_dataset_NDC.py:332, :389).
      ``create_mask_table(300)`` fills ``_time_map[i] = i/300``
      (feature_gaussian_model.py:186-191), so the table row index IS the frame
      index. The script still asserts this per camera and records the worst
      deviation in ``rowset.json``.
    * ``points_inside_convex_hull`` (utils/segment_utils.py:43-82) DOES NOT
      COMPUTE A CONVEX HULL IN THIS COMMIT. The Delaunay limb is commented out
      (:72-81) and the function returns an IQR outlier-rejection mask
      ``~any(masked_points outside [Q1 - f*IQR, Q3 + f*IQR])`` of length
      ``mask.sum()``, ignoring every unmasked point. It also never touches the
      ``pytorch3d.ops`` it imports. It is still called here, unmodified, because
      fidelity to the published pipeline matters more than the name being
      wrong -- but it is an axis-aligned IQR box filter and is reported as
      ``iqr_box`` in the counts.
    * The radius filter is ``radii <= radii.mean() + 7 * radii.std()`` on
      ``render_pkg["radii"]`` (recycler/ie_cut_roasted_beef.ipynb cell 18).
    * ``get_combined_args`` from ``utils.segment_utils`` (:84) parses an
      EXPLICIT ``['--model_path', p]`` list and reads ``feature_cfg_args``; the
      same-named function in ``arguments`` (:178) reads ``sys.argv[1:]`` and
      would eat this script's own CLI. The segment_utils one is used.

ASSUMED, NOT EXECUTED (each would fail loudly, not silently, if wrong)
    * That ``render_segmentation`` tolerates its own size mismatch: it builds
      ``screenspace_points`` at the FULL row count N (:161) but passes
      ``means3D`` of length ``mask.sum()``. The published notebooks run this
      way, so the CUDA rasterizer must size itself from ``means3D``. Not
      verified here.
    * That an all-False keep mask would crash the rasterizer. This script never
      finds out: ``_render_keep`` short-circuits an empty keep set to the
      analytic answer (a constant background image).
    * That ``--real_images`` holds ``camXX_FFFF.png`` at, or above, the
      1352x1014 render resolution. A different size is LANCZOS-resized -- the
      same resize ``Neural3D_NDC_Dataset.__getitem__`` applies (:406) -- and
      every resized file is recorded in the output JSON rather than passing
      silently.
    * That every requested ``camXX/images/FFFF.png`` is readable.
      ``FourDGSdataset.__getitem__`` wraps the whole unpack in a bare ``except:``
      (scene/dataset.py:25-41), so a missing or corrupt frame does not surface
      as a file error -- it falls through to the ``caminfo`` branch and dies
      with an ``AttributeError`` on a tuple. Loud, but misleading; check the
      file before believing the traceback.
    * That ``dataset.object_masks = False`` is safe. It is set here so the run
      does not require DEVA pseudo-labels on disk for all 20 cameras; the
      consequence is ``camera.objects is None``, which this script never reads
      (object ids come from ``--ids``).
    * That the composited RGB source should be the BLACK-background render.
      There is no neutral choice: the object-free render has genuine holes, and
      any constant fill is a lie about them. Black is used for BOTH
      ``images_edited`` and ``null_composite`` so the two are comparable, and
      the holes stay visible and are quantified in ``support/`` and
      ``hole_fraction.json`` instead of being cosmetically filled.

DELIBERATE DEVIATIONS FROM THE NOTEBOOKS
    * The notebooks recompute the row set at every timestamp, so the deleted
      set flickers frame to frame. This script's build mode takes a MAJORITY
      vote over the timestamps in ``[A-margin, B+margin]`` and freezes ONE
      canonical set, because an absence edit whose removed set changes per
      frame is not an absence of one object. ``rowset.json`` reports the
      per-timestamp Jaccard against that canonical set so the cost of freezing
      is measurable.
    * The radius filter needs a render, hence a viewpoint. The notebook uses
      "whatever view this timestamp is". A canonical set needs a canonical
      reference, so this script uses ``--cam_view`` at the midpoint frame of
      ``[A, B]`` and records that in ``edit_params.json``.
    * ``points_inside_convex_hull`` is applied to the UNDEFORMED ``get_xyz``
      (as recycler cell 18 does), not to the deformed means (as cell 13 does),
      because the canonical set must not depend on a timestamp.

Run contract: Linux, inside the SA4D venv, with ``cwd`` and ``PYTHONPATH`` both
set to the SA4D checkout (``scene/neural_3D_dataset_NDC.py:14`` does
``sys.path.append('./utils')``, which only works from that cwd). Every heavy
import lives inside the runtime functions so that the pure helpers below --
and their tests -- import with nothing but numpy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

# The frame->time divisor is hard-coded as `countss = 300` in
# scene/neural_3D_dataset_NDC.py:332, and used at :389 as `idx/countss`.
TIME_DIVISOR = 300

# Fixed sensitivity probes for the row-set report. These are NOT tunable: they
# exist so that every run reports how much the selection depends on the soft
# threshold, and a knob would let that report be tuned to look stable.
SENSITIVITY_SOFT_LOW = 0.001
SENSITIVITY_SOFT_HIGH = 0.05

# recycler/ie_cut_roasted_beef.ipynb cell 18.
RADIUS_STD_FACTOR = 7.0
IQR_OUTLIER_FACTOR = 1.0


# --------------------------------------------------------------------------
# Pure helpers. numpy only, no torch, no I/O. These are what tests/ exercises.
# --------------------------------------------------------------------------


def parse_frame_list(spec):
    """Parse a frame specification like ``"25-105,215-275"`` or ``"3,7,9"``.

    Ranges are INCLUSIVE at both ends. The result is sorted and de-duplicated.
    """
    frames = []
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            bounds = chunk.split("-")
            if len(bounds) != 2:
                raise ValueError("malformed frame range: %r" % (chunk,))
            lo, hi = int(bounds[0]), int(bounds[1])
            if hi < lo:
                raise ValueError("descending frame range: %r" % (chunk,))
            frames.extend(range(lo, hi + 1))
        else:
            frames.append(int(chunk))
    if not frames:
        raise ValueError("empty frame specification: %r" % (spec,))
    if min(frames) < 0:
        raise ValueError("negative frame in %r" % (spec,))
    return sorted(set(frames))


def alpha_from_two_renders(img_white, img_black, channel_axis=0):
    """Exact per-pixel alpha from the same content rendered on white and black.

    ``I = C + (1 - a) * bg`` for a constant background, so
    ``I_white - I_black = 1 - a`` per channel. Inputs must be the RAW (unclipped)
    renders; the caller clips only when writing a PNG. Default ``channel_axis=0``
    matches the rasterizer's CHW output.
    """
    white = np.asarray(img_white, dtype=np.float64)
    black = np.asarray(img_black, dtype=np.float64)
    if white.shape != black.shape:
        raise ValueError("render shape mismatch: %r vs %r" % (white.shape, black.shape))
    diff = np.abs(white - black).mean(axis=channel_axis)
    return np.clip(1.0 - diff, 0.0, 1.0)


def binarise(alpha, threshold=0.5):
    """``alpha > threshold`` as a bool array."""
    return np.asarray(alpha, dtype=np.float64) > float(threshold)


def dilate_binary(mask, radius):
    """Binary dilation by a disk of the given pixel radius. ``radius <= 0`` copies."""
    src = np.asarray(mask, dtype=bool)
    if src.ndim != 2:
        raise ValueError("dilate_binary expects a 2-D mask, got %r" % (src.shape,))
    rad = int(radius)
    if rad <= 0:
        return src.copy()
    height, width = src.shape
    out = np.zeros_like(src)
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            if dx * dx + dy * dy > rad * rad:
                continue
            ty0, ty1 = max(0, dy), min(height, height + dy)
            tx0, tx1 = max(0, dx), min(width, width + dx)
            if ty0 >= ty1 or tx0 >= tx1:
                continue
            out[ty0:ty1, tx0:tx1] |= src[ty0 - dy:ty1 - dy, tx0 - dx:tx1 - dx]
    return out


def _convolve1d(array, kernel, axis):
    radius = (len(kernel) - 1) // 2
    pad = [(0, 0)] * array.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(array, pad, mode="edge")
    out = np.zeros_like(array, dtype=np.float64)
    for offset, weight in enumerate(kernel):
        take = [slice(None)] * array.ndim
        take[axis] = slice(offset, offset + array.shape[axis])
        out += weight * padded[tuple(take)]
    return out


def gaussian_feather(mask, sigma):
    """Separable Gaussian blur of a 0/1 mask, truncated at 3 sigma.

    Edge padding is ``"edge"``, so an all-ones mask stays all ones -- a
    feathered full-frame mask must still fully replace the frame.
    """
    arr = np.asarray(mask, dtype=np.float64)
    if sigma is None or float(sigma) <= 0.0:
        return np.clip(arr, 0.0, 1.0)
    sigma = float(sigma)
    radius = int(math.ceil(3.0 * sigma))
    grid = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    kernel /= kernel.sum()
    out = _convolve1d(arr, kernel, axis=1)
    out = _convolve1d(out, kernel, axis=0)
    return np.clip(out, 0.0, 1.0)


def edit_alpha(alpha_obj, threshold=0.5, dilate=0, feather=0.0):
    """The composite's alpha: binarise, dilate, feather. Returns float [0, 1]."""
    hard = binarise(alpha_obj, threshold)
    grown = dilate_binary(hard, dilate)
    return gaussian_feather(grown.astype(np.float64), feather)


def composite_alpha(real_rgb, replacement_rgb, alpha):
    """``edited = real * (1 - a) + replacement * a`` -> HWC uint8.

    ``real_rgb`` and ``replacement_rgb`` are HWC floats in [0, 1]; ``alpha`` is
    HW in [0, 1].
    """
    real = np.asarray(real_rgb, dtype=np.float64)
    rep = np.asarray(replacement_rgb, dtype=np.float64)
    if real.shape != rep.shape:
        raise ValueError("composite shape mismatch: %r vs %r" % (real.shape, rep.shape))
    weight = np.asarray(alpha, dtype=np.float64)
    if weight.shape != real.shape[:2]:
        raise ValueError("alpha shape %r does not match image %r" % (weight.shape, real.shape))
    out = real * (1.0 - weight[..., None]) + rep * weight[..., None]
    return to_uint8_map(out)


def to_uint8_map(array):
    """Clip a float array in [0, 1] and round to uint8."""
    return np.clip(np.asarray(array, dtype=np.float64) * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


def hole_fraction(alpha_obj, alpha_bg, object_threshold=0.5, support_threshold=0.5):
    """Share of object-mask pixels with no background support behind them.

    Returns ``hole_fraction: None`` when the object mask is empty -- a fraction
    with no denominator is not a measurement.
    """
    obj = binarise(alpha_obj, object_threshold)
    count = int(obj.sum())
    if count == 0:
        return {"object_px": 0, "hole_px": 0, "hole_fraction": None}
    support = np.asarray(alpha_bg, dtype=np.float64)
    if support.shape != obj.shape:
        raise ValueError("alpha shape mismatch: %r vs %r" % (support.shape, obj.shape))
    holes = int((support[obj] < float(support_threshold)).sum())
    return {"object_px": count, "hole_px": holes, "hole_fraction": holes / count}


def mask_bbox_centroid(mask):
    """Pixel count, inclusive ``[x0, y0, x1, y1]`` bbox and ``[cx, cy]`` centroid."""
    arr = np.asarray(mask, dtype=bool)
    count = int(arr.sum())
    if count == 0:
        return {"count": 0, "bbox": None, "centroid": None}
    ys, xs = np.nonzero(arr)
    return {
        "count": count,
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "centroid": [float(xs.mean()), float(ys.mean())],
    }


def centroid_velocity(previous_centroid, centroid, frame_gap):
    """Per-frame centroid velocity, or ``None`` when either endpoint is missing."""
    if previous_centroid is None or centroid is None:
        return None
    gap = int(frame_gap)
    if gap <= 0:
        return None
    dx = (float(centroid[0]) - float(previous_centroid[0])) / gap
    dy = (float(centroid[1]) - float(previous_centroid[1])) / gap
    return {
        "frame_gap": gap,
        "dx_per_frame": dx,
        "dy_per_frame": dy,
        "speed_px_per_frame": math.hypot(dx, dy),
    }


def majority_rowset(table):
    """Strict majority over the timestamps of a ``[T, N]`` bool table.

    A row is kept when it is selected in MORE than half the timestamps; an
    exact tie is NOT a majority and is rejected.
    """
    arr = np.asarray(table, dtype=bool)
    if arr.ndim != 2:
        raise ValueError("majority_rowset expects [T, N], got %r" % (arr.shape,))
    n_times = arr.shape[0]
    if n_times == 0:
        raise ValueError("majority_rowset over zero timestamps")
    return (arr.sum(axis=0) * 2) > n_times


def jaccard(left, right):
    """Jaccard index of two boolean row sets; two empty sets score 1.0."""
    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("jaccard shape mismatch: %r vs %r" % (a.shape, b.shape))
    union = int((a | b).sum())
    if union == 0:
        return 1.0
    return float((a & b).sum()) / union


def apply_subset_filter(keep, sub_keep):
    """``keep[keep] = sub_keep`` without mutating ``keep``.

    ``sub_keep`` is indexed by position WITHIN the currently kept rows -- the
    shape both ``points_inside_convex_hull`` and the ``radii`` filter return.
    Works on numpy bool arrays and on torch bool tensors alike (torch exposes
    ``.clone()``), so the runtime path and the tested path are one function.
    """
    out = keep.clone() if hasattr(keep, "clone") else np.array(keep, dtype=bool, copy=True)
    out[keep] = sub_keep
    return out


def rowset_summary(hard, soft_low, soft_high):
    """Sizes of (argmax-only), (argmax | soft_low), (argmax | soft_high)."""
    hard_a = np.asarray(hard, dtype=bool)
    low_a = np.asarray(soft_low, dtype=bool)
    high_a = np.asarray(soft_high, dtype=bool)
    return {
        "argmax_only": int(hard_a.sum()),
        "argmax_or_soft_low": int((hard_a | low_a).sum()),
        "argmax_or_soft_high": int((hard_a | high_a).sum()),
    }


def camera_frame_from_image_path(path):
    """``<...>/cam15/images/0060.png`` -> ``("cam15", 60)``.

    This is the only reliable camera/frame identity available:
    ``camera.image_name`` is the flat dataset index (scene/dataset.py:44).
    """
    parts = str(path).replace("\\", "/").rstrip("/").split("/")
    if len(parts) < 3:
        raise ValueError("cannot parse camera/frame from %r" % (path,))
    if parts[-2] != "images":
        raise ValueError("expected .../camXX/images/FFFF.png, got %r" % (path,))
    return parts[-3], int(parts[-1].rsplit(".", 1)[0])


def frame_name(camera_name, frame):
    """``("cam15", 60)`` -> ``"cam15_0060"``."""
    return "%s_%04d" % (camera_name, int(frame))


def sha256_file(path):
    """Hex sha256 of a file, or ``None`` if it does not exist."""
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Runtime. Everything below imports torch / SA4D and needs a GPU.
# --------------------------------------------------------------------------


def _load_sa4d(args):
    """Build ``(dataset, pipe, gaussians, scene)`` exactly as delete.ipynb does."""
    from argparse import ArgumentParser

    from arguments import ModelParams, PipelineParams, ModelHiddenParams
    from scene import Scene, GaussianModel
    from utils.segment_utils import get_combined_args

    parser = ArgumentParser(description="sa4d_absence_edit_render")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hidden = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=args.iteration, type=int)
    parser.add_argument("--mode", default="feature", choices=["scene", "feature"])
    parser.add_argument("--configs", type=str, default=args.configs)

    # utils.segment_utils.get_combined_args:84 parses an EXPLICIT argv list, so
    # this script's own flags are invisible to it, and reads `feature_cfg_args`.
    sa4d_args = get_combined_args(parser, args.model_path, "feature")
    if args.configs:
        import mmcv

        from utils.params_utils import merge_hparams

        sa4d_args = merge_hparams(sa4d_args, mmcv.Config.fromfile(args.configs))

    dataset = model.extract(sa4d_args)
    hyperparam = hidden.extract(sa4d_args)
    pipe = pipeline.extract(sa4d_args)

    # Object ids come from --ids, so no DEVA pseudo-labels are needed on disk.
    dataset.object_masks = False
    dataset.need_gt_masks = False

    gaussians = GaussianModel(dataset.sh_degree, "feature", hyperparam, dataset.feature_dim)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=sa4d_args.iteration,
        mode="feature",
        shuffle=False,
        cam_view=args.cam_view,
    )
    return dataset, pipe, gaussians, scene, sa4d_args


def _build_camera_index(dataset, scene):
    """Map ``camera_name -> {frame: (holder, index)}`` over the whole rig.

    ``scene.getTrainCameras()`` is only ``--cam_view`` (neural_3D_dataset_NDC.py
    :369), so the 19 non-held-out cameras are re-read from a second
    ``Neural3D_NDC_Dataset`` built with the arguments of
    scene/dataset_readers.py:463-465 but ``cam_view=None``. cam00 comes from the
    test split (``eval_index=0``).
    """
    from scene.dataset import FourDGSdataset
    from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset

    rig_raw = Neural3D_NDC_Dataset(
        dataset.source_path,
        "train",
        1.0,
        time_scale=1,
        scene_bbox_min=[-2.5, -2.0, -1.0],
        scene_bbox_max=[2.5, 2.0, 1.0],
        eval_index=0,
        object_masks=False,
        mode="feature",
        cam_view=None,
    )
    rig = FourDGSdataset(rig_raw, dataset, "dynerf")
    held_out = scene.getTestCameras()

    index = {}
    for holder, raw in ((rig, rig_raw), (held_out, held_out.dataset)):
        for position, path in enumerate(raw.image_paths):
            camera_name, frame = camera_frame_from_image_path(path)
            index.setdefault(camera_name, {})[frame] = (holder, position)
    return index


def _get_view(index, camera_name, frame):
    try:
        holder, position = index[camera_name][frame]
    except KeyError:
        raise SystemExit("camera %s frame %d is not in the dataset" % (camera_name, frame))
    return holder[position]


def _row_sets(gaussians, time_value, ids_tensor, thresholds):
    """delete.ipynb cell 15, verbatim, plus extra soft thresholds.

    Returns ``(hard[N], {threshold: soft[N]})`` as bool cuda tensors.
    """
    import torch

    means3d = gaussians.get_xyz
    times = torch.tensor(time_value).to(means3d.device).repeat(means3d.shape[0], 1)
    identity_encoding = gaussians._mlp(means3d, times)
    logits3d = gaussians._classifier(identity_encoding.unsqueeze(1).permute(2, 0, 1))
    prob_obj3d = torch.softmax(logits3d, dim=0)
    obj3d = torch.argmax(prob_obj3d, dim=0)
    hard = (obj3d[..., None] == ids_tensor[None, :]).any(dim=-1).squeeze()
    soft = {}
    for threshold in thresholds:
        soft[threshold] = (prob_obj3d[ids_tensor, :, :] > threshold).any(dim=0).squeeze()
    return hard, soft


def _render_keep(view, gaussians, pipe, keep_mask, background, background_value, height, width):
    """Raw (UNCLIPPED) CHW float render of the kept rows over a constant bg.

    An empty keep set is answered analytically instead of handing the CUDA
    rasterizer zero primitives.
    """
    from gaussian_renderer import render_segmentation

    if int(keep_mask.sum().item()) == 0:
        return np.full((3, height, width), float(background_value), dtype=np.float64)
    rendered = render_segmentation(view, gaussians, pipe, background, keep_mask.bool())["render"]
    return rendered.detach().float().cpu().numpy().astype(np.float64)


def _chw_to_hwc_uint8(chw):
    return to_uint8_map(np.asarray(chw, dtype=np.float64)).transpose(1, 2, 0)


def _chw_to_hwc_float(chw):
    return np.clip(np.asarray(chw, dtype=np.float64), 0.0, 1.0).transpose(1, 2, 0)


def _save_png(path, array):
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(array).save(path)


def _load_real_image(real_images_dir, camera_name, frame, height, width):
    """Load ``<dir>/camXX_FFFF.png`` as HWC float in [0, 1]. Returns (image, resized)."""
    from PIL import Image

    path = os.path.join(real_images_dir, frame_name(camera_name, frame) + ".png")
    if not os.path.isfile(path):
        raise SystemExit("real image not found: %s" % (path,))
    image = Image.open(path).convert("RGB")
    resized = False
    if image.size != (width, height):
        # The same LANCZOS resize Neural3D_NDC_Dataset.__getitem__:406 applies.
        image = image.resize((width, height), Image.LANCZOS)
        resized = True
    return np.asarray(image, dtype=np.float64) / 255.0, resized


def _deformed_rows(gaussians, keep_mask, time_value):
    """Deformed, ACTIVATED state of the kept rows at ``time_value``.

    Mirrors gaussian_renderer/__init__.py:213-226: deform, then apply
    ``scaling_activation`` / ``rotation_activation`` / ``opacity_activation``,
    so these are the quantities the rasterizer actually consumed.
    """
    import torch

    row_index = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
    means3d = gaussians.get_xyz[keep_mask]
    scales = gaussians._scaling[keep_mask]
    rotations = gaussians._rotation[keep_mask]
    opacity = gaussians._opacity[keep_mask]
    shs = gaussians.get_features[keep_mask]
    times = torch.tensor(time_value).to(means3d.device).repeat(means3d.shape[0], 1)
    xyz, scale_f, rot_f, opacity_f, _ = gaussians._deformation(
        means3d, scales, rotations, opacity, shs, times
    )
    return {
        "row_index": row_index.detach().cpu().numpy().astype(np.int64),
        "xyz": xyz.detach().float().cpu().numpy(),
        "scales": gaussians.scaling_activation(scale_f).detach().float().cpu().numpy(),
        "rotations": gaussians.rotation_activation(rot_f).detach().float().cpu().numpy(),
        "opacity": gaussians.opacity_activation(opacity_f).detach().float().cpu().numpy(),
    }


def _provenance(args, sa4d_args, iteration):
    """sha256 of every loaded model file and of every imported SA4D source file."""
    import torch

    import scene as _scene_pkg

    sa4d_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(_scene_pkg.__file__))))

    model_dir = args.model_path
    point_cloud_dir = os.path.join(model_dir, "point_cloud", "iteration_%d" % iteration)
    model_files = {}
    for relative in (
        "scene_point_cloud.ply",
        "deformation.pth",
        "deformation_table.pth",
        "deformation_accum.pth",
        os.path.join(args.cam_view, "mlp.pt"),
        os.path.join(args.cam_view, "classifier.pt"),
    ):
        key = ("point_cloud/iteration_%d/%s" % (iteration, relative)).replace("\\", "/")
        model_files[key] = sha256_file(os.path.join(point_cloud_dir, relative))
    for relative in ("cfg_args", "feature_cfg_args"):
        model_files[relative] = sha256_file(os.path.join(model_dir, relative))

    sources = {}
    for _name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if not path:
            continue
        absolute = os.path.abspath(path)
        if absolute.startswith(sa4d_root + os.sep) and absolute.endswith(".py"):
            sources[os.path.relpath(absolute, sa4d_root).replace("\\", "/")] = sha256_file(absolute)

    try:
        import pytorch3d

        pytorch3d_version = getattr(pytorch3d, "__version__", None)
    except Exception:
        pytorch3d_version = None

    return {
        "sa4d_root": sa4d_root,
        "cwd": os.getcwd(),
        "argv": list(sys.argv),
        "model_path": model_dir,
        "iteration": iteration,
        "cam_view": args.cam_view,
        "source_path": getattr(sa4d_args, "source_path", None),
        "model_files_sha256": model_files,
        "sa4d_sources_sha256": dict(sorted(sources.items())),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "pytorch3d_version": pytorch3d_version,
        "python_version": sys.version,
    }


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _camera_names(args, index):
    """Resolve ``--cameras 0 15 8`` to dataset camera names, in the order given."""
    names = []
    for number in args.cameras:
        name = "cam%02d" % int(number)
        if name not in index:
            raise SystemExit("camera %s is not in the dataset (have: %s)"
                             % (name, ", ".join(sorted(index))))
        names.append(name)
    return names


def _time_check(view, frame, worst):
    """Track the worst deviation of ``camera.time`` from ``frame / 300``."""
    deviation = abs(float(view.time) - frame / float(TIME_DIVISOR))
    return max(worst, deviation)


# --------------------------------------------------------------------------


def run_preview(args):
    import torch

    dataset, pipe, gaussians, scene, sa4d_args = _load_sa4d(args)
    index = _build_camera_index(dataset, scene)
    cameras = _camera_names(args, index)
    frames = parse_frame_list(args.frames)

    ids_tensor = torch.tensor(list(args.ids)).cuda()
    white = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    black = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    thresholds = sorted({float(args.soft_thresh), SENSITIVITY_SOFT_LOW, SENSITIVITY_SOFT_HIGH})

    per_timestamp = []
    per_view = []
    previous_selection = None
    previous_centroids = {name: (None, None) for name in cameras}
    worst_time_deviation = 0.0
    resized_real = []

    with torch.no_grad():
        for frame in frames:
            time_value = frame / float(TIME_DIVISOR)
            hard, soft = _row_sets(gaussians, time_value, ids_tensor, thresholds)
            selection = hard | soft[float(args.soft_thresh)]
            selection_np = selection.detach().cpu().numpy().astype(bool)

            summary = rowset_summary(
                hard.detach().cpu().numpy(),
                soft[SENSITIVITY_SOFT_LOW].detach().cpu().numpy(),
                soft[SENSITIVITY_SOFT_HIGH].detach().cpu().numpy(),
            )
            per_timestamp.append({
                "frame": frame,
                "time": time_value,
                "selected_rows": int(selection_np.sum()),
                "jaccard_with_previous": (
                    None if previous_selection is None else jaccard(previous_selection, selection_np)
                ),
                "sensitivity": summary,
                "soft_thresh": float(args.soft_thresh),
            })
            previous_selection = selection_np

            keep_object = selection.bool()
            keep_background = ~keep_object

            for camera_name in cameras:
                view = _get_view(index, camera_name, frame)
                worst_time_deviation = _time_check(view, frame, worst_time_deviation)
                height = int(view.image_height)
                width = int(view.image_width)

                obj_w = _render_keep(view, gaussians, pipe, keep_object, white, 1.0, height, width)
                obj_b = _render_keep(view, gaussians, pipe, keep_object, black, 0.0, height, width)
                bg_w = _render_keep(view, gaussians, pipe, keep_background, white, 1.0, height, width)
                bg_b = _render_keep(view, gaussians, pipe, keep_background, black, 0.0, height, width)

                alpha_obj = alpha_from_two_renders(obj_w, obj_b)
                alpha_bg = alpha_from_two_renders(bg_w, bg_b)

                if args.real_images:
                    real, resized = _load_real_image(args.real_images, camera_name, frame, height, width)
                    if resized:
                        resized_real.append(frame_name(camera_name, frame))
                else:
                    real = view.original_image.detach().float().cpu().numpy().transpose(1, 2, 0)
                    real = np.clip(real.astype(np.float64), 0.0, 1.0)

                stem = frame_name(camera_name, frame)
                # The black-background render IS the premultiplied colour C, so
                # `real * (1 - a) + C` is a correct over-composite.
                _save_png(os.path.join(args.out, "preview", stem + "_obj.png"),
                          to_uint8_map(real * (1.0 - alpha_obj[..., None]) + _chw_to_hwc_float(obj_b)))
                _save_png(os.path.join(args.out, "preview", stem + "_bg.png"),
                          to_uint8_map(real * (1.0 - alpha_bg[..., None]) + _chw_to_hwc_float(bg_b)))
                _save_png(os.path.join(args.out, "alpha_obj", stem + ".png"), to_uint8_map(alpha_obj))
                _save_png(os.path.join(args.out, "support", stem + ".png"), to_uint8_map(alpha_bg))

                geometry = mask_bbox_centroid(binarise(alpha_obj, 0.5))
                previous_frame, previous_centroid = previous_centroids[camera_name]
                velocity = centroid_velocity(
                    previous_centroid,
                    geometry["centroid"],
                    0 if previous_frame is None else frame - previous_frame,
                )
                previous_centroids[camera_name] = (frame, geometry["centroid"])

                per_view.append({
                    "camera": camera_name,
                    "frame": frame,
                    "time": float(view.time),
                    "object_mask_px": geometry["count"],
                    "bbox": geometry["bbox"],
                    "centroid": geometry["centroid"],
                    "centroid_velocity": velocity,
                    "hole": hole_fraction(alpha_obj, alpha_bg),
                })

    _write_json(os.path.join(args.out, "preview.json"), {
        "mode": "preview",
        "ids": list(args.ids),
        "cameras": cameras,
        "frames": frames,
        "soft_thresh": float(args.soft_thresh),
        "sensitivity_thresholds": {
            "soft_low": SENSITIVITY_SOFT_LOW,
            "soft_high": SENSITIVITY_SOFT_HIGH,
        },
        "per_timestamp": per_timestamp,
        "per_view": per_view,
        "worst_camera_time_deviation": worst_time_deviation,
        "real_images_resized": resized_real,
        "provenance": _provenance(args, sa4d_args, int(scene.loaded_iter)),
    })
    print("preview written to %s" % (args.out,))


def run_build(args):
    import torch

    from gaussian_renderer import render_segmentation
    from utils.segment_utils import points_inside_convex_hull

    window_lo, window_hi = int(args.window[0]), int(args.window[1])
    if window_hi < window_lo:
        raise SystemExit("--window expects A <= B, got %d %d" % (window_lo, window_hi))
    margin = int(args.margin)
    render_lo = max(0, window_lo - margin)
    render_hi = min(TIME_DIVISOR - 1, window_hi + margin)
    render_frames = list(range(render_lo, render_hi + 1))

    dataset, pipe, gaussians, scene, sa4d_args = _load_sa4d(args)
    index = _build_camera_index(dataset, scene)
    camera_names = sorted(index)

    ids_tensor = torch.tensor(list(args.ids)).cuda()
    white = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    black = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    thresholds = sorted({float(args.soft_thresh), SENSITIVITY_SOFT_LOW, SENSITIVITY_SOFT_HIGH})

    n_rows = int(gaussians.get_xyz.shape[0])
    per_timestamp = []
    table = np.zeros((len(render_frames), n_rows), dtype=bool)

    with torch.no_grad():
        # ---- 1. per-timestamp selections over [A-margin, B+margin] ----------
        for position, frame in enumerate(render_frames):
            time_value = frame / float(TIME_DIVISOR)
            hard, soft = _row_sets(gaussians, time_value, ids_tensor, thresholds)
            selection = (hard | soft[float(args.soft_thresh)]).detach().cpu().numpy().astype(bool)
            table[position] = selection
            per_timestamp.append({
                "frame": frame,
                "time": time_value,
                "selected_rows": int(selection.sum()),
                "sensitivity": rowset_summary(
                    hard.detach().cpu().numpy(),
                    soft[SENSITIVITY_SOFT_LOW].detach().cpu().numpy(),
                    soft[SENSITIVITY_SOFT_HIGH].detach().cpu().numpy(),
                ),
            })

        # ---- 2. canonical set: majority, then IQR box, then radius ----------
        canonical_np = majority_rowset(table)
        count_majority = int(canonical_np.sum())
        if count_majority == 0:
            raise SystemExit(
                "majority vote over frames %d-%d selected zero rows for ids %s"
                % (render_lo, render_hi, args.ids)
            )

        canonical = torch.from_numpy(canonical_np).cuda()
        keep_sub = points_inside_convex_hull(
            gaussians.get_xyz, canonical, outlier_factor=IQR_OUTLIER_FACTOR
        )
        canonical = apply_subset_filter(canonical, keep_sub)
        count_after_iqr = int(canonical.sum().item())
        if count_after_iqr == 0:
            raise SystemExit("the IQR box filter emptied the canonical set")

        reference_frame = (window_lo + window_hi) // 2
        reference_view = _get_view(index, args.cam_view, reference_frame)
        radii = render_segmentation(
            reference_view, gaussians, pipe, black, canonical.bool()
        )["radii"].float()
        keep_sub = radii <= radii.mean() + RADIUS_STD_FACTOR * radii.std()
        canonical = apply_subset_filter(canonical, keep_sub)
        count_after_radius = int(canonical.sum().item())
        if count_after_radius == 0:
            raise SystemExit("the radius filter emptied the canonical set")

        canonical_np = canonical.detach().cpu().numpy().astype(bool)
        for position, record in enumerate(per_timestamp):
            record["jaccard_with_canonical"] = jaccard(table[position], canonical_np)
        del table

        # ---- 3. the keep table: edited inside [A, B], keep-all outside ------
        gaussians.create_mask_table(TIME_DIVISOR)
        keep_all = torch.ones(n_rows, dtype=torch.bool, device="cuda")
        keep_edited = ~canonical.bool()
        for row in range(TIME_DIVISOR):
            gaussians._time_map[row] = row / float(TIME_DIVISOR)
            gaussians._mask_table[row] = keep_edited if window_lo <= row <= window_hi else keep_all

        # ---- 4. render every rig camera over the render window --------------
        holes = {}
        worst_time_deviation = 0.0
        resized_real = []
        # The construction mask is meant to be non-empty on every frame. It can
        # still come out empty if the object leaves a camera's frustum, so that
        # is COUNTED and reported rather than asserted away.
        empty_construction = []
        for camera_name in camera_names:
            holes[camera_name] = {}
            for frame in render_frames:
                view = _get_view(index, camera_name, frame)
                worst_time_deviation = _time_check(view, frame, worst_time_deviation)
                height = int(view.image_height)
                width = int(view.image_width)
                stem = frame_name(camera_name, frame)
                in_window = window_lo <= frame <= window_hi

                # Nearest-timestamp table lookup, exactly delete.ipynb cell 16.
                difference = torch.abs(gaussians._time_map - view.time)
                table_row = int(torch.argmin(difference).item())
                keep_background = gaussians._mask_table[table_row]

                obj_w = _render_keep(view, gaussians, pipe, canonical, white, 1.0, height, width)
                obj_b = _render_keep(view, gaussians, pipe, canonical, black, 0.0, height, width)
                bg_w = _render_keep(view, gaussians, pipe, keep_background, white, 1.0, height, width)
                bg_b = _render_keep(view, gaussians, pipe, keep_background, black, 0.0, height, width)

                alpha_obj = alpha_from_two_renders(obj_w, obj_b)
                alpha_bg = alpha_from_two_renders(bg_w, bg_b)

                render_dir = os.path.join(args.out, "render", camera_name)
                _save_png(os.path.join(render_dir, "%04d_bgW.png" % frame), _chw_to_hwc_uint8(bg_w))
                _save_png(os.path.join(render_dir, "%04d_bgB.png" % frame), _chw_to_hwc_uint8(bg_b))
                _save_png(os.path.join(render_dir, "%04d_objW.png" % frame), _chw_to_hwc_uint8(obj_w))
                _save_png(os.path.join(render_dir, "%04d_objB.png" % frame), _chw_to_hwc_uint8(obj_b))
                _save_png(os.path.join(args.out, "alpha_obj", stem + ".png"), to_uint8_map(alpha_obj))
                _save_png(os.path.join(args.out, "support", stem + ".png"), to_uint8_map(alpha_bg))

                construction = binarise(alpha_obj, 0.5)
                if not construction.any():
                    empty_construction.append(stem)
                _save_png(os.path.join(args.out, "construction_masks", stem + ".png"),
                          (construction.astype(np.uint8) * 255))
                visible = np.zeros_like(construction) if in_window else construction
                _save_png(os.path.join(args.out, "visible_object", stem + ".png"),
                          (visible.astype(np.uint8) * 255))

                holes[camera_name]["%04d" % frame] = hole_fraction(alpha_obj, alpha_bg)

                if in_window:
                    real, resized = _load_real_image(
                        args.real_images, camera_name, frame, height, width
                    )
                    if resized:
                        resized_real.append(stem)
                    weight = edit_alpha(alpha_obj, 0.5, args.dilate, args.feather)
                    _save_png(os.path.join(args.out, "images_edited", stem + ".png"),
                              composite_alpha(real, _chw_to_hwc_float(bg_b), weight))
                    if camera_name == "cam00":
                        # Same mask, same dilate/feather, but the UNEDITED scene:
                        # any difference from the real image is seam + render
                        # domain gap, not the removal.
                        full_b = _render_keep(view, gaussians, pipe, keep_all, black, 0.0,
                                              height, width)
                        _save_png(os.path.join(args.out, "null_composite", stem + ".png"),
                                  composite_alpha(real, _chw_to_hwc_float(full_b), weight))

        # ---- 5. removed-row ground truth, one file per frame ----------------
        for frame in render_frames:
            payload = _deformed_rows(gaussians, canonical.bool(), frame / float(TIME_DIVISOR))
            target = os.path.join(args.out, "truth3d", "removed_%04d.npz" % frame)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            np.savez_compressed(target, **payload)

    _write_json(os.path.join(args.out, "hole_fraction.json"), holes)
    _write_json(os.path.join(args.out, "rowset.json"), {
        "n_rows_total": n_rows,
        "render_frames": [render_lo, render_hi],
        "window": [window_lo, window_hi],
        "canonical_counts": {
            "majority": count_majority,
            "after_iqr_box": count_after_iqr,
            "after_radius": count_after_radius,
        },
        "canonical_filters": {
            "majority_rule": "selected in strictly more than half of the render-window timestamps",
            "iqr_box": {
                "function": "utils.segment_utils.points_inside_convex_hull",
                "outlier_factor": IQR_OUTLIER_FACTOR,
                "note": "IQR box on the UNDEFORMED get_xyz; the hull limb is commented out upstream",
            },
            "radius": {
                "rule": "radii <= mean + %g * std" % RADIUS_STD_FACTOR,
                "reference_camera": args.cam_view,
                "reference_frame": reference_frame,
            },
        },
        "per_timestamp": per_timestamp,
        "worst_camera_time_deviation": worst_time_deviation,
        "real_images_resized": resized_real,
        "empty_construction_masks": empty_construction,
    })
    _write_json(os.path.join(args.out, "edit_params.json"), {
        "args": {key: value for key, value in sorted(vars(args).items())},
        "ids": list(args.ids),
        "window": [window_lo, window_hi],
        "render_frames": [render_lo, render_hi],
        "soft_thresh": float(args.soft_thresh),
        "sensitivity_thresholds": {
            "soft_low": SENSITIVITY_SOFT_LOW,
            "soft_high": SENSITIVITY_SOFT_HIGH,
        },
        "dilate_px": int(args.dilate),
        "feather_sigma_px": float(args.feather),
        "object_alpha_threshold": 0.5,
        "support_threshold": 0.5,
        "composite_source": "black-background SA4D render (premultiplied colour)",
        "cameras": camera_names,
    })
    _write_json(os.path.join(args.out, "sa4d_provenance.json"),
                _provenance(args, sa4d_args, int(scene.loaded_iter)))
    print("build written to %s (%d cameras x %d frames)"
          % (args.out, len(camera_names), len(render_frames)))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Render an SA4D object-removal edit of an N3V scene.")
    parser.add_argument("--mode", required=True, choices=["preview", "build"])
    parser.add_argument("--model_path", required=True,
                        help="SA4D output dir holding point_cloud/ and feature_cfg_args")
    parser.add_argument("--iteration", type=int, default=14000)
    parser.add_argument("--cam_view", required=True,
                        help="reference camera whose mlp.pt/classifier.pt are loaded")
    parser.add_argument("--configs", type=str, default=None,
                        help="mmcv config, e.g. arguments/dynerf/cut_roasted_beef.py")
    parser.add_argument("--ids", type=int, nargs="+", required=True,
                        help="DEVA short ids of the object, in the cam_view classifier's 256 classes")
    parser.add_argument("--soft_thresh", type=float, default=SENSITIVITY_SOFT_LOW,
                        help="softmax threshold for the soft limb of the row set")
    parser.add_argument("--real_images", type=str, default=None,
                        help="directory of camXX_FFFF.png real frames (required for --mode build)")
    parser.add_argument("--out", required=True)
    # preview only
    parser.add_argument("--cameras", type=int, nargs="+",
                        help="preview: camera numbers, e.g. 0 15 8 (0 is the held-out cam00)")
    parser.add_argument("--frames", type=str,
                        help="preview: frame list, e.g. 25-105,215-275")
    # build only
    parser.add_argument("--window", type=int, nargs=2, metavar=("A", "B"),
                        help="build: inclusive absence window")
    parser.add_argument("--margin", type=int, default=20,
                        help="build: frames rendered either side of the window")
    parser.add_argument("--dilate", type=int, default=6,
                        help="build: composite mask dilation radius in pixels")
    parser.add_argument("--feather", type=float, default=2.0,
                        help="build: composite mask Gaussian feather sigma in pixels")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "preview":
        if not args.cameras or not args.frames:
            raise SystemExit("--mode preview needs --cameras and --frames")
        run_preview(args)
    else:
        if args.window is None:
            raise SystemExit("--mode build needs --window A B")
        if not args.real_images:
            raise SystemExit("--mode build needs --real_images for the composites")
        run_build(args)


if __name__ == "__main__":
    main()
