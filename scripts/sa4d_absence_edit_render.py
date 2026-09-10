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

ROW-SET NARROWING (``--argmax_only`` / ``--scale_max_factor`` /
``--box_percentile`` / ``--mask_consistency``)
-------------------------------------------------------------------------
The unfiltered selection is not the object. Measured on
``cut_roasted_beef`` at iteration 14000, ``--cam_view cam15``, ``--ids 95``
(the dog on a stool): 1,579 rows argmax-only, 1,854 at soft > 0.05, 2,773 at
soft > 0.001, and the object-only render covered ~236,000 px of cam00 (bbox
[0, 380, 652, 1013]) as a diffuse smear, against a real dog silhouette of
roughly 40-60k px. The background-without-object render also lost the stool.
So the selection carries large diffuse Gaussians and neighbouring content.

Four OPTIONAL narrowing steps are available, applied in this fixed order.
All four are OFF by default, and with all four absent this script behaves
exactly as it did before them. Every step's surviving row count is reported
under ``filters`` in ``preview.json`` / ``rowset.json``.

    1. ``--argmax_only`` -- the base row set is argmax-in-``--ids`` only; the
       soft limb is not unioned in. The soft SENSITIVITY numbers are still
       reported, so the cost of dropping it stays visible.
    2. ``--scale_max_factor F`` -- drop rows whose LARGEST ACTIVATED scale at
       that timestamp exceeds ``F x median(largest activated scale)`` of the
       base set. The activated, deformed scale is used because that is what
       the rasterizer consumed (gaussian_renderer/__init__.py:213-226). The
       base set's scale distribution (median, p90, p99, max) is reported
       whether or not the filter is on.
    3. ``--box_percentile P`` (with ``--box_pad``, default 0.1) -- keep rows
       inside the per-axis ``[P, 100-P]`` percentile box of the base set's
       CANONICAL (undeformed ``get_xyz``) positions, each axis grown by
       ``box_pad`` times that axis's extent on BOTH sides. Canonical, not
       deformed, so the box does not depend on a timestamp -- the same reason
       ``points_inside_convex_hull`` is applied to ``get_xyz``.
    4. ``--mask_consistency <deva_root> --mask_min_cams K --mask_frames L``
       -- a multi-view DEVA-mask vote. At each anchor frame in ``L`` the base
       rows (after steps 2 and 3) are deformed to that timestamp and projected
       into all 19 TRAINING cameras; each camera's DEVA id map is read from
       ``<deva_root>/camXX/pseudo_label/object_mask/FFFF.png``. Per camera the
       HARMONISED id is the modal id the projected rows land on -- DEVA ids
       are per-camera, so cam15's id 95 is not cam07's id 95 and they must be
       matched by agreement rather than by number. A row is consistent in a
       camera when its projection lands inside the image on that camera's
       harmonised id, and survives the anchor frame when it is consistent in
       at least ``K`` cameras. The final mask keeps rows that survive at least
       ``--mask_min_frames`` (default 1.0, i.e. EVERY) anchor frame. The
       resulting row mask is frozen once and intersected into every timestamp,
       in preview and in build alike.

THE PROJECTION USED BY STEP 4, AND THE PRECONDITION THAT GUARDS IT
------------------------------------------------------------------
``project_points`` reproduces the rasterizer's own convention exactly, read
out of the vendored ``diff-gaussian-rasterization`` in THIS repository:

    p_hom  = [x, y, z, 1] @ full_proj_transform
             (cuda_rasterizer/auxiliary.h:69-78 -- ``transformPoint4x4``
             indexes ``matrix[0], matrix[4], matrix[8], matrix[12]`` for the
             first output, i.e. a ROW-vector times the matrix as stored, and
             ``full_proj_transform`` is already stored transposed by
             scene/cameras.py; called at cuda_rasterizer/forward.cu:148)
    p_proj = p_hom.xyz / (p_hom.w + 1e-7)      (forward.cu:149-150)
    pixel  = ((p_proj + 1) * S - 1) * 0.5      (auxiliary.h:42-45, applied at
                                                forward.cu:467)
    in front of the camera iff  ([x,y,z,1] @ world_view_transform).z > 0.2
                                               (forward.cu:151-153)

Note this is NOT ``(ndc + 1) * 0.5 * (S - 1)``, which the ADAGS helper
``utils/motion_prior_utils.py:147-148`` uses; the two differ by up to half a
pixel at the frame edge and the rasterizer's form is the one used here.

Because a wrong projection would silently make EVERY row inconsistent and
return a clean, small, entirely fictitious row set, step 4 carries a
PRECONDITION rather than only a reading rule: on the first (camera, anchor
frame) pair the analytic projection is compared against the ``points2d`` that
``render_segmentation`` returns, and a median disagreement above
``PROJECTION_CHECK_TOL_PX`` ABORTS the run. Whether the check actually ran,
and its statistics, are recorded in ``filters.mask_consistency.
projection_check`` -- if ``points2d`` turns out to be unavailable or of an
unexpected shape the run continues with the analytic projection and says so,
so an unexercised precondition is never mistaken for a passed one.

WHAT DEFINES THE EDIT REGION (``--edit_region``)
------------------------------------------------
``alpha`` (the default, and the behaviour this script had before the flag
existed) takes the edit region from the object-only render's own alpha. On
``cut_roasted_beef`` id 95 that region is NOT the dog: it is a diffuse smear of
roughly 236,000 px -- about a quarter of the frame -- against a real dog
silhouette of some 40-60k px. An alpha that wide cannot define an absence edit;
it defines a hole in the picture.

``deva`` takes the edit region from the SEGMENTER instead. For each camera and
frame the silhouette ``S`` is the union of the listed DEVA ids' pixels in
``<deva_root>/camXX/pseudo_label/object_mask/FFFF.png``, read with the same
palette-safe, NEAREST-resized reader the mask vote uses. DEVA ids are
PER-CAMERA -- id 95 is the dog on cam15 and id 125 is the dog on cam00 -- so the
ids come from ``--deva_ids_by_camera``, and where a camera is absent from that
mapping the id harmonised by ``--mask_consistency`` is used instead. cam00 is
held out and never harmonised, so cam00 must always be named explicitly.

``--alpha_guard_px P`` (default 0, off) additionally intersects ``S`` with the
object alpha dilated by ``P`` px. It guards against a DEVA leak -- a frame where
the segmenter's id spills onto something else -- at the cost of reintroducing a
dependence on the alpha it was introduced to escape, so it is off unless asked
for and its effect is reported per camera and frame.

In ``deva`` mode ``S`` replaces the alpha-derived mask everywhere the region is
used: ``construction_masks/`` is ``S`` undilated, ``visible_object/`` is ``S``
outside the absence window, and the composite weight is ``S`` dilated by
``--dilate`` and feathered by ``--feather``. The numbers change with it, and
BOTH are reported, under names that cannot be confused:

    * ``hole`` / ``object_px`` / ``hole_px`` / ``hole_fraction`` -- the OLD,
      alpha-based numbers, unchanged, still present in both modes.
    * ``silhouette`` -- ``silhouette_px``, the hole count and fraction ON ``S``,
      ``support_inside_silhouette`` (the MEAN ``alpha_bg`` over ``S``, which is
      the quantity that decides whether the model has anything to show once the
      object is removed), and ``alpha_cover_of_silhouette`` (the share of ``S``
      that SA4D's own rows cover). A low ``alpha_cover_of_silhouette`` is a
      disagreement between the segmenter and the selected rows, and it is
      reported rather than hidden precisely because the two need not agree.

The per-frame ``hole_fraction.json`` entry is a FLAT dict in ``alpha`` mode
(exactly as before) and a two-key ``{"alpha": ..., "silhouette": ...}`` dict in
``deva`` mode; ``edit_params.json`` records which, along with the mode, the ids
used per camera, and the ``S`` pixel count for every camera and frame.

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

# Fixed reporting thresholds for the object-only alpha footprint. Like the
# soft-threshold probes above these are NOT tunable: they exist so that every
# preview reports how sharp the rendered object mask is, and a knob would let
# that report be tuned.
ALPHA_REPORT_THRESHOLDS = (0.25, 0.5, 0.75)

# DEVA writes 0 for "no object here". It is excluded from the harmonisation
# argmax: if most projected rows land on unlabelled pixels the modal id would
# be 0 and "consistent" would come to mean "agrees about being background",
# which is the exact opposite of the intended test. The 0 count is still
# reported per camera.
DEVA_UNLABELLED_ID = 0

# cuda_rasterizer/forward.cu:149-150 and :151-153.
HOMOGENEOUS_W_EPS = 1e-7
NEAR_CLIP_Z = 0.2

# Median |analytic - points2d| above this many pixels aborts the run: it means
# the projection convention below does not match the rasterizer's, and every
# mask-consistency number would be fiction.
PROJECTION_CHECK_TOL_PX = 1.0


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


def edit_alpha_from_mask(mask, dilate=0, feather=0.0):
    """The composite's alpha from an ALREADY BINARY region: dilate, then feather.

    This is the second half of ``edit_alpha``, split out because in
    ``--edit_region deva`` the region arrives as a segmenter silhouette rather
    than as a soft alpha and must not be re-thresholded.
    """
    grown = dilate_binary(np.asarray(mask, dtype=bool), dilate)
    return gaussian_feather(grown.astype(np.float64), feather)


def edit_alpha(alpha_obj, threshold=0.5, dilate=0, feather=0.0):
    """The composite's alpha: binarise, dilate, feather. Returns float [0, 1]."""
    return edit_alpha_from_mask(binarise(alpha_obj, threshold), dilate, feather)


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


def centroid_velocity_detail(previous_centroid, centroid, frame_gap):
    """Signed per-frame centroid motion, or ``None`` without two endpoints and a gap."""
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


def centroid_velocity(previous_centroid, centroid, frame_gap):
    """Centroid speed in px/frame as a FLOAT, or ``None`` on the first frame.

    A scalar, because a consumer that plots "velocity" against frame must not
    have to know whether this field is a number or a dict. The signed
    components live in ``centroid_velocity_detail``.
    """
    detail = centroid_velocity_detail(previous_centroid, centroid, frame_gap)
    return None if detail is None else float(detail["speed_px_per_frame"])


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


def scale_distribution(max_scale):
    """``count / median / p90 / p99 / max`` of a per-row largest-scale vector."""
    values = np.asarray(max_scale, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"count": 0, "median": None, "p90": None, "p99": None, "max": None}
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90.0)),
        "p99": float(np.percentile(values, 99.0)),
        "max": float(values.max()),
    }


def scale_max_keep(max_scale, factor):
    """Keep rows with ``largest scale <= factor * median(largest scale)``.

    Returns ``(sub_keep, stats)`` where ``sub_keep`` is indexed WITHIN the base
    rows, and ``stats`` always carries the distribution -- including when
    ``factor is None`` and nothing is dropped, so the number that would have
    been used is on the record either way.
    """
    values = np.asarray(max_scale, dtype=np.float64).reshape(-1)
    stats = scale_distribution(values)
    stats["factor"] = None if factor is None else float(factor)
    stats["threshold"] = None
    if factor is None:
        stats["dropped"] = 0
        return np.ones(values.shape, dtype=bool), stats
    if float(factor) <= 0.0:
        raise ValueError("scale_max_factor must be positive, got %r" % (factor,))
    if values.size == 0:
        stats["dropped"] = 0
        return np.zeros(0, dtype=bool), stats
    threshold = float(factor) * float(stats["median"])
    stats["threshold"] = threshold
    keep = values <= threshold
    stats["dropped"] = int((~keep).sum())
    return keep, stats


def percentile_box(positions, percentile, pad=0.0):
    """Per-axis ``[P, 100-P]`` box of ``positions``, grown by ``pad * extent`` per side."""
    points = np.asarray(positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("percentile_box expects [M, 3], got %r" % (points.shape,))
    if points.shape[0] == 0:
        raise ValueError("percentile_box over zero points")
    value = float(percentile)
    if not 0.0 <= value < 50.0:
        raise ValueError("box_percentile must be in [0, 50), got %r" % (percentile,))
    low = np.percentile(points, value, axis=0)
    high = np.percentile(points, 100.0 - value, axis=0)
    grow = float(pad) * (high - low)
    return low - grow, high + grow


def inside_box(positions, low, high):
    """Inclusive per-axis membership test against a box."""
    points = np.asarray(positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("inside_box expects [M, 3], got %r" % (points.shape,))
    low_a = np.asarray(low, dtype=np.float64)
    high_a = np.asarray(high, dtype=np.float64)
    return np.all((points >= low_a[None, :]) & (points <= high_a[None, :]), axis=1)


def project_points(points_xyz, full_proj_transform, world_view_transform, width, height):
    """World -> pixel with the RASTERIZER's convention. Returns ``(xy[M, 2], valid[M])``.

    See the module docstring for the file:line provenance of each step.
    ``valid`` means "in front of the near clip and finite"; it does NOT mean
    "inside the image" -- ``sample_id_map`` applies the bounds test, so the two
    reasons a row can be unusable stay separable.
    """
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("project_points expects [M, 3], got %r" % (points.shape,))
    proj = np.asarray(full_proj_transform, dtype=np.float64)
    if proj.shape != (4, 4):
        raise ValueError("full_proj_transform must be 4x4, got %r" % (proj.shape,))
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    clip = homogeneous @ proj
    weight = clip[:, 3] + HOMOGENEOUS_W_EPS
    with np.errstate(divide="ignore", invalid="ignore"):
        ndc_x = clip[:, 0] / weight
        ndc_y = clip[:, 1] / weight
    x = ((ndc_x + 1.0) * float(width) - 1.0) * 0.5
    y = ((ndc_y + 1.0) * float(height) - 1.0) * 0.5
    if world_view_transform is None:
        in_front = np.ones(points.shape[0], dtype=bool)
    else:
        view = np.asarray(world_view_transform, dtype=np.float64)
        if view.shape != (4, 4):
            raise ValueError("world_view_transform must be 4x4, got %r" % (view.shape,))
        in_front = (homogeneous @ view[:, 2]) > NEAR_CLIP_Z
    valid = in_front & np.isfinite(x) & np.isfinite(y)
    return np.stack([x, y], axis=1), valid


def sample_id_map(id_map, xy, valid):
    """Nearest-pixel id lookup. Returns ``(ids[M], hit[M])``.

    ``hit`` is ``valid`` further restricted to projections that land inside the
    image; ``ids`` is 0 wherever ``hit`` is False and must not be read there.
    """
    ids = np.asarray(id_map)
    if ids.ndim != 2:
        raise ValueError("sample_id_map expects a 2-D id map, got %r" % (ids.shape,))
    points = np.asarray(xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("sample_id_map expects [M, 2] coordinates, got %r" % (points.shape,))
    height, width = ids.shape
    finite = np.isfinite(points).all(axis=1)
    safe = np.where(finite[:, None], points, -1.0)
    column = np.rint(safe[:, 0]).astype(np.int64)
    row = np.rint(safe[:, 1]).astype(np.int64)
    hit = (np.asarray(valid, dtype=bool) & finite
           & (column >= 0) & (column < width) & (row >= 0) & (row < height))
    out = np.zeros(points.shape[0], dtype=np.int64)
    out[hit] = ids[row[hit], column[hit]].astype(np.int64)
    return out, hit


def harmonise_id(sampled_ids, hit, ignore_id=DEVA_UNLABELLED_ID):
    """Modal non-ignored id among the rows that landed in the image.

    DEVA ids are per-camera, so the id the object carries in the reference
    camera means nothing in any other camera; the id is recovered from where
    the projected rows actually land. Ties go to the SMALLEST id so the choice
    is deterministic. Returns ``(chosen_id_or_None, {id: count})`` with the
    ignored id present in the counts.
    """
    ids = np.asarray(sampled_ids, dtype=np.int64)
    mask = np.asarray(hit, dtype=bool)
    if ids.shape != mask.shape:
        raise ValueError("harmonise_id shape mismatch: %r vs %r" % (ids.shape, mask.shape))
    counts = {}
    if mask.any():
        values, tally = np.unique(ids[mask], return_counts=True)
        counts = {int(value): int(count) for value, count in zip(values, tally)}
    ranked = sorted(
        ((count, -identifier) for identifier, count in counts.items()
         if identifier != int(ignore_id)),
        reverse=True,
    )
    chosen = None if not ranked else int(-ranked[0][1])
    return chosen, counts


def consistency_counts(per_camera_hits):
    """``[C, M]`` bool -> per-row count of cameras that agreed."""
    arr = np.asarray(per_camera_hits, dtype=bool)
    if arr.ndim != 2:
        raise ValueError("consistency_counts expects [C, M], got %r" % (arr.shape,))
    return arr.sum(axis=0).astype(np.int64)


def consistency_keep(counts, min_cameras):
    """Rows agreed on by at least ``min_cameras`` cameras."""
    return np.asarray(counts, dtype=np.int64) >= int(min_cameras)


def count_histogram(counts):
    """``{"<count>": rows}`` -- JSON-safe, so the vote's shape is auditable."""
    values, tally = np.unique(np.asarray(counts, dtype=np.int64), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, tally)}


def frame_survivors(per_frame_keep, min_fraction):
    """``[F, N]`` bool -> rows surviving at least ``min_fraction`` of the anchor frames."""
    arr = np.asarray(per_frame_keep, dtype=bool)
    if arr.ndim != 2:
        raise ValueError("frame_survivors expects [F, N], got %r" % (arr.shape,))
    frames = arr.shape[0]
    if frames == 0:
        raise ValueError("frame_survivors over zero anchor frames")
    fraction = float(min_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("mask_min_frames must be in (0, 1], got %r" % (min_fraction,))
    # 1e-12 so that fraction == 1.0 is not defeated by float division.
    return (arr.sum(axis=0) / float(frames)) >= fraction - 1e-12


def alpha_threshold_counts(alpha, thresholds=ALPHA_REPORT_THRESHOLDS):
    """``{"0.25": px, "0.5": px, "0.75": px}`` for one alpha map."""
    arr = np.asarray(alpha, dtype=np.float64)
    return {("%g" % float(t)): int((arr > float(t)).sum()) for t in thresholds}


def mask_fit(alpha_obj, id_map, target_id, threshold=0.5):
    """How well the rendered object mask agrees with the segmenter's mask.

    ``inside_fraction`` is the share of object-mask pixels that land on the
    camera's harmonised DEVA id -- the number asked for. ``recall_of_deva_id``
    is its denominator's mirror image and is reported alongside because on its
    own ``inside_fraction`` cannot tell a small precise blob from a mask that
    actually covers the object.
    """
    obj = binarise(alpha_obj, threshold)
    ids = np.asarray(id_map)
    if ids.shape != obj.shape:
        raise ValueError("mask_fit shape mismatch: %r vs %r" % (ids.shape, obj.shape))
    object_px = int(obj.sum())
    if target_id is None:
        return {"object_px": object_px, "deva_id": None, "deva_id_px": None,
                "inside_px": None, "inside_fraction": None, "recall_of_deva_id": None}
    target = int(target_id)
    deva_px = int((ids == target).sum())
    if object_px == 0:
        return {"object_px": 0, "deva_id": target, "deva_id_px": deva_px,
                "inside_px": 0, "inside_fraction": None, "recall_of_deva_id": None}
    inside = int((ids[obj] == target).sum())
    return {
        "object_px": object_px,
        "deva_id": target,
        "deva_id_px": deva_px,
        "inside_px": inside,
        "inside_fraction": inside / object_px,
        "recall_of_deva_id": (inside / deva_px) if deva_px else None,
    }


def parse_ids_by_camera(text):
    """Parse ``--deva_ids_by_camera``: ``{"cam00": [125], "cam15": [95]}``.

    Accepts either the JSON text itself or a path to a file containing it -- the
    inline form is unquotable in some shells and a mapping that silently parsed
    as a filename, or the reverse, would pick the wrong ids without saying so.
    Values may be a single id or a list of ids. ``0`` is refused: it is DEVA's
    "no object here" label (:data:`DEVA_UNLABELLED_ID`), so a silhouette
    containing it would be most of the frame.
    """
    raw = str(text)
    if os.path.isfile(raw):
        with open(raw, "r", encoding="utf-8") as handle:
            raw = handle.read()
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise ValueError("--deva_ids_by_camera is neither a readable file nor valid "
                         "JSON: %s" % (exc,))
    if not isinstance(parsed, dict):
        raise ValueError("--deva_ids_by_camera must be a JSON object mapping "
                         '"camXX" -> [ids], got %r' % (type(parsed).__name__,))
    out = {}
    for camera_name, value in parsed.items():
        if not str(camera_name).strip():
            raise ValueError("--deva_ids_by_camera has an empty camera name")
        entries = value if isinstance(value, (list, tuple)) else [value]
        if not entries:
            raise ValueError("--deva_ids_by_camera gives camera %r no ids" % (camera_name,))
        ids = []
        for entry in entries:
            if isinstance(entry, bool) or not isinstance(entry, int):
                raise ValueError("--deva_ids_by_camera id %r for camera %r is not an integer"
                                 % (entry, camera_name))
            if entry == DEVA_UNLABELLED_ID:
                raise ValueError(
                    "--deva_ids_by_camera lists id %d for camera %r, which is DEVA's "
                    "UNLABELLED label; its silhouette would be the background"
                    % (DEVA_UNLABELLED_ID, camera_name))
            ids.append(int(entry))
        out[str(camera_name)] = sorted(set(ids))
    return out


def resolve_deva_ids(ids_by_camera, camera_name, harmonised_by_camera=None):
    """The DEVA ids to use for one camera: explicit mapping first, harmonised second.

    Raises ``ValueError`` when neither source supplies an id -- a camera whose
    silhouette cannot be built must stop the run, not contribute an empty mask
    that would read as "the object is not visible here".
    """
    explicit = (ids_by_camera or {}).get(camera_name)
    if explicit:
        return list(explicit)
    harmonised = (harmonised_by_camera or {}).get(camera_name)
    if harmonised is not None:
        return [int(harmonised)]
    raise ValueError(
        "no DEVA id for camera %s: name it in --deva_ids_by_camera, or let "
        "--mask_consistency harmonise it (cam00 is held out and is never "
        "harmonised, so cam00 must always be named explicitly)" % (camera_name,))


def silhouette_from_id_map(id_map, ids):
    """Union of the listed ids' pixels in a DEVA id map, as a bool mask."""
    values = [int(value) for value in ids]
    if not values:
        raise ValueError("silhouette_from_id_map needs at least one id")
    if any(value == DEVA_UNLABELLED_ID for value in values):
        raise ValueError("id %d is DEVA's UNLABELLED label and cannot form a silhouette"
                         % (DEVA_UNLABELLED_ID,))
    array = np.asarray(id_map)
    if array.ndim != 2:
        raise ValueError("silhouette_from_id_map expects a 2-D id map, got %r" % (array.shape,))
    out = np.zeros(array.shape, dtype=bool)
    for value in values:
        out |= (array == value)
    return out


def guard_silhouette(silhouette, alpha_obj, guard_px, threshold=0.5):
    """Optionally intersect ``S`` with the object alpha dilated by ``guard_px``.

    ``guard_px <= 0`` is OFF and returns ``S`` untouched -- note that 0 means
    "no intersection at all", not "intersect with the undilated alpha", because
    an unrequested intersection would silently reintroduce the very alpha the
    silhouette exists to replace. Returns ``(mask, stats)``; ``stats`` always
    records whether the guard ran and what it cost.
    """
    sil = np.asarray(silhouette, dtype=bool)
    radius = int(guard_px)
    stats = {"enabled": radius > 0, "guard_px": radius, "alpha_threshold": float(threshold),
             "silhouette_px": int(sil.sum()), "guard_px_count": None, "kept_px": int(sil.sum()),
             "dropped_px": 0}
    if radius <= 0:
        return sil.copy(), stats
    alpha = np.asarray(alpha_obj, dtype=np.float64)
    if alpha.shape != sil.shape:
        raise ValueError("guard_silhouette shape mismatch: %r vs %r" % (alpha.shape, sil.shape))
    guard = dilate_binary(binarise(alpha, threshold), radius)
    kept = sil & guard
    stats["guard_px_count"] = int(guard.sum())
    stats["kept_px"] = int(kept.sum())
    stats["dropped_px"] = int(sil.sum()) - int(kept.sum())
    return kept, stats


def silhouette_report(silhouette, alpha_obj, alpha_bg, support_threshold=0.5,
                      alpha_threshold=0.5):
    """Support and coverage ON the silhouette -- the numbers the edit turns on.

    ``support_inside_silhouette`` is the MEAN ``alpha_bg`` over ``S``: what the
    model actually has to show once the object is removed.
    ``alpha_cover_of_silhouette`` is the share of ``S`` that SA4D's own selected
    rows cover, so a disagreement between the segmenter and the row set is
    visible rather than absorbed. Every fraction is ``None`` on an empty ``S``.
    """
    sil = np.asarray(silhouette, dtype=bool)
    count = int(sil.sum())
    support = np.asarray(alpha_bg, dtype=np.float64)
    alpha = np.asarray(alpha_obj, dtype=np.float64)
    if support.shape != sil.shape or alpha.shape != sil.shape:
        raise ValueError("silhouette_report shape mismatch: S %r, alpha_obj %r, alpha_bg %r"
                         % (sil.shape, alpha.shape, support.shape))
    if count == 0:
        return {"silhouette_px": 0, "hole_px": 0, "hole_fraction": None,
                "support_inside_silhouette": None, "alpha_cover_of_silhouette": None}
    holes = int((support[sil] < float(support_threshold)).sum())
    return {
        "silhouette_px": count,
        "hole_px": holes,
        "hole_fraction": holes / count,
        "support_inside_silhouette": float(support[sil].mean()),
        "alpha_cover_of_silhouette": float((alpha[sil] > float(alpha_threshold)).mean()),
    }


def mask_outline(mask, width=1):
    """The INNER boundary of a mask: pixels of the mask adjacent to its outside.

    Inner, not outer, so the outline drawn on a preview marks the silhouette's
    own extent and never claims a pixel the segmenter did not.
    """
    src = np.asarray(mask, dtype=bool)
    thickness = max(1, int(width))
    return src & dilate_binary(~src, thickness)


def overlay_outline(image_rgb, outline, colour=(1.0, 0.0, 0.0)):
    """Draw a boolean outline over an HWC float image. Returns HWC uint8."""
    image = np.asarray(image_rgb, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("overlay_outline expects HWC RGB, got %r" % (image.shape,))
    edge = np.asarray(outline, dtype=bool)
    if edge.shape != image.shape[:2]:
        raise ValueError("outline shape %r does not match image %r" % (edge.shape, image.shape))
    out = image.copy()
    out[edge] = np.asarray(colour, dtype=np.float64)
    return to_uint8_map(out)


def modal_value(values):
    """Most common entry of a list, ties to the smallest; ``None`` if all are ``None``."""
    present = [int(value) for value in values if value is not None]
    if not present:
        return None
    unique, tally = np.unique(np.asarray(present, dtype=np.int64), return_counts=True)
    ranked = sorted(((int(count), -int(value)) for value, count in zip(unique, tally)),
                    reverse=True)
    return int(-ranked[0][1])


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


def _numpy_matrix(tensor):
    """A torch 4x4 camera matrix as a float64 numpy array."""
    return np.asarray(tensor.detach().cpu().numpy(), dtype=np.float64)


def _load_deva_ids(deva_root, camera_name, frame, height, width, cache=None):
    """``<root>/camXX/pseudo_label/object_mask/FFFF.png`` as a 2-D id array.

    DEVA writes SHORT IDS, one uint8 per pixel, per camera. A palette ("P")
    PNG carries those ids as palette INDICES, so it is read with
    ``np.asarray`` and never through ``convert("L")``, which would map them
    through the palette and destroy them. An RGB file is a colourised
    visualisation, not an id map, and is refused rather than silently
    reinterpreted. Returns ``(ids, was_resized)``.
    """
    from PIL import Image

    # The root is part of the key: --mask_consistency and --deva_root may point
    # at DIFFERENT id-map trees in the same run, and a root-blind cache would
    # serve one root's map for the other's request without saying so.
    key = (str(deva_root), camera_name, int(frame), int(height), int(width))
    if cache is not None and key in cache:
        return cache[key]
    path = os.path.join(deva_root, camera_name, "pseudo_label", "object_mask",
                        "%04d.png" % int(frame))
    if not os.path.isfile(path):
        raise SystemExit("DEVA id map not found: %s" % (path,))
    image = Image.open(path)
    if image.mode not in ("L", "P", "I", "I;16"):
        raise SystemExit(
            "DEVA id map %s has mode %s; expected a single-channel id map. An RGB "
            "file is a colourised visualisation and its ids cannot be recovered."
            % (path, image.mode))
    was_resized = False
    if image.size != (int(width), int(height)):
        # NEAREST, always: any interpolation between two ids invents a third.
        image = image.resize((int(width), int(height)), Image.NEAREST)
        was_resized = True
    # Kept at the file's own dtype (uint8 for a DEVA short-id map). Upcasting to
    # int64 here would make the cache eight times larger for no gain: every
    # consumer either indexes it or compares it against a small integer.
    result = (np.asarray(image), was_resized)
    if cache is not None:
        cache[key] = result
    return result


def _resolve_ids_for_cameras(args, camera_names, harmonised, known_cameras=None):
    """``{camera: [deva ids]}`` for every camera that will be rendered.

    Resolved ONCE, up front, before any rendering: a camera with no id is a
    stop, and finding that out after a full rig build would waste the build.
    ``known_cameras`` is every camera in the DATASET, not only the ones being
    rendered, so that one mapping serves a preview of two cameras and a build of
    the whole rig -- while a name that is in neither is still a typo and stops
    the run rather than falling through to a harmonised id for some other camera.
    """
    mapping = parse_ids_by_camera(args.deva_ids_by_camera) if args.deva_ids_by_camera else {}
    unknown = sorted(set(mapping) - set(known_cameras if known_cameras is not None else camera_names))
    if unknown:
        raise SystemExit("--deva_ids_by_camera names cameras that are not in the dataset: %s"
                         % (", ".join(unknown),))
    resolved = {}
    for name in camera_names:
        try:
            resolved[name] = resolve_deva_ids(mapping, name, harmonised)
        except ValueError as exc:
            raise SystemExit("--edit_region deva: %s" % (exc,))
    return mapping, resolved


def _silhouette_for_view(args, camera_name, frame, height, width, alpha_obj, ids_for_camera,
                         cache=None):
    """The segmenter's silhouette ``S`` for one (camera, frame), optionally guarded."""
    id_map, was_resized = _load_deva_ids(args.deva_root, camera_name, frame, height, width, cache)
    raw = silhouette_from_id_map(id_map, ids_for_camera)
    mask, guard = guard_silhouette(raw, alpha_obj, args.alpha_guard_px)
    info = {
        "deva_ids": list(ids_for_camera),
        "deva_silhouette_px": int(mask.sum()),
        "deva_silhouette_px_before_guard": int(raw.sum()),
        "deva_resized": bool(was_resized),
        "alpha_guard": guard,
    }
    return mask, info


def _geometry_filters(gaussians, selection, time_value, args):
    """Narrowing steps 2 (scale) and 3 (canonical percentile box).

    Both are skipped entirely -- no deformation call, no GPU work -- when
    neither flag is set, so the default path is byte-for-byte the old one.
    """
    import torch

    base = int(selection.sum().item())
    report = {"base": base, "after_scale": base, "after_box": base,
              "scale": None, "box": None}
    if base == 0 or (args.scale_max_factor is None and args.box_percentile is None):
        return selection, report

    keep = selection.bool()
    state = _deformed_rows(gaussians, keep, time_value)
    # Only the three SPATIAL axes: a fourth column, where a model carries one,
    # is a temporal extent and is not a screen footprint.
    scales = np.asarray(state["scales"], dtype=np.float64)[:, :3]
    sub_scale, scale_stats = scale_max_keep(scales.max(axis=1), args.scale_max_factor)
    scale_stats["scale_columns_used"] = int(scales.shape[1])
    report["scale"] = scale_stats
    report["after_scale"] = int(sub_scale.sum())

    canonical = gaussians.get_xyz[keep].detach().float().cpu().numpy().astype(np.float64)
    if args.box_percentile is None:
        sub = sub_scale
    else:
        low, high = percentile_box(canonical, args.box_percentile, args.box_pad)
        sub_box = inside_box(canonical, low, high)
        report["box"] = {
            "percentile": float(args.box_percentile),
            "pad": float(args.box_pad),
            "low": [float(value) for value in low],
            "high": [float(value) for value in high],
            "dropped_from_base": int((~sub_box).sum()),
        }
        sub = sub_scale & sub_box
    report["after_box"] = int(sub.sum())
    filtered = apply_subset_filter(keep, torch.from_numpy(sub).to(keep.device))
    return filtered, report


def _selection_bundle(gaussians, ids_tensor, thresholds, args, frame):
    """``(hard, soft, filtered_after_steps_1_to_3, geometry_report)`` at one frame."""
    time_value = frame / float(TIME_DIVISOR)
    hard, soft = _row_sets(gaussians, time_value, ids_tensor, thresholds)
    base = hard.bool() if args.argmax_only else (hard | soft[float(args.soft_thresh)]).bool()
    filtered, report = _geometry_filters(gaussians, base, time_value, args)
    return hard, soft, filtered, report


def _projection_check(view, camera_name, frame, gaussians, pipe, background, keep_mask,
                      row_index, n_rows, expected_xy):
    """PRECONDITION for step 4: does ``project_points`` match the rasterizer?

    A wrong convention makes every row inconsistent and returns a small, clean,
    entirely fictitious row set -- a favourable-looking answer an instrument
    could not have failed to give. So the analytic projection is compared
    against ``render_segmentation``'s own ``points2d`` and a disagreement above
    ``PROJECTION_CHECK_TOL_PX`` raises. When ``points2d`` cannot be used the
    reason is RECORDED and ``ran`` stays False, so an unexercised precondition
    is never read as a passed one.
    """
    from gaussian_renderer import render_segmentation

    report = {
        "ran": False,
        "reason": None,
        "camera": camera_name,
        "frame": int(frame),
        "tolerance_px": PROJECTION_CHECK_TOL_PX,
        "compared_rows": 0,
        "median_px": None,
        "p95_px": None,
        "max_px": None,
    }
    try:
        rendered = render_segmentation(view, gaussians, pipe, background, keep_mask.bool())
        points2d = rendered.get("points2d", None) if hasattr(rendered, "get") else None
        if points2d is None:
            report["reason"] = "render_segmentation returned no usable 'points2d'"
            return report
        array = np.asarray(points2d.detach().float().cpu().numpy(), dtype=np.float64)
    except Exception as exc:
        # Diagnostic-only limb: the exception is REPORTED, never swallowed, and
        # the run continues on the analytic projection with `ran` False.
        report["reason"] = "points2d unavailable: %r" % (exc,)
        return report

    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] < 2:
        report["reason"] = "points2d has unexpected shape %r" % (array.shape,)
        return report
    if array.shape[0] == expected_xy.shape[0]:
        observed = array[:, :2]
    elif array.shape[0] == n_rows:
        observed = array[row_index, :2]
    else:
        report["reason"] = ("points2d has %d rows, expected %d (kept) or %d (all)"
                            % (array.shape[0], expected_xy.shape[0], n_rows))
        return report

    magnitude = np.abs(observed[np.isfinite(observed).all(axis=1)])
    if magnitude.size == 0:
        report["reason"] = "points2d is entirely non-finite"
        return report
    if float(np.percentile(magnitude, 90.0)) <= 2.0:
        # Values confined to about [-1, 1] are NDC, not pixels; comparing them
        # against pixel coordinates would abort a perfectly good run.
        report["reason"] = ("points2d is not in pixel units (p90 |value| = %.4f); "
                            "check skipped" % (float(np.percentile(magnitude, 90.0)),))
        return report

    distance = np.linalg.norm(observed - expected_xy, axis=1)
    finite = distance[np.isfinite(distance)]
    if finite.size == 0:
        report["reason"] = "no finite row pairs to compare"
        return report
    report.update({
        "ran": True,
        "compared_rows": int(finite.size),
        "median_px": float(np.median(finite)),
        "p95_px": float(np.percentile(finite, 95.0)),
        "max_px": float(finite.max()),
    })
    if report["median_px"] > PROJECTION_CHECK_TOL_PX:
        raise SystemExit(
            "projection precondition FAILED: median |analytic - points2d| = %.4f px "
            "> %.4f px tolerance over %d rows. The mask-consistency vote would be "
            "meaningless; refusing to run it."
            % (report["median_px"], PROJECTION_CHECK_TOL_PX, report["compared_rows"]))
    return report


def _mask_consistency(gaussians, pipe, args, index, selection_at, background, deva_cache):
    """Narrowing step 4. Returns ``(keep[N] bool numpy, report)``."""
    anchor_frames = parse_frame_list(args.mask_frames)
    cameras = [name for name in sorted(index) if name != "cam00"]
    if not cameras:
        raise SystemExit("--mask_consistency found no training cameras in the dataset")
    min_cameras = int(args.mask_min_cams)
    if min_cameras < 1:
        raise SystemExit("--mask_min_cams must be at least 1")
    if min_cameras > len(cameras):
        raise SystemExit("--mask_min_cams %d exceeds the %d training cameras available"
                         % (min_cameras, len(cameras)))

    n_rows = int(gaussians.get_xyz.shape[0])
    table = np.zeros((len(anchor_frames), n_rows), dtype=bool)
    chosen_by_camera = {name: [] for name in cameras}
    per_frame = []
    resized = []
    projection_check = {"ran": False, "reason": "no non-empty anchor frame to check on",
                        "tolerance_px": PROJECTION_CHECK_TOL_PX}
    checked = False

    for position, frame in enumerate(anchor_frames):
        selection, _ = selection_at(frame)
        base_rows = int(selection.sum().item())
        if base_rows == 0:
            per_frame.append({"frame": frame, "base_rows": 0, "survivors": 0,
                              "consistency_histogram": {}, "cameras": []})
            continue
        state = _deformed_rows(gaussians, selection.bool(), frame / float(TIME_DIVISOR))
        row_index = np.asarray(state["row_index"], dtype=np.int64)
        xyz = np.asarray(state["xyz"], dtype=np.float64)

        hits = np.zeros((len(cameras), row_index.size), dtype=bool)
        camera_records = []
        for slot, camera_name in enumerate(cameras):
            view = _get_view(index, camera_name, frame)
            height, width = int(view.image_height), int(view.image_width)
            id_map, was_resized = _load_deva_ids(
                args.mask_consistency, camera_name, frame, height, width, deva_cache)
            if was_resized:
                resized.append("%s/%04d" % (camera_name, frame))
            xy, valid = project_points(
                xyz,
                _numpy_matrix(view.full_proj_transform),
                _numpy_matrix(view.world_view_transform),
                width,
                height,
            )
            if not checked:
                checked = True
                projection_check = _projection_check(
                    view, camera_name, frame, gaussians, pipe, background, selection,
                    row_index, n_rows, xy)
            ids_at, hit = sample_id_map(id_map, xy, valid)
            chosen, counts = harmonise_id(ids_at, hit)
            chosen_by_camera[camera_name].append(chosen)
            if chosen is not None:
                hits[slot] = hit & (ids_at == chosen)
            camera_records.append({
                "camera": camera_name,
                "chosen_id": chosen,
                "in_front_of_camera": int(valid.sum()),
                "landed_in_image": int(hit.sum()),
                "unlabelled_rows": int(counts.get(DEVA_UNLABELLED_ID, 0)),
                "consistent_rows": int(hits[slot].sum()),
                "top_ids": dict(sorted(counts.items(),
                                       key=lambda item: (-item[1], item[0]))[:8]),
                "deva_resized": bool(was_resized),
            })

        counts_per_row = consistency_counts(hits)
        keep_rows = consistency_keep(counts_per_row, min_cameras)
        table[position, row_index[keep_rows]] = True
        per_frame.append({
            "frame": frame,
            "base_rows": base_rows,
            "survivors": int(keep_rows.sum()),
            "consistency_histogram": count_histogram(counts_per_row),
            "cameras": camera_records,
        })

    # Precondition, not a reading rule: a vote that never had a row to look at
    # would return a clean empty mask and say nothing about anything.
    if not any(record["base_rows"] for record in per_frame):
        raise SystemExit(
            "--mask_consistency: every anchor frame (%s) had an EMPTY base row set, so "
            "the vote was never exercised" % (args.mask_frames,))

    survivors = frame_survivors(table, args.mask_min_frames)
    report = {
        "enabled": True,
        "deva_root": args.mask_consistency,
        "min_cameras": min_cameras,
        "min_frames_fraction": float(args.mask_min_frames),
        "anchor_frames": anchor_frames,
        "cameras": cameras,
        "harmonised_id_by_camera": {name: modal_value(values)
                                    for name, values in chosen_by_camera.items()},
        "per_frame": per_frame,
        "survivors": int(survivors.sum()),
        "deva_maps_resized": sorted(set(resized)),
        "projection_check": projection_check,
    }
    return survivors, report


def _filter_config(args):
    """The narrowing knobs, exactly as given, for the output JSON."""
    return {
        "argmax_only": bool(args.argmax_only),
        "scale_max_factor": None if args.scale_max_factor is None else float(args.scale_max_factor),
        "box_percentile": None if args.box_percentile is None else float(args.box_percentile),
        "box_pad": float(args.box_pad),
        "mask_consistency_root": args.mask_consistency,
        "mask_min_cams": None if args.mask_consistency is None else int(args.mask_min_cams),
        "mask_frames": args.mask_frames,
        "mask_min_frames": float(args.mask_min_frames),
        "order": ["argmax_only", "scale_max_factor", "box_percentile", "mask_consistency"],
    }


def _edit_region_config(args, ids_mapping, ids_by_camera, silhouette_px):
    """What defined the edit region, and -- in ``deva`` mode -- what it measured."""
    config = {
        "mode": args.edit_region,
        "deva_root": args.deva_root,
        "alpha_guard_px": int(args.alpha_guard_px),
        "dilate_px": int(args.dilate),
        "feather_sigma_px": float(args.feather),
        "object_alpha_threshold": 0.5,
    }
    if args.edit_region != "deva":
        return config
    config.update({
        "ids_explicit": {name: list(value) for name, value in sorted(ids_mapping.items())},
        "ids_by_camera": {name: list(value) for name, value in sorted(ids_by_camera.items())},
        "ids_source": {
            name: ("explicit" if name in ids_mapping else "mask_consistency_harmonised")
            for name in sorted(ids_by_camera)
        },
        "silhouette_px": {name: dict(sorted(frames.items()))
                          for name, frames in sorted(silhouette_px.items())},
    })
    return config


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
    deva_cache = {}
    filter_counts = []

    with torch.no_grad():
        # ---- step 4, once, frozen, and intersected into every timestamp -----
        if args.mask_consistency:
            consistency_keep_np, consistency_report = _mask_consistency(
                gaussians,
                pipe,
                args,
                index,
                lambda frame: _selection_bundle(gaussians, ids_tensor, thresholds, args, frame)[2:],
                black,
                deva_cache,
            )
            harmonised = consistency_report["harmonised_id_by_camera"]
        else:
            consistency_keep_np = None
            consistency_report = {"enabled": False}
            harmonised = {}

        # ---- the edit region, resolved before any rendering -----------------
        if args.edit_region == "deva":
            ids_mapping, ids_by_camera = _resolve_ids_for_cameras(
                args, cameras, harmonised, sorted(index))
        else:
            ids_mapping, ids_by_camera = {}, {}
        silhouette_px = {}

        for frame in frames:
            time_value = frame / float(TIME_DIVISOR)
            hard, soft, filtered, geometry = _selection_bundle(
                gaussians, ids_tensor, thresholds, args, frame)
            if consistency_keep_np is None:
                selection = filtered
                geometry["after_mask_consistency"] = geometry["after_box"]
            else:
                selection = filtered & torch.from_numpy(consistency_keep_np).to(filtered.device)
                geometry["after_mask_consistency"] = int(selection.sum().item())
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
                "filters": geometry,
            })
            filter_counts.append({
                "frame": frame,
                "base": geometry["base"],
                "after_scale": geometry["after_scale"],
                "after_box": geometry["after_box"],
                "after_mask_consistency": geometry["after_mask_consistency"],
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

                # ---- the edit region for this view ----------------------
                if args.edit_region == "deva":
                    region, region_info = _silhouette_for_view(
                        args, camera_name, frame, height, width, alpha_obj,
                        ids_by_camera[camera_name], deva_cache)
                    silhouette_px.setdefault(camera_name, {})["%04d" % frame] = \
                        region_info["deva_silhouette_px"]
                    silhouette = silhouette_report(region, alpha_obj, alpha_bg)
                else:
                    region, region_info, silhouette = None, None, None

                # The black-background render IS the premultiplied colour C, so
                # `real * (1 - a) + C` is a correct over-composite.
                _save_png(os.path.join(args.out, "preview", stem + "_obj.png"),
                          to_uint8_map(real * (1.0 - alpha_obj[..., None]) + _chw_to_hwc_float(obj_b)))
                _save_png(os.path.join(args.out, "preview", stem + "_bg.png"),
                          to_uint8_map(real * (1.0 - alpha_bg[..., None]) + _chw_to_hwc_float(bg_b)))
                _save_png(os.path.join(args.out, "alpha_obj", stem + ".png"), to_uint8_map(alpha_obj))
                _save_png(os.path.join(args.out, "support", stem + ".png"), to_uint8_map(alpha_bg))

                if region is not None:
                    # The actual composite, so the edit can be judged BY EYE and
                    # not only through a table of fractions: real outside `a`,
                    # the object-free render inside it.
                    weight = edit_alpha_from_mask(region, args.dilate, args.feather)
                    _save_png(os.path.join(args.out, "preview", stem + "_edit.png"),
                              composite_alpha(real, _chw_to_hwc_float(bg_b), weight))
                    _save_png(os.path.join(args.out, "preview", stem + "_sil.png"),
                              overlay_outline(real, mask_outline(region)))

                shape = mask_bbox_centroid(binarise(alpha_obj, 0.5))
                previous_frame, previous_centroid = previous_centroids[camera_name]
                gap = 0 if previous_frame is None else frame - previous_frame
                velocity = centroid_velocity(previous_centroid, shape["centroid"], gap)
                velocity_detail = centroid_velocity_detail(
                    previous_centroid, shape["centroid"], gap)
                previous_centroids[camera_name] = (frame, shape["centroid"])

                # How well does the rendered object mask fit the segmenter's?
                # cam00 is the held-out camera and has no training-time DEVA
                # harmonisation, so it is skipped by design.
                if args.mask_consistency and camera_name != "cam00":
                    id_map, _ = _load_deva_ids(args.mask_consistency, camera_name, frame,
                                               height, width, deva_cache)
                    fit = mask_fit(alpha_obj, id_map, harmonised.get(camera_name))
                else:
                    fit = None

                record = {
                    "camera": camera_name,
                    "frame": frame,
                    "time": float(view.time),
                    "object_mask_px": shape["count"],
                    "object_alpha_px": alpha_threshold_counts(alpha_obj),
                    "bbox": shape["bbox"],
                    "centroid": shape["centroid"],
                    "centroid_velocity": velocity,
                    "centroid_velocity_detail": velocity_detail,
                    "deva_mask_fit": fit,
                    # `hole` is the OLD, alpha-based number, in both modes; the
                    # silhouette numbers live under `silhouette` and never
                    # overwrite it.
                    "hole": hole_fraction(alpha_obj, alpha_bg),
                }
                if region is not None:
                    record["silhouette"] = dict(region_info, **silhouette)
                per_view.append(record)

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
        "filters": dict(_filter_config(args),
                        per_timestamp_counts=filter_counts,
                        mask_consistency=consistency_report),
        "edit_region": _edit_region_config(args, ids_mapping, ids_by_camera, silhouette_px),
        "alpha_report_thresholds": list(ALPHA_REPORT_THRESHOLDS),
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
    deva_cache = {}
    filter_counts = []

    with torch.no_grad():
        # ---- 0. step 4, once, frozen (it needs its own anchor frames) -------
        if args.mask_consistency:
            consistency_keep_np, consistency_report = _mask_consistency(
                gaussians,
                pipe,
                args,
                index,
                lambda frame: _selection_bundle(gaussians, ids_tensor, thresholds, args, frame)[2:],
                black,
                deva_cache,
            )
            harmonised = consistency_report["harmonised_id_by_camera"]
        else:
            consistency_keep_np = None
            consistency_report = {"enabled": False}
            harmonised = {}

        # ---- 0b. the edit region, resolved before any rendering -------------
        if args.edit_region == "deva":
            ids_mapping, ids_by_camera = _resolve_ids_for_cameras(args, camera_names, harmonised)
        else:
            ids_mapping, ids_by_camera = {}, {}
        silhouette_px = {}

        # ---- 1. per-timestamp selections over [A-margin, B+margin] ----------
        for position, frame in enumerate(render_frames):
            time_value = frame / float(TIME_DIVISOR)
            hard, soft, filtered, geometry = _selection_bundle(
                gaussians, ids_tensor, thresholds, args, frame)
            selection = filtered.detach().cpu().numpy().astype(bool)
            table[position] = selection
            # Step 4 is NOT recorded per timestamp here: in build mode it is
            # applied once, to the canonical set, after the majority vote. Its
            # count is `canonical_counts.after_mask_consistency`.
            per_timestamp.append({
                "frame": frame,
                "time": time_value,
                "selected_rows": int(selection.sum()),
                "sensitivity": rowset_summary(
                    hard.detach().cpu().numpy(),
                    soft[SENSITIVITY_SOFT_LOW].detach().cpu().numpy(),
                    soft[SENSITIVITY_SOFT_HIGH].detach().cpu().numpy(),
                ),
                "filters": geometry,
            })
            filter_counts.append({
                "frame": frame,
                "base": geometry["base"],
                "after_scale": geometry["after_scale"],
                "after_box": geometry["after_box"],
            })

        # ---- 2. canonical set: majority, consistency, IQR box, radius -------
        canonical_np = majority_rowset(table)
        count_majority = int(canonical_np.sum())
        if count_majority == 0:
            raise SystemExit(
                "majority vote over frames %d-%d selected zero rows for ids %s"
                % (render_lo, render_hi, args.ids)
            )

        if consistency_keep_np is not None:
            canonical_np = canonical_np & consistency_keep_np
        count_after_consistency = int(canonical_np.sum())
        if count_after_consistency == 0:
            raise SystemExit("the DEVA mask-consistency vote emptied the canonical set")

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

                # The edit region: the object alpha, or -- in `deva` mode -- the
                # segmenter's silhouette for THIS camera at THIS frame.
                alpha_hole = hole_fraction(alpha_obj, alpha_bg)
                if args.edit_region == "deva":
                    # No id-map cache here: a full rig build touches every
                    # camera x every frame exactly once, and caching them all
                    # would hold the whole DEVA window in memory for no reuse.
                    construction, region_info = _silhouette_for_view(
                        args, camera_name, frame, height, width, alpha_obj,
                        ids_by_camera[camera_name], None)
                    silhouette_px.setdefault(camera_name, {})["%04d" % frame] = \
                        region_info["deva_silhouette_px"]
                    holes[camera_name]["%04d" % frame] = {
                        "alpha": alpha_hole,
                        "silhouette": dict(
                            region_info,
                            **silhouette_report(construction, alpha_obj, alpha_bg)),
                    }
                else:
                    construction = binarise(alpha_obj, 0.5)
                    holes[camera_name]["%04d" % frame] = alpha_hole
                if not construction.any():
                    empty_construction.append(stem)
                _save_png(os.path.join(args.out, "construction_masks", stem + ".png"),
                          (construction.astype(np.uint8) * 255))
                visible = np.zeros_like(construction) if in_window else construction
                _save_png(os.path.join(args.out, "visible_object", stem + ".png"),
                          (visible.astype(np.uint8) * 255))

                if in_window:
                    real, resized = _load_real_image(
                        args.real_images, camera_name, frame, height, width
                    )
                    if resized:
                        resized_real.append(stem)
                    weight = edit_alpha_from_mask(construction, args.dilate, args.feather)
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
            "after_mask_consistency": count_after_consistency,
            "after_iqr_box": count_after_iqr,
            "after_radius": count_after_radius,
        },
        "filters": dict(_filter_config(args),
                        per_timestamp_counts=filter_counts,
                        mask_consistency=consistency_report,
                        canonical_after_mask_consistency=count_after_consistency),
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
        "edit_region": _edit_region_config(args, ids_mapping, ids_by_camera, silhouette_px),
        "hole_fraction_json_shape": (
            "per camera/frame: {alpha: {...}, silhouette: {...}}"
            if args.edit_region == "deva"
            else "per camera/frame: the flat alpha-based hole record"),
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
    # row-set narrowing, applied in this order; every one of them is OFF by
    # default and the default behaviour is unchanged when they are all absent.
    parser.add_argument("--argmax_only", action="store_true",
                        help="base row set is argmax-in-ids only; drop the soft limb")
    parser.add_argument("--scale_max_factor", type=float, default=None,
                        help="drop rows whose largest activated scale exceeds F x the "
                             "base set's median largest scale (default: off)")
    parser.add_argument("--box_percentile", type=float, default=None,
                        help="keep rows inside the per-axis [P, 100-P] percentile box of "
                             "the base set's canonical positions (default: off)")
    parser.add_argument("--box_pad", type=float, default=0.1,
                        help="grow each axis of the percentile box by this fraction of "
                             "its extent, on both sides")
    parser.add_argument("--mask_consistency", type=str, default=None,
                        help="DEVA root holding camXX/pseudo_label/object_mask/FFFF.png; "
                             "enables the multi-view mask vote (default: off)")
    parser.add_argument("--mask_min_cams", type=int, default=None,
                        help="rows must agree with the harmonised DEVA id in at least K "
                             "training cameras (required with --mask_consistency)")
    parser.add_argument("--mask_frames", type=str, default=None,
                        help="anchor frames for the mask vote, e.g. 30,60,90 or 30-40 "
                             "(required with --mask_consistency)")
    parser.add_argument("--mask_min_frames", type=float, default=1.0,
                        help="fraction of anchor frames a row must survive (default 1.0, "
                             "i.e. every anchor frame)")
    # what defines the EDIT REGION. `alpha` is the default and is the behaviour
    # this script had before these flags existed.
    parser.add_argument("--edit_region", choices=["alpha", "deva"], default="alpha",
                        help="region edited: the object-only render's alpha (default), or "
                             "the DEVA silhouette of --deva_ids_by_camera")
    parser.add_argument("--deva_root", type=str, default=None,
                        help="DEVA root holding camXX/pseudo_label/object_mask/FFFF.png for "
                             "the silhouette (required with --edit_region deva)")
    parser.add_argument("--deva_ids_by_camera", type=str, default=None,
                        help='JSON (inline or a file) mapping "camXX" -> [deva ids], e.g. '
                             '\'{"cam00": [125], "cam15": [95]}\'. Cameras left out fall back '
                             "to the id harmonised by --mask_consistency; cam00 is never "
                             "harmonised and must always be named here")
    parser.add_argument("--alpha_guard_px", type=int, default=0,
                        help="intersect the DEVA silhouette with the object alpha dilated by "
                             "this many pixels (default 0 = no intersection at all)")
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
                        help="composite mask dilation radius in pixels (build, and the "
                             "--edit_region deva preview composite)")
    parser.add_argument("--feather", type=float, default=2.0,
                        help="composite mask Gaussian feather sigma in pixels (build, and "
                             "the --edit_region deva preview composite)")
    return parser


def _validate_filter_args(args):
    """Reject narrowing knobs that cannot mean anything, before any GPU work."""
    if args.scale_max_factor is not None and args.scale_max_factor <= 0.0:
        raise SystemExit("--scale_max_factor must be positive, got %r" % (args.scale_max_factor,))
    if args.box_percentile is not None and not 0.0 <= args.box_percentile < 50.0:
        raise SystemExit("--box_percentile must be in [0, 50), got %r" % (args.box_percentile,))
    if args.box_pad < 0.0:
        raise SystemExit("--box_pad must be non-negative, got %r" % (args.box_pad,))
    if not 0.0 < args.mask_min_frames <= 1.0:
        raise SystemExit("--mask_min_frames must be in (0, 1], got %r" % (args.mask_min_frames,))
    if args.mask_consistency:
        # No default is supplied for either: both are load-bearing thresholds
        # and a silent default would decide the row set on the run's behalf.
        if args.mask_min_cams is None:
            raise SystemExit("--mask_consistency needs --mask_min_cams K")
        if args.mask_min_cams < 1:
            raise SystemExit("--mask_min_cams must be at least 1")
        if not args.mask_frames:
            raise SystemExit("--mask_consistency needs --mask_frames")
        try:
            parse_frame_list(args.mask_frames)
        except ValueError as exc:
            raise SystemExit("--mask_frames %r: %s" % (args.mask_frames, exc))
    elif args.mask_min_cams is not None or args.mask_frames:
        raise SystemExit("--mask_min_cams/--mask_frames need --mask_consistency <deva_root>")


def _validate_edit_region_args(args):
    """Reject an edit region that cannot be built, before any GPU work."""
    if args.alpha_guard_px < 0:
        raise SystemExit("--alpha_guard_px must be non-negative, got %r" % (args.alpha_guard_px,))
    if args.edit_region != "deva":
        # Refused rather than ignored: a deva knob on an alpha run means the
        # caller believes the silhouette is in use, and it is not.
        if args.deva_root or args.deva_ids_by_camera or args.alpha_guard_px:
            raise SystemExit("--deva_root/--deva_ids_by_camera/--alpha_guard_px need "
                             "--edit_region deva")
        return
    if not args.deva_root:
        raise SystemExit("--edit_region deva needs --deva_root <dir>")
    if not args.deva_ids_by_camera and not args.mask_consistency:
        raise SystemExit("--edit_region deva needs --deva_ids_by_camera, or "
                         "--mask_consistency to harmonise the ids")
    if args.deva_ids_by_camera:
        try:
            parse_ids_by_camera(args.deva_ids_by_camera)
        except ValueError as exc:
            raise SystemExit(str(exc))


def main(argv=None):
    args = build_parser().parse_args(argv)
    _validate_filter_args(args)
    _validate_edit_region_args(args)
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
