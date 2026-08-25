#!/usr/bin/env python3
"""Per-row membership SUPERVISABILITY, measured on the ACTUAL carrier channel.

MEASUREMENT ONLY. This script trains nothing, writes no checkpoint, emits no
program, and mutates no repository file. It renders a frozen checkpoint and
reports curves, distributions and quantiles. It chooses no threshold.


WHY THIS IS A REWRITE, AND WHAT THE PREVIOUS PROBE GOT WRONG
------------------------------------------------------------
Version 1 of this instrument backpropagated ``rendered_image.sum()`` into
``_features_dc`` and read the per-row ``|grad|`` as the compositing weight a
membership channel would receive. An adversarial review refuted that, and the
refutation was verified against source:

* ``gaussian_renderer/__init__.py:305`` binds ``shs = pc.get_features`` and
  ``:357`` binds ``sh_static = pc.get_features``. Under soft routing
  (``enable_soft_routing: true`` in ``configs/lrv3/a0_local_control.yaml:84``)
  the SAME parameter feeds the dynamic branch and the static twin.
* ``gaussian_renderer/diff_gaussian_rasterization.py:250-270`` returns BOTH
  ``grad_sh`` and ``grad_sh_static``; autograd SUMS them into
  ``_features_dc.grad``.
* ``opacity_static = base_opacity * static_probability``
  (``gaussian_renderer/__init__.py:351``) carries NO temporal marginal and
  renders at the UNDEFORMED ``pc.get_xyz``.

The membership carrier is DYNAMIC-ONLY: ``forward.cu`` accumulates
``Flow[ch] += flows[collected_id[j] * 2 + ch] * alpha * T`` strictly inside
``if (collected_id[j] < P)``, and ``backward.cu`` accumulates
``dL_dflows`` strictly inside ``if (gaussian_idx < P)``. So a row culled by the
temporal marginal (``forward.cu`` opacity path) has ZERO dynamic weight but
NONZERO static weight -- and version 1's probe would have called it
supervisable. It could not fire for the exact condition it existed to detect.
Separately the SH path clamps rendered RGB at zero and zeroes the clamped
channel's gradient (``forward.cu:67-70``, ``backward.cu:32-34``), a second
error in the opposite direction.


THE REPLACEMENT: PROBE THE CARRIER ITSELF
-----------------------------------------
The membership channel composites exactly as

    M_c(pixel) = sum_i  f_ic * alpha_i * T_i                                (1)

with ``f`` the per-row ``flow_2d`` input: two channels, dynamic-only, NO
background term, VJP repaired and pinned against
``tests/ref_impls/flow_compositing_reference.py``. Bind ``flow_2d`` to a
per-row LEAF tensor of ones. Then for any upstream gradient ``g(pixel)`` on
``M_0``:

    dL/df_i0 = sum over pixels of  g(pixel) * alpha_i * T_i                 (2)

so ONE forward pass plus SEVERAL backward passes with different ``g`` yields,
per row:

    g = 1 everywhere   ->  w_total_i      = total accumulated compositing weight
    g = mask_k         ->  w_in_mask_i(k) = weight accumulated inside class k

``flow_2d`` runs through no SH evaluation, hence no clamp, and reaches no
static primitive, hence no static contamination. No gradient-identity argument
is needed: this IS the channel a membership head would use.


THE FIVE ANSWERS FROM ESSENTIALLY ONE FORWARD PASS
--------------------------------------------------
1. SUPERVISABILITY CEILING AS A CURVE. ``achievable_recall_ceiling(e_min)``
   over the FROZEN absolute grid {0, 1e-6, 1e-4, 1e-2, 1e-1, 1}, plus a
   quantile limb whose ``e_min`` values are quantiles of ``w_total`` over ALL
   ROWS OF THE CLOUD at q = {0.50, 0.75, 0.90, 0.95, 0.99} -- never over the
   in-sphere rows, which would be a cut chosen with the oracle in hand -- plus
   the strict ``w_total > 0`` point and the ``w_total`` distribution over the
   target's rows.
2. THE ZERO-PARAMETER CLOSED-FORM VOTE. ``score_i(k) = w_in_mask_i(k)``,
   assign ``argmax_k``, at the FROZEN operating point: eligible iff
   ``w_total_i > 0``, assigned iff ``max_k w_in_mask_i(k) >= 0.50 *
   w_total_i``, else ABSTAIN. Per-row precision and recall against the
   authored sphere. This baseline needs no optimizer; if it clears the
   commissioned reference (precision 0.80 / recall 0.90) then learned training
   is unnecessary. The sweep over tau in {0, 0.25, 0.50, 2/3, 0.75, 0.90} is
   reported beside it and is labelled CEILING INFORMATION ONLY: reading its
   best row as the vote's result would be choosing the threshold after seeing
   the outcome.
3. PER-CELL BREAKDOWN over the recorded 8^3 grid, including cells 420 and 429
   by name.
4. THE STATIC-SPHERE CONTROL. The same instrument aimed at a static sphere
   whose id lives in the same identity buffers and which is supervised at
   every frame with NO temporal-marginal attenuation. This separates "per-row
   membership is learnable on this substrate" from "EVENT membership is
   learnable despite temporal-support gradient starvation". It costs no extra
   render: the class scores for every id are accumulated in the same pass.
5. C1 -- THE STATIC-TWIN SHARE. Both branches' SH inputs are bound to SEPARATE
   leaves so their gradients no longer sum, and the per-row static share is
   reported as a DISTRIBUTION over the target's rows. No threshold is applied.


WHAT THIS SCRIPT DOES NOT SETTLE
--------------------------------
* The ceiling bounds RECALL. A row that CAN receive gradient is not thereby a
  row a head would label correctly -- which is why the closed-form vote is
  measured alongside it, and why the vote's precision is reported.
* It reads the FROZEN cloud only. A retrained or densified cloud is a
  different cloud; ``cloud_fingerprint`` names the one a number binds to.
* The static-twin share (answer 5) is measured on the SH channel, which is the
  only per-row handle the static branch exposes -- the flow channel does not
  reach static primitives at all. It therefore inherits the RGB clamp. The
  ``clamp_exposure`` block counts the rows the DC-order clamp could zero, and
  ``dc_vs_flow_consistency`` measures the clamp's actual bite on the dynamic
  branch by comparing the clamp-exposed DC gradient against the clamp-free
  flow weight.


FROZEN PRECONDITIONS
--------------------
Every precondition is a statement about the SETUP or about whether the
MECHANISM was exercised. None reads a score, so none can leak the outcome, and
all are evaluated before any ratio is formed or printed. Freezing a reading
rule is not enough; a frozen rule needs a frozen precondition asserting the
mechanism it reads was actually exercised.

  P1  render_ran                    -- rows with nonzero ``w_total`` exist.
  P2  rows_in_event_target_positive -- the authored event region contains rows.
  P3  rows_in_static_target_positive-- the CONTROL is not vacuous either.
  P4  frame_set_within_presence     -- every measured frame lies inside an
                                       authored presence window.
  P5  any_view_rendered_nonzero     -- the renders are not blank.
  P6  cameras_disjoint_from_test    -- by parsed id AND by object identity
                                       against the train split; an empty or
                                       absent held-out roster is REFUSED, not
                                       passed vacuously.
  P7  row_count_matches_checkpoint  -- the loaded cloud is the checkpoint's.
  P8  topology_invariant            -- the row count is unchanged across the
                                       whole pass.
  P9  flow_leaf_bound_every_view    -- the leaf substitution was verified at
                                       the rasterizer call boundary once per
                                       render, and the counts agree.
  P10 mask_partition_consistent     -- for every view, sum_k w_in_mask_k
                                       reproduces the independently measured
                                       w_total. The identity buffer partitions
                                       the image, so this is an identity; a
                                       failure means a mask, a class list or an
                                       upstream gradient is wrong.
  P11 backward_repeatable           -- the g = 1 backward, repeated AFTER every
                                       class backward on the same retained
                                       graph, reproduces its first result BIT
                                       FOR BIT. Several backward passes over
                                       one forward is the load-bearing trick of
                                       this instrument; this is the observation
                                       that it is sound.
  P12 static_branch_shares_features -- the static twin really does read the
                                       same features as the dynamic branch
                                       (the fact the refutation rests on), so
                                       answer 5 is measuring what it claims.
  P13 identity_masks_complete       -- every requested (camera, frame) buffer
                                       exists, has the render's shape, and its
                                       recorded sha256 matches on load.
  P14 camera_mask_supply            -- no training camera supplies fewer event
                                       pixels than ``--min-camera-mask-px``.
                                       DEFAULT 0, so this is REPORT ONLY: the
                                       per-camera counts are printed and the
                                       scientific floor is the primary's to
                                       choose, never this script's.
  P15 fingerprint_as_expected       -- when ``--expect-fingerprint`` is given,
                                       the cloud is the named one. When it is
                                       not given the check is recorded as not
                                       requested and cannot fail.

If any precondition fails the script raises before reporting the answers; the
failure list is also written into the JSON's ``preconditions`` block.


ANTI-LEAKAGE
------------
Training cameras only. ``scene.getTestCameras`` is replaced by a raiser for the
whole measurement, the consumed cameras are checked to be elements of the train
split by object identity, and their parsed ids are checked disjoint from
``event_spec.json``'s ``test_cameras``. Supervision masks are read from
``train_identity/`` and a directory whose final component is ``gt_identity`` is
REFUSED. The authored spheres are read from the fixture and from the
generator's own constants, and are used ONLY to label rows after every weight
already exists; they never enter a render or a gradient.


NO THRESHOLD IS CHOSEN HERE
---------------------------
The ceiling is reported as a CURVE over a frozen ``e_min`` grid plus the exact
``w_total > 0`` point. The static-twin share is reported as QUANTILES. The
per-camera supply is reported per camera. Selecting any of these is a
scientific decision this instrument does not make.

The ONE cut that is applied -- the vote's ``VOTE_TAU`` -- is not an exception
to that: it is frozen in this file ahead of the run and the sweep around it is
labelled as ceiling information precisely so that it stays frozen.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA = "membership-supervisability-v2"

#: FROZEN absolute ``e_min`` grid for the ceiling CURVE. Declared here, never
#: derived from the measured distribution, and never reduced to one value.
CEILING_E_MIN_GRID = (0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0)

#: FROZEN quantile grid for the SECOND limb of the ceiling curve. The e_min
#: values these produce are quantiles of ``w_total`` over **ALL ROWS OF THE
#: CLOUD**, never over the in-sphere rows: a cut derived from the target's own
#: distribution would be a cut chosen with the oracle in hand, and the ceiling
#: it produced would be partly a restatement of the labels. Over all rows the
#: e_min values are a property of the substrate, so the same six numbers apply
#: to the event target, to the static control and to every cell, which is also
#: what makes those three comparable.
CEILING_QUANTILE_GRID = (0.50, 0.75, 0.90, 0.95, 0.99)

#: Quantiles of the ``w_total`` distribution over a target's rows.
WEIGHT_QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0)

#: Quantiles of the per-row static-twin share (answer 5). Reported as a
#: distribution, never pooled into a mean and never thresholded.
SHARE_QUANTILES = (0.50, 0.90, 0.95, 0.99, 1.0)

#: The reference the vote is READ against. Supplied in this instrument's
#: commission, not chosen by this script, and reported as a plain comparison.
VOTE_REFERENCE = {"precision": 0.80, "recall": 0.90}

#: THE FROZEN OPERATING POINT of the vote (answer 2). A row is ELIGIBLE iff
#: ``w_total_i > 0`` and is ASSIGNED its argmax class iff
#: ``max_k w_in_mask_i(k) >= VOTE_TAU * w_total_i``; otherwise it ABSTAINS.
#: This single value is the score. It is declared here, ahead of any run.
VOTE_TAU = 0.50

#: The tau sweep. Reported as CEILING INFORMATION ONLY -- it says what the vote
#: could reach at other operating points, and reading the best entry of it as
#: the vote's result would be choosing a threshold after seeing the outcome,
#: which is exactly what freezing VOTE_TAU exists to prevent. 2/3 is included
#: because it is the smallest tau that forces a strict majority of a row's
#: weight into one class when three classes carry weight.
VOTE_TAU_GRID = (0.00, 0.25, 0.50, 2.0 / 3.0, 0.75, 0.90)

#: Cell keys called out by name in the per-cell breakdown: the 2,036 rows
#: (19.12% of the object) that produced the structural 0.8088 recall cap.
NAMED_CELL_KEYS = (420, 429)

#: Cells per axis of the recorded grid (`scripts/estimate_episodes.py`
#: VOXEL_CELLS_PER_AXIS). Overridden by a supplied episode program.
DEFAULT_CELLS_PER_AXIS = 8

#: Sentinel for "this row had no positive evidence for any class".
ABSTAIN_CLASS = -9999

#: The directory name this instrument refuses to read supervision from.
FORBIDDEN_IDENTITY_DIRNAME = "gt_identity"

_CAMERA_ID = re.compile(r"cam(\d+)")
_FRAME_ID = re.compile(r"_f(\d+)")
_IDENTITY_NAME = re.compile(r"^cam(\d+)_f(\d+)\.npy$")


class ContractError(RuntimeError):
    """A setup the instrument refuses to measure."""


class PreconditionError(RuntimeError):
    """A frozen precondition about the setup did not hold."""


class LeakageError(RuntimeError):
    """The measurement reached for a held-out input."""


# ---------------------------------------------------------------------------
# PURE LOGIC -- numpy and stdlib only, no torch, no CUDA, no I/O beyond
# reading .npy. Everything below is exercised by `--self-test` and by
# tests/test_membership_supervisability.py.
# ---------------------------------------------------------------------------


def expand_inclusive_pair(pair, label):
    """``[start, end]`` (inclusive, as written by the fixture generator) -> list.

    ``build_synthetic_reveal_scene.py`` writes ``presence_frames`` as a PAIR of
    endpoints, not an enumeration; ``return_frames`` in the same file IS an
    enumeration. The two are read by different functions on purpose so neither
    convention can silently absorb the other.
    """
    values = [int(v) for v in pair]
    if len(values) != 2:
        raise ContractError(
            "%s must be an inclusive [start, end] pair; got %r" % (label, pair))
    start, end = values
    if end < start:
        raise ContractError("%s has end %d before start %d" % (label, end, start))
    if start < 0:
        raise ContractError("%s has negative start %d" % (label, start))
    return list(range(start, end + 1))


def default_frame_set(spec):
    """The declared default: episode-1 frames plus the return frames.

    Both come from the fixture's own ``event_spec.json``. Nothing is
    hardcoded -- feeding this a fixture whose return sits elsewhere (LRV4's
    one-frame return) yields that fixture's frames, not LRV3's.
    """
    presence = spec.get("presence_frames")
    if not isinstance(presence, dict) or "episode_1" not in presence:
        raise ContractError("event_spec.json carries no presence_frames.episode_1")
    episode_1 = expand_inclusive_pair(presence["episode_1"],
                                      "presence_frames.episode_1")
    returns = spec.get("return_frames")
    if not isinstance(returns, (list, tuple)) or not returns:
        raise ContractError("event_spec.json carries no non-empty return_frames")
    return sorted(set(episode_1) | set(int(f) for f in returns))


def presence_windows(spec):
    """Inclusive [start, end] windows in which the event object EXISTS."""
    presence = spec.get("presence_frames") or {}
    windows = []
    for key in sorted(presence):
        if key == "gap":
            continue
        pair = [int(v) for v in presence[key]]
        if len(pair) != 2:
            raise ContractError(
                "presence_frames.%s must be an inclusive pair; got %r"
                % (key, presence[key]))
        windows.append((pair[0], pair[1]))
    if not windows:
        raise ContractError("event_spec.json declares no presence windows")
    return windows


def frames_within_presence(frames, windows):
    """(ok, offending_frames) for P4. A statement about the FRAME SET only."""
    bad = []
    for frame in frames:
        frame = int(frame)
        if not any(start <= frame <= end for start, end in windows):
            bad.append(frame)
    return (not bad), bad


def parse_frame_spec(text):
    """``"0-29,57,58,59"`` -> sorted unique frame list. Empty text -> None."""
    text = str(text or "").strip()
    if not text:
        return None
    frames = set()
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk.lstrip("-"):
            lo_text, _, hi_text = chunk.partition("-")
            lo, hi = int(lo_text), int(hi_text)
            if hi < lo:
                raise ContractError("frame range %r runs backwards" % chunk)
            frames.update(range(lo, hi + 1))
        else:
            frames.add(int(chunk))
    if not frames:
        raise ContractError("--frames parsed to an empty set")
    return sorted(frames)


def in_sphere_flags(points, centre, radius):
    """``(xyz - centre).norm(dim=1) <= radius`` as a boolean numpy array.

    ONE implementation, used by the torch limb as well: the labels are formed
    on CPU from the detached xyz, so there is no second expression that could
    drift from this one.
    """
    cx, cy, cz = (float(centre[0]), float(centre[1]), float(centre[2]))
    radius = float(radius)
    if radius < 0.0:
        raise ContractError("sphere radius %r is negative" % radius)
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ContractError("points must be (N, 3); got %r" % (p.shape,))
    delta = p - np.asarray([cx, cy, cz], dtype=np.float64)
    return np.sqrt((delta * delta).sum(axis=1)) <= radius


def voxel_keys(xyz, lo=None, span=None, cells_per_axis=DEFAULT_CELLS_PER_AXIS):
    """(lo, span, keys) bit-faithful to ``estimate_episodes.voxel_grid``.

    The arithmetic is done in FLOAT32 because torch does it in float32; a
    float64 reproduction can place a boundary row in a different cell. ``clamp``
    is applied to the float BEFORE truncation, exactly as torch's
    ``.clamp(0, cells - 1).long()`` does. Cell ids are
    ``ix * cells^2 + iy * cells + iz``.

    ``lo``/``span`` may be supplied (from a recorded episode program, which
    carries the ABSOLUTE world-space grid) or recomputed from these points'
    own bounding box.
    """
    cells = int(cells_per_axis)
    if cells < 1:
        raise ContractError("cells_per_axis must be >= 1")
    p = np.ascontiguousarray(xyz, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ContractError("xyz must be (N, 3); got %r" % (p.shape,))
    if lo is None or span is None:
        lo = p.min(axis=0)
        span = np.maximum(p.max(axis=0) - lo, np.float32(1e-6)).astype(np.float32)
    else:
        lo = np.asarray(lo, dtype=np.float32).reshape(3)
        span = np.asarray(span, dtype=np.float32).reshape(3)
        if bool((span <= 0).any()):
            raise ContractError("supplied grid span has a non-positive axis")
    voxel = ((p - lo) / span * np.float32(cells)).astype(np.float32)
    voxel = np.clip(voxel, np.float32(0.0), np.float32(cells - 1))
    idx = voxel.astype(np.int64)
    keys = idx[:, 0] * cells * cells + idx[:, 1] * cells + idx[:, 2]
    return lo, span, keys


def decode_cell_key(key, cells_per_axis=DEFAULT_CELLS_PER_AXIS):
    """``420`` -> ``(6, 4, 4)``. The inverse of ``voxel_keys``' encoding."""
    cells = int(cells_per_axis)
    key = int(key)
    return (key // (cells * cells), (key // cells) % cells, key % cells)


def quantiles_of(values, qs):
    """Linear-interpolated quantiles, keyed ``pNNN``. Empty -> all None."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    out = {}
    for q in qs:
        q = float(q)
        if not 0.0 <= q <= 1.0:
            raise ContractError("quantile %r outside [0, 1]" % q)
        key = "p%03d" % int(round(q * 100))
        out[key] = (None if array.size == 0
                    else float(np.quantile(array, q, method="linear")))
    return out


def weight_distribution(values, qs=WEIGHT_QUANTILES):
    """min / declared quantiles / max, plus the exact-zero count."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "n": int(array.size),
        "n_exactly_zero": int((array == 0.0).sum()),
        "min": (float(array.min()) if array.size else None),
        "max": (float(array.max()) if array.size else None),
        "mean": (float(array.mean()) if array.size else None),
        "quantiles": quantiles_of(array, qs),
    }


def quantile_e_min_points(all_row_weights, qs=CEILING_QUANTILE_GRID):
    """The ceiling curve's quantile limb: e_min values from **ALL** rows.

    ``all_row_weights`` must be the ``w_total`` of the WHOLE cloud, not of the
    in-sphere rows. Quantiles of the target's own distribution would be cuts
    chosen with the oracle in hand; quantiles of the cloud are a property of the
    substrate and know nothing about which rows are the object. Computing them
    once, here, is also what lets the event target, the static control and every
    cell be read against the SAME six numbers.

    Returns a list of ``{"q", "e_min"}`` in the declared order. An empty cloud
    yields an empty list rather than a fabricated cut.
    """
    w = np.asarray(all_row_weights, dtype=np.float64).reshape(-1)
    if w.size and bool((w < 0).any()):
        raise ContractError(
            "a row carries a negative accumulated weight; the accumulator "
            "reads magnitudes only")
    points = []
    for q in qs:
        q = float(q)
        if not 0.0 <= q <= 1.0:
            raise ContractError("quantile %r outside [0, 1]" % q)
        if w.size == 0:
            continue
        points.append({
            "q": q,
            "e_min": float(np.quantile(w, q, method="linear")),
        })
    return points


def ceiling_curve(target_weights, grid=CEILING_E_MIN_GRID, quantile_points=None):
    """Answer 1: the supervisability ceiling as a CURVE, not a scalar.

    ``achievable_recall_ceiling(e_min)`` = |{target rows with
    ``w_total >= e_min``}| / |target rows|, over the FROZEN absolute grid, plus
    the exact ``w_total > 0`` point which is the mathematical statement of (2)
    and not a tuned cut, plus a quantile limb whose e_min values come from
    ``quantile_e_min_points`` over ALL rows of the cloud.

    Returns ``None`` for an empty target: a ratio without its n is not a
    measurement.
    """
    w = np.asarray(target_weights, dtype=np.float64).reshape(-1)
    if w.size and bool((w < 0).any()):
        raise ContractError(
            "a target row carries a negative accumulated weight; the "
            "accumulator reads magnitudes only")
    n = int(w.size)

    def entry(e_min, strict):
        e_min = float(e_min)
        if e_min < 0.0:
            raise ContractError("e_min %r is negative" % e_min)
        at_or_above = int((w > 0.0).sum() if strict else (w >= e_min).sum())
        return {
            "e_min": e_min,
            "predicate": "w_total > 0" if strict else "w_total >= e_min",
            "n_at_or_above": at_or_above,
            "n_below": n - at_or_above,
            "achievable_recall_ceiling": (at_or_above / float(n) if n else None),
        }

    quantile_curve = []
    for point in (quantile_points or []):
        item = entry(point["e_min"], False)
        item["q"] = float(point["q"])
        item["e_min_source"] = "quantile of w_total over ALL rows of the cloud"
        quantile_curve.append(item)

    return {
        "n_target_rows": n,
        "strict_positive": entry(0.0, True),
        "curve": [entry(e, e == 0.0) for e in grid],
        "quantile_curve": quantile_curve,
        "quantile_curve_provenance": (
            "e_min values are quantiles of w_total over ALL rows, never over "
            "the target's rows: a cut read off the target's own distribution "
            "would be chosen with the oracle in hand"),
    }


def argmax_vote(class_ids, scores, w_total, tau=VOTE_TAU):
    """Answer 2: the zero-parameter closed-form vote at its FROZEN operating point.

    ``score_i(k) = w_in_mask_i(k)``; assign ``argmax_k``. The operating point is
    the one frozen in ``VOTE_TAU`` and is applied here in full:

    * a row is ELIGIBLE iff ``w_total_i > 0`` -- a row that received no
      compositing weight at all was never supervised and has nothing to vote
      with;
    * an eligible row is ASSIGNED iff ``max_k w_in_mask_i(k) >= tau *
      w_total_i`` -- its best class must own at least that share of the weight
      the row actually accumulated;
    * every other row ABSTAINS, taking ``ABSTAIN_CLASS``.

    Ties go to the first class in the supplied order and are counted. Note that
    the rule is a SHARE, not a magnitude: it is scale-free in the row's own
    supply, so a faintly-supervised row is judged on the same terms as a
    strongly-supervised one, and the magnitude question is answered separately
    by the ceiling curve.

    ``scores`` is (N, K) with the columns in the order of ``class_ids``.
    """
    ids = np.asarray([int(c) for c in class_ids], dtype=np.int64)
    if ids.size == 0:
        raise ContractError("the vote needs at least one class")
    if len(set(ids.tolist())) != int(ids.size):
        raise ContractError("duplicate class id in %r" % (ids.tolist(),))
    if ABSTAIN_CLASS in set(ids.tolist()):
        raise ContractError(
            "class id %d collides with the abstention sentinel" % ABSTAIN_CLASS)
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != int(ids.size):
        raise ContractError(
            "scores must be (N, %d); got %r" % (int(ids.size), (s.shape,)))
    if s.size and bool((s < 0).any()):
        raise ContractError("a class score is negative; weights are magnitudes")
    total = np.asarray(w_total, dtype=np.float64).reshape(-1)
    if total.size != s.shape[0]:
        raise ContractError(
            "w_total has %d entries but scores has %d rows"
            % (int(total.size), int(s.shape[0])))
    if total.size and bool((total < 0).any()):
        raise ContractError("w_total is negative; weights are magnitudes")
    tau = float(tau)
    if not 0.0 <= tau <= 1.0:
        raise ContractError("tau %r outside [0, 1]" % tau)

    best_j = s.argmax(axis=1) if s.shape[0] else np.zeros(0, dtype=np.int64)
    best = (s[np.arange(s.shape[0]), best_j] if s.shape[0]
            else np.zeros(0, dtype=np.float64))

    eligible = total > 0.0
    meets_tau = best >= tau * total
    assign = eligible & meets_tau
    assigned = np.where(assign, ids[best_j], np.int64(ABSTAIN_CLASS))
    tied = ((s == best[:, None]).sum(axis=1) > 1) & assign

    stats = {
        "n_rows": int(s.shape[0]),
        "tau": tau,
        "rule": ("eligible iff w_total > 0; assigned iff "
                 "max_k w_in_mask(k) >= tau * w_total; else ABSTAIN"),
        "tau_provenance": ("FROZEN in VOTE_TAU ahead of the run; the tau sweep "
                           "is reported separately as ceiling information and "
                           "is never read as the score"),
        "n_eligible": int(eligible.sum()),
        "n_assigned": int(assign.sum()),
        "n_abstained": int((~assign).sum()),
        "n_abstained_ineligible": int((~eligible).sum()),
        "n_abstained_below_tau": int((eligible & ~meets_tau).sum()),
        # Under P10 the class weights re-sum to w_total, so an eligible row
        # always has a strictly positive best and this reads zero. It is
        # reported rather than assumed: at tau = 0.00 the rule alone would let
        # such a row be assigned by tie-break, and a reader should see that it
        # did not happen rather than take it on trust.
        "n_assigned_with_zero_best": int((assign & (best <= 0.0)).sum()),
        "n_tied": int(tied.sum()),
        "class_order": ids.tolist(),
        "tie_break": "first class in class_order",
    }
    return assigned, stats


def vote_tau_curve(class_ids, scores, w_total, truth, target_class,
                   grid=VOTE_TAU_GRID):
    """The vote's precision/recall across the tau sweep. CEILING INFORMATION.

    This exists so the frozen operating point can be SITUATED -- to show whether
    ``VOTE_TAU`` sits on a plateau or on a cliff. It is not the score, and the
    best row of it is not a result: picking a tau after seeing this table would
    be choosing a threshold with the oracle in hand, which is the failure the
    frozen operating point exists to prevent. Every entry is labelled.
    """
    rows = []
    for tau in grid:
        assigned, stats = argmax_vote(class_ids, scores, w_total, tau=tau)
        metrics = precision_recall(assigned == int(target_class), truth)
        rows.append({
            "tau": float(tau),
            "is_frozen_operating_point": bool(float(tau) == float(VOTE_TAU)),
            "n_assigned": stats["n_assigned"],
            "n_abstained": stats["n_abstained"],
            "n_abstained_below_tau": stats["n_abstained_below_tau"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"],
        })
    return {
        "reading": "CEILING INFORMATION ONLY -- never the score",
        "frozen_operating_point": float(VOTE_TAU),
        "grid": [float(t) for t in grid],
        "curve": rows,
    }


def precision_recall(predicted, truth):
    """Per-row precision and recall. ``None`` when a denominator is zero."""
    p = np.asarray(predicted, dtype=bool).reshape(-1)
    t = np.asarray(truth, dtype=bool).reshape(-1)
    if p.shape != t.shape:
        raise ContractError(
            "predicted/truth length mismatch: %d vs %d" % (p.size, t.size))
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    tn = int((~p & ~t).sum())
    return {
        "n": int(p.size),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_predicted_positive": tp + fp,
        "n_truth_positive": tp + fn,
        "precision": (tp / float(tp + fp) if (tp + fp) else None),
        "recall": (tp / float(tp + fn) if (tp + fn) else None),
    }


def clears_reference(metrics, reference=VOTE_REFERENCE):
    """Plain comparison against the COMMISSIONED reference pair.

    This is not a threshold this instrument chose; it is the pair the
    instrument was asked to read the vote against, reported as a boolean
    beside the raw numbers.
    """
    precision = metrics.get("precision")
    recall = metrics.get("recall")
    if precision is None or recall is None:
        return None
    return bool(precision >= float(reference["precision"])
                and recall >= float(reference["recall"]))


def restricted_metrics(mask, weights, predicted, truth, grid=CEILING_E_MIN_GRID,
                       quantile_points=None):
    """Ceiling + vote precision/recall restricted to the rows ``mask`` selects.

    ``quantile_points`` are the cloud-wide e_min values from
    ``quantile_e_min_points``. They are passed in rather than recomputed per
    cell for two reasons: a per-cell quantile would be a cut derived from the
    cell's own rows, and cells read against different cuts are not comparable.
    """
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=bool).reshape(-1)
    truth = np.asarray(truth, dtype=bool).reshape(-1)
    if not (mask.shape == weights.shape == predicted.shape == truth.shape):
        raise ContractError("restricted_metrics received ragged inputs")
    target = mask & truth
    return {
        "n_rows_in_restriction": int(mask.sum()),
        "n_target_rows_in_restriction": int(target.sum()),
        "ceiling": ceiling_curve(weights[target], grid, quantile_points),
        "w_total_distribution": weight_distribution(weights[target]),
        "vote": precision_recall(predicted[mask], truth[mask]),
    }


def camera_id_of_name(image_name):
    """``"cam07_f012"`` -> 7. Refuses a name it cannot parse."""
    match = _CAMERA_ID.search(str(image_name or ""))
    if match is None:
        raise ContractError(
            "cannot parse a camera id from image_name %r" % (image_name,))
    return int(match.group(1))


def frame_index_of_name(image_name):
    """``"cam07_f012"`` -> 12, or None when the name carries no frame tag."""
    match = _FRAME_ID.search(str(image_name or ""))
    if match is None:
        return None
    return int(match.group(1))


def assert_cameras_are_training(used_camera_ids, train_camera_ids, test_camera_ids):
    """Refuse any camera that is held out or not a declared training camera.

    Both rosters are REQUIRED. An absent or empty ``test_cameras`` would make
    the disjointness test vacuously true, which is exactly the failure mode
    where a guard degrades to "protects nothing" -- so it refuses instead.
    """
    if not train_camera_ids:
        raise ContractError(
            "event_spec.json declares no train_cameras; the training-only "
            "guard would have nothing to check against")
    if not test_camera_ids:
        raise ContractError(
            "event_spec.json declares no test_cameras; the held-out "
            "disjointness check would be vacuously true")
    used = sorted(set(int(c) for c in used_camera_ids))
    train = set(int(c) for c in train_camera_ids)
    held_out = set(int(c) for c in test_camera_ids)
    overlap = sorted(set(used) & held_out)
    if overlap:
        raise LeakageError(
            "anti-leakage: cameras %r are held out in event_spec.json" % (overlap,))
    stray = sorted(set(used) - train)
    if stray:
        raise LeakageError(
            "anti-leakage: cameras %r are not declared training cameras" % (stray,))
    return True


def check_flow_binding(tensor, n_rows):
    """The leaf substitution really is a LEAF of the right shape.

    Duck-typed on ``.is_leaf`` / ``.requires_grad`` / ``.shape`` so the
    self-test can exercise it without torch. Bound to a NON-leaf (an
    expression, a slice, a detached-then-operated tensor) no gradient would
    ever arrive at it and every ``w_total`` would read zero -- which P1 then
    catches. This refuses one step earlier, at the binding.
    """
    shape = tuple(int(v) for v in getattr(tensor, "shape", ()))
    if shape != (int(n_rows), 2):
        raise ContractError(
            "flow_2d leaf must be (%d, 2); got %r" % (int(n_rows), (shape,)))
    if not bool(getattr(tensor, "requires_grad", False)):
        raise ContractError("flow_2d leaf does not require grad")
    if not bool(getattr(tensor, "is_leaf", False)):
        raise ContractError(
            "flow_2d was bound to a NON-LEAF: autograd would deliver no "
            "gradient to it and every accumulated weight would read zero")
    return True


def evaluate_preconditions(
    n_rows_nonzero_w_total,
    n_rows_in_event_target,
    n_rows_in_static_target,
    frame_set_ok,
    offending_frames,
    n_views_nonzero_image,
    n_views,
    camera_ids_used,
    test_camera_ids,
    cameras_are_train_objects,
    n_rows_loaded,
    n_rows_checkpoint,
    n_rows_after_pass,
    n_rasterizer_calls,
    n_expected_rasterizer_calls,
    mask_partition_ok,
    backward_repeat_bitwise_identical,
    static_branch_shares_features,
    identity_masks_complete,
    cameras_below_mask_floor,
    min_camera_mask_px,
    fingerprint_expected,
    fingerprint_measured,
):
    """The frozen precondition block. Booleans only; no score is read.

    Returns ``(block, failures)``. ``block`` goes into the JSON verbatim so a
    reader can see every check that ran, not only the ones that failed.
    """
    used = sorted(int(c) for c in camera_ids_used)
    held_out = sorted(int(c) for c in test_camera_ids)
    overlap = sorted(set(used) & set(held_out))
    expected = (None if fingerprint_expected is None
                else str(fingerprint_expected).strip().lower())
    measured = str(fingerprint_measured or "").strip().lower()

    block = {
        "P1_render_ran": bool(int(n_rows_nonzero_w_total) > 0),
        "P2_rows_in_event_target_positive": bool(int(n_rows_in_event_target) > 0),
        "P3_rows_in_static_target_positive": bool(int(n_rows_in_static_target) > 0),
        "P4_frame_set_within_presence": bool(frame_set_ok),
        "P5_any_view_rendered_nonzero": bool(int(n_views_nonzero_image) > 0),
        "P6_cameras_disjoint_from_test": (bool(not overlap)
                                          and bool(cameras_are_train_objects)
                                          and bool(held_out)),
        "P7_row_count_matches_checkpoint": bool(
            int(n_rows_loaded) == int(n_rows_checkpoint)),
        "P8_topology_invariant": bool(int(n_rows_after_pass) == int(n_rows_loaded)),
        "P9_flow_leaf_bound_every_view": bool(
            int(n_rasterizer_calls) == int(n_expected_rasterizer_calls)
            and int(n_expected_rasterizer_calls) > 0),
        "P10_mask_partition_consistent": bool(mask_partition_ok),
        "P11_backward_repeatable": bool(backward_repeat_bitwise_identical),
        "P12_static_branch_shares_features": bool(static_branch_shares_features),
        "P13_identity_masks_complete": bool(identity_masks_complete),
        "P14_camera_mask_supply": bool(not list(cameras_below_mask_floor)),
        "P15_fingerprint_as_expected": bool(expected is None or expected == measured),
        "detail": {
            "n_rows_nonzero_w_total": int(n_rows_nonzero_w_total),
            "n_rows_in_event_target": int(n_rows_in_event_target),
            "n_rows_in_static_target": int(n_rows_in_static_target),
            "frames_outside_presence": [int(f) for f in offending_frames],
            "n_views": int(n_views),
            "n_views_rendered_nonzero": int(n_views_nonzero_image),
            "n_views_rendered_all_zero": int(n_views) - int(n_views_nonzero_image),
            "camera_ids_used": used,
            "test_camera_ids": held_out,
            "camera_id_overlap_with_test": overlap,
            "cameras_are_train_split_objects": bool(cameras_are_train_objects),
            "n_rows_loaded": int(n_rows_loaded),
            "n_rows_checkpoint": int(n_rows_checkpoint),
            "n_rows_after_pass": int(n_rows_after_pass),
            "n_rasterizer_calls_intercepted": int(n_rasterizer_calls),
            "n_rasterizer_calls_expected": int(n_expected_rasterizer_calls),
            "mask_partition_consistent": bool(mask_partition_ok),
            "backward_repeat_bitwise_identical": bool(backward_repeat_bitwise_identical),
            "static_branch_shares_features": bool(static_branch_shares_features),
            "identity_masks_complete": bool(identity_masks_complete),
            "min_camera_mask_px": int(min_camera_mask_px),
            "cameras_below_mask_floor": [int(c) for c in cameras_below_mask_floor],
            "fingerprint_expected": expected,
            "fingerprint_measured": measured,
            "fingerprint_check_requested": bool(expected is not None),
        },
    }
    failures = [key for key, value in block.items()
                if key != "detail" and not value]
    return block, failures


# ---------------------------------------------------------------------------
# IDENTITY MASKS -- numpy only, so the census runs (and is testable) without
# torch or a GPU.
# ---------------------------------------------------------------------------


def resolve_identity_dir(source_path, identity_dir):
    """``<source>/train_identity`` by default; ``gt_identity`` is REFUSED.

    ``gt_identity/`` means HELD-OUT ONLY in this repository and existing
    consumers read that meaning off the directory name. Reading supervision
    masks from it would be a leak, so it refuses by construction.
    """
    path = (Path(source_path) / "train_identity" if not identity_dir
            else Path(identity_dir))
    if path.name.strip().lower() == FORBIDDEN_IDENTITY_DIRNAME:
        raise LeakageError(
            "anti-leakage: %r names the HELD-OUT identity directory; "
            "supervision masks come from train_identity/ only" % str(path))
    if not path.is_dir():
        raise ContractError("no identity mask directory at %s" % path)
    return path


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def identity_census(identity_dir, wanted_views, reference_frame, event_object_id):
    """Read every buffer in the directory once and report what is there.

    Returns ``(census, by_view, by_digest)`` where ``by_view`` maps
    ``(camera, frame) -> sha256`` and ``by_digest`` maps ``sha256 -> int16
    array``. Distinct buffers are counted by CONTENT, which is the fact that
    matters: 960 files carrying only 32 distinct images means the effective
    supervision n is the number of distinct masks, not the number of
    observations.
    """
    identity_dir = Path(identity_dir)
    files = sorted(p for p in identity_dir.iterdir()
                   if _IDENTITY_NAME.match(p.name))
    if not files:
        raise ContractError("no camNN_fFFF.npy buffers under %s" % identity_dir)

    by_view = {}
    by_digest = {}
    per_camera_digests = {}
    shapes, dtypes = set(), set()
    class_ids = set()
    for path in files:
        match = _IDENTITY_NAME.match(path.name)
        cam, frame = int(match.group(1)), int(match.group(2))
        payload = path.read_bytes()
        digest = sha256_bytes(payload)
        by_view[(cam, frame)] = digest
        per_camera_digests.setdefault(cam, set()).add(digest)
        if digest not in by_digest:
            array = np.load(str(path), allow_pickle=False)
            by_digest[digest] = array
            shapes.add(tuple(int(v) for v in array.shape))
            dtypes.add(str(array.dtype))
            class_ids.update(int(v) for v in np.unique(array).tolist())

    if len(shapes) != 1:
        raise ContractError("identity buffers disagree on shape: %r" % (shapes,))
    if len(dtypes) != 1:
        raise ContractError("identity buffers disagree on dtype: %r" % (dtypes,))

    missing = sorted(v for v in wanted_views if v not in by_view)
    per_camera_event_px = {}
    per_camera_event_px_total = {}
    for cam, frame in sorted(wanted_views):
        array = by_digest[by_view[(cam, frame)]] if (cam, frame) in by_view else None
        if array is None:
            continue
        count = int((array == int(event_object_id)).sum())
        per_camera_event_px_total[cam] = per_camera_event_px_total.get(cam, 0) + count
        if int(frame) == int(reference_frame):
            per_camera_event_px[cam] = count

    census = {
        "dir": str(identity_dir).replace("\\", "/"),
        "n_files": len(files),
        "n_distinct_buffers_by_content": len(by_digest),
        "distinct_buffers_per_camera": {
            str(c): len(s) for c, s in sorted(per_camera_digests.items())},
        "n_cameras_in_dir": len(per_camera_digests),
        "shape": list(shapes)[0],
        "dtype": list(dtypes)[0],
        "class_ids_present": sorted(class_ids),
        "event_object_id": int(event_object_id),
        "reference_frame": int(reference_frame),
        "event_pixels_per_camera_at_reference_frame": {
            str(c): int(v) for c, v in sorted(per_camera_event_px.items())},
        "event_pixels_per_camera_over_measured_views": {
            str(c): int(v) for c, v in sorted(per_camera_event_px_total.items())},
        "event_pixels_total_at_reference_frame": int(sum(per_camera_event_px.values())),
        "missing_requested_views": [list(v) for v in missing],
        "effective_supervision_note": (
            "%d files carry only %d distinct images by content, i.e. %s per "
            "camera. The effective supervision n is the number of DISTINCT "
            "masks, not the number of observations."
            % (len(files), len(by_digest),
               sorted({len(s) for s in per_camera_digests.values()}))),
    }
    return census, by_view, by_digest


def cameras_below_supply_floor(per_camera_counts, min_camera_mask_px):
    """Cameras whose event-pixel supply is below a DECLARED floor.

    The default floor is 0, so nothing is ever below it and this is REPORT
    ONLY. Choosing a scientific floor is the primary's decision; this
    instrument reports every camera's count and flags against whatever floor it
    is handed.
    """
    floor = int(min_camera_mask_px)
    if floor < 0:
        raise ContractError("--min-camera-mask-px must be >= 0")
    return sorted(int(c) for c, v in per_camera_counts.items() if int(v) < floor)


# ---------------------------------------------------------------------------
# FIXTURE GEOMETRY -- read from the fixture and from the generator's own
# constants. Nothing here is a guessed literal.
# ---------------------------------------------------------------------------


def load_event_spec(source_path):
    path = Path(source_path) / "event_spec.json"
    if not path.is_file():
        raise ContractError("no event_spec.json under %s" % source_path)
    return json.loads(path.read_text(encoding="utf-8")), str(path)


def sphere_from_spec(spec):
    """Centre and radius of the authored event object, from the fixture."""
    obj = spec.get("event_object")
    if not isinstance(obj, dict):
        raise ContractError("event_spec.json carries no event_object")
    centre = [float(v) for v in obj["centre"]]
    radius = float(obj["radius"])
    if len(centre) != 3:
        raise ContractError("event_object.centre must have three components")
    if radius <= 0.0:
        raise ContractError("event_object.radius must be positive; got %r" % radius)
    return centre, radius


def generator_identity_constants():
    """Object ids and static-sphere geometry, READ from the generator.

    ``scripts/build_synthetic_reveal_scene.py`` gives the static spheres no id
    constants: they are an anonymous ``STATIC_SPHERES`` tuple whose ids are
    derived positionally at render time as ``idx + 1``
    (``ids.append(idx + 1)``). This function reproduces exactly that derivation
    from the imported module, so a change to the generator's tuple is followed
    rather than silently contradicted by a hardcoded literal.
    """
    from scripts import build_synthetic_reveal_scene as gen

    spheres = []
    for index, entry in enumerate(gen.STATIC_SPHERES):
        centre, radius = entry[0], entry[1]
        spheres.append({
            "index": int(index),
            "id": int(index) + 1,
            "centre": [float(v) for v in centre],
            "radius": float(radius),
        })
    return {
        "event_object_id": int(gen.EVENT_OBJECT_ID),
        "background_id": int(gen.BACKGROUND_ID),
        "ground_id": int(gen.GROUND_ID),
        "static_spheres": spheres,
        "source": "scripts/build_synthetic_reveal_scene.py",
        "static_sphere_id_rule": "positional: ids.append(idx + 1)",
    }


def select_static_control(constants, index):
    """The static sphere the control targets. Never a guessed id."""
    spheres = constants["static_spheres"]
    if not spheres:
        raise ContractError("the generator declares no static spheres")
    index = int(index)
    if not 0 <= index < len(spheres):
        raise ContractError(
            "--static-sphere-index %d outside [0, %d)" % (index, len(spheres)))
    return dict(spheres[index])


def load_grid_definition(program_path, cells_per_axis_default=DEFAULT_CELLS_PER_AXIS):
    """``(lo, span, cells, provenance)`` from a recorded episode program.

    ``build_v2_program`` (``scripts/estimate_episodes.py``) records the
    ABSOLUTE world-space grid under ``spatial`` as ``kind``,
    ``cells_per_axis``, ``lo``, ``span``. Returns ``(None, None, default, ...)``
    when no program is supplied or found, which makes the caller recompute the
    grid over the cloud's own bounds AND say so in the report.
    """
    if not program_path:
        return None, None, int(cells_per_axis_default), {
            "grid_source": "recomputed_from_cloud_bounds",
            "reason": "no episode program supplied or discovered",
            "program_path": None,
        }
    path = Path(program_path)
    if not path.is_file():
        raise ContractError("episode program not found: %s" % path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    spatial = payload.get("spatial") or {}
    if spatial.get("kind") != "voxel_grid":
        raise ContractError(
            "episode program %s carries spatial.kind %r, not 'voxel_grid'"
            % (path, spatial.get("kind")))
    lo = [float(v) for v in spatial["lo"]]
    span = [float(v) for v in spatial["span"]]
    cells = int(spatial.get("cells_per_axis", cells_per_axis_default))
    provenance = {
        "grid_source": "episode_program",
        "program_path": str(path).replace("\\", "/"),
        "program_sha256": sha256_bytes(path.read_bytes()),
        "cells_per_axis": cells,
        "lo": lo,
        "span": span,
        "program_cloud_sha256": (payload.get("cloud") or {}).get("xyz_sha256"),
        "program_cloud_rows": (payload.get("cloud") or {}).get("n_rows"),
    }
    return lo, span, cells, provenance


def discover_episode_program(*directories):
    """First ``*.json`` in the given directories carrying a voxel_grid spatial
    block. Deterministic (sorted) and silent about anything it cannot parse."""
    for directory in directories:
        if not directory:
            continue
        root = Path(directory)
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            spatial = payload.get("spatial") if isinstance(payload, dict) else None
            if isinstance(spatial, dict) and spatial.get("kind") == "voxel_grid":
                return str(path)
    return None


def sha256_file(path):
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


# ---------------------------------------------------------------------------
# TORCH LIMB. Every import is lazy so the module stays importable, testable
# and self-testable on a workstation without torch.
# ---------------------------------------------------------------------------


def build_parser():
    from arguments import ModelParams, OptimizationParams, PipelineParams

    parser = ArgumentParser(
        description="Per-row membership supervisability on the flow carrier "
                    "channel (measurement only)")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", required=True)
    parser.add_argument("--start_checkpoint", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument(
        "--frames", default="",
        help="explicit frame set, e.g. '0-29,57-59'. Default: episode-1 plus "
             "return frames READ FROM the fixture's event_spec.json.")
    parser.add_argument(
        "--max_cameras", type=int, default=0,
        help="0 (default) uses EVERY training camera, which is what the "
             "ceiling is defined over. A positive value is a smoke-test "
             "subsample and makes every number a lower bound on the ceiling.")
    parser.add_argument(
        "--identity-dir", dest="identity_dir", default="",
        help="supervision masks; default <source_path>/train_identity. A path "
             "whose final component is gt_identity is REFUSED.")
    parser.add_argument(
        "--episode-program", dest="episode_program", default="",
        help="recorded episode program v2 carrying spatial.lo/span/"
             "cells_per_axis. Default: discover one in the run dir, then in "
             "configs/lrv3. Absent -> the 8^3 grid is recomputed over the "
             "cloud's own bounds and the report SAYS SO.")
    parser.add_argument(
        "--no-discover-program", dest="no_discover_program", action="store_true",
        help="do not auto-discover an episode program; force recomputation.")
    parser.add_argument(
        "--static-sphere-index", dest="static_sphere_index", type=int, default=0,
        help="which of the generator's STATIC_SPHERES the control targets "
             "(0-based). Its id, centre and radius are read from the "
             "generator module, never hardcoded.")
    parser.add_argument(
        "--min-camera-mask-px", dest="min_camera_mask_px", type=int, default=0,
        help="DECLARED per-camera event-pixel floor. Default 0 = REPORT ONLY: "
             "every camera's count is reported and nothing is refused. This "
             "script does not choose a scientific floor.")
    parser.add_argument(
        "--expect-fingerprint", dest="expect_fingerprint", default="",
        help="refuse unless the loaded cloud's fingerprint equals this value.")
    parser.add_argument("--hash_checkpoint", action="store_true")
    parser.add_argument("--gaussian_dim", type=int, default=4)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[0.0, 10.0])
    parser.add_argument("--num_pts", type=int, default=50_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--exhaust_test", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser, lp, op, pp


def restore_frozen_model(args, lp, op, pp):
    """Build the Scene and restore the checkpoint on the EXISTING frozen path.

    Mirrors ``main.py``'s ``validation()``: construct ``GaussianModel``, build
    ``Scene``, ``torch.load`` the checkpoint and ``restore(model_params,
    None)``. Passing ``None`` skips ``training_setup`` and the optimizer
    entirely -- nothing here ever takes a step.
    """
    import torch
    from scene import Scene
    from scene.gaussian_model import GaussianModel

    torch.manual_seed(args.seed)
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    if bool(getattr(opt, "elgs_enable", False)):
        raise ContractError(
            "elgs_enable is set: an EL-GS presence program gates rows and "
            "would change the very compositing weights this instrument "
            "measures. Point it at the substrate cell instead.")
    for key in ("event_candidate_manifest", "event_boundary_support_manifest"):
        if str(getattr(opt, key, "") or "").strip():
            raise ContractError(
                "%s must be empty: a runtime visibility gate would suppress "
                "rows and deflate every reported weight for a reason that is "
                "not about supervisability" % key)

    # The leaf substitution happens at the rasterizer call boundary, where
    # `flow_2d` arrives with one row per Gaussian. With
    # `compute_cov3D_python` the renderer PREFILTERS every per-row argument by
    # `marginal_t > 0.05` before that boundary, so the tensor reaching the
    # rasterizer is a SUBSET whose rows can no longer be mapped back. Refuse
    # rather than measure a permutation of the wrong rows.
    if bool(getattr(pipe, "compute_cov3D_python", False)):
        raise ContractError(
            "compute_cov3D_python is set: gaussian_renderer.render prefilters "
            "flow_2d by marginal_t > 0.05 before the rasterizer call, so the "
            "per-row leaf could not be bound one-to-one. Measure on the "
            "non-prefiltered path.")
    if bool(getattr(pipe, "convert_SHs_python", False)):
        raise ContractError(
            "convert_SHs_python is set: the renderer then passes "
            "colors_precomp and no SH tensor, so the static twin's SH leaf "
            "(answer 5) has nothing to bind to.")

    gaussians = GaussianModel(
        dataset.sh_degree, gaussian_dim=args.gaussian_dim,
        time_duration=args.time_duration, rot_4d=args.rot_4d,
        force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0,
    )
    scene = Scene(dataset, gaussians, shuffle=False, num_pts=args.num_pts,
                  num_pts_ratio=args.num_pts_ratio,
                  time_duration=args.time_duration)
    model_params, first_iter = torch.load(args.start_checkpoint)
    n_rows_checkpoint = int(model_params[1].shape[0])
    gaussians.restore(model_params, None)

    # Answer 5 is only MEASURABLE under soft routing. With soft routing OFF the
    # static twin is a separate parameter set (`pc.get_static_features`) and
    # there is no shared-features question to ask; with it ON both branches
    # read `_features_dc`, which is exactly the contamination the refutation
    # named. `restore(model_params, None)` skips `training_setup`, so this flag
    # comes from the checkpoint's own routing state. Disagreement with the
    # config is itself a finding and is refused, not reconciled.
    restored_routing = bool(getattr(gaussians, "enable_soft_routing", False))
    configured_routing = bool(getattr(opt, "enable_soft_routing", False))
    if not restored_routing:
        raise ContractError(
            "the checkpoint restored with enable_soft_routing=False: the "
            "static twin then reads get_static_features rather than the "
            "dynamic branch's features, so C1 -- the static-twin share -- is "
            "not a question this setup can be asked")
    if restored_routing != configured_routing:
        raise ContractError(
            "enable_soft_routing disagrees between the checkpoint (%r) and "
            "the config (%r); the restored substrate is not the one this "
            "config describes" % (restored_routing, configured_routing))

    return gaussians, scene, dataset, opt, pipe, int(first_iter), n_rows_checkpoint


def select_training_views(scene, spec, frames, max_cameras):
    """(views, camera_ids) over TRAINING cameras only.

    ``views`` is a list of ``(camera_id, frame, camera_object)`` in a
    deterministic camera-major order. Cameras are taken from
    ``scene.train_cameras[1.0]`` directly rather than from
    ``getTrainCameras()``: that returns a ``CameraDataset`` which iterates as
    ``(image, camera)`` tuples, and handing such a tuple to a roster builder
    has previously produced a SILENTLY EMPTY roster in this repository.
    """
    stack = scene.train_cameras[1.0]
    if not stack:
        raise ContractError("the train split is empty")
    wanted = set(int(f) for f in frames)

    by_camera = {}
    for camera in stack:
        name = getattr(camera, "image_name", None)
        cam_id = camera_id_of_name(name)
        frame = frame_index_of_name(name)
        if frame is None:
            raise ContractError(
                "cannot parse a frame index from image_name %r" % (name,))
        by_camera.setdefault(cam_id, {})[frame] = camera

    available = sorted(by_camera)
    assert_cameras_are_training(
        available, spec.get("train_cameras"), spec.get("test_cameras"))

    chosen = available
    if int(max_cameras) > 0 and int(max_cameras) < len(available):
        step = len(available) / float(int(max_cameras))
        chosen = [available[int(i * step)] for i in range(int(max_cameras))]

    views, missing = [], []
    for cam_id in chosen:
        for frame in sorted(wanted):
            camera = by_camera[cam_id].get(frame)
            if camera is None:
                missing.append((cam_id, frame))
                continue
            views.append((cam_id, frame, camera))
    if missing:
        raise ContractError(
            "the train split is missing %d requested (camera, frame) views, "
            "first few %r" % (len(missing), missing[:5]))
    if not views:
        raise ContractError("no training views matched the requested frame set")
    return views, chosen


@contextlib.contextmanager
def bind_render_leaves(flow_leaf, sh_leaf, sh_static_leaf, n_rows, record):
    """Bind ``flow_2d`` and both SH inputs to LEAVES at the rasterizer boundary.

    ``gaussian_renderer.render`` builds ``flow_2d`` itself -- either from
    projected motion or as ``torch.zeros_like(pc.get_xyz[:, :2])`` -- so it is
    never a leaf and never carries a gradient path back to anything this
    instrument can read. Rather than reimplement ``render`` (which would risk
    silently diverging from the training path in exactly the opacity chain the
    measurement depends on), this installs a TEST DOUBLE at the one call
    boundary where the tensors are handed to the rasterizer, and substitutes
    the three arguments. Everything upstream -- the temporal marginal, soft
    routing, the opacity chain, the runtime gates -- is computed by the real
    ``render``. No repository file is modified; the symbol is restored on exit.

    Each interception VERIFIES, and refuses rather than silently proceeding:

    * ``flow_2d`` arrives with one row per Gaussian (a prefiltered subset would
      make the per-row binding a lie);
    * the dynamic and static row counts both equal the cloud's;
    * both SH tensors arrive (the ``colors_precomp`` path is refused earlier);
    * the static twin reads the SAME feature values as the dynamic branch,
      which is the fact answer 5 exists to quantify.
    """
    import torch
    import gaussian_renderer as gr

    real = gr.GaussianRasterizer

    class _LeafBindingRasterizer:
        def __init__(self, raster_settings):
            self._inner = real(raster_settings=raster_settings)

        def __call__(self, **kwargs):
            flow = kwargs.get("flow_2d")
            if flow is None:
                raise ContractError(
                    "the renderer passed no flow_2d; the membership carrier "
                    "channel is not being fed at all")
            shape = tuple(int(v) for v in flow.shape)
            if shape != (int(n_rows), 2):
                raise ContractError(
                    "flow_2d reached the rasterizer with shape %r, not "
                    "(%d, 2): a prefilter has subset the rows and the per-row "
                    "leaf binding would not be one-to-one" % (shape, n_rows))
            for key in ("means3D", "opacities", "means3D_static", "opacities_static"):
                tensor = kwargs.get(key)
                if tensor is None or int(tensor.shape[0]) != int(n_rows):
                    raise ContractError(
                        "%s reached the rasterizer with %r rows, not %d"
                        % (key, None if tensor is None else tensor.shape[0], n_rows))
            shs, shs_static = kwargs.get("shs"), kwargs.get("shs_static")
            if shs is None or shs_static is None:
                raise ContractError(
                    "the renderer passed no SH tensor for one branch; the "
                    "separate-leaf static share cannot be measured")
            if tuple(shs.shape) != tuple(sh_leaf.shape):
                raise ContractError(
                    "shs shape %r does not match the bound leaf %r"
                    % (tuple(shs.shape), tuple(sh_leaf.shape)))
            if tuple(shs_static.shape) != tuple(sh_static_leaf.shape):
                raise ContractError(
                    "shs_static shape %r does not match the bound leaf %r"
                    % (tuple(shs_static.shape), tuple(sh_static_leaf.shape)))
            record["shares_features"] = bool(
                record.get("shares_features", True)
                and bool(torch.equal(shs.detach(), shs_static.detach())))
            record["calls"] = int(record.get("calls", 0)) + 1

            kwargs["flow_2d"] = flow_leaf
            kwargs["shs"] = sh_leaf
            kwargs["shs_static"] = sh_static_leaf
            return self._inner(**kwargs)

    gr.GaussianRasterizer = _LeafBindingRasterizer
    try:
        yield
    finally:
        gr.GaussianRasterizer = real


def measure(gaussians, views, pipe, background, class_ids, mask_source,
            check_repeat_at, verbose=False):
    """One forward per view; several backward passes per forward.

    Returns per-row accumulators plus the observations the mechanism
    preconditions read.
    """
    import torch
    from gaussian_renderer import render

    n_rows = int(gaussians._xyz.shape[0])
    device = gaussians._xyz.device
    features = gaussians.get_features.detach()

    flow_leaf = torch.ones((n_rows, 2), dtype=torch.float32,
                           device=device, requires_grad=True)
    check_flow_binding(flow_leaf, n_rows)
    # TWO independent leaves holding the SAME values. Because they are
    # separate leaves the rasterizer's `grad_sh` and `grad_sh_static` no
    # longer sum into one tensor, which is precisely what made C1
    # unmeasurable before.
    sh_leaf = features.clone().requires_grad_(True)
    sh_static_leaf = features.clone().requires_grad_(True)

    n_classes = len(class_ids)
    w_total = torch.zeros(n_rows, dtype=torch.float64, device=device)
    w_in_mask = torch.zeros((n_rows, n_classes), dtype=torch.float64, device=device)
    w_sh_dynamic = torch.zeros(n_rows, dtype=torch.float64, device=device)
    w_sh_static = torch.zeros(n_rows, dtype=torch.float64, device=device)
    w_sh_dynamic_dc = torch.zeros(n_rows, dtype=torch.float64, device=device)

    per_view = []
    n_views_nonzero_image = 0
    max_partition_rel_dev = 0.0
    repeat_checks = []
    record = {"calls": 0, "shares_features": True}

    with bind_render_leaves(flow_leaf, sh_leaf, sh_static_leaf, n_rows, record):
        for index, (cam_id, frame, camera) in enumerate(views):
            identity = mask_source(cam_id, frame)          # int16 (H, W) on device
            camera_on_device = camera.cuda() if torch.cuda.is_available() else camera
            out = render(camera_on_device, gaussians, pipe, background)
            flow_image = out["flow"]
            image = out["render"]
            if tuple(flow_image.shape[-2:]) != tuple(identity.shape[-2:]):
                raise ContractError(
                    "identity buffer %r does not match the rendered image %r "
                    "for cam%02d f%03d"
                    % (tuple(identity.shape), tuple(flow_image.shape[-2:]),
                       cam_id, frame))
            if not bool(flow_image.requires_grad):
                raise ContractError(
                    "the rendered flow image carries no gradient path; the "
                    "flow_2d leaf binding did not take effect")

            image_abs_sum = float(image.detach().abs().sum())
            if image_abs_sum > 0.0:
                n_views_nonzero_image += 1

            def flow_grad(upstream_hw):
                """d(sum_pixels upstream * M_0) / d(flow_leaf[:, 0]).

                `torch.autograd.grad` RETURNS the gradient instead of
                accumulating into `.grad`, so no clearing discipline exists to
                get wrong and no pass can contaminate another.
                """
                grad_out = torch.zeros_like(flow_image)
                grad_out[0] = upstream_hw
                (grad,) = torch.autograd.grad(
                    outputs=flow_image, inputs=(flow_leaf,),
                    grad_outputs=grad_out, retain_graph=True, allow_unused=False)
                return grad[:, 0]

            ones_hw = torch.ones(flow_image.shape[-2:], dtype=flow_image.dtype,
                                 device=flow_image.device)
            view_total = flow_grad(ones_hw)

            view_class = []
            for class_id in class_ids:
                mask = (identity == int(class_id)).to(flow_image.dtype)
                view_class.append(flow_grad(mask))
            stacked = torch.stack(view_class, dim=1)

            # P10: the identity buffer partitions the image, so the class
            # weights must re-sum to the independently measured total. A
            # failure means a mask, the class list or an upstream gradient is
            # wrong -- it cannot be read as a result.
            summed = stacked.sum(dim=1)
            denominator = view_total.abs().clamp_min(1e-30)
            rel_dev = float(((summed - view_total).abs() / denominator).max())
            max_partition_rel_dev = max(max_partition_rel_dev, rel_dev)

            if index in check_repeat_at:
                # P11: repeat the g = 1 backward AFTER every class backward on
                # the same retained graph. Several backwards over one forward
                # is this instrument's load-bearing trick; this is the
                # observation that the retained graph is not being consumed or
                # corrupted, not a restatement of the intent.
                repeat = flow_grad(ones_hw)
                repeat_checks.append({
                    "view_index": int(index),
                    "camera": int(cam_id), "frame": int(frame),
                    "bitwise_identical": bool(torch.equal(repeat, view_total)),
                    "max_abs_difference": float((repeat - view_total).abs().max()),
                })

            grad_dyn, grad_stat = torch.autograd.grad(
                outputs=image, inputs=(sh_leaf, sh_static_leaf),
                grad_outputs=torch.ones_like(image),
                retain_graph=True, allow_unused=False)

            w_total += view_total.detach().to(torch.float64)
            w_in_mask += stacked.detach().to(torch.float64)
            w_sh_dynamic += grad_dyn.detach().abs().reshape(n_rows, -1).sum(
                dim=1).to(torch.float64)
            w_sh_static += grad_stat.detach().abs().reshape(n_rows, -1).sum(
                dim=1).to(torch.float64)
            w_sh_dynamic_dc += grad_dyn.detach()[:, 0, :].abs().sum(
                dim=1).to(torch.float64)

            per_view.append({
                "camera": int(cam_id), "frame": int(frame),
                "image_abs_sum": image_abs_sum,
                "rows_with_nonzero_weight": int((view_total.detach() != 0).sum()),
                "partition_rel_deviation": rel_dev,
            })
            if verbose and (index % 50 == 0 or index == len(views) - 1):
                print("  view %d/%d cam%02d f%03d: %d rows carry weight"
                      % (index + 1, len(views), cam_id, frame,
                         per_view[-1]["rows_with_nonzero_weight"]))

            del out, flow_image, image, view_total, stacked, view_class

    return {
        "n_rows": n_rows,
        "w_total": w_total,
        "w_in_mask": w_in_mask,
        "w_sh_dynamic": w_sh_dynamic,
        "w_sh_static": w_sh_static,
        "w_sh_dynamic_dc": w_sh_dynamic_dc,
        "per_view": per_view,
        "n_views_nonzero_image": n_views_nonzero_image,
        "max_partition_rel_deviation": max_partition_rel_dev,
        "repeat_checks": repeat_checks,
        "n_rasterizer_calls": int(record["calls"]),
        "static_branch_shares_features": bool(record["shares_features"]),
        "flow_leaf_shape": [int(v) for v in flow_leaf.shape],
        "sh_leaf_shape": [int(v) for v in sh_leaf.shape],
    }


def clamp_exposure_block(gaussians):
    """How many rows the rasterizer's RGB clamp could zero at DC order.

    Relevant ONLY to answer 5, which is measured on the SH channel because the
    static branch exposes no flow handle. The flow-derived ``w_total`` that
    every other answer rests on runs through no SH evaluation and is therefore
    clamp-free by construction.
    """
    import torch
    from utils.sh_utils import C0

    with torch.no_grad():
        dc = gaussians._features_dc.detach().reshape(
            int(gaussians._features_dc.shape[0]), -1)
        rgb = float(C0) * dc.to(torch.float64) + 0.5
        nonpositive = rgb <= 0.0
        any_channel = int(nonpositive.any(dim=1).sum())
        all_channels = int(nonpositive.all(dim=1).sum())
    return {
        "sh_c0": float(C0),
        "expression": "SH_C0 * _features_dc + 0.5 (order-0 radiance)",
        "rows_nonpositive_in_any_channel": any_channel,
        "rows_nonpositive_in_all_channels": all_channels,
        "dc_order_clamp_can_zero_a_row": bool(all_channels > 0),
        "applies_to": "answer 5 (the SH-mediated static share) only",
    }


# ---------------------------------------------------------------------------
# SELF TEST -- runs the pure limbs only, so it works without torch.
# ---------------------------------------------------------------------------


def self_test():
    checks = []

    def check(label, condition):
        checks.append((label, bool(condition)))

    # -- the sphere test ---------------------------------------------------
    flags = in_sphere_flags(
        [[0.0, 0.0, 0.0],       # centre, inside
         [0.5, 0.0, 0.0],       # EXACTLY at the radius, inclusive
         [0.0, -0.5, 0.0],      # exactly at the radius on another axis
         [0.5, 0.5, 0.0],       # outside
         [0.25, 0.25, 0.25]],   # inside
        [0.0, 0.0, 0.0], 0.5)
    check("sphere: centre is inside", bool(flags[0]))
    check("sphere: exactly at radius is INSIDE (<=)", bool(flags[1]))
    check("sphere: exactly at radius on -y is INSIDE", bool(flags[2]))
    check("sphere: outside is outside", not bool(flags[3]))
    check("sphere: 4 of 5 points inside", int(flags.sum()) == 4)

    centre, radius = [0.7, 0.1, 0.35], 0.2
    points = [[0.7, 0.1, 0.35], [0.7, 0.1, 0.475],
              [0.7, 0.1, 0.6], [0.0, 0.0, 0.0]]
    offset = in_sphere_flags(points, centre, radius)
    check("sphere: offset centre, 2 of 4 inside", int(offset.sum()) == 2)
    check("sphere: offset centre keeps ordering",
          offset.tolist() == [True, True, False, False])
    try:
        in_sphere_flags(points, centre, -1.0)
        check("sphere: refuses a negative radius", False)
    except ContractError:
        check("sphere: refuses a negative radius", True)
    check("sphere: radius 0 admits only the exact centre",
          int(in_sphere_flags(points, centre, 0.0).sum()) == 1)

    # -- the ceiling CURVE -------------------------------------------------
    weights = [0.0, 0.0, 1e-9, 1e-4, 1e-2, 0.5, 2.0]
    curve = ceiling_curve(weights)
    check("ceiling: strict w>0 counts 5 of 7",
          curve["strict_positive"]["n_at_or_above"] == 5)
    check("ceiling: strict w>0 gives 5/7",
          abs(curve["strict_positive"]["achievable_recall_ceiling"] - 5.0 / 7.0) < 1e-15)
    check("ceiling: the e_min entry 0.0 reads STRICTLY",
          [e for e in curve["curve"] if e["e_min"] == 0.0][0]["n_at_or_above"] == 5)
    at_1e2 = [e for e in curve["curve"] if e["e_min"] == 1e-2][0]
    check("ceiling: e_min 1e-2 admits 3 rows", at_1e2["n_at_or_above"] == 3)
    check("ceiling: e_min 1e-2 recall is 3/7",
          abs(at_1e2["achievable_recall_ceiling"] - 3.0 / 7.0) < 1e-15)
    counts = [e["n_at_or_above"] for e in curve["curve"]]
    check("ceiling: counts are non-increasing in e_min",
          all(b <= a for a, b in zip(counts, counts[1:])))
    check("ceiling: every entry partitions the target",
          all(e["n_at_or_above"] + e["n_below"] == 7 for e in curve["curve"]))
    check("ceiling: a curve is reported, not a scalar", len(curve["curve"]) > 1)
    check("ceiling: the FROZEN absolute grid is exactly the declared six",
          [e["e_min"] for e in curve["curve"]]
          == [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0])
    check("ceiling: an EMPTY target returns None, never 0.0 or 1.0",
          ceiling_curve([])["strict_positive"]["achievable_recall_ceiling"] is None)
    try:
        ceiling_curve([-1e-30])
        check("ceiling: refuses a negative weight", False)
    except ContractError:
        check("ceiling: refuses a negative weight", True)

    # -- the ceiling's QUANTILE limb, taken over ALL rows ------------------
    # 100 cloud rows: 60 dead, then 1..40. Quantiles of THIS are the cuts; the
    # target below is a different, deliberately better-supervised subset, so if
    # the cuts were (wrongly) taken from the target the numbers would differ
    # visibly and the checks below would fail.
    all_rows = [0.0] * 60 + [float(v) for v in range(1, 41)]
    points = quantile_e_min_points(all_rows)
    check("ceiling quantiles: one e_min per declared q",
          [p["q"] for p in points] == list(CEILING_QUANTILE_GRID))
    check("ceiling quantiles: the frozen q grid is 0.50/0.75/0.90/0.95/0.99",
          list(CEILING_QUANTILE_GRID) == [0.50, 0.75, 0.90, 0.95, 0.99])
    check("ceiling quantiles: q50 of a half-dead cloud is 0.0",
          points[0]["e_min"] == 0.0)
    check("ceiling quantiles: e_min values are non-decreasing in q",
          all(b["e_min"] >= a["e_min"] for a, b in zip(points, points[1:])))
    check("ceiling quantiles: an EMPTY cloud fabricates no cut",
          quantile_e_min_points([]) == [])
    try:
        quantile_e_min_points([-1.0])
        check("ceiling quantiles: refuse a negative weight", False)
    except ContractError:
        check("ceiling quantiles: refuse a negative weight", True)

    target_subset = [30.0, 36.0, 40.0, 0.0]
    with_quantiles = ceiling_curve(target_subset, CEILING_E_MIN_GRID, points)
    check("ceiling quantiles: the limb is reported alongside the absolute grid",
          len(with_quantiles["quantile_curve"]) == len(CEILING_QUANTILE_GRID)
          and len(with_quantiles["curve"]) == len(CEILING_E_MIN_GRID))
    check("ceiling quantiles: each limb entry carries its q AND its provenance",
          all(("q" in e and "ALL rows" in e["e_min_source"])
              for e in with_quantiles["quantile_curve"]))
    check("ceiling quantiles: the limb reads the TARGET's rows against the "
          "CLOUD's cuts",
          all(e["n_at_or_above"] + e["n_below"] == 4
              for e in with_quantiles["quantile_curve"]))
    # THE LEAKAGE CHECK. The cloud is 60% dead, so its q50 is 0.0 and all four
    # target rows clear it. The TARGET's own q50 would be 33.0, which only two
    # of them clear. Reading 4 here is therefore positive evidence that the cut
    # came from the cloud; reading 2 would mean the oracle had entered the cut.
    check("ceiling quantiles: the cloud's q50 is 0.0 on a 60%-dead cloud",
          points[0]["e_min"] == 0.0)
    check("ceiling quantiles: at the CLOUD's q50 the whole target clears",
          with_quantiles["quantile_curve"][0]["n_at_or_above"] == 4)
    check("ceiling quantiles: ... whereas the TARGET's own q50 would admit "
          "only 2, so the cut demonstrably did not come from the target",
          int(np.sum(np.asarray(target_subset)
                     >= float(np.quantile(target_subset, 0.50)))) == 2)
    check("ceiling quantiles: at the cloud's q99 only the top target row clears",
          with_quantiles["quantile_curve"][-1]["n_at_or_above"] == 1)
    check("ceiling quantiles: omitting the limb leaves it EMPTY, never faked",
          ceiling_curve(target_subset)["quantile_curve"] == [])
    check("ceiling quantiles: the provenance says the cuts avoid the target",
          "never over" in with_quantiles["quantile_curve_provenance"])

    # -- the w_total distribution -----------------------------------------
    dist = weight_distribution(weights)
    check("distribution: counts the exact zeros", dist["n_exactly_zero"] == 2)
    check("distribution: min is 0.0", dist["min"] == 0.0)
    check("distribution: max is 2.0", dist["max"] == 2.0)
    check("distribution: median is 1e-4", dist["quantiles"]["p050"] == 1e-4)
    for q in WEIGHT_QUANTILES:
        check("distribution: reports p%03d" % int(round(q * 100)),
              ("p%03d" % int(round(q * 100))) in dist["quantiles"])
    check("distribution: an empty input yields all None",
          weight_distribution([])["quantiles"]["p050"] is None)
    share = quantiles_of([0.0, 0.25, 0.5, 0.75, 1.0], SHARE_QUANTILES)
    check("share quantiles: p050 is 0.5", share["p050"] == 0.5)
    check("share quantiles: p100 is 1.0", share["p100"] == 1.0)
    check("share quantiles: q90/q95/q99 are all reported",
          all(k in share for k in ("p090", "p095", "p099")))
    try:
        quantiles_of([0.0, 1.0], (1.5,))
        check("share quantiles: refuse q outside [0, 1]", False)
    except ContractError:
        check("share quantiles: refuse q outside [0, 1]", True)

    # -- the closed-form vote ---------------------------------------------
    # 12 rows, 3 classes. Rows 0-3 are truly the event object.
    class_ids = [-1, 0, 100]
    truth = [True] * 4 + [False] * 8
    scores = ([[0.1, 0.2, 5.0]] * 4        # event rows: class 100 dominates
              + [[7.0, 0.3, 0.1]] * 4      # background rows
              + [[0.2, 9.0, 0.1]] * 3      # ground rows
              + [[0.0, 0.0, 0.0]])         # a row with no evidence at all
    # w_total is the row sum, exactly as P10's partition identity guarantees on
    # a real pass: sum_k w_in_mask_k == w_total.
    w_total_rows = [float(sum(row)) for row in scores]
    assigned, stats = argmax_vote(class_ids, scores, w_total_rows)
    predicted = (assigned == 100)
    metrics = precision_recall(predicted, truth)
    check("vote: the event rows are assigned class 100",
          assigned[:4].tolist() == [100] * 4)
    check("vote: an evidence-free row ABSTAINS, never wins by tie-break",
          int(assigned[-1]) == ABSTAIN_CLASS)
    check("vote: one abstention is counted", stats["n_abstained"] == 1)
    check("vote: the evidence-free row abstains on ELIGIBILITY (w_total = 0), "
          "not on tau", stats["n_abstained_ineligible"] == 1
          and stats["n_abstained_below_tau"] == 0)
    check("vote: the frozen tau is reported with the result",
          stats["tau"] == VOTE_TAU and VOTE_TAU == 0.50)
    check("vote: eleven of twelve rows are eligible and assigned",
          stats["n_eligible"] == 11 and stats["n_assigned"] == 11)
    check("vote: no row was assigned on a zero best score",
          stats["n_assigned_with_zero_best"] == 0)
    check("vote: precision is 1.0", metrics["precision"] == 1.0)
    check("vote: recall is 1.0", metrics["recall"] == 1.0)
    check("vote: the confusion cells add up",
          metrics["tp"] + metrics["fp"] + metrics["fn"] + metrics["tn"] == 12)
    check("vote: clears the commissioned reference here",
          clears_reference(metrics) is True)
    check("vote: an undefined precision does not clear anything",
          clears_reference({"precision": None, "recall": 1.0}) is None)
    try:
        argmax_vote([-1, -1], [[1.0, 2.0]], [3.0])
        check("vote: refuses duplicate class ids", False)
    except ContractError:
        check("vote: refuses duplicate class ids", True)
    try:
        argmax_vote(class_ids, [[1.0, -1.0, 0.0]], [0.0])
        check("vote: refuses a negative class score", False)
    except ContractError:
        check("vote: refuses a negative class score", True)
    try:
        argmax_vote([ABSTAIN_CLASS, 0], [[1.0, 2.0]], [3.0])
        check("vote: refuses a class colliding with the abstain sentinel", False)
    except ContractError:
        check("vote: refuses a class colliding with the abstain sentinel", True)
    try:
        argmax_vote(class_ids, [[1.0, 2.0, 3.0]], [6.0, 6.0])
        check("vote: refuses a w_total of the wrong length", False)
    except ContractError:
        check("vote: refuses a w_total of the wrong length", True)
    try:
        argmax_vote(class_ids, [[1.0, 2.0, 3.0]], [-6.0])
        check("vote: refuses a negative w_total", False)
    except ContractError:
        check("vote: refuses a negative w_total", True)
    for bad_tau in (-0.1, 1.1):
        try:
            argmax_vote(class_ids, [[1.0, 2.0, 3.0]], [6.0], tau=bad_tau)
            check("vote: refuses tau %r" % bad_tau, False)
        except ContractError:
            check("vote: refuses tau %r" % bad_tau, True)

    # -- THE FROZEN OPERATING POINT actually bites --------------------------
    # Three rows, all with w_total = 1.0, whose best-class SHARE straddles 0.50:
    # 0.80 (assigned at every tau up to 0.75), 0.45 (assigned only below 0.50)
    # and 0.34 (assigned only at tau = 0.25 or below). If the tau rule were
    # dropped and the vote fell back to bare argmax, all three would be
    # assigned at every tau and the sweep below would be flat.
    straddle_scores = [[0.80, 0.15, 0.05],
                       [0.45, 0.35, 0.20],
                       [0.34, 0.33, 0.33]]
    straddle_total = [1.0, 1.0, 1.0]
    at_frozen, frozen_stats = argmax_vote([-1, 0, 100], straddle_scores,
                                          straddle_total)
    check("tau rule: at the frozen 0.50 only the 0.80-share row is assigned",
          frozen_stats["n_assigned"] == 1
          and int(at_frozen[0]) == -1
          and int(at_frozen[1]) == ABSTAIN_CLASS
          and int(at_frozen[2]) == ABSTAIN_CLASS)
    check("tau rule: the two rejects abstain BELOW TAU, not on eligibility",
          frozen_stats["n_abstained_below_tau"] == 2
          and frozen_stats["n_abstained_ineligible"] == 0)
    assigned_by_tau = [
        argmax_vote([-1, 0, 100], straddle_scores, straddle_total,
                    tau=t)[1]["n_assigned"]
        for t in VOTE_TAU_GRID]
    check("tau rule: the sweep is 3/3/1/1/1/0 -- it BITES, it is not flat",
          assigned_by_tau == [3, 3, 1, 1, 1, 0])
    check("tau rule: assignments are non-increasing in tau",
          all(b <= a for a, b in zip(assigned_by_tau, assigned_by_tau[1:])))
    check("tau rule: the frozen grid is 0/0.25/0.50/2-thirds/0.75/0.90",
          [round(t, 4) for t in VOTE_TAU_GRID]
          == [0.0, 0.25, 0.5, 0.6667, 0.75, 0.9])
    # The rule is a SHARE, so scaling a row's whole supply changes nothing.
    scaled, _ = argmax_vote([-1, 0, 100],
                            [[v * 1e-9 for v in row] for row in straddle_scores],
                            [1e-9, 1e-9, 1e-9])
    check("tau rule: it is scale-free -- shrinking every weight 1e9x does not "
          "change one assignment", scaled.tolist() == at_frozen.tolist())

    tau_curve = vote_tau_curve([-1, 0, 100], straddle_scores, straddle_total,
                               [True, True, True], -1)
    check("tau curve: one entry per declared tau",
          [e["tau"] for e in tau_curve["curve"]] == list(VOTE_TAU_GRID))
    check("tau curve: exactly one entry is flagged the frozen operating point",
          sum(1 for e in tau_curve["curve"]
              if e["is_frozen_operating_point"]) == 1)
    check("tau curve: it is LABELLED ceiling information, never the score",
          "CEILING INFORMATION ONLY" in tau_curve["reading"])
    check("tau curve: recall falls as tau rises (3/3 -> 1/3 -> 0/3)",
          [e["recall"] for e in tau_curve["curve"]]
          == [1.0, 1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0])

    # -- NEUTER (a): a vote computed from w_total DISCARDS class information
    # and must collapse. w_total is one number per row, so replicating it
    # across the class columns ties every class on every row and the whole
    # decision falls to the tie-break. Both orderings are exercised because
    # which way it collapses depends on where the event class sits.
    neutered_scores = [[value] * len(class_ids) for value in w_total_rows]
    event_last, _ = argmax_vote([-1, 0, 100], neutered_scores, w_total_rows)
    last_metrics = precision_recall(event_last == 100, truth)
    event_first, neutered_stats = argmax_vote([100, -1, 0], neutered_scores,
                                              w_total_rows)
    first_metrics = precision_recall(event_first == 100, truth)
    check("NEUTER a: a w_total vote with the event class FIRST collapses "
          "precision to the base rate",
          first_metrics["precision"] is not None
          and first_metrics["precision"] <= 0.4
          and first_metrics["precision"] < metrics["precision"])
    check("NEUTER a: ... and it is detected as a collapse, not a pass",
          clears_reference(first_metrics) is False)
    check("NEUTER a: a w_total vote with the event class LAST collapses recall "
          "to zero", last_metrics["recall"] == 0.0)
    check("NEUTER a: ... predicting no event row at all, so its precision is "
          "UNDEFINED rather than 0.0",
          last_metrics["n_predicted_positive"] == 0
          and last_metrics["precision"] is None)
    check("NEUTER a: ... and it certainly does not clear the reference",
          clears_reference(last_metrics) is not True)
    check("NEUTER a: the CORRECT w_in_mask vote is unaffected",
          metrics["precision"] == 1.0 and metrics["recall"] == 1.0)
    check("NEUTER a: ... and the tau rule cannot rescue it, because a w_total "
          "vote's best score IS w_total, so every eligible row clears every "
          "tau <= 1 and the whole decision is the tie-break",
          neutered_stats["n_assigned"] == 11
          and neutered_stats["n_abstained_below_tau"] == 0
          and neutered_stats["n_tied"] == 11)

    # -- precision/recall edge cases ---------------------------------------
    empty = precision_recall([False, False], [False, False])
    check("metrics: no prediction and no truth gives None, not 0.0",
          empty["precision"] is None and empty["recall"] is None)
    none_predicted = precision_recall([False, False], [True, False])
    check("metrics: recall 0.0 with an undefined precision",
          none_predicted["recall"] == 0.0 and none_predicted["precision"] is None)
    try:
        precision_recall([True], [True, False])
        check("metrics: refuses a length mismatch", False)
    except ContractError:
        check("metrics: refuses a length mismatch", True)

    # -- the voxel grid ----------------------------------------------------
    check("grid: 420 decodes to (6, 4, 4)", decode_cell_key(420) == (6, 4, 4))
    check("grid: 429 decodes to (6, 5, 5)", decode_cell_key(429) == (6, 5, 5))
    check("grid: 364/365 decode to the two accepted cells",
          decode_cell_key(364) == (5, 5, 4) and decode_cell_key(365) == (5, 5, 5))
    unit = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]],
                    dtype=np.float32)
    lo, span, keys = voxel_keys(unit)
    check("grid: the min corner lands in cell 0", int(keys[0]) == 0)
    check("grid: the MAX corner is clamped to (7, 7, 7) = 511",
          int(keys[1]) == 511)
    check("grid: the centre lands in (4, 4, 4) = 292", int(keys[2]) == 292)
    check("grid: recomputed lo is the per-axis min", lo.tolist() == [0.0, 0.0, 0.0])
    check("grid: recomputed span is the per-axis extent",
          span.tolist() == [1.0, 1.0, 1.0])
    lo2, span2, keys2 = voxel_keys(unit, lo=[0.0, 0.0, 0.0], span=[2.0, 2.0, 2.0])
    check("grid: a SUPPLIED lo/span is used, not the points' own bounds",
          int(keys2[1]) == (4 * 64 + 4 * 8 + 4))
    check("grid: a supplied lo/span is echoed back", span2.tolist() == [2.0, 2.0, 2.0])
    outside = voxel_keys(np.array([[-5.0, -5.0, -5.0]], dtype=np.float32),
                         lo=[0.0, 0.0, 0.0], span=[1.0, 1.0, 1.0])[2]
    check("grid: a point below the supplied box clamps into cell 0",
          int(outside[0]) == 0)
    try:
        voxel_keys(unit, lo=[0.0, 0.0, 0.0], span=[0.0, 1.0, 1.0])
        check("grid: refuses a degenerate supplied span", False)
    except ContractError:
        check("grid: refuses a degenerate supplied span", True)
    try:
        voxel_keys(unit, cells_per_axis=0)
        check("grid: refuses cells_per_axis < 1", False)
    except ContractError:
        check("grid: refuses cells_per_axis < 1", True)

    # -- per-cell restriction ---------------------------------------------
    cell_mask = [True] * 4 + [False] * 8
    restricted = restricted_metrics(cell_mask, w_total_rows, predicted, truth)
    check("per-cell: the restriction counts its own rows",
          restricted["n_rows_in_restriction"] == 4)
    check("per-cell: the restriction counts its own TARGET rows",
          restricted["n_target_rows_in_restriction"] == 4)
    check("per-cell: a restricted ceiling is computed over those rows only",
          restricted["ceiling"]["n_target_rows"] == 4)
    shared_cut = restricted_metrics(cell_mask, w_total_rows, predicted, truth,
                                    quantile_points=points)
    check("per-cell: a cell is read against the CLOUD's e_min values, so two "
          "cells are comparable",
          [e["e_min"] for e in shared_cut["ceiling"]["quantile_curve"]]
          == [p["e_min"] for p in points])
    check("per-cell: the restricted vote is perfect here",
          restricted["vote"]["precision"] == 1.0
          and restricted["vote"]["recall"] == 1.0)
    empty_cell = restricted_metrics([False] * 12, w_total_rows, predicted, truth)
    check("per-cell: an EMPTY cell reports None, not a fabricated ratio",
          empty_cell["ceiling"]["strict_positive"]["achievable_recall_ceiling"] is None
          and empty_cell["vote"]["precision"] is None)
    try:
        restricted_metrics([True], w_total_rows, predicted, truth)
        check("per-cell: refuses ragged inputs", False)
    except ContractError:
        check("per-cell: refuses ragged inputs", True)

    # -- the static-sphere control reads the GENERATOR, not a literal ------
    constants = generator_identity_constants()
    check("static control: the event object id comes from the generator",
          constants["event_object_id"] == 100)
    check("static control: three static spheres are declared",
          len(constants["static_spheres"]) == 3)
    check("static control: their ids are the positional idx + 1",
          [s["id"] for s in constants["static_spheres"]] == [1, 2, 3])
    check("static control: no static id collides with the event id",
          constants["event_object_id"] not in
          {s["id"] for s in constants["static_spheres"]})
    chosen = select_static_control(constants, 0)
    check("static control: index 0 selects id 1", chosen["id"] == 1)
    check("static control: it carries a positive radius", chosen["radius"] > 0.0)
    check("static control: it carries a 3-vector centre", len(chosen["centre"]) == 3)
    for bad in (-1, len(constants["static_spheres"])):
        try:
            select_static_control(constants, bad)
            check("static control: refuses index %d" % bad, False)
        except ContractError:
            check("static control: refuses index %d" % bad, True)

    # -- the flow-leaf binding check --------------------------------------
    class _Stand:
        def __init__(self, shape, requires_grad, is_leaf):
            self.shape = shape
            self.requires_grad = requires_grad
            self.is_leaf = is_leaf

    check("binding: a real leaf of the right shape is admitted",
          check_flow_binding(_Stand((10, 2), True, True), 10) is True)
    for label, stand in (
            ("wrong shape", _Stand((10, 3), True, True)),
            ("wrong row count", _Stand((9, 2), True, True)),
            ("requires_grad False", _Stand((10, 2), False, True)),
            ("NON-LEAF", _Stand((10, 2), True, False))):
        try:
            check_flow_binding(stand, 10)
            check("binding: refuses a %s flow_2d" % label, False)
        except ContractError:
            check("binding: refuses a %s flow_2d" % label, True)

    # -- NEUTER (b): a non-leaf flow_2d delivers NO gradient, so every
    # accumulated weight reads zero and P1 must fire.
    starved = ceiling_curve([0.0] * 84)
    check("NEUTER b: a starved run reads ceiling 0.0, not None",
          starved["strict_positive"]["achievable_recall_ceiling"] == 0.0)
    _, starved_failures = evaluate_preconditions(**_healthy_precondition_kwargs(
        n_rows_nonzero_w_total=0))
    check("NEUTER b: a starved run FAILS P1_render_ran",
          starved_failures == ["P1_render_ran"])
    starved_vote, starved_stats = argmax_vote(
        [-1, 0, 100], [[0.0, 0.0, 0.0]] * 5, [0.0] * 5)
    check("NEUTER b: a starved run's vote abstains on every row",
          set(starved_vote.tolist()) == {ABSTAIN_CLASS})
    check("NEUTER b: ... on ELIGIBILITY, so no row is assigned by tie-break "
          "even at tau = 0",
          starved_stats["n_abstained_ineligible"] == 5
          and argmax_vote([-1, 0, 100], [[0.0, 0.0, 0.0]] * 5, [0.0] * 5,
                          tau=0.0)[1]["n_assigned"] == 0)
    check("NEUTER b: ... and the starved tau curve is flat at zero recall",
          all(e["recall"] == 0.0 for e in vote_tau_curve(
              [-1, 0, 100], [[0.0, 0.0, 0.0]] * 5, [0.0] * 5,
              [True] * 5, 100)["curve"]))

    # -- the per-camera supply floor --------------------------------------
    supply = {0: 8005, 1: 8201, 13: 316, 14: 16}
    check("supply: the DEFAULT floor 0 flags nobody (report only)",
          cameras_below_supply_floor(supply, 0) == [])
    check("supply: a declared floor flags the starved cameras",
          cameras_below_supply_floor(supply, 1000) == [13, 14])
    check("supply: a floor above every camera flags every camera",
          cameras_below_supply_floor(supply, 100000) == [0, 1, 13, 14])
    try:
        cameras_below_supply_floor(supply, -1)
        check("supply: refuses a negative floor", False)
    except ContractError:
        check("supply: refuses a negative floor", True)

    # -- frame handling ----------------------------------------------------
    spec = {"presence_frames": {"episode_1": [0, 29], "gap": [30, 56],
                                "episode_2": [57, 59]},
            "return_frames": [57, 58, 59],
            "event_object": {"centre": [0.7, 0.1, 0.35], "radius": 0.2},
            "train_cameras": [0, 1, 3], "test_cameras": [2, 7]}
    frames = default_frame_set(spec)
    check("frames: default is 33 frames", len(frames) == 33)
    check("frames: default is episode 1 plus the return",
          frames == list(range(0, 30)) + [57, 58, 59])
    windows = presence_windows(spec)
    check("frames: the gap window is excluded from presence", (30, 56) not in windows)
    ok, bad = frames_within_presence(frames, windows)
    check("P4: the default frame set is inside presence", ok and not bad)
    ok_gap, bad_gap = frames_within_presence([29, 30, 40], windows)
    check("P4: gap frames are refused", (not ok_gap) and bad_gap == [30, 40])
    check("frames: --frames parses ranges",
          parse_frame_spec("0-2,57,58") == [0, 1, 2, 57, 58])
    check("frames: empty --frames means 'use the spec'", parse_frame_spec("") is None)
    check("frames: a different fixture yields ITS frames",
          default_frame_set({"presence_frames": {"episode_1": [0, 29],
                                                 "episode_2": [59, 59]},
                             "return_frames": [59]})
          == list(range(0, 30)) + [59])

    # -- anti-leakage ------------------------------------------------------
    try:
        assert_cameras_are_training([0, 1, 2], [0, 1, 3], [2, 7])
        check("leakage: a held-out camera is refused", False)
    except LeakageError:
        check("leakage: a held-out camera is refused", True)
    check("leakage: training cameras are admitted",
          assert_cameras_are_training([0, 1, 3], [0, 1, 3], [2, 7]) is True)
    for label, train_ids, test_ids in (("test", [0, 1, 3], []),
                                       ("train", [], [2, 7])):
        try:
            assert_cameras_are_training([0], train_ids, test_ids)
            check("leakage: an empty %s roster is refused" % label, False)
        except ContractError:
            check("leakage: an empty %s roster is refused" % label, True)
    try:
        resolve_identity_dir("data/synthetic/lrv3", "data/synthetic/lrv3/gt_identity")
        check("leakage: the held-out identity directory is refused", False)
    except LeakageError:
        check("leakage: the held-out identity directory is refused", True)
    check("names: camera and frame ids parse from the fixture naming",
          camera_id_of_name("cam07_f012") == 7
          and frame_index_of_name("cam19_f059") == 59)
    for bad_name in ("", None):
        try:
            camera_id_of_name(bad_name)
            check("names: refuses the unparseable name %r" % (bad_name,), False)
        except ContractError:
            check("names: refuses the unparseable name %r" % (bad_name,), True)

    # -- every precondition fires when violated ---------------------------
    block, failures = evaluate_preconditions(**_healthy_precondition_kwargs())
    check("preconditions: a healthy setup passes all fifteen", failures == [])
    check("preconditions: fifteen checks are reported",
          len([k for k in block if k != "detail"]) == 15)

    violations = {
        "P1_render_ran": {"n_rows_nonzero_w_total": 0},
        "P2_rows_in_event_target_positive": {"n_rows_in_event_target": 0},
        "P3_rows_in_static_target_positive": {"n_rows_in_static_target": 0},
        "P4_frame_set_within_presence": {"frame_set_ok": False,
                                         "offending_frames": [40]},
        "P5_any_view_rendered_nonzero": {"n_views_nonzero_image": 0},
        "P6_cameras_disjoint_from_test": {"camera_ids_used": [0, 2]},
        "P7_row_count_matches_checkpoint": {"n_rows_checkpoint": 999},
        "P8_topology_invariant": {"n_rows_after_pass": 999},
        "P9_flow_leaf_bound_every_view": {"n_rasterizer_calls": 527},
        "P10_mask_partition_consistent": {"mask_partition_ok": False},
        "P11_backward_repeatable": {"backward_repeat_bitwise_identical": False},
        "P12_static_branch_shares_features": {"static_branch_shares_features": False},
        "P13_identity_masks_complete": {"identity_masks_complete": False},
        "P14_camera_mask_supply": {"cameras_below_mask_floor": [13, 14],
                                   "min_camera_mask_px": 1000},
        "P15_fingerprint_as_expected": {"expect_wrong_fingerprint": True},
    }
    for key, override in violations.items():
        _, broken = evaluate_preconditions(**_healthy_precondition_kwargs(**override))
        check("preconditions: %s fires when violated" % key, broken == [key])

    _, off_split = evaluate_preconditions(
        **_healthy_precondition_kwargs(cameras_are_train_objects=False))
    check("preconditions: P6 also fires on a non-train-split camera object",
          off_split == ["P6_cameras_disjoint_from_test"])
    _, vacuous = evaluate_preconditions(
        **_healthy_precondition_kwargs(test_camera_ids=[]))
    check("preconditions: P6 fires on an EMPTY held-out roster rather than "
          "passing vacuously", vacuous == ["P6_cameras_disjoint_from_test"])
    _, no_calls = evaluate_preconditions(**_healthy_precondition_kwargs(
        n_rasterizer_calls=0, n_expected_rasterizer_calls=0))
    check("preconditions: P9 fires when NO render was intercepted at all",
          no_calls == ["P9_flow_leaf_bound_every_view"])
    matched_block, matched = evaluate_preconditions(
        **_healthy_precondition_kwargs(expect_matching_fingerprint=True))
    check("preconditions: a MATCHING --expect-fingerprint passes", matched == [])
    check("preconditions: the fingerprint check records that it was requested",
          matched_block["detail"]["fingerprint_check_requested"] is True)
    default_block, _ = evaluate_preconditions(**_healthy_precondition_kwargs())
    check("preconditions: without --expect-fingerprint the check is recorded "
          "as NOT requested",
          default_block["detail"]["fingerprint_check_requested"] is False)

    import inspect
    names = set(inspect.signature(evaluate_preconditions).parameters)
    forbidden = {"ceiling", "recall", "precision", "w_total", "w_in_mask",
                 "weights", "vote", "share", "curve"}
    check("preconditions: read the SETUP only, never a score",
          names & forbidden == set())

    failed = [label for label, ok in checks if not ok]
    for label, ok in checks:
        print("  %s %s" % ("PASS" if ok else "FAIL", label))
    print("self-test: %d checks, %d failed" % (len(checks), len(failed)))
    return 0 if not failed else 1


_GOOD_FINGERPRINT = "a" * 64


def _healthy_precondition_kwargs(expect_matching_fingerprint=False,
                                 expect_wrong_fingerprint=False, **overrides):
    """A passing precondition setup, so a violation can be injected one at a
    time. Shared by ``self_test`` and the pytest suite."""
    kwargs = dict(
        n_rows_nonzero_w_total=149_794,
        n_rows_in_event_target=10_648,
        n_rows_in_static_target=4_000,
        frame_set_ok=True,
        offending_frames=[],
        n_views_nonzero_image=528,
        n_views=528,
        camera_ids_used=[0, 1, 3],
        test_camera_ids=[2, 7, 12, 17],
        cameras_are_train_objects=True,
        n_rows_loaded=149_794,
        n_rows_checkpoint=149_794,
        n_rows_after_pass=149_794,
        n_rasterizer_calls=528,
        n_expected_rasterizer_calls=528,
        mask_partition_ok=True,
        backward_repeat_bitwise_identical=True,
        static_branch_shares_features=True,
        identity_masks_complete=True,
        cameras_below_mask_floor=[],
        min_camera_mask_px=0,
        fingerprint_expected=None,
        fingerprint_measured=_GOOD_FINGERPRINT,
    )
    if expect_matching_fingerprint:
        kwargs["fingerprint_expected"] = _GOOD_FINGERPRINT
    if expect_wrong_fingerprint:
        kwargs["fingerprint_expected"] = "b" * 64
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv or "--self_test" in argv:
        return self_test()

    import torch
    from scripts.falsify_b2_edit import _merge_config, resolve_model_path
    from elgs.trainer_hooks import cloud_fingerprint

    parser, lp, op, pp = build_parser()
    args = parser.parse_args(argv)
    _merge_config(args, args.config)
    resolve_model_path(args)
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)

    spec, spec_path = load_event_spec(args.source_path)
    event_centre, event_radius = sphere_from_spec(spec)
    constants = generator_identity_constants()
    static_target = select_static_control(constants, args.static_sphere_index)

    frames = parse_frame_spec(args.frames)
    frame_source = "--frames"
    if frames is None:
        frames = default_frame_set(spec)
        frame_source = "event_spec.json:presence_frames.episode_1+return_frames"
    windows = presence_windows(spec)
    frame_set_ok, offending_frames = frames_within_presence(frames, windows)

    gaussians, scene, dataset, opt, pipe, first_iter, n_rows_checkpoint = (
        restore_frozen_model(args, lp, op, pp))

    # Held-out views are unreachable for the whole measurement, not merely
    # unused. A guard that can degrade to "protects nothing" is worse than no
    # guard, so this replaces the accessor rather than trusting the call site.
    def _refuse_test_cameras(*_args, **_kwargs):
        raise LeakageError(
            "anti-leakage: getTestCameras() called during a supervisability "
            "measurement; every reported number is defined over TRAINING "
            "views only")

    scene.getTestCameras = _refuse_test_cameras

    fingerprint = cloud_fingerprint(gaussians._xyz)
    n_rows_loaded = int(gaussians._xyz.shape[0])
    views, camera_ids = select_training_views(scene, spec, frames, args.max_cameras)
    train_stack_ids = set(id(camera) for camera in scene.train_cameras[1.0])
    cameras_are_train_objects = all(
        id(camera) in train_stack_ids for _, _, camera in views)

    identity_dir = resolve_identity_dir(args.source_path, args.identity_dir)
    wanted_views = [(int(c), int(f)) for c, f, _ in views]
    census, digest_by_view, array_by_digest = identity_census(
        identity_dir, wanted_views, min(frames), constants["event_object_id"])
    identity_masks_complete = not census["missing_requested_views"]
    render_shape = tuple(census["shape"])
    class_ids = list(census["class_ids_present"])
    if constants["event_object_id"] not in class_ids:
        raise ContractError(
            "the identity buffers never carry the event object id %d; the "
            "vote would have no event class to assign"
            % constants["event_object_id"])
    below_floor = cameras_below_supply_floor(
        {int(k): v for k, v
         in census["event_pixels_per_camera_at_reference_frame"].items()},
        args.min_camera_mask_px)

    program_path = str(args.episode_program or "").strip()
    if not program_path and not args.no_discover_program:
        program_path = discover_episode_program(
            args.model_path, str(REPO_ROOT / "configs" / "lrv3")) or ""
    grid_lo, grid_span, cells_per_axis, grid_provenance = load_grid_definition(
        program_path)

    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")

    device = gaussians._xyz.device
    mask_cache = {}

    def mask_source(cam_id, frame):
        digest = digest_by_view[(int(cam_id), int(frame))]
        cached = mask_cache.get(digest)
        if cached is None:
            array = array_by_digest[digest]
            if tuple(array.shape) != render_shape:
                raise ContractError(
                    "identity buffer for cam%02d f%03d has shape %r, not %r"
                    % (cam_id, frame, tuple(array.shape), render_shape))
            cached = torch.from_numpy(np.ascontiguousarray(array, dtype=np.int32)
                                      ).to(device)
            mask_cache[digest] = cached
        return cached

    print("cloud %s | rows %d | cameras %s | frames %d | views %d | classes %s"
          % (fingerprint[:16], n_rows_loaded, camera_ids, len(frames),
             len(views), class_ids))
    print("identity %s | %d files | %d distinct buffers by content"
          % (census["dir"], census["n_files"],
             census["n_distinct_buffers_by_content"]))

    measured = measure(
        gaussians, views, pipe, background, class_ids, mask_source,
        check_repeat_at={0, len(views) - 1}, verbose=bool(args.verbose))

    w_total = measured["w_total"].to("cpu").numpy()
    w_in_mask = measured["w_in_mask"].to("cpu").numpy()
    w_sh_dynamic = measured["w_sh_dynamic"].to("cpu").numpy()
    w_sh_static = measured["w_sh_static"].to("cpu").numpy()
    w_sh_dynamic_dc = measured["w_sh_dynamic_dc"].to("cpu").numpy()
    xyz = gaussians._xyz.detach().to("cpu").numpy()
    n_rows_after_pass = int(gaussians._xyz.shape[0])

    # The authored spheres enter HERE and nowhere earlier: they label rows
    # after every weight already exists, through the one shared predicate.
    in_event = in_sphere_flags(xyz, event_centre, event_radius)
    in_static = in_sphere_flags(xyz, static_target["centre"], static_target["radius"])

    # The ceiling's quantile limb is computed ONCE, over ALL rows of the cloud.
    # Passing the same six e_min values to the event target, the static control
    # and every cell is what makes those readings comparable, and taking them
    # from the whole cloud rather than the in-sphere rows is what keeps the
    # oracle out of the cut.
    quantile_points = quantile_e_min_points(w_total)

    assigned, vote_stats = argmax_vote(class_ids, w_in_mask, w_total, tau=VOTE_TAU)
    predicted_event = assigned == int(constants["event_object_id"])
    predicted_static = assigned == int(static_target["id"])

    repeat_ok = bool(measured["repeat_checks"]) and all(
        r["bitwise_identical"] for r in measured["repeat_checks"])
    partition_ok = bool(measured["max_partition_rel_deviation"] <= 1e-6)

    preconditions, failures = evaluate_preconditions(
        n_rows_nonzero_w_total=int((w_total > 0.0).sum()),
        n_rows_in_event_target=int(in_event.sum()),
        n_rows_in_static_target=int(in_static.sum()),
        frame_set_ok=frame_set_ok,
        offending_frames=offending_frames,
        n_views_nonzero_image=measured["n_views_nonzero_image"],
        n_views=len(views),
        camera_ids_used=camera_ids,
        test_camera_ids=spec.get("test_cameras", []),
        cameras_are_train_objects=cameras_are_train_objects,
        n_rows_loaded=n_rows_loaded,
        n_rows_checkpoint=n_rows_checkpoint,
        n_rows_after_pass=n_rows_after_pass,
        n_rasterizer_calls=measured["n_rasterizer_calls"],
        n_expected_rasterizer_calls=len(views),
        mask_partition_ok=partition_ok,
        backward_repeat_bitwise_identical=repeat_ok,
        static_branch_shares_features=measured["static_branch_shares_features"],
        identity_masks_complete=identity_masks_complete,
        cameras_below_mask_floor=below_floor,
        min_camera_mask_px=args.min_camera_mask_px,
        fingerprint_expected=(str(args.expect_fingerprint).strip() or None),
        fingerprint_measured=fingerprint,
    )

    # ---- answer 3: the per-cell breakdown -------------------------------
    grid_lo_used, grid_span_used, keys = voxel_keys(
        xyz, lo=grid_lo, span=grid_span, cells_per_axis=cells_per_axis)
    overlapping = sorted({int(k) for k in keys[in_event].tolist()})
    per_cell = {}
    for key in overlapping:
        cell_mask = keys == key
        block = restricted_metrics(cell_mask, w_total, predicted_event, in_event,
                                   quantile_points=quantile_points)
        block["cell_index"] = list(decode_cell_key(key, cells_per_axis))
        per_cell[str(key)] = block
    named_cells = {}
    for key in NAMED_CELL_KEYS:
        cell_mask = keys == int(key)
        block = restricted_metrics(cell_mask, w_total, predicted_event, in_event,
                                   quantile_points=quantile_points)
        block["cell_index"] = list(decode_cell_key(key, cells_per_axis))
        block["is_object_overlapping"] = bool(int(key) in overlapping)
        named_cells[str(int(key))] = block
    named_union = np.isin(keys, np.asarray([int(k) for k in NAMED_CELL_KEYS]))
    named_union_block = restricted_metrics(
        named_union, w_total, predicted_event, in_event,
        quantile_points=quantile_points)

    # ---- answer 5: the static-twin share --------------------------------
    denominator = w_sh_dynamic + w_sh_static
    positive = denominator > 0.0
    share = np.zeros_like(denominator)
    share[positive] = w_sh_static[positive] / denominator[positive]
    share_target = share[in_event & positive]

    # The clamp's actual bite on the DYNAMIC branch: with g = 1 on every RGB
    # channel and no clamp, the DC gradient is 3 * SH_C0 * w_total. Measured
    # against the clamp-free flow weight, so the caveat carries a number.
    from utils.sh_utils import C0
    expected_dc = 3.0 * float(C0) * w_total
    comparable = in_event & (expected_dc > 0.0)
    dc_ratio = (w_sh_dynamic_dc[comparable] / expected_dc[comparable]
                if int(comparable.sum()) else np.zeros(0))

    config_hash, config_bytes = sha256_file(args.config)
    checkpoint_hash, checkpoint_bytes = (None, None)
    if args.hash_checkpoint:
        checkpoint_hash, checkpoint_bytes = sha256_file(args.start_checkpoint)

    event_vote = precision_recall(predicted_event, in_event)
    static_vote = precision_recall(predicted_static, in_static)

    report = {
        "schema": SCHEMA,
        "evidence_bearing": False,
        "measurement_only": True,
        "carrier": {
            "channel": "flow_2d (the membership carrier)",
            "identity": "dL/df_i0 = sum_pixels g(pixel) * alpha_i * T_i",
            "dynamic_only": (
                "forward.cu accumulates Flow[] strictly inside "
                "if (collected_id[j] < P); backward.cu accumulates dL_dflows "
                "strictly inside if (gaussian_idx < P). No static primitive "
                "contributes and there is no background term."),
            "clamp_free": (
                "flow_2d runs through no SH evaluation, so the RGB clamp "
                "(forward.cu:67-70, backward.cu:32-34) cannot zero it."),
            "supersedes": (
                "the _features_dc probe of membership-supervisability-v1, "
                "which summed the dynamic and static branches' SH gradients "
                "and could not fire for a temporal-marginal-culled row"),
        },
        "checkpoint": {"path": str(args.start_checkpoint), "iteration": first_iter,
                       "sha256": checkpoint_hash, "bytes": checkpoint_bytes,
                       "rows": n_rows_checkpoint},
        "config": {"path": str(args.config), "sha256": config_hash,
                   "bytes": config_bytes},
        "cloud": {"fingerprint_sha256": fingerprint, "rows": n_rows_loaded,
                  "rows_after_pass": n_rows_after_pass,
                  "expected_fingerprint": (str(args.expect_fingerprint).strip()
                                           or None)},
        "targets": {
            "event": {"id": constants["event_object_id"], "centre": event_centre,
                      "radius": event_radius, "source": spec_path,
                      "n_rows": int(in_event.sum())},
            "static_control": dict(static_target,
                                   n_rows=int(in_static.sum()),
                                   source=constants["source"],
                                   id_rule=constants["static_sphere_id_rule"]),
            "predicate": "(xyz - centre).norm(dim=1) <= radius",
        },
        "frame_set": {"frames": [int(f) for f in frames], "n_frames": len(frames),
                      "source": frame_source,
                      "presence_windows": [list(w) for w in windows]},
        "camera_set": {"camera_ids": [int(c) for c in camera_ids],
                       "n_cameras": len(camera_ids),
                       "train_cameras_declared": [int(c) for c in
                                                  spec.get("train_cameras", [])],
                       "test_cameras_declared": [int(c) for c in
                                                 spec.get("test_cameras", [])],
                       "max_cameras_arg": int(args.max_cameras)},
        "identity_masks": dict(census,
                               min_camera_mask_px=int(args.min_camera_mask_px),
                               cameras_below_floor=below_floor,
                               floor_policy=("DEFAULT 0 = report only; this "
                                             "script chooses no scientific "
                                             "floor")),
        "binding": {
            "mechanism": ("gaussian_renderer.GaussianRasterizer is replaced by "
                          "a verifying test double for the duration of the "
                          "measurement; flow_2d and BOTH SH inputs are "
                          "substituted with leaves at the call boundary. No "
                          "repository file is modified."),
            "flow_leaf_shape": measured["flow_leaf_shape"],
            "sh_leaf_shape": measured["sh_leaf_shape"],
            "n_rasterizer_calls_intercepted": measured["n_rasterizer_calls"],
            "n_views": len(views),
            "static_branch_shares_features": measured["static_branch_shares_features"],
            "compute_cov3D_python": bool(getattr(pipe, "compute_cov3D_python", False)),
            "convert_SHs_python": bool(getattr(pipe, "convert_SHs_python", False)),
            "enable_soft_routing": bool(getattr(gaussians, "enable_soft_routing", False)),
            "enable_rendered_flow_in_config": bool(
                getattr(opt, "enable_rendered_flow", False)),
        },
        "preconditions": preconditions,
        "precondition_failures": failures,
        "answer_1_supervisability_ceiling": {
            "target": "event object rows",
            "ceiling": ceiling_curve(w_total[in_event], CEILING_E_MIN_GRID,
                                     quantile_points),
            "w_total_distribution": weight_distribution(w_total[in_event]),
            "e_min_grid_absolute": [float(e) for e in CEILING_E_MIN_GRID],
            "e_min_grid_quantiles_of_all_rows": [
                float(q) for q in CEILING_QUANTILE_GRID],
            "quantile_e_min_values": quantile_points,
            "w_total_distribution_over_ALL_rows": weight_distribution(w_total),
            "note": ("a CURVE plus the exact w_total > 0 point; no e_min is "
                     "preferred over any other by this script. The quantile "
                     "limb's cuts come from ALL rows of the cloud, never from "
                     "the in-sphere rows, so the oracle does not enter the "
                     "cut -- and the same six values are applied to the static "
                     "control and to every cell, which is what makes those "
                     "readings comparable to this one"),
        },
        "answer_2_closed_form_vote": {
            "score": "score_i(k) = w_in_mask_i(k); assign argmax_k",
            "parameters": 0,
            "classes": [int(c) for c in class_ids],
            "frozen_operating_point": {
                "tau": float(VOTE_TAU),
                "eligibility": "w_total_i > 0",
                "assignment": "max_k w_in_mask_i(k) >= tau * w_total_i",
                "otherwise": "ABSTAIN",
                "provenance": ("declared in VOTE_TAU ahead of the run; this "
                               "single point is the vote's result"),
            },
            "stats": vote_stats,
            "event": event_vote,
            "commissioned_reference": VOTE_REFERENCE,
            "clears_commissioned_reference": clears_reference(event_vote),
            "reference_provenance": (
                "supplied in this instrument's commission, not chosen here; "
                "reported as a plain comparison beside the raw numbers"),
            "tau_curve": vote_tau_curve(
                class_ids, w_in_mask, w_total, in_event,
                constants["event_object_id"]),
        },
        "answer_3_per_cell": {
            "grid": dict(grid_provenance,
                         cells_per_axis_used=int(cells_per_axis),
                         lo_used=[float(v) for v in grid_lo_used],
                         span_used=[float(v) for v in grid_span_used],
                         encoding="key = ix * cells^2 + iy * cells + iz",
                         arithmetic=("float32, clamp before truncation, "
                                     "bit-faithful to "
                                     "scripts/estimate_episodes.voxel_grid")),
            "n_object_overlapping_cells": len(overlapping),
            "object_overlapping_cell_keys": overlapping,
            "per_cell": per_cell,
            "named_cells": named_cells,
            "named_cells_union": named_union_block,
            "named_cells_note": (
                "cells 420 and 429 carry abstain_reason no_interior_gap and "
                "zero agreeing cameras in the recorded T1 estimate; their "
                "2,036 in-sphere rows (19.12% of the object) are what "
                "produced the structural 0.8088 recall cap"),
        },
        "answer_4_static_control": {
            "target": dict(static_target),
            "why": ("the static spheres carry their own ids in the same "
                    "identity buffers and are supervised at every frame with "
                    "NO temporal-marginal attenuation, so this separates "
                    "'per-row membership is learnable on this substrate' from "
                    "'EVENT membership is learnable despite temporal-support "
                    "gradient starvation'"),
            "ceiling": ceiling_curve(w_total[in_static], CEILING_E_MIN_GRID,
                                     quantile_points),
            "w_total_distribution": weight_distribution(w_total[in_static]),
            "vote": static_vote,
            "clears_commissioned_reference": clears_reference(static_vote),
            "tau_curve": vote_tau_curve(
                class_ids, w_in_mask, w_total, in_static, static_target["id"]),
            "read_against": ("the SAME frozen operating point and the SAME "
                             "cloud-wide quantile e_min values as answer 1, "
                             "which is what makes the comparison a comparison"),
        },
        "answer_5_static_twin_share": {
            "definition": ("per row, |grad_sh_static| / (|grad_sh_dynamic| + "
                           "|grad_sh_static|), with the two branches bound to "
                           "SEPARATE leaves so their gradients no longer sum "
                           "into _features_dc.grad"),
            "n_target_rows": int(in_event.sum()),
            "n_target_rows_with_positive_denominator": int(share_target.size),
            "quantiles": quantiles_of(share_target, SHARE_QUANTILES),
            "no_threshold_applied": True,
            "sh_mediated_caveat": (
                "the static branch exposes no flow handle, so this share is "
                "necessarily measured on the SH channel and inherits the RGB "
                "clamp; see clamp_exposure and dc_vs_flow_consistency"),
        },
        "consistency": {
            "mask_partition_max_rel_deviation": measured["max_partition_rel_deviation"],
            "mask_partition_tolerance": 1e-6,
            "backward_repeat_checks": measured["repeat_checks"],
            "dc_vs_flow_consistency": {
                "expression": "|grad_sh_dynamic[:, 0, :]|.sum() / (3 * SH_C0 * w_total)",
                "reads_1_when": "the RGB clamp zeroes no channel on any ray",
                "n": int(dc_ratio.size),
                "distribution": weight_distribution(
                    dc_ratio, (0.0, 0.01, 0.05, 0.50, 1.0)),
            },
            "clamp_exposure": clamp_exposure_block(gaussians),
        },
        "views": {"n_views": len(views), "per_view": measured["per_view"]},
        "notes": {
            "bounds": ("answer 1 bounds RECALL only; answer 2 measures the "
                       "precision a zero-parameter reader actually achieves"),
            "thresholds": ("none chosen here: curves, distributions and "
                           "quantiles are reported and the primary decides"),
        },
    }

    with open(args.out_report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, sort_keys=True)

    if failures:
        raise PreconditionError(
            "FROZEN PRECONDITIONS FAILED: %s. No answer is reported. Detail "
            "written to %s" % (", ".join(failures), args.out_report))

    print(json.dumps({
        "cloud_fingerprint": fingerprint,
        "total_rows": n_rows_loaded,
        "rows_in_event_target": int(in_event.sum()),
        "ceiling_strict_positive": report["answer_1_supervisability_ceiling"][
            "ceiling"]["strict_positive"]["achievable_recall_ceiling"],
        "vote_tau_frozen": float(VOTE_TAU),
        "vote_event_precision": event_vote["precision"],
        "vote_event_recall": event_vote["recall"],
        "vote_n_assigned": vote_stats["n_assigned"],
        "vote_n_abstained": vote_stats["n_abstained"],
        "vote_clears_commissioned_reference":
            report["answer_2_closed_form_vote"]["clears_commissioned_reference"],
        "static_control_ceiling": report["answer_4_static_control"][
            "ceiling"]["strict_positive"]["achievable_recall_ceiling"],
        "static_control_precision": static_vote["precision"],
        "static_control_recall": static_vote["recall"],
        "static_twin_share_quantiles":
            report["answer_5_static_twin_share"]["quantiles"],
        "named_cells_420_429_ceiling": named_union_block["ceiling"][
            "strict_positive"]["achievable_recall_ceiling"],
        "grid_source": grid_provenance["grid_source"],
        "out_report": str(args.out_report),
    }, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
