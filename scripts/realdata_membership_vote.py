#!/usr/bin/env python3
"""Per-primitive membership on a REAL N3V scene, from TRAINING VIEWS ONLY.

MEASUREMENT + EMISSION. This script trains nothing and writes no checkpoint.
It renders a frozen checkpoint, votes per row, writes one JSON report, and
optionally emits two `adags-episode-program-v2` artifacts.

WHY THIS SCRIPT EXISTS
----------------------
`scripts/membership_supervisability.py` already contains the closed-form
per-row membership vote, but it is welded to the synthetic reveal fixture: it
imports `scripts/build_synthetic_reveal_scene`, requires
`<source_path>/event_spec.json`, reads `train_identity/camNN_fFFF.npy` int16
ORACLE identity buffers, parses `_f<digits>` image names, and has no program
emitter. None of those exist on N3V. This script re-implements the SMALLEST
part of that instrument that transfers -- the leaf binding at the rasterizer
call boundary and the two mask-weighted backward passes -- and replaces the
oracle identity buffer with DEVA/SAM automatic id maps, which are derived
from the TRAINING videos alone.

The refuted alternative is on the record and is what makes this worth doing:
voxel-cell membership on the LRV3 fixture measured precision 0.0446 /
recall 0.1786 on the fresh seeding cloud
(research-wiki/operations/lrv3-membership-diagnostic-2026-08-23.md), and
PARTIAL membership scored -2.469 dB, WORSE than not gating at all
(research-wiki/operations/nonoracle-timing-t2-result-2026-08-23.md). So a
membership instrument must be scored BEFORE any retraining, and it must be
per-row rather than per-cell.

THE SIGNAL, VERBATIM FROM THE WORKING INSTRUMENT
------------------------------------------------
`gaussian_renderer.render` hands the rasterizer a `flow_2d` tensor with one
row per Gaussian (gaussian_renderer/__init__.py:315-317, :381) and the CUDA
forward accumulates, DYNAMIC-BRANCH ONLY,

    Flow[ch](pixel) += flows[i * 2 + ch] * alpha_i * T_i

(diff-gaussian-rasterization/cuda_rasterizer/forward.cu:760, inside
`if (collected_id[j] < P)`). Substituting a per-row LEAF for `flow_2d` at the
rasterizer call boundary therefore makes

    d / d flows[i, 0]  of  sum_{pixels p in A} Flow[0](p)   =   sum_{p in A} alpha_i(p) T_i(p)

exactly the rendered compositing weight row i deposits inside the pixel set
A. With A = the harmonized object mask this is `w_in_i`; with A = its
complement it is `w_out_i`; with A = the whole image it is `w_total_i`, and
the three must satisfy `w_in + w_out == w_total` up to float error. That
partition identity is checked per view and a failure REFUSES rather than
being reported (mirrors P10 of the fixture instrument).

The leaf binding is a test double installed around
`gaussian_renderer.GaussianRasterizer`, adapted from
scripts/membership_supervisability.py:2075-2153. Nothing upstream of the
rasterizer -- the temporal marginal, soft routing, the opacity chain -- is
reimplemented, so the measured weights are the ones the training path
actually composites. `compute_cov3D_python` is REFUSED because it prefilters
every per-row argument by `marginal_t > 0.05` before that boundary
(gaussian_renderer/__init__.py:319-346) and the per-row binding would no
longer be one-to-one; `convert_SHs_python` is refused for symmetry with the
fixture instrument's precondition even though this script does not read the
SH leaves.

The SEED CONTRIBUTION MAP reuses the same channel in the forward direction:
setting the leaf's channel 0 to the seed indicator makes the rendered
`flow[0]` equal `S(p) = sum_{i in seed} alpha_i(p) T_i(p)`, which is what the
DEVA id harmonization thresholds. Leaf VALUES do not affect the gradient
(the derivative of a linear accumulation w.r.t. its own coefficient), so one
forward can serve both purposes.

ANTI-LEAKAGE
------------
cam00 is the held-out N3V viewpoint. It is refused three ways: it is removed
from the camera roster by id, `scene.getTestCameras` is replaced by a raiser
for the whole measurement stage, and `builtins.open` / `os.open` refuse any
path naming `cam00` (which also covers the DEVA mask tree). The oracle-path
refusals of scripts/estimate_episodes.py:209-224 (`gt_identity`,
`train_identity`, `event_spec.json`, `oracle_*.json`) are carried over
unchanged. Every consumed camera must be an ELEMENT of `scene.train_cameras[1.0]`
by object identity, not by equality -- the check from
scripts/estimate_episodes.py:270-284.

The guard is installed AROUND the measurement stage only, exactly as
`estimate_episodes.main` does: `Scene(...)` construction legitimately reads
the test split from disk before the guard exists, and the guard then makes
those objects unreachable.

DIAGNOSTIC MODE AND THE SECOND HARMONISATION RULE
--------------------------------------------------
The first real-checkpoint run refused at harmonisation: with a 1,483-row seed
inside a 599,478-row cloud, `S(p) = sum_{i in seed} alpha_i T_i` almost never
exceeds `--s_thresh 0.5`, so "the fraction of an id's OWN pixels carrying
S > s_thresh" cannot reach `--id_overlap` for any id and the instrument fails
closed with every fraction at ~0. That refusal is correct -- an empty object
mask would report every row a non-member -- but it says nothing about WHERE
the seed actually lands.

`--diagnose` answers that question and nothing else: it renders the same S map
the chooser thresholds, writes it as an image, writes the DEVA id map beside
it, tabulates the ids by the S-MASS they carry rather than by a threshold
crossing, projects the seed rows into every camera, and exits 0 without voting
or emitting anything.

`--id_rule mass` is the alternative chooser that follows from the same
observation: rank ids by the S-mass they carry and take them in descending
order until `--mass_cover` of the view's S-mass is covered. It is
scale-invariant in S, so a seed that deposits a thin, diffuse contribution
still selects the id it is concentrated on. It is NOT the frozen operating
point: `--id_rule fraction` remains the default and its behaviour, its chosen
ids and its emitted program bytes are untouched by this addition.

WHAT THIS SCRIPT DOES NOT DO
-----------------------------
It does not score against ground truth (there is none on N3V), it does not
choose whether the emitted programs should be trained, and it does not
retrain. The vote threshold `--tau` is exposed but its default 0.5 is the
frozen operating point of scripts/membership_supervisability.py:280.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402


REPORT_SCHEMA = "adags-realdata-membership-vote-v1"

#: Frozen operating point, copied from
#: scripts/membership_supervisability.py:280. A row is a MEMBER iff its
#: pooled in-mask share is at least this. Exposed as `--tau` so a sweep can
#: be reported as ceiling information, never as the score.
VOTE_TAU = 0.50

#: Sentinel group id for the single emitted group. 1 rather than 0 so a
#: -1/0 confusion in any consumer is loud.
EMITTED_GROUP_ID = 1

#: The N3V held-out viewpoint. Never rendered, never opened.
DEFAULT_HELD_OUT_CAMERAS = (0,)

#: Relative tolerance on the per-view partition identity w_in + w_out == w_total.
#: The three sums are accumulated by `atomicAdd` over ~1.4M pixels in
#: nondeterministic order in float32, and they visit the pixels in three
#: different groupings, so exact agreement is not available and a tolerance
#: at float32 epsilon would refuse every real view. 1e-3 is loose enough to
#: survive that and tight enough that a wrong mask, a wrong complement or an
#: unbound leaf -- all of which are O(1) errors -- still fails closed.
#: Overridable with `--partition_rel_tol`; the observed maximum is always
#: reported so the margin is visible rather than assumed.
PARTITION_REL_TOL = 1e-3

#: Weights are magnitudes. A gradient more negative than this fraction of the
#: view's own maximum means the binding is wrong, not that a row is negative.
NEGATIVE_WEIGHT_REL_TOL = 1e-6

#: The two harmonisation rules. `fraction` is the frozen default and is what
#: every historical invocation of this script used.
ID_RULES = ("fraction", "mass")

#: `--diagnose` artifacts.
DIAG_SCHEMA = "adags-realdata-membership-diagnose-v1"

#: The S levels the diagnostic counts pixels at. 0.5 is `--s_thresh`'s default
#: so the table always contains the level the failing rule actually used.
DIAG_S_LEVELS = (0.02, 0.05, 0.1, 0.2, 0.5)

#: The overlay marks pixels above this. Deliberately far below `--s_thresh`:
#: the whole point of the overlay is to show a contribution too weak for the
#: fraction rule to see.
DIAG_OVERLAY_S = 0.05

#: The overlay's marker colour. `id_colour` never emits a channel outside
#: [32, 223], so this can never collide with a DEVA id's colour.
DIAG_MARKER_RGB = (255, 0, 255)

#: How many ids the diagnostic tabulates per view.
DIAG_TOP_K = 10

#: The scoring-side frame box, x0 y0 x1 y1 INCLUSIVE. Supplied by the caller;
#: this script never derives it and never scores against it.
DEFAULT_FBOX = (664, 912, 744, 976)
DEFAULT_FBOX_FRAME = 150


class LeakageError(RuntimeError):
    """Raised when the measurement stage reaches for a forbidden input."""


# ---------------------------------------------------------------------------
# anti-leakage guard (mirrors scripts/estimate_episodes.py:205-344)
# ---------------------------------------------------------------------------


def _normalise(path):
    return str(path).replace("\\", "/").lower()


def is_forbidden_path(path, held_out_cameras=DEFAULT_HELD_OUT_CAMERAS):
    """True for any path the measurement stage must never open.

    The first four rules are carried over verbatim from
    scripts/estimate_episodes.py:209-224. The fifth is new and specific to
    this script: any path naming a held-out camera. On N3V that is `cam00`,
    and it covers the image tree, the DEVA mask tree, and anything else.
    """
    low = _normalise(path)
    if "gt_identity" in low:
        return True
    if "train_identity" in low:
        return True
    if low.endswith("event_spec.json"):
        return True
    if re.search(r"(^|/)oracle_[^/]*\.json$", low):
        return True
    for cam_id in held_out_cameras:
        if ("cam%02d" % int(cam_id)) in low:
            return True
    return False


class HeldOutGuard(object):
    """Runtime enforcement of the training-views-only contract.

    Identical in structure to `estimate_episodes.LeakageGuard`; the addition
    is `held_out_cameras`, which turns the path filter into a held-out-view
    filter as well.
    """

    def __init__(self, scene=None, opt=None,
                 held_out_cameras=DEFAULT_HELD_OUT_CAMERAS):
        self._scene = scene
        self._opt = opt
        self._held_out = tuple(int(c) for c in held_out_cameras)
        self.active = False
        self.checks = {}
        self._saved_open = None
        self._saved_os_open = None
        self._saved_get_test = None
        self._audit_installed = False

    # -- static assertions (run before measurement) -----------------------

    def assert_manifests_empty(self):
        """A runtime visibility gate would suppress rows and deflate weights.

        Same refusal as scripts/membership_supervisability.py:1953-1959.
        """
        for key in ("event_candidate_manifest", "event_boundary_support_manifest"):
            value = str(getattr(self._opt, key, "") or "").strip()
            if value:
                raise LeakageError(
                    "anti-leakage: %s must be empty for a measurement run; got %r"
                    % (key, value))
        self.checks["event_manifests_empty"] = True

    def assert_train_only(self, cameras, train_stack):
        """Every consumed camera must BE an element of the train stack.

        Identity, not equality (scripts/estimate_episodes.py:270-284): a
        Camera drawn from the test split can never pass this.
        """
        allowed = set(id(cam) for cam in train_stack)
        for cam in cameras:
            if id(cam) not in allowed:
                raise LeakageError(
                    "anti-leakage: camera %r is not an element of the train split"
                    % (getattr(cam, "image_name", "<unnamed>"),))
        self.checks["cameras_are_train_split_objects"] = True

    def assert_no_held_out_ids(self, camera_ids):
        overlap = sorted(set(int(c) for c in camera_ids) & set(self._held_out))
        if overlap:
            raise LeakageError(
                "anti-leakage: cameras %r are held out and must never be read"
                % (overlap,))
        self.checks["no_held_out_camera_ids"] = True

    # -- context manager ---------------------------------------------------

    def __enter__(self):
        self.active = True
        self._saved_open = builtins.open
        self._saved_os_open = os.open
        guard = self
        held = self._held_out

        def guarded_open(file, *args, **kwargs):
            if guard.active and is_forbidden_path(file, held):
                raise LeakageError(
                    "anti-leakage: the measurement stage tried to open %r"
                    % (str(file),))
            return guard._saved_open(file, *args, **kwargs)

        def guarded_os_open(path, *args, **kwargs):
            if guard.active and is_forbidden_path(path, held):
                raise LeakageError(
                    "anti-leakage: the measurement stage tried to open %r"
                    % (str(path),))
            return guard._saved_os_open(path, *args, **kwargs)

        builtins.open = guarded_open
        os.open = guarded_os_open
        self.checks["forbidden_path_open_guard"] = True

        if hasattr(sys, "addaudithook") and not self._audit_installed:
            def _hook(event, args):
                if not guard.active:
                    return
                if event in ("open", "os.open", "io.open") and args:
                    if is_forbidden_path(args[0], held):
                        raise LeakageError(
                            "anti-leakage: audit hook blocked %r" % (str(args[0]),))
            sys.addaudithook(_hook)
            self._audit_installed = True
        self.checks["audit_hook_installed"] = bool(self._audit_installed)

        if self._scene is not None:
            self._saved_get_test = self._scene.getTestCameras

            def _refuse(*_args, **_kwargs):
                raise LeakageError(
                    "anti-leakage: getTestCameras() called during measurement")

            self._scene.getTestCameras = _refuse
            self.checks["get_test_cameras_disabled"] = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.active = False
        if self._saved_open is not None:
            builtins.open = self._saved_open
        if self._saved_os_open is not None:
            os.open = self._saved_os_open
        if self._scene is not None and self._saved_get_test is not None:
            self._scene.getTestCameras = self._saved_get_test
        return False


# ---------------------------------------------------------------------------
# pure helpers: specs, harmonization, vote, grid, program. No torch, no I/O.
# ---------------------------------------------------------------------------


def parse_int_ranges(text):
    """`"150-156,195-205,300"` -> sorted unique ints. Refuses malformed input."""
    out = set()
    raw = str(text or "").strip()
    if not raw:
        raise ContractError("empty frame/camera specification")
    for chunk in raw.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        if "-" in piece.lstrip("-"):
            lo_text, _, hi_text = piece.partition("-")
            lo, hi = int(lo_text), int(hi_text)
            if hi < lo:
                raise ContractError("descending range %r" % (piece,))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(piece))
    if not out:
        raise ContractError("specification %r selected nothing" % (text,))
    return sorted(out)


def parse_camera_spec(text, available):
    """`"all"` -> every available id; otherwise an explicit subset."""
    ordered = sorted(int(c) for c in available)
    if str(text or "").strip().lower() == "all":
        return ordered
    wanted = parse_int_ranges(text)
    missing = sorted(set(wanted) - set(ordered))
    if missing:
        raise ContractError(
            "cameras %r are not in the training split %r" % (missing, ordered))
    return [c for c in ordered if c in set(wanted)]


def choose_ids_for_view(id_map, s_map, s_thresh, id_overlap, min_id_pixels,
                        ignore_id=0):
    """Which DEVA ids in ONE view belong to the seeded object.

    `id_map` is the uint8 per-camera-video id image; `s_map` is the seed
    contribution map `S(p) = sum_{i in seed} alpha_i T_i` at the same
    resolution. An id is CHOSEN iff the fraction of its pixels carrying
    `S > s_thresh` is at least `id_overlap`, and iff it covers at least
    `min_id_pixels` pixels.

    DEVA ids are per-camera-video and are NOT consistent across cameras, and
    one object is frequently split into several ids -- which is exactly why
    this is a per-camera set-valued choice and not a single label lookup.

    Returns `(chosen_ids, per_id_stats)`; `per_id_stats` lists every id
    considered so a report reader can see the near-misses rather than only
    the survivors.
    """
    ids_arr = np.asarray(id_map)
    s_arr = np.asarray(s_map)
    if ids_arr.shape != s_arr.shape:
        raise ContractError(
            "id map %r and contribution map %r disagree on shape"
            % (tuple(ids_arr.shape), tuple(s_arr.shape)))
    if not 0.0 <= float(id_overlap) <= 1.0:
        raise ContractError("--id_overlap %r outside [0, 1]" % (id_overlap,))
    hot = s_arr > float(s_thresh)
    chosen, stats = [], []
    for raw in np.unique(ids_arr):
        value = int(raw)
        if value == int(ignore_id):
            continue
        region = ids_arr == value
        pixels = int(region.sum())
        covered = int(np.logical_and(region, hot).sum())
        fraction = float(covered) / float(pixels) if pixels else 0.0
        big_enough = pixels >= int(min_id_pixels)
        take = bool(big_enough and fraction >= float(id_overlap))
        stats.append({
            "id": value,
            "pixels": pixels,
            "pixels_over_s_thresh": covered,
            "fraction": fraction,
            "meets_min_pixels": big_enough,
            "chosen": take,
        })
        if take:
            chosen.append(value)
    return sorted(chosen), stats


def summarise_ids_by_mass(id_map, s_map, s_thresh, ignore_id=0):
    """Every DEVA id in ONE view, ranked by the S-MASS it carries.

    `s_mass` is `sum_{p in id} S(p)` -- the total rendered compositing weight
    the seed deposits inside that id. It is the quantity `choose_ids_for_view`
    does NOT look at: that rule asks how many of an id's pixels cross an
    ABSOLUTE level, so a seed of 1,483 rows inside a 599,478-row cloud, whose
    S is everywhere below `--s_thresh`, reads exactly 0.0 for every id and
    carries no information at all. S-mass is scale-free in S and still ranks
    the ids correctly in that regime.

    `mass_fraction` is normalised by the WHOLE VIEW's S-mass, INCLUDING the
    ignored id, not by the labelled part. That is the honest denominator: if
    the seed's contribution lands mostly on unlabelled background then no id
    can cover it, and the caller must be able to see that rather than have it
    renormalised away. `s_mass_ignored_id_fraction` reports exactly that share.

    Returns `(rows, totals)` with `rows` sorted by descending `s_mass`, ties
    broken by ascending id so the order is deterministic.
    """
    ids_arr = np.asarray(id_map)
    s_arr = np.asarray(s_map, dtype=np.float64)
    if ids_arr.shape != s_arr.shape:
        raise ContractError(
            "id map %r and contribution map %r disagree on shape"
            % (tuple(ids_arr.shape), tuple(s_arr.shape)))
    hot = s_arr > float(s_thresh)
    total_view = float(s_arr.sum())
    labelled = 0.0
    rows = []
    for raw in np.unique(ids_arr):
        value = int(raw)
        if value == int(ignore_id):
            continue
        region = ids_arr == value
        pixels = int(region.sum())
        mass = float(s_arr[region].sum())
        labelled += mass
        covered = int(np.logical_and(region, hot).sum())
        rows.append({
            "id": value,
            "pixels": pixels,
            "s_mass": mass,
            "mass_fraction": (mass / total_view) if total_view > 0.0 else 0.0,
            "pixels_over_s_thresh": covered,
            "fraction": (float(covered) / float(pixels)) if pixels else 0.0,
        })
    rows.sort(key=lambda row: (-row["s_mass"], row["id"]))
    totals = {
        "ignore_id": int(ignore_id),
        "n_ids": len(rows),
        "s_mass_view": total_view,
        "s_mass_labelled": labelled,
        "s_mass_ignored_id_fraction": (
            (total_view - labelled) / total_view if total_view > 0.0 else 0.0),
        "mass_fraction_denominator": (
            "the whole view's S-mass, including the ignored id"),
    }
    return rows, totals


def choose_ids_by_mass(id_map, s_map, s_thresh, mass_cover, id_min_mass_frac,
                       min_id_pixels, ignore_id=0):
    """The `mass` harmonisation rule for ONE view.

    Walk the ids in descending S-mass and take them until `mass_cover` of the
    view's S-mass is covered by the taken ids. An id is admissible only if it
    carries at least `id_min_mass_frac` of the view's S-mass AND covers at
    least `min_id_pixels` pixels.

    `mass_fraction` is monotone decreasing along the walk, so the
    `id_min_mass_frac` bar truncates it; the `min_id_pixels` bar does not, so
    an id that is heavy but tiny is SKIPPED and the walk continues rather than
    stopping. A skipped id contributes nothing to the covered mass, so the
    cover target is never met by mass the rule refused to admit.

    Returns `(chosen, stats, totals)`. `stats` is every id considered, in the
    walk's own order, so the report shows the near-misses and the reason each
    was refused. Fail-closed behaviour is the CALLER's: an empty `chosen` is
    returned, not an exception, so the caller can raise with its own context.
    """
    if not 0.0 <= float(mass_cover) <= 1.0:
        raise ContractError("--mass_cover %r outside [0, 1]" % (mass_cover,))
    if not 0.0 <= float(id_min_mass_frac) <= 1.0:
        raise ContractError(
            "--id_min_mass_frac %r outside [0, 1]" % (id_min_mass_frac,))
    rows, totals = summarise_ids_by_mass(id_map, s_map, s_thresh, ignore_id)
    chosen, stats = [], []
    covered = 0.0
    satisfied = False
    for row in rows:
        entry = dict(row)
        big_enough = bool(row["pixels"] >= int(min_id_pixels))
        heavy_enough = bool(row["mass_fraction"] >= float(id_min_mass_frac))
        take = bool(big_enough and heavy_enough and not satisfied)
        if take:
            covered += float(row["mass_fraction"])
            chosen.append(int(row["id"]))
            if covered >= float(mass_cover):
                satisfied = True
        entry["meets_min_pixels"] = big_enough
        entry["meets_min_mass_frac"] = heavy_enough
        entry["chosen"] = take
        entry["cumulative_mass_fraction"] = float(covered)
        stats.append(entry)
    totals = dict(totals)
    totals["mass_cover_target"] = float(mass_cover)
    totals["mass_covered_by_chosen"] = float(covered)
    totals["mass_cover_reached"] = bool(satisfied)
    return sorted(chosen), stats, totals


def choose_ids_by_rule(rule, id_map, s_map, s_thresh, id_overlap,
                       min_id_pixels, mass_cover, id_min_mass_frac,
                       ignore_id=0):
    """Dispatch to one harmonisation rule; `(chosen, stats, rule_info)`.

    `rule == "fraction"` delegates VERBATIM to `choose_ids_for_view` with the
    same arguments in the same order, so the frozen default cannot drift: the
    dispatcher adds a label and nothing else.
    """
    name = str(rule)
    if name == "fraction":
        chosen, stats = choose_ids_for_view(
            id_map, s_map, s_thresh, id_overlap, min_id_pixels, ignore_id)
        return chosen, stats, {"rule": "fraction"}
    if name == "mass":
        chosen, stats, totals = choose_ids_by_mass(
            id_map, s_map, s_thresh, mass_cover, id_min_mass_frac,
            min_id_pixels, ignore_id)
        info = {"rule": "mass"}
        info.update(totals)
        return chosen, stats, info
    raise ContractError(
        "unknown --id_rule %r; expected one of %r" % (rule, list(ID_RULES)))


def diagnose_view_summary(id_map, s_map, s_thresh, top_k=DIAG_TOP_K,
                          ignore_id=0, levels=DIAG_S_LEVELS):
    """The per-view diagnostic record. Pure numpy; no I/O, no torch.

    Reports the S map's own scale (max, mean, sum), how many pixels clear each
    level in `levels`, and the `top_k` ids by S-mass with their area, mass,
    mass fraction and -- so the failing rule's own quantity is visible beside
    the replacement's -- the fraction of each id's pixels above `s_thresh`.
    """
    s_arr = np.asarray(s_map, dtype=np.float64)
    rows, totals = summarise_ids_by_mass(id_map, s_arr, s_thresh, ignore_id)
    return {
        "s_thresh": float(s_thresh),
        "n_pixels": int(s_arr.size),
        "s_max": float(s_arr.max()) if s_arr.size else 0.0,
        "s_mean": float(s_arr.mean()) if s_arr.size else 0.0,
        "s_sum": float(s_arr.sum()),
        "pixels_over": {
            ("%g" % float(level)): int((s_arr > float(level)).sum())
            for level in levels},
        "id_totals": totals,
        "top_ids_by_s_mass": [dict(row) for row in rows[:int(top_k)]],
    }


def scale_to_uint8(values, vmax=1.0):
    """Clip to `[0, vmax]`, scale to 0-255, round. A non-positive or
    non-finite `vmax` yields an all-zero image rather than a division error."""
    arr = np.asarray(values, dtype=np.float64)
    top = float(vmax)
    if not np.isfinite(top) or top <= 0.0:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.rint(np.clip(arr, 0.0, top) / top * 255.0).astype(np.uint8)


def id_colour(value, ignore_id=0):
    """A deterministic pseudo-random RGB for one DEVA id.

    Every channel lands in [32, 223], which is what makes `DIAG_MARKER_RGB`
    unambiguous: no id can ever be drawn in the marker's colour.
    """
    if int(value) == int(ignore_id):
        return (0, 0, 0)
    h = (int(value) * 2654435761 + 40503) % (1 << 32)
    return (32 + (h >> 2) % 192, 32 + (h >> 11) % 192, 32 + (h >> 20) % 192)


def id_palette_rgb(id_map, marked=None, marker=DIAG_MARKER_RGB, ignore_id=0):
    """The id map as an RGB image, with `marked` pixels painted `marker`."""
    ids_arr = np.asarray(id_map)
    out = np.zeros(tuple(ids_arr.shape) + (3,), dtype=np.uint8)
    for raw in np.unique(ids_arr):
        out[ids_arr == raw] = np.asarray(id_colour(int(raw), ignore_id),
                                         dtype=np.uint8)
    if marked is not None:
        flags = np.asarray(marked, dtype=bool)
        if flags.shape != ids_arr.shape:
            raise ContractError(
                "marker mask %r and id map %r disagree on shape"
                % (tuple(flags.shape), tuple(ids_arr.shape)))
        out[flags] = np.asarray(marker, dtype=np.uint8)
    return out


def projected_bbox(xy, valid=None):
    """Axis-aligned pixel bbox of the projected points selected by `valid`."""
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if valid is None:
        sel = np.ones(pts.shape[0], dtype=bool)
    else:
        sel = np.asarray(valid, dtype=bool).reshape(-1)
        if sel.shape[0] != pts.shape[0]:
            raise ContractError(
                "validity flags (%d) and projected points (%d) disagree"
                % (int(sel.shape[0]), int(pts.shape[0])))
    kept = int(sel.sum())
    out = {"n_points": int(pts.shape[0]), "n_selected": kept,
           "x_min": None, "x_max": None, "y_min": None, "y_max": None}
    if kept:
        sub = pts[sel]
        out.update({
            "x_min": float(sub[:, 0].min()), "x_max": float(sub[:, 0].max()),
            "y_min": float(sub[:, 1].min()), "y_max": float(sub[:, 1].max()),
        })
    return out


def count_in_fbox(xy, valid, fbox):
    """How many projected points land inside `fbox = (x0, y0, x1, y1)`.

    The box is INCLUSIVE on both ends and is applied to the ROUNDED pixel
    coordinates, which is the same rounding `estimate_episodes.build_footprints`
    uses to scatter a projected row into a pixel
    (scripts/estimate_episodes.py:866-868).
    """
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    sel = np.asarray(valid, dtype=bool).reshape(-1)
    if sel.shape[0] != pts.shape[0]:
        raise ContractError(
            "validity flags (%d) and projected points (%d) disagree"
            % (int(sel.shape[0]), int(pts.shape[0])))
    box = [int(v) for v in fbox]
    if len(box) != 4:
        raise ContractError("--fbox needs exactly four integers x0 y0 x1 y1")
    x0, y0, x1, y1 = box
    if x1 < x0 or y1 < y0:
        raise ContractError(
            "--fbox %r is not x0 y0 x1 y1 with x1 >= x0 and y1 >= y0" % (box,))
    xs = np.rint(pts[:, 0])
    ys = np.rint(pts[:, 1])
    inside = sel & (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    return int(inside.sum())


def aggregate_chosen_ids(per_frame_chosen, id_min_frames):
    """Pool one camera's per-anchor-frame id choices into one id set.

    An id enters the camera's set iff it was chosen on at least
    `id_min_frames` anchor frames. With the default 1 this is the union,
    which is the permissive reading; raising it is a stricter, and reportable,
    operating point.
    """
    counts = {}
    for chosen in per_frame_chosen:
        for value in set(int(v) for v in chosen):
            counts[value] = counts.get(value, 0) + 1
    keep = sorted(k for k, n in counts.items() if n >= int(id_min_frames))
    return keep, {str(k): int(v) for k, v in sorted(counts.items())}


def mask_from_ids(id_map, chosen_ids):
    """Boolean object mask = union of the chosen ids."""
    ids_arr = np.asarray(id_map)
    if not chosen_ids:
        return np.zeros(ids_arr.shape, dtype=bool)
    wanted = np.asarray(sorted(int(v) for v in chosen_ids))
    return np.isin(ids_arr, wanted)


def membership_vote(w_in, w_out, tau=VOTE_TAU):
    """The closed-form vote. Pooled over views, strict `e_min = 0`.

    * a row is ELIGIBLE iff `w_in + w_out > 0` -- a row that received no
      compositing weight at all in any measured view was never observed and
      has nothing to vote with, so it ABSTAINS;
    * an eligible row is a MEMBER iff `w_in >= tau * (w_in + w_out)`.

    The `>=` matches `scripts/membership_supervisability.py:687-690`
    (`max_k w_in_mask_i(k) >= tau * w_total_i`). At `tau = 0.5` that admits a
    row whose weight splits exactly evenly, so the count of rows sitting
    EXACTLY at tau is reported and the strict-inequality count is reported
    alongside; if they differ the choice mattered and the reader can see it.
    """
    a = np.asarray(w_in, dtype=np.float64).reshape(-1)
    b = np.asarray(w_out, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ContractError("w_in and w_out disagree on length")
    if a.size and (bool((a < 0).any()) or bool((b < 0).any())):
        raise ContractError("a weight is negative; compositing weights are magnitudes")
    tau = float(tau)
    if not 0.0 <= tau <= 1.0:
        raise ContractError("tau %r outside [0, 1]" % (tau,))
    total = a + b
    eligible = total > 0.0
    share = np.zeros(a.shape, dtype=np.float64)
    np.divide(a, total, out=share, where=eligible)
    member = eligible & (share >= tau)
    member_strict = eligible & (share > tau)
    stats = {
        "n_rows": int(a.size),
        "tau": tau,
        "rule": ("eligible iff w_in + w_out > 0; member iff "
                 "w_in >= tau * (w_in + w_out); else ABSTAIN"),
        "e_min": 0.0,
        "n_eligible": int(eligible.sum()),
        "n_members": int(member.sum()),
        "n_members_strict_gt_tau": int(member_strict.sum()),
        "n_rows_exactly_at_tau": int((eligible & (share == tau)).sum()),
        "n_abstained": int((~member).sum()),
        "n_abstained_ineligible": int((~eligible).sum()),
        "n_abstained_below_tau": int((eligible & ~member).sum()),
    }
    return member, share, stats


def jaccard(a, b):
    """Jaccard index of two boolean membership vectors; 1.0 for two empties."""
    left = np.asarray(a, dtype=bool)
    right = np.asarray(b, dtype=bool)
    union = int(np.logical_or(left, right).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum()) / float(union)


def padded_bbox(points, pad):
    """(lo, span) of a padded axis-aligned box around `points`.

    `pad` is a fraction of each axis' own extent, applied on BOTH sides, so
    the members are strictly interior. A degenerate axis (all members share a
    coordinate) is given a small absolute span rather than zero, because
    `resolve_v2_membership` refuses a non-positive span
    (elgs/trainer_hooks.py:797-798).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ContractError("points must be (N, 3); got %r" % (tuple(pts.shape),))
    if pts.shape[0] == 0:
        raise ContractError("cannot bound an empty member set")
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    extent = hi - lo
    margin = np.maximum(extent * float(pad), 1e-6)
    lo = lo - margin
    span = (extent + 2.0 * margin)
    span = np.maximum(span, 1e-6)
    return lo, span


def voxel_keys(points, lo, span, cells_per_axis):
    """Cell keys under the EXACT grid `resolve_v2_membership` reapplies.

    Replicated from elgs/trainer_hooks.py:799-802, which is itself the grid
    `estimate_episodes.voxel_grid` (scripts/estimate_episodes.py:577-594)
    emits:

        voxel = ((xyz - lo) / span * cells).clamp(0, cells - 1).long()
        key   = x * cells^2 + y * cells + z

    THE ARITHMETIC IS DONE IN float32 ON PURPOSE. `resolve_v2_membership`
    builds `lo` and `span` with `dtype=xyz.dtype`, which is the model's
    float32, so doing this in float64 here would put a row sitting on a cell
    boundary in a different cell than the seeding run does -- and at 64 cells
    per axis the boundaries are dense. `clamp` runs BEFORE `.long()` there, so
    every value is already non-negative and truncation equals floor; `clip`
    then `astype` reproduces that exactly.
    """
    cells = int(cells_per_axis)
    if cells < 1:
        raise ContractError("cells_per_axis must be >= 1")
    pts = np.asarray(points, dtype=np.float32)
    lo_arr = np.asarray(lo, dtype=np.float32).reshape(3)
    span_arr = np.asarray(span, dtype=np.float32).reshape(3)
    if bool((span_arr <= 0).any()):
        raise ContractError("span must be positive on every axis")
    voxel = np.clip((pts - lo_arr) / span_arr * np.float32(cells),
                    0, cells - 1).astype(np.int64)
    return voxel[:, 0] * cells * cells + voxel[:, 1] * cells + voxel[:, 2]


def voxel_inside(points, lo, span):
    """Rows inside the originating box; the rest are forced to -1 downstream.

    elgs/trainer_hooks.py:812-815 recomputes exactly this and sets those rows
    ungated, so a fine grid built from a padded member bbox gates NOTHING
    outside that box, however far a foreign cloud's rows stray. float32 for
    the same reason as `voxel_keys`.
    """
    pts = np.asarray(points, dtype=np.float32)
    lo_arr = np.asarray(lo, dtype=np.float32).reshape(3)
    span_arr = np.asarray(span, dtype=np.float32).reshape(3)
    return np.all((pts >= lo_arr) & (pts <= lo_arr + span_arr), axis=1)


def inset_gap_seconds(offset_frame, onset_frame, frame_dt, w):
    """Absence gap in model seconds, boundaries INSET by exactly w.

    REPLICATED VERBATIM from scripts/estimate_episodes.py:502-520 (the
    function body is lines 517-519) so this script does not have to import a
    module that pulls torch at call time. The convention:

        gap_start = t(last present frame)  + w = (offset_frame - 1) * dt + w
        gap_end   = t(first present frame) - w =  onset_frame      * dt - w

    so `offset_frame` is the FIRST ABSENT frame and `onset_frame` is the
    FIRST PRESENT frame after the gap. `--gap_frames A B` names the absent
    frames INCLUSIVELY, hence `offset_frame = A` and `onset_frame = B + 1`.
    """
    gap_start = (int(offset_frame) - 1) * float(frame_dt) + float(w)
    gap_end = int(onset_frame) * float(frame_dt) - float(w)
    return gap_start, gap_end


def cloud_fingerprint_from_array(xyz):
    """sha256 over the float32 xyz bytes.

    Byte-identical to `elgs.trainer_hooks.cloud_fingerprint`
    (elgs/trainer_hooks.py:700-705), which hashes
    `xyz.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()`.
    """
    arr = np.ascontiguousarray(np.asarray(xyz, dtype=np.float32))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def build_v2_program(membership_mode, row_group_ids, n_rows, xyz_sha256,
                     lo, span, cells_per_axis, group_cell_keys,
                     gap_seconds, offset_frame, onset_frame, rows_at_estimation,
                     w, frame_dt, source=None, group_id=EMITTED_GROUP_ID):
    """One `adags-episode-program-v2` payload with exactly one gated group.

    Field-for-field the shape `estimate_episodes.build_v2_program`
    (scripts/estimate_episodes.py:1023-1104) emits, so the same consumer
    (`elgs.trainer_hooks._load_episode_program_v2`, lines 707-772) parses it.
    The `spatial` block is written in BOTH modes -- the loader validates it
    whenever it is present, and carrying it lets one artifact be re-declared
    for the other protocol without re-measuring.
    """
    if membership_mode not in ("row_ids", "spatial_voxel"):
        raise ContractError("unknown membership_mode %r" % (membership_mode,))
    payload = {
        "schema_version": "adags-episode-program-v2",
        "units": "model_time_seconds",
        "membership_mode": membership_mode,
        "frame_dt": float(frame_dt),
        "presence_edge_half_width_w": float(w),
        "cloud": {
            "n_rows": int(n_rows),
            "xyz_sha256": str(xyz_sha256),
        },
        "spatial": {
            "kind": "voxel_grid",
            "cells_per_axis": int(cells_per_axis),
            "lo": [float(v) for v in np.asarray(lo).reshape(3)],
            "span": [float(v) for v in np.asarray(span).reshape(3)],
            "group_cell_keys": {
                str(int(group_id)): [int(k) for k in group_cell_keys]},
        },
        "groups": [
            {
                "group": int(group_id),
                "gaps": [[float(gap_seconds[0]), float(gap_seconds[1])]],
                "offset_frame": int(offset_frame),
                "onset_frame": int(onset_frame),
                "rows_at_estimation": int(rows_at_estimation),
            }
        ],
        "source": dict(source or {}),
    }
    if membership_mode == "row_ids":
        payload["row_group_ids"] = [int(v) for v in row_group_ids]
        if len(payload["row_group_ids"]) != int(n_rows):
            raise ContractError(
                "row_group_ids has %d entries but the cloud has %d rows"
                % (len(payload["row_group_ids"]), int(n_rows)))
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return payload, hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def mask_path(mask_root, camera_id, frame,
              camera_format="cam{cam:02d}",
              subdir="pseudo_label/object_mask",
              frame_format="{frame:04d}.png"):
    return str(Path(mask_root)
               / camera_format.format(cam=int(camera_id))
               / subdir
               / frame_format.format(frame=int(frame)))


def load_id_map(path):
    """Read a DEVA/SAM automatic id map as a 2-D uint8 array.

    `PIL` is what the repository's data path already uses; `imageio` is the
    fallback. A colour PNG is refused rather than silently reduced -- an id
    map that is not single-channel is not the artifact this expects.
    """
    try:
        from PIL import Image
        with Image.open(path) as handle:
            arr = np.array(handle)
    except ImportError:  # pragma: no cover - environment dependent
        import imageio.v2 as imageio
        arr = np.asarray(imageio.imread(path))
    if arr.ndim == 3:
        if arr.shape[2] in (3, 4) and bool(
                (arr[..., 0] == arr[..., 1]).all() and (arr[..., 0] == arr[..., 2]).all()):
            arr = arr[..., 0]
        else:
            raise ContractError(
                "id map %s has shape %r: a multi-channel non-grey PNG is not a "
                "single-channel id map" % (path, tuple(arr.shape)))
    if arr.ndim != 2:
        raise ContractError(
            "id map %s has shape %r, expected 2-D" % (path, tuple(arr.shape)))
    return arr.astype(np.int64)


def save_png(path, array):
    """Write a uint8 2-D (grey) or 3-D (RGB) array. Same backends as
    `load_id_map`, in the same order, so the diagnostic never needs a
    dependency the reader does not already have."""
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.uint8))
    if arr.ndim not in (2, 3):
        raise ContractError(
            "cannot write a PNG from a %d-D array" % (int(arr.ndim),))
    try:
        from PIL import Image
        Image.fromarray(arr).save(str(path))
    except ImportError:  # pragma: no cover - environment dependent
        import imageio.v2 as imageio
        imageio.imwrite(str(path), arr)
    return str(path)


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=20)
        if out.returncode == 0:
            return out.stdout.decode("utf-8", "replace").strip()
    except Exception:  # pragma: no cover - best effort provenance only
        pass
    return ""


def _merge_config(args, config_path):
    """YAML-over-argparse merge, identical to scripts/estimate_episodes.py:1240."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    def recursive_merge(key, host):
        if isinstance(host[key], dict):
            for kk in host[key]:
                recursive_merge(kk, host[key])
        else:
            assert hasattr(args, key), "unknown config key %s" % key
            setattr(args, key, host[key])

    for key in config:
        recursive_merge(key, config)


_CAMERA_ID = re.compile(r"cam(\d+)")
_FRAME_SUFFIX = re.compile(r"cam\d+_(\d+)$")


def camera_id_of(camera):
    """`"cam07_0150"` -> 7. Refuses a name it cannot parse."""
    match = _CAMERA_ID.search(str(getattr(camera, "image_name", "")))
    if match is None:
        raise ContractError(
            "cannot parse a camera id from image_name %r"
            % (getattr(camera, "image_name", None),))
    return int(match.group(1))


def frame_index_of_name(image_name):
    """`"cam07_0150"` -> 150, or None when the name carries no frame tag."""
    match = _FRAME_SUFFIX.search(str(image_name or ""))
    if match is None:
        return None
    return int(match.group(1))


def frame_index_of(camera, frame_dt):
    """The frame index, taken from the timestamp and CHECKED against the name.

    `n3v2blender.py:343-345` writes `time = int(name[-4:]) / 30.0`, and the
    blender reader carries that straight through
    (scene/dataset_readers.py:380, :400). The DEVA masks are named by the
    SAME 4-digit index. So the two routes must agree, and a disagreement
    means the timestamp convention shifted under this script -- which would
    silently pair every render with the wrong mask. Refuse instead.
    """
    frame = int(round(float(camera.timestamp) / float(frame_dt)))
    from_name = frame_index_of_name(getattr(camera, "image_name", ""))
    if from_name is not None and from_name != frame:
        raise ContractError(
            "camera %r has timestamp %r -> frame %d but its name says frame "
            "%d; the mask filenames follow the NAME, so this pairing cannot "
            "be trusted" % (getattr(camera, "image_name", None),
                            float(camera.timestamp), frame, from_name))
    return frame, from_name is not None


# ---------------------------------------------------------------------------
# torch-dependent: leaf binding and the two measurement passes
# ---------------------------------------------------------------------------


def bind_flow_leaf(flow_leaf, n_rows, record):
    """Bind `flow_2d` to a per-row LEAF at the rasterizer call boundary.

    Adapted from scripts/membership_supervisability.py:2075-2153, minus the
    two SH leaves (this instrument does not ask the static-share question).
    Every interception VERIFIES and refuses rather than proceeding:

    * `flow_2d` arrives with one row per Gaussian -- a prefiltered subset
      would make the per-row binding a lie;
    * `means3D` / `opacities` / `means3D_static` / `opacities_static` all
      carry the cloud's row count.

    `gaussian_renderer.render` builds `flow_2d` itself
    (gaussian_renderer/__init__.py:315-317) so it is never a leaf and carries
    no gradient path; substituting at this one boundary leaves every upstream
    computation -- temporal marginal, soft routing, opacity chain, runtime
    gates -- to the real `render`. No repository file is modified and the
    symbol is restored on exit.
    """
    import contextlib
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
                    "flow_2d reached the rasterizer with shape %r, not (%d, 2): "
                    "a prefilter has subset the rows and the per-row leaf "
                    "binding would not be one-to-one" % (shape, n_rows))
            for key in ("means3D", "opacities", "means3D_static",
                        "opacities_static"):
                tensor = kwargs.get(key)
                if tensor is None or int(tensor.shape[0]) != int(n_rows):
                    raise ContractError(
                        "%s reached the rasterizer with %r rows, not %d"
                        % (key, None if tensor is None else tensor.shape[0],
                           n_rows))
            record["calls"] = int(record.get("calls", 0)) + 1
            kwargs["flow_2d"] = flow_leaf
            return self._inner(**kwargs)

    @contextlib.contextmanager
    def _ctx():
        gr.GaussianRasterizer = _LeafBindingRasterizer
        try:
            yield
        finally:
            gr.GaussianRasterizer = real

    return _ctx()


def check_flow_binding(tensor, n_rows):
    """The leaf substitution really is a LEAF of the right shape.

    scripts/membership_supervisability.py:899-919. Bound to a NON-leaf no
    gradient would ever arrive and every weight would read zero.
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


def _render_flow(camera, gaussians, pipe, background):
    from gaussian_renderer import render

    out = render(camera, gaussians, pipe, background)
    return out["flow"]


def contribution_map(camera, gaussians, pipe, background, flow_leaf, seed_mask):
    """`S(p) = sum_{i in seed} alpha_i(p) T_i(p)`, no gradient.

    Channel 0 of the leaf carries the seed indicator, so the rendered flow
    channel 0 IS the seed contribution map by the forward accumulation
    `Flow[ch] += flows[i*2+ch] * alpha * T`.
    """
    import torch

    with torch.no_grad():
        flow_leaf.data[:, 0] = seed_mask.to(flow_leaf.dtype)
        flow_leaf.data[:, 1] = 0.0
        flow_image = _render_flow(camera, gaussians, pipe, background)
        return flow_image[0].detach().clone()


def view_weights(camera, gaussians, pipe, background, flow_leaf, mask_hw):
    """`(w_in, w_out, w_total, partition_rel_deviation)` for ONE view.

    Three backward passes over ONE retained forward: the whole image, the
    mask, and its complement. `w_in + w_out` is then checked against the
    independently measured `w_total`; the check is only meaningful BECAUSE
    the complement is measured rather than subtracted.
    """
    import torch

    with torch.no_grad():
        flow_leaf.data.fill_(1.0)
    flow_image = _render_flow(camera, gaussians, pipe, background)
    if not bool(flow_image.requires_grad):
        raise ContractError(
            "the rendered flow image carries no gradient path; the flow_2d "
            "leaf binding did not take effect")

    def flow_grad(upstream_hw):
        grad_out = torch.zeros_like(flow_image)
        grad_out[0] = upstream_hw
        (grad,) = torch.autograd.grad(
            outputs=flow_image, inputs=(flow_leaf,), grad_outputs=grad_out,
            retain_graph=True, allow_unused=False)
        return grad[:, 0]

    ones_hw = torch.ones(flow_image.shape[-2:], dtype=flow_image.dtype,
                         device=flow_image.device)
    mask = mask_hw.to(device=flow_image.device, dtype=flow_image.dtype)
    w_total = flow_grad(ones_hw)
    w_in = flow_grad(mask)
    w_out = flow_grad(ones_hw - mask)

    # Floor the denominator at a fraction of the view's own largest weight
    # rather than at 1e-30: a row that received essentially no weight would
    # otherwise turn float32 dust into an enormous relative deviation and
    # refuse a healthy view.
    scale = w_total.abs().max().clamp_min(1e-30)
    denominator = w_total.abs().clamp_min(1e-6 * scale)
    rel_dev = float(((w_in + w_out - w_total).abs() / denominator).max())
    return (w_in.detach(), w_out.detach(), w_total.detach(), rel_dev)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_parser():
    from arguments import ModelParams, OptimizationParams, PipelineParams

    parser = ArgumentParser(
        description="Training-view-only per-primitive membership on real N3V")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", required=True)
    parser.add_argument("--start_checkpoint", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument(
        "--mask_root", required=True,
        help="root of the DEVA/SAM id maps: <mask_root>/<camXX>/"
             "pseudo_label/object_mask/<NNNN>.png")
    parser.add_argument("--mask_camera_format", default="cam{cam:02d}")
    parser.add_argument("--mask_subdir", default="pseudo_label/object_mask")
    parser.add_argument("--mask_frame_format", default="{frame:04d}.png")
    parser.add_argument(
        "--seed_program", default="",
        help="the T1 estimator's emitted v2 program (row_ids mode) on THIS "
             "checkpoint; its gated rows seed the vote")
    parser.add_argument(
        "--seed_groups", default="",
        help="restrict the seed to these group ids from --seed_program")
    parser.add_argument(
        "--seed_bbox3d", nargs=6, type=float, default=None,
        metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"),
        help="AUTHORED fallback seed box; recorded as authored in the report")
    parser.add_argument("--cameras", default="all")
    parser.add_argument("--held_out_cameras", default="0")
    parser.add_argument("--anchor_frames", default="150-156")
    parser.add_argument("--frames", default="150-156,195-205")
    parser.add_argument(
        "--gap_frames", nargs=2, type=int, default=None, metavar=("A", "B"),
        help="INCLUSIVE absent (occluded) frames; offset_frame = A, "
             "onset_frame = B + 1")
    parser.add_argument("--s_thresh", type=float, default=0.5)
    parser.add_argument("--id_overlap", type=float, default=0.5)
    parser.add_argument("--min_id_pixels", type=int, default=64)
    parser.add_argument("--id_min_frames", type=int, default=1)
    parser.add_argument(
        "--id_rule", default="fraction", choices=list(ID_RULES),
        help="how an anchor view's DEVA ids are harmonized. 'fraction' (the "
             "frozen default) takes an id when at least --id_overlap of ITS "
             "OWN pixels carry S > --s_thresh. 'mass' ranks ids by the S-mass "
             "they carry and takes them in descending order until "
             "--mass_cover of the view's S-mass is covered; it is scale-free "
             "in S and therefore still discriminates when a thin seed puts "
             "every pixel below --s_thresh.")
    parser.add_argument(
        "--mass_cover", type=float, default=0.8,
        help="--id_rule mass only: stop taking ids once they cover this "
             "fraction of the view's total S-mass")
    parser.add_argument(
        "--id_min_mass_frac", type=float, default=0.05,
        help="--id_rule mass only: an id must carry at least this fraction of "
             "the view's total S-mass to be admissible")
    parser.add_argument(
        "--diagnose", action="store_true",
        help="write the S maps, the id overlays, a per-view id table and the "
             "seed's projection into every camera next to --out_report, then "
             "exit 0 WITHOUT voting and WITHOUT emitting any program")
    parser.add_argument(
        "--fbox", nargs=4, type=int, default=list(DEFAULT_FBOX),
        metavar=("X0", "Y0", "X1", "Y1"),
        help="--diagnose only: an INCLUSIVE pixel box the caller supplies; "
             "the diagnostic counts how many seed rows project into it")
    parser.add_argument(
        "--fbox_frame", type=int, default=DEFAULT_FBOX_FRAME,
        help="--diagnose only: the frame at which --fbox is evaluated")
    parser.add_argument("--tau", type=float, default=VOTE_TAU)
    parser.add_argument("--partition_rel_tol", type=float,
                        default=PARTITION_REL_TOL,
                        help="refuse a view whose w_in + w_out departs from "
                             "the independently measured w_total by more than "
                             "this relative amount")
    parser.add_argument("--emit_program_rows", default="")
    parser.add_argument("--emit_program_spatial", default="")
    parser.add_argument("--fine_cells", type=int, default=64)
    parser.add_argument("--pad", type=float, default=0.05)
    parser.add_argument("--skip_checkpoint_hash", action="store_true")
    # Scene-shape arguments, matching scripts/estimate_episodes.py:1336-1344.
    # `_merge_config` asserts every YAML key already exists on `args`, so
    # omitting these makes any real config fail at merge time.
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
    # NOTE: --elgs_prereg_dir is NOT declared: it is already an
    # OptimizationParams field (arguments/__init__.py:263) and redeclaring it
    # is an argparse conflict. It is read off `opt` below.
    return parser, lp, op, pp


def resolve_seed_rows(args, xyz_np, fingerprint):
    """(seed boolean mask, provenance dict). Fails closed on any mismatch."""
    if bool(args.seed_program) == (args.seed_bbox3d is not None):
        raise ContractError(
            "supply exactly one of --seed_program or --seed_bbox3d")
    n_rows = int(xyz_np.shape[0])
    if args.seed_program:
        with open(args.seed_program, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != "adags-episode-program-v2":
            raise ContractError(
                "--seed_program is not an adags-episode-program-v2 artifact: %r"
                % (payload.get("schema_version"),))
        if payload.get("membership_mode") != "row_ids":
            raise ContractError(
                "--seed_program must be in row_ids mode so the seed binds to "
                "THIS checkpoint's rows; got %r"
                % (payload.get("membership_mode"),))
        cloud = dict(payload.get("cloud") or {})
        if int(cloud.get("n_rows", -1)) != n_rows:
            raise ContractError(
                "--seed_program was computed on a %r-row cloud but this "
                "checkpoint has %d rows" % (cloud.get("n_rows"), n_rows))
        if str(cloud.get("xyz_sha256", "")) != fingerprint:
            raise ContractError(
                "--seed_program cloud fingerprint %r does not match this "
                "checkpoint's %s; the seed column would bind to the wrong "
                "primitives" % (cloud.get("xyz_sha256"), fingerprint))
        column = np.asarray(payload.get("row_group_ids") or [], dtype=np.int64)
        if column.size != n_rows:
            raise ContractError(
                "row_group_ids has %d entries but the cloud has %d rows"
                % (int(column.size), n_rows))
        gated = sorted(int(g["group"]) for g in payload.get("groups") or [])
        if not gated:
            raise ContractError("--seed_program carries no gated group")
        if str(args.seed_groups or "").strip():
            wanted = parse_int_ranges(args.seed_groups)
            missing = sorted(set(wanted) - set(gated))
            if missing:
                raise ContractError(
                    "--seed_groups %r are not gated in the program (gated: %r)"
                    % (missing, gated))
            gated = [g for g in gated if g in set(wanted)]
        seed = np.isin(column, np.asarray(gated, dtype=np.int64))
        provenance = {
            "kind": "seed_program",
            "authored": False,
            "path": str(args.seed_program),
            "program_sha256": sha256_file(args.seed_program),
            "groups_used": gated,
            "seed_rows": int(seed.sum()),
        }
    else:
        box = [float(v) for v in args.seed_bbox3d]
        lo = np.asarray(box[:3], dtype=np.float64)
        hi = np.asarray(box[3:], dtype=np.float64)
        if bool((hi <= lo).any()):
            raise ContractError(
                "--seed_bbox3d must be strictly increasing per axis: %r" % (box,))
        seed = np.all((xyz_np >= lo) & (xyz_np <= hi), axis=1)
        provenance = {
            "kind": "seed_bbox3d",
            # LOUD: this seed is not derived from any measurement. Any claim
            # downstream inherits an authored geometric prior.
            "authored": True,
            "bbox": box,
            "seed_rows": int(seed.sum()),
        }
    if int(provenance["seed_rows"]) == 0:
        raise ContractError(
            "the seed selected zero rows; the contribution map would be "
            "identically zero and every id would be rejected")
    return seed, provenance


def run_diagnostics(args, scene, gaussians, pipe, background, flow_leaf,
                    seed_mask, seed_np, seed_provenance, guard, record,
                    by_camera, chosen_cameras, anchor_frames, frame_dt,
                    n_rows, held_out, train_stack,
                    provenance):  # pragma: no cover - needs torch + a checkpoint
    """`--diagnose`: describe the S map and the seed's projection, then STOP.

    Two limbs, deliberately separated.

    LIMB 1 runs UNDER the held-out guard and under the leaf binding, on the
    training anchor views only. It renders exactly the S map
    `choose_ids_for_view` thresholds and writes it, the DEVA id map and a
    per-view id table. It never calls a chooser, so the fail-closed refusal
    that motivated this mode cannot fire and hide the evidence.

    LIMB 2 runs OUTSIDE the guard, on purpose, and projects the seed rows into
    EVERY camera including the held-out cam00. It is pure geometry -- the same
    `get_dynamic_xyz` / `project_points_to_screen` pair
    `estimate_episodes.build_footprints` uses (scripts/estimate_episodes.py:846,
    :863-864) -- and reads no image, no mask and no rendered pixel of a
    held-out view. The bypass is structural (the limb sits after the guard's
    scope) rather than a flag poked into the guard, and `diag.json` says so in
    a top-level key so no consumer can read a cam00 number as a
    training-view-only measurement.
    """
    import torch

    from utils.motion_prior_utils import project_points_to_screen

    out_dir = Path(args.out_report).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- limb 1: the S map on the training anchor views ------------------
    views = []
    with guard, bind_flow_leaf(flow_leaf, n_rows, record):
        for cam_id in chosen_cameras:
            for frame in anchor_frames:
                camera = by_camera[cam_id][frame]
                camera_on_device = (camera.cuda() if torch.cuda.is_available()
                                    else camera)
                s_map = contribution_map(camera_on_device, gaussians, pipe,
                                         background, flow_leaf, seed_mask)
                s_np = s_map.to("cpu", torch.float32).numpy().astype(np.float64)
                path = mask_path(args.mask_root, cam_id, frame,
                                 args.mask_camera_format, args.mask_subdir,
                                 args.mask_frame_format)
                id_map = load_id_map(path)
                if id_map.shape != s_np.shape:
                    raise ContractError(
                        "id map %s has shape %r but the render is %r"
                        % (path, tuple(id_map.shape), tuple(s_np.shape)))
                summary = diagnose_view_summary(id_map, s_np, args.s_thresh)
                stem = "diag_cam%02d_%04d" % (int(cam_id), int(frame))
                linear = out_dir / (stem + "_S.png")
                by_max = out_dir / (stem + "_Smax.png")
                overlay = out_dir / (stem + "_overlay.png")
                save_png(linear, scale_to_uint8(s_np, 1.0))
                save_png(by_max, scale_to_uint8(s_np, summary["s_max"]))
                save_png(overlay, id_palette_rgb(
                    id_map, marked=(s_np > DIAG_OVERLAY_S)))
                summary.update({
                    "camera": int(cam_id),
                    "frame": int(frame),
                    "mask_path": path,
                    "images": {
                        "s_clipped_to_unit": linear.name,
                        "s_scaled_by_view_max": by_max.name,
                        "id_overlay": overlay.name,
                        "overlay_marks_s_above": float(DIAG_OVERLAY_S),
                        "overlay_marker_rgb": list(DIAG_MARKER_RGB),
                    },
                })
                views.append(summary)
                top = summary["top_ids_by_s_mass"]
                print("  cam%02d f%04d S max %.4g mean %.4g | >%.2f %d px | "
                      "top id %s mass_frac %.4f"
                      % (int(cam_id), int(frame), summary["s_max"],
                         summary["s_mean"], DIAG_OVERLAY_S,
                         summary["pixels_over"]["%g" % DIAG_OVERLAY_S],
                         (top[0]["id"] if top else None),
                         (top[0]["mass_fraction"] if top else 0.0)))

    # ---- limb 2: pure projection, guard deliberately out of scope --------
    all_cameras = {}
    for camera in list(train_stack) + list(scene.test_cameras.get(1.0, []) or []):
        cam_id = camera_id_of(camera)
        frame, _ = frame_index_of(camera, frame_dt)
        all_cameras.setdefault(cam_id, {})[frame] = camera
    seed_rows = np.nonzero(np.asarray(seed_np, dtype=bool))[0]
    seed_index = torch.from_numpy(
        np.ascontiguousarray(seed_rows.astype(np.int64))).to(
            gaussians._xyz.device)
    want_frames = sorted(set(int(f) for f in anchor_frames)
                         | {int(args.fbox_frame)})
    held_set = set(int(c) for c in held_out)
    projection_cameras = []
    for cam_id in sorted(all_cameras):
        per_frame = []
        for frame in want_frames:
            camera = all_cameras[cam_id].get(frame)
            if camera is None:
                per_frame.append({"frame": int(frame), "present": False})
                continue
            # No `.cuda()` and no dataset indexing: `full_proj_transform`,
            # `image_width` and `image_height` are all this needs, so not one
            # byte of a held-out image is touched.
            with torch.no_grad():
                points = gaussians.get_dynamic_xyz(
                    float(camera.timestamp)).detach()
                xy, valid = project_points_to_screen(points[seed_index], camera)
            xy_np = xy.to("cpu", torch.float32).numpy().reshape(-1, 2)
            valid_np = valid.reshape(-1).to("cpu").numpy().astype(bool)
            per_frame.append({
                "frame": int(frame),
                "present": True,
                "image_width": int(camera.image_width),
                "image_height": int(camera.image_height),
                "bbox_on_screen": projected_bbox(xy_np, valid_np),
                "bbox_all_projected": projected_bbox(xy_np, None),
                "n_seed_rows_in_fbox": count_in_fbox(xy_np, valid_np, args.fbox),
            })
        projection_cameras.append({
            "camera": int(cam_id),
            "is_held_out": bool(int(cam_id) in held_set),
            "frames": per_frame,
        })

    diagnostic = {
        "schema": DIAG_SCHEMA,
        "held_out_projection_is_scoring_side_bookkeeping": True,
        "held_out_projection_note": (
            "LOUD: the `projection` block below covers EVERY camera, the "
            "HELD-OUT cam00 included, with the held-out guard deliberately "
            "out of scope for that limb. It is pure geometry -- "
            "get_dynamic_xyz + project_points_to_screen -- and reads no "
            "image, no DEVA mask and no rendered pixel of any held-out view. "
            "It exists to locate a caller-supplied frame box on the scoring "
            "side. NO number in this block may enter a training-view-only "
            "claim, and nothing in this file is a measurement of anything."),
        "voted": False,
        "programs_emitted": False,
        "provenance": dict(provenance),
        "settings": {
            "cameras": [int(c) for c in chosen_cameras],
            "held_out_cameras": sorted(held_set),
            "anchor_frames": [int(f) for f in anchor_frames],
            "s_thresh": float(args.s_thresh),
            "id_rule": str(args.id_rule),
            "mass_cover": float(args.mass_cover),
            "id_min_mass_frac": float(args.id_min_mass_frac),
            "min_id_pixels": int(args.min_id_pixels),
            "id_overlap": float(args.id_overlap),
            "diag_s_levels": [float(v) for v in DIAG_S_LEVELS],
            "overlay_s": float(DIAG_OVERLAY_S),
            "top_k": int(DIAG_TOP_K),
            "fbox": [int(v) for v in args.fbox],
            "fbox_frame": int(args.fbox_frame),
        },
        "seed": dict(seed_provenance),
        "guard": dict(guard.checks),
        "rasterizer_intercepts": int(record.get("calls", 0)),
        "views": views,
        "projection": {
            "position_source": (
                "gaussians.get_dynamic_xyz(camera.timestamp), the same call "
                "scripts/estimate_episodes.build_footprints makes at "
                "scripts/estimate_episodes.py:863"),
            "projection_fn": (
                "utils.motion_prior_utils.project_points_to_screen; "
                "`bbox_on_screen` keeps only points whose NDC lies in "
                "[-1, 1]^2, `bbox_all_projected` keeps every seed row"),
            "fbox": [int(v) for v in args.fbox],
            "fbox_convention": "x0 y0 x1 y1, INCLUSIVE, on rounded pixels",
            "fbox_frame": int(args.fbox_frame),
            "n_seed_rows": int(seed_rows.size),
            "cameras": projection_cameras,
        },
    }
    diag_path = out_dir / "diag.json"
    with open(str(diag_path), "w", encoding="utf-8") as handle:
        json.dump(diagnostic, handle, indent=1, sort_keys=True)

    for entry in projection_cameras:
        for frame_entry in entry["frames"]:
            if frame_entry.get("present") and int(
                    frame_entry["frame"]) == int(args.fbox_frame):
                print("cam%02d%s frame %d: %d / %d seed rows in fbox %r"
                      % (entry["camera"], " (HELD OUT)" if entry["is_held_out"]
                         else "", frame_entry["frame"],
                         frame_entry["n_seed_rows_in_fbox"], int(seed_rows.size),
                         [int(v) for v in args.fbox]))
    print("diagnose: %d anchor views, %d cameras projected -> %s"
          % (len(views), len(projection_cameras), diag_path))
    print("diagnose: NO vote, NO program emitted; exiting 0")
    return 0


def main(argv=None):  # pragma: no cover - requires torch + CUDA + a checkpoint
    import torch

    from elgs.trainer_hooks import (
        _load_episode_program_v2,
        build_interval_config,
        infer_frame_dt,
        load_structural_prereg,
    )

    parser, lp, op, pp = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    _merge_config(args, args.config)

    held_out = tuple(parse_int_ranges(args.held_out_cameras))
    anchor_frames = parse_int_ranges(args.anchor_frames)
    measure_frames = parse_int_ranges(args.frames)

    run_dir = os.environ.get("ADAGS_RUN_DIR", "").strip()
    if not str(getattr(args, "model_path", "") or "").strip():
        if not run_dir:
            raise ContractError("--model_path is required when ADAGS_RUN_DIR is unset")
        args.model_path = run_dir
    os.makedirs(args.model_path, exist_ok=True)
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    dataset_params = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    # Refusals BEFORE the scene is built: both flags break the per-row binding
    # (scripts/membership_supervisability.py:1971-1985).
    if bool(getattr(pipe, "compute_cov3D_python", False)):
        raise ContractError(
            "compute_cov3D_python is set: gaussian_renderer.render prefilters "
            "flow_2d by marginal_t > 0.05 before the rasterizer call "
            "(gaussian_renderer/__init__.py:319-346), so the per-row leaf "
            "could not be bound one-to-one.")
    if bool(getattr(pipe, "convert_SHs_python", False)):
        raise ContractError(
            "convert_SHs_python is set: the renderer then passes "
            "colors_precomp and the SH path this instrument's preconditions "
            "assume is absent.")

    from scene import Scene
    from scene.gaussian_model import GaussianModel

    gaussians = GaussianModel(
        dataset_params.sh_degree, gaussian_dim=args.gaussian_dim,
        time_duration=args.time_duration, rot_4d=args.rot_4d,
        force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0,
    )
    scene = Scene(dataset_params, gaussians, num_pts=args.num_pts,
                  num_pts_ratio=args.num_pts_ratio,
                  time_duration=args.time_duration, shuffle=False)
    scene.opt = opt
    gaussians.training_setup(opt)
    model_params, _ = torch.load(args.start_checkpoint)
    gaussians.restore(model_params, opt)

    if int(gaussians.gaussian_dim) != 4:
        raise ContractError(
            "the flow carrier channel exists only on the 4D path; got dim %d"
            % gaussians.gaussian_dim)
    if not bool(getattr(gaussians, "enable_soft_routing", False)):
        raise ContractError(
            "soft routing is off: with it off the static twin is a SEPARATE "
            "parameter set (gaussian_renderer/__init__.py:363-370) and the "
            "dynamic-only flow channel no longer accounts for the row's "
            "rendered contribution")
    if bool(getattr(gaussians, "enable_soft_routing", False)) != bool(
            getattr(opt, "enable_soft_routing", False)):
        raise ContractError(
            "enable_soft_routing disagrees between the checkpoint and the "
            "config; the restored substrate is not the one this config names")

    device = gaussians._xyz.device
    n_rows = int(gaussians._xyz.shape[0])
    xyz_np = gaussians._xyz.detach().to("cpu", torch.float32).numpy()
    fingerprint = cloud_fingerprint_from_array(xyz_np)

    seed_np, seed_provenance = resolve_seed_rows(args, xyz_np.astype(np.float64),
                                                 fingerprint)
    seed_mask = torch.from_numpy(np.ascontiguousarray(seed_np)).to(device)

    train_stack = scene.train_cameras[1.0]
    if not train_stack:
        raise ContractError("the train split is empty")
    timestamps = [float(c.timestamp) for c in train_stack]
    frame_dt = infer_frame_dt(timestamps)
    time_span = max(timestamps) - min(timestamps)
    prereg = load_structural_prereg(str(opt.elgs_prereg_dir))
    interval_config = build_interval_config(prereg, time_span, frame_dt)

    by_camera = {}
    n_name_checked = 0
    for camera in train_stack:
        cam_id = camera_id_of(camera)
        frame, name_checked = frame_index_of(camera, frame_dt)
        n_name_checked += int(name_checked)
        by_camera.setdefault(cam_id, {})[frame] = camera
    available = sorted(by_camera)
    chosen_cameras = parse_camera_spec(args.cameras, available)
    chosen_cameras = [c for c in chosen_cameras if c not in set(held_out)]
    if not chosen_cameras:
        raise ContractError("no training camera survived the held-out filter")

    used_cameras = []
    missing = []
    for cam_id in chosen_cameras:
        for frame in sorted(set(anchor_frames) | set(measure_frames)):
            camera = by_camera[cam_id].get(frame)
            if camera is None:
                missing.append((cam_id, frame))
            else:
                used_cameras.append(camera)
    if missing:
        raise ContractError(
            "the train split is missing %d requested (camera, frame) views, "
            "first few %r" % (len(missing), missing[:5]))

    background = torch.tensor(
        [1, 1, 1] if dataset_params.white_background else [0, 0, 0],
        dtype=torch.float32, device=device)

    guard = HeldOutGuard(scene=scene, opt=opt, held_out_cameras=held_out)
    guard.assert_manifests_empty()
    guard.assert_no_held_out_ids(chosen_cameras)
    guard.assert_train_only(used_cameras, train_stack)

    flow_leaf = torch.ones((n_rows, 2), dtype=torch.float32, device=device,
                           requires_grad=True)
    check_flow_binding(flow_leaf, n_rows)
    record = {"calls": 0}

    if args.diagnose:
        print("cameras %s | anchor %s | seed rows %d / %d | DIAGNOSE ONLY"
              % (chosen_cameras, anchor_frames, int(seed_np.sum()), n_rows))
        return run_diagnostics(
            args, scene, gaussians, pipe, background, flow_leaf, seed_mask,
            seed_np, seed_provenance, guard, record, by_camera,
            chosen_cameras, anchor_frames, frame_dt, n_rows, held_out,
            train_stack,
            provenance={
                "checkpoint": str(args.start_checkpoint),
                "config": str(args.config),
                "source_path": str(getattr(args, "source_path", "")),
                "model_path": str(args.model_path),
                "mask_root": str(args.mask_root),
                "commit": git_commit(),
                "n_rows": n_rows,
                "cloud_xyz_sha256": fingerprint,
                "frame_dt": float(frame_dt),
                "instrument": REPORT_SCHEMA,
            })

    w_in_by_camera = {}
    w_out_by_camera = {}
    harmonization = []
    per_view = []
    counters = {"forward_renders": 0, "backward_passes": 0, "masks_loaded": 0}
    max_partition_rel_dev = 0.0
    empty_mask_views = []
    started = time.perf_counter()

    print("cameras %s | anchor %s | frames %s | seed rows %d / %d"
          % (chosen_cameras, anchor_frames, measure_frames,
             int(seed_np.sum()), n_rows))

    with guard, bind_flow_leaf(flow_leaf, n_rows, record):
        for cam_id in chosen_cameras:
            # --- stage A: harmonize this camera's DEVA ids --------------
            per_frame_chosen, per_frame_stats = [], []
            for frame in anchor_frames:
                camera = by_camera[cam_id][frame]
                camera_on_device = (camera.cuda() if torch.cuda.is_available()
                                    else camera)
                s_map = contribution_map(camera_on_device, gaussians, pipe,
                                         background, flow_leaf, seed_mask)
                counters["forward_renders"] += 1
                path = mask_path(args.mask_root, cam_id, frame,
                                 args.mask_camera_format, args.mask_subdir,
                                 args.mask_frame_format)
                id_map = load_id_map(path)
                counters["masks_loaded"] += 1
                s_np = s_map.to("cpu", torch.float32).numpy()
                if id_map.shape != s_np.shape:
                    raise ContractError(
                        "id map %s has shape %r but the render is %r"
                        % (path, tuple(id_map.shape), tuple(s_np.shape)))
                chosen, stats, rule_info = choose_ids_by_rule(
                    args.id_rule, id_map, s_np, args.s_thresh, args.id_overlap,
                    args.min_id_pixels, args.mass_cover, args.id_min_mass_frac)
                if not chosen:
                    # FAIL CLOSED. A camera/frame with no chosen id has no
                    # object mask, and measuring it against an empty mask
                    # would silently report every row as a non-member.
                    if args.id_rule == "mass":
                        raise ContractError(
                            "no DEVA id met the mass bar for cam%02d frame %d "
                            "(mass_cover %r, id_min_mass_frac %r, "
                            "min_id_pixels %r). The view's total S-mass is "
                            "%.6g, of which %.4f sits on the ignored id. Top "
                            "5 ids by S-mass (s_mass, mass_fraction, pixels, "
                            "id): %r"
                            % (cam_id, frame, args.mass_cover,
                               args.id_min_mass_frac, args.min_id_pixels,
                               rule_info.get("s_mass_view", 0.0),
                               rule_info.get("s_mass_ignored_id_fraction", 0.0),
                               [(round(s["s_mass"], 6),
                                 round(s["mass_fraction"], 6),
                                 s["pixels"], s["id"]) for s in stats[:5]]))
                    raise ContractError(
                        "no DEVA id met the overlap bar for cam%02d frame %d "
                        "(s_thresh %r, id_overlap %r, min_id_pixels %r). "
                        "Best fractions: %r"
                        % (cam_id, frame, args.s_thresh, args.id_overlap,
                           args.min_id_pixels,
                           sorted((round(s["fraction"], 4), s["id"])
                                  for s in stats)[-5:]))
                per_frame_chosen.append(chosen)
                per_frame_stats.append({
                    "frame": int(frame),
                    "mask_path": path,
                    "chosen_ids": chosen,
                    "pixels_over_s_thresh": int((s_np > args.s_thresh).sum()),
                    "ids": stats,
                    "id_rule_info": dict(rule_info),
                })
            camera_ids_chosen, hit_counts = aggregate_chosen_ids(
                per_frame_chosen, args.id_min_frames)
            if not camera_ids_chosen:
                raise ContractError(
                    "cam%02d: no DEVA id was chosen on at least %d anchor "
                    "frames" % (cam_id, int(args.id_min_frames)))
            harmonization.append({
                "camera": int(cam_id),
                "chosen_ids": camera_ids_chosen,
                "anchor_hit_counts": hit_counts,
                "anchor_frames": per_frame_stats,
            })
            if args.verbose:
                print("  cam%02d ids %r" % (cam_id, camera_ids_chosen))

            # --- stage B: per-row weights on the measurement frames -----
            acc_in = torch.zeros(n_rows, dtype=torch.float64, device=device)
            acc_out = torch.zeros(n_rows, dtype=torch.float64, device=device)
            for frame in measure_frames:
                camera = by_camera[cam_id][frame]
                camera_on_device = (camera.cuda() if torch.cuda.is_available()
                                    else camera)
                path = mask_path(args.mask_root, cam_id, frame,
                                 args.mask_camera_format, args.mask_subdir,
                                 args.mask_frame_format)
                id_map = load_id_map(path)
                counters["masks_loaded"] += 1
                mask_np = mask_from_ids(id_map, camera_ids_chosen)
                mask_pixels = int(mask_np.sum())
                if mask_pixels == 0:
                    empty_mask_views.append([int(cam_id), int(frame)])
                    raise ContractError(
                        "cam%02d frame %d: the harmonized id set %r covers no "
                        "pixel. The DEVA track for this object is absent in "
                        "this frame; restrict --frames or --cameras rather "
                        "than measuring an empty mask."
                        % (cam_id, frame, camera_ids_chosen))
                mask_t = torch.from_numpy(
                    np.ascontiguousarray(mask_np, dtype=np.float32)).to(device)
                w_in, w_out, w_total, rel_dev = view_weights(
                    camera_on_device, gaussians, pipe, background, flow_leaf,
                    mask_t)
                counters["forward_renders"] += 1
                counters["backward_passes"] += 3
                if rel_dev > float(args.partition_rel_tol):
                    raise ContractError(
                        "cam%02d frame %d: w_in + w_out deviates from w_total "
                        "by relative %.3e > %.3e. The mask, the complement or "
                        "the leaf binding is wrong; this cannot be read as a "
                        "result." % (cam_id, frame, rel_dev,
                                     float(args.partition_rel_tol)))
                max_partition_rel_dev = max(max_partition_rel_dev, rel_dev)
                floor = -NEGATIVE_WEIGHT_REL_TOL * float(
                    w_total.abs().max().clamp_min(1e-30))
                if float(w_in.min()) < floor or float(w_out.min()) < floor:
                    raise ContractError(
                        "cam%02d frame %d: a compositing weight is materially "
                        "negative (min in %.3e, min out %.3e); the binding is "
                        "wrong" % (cam_id, frame, float(w_in.min()),
                                   float(w_out.min())))
                acc_in += w_in.clamp_min(0).to(torch.float64)
                acc_out += w_out.clamp_min(0).to(torch.float64)
                per_view.append({
                    "camera": int(cam_id), "frame": int(frame),
                    "mask_pixels": mask_pixels,
                    "rows_with_nonzero_weight": int(
                        ((w_in + w_out).detach() != 0).sum()),
                    "partition_rel_deviation": rel_dev,
                })
            w_in_by_camera[cam_id] = acc_in.to("cpu").numpy()
            w_out_by_camera[cam_id] = acc_out.to("cpu").numpy()

    elapsed = time.perf_counter() - started

    # ---- the vote -------------------------------------------------------
    w_in_total = np.zeros(n_rows, dtype=np.float64)
    w_out_total = np.zeros(n_rows, dtype=np.float64)
    for cam_id in chosen_cameras:
        w_in_total += w_in_by_camera[cam_id]
        w_out_total += w_out_by_camera[cam_id]
    members, share, vote_stats = membership_vote(w_in_total, w_out_total,
                                                 tau=args.tau)
    n_members = int(members.sum())
    if n_members == 0:
        raise ContractError(
            "the vote admitted zero rows; there is no program to emit and "
            "`seed_families` would refuse the artifact anyway "
            "(elgs/trainer_hooks.py:950-957)")

    per_camera_agreement = []
    for cam_id in chosen_cameras:
        a = w_in_by_camera[cam_id][members]
        b = w_out_by_camera[cam_id][members]
        observed = (a + b) > 0
        per_camera_agreement.append({
            "camera": int(cam_id),
            "members_observed": int(observed.sum()),
            # Precision-LIKE, not precision: there is no ground truth here.
            # It is the fraction of voted members this camera alone would
            # also have called members.
            "fraction_w_in_gt_w_out": (
                float((a[observed] > b[observed]).mean())
                if int(observed.sum()) else None),
            "member_w_in_sum": float(a.sum()),
            "member_w_out_sum": float(b.sum()),
        })

    loco = []
    for cam_id in chosen_cameras:
        left_in = w_in_total - w_in_by_camera[cam_id]
        left_out = w_out_total - w_out_by_camera[cam_id]
        sub_members, _, sub_stats = membership_vote(left_in, left_out,
                                                    tau=args.tau)
        loco.append({
            "left_out_camera": int(cam_id),
            "n_members": int(sub_members.sum()),
            "jaccard_with_full": jaccard(members, sub_members),
            "n_added": int(np.logical_and(sub_members, ~members).sum()),
            "n_dropped": int(np.logical_and(members, ~sub_members).sum()),
            "n_eligible": int(sub_stats["n_eligible"]),
        })

    member_xyz = xyz_np[members].astype(np.float64)
    geometry = {
        "n_members": n_members,
        "bbox_min": [float(v) for v in member_xyz.min(axis=0)],
        "bbox_max": [float(v) for v in member_xyz.max(axis=0)],
        "centroid": [float(v) for v in member_xyz.mean(axis=0)],
        "seed_rows": int(seed_np.sum()),
        "members_in_seed": int(np.logical_and(members, seed_np).sum()),
        "seed_rows_not_members": int(np.logical_and(seed_np, ~members).sum()),
    }

    # ---- programs -------------------------------------------------------
    programs = {}
    if args.emit_program_rows or args.emit_program_spatial:
        if args.gap_frames is None:
            raise ContractError("--gap_frames A B is required to emit a program")
        absent_lo, absent_hi = int(args.gap_frames[0]), int(args.gap_frames[1])
        if absent_hi < absent_lo:
            raise ContractError("--gap_frames must be ascending")
        offset_frame, onset_frame = absent_lo, absent_hi + 1
        gap = inset_gap_seconds(offset_frame, onset_frame, frame_dt,
                                interval_config.w)
        lo, span = padded_bbox(member_xyz, args.pad)
        keys_all = voxel_keys(xyz_np.astype(np.float64), lo, span,
                              args.fine_cells)
        inside_all = voxel_inside(xyz_np.astype(np.float64), lo, span)
        member_keys = sorted(set(int(k) for k in keys_all[members]))
        # SELF-CHECK: what the spatial program would gate on THIS very cloud.
        # It is an upper bound on its fidelity -- a fresh create_from_pcd
        # cloud can only do worse, because it puts different rows in the same
        # cells.
        spatial_hits = np.logical_and(
            inside_all, np.isin(keys_all, np.asarray(member_keys, dtype=np.int64)))
        if not bool(np.all(spatial_hits[members])):
            # The padding exists precisely so every member is strictly
            # interior. If one is not, `resolve_v2_membership` would force it
            # to -1 and the spatial program would silently drop a voted row.
            raise ContractError(
                "%d voted members fall outside the padded box or its cell "
                "set; raise --pad" % int((~spatial_hits[members]).sum()))
        source = {
            "checkpoint": str(args.start_checkpoint),
            "config": str(args.config),
            "instrument": REPORT_SCHEMA,
            "seed": dict(seed_provenance),
            "cameras": [int(c) for c in chosen_cameras],
            "frames": [int(f) for f in measure_frames],
            "tau": float(args.tau),
        }
        if args.id_rule != "fraction":
            # The default rule is the one the schema was frozen with, so its
            # ABSENCE means `fraction` and a default artifact keeps the bytes
            # -- and therefore the sha256 -- it had before this rule existed.
            # A non-default rule is stamped into the artifact that carries it.
            source["id_rule"] = str(args.id_rule)
            source["mass_cover"] = float(args.mass_cover)
            source["id_min_mass_frac"] = float(args.id_min_mass_frac)
        row_column = np.where(members, EMITTED_GROUP_ID, -1).astype(np.int64)

        emitted = []
        if args.emit_program_rows:
            payload, digest = build_v2_program(
                "row_ids", row_column.tolist(), n_rows, fingerprint, lo, span,
                args.fine_cells, member_keys, gap, offset_frame, onset_frame,
                n_members, interval_config.w, frame_dt, source=source)
            emitted.append((args.emit_program_rows, payload, digest, "row_ids"))
        if args.emit_program_spatial:
            payload, digest = build_v2_program(
                "spatial_voxel", None, n_rows, fingerprint, lo, span,
                args.fine_cells, member_keys, gap, offset_frame, onset_frame,
                n_members, interval_config.w, frame_dt, source=source)
            emitted.append((args.emit_program_spatial, payload, digest,
                            "spatial_voxel"))

        for path, payload, digest, mode in emitted:
            # Validate against the REAL consumer before writing: this is the
            # function `seed_families` calls, so an inexpressible gap or a
            # malformed block fails here rather than at seeding time.
            _load_episode_program_v2(payload, interval_config)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=1, sort_keys=True)
            programs[mode] = {
                "path": str(path),
                "sha256": digest,
                "validated_by": "elgs.trainer_hooks._load_episode_program_v2",
            }
            print("emitted %s program %s sha256 %s" % (mode, path, digest))

        programs["gap"] = {
            "absent_frames_inclusive": [absent_lo, absent_hi],
            "offset_frame": offset_frame,
            "onset_frame": onset_frame,
            "gap_seconds": [float(gap[0]), float(gap[1])],
            "presence_edge_half_width_w": float(interval_config.w),
            "frame_dt": float(frame_dt),
            "convention": ("scripts/estimate_episodes.py:502-520; offset_frame "
                           "is the first ABSENT frame, onset_frame the first "
                           "PRESENT frame after the gap"),
        }
        programs["fine_grid"] = {
            "cells_per_axis": int(args.fine_cells),
            "pad_fraction": float(args.pad),
            "lo": [float(v) for v in lo],
            "span": [float(v) for v in span],
            "n_member_cells": len(member_keys),
            "self_check_rows_gated_on_this_cloud": int(spatial_hits.sum()),
            "self_check_jaccard_with_members": jaccard(members, spatial_hits),
            "self_check_note": (
                "an UPPER bound on spatial_voxel fidelity: a fresh "
                "create_from_pcd cloud puts different rows in these cells"),
        }

    report = {
        "schema": REPORT_SCHEMA,
        "provenance": {
            "checkpoint": str(args.start_checkpoint),
            "checkpoint_sha256": ("" if args.skip_checkpoint_hash
                                  else sha256_file(args.start_checkpoint)),
            "config": str(args.config),
            "source_path": str(getattr(args, "source_path", "")),
            "model_path": str(args.model_path),
            "mask_root": str(args.mask_root),
            "commit": git_commit(),
            "n_rows": n_rows,
            "cloud_xyz_sha256": fingerprint,
            "frame_dt": float(frame_dt),
            "time_span": float(time_span),
            "elgs_prereg_dir": str(opt.elgs_prereg_dir),
            "elapsed_seconds": float(elapsed),
            "train_views": len(train_stack),
            "train_views_with_name_frame_check": int(n_name_checked),
        },
        "settings": {
            "cameras": [int(c) for c in chosen_cameras],
            "held_out_cameras": [int(c) for c in held_out],
            "anchor_frames": [int(f) for f in anchor_frames],
            "frames": [int(f) for f in measure_frames],
            "s_thresh": float(args.s_thresh),
            "id_overlap": float(args.id_overlap),
            "min_id_pixels": int(args.min_id_pixels),
            "id_min_frames": int(args.id_min_frames),
            "id_rule": str(args.id_rule),
            "id_rule_default": "fraction",
            "mass_cover": float(args.mass_cover),
            "id_min_mass_frac": float(args.id_min_mass_frac),
            "tau": float(args.tau),
            "tau_default_provenance": (
                "0.50, the frozen operating point of "
                "scripts/membership_supervisability.py:280"),
            "fine_cells": int(args.fine_cells),
            "pad": float(args.pad),
        },
        "seed": seed_provenance,
        "guard": dict(guard.checks),
        "counters": dict(counters),
        "rasterizer_intercepts": int(record.get("calls", 0)),
        "max_partition_rel_deviation": float(max_partition_rel_dev),
        "partition_rel_tol": float(args.partition_rel_tol),
        "partition_rel_tol_default": PARTITION_REL_TOL,
        "empty_mask_views": empty_mask_views,
        "harmonization": harmonization,
        "per_view": per_view,
        "vote": vote_stats,
        "per_camera_agreement": per_camera_agreement,
        "leave_one_camera_out": loco,
        "geometry": geometry,
        "share_quantiles": {
            str(q): float(np.quantile(share[(w_in_total + w_out_total) > 0], q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "programs": programs,
    }
    with open(args.out_report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, sort_keys=True)

    print("members %d / %d (eligible %d, abstain %d) in %.1f s"
          % (n_members, n_rows, vote_stats["n_eligible"],
             vote_stats["n_abstained"], elapsed))
    print("LOCO jaccard min %.4f" % min(e["jaccard_with_full"] for e in loco))
    print("report %s" % args.out_report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
