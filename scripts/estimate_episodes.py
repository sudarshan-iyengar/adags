#!/usr/bin/env python3
"""Phase T1: training-view-only episode-boundary estimation, MEASUREMENT ONLY.

EXPLORATORY. This script answers exactly one question and changes nothing:

    can a training-view-only estimator place episode onset/offset to
    FRAME accuracy?

It does NOT retrain, does NOT emit an `elgs-oracle-episodes-v1` config, and
does NOT touch `seed_families`. Feeding an estimated program back into
training is phase T2 and is justified only if T1 passes. That sequencing is
deliberate: the measured small-mistiming control (experiment 191,
research-wiki/operations/lrv3-local-presence-corrected-cell-2026-08-20.md
RESULT 2) shows a 2-frame timing error costs -2.39 dB at the return, i.e. is
WORSE than not gating at all. An estimator that is not frame-accurate has
negative value and must not be retrained.


THE SIGNAL
----------
Per-primitive contribution mass does not exist in this codebase: the forward
rasterizer contains zero `atomicAdd` (diff-gaussian-rasterization/
cuda_rasterizer/forward.cu), so no per-Gaussian accumulator is written during
blending. Extracting one would require new CUDA.

Instead the TOTAL OPACITY GATE is used as an ablation operator, through the
machinery that already exists and is already tested:

  * `gaussians._elgs_family_ids` carries one family id per row, -1 =
    unassigned (scene/gaussian_model.py:234);
  * `ElgsRuntime.gated_row_mask` gates exactly the rows whose family carries
    K >= 2 (elgs/runtime.py:215-237);
  * `local_presence_multipliers` applies presence to BOTH routed branches for
    gated rows and leaves every other row at the ordinary temporal marginal
    with an unmodulated static twin (elgs/presence.py:80-103);
  * `_elgs_presence_override` scores a counterfactual without mutating live
    state (scene/gaussian_model.py:245-257, elgs/runtime.py:154-164).

Ablating one group G therefore means: put ONLY G's rows in a gated family,
leave every other row at -1, and override G's family with the K=0 empty
program so its presence is exactly zero. The result is byte-faithful to the
substrate everywhere outside G, and an exact total gate on G. The unablated
base render simply detaches the runtime, so it is the pure substrate.

The per-frame explanatory series is

    E_G(t) = mean over sampled train cameras of
             [ L1(G ablated) - L1(G present) ] restricted to G's footprint

so E_G is large while G explains the training views and ~0 while it does not.


ANTI-LEAKAGE: THE STAGE BOUNDARY IS LOAD-BEARING
------------------------------------------------
Allowed estimator inputs: the trained checkpoint, TRAIN cameras only,
training-view GT RGB, and renderer outputs.

Forbidden as estimator input: `scene.getTestCameras()`, anything under
`gt_identity/`, `event_spec.json`'s `presence_frames` / `event_object`, and
BOTH the `region` AND the `gaps` of any `configs/lrv3/oracle_*.json`. The
oracle `region` is the event object's true geometry, identical to
`event_spec.json`; an estimator that computes boundaries but inherits that
region is still oracle-supervised and would silently fake a success. Group
membership here is derived from the trained cloud's own bounding box and
nothing else.

The two stages are physically separated:

    estimate_episode_program(...)   sees no ground truth. It is not handed
                                    the Scene, the source path, the event
                                    spec, or any identity buffer. Its whole
                                    argument list is checked by
                                    tests/test_estimate_episodes.py.

    score_program(...)              runs strictly afterwards, on an already
                                    frozen program, and is the only place
                                    ground truth is opened.

REORDERING THESE TWO STAGES, OR PASSING ANY SCORING INPUT INTO THE
ESTIMATION STAGE, INVALIDATES THE RESULT. The estimate is frozen and hashed
before scoring begins; the hash is recorded in the report so that a program
scored under one hash can never be confused with a program revised after
seeing ground truth.

The separation is additionally enforced at runtime by `LeakageGuard`, which
during estimation only: refuses any file open under `gt_identity`, any
`event_spec.json`, and any `oracle_*.json`; replaces `Scene.getTestCameras`
with a raiser; and asserts the two external event manifests are empty.


GROUPING CHOICE
---------------
Candidate groups are a FIXED VOXEL GRID over the trained cloud's own bounding
box (`VOXEL_CELLS_PER_AXIS` per axis), with cells below `MIN_GROUP_ROWS` rows
discarded to substrate. Chosen over the epsilon-graph connected components of
`scene/packet_birth.py:433-456` because:

  * it is provably oracle-free -- it depends only on the cloud's own extent,
    never on any authored region;
  * it is the SAME idiom `seed_families` uses (elgs/trainer_hooks.py:686-693),
    so a phase-T2 pass feeding a computed program into `seed_families`
    inherits identical granularity and the preregistered `max_families = 512`
    cap binds identically -- T1 and T2 stay comparable by construction;
  * it is deterministic and O(N). `packet_components` builds an N x N `cdist`
    (packet_birth.py:450) and derives its epsilon from the median
    nearest-neighbour distance, which is O(N^2) memory at 150k rows and, being
    density-dependent, would produce different groups for the substrate and
    for any retrained model -- breaking exactly the T1 -> T2 comparability
    this pass exists to establish.

`MIN_GROUP_ROWS` is imported from `scene.packet_birth.MIN_PACKET_ROWS` so the
sub-threshold rule stays single-sourced.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import re
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

# INSERT, not append: the admitted image ships a `pointops2` whose `functions`
# subpackage is absent and `utils.general_utils` imports it unconditionally.
# Same rationale as scripts/eval_lrv1_event.py and scripts/consolidate_packets.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from depth_visibility.errors import ContractError  # noqa: E402
from elgs.families import FamilyRegistry  # noqa: E402
from elgs.intervals import IntervalState, inverse  # noqa: E402
from elgs.runtime import ElgsRuntime, ScheduleAnchors  # noqa: E402

# NOTE ON IMPORT WEIGHT. Everything above is importable without a CUDA
# extension, so the whole decision rule, the leakage guard, the grouping and
# the ablation wiring are unit-testable on a workstation. `scene` (and
# therefore `scene.packet_birth`), `gaussian_renderer`, `arguments`,
# `utils.motion_prior_utils`, `yaml` and `numpy` are imported LAZILY inside the
# functions that need them: importing `scene` pulls the compiled
# `pointops2_cuda`, which is unavailable outside the admitted image. Keep it
# that way -- the auditable logic must not require a GPU to read or to test.

REPORT_SCHEMA = "adags-episode-estimate-v1"

# ---------------------------------------------------------------------------
# Frozen constants. Any change here is a NEW specification, not a flag.
# ---------------------------------------------------------------------------

#: Voxel cells per axis for candidate grouping; matches the preregistered
#: family-seeding grid (configs/elgs/prereg_structural_v1.json).
VOXEL_CELLS_PER_AXIS = 8
#: Groups below this many rows stay substrate. This is a MIRROR of
#: `scene.packet_birth.MIN_PACKET_ROWS`, restated here so the decision rule
#: stays importable without the CUDA extension that `scene` pulls in.
#: `assert_min_group_rows_single_sourced()` fails closed if the two ever
#: diverge, and `main()` calls it before any measurement runs.
MIN_GROUP_ROWS = 4
#: Contrast test: presence-varying iff the separation between the two modes
#: is at least this multiple of the WITHIN-mode robust scale. See
#: `contrast_statistics` for the rule and for why the scale is measured
#: within each mode rather than over the whole series.
CONTRAST_MAD_MULTIPLE = 4.0
#: Minimum samples in EACH mode for the contrast test to be evaluable. A flat
#: series has an empty high mode and abstains here.
MIN_MODE_SAMPLES = 3
#: Hysteresis half-band as a fraction of the series half-range. FREE
#: PARAMETER, chosen at implementation time on 2026-08-23 because the rule as
#: specified did not fix a band width. It is NOT derived from any measurement
#: and nothing in the evidence selects it; it is frozen only so the pass is
#: reproducible.
HYSTERESIS_FRACTION = 0.25
#: A crossing is accepted only if at least this many sampled train cameras
#: independently place BOTH boundaries within AGREEMENT_TOLERANCE_FRAMES.
MIN_AGREEING_CAMERAS = 3
AGREEMENT_TOLERANCE_FRAMES = 1
#: Screen-footprint dilation (pixels) applied to a group's projected extent.
FOOTPRINT_DILATE_PX = 4
#: Chunk size for the footprint dilation, to bound peak memory.
_DILATE_CHUNK = 64

#: Abstention reasons (the group stays ungated, family_id = -1).
ABSTAIN_CONTRAST = "contrast"
ABSTAIN_SHAPE = "no_interior_gap"
ABSTAIN_AGREEMENT = "camera_disagreement"
ABSTAIN_INADMISSIBLE = "inadmissible_interval"
ABSTAIN_NOT_DENSE = "boundary_not_densely_evaluated"
ABSTAIN_NO_FOOTPRINT = "empty_footprint"


class LeakageError(RuntimeError):
    """Raised when the estimation stage reaches for a forbidden input."""


# ---------------------------------------------------------------------------
# anti-leakage guard
# ---------------------------------------------------------------------------


def _normalise(path):
    return str(path).replace("\\", "/").lower()


def is_forbidden_path(path):
    """True for any path the ESTIMATION stage must never open."""
    low = _normalise(path)
    if "gt_identity" in low:
        return True
    if low.endswith("event_spec.json"):
        return True
    if re.search(r"(^|/)oracle_[^/]*\.json$", low):
        return True
    return False


class LeakageGuard(object):
    """Runtime enforcement of the anti-leakage contract, estimation only.

    Deliberately NOT passed into `estimate_episode_program`: it is installed
    around the call by `main`, so the estimation signature stays free of any
    object that could reach ground truth (see the module docstring).

    `sys.addaudithook` is used when available (Python 3.8+; it cannot be
    uninstalled, so it is gated on the active flag). The container that runs
    this pass may be Python 3.7, where audit hooks do not exist, so
    `builtins.open` and `os.open` are patched unconditionally as the portable
    path.
    """

    def __init__(self, scene=None, opt=None):
        self._scene = scene
        self._opt = opt
        self.active = False
        self.checks = {}
        self._saved_open = None
        self._saved_os_open = None
        self._saved_get_test = None
        self._audit_installed = False

    # -- static assertions (run before estimation) ------------------------

    def assert_manifests_empty(self):
        for key in ("event_candidate_manifest", "event_boundary_support_manifest"):
            value = str(getattr(self._opt, key, "") or "").strip()
            if value:
                raise LeakageError(
                    "anti-leakage: %s must be empty for an estimation run; got %r"
                    % (key, value)
                )
        self.checks["event_manifests_empty"] = True

    def assert_train_only(self, cameras, train_stack):
        """Every consumed camera must BE an element of the train stack.

        Identity, not equality: `Scene.getTrainCameras` copies the list but
        not the Camera objects (utils/data_utils.py:19), so an object drawn
        from the test split can never pass this.
        """
        allowed = set(id(cam) for cam in train_stack)
        for cam in cameras:
            if id(cam) not in allowed:
                raise LeakageError(
                    "anti-leakage: camera %r is not an element of the train split"
                    % (getattr(cam, "image_name", "<unnamed>"),)
                )
        self.checks["cameras_are_train_split_objects"] = True

    # -- context manager ---------------------------------------------------

    def __enter__(self):
        self.active = True
        self._saved_open = builtins.open
        self._saved_os_open = os.open
        guard = self

        def guarded_open(file, *args, **kwargs):
            if guard.active and is_forbidden_path(file):
                raise LeakageError(
                    "anti-leakage: the estimation stage tried to open %r" % (str(file),)
                )
            return guard._saved_open(file, *args, **kwargs)

        def guarded_os_open(path, *args, **kwargs):
            if guard.active and is_forbidden_path(path):
                raise LeakageError(
                    "anti-leakage: the estimation stage tried to open %r" % (str(path),)
                )
            return guard._saved_os_open(path, *args, **kwargs)

        builtins.open = guarded_open
        os.open = guarded_os_open
        self.checks["forbidden_path_open_guard"] = True

        if hasattr(sys, "addaudithook") and not self._audit_installed:
            def _hook(event, args):
                if not guard.active:
                    return
                if event in ("open", "os.open", "io.open") and args:
                    if is_forbidden_path(args[0]):
                        raise LeakageError(
                            "anti-leakage: audit hook blocked %r" % (str(args[0]),)
                        )
            sys.addaudithook(_hook)
            self._audit_installed = True
        self.checks["audit_hook_installed"] = bool(self._audit_installed)

        if self._scene is not None:
            self._saved_get_test = self._scene.getTestCameras

            def _refuse(*_args, **_kwargs):
                raise LeakageError(
                    "anti-leakage: getTestCameras() called during estimation"
                )

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
# decision rule (pure; no torch, no I/O -- hand-checkable in tests)
# ---------------------------------------------------------------------------


def median(values):
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    if n == 0:
        raise ContractError("median of an empty series")
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def mad(values):
    """Median absolute deviation about the median."""
    centre = median(values)
    return median([abs(float(v) - centre) for v in values])


def split_modes(values):
    """Partition a series at the midpoint of its range: (high, low, midpoint).

    `high` is strictly above the midpoint, `low` is at or below it, so the two
    are disjoint and exhaust the series.
    """
    midpoint = 0.5 * (max(values) + min(values))
    high = [float(v) for v in values if v > midpoint]
    low = [float(v) for v in values if v <= midpoint]
    return high, low, midpoint


def contrast_statistics(values):
    """(admit, detail) for the WITHIN-MODE contrast test.

    Frozen 2026-08-23, REPLACING an earlier rule that compared the series
    range against the MAD of the WHOLE series. That rule was defective in both
    directions and the replacement is not a tuning of it:

      * for a balanced series the median sits at the midpoint, so every
        absolute deviation equals half the range, MAD = range/2, and the rule
        demanded range >= 2*range -- unsatisfiable. It would have abstained on
        everything in a scene with a ~50% absence fraction;
      * it admitted the LRV3 33/27 split only because that imbalance puts the
        median ON the high plateau and drives the whole-series MAD toward 0 --
        at which point the test also passes pure noise, since range >= 0 holds
        trivially.

    MAD over a bimodal series is not a scale estimate; it is a measure of the
    split ratio. The amended rule measures the scale where the statistical
    intent lives -- WITHIN each mode -- which makes it invariant to the
    present/absent ratio:

        m = (max E + min E) / 2
        H = {E : E > m},  L = {E : E <= m}
        require |H| >= MIN_MODE_SAMPLES and |L| >= MIN_MODE_SAMPLES
        s = max(MAD(H), MAD(L))
        admit iff mean(H) - mean(L) >= CONTRAST_MAD_MULTIPLE * s

    A flat series has an empty H and abstains at the population requirement,
    so the old degenerate MAD == 0 case is excluded structurally rather than
    by a patched precondition. When both modes are populated the separation is
    strictly positive by construction (every element of H exceeds m and every
    element of L does not), so s == 0 means an exact step and is admitted.
    """
    detail = {
        "n_high": 0, "n_low": 0, "separation": None,
        "within_mode_scale": None, "midpoint": None,
    }
    if not values:
        return False, detail
    high, low, midpoint = split_modes(values)
    detail["n_high"] = len(high)
    detail["n_low"] = len(low)
    detail["midpoint"] = midpoint
    if len(high) < MIN_MODE_SAMPLES or len(low) < MIN_MODE_SAMPLES:
        return False, detail
    scale = max(mad(high), mad(low))
    separation = (sum(high) / len(high)) - (sum(low) / len(low))
    detail["separation"] = separation
    detail["within_mode_scale"] = scale
    if scale <= 0.0:
        return separation > 0.0, detail
    return separation >= CONTRAST_MAD_MULTIPLE * scale, detail


def contrast_ok(values):
    """Whether a series is presence-varying under the frozen contrast test."""
    return contrast_statistics(values)[0]


def hysteresis_band(values):
    """(low, high) thresholds about the midpoint of the series range."""
    lo = min(values)
    hi = max(values)
    midpoint = 0.5 * (hi + lo)
    half_band = HYSTERESIS_FRACTION * 0.5 * (hi - lo)
    return midpoint - half_band, midpoint + half_band


def hysteresis_transitions(frames, values):
    """Frames at which the series crosses out of one state into the other.

    A sample inside the band leaves the state unchanged -- that is the
    hysteresis. The reported frame is the FIRST frame on the new side, never
    an interpolated position.
    """
    if len(frames) != len(values):
        raise ContractError("frames and values must be the same length")
    low, high = hysteresis_band(values)
    transitions = []
    state = None
    for frame, value in zip(frames, values):
        if value >= high:
            new_state = "high"
        elif value <= low:
            new_state = "low"
        else:
            continue
        if state is None:
            state = new_state
            continue
        if new_state != state:
            transitions.append((state, new_state, int(frame)))
            state = new_state
    return transitions


def detect_gap(frames, values):
    """(offset_frame, onset_frame, reason).

    offset_frame is the first ABSENT frame; onset_frame is the first PRESENT
    frame after the gap. Returns (None, None, reason) when the series does not
    describe exactly one interior absence, which is the only shape the episode
    program can express (interior gaps only, elgs/trainer_hooks.py:640-641).
    """
    if not contrast_ok(values):
        return None, None, ABSTAIN_CONTRAST
    transitions = hysteresis_transitions(frames, values)
    if len(transitions) != 2:
        return None, None, ABSTAIN_SHAPE
    first, second = transitions
    if first[0] != "high" or first[1] != "low":
        return None, None, ABSTAIN_SHAPE
    if second[0] != "low" or second[1] != "high":
        return None, None, ABSTAIN_SHAPE
    return first[2], second[2], None


def boundary_is_dense(evaluated_frames, frame):
    """True iff the frame immediately before the crossing was also evaluated.

    This is what forbids interpolating a boundary across a coarse stride: a
    crossing reported at frame f is only trustworthy if f-1 was measured, so
    the transition is localized to a single frame interval.
    """
    return (int(frame) - 1) in set(int(f) for f in evaluated_frames)


def inset_gap_seconds(offset_frame, onset_frame, frame_dt, w):
    """Absence gap in model seconds, boundaries INSET by exactly w.

    The presence plateau of an episode is [b + w, d - w]
    (elgs/presence.py:11-12), so putting the interval endpoints one edge
    half-width OUTSIDE the last/first present frame makes the plateau cover
    every truly-present frame at presence 1.0 and forces the whole smoothstep
    ramp to fall inside the absence gap. This is the inset convention of
    oracle_correct.json, rederived here from the substrate semantics rather
    than copied from it.

        gap_start = t(last present frame)  + w = (offset_frame - 1)*dt + w
        gap_end   = t(first present frame) - w =  onset_frame     *dt - w
    """
    gap_start = (int(offset_frame) - 1) * float(frame_dt) + float(w)
    gap_end = int(onset_frame) * float(frame_dt) - float(w)
    return gap_start, gap_end


def build_gap_interval(gap_start, gap_end, config):
    """The K=2 IntervalState for one interior gap, or ContractError.

    `inverse` performs the floor and admissibility checks, so an inexpressible
    gap fails closed here. Callers ABSTAIN on failure; they never repair.
    """
    lo, hi = -config.w_m, config.T + config.w_m
    edges = [lo, float(gap_start), float(gap_end), hi]
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ContractError("gap boundaries are not strictly increasing: %s" % (edges,))
    lens = [edges[1] - edges[0], edges[3] - edges[2]]
    gaps = [edges[2] - edges[1]]
    return inverse(2, True, True, 0.0, lens, gaps, 0.0, config, dtype=torch.float32)


def marker_interval(config):
    """A neutral admissible K=2 program used ONLY to mark a family gated.

    `gated_row_mask` reads the REGISTRY's K (elgs/runtime.py:225-228), so an
    ablation family must carry K >= 2 to be gated at all. Its boundaries are
    never rendered: every render overrides the family, with the K=0 empty
    program for the ablated pass. The gap is placed at the centre of the
    window and the two episodes split the remainder.
    """
    gap_len = 1.5 * config.floor_gap
    span = (config.omega - gap_len) / 2.0
    if span <= config.floor_len:
        raise ContractError(
            "scene window too short to build a K=2 ablation marker: "
            "episode span %.6f <= floor_len %.6f" % (span, config.floor_len)
        )
    return inverse(2, True, True, 0.0, [span, span], [gap_len], 0.0, config,
                   dtype=torch.float32)


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------


def assert_min_group_rows_single_sourced():
    """Fail closed if the mirrored row floor has drifted from packet birth.

    Imports `scene`, so it only runs where the CUDA extension exists; `main()`
    calls it before any measurement.
    """
    from scene.packet_birth import MIN_PACKET_ROWS

    if int(MIN_GROUP_ROWS) != int(MIN_PACKET_ROWS):
        raise ContractError(
            "MIN_GROUP_ROWS=%d has drifted from scene.packet_birth."
            "MIN_PACKET_ROWS=%d" % (MIN_GROUP_ROWS, MIN_PACKET_ROWS)
        )
    return True


def voxel_grid(xyz, cells_per_axis=VOXEL_CELLS_PER_AXIS):
    """(lo, span, keys) for the ABSOLUTE grid over the cloud's bounding box.

    Factored out so the emitted v2 program can carry `lo`/`span` and reapply
    the very same world-space grid to a DIFFERENT cloud -- which is what a
    fresh training run needs, since it seeds a `create_from_pcd` cloud rather
    than the trained one the estimate was computed on.
    """
    cells = int(cells_per_axis)
    if cells < 1:
        raise ContractError("cells_per_axis must be >= 1")
    points = xyz.detach()
    lo = points.min(dim=0).values
    span = (points.max(dim=0).values - lo).clamp_min(1e-6)
    voxel = ((points - lo) / span * cells).clamp(0, cells - 1).long()
    keys = voxel[:, 0] * cells * cells + voxel[:, 1] * cells + voxel[:, 2]
    return lo, span, keys


def build_voxel_groups(xyz, cells_per_axis=VOXEL_CELLS_PER_AXIS,
                       min_rows=MIN_GROUP_ROWS):
    """Per-row group labels over the cloud's own bounding box; -1 = substrate.

    Identical construction to `seed_families` (elgs/trainer_hooks.py:686-693),
    with sub-threshold cells demoted to -1. Deterministic for a fixed cloud.
    """
    points = xyz.detach()
    count = int(points.shape[0])
    labels = torch.full((count,), -1, dtype=torch.long, device=points.device)
    if count == 0:
        return labels, 0
    _, _, keys = voxel_grid(points, cells_per_axis)
    unique_keys, inverse_map = torch.unique(keys, sorted=True, return_inverse=True)
    counts = torch.bincount(inverse_map, minlength=int(unique_keys.numel()))
    keep = counts >= int(min_rows)
    remap = torch.full((int(unique_keys.numel()),), -1, dtype=torch.long,
                       device=points.device)
    kept = int(keep.sum())
    if kept:
        remap[keep] = torch.arange(kept, dtype=torch.long, device=points.device)
    labels = remap[inverse_map]
    return labels, kept


# ---------------------------------------------------------------------------
# ablation runtime
# ---------------------------------------------------------------------------


class AblationRuntime(object):
    """One gated family per candidate group, driven through the total gate.

    `attach(group)` installs the per-row column in which ONLY that group's
    rows carry a family id; every other row stays at -1 and therefore keeps
    the ordinary temporal marginal and an unmodulated static twin
    (elgs/presence.py:101-102). `detach()` restores the pure substrate.
    """

    #: The K=0 empty program: presence is exactly zero (elgs/runtime.py:182-184).
    EMPTY = IntervalState(K=0, latch_pre=None, latch_post=None, a=None)

    def __init__(self, gaussians, labels, n_groups, config, schedule):
        self._gaussians = gaussians
        self._labels = labels.to("cpu")
        self._n_groups = int(n_groups)
        self.config = config
        registry = FamilyRegistry()
        marker = marker_interval(config)
        self._family_of_group = {}
        for group in range(self._n_groups):
            record = registry.create_family(
                birth_time=0.0,
                birth_site=(0.0, 0.0, 0.0),
                lineage_key="ablation-group-%d" % group,
                interval=marker,
                tau=(0.0, 0.0),
            )
            self._family_of_group[group] = record.family_id
        device = gaussians._xyz.device if gaussians._xyz.numel() else "cpu"
        self._runtime = ElgsRuntime(registry, config, schedule, device=device,
                                    dtype=torch.float32)
        # Row indices per group, resolved once. Caching a full-length column
        # PER GROUP would cost n_groups x n_rows longs (about 600 MB at 512
        # groups and 150k rows); one shared buffer plus an index list is
        # ~1/512 of that, and makes attach O(|G|) rather than O(N).
        self._rows = {}
        for group in range(self._n_groups):
            self._rows[group] = (self._labels == group).nonzero(as_tuple=True)[0]
        self._column = torch.full_like(self._labels, -1)
        self._attached = None

    def rows_of(self, group):
        return self._rows[group]

    def attach(self, group):
        """Gate `group` off entirely; everything else stays substrate."""
        if self._attached is not None:
            self._column[self._rows[self._attached]] = -1
        self._column[self._rows[group]] = self._family_of_group[group]
        self._attached = group
        gaussians = self._gaussians
        gaussians._elgs_family_ids = self._column
        gaussians._elgs_local_presence = True
        gaussians.elgs_runtime = self._runtime
        gaussians._elgs_presence_override = {
            self._family_of_group[group]: self.EMPTY
        }

    def detach(self):
        """Pure substrate: no runtime, so the renderer uses `get_marginal_t`."""
        if self._attached is not None:
            self._column[self._rows[self._attached]] = -1
            self._attached = None
        gaussians = self._gaussians
        gaussians.elgs_runtime = None
        gaussians._elgs_local_presence = False
        gaussians._elgs_presence_override = None


# ---------------------------------------------------------------------------
# footprints and measurement
# ---------------------------------------------------------------------------


def _dilate(mask, pixels):
    """Boolean dilation by a square structuring element, chunked over groups."""
    if pixels <= 0:
        return mask
    kernel = 2 * int(pixels) + 1
    out = torch.zeros_like(mask)
    for start in range(0, mask.shape[0], _DILATE_CHUNK):
        chunk = mask[start:start + _DILATE_CHUNK].float().unsqueeze(1)
        pooled = F.max_pool2d(chunk, kernel_size=kernel, stride=1,
                              padding=kernel // 2)
        out[start:start + _DILATE_CHUNK] = pooled.squeeze(1) > 0.5
    return out


def build_footprints(gaussians, labels, n_groups, views, dataset, height, width,
                     dilate=FOOTPRINT_DILATE_PX):
    """(n_groups, H, W) bool per camera: where a group could ever contribute.

    The union over EVERY frame, so the mask is fixed in time and the per-frame
    series is comparable across frames. Pure projection -- no render. All rows
    are projected once per (camera, frame) and scattered into the per-group
    accumulator, so the cost is one projection per view, not one per group.
    """
    from utils.motion_prior_utils import project_points_to_screen

    device = gaussians._xyz.device
    labels_dev = labels.to(device)
    flat_pixels = height * width
    footprints = {}
    for camera_id in sorted(views.keys()):
        acc = torch.zeros((n_groups, flat_pixels), dtype=torch.bool, device=device)
        for frame in sorted(views[camera_id].keys()):
            # `viewpoint_stack` directly, NOT dataset[i]: indexing the dataset
            # decodes the training image (utils/data_utils.py:20-32) and this
            # pass needs geometry only.
            camera = dataset.viewpoint_stack[views[camera_id][frame]]
            points = gaussians.get_dynamic_xyz(float(camera.timestamp)).detach()
            xy, valid = project_points_to_screen(points, camera)
            valid = valid.reshape(-1)
            xs = xy[:, 0].round().long().clamp(0, width - 1)
            ys = xy[:, 1].round().long().clamp(0, height - 1)
            flat = ys * width + xs
            keep = valid & (labels_dev >= 0)
            if not bool(keep.any()):
                continue
            linear = labels_dev[keep] * flat_pixels + flat[keep]
            acc.view(-1)[linear] = True
        mask = acc.view(n_groups, height, width)
        footprints[camera_id] = _dilate(mask, dilate)
    return footprints


def _render_error(camera, gt_image, gaussians, pipe, background):
    """Per-pixel mean-absolute error of the current model against the GT view."""
    from gaussian_renderer import render  # lazy: keeps the module CPU-importable

    out = render(camera, gaussians, pipe, background)
    return (out["render"] - gt_image).abs().mean(dim=0)


def measure_series(gaussians, ablation, footprints, views, dataset, frames,
                   pipe, background, counters, verbose=False):
    """{(group, camera_id, frame): E} over the requested frames.

    The base render is shared across groups at a given view -- rendered once,
    then every group's ablated render is differenced against it.
    """
    values = {}
    n_groups = ablation._n_groups
    for camera_id in sorted(views.keys()):
        footprint = footprints[camera_id]
        active = [g for g in range(n_groups) if bool(footprint[g].any())]
        for frame in frames:
            if frame not in views[camera_id]:
                continue
            gt_image, camera = dataset[views[camera_id][frame]]
            camera = camera.cuda() if torch.cuda.is_available() else camera
            gt_image = gt_image.to(gaussians._xyz.device)
            ablation.detach()
            base_error = _render_error(camera, gt_image, gaussians, pipe, background)
            counters["base_renders"] += 1
            for group in active:
                ablation.attach(group)
                ablated_error = _render_error(camera, gt_image, gaussians, pipe,
                                              background)
                counters["ablated_renders"] += 1
                delta = ablated_error - base_error
                mask = footprint[group]
                values[(group, camera_id, int(frame))] = float(delta[mask].mean())
            ablation.detach()
        if verbose:
            print("  camera %d: %d groups with a footprint" % (camera_id, len(active)))
    return values


# ---------------------------------------------------------------------------
# STAGE 1 -- ESTIMATION. No ground-truth object is reachable from here.
# ---------------------------------------------------------------------------


def estimate_episode_program(
    gaussians,
    dataset,
    views,
    frame_dt,
    interval_config,
    schedule,
    pipe,
    background,
    height,
    width,
    coarse_stride,
    n_frames,
    verbose=False,
):
    """Estimate one interior absence gap per candidate group.

    Every argument is either the trained model, the TRAIN camera dataset, the
    train view index, or a numeric/render parameter. There is deliberately no
    `scene`, no `source_path`, no event spec and no identity buffer: the
    scoring inputs are unreachable from this call. See the module docstring;
    tests/test_estimate_episodes.py asserts this signature.
    """
    labels, n_groups = build_voxel_groups(gaussians._xyz)
    if n_groups == 0:
        raise ContractError("voxel grouping produced no group above the row floor")
    if verbose:
        print("grouping: %d groups, %d rows gated, %d rows substrate"
              % (n_groups, int((labels >= 0).sum()), int((labels < 0).sum())))

    ablation = AblationRuntime(gaussians, labels, n_groups, interval_config, schedule)
    footprints = build_footprints(gaussians, labels, n_groups, views, dataset,
                                  height, width)
    counters = {"base_renders": 0, "ablated_renders": 0}

    coarse_frames = list(range(0, n_frames, int(coarse_stride)))
    if (n_frames - 1) not in coarse_frames:
        coarse_frames.append(n_frames - 1)

    # Up-front cost signal: the coarse stage dominates, and an operator who
    # sees an unaffordable projection here can abort before any render.
    per_view_groups = sum(int(footprints[cam].any(dim=2).any(dim=1).sum())
                          for cam in sorted(views.keys()))
    print("projected coarse renders: %d base + %d ablated (%d group-views "
          "with a footprint across %d cameras, %d coarse frames)"
          % (len(views) * len(coarse_frames),
             per_view_groups * len(coarse_frames),
             per_view_groups, len(views), len(coarse_frames)))
    if verbose:
        print("coarse stage: %d frames (stride %d)" % (len(coarse_frames), coarse_stride))
    values = measure_series(gaussians, ablation, footprints, views, dataset,
                            coarse_frames, pipe, background, counters, verbose)
    evaluated = set(coarse_frames)

    # Fine stage: every frame in the neighbourhood of a coarse-detected
    # transition. Never interpolated -- frame accuracy is the whole point.
    camera_ids = sorted(views.keys())
    fine_frames = set()
    for group in range(n_groups):
        frames, series = _pooled_series(values, group, camera_ids, sorted(evaluated))
        if len(series) < 3:
            continue
        offset, onset, reason = detect_gap(frames, series)
        if reason is not None:
            continue
        for boundary in (offset, onset):
            low = max(0, boundary - int(coarse_stride))
            high = min(n_frames - 1, boundary + int(coarse_stride))
            for frame in range(low, high + 1):
                if frame not in evaluated:
                    fine_frames.add(frame)
    if fine_frames:
        if verbose:
            print("fine stage: %d additional frames" % len(fine_frames))
        values.update(measure_series(gaussians, ablation, footprints, views,
                                     dataset, sorted(fine_frames), pipe,
                                     background, counters, verbose))
        evaluated |= fine_frames
    ablation.detach()

    evaluated_frames = sorted(evaluated)
    decisions = []
    for group in range(n_groups):
        decisions.append(_decide_group(
            group, values, camera_ids, evaluated_frames, ablation, footprints,
            frame_dt, interval_config,
        ))
    return {
        "labels": labels,
        "n_groups": n_groups,
        "decisions": decisions,
        "evaluated_frames": evaluated_frames,
        "coarse_frames": sorted(coarse_frames),
        "fine_frames": sorted(fine_frames),
        "render_counts": dict(counters),
        "series": values,
    }


def _pooled_series(values, group, camera_ids, frames):
    """(frames, mean-over-cameras) for one group, over frames that were measured."""
    out_frames = []
    out_values = []
    for frame in frames:
        samples = [values[(group, cam, frame)] for cam in camera_ids
                   if (group, cam, frame) in values]
        if not samples:
            continue
        out_frames.append(int(frame))
        out_values.append(sum(samples) / float(len(samples)))
    return out_frames, out_values


def _decide_group(group, values, camera_ids, evaluated_frames, ablation,
                  footprints, frame_dt, interval_config):
    """Apply the frozen decision rule to one group."""
    rows = ablation.rows_of(group)
    record = {
        "group": int(group),
        "rows": int(rows.numel()),
        "gated": False,
        "offset_frame": None,
        "onset_frame": None,
        "gap_seconds": None,
        "abstain_reason": None,
        "agreeing_cameras": 0,
        "per_camera": {},
    }
    cameras_with_footprint = [cam for cam in camera_ids
                              if bool(footprints[cam][group].any())]
    if not cameras_with_footprint:
        record["abstain_reason"] = ABSTAIN_NO_FOOTPRINT
        return record

    frames, series = _pooled_series(values, group, camera_ids, evaluated_frames)
    if len(series) < 3:
        record["abstain_reason"] = ABSTAIN_CONTRAST
        return record
    record["contrast"] = contrast_statistics(series)[1]
    record["contrast"]["range"] = max(series) - min(series)

    offset, onset, reason = detect_gap(frames, series)
    if reason is not None:
        record["abstain_reason"] = reason
        return record
    record["offset_frame"] = int(offset)
    record["onset_frame"] = int(onset)

    if not (boundary_is_dense(evaluated_frames, offset)
            and boundary_is_dense(evaluated_frames, onset)):
        record["abstain_reason"] = ABSTAIN_NOT_DENSE
        return record

    agreeing = 0
    for cam in cameras_with_footprint:
        cam_frames = []
        cam_values = []
        for frame in evaluated_frames:
            key = (group, cam, frame)
            if key in values:
                cam_frames.append(int(frame))
                cam_values.append(values[key])
        if len(cam_values) < 3:
            continue
        cam_offset, cam_onset, cam_reason = detect_gap(cam_frames, cam_values)
        record["per_camera"][str(cam)] = {
            "offset_frame": cam_offset,
            "onset_frame": cam_onset,
            "reason": cam_reason,
        }
        if cam_reason is not None:
            continue
        if (abs(cam_offset - offset) <= AGREEMENT_TOLERANCE_FRAMES
                and abs(cam_onset - onset) <= AGREEMENT_TOLERANCE_FRAMES):
            agreeing += 1
    record["agreeing_cameras"] = agreeing
    if agreeing < MIN_AGREEING_CAMERAS:
        record["abstain_reason"] = ABSTAIN_AGREEMENT
        return record

    gap_start, gap_end = inset_gap_seconds(offset, onset, frame_dt,
                                           interval_config.w)
    try:
        build_gap_interval(gap_start, gap_end, interval_config)
    except ContractError as error:
        record["abstain_reason"] = ABSTAIN_INADMISSIBLE
        record["inadmissible_detail"] = str(error)
        return record

    record["gated"] = True
    record["gap_seconds"] = [gap_start, gap_end]
    return record


def freeze_program(estimate, frame_dt, interval_config):
    """The frozen, hashable estimated program. Nothing may change after this."""
    groups = []
    for record in estimate["decisions"]:
        groups.append({
            "group": record["group"],
            "rows": record["rows"],
            "gated": record["gated"],
            "offset_frame": record["offset_frame"],
            "onset_frame": record["onset_frame"],
            "gap_seconds": record["gap_seconds"],
            "abstain_reason": record["abstain_reason"],
            "agreeing_cameras": record["agreeing_cameras"],
        })
    program = {
        "schema": REPORT_SCHEMA + "/program",
        "units": "model_time_seconds",
        "frame_dt": float(frame_dt),
        "presence_edge_half_width_w": float(interval_config.w),
        "floor_len": float(interval_config.floor_len),
        "floor_gap": float(interval_config.floor_gap),
        "n_groups": estimate["n_groups"],
        "groups": groups,
    }
    blob = json.dumps(program, sort_keys=True).encode("utf-8")
    return program, hashlib.sha256(blob).hexdigest()


def build_v2_program(decisions, labels, xyz, frame_dt, interval_config,
                     membership_mode, cells_per_axis=VOXEL_CELLS_PER_AXIS,
                     source=None):
    """The STANDALONE trainable artifact: schema `adags-episode-program-v2`.

    Derived only from the frozen decisions and the cloud, so it is emitted
    BEFORE scoring and carries nothing from the ground truth.

    Membership is written BOTH ways and `membership_mode` declares which one
    binds at seeding:

      row_ids       an explicit per-row column, exact but bound to this cloud
                    by a row count and an xyz fingerprint. `seed_families`
                    fails closed if the seeding cloud differs -- which it
                    always does for a fresh `create_from_pcd` run.
      spatial_voxel the absolute world-space grid plus each gated group's cell
                    keys, which reapplies to any cloud.

    The spatial block is emitted in both modes so one artifact can be
    re-declared for either run protocol without recomputing the estimate.
    """
    from elgs.trainer_hooks import EPISODE_PROGRAM_SCHEMA_V2, cloud_fingerprint

    if membership_mode not in ("row_ids", "spatial_voxel"):
        raise ContractError("unknown membership_mode %r" % (membership_mode,))
    gated = [d for d in decisions if d.get("gated")]
    if not gated:
        raise ContractError(
            "no group was gated: there is no program to emit. Phase T2 needs a "
            "T1 pass that admitted at least one group."
        )
    gated_ids = sorted(int(d["group"]) for d in gated)
    labels_cpu = labels.to("cpu")
    wanted = torch.tensor(gated_ids, dtype=labels_cpu.dtype)
    row_ids = torch.where(torch.isin(labels_cpu, wanted), labels_cpu,
                          torch.full_like(labels_cpu, -1))
    lo, span, keys = voxel_grid(xyz, cells_per_axis)
    keys_cpu = keys.to("cpu")
    group_cell_keys = {}
    for group_id in gated_ids:
        member = labels_cpu == group_id
        group_cell_keys[str(group_id)] = sorted(
            set(int(k) for k in keys_cpu[member].tolist()))
    program = {
        "schema_version": EPISODE_PROGRAM_SCHEMA_V2,
        "units": "model_time_seconds",
        "membership_mode": membership_mode,
        "frame_dt": float(frame_dt),
        "presence_edge_half_width_w": float(interval_config.w),
        "cloud": {
            "n_rows": int(xyz.shape[0]),
            "xyz_sha256": cloud_fingerprint(xyz),
        },
        "spatial": {
            "kind": "voxel_grid",
            "cells_per_axis": int(cells_per_axis),
            "lo": [float(v) for v in lo.detach().cpu().tolist()],
            "span": [float(v) for v in span.detach().cpu().tolist()],
            "group_cell_keys": group_cell_keys,
        },
        "groups": [
            {
                "group": int(d["group"]),
                "gaps": [[float(d["gap_seconds"][0]), float(d["gap_seconds"][1])]],
                "offset_frame": int(d["offset_frame"]),
                "onset_frame": int(d["onset_frame"]),
                "rows_at_estimation": int(d["rows"]),
            }
            for d in sorted(gated, key=lambda r: int(r["group"]))
        ],
        "source": dict(source or {}),
    }
    if membership_mode == "row_ids":
        program["row_group_ids"] = [int(v) for v in row_ids.tolist()]
    blob = json.dumps(program, sort_keys=True).encode("utf-8")
    return program, hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# STAGE 2 -- SCORING. Ground truth is opened here and ONLY here, against an
# already-frozen program.
# ---------------------------------------------------------------------------


def score_program(program, labels, xyz, source_path, train_camera_ids):
    """Score the frozen estimate against the authored ground truth."""
    import numpy as np

    root = Path(source_path)
    spec = json.loads((root / "event_spec.json").read_text())
    presence = spec["presence_frames"]
    gap = presence["gap"]
    authored_offset = int(gap[0])
    authored_onset = int(gap[1]) + 1
    event = spec["event_object"]
    centre = torch.tensor(event["centre"], dtype=xyz.dtype, device=xyz.device)
    radius = float(event["radius"])
    inside = (xyz - centre).norm(dim=1) <= radius

    test_camera_ids = sorted(int(c) for c in spec.get("test_cameras", []))
    overlap = sorted(set(train_camera_ids) & set(test_camera_ids))
    if overlap:
        raise ContractError(
            "anti-leakage: the estimator consumed held-out cameras %s" % (overlap,)
        )

    # Identity buffers, evaluation only: recover the absence gap empirically
    # and cross-check it against the authored one.
    identity_gap = None
    gt_dir = root / "gt_identity"
    if gt_dir.is_dir():
        object_id = int(event["id"])
        absent = []
        for frame in range(int(spec.get("n_frames", 0)) or 0):
            files = sorted(gt_dir.glob("cam*_f%03d.npy" % frame))
            if not files:
                continue
            pixels = 0
            for path in files:
                pixels += int((np.load(str(path)) == object_id).sum())
            if pixels == 0:
                absent.append(frame)
        if absent:
            identity_gap = [min(absent), max(absent)]

    n_groups = int(program["n_groups"])
    labels_cpu = labels.to("cpu")
    rows_in_sphere = {}
    for group in range(n_groups):
        selector = (labels_cpu == group).to(inside.device)
        rows_in_sphere[group] = int(inside[selector].sum())

    overlapping = [g for g in range(n_groups) if rows_in_sphere[g] > 0]
    gated = [r for r in program["groups"] if r["gated"]]
    abstained = [r for r in program["groups"] if not r["gated"]]

    per_group = []
    offset_errors = []
    onset_errors = []
    false_activations = []
    for record in gated:
        group = record["group"]
        offset_error = int(record["offset_frame"]) - authored_offset
        onset_error = int(record["onset_frame"]) - authored_onset
        overlaps = rows_in_sphere[group] > 0
        if not overlaps:
            false_activations.append(group)
        else:
            offset_errors.append(offset_error)
            onset_errors.append(onset_error)
        per_group.append({
            "group": group,
            "rows": record["rows"],
            "rows_in_event_sphere": rows_in_sphere[group],
            "overlaps_event_object": overlaps,
            "offset_frame": record["offset_frame"],
            "onset_frame": record["onset_frame"],
            "offset_error_frames": offset_error,
            "onset_error_frames": onset_error,
        })

    reasons = {}
    for record in abstained:
        key = record["abstain_reason"] or "unknown"
        reasons[key] = reasons.get(key, 0) + 1

    return {
        "authored": {
            "offset_frame": authored_offset,
            "onset_frame": authored_onset,
            "gap_frames": [int(gap[0]), int(gap[1])],
            "identity_buffer_gap_frames": identity_gap,
            "identity_cross_check_agrees": (
                identity_gap == [int(gap[0]), int(gap[1])]
                if identity_gap is not None else None
            ),
            "event_object_centre": event["centre"],
            "event_object_radius": radius,
            "test_cameras": test_camera_ids,
        },
        "n_groups": n_groups,
        "opportunity_denominator": len(overlapping),
        "groups_overlapping_event_object": len(overlapping),
        "n_gated": len(gated),
        "n_abstained": len(abstained),
        "abstention_rate": (len(abstained) / float(n_groups)) if n_groups else None,
        "abstention_reasons": reasons,
        "false_activations": len(false_activations),
        "false_activation_groups": false_activations,
        "offset_error_frames": _error_summary(offset_errors),
        "onset_error_frames": _error_summary(onset_errors),
        "per_group": per_group,
    }


def _error_summary(errors):
    if not errors:
        return {"count": 0, "exact": 0, "within_1": 0, "median": None,
                "max_abs": None, "histogram": {}}
    histogram = {}
    for error in errors:
        histogram[str(error)] = histogram.get(str(error), 0) + 1
    return {
        "count": len(errors),
        "exact": sum(1 for e in errors if e == 0),
        "within_1": sum(1 for e in errors if abs(e) <= 1),
        "median": median(errors),
        "max_abs": max(abs(e) for e in errors),
        "histogram": histogram,
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _merge_config(args, config_path):
    """YAML-over-argparse merge, identical to scripts/eval_lrv1_event.py."""
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


def camera_id_of(camera):
    match = _CAMERA_ID.search(str(getattr(camera, "image_name", "")))
    if match is None:
        raise ContractError(
            "cannot parse a camera id from image_name %r"
            % (getattr(camera, "image_name", None),)
        )
    return int(match.group(1))


def build_view_index(dataset, frame_dt, camera_ids=None):
    """{camera_id: {frame_index: dataset_index}} plus the camera objects used."""
    views = {}
    cameras = []
    for index in range(len(dataset)):
        camera = dataset.viewpoint_stack[index]
        cam_id = camera_id_of(camera)
        if camera_ids is not None and cam_id not in camera_ids:
            continue
        frame = int(round(float(camera.timestamp) / float(frame_dt)))
        views.setdefault(cam_id, {})[frame] = index
        cameras.append(camera)
    return views, cameras


def select_cameras(all_ids, count):
    """Deterministic even spread over the sorted train camera ids."""
    ordered = sorted(all_ids)
    if count >= len(ordered):
        return ordered
    step = len(ordered) / float(count)
    return [ordered[int(i * step)] for i in range(count)]


def main(argv=None):
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from elgs.trainer_hooks import (
        build_interval_config,
        infer_frame_dt,
        load_structural_prereg,
    )

    parser = ArgumentParser(description="Phase T1 episode-boundary estimation")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", required=True)
    parser.add_argument("--start_checkpoint", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument(
        "--emit_program", default="",
        help="write the standalone trainable v2 program to this path")
    parser.add_argument(
        "--membership_mode", choices=("row_ids", "spatial_voxel"),
        default="row_ids",
        help=("which membership binds at seeding. 'row_ids' is exact but only "
              "valid when the training run seeds THIS checkpoint's cloud; a "
              "fresh create_from_pcd run needs 'spatial_voxel'"))
    parser.add_argument("--cameras", type=int, default=4,
                        help="how many train cameras to sample (>= %d)"
                             % MIN_AGREEING_CAMERAS)
    parser.add_argument("--coarse_stride", type=int, default=4)
    parser.add_argument(
        "--skip-scoring", dest="skip_scoring", action="store_true",
        help=("omit the ground-truth scoring stage. It builds a SPHERE "
              "membership test from event_spec.json's event_object "
              "centre/radius, which a non-spherical fixture deliberately "
              "omits; such a fixture is scored by a separate tool"))
    # NOTE: --elgs_prereg_dir is NOT declared here. It is already an
    # OptimizationParams field (arguments/__init__.py:257), so declaring it
    # again is an argparse conflict; it is read off `opt` below.
    # Scene-shape arguments, matching scripts/consolidate_packets.py:333-341
    # and scripts/eval_lrv1_event.py:135-143. `_merge_config` asserts that
    # every YAML key already exists on `args`, so omitting these makes any
    # real config fail at merge time.
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
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    _merge_config(args, args.config)

    if int(args.cameras) < MIN_AGREEING_CAMERAS:
        raise ContractError(
            "--cameras must be at least %d: the frozen agreement rule needs "
            "%d independent cameras" % (MIN_AGREEING_CAMERAS, MIN_AGREEING_CAMERAS)
        )

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

    from scene import Scene  # lazy, mirrors the render import
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

    assert_min_group_rows_single_sourced()

    if int(gaussians.gaussian_dim) != 4:
        raise ContractError("the total gate requires a 4D model; got dim %d"
                            % gaussians.gaussian_dim)
    if not bool(getattr(gaussians, "enable_soft_routing", False)):
        raise ContractError(
            "the total gate needs soft routing: without it the static twin is a "
            "separate hard-static set and ablation would leak through it "
            "(gaussian_renderer/__init__.py:349-356)"
        )

    train_dataset = scene.getTrainCameras()
    train_stack = scene.train_cameras[1.0]
    timestamps = [float(c.timestamp) for c in train_stack]
    frame_dt = infer_frame_dt(timestamps)
    time_span = max(timestamps) - min(timestamps)
    prereg = load_structural_prereg(str(opt.elgs_prereg_dir))
    interval_config = build_interval_config(prereg, time_span, frame_dt)
    sched = prereg["schedule"]["full"]
    schedule = ScheduleAnchors(
        seed_iteration=int(sched["seed_iteration"]),
        audit_iteration=int(sched["audit_iteration"]),
        round_iterations=tuple(int(v) for v in sched["round_iterations"]),
        refit_until=int(sched["refit_until"]),
    )
    n_frames = int(round(time_span / frame_dt)) + 1

    all_views, _ = build_view_index(train_dataset, frame_dt)
    chosen = select_cameras(all_views.keys(), int(args.cameras))
    views, used_cameras = build_view_index(train_dataset, frame_dt, set(chosen))

    background = torch.tensor(
        [1, 1, 1] if dataset_params.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu",
    )
    sample_camera = train_stack[0]
    height = int(sample_camera.image_height)
    width = int(sample_camera.image_width)

    guard = LeakageGuard(scene=scene, opt=opt)
    guard.assert_manifests_empty()
    guard.assert_train_only(used_cameras, train_stack)

    print("groups grid %d^3 | cameras %s | frames %d | coarse stride %d"
          % (VOXEL_CELLS_PER_AXIS, chosen, n_frames, args.coarse_stride))

    started = time.perf_counter()
    with guard:
        estimate = estimate_episode_program(
            gaussians=gaussians,
            dataset=train_dataset,
            views=views,
            frame_dt=frame_dt,
            interval_config=interval_config,
            schedule=schedule,
            pipe=pipe,
            background=background,
            height=height,
            width=width,
            coarse_stride=int(args.coarse_stride),
            n_frames=n_frames,
            verbose=bool(args.verbose),
        )
    elapsed = time.perf_counter() - started

    # FREEZE before any ground truth is opened.
    program, program_hash = freeze_program(estimate, frame_dt, interval_config)
    print("estimate frozen: sha256 %s" % program_hash)

    # The trainable artifact is derived from the FROZEN decisions and the
    # cloud only, and is written here -- still before scoring -- so it
    # provably carries nothing from the ground truth.
    v2_program = None
    v2_hash = None
    if args.emit_program:
        Path(args.emit_program).parent.mkdir(parents=True, exist_ok=True)
        v2_program, v2_hash = build_v2_program(
            estimate["decisions"], estimate["labels"], gaussians._xyz.detach(),
            frame_dt, interval_config, args.membership_mode,
            source={
                "checkpoint": str(args.start_checkpoint),
                "config": str(args.config),
                "estimate_sha256": program_hash,
            },
        )
        with open(args.emit_program, "w", encoding="utf-8") as handle:
            json.dump(v2_program, handle, indent=1, sort_keys=True)
        print("v2 program written: %s (membership_mode %s, sha256 %s)"
              % (args.emit_program, args.membership_mode, v2_hash))

    if args.skip_scoring:
        scoring = {"skipped": True, "reason": (
            "--skip-scoring was passed: the scoring stage reads "
            "event_object.centre/radius from the fixture's spec and builds a "
            "sphere membership test from them, which a non-spherical fixture "
            "deliberately omits. Membership must be scored separately.")}
    else:
        scoring = score_program(program, estimate["labels"], gaussians._xyz.detach(),
                                dataset_params.source_path, chosen)

    report = {
        "schema": REPORT_SCHEMA,
        "checkpoint": str(args.start_checkpoint),
        "config": str(args.config),
        "source_path": str(dataset_params.source_path),
        "program": program,
        "program_sha256": program_hash,
        "v2_program": {
            "path": str(args.emit_program) or None,
            "membership_mode": args.membership_mode if args.emit_program else None,
            "sha256": v2_hash,
        },
        "grouping": {
            "method": "voxel_grid_over_cloud_bounding_box",
            "cells_per_axis": VOXEL_CELLS_PER_AXIS,
            "min_group_rows": MIN_GROUP_ROWS,
            "n_groups": estimate["n_groups"],
            "n_rows": int(gaussians._xyz.shape[0]),
        },
        "sampling": {
            "train_camera_ids_available": sorted(all_views.keys()),
            "train_camera_ids_used": chosen,
            "n_frames": n_frames,
            "coarse_stride": int(args.coarse_stride),
            "coarse_frames": estimate["coarse_frames"],
            "fine_frames": estimate["fine_frames"],
            "evaluated_frames": estimate["evaluated_frames"],
        },
        "decision_rule": {
            "contrast_rule": "within_mode_separation_over_within_mode_mad",
            "contrast_mad_multiple": CONTRAST_MAD_MULTIPLE,
            "min_mode_samples": MIN_MODE_SAMPLES,
            "hysteresis_fraction": HYSTERESIS_FRACTION,
            "min_agreeing_cameras": MIN_AGREEING_CAMERAS,
            "agreement_tolerance_frames": AGREEMENT_TOLERANCE_FRAMES,
            "footprint_dilate_px": FOOTPRINT_DILATE_PX,
        },
        "render_counts": estimate["render_counts"],
        # Per-group diagnostics, OUTSIDE the hashed program: contrast
        # statistics and the per-camera crossings behind each decision.
        "diagnostics": {"decisions": estimate["decisions"]},
        "timing": {
            "estimation_seconds": elapsed,
            "seconds_per_render": elapsed / float(max(
                1, estimate["render_counts"]["base_renders"]
                + estimate["render_counts"]["ablated_renders"])),
        },
        "anti_leakage": {
            "checks_passed": guard.checks,
            "estimation_saw_ground_truth": False,
            "program_frozen_before_scoring": True,
        },
        "scoring": scoring,
    }
    with open(args.out_report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
    _print_table(report)
    return report


def _print_table(report):
    scoring = report["scoring"]
    counts = report["render_counts"]
    if scoring.get("skipped"):
        print("")
        print("=== phase T1 episode-boundary estimation (SCORING SKIPPED) ===")
        print("groups                  %d" % report["grouping"]["n_groups"])
        print("gated groups            %d"
              % sum(1 for r in report["program"]["groups"] if r["gated"]))
        print("scoring skipped         %s" % scoring["reason"])
        print("program sha256          %s" % report["program_sha256"])
        return
    print("")
    print("=== phase T1 episode-boundary estimation ===")
    print("groups                  %d" % report["grouping"]["n_groups"])
    print("gated / abstained       %d / %d"
          % (scoring["n_gated"], scoring["n_abstained"]))
    print("abstention rate         %.4f" % (scoring["abstention_rate"] or 0.0))
    print("opportunity denominator %d" % scoring["opportunity_denominator"])
    print("false activations       %d" % scoring["false_activations"])
    print("authored offset/onset   %d / %d"
          % (scoring["authored"]["offset_frame"], scoring["authored"]["onset_frame"]))
    for label, key in (("offset", "offset_error_frames"),
                       ("onset", "onset_error_frames")):
        summary = scoring[key]
        print("%-7s error  n=%-4d exact=%-4d |e|<=1=%-4d median=%s max|e|=%s"
              % (label, summary["count"], summary["exact"], summary["within_1"],
                 summary["median"], summary["max_abs"]))
    print("renders base/ablated    %d / %d"
          % (counts["base_renders"], counts["ablated_renders"]))
    print("estimation seconds      %.1f" % report["timing"]["estimation_seconds"])
    print("program sha256          %s" % report["program_sha256"])
    print("abstention reasons      %s" % json.dumps(scoring["abstention_reasons"],
                                                    sort_keys=True))


if __name__ == "__main__":
    main()
