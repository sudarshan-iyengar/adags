#!/usr/bin/env python3
"""LRV5-NCX: score axis-aligned CELL-HULL COMPLETION against a non-convex object.

MEASUREMENT ONLY. Pure numpy + stdlib, ZERO GPU, no torch, no rendering, no
training. It reads three primary artifacts -- the fixture's `event_spec.json`,
the TRAINED substrate `point_cloud.ply`, and the phase-T1 estimator report --
and emits precision, recall and the false-activation count three times: for the
frozen base rule, for base + H1, and for base + H2.

It implements section 9 of the frozen spec

    research-wiki/operations/nonconvex-hull-falsifier-spec-2026-08-24.md

and nothing else. No threshold, metric definition or hull operator in that
document may be changed here; if one is wrong, the spec is amended first.


WHAT IS UNDER TEST
------------------
`lrv3-membership-candidates-result-2026-08-23` section 7 records an axis-aligned
hull-completion step that would have taken candidate A2+B from 0.9375 / 0.8088
to 0.9400 / 1.0000 on LRV3. It was NOT adopted, and one of the two stated
reasons is inherited verbatim by this script: *hull completion assumes the
object is cell-convex; that is true of a sphere and need not be true of anything
else, so it is a fixture-shaped rule until tested on a non-convex object.*

LRV5-NCX is that test. The event object is an L whose axis-aligned CELL hull
provably contains cells with zero object volume, and those cells are occupied by
a persistent, always-visible static object (the notch filler, identity id 200).
If hull completion fills the concavity it suppresses that object for the whole
27-frame absence gap, which is a rendering error, not a bookkeeping deficit.


WHY THE DELTA, NOT THE ABSOLUTE POST-HULL NUMBER, IS THE VERDICT
----------------------------------------------------------------
Spec section 9, frozen attribution rule. Rows modelling the L's INNER FACES may
drift into the concavity cells on their own and depress the base rule's
precision before any hull operator runs. An absolute post-hull precision
therefore confounds "hull completion over-reached" with "the base rule was
already imprecise here". The delta base -> base+H1 is immune to that, because
both terms carry the same base contamination. This script reports all three
absolute rows AND both deltas, and states which one decides.


THE BASE RULE (frozen, `lrv3-membership-gate-spec-2026-08-23` sections 3-4)
--------------------------------------------------------------------------
Candidate **A2** -- transitive flood fill seeded by T1's accepted cells. A
rejected cell joins the growing component only if ALL of:

  (i)   face adjacency (6-connectivity) to the growing component;
  (ii)  its inferred onset AND offset equal the component's EXACTLY (no
        tolerance; a cell with no boundary estimate can never qualify);
  (iii) `agreeing_cameras >= 2`.

combined with candidate **B** -- same-cloud `row_ids` binding, i.e. the accepted
cells are read as ROWS OF THE TRAINED SUBSTRATE, the cloud the estimator ran on
and the cloud a `membership_mode: row_ids` program would bind to.


ROW ATTRIBUTION, DECLARED
-------------------------
A hull operator adds CELLS. A cell below `MIN_GROUP_ROWS` is not a group, and
candidate B's `row_ids` column (built by
`scripts/estimate_episodes.py::build_v2_program`) names only rows carrying a
GROUP label. Two readings therefore exist and both are reported:

  primary   -- rows of accepted cells THAT ARE GROUPS. This is what candidate B
               would actually bind. It is also the conservative choice for a
               rejection verdict, because it counts FEWER wrongly-gated rows.
  secondary -- every row in an accepted cell, including sub-threshold substrate
               rows. Reported as `rows_including_substrate`, never used for the
               verdict.

The primary reading is used for every gate number so the gate is not made easier
or harder by an undeclared choice.


ANTI-LEAKAGE
------------
This is the SCORING stage and nothing else. It opens ground truth by design --
that is its whole job -- and it must therefore never be imported by, or run
before, `scripts/estimate_episodes.py`'s estimation stage. The T1 report it
reads is already frozen and hashed; this script records that hash and never
alters it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from argparse import ArgumentParser
from collections import deque
from pathlib import Path

# INSERT, not append: same rationale as scripts/estimate_episodes.py -- the
# admitted image ships a `pointops2` whose `functions` subpackage is absent.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from depth_visibility.errors import ContractError  # noqa: E402

MANIFEST_SCHEMA = "adags-nonconvex-membership-score-v1"

# ---------------------------------------------------------------------------
# Frozen constants. MIRRORED, never redefined. Any change is a NEW spec.
# ---------------------------------------------------------------------------

#: Mirror of scripts/estimate_episodes.py::VOXEL_CELLS_PER_AXIS. Cross-checked
#: against the T1 report's own `grouping` block before anything is scored.
VOXEL_CELLS_PER_AXIS = 8
#: Mirror of scripts/estimate_episodes.py::MIN_GROUP_ROWS.
MIN_GROUP_ROWS = 4
#: Candidate A condition (iii): the JOIN bar, not T1's solo bar of 3.
#: `lrv3-membership-gate-spec-2026-08-23` section 3.
MIN_AGREEING_CAMERAS_JOIN = 2

#: THE GATE (spec section 10). No floor here may be moved after any score is
#: read, and it is not weakened if the fixture turns out to be hard.
GATE_PRECISION_FLOOR = 0.80
GATE_RECALL_FLOOR = 0.90
GATE_MAX_FALSE_ACTIVATIONS = 0

#: Precondition V2 (spec section 8): the concavity must be POPULATED.
V2_MIN_ROWS_IN_EMPTY_CELLS = 200
#: Precondition V4 (spec section 8): the bite must be real.
V4_MIN_ADDED_ROW_FRACTION = 0.25

#: Spec section 10.1, quoted verbatim. Printed when the rejection fires.
REJECTION_SENTENCE = (
    "If hull completion fills the concavity and violates precision >= 0.80, "
    "OR produces any false activation, hull completion is REJECTED as "
    "fixture-shaped."
)

_FACE_NEIGHBOURS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                    (0, -1, 0), (0, 0, 1), (0, 0, -1))
_CHEBYSHEV_1 = tuple((di, dj, dk)
                     for di in (-1, 0, 1)
                     for dj in (-1, 0, 1)
                     for dk in (-1, 0, 1))


# ---------------------------------------------------------------------------
# PLY reading (stdlib + numpy; `plyfile` is not assumed present)
# ---------------------------------------------------------------------------

_PLY_SCALARS = {
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
}


def read_ply_xyz(path):
    """(N, 3) float32 xyz from a PLY `vertex` element.

    `scene/gaussian_model.py::save_ply` writes every attribute as `f4` in the
    order `x, y, z, nx, ny, nz, f_dc..., [f_rest...], opacity, scale..., rot...`
    and `get_xyz` IS `_xyz`, so the first three columns are the exact tensor the
    estimator grouped. Both `binary_little_endian` and `ascii` are accepted;
    `binary_big_endian` and list properties are refused rather than guessed at.
    """
    raw = Path(path).read_bytes()
    marker = b"end_header"
    at = raw.find(marker)
    if at < 0:
        raise ContractError("%s: no PLY end_header" % (path,))
    line_end = raw.find(b"\n", at)
    header = raw[:line_end].decode("ascii", "replace").replace("\r\n", "\n")
    body = raw[line_end + 1:]

    fmt = None
    elements = []          # list of (name, count, [(prop_name, dtype_str)])
    for line in header.split("\n"):
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            elements.append((parts[1], int(parts[2]), []))
        elif parts[0] == "property":
            if not elements:
                raise ContractError("%s: property before element" % (path,))
            if parts[1] == "list":
                raise ContractError(
                    "%s: list properties are not supported; the trained cloud "
                    "has none" % (path,))
            if parts[1] not in _PLY_SCALARS:
                raise ContractError("%s: unknown PLY scalar %r"
                                    % (path, parts[1]))
            elements[-1][2].append((parts[2], _PLY_SCALARS[parts[1]]))
    if fmt not in ("binary_little_endian", "ascii"):
        raise ContractError("%s: unsupported PLY format %r" % (path, fmt))

    offset = 0
    for name, count, props in elements:
        dtype = np.dtype([(p, d) for p, d in props])
        if fmt == "ascii":
            if name != "vertex":
                raise ContractError(
                    "%s: ascii PLY with a non-vertex element is not supported"
                    % (path,))
            text = body.decode("ascii", "replace").split()
            values = np.asarray(text, dtype=np.float64)
            values = values.reshape(count, len(props))
            table = {p: values[:, i] for i, (p, _) in enumerate(props)}
        else:
            chunk = np.frombuffer(body, dtype=dtype, count=count, offset=offset)
            offset += count * dtype.itemsize
            table = {p: chunk[p] for p, _ in props}
        if name != "vertex":
            continue
        for axis in ("x", "y", "z"):
            if axis not in table:
                raise ContractError("%s: vertex element has no %r" % (path, axis))
        xyz = np.empty((count, 3), dtype=np.float32)
        for i, axis in enumerate(("x", "y", "z")):
            xyz[:, i] = np.asarray(table[axis], dtype=np.float32)
        return xyz
    raise ContractError("%s: no vertex element" % (path,))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cloud_fingerprint(xyz):
    """sha256 over the float32 xyz bytes, matching `elgs.trainer_hooks`."""
    data = np.ascontiguousarray(xyz, dtype=np.float32).tobytes()
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# The membership predicate: CLOSED BOX UNION, read from the artifact
# ---------------------------------------------------------------------------


def read_event_boxes(spec, key="event_object"):
    """Boxes of a closed-box-union object, refusing a sphere fixture loudly.

    LRV5's generator DELIBERATELY omits `centre`/`radius` so that a
    sphere-shaped consumer fails rather than silently mis-scoring. This
    function refuses the opposite mistake too: a sphere fixture handed to the
    non-convex scorer.
    """
    obj = spec.get(key)
    if not isinstance(obj, dict):
        raise ContractError("event_spec.json has no %r object" % (key,))
    if "centre" in obj or "radius" in obj:
        raise ContractError(
            "REFUSING: %r carries 'centre'/'radius', i.e. this is a SPHERE "
            "fixture (LRV1-LRV4). This scorer implements the closed-box-union "
            "predicate of the LRV5-NCX spec and must not be pointed at a "
            "sphere; score that fixture with "
            "scripts/estimate_episodes.py's own scoring stage instead."
            % (key,))
    if obj.get("kind") != "axis_aligned_box_union":
        raise ContractError("%r kind is %r, expected 'axis_aligned_box_union'"
                            % (key, obj.get("kind")))
    if obj.get("membership_predicate") != "closed_box_union":
        raise ContractError(
            "%r membership_predicate is %r, expected 'closed_box_union'"
            % (key, obj.get("membership_predicate")))
    boxes = obj.get("boxes") or []
    if not boxes:
        raise ContractError("%r carries no boxes" % (key,))
    out = []
    for entry in boxes:
        lo = np.asarray(entry["lo"], dtype=np.float64)
        hi = np.asarray(entry["hi"], dtype=np.float64)
        if lo.shape != (3,) or hi.shape != (3,):
            raise ContractError("%r box %r is not 3-D" % (key, entry.get("name")))
        if not np.all(hi >= lo):
            raise ContractError("%r box %r has hi < lo" % (key, entry.get("name")))
        out.append((str(entry.get("name", "box%d" % len(out))), lo, hi))
    return out


def in_box_union(points, boxes):
    """Closed, inclusive on both sides: OR over boxes of (lo <= p <= hi)."""
    p = np.asarray(points, dtype=np.float64)
    mask = np.zeros(p.shape[0], dtype=bool)
    for _, lo, hi in boxes:
        mask |= np.all((p >= lo) & (p <= hi), axis=1)
    return mask


def box_cell_volume(lo, hi, cell_lo, cell_hi):
    """Exact volume of an axis-aligned box intersected with an axis-aligned cell."""
    lengths = np.minimum(hi, cell_hi) - np.maximum(lo, cell_lo)
    if np.any(lengths <= 0.0):
        return 0.0
    return float(np.prod(lengths))


# ---------------------------------------------------------------------------
# The voxel grid, reproduced EXACTLY from scripts/estimate_episodes.py
# ---------------------------------------------------------------------------


def voxel_grid(xyz, cells_per_axis=VOXEL_CELLS_PER_AXIS):
    """(lo, span, keys), bit-faithful to `estimate_episodes.voxel_grid`.

    The arithmetic is done in FLOAT32 because torch does it in float32; a
    float64 reproduction can place a boundary row in a different cell.
    `clamp` is applied to the float before truncation, exactly as torch does.
    """
    cells = int(cells_per_axis)
    if cells < 1:
        raise ContractError("cells_per_axis must be >= 1")
    p = np.ascontiguousarray(xyz, dtype=np.float32)
    lo = p.min(axis=0)
    span = np.maximum(p.max(axis=0) - lo, np.float32(1e-6)).astype(np.float32)
    voxel = ((p - lo) / span * np.float32(cells)).astype(np.float32)
    voxel = np.clip(voxel, np.float32(0.0), np.float32(cells - 1))
    idx = voxel.astype(np.int64)
    keys = idx[:, 0] * cells * cells + idx[:, 1] * cells + idx[:, 2]
    return lo, span, keys


def build_voxel_groups(keys, min_rows=MIN_GROUP_ROWS):
    """(labels, n_groups, kept_keys); -1 = substrate.

    Faithful to `estimate_episodes.build_voxel_groups`: group ids are assigned
    in ASCENDING CELL-KEY order over the kept cells, which is what makes the T1
    report's integer group ids decodable back to cell indices.
    """
    unique, inverse, counts = np.unique(keys, return_inverse=True,
                                        return_counts=True)
    keep = counts >= int(min_rows)
    remap = np.full(unique.shape[0], -1, dtype=np.int64)
    kept = int(keep.sum())
    if kept:
        remap[keep] = np.arange(kept, dtype=np.int64)
    return remap[inverse], kept, unique[keep]


def decode_key(key, cells_per_axis=VOXEL_CELLS_PER_AXIS):
    cells = int(cells_per_axis)
    key = int(key)
    return (key // (cells * cells), (key // cells) % cells, key % cells)


def encode_cell(cell, cells_per_axis=VOXEL_CELLS_PER_AXIS):
    cells = int(cells_per_axis)
    i, j, k = cell
    return int(i) * cells * cells + int(j) * cells + int(k)


# ---------------------------------------------------------------------------
# THE HULL OPERATORS -- spec section 9, transcribed
# ---------------------------------------------------------------------------


def six_connected_components(cells):
    """6-connected components of a set of integer cell indices."""
    remaining = set(cells)
    out = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        queue = deque([seed])
        while queue:
            i, j, k = queue.popleft()
            for di, dj, dk in _FACE_NEIGHBOURS:
                nb = (i + di, j + dj, k + dk)
                if nb in remaining:
                    remaining.discard(nb)
                    component.add(nb)
                    queue.append(nb)
        out.append(component)
    return out


def bbox_cells(cells):
    """Every cell in the axis-aligned INDEX bounding box of `cells`."""
    if not cells:
        return set()
    i0 = min(c[0] for c in cells); i1 = max(c[0] for c in cells)
    j0 = min(c[1] for c in cells); j1 = max(c[1] for c in cells)
    k0 = min(c[2] for c in cells); k1 = max(c[2] for c in cells)
    return set((i, j, k)
               for i in range(i0, i1 + 1)
               for j in range(j0, j1 + 1)
               for k in range(k0, k1 + 1))


def hull_h1(accepted):
    """H1 -- axis-aligned bounding-box fill of the accepted component.

    PER 6-CONNECTED COMPONENT, then unioned. Not the bbox of the whole set:
    that distinction is what makes H1's declared vacuity mode (a component
    confined to one arm) possible, and precondition V3 exists to catch it.
    """
    out = set()
    for component in six_connected_components(accepted):
        out |= bbox_cells(component)
    return out


def hull_h2(accepted):
    """H2 -- 3x3x3 morphological closing, clipped to the accepted bbox.

    Dilation and erosion both use the full 26-neighbour structuring element
    (Chebyshev radius 1). The result is unioned with A (closing is extensive)
    and intersected with bbox(A) so it can never grow outward.
    """
    if not accepted:
        return set()
    dilated = set()
    for i, j, k in accepted:
        for di, dj, dk in _CHEBYSHEV_1:
            dilated.add((i + di, j + dj, k + dk))
    eroded = set()
    for cell in dilated:
        i, j, k = cell
        if all((i + di, j + dj, k + dk) in dilated
               for di, dj, dk in _CHEBYSHEV_1):
            eroded.add(cell)
    return (eroded | set(accepted)) & bbox_cells(accepted)


# ---------------------------------------------------------------------------
# THE BASE RULE -- candidate A2, on cells
# ---------------------------------------------------------------------------


def a2_flood_fill(seed_cells, decisions_by_cell):
    """Candidate A2: transitive flood fill under FACE adjacency.

    `decisions_by_cell` maps a cell index to a dict carrying `onset_frame`,
    `offset_frame` and `agreeing_cameras`. A cell joins only if it is face
    adjacent to the growing component, reproduces the component's onset AND
    offset EXACTLY, and carries `agreeing_cameras >= MIN_AGREEING_CAMERAS_JOIN`.

    The flood is run once PER 6-connected component of the seed set, using that
    component's own reference boundaries, and the results are unioned. A seed
    component whose own cells disagree about the boundaries leaves the frozen
    rule undefined, so that is refused rather than resolved by invention.
    """
    seeds = set(seed_cells)
    accepted = set(seeds)
    references = []
    for component in six_connected_components(seeds):
        pairs = set()
        for cell in component:
            record = decisions_by_cell[cell]
            pairs.add((record.get("offset_frame"), record.get("onset_frame")))
        if len(pairs) != 1:
            raise ContractError(
                "candidate A2 is undefined: the T1-accepted component %s "
                "carries more than one (offset, onset) pair %s. The frozen "
                "rule agrees a joining cell against 'the component's' "
                "boundaries and there is no such thing here."
                % (sorted(component), sorted(pairs)))
        reference = pairs.pop()
        references.append({"cells": sorted(component),
                           "offset_frame": reference[0],
                           "onset_frame": reference[1]})
        if reference[0] is None or reference[1] is None:
            continue
        queue = deque(component)
        grown = set(component)
        while queue:
            i, j, k = queue.popleft()
            for di, dj, dk in _FACE_NEIGHBOURS:
                nb = (i + di, j + dj, k + dk)
                if nb in grown:
                    continue
                record = decisions_by_cell.get(nb)
                if record is None:
                    continue
                if record.get("offset_frame") != reference[0]:
                    continue
                if record.get("onset_frame") != reference[1]:
                    continue
                cams = record.get("agreeing_cameras")
                if cams is None or int(cams) < MIN_AGREEING_CAMERAS_JOIN:
                    continue
                grown.add(nb)
                queue.append(nb)
        accepted |= grown
    return accepted, references


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


class CellTable(object):
    """Per-cell row counts on ONE cloud, plus the cell -> group mapping."""

    def __init__(self, cells_per_axis, keys, labels, event_mask, filler_mask):
        self.cells_per_axis = int(cells_per_axis)
        self.rows = {}
        self.event_rows = {}
        self.filler_rows = {}
        self.group_of_cell = {}
        self.cell_of_group = {}
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        bounds = np.searchsorted(sorted_keys, np.unique(sorted_keys))
        edges = list(bounds) + [len(sorted_keys)]
        for n, start in enumerate(edges[:-1]):
            stop = edges[n + 1]
            rows = order[start:stop]
            cell = decode_key(sorted_keys[start], self.cells_per_axis)
            self.rows[cell] = int(rows.shape[0])
            self.event_rows[cell] = int(event_mask[rows].sum())
            self.filler_rows[cell] = int(filler_mask[rows].sum())
            label = int(labels[rows[0]])
            if label >= 0:
                self.group_of_cell[cell] = label
                self.cell_of_group[label] = cell
        self.total_event_rows = int(event_mask.sum())

    def score(self, cell_set):
        """Precision, recall and the false-activation count for a cell set."""
        cell_set = set(cell_set)
        gated_groups = sorted(self.group_of_cell[c] for c in cell_set
                              if c in self.group_of_cell)
        rows = sum(self.rows[self.cell_of_group[g]] for g in gated_groups)
        hits = sum(self.event_rows[self.cell_of_group[g]] for g in gated_groups)
        filler = sum(self.filler_rows[self.cell_of_group[g]] for g in gated_groups)
        false_groups = [g for g in gated_groups
                        if self.event_rows[self.cell_of_group[g]] == 0]
        rows_all = sum(self.rows.get(c, 0) for c in cell_set)
        zero_object_cells = sorted(c for c in cell_set
                                   if self.event_rows.get(c, 0) == 0)
        return {
            "cells": len(cell_set),
            "cells_that_are_groups": len(gated_groups),
            "cells_without_a_group": len(cell_set) - len(gated_groups),
            "rows_gated": rows,
            "rows_including_substrate": rows_all,
            "rows_in_event_object": hits,
            "rows_in_notch_filler": filler,
            "precision": (hits / float(rows)) if rows else None,
            "recall": ((hits / float(self.total_event_rows))
                       if self.total_event_rows else None),
            "false_activations": len(false_groups),
            "false_activation_groups": false_groups,
            "gated_groups": gated_groups,
            "zero_object_cells": [list(c) for c in zero_object_cells],
            "n_zero_object_cells": len(zero_object_cells),
        }


def _delta(before, after):
    def diff(key):
        a, b = before.get(key), after.get(key)
        if a is None or b is None:
            return None
        return b - a
    return {
        "cells": diff("cells"),
        "rows_gated": diff("rows_gated"),
        "rows_in_event_object": diff("rows_in_event_object"),
        "rows_in_notch_filler": diff("rows_in_notch_filler"),
        "precision": diff("precision"),
        "recall": diff("recall"),
        "false_activations": diff("false_activations"),
        "n_zero_object_cells": diff("n_zero_object_cells"),
    }


def gate_verdict(score):
    """The three frozen gate conditions, evaluated without interpretation."""
    precision = score["precision"]
    recall = score["recall"]
    return {
        "precision_ok": precision is not None and precision >= GATE_PRECISION_FLOOR,
        "recall_ok": recall is not None and recall >= GATE_RECALL_FLOOR,
        "false_activations_ok": (score["false_activations"]
                                 <= GATE_MAX_FALSE_ACTIVATIONS),
        "precision_floor": GATE_PRECISION_FLOOR,
        "recall_floor": GATE_RECALL_FLOOR,
        "max_false_activations": GATE_MAX_FALSE_ACTIVATIONS,
    }


# ---------------------------------------------------------------------------
# fixture-validity preconditions V1-V4 (spec section 8)
# ---------------------------------------------------------------------------


def check_preconditions(table, base_cells, base_score, h1_score, event_boxes,
                        lo, span, cells_per_axis):
    """V1-V4 on the REALIZED trained cloud and its REALIZED grid."""
    occupied = sorted(c for c, n in table.event_rows.items() if n > 0)
    hull = bbox_cells(occupied) if occupied else set()
    empty_in_hull = sorted(c for c in hull if table.event_rows.get(c, 0) == 0)

    v1 = len(empty_in_hull) >= 1

    rows_in_empty = sum(table.rows.get(c, 0) for c in empty_in_hull)
    max_rows_in_one = max([table.rows.get(c, 0) for c in empty_in_hull] or [0])
    v2 = (rows_in_empty >= V2_MIN_ROWS_IN_EMPTY_CELLS
          and max_rows_in_one >= MIN_GROUP_ROWS)

    cell_size = (np.asarray(span, dtype=np.float64)
                 / float(int(cells_per_axis)))
    origin = np.asarray(lo, dtype=np.float64)
    by_name = {}
    arm_only = {}
    for name, box_lo, box_hi in event_boxes:
        by_name[name] = (box_lo, box_hi)
    names = [n for n, _, _ in event_boxes]
    for cell in sorted(base_cells):
        cell_lo = origin + np.asarray(cell, dtype=np.float64) * cell_size
        cell_hi = cell_lo + cell_size
        volumes = {name: box_cell_volume(by_name[name][0], by_name[name][1],
                                         cell_lo, cell_hi)
                   for name in names}
        positive = [name for name in names if volumes[name] > 0.0]
        if len(positive) == 1:
            arm_only.setdefault(positive[0], []).append(list(cell))
    v3 = len(names) >= 2 and all(len(arm_only.get(name, [])) >= 1
                                 for name in names)

    base_rows = base_score["rows_gated"]
    added_rows = h1_score["rows_gated"] - base_rows
    v4 = (base_rows > 0
          and added_rows >= V4_MIN_ADDED_ROW_FRACTION * base_rows)

    return {
        "V1_concavity_exists_at_the_realized_grid": {
            "passed": bool(v1),
            "statement": ("the index bbox of the event object's occupied cells "
                          "contains >= 1 cell with ZERO rows satisfying "
                          "is_event"),
            "occupied_cells": len(occupied),
            "hull_cells": len(hull),
            "empty_in_hull_cells": len(empty_in_hull),
            "empty_in_hull": [list(c) for c in empty_in_hull],
        },
        "V2_concavity_is_populated": {
            "passed": bool(v2),
            "statement": ("the union of V1's cells holds >= %d rows and >= 1 of "
                          "them individually holds >= %d rows"
                          % (V2_MIN_ROWS_IN_EMPTY_CELLS, MIN_GROUP_ROWS)),
            "rows_in_empty_hull_cells": int(rows_in_empty),
            "max_rows_in_a_single_empty_cell": int(max_rows_in_one),
            "min_rows_required": V2_MIN_ROWS_IN_EMPTY_CELLS,
        },
        "V3_operator_is_not_vacuous": {
            "passed": bool(v3),
            "statement": ("the accepted component holds >= 1 cell whose object "
                          "volume comes ONLY from each named arm; otherwise H1's "
                          "per-component bbox never covers the notch"),
            "cells_only_from": {name: arm_only.get(name, []) for name in names},
            "counts": {name: len(arm_only.get(name, [])) for name in names},
        },
        "V4_bite_is_real": {
            "passed": bool(v4),
            "statement": ("rows added by H1 are >= %.2f x rows already gated by "
                          "the base rule" % V4_MIN_ADDED_ROW_FRACTION),
            "base_rows_gated": int(base_rows),
            "rows_added_by_h1": int(added_rows),
            "required_added_rows": (V4_MIN_ADDED_ROW_FRACTION * base_rows),
            "measured_fraction": ((added_rows / float(base_rows))
                                  if base_rows else None),
        },
    }


def decide(preconditions, h1_score):
    """Spec section 10.1, applied literally. The gate is NEVER weakened here."""
    failed = [name for name, block in sorted(preconditions.items())
              if not block["passed"]]
    if failed:
        return {
            "verdict": "INVALID",
            "clause": "spec 8: a fixture-validity precondition failed",
            "failed_preconditions": failed,
            "note": ("No verdict on hull completion may be read from this run. "
                     "It is recorded as an invalid instrument, exactly as "
                     "lrv4-starved-fixture-result-2026-08-23 was."),
        }
    gate = gate_verdict(h1_score)
    fills_concavity = h1_score["n_zero_object_cells"] >= 1
    violates = (not gate["precision_ok"]) or (not gate["false_activations_ok"])
    if fills_concavity and violates:
        return {
            "verdict": "REJECTED",
            "clause": ("spec 10.1: base+H1 gates >= 1 cell with zero "
                       "event-object rows AND (precision < %.2f OR false "
                       "activations > %d)"
                       % (GATE_PRECISION_FLOOR, GATE_MAX_FALSE_ACTIVATIONS)),
            "failed_preconditions": [],
            "note": REJECTION_SENTENCE,
        }
    if gate["precision_ok"] and gate["recall_ok"] and gate["false_activations_ok"]:
        return {
            "verdict": "SURVIVES ROUND 1",
            "clause": ("spec 10.1: base+H1 met all three gate conditions and "
                       "V1-V4 all held"),
            "failed_preconditions": [],
            "note": ("NOT admission. Spec section 11 requires orientation O2 to "
                     "be built, trained and scored under this same spec before "
                     "hull completion may be admitted."),
        }
    return {
        "verdict": "INCONCLUSIVE",
        "clause": ("neither section 10.1 clause fired: base+H1 did not meet the "
                   "gate, and the rejection antecedent (>= 1 gated cell with "
                   "zero event-object rows) did not hold"),
        "failed_preconditions": [],
        "note": ("The operator was not exercised as designed on this run. No "
                 "verdict on hull completion. Do NOT read this as a pass, and "
                 "do NOT move a floor to resolve it."),
    }


# ---------------------------------------------------------------------------
# self-test: H1, H2 and the predicate on hand-constructed inputs
# ---------------------------------------------------------------------------


def _expect(name, got, want, failures):
    ok = got == want
    print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
    if not ok:
        print("        got  %r" % (got,))
        print("        want %r" % (want,))
        failures.append(name)
    return ok


def self_test():
    """Hand-checked H1/H2 answers. Needs NO data files and no GPU."""
    failures = []
    print("=== self-test: hull operators (spec section 9) ===")

    # -- H1 ---------------------------------------------------------------
    # A minimal L in the j = 0 plane: arms of length 3, thickness 1. Its
    # index bbox is the full 3 x 3 block, so H1 fills the entire 2 x 2 notch.
    ell = {(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 0, 1), (0, 0, 2)}
    notch = {(1, 0, 1), (1, 0, 2), (2, 0, 1), (2, 0, 2)}
    _expect("H1 fills the whole 2x2 notch of an L",
            hull_h1(ell), ell | notch, failures)
    _expect("H1 is extensive on the L", ell <= hull_h1(ell), True, failures)

    # The spec's DECLARED VACUITY MODE (precondition V3 exists for it): a
    # component confined to ONE arm has a bbox that never covers the notch.
    arm_only = {(0, 0, 0), (0, 0, 1), (0, 0, 2)}
    _expect("H1 VACUITY MODE: a one-arm component adds nothing",
            hull_h1(arm_only), arm_only, failures)
    _expect("H1 vacuity mode adds exactly 0 cells",
            len(hull_h1(arm_only) - arm_only), 0, failures)

    # PER-COMPONENT, not the bbox of the union. Two separated components must
    # not be bridged: a whole-set bbox would return 49 cells here.
    two = {(0, 0, 0), (0, 0, 1), (5, 0, 5), (5, 0, 6), (6, 0, 5)}
    _expect("H1 is per 6-connected component, not one global bbox",
            hull_h1(two),
            {(0, 0, 0), (0, 0, 1), (5, 0, 5), (5, 0, 6), (6, 0, 5), (6, 0, 6)},
            failures)
    _expect("H1 global-bbox reading is NOT what is implemented",
            len(hull_h1(two)) != len(bbox_cells(two)), True, failures)

    # The LRV3 case the operator came from: 6 of the 8 cells of a contiguous
    # 2 x 2 x 2 block; H1 fills exactly the two holes.
    block = bbox_cells({(5, 4, 4), (6, 5, 5)})
    lrv3 = block - {(6, 4, 4), (6, 5, 4)}
    _expect("H1 fills the LRV3 2x2x2 block's two holes",
            hull_h1(lrv3), block, failures)

    _expect("H1 of the empty set is empty", hull_h1(set()), set(), failures)
    single = {(3, 3, 3)}
    _expect("H1 of one cell is that cell", hull_h1(single), single, failures)

    # -- H2 ---------------------------------------------------------------
    # An ENCLOSED hole is closed: the 8 cells of a 3 x 3 ring, hole at centre.
    ring = set((i, 0, k) for i in range(3) for k in range(3)) - {(1, 0, 1)}
    _expect("H2 closes an enclosed 1-cell hole",
            hull_h2(ring), ring | {(1, 0, 1)}, failures)

    # A 3 x 3 x 3 solid minus its centre closes the same way.
    solid = set((i, j, k) for i in range(3) for j in range(3) for k in range(3))
    _expect("H2 closes the centre of a 3x3x3 solid",
            hull_h2(solid - {(1, 1, 1)}), solid, failures)

    # THE PREDICTED RESULT (spec section 10): a 3 x 3 x 3 closing cannot fill a
    # notch two cells wide. Same L as the H1 case above.
    _expect("H2 adds NOTHING to the L's 2-cell-wide notch",
            hull_h2(ell), ell, failures)
    _expect("H2 leaves the L's notch cells ungated",
            hull_h2(ell) & notch, set(), failures)

    # H2 is extensive, and clipped so it can never grow outward.
    _expect("H2 is extensive on the ring", ring <= hull_h2(ring), True, failures)
    _expect("H2 never leaves bbox(A)",
            hull_h2(ring) <= bbox_cells(ring), True, failures)
    _expect("H2 of one cell is that cell", hull_h2(single), single, failures)
    _expect("H2 of the empty set is empty", hull_h2(set()), set(), failures)

    # -- candidate A2 ------------------------------------------------------
    print("=== self-test: the base rule (candidate A2) ===")
    decisions = {
        (0, 0, 0): {"offset_frame": 30, "onset_frame": 57, "agreeing_cameras": 4},
        (1, 0, 0): {"offset_frame": 30, "onset_frame": 57, "agreeing_cameras": 2},
        (2, 0, 0): {"offset_frame": 30, "onset_frame": 57, "agreeing_cameras": 2},
        (3, 0, 0): {"offset_frame": 30, "onset_frame": 57, "agreeing_cameras": 1},
        (0, 0, 1): {"offset_frame": 31, "onset_frame": 57, "agreeing_cameras": 4},
        (0, 1, 0): {"offset_frame": None, "onset_frame": None,
                    "agreeing_cameras": 0},
        (7, 7, 7): {"offset_frame": 30, "onset_frame": 57, "agreeing_cameras": 4},
    }
    accepted, refs = a2_flood_fill({(0, 0, 0)}, decisions)
    _expect("A2 grows TRANSITIVELY through a 2-camera chain",
            accepted, {(0, 0, 0), (1, 0, 0), (2, 0, 0)}, failures)
    _expect("A2 refuses a 1-camera cell", (3, 0, 0) in accepted, False, failures)
    _expect("A2 refuses a cell whose offset differs by 1",
            (0, 0, 1) in accepted, False, failures)
    _expect("A2 refuses a cell with no boundary estimate",
            (0, 1, 0) in accepted, False, failures)
    _expect("A2 refuses a non-adjacent cell", (7, 7, 7) in accepted, False,
            failures)
    _expect("A2 reports one seed component", len(refs), 1, failures)

    ok = True
    try:
        a2_flood_fill({(0, 0, 0), (1, 0, 0)},
                      {(0, 0, 0): {"offset_frame": 30, "onset_frame": 57,
                                   "agreeing_cameras": 4},
                       (1, 0, 0): {"offset_frame": 29, "onset_frame": 57,
                                   "agreeing_cameras": 4}})
        ok = False
    except ContractError:
        pass
    _expect("A2 refuses a seed component with disagreeing boundaries", ok, True,
            failures)

    # -- the membership predicate -----------------------------------------
    print("=== self-test: the closed-box-union predicate ===")
    boxes = [("arm_a_z", np.array([-0.64, -0.30, -0.64]),
              np.array([-0.24, 0.30, 0.64])),
             ("arm_b_x", np.array([-0.64, -0.30, -0.64]),
              np.array([0.64, 0.30, -0.24]))]
    points = np.array([
        [-0.64, -0.30, -0.64],   # a corner, CLOSED -> inside
        [-0.24, 0.30, 0.64],     # arm A's far corner, CLOSED -> inside
        [-0.44, 0.00, 0.00],     # deep inside arm A
        [0.00, 0.00, -0.44],     # deep inside arm B
        [0.20, 0.00, 0.20],      # THE NOTCH -> outside
        [0.20, 0.00, 0.20],      # (repeated; the notch filler sits here)
        [-0.44, 0.31, 0.00],     # just outside in y
        [0.65, 0.00, -0.44],     # just outside in x
    ])
    _expect("closed box union: faces included, notch excluded",
            in_box_union(points, boxes).tolist(),
            [True, True, True, True, False, False, False, False], failures)

    spec_like = {"event_object": {"id": 100, "kind": "axis_aligned_box_union",
                                  "membership_predicate": "closed_box_union",
                                  "boxes": [{"name": "arm_a_z",
                                             "lo": [-0.64, -0.30, -0.64],
                                             "hi": [-0.24, 0.30, 0.64]}]}}
    _expect("read_event_boxes accepts a box-union spec",
            len(read_event_boxes(spec_like)), 1, failures)
    ok = True
    try:
        read_event_boxes({"event_object": {"id": 100, "centre": [0.7, 0.1, 0.35],
                                           "radius": 0.2}})
        ok = False
    except ContractError:
        pass
    _expect("read_event_boxes REFUSES a sphere fixture", ok, True, failures)

    # -- exact box/cell volume --------------------------------------------
    print("=== self-test: exact box/cell volume and the grid ===")
    _expect("box/cell volume of a disjoint pair is exactly zero",
            box_cell_volume(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]),
                            np.array([2.0, 2.0, 2.0]), np.array([3.0, 3.0, 3.0])),
            0.0, failures)
    _expect("box/cell volume of a touching pair is exactly zero",
            box_cell_volume(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]),
                            np.array([1.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0])),
            0.0, failures)
    _expect("box/cell volume of a half overlap",
            box_cell_volume(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]),
                            np.array([0.5, 0.0, 0.0]), np.array([1.5, 1.0, 1.0])),
            0.5, failures)

    # -- the grid and the grouping ----------------------------------------
    cloud = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0],
                      [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=np.float32)
    lo, span, keys = voxel_grid(cloud, cells_per_axis=2)
    _expect("the grid lo is the cloud minimum", lo.tolist(), [0.0, 0.0, 0.0],
            failures)
    _expect("the grid span is the cloud extent", span.tolist(), [1.0, 1.0, 1.0],
            failures)
    _expect("the cloud maximum is CLAMPED into the last cell, not cells+1",
            decode_key(keys[1], 2), (1, 1, 1), failures)
    _expect("cell keys round-trip through decode/encode",
            [encode_key for encode_key in
             (encode_cell(decode_key(int(k), 2), 2) for k in keys)],
            [int(k) for k in keys], failures)
    labels, kept, kept_keys = build_voxel_groups(keys, min_rows=2)
    _expect("only the cell holding >= min_rows rows becomes a group", kept, 1,
            failures)
    _expect("sub-threshold rows are demoted to substrate (-1)",
            sorted(set(int(v) for v in labels)), [-1, 0], failures)
    _expect("the kept group is the 2-row cell",
            decode_key(int(kept_keys[0]), 2), (1, 1, 1), failures)

    # -- scoring on a hand-built table -------------------------------------
    print("=== self-test: precision / recall / false activations ===")
    # 3 cells of 4 rows each: cell A all event, cell B half event, cell C none.
    keys = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    labels = np.array([0] * 4 + [1] * 4 + [2] * 4, dtype=np.int64)
    event = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    filler = np.array([0] * 8 + [1, 1, 1, 1], dtype=bool)
    table = CellTable(8, keys, labels, event, filler)
    both = table.score([decode_key(0, 8), decode_key(1, 8)])
    _expect("precision over two cells", round(both["precision"], 6),
            round(6 / 8.0, 6), failures)
    _expect("recall over two cells", round(both["recall"], 6),
            round(6 / 6.0, 6), failures)
    _expect("no false activation when every gated group holds event rows",
            both["false_activations"], 0, failures)
    all_three = table.score([decode_key(0, 8), decode_key(1, 8), decode_key(2, 8)])
    _expect("a gated group with ZERO event rows is a false activation",
            all_three["false_activations"], 1, failures)
    _expect("the false activation names the right group",
            all_three["false_activation_groups"], [2], failures)
    _expect("precision falls when the empty cell is gated",
            round(all_three["precision"], 6), round(6 / 12.0, 6), failures)
    _expect("recall is unchanged by gating an event-free cell",
            all_three["recall"], both["recall"], failures)
    _expect("the notch-filler column counts the filler rows",
            all_three["rows_in_notch_filler"], 4, failures)
    _expect("the zero-object cell is identified",
            all_three["zero_object_cells"], [[0, 0, 2]], failures)
    delta = _delta(both, all_three)
    _expect("the DELTA reports the precision drop",
            round(delta["precision"], 6), round(6 / 12.0 - 6 / 8.0, 6), failures)
    _expect("the DELTA reports the added false activation",
            delta["false_activations"], 1, failures)

    # -- the gate is not weakened -----------------------------------------
    print("=== self-test: the frozen gate and the verdict ===")
    _expect("the precision floor is 0.80", GATE_PRECISION_FLOOR, 0.80, failures)
    _expect("the recall floor is 0.90", GATE_RECALL_FLOOR, 0.90, failures)
    _expect("zero false activations are allowed", GATE_MAX_FALSE_ACTIVATIONS, 0,
            failures)
    passing = {name: {"passed": True} for name in
               ("V1", "V2", "V3", "V4")}
    failing = dict(passing, V3={"passed": False})
    _expect("a failed precondition yields INVALID and no verdict",
            decide(failing, all_three)["verdict"], "INVALID", failures)
    _expect("a filled concavity at precision 0.50 is REJECTED",
            decide(passing, all_three)["verdict"], "REJECTED", failures)
    # A clean instrument: two gated cells, 9 of 10 rows in the event object and
    # every event row gated -- precision 0.90, recall 1.00, no false activation.
    clean_keys = np.array([0] * 5 + [1] * 5, dtype=np.int64)
    clean_labels = np.array([0] * 5 + [1] * 5, dtype=np.int64)
    clean_event = np.array([1, 1, 1, 1, 0] + [1] * 5, dtype=bool)
    clean_table = CellTable(8, clean_keys, clean_labels, clean_event,
                            np.zeros(10, dtype=bool))
    clean = clean_table.score([decode_key(0, 8), decode_key(1, 8)])
    _expect("a clean gate at full recall SURVIVES ROUND 1",
            decide(passing, clean)["verdict"], "SURVIVES ROUND 1", failures)
    # Recall below the floor with NO zero-object cell fires neither clause.
    partial = table.score([decode_key(0, 8)])
    _expect("a partial gate that fills no concavity is INCONCLUSIVE",
            decide(passing, partial)["verdict"], "INCONCLUSIVE", failures)

    print("")
    if failures:
        print("SELF-TEST FAILED: %d check(s): %s" % (len(failures), failures))
        return 1
    print("SELF-TEST PASSED")
    return 0


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def load_t1_groups(report):
    """The per-group decisions, from a T1 report or a bare frozen program."""
    program = report.get("program") if isinstance(report, dict) else None
    if program is None and isinstance(report, dict) and "groups" in report:
        program = report
    if not isinstance(program, dict) or "groups" not in program:
        raise ContractError(
            "the estimator report carries no frozen program with a 'groups' "
            "array; expected scripts/estimate_episodes.py's --out_report")
    return program


def _fmt(value, spec="%.4f"):
    return "n/a" if value is None else (spec % value)


def _print_row(label, score):
    print("%-22s %6d %10d %11s %9s %8d %10d"
          % (label, score["cells"], score["rows_gated"],
             _fmt(score["precision"]), _fmt(score["recall"]),
             score["false_activations"], score["n_zero_object_cells"]))


def main(argv=None):
    parser = ArgumentParser(
        description=("score the LRV5-NCX non-convex hull-completion falsifier "
                     "(frozen spec section 9); CPU only"))
    parser.add_argument("--self-test", dest="self_test", action="store_true",
                        help="validate H1/H2 and the metrics on hand-built "
                             "inputs; needs no data files")
    parser.add_argument("--source_path",
                        help="the LRV5 fixture directory holding event_spec.json")
    parser.add_argument("--point_cloud",
                        help="the TRAINED substrate point_cloud.ply")
    parser.add_argument("--estimate_report",
                        help="scripts/estimate_episodes.py --out_report JSON")
    parser.add_argument("--v2_program", default="",
                        help="optional --emit_program JSON; when given, its "
                             "cloud fingerprint and grid are cross-checked")
    parser.add_argument("--out_manifest",
                        help="where to write the JSON manifest")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.self_test:
        return self_test()

    missing = [name for name in ("source_path", "point_cloud",
                                 "estimate_report", "out_manifest")
               if not getattr(args, name)]
    if missing:
        raise ContractError("missing required arguments: %s"
                            % ", ".join("--" + m for m in missing))

    spec_path = Path(args.source_path) / "event_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    event_boxes = read_event_boxes(spec, "event_object")
    filler_boxes = read_event_boxes(spec, "notch_filler")

    report = json.loads(Path(args.estimate_report).read_text(encoding="utf-8"))
    program = load_t1_groups(report)

    grouping = report.get("grouping") or {}
    cells_per_axis = int(grouping.get("cells_per_axis", VOXEL_CELLS_PER_AXIS))
    min_group_rows = int(grouping.get("min_group_rows", MIN_GROUP_ROWS))
    if cells_per_axis != VOXEL_CELLS_PER_AXIS or min_group_rows != MIN_GROUP_ROWS:
        raise ContractError(
            "the T1 report was produced at cells_per_axis=%d min_group_rows=%d, "
            "but the frozen fixture constants are %d / %d. These are frozen "
            "constants of scripts/estimate_episodes.py, not flags."
            % (cells_per_axis, min_group_rows, VOXEL_CELLS_PER_AXIS,
               MIN_GROUP_ROWS))

    xyz = read_ply_xyz(args.point_cloud)
    fingerprint = cloud_fingerprint(xyz)
    if grouping.get("n_rows") is not None and int(grouping["n_rows"]) != xyz.shape[0]:
        raise ContractError(
            "the point cloud has %d rows but the T1 report grouped %d. This is "
            "NOT the cloud the estimate was computed on."
            % (xyz.shape[0], int(grouping["n_rows"])))

    lo, span, keys = voxel_grid(xyz, cells_per_axis)
    labels, n_groups, kept_keys = build_voxel_groups(keys, min_group_rows)
    if n_groups != int(program["n_groups"]):
        raise ContractError(
            "the grouping reproduced %d groups but the frozen program declares "
            "%d. The cloud or the grid does not match the estimate."
            % (n_groups, int(program["n_groups"])))

    # Per-group row counts must reproduce the frozen program EXACTLY. This is
    # the strong check that the cloud, the grid and the estimate agree.
    counts = np.bincount(labels[labels >= 0], minlength=n_groups)
    for record in program["groups"]:
        group = int(record["group"])
        if int(record["rows"]) != int(counts[group]):
            raise ContractError(
                "group %d holds %d rows in this cloud but the frozen program "
                "recorded %d" % (group, int(counts[group]), int(record["rows"])))

    v2_checks = {"provided": bool(args.v2_program)}
    if args.v2_program:
        v2 = json.loads(Path(args.v2_program).read_text(encoding="utf-8"))
        declared = str((v2.get("cloud") or {}).get("xyz_sha256", ""))
        v2_checks["xyz_sha256_declared"] = declared
        v2_checks["xyz_sha256_matches"] = (declared == fingerprint)
        spatial = v2.get("spatial") or {}
        v2_checks["lo_matches"] = (
            [float(v) for v in spatial.get("lo", [])]
            == [float(v) for v in lo.tolist()])
        v2_checks["span_matches"] = (
            [float(v) for v in spatial.get("span", [])]
            == [float(v) for v in span.tolist()])
        for key in ("xyz_sha256_matches", "lo_matches", "span_matches"):
            if not v2_checks[key]:
                raise ContractError(
                    "the v2 program does not describe this cloud: %s is False. "
                    "Refusing to score a mismatched pair." % key)

    cell_of_group = {}
    decisions_by_cell = {}
    for n, key in enumerate(kept_keys):
        cell_of_group[n] = decode_key(int(key), cells_per_axis)
    for record in program["groups"]:
        decisions_by_cell[cell_of_group[int(record["group"])]] = record

    event_mask = in_box_union(xyz, event_boxes)
    filler_mask = in_box_union(xyz, filler_boxes)
    table = CellTable(cells_per_axis, keys, labels, event_mask, filler_mask)

    seeds = set(cell_of_group[int(r["group"])] for r in program["groups"]
                if r.get("gated"))
    if not seeds:
        raise ContractError(
            "the T1 estimator gated NO group, so candidate A2 has no seed and "
            "the base rule is empty. There is nothing to hull-complete and no "
            "verdict on hull completion is available from this run.")
    base_cells, references = a2_flood_fill(seeds, decisions_by_cell)
    h1_cells = hull_h1(base_cells)
    h2_cells = hull_h2(base_cells)

    base = table.score(base_cells)
    h1 = table.score(h1_cells)
    h2 = table.score(h2_cells)
    preconditions = check_preconditions(table, base_cells, base, h1, event_boxes,
                                        lo, span, cells_per_axis)
    verdict = decide(preconditions, h1)

    # If the base rule ALREADY gates a zero-object cell, section 10.1's
    # antecedent can fire without H1 having added one. Say so rather than let
    # the reading be ambiguous.
    verdict["base_already_gated_zero_object_cells"] = base["n_zero_object_cells"]
    verdict["zero_object_cells_added_by_h1"] = (
        h1["n_zero_object_cells"] - base["n_zero_object_cells"])

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "spec_document": ("research-wiki/operations/"
                          "nonconvex-hull-falsifier-spec-2026-08-24.md"),
        "evidence_bearing": False,
        "scene_id": spec.get("scene_id"),
        "orientation": spec.get("orientation"),
        "inputs": {
            "event_spec": {"path": str(spec_path),
                           "sha256": sha256_file(spec_path)},
            "point_cloud": {"path": str(args.point_cloud),
                            "sha256": sha256_file(args.point_cloud),
                            "n_rows": int(xyz.shape[0]),
                            "xyz_sha256": fingerprint},
            "estimate_report": {"path": str(args.estimate_report),
                                "sha256": sha256_file(args.estimate_report),
                                "program_sha256": report.get("program_sha256")},
            "v2_program": v2_checks,
        },
        "membership_predicate": {
            "kind": "closed_box_union",
            "event_boxes": [{"name": n, "lo": l.tolist(), "hi": h.tolist()}
                            for n, l, h in event_boxes],
            "notch_filler_boxes": [{"name": n, "lo": l.tolist(), "hi": h.tolist()}
                                   for n, l, h in filler_boxes],
            "note": ("is_event(p) = OR over boxes of (lo <= p <= hi), "
                     "componentwise and INCLUSIVE. Spec sections 1.1 and 9."),
        },
        "grid": {
            "cells_per_axis": cells_per_axis,
            "min_group_rows": min_group_rows,
            "lo": [float(v) for v in lo.tolist()],
            "span": [float(v) for v in span.tolist()],
            "n_groups": int(n_groups),
            "n_rows": int(xyz.shape[0]),
            "rows_in_event_object": int(table.total_event_rows),
            "rows_in_notch_filler": int(filler_mask.sum()),
        },
        "base_rule": {
            "candidate": "A2 (transitive face-adjacency flood fill) + B "
                         "(same-cloud row_ids binding on the trained substrate)",
            "min_agreeing_cameras_to_join": MIN_AGREEING_CAMERAS_JOIN,
            "t1_gated_cells": sorted(list(c) for c in seeds),
            "seed_components": references,
            "accepted_cells": sorted(list(c) for c in base_cells),
        },
        "row_attribution": {
            "primary": ("rows of accepted cells THAT ARE GROUPS -- what "
                        "candidate B's row_ids column binds"),
            "secondary": ("rows_including_substrate additionally counts rows in "
                          "accepted cells below min_group_rows; reported, never "
                          "used for the verdict"),
        },
        "scores": {"base": base, "base_plus_h1": h1, "base_plus_h2": h2},
        "deltas": {
            "base_to_base_plus_h1": _delta(base, h1),
            "base_to_base_plus_h2": _delta(base, h2),
            "why_the_delta_decides": (
                "Spec section 9, frozen attribution rule: rows modelling the "
                "L's INNER FACES may drift into the concavity cells and depress "
                "the base rule's precision on their own. The delta is immune to "
                "that because both terms carry the same base contamination; an "
                "absolute post-hull number is not. The verdict on hull "
                "completion is read from the delta."),
        },
        "gate": {
            "base": gate_verdict(base),
            "base_plus_h1": gate_verdict(h1),
            "base_plus_h2": gate_verdict(h2),
            "rejection_sentence": REJECTION_SENTENCE,
        },
        "false_activation_control": {
            "scope": "ALL groups on the trained cloud, not only the object's",
            "n_groups_total": int(n_groups),
            "groups_overlapping_event_object": int(sum(
                1 for g, cell in cell_of_group.items()
                if table.event_rows.get(cell, 0) > 0)),
            "base": {"gated_groups": base["cells_that_are_groups"],
                     "false_activations": base["false_activations"],
                     "false_activation_groups": base["false_activation_groups"],
                     "abstention_rate": 1.0 - base["cells_that_are_groups"]
                     / float(n_groups)},
            "base_plus_h1": {"gated_groups": h1["cells_that_are_groups"],
                             "false_activations": h1["false_activations"],
                             "false_activation_groups": h1["false_activation_groups"],
                             "abstention_rate": 1.0 - h1["cells_that_are_groups"]
                             / float(n_groups)},
            "base_plus_h2": {"gated_groups": h2["cells_that_are_groups"],
                             "false_activations": h2["false_activations"],
                             "false_activation_groups": h2["false_activation_groups"],
                             "abstention_rate": 1.0 - h2["cells_that_are_groups"]
                             / float(n_groups)},
        },
        "preconditions": preconditions,
        "verdict": verdict,
        "forbidden_readings": [
            "that LRV5 establishes anything about membership RECALL",
            "that a SURVIVES verdict admits hull completion (spec section 11 "
            "requires orientation O2 first)",
            "that any LRV3 number is retracted",
            "that this fixture's absence transfers to real data",
        ],
    }
    Path(args.out_manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_manifest, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)

    print("")
    print("=== LRV5-NCX hull-completion falsifier (spec section 9) ===")
    print("scene / orientation    %s / %s"
          % (spec.get("scene_id"), spec.get("orientation")))
    print("trained cloud          %d rows | %d in the event object | %d in the filler"
          % (xyz.shape[0], table.total_event_rows, int(filler_mask.sum())))
    print("groups                 %d | T1 gated cells %d"
          % (n_groups, len(seeds)))
    print("")
    print("%-22s %6s %10s %11s %9s %8s %10s"
          % ("rule", "cells", "rows", "precision", "recall", "false", "0-obj"))
    _print_row("base (A2 + B)", base)
    _print_row("base + H1", h1)
    _print_row("base + H2", h2)
    print("")
    for label, key in (("base -> base+H1", "base_to_base_plus_h1"),
                       ("base -> base+H2", "base_to_base_plus_h2")):
        d = manifest["deltas"][key]
        print("DELTA %-16s cells %+d | rows %+d | precision %s | recall %s | "
              "false %+d | 0-obj cells %+d"
              % (label, d["cells"], d["rows_gated"], _fmt(d["precision"], "%+.4f"),
                 _fmt(d["recall"], "%+.4f"), d["false_activations"],
                 d["n_zero_object_cells"]))
    print("")
    print("THE DELTA IS THE VERDICT-BEARING QUANTITY, not the absolute "
          "post-hull number.")
    print(manifest["deltas"]["why_the_delta_decides"])
    print("")
    print("notch-filler rows gated  base %d | +H1 %d | +H2 %d"
          % (base["rows_in_notch_filler"], h1["rows_in_notch_filler"],
             h2["rows_in_notch_filler"]))
    print("")
    for name in sorted(preconditions):
        block = preconditions[name]
        print("  [%s] %s" % ("PASS" if block["passed"] else "FAIL", name))
    print("")
    print("VERDICT: %s" % verdict["verdict"])
    print("  clause: %s" % verdict["clause"])
    if verdict["verdict"] == "REJECTED":
        print("  %s" % REJECTION_SENTENCE)
        if verdict["zero_object_cells_added_by_h1"] <= 0:
            print("  CAVEAT: H1 added no zero-object cell of its own; the base "
                  "rule already gated %d. Read the delta before citing this."
                  % verdict["base_already_gated_zero_object_cells"])
    else:
        print("  %s" % verdict["note"])
    print("")
    print("manifest written: %s" % args.out_manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
