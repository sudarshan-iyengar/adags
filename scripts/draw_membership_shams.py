#!/usr/bin/env python
"""Draw the three wave-2 wrong-membership sham row sets (spec v2.0.0 11.2).

Three shams, all count-matched to the construction-derived ("truth") row set
and all with ZERO overlap with it:

    GWRONGMEM_A  a uniform draw over the rows OUTSIDE the truth set,
                 seed 1000 * draw_index + prefix_seed
    GWRONGMEM_B  a second independent uniform draw,
                 seed 1000 * (draw_index + 1) + prefix_seed
    GWRONGMEM_L  the LOCAL contribution-matched sham: rows outside the truth
                 set that project inside the dilated (+20 px) construction
                 silhouette on >= 8 of 19 training cameras at frames 50 and
                 95, taken greedily by descending rendered contribution until
                 the set's total contribution mass at frame 50 on cam15 is
                 within +- `--mass_tolerance` of the truth set's.

A and B answer "does deleting ANY 1,700 rows buy the same thing"; L answers
the sharper question the wave-1 random sham could not, "does deleting 1,700
rows that paint the SAME PLACE buy the same thing". L is the one that can
refuse: if the eligible local rows cannot carry enough contribution mass to
reach the band, there is no contribution-matched local set and the script
exits non-zero rather than emitting an unmatched one.

INPUTS, and why each is taken rather than recomputed here:

``--truth_program``   the construction-derived program
                      (`adags-episode-program-v2`, `membership_mode:
                      row_ids`) that `scripts/realdata_membership_vote.py
                      --emit_program_rows` wrote for this prefix. It supplies
                      the truth row set, the cloud's row count and xyz
                      sha256, the gap, and every field the sham programs
                      inherit unchanged. The shams differ from it in
                      `row_group_ids` (and the bookkeeping under `source`)
                      ALONE, exactly as the wave-1 sham did.
``--contribution``    per-row rendered contribution at frame 50 on cam15,
                      from `realdata_membership_vote.py --emit_row_weights`
                      (an .npz carrying `w_in`; a .npy array or a JSON list
                      are also accepted). Re-deriving it here would mean a
                      second, unvalidated implementation of the rasterizer
                      leaf binding.
``--local_eligible``  a precomputed boolean per row: True where the row's
                      deformed centre projects inside the +20 px
                      construction silhouette on >= 8 of 19 cameras at frames
                      50 AND 95. Taken as a boolean column from the vote or
                      preview tooling rather than reprojected here, for the
                      same reason.
``--gap A B``         the arm's absent-frame window, checked against the
                      truth program's own `offset_frame`/`onset_frame`. The
                      shams carry the truth program's gap seconds unchanged,
                      so this is an assertion, never a conversion.

OUTPUTS, per draw, in ``--out_dir``: the program json and a counts sidecar
carrying `truth_n`, `draw_n`, `overlap_n` (asserted 0), `eligible_n`,
`contribution_truth`, `contribution_draw`, `seed` and the sha256 of the drawn
row-id list. `scripts/realdata_gate_analysis.py` reads `truth_n`, `draw_n`
and `overlap_n` from these sidecars for the zero-overlap assertion.

CPU only; numpy only. torch is never imported.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys

import numpy as np

DRAW_A = "GWRONGMEM_A"
DRAW_B = "GWRONGMEM_B"
DRAW_L = "GWRONGMEM_L"
DRAWS = (DRAW_A, DRAW_B, DRAW_L)

#: seed = SEED_STRIDE * draw index + prefix seed (spec v2.0.0 section 11.2)
SEED_STRIDE = 1000
DEFAULT_DRAW_INDEX = 1
DEFAULT_MASS_TOLERANCE = 0.10
PROGRAM_SCHEMA = "adags-episode-program-v2"
CONTRIBUTION_KEYS = ("w_in", "contribution", "mass")
ELIGIBLE_KEYS = ("eligible", "local_eligible", "mask")


class DrawRefused(Exception):
    """The draw cannot be made as specified; nothing is written."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_program(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != PROGRAM_SCHEMA:
        raise DrawRefused(
            "%s: schema_version is %r, expected %r"
            % (path, payload.get("schema_version"), PROGRAM_SCHEMA))
    if payload.get("membership_mode") != "row_ids":
        raise DrawRefused(
            "%s: membership_mode is %r; a sham draw can only replace an "
            "explicit row column" % (path, payload.get("membership_mode")))
    rows = payload.get("row_group_ids")
    if not isinstance(rows, list) or not rows:
        raise DrawRefused("%s: row_group_ids is missing or empty" % path)
    n_rows = int((payload.get("cloud") or {}).get("n_rows", len(rows)))
    if len(rows) != n_rows:
        raise DrawRefused(
            "%s: row_group_ids has %d entries but cloud.n_rows is %d"
            % (path, len(rows), n_rows))
    groups = payload.get("groups") or []
    if len(groups) != 1:
        raise DrawRefused(
            "%s: expected exactly one gated group, got %d" % (path, len(groups)))
    return payload


def _array_from_file(path, keys, what):
    """A 1-D array from an .npz (first of `keys` present), .npy or json."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        with np.load(path, allow_pickle=False) as data:
            for key in keys:
                if key in data:
                    return np.asarray(data[key]).reshape(-1)
            raise DrawRefused(
                "%s: none of %s is in the archive (found %s)"
                % (path, list(keys), sorted(data.files)))
    if ext == ".npy":
        return np.asarray(np.load(path, allow_pickle=False)).reshape(-1)
    if ext == ".json":
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            for key in keys:
                if key in payload:
                    return np.asarray(payload[key]).reshape(-1)
            raise DrawRefused(
                "%s: none of %s is in the object (found %s)"
                % (path, list(keys), sorted(payload)))
        return np.asarray(payload).reshape(-1)
    raise DrawRefused("%s: cannot read a %s from a %r file" % (path, what, ext))


def load_contribution(path, n_rows):
    values = _array_from_file(path, CONTRIBUTION_KEYS, "contribution column")
    if values.size != n_rows:
        raise DrawRefused(
            "%s: contribution has %d entries, the cloud has %d rows"
            % (path, values.size, n_rows))
    values = values.astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise DrawRefused("%s: the contribution column is not all finite" % path)
    if values.min() < 0.0:
        raise DrawRefused(
            "%s: the contribution column has a negative entry (min %.6g); a "
            "compositing weight cannot be negative" % (path, values.min()))
    return values


def load_local_eligible(path, n_rows):
    values = _array_from_file(path, ELIGIBLE_KEYS, "boolean eligibility column")
    if values.size != n_rows:
        raise DrawRefused(
            "%s: eligibility has %d entries, the cloud has %d rows"
            % (path, values.size, n_rows))
    return values.astype(bool)


def truth_mask(program):
    """The boolean truth membership column of a row_ids program."""
    return np.asarray(program["row_group_ids"], dtype=np.int64) >= 0


def check_gap(program, gap):
    """The truth program's own gap is the one the shams inherit."""
    if gap is None:
        return
    group = program["groups"][0]
    # The estimator's `offset_frame` is the first ABSENT frame and its
    # `onset_frame` the first PRESENT frame after the gap, so the absent
    # window `--gap A B` (B = the LAST absent frame, spec v1.2.0 anchor
    # semantics) is [offset_frame, onset_frame - 1].
    declared = [int(group.get("offset_frame", -1)),
                int(group.get("onset_frame", 0)) - 1]
    if declared != [int(gap[0]), int(gap[1])]:
        raise DrawRefused(
            "the truth program's gap is %r but --gap says %r; the shams carry "
            "the truth program's gap unchanged, so this mismatch means the "
            "wrong program was handed in" % (declared, [int(gap[0]), int(gap[1])]))


# ---------------------------------------------------------------------------
# The draws
# ---------------------------------------------------------------------------


def uniform_draw(truth, seed):
    """`truth.sum()` rows drawn uniformly from the rows OUTSIDE `truth`."""
    outside = np.flatnonzero(~truth)
    want = int(truth.sum())
    if outside.size < want:
        raise DrawRefused(
            "a count-matched draw needs %d rows outside the truth set, only "
            "%d exist" % (want, outside.size))
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(outside, size=want, replace=False)
    return np.sort(chosen), outside.size


def local_contribution_matched_draw(truth, eligible, contribution,
                                    tolerance=DEFAULT_MASS_TOLERANCE):
    """Greedy by descending contribution over eligible non-truth rows.

    Rows are taken in descending contribution while they keep the running
    mass at or below `(1 + tolerance) * target`, and the walk stops as soon
    as the mass reaches `(1 - tolerance) * target`. A row that would overshoot
    the upper edge is SKIPPED rather than ending the walk, so a single heavy
    row cannot strand the draw below the band while lighter rows remain. If
    the walk ends below the lower edge there is no matched local set and the
    draw is refused.

    Ties are broken by ascending row index, so the draw is a function of the
    inputs alone.
    """
    target = float(contribution[truth].sum())
    if target <= 0.0:
        raise DrawRefused(
            "the truth set carries no contribution mass at the recorded view; "
            "there is nothing to match")
    pool = np.flatnonzero(eligible & ~truth)
    if pool.size == 0:
        raise DrawRefused(
            "no row is both locally eligible and outside the truth set")
    order = pool[np.lexsort((pool, -contribution[pool]))]
    lower = (1.0 - float(tolerance)) * target
    upper = (1.0 + float(tolerance)) * target
    taken, mass = [], 0.0
    for row in order:
        value = float(contribution[row])
        if mass + value > upper:
            continue
        taken.append(int(row))
        mass += value
        if mass >= lower:
            break
    if mass < lower:
        raise DrawRefused(
            "the eligible local rows carry %.6g of contribution mass against "
            "the truth set's %.6g; the +-%.0f%% band [%.6g, %.6g] is "
            "unreachable, so no contribution-matched local sham exists"
            % (float(contribution[pool].sum()), target, 100.0 * tolerance,
               lower, upper))
    return np.sort(np.asarray(taken, dtype=np.int64)), pool.size


def load_xyz(path, n_rows):
    """[n_rows, 3] canonical positions from a .npy or an .npz (key `xyz`)."""
    xyz = _array_from_file(path, ("xyz", "_xyz", "positions"), "xyz")
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[0] != n_rows or xyz.shape[1] != 3:
        raise DrawRefused(
            "xyz has shape %r; the program has %d rows and needs [%d, 3]"
            % (xyz.shape, n_rows, n_rows))
    if not np.all(np.isfinite(xyz)):
        raise DrawRefused("xyz carries non-finite values")
    return xyz


def radius_contribution_matched_draw(truth, xyz, contribution,
                                     tolerance=DEFAULT_MASS_TOLERANCE):
    """Spec v2.0.0 section 13.18: mass-matched, locality relaxed by radius.

    Non-truth rows are ranked by 3D distance from the truth set's centroid.
    The radius R is the smallest distance at which the non-truth rows within
    R carry at least `(1 - tolerance) * target` contribution mass; within R
    the rows are taken by descending contribution (ties by row index),
    skipping a row that would overshoot `(1 + tolerance) * target`, stopping
    once the lower edge is reached. Deterministic; no seed.

    Returns (rows, info) with info = {radius, n_within_radius, centroid}.
    """
    target = float(contribution[truth].sum())
    if target <= 0.0:
        raise DrawRefused(
            "the truth set carries no contribution mass at the recorded view; "
            "there is nothing to match")
    lower = (1.0 - float(tolerance)) * target
    upper = (1.0 + float(tolerance)) * target
    centroid = xyz[truth].mean(axis=0)
    pool = np.flatnonzero(~truth)
    if pool.size == 0:
        raise DrawRefused("every row is in the truth set; no sham is possible")
    dist = np.linalg.norm(xyz[pool] - centroid[None, :], axis=1)
    by_dist = np.lexsort((pool, dist))
    cum = np.cumsum(np.maximum(contribution[pool[by_dist]], 0.0))
    if cum[-1] < lower:
        raise DrawRefused(
            "all non-truth rows together carry %.6g of contribution mass "
            "against the truth set's %.6g; the +-%.0f%% band [%.6g, %.6g] is "
            "unreachable at any radius" % (float(cum[-1]), target,
                                            100.0 * tolerance, lower, upper))
    k = int(np.searchsorted(cum, lower, side="left"))
    radius = float(dist[by_dist[k]])
    # every non-truth row at distance <= radius (ties at the boundary included)
    within = pool[dist <= radius]
    order = within[np.lexsort((within, -contribution[within]))]
    taken, mass = [], 0.0
    for row in order:
        value = float(contribution[row])
        if mass + value > upper:
            continue
        taken.append(int(row))
        mass += value
        if mass >= lower:
            break
    if mass < lower:  # pragma: no cover - the prefix sum guarantees reach
        raise DrawRefused("the greedy walk within radius %.6g ended at %.6g "
                          "below the lower edge %.6g" % (radius, mass, lower))
    info = {
        "radius": radius,
        "n_within_radius": int(within.size),
        "centroid": [float(v) for v in centroid],
    }
    return np.sort(np.asarray(taken, dtype=np.int64)), info


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def _row_ids_sha256(row_ids):
    blob = json.dumps([int(v) for v in row_ids], separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_sham_program(truth_program, rows, arm, seed, extra=None):
    """The truth program with its row column replaced. Nothing else moves."""
    payload = copy.deepcopy(truth_program)
    group_id = int(payload["groups"][0]["group"])
    column = np.full(len(payload["row_group_ids"]), -1, dtype=np.int64)
    column[rows] = group_id
    payload["row_group_ids"] = [int(v) for v in column]
    payload["groups"][0]["rows_at_estimation"] = int(len(rows))
    source = payload.setdefault("source", {})
    source["wrongmem"] = dict(
        {"arm": arm, "kind": "count_matched_zero_overlap_draw", "seed": int(seed),
         "count": int(len(rows)),
         "drawn_by": "scripts/draw_membership_shams.py"},
        **(extra or {}))
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return payload, hashlib.sha256(blob).hexdigest()


def counts_sidecar(arm, truth, rows, eligible_n, contribution, seed,
                   program_sha256, extra=None):
    truth_rows = np.flatnonzero(truth)
    overlap = int(np.intersect1d(truth_rows, rows, assume_unique=True).size)
    payload = {
        "arm": arm,
        "truth_n": int(truth.sum()),
        "draw_n": int(len(rows)),
        "overlap_n": overlap,
        "eligible_n": int(eligible_n),
        "contribution_truth": float(contribution[truth].sum()),
        "contribution_draw": float(contribution[rows].sum()),
        "seed": int(seed),
        "row_ids_sha256": _row_ids_sha256(rows),
        "program_sha256": program_sha256,
    }
    payload.update(extra or {})
    return payload


def draw_all(truth_program, contribution, eligible, prefix_seed,
             draw_index=DEFAULT_DRAW_INDEX,
             tolerance=DEFAULT_MASS_TOLERANCE, xyz=None, l_mode="local"):
    """{arm: (program, program_sha256, counts)} for the three shams.

    `l_mode` is "local" (section 11.2 as first frozen: locally eligible rows,
    which cannot reach the band on the real prefixes) or "radius" (section
    13.18: mass matching kept, locality relaxed by 3D radius from the truth
    centroid; needs `xyz`). In radius mode the local-only mass ratio is still
    computed from `eligible` and recorded as a finding.
    """
    truth = truth_mask(truth_program)
    if l_mode not in ("local", "radius"):
        raise DrawRefused("unknown l_mode %r" % (l_mode,))
    if l_mode == "radius" and xyz is None:
        raise DrawRefused("l_mode radius needs the cloud's xyz")
    seeds = {
        DRAW_A: SEED_STRIDE * int(draw_index) + int(prefix_seed),
        DRAW_B: SEED_STRIDE * (int(draw_index) + 1) + int(prefix_seed),
        DRAW_L: SEED_STRIDE * int(draw_index) + int(prefix_seed),
    }
    out = {}
    for arm in (DRAW_A, DRAW_B):
        rows, eligible_n = uniform_draw(truth, seeds[arm])
        extra = {"draw": "uniform_outside_truth"}
        program, digest = build_sham_program(
            truth_program, rows, arm, seeds[arm], extra)
        out[arm] = (program, digest, counts_sidecar(
            arm, truth, rows, eligible_n, contribution, seeds[arm], digest,
            extra))
    if l_mode == "radius":
        local_pool = np.flatnonzero(eligible & ~truth)
        local_only_ratio = (float(contribution[local_pool].sum())
                            / float(contribution[truth].sum()))
        rows, info = radius_contribution_matched_draw(
            truth, xyz, contribution, tolerance)
        eligible_n = info["n_within_radius"]
        extra = {
            "draw": "greedy_descending_contribution_within_radius",
            "mass_tolerance": float(tolerance),
            "radius_reached": info["radius"],
            "n_within_radius": info["n_within_radius"],
            "truth_centroid_xyz": info["centroid"],
            "local_only_mass_ratio_retired_rule": local_only_ratio,
            "local_only_eligible_n": int(local_pool.size),
        }
    else:
        rows, eligible_n = local_contribution_matched_draw(
            truth, eligible, contribution, tolerance)
        extra = {
            "draw": "greedy_descending_contribution_local",
            "mass_tolerance": float(tolerance),
        }
    program, digest = build_sham_program(
        truth_program, rows, DRAW_L, seeds[DRAW_L], extra)
    out[DRAW_L] = (program, digest, counts_sidecar(
        DRAW_L, truth, rows, eligible_n, contribution, seeds[DRAW_L], digest,
        dict(extra, mass_ratio_draw_over_truth=(
            float(contribution[rows].sum())
            / float(contribution[truth].sum())))))
    for arm, (_, _, counts) in out.items():
        if counts["overlap_n"] != 0:
            raise DrawRefused(
                "%s: the draw overlaps the truth set in %d rows; spec v2.0.0 "
                "requires zero" % (arm, counts["overlap_n"]))
        if arm in (DRAW_A, DRAW_B) and counts["draw_n"] != counts["truth_n"]:
            raise DrawRefused(
                "%s: drew %d rows against a truth set of %d"
                % (arm, counts["draw_n"], counts["truth_n"]))
    return out


def write_draws(out_dir, draws):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for arm, (program, _, counts) in sorted(draws.items()):
        stem = arm.lower()
        program_path = os.path.join(out_dir, "program_%s.json" % stem)
        counts_path = os.path.join(out_dir, "counts_%s.json" % stem)
        with open(program_path, "w", encoding="utf-8") as handle:
            json.dump(program, handle, indent=1, sort_keys=True)
        with open(counts_path, "w", encoding="utf-8") as handle:
            json.dump(counts, handle, indent=1, sort_keys=True)
        written.append((arm, program_path, counts_path))
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--truth_program", required=True,
                        help="the construction-derived row_ids program")
    parser.add_argument("--contribution", required=True,
                        help="per-row contribution at frame 50 on cam15 "
                             "(realdata_membership_vote.py --emit_row_weights)")
    parser.add_argument("--local_eligible", required=True,
                        help="boolean per row: projects inside the +20 px "
                             "construction silhouette on >= 8 of 19 cameras "
                             "at frames 50 and 95")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prefix_seed", type=int, required=True)
    parser.add_argument("--draw_index", type=int, default=DEFAULT_DRAW_INDEX,
                        help="A takes seed %d*index+prefix_seed, B takes "
                             "%d*(index+1)+prefix_seed (default index %d, so "
                             "1000+seed and 2000+seed)"
                             % (SEED_STRIDE, SEED_STRIDE, DEFAULT_DRAW_INDEX))
    parser.add_argument("--gap", nargs=2, type=int, default=None,
                        metavar=("A", "B"),
                        help="the arm's absent-frame window, checked against "
                             "the truth program's own gap")
    parser.add_argument("--mass_tolerance", type=float,
                        default=DEFAULT_MASS_TOLERANCE,
                        help="the L draw's contribution band (default %.2f)"
                             % DEFAULT_MASS_TOLERANCE)
    parser.add_argument("--l_mode", choices=("local", "radius"),
                        default="local",
                        help="L draw rule: 'local' = section 11.2 as first "
                             "frozen (eligible local rows; refuses when they "
                             "cannot reach the band); 'radius' = section "
                             "13.18 (mass matched, locality relaxed by 3D "
                             "radius from the truth centroid; needs --xyz)")
    parser.add_argument("--xyz", default=None,
                        help="[n_rows, 3] canonical xyz (.npy, or .npz with "
                             "key xyz); required for --l_mode radius")
    args = parser.parse_args(argv)

    try:
        program = load_program(args.truth_program)
        check_gap(program, args.gap)
        n_rows = len(program["row_group_ids"])
        contribution = load_contribution(args.contribution, n_rows)
        eligible = load_local_eligible(args.local_eligible, n_rows)
        xyz = load_xyz(args.xyz, n_rows) if args.xyz else None
        draws = draw_all(program, contribution, eligible, args.prefix_seed,
                         draw_index=args.draw_index,
                         tolerance=args.mass_tolerance,
                         xyz=xyz, l_mode=args.l_mode)
    except DrawRefused as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 2
    for arm, program_path, counts_path in write_draws(args.out_dir, draws):
        counts = draws[arm][2]
        print("%s rows %d (truth %d, overlap %d, eligible %d, mass %.6g vs "
              "%.6g, seed %d) -> %s, %s"
              % (arm, counts["draw_n"], counts["truth_n"], counts["overlap_n"],
                 counts["eligible_n"], counts["contribution_draw"],
                 counts["contribution_truth"], counts["seed"],
                 program_path, counts_path))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
