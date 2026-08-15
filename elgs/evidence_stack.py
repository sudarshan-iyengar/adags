"""Evidence-path assembly: sealed tracks -> clusters -> windows -> Phi.

This is the GLUE that was disclosed as M1-gated in
`research-wiki/operations/elgs-m0-implementation-record.md` ("the evidence
stack (clusters/bridges/observability/energy/SlotGrid full path) is
implemented + unit-tested but unwired pending the M1 track artifacts").
Every scientific primitive it calls already exists and is unchanged:
`elgs.clusters`, `elgs.bridges`, `elgs.observability`, `elgs.evidence`,
`elgs.energy`. Nothing here re-derives them.

What it adds is exactly the composition:

  sealed artifact (elgs.tracks_loader)
    -> per-seed r_u / d_u / n_cam            (the artifact's own frozen
                                              r_u mapping + §2 d_u formula)
    -> seed-overlap clusters                 (elgs.clusters.form_clusters)
    -> bind(u) = nearest family              (elgs.clusters.nearest_family)
    -> per-family anchors and windows        (elgs.bridges)
    -> evidence-independent bridge family    (below; stop-gradient)
    -> q_tilde round snapshot                (elgs.observability + a probe)
    -> EvidenceBatch -> Phi                  (elgs.energy)
    -> beta * (Phi_candidate - Phi_incumbent) as the acceptance
       `exact_deltas` term                   (elgs.acceptance.decide)

WHERE THE EVIDENCE ENTERS. `elgs.acceptance`'s module docstring fixes
this: "'Exact' survives only for the non-sampled closed-form
tracker/prior deltas added outside the sampled render estimate". Phi is
exactly such a closed-form term, so it enters through
`run_pass(..., exact_deltas_fn=...)` and NOT through the SNIS render
samples. The photometric path is untouched.

HEADS ARE GATED, NOT ASSUMED. `configs/elgs/prereg_evidence_heads_v1.json`
still carries `"status": "unfrozen"` for g_v, h_c, h_o, pi_miss and
g_pos_sigma (freeze point: "before the first evidence-bearing job (B2)"),
and `prereg_observability_v1.json` for `alpha_u.rho`. `load_evidence_heads`
therefore REFUSES to build heads from the frozen prereg while any required
entry is unfrozen, and accepts the declared smoke-tier constants ONLY for
a run that has set `elgs_smoke_schedule` (which the submission wrapper
stamps `evidence_bearing: false`). An evidence-bearing run cannot reach
the evidence path until the heads are actually frozen.

DISCLOSED READINGS (recorded here, not silently applied):
  D1. `d_u` is the §2 "cluster's visible-report rate in its anchor
      intervals". Anchors are per-FAMILY but `form_clusters` needs a
      per-SEED d before binding exists, so d is computed per seed over
      that seed's OWN plateau frames (the frames where it carries at
      least `min_occupancy_cameras` visible reports) and the cluster
      takes the min over members -- the same conservative rule §2
      prescribes for merged clusters.
  D2. The smoke-tier bridge family is the three-hypothesis
      {hold-left, linear, hold-right} interpolation between the two
      bounding anchors' consensus points. It reads committed anchor
      state ONLY, never the reports it will score, satisfying the §3
      evidence-independence requirement. It is declared smoke-tier: a
      claim-grade constructor is a category-2 preregistration item.
  D3. A family with fewer than two anchors has no evidence windows and
      contributes Phi = 0.0 exactly (`elgs.bridges` returns no windows;
      the family is reported photometric_only in coverage). This is the
      INSUFFICIENT-EVIDENCE FALLBACK: it degrades to the verified M0
      behaviour for that family rather than refusing the round.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch

from depth_visibility.errors import ContractError, SchemaError

from .bridges import (
    AnchorInterval,
    Window,
    coverage_entry,
    find_anchor_intervals,
    validate_bridge_ids,
    windows_between_anchors,
)
from .clusters import BindingTable, SeedCluster, form_clusters, nearest_family
from .energy import (
    EvidenceBatch,
    Segment,
    phi,
    stream_log_likelihoods,
)
from .evidence import EvidenceHeads, EvidenceParams, Report
from .observability import (
    QSnapshot,
    SigmaPoint,
    compute_q,
    compute_q_batch,
    q_tilde,
)
from .tracks_loader import SealedTracks

SMOKE_HEADS_FILENAME = "smoke_evidence_heads_v1.json"
HEADS_PREREG_FILENAME = "prereg_evidence_heads_v1.json"
SMOKE_HEADS_SCHEMA = "elgs-smoke-evidence-heads-v1"

#: The frozen §3 sigma-point scheme: centre (0.4) + 6 axis offsets (0.1).
SIGMA_CENTER_WEIGHT = 0.4
SIGMA_AXIS_WEIGHT = 0.1


# ---------------------------------------------------------------------------
# Head loading (the unfrozen-prereg gate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceConstants:
    """Head bundle plus the §3/§4 constants the assembly needs."""

    heads: EvidenceHeads
    rho: float
    anchor_report_floor: int
    min_occupancy_cameras: int
    plateau_seed_fraction: float
    vis_threshold: float
    sigma_world: float
    tier: str

    def alpha_model(self, n_cam: int) -> float:
        """alpha_u = n_cam^(-rho) (prereg_observability alpha_u.family)."""
        if n_cam < 1:
            raise ContractError("n_cam must be >= 1")
        return float(n_cam) ** (-self.rho)


def _beta_density(a: float, b: float, lo: float, hi: float):
    """The frozen g_v family: beta(a,b) on [v_min, v_max], a >= 1."""
    if a < 1.0:
        raise ContractError("g_v beta density requires a >= 1 (monotone constraint)")
    if hi <= lo:
        raise ContractError("g_v support must be non-empty")
    log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    span = hi - lo

    def density(v: float) -> float:
        u = (float(v) - lo) / span
        if u <= 0.0 or u >= 1.0:
            # Endpoint mass is not representable in a density; the §2
            # floors/caps are constraints, so return the limiting value
            # rather than clamping to zero (which would violate the cap
            # check's "floors are constraints" ruling).
            u = min(max(u, 1e-12), 1.0 - 1e-12)
        return math.exp(log_norm + (a - 1.0) * math.log(u) + (b - 1.0) * math.log1p(-u)) / span

    return density


def _uniform_density(lo: float, hi: float, floor: float):
    value = 1.0 / (hi - lo)
    if value < floor:
        raise ContractError(
            f"uniform value head density {value} is below its floor {floor}"
        )

    def density(_v: float) -> float:
        return value

    return density


def load_evidence_constants(prereg_dir: str, *, smoke: bool) -> EvidenceConstants:
    """Build the head bundle, or refuse.

    ``smoke=False`` reads the FROZEN prereg and raises if any required
    entry is still ``"status": "unfrozen"`` -- the gate that keeps an
    evidence-bearing run out of the evidence path before B2.
    ``smoke=True`` reads the declared smoke-tier constants instead.
    """

    if not smoke:
        path = os.path.join(prereg_dir, HEADS_PREREG_FILENAME)
        if not os.path.isfile(path):
            raise ContractError(f"evidence-heads prereg missing: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        unfrozen = sorted(
            name
            for name, entry in (payload.get("head_families") or {}).items()
            if isinstance(entry, Mapping) and entry.get("status") == "unfrozen"
        )
        reliability = payload.get("reliability") or {}
        for name, entry in reliability.items():
            if isinstance(entry, Mapping) and entry.get("status") == "unfrozen":
                unfrozen.append(f"reliability.{name}")
        if unfrozen:
            raise ContractError(
                "the evidence heads are not frozen "
                f"({sorted(unfrozen)}); prereg_evidence_heads_v1 fixes their "
                "freeze point at 'before the first evidence-bearing job "
                "(B2)'. An evidence-bearing run may not consume the evidence "
                "path until they are frozen; use elgs_smoke_schedule for a "
                "declared non-evidence-bearing smoke."
            )
        frozen_values = payload.get("frozen_values")
        if not isinstance(frozen_values, Mapping):
            raise ContractError(
                "prereg_evidence_heads_v1 reports every head frozen but "
                "carries no 'frozen_values' block with their numeric "
                "materialization; refusing to invent head parameters"
            )
        return _constants_from_values(frozen_values, tier="frozen")

    path = os.path.join(prereg_dir, SMOKE_HEADS_FILENAME)
    if not os.path.isfile(path):
        raise ContractError(f"smoke evidence-heads constants missing: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != SMOKE_HEADS_SCHEMA:
        raise SchemaError(
            f"unsupported smoke heads schema: {payload.get('schema_version')!r}"
        )
    if payload.get("tier") != "smoke" or payload.get("evidence_bearing") is not False:
        raise ContractError(
            "smoke heads file must declare tier='smoke' and "
            "evidence_bearing=false"
        )
    return _constants_from_values(payload, tier="smoke")


def _constants_from_values(payload: Mapping, *, tier: str) -> EvidenceConstants:
    """Materialize the head bundle from an explicit numeric block."""
    params = EvidenceParams(**payload["params"])
    gv = payload["g_v"]
    heads = EvidenceHeads(
        params=params,
        g_v=_beta_density(float(gv["a"]), float(gv["b"]), params.v_min, params.v_max),
        h_c=_uniform_density(params.v_min, params.v_max, params.h_floor),
        h_o=_uniform_density(params.v_min, params.v_max, params.h_floor),
    )
    return EvidenceConstants(
        heads=heads,
        rho=float(payload["rho"]),
        anchor_report_floor=int(payload["anchor_report_floor"]),
        min_occupancy_cameras=int(payload["min_occupancy_cameras"]),
        plateau_seed_fraction=float(payload["plateau_seed_fraction"]),
        vis_threshold=float(payload["vis_threshold"]),
        sigma_world=float(payload["sigma_world"]),
        tier=tier,
    )


# ---------------------------------------------------------------------------
# Per-seed derivation from the sealed artifact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedEvidence:
    """Everything one seed contributes before clustering."""

    seed_id: int
    point: tuple[float, float, float]
    n_cam: int
    r_u: float
    d_u: float


def _clip(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def derive_seed_evidence(
    sealed: SealedTracks, *, r_min: float, vis_threshold: float, min_cameras: int
) -> tuple[SeedEvidence, ...]:
    """Per-seed r_u and d_u from the artifact's OWN frozen diagnostics.

    r_u follows the artifact manifest's declared mapping verbatim
    (`build_elgs_tracks.R_U_MAPPING`): clip(min(1 - fb_rms/8,
    1 - reproj_rms/8), r_min, 1); a missing diagnostic yields r_min.
    d_u is disclosed reading D1 (module docstring).
    """

    diagnostics = sealed.artifact.get("diagnostics") or {}
    fb = diagnostics.get("fb_rms_px") or {}
    reproj = diagnostics.get("reproj_rms_px") or {}

    # Per (seed, frame) visible-camera counts, for the D1 plateau.
    visible: dict[int, dict[float, int]] = {}
    queried: dict[int, set[int]] = {}
    for track in sealed.tracks:
        seed_id = int(track["seed_id"])
        queried.setdefault(seed_id, set()).add(int(track["camera_id"]))
        per_frame = visible.setdefault(seed_id, {})
        for report in track["reports"]:
            if report.get("is_miss", False):
                continue
            if float(report["v"]) >= vis_threshold:
                frame = float(report["frame"])
                per_frame[frame] = per_frame.get(frame, 0) + 1

    out: list[SeedEvidence] = []
    for seed in sealed.seeds:
        seed_id = int(seed["seed_id"])
        fb_values = [
            float(v) for key, v in fb.items() if key.split(":", 1)[0] == str(seed_id)
        ]
        fb_rms = max(fb_values) if fb_values else None
        reproj_rms = reproj.get(str(seed_id))
        candidates = []
        if fb_rms is not None:
            candidates.append(1.0 - fb_rms / 8.0)
        if reproj_rms is not None:
            candidates.append(1.0 - float(reproj_rms) / 8.0)
        r_u = _clip(min(candidates), r_min, 1.0) if candidates else r_min

        per_frame = visible.get(seed_id, {})
        plateau = [count for count in per_frame.values() if count >= min_cameras]
        n_queried = max(1, len(queried.get(seed_id, ())))
        d_u = _clip(
            (sum(plateau) / (len(plateau) * n_queried)) if plateau else 0.0, 0.0, 1.0
        )
        out.append(
            SeedEvidence(
                seed_id=seed_id,
                point=tuple(float(c) for c in seed["point"]),
                n_cam=int(seed["n_cam"]),
                r_u=r_u,
                d_u=d_u,
            )
        )
    return tuple(sorted(out, key=lambda s: s.seed_id))


def seed_overlap_edges(
    seeds: Sequence[SeedEvidence], radius: float
) -> tuple[tuple[int, int], ...]:
    """The seed-overlap graph: seeds within `radius` share a cluster.

    O(n^2) over at most `max_seeds` (512) seeds -- 131k distance
    evaluations, evaluated once per run at binding, not per round.
    """
    if radius <= 0.0:
        raise ContractError("seed overlap radius must be positive")
    edges: list[tuple[int, int]] = []
    for i, a in enumerate(seeds):
        for b in seeds[i + 1 :]:
            dx = a.point[0] - b.point[0]
            dy = a.point[1] - b.point[1]
            dz = a.point[2] - b.point[2]
            if dx * dx + dy * dy + dz * dz <= radius * radius:
                edges.append((a.seed_id, b.seed_id))
    return tuple(edges)


def build_clusters(
    seeds: Sequence[SeedEvidence], *, radius: float, constants: EvidenceConstants
) -> tuple[SeedCluster, ...]:
    """Connected components of the seed-overlap graph (elgs.clusters)."""
    if not seeds:
        return ()
    return form_clusters(
        {s.seed_id: s.point for s in seeds},
        seed_overlap_edges(seeds, radius),
        {s.seed_id: s.r_u for s in seeds},
        {s.seed_id: s.d_u for s in seeds},
        {s.seed_id: s.n_cam for s in seeds},
        constants.alpha_model,
    )


def bind_clusters(
    clusters: Sequence[SeedCluster],
    positions: torch.Tensor,
    family_ids: torch.Tensor,
    threshold: float,
) -> BindingTable:
    """bind(u) for every cluster; unassigned => inactive (spec §2)."""
    table = BindingTable()
    for cluster in clusters:
        table.bind(
            cluster.cluster_id,
            nearest_family(cluster.canonical_point, positions, family_ids, threshold),
        )
    return table


# ---------------------------------------------------------------------------
# Anchors and windows (per family)
# ---------------------------------------------------------------------------


def family_anchor_series(
    sealed: SealedTracks,
    track_ids: Sequence[int],
    *,
    c_cap: int,
    vis_threshold: float,
    min_cameras: int,
    seed_fraction: float,
) -> tuple[tuple[float, ...], tuple[bool, ...], tuple[int, ...]]:
    """(frames, plateau_series, capped_visible_counts) for one family.

    PLATEAU IS A FRACTION OF SEEDS, NOT A POOLED REPORT COUNT.

    A SEED is observed at t iff at least `min_cameras` cameras carry a
    visible report for it there — the census's own per-seed notion of
    presence. The family is on a plateau at t iff at least
    `plateau_seed_fraction` of its seeds are observed.

    An earlier revision pooled visible reports across ALL the family's
    tracks and tested that count against `min_cameras`. That predicate
    SATURATES with family size: measured on the real scissor artifact,
    one cluster held 512 seeds and 10,995 tracks, so at least two tracks
    were visible at every single frame, the plateau never broke, and the
    family produced ONE anchor and ZERO windows — Phi identically 0 on
    the richest absence sequence in the tranche (census: 343 windows).
    The synthetic fixture could not show this: three seeds that all went
    dark together satisfy either predicate.

    A fraction is scale-invariant and therefore cannot saturate.
    """
    frames = tuple(float(i) for i in sealed.frame_indices)
    wanted = {int(j) for j in track_ids}
    # (seed, frame) -> number of cameras with a visible report.
    per_seed: dict[int, dict[float, int]] = {}
    for track in sealed.tracks:
        if int(track["track_id"]) not in wanted:
            continue
        seed_id = int(track["seed_id"])
        by_frame = per_seed.setdefault(seed_id, {})
        for report in track["reports"]:
            if report.get("is_miss", False):
                continue
            if float(report["v"]) >= vis_threshold:
                frame = float(report["frame"])
                by_frame[frame] = by_frame.get(frame, 0) + 1

    n_seeds = len(per_seed)
    if n_seeds == 0:
        return frames, tuple(False for _ in frames), tuple(0 for _ in frames)
    needed = max(1, math.ceil(seed_fraction * n_seeds))

    plateau: list[bool] = []
    capped: list[int] = []
    for frame in frames:
        observed = 0
        reports = 0
        for by_frame in per_seed.values():
            cameras = by_frame.get(frame, 0)
            if cameras >= min_cameras:
                observed += 1
            # §4 cap operator: at most C_cap cameras per (seed, frame).
            reports += min(cameras, c_cap)
        plateau.append(observed >= needed)
        # `find_anchor_intervals` sums this over a run and compares it
        # with `report_floor`, and elgs.bridges specifies a capped
        # VISIBLE-REPORT count. Only the PLATEAU predicate becomes a seed
        # fraction; counting seeds here too would silently rescale the
        # floor by the cameras-per-seed factor (~21 on scissor) and, on
        # singleton clusters, put every run below any sane floor.
        capped.append(reports)
    return frames, tuple(plateau), tuple(capped)


def family_windows(
    sealed: SealedTracks,
    track_ids: Sequence[int],
    *,
    c_cap: int,
    constants: EvidenceConstants,
) -> tuple[tuple[AnchorInterval, ...], tuple[Window, ...]]:
    """Anchors and the windows between them for one family."""
    frames, plateau, capped = family_anchor_series(
        sealed,
        track_ids,
        c_cap=c_cap,
        vis_threshold=constants.vis_threshold,
        min_cameras=constants.min_occupancy_cameras,
        seed_fraction=constants.plateau_seed_fraction,
    )
    anchors = find_anchor_intervals(
        frames, plateau, capped, constants.anchor_report_floor
    )
    return anchors, windows_between_anchors(anchors)


# ---------------------------------------------------------------------------
# Bridges (disclosed reading D2)
# ---------------------------------------------------------------------------


def _consensus_point(
    sealed: SealedTracks, seed_id: int, frame: float
) -> tuple[float, float, float] | None:
    for entry in sealed.consensus.get(str(seed_id), ()):
        if float(entry["frame"]) == frame and entry.get("point") is not None:
            return tuple(float(c) for c in entry["point"])
    return None


def _nearest_consensus(
    sealed: SealedTracks, seed_id: int, frame: float
) -> tuple[float, float, float] | None:
    """The consensus point at `frame`, else the nearest frame that has
    one (anchor endpoints can fall on a frame with < 2 visible cameras)."""
    exact = _consensus_point(sealed, seed_id, frame)
    if exact is not None:
        return exact
    best = None
    best_gap = None
    for entry in sealed.consensus.get(str(seed_id), ()):
        if entry.get("point") is None:
            continue
        gap = abs(float(entry["frame"]) - frame)
        if best_gap is None or gap < best_gap:
            best_gap, best = gap, tuple(float(c) for c in entry["point"])
    return best


def build_bridge_family(
    sealed: SealedTracks,
    seed_id: int,
    window: Window,
) -> dict[int, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """B(W): {bridge_id: (p_start, p_end)} -- disclosed reading D2.

    Evidence-independent by construction: reads only the consensus
    points at the two bounding ANCHOR frames, never any report inside
    the window it will score. Returns an empty family when neither
    endpoint has a consensus point (the caller then treats the window
    as photometric-only).
    """
    left = _nearest_consensus(sealed, seed_id, window.start_frame)
    right = _nearest_consensus(sealed, seed_id, window.end_frame)
    if left is None and right is None:
        return {}
    if left is None:
        left = right
    if right is None:
        right = left
    family = {0: (left, left), 1: (left, right), 2: (right, right)}
    validate_bridge_ids(sorted(family))
    return family


def bridge_point(
    endpoints: tuple[tuple[float, float, float], tuple[float, float, float]],
    window: Window,
    frame: float,
) -> tuple[float, float, float]:
    """Constant-velocity position of one bridge hypothesis at `frame`."""
    span = window.end_frame - window.start_frame
    t = 0.0 if span <= 0.0 else (frame - window.start_frame) / span
    t = _clip(t, 0.0, 1.0)
    start, end = endpoints
    return tuple(start[i] + t * (end[i] - start[i]) for i in range(3))


def sigma_grid(
    point: tuple[float, float, float],
    sigma: float,
) -> tuple[tuple[tuple[float, float, float], float], ...]:
    """The frozen 7-point world grid: centre first, then +/- per axis.

    Pure geometry, no projection. Split out of `sigma_points_for` so a
    caller that batches projections can obtain the SAME world points in
    the SAME order and project them together; `sigma_points_for` is that
    function composed with a per-point projector, unchanged.
    """
    offsets = [(point, SIGMA_CENTER_WEIGHT)]
    for axis in range(3):
        for sign in (-1.0, 1.0):
            shifted = list(point)
            shifted[axis] += sign * sigma
            offsets.append((tuple(shifted), SIGMA_AXIS_WEIGHT))
    return tuple(offsets)


def sigma_points_from_projections(
    grid: Sequence[tuple[tuple[float, float, float], float]],
    projections: Sequence,
) -> tuple[SigmaPoint, ...]:
    """Assemble sigma points from an already-projected grid.

    `projections[i]` is `(pixel, depth)` or None for `grid[i]`. A point
    that does not project is dropped and its weight is redistributed
    over the survivors so the weights still sum to one
    (validate_sigma_points is a hard contract).
    """
    if len(grid) != len(projections):
        raise ContractError(
            f"sigma grid has {len(grid)} points but {len(projections)} projections"
        )
    survivors = []
    for (world, weight), projected in zip(grid, projections):
        if projected is None:
            continue
        pixel, depth = projected
        if depth <= 0.0:
            continue
        survivors.append((world, weight, pixel, depth))
    if not survivors:
        return ()
    total = sum(w for _, w, _, _ in survivors)
    return tuple(
        SigmaPoint(weight=w / total, point=world, pixel=pixel, depth=depth)
        for world, w, pixel, depth in survivors
    )


def sigma_points_for(
    point: tuple[float, float, float],
    sigma: float,
    project,
) -> tuple[SigmaPoint, ...]:
    """The frozen 7-point grid (prereg_observability sigma_points).

    `project(world_point) -> (pixel, depth) | None`; a point that does
    not project is dropped and its weight is redistributed over the
    survivors so the weights still sum to one (validate_sigma_points is
    a hard contract).
    """
    grid = sigma_grid(point, sigma)
    return sigma_points_from_projections(
        grid, [project(world) for world, _ in grid]
    )


# ---------------------------------------------------------------------------
# Round context: the object the trainer holds across a round
# ---------------------------------------------------------------------------


@dataclass
class EvidenceContext:
    """Bound evidence state for one run (persisted across checkpoints)."""

    sealed: SealedTracks
    constants: EvidenceConstants
    clusters: tuple[SeedCluster, ...]
    binding: BindingTable
    tracks_of_cluster: dict[int, tuple[int, ...]]
    seed_of_track: dict[int, int]
    c_cap: int
    beta: float
    tau_b: float
    coverage: dict = field(default_factory=dict)
    #: SMOKE-ONLY per-window evaluation bound; 0 = disabled (full-report
    #: semantics). Set only via resolve_smoke_report_cap.
    smoke_max_reports_per_window: int = 0
    _window_counts: dict = field(default_factory=dict)

    def window_count(self, family_id: int) -> int:
        """How many evidence windows family `family_id` has.

        STRUCTURAL AVAILABILITY ONLY. This counts windows; it never
        evaluates a likelihood, a q value, an evidence delta, or an
        acceptance outcome, so a caller may use it to choose WHERE to
        look without any result-bearing quantity entering the choice.

        Memoized: the count depends only on the sealed artifact, the
        frozen constants and the binding, none of which change while a
        run is training.
        """
        family_id = int(family_id)
        if family_id in self._window_counts:
            return self._window_counts[family_id]
        track_ids = self.track_ids_of_family(family_id)
        if not track_ids:
            self._window_counts[family_id] = 0
            return 0
        _, windows = family_windows(
            self.sealed, track_ids, c_cap=self.c_cap, constants=self.constants
        )
        self._window_counts[family_id] = len(windows)
        return len(windows)

    def families_with_windows(self, family_ids: Sequence[int]) -> tuple[int, ...]:
        """The subset of `family_ids` carrying at least one window."""
        return tuple(sorted(f for f in family_ids if self.window_count(f) > 0))

    def clusters_of_family(self, family_id: int) -> tuple[SeedCluster, ...]:
        wanted = set(self.binding.clusters_of(family_id))
        return tuple(c for c in self.clusters if c.cluster_id in wanted)

    def track_ids_of_family(self, family_id: int) -> tuple[int, ...]:
        ids: list[int] = []
        for cluster in self.clusters_of_family(family_id):
            ids.extend(self.tracks_of_cluster.get(cluster.cluster_id, ()))
        return tuple(sorted(ids))

    def to_state(self) -> dict:
        """Checkpoint payload.

        CORRECTED: an earlier revision said the binding table "already
        rides in the M0 `elgs_state.binding` slot". It does NOT — that
        slot holds `StateBundle.binding`, a DIFFERENT object the evidence
        path never writes. The evidence binding is its own table and is
        serialized here, so a resume restores the bindings the run
        actually used instead of re-deriving them against drifted
        primitive positions.
        """
        return {
            "tracks_provenance": self.sealed.provenance(),
            "tier": self.constants.tier,
            "binding": self.binding.to_state(),
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "seed_ids": list(c.seed_ids),
                    "canonical_point": list(c.canonical_point),
                    "r_u": c.r_u,
                    "d_u": c.d_u,
                    "n_cam": c.n_cam,
                    "alpha_u": c.alpha_u,
                }
                for c in self.clusters
            ],
            "tracks_of_cluster": {
                str(k): list(v) for k, v in sorted(self.tracks_of_cluster.items())
            },
            "coverage": dict(self.coverage),
        }


def build_evidence_context(
    sealed: SealedTracks,
    constants: EvidenceConstants,
    positions: torch.Tensor,
    family_ids: torch.Tensor,
    *,
    r_site: float,
    binding_threshold: float,
    c_cap: int,
    beta: float,
    tau_b: float,
    report_raster: tuple[float, float] | None = None,
) -> EvidenceContext:
    """One-shot binding pass: clusters, bind(u), and the track index.

    `report_raster` is the (width, height) the artifact's report
    positions live in. It is checked against the heads' declared D_img:
    `g_pos` compares a report position with a bridge projection, so a
    D_img in one raster and reports in another silently zeroes the
    visible head for every report and gives the evidence term a constant
    sign. Passing None skips the check and is only valid for a fixture.
    """
    if report_raster is not None:
        params = constants.heads.params
        width, height = float(report_raster[0]), float(report_raster[1])
        declared = (params.d_img_x1 - params.d_img_x0,
                    params.d_img_y1 - params.d_img_y0)
        if abs(declared[0] - width) > 1.0 or abs(declared[1] - height) > 1.0:
            raise ContractError(
                f"declared D_img {declared[0]:.0f}x{declared[1]:.0f} does not "
                f"match the report raster {width:.0f}x{height:.0f}; g_pos "
                "would compare positions in two different pixel domains"
            )
    seeds = derive_seed_evidence(
        sealed,
        r_min=constants.heads.params.r_min,
        vis_threshold=constants.vis_threshold,
        min_cameras=constants.min_occupancy_cameras,
    )
    clusters = build_clusters(seeds, radius=r_site, constants=constants)
    binding = bind_clusters(clusters, positions, family_ids, binding_threshold)

    seed_of_track = {int(t["track_id"]): int(t["seed_id"]) for t in sealed.tracks}
    tracks_by_seed: dict[int, list[int]] = {}
    for track_id, seed_id in seed_of_track.items():
        tracks_by_seed.setdefault(seed_id, []).append(track_id)
    tracks_of_cluster = {
        c.cluster_id: tuple(
            sorted(tid for sid in c.seed_ids for tid in tracks_by_seed.get(sid, ()))
        )
        for c in clusters
    }
    return EvidenceContext(
        sealed=sealed,
        constants=constants,
        clusters=clusters,
        binding=binding,
        tracks_of_cluster=tracks_of_cluster,
        seed_of_track=seed_of_track,
        c_cap=c_cap,
        beta=beta,
        tau_b=tau_b,
    )


# ---------------------------------------------------------------------------
# Phi for one family under one interval hypothesis
# ---------------------------------------------------------------------------


def _reports_in_window(
    sealed: SealedTracks,
    track_ids: Sequence[int],
    window: Window,
    params: EvidenceParams,
) -> dict[tuple[int, int, float], Report]:
    """Raw reports STRICTLY INSIDE the window, keyed (track, camera, frame).

    Out-of-domain positions never reach here: the builder already
    converted them to miss tokens (`build_elgs_tracks` step 5), which is
    the heads prereg's `off_domain_reports` ruling.

    ENDPOINTS ARE EXCLUDED. `windows_between_anchors` builds
    `Window(earlier.end_frame, later.start_frame)`, so both endpoints ARE
    anchor frames, and the bridge family for the window is constructed
    FROM the consensus points at exactly those frames. Including them
    would (a) score anchor-frame reports against a bridge fitted to them
    and (b) double-count every frame shared by two windows whenever an
    anchor is a single frame -- in which case `earlier.end_frame ==
    later.start_frame` and consecutive windows touch. A window is the
    span BETWEEN anchors; its interior is what the evidence is about.
    """
    wanted = {int(j) for j in track_ids}
    out: dict[tuple[int, int, float], Report] = {}
    for track in sealed.tracks:
        track_id = int(track["track_id"])
        if track_id not in wanted:
            continue
        camera_id = int(track["camera_id"])
        for report in track["reports"]:
            frame = float(report["frame"])
            if not (window.start_frame < frame < window.end_frame):
                continue
            if report.get("is_miss", False):
                out[(track_id, camera_id, frame)] = Report(is_miss=True)
            else:
                out[(track_id, camera_id, frame)] = Report(
                    is_miss=False,
                    v=_clip(float(report["v"]), params.v_min, params.v_max),
                    x=_clip(float(report["x"]), params.d_img_x0, params.d_img_x1),
                    y=_clip(float(report["y"]), params.d_img_y0, params.d_img_y1),
                )
    return out


#: Frozen description of the smoke-only selection rule, emitted in the
#: round log so the bound is never silent.
SMOKE_SELECTION_RULE = (
    "smoke-only: frames evenly spaced across the window's temporal extent, "
    "then reports evenly spaced by (camera_id, track_id) within each "
    "selected frame; deterministic; selection reads ONLY track/camera/frame "
    "identifiers"
)


def resolve_smoke_report_cap(
    cap: int, *, smoke_schedule: bool, evidence_bearing: bool
) -> int:
    """Validate the SMOKE-ONLY per-window report cap, or refuse.

    Returns 0 (disabled, full-report semantics) when no cap is declared.
    A declared cap is admissible ONLY on a run that both sets
    `elgs_smoke_schedule` AND is stamped `evidence_bearing: false`. It is
    an evaluation bound for an integration smoke, NOT a production report
    cap: capping reports changes what the evidence estimator sees, and a
    production cap needs its own design, calibration, review and
    preregistration. Fail-closed on either condition.
    """
    cap = int(cap or 0)
    if cap <= 0:
        return 0
    if not smoke_schedule:
        raise ContractError(
            "elgs_smoke_max_reports_per_window is a SMOKE-ONLY evaluation "
            "bound and requires elgs_smoke_schedule; it is not a production "
            "report cap"
        )
    if evidence_bearing:
        raise ContractError(
            "elgs_smoke_max_reports_per_window may not be used on an "
            "evidence-bearing run: it truncates the reports the evidence "
            "estimator sees, which is a scientific change requiring its own "
            "design, calibration and preregistration"
        )
    return cap


def select_smoke_reports(reports: Mapping, max_reports: int):
    """A small deterministic, OUTCOME-BLIND subset of `reports`.

    Stratified across the window's temporal extent (evenly spaced frames)
    and across cameras (evenly spaced by camera id within each selected
    frame). Selection reads ONLY the (track_id, camera_id, frame) key —
    never q, never a likelihood, never the report's visibility value,
    never a delta sign — so it cannot select for a preferred result.

    Returns `(retained, stats)`. `max_reports <= 0` or a window already
    within budget returns the reports UNCHANGED, so the uncapped path is
    bit-identical to full-report semantics.
    """
    keys = sorted(reports)
    total = len(keys)
    if max_reports <= 0 or total <= max_reports:
        return dict(reports), {
            "total": total, "retained": total, "dropped": 0,
            "frames": len({k[2] for k in keys}),
            "cameras": len({k[1] for k in keys}),
        }

    by_frame: dict[float, list] = {}
    for key in keys:
        by_frame.setdefault(key[2], []).append(key)
    frames = sorted(by_frame)

    n_frames = min(len(frames), max_reports)
    picked_frames = sorted(
        dict.fromkeys(frames[(i * len(frames)) // n_frames] for i in range(n_frames))
    )
    budget = max(1, max_reports // len(picked_frames))

    chosen: list = []
    for index, frame in enumerate(picked_frames):
        group = sorted(by_frame[frame], key=lambda k: (k[1], k[0]))
        take = min(len(group), budget)
        # ROTATE the within-frame start by the frame's position. Without
        # it a budget of one report per frame always lands on the lowest
        # (camera_id, track_id) and the subset covers exactly ONE camera —
        # temporally spread but camera-degenerate. The rotation is pure
        # index arithmetic on identifiers, so it stays deterministic and
        # outcome-blind while spreading the subset over cameras too.
        offset = index % len(group)
        chosen.extend(
            group[(offset + (i * len(group)) // take) % len(group)]
            for i in range(take)
        )
    chosen = sorted(dict.fromkeys(chosen))[:max_reports]

    return {key: reports[key] for key in chosen}, {
        "total": total,
        "retained": len(chosen),
        "dropped": total - len(chosen),
        "frames": len({k[2] for k in chosen}),
        "cameras": len({k[1] for k in chosen}),
    }


def window_phi(
    context: EvidenceContext,
    clusters: Sequence[SeedCluster],
    reports: Mapping[tuple[int, int, float], Report],
    q_snapshot: QSnapshot,
    bridge_centers: Mapping[int, Mapping[tuple[int, int, float], tuple[float, float]]],
    z_series: Sequence[bool],
) -> float:
    """Phi_{f,W} for one family/window (elgs.energy, unchanged).

    `z_series` is the family's presence indicator over the window's
    frames under the interval hypothesis being scored -- this is the ONLY
    place the candidate's structure enters, which is what makes
    Phi_candidate - Phi_incumbent a well-defined closed-form delta.
    """
    frames = sorted({f for (_, _, f) in reports})
    if not frames:
        return 0.0
    segments = _build_segments(frames, z_series)
    if not segments:
        return 0.0

    q_maps = q_snapshot.as_bridge_maps()
    if not q_maps:
        return 0.0
    batch = EvidenceBatch(
        reports=dict(reports),
        q_tilde={b: dict(m) for b, m in q_maps.items()},
        bridge_centers={b: dict(m) for b, m in bridge_centers.items()},
    )
    stream = stream_log_likelihoods(
        batch,
        clusters,
        context.tracks_of_cluster,
        segments,
        context.constants.heads,
        context.c_cap,
    )
    return phi(stream, context.tau_b)


def _build_segments(frames: Sequence[float], z_series: Sequence[bool]) -> tuple[Segment, ...]:
    """S(P_f, W) with no edge band declared at smoke tier."""
    from .energy import build_segments

    if len(frames) != len(z_series):
        raise ContractError("z_series must align with the window's frames")
    return build_segments(frames, z_series, [False] * len(frames))


def presence_series(
    realization, frames: Sequence[float], frame_time: Mapping[int, float]
) -> tuple[bool, ...]:
    """z_f(t): whether an episode covers each frame under a realization.

    `frame_time` maps a DiVa-360 frame INDEX to the MODEL time the scene
    actually stamped on that frame's cameras. It is an explicit lookup,
    never a formula: the converter writes `time = index / fps` but the
    dataset reader then divides by `frame_ratio`
    (`scene/dataset_readers.py::readCamerasFromTransforms`), so
    `1 / artifact_fps` is NOT the model dt whenever `frame_ratio > 1`.
    Deriving it would silently mis-scale every interval comparison; a
    frame index absent from the map fails closed.
    """
    out: list[bool] = []
    for frame in frames:
        index = int(frame)
        if index not in frame_time:
            raise ContractError(
                f"artifact frame {index} has no model timestamp in the "
                "loaded scene; refusing to guess a frame->time mapping"
            )
        t = frame_time[index]
        covered = any(
            float(b) <= t <= float(d) for b, d in zip(realization.b, realization.d)
        )
        out.append(covered)
    return tuple(out)


# ---------------------------------------------------------------------------
# Round-boundary q refresh and the acceptance `exact_deltas` term
# ---------------------------------------------------------------------------


@dataclass
class WindowEvidence:
    """One (family, window) pair's frozen inputs for the round."""

    family_id: int
    window: Window
    clusters: tuple[SeedCluster, ...]
    reports: dict
    bridge_centers: dict
    frames: tuple[float, ...]


@dataclass
class RoundEvidence:
    """Everything the round needs, frozen at the round boundary."""

    round_index: int
    q_snapshot: QSnapshot
    windows: tuple[WindowEvidence, ...]
    coverage: dict
    n_q_values: int
    report_accounting: dict = field(default_factory=dict)

    def windows_of_family(self, family_id: int) -> tuple[WindowEvidence, ...]:
        return tuple(w for w in self.windows if w.family_id == family_id)


@dataclass(frozen=True)
class _QRequest:
    """One (report, bridge) pair the round must score, in snapshot order.

    `point` is the bridge's world position at this frame, or None when
    the report's seed carries no endpoints for this bridge -- the case
    the scalar loop records as `value = 0.0`, `centre = (0, 0)`.
    """

    bridge_id: int
    track_id: int
    camera_id: int
    frame: float
    d_u: float
    point: tuple[float, float, float] | None


def _plan_window_requests(
    reports,
    bridge_ids,
    bridges_by_seed,
    seed_of_track,
    d_by_cluster,
    cluster_of_seed,
    window,
) -> list[_QRequest]:
    """The window's (report, bridge) pairs, REPORT-major then bridge.

    The order is load-bearing: it is the order `snapshot.put` sees, and
    the double-write contract is defined against it.
    """
    requests: list[_QRequest] = []
    for (track_id, camera_id, frame) in reports:
        seed_id = seed_of_track[track_id]
        d_u = d_by_cluster.get(cluster_of_seed.get(seed_id, -1), 0.0)
        seed_bridges = bridges_by_seed.get(seed_id)
        for bridge_id in bridge_ids:
            endpoints = (seed_bridges or {}).get(bridge_id)
            requests.append(
                _QRequest(
                    bridge_id=bridge_id,
                    track_id=track_id,
                    camera_id=camera_id,
                    frame=frame,
                    d_u=d_u,
                    point=(
                        None if endpoints is None
                        else bridge_point(endpoints, window, frame)
                    ),
                )
            )
    return requests


def _resolve_requests_scalar(
    requests: Sequence[_QRequest], probe, *, sigma_world: float, family_id: int
) -> list[tuple[float, tuple[float, float]]]:
    """(q_tilde, bridge centre) per request, one probe call at a time.

    THE ORACLE. Kept verbatim from the pre-batching implementation and
    still the path any probe without the batched surface takes.
    """
    out: list[tuple[float, tuple[float, float]]] = []
    for request in requests:
        value = 0.0
        center = (0.0, 0.0)
        if request.point is not None:
            projected = probe.project(request.camera_id, request.point)
            if projected is not None:
                center = projected[0]
                sigmas = sigma_points_for(
                    request.point,
                    sigma_world,
                    lambda p, cid=request.camera_id: probe.project(cid, p),
                )
                if sigmas:
                    q = compute_q(
                        sigmas,
                        probe,
                        request.camera_id,
                        request.frame,
                        exclude_track=request.track_id,
                        present_family=family_id,
                        kappa_res=1.0,
                    )
                    value = q_tilde(q, request.d_u)
        out.append((value, center))
    return out


def _resolve_requests_batched(
    requests: Sequence[_QRequest], probe, *, sigma_world: float, family_id: int
) -> list[tuple[float, tuple[float, float]]]:
    """`_resolve_requests_scalar` with the probe calls grouped.

    Two batched passes replace the scalar path's fifteen projections and
    seven transmittance calls PER q value:

    1. PROJECTION, grouped by camera. Every request's 7-point sigma grid
       is projected in one call per camera instead of one call per point
       (the scalar path also re-projects the bridge centre separately;
       here it is grid element 0, the same world point through the same
       projector).
    2. TRANSMITTANCE, grouped by (camera, frame). Every sigma point of
       every report and bridge sharing a (camera, frame) composites
       against the SAME geometry and the SAME alpha columns, so one
       bounded-chunk reduction answers all of them.

    Nothing about WHICH reports, frames, cameras, bridges or sigma
    points contribute changes -- the request list, the grid, the
    drop-and-renormalize rule and the accumulation order are the scalar
    path's own.

    Grouping across reports at one (camera, frame) is valid only because
    the probe declares `excludes_track_rows = False`: `exclude_track` is
    recorded but filters no rows, so two reports at the same
    (camera, frame) composite the identical splat set. A probe that does
    filter by track does not set the attribute and is grouped per track.
    """
    ignores_track = getattr(probe, "excludes_track_rows", True) is False

    # -- pass 1: grids, then one projection call per camera -------------
    grids: dict[int, tuple] = {}
    by_camera: dict[int, list[tuple[int, int]]] = {}
    world_points: dict[int, list] = {}
    for index, request in enumerate(requests):
        if request.point is None:
            continue
        grid = sigma_grid(request.point, sigma_world)
        grids[index] = grid
        slots = by_camera.setdefault(request.camera_id, [])
        points = world_points.setdefault(request.camera_id, [])
        for grid_index, (world, _weight) in enumerate(grid):
            slots.append((index, grid_index))
            points.append(world)

    projections: dict[int, list] = {
        index: [None] * len(grid) for index, grid in grids.items()
    }
    for camera_id, slots in by_camera.items():
        results = probe.project_batch(camera_id, world_points[camera_id])
        for (index, grid_index), projected in zip(slots, results):
            projections[index][grid_index] = projected

    # -- pass 2: sigma points, then one reduction per (camera, frame) ---
    sigmas_by_request: dict[int, tuple[SigmaPoint, ...]] = {}
    centers: dict[int, tuple[float, float]] = {}
    groups: dict[tuple, list[int]] = {}
    for index, request in enumerate(requests):
        if index not in grids:
            continue
        projected = projections[index]
        # The scalar path projects the CENTRE first and, when that fails,
        # records value 0 / centre (0, 0) without consulting the offsets.
        if projected[0] is None:
            continue
        centers[index] = projected[0][0]
        sigmas = sigma_points_from_projections(grids[index], projected)
        if not sigmas:
            continue
        sigmas_by_request[index] = sigmas
        key = (
            (request.camera_id, request.frame)
            if ignores_track
            else (request.camera_id, request.frame, request.track_id)
        )
        groups.setdefault(key, []).append(index)

    transmittances: dict[int, list[float]] = {}
    for key, members in groups.items():
        camera_id, frame = key[0], key[1]
        pixels: list[tuple[float, float]] = []
        depths: list[float] = []
        for index in members:
            for point in sigmas_by_request[index]:
                pixels.append(point.pixel)
                depths.append(point.depth)
        values = probe.transmittance_batch(
            camera_id,
            frame,
            pixels,
            depths,
            # None is exact when the probe ignores the argument; the
            # group key carries the track otherwise.
            exclude_track=None if ignores_track else key[2],
            present_family=family_id,
        ).tolist()
        cursor = 0
        for index in members:
            width = len(sigmas_by_request[index])
            transmittances[index] = values[cursor:cursor + width]
            cursor += width

    # -- assemble, in request order -------------------------------------
    out: list[tuple[float, tuple[float, float]]] = []
    for index, request in enumerate(requests):
        value = 0.0
        center = centers.get(index, (0.0, 0.0))
        sigmas = sigmas_by_request.get(index)
        if sigmas is not None:
            # Every surviving sigma point projected, and `project`
            # rejects on exactly the znear + raster-bounds conditions
            # `in_frustum` rejects on, so the scalar path's per-point
            # in_frustum test admits all of them. `ModelProbe`'s two
            # predicates are pinned equal by test.
            q = compute_q_batch(
                sigmas,
                probe,
                request.camera_id,
                request.frame,
                exclude_track=request.track_id,
                present_family=family_id,
                kappa_res=1.0,
                transmittances=transmittances[index],
            )
            value = q_tilde(q, request.d_u)
        out.append((value, center))
    return out


def resolve_window_requests(
    requests: Sequence[_QRequest], probe, *, sigma_world: float, family_id: int
) -> list[tuple[float, tuple[float, float]]]:
    """Batched when the probe offers the batched surface, else scalar."""
    batched = all(
        callable(getattr(probe, name, None))
        for name in ("project_batch", "transmittance_batch")
    )
    resolve = _resolve_requests_batched if batched else _resolve_requests_scalar
    return resolve(requests, probe, sigma_world=sigma_world, family_id=family_id)


def refresh_round_evidence(
    context: EvidenceContext,
    probe,
    *,
    round_index: int,
    family_ids: Sequence[int],
) -> RoundEvidence:
    """Recompute the q snapshot and freeze it for this round (spec §3).

    A family with fewer than two anchors yields no windows and is
    recorded `photometric_only` in coverage (disclosed reading D3): it
    contributes Phi = 0.0 and the round proceeds. That is the
    INSUFFICIENT-EVIDENCE FALLBACK, not a refusal.
    """
    snapshot = QSnapshot(round_index)
    accounting = {
        "total": 0, "retained": 0, "dropped": 0,
        "frames_covered": 0, "cameras_covered": 0,
        "max_reports_per_window": int(context.smoke_max_reports_per_window),
        "selection_rule": (
            SMOKE_SELECTION_RULE
            if context.smoke_max_reports_per_window > 0
            else "uncapped: every report in the window"
        ),
    }
    windows: list[WindowEvidence] = []
    coverage: dict[str, dict] = {}
    written: set[tuple[int, int, int, float]] = set()

    for family_id in sorted(int(f) for f in family_ids):
        clusters = context.clusters_of_family(family_id)
        if not clusters:
            coverage[str(family_id)] = {
                "n_anchors": 0, "n_windows": 0, "photometric_only": True,
                "reason": "no bound clusters",
            }
            continue
        track_ids = context.track_ids_of_family(family_id)
        anchors, family_window_set = family_windows(
            context.sealed, track_ids, c_cap=context.c_cap, constants=context.constants
        )
        entry = coverage_entry(family_id, anchors)
        # `n_windows_retained` is counted AFTER the drops below, so a
        # family whose windows all carry zero reports or no bridge family
        # is not logged as evidence-covered while contributing Phi = 0.
        record = {
            "n_anchors": entry.n_anchors,
            "n_windows": entry.n_windows,
            "n_windows_retained": 0,
            "photometric_only": entry.photometric_only,
        }
        coverage[str(family_id)] = record
        if entry.photometric_only:
            continue

        for window in family_window_set:
            reports = _reports_in_window(
                context.sealed, track_ids, window, context.constants.heads.params
            )
            if not reports:
                continue
            reports, selection = select_smoke_reports(
                reports, context.smoke_max_reports_per_window
            )
            for field_name in ("total", "retained", "dropped"):
                accounting[field_name] += selection[field_name]
            accounting["frames_covered"] += selection["frames"]
            accounting["cameras_covered"] = max(
                accounting["cameras_covered"], selection["cameras"]
            )
            bridges_by_seed: dict[int, dict] = {}
            for cluster in clusters:
                for seed_id in cluster.seed_ids:
                    family_bridges = build_bridge_family(context.sealed, seed_id, window)
                    if family_bridges:
                        bridges_by_seed[seed_id] = family_bridges
            if not bridges_by_seed:
                continue
            bridge_ids = sorted({b for fam in bridges_by_seed.values() for b in fam})
            centers: dict[int, dict] = {b: {} for b in bridge_ids}

            d_by_cluster = {c.cluster_id: c.d_u for c in clusters}
            cluster_of_seed = {
                sid: c.cluster_id for c in clusters for sid in c.seed_ids
            }
            requests = _plan_window_requests(
                reports,
                bridge_ids,
                bridges_by_seed,
                context.seed_of_track,
                d_by_cluster,
                cluster_of_seed,
                window,
            )
            resolved = resolve_window_requests(
                requests,
                probe,
                sigma_world=context.constants.sigma_world,
                family_id=family_id,
            )
            for request, (value, center) in zip(requests, resolved):
                key = (
                    request.bridge_id, request.track_id,
                    request.camera_id, request.frame,
                )
                if key in written:
                    # With exclusive window endpoints two windows of
                    # one family cannot share a frame, so this is
                    # unreachable. It is a HARD failure rather than a
                    # silent skip: the previous revision skipped, and
                    # the skip handed the second window a q computed
                    # under the FIRST window's bridge geometry while
                    # recomputing its centre -- q and g_pos then
                    # referred to different world points.
                    raise ContractError(
                        f"q snapshot key {key} written twice: two windows "
                        "of one family share a frame, which the exclusive "
                        "window-endpoint rule must prevent"
                    )
                snapshot.put(
                    request.bridge_id, request.track_id,
                    request.camera_id, request.frame, value,
                )
                written.add(key)
                centers[request.bridge_id][
                    (request.track_id, request.camera_id, request.frame)
                ] = center

            windows.append(
                WindowEvidence(
                    family_id=family_id,
                    window=window,
                    clusters=clusters,
                    reports=reports,
                    bridge_centers=centers,
                    frames=tuple(sorted({f for (_, _, f) in reports})),
                )
            )
            record["n_windows_retained"] += 1
            record["photometric_only"] = False
        if record["n_windows_retained"] == 0:
            record["photometric_only"] = True
            record.setdefault("reason", "no window retained reports or bridges")

    snapshot.freeze()
    context.coverage = coverage
    return RoundEvidence(
        round_index=round_index,
        q_snapshot=snapshot,
        windows=tuple(windows),
        coverage=coverage,
        n_q_values=len(written),
        report_accounting=accounting,
    )


def _window_q_subset(round_evidence: RoundEvidence, entry: WindowEvidence) -> QSnapshot:
    """A frozen QSnapshot restricted to one window's report keys.

    `EvidenceBatch.validate_bridge_independent_eligibility` requires each
    bridge's q map to cover EXACTLY the window's report key set; the
    round snapshot spans every window, so it is projected down here.
    Values are copied, never recomputed -- the round snapshot stays the
    single source of q (spec §3: no mid-round refresh).
    """
    subset = QSnapshot(round_evidence.round_index)
    for bridge_id in sorted(entry.bridge_centers):
        for (track_id, camera_id, frame) in entry.reports:
            subset.put(
                bridge_id,
                track_id,
                camera_id,
                frame,
                round_evidence.q_snapshot.get(bridge_id, track_id, camera_id, frame),
            )
    subset.freeze()
    return subset


def family_phi(
    context: EvidenceContext,
    round_evidence: RoundEvidence,
    family_id: int,
    realization,
    *,
    frame_time: Mapping[int, float],
) -> float:
    """sum_W Phi_{f,W} under one interval realization (0.0 if no windows).

    `realization=None` scores the family as absent everywhere (the
    TRUNCATE_DELETE / retirement hypothesis).
    """
    total = 0.0
    for entry in round_evidence.windows_of_family(family_id):
        if realization is None:
            z_series = tuple(False for _ in entry.frames)
        else:
            z_series = presence_series(realization, entry.frames, frame_time)
        total += window_phi(
            context,
            entry.clusters,
            entry.reports,
            _window_q_subset(round_evidence, entry),
            entry.bridge_centers,
            z_series,
        )
    return total


def evidence_exact_delta(
    context: EvidenceContext,
    round_evidence: RoundEvidence,
    plan,
    registry,
    config,
    *,
    frame_time: Mapping[int, float],
) -> float:
    """beta * (Phi_candidate - Phi_incumbent) over the plan's families.

    This is the acceptance path's `exact_deltas` term: a NON-SAMPLED
    closed-form quantity added outside the SNIS render estimate
    (`elgs.acceptance` module docstring). A plan touching only families
    with no evidence windows returns exactly 0.0, so the decision
    reduces to the verified M0 photometric comparison.
    """
    from .intervals import forward

    delta = 0.0
    for family_id in plan.family_ids:
        if not round_evidence.windows_of_family(family_id):
            continue
        record = registry.get(family_id)
        incumbent = (
            None
            if record.retired or record.interval.K == 0
            else forward(record.interval, config)
        )
        candidate_state = plan.child_intervals.get(family_id)
        candidate = (
            None
            if candidate_state is None or candidate_state.K == 0
            else forward(candidate_state, config)
        )
        phi_inc = family_phi(
            context, round_evidence, family_id, incumbent, frame_time=frame_time
        )
        phi_cand = family_phi(
            context, round_evidence, family_id, candidate, frame_time=frame_time
        )
        delta += phi_cand - phi_inc
    return context.beta * delta


__all__ = [
    "EvidenceConstants",
    "EvidenceContext",
    "RoundEvidence",
    "WindowEvidence",
    "evidence_exact_delta",
    "family_phi",
    "refresh_round_evidence",
    "SeedEvidence",
    "bind_clusters",
    "bridge_point",
    "build_bridge_family",
    "build_clusters",
    "build_evidence_context",
    "derive_seed_evidence",
    "family_anchor_series",
    "family_windows",
    "SMOKE_SELECTION_RULE",
    "load_evidence_constants",
    "presence_series",
    "resolve_smoke_report_cap",
    "resolve_window_requests",
    "select_smoke_reports",
    "seed_overlap_edges",
    "sigma_grid",
    "sigma_points_for",
    "sigma_points_from_projections",
    "window_phi",
]
