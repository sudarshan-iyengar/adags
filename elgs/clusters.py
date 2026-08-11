"""Seed clusters, canonical points, family binding (spec §2, rev-3 A2).

Report ownership is BY CONSTRUCTION: each track j comes from exactly
one seed s(j); each seed belongs to exactly one cluster — the
connected components of the seed-overlap graph at binding — so the
streams J(u) are pairwise disjoint.

Canonical cluster point (rev-3 A2): x_u := the seed surface point of
the LOWEST-ID seed in the component (deterministic, persisted);
bind(u) = the family of the nearest primitive to x_u within the
preregistered threshold (single-valued by construction); unassigned
=> cluster inactive. Component re-formation at MERGE recomputes x_u
by the same lowest-ID rule over the merged component.

Merged-cluster aggregation (spec §2, deterministic): r_u = min over
members, d_u = min over members, alpha_u recomputed from the merged
n_cam under the preregistered correlation model (injected — the model
itself is a category-2 preregistration item).

Binding freeze (plan §2.4a): the 2.8k audit freezes the audited
binding set. Permitted post-audit mutations are EXACTLY: MERGE
redirection to the surviving family ID, and late-birth lineage-local
seeding for families born after the audit. Anything else raises.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence

import torch

from depth_visibility.errors import ContractError


@dataclass(frozen=True)
class SeedCluster:
    """One ownership cluster u (a connected component of seeds)."""

    cluster_id: int
    seed_ids: tuple[int, ...]
    canonical_point: tuple[float, float, float]
    r_u: float
    d_u: float
    n_cam: int
    alpha_u: float

    def __post_init__(self) -> None:
        if not self.seed_ids:
            raise ContractError("a cluster must own at least one seed")
        if list(self.seed_ids) != sorted(self.seed_ids):
            raise ContractError("cluster seed_ids must be sorted ascending")
        if not 0.0 < self.r_u <= 1.0:
            raise ContractError(
                f"r_u={self.r_u} outside (0,1] (spec §2: r_u in [r_min,1]; "
                "the head-level r_min bound is enforced at likelihood "
                "evaluation, never by clamping)"
            )
        if not 0.0 <= self.d_u <= 1.0:
            raise ContractError(f"d_u={self.d_u} outside [0,1]")
        if not 0.0 < self.alpha_u <= 1.0:
            raise ContractError(f"alpha_u={self.alpha_u} outside (0,1]")
        if self.n_cam < 1:
            raise ContractError("n_cam must be >= 1")


class _UnionFind:
    def __init__(self, ids: Sequence[int]) -> None:
        self._parent = {i: i for i in ids}

    def find(self, i: int) -> int:
        root = i
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[i] != root:
            self._parent[i], i = root, self._parent[i]
        return root

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            # Deterministic: lower root wins.
            lo, hi = (ri, rj) if ri < rj else (rj, ri)
            self._parent[hi] = lo


def form_clusters(
    seed_points: Mapping[int, tuple[float, float, float]],
    overlap_edges: Sequence[tuple[int, int]],
    seed_r: Mapping[int, float],
    seed_d: Mapping[int, float],
    seed_n_cam: Mapping[int, int],
    alpha_model: Callable[[int], float],
) -> tuple[SeedCluster, ...]:
    """Connected components of the seed-overlap graph at binding.

    Cluster IDs are assigned by ascending lowest member seed ID, so
    the result is deterministic for a fixed input regardless of edge
    order. Per-cluster r/d take the min over members (the same
    conservative rule §2 prescribes for merged clusters); n_cam is
    the max over members and alpha_u = alpha_model(n_cam).
    """
    ids = sorted(seed_points)
    if not ids:
        return ()
    for mapping, name in ((seed_r, "seed_r"), (seed_d, "seed_d"), (seed_n_cam, "seed_n_cam")):
        missing = set(ids) - set(mapping)
        if missing:
            raise ContractError(f"{name} missing entries for seeds {sorted(missing)}")
    uf = _UnionFind(ids)
    for a, b in overlap_edges:
        if a not in seed_points or b not in seed_points:
            raise ContractError(f"overlap edge ({a},{b}) references unknown seed")
        uf.union(a, b)

    components: dict[int, list[int]] = {}
    for i in ids:
        components.setdefault(uf.find(i), []).append(i)

    clusters = []
    for cluster_id, root in enumerate(sorted(components, key=lambda r: min(components[r]))):
        members = tuple(sorted(components[root]))
        lowest = members[0]
        n_cam = max(seed_n_cam[m] for m in members)
        clusters.append(
            SeedCluster(
                cluster_id=cluster_id,
                seed_ids=members,
                canonical_point=tuple(float(v) for v in seed_points[lowest]),
                r_u=min(seed_r[m] for m in members),
                d_u=min(seed_d[m] for m in members),
                n_cam=n_cam,
                alpha_u=alpha_model(n_cam),
            )
        )
    return tuple(clusters)


def merge_clusters(
    members: Sequence[SeedCluster],
    seed_points: Mapping[int, tuple[float, float, float]],
    alpha_model: Callable[[int], float],
    *,
    cluster_id: int,
) -> SeedCluster:
    """Re-form one cluster from merged components (spec §2 rules).

    Recorded reading: §2 says alpha_u is "recomputed from the merged
    n_cam"; the merged n_cam here is the MAX over member n_cam (the
    cameras that can see any member can see the merged cluster's
    canonical region at least as well as its best-covered member) —
    disclosed, same rule form_clusters uses.
    """
    if len(members) < 2:
        raise ContractError("merge_clusters needs at least two clusters")
    seed_ids = tuple(sorted(sid for c in members for sid in c.seed_ids))
    if len(seed_ids) != len(set(seed_ids)):
        raise ContractError("merged clusters share seeds: ownership violated")
    lowest = seed_ids[0]
    if lowest not in seed_points:
        raise ContractError(f"canonical seed {lowest} has no persisted surface point")
    n_cam = max(c.n_cam for c in members)
    return SeedCluster(
        cluster_id=cluster_id,
        seed_ids=seed_ids,
        canonical_point=tuple(float(v) for v in seed_points[lowest]),
        r_u=min(c.r_u for c in members),
        d_u=min(c.d_u for c in members),
        n_cam=n_cam,
        alpha_u=alpha_model(n_cam),
    )


def nearest_family(
    point: tuple[float, float, float],
    primitive_positions: torch.Tensor,
    primitive_family_ids: torch.Tensor,
    threshold: float,
) -> int | None:
    """bind(u): family of the nearest primitive to x_u within the
    preregistered threshold; None => cluster inactive (spec §2)."""
    if primitive_positions.dim() != 2 or primitive_positions.shape[1] != 3:
        raise ContractError("primitive_positions must be (N, 3)")
    if primitive_family_ids.shape[0] != primitive_positions.shape[0]:
        raise ContractError("family-id column length mismatch")
    if threshold <= 0:
        raise ContractError("binding threshold must be positive")
    if primitive_positions.shape[0] == 0:
        return None
    x = primitive_positions.new_tensor(point)
    distances = torch.linalg.norm(primitive_positions - x, dim=1)
    best = int(torch.argmin(distances))
    if float(distances[best]) > threshold:
        return None
    return int(primitive_family_ids[best])


class BindingTable:
    """cluster_id -> family_id (or None = inactive), with audit freeze.

    After freeze_audited() the ONLY permitted mutations are
    redirect_for_merge (surviving-ID redirection) and bind_late_birth
    (families born after the audit, same delay/audit protocol); both
    are logged. Everything else raises (plan §2.4a).
    """

    def __init__(self) -> None:
        self._binding: dict[int, int | None] = {}
        self._frozen = False
        self._mutation_log: list[dict] = []

    def bind(self, cluster_id: int, family_id: int | None) -> None:
        if self._frozen:
            raise ContractError(
                "binding table is frozen after the audit; only MERGE "
                "redirection and late-birth seeding are permitted"
            )
        self._binding[cluster_id] = family_id

    def freeze_audited(self) -> None:
        if self._frozen:
            raise ContractError("binding table already frozen")
        self._frozen = True

    @property
    def frozen(self) -> bool:
        return self._frozen

    def family_of(self, cluster_id: int) -> int | None:
        if cluster_id not in self._binding:
            raise ContractError(f"cluster {cluster_id} has no binding entry")
        return self._binding[cluster_id]

    def clusters_of(self, family_id: int) -> tuple[int, ...]:
        """U(f) = bind^{-1}(f)."""
        return tuple(
            sorted(c for c, f in self._binding.items() if f == family_id)
        )

    def redirect_for_merge(self, retired_family: int, surviving_family: int) -> tuple[int, ...]:
        """Permitted post-audit mutation 1: both parents' bindings are
        redirected to the surviving ID (rev-3 A3)."""
        moved = tuple(
            sorted(c for c, f in self._binding.items() if f == retired_family)
        )
        for cluster_id in moved:
            self._binding[cluster_id] = surviving_family
        self._mutation_log.append(
            {
                "kind": "merge_redirect",
                "retired_family": retired_family,
                "surviving_family": surviving_family,
                "clusters": list(moved),
            }
        )
        return moved

    def bind_late_birth(self, cluster_id: int, family_id: int) -> None:
        """Permitted post-audit mutation 2: late-birth lineage-local
        seeding binds a NEW cluster; rebinding an existing one raises."""
        if cluster_id in self._binding:
            raise ContractError(
                f"cluster {cluster_id} already bound; late-birth seeding "
                "may only bind new clusters"
            )
        self._binding[cluster_id] = family_id
        self._mutation_log.append(
            {"kind": "late_birth_bind", "cluster_id": cluster_id, "family_id": family_id}
        )

    @property
    def mutation_log(self) -> tuple[dict, ...]:
        return tuple(self._mutation_log)

    # -- serialization (composed by elgs.state_io) ------------------------

    def to_state(self) -> dict:
        return {
            "binding": {str(c): f for c, f in sorted(self._binding.items())},
            "frozen": self._frozen,
            "mutation_log": list(self._mutation_log),
        }

    @classmethod
    def from_state(cls, payload: dict) -> "BindingTable":
        table = cls()
        for cluster_str, family in payload["binding"].items():
            table._binding[int(cluster_str)] = None if family is None else int(family)
        table._frozen = bool(payload["frozen"])
        table._mutation_log = [dict(e) for e in payload.get("mutation_log", [])]
        return table


__all__ = [
    "BindingTable",
    "SeedCluster",
    "form_clusters",
    "merge_clusters",
    "nearest_family",
]
