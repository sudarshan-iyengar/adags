"""Segments, cap operator, evidence streams, tempered Phi, energy (spec §4).

Segment set S(P_f, W): the maximal runs of frames {t in W : t not in
X_f}, each run carrying its constant z_f(t). Cap operator (per
bridge): A_{b,j,t} = the C_cap cameras of highest q_tilde, ties broken
by ASCENDING CAMERA ID (bridge-independent tie-break); fewer than
C_cap eligible => ALL eligible cameras are taken (both streams);
report ELIGIBILITY is bridge-independent by construction.

  l_b(P_f)      = sum_u alpha_u sum_{j in J(u)} sum_{t in S} sum_{c in A}
                  log L_{z_f(t)}(y_{j,c,t} | b, q_tilde, r_u)
  l_b_cens      identical with L0 in place of L_z (same A)
  Phi_{f,W}     = -tau_B * log[ sum_b e^{l_b/tau_B} / sum_b e^{l_b_cens/tau_B} ]

PROP 1 (censored zero) is honored EXACTLY by code path: a segment
whose capped reports have q_tilde identically zero under EVERY bridge
adds the same constant to every l_b and every l_b_cens (censoring
equality + bridge-independent tie-break), and F(x + c*1) = F(x) - c
cancels it in Phi — so such segments are DETECTED and contribute
exactly 0.0, under any program and any transition re-partitioning.

TERMINOLOGY (spec §4): Phi is a TEMPERED BRIDGE AGGREGATION, an
engineered energy — not a marginal likelihood.

State energy + transaction ledger:
  E = L_render + beta * sum Phi
      + sum_f [ kappa*K_f + sum psi_dur(len) + sum psi_gap(gap) ]
      + C(H),  C(H) = chi*N_returnbirth + mu*N_merge  (ledgered,
      never refunded; kappa is a state term, auto-refunded on
      deletion). The psi barriers are preregistered quadratic forms
      (category-B); they enter here as injected callables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from depth_visibility.errors import ContractError

from .clusters import SeedCluster
from .evidence import EvidenceHeads, Report, likelihood_l0, likelihood_l1

ReportKey = tuple[int, int, float]  # (track_id, camera_id, frame)


@dataclass(frozen=True)
class Segment:
    """One maximal edge-band-free run inside a window, constant z."""

    frames: tuple[float, ...]
    z: bool

    def __post_init__(self) -> None:
        if not self.frames:
            raise ContractError("a segment must contain at least one frame")


def build_segments(
    frames: Sequence[float],
    z_series: Sequence[bool],
    edge_band_series: Sequence[bool],
) -> tuple[Segment, ...]:
    """S(P_f, W): maximal runs of non-edge-band frames, constant z.

    A run breaks where an edge-band frame intervenes OR where z
    changes (each run carries its constant z_f(t) — spec §4).
    """
    if not (len(frames) == len(z_series) == len(edge_band_series)):
        raise ContractError("segment inputs must have equal length")
    segments: list[Segment] = []
    run: list[float] = []
    run_z: bool | None = None
    for frame, z, banded in zip(frames, z_series, edge_band_series):
        if banded or (run and z != run_z):
            if run:
                segments.append(Segment(frames=tuple(run), z=bool(run_z)))
                run = []
            run_z = None
        if not banded:
            if not run:
                run_z = z
            run.append(frame)
    if run:
        segments.append(Segment(frames=tuple(run), z=bool(run_z)))
    return tuple(segments)


def cap_select(
    q_tilde_by_camera: Mapping[int, float], c_cap: int
) -> tuple[int, ...]:
    """A_{b,j,t}: C_cap cameras of highest q_tilde, ties by ascending
    camera ID; fewer eligible => all. When every q_tilde is equal
    (e.g. all zero) the selection reduces to the camera-ID tie-break
    alone, which is bridge-independent — PROP 1 condition (i)."""
    if c_cap < 1:
        raise ContractError("C_cap must be >= 1")
    ranked = sorted(q_tilde_by_camera, key=lambda c: (-q_tilde_by_camera[c], c))
    return tuple(sorted(ranked[:c_cap]))


@dataclass(frozen=True)
class EvidenceBatch:
    """Frozen per-window evidence inputs at a round snapshot.

    reports:   (j, c, t) -> Report          (eligibility = key set,
               bridge-independent by construction)
    q_tilde:   bridge_id -> {(j, c, t) -> float}  (round-snapshot
               values; a missing key means camera ineligible)
    bridge_centers: bridge_id -> {(j, c, t) -> (x, y)} projected points
    """

    reports: Mapping[ReportKey, Report]
    q_tilde: Mapping[int, Mapping[ReportKey, float]]
    bridge_centers: Mapping[int, Mapping[ReportKey, tuple[float, float]]]

    def bridges(self) -> tuple[int, ...]:
        if not self.q_tilde:
            raise ContractError("evidence batch carries no bridges")
        return tuple(sorted(self.q_tilde))

    def validate_bridge_independent_eligibility(self) -> None:
        """Report ELIGIBILITY must be bridge-independent by
        construction (spec §4; PROP 1 condition (i)): every bridge's
        q map covers exactly the report key set."""
        report_keys = set(self.reports)
        for bridge_id in self.bridges():
            keys = set(self.q_tilde[bridge_id])
            if keys != report_keys:
                missing = sorted(report_keys - keys)[:3]
                extra = sorted(keys - report_keys)[:3]
                raise ContractError(
                    f"bridge {bridge_id} q map breaks bridge-independent "
                    f"eligibility (missing~{missing}, extra~{extra})"
                )


def _capped_keys_for(
    batch: EvidenceBatch,
    bridge_id: int,
    track_ids: Sequence[int],
    segment: Segment,
    c_cap: int,
) -> tuple[ReportKey, ...]:
    q_b = batch.q_tilde[bridge_id]
    keys: list[ReportKey] = []
    for j in track_ids:
        for t in segment.frames:
            cams = {
                c: q_b[(j, c, t)]
                for (jj, c, tt) in batch.reports
                if jj == j and tt == t and (j, c, t) in q_b
            }
            if not cams:
                continue
            for c in cap_select(cams, c_cap):
                keys.append((j, c, t))
    return tuple(keys)


def segment_is_censored(
    batch: EvidenceBatch,
    track_ids: Sequence[int],
    segment: Segment,
) -> bool:
    """True iff q_tilde == 0.0 exactly for ALL bridges over the
    segment's eligible reports (PROP 1 antecedent)."""
    for bridge_id in batch.bridges():
        q_b = batch.q_tilde[bridge_id]
        for j in track_ids:
            for t in segment.frames:
                for (jj, c, tt), q in q_b.items():
                    if jj == j and tt == t and q != 0.0:
                        return False
    return True


@dataclass(frozen=True)
class StreamLogLik:
    """Per-bridge scored and censored stream log-likelihoods."""

    scored: dict
    censored: dict


def stream_log_likelihoods(
    batch: EvidenceBatch,
    clusters: Sequence[SeedCluster],
    tracks_of_cluster: Mapping[int, Sequence[int]],
    segments: Sequence[Segment],
    heads: EvidenceHeads,
    c_cap: int,
) -> StreamLogLik:
    """l_b and l_b^cens over U(f)'s clusters. Censored segments are
    omitted from BOTH streams (their contribution is exactly the same
    constant in each — see module docstring), which realizes PROP 1
    as an exact 0.0 rather than a float cancellation."""
    batch.validate_bridge_independent_eligibility()
    scored: dict[int, float] = {}
    censored: dict[int, float] = {}
    for bridge_id in batch.bridges():
        l_scored = 0.0
        l_cens = 0.0
        for cluster in clusters:
            track_ids = tuple(tracks_of_cluster.get(cluster.cluster_id, ()))
            if not track_ids:
                continue
            # r_u is validated at SeedCluster construction; a value
            # below r_min fails closed in the likelihood call — no
            # silent clamping (spec §2 range is a constraint).
            r_u = cluster.r_u
            for segment in segments:
                if segment_is_censored(batch, track_ids, segment):
                    continue
                for key in _capped_keys_for(batch, bridge_id, track_ids, segment, c_cap):
                    report = batch.reports[key]
                    q = batch.q_tilde[bridge_id][key]
                    if segment.z:
                        center = batch.bridge_centers[bridge_id][key]
                        l1 = likelihood_l1(report, center, q, r_u, heads)
                    else:
                        # z = 0 on the segment: the scored stream uses
                        # L0 (absence hypothesis) — L_{z_f(t)} with z=0.
                        l1 = likelihood_l0(report, r_u, heads)
                    l0 = likelihood_l0(report, r_u, heads)
                    # p_floor, not a raise. A previous revision raised
                    # here, which was wrong twice over: math.log(0) gives
                    # ValueError (not a ContractError, so it escapes the
                    # round driver and kills the job), but raising at all
                    # turns a REACHABLE and MEANINGFUL state into a fatal
                    # error. L1 -> 0 means "the report is far from the
                    # bridge, so presence is strongly disfavoured" — valid
                    # evidence, not a fault. The heads prereg already
                    # declares a p_floor for exactly this
                    # ("derived: inf L0 = r_min*pi_floor at the miss token
                    # or r_min*(1-pi_m^c)*h_floor/|D_img| positionally,
                    # whichever is smaller"), so the floor is applied.
                    floor = _p_floor(heads)
                    l_scored += cluster.alpha_u * math.log(max(l1, floor))
                    l_cens += cluster.alpha_u * math.log(max(l0, floor))
        scored[bridge_id] = l_scored
        censored[bridge_id] = l_cens
    return StreamLogLik(scored=scored, censored=censored)


def _p_floor(heads: EvidenceHeads) -> float:
    """The prereg's declared likelihood floor (evidence-heads
    `floors_caps.p_floor`): the smaller of the miss-token and positional
    infima of L0. Both limbs are strictly positive for admissible heads,
    so the floor never fires on well-posed evidence — it exists so a
    vanishing position head degrades the report to "maximally
    uninformative" instead of to an undefined logarithm."""
    params = heads.params
    miss_limb = params.r_min * params.pi_floor
    positional_limb = (
        params.r_min * (1.0 - params.pi_miss_cens) * params.h_floor / params.d_img_area
    )
    floor = min(miss_limb, positional_limb)
    if floor <= 0.0:
        raise ContractError(
            "derived p_floor is non-positive; the frozen head constants are "
            "inadmissible"
        )
    return floor


def _tempered_lse(values: Sequence[float], tau: float) -> float:
    """tau * log sum_b e^{v/tau}, max-stabilized."""
    peak = max(values)
    return peak + tau * math.log(sum(math.exp((v - peak) / tau) for v in values))


def phi(stream: StreamLogLik, tau_b: float) -> float:
    """Phi = -tau_B log[sum_b e^{l_b/tau_B} / sum_b e^{l_cens/tau_B}]."""
    if tau_b <= 0.0:
        raise ContractError("tau_B must be strictly positive")
    bridges = sorted(stream.scored)
    if bridges != sorted(stream.censored):
        raise ContractError("scored and censored streams disagree on bridges")
    if not bridges:
        raise ContractError("Phi undefined with no bridges")
    scored = [stream.scored[b] for b in bridges]
    cens = [stream.censored[b] for b in bridges]
    return -(_tempered_lse(scored, tau_b) - _tempered_lse(cens, tau_b))


@dataclass(frozen=True)
class PriorTerms:
    """kappa*K + psi barriers; psi forms are preregistered callables."""

    kappa: float
    psi_dur: Callable[[float], float]
    psi_gap: Callable[[float], float]

    def state_prior(self, lens: Sequence[float], gaps: Sequence[float]) -> float:
        if self.kappa < 0.0:
            raise ContractError("kappa must be non-negative")
        return (
            self.kappa * len(lens)
            + sum(self.psi_dur(v) for v in lens)
            + sum(self.psi_gap(v) for v in gaps)
        )


@dataclass(frozen=True)
class TransactionCharges:
    """C(H) = chi*N_returnbirth + mu*N_merge — ledgered, never refunded."""

    chi: float
    mu: float

    def ledger_charge(self, n_return_birth: int, n_merge: int) -> float:
        if n_return_birth < 0 or n_merge < 0:
            raise ContractError("ledger event counts must be non-negative")
        return self.chi * n_return_birth + self.mu * n_merge


def total_energy(
    l_render: float,
    beta: float,
    phi_values: Sequence[float],
    per_family_priors: Sequence[float],
    ledger_charge: float,
) -> float:
    """E = L_render + beta*sum Phi + sum priors + C(H)."""
    if beta < 0.0:
        raise ContractError("beta must be non-negative")
    return l_render + beta * sum(phi_values) + sum(per_family_priors) + ledger_charge


__all__ = [
    "EvidenceBatch",
    "PriorTerms",
    "Segment",
    "StreamLogLik",
    "TransactionCharges",
    "build_segments",
    "cap_select",
    "phi",
    "segment_is_censored",
    "stream_log_likelihoods",
    "total_energy",
]
