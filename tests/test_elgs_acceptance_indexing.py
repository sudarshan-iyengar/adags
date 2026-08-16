"""Parity tests for the exact (track, frame) indexing in elgs/energy.py.

`_capped_keys_for` and `segment_is_censored` used to recover "the
cameras of this (track, frame)" by rescanning the whole report
population, making both O(tracks x frames x reports). The indexed
versions answer the same questions by lookup. These tests pin that the
answers are IDENTICAL — not close, identical — against the preserved
pre-index bodies (`_capped_keys_for_scan`, `_segment_is_censored_scan`),
which are the oracle.

The oracle is the code that actually shipped and produced experiments
78-83, so parity here is parity with recorded behaviour rather than with
a re-derivation of it.

Scale: these fixtures are deliberately small, because the scanning
oracle they compare against is quadratic. The complexity and throughput
claims are measured by scripts/profile_acceptance_indexing.py, not here.

CPU only. elgs/energy.py imports `math` and nothing else numerical — it
holds no tensors and touches no device — so there is no CPU/GPU split to
test at this layer; `test_energy_layer_is_device_free` pins that fact so
the absence of a GPU case is a checked property rather than an omission.
"""

import math
import unittest

from depth_visibility.errors import ContractError
from elgs.clusters import SeedCluster
from elgs.energy import (
    EvidenceBatch,
    Segment,
    _capped_keys_for,
    _capped_keys_for_scan,
    _segment_is_censored_scan,
    build_evidence_index,
    phi,
    segment_is_censored,
    stream_log_likelihoods,
    total_energy,
)
from elgs.evidence import EvidenceHeads, EvidenceParams, Report

PARAMS = EvidenceParams(
    pi_miss_vis=0.2,
    pi_miss_cens=0.6,
    pi_miss_out=0.4,
    d_img_x0=0.0,
    d_img_x1=10.0,
    d_img_y0=0.0,
    d_img_y1=8.0,
    g_pos_sigma=1.5,
    v_min=0.0,
    v_max=1.0,
    r_min=0.05,
    h_floor=0.01,
    pi_floor=0.01,
    g_cap=50.0,
    pos_cap=50.0,
)

HEADS = EvidenceHeads(
    params=PARAMS,
    g_v=lambda v: 2.0 * v if v > 0 else 0.0,
    h_c=lambda v: 1.0,
    h_o=lambda v: 1.0,
)


def _deterministic_q(bridge: int, j: int, c: int, t: float) -> float:
    """A spread of q values in [0,1] that depends on every coordinate,
    so a bridge/track/camera/frame mix-up cannot pass unnoticed."""
    raw = math.sin(1.7 * bridge + 2.3 * j + 0.9 * c + 0.41 * t)
    return round(0.5 * (raw + 1.0), 9)


def build_batch(
    *,
    n_tracks: int = 4,
    cameras: tuple[int, ...] = (0, 2, 5, 7),
    frames: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0),
    n_bridges: int = 3,
    zero_before: float = 1.0,
    drop_q_keys: frozenset = frozenset(),
    drop_report_keys: frozenset = frozenset(),
) -> EvidenceBatch:
    """A batch with several tracks, cameras, frames and bridges.

    Frames strictly below `zero_before` carry q == 0.0 under EVERY
    bridge, so they form a genuinely censored span; everything at or
    after it is active. `drop_q_keys` removes keys from the q maps only
    (exercising the `(j, c, t) in q_b` guard), `drop_report_keys`
    removes them from the reports (exercising ragged camera coverage).
    """
    reports = {}
    q_tilde = {b: {} for b in range(n_bridges)}
    centers = {b: {} for b in range(n_bridges)}
    for j in range(n_tracks):
        for t in frames:
            for c in cameras:
                key = (j, c, t)
                if key in drop_report_keys:
                    continue
                reports[key] = Report(
                    is_miss=((j + c) % 5 == 0),
                    v=0.3 + 0.06 * ((j + c) % 8),
                    x=1.0 + (c % 4),
                    y=1.0 + (j % 5),
                )
                for b in range(n_bridges):
                    if key in drop_q_keys:
                        continue
                    q_tilde[b][key] = (
                        0.0 if t < zero_before else _deterministic_q(b, j, c, t)
                    )
                    centers[b][key] = (1.2 + 0.3 * b + (c % 4), 1.4 + 0.2 * b + (j % 5))
    return EvidenceBatch(reports=reports, q_tilde=q_tilde, bridge_centers=centers)


def build_clusters(n_tracks: int, n_clusters: int):
    """`n_clusters` clusters partitioning `n_tracks` tracks round-robin.

    Distinct r_u and alpha_u per cluster so a cluster mix-up changes the
    numbers rather than cancelling out.
    """
    assignment: dict[int, list[int]] = {u: [] for u in range(n_clusters)}
    for j in range(n_tracks):
        assignment[j % n_clusters].append(j)
    clusters = tuple(
        SeedCluster(
            cluster_id=u,
            seed_ids=(u,),
            canonical_point=(0.1 * u, 0.2 * u, 0.3 * u),
            r_u=0.5 + 0.1 * u,
            d_u=0.5,
            n_cam=2,
            alpha_u=0.4 + 0.15 * u,
        )
        for u in range(n_clusters)
    )
    return clusters, {u: tuple(v) for u, v in assignment.items()}


def stream_log_likelihoods_scan(batch, clusters, tracks_of_cluster, segments, heads, c_cap):
    """REFERENCE reduction: the pre-index loop, driven entirely by the
    scanning oracles and with no memoization. Independent of the
    production path except for the shared per-report likelihood calls,
    which this change does not touch."""
    from elgs.energy import _p_floor
    from elgs.evidence import likelihood_l0, likelihood_l1

    batch.validate_bridge_independent_eligibility()
    scored, censored = {}, {}
    for bridge_id in batch.bridges():
        l_scored = 0.0
        l_cens = 0.0
        for cluster in clusters:
            track_ids = tuple(tracks_of_cluster.get(cluster.cluster_id, ()))
            if not track_ids:
                continue
            r_u = cluster.r_u
            for segment in segments:
                if _segment_is_censored_scan(batch, track_ids, segment):
                    continue
                for key in _capped_keys_for_scan(
                    batch, bridge_id, track_ids, segment, c_cap
                ):
                    report = batch.reports[key]
                    q = batch.q_tilde[bridge_id][key]
                    if segment.z:
                        center = batch.bridge_centers[bridge_id][key]
                        l1 = likelihood_l1(report, center, q, r_u, heads)
                    else:
                        l1 = likelihood_l0(report, r_u, heads)
                    l0 = likelihood_l0(report, r_u, heads)
                    floor = _p_floor(heads)
                    l_scored += cluster.alpha_u * math.log(max(l1, floor))
                    l_cens += cluster.alpha_u * math.log(max(l0, floor))
        scored[bridge_id] = l_scored
        censored[bridge_id] = l_cens
    from elgs.energy import StreamLogLik

    return StreamLogLik(scored=scored, censored=censored)


class CappedKeyParityTests(unittest.TestCase):
    """`_capped_keys_for` must equal its scanning oracle as an ORDERED
    tuple — accumulation order in `stream_log_likelihoods` is float
    addition order, so set equality would not be enough."""

    def _assert_parity(self, batch, track_ids, segment, c_cap):
        index = build_evidence_index(batch)
        for bridge_id in batch.bridges():
            expected = _capped_keys_for_scan(batch, bridge_id, track_ids, segment, c_cap)
            with_index = _capped_keys_for(
                batch, bridge_id, track_ids, segment, c_cap, index
            )
            without_index = _capped_keys_for(
                batch, bridge_id, track_ids, segment, c_cap
            )
            self.assertEqual(with_index, expected, f"bridge {bridge_id}")
            self.assertEqual(without_index, expected, f"bridge {bridge_id} (no index)")

    def test_multiple_tracks_cameras_frames_and_bridges(self):
        batch = build_batch()
        segment = Segment(frames=(1.0, 2.0, 3.0, 4.0), z=True)
        for c_cap in (1, 2, 3, 4, 9):
            with self.subTest(c_cap=c_cap):
                self._assert_parity(batch, (0, 1, 2, 3), segment, c_cap)

    def test_cap_larger_than_available_cameras(self):
        batch = build_batch(cameras=(3,))
        self._assert_parity(batch, (0, 1), Segment(frames=(2.0,), z=True), 8)

    def test_duplicate_track_ids_duplicate_the_keys(self):
        """`track_ids` is a Sequence, not a set. A repeated track really
        does contribute its reports twice in the oracle, and the indexed
        path must reproduce that rather than silently deduplicating."""
        batch = build_batch(n_tracks=2)
        segment = Segment(frames=(2.0, 3.0), z=True)
        keys = _capped_keys_for_scan(batch, 0, (1, 1), segment, 2)
        self.assertEqual(len(keys), 2 * len(_capped_keys_for_scan(batch, 0, (1,), segment, 2)))
        self._assert_parity(batch, (1, 1), segment, 2)
        self._assert_parity(batch, (0, 1, 0), segment, 3)

    def test_repeated_frames_in_a_segment(self):
        batch = build_batch(n_tracks=2)
        self._assert_parity(batch, (0, 1), Segment(frames=(2.0, 2.0, 3.0), z=True), 2)

    def test_missing_q_keys_are_skipped_identically(self):
        """A report with no q entry under a bridge is ineligible. The
        indexed path must apply the same `(j, c, t) in q_b` guard."""
        dropped = frozenset({(0, 2, 2.0), (1, 5, 3.0), (2, 0, 4.0)})
        batch = build_batch(drop_q_keys=dropped)
        segment = Segment(frames=(1.0, 2.0, 3.0, 4.0), z=True)
        self._assert_parity(batch, (0, 1, 2, 3), segment, 2)
        # the guard is load-bearing: at C_cap covering everything, the
        # dropped keys must be absent from the result.
        index = build_evidence_index(batch)
        keys = set(_capped_keys_for(batch, 0, (0, 1, 2, 3), segment, 99, index))
        self.assertEqual(keys & dropped, set())

    def test_ragged_camera_coverage(self):
        """Not every (track, frame) sees every camera."""
        dropped = frozenset(
            {(0, 0, 1.0), (0, 2, 1.0), (0, 5, 1.0), (0, 7, 1.0), (1, 7, 3.0)}
        )
        batch = build_batch(drop_report_keys=dropped)
        self._assert_parity(
            batch, (0, 1, 2), Segment(frames=(1.0, 2.0, 3.0), z=True), 2
        )

    def test_track_or_frame_absent_from_every_report(self):
        """A (track, frame) with no reports at all contributes nothing —
        the `if not cams: continue` limb."""
        batch = build_batch(n_tracks=2, frames=(1.0, 2.0))
        self._assert_parity(batch, (0, 1, 99), Segment(frames=(1.0, 2.0), z=True), 2)
        self._assert_parity(batch, (0, 1), Segment(frames=(1.0, 42.0), z=True), 2)
        index = build_evidence_index(batch)
        self.assertEqual(
            _capped_keys_for(batch, 0, (99,), Segment(frames=(7.0,), z=True), 2, index),
            (),
        )

    def test_frame_and_segment_boundaries(self):
        """First frame, last frame, and the exact censored/active seam."""
        batch = build_batch(zero_before=2.0)
        for frames in ((0.0,), (4.0,), (1.0, 2.0), (0.0, 1.0), (3.0, 4.0)):
            with self.subTest(frames=frames):
                self._assert_parity(batch, (0, 1, 2, 3), Segment(frames=frames, z=True), 2)

    def test_empty_report_set(self):
        batch = EvidenceBatch(reports={}, q_tilde={0: {}, 1: {}}, bridge_centers={0: {}, 1: {}})
        index = build_evidence_index(batch)
        segment = Segment(frames=(0.0, 1.0), z=True)
        self.assertEqual(_capped_keys_for(batch, 0, (0, 1), segment, 2, index), ())
        self.assertEqual(_capped_keys_for_scan(batch, 0, (0, 1), segment, 2), ())
        self.assertTrue(segment_is_censored(batch, (0, 1), segment, index))
        self.assertTrue(_segment_is_censored_scan(batch, (0, 1), segment))


class CensoringParityTests(unittest.TestCase):
    def _assert_parity(self, batch, track_ids, segment):
        index = build_evidence_index(batch)
        expected = _segment_is_censored_scan(batch, track_ids, segment)
        self.assertIs(segment_is_censored(batch, track_ids, segment, index), expected)
        self.assertIs(segment_is_censored(batch, track_ids, segment), expected)
        return expected

    def test_fully_censored_span_detected(self):
        batch = build_batch(zero_before=2.0)
        self.assertTrue(self._assert_parity(batch, (0, 1, 2, 3), Segment(frames=(0.0, 1.0), z=True)))

    def test_active_span_not_censored(self):
        batch = build_batch(zero_before=2.0)
        self.assertFalse(self._assert_parity(batch, (0, 1, 2, 3), Segment(frames=(2.0, 3.0), z=True)))

    def test_mixed_span_is_not_censored(self):
        """One nonzero q anywhere in the span, under any bridge, breaks
        censoring — the oracle short-circuits on the first one."""
        batch = build_batch(zero_before=2.0)
        self.assertFalse(self._assert_parity(batch, (0, 1), Segment(frames=(1.0, 2.0), z=True)))

    def test_boundary_frames(self):
        batch = build_batch(zero_before=2.0)
        self.assertTrue(self._assert_parity(batch, (0,), Segment(frames=(1.0,), z=True)))
        self.assertFalse(self._assert_parity(batch, (0,), Segment(frames=(2.0,), z=True)))

    def test_track_subset_can_be_censored_while_another_is_not(self):
        """Censoring is per (track set, segment). Zeroing one track's q
        must not censor a segment scored for a different track."""
        batch = build_batch(n_tracks=3, zero_before=0.0)
        q = {b: dict(m) for b, m in batch.q_tilde.items()}
        for b in q:
            for key in list(q[b]):
                if key[0] == 1:
                    q[b][key] = 0.0
        batch = EvidenceBatch(
            reports=batch.reports, q_tilde=q, bridge_centers=batch.bridge_centers
        )
        segment = Segment(frames=(2.0, 3.0), z=True)
        self.assertTrue(self._assert_parity(batch, (1,), segment))
        self.assertFalse(self._assert_parity(batch, (0,), segment))
        self.assertFalse(self._assert_parity(batch, (0, 1), segment))

    def test_censoring_reads_q_maps_not_report_keys(self):
        """The oracle scans `q_b.items()`, so a q entry with no matching
        report still counts. The index is built from the q maps for
        exactly this reason."""
        batch = build_batch(n_tracks=1, cameras=(0,), frames=(0.0,), zero_before=99.0)
        q = {b: dict(m) for b, m in batch.q_tilde.items()}
        q[0][(0, 3, 0.0)] = 0.75  # no report at this key
        batch = EvidenceBatch(
            reports=batch.reports, q_tilde=q, bridge_centers=batch.bridge_centers
        )
        self.assertFalse(self._assert_parity(batch, (0,), Segment(frames=(0.0,), z=True)))

    def test_no_bridges_fails_closed_on_both_paths(self):
        batch = EvidenceBatch(reports={}, q_tilde={}, bridge_centers={})
        segment = Segment(frames=(0.0,), z=True)
        with self.assertRaises(ContractError):
            _segment_is_censored_scan(batch, (0,), segment)
        with self.assertRaises(ContractError):
            segment_is_censored(batch, (0,), segment)
        with self.assertRaises(ContractError):
            build_evidence_index(batch)


class DownstreamEnergyParityTests(unittest.TestCase):
    """Parity must survive all the way to the numbers acceptance reads:
    the two stream dicts, Phi, and the total energy."""

    def _fixture(self, *, n_tracks=4, n_clusters=2, zero_before=2.0, n_bridges=3):
        batch = build_batch(
            n_tracks=n_tracks, zero_before=zero_before, n_bridges=n_bridges
        )
        clusters, tracks = build_clusters(n_tracks, n_clusters)
        segments = (
            Segment(frames=(0.0, 1.0), z=True),  # censored span
            Segment(frames=(2.0, 3.0), z=True),  # active, present
            Segment(frames=(4.0,), z=False),  # active, absent
        )
        return batch, clusters, tracks, segments

    def test_stream_log_likelihoods_are_bit_identical(self):
        batch, clusters, tracks, segments = self._fixture()
        for c_cap in (1, 2, 4):
            with self.subTest(c_cap=c_cap):
                got = stream_log_likelihoods(batch, clusters, tracks, segments, HEADS, c_cap)
                ref = stream_log_likelihoods_scan(
                    batch, clusters, tracks, segments, HEADS, c_cap
                )
                self.assertEqual(sorted(got.scored), sorted(ref.scored))
                for b in ref.scored:
                    self.assertEqual(got.scored[b], ref.scored[b], f"scored bridge {b}")
                    self.assertEqual(got.censored[b], ref.censored[b], f"censored bridge {b}")

    def test_phi_is_bit_identical(self):
        batch, clusters, tracks, segments = self._fixture()
        got = phi(stream_log_likelihoods(batch, clusters, tracks, segments, HEADS, 2), 0.7)
        ref = phi(
            stream_log_likelihoods_scan(batch, clusters, tracks, segments, HEADS, 2), 0.7
        )
        self.assertEqual(got, ref)
        self.assertTrue(math.isfinite(got))
        self.assertNotEqual(got, 0.0)

    def test_total_energy_is_bit_identical(self):
        """The acceptance path consumes Phi through `total_energy`; pin
        the value acceptance would actually see."""
        batch, clusters, tracks, segments = self._fixture()
        got = phi(stream_log_likelihoods(batch, clusters, tracks, segments, HEADS, 2), 0.7)
        ref = phi(
            stream_log_likelihoods_scan(batch, clusters, tracks, segments, HEADS, 2), 0.7
        )
        self.assertEqual(
            total_energy(1.25, 0.1, [got], [0.3, 0.4], 0.05),
            total_energy(1.25, 0.1, [ref], [0.3, 0.4], 0.05),
        )

    def test_exact_delta_between_two_presence_hypotheses_is_identical(self):
        """`evidence_exact_delta` is beta*(Phi_candidate - Phi_incumbent).
        Both terms come from this path, so pin the DIFFERENCE too — a
        bias common to both would cancel and hide there."""
        batch, clusters, tracks, _ = self._fixture()
        incumbent = (Segment(frames=(0.0, 1.0), z=True), Segment(frames=(2.0, 3.0, 4.0), z=True))
        candidate = (
            Segment(frames=(0.0, 1.0), z=True),
            Segment(frames=(2.0,), z=True),
            Segment(frames=(3.0, 4.0), z=False),
        )
        def _phi(segments, fn):
            return phi(fn(batch, clusters, tracks, segments, HEADS, 2), 0.7)

        got = _phi(candidate, stream_log_likelihoods) - _phi(incumbent, stream_log_likelihoods)
        ref = _phi(candidate, stream_log_likelihoods_scan) - _phi(
            incumbent, stream_log_likelihoods_scan
        )
        self.assertEqual(got, ref)
        self.assertNotEqual(got, 0.0)

    def test_multiple_families_and_bridge_counts(self):
        for n_bridges in (1, 2, 5):
            for n_clusters in (1, 3, 4):
                with self.subTest(bridges=n_bridges, clusters=n_clusters):
                    batch, clusters, tracks, segments = self._fixture(
                        n_clusters=n_clusters, n_bridges=n_bridges
                    )
                    got = stream_log_likelihoods(batch, clusters, tracks, segments, HEADS, 2)
                    ref = stream_log_likelihoods_scan(
                        batch, clusters, tracks, segments, HEADS, 2
                    )
                    self.assertEqual(got.scored, ref.scored)
                    self.assertEqual(got.censored, ref.censored)

    def test_all_segments_censored_still_gives_exactly_zero(self):
        """PROP 1 through the indexed path: a wholly censored window
        contributes exactly 0.0, not a small float residue."""
        batch, clusters, tracks, _ = self._fixture(zero_before=99.0)
        segments = (Segment(frames=(0.0, 1.0, 2.0, 3.0, 4.0), z=True),)
        got = phi(stream_log_likelihoods(batch, clusters, tracks, segments, HEADS, 2), 0.7)
        self.assertEqual(got, 0.0)
        self.assertEqual(
            got, phi(stream_log_likelihoods_scan(batch, clusters, tracks, segments, HEADS, 2), 0.7)
        )

    def test_cluster_with_no_tracks_is_skipped_identically(self):
        batch, clusters, tracks, segments = self._fixture()
        tracks = dict(tracks)
        tracks[1] = ()
        got = stream_log_likelihoods(batch, clusters, tracks, segments, HEADS, 2)
        ref = stream_log_likelihoods_scan(batch, clusters, tracks, segments, HEADS, 2)
        self.assertEqual(got.scored, ref.scored)
        self.assertEqual(got.censored, ref.censored)

    def test_eligibility_validation_still_fires(self):
        """The index must not paper over a bridge whose q map does not
        cover the report key set — that check is the PROP 1 condition."""
        batch = build_batch(n_tracks=2, drop_q_keys=frozenset({(0, 0, 2.0)}))
        clusters, tracks = build_clusters(2, 1)
        with self.assertRaises(ContractError):
            stream_log_likelihoods(
                batch, clusters, tracks, (Segment(frames=(2.0,), z=True),), HEADS, 2
            )


class IndexStructureTests(unittest.TestCase):
    def test_camera_map_matches_the_report_keys_exactly(self):
        batch = build_batch(n_tracks=3)
        index = build_evidence_index(batch)
        rebuilt = set()
        for (j, t), cams in index.cameras_by_track_frame.items():
            self.assertEqual(len(set(cams)), len(cams), "cameras must not repeat")
            for c in cams:
                rebuilt.add((j, c, t))
        self.assertEqual(rebuilt, set(batch.reports))

    def test_camera_map_preserves_report_iteration_order(self):
        batch = build_batch(n_tracks=2, cameras=(7, 2, 5, 0))
        index = build_evidence_index(batch)
        for (j, t), cams in index.cameras_by_track_frame.items():
            expected = tuple(c for (jj, c, tt) in batch.reports if jj == j and tt == t)
            self.assertEqual(cams, expected)

    def test_nonzero_set_matches_a_direct_reduction(self):
        batch = build_batch(n_tracks=3, zero_before=2.0)
        index = build_evidence_index(batch)
        expected = {
            (j, t)
            for b in batch.bridges()
            for (j, _c, t), q in batch.q_tilde[b].items()
            if q != 0.0
        }
        self.assertEqual(set(index.nonzero_track_frames), expected)

    def test_energy_layer_is_device_free(self):
        """elgs.energy holds no tensors and selects no device, so there
        is no CPU/GPU behaviour to diverge at this layer. Pinned rather
        than assumed: if a tensor ever enters this module, the missing
        GPU test becomes a real gap and this fails."""
        import elgs.energy as energy

        self.assertNotIn("torch", vars(energy))
        source = open(energy.__file__, encoding="utf-8").read()
        self.assertNotIn("import torch", source)


if __name__ == "__main__":
    unittest.main()
