"""Unit tests for elgs/clusters.py (spec §2 ownership, rev-3 A2 binding).

CPU only, unittest. Oracles: enumerated component structures, the
lowest-ID canonical-point rule, min-aggregation on merge, and the
plan-§2.4a binding-freeze semantics.
"""

import unittest

import torch

from depth_visibility.errors import ContractError
from elgs.clusters import (
    BindingTable,
    SeedCluster,
    form_clusters,
    merge_clusters,
    nearest_family,
)


def _alpha(n_cam: int) -> float:
    # Injected stand-in correlation model (the real one is category-2
    # preregistration): alpha shrinks with camera count.
    return 1.0 / n_cam


def _inputs():
    points = {
        0: (0.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (5.0, 0.0, 0.0),
        3: (6.0, 0.0, 0.0),
        4: (9.0, 9.0, 9.0),
    }
    r = {0: 0.9, 1: 0.8, 2: 0.95, 3: 0.7, 4: 1.0}
    d = {0: 0.5, 1: 0.6, 2: 0.4, 3: 0.9, 4: 1.0}
    n_cam = {0: 2, 1: 3, 2: 2, 3: 2, 4: 5}
    return points, r, d, n_cam


class FormClustersTests(unittest.TestCase):
    def test_components_and_lowest_id_canonical_point(self):
        points, r, d, n_cam = _inputs()
        clusters = form_clusters(points, [(0, 1), (2, 3)], r, d, n_cam, _alpha)
        self.assertEqual(len(clusters), 3)
        by_seeds = {c.seed_ids: c for c in clusters}
        self.assertIn((0, 1), by_seeds)
        self.assertIn((2, 3), by_seeds)
        self.assertIn((4,), by_seeds)
        # Canonical point = surface point of the LOWEST-ID member seed.
        self.assertEqual(by_seeds[(0, 1)].canonical_point, points[0])
        self.assertEqual(by_seeds[(2, 3)].canonical_point, points[2])
        # min-aggregation for r/d; max n_cam feeds the alpha model.
        self.assertEqual(by_seeds[(0, 1)].r_u, 0.8)
        self.assertEqual(by_seeds[(0, 1)].d_u, 0.5)
        self.assertEqual(by_seeds[(0, 1)].n_cam, 3)
        self.assertAlmostEqual(by_seeds[(0, 1)].alpha_u, 1.0 / 3.0, places=12)

    def test_edge_order_does_not_change_result(self):
        points, r, d, n_cam = _inputs()
        a = form_clusters(points, [(0, 1), (2, 3)], r, d, n_cam, _alpha)
        b = form_clusters(points, [(3, 2), (1, 0)], r, d, n_cam, _alpha)
        self.assertEqual([c.seed_ids for c in a], [c.seed_ids for c in b])
        self.assertEqual([c.cluster_id for c in a], [c.cluster_id for c in b])

    def test_streams_pairwise_disjoint_by_construction(self):
        points, r, d, n_cam = _inputs()
        clusters = form_clusters(points, [(0, 1)], r, d, n_cam, _alpha)
        seen: set[int] = set()
        for c in clusters:
            self.assertFalse(seen & set(c.seed_ids))
            seen |= set(c.seed_ids)
        self.assertEqual(seen, set(points))

    def test_unknown_edge_and_missing_metadata_rejected(self):
        points, r, d, n_cam = _inputs()
        with self.assertRaises(ContractError):
            form_clusters(points, [(0, 99)], r, d, n_cam, _alpha)
        with self.assertRaises(ContractError):
            form_clusters(points, [], {0: 1.0}, d, n_cam, _alpha)


class MergeClustersTests(unittest.TestCase):
    def test_merge_reforms_with_lowest_id_point_and_min_aggregation(self):
        points, r, d, n_cam = _inputs()
        clusters = form_clusters(points, [(0, 1), (2, 3)], r, d, n_cam, _alpha)
        by_seeds = {c.seed_ids: c for c in clusters}
        merged = merge_clusters(
            [by_seeds[(2, 3)], by_seeds[(0, 1)]], points, _alpha, cluster_id=7
        )
        self.assertEqual(merged.seed_ids, (0, 1, 2, 3))
        self.assertEqual(merged.canonical_point, points[0])  # recomputed, lowest ID
        self.assertEqual(merged.r_u, 0.7)
        self.assertEqual(merged.d_u, 0.4)
        self.assertEqual(merged.n_cam, 3)

    def test_shared_seed_rejected(self):
        a = SeedCluster(0, (0, 1), (0.0, 0.0, 0.0), 1.0, 1.0, 2, 0.5)
        b = SeedCluster(1, (1, 2), (1.0, 0.0, 0.0), 1.0, 1.0, 2, 0.5)
        with self.assertRaises(ContractError):
            merge_clusters([a, b], {0: (0.0, 0.0, 0.0)}, _alpha, cluster_id=2)


class NearestFamilyTests(unittest.TestCase):
    def test_bind_within_threshold_single_valued(self):
        positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        fam = torch.tensor([3, 3, 8])
        self.assertEqual(nearest_family((0.1, 0.0, 0.0), positions, fam, 1.0), 3)
        self.assertEqual(nearest_family((9.5, 0.0, 0.0), positions, fam, 1.0), 8)

    def test_beyond_threshold_is_inactive(self):
        positions = torch.tensor([[0.0, 0.0, 0.0]])
        fam = torch.tensor([3])
        self.assertIsNone(nearest_family((5.0, 0.0, 0.0), positions, fam, 1.0))

    def test_empty_bank_is_inactive(self):
        self.assertIsNone(
            nearest_family((0.0, 0.0, 0.0), torch.zeros((0, 3)), torch.zeros((0,)), 1.0)
        )


class BindingTableTests(unittest.TestCase):
    def _table(self):
        table = BindingTable()
        table.bind(0, 10)
        table.bind(1, 10)
        table.bind(2, 20)
        table.bind(3, None)
        return table

    def test_u_of_f_inverse_map(self):
        table = self._table()
        self.assertEqual(table.clusters_of(10), (0, 1))
        self.assertEqual(table.clusters_of(20), (2,))
        self.assertIsNone(table.family_of(3))

    def test_freeze_blocks_ordinary_rebinding(self):
        table = self._table()
        table.freeze_audited()
        with self.assertRaises(ContractError):
            table.bind(0, 20)
        with self.assertRaises(ContractError):
            table.bind(9, 20)
        with self.assertRaises(ContractError):
            table.freeze_audited()

    def test_merge_redirection_permitted_after_freeze_and_logged(self):
        table = self._table()
        table.freeze_audited()
        moved = table.redirect_for_merge(retired_family=20, surviving_family=10)
        self.assertEqual(moved, (2,))
        self.assertEqual(table.clusters_of(10), (0, 1, 2))
        self.assertEqual(table.clusters_of(20), ())
        kinds = [entry["kind"] for entry in table.mutation_log]
        self.assertEqual(kinds, ["merge_redirect"])

    def test_late_birth_bind_permitted_only_for_new_clusters(self):
        table = self._table()
        table.freeze_audited()
        table.bind_late_birth(7, 30)
        self.assertEqual(table.family_of(7), 30)
        with self.assertRaises(ContractError):
            table.bind_late_birth(0, 30)  # existing cluster: forbidden
        kinds = [entry["kind"] for entry in table.mutation_log]
        self.assertEqual(kinds, ["late_birth_bind"])


class SeedClusterValidationTests(unittest.TestCase):
    def test_invalid_clusters_rejected(self):
        with self.assertRaises(ContractError):
            SeedCluster(0, (), (0.0, 0.0, 0.0), 1.0, 1.0, 1, 0.5)
        with self.assertRaises(ContractError):
            SeedCluster(0, (2, 1), (0.0, 0.0, 0.0), 1.0, 1.0, 1, 0.5)
        with self.assertRaises(ContractError):
            SeedCluster(0, (0,), (0.0, 0.0, 0.0), 1.0, 1.5, 1, 0.5)
        with self.assertRaises(ContractError):
            SeedCluster(0, (0,), (0.0, 0.0, 0.0), 1.0, 1.0, 1, 0.0)


if __name__ == "__main__":
    unittest.main()
