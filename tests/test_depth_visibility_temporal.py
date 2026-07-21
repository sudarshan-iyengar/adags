import unittest

import numpy as np

from depth_visibility.errors import FlowSemanticsError
from depth_visibility.fixtures import (
    constant_flow,
    flow_manifest,
    temporal_patch_candidates,
)
from depth_visibility.flow import (
    adapt_declared_flow,
    forward_backward_cycle,
    reciprocal_node_flow_matches,
    validate_flow_manifest,
)
from depth_visibility.tracking import (
    advance_dormant_tracks,
    aggregate_track_time,
    derive_states,
    identity_risk,
    patch_pair_terms,
    propagate_tracks,
    reciprocal_patch_edges,
    reidentify_track,
    split_merge_components,
    transition_risk,
)


class FlowTests(unittest.TestCase):
    def test_manifest_direction_and_resize_are_fail_closed(self):
        forward_record = validate_flow_manifest(flow_manifest())
        backward_record = validate_flow_manifest(
            flow_manifest(direction="backward_t_to_t_minus_1")
        )
        self.assertEqual(forward_record.source_frame, 0)
        self.assertEqual(backward_record.source_frame, 1)

        invalid = flow_manifest()
        invalid["direction"] = "backward_t_to_t_minus_1"
        with self.assertRaises(FlowSemanticsError):
            validate_flow_manifest(invalid)
        invalid = flow_manifest()
        invalid["occlusion_semantics"] = "unknown"
        with self.assertRaises(FlowSemanticsError):
            validate_flow_manifest(invalid)

        small_record = validate_flow_manifest(flow_manifest(height=2, width=4))
        resized = adapt_declared_flow(
            constant_flow(2, 4, 1.0, 0.5),
            small_record,
            (4, 8),
        )
        np.testing.assert_allclose(resized[..., 0], 2.0)
        np.testing.assert_allclose(resized[..., 1], 1.0)

    def test_forward_backward_cycle_and_mutual_nearest(self):
        forward = constant_flow(12, 12, 1.0, 0.0)
        backward = constant_flow(12, 12, -1.0, 0.0)
        cycle = forward_backward_cycle((2.0, 3.0), forward, backward)
        self.assertTrue(cycle["accepted"])
        self.assertEqual(cycle["cycle_error"], 0.0)
        np.testing.assert_array_equal(cycle["destination"], [3.0, 3.0])

        sources = [
            {"node_id": "s0", "xy": [2.0, 2.0]},
            {"node_id": "s1", "xy": [5.0, 2.0]},
        ]
        destinations = [
            {"node_id": "d0", "xy": [3.0, 2.0]},
            {"node_id": "d1", "xy": [6.0, 2.0]},
        ]
        matches = reciprocal_node_flow_matches(
            sources, destinations, forward, backward
        )
        self.assertEqual(
            [(item["source_id"], item["destination_id"]) for item in matches],
            [("s0", "d0"), ("s1", "d1")],
        )
        bad_backward = constant_flow(12, 12, 1.0, 0.0)
        self.assertEqual(
            reciprocal_node_flow_matches(sources, destinations, forward, bad_backward),
            [],
        )


class TrackingTests(unittest.TestCase):
    def test_temporal_terms_risk_and_identity_ratio(self):
        terms = patch_pair_terms(1.0, 0.75, 0.2, 0.01, r_scene=1.0)
        self.assertLessEqual(terms["cost"], 1.0)
        self.assertEqual(identity_risk(0.2, None), 0.0)
        self.assertEqual(identity_risk(0.0, 0.0), 1.0)
        self.assertEqual(transition_risk(0.2, 0.4, 0.3), 0.4)

    def test_reciprocal_edges_split_merge_and_propagation(self):
        candidates = temporal_patch_candidates()
        edges = reciprocal_patch_edges(candidates)
        self.assertEqual(len(edges), 1)
        insufficient = [dict(candidates[0], camera_node_match_counts={"cam01": 3})]
        self.assertEqual(reciprocal_patch_edges(insufficient), [])
        self.assertEqual(split_merge_components(insufficient), [])
        components = split_merge_components(candidates)
        self.assertFalse(components[0]["ambiguous"])
        previous = [
            {
                "patch_id": "p0",
                "track_id": "track-old",
                "scene": "fixture",
                "scored_target": "cam00",
                "frame": 0,
            }
        ]
        current = [
            {
                "patch_id": "p1",
                "scene": "fixture",
                "scored_target": "cam00",
                "frame": 1,
            }
        ]
        propagated = propagate_tracks(previous, current, edges, components)
        self.assertEqual(propagated[0]["track_id"], "track-old")
        self.assertEqual(propagated[0]["identity_state"], "propagated")
        self.assertEqual(propagated[0]["temporal_edge_id"], edges[0]["edge_id"])

        split_candidates = temporal_patch_candidates(split=True)
        split_edges = reciprocal_patch_edges(split_candidates)
        split_components = split_merge_components(split_candidates)
        self.assertTrue(split_components[0]["ambiguous"])
        split_current = [
            {
                "patch_id": patch_id,
                "scene": "fixture",
                "scored_target": "cam00",
                "frame": 1,
            }
            for patch_id in ("p1", "p2")
        ]
        ambiguous = propagate_tracks(
            previous, split_current, split_edges, split_components
        )
        self.assertTrue(
            all(item["identity_state"] == "uncertain_split_merge" for item in ambiguous)
        )
        self.assertTrue(all(item["track_id"] is None for item in ambiguous))

    def test_dormancy_reappearance_and_state_semantics(self):
        dormant = advance_dormant_tracks(
            {
                "track": {
                    "last_frame": 2,
                    "last_visible_projections": {"cam01": [2.0, 3.0]},
                    "linear_rgb": [0.2, 0.3, 0.4],
                    "normal": [0.0, 0.0, 1.0],
                    "identity_descriptor": "sealed",
                }
            },
            5,
        )
        self.assertEqual(dormant["track"]["dormant_age"], 3)
        with self.assertRaises(ValueError):
            advance_dormant_tracks(
                {"track": {"last_frame": 2, "world_point": [0, 0, 1]}},
                3,
            )

        candidate = {
            "track_id": "track",
            "camera_count": 2,
            "endpoint_error_pixels": 1.0,
            "ncc": 0.8,
            "normal_angle_degrees": 10.0,
            "rgb_l2": 0.1,
            "cost": 0.4,
            "reciprocal": True,
            "one_to_one": True,
            "complete_flow_chain": True,
        }
        match = reidentify_track([candidate])
        self.assertEqual(match["event"], "reappearance")
        self.assertEqual(match["track_id"], "track")

        self.assertEqual(derive_states("occluded", "visible"), "reveal")
        self.assertEqual(derive_states("visible", "occluded"), "hide")
        self.assertEqual(derive_states(None, "visible"), "uncertain")
        self.assertEqual(
            aggregate_track_time(["occluded", "visible"], hypothesis_exists=True),
            "observed",
        )
        self.assertEqual(
            aggregate_track_time([], hypothesis_exists=False),
            "unobserved",
        )
        self.assertEqual(
            aggregate_track_time(["occluded"], hypothesis_exists=True),
            "uncertain",
        )


if __name__ == "__main__":
    unittest.main()
