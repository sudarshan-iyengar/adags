"""B0 tracker-pipeline dry run on synthetic fixtures (experiment plan
B0; the real CoTracker builder lands with M1 and must emit exactly
this schema). CPU only, unittest."""

import unittest

from depth_visibility.errors import ContractError, SchemaError
from elgs.tracks_schema import (
    TRACKS_SCHEMA,
    build_synthetic_fixture,
    validate_tracks_artifact,
)


class DryRunTests(unittest.TestCase):
    def test_fixture_builds_and_validates(self):
        artifact = validate_tracks_artifact(build_synthetic_fixture())
        self.assertEqual(artifact["schema_version"], TRACKS_SCHEMA)
        self.assertGreaterEqual(len(artifact["seeds"]), 4)
        self.assertTrue(all(s["n_cam"] >= 2 for s in artifact["seeds"]))
        # Identity by common seed: every track resolves to one seed.
        seed_ids = {s["seed_id"] for s in artifact["seeds"]}
        self.assertTrue(all(t["seed_id"] in seed_ids for t in artifact["tracks"]))

    def test_fixture_is_deterministic(self):
        self.assertEqual(build_synthetic_fixture(), build_synthetic_fixture())

    def test_held_out_camera_leak_rejected(self):
        artifact = build_synthetic_fixture()
        artifact["tracks"][0]["camera_id"] = 4  # the held-out camera
        with self.assertRaises(ContractError):
            validate_tracks_artifact(artifact)

    def test_single_camera_seed_rejected(self):
        artifact = build_synthetic_fixture()
        artifact["seeds"][0]["n_cam"] = 1
        with self.assertRaises(ContractError):
            validate_tracks_artifact(artifact)

    def test_schema_violations_rejected(self):
        with self.assertRaises(SchemaError):
            validate_tracks_artifact({"schema_version": "wrong"})
        artifact = build_synthetic_fixture()
        del artifact["manifest"]["held_out_cameras"]
        with self.assertRaises(SchemaError):
            validate_tracks_artifact(artifact)
        artifact2 = build_synthetic_fixture()
        artifact2["tracks"][0]["seed_id"] = 999
        with self.assertRaises(ContractError):
            validate_tracks_artifact(artifact2)

    def test_duplicate_ids_rejected(self):
        artifact = build_synthetic_fixture()
        artifact["tracks"][1]["track_id"] = artifact["tracks"][0]["track_id"]
        with self.assertRaises(ContractError):
            validate_tracks_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
