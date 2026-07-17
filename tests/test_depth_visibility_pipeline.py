import unittest

from depth_visibility.errors import ProvenanceError
from depth_visibility.fixtures import two_plane_track_pixels
from depth_visibility.ledger import (
    build_scene_ledger,
    build_target_frame_ledger,
    freeze_train_sidecars,
    recursive_provenance_check,
    seal_scene_ledger,
)


def complete_provenance():
    cameras = [f"cam{index:02d}" for index in range(1, 8)]
    return {
        "physical_ancestry": cameras,
        "dependencies": [
            {
                "kind": "source_nodes",
                "physical_ancestry": cameras[:3],
                "payload": {"array_hash": "a" * 64},
            },
            {
                "kind": "fused_hypotheses",
                "physical_ancestry": cameras[3:6],
                "payload": {"array_hash": "b" * 64},
            },
            {
                "kind": "witness",
                "physical_ancestry": cameras[6:],
                "payload": {"array_hash": "c" * 64},
            },
        ],
    }


class ProvenanceTests(unittest.TestCase):
    def test_recursive_union_and_target_exclusion(self):
        provenance = complete_provenance()
        ancestry = recursive_provenance_check(
            provenance,
            scored_target="cam00",
            target_image_hashes=["f" * 64],
        )
        self.assertEqual(ancestry, {f"cam{index:02d}" for index in range(1, 8)})

        contaminated = complete_provenance()
        contaminated["dependencies"][0]["physical_ancestry"].append("cam00")
        with self.assertRaises(ProvenanceError):
            recursive_provenance_check(contaminated, scored_target="cam00")

        missing = complete_provenance()
        del missing["dependencies"][0]["physical_ancestry"]
        with self.assertRaises(ProvenanceError):
            recursive_provenance_check(missing, scored_target="cam00")

        leaked_hash = complete_provenance()
        leaked_hash["dependencies"][0]["payload"]["nested"] = {"value": "f" * 64}
        with self.assertRaises(ProvenanceError):
            recursive_provenance_check(
                leaked_hash,
                scored_target="cam00",
                target_image_hashes=["f" * 64],
            )

    def test_cyclic_provenance_fails_closed(self):
        provenance = {"physical_ancestry": ["cam01"]}
        provenance["dependencies"] = [provenance]
        with self.assertRaises(ProvenanceError):
            recursive_provenance_check(provenance, scored_target="cam00")


class LedgerPipelineTests(unittest.TestCase):
    def test_frame_scene_seal_and_freeze_are_deterministic(self):
        tracks, witnesses = two_plane_track_pixels()
        provenance = complete_provenance()
        first = build_target_frame_ledger(
            scene="fixture",
            frame=0,
            scored_target="cam00",
            track_pixels=tracks,
            visible_witnesses=witnesses,
            provenance=provenance,
            target_image_hashes=["f" * 64],
        )
        second = build_target_frame_ledger(
            scene="fixture",
            frame=0,
            scored_target="cam00",
            track_pixels=dict(reversed(list(tracks.items()))),
            visible_witnesses=witnesses,
            provenance=provenance,
            target_image_hashes=["f" * 64],
        )
        self.assertEqual(first["ledger_id"], second["ledger_id"])
        self.assertEqual(
            {(region["track_id"], region["state"]) for region in first["regions"]},
            {("front", "visible"), ("rear", "occluded")},
        )

        scene = build_scene_ledger([first])
        seal = seal_scene_ledger(scene)
        self.assertTrue(seal["sealed"])
        self.assertEqual(seal["scene_ledger_id"], scene["scene_ledger_id"])
        freeze = freeze_train_sidecars(scene)
        self.assertEqual(freeze["scene_ledger_id"], scene["scene_ledger_id"])
        self.assertIn("human_labels", freeze["prohibited_read_proof"])

    def test_ledger_rejects_unsealed_track_or_witness_ancestry(self):
        tracks, witnesses = two_plane_track_pixels()
        provenance = complete_provenance()
        tracks["front"][(8, 10)]["physical_ancestry"] = ("cam99",)
        with self.assertRaises(ProvenanceError):
            build_target_frame_ledger(
                scene="fixture",
                frame=0,
                scored_target="cam00",
                track_pixels=tracks,
                visible_witnesses=witnesses,
                provenance=provenance,
            )

    def test_train_freeze_rejects_reference_side_information(self):
        scene = {
            "scene": "fixture",
            "scene_ledger_id": "ledger",
            "frame_ledger_ids": ["frame"],
            "frames": [{"R009_source_sha": "historical"}],
        }
        with self.assertRaises(ProvenanceError):
            freeze_train_sidecars(scene)
        scene["frames"] = [{"human_labels": []}]
        with self.assertRaises(ProvenanceError):
            freeze_train_sidecars(scene)


if __name__ == "__main__":
    unittest.main()
