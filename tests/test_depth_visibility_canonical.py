import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from depth_visibility.artifacts import (
    atomic_write_json_immutable,
    build_inventory,
    load_verified_array,
    write_canonical_array,
    write_terminal_last,
)
from depth_visibility.canonical import (
    array_semantic_ref,
    binary64_hex,
    canonical_json_bytes,
    canonicalize,
    domain_id,
    sha256_file,
    verify_array_ref,
)
from depth_visibility.errors import ArtifactError, NonFiniteError, SchemaError
from depth_visibility.n3v import compute_r_scene, load_scene_index, validate_split_binding
from depth_visibility.schema import load_config, validate_config


class CanonicalTests(unittest.TestCase):
    def test_binary64_signed_zero_and_float_identity(self):
        self.assertEqual(binary64_hex(0.0), "0000000000000000")
        self.assertEqual(binary64_hex(-0.0), "8000000000000000")
        self.assertEqual(canonicalize({"x": 1.5}), {"x": "3ff8000000000000"})
        with self.assertRaises(SchemaError):
            binary64_hex(True)
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(NonFiniteError):
                binary64_hex(value)

    def test_dict_order_domain_separation_and_nfc(self):
        left = canonical_json_bytes({"b": 2, "a": 1.0})
        right = canonical_json_bytes({"a": 1.0, "b": 2})
        self.assertEqual(left, right)
        self.assertNotEqual(domain_id("domain-a", {"x": 1}), domain_id("domain-b", {"x": 1}))
        self.assertEqual(canonicalize("café"), "café")
        with self.assertRaises(SchemaError):
            canonicalize("café")
        with self.assertRaises(SchemaError):
            canonicalize(np.zeros(1))

    def test_array_endian_shape_and_semantic_hash(self):
        little = np.array([[1.0, 2.5]], dtype="<f4")
        big = little.astype(">f4")
        little_ref = array_semantic_ref(little, "fixture/depth")
        big_ref = array_semantic_ref(big, "fixture/depth")
        self.assertEqual(little_ref, big_ref)
        self.assertEqual(little_ref["dtype"], "<f4")
        self.assertEqual(little_ref["byte_order"], "little")
        verify_array_ref(little, little_ref)
        changed = little.copy()
        changed[0, 0] = 9.0
        with self.assertRaises(SchemaError):
            verify_array_ref(changed, little_ref)
        with self.assertRaises(NonFiniteError):
            array_semantic_ref(np.array([np.nan]), "fixture/nonfinite")

    def test_artifact_roundtrip_is_stable_and_terminal_is_last(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            array = np.arange(12, dtype=np.float32).reshape(3, 4)
            first = write_canonical_array(root / "a.npy", array, "fixture/a", relative_to=root)
            second = write_canonical_array(root / "b.npy", array, "fixture/a", relative_to=root)
            self.assertEqual(first["file_sha256"], second["file_sha256"])
            np.testing.assert_array_equal(load_verified_array(root / "a.npy", first), array)
            inventory = build_inventory(root)
            self.assertEqual([item["path"] for item in inventory], ["a.npy", "b.npy"])
            terminal = root / "terminal.json"
            write_terminal_last(
                terminal,
                {"schema_version": "phase9-terminal-v1", "status": "succeeded"},
                [root / "a.npy", root / "b.npy"],
            )
            self.assertTrue(terminal.is_file())
            with self.assertRaises(ArtifactError):
                write_terminal_last(terminal, {"status": "again"}, [root / "a.npy"])
            immutable = root / "stage1-ledger.json"
            atomic_write_json_immutable(immutable, {"risk": 0.25})
            self.assertEqual(json.loads(immutable.read_text(encoding="utf-8"))["risk"], 0.25)
            with self.assertRaises(ArtifactError):
                atomic_write_json_immutable(immutable, {"risk": 0.5})


    def test_frozen_config_strictness(self):
        path = Path("configs/depth_visibility/csvl_isr_v1.json")
        config = load_config(path, expected_sha256=sha256_file(path))
        self.assertEqual(config["method_id"], "csvl-isr-v1")
        changed = copy.deepcopy(config)
        changed["unexpected"] = True
        with self.assertRaises(SchemaError):
            validate_config(changed)
        changed = copy.deepcopy(config)
        changed["da3"]["process_res"] = 512
        with self.assertRaises(SchemaError):
            validate_config(changed)

    def test_split_source_binding_does_not_claim_legacy_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fixture_scene"
            (root / "images").mkdir(parents=True)
            for stem in ("cam01_0000", "cam02_0000"):
                (root / "images" / f"{stem}.png").write_bytes(b"fixture")
            base = {
                "w": 64,
                "h": 48,
                "fl_x": 50.0,
                "fl_y": 52.0,
                "cx": 31.5,
                "cy": 23.5,
            }
            def frame(camera, x):
                matrix = np.eye(4)
                matrix[0, 3] = x
                return {
                    "file_path": f"images/{camera}_0000",
                    "transform_matrix": matrix.tolist(),
                    "time": 0.0,
                }
            train = {**base, "frames": [frame("cam01", 0.0), frame("cam02", 1.0)]}
            test = {**base, "frames": [frame("cam00", 0.5)]}
            (root / "transforms_train.json").write_text(json.dumps(train), encoding="utf-8")
            (root / "transforms_test.json").write_text(json.dumps(test), encoding="utf-8")
            index = load_scene_index(root)
            manifest = {
                "schema_version": "n3v-split-v1",
                "dataset_root_env": "$WORK/fixture",
                "scenes": {
                    "fixture_scene": {
                        "train": {
                            "camera_ids": ["cam01", "cam02"],
                            "record_count": 2,
                            "record_identity_sha256": "0" * 64,
                            "source_path": "transforms_train.json",
                            "source_sha256": sha256_file(root / "transforms_train.json"),
                        },
                        "test": {
                            "camera_ids": ["cam00"],
                            "record_count": 1,
                            "record_identity_sha256": "f" * 64,
                            "source_path": "transforms_test.json",
                            "source_sha256": sha256_file(root / "transforms_test.json"),
                        },
                    }
                },
            }
            report = validate_split_binding(index, manifest)
            self.assertEqual(report["train"]["legacy_record_identity_status"], "unverified_encoder")
            self.assertNotEqual(
                report["train"]["canonical_record_identity_sha256"],
                manifest["scenes"]["fixture_scene"]["train"]["record_identity_sha256"],
            )
            self.assertIsNone(index.split("test")[0].image_path)
            calibration_only = load_scene_index(
                root, expose_train_images=False, expose_test_images=False
            )
            self.assertTrue(all(record.image_path is None for record in calibration_only.split("train")))
            self.assertTrue(all(record.image_path is None for record in calibration_only.split("test")))
            self.assertAlmostEqual(compute_r_scene(list(index.split("train"))), 0.5)


if __name__ == "__main__":
    unittest.main()
