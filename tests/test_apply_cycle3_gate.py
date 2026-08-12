"""CPU tests for scripts/apply_cycle3_gate.py (split G-R/G-OA verdicts).

Run with:
    C:/Users/sucar/venvs/elgs-cpu/Scripts/python.exe -m unittest tests.test_apply_cycle3_gate

Fixtures are hand-built census artifacts with exactly the fields the
gate reads (halves attribution, per-half coverage tallies, candidate
records), so every verdict has a closed-form expected value.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402
from scripts import apply_cycle3_gate as gate  # noqa: E402


def _census(
    *,
    split: int,
    second_union: int,
    second_primary: int,
    second_occl: int,
    second_absence: int,
    second_cov: tuple[int, int],
    identities: int = 2,
) -> dict:
    """One synthetic single-sequence census with consistent records/halves."""

    records = []
    for i in range(second_union):
        label = (
            "same_object_return_primary" if i < second_primary else "same_object_return_r2prime"
        )
        records.append(
            {
                "seed_id": i % identities,
                "first_frame": split + 1 + i,
                "return_run_start": split + 5 + i,
                "return": label,
            }
        )
    # absence candidates beyond the returns (returns are a subset of absences)
    for i in range(second_absence - second_union):
        records.append(
            {
                "seed_id": 0,
                "first_frame": split + 1 + i,
                "return_run_start": split + 2 + i,
                "return": "beyond_r_site",
            }
        )
    covered, total = second_cov
    return {
        "per_sequence": {
            "seq": {
                "halves": {
                    "split_frame": split,
                    "occlusion": {"first_half": 0, "second_half": second_occl},
                    "true_absence": {"first_half": 0, "second_half": second_absence},
                    "union_returns": {"first_half": 0, "second_half": second_union},
                    "primary_returns": {"first_half": 0, "second_half": second_primary},
                },
                "coverage_tallies": {
                    "components_total": total * 2,
                    "components_covered": covered * 2,
                    "by_half": {
                        "first_half": {"components_total": total, "components_covered": covered},
                        "second_half": {"components_total": total, "components_covered": covered},
                    },
                },
                "records": {"true_absence_candidates": records},
            }
        }
    }


class Cycle3GateTests(unittest.TestCase):
    def _write(self, tmp: Path, name: str, census: dict) -> Path:
        path = tmp / f"{name}.json"
        path.write_text(json.dumps(census), encoding="utf-8")
        return path

    def _run(self, writing_2: dict, pour_tea: dict, tambourine: dict) -> dict:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            return gate.apply_gate(
                {
                    "writing_2": self._write(tmp, "w2", writing_2),
                    "pour_tea": self._write(tmp, "pt", pour_tea),
                    "tambourine": self._write(tmp, "tb", tambourine),
                }
            )

    def test_both_pass(self):
        result = self._run(
            _census(split=240, second_union=45, second_primary=40, second_occl=100,
                    second_absence=60, second_cov=(80, 100)),
            _census(split=226, second_union=0, second_primary=0, second_occl=50,
                    second_absence=30, second_cov=(60, 100)),
            _census(split=128, second_union=0, second_primary=0, second_occl=20,
                    second_absence=15, second_cov=(70, 100)),
        )
        g_r = result["G_R_reactivation"]
        self.assertTrue(g_r["pass"])
        self.assertEqual(g_r["union_returns_second_half"], 45)
        self.assertEqual(g_r["primary_returns_second_half"], 40)
        self.assertEqual(sum(g_r["per_identity_decomposition"].values()), 45)
        g_oa = result["G_OA_occlusion_absence"]
        self.assertTrue(g_oa["pass"])
        self.assertEqual(g_oa["pooled_occlusion_second_half"], 170)
        self.assertEqual(g_oa["pooled_true_absence_second_half"], 105)

    def test_g_r_fails_alone_when_returns_short(self):
        result = self._run(
            _census(split=240, second_union=20, second_primary=20, second_occl=100,
                    second_absence=60, second_cov=(80, 100)),
            _census(split=226, second_union=0, second_primary=0, second_occl=50,
                    second_absence=30, second_cov=(60, 100)),
            _census(split=128, second_union=0, second_primary=0, second_occl=20,
                    second_absence=15, second_cov=(70, 100)),
        )
        self.assertFalse(result["G_R_reactivation"]["pass"])
        self.assertTrue(result["G_OA_occlusion_absence"]["pass"])  # decoupled

    def test_g_oa_fails_alone_on_companion_free_ride(self):
        # tambourine contributes 5 absences (< 12 per-sequence floor):
        # G-OA fails on per_sequence_floors while G-R still passes.
        result = self._run(
            _census(split=240, second_union=45, second_primary=40, second_occl=100,
                    second_absence=60, second_cov=(80, 100)),
            _census(split=226, second_union=0, second_primary=0, second_occl=50,
                    second_absence=40, second_cov=(60, 100)),
            _census(split=128, second_union=0, second_primary=0, second_occl=20,
                    second_absence=5, second_cov=(70, 100)),
        )
        self.assertTrue(result["G_R_reactivation"]["pass"])
        g_oa = result["G_OA_occlusion_absence"]
        self.assertFalse(g_oa["pass"])
        self.assertFalse(g_oa["comparisons"]["per_sequence_floors"])
        self.assertFalse(g_oa["per_sequence"]["tambourine"]["per_sequence_ok"])

    def test_records_halves_cross_check(self):
        # Corrupt the halves attribution so it disagrees with the records:
        # the gate must fail closed.
        broken = _census(split=240, second_union=45, second_primary=40, second_occl=100,
                         second_absence=60, second_cov=(80, 100))
        broken["per_sequence"]["seq"]["halves"]["union_returns"]["second_half"] = 44
        with self.assertRaises(ContractError):
            self._run(
                broken,
                _census(split=226, second_union=0, second_primary=0, second_occl=50,
                        second_absence=30, second_cov=(60, 100)),
                _census(split=128, second_union=0, second_primary=0, second_occl=20,
                        second_absence=15, second_cov=(70, 100)),
            )

    def test_first_half_returns_never_count(self):
        # A return whose terminating run starts BEFORE the split must not
        # enter the G-R count (frozen boundary convention).
        census = _census(split=240, second_union=36, second_primary=36, second_occl=100,
                         second_absence=60, second_cov=(80, 100))
        census["per_sequence"]["seq"]["records"]["true_absence_candidates"].append(
            {"seed_id": 0, "first_frame": 10, "return_run_start": 100,
             "return": "same_object_return_primary"}
        )
        census["per_sequence"]["seq"]["halves"]["union_returns"]["first_half"] = 1
        result = self._run(
            census,
            _census(split=226, second_union=0, second_primary=0, second_occl=50,
                    second_absence=30, second_cov=(60, 100)),
            _census(split=128, second_union=0, second_primary=0, second_occl=20,
                    second_absence=15, second_cov=(70, 100)),
        )
        self.assertEqual(result["G_R_reactivation"]["union_returns_second_half"], 36)
        self.assertTrue(result["G_R_reactivation"]["pass"])


if __name__ == "__main__":
    unittest.main()
