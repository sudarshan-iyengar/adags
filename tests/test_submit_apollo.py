"""Unit tests for scripts/submit_apollo.py (CPU-only, no `det` CLI needed).

Run with:
    C:/Users/sucar/venvs/elgs-cpu/Scripts/python.exe -m unittest tests.test_submit_apollo

Every test below exercises a PURE function directly (no subprocess call to
`det`, no live Determined cluster). Closure/materialize tests build a
throw-away scratch git repository per test class in ``setUp`` (``git init``
plus one commit), matching the plan's requirement that the wrapper's
provenance guarantees are checked against real ``git`` behavior rather than
mocked out.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError, SchemaError  # noqa: E402
from scripts import submit_apollo as wrapper  # noqa: E402


# ---------------------------------------------------------------------------
# Scratch-repo helpers
# ---------------------------------------------------------------------------


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout.strip()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _seed_repo_tree(root: Path) -> None:
    """Populate every EXECUTION_DIRS/EXECUTION_FILES entry with a placeholder file."""

    for directory in wrapper.EXECUTION_DIRS:
        _write(root / directory / "__init__.py", f"# {directory} placeholder\n")
    _write(root / "main.py", "# main placeholder\n")
    _write(root / "scripts" / "submit_apollo.py", "# wrapper placeholder (scratch copy)\n")
    _write(root / "det_exp_apollo.yaml", 'name: "{{NAME}}"\n')
    _write(
        root / "configs" / "elgs" / "smoke.json",
        json.dumps({"iterations": 10, "seed": 1}),
    )
    _write(
        root / "configs" / "elgs" / "prereg_structural_v1.json",
        json.dumps({"k_max": 4}),
    )
    _write(root / "research-wiki" / "log.md", "# durable wiki log\n")


def _init_scratch_repo(root: Path) -> str:
    """Seed, ``git init``, and commit a scratch repo. Returns the commit SHA."""

    _seed_repo_tree(root)
    _git("init", "-q", cwd=root)
    _git("config", "user.email", "elgs-test@example.invalid", cwd=root)
    _git("config", "user.name", "ELGS Test", cwd=root)
    _git("add", "-A", cwd=root)
    _git("commit", "-q", "-m", "initial scratch commit", cwd=root)
    return _git("rev-parse", "HEAD", cwd=root)


class _ScratchRepoTestCase(unittest.TestCase):
    """Base class providing a committed scratch repo at ``self.repo_root``."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="elgs-wrapper-test-")
        self.repo_root = Path(self._tmp.name) / "repo"
        self.repo_root.mkdir()
        self.commit = _init_scratch_repo(self.repo_root)
        self.config_path = self.repo_root / "configs" / "elgs" / "smoke.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()


# ---------------------------------------------------------------------------
# resolve_execution_set
# ---------------------------------------------------------------------------


class ResolveExecutionSetTests(_ScratchRepoTestCase):
    def test_includes_declared_dirs_files_and_named_config(self):
        execution_set = wrapper.resolve_execution_set(self.repo_root, [self.config_path])
        for directory in wrapper.EXECUTION_DIRS:
            self.assertIn(directory, execution_set)
        for file_entry in wrapper.EXECUTION_FILES:
            self.assertIn(file_entry, execution_set)
        self.assertIn("configs/elgs/smoke.json", execution_set)
        # Deterministic, de-duplicated, sorted.
        self.assertEqual(execution_set, sorted(set(execution_set)))

    def test_accepts_relative_and_absolute_config_paths_identically(self):
        via_absolute = wrapper.resolve_execution_set(self.repo_root, [self.config_path])
        via_relative = wrapper.resolve_execution_set(self.repo_root, ["configs/elgs/smoke.json"])
        self.assertEqual(via_absolute, via_relative)

    def test_missing_declared_path_raises(self):
        (self.repo_root / "elgs").rename(self.repo_root / "elgs_renamed")
        with self.assertRaises(ContractError):
            wrapper.resolve_execution_set(self.repo_root, [self.config_path])

    def test_config_path_outside_repo_raises(self):
        outside = Path(self._tmp.name) / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        with self.assertRaises(ContractError):
            wrapper.resolve_execution_set(self.repo_root, [outside])


# ---------------------------------------------------------------------------
# check_execution_closure
# ---------------------------------------------------------------------------


class CheckExecutionClosureTests(_ScratchRepoTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.execution_set = wrapper.resolve_execution_set(self.repo_root, [self.config_path])

    def test_clean_tree_is_evidence_bearing(self):
        report = wrapper.check_execution_closure(self.repo_root, self.execution_set)
        self.assertEqual(report.dirty_inside, ())
        self.assertEqual(report.dirty_outside, ())
        self.assertTrue(report.evidence_bearing)
        self.assertFalse(report.dirty_smoke)

    def test_dirty_inside_refuses_by_default(self):
        (self.repo_root / "elgs" / "__init__.py").write_text("# modified\n", encoding="utf-8")
        with self.assertRaises(ContractError):
            wrapper.check_execution_closure(self.repo_root, self.execution_set)

    def test_dirty_inside_with_dirty_smoke_is_allowed_and_marked_non_evidence_bearing(self):
        (self.repo_root / "elgs" / "__init__.py").write_text("# modified\n", encoding="utf-8")
        report = wrapper.check_execution_closure(
            self.repo_root, self.execution_set, dirty_smoke=True
        )
        self.assertIn("elgs/__init__.py", report.dirty_inside)
        self.assertFalse(report.evidence_bearing)
        self.assertTrue(report.dirty_smoke)

    def test_untracked_user_owned_wiki_files_are_outside_and_allowed(self):
        # Mirrors the real repo's two never-touched untracked files:
        # research-wiki/deep-dive-prompt.txt and research-wiki/run-deep-dive.ps1.
        _write(self.repo_root / "research-wiki" / "deep-dive-prompt.txt", "prompt text\n")
        _write(self.repo_root / "research-wiki" / "run-deep-dive.ps1", "# script\n")
        report = wrapper.check_execution_closure(self.repo_root, self.execution_set)
        self.assertTrue(report.evidence_bearing)
        self.assertEqual(report.dirty_inside, ())
        self.assertIn("research-wiki/deep-dive-prompt.txt", report.dirty_outside)
        self.assertIn("research-wiki/run-deep-dive.ps1", report.dirty_outside)

    def test_new_untracked_file_inside_execution_dir_is_dirty_inside(self):
        _write(self.repo_root / "elgs" / "new_module.py", "# new\n")
        with self.assertRaises(ContractError) as ctx:
            wrapper.check_execution_closure(self.repo_root, self.execution_set)
        self.assertIn("elgs/new_module.py", str(ctx.exception))

    def test_wholly_untracked_directory_git_collapses_is_still_classified_inside(self):
        # git status --porcelain reports an entirely-untracked directory as a
        # single "dirname/" line, not one line per file inside it. The
        # execution set names a specific file under that new directory; the
        # closure check must still classify the collapsed directory line as
        # "inside" (this is exactly the situation the real repo's untracked
        # configs/elgs/ directory exercises).
        _write(self.repo_root / "configs" / "elgs_new" / "prereg_extra_v1.json", "{}")
        execution_set = wrapper.resolve_execution_set(
            self.repo_root, [self.config_path]
        ) + ["configs/elgs_new/prereg_extra_v1.json"]
        with self.assertRaises(ContractError) as ctx:
            wrapper.check_execution_closure(self.repo_root, execution_set)
        self.assertIn("configs/elgs_new", str(ctx.exception))

    def test_evidence_bearing_stays_true_when_only_outside_dirt_and_dirty_smoke_passed(self):
        _write(self.repo_root / "research-wiki" / "scratch.txt", "note\n")
        report = wrapper.check_execution_closure(
            self.repo_root, self.execution_set, dirty_smoke=True
        )
        self.assertTrue(report.evidence_bearing)


# ---------------------------------------------------------------------------
# materialize_context
# ---------------------------------------------------------------------------


class MaterializeContextTests(_ScratchRepoTestCase):
    def test_extracts_files_and_returns_stable_sha256(self):
        out_dir_a = Path(self._tmp.name) / "context_a"
        out_dir_a.mkdir()
        out_dir_b = Path(self._tmp.name) / "context_b"
        out_dir_b.mkdir()

        sha_a = wrapper.materialize_context(self.repo_root, self.commit, out_dir_a)
        sha_b = wrapper.materialize_context(self.repo_root, self.commit, out_dir_b)

        self.assertEqual(sha_a, sha_b)
        self.assertEqual(len(sha_a), 64)
        self.assertTrue((out_dir_a / "main.py").is_file())
        self.assertTrue((out_dir_a / "elgs" / "__init__.py").is_file())
        self.assertEqual(
            (out_dir_a / "main.py").read_text(encoding="utf-8"),
            (self.repo_root / "main.py").read_text(encoding="utf-8"),
        )

    def test_refuses_nonempty_target_directory(self):
        out_dir = Path(self._tmp.name) / "context_dirty"
        out_dir.mkdir()
        (out_dir / "stray.txt").write_text("pre-existing\n", encoding="utf-8")
        with self.assertRaises(ContractError):
            wrapper.materialize_context(self.repo_root, self.commit, out_dir)

    def test_refuses_missing_target_directory(self):
        missing = Path(self._tmp.name) / "does_not_exist"
        with self.assertRaises(ContractError):
            wrapper.materialize_context(self.repo_root, self.commit, missing)

    def test_unknown_commit_raises(self):
        out_dir = Path(self._tmp.name) / "context_bad_commit"
        out_dir.mkdir()
        with self.assertRaises(ContractError):
            wrapper.materialize_context(self.repo_root, "0" * 40, out_dir)


# ---------------------------------------------------------------------------
# canonical_config_hash
# ---------------------------------------------------------------------------


class CanonicalConfigHashTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="elgs-cfg-hash-test-")
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_json_hash_is_stable_across_formatting_and_key_order(self):
        # Same path, rewritten with different formatting/key order, so the
        # path-identity component of the hash cannot mask the comparison.
        path = self.dir / "config.json"
        path.write_text('{"a":1,"b":2}', encoding="utf-8")
        first = wrapper.canonical_config_hash([path])
        path.write_text('{\n  "b": 2,\n  "a": 1\n}\n', encoding="utf-8")
        second = wrapper.canonical_config_hash([path])
        self.assertEqual(first, second)

    def test_json_hash_changes_with_content(self):
        one = self.dir / "one.json"
        two = self.dir / "two.json"
        one.write_text('{"a": 1}', encoding="utf-8")
        two.write_text('{"a": 2}', encoding="utf-8")
        self.assertNotEqual(
            wrapper.canonical_config_hash([one]), wrapper.canonical_config_hash([two])
        )

    def test_yaml_hash_is_raw_bytes_not_canonicalized(self):
        # Same path, same semantic content, only a trailing blank line
        # differs -- YAML has no canonicalization in this repo, so the hash
        # MUST differ. This is a deliberate contract, not an oversight;
        # documented in canonical_config_hash's docstring.
        path = self.dir / "config.yaml"
        path.write_text("iterations: 10\nseed: 1\n", encoding="utf-8")
        first = wrapper.canonical_config_hash([path])
        path.write_text("iterations: 10\nseed: 1\n\n", encoding="utf-8")
        second = wrapper.canonical_config_hash([path])
        self.assertNotEqual(first, second)

    def test_order_of_input_paths_does_not_matter(self):
        first = self.dir / "first.json"
        second = self.dir / "second.json"
        first.write_text('{"a": 1}', encoding="utf-8")
        second.write_text('{"b": 2}', encoding="utf-8")
        self.assertEqual(
            wrapper.canonical_config_hash([first, second]),
            wrapper.canonical_config_hash([second, first]),
        )

    def test_missing_file_raises(self):
        with self.assertRaises(ContractError):
            wrapper.canonical_config_hash([self.dir / "missing.json"])

    def test_unsupported_extension_raises(self):
        bad = self.dir / "config.toml"
        bad.write_text("a = 1\n", encoding="utf-8")
        with self.assertRaises(ContractError):
            wrapper.canonical_config_hash([bad])

    def test_invalid_json_raises_schema_error(self):
        bad = self.dir / "broken.json"
        bad.write_text("{not valid json", encoding="utf-8")
        with self.assertRaises(SchemaError):
            wrapper.canonical_config_hash([bad])


# ---------------------------------------------------------------------------
# build_manifest
# ---------------------------------------------------------------------------


class BuildManifestTests(_ScratchRepoTestCase):
    REQUIRED_KEYS = {
        "schema",
        "commit",
        "branch",
        "config_files",
        "config_canonical_hash",
        "prereg_files",
        "image_ref",
        "pool",
        "slots",
        "seed",
        "dataset_manifest",
        "run_dir",
        "entrypoint_script",
        "wrapper",
        "submitter_host",
        "utc_stamp",
        "evidence_bearing",
        "projected_gpu_hours",
    }

    def _build(self, **overrides):
        kwargs = dict(
            repo_root=self.repo_root,
            commit=self.commit,
            branch="apollo/csvl-vpl-v2-exploratory",
            config_paths=[self.config_path],
            image_ref="sudarshaniyengar/adags@sha256:" + "a" * 64,
            pool="hopper",
            slots=1,
            seed=7,
            run_dir="/apollo/users/sri/proj_adags/runs/elgs/20260811T000000Z_m0-s1_7_abc1234",
            wrapper_argv=["submit_apollo.py", "submit", "--cell", "m0-s1"],
            evidence_bearing=True,
            projected_gpu_hours=2.0,
        )
        kwargs.update(overrides)
        return wrapper.build_manifest(**kwargs)

    def test_manifest_has_every_required_key(self):
        manifest = self._build()
        self.assertEqual(self.REQUIRED_KEYS, set(manifest.keys()))

    def test_manifest_is_plain_json_serializable(self):
        manifest = self._build()
        json.dumps(manifest, allow_nan=False)  # must not raise

    def test_manifest_schema_and_scalars(self):
        manifest = self._build()
        self.assertEqual(manifest["schema"], wrapper.MANIFEST_SCHEMA)
        self.assertEqual(manifest["commit"], self.commit)
        self.assertEqual(manifest["pool"], "hopper")
        self.assertEqual(manifest["slots"], 1)
        self.assertEqual(manifest["seed"], 7)
        self.assertTrue(manifest["evidence_bearing"])
        self.assertEqual(manifest["projected_gpu_hours"], 2.0)

    def test_config_canonical_hash_matches_direct_call(self):
        # The manifest keys entries on the RELATIVE path (the
        # root-independence contract the S1 smokes forced), so the
        # direct call must use the same relative_to root.
        manifest = self._build()
        expected = wrapper.canonical_config_hash(
            [self.config_path], relative_to=self.repo_root
        )
        self.assertEqual(manifest["config_canonical_hash"], expected)

    def test_config_files_entry_has_path_and_sha256(self):
        manifest = self._build()
        self.assertEqual(len(manifest["config_files"]), 1)
        entry = manifest["config_files"][0]
        self.assertEqual(entry["path"], "configs/elgs/smoke.json")
        self.assertEqual(len(entry["sha256"]), 64)

    def test_prereg_files_are_discovered_under_configs_elgs(self):
        manifest = self._build()
        prereg_paths = {entry["path"] for entry in manifest["prereg_files"]}
        self.assertIn("configs/elgs/prereg_structural_v1.json", prereg_paths)
        # The smoke config itself is not a prereg file and must not leak in.
        self.assertNotIn("configs/elgs/smoke.json", prereg_paths)

    def test_dataset_manifest_defaults_to_none_for_a_smoke_run(self):
        manifest = self._build()
        self.assertIsNone(manifest["dataset_manifest"])

    def test_dataset_manifest_recorded_when_provided(self):
        dataset_manifest = self.repo_root / "data_manifest.json"
        dataset_manifest.write_text('{"files": []}', encoding="utf-8")
        manifest = self._build(dataset_manifest_path=dataset_manifest)
        self.assertIsNotNone(manifest["dataset_manifest"])
        self.assertEqual(len(manifest["dataset_manifest"]["sha256"]), 64)

    def test_wrapper_block_has_argv_and_own_file_hash(self):
        manifest = self._build()
        self.assertEqual(
            manifest["wrapper"]["argv"], ["submit_apollo.py", "submit", "--cell", "m0-s1"]
        )
        self.assertEqual(len(manifest["wrapper"]["file_sha256"]), 64)


# ---------------------------------------------------------------------------
# claim_cell
# ---------------------------------------------------------------------------


class ClaimCellTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="elgs-claims-test-")
        self.claims_dir = Path(self._tmp.name) / "claims"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_claim_creates_expected_filename(self):
        path = wrapper.claim_cell(self.claims_dir, "m0-s1", 0)
        self.assertEqual(path.name, "m0-s1__r0.json")
        self.assertTrue(path.is_file())
        body = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(body["schema"], wrapper.CLAIM_SCHEMA)
        self.assertEqual(body["cell_name"], "m0-s1")
        self.assertEqual(body["retry"], 0)

    def test_duplicate_claim_raises_eexist_style_error(self):
        wrapper.claim_cell(self.claims_dir, "m0-s1", 0)
        with self.assertRaises(ContractError) as ctx:
            wrapper.claim_cell(self.claims_dir, "m0-s1", 0)
        self.assertIn("already claimed", str(ctx.exception))

    def test_different_retry_counter_does_not_collide(self):
        first = wrapper.claim_cell(self.claims_dir, "m0-s1", 0)
        second = wrapper.claim_cell(self.claims_dir, "m0-s1", 1)
        self.assertNotEqual(first, second)
        self.assertTrue(first.is_file())
        self.assertTrue(second.is_file())

    def test_payload_is_merged_into_claim_body(self):
        path = wrapper.claim_cell(self.claims_dir, "m0-s1", 0, payload={"pool": "hopper"})
        body = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(body["pool"], "hopper")


# ---------------------------------------------------------------------------
# append_ledger
# ---------------------------------------------------------------------------


class AppendLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="elgs-ledger-test-")
        self.ledger_path = Path(self._tmp.name) / "state" / "experiment-ledger.jsonl"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _lines(self):
        return [
            json.loads(line)
            for line in self.ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_append_creates_parent_and_writes_one_line(self):
        wrapper.append_ledger(self.ledger_path, {"event": "submitted", "cell_name": "m0-s1"})
        lines = self._lines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["event"], "submitted")
        self.assertEqual(lines[0]["schema"], wrapper.LEDGER_SCHEMA)

    def test_second_append_preserves_the_first_line(self):
        wrapper.append_ledger(self.ledger_path, {"event": "submitted", "cell_name": "m0-s1"})
        wrapper.append_ledger(self.ledger_path, {"event": "cancelled", "cell_name": "m0-s1"})
        lines = self._lines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["event"], "submitted")
        self.assertEqual(lines[1]["event"], "cancelled")

    def test_caller_supplied_schema_is_overridden(self):
        wrapper.append_ledger(self.ledger_path, {"event": "submitted", "schema": "bogus-v0"})
        lines = self._lines()
        self.assertEqual(lines[0]["schema"], wrapper.LEDGER_SCHEMA)

    def test_non_finite_record_raises_schema_error(self):
        with self.assertRaises(SchemaError):
            wrapper.append_ledger(self.ledger_path, {"event": "submitted", "bad": float("nan")})


# ---------------------------------------------------------------------------
# render_template
# ---------------------------------------------------------------------------


class RenderTemplateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="elgs-template-test-")
        self.template_path = Path(self._tmp.name) / "det_exp_apollo.yaml"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_the_real_template_renders_completely(self):
        substitutions = {
            "NAME": "m0-s1",
            "POOL": "hopper",
            "IMAGE_REF": "sudarshaniyengar/adags@sha256:" + "a" * 64,
            "ENTRYPOINT_SCRIPT": "main.py",
            "ENTRYPOINT_ARGS": "--config configs/elgs/smoke.json --model_path /apollo/x",
            "RUN_DIR": "/apollo/users/sri/proj_adags/runs/elgs/x",
        }
        rendered = wrapper.render_template(REPO_ROOT / "det_exp_apollo.yaml", substitutions)
        # No *actual* placeholder token survives. (The template's own header
        # comments legitimately mention the literal string "{{...}}" while
        # explaining the substitution syntax -- that is prose, not a
        # placeholder, and render_template's own placeholder regex correctly
        # ignores it; assert on the six real tokens instead of a blanket
        # substring search.)
        for key in ("NAME", "POOL", "IMAGE_REF", "ENTRYPOINT_SCRIPT", "ENTRYPOINT_ARGS", "RUN_DIR"):
            self.assertNotIn("{{" + key + "}}", rendered)
        self.assertIn("m0-s1", rendered)
        self.assertIn("resource_pool: \"hopper\"", rendered)

    def test_missing_substitution_raises(self):
        self.template_path.write_text("name: {{NAME}}\npool: {{POOL}}\n", encoding="utf-8")
        with self.assertRaises(ContractError):
            wrapper.render_template(self.template_path, {"NAME": "m0-s1"})

    def test_full_substitution_leaves_no_placeholder(self):
        self.template_path.write_text("name: {{NAME}}\npool: {{POOL}}\n", encoding="utf-8")
        rendered = wrapper.render_template(
            self.template_path, {"NAME": "m0-s1", "POOL": "hopper"}
        )
        self.assertEqual(rendered, "name: m0-s1\npool: hopper\n")

    def test_leftover_placeholder_reintroduced_by_a_substituted_value_is_caught(self):
        self.template_path.write_text("entrypoint: {{ENTRYPOINT_ARGS}}\n", encoding="utf-8")
        with self.assertRaises(ContractError):
            wrapper.render_template(
                self.template_path, {"ENTRYPOINT_ARGS": "--extra {{POOL}}"}
            )


# ---------------------------------------------------------------------------
# runtime_assertions' pure helpers
# ---------------------------------------------------------------------------


class RuntimeAssertionsPureHelperTests(unittest.TestCase):
    def test_forbidden_worktree_exact_path_refused(self):
        with self.assertRaises(ContractError):
            wrapper._refuse_forbidden_worktree(wrapper.FORBIDDEN_WORKTREE)

    def test_forbidden_worktree_nested_path_refused(self):
        with self.assertRaises(ContractError):
            wrapper._refuse_forbidden_worktree(wrapper.FORBIDDEN_WORKTREE + "/elgs")

    def test_other_paths_are_not_refused(self):
        # Must not raise.
        wrapper._refuse_forbidden_worktree("/apollo/users/sri/proj_adags/runs/elgs/some-run")

    def test_assert_path_within_root_passes_for_a_contained_path(self):
        with tempfile.TemporaryDirectory(prefix="elgs-within-root-") as tmp:
            root = Path(tmp)
            child = root / "elgs" / "intervals.py"
            child.parent.mkdir(parents=True)
            child.write_text("# module\n", encoding="utf-8")
            wrapper._assert_path_within_root(child, root, "elgs.__file__")  # must not raise

    def test_assert_path_within_root_raises_for_an_escaping_path(self):
        with tempfile.TemporaryDirectory(prefix="elgs-within-root-a-") as tmp_a, \
                tempfile.TemporaryDirectory(prefix="elgs-within-root-b-") as tmp_b:
            outside_file = Path(tmp_b) / "main.py"
            outside_file.write_text("# module\n", encoding="utf-8")
            with self.assertRaises(ContractError):
                wrapper._assert_path_within_root(outside_file, Path(tmp_a), "main.__file__")


class CanonicalHashRootIndependenceTests(unittest.TestCase):
    """Regression for the second S1 smoke catch: identical bytes under
    different absolute roots (temp context dir on the submitter vs
    /run/determined/workdir in-container) must hash identically when
    relative_to is given — the entry key is the relative path, never
    the absolute string."""

    def test_same_bytes_different_roots_same_hash(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            for root in (a, b):
                target = Path(root) / "configs" / "elgs"
                target.mkdir(parents=True)
                (target / "run.yaml").write_bytes(b"iterations: 600\n")
            hash_a = wrapper.canonical_config_hash(
                [Path(a) / "configs" / "elgs" / "run.yaml"], relative_to=a
            )
            hash_b = wrapper.canonical_config_hash(
                [Path(b) / "configs" / "elgs" / "run.yaml"], relative_to=b
            )
            self.assertEqual(hash_a, hash_b)
            # Without relative_to the absolute strings differ: the
            # legacy behavior that produced the mismatch.
            legacy_a = wrapper.canonical_config_hash(
                [Path(a) / "configs" / "elgs" / "run.yaml"]
            )
            legacy_b = wrapper.canonical_config_hash(
                [Path(b) / "configs" / "elgs" / "run.yaml"]
            )
            self.assertNotEqual(legacy_a, legacy_b)


class EntrypointScriptTests(unittest.TestCase):
    """M1 extension: wrapper-controlled entrypoint script (allowlisted)."""

    def test_main_py_args_unchanged(self):
        args = wrapper._build_entrypoint_args(
            config_path="configs/elgs/x.yaml",
            run_dir="/apollo/run",
            extra_args=["--seed", "1"],
        )
        self.assertEqual(args, "--config configs/elgs/x.yaml --model_path /apollo/run --seed 1")

    def test_non_main_entrypoint_takes_only_extra_args(self):
        args = wrapper._build_entrypoint_args(
            config_path="configs/elgs/prereg_m1_census_v1.json",
            run_dir="/apollo/run",
            extra_args=["--scene-dir", "/apollo/scene", "--out", "/apollo/run/census.json"],
            entrypoint_script="scripts/build_m1_census.py",
        )
        self.assertEqual(args, "--scene-dir /apollo/scene --out /apollo/run/census.json")
        self.assertNotIn("--config", args)
        self.assertNotIn("--model_path", args)

    def test_non_main_entrypoint_requires_extra_args(self):
        with self.assertRaises(ContractError):
            wrapper._build_entrypoint_args(
                config_path="c.yaml",
                run_dir="/r",
                extra_args=[],
                entrypoint_script="scripts/build_elgs_tracks.py",
            )

    def test_unlisted_entrypoint_script_is_rejected(self):
        with self.assertRaises(ContractError):
            wrapper._build_entrypoint_args(
                config_path="c.yaml",
                run_dir="/r",
                extra_args=["--x"],
                entrypoint_script="scripts/unknown.py",
            )

    def test_template_renders_with_census_entrypoint(self):
        substitutions = {
            "NAME": "m1_a0_unlock_0",
            "POOL": "hopper",
            "IMAGE_REF": "sudarshaniyengar/adags@sha256:" + "a" * 64,
            "ENTRYPOINT_SCRIPT": "scripts/build_m1_census.py",
            "ENTRYPOINT_ARGS": "--scene-dir /apollo/scene --out /apollo/run/census.json",
            "RUN_DIR": "/apollo/users/sri/proj_adags/runs/elgs/x",
        }
        rendered = wrapper.render_template(REPO_ROOT / "det_exp_apollo.yaml", substitutions)
        self.assertIn("exec python3 scripts/build_m1_census.py", rendered)

    def test_manifest_records_entrypoint_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            config = repo / "configs" / "elgs"
            config.mkdir(parents=True)
            (config / "cell.json").write_text('{"a": 1}', encoding="utf-8")
            manifest = wrapper.build_manifest(
                repo_root=repo,
                commit="c" * 40,
                branch="b",
                config_paths=[config / "cell.json"],
                image_ref="img@sha256:" + "a" * 64,
                pool="hopper",
                slots=1,
                seed=0,
                run_dir="/r",
                wrapper_argv=["x"],
                evidence_bearing=True,
                projected_gpu_hours=0.1,
                entrypoint_script="scripts/build_m1_census.py",
            )
            self.assertEqual(manifest["entrypoint_script"], "scripts/build_m1_census.py")


if __name__ == "__main__":
    unittest.main()
