"""Pointer-column payload edit (CCR v3): composition, sidecar,
one-hop invariant, and the deployed-state-equals-tested-state property.

Covers both payload families that share the single pointer column:
the appearance arms ("dc"/"full") and the opacity arm ("opacity").

Run with:
    python -m unittest tests.test_appearance_edit
    python -m pytest tests/test_appearance_edit.py

`scene/appearance_edit.py` is loaded STANDALONE rather than as
`scene.appearance_edit`, because importing the `scene` package executes
`scene/__init__.py`, which pulls in compiled CUDA extensions
(`simple_knn._C`, `pointops2_cuda`) that are not buildable on every
workstation. The module itself depends only on `hashlib`, `torch` and
`depth_visibility.errors`, so loading the file directly runs exactly the
same code and keeps these numerics auditable without a GPU image. The
few tests that genuinely need `GaussianModel` import it lazily and skip
when the extensions are absent.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402


def _load_appearance_edit():
    path = REPO_ROOT / "scene" / "appearance_edit.py"
    spec = importlib.util.spec_from_file_location(
        "adags_appearance_edit_under_test", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ae = _load_appearance_edit()

apply_appearance_edit = ae.apply_appearance_edit
build_edit_payload = ae.build_edit_payload
clear_appearance_edit = ae.clear_appearance_edit
compose_shared_features = ae.compose_shared_features
compose_shared_opacity = ae.compose_shared_opacity
load_edit_payload = ae.load_edit_payload
non_pointer_state_hash = ae.non_pointer_state_hash
redirects_features = ae.redirects_features
redirects_opacity = ae.redirects_opacity


def _gaussian_model_property(name):
    """The getter of a `GaussianModel` property, or a clean skip.

    Importing `scene.gaussian_model` transitively requires compiled CUDA
    extensions; where they exist the property logic is worth exercising,
    and where they do not the rest of this file still runs.
    """
    try:
        from scene.gaussian_model import GaussianModel
    except Exception as exc:                    # pragma: no cover
        raise unittest.SkipTest(
            "scene.gaussian_model unavailable ({}: {})".format(
                type(exc).__name__, exc))
    return getattr(GaussianModel, name).fget


def _class_member_sources(path, class_name, member_names):
    """Slice named members out of a class body, by line, from the source.

    Used to exercise the REAL `get_features` / `get_opacity` property
    bodies on a workstation where importing `GaussianModel` is
    impossible: the text under test comes from the repository file, so a
    property that stops doing what it claims still fails this test. Only
    the surrounding class is synthetic.
    """
    source = Path(path).read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    node = None
    for candidate in tree.body:
        if isinstance(candidate, ast.ClassDef) and candidate.name == class_name:
            node = candidate
            break
    if node is None:
        raise AssertionError("{} not found in {}".format(class_name, path))

    def start_of(member):
        starts = [member.lineno]
        for decorator in getattr(member, "decorator_list", []):
            starts.append(decorator.lineno)
        return min(starts)

    members = [m for m in node.body
               if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
    members.sort(key=start_of)
    out = {}
    for index, member in enumerate(members):
        if member.name not in member_names:
            continue
        begin = start_of(member) - 1
        end = (start_of(members[index + 1]) - 1
               if index + 1 < len(members) else len(lines))
        out[member.name] = "\n".join(lines[begin:end]).rstrip() + "\n"
    missing = set(member_names) - set(out)
    if missing:
        raise AssertionError("members not found: {}".format(sorted(missing)))
    return out


def _build_property_host():
    """A stand-in class carrying the repository's real property bodies.

    The properties lazily `from scene.appearance_edit import ...`. Where
    the `scene` package imports (a GPU image) that resolves to the real
    module; where it does not, the already-loaded standalone module is
    published under that name for the duration of the test and removed
    afterwards, so no other test in the session sees a stub.
    """
    sources = _class_member_sources(
        REPO_ROOT / "scene" / "gaussian_model.py", "GaussianModel",
        ("get_features", "get_opacity"))
    body = textwrap.dedent(sources["get_features"] + "\n"
                           + sources["get_opacity"])
    namespace = {"torch": torch}
    exec("class _PropertyHost:\n" + textwrap.indent(body, "    "),
         namespace)                                     # noqa: S102
    return namespace["_PropertyHost"]


class _scene_module_shim(object):
    """Publish the standalone appearance-edit module as
    `scene.appearance_edit` only if the real package cannot be imported."""

    def __init__(self):
        self.installed = []

    def __enter__(self):
        try:
            importlib.import_module("scene.appearance_edit")
            return self
        except Exception:                               # pragma: no cover
            pass
        package = sys.modules.get("scene")
        if package is None:
            package = types.ModuleType("scene")
            package.__path__ = [str(REPO_ROOT / "scene")]
            sys.modules["scene"] = package
            self.installed.append("scene")
        if "scene.appearance_edit" not in sys.modules:
            sys.modules["scene.appearance_edit"] = ae
            self.installed.append("scene.appearance_edit")
        return self

    def __exit__(self, *exc_info):
        for name in reversed(self.installed):
            sys.modules.pop(name, None)
        return False


def _features(n=6, seed=3):
    gen = torch.Generator().manual_seed(seed)
    dc = torch.rand(n, 1, 3, generator=gen)
    rest = torch.rand(n, 15, 3, generator=gen)
    return dc, rest


def _opacity(n=6, seed=11):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, 1, generator=gen)


class ComposeTests(unittest.TestCase):
    def test_identity_pointer_is_a_no_op(self):
        dc, rest = _features()
        idx = torch.arange(6)
        out_dc, out_rest = compose_shared_features(dc, rest, idx, "dc")
        self.assertTrue(torch.equal(out_dc, dc))
        self.assertTrue(torch.equal(out_rest, rest))

    def test_dc_mode_redirects_dc_only(self):
        dc, rest = _features()
        idx = torch.arange(6)
        idx[4] = 1  # row 4 reuses row 1's base radiance
        out_dc, out_rest = compose_shared_features(dc, rest, idx, "dc")
        self.assertTrue(torch.equal(out_dc[4], dc[1]))
        self.assertTrue(torch.equal(out_rest[4], rest[4]))  # own SH kept
        self.assertTrue(torch.equal(out_dc[:4], dc[:4]))

    def test_full_mode_redirects_both(self):
        dc, rest = _features()
        idx = torch.arange(6)
        idx[4] = 1
        out_dc, out_rest = compose_shared_features(dc, rest, idx, "full")
        self.assertTrue(torch.equal(out_dc[4], dc[1]))
        self.assertTrue(torch.equal(out_rest[4], rest[1]))

    def test_one_hop_invariant_fails_closed(self):
        dc, rest = _features()
        idx = torch.arange(6)
        idx[4] = 1
        idx[1] = 0  # row 1 is a donor AND a recipient -> chain
        with self.assertRaises(ContractError):
            compose_shared_features(dc, rest, idx, "dc")

    def test_bad_mode_and_bad_shape_fail_closed(self):
        dc, rest = _features()
        with self.assertRaises(ContractError):
            compose_shared_features(dc, rest, torch.arange(6), "chroma")
        with self.assertRaises(ContractError):
            compose_shared_features(dc, rest, torch.arange(5), "dc")

    def test_appearance_composer_refuses_the_opacity_mode(self):
        dc, rest = _features()
        with self.assertRaises(ContractError):
            compose_shared_features(dc, rest, torch.arange(6), "opacity")


class OpacityComposeTests(unittest.TestCase):
    def test_identity_pointer_is_a_no_op(self):
        opacity = _opacity()
        out = compose_shared_opacity(opacity, torch.arange(6), "opacity")
        self.assertTrue(torch.equal(out, opacity))

    def test_redirected_rows_take_the_donor_logit(self):
        opacity = _opacity()
        idx = torch.arange(6)
        idx[4] = 1
        idx[5] = 0
        out = compose_shared_opacity(opacity, idx, "opacity")
        self.assertTrue(torch.equal(out[4], opacity[1]))
        self.assertTrue(torch.equal(out[5], opacity[0]))
        self.assertTrue(torch.equal(out[:4], opacity[:4]))

    def test_the_source_column_is_never_written(self):
        opacity = _opacity()
        before = opacity.clone()
        idx = torch.arange(6)
        idx[3] = 2
        compose_shared_opacity(opacity, idx, "opacity")
        self.assertTrue(torch.equal(opacity, before))

    def test_one_hop_invariant_fails_closed(self):
        opacity = _opacity()
        idx = torch.arange(6)
        idx[4] = 1
        idx[1] = 0  # row 1 is a donor AND a recipient -> chain
        with self.assertRaises(ContractError):
            compose_shared_opacity(opacity, idx, "opacity")

    def test_bad_mode_and_bad_shape_fail_closed(self):
        opacity = _opacity()
        with self.assertRaises(ContractError):
            compose_shared_opacity(opacity, torch.arange(6), "dc")
        with self.assertRaises(ContractError):
            compose_shared_opacity(opacity, torch.arange(6), "chroma")
        with self.assertRaises(ContractError):
            compose_shared_opacity(opacity, torch.arange(5), "opacity")
        with self.assertRaises(ContractError):
            compose_shared_opacity(opacity, torch.zeros(6, 1, dtype=torch.long),
                                   "opacity")


class ModeDispatchTests(unittest.TestCase):
    def test_every_mode_belongs_to_exactly_one_family(self):
        self.assertEqual(ae.EDIT_MODES,
                         ae.APPEARANCE_EDIT_MODES + ae.OPACITY_EDIT_MODES)
        for mode in ae.EDIT_MODES:
            self.assertNotEqual(redirects_features(mode),
                                redirects_opacity(mode))

    def test_appearance_modes_are_unchanged(self):
        self.assertEqual(ae.APPEARANCE_EDIT_MODES, ("dc", "full"))
        self.assertTrue(redirects_features("dc"))
        self.assertTrue(redirects_features("full"))
        self.assertFalse(redirects_features("opacity"))
        self.assertTrue(redirects_opacity("opacity"))
        self.assertFalse(redirects_opacity("chroma"))


class SidecarTests(unittest.TestCase):
    def test_roundtrip_and_apply(self):
        pointer = torch.arange(6)
        pointer[5] = 2
        payload = build_edit_payload(
            pointer, "dc", source_checkpoint="chkpnt_best.pth",
            funnel={"evaluated": 3, "admitted": 1},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "edit.pt"
            torch.save(payload, path)
            loaded = load_edit_payload(path)
        self.assertEqual(loaded["num_redirected"], 1)
        self.assertEqual(loaded["funnel"]["admitted"], 1)

        g = types.SimpleNamespace(
            _features_dc=torch.zeros(6, 1, 3),
        )
        apply_appearance_edit(g, loaded)
        self.assertTrue(torch.equal(g._appearance_source_idx, pointer))
        self.assertEqual(g._appearance_share_mode, "dc")

    def test_opacity_sidecar_roundtrips_and_applies(self):
        pointer = torch.arange(6)
        pointer[5] = 2
        payload = build_edit_payload(
            pointer, "opacity", source_checkpoint="chkpnt6000.pth",
            funnel={"admitted": 2},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "edit.pt"
            torch.save(payload, path)
            loaded = load_edit_payload(path)
        self.assertEqual(loaded["mode"], "opacity")
        self.assertEqual(loaded["num_redirected"], 1)

        g = types.SimpleNamespace(_features_dc=torch.zeros(6, 1, 3))
        apply_appearance_edit(g, loaded)
        self.assertEqual(g._appearance_share_mode, "opacity")

    def test_unknown_mode_is_refused_on_build_and_load(self):
        with self.assertRaises(ContractError):
            build_edit_payload(torch.arange(4), "chroma",
                               source_checkpoint="x")
        payload = build_edit_payload(torch.arange(4), "dc",
                                     source_checkpoint="x")
        payload["mode"] = "chroma"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "edit.pt"
            torch.save(payload, path)
            with self.assertRaises(ContractError):
                load_edit_payload(path)

    def test_apply_refuses_row_count_mismatch(self):
        payload = build_edit_payload(
            torch.arange(6), "dc", source_checkpoint="x"
        )
        g = types.SimpleNamespace(_features_dc=torch.zeros(7, 1, 3))
        with self.assertRaises(ContractError):
            apply_appearance_edit(g, payload)


def _hashable_model(seed=9, rows=4, extras=True):
    gen = torch.Generator().manual_seed(seed)
    model = types.SimpleNamespace(
        _xyz=torch.rand(rows, 3, generator=gen),
        _features_dc=torch.rand(rows, 1, 3, generator=gen),
        _features_rest=torch.rand(rows, 15, 3, generator=gen),
        _opacity=torch.rand(rows, 1, generator=gen),
        _scaling=torch.rand(rows, 3, generator=gen),
        _rotation=torch.rand(rows, 4, generator=gen),
        _t=torch.rand(rows, 1, generator=gen),
        _scaling_t=torch.rand(rows, 1, generator=gen),
        _route_logit=torch.rand(rows, 1, generator=gen),
        _motion_lora_coeff=torch.rand(rows, 8, generator=gen),
    )
    if extras:
        model._packet_ids = torch.arange(rows, dtype=torch.long)
        model._rotation_r = torch.rand(rows, 4, generator=gen)
        model._motion_v = torch.rand(rows, 3, generator=gen)
        model._motion_a = torch.rand(rows, 3, generator=gen)
    return model


class InvariantHashTests(unittest.TestCase):
    def test_hash_ignores_pointer_and_mode_but_not_parameters(self):
        a = _hashable_model(extras=False)
        b = _hashable_model(extras=False)
        b._appearance_source_idx = torch.tensor([0, 0, 2, 3])
        b._appearance_share_mode = "full"
        self.assertEqual(non_pointer_state_hash(a), non_pointer_state_hash(b))
        b._features_dc = b._features_dc + 1e-6
        self.assertNotEqual(
            non_pointer_state_hash(a), non_pointer_state_hash(b)
        )

    def test_clear_restores_identity_semantics(self):
        g = types.SimpleNamespace(_features_dc=torch.zeros(3, 1, 3))
        apply_appearance_edit(
            g, build_edit_payload(torch.tensor([0, 0, 2]), "dc",
                                  source_checkpoint="x"),
        )
        clear_appearance_edit(g)
        self.assertEqual(g._appearance_source_idx.numel(), 0)

    def test_clear_restores_identity_semantics_for_the_opacity_mode(self):
        g = types.SimpleNamespace(_features_dc=torch.zeros(3, 1, 3))
        apply_appearance_edit(
            g, build_edit_payload(torch.tensor([0, 0, 2]), "opacity",
                                  source_checkpoint="x"),
        )
        self.assertEqual(g._appearance_share_mode, "opacity")
        clear_appearance_edit(g)
        self.assertEqual(g._appearance_source_idx.numel(), 0)
        self.assertEqual(g._appearance_share_mode, "dc")

    def test_install_compose_clear_writes_no_parameter_tensor(self):
        model = _hashable_model()
        before = non_pointer_state_hash(model)
        pointer = torch.arange(4)
        pointer[3] = 1
        apply_appearance_edit(
            model, build_edit_payload(pointer, "opacity",
                                      source_checkpoint="x"))
        composed = compose_shared_opacity(
            model._opacity, model._appearance_source_idx, "opacity")
        self.assertTrue(torch.equal(composed[3], model._opacity[1]))
        clear_appearance_edit(model)
        self.assertEqual(non_pointer_state_hash(model), before)

    def test_hash_covers_the_four_newly_added_columns(self):
        for name in ("_packet_ids", "_rotation_r", "_motion_v", "_motion_a"):
            self.assertIn(name, ae.HASHED_ROW_COLUMNS)
            base = _hashable_model()
            mutated = _hashable_model()
            original = getattr(mutated, name)
            if original.dtype == torch.long:
                setattr(mutated, name, original + 1)
            else:
                setattr(mutated, name, original + 1e-6)
            self.assertNotEqual(
                non_pointer_state_hash(base), non_pointer_state_hash(mutated),
                "{} is not covered by the invariant hash".format(name))

    def test_hash_still_skips_empty_and_missing_columns(self):
        without = _hashable_model(extras=False)
        with_empty = _hashable_model(extras=False)
        with_empty._packet_ids = torch.empty(0, dtype=torch.long)
        with_empty._rotation_r = torch.empty(0)
        with_empty._motion_v = torch.empty(0)
        with_empty._motion_a = torch.empty(0)
        with_empty._motion_lora_basis = torch.rand(8, 32, 3)   # not hashed
        self.assertEqual(non_pointer_state_hash(without),
                         non_pointer_state_hash(with_empty))


class PropertyRedirectTests(unittest.TestCase):
    """`GaussianModel.get_opacity` / `get_features` under a pointer.

    Skipped where the compiled CUDA extensions are unavailable; the
    composition itself is covered unconditionally above.
    """

    @staticmethod
    def _model(rows=6, mode=None, pointer=None):
        gen = torch.Generator().manual_seed(5)
        model = types.SimpleNamespace(
            _opacity=torch.randn(rows, 1, generator=gen),
            _features_dc=torch.rand(rows, 1, 3, generator=gen),
            _features_rest=torch.rand(rows, 15, 3, generator=gen),
            opacity_activation=torch.sigmoid,
        )
        if pointer is not None:
            model._appearance_source_idx = pointer
            model._appearance_share_mode = mode
        return model

    def test_opacity_mode_returns_the_donor_activated_opacity(self):
        get_opacity = _gaussian_model_property("get_opacity")
        pointer = torch.arange(6)
        pointer[4] = 1
        pointer[5] = 0
        model = self._model(mode="opacity", pointer=pointer)
        out = get_opacity(model)
        expected = torch.sigmoid(model._opacity)
        self.assertTrue(torch.equal(out[4], expected[1]))
        self.assertTrue(torch.equal(out[5], expected[0]))
        self.assertTrue(torch.equal(out[:4], expected[:4]))

    def test_empty_pointer_is_bit_identical_to_the_unedited_property(self):
        get_opacity = _gaussian_model_property("get_opacity")
        plain = self._model()
        edited = self._model(mode="opacity",
                             pointer=torch.empty(0, dtype=torch.long))
        self.assertTrue(torch.equal(get_opacity(plain), get_opacity(edited)))
        self.assertTrue(torch.equal(get_opacity(plain),
                                    torch.sigmoid(plain._opacity)))

    def test_an_appearance_pointer_leaves_opacity_untouched(self):
        get_opacity = _gaussian_model_property("get_opacity")
        pointer = torch.arange(6)
        pointer[4] = 1
        edited = self._model(mode="dc", pointer=pointer)
        self.assertTrue(torch.equal(get_opacity(edited),
                                    torch.sigmoid(edited._opacity)))

    def test_an_opacity_pointer_leaves_appearance_untouched(self):
        get_features = _gaussian_model_property("get_features")
        pointer = torch.arange(6)
        pointer[4] = 1
        edited = self._model(mode="opacity", pointer=pointer)
        plain = self._model()
        self.assertTrue(torch.equal(get_features(edited),
                                    get_features(plain)))

    def test_a_dc_pointer_still_redirects_appearance(self):
        get_features = _gaussian_model_property("get_features")
        pointer = torch.arange(6)
        pointer[4] = 1
        edited = self._model(mode="dc", pointer=pointer)
        out = get_features(edited)
        self.assertTrue(torch.equal(out[4, 0], edited._features_dc[1, 0]))
        self.assertTrue(torch.equal(out[4, 1:], edited._features_rest[4]))


class PropertySourceRedirectTests(unittest.TestCase):
    """The same behaviour, on the property bodies read out of
    `scene/gaussian_model.py`, so the redirect at the render read point
    is verifiable without the compiled CUDA extensions."""

    @classmethod
    def setUpClass(cls):
        cls.host = _build_property_host()

    def _model(self, rows=6, mode=None, pointer=None):
        gen = torch.Generator().manual_seed(5)
        model = self.host()
        model._opacity = torch.randn(rows, 1, generator=gen)
        model._features_dc = torch.rand(rows, 1, 3, generator=gen)
        model._features_rest = torch.rand(rows, 15, 3, generator=gen)
        model.opacity_activation = torch.sigmoid
        if pointer is not None:
            model._appearance_source_idx = pointer
            model._appearance_share_mode = mode
        return model

    def test_opacity_mode_returns_the_donor_activated_opacity(self):
        pointer = torch.arange(6)
        pointer[4] = 1
        pointer[5] = 0
        model = self._model(mode="opacity", pointer=pointer)
        with _scene_module_shim():
            out = model.get_opacity
        expected = torch.sigmoid(model._opacity)
        self.assertTrue(torch.equal(out[4], expected[1]))
        self.assertTrue(torch.equal(out[5], expected[0]))
        self.assertTrue(torch.equal(out[:4], expected[:4]))

    def test_empty_pointer_is_bit_identical_to_the_unedited_property(self):
        plain = self._model()
        edited = self._model(mode="opacity",
                             pointer=torch.empty(0, dtype=torch.long))
        with _scene_module_shim():
            self.assertTrue(torch.equal(plain.get_opacity,
                                        edited.get_opacity))
            self.assertTrue(torch.equal(plain.get_opacity,
                                        torch.sigmoid(plain._opacity)))

    def test_an_appearance_pointer_leaves_opacity_untouched(self):
        pointer = torch.arange(6)
        pointer[4] = 1
        edited = self._model(mode="dc", pointer=pointer)
        with _scene_module_shim():
            self.assertTrue(torch.equal(edited.get_opacity,
                                        torch.sigmoid(edited._opacity)))

    def test_an_opacity_pointer_leaves_appearance_untouched(self):
        pointer = torch.arange(6)
        pointer[4] = 1
        edited = self._model(mode="opacity", pointer=pointer)
        plain = self._model()
        with _scene_module_shim():
            self.assertTrue(torch.equal(edited.get_features,
                                        plain.get_features))

    def test_a_dc_pointer_still_redirects_appearance(self):
        pointer = torch.arange(6)
        pointer[4] = 1
        edited = self._model(mode="dc", pointer=pointer)
        with _scene_module_shim():
            out = edited.get_features
        self.assertTrue(torch.equal(out[4, 0], edited._features_dc[1, 0]))
        self.assertTrue(torch.equal(out[4, 1:], edited._features_rest[4]))

    def test_the_one_hop_invariant_reaches_the_render_read_point(self):
        pointer = torch.arange(6)
        pointer[4] = 1
        pointer[1] = 0                       # chain: row 1 donates AND receives
        model = self._model(mode="opacity", pointer=pointer)
        with _scene_module_shim():
            with self.assertRaises(ContractError):
                model.get_opacity

    def test_the_extracted_bodies_are_the_repository_properties(self):
        sources = _class_member_sources(
            REPO_ROOT / "scene" / "gaussian_model.py", "GaussianModel",
            ("get_features", "get_opacity"))
        self.assertIn("compose_shared_opacity", sources["get_opacity"])
        self.assertIn("redirects_opacity", sources["get_opacity"])
        self.assertIn("opacity_activation", sources["get_opacity"])
        self.assertIn("compose_shared_features", sources["get_features"])
        self.assertIn("redirects_features", sources["get_features"])
        for body in sources.values():
            self.assertTrue(body.lstrip().startswith("@property"))


if __name__ == "__main__":
    unittest.main()
