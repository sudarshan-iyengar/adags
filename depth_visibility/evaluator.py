"""Artifact validation, execution binding, and conservative gate decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from .canonical import canonical_json_bytes, domain_id, sha256_file
from .errors import ContractError, NonFiniteError, ProvenanceError, SchemaError


TERMINAL_SCHEMA = "phase9-terminal-manifest-v1"
SCORE_SCHEMA = "phase9-gate-a-score-v1"


def _load(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _finite_tree(value: Any, where: str = "artifact") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise NonFiniteError(f"{where} contains a nonfinite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, f"{where}[{index}]")


def validate_schema(artifact: Any, schema: Mapping[str, Any], where: str = "$") -> None:
    """Validate the strict JSON-schema subset used by the tracked bundle."""

    if "const" in schema and artifact != schema["const"]:
        raise SchemaError(f"{where} must equal {schema['const']!r}")
    if "enum" in schema and artifact not in schema["enum"]:
        raise SchemaError(f"{where} not in frozen enum")
    expected = schema.get("type")
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        matches = {
            "object": isinstance(artifact, Mapping),
            "array": isinstance(artifact, list),
            "string": isinstance(artifact, str),
            "integer": isinstance(artifact, int) and not isinstance(artifact, bool),
            "number": isinstance(artifact, (int, float)) and not isinstance(artifact, bool),
            "boolean": isinstance(artifact, bool),
            "null": artifact is None,
        }
        if not any(matches.get(value, False) for value in types):
            raise SchemaError(f"{where} has wrong type; expected {types}")
    if isinstance(artifact, Mapping):
        required = set(schema.get("required", []))
        missing = required - set(artifact)
        if missing:
            raise SchemaError(f"{where} missing keys: {sorted(missing)}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unknown = set(artifact) - set(properties)
            if unknown:
                raise SchemaError(f"{where} unknown keys: {sorted(unknown)}")
        for key, child_schema in properties.items():
            if key in artifact:
                validate_schema(artifact[key], child_schema, f"{where}.{key}")
    if isinstance(artifact, list):
        if "minItems" in schema and len(artifact) < int(schema["minItems"]):
            raise SchemaError(f"{where} has too few items")
        if "maxItems" in schema and len(artifact) > int(schema["maxItems"]):
            raise SchemaError(f"{where} has too many items")
        if schema.get("uniqueItems") and len({json.dumps(item, sort_keys=True) for item in artifact}) != len(artifact):
            raise SchemaError(f"{where} items are not unique")
        item_schema = schema.get("items")
        if item_schema:
            for index, child in enumerate(artifact):
                validate_schema(child, item_schema, f"{where}[{index}]")
    if isinstance(artifact, (int, float)) and not isinstance(artifact, bool):
        value = float(artifact)
        if not math.isfinite(value):
            raise NonFiniteError(f"{where} is nonfinite")
        if "minimum" in schema and value < float(schema["minimum"]):
            raise SchemaError(f"{where} is below minimum")
        if "maximum" in schema and value > float(schema["maximum"]):
            raise SchemaError(f"{where} is above maximum")
    if isinstance(artifact, str):
        if "minLength" in schema and len(artifact) < int(schema["minLength"]):
            raise SchemaError(f"{where} is too short")
        if "pattern" in schema:
            import re
            if re.fullmatch(str(schema["pattern"]), artifact) is None:
                raise SchemaError(f"{where} does not match pattern")


def load_schema_bundle(path: str | Path) -> dict[str, Any]:
    bundle = _load(path)
    if bundle.get("schema_version") != "phase9-schema-bundle-v1":
        raise SchemaError("wrong Phase 9 schema bundle")
    if not isinstance(bundle.get("schemas"), Mapping):
        raise SchemaError("schema bundle has no schemas")
    return bundle


def validate_named_artifact(
    artifact: Mapping[str, Any],
    schema_name: str,
    bundle: Mapping[str, Any],
) -> None:
    try:
        schema = bundle["schemas"][schema_name]
    except KeyError as exc:
        raise SchemaError(f"unregistered Phase 9 schema: {schema_name}") from exc
    validate_schema(artifact, schema)
    _finite_tree(artifact)


def write_json_atomic(path: str | Path, artifact: Mapping[str, Any]) -> str:
    """Write finite sorted JSON atomically and return its byte SHA-256."""

    _finite_tree(artifact)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return hashlib.sha256(encoded).hexdigest()


def load_run_entry(matrix_path: str | Path, run_id: str) -> dict[str, Any]:
    matrix = _load(matrix_path)
    matches = [entry for entry in matrix.get("runs", []) if entry.get("run_id") == run_id]
    if len(matches) != 1:
        raise ContractError(f"run matrix contains {len(matches)} entries for {run_id}")
    return matches[0]


def resolved_python_argv(
    manifest: Mapping[str, Any],
    *,
    run_id: str,
    action: str,
    launcher_path: str,
    execution_manifest_path: str | None = None,
) -> tuple[str, ...]:
    """Return the exact registered Python argv after fail-closed identity checks."""

    if manifest.get("run_id") != run_id or manifest.get("action") != action:
        raise ProvenanceError("resolved execution manifest run/action mismatch")
    raw = manifest.get("resolved_argv")
    if not isinstance(raw, list) or not raw or any(not isinstance(value, str) for value in raw):
        raise SchemaError("resolved argv must be a nonempty string list")
    argv = tuple(raw)
    if any("\0" in value for value in argv):
        raise ProvenanceError("resolved argv contains a NUL byte")
    if len(argv) < 7 or argv[0] != "python" or argv[1] != launcher_path or argv[2] != action:
        raise ProvenanceError("resolved argv interpreter/launcher/action mismatch")

    def option(name: str) -> str:
        positions = [index for index, value in enumerate(argv) if value == name]
        if len(positions) != 1 or positions[0] + 1 >= len(argv):
            raise ProvenanceError(f"resolved argv must contain exactly one {name}")
        return argv[positions[0] + 1]

    if option("--run-id") != run_id:
        raise ProvenanceError("resolved argv run ID mismatch")
    bound_manifest = option("--execution-manifest")
    if execution_manifest_path is not None and Path(bound_manifest).resolve() != Path(execution_manifest_path).resolve():
        raise ProvenanceError("resolved argv execution-manifest path mismatch")
    return argv


def validate_execution_manifest(
    manifest: Mapping[str, Any],
    *,
    run_id: str,
    action: str,
    run_entry: Mapping[str, Any],
    require_resolved: bool = True,
) -> dict[str, Any]:
    """Fail closed on run/action/config/implementation drift."""

    required = {
        "schema_version",
        "run_id",
        "action",
        "resolved_argv",
        "resolved_argv_sha256",
        "launcher_sha256",
        "implementation",
        "configuration",
        "expected_outputs",
        "input_artifacts",
        "external_inputs",
        "input_binding_status",
    }
    missing = required - set(manifest)
    if missing:
        raise SchemaError(f"execution manifest missing keys: {sorted(missing)}")
    if manifest["schema_version"] != "phase9-resolved-execution-v1":
        raise SchemaError("wrong execution-manifest schema")
    if manifest["run_id"] != run_id or manifest["action"] != action:
        raise ProvenanceError("execution manifest run/action mismatch")
    if run_entry["run_id"] != run_id or run_entry["action"] != action:
        raise ProvenanceError("run matrix run/action mismatch")
    encoded_argv = json.dumps(manifest["resolved_argv"], separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    if hashlib.sha256(encoded_argv).hexdigest() != manifest["resolved_argv_sha256"]:
        raise ProvenanceError("resolved argv hash mismatch")
    resolved_python_argv(
        manifest,
        run_id=run_id,
        action=action,
        launcher_path=run_entry["command"]["launcher_path"],
    )
    expected_argv = [
        os.path.expandvars(value) if "$" in value else value
        for value in run_entry["command"]["argv_template"]
    ]
    if any("$" in value for value in expected_argv):
        raise ProvenanceError("run-matrix argv contains an unresolved environment variable")
    if manifest["resolved_argv"] != expected_argv:
        raise ProvenanceError("resolved argv differs from the run matrix")
    expected_outputs = sorted(
        (item["path"], item["schema"])
        for item in run_entry["expected_outputs"] if item.get("required")
    )
    actual_outputs = sorted(
        (item["path"], item["schema"])
        for item in manifest["expected_outputs"] if item.get("required")
    )
    if expected_outputs != actual_outputs:
        raise ProvenanceError("resolved expected outputs differ from run matrix")
    for key in ("input_artifacts", "external_inputs"):
        expected_inputs = sorted(
            (item.get("path"), item.get("producer_run_id"), item.get("schema"))
            for item in run_entry[key]
        )
        actual_inputs = sorted(
            (item.get("path"), item.get("producer_run_id"), item.get("schema"))
            for item in manifest[key]
        )
        if expected_inputs != actual_inputs:
            raise ProvenanceError(f"resolved {key} differ from run matrix")
    if require_resolved:
        if not str(manifest.get("input_binding_status", "")).startswith("resolved_exact"):
            raise ProvenanceError("execution input artifacts remain unresolved")
        for section, keys in (
            (manifest["implementation"], ("commit", "implementation_manifest_sha256")),
            (manifest["configuration"], ("resolved_merged_config_sha256", "base_config_sha256")),
        ):
            if any(not section.get(key) for key in keys):
                raise ProvenanceError("execution binding remains unresolved")
    return {"valid": True, "run_id": run_id, "action": action}


def terminal_manifest(
    *,
    run_id: str,
    action: str,
    status: str,
    exit_code: int,
    execution_manifest_sha256: str,
    produced_artifacts: Sequence[Mapping[str, Any]],
    scientific_payload: Mapping[str, Any] | None = None,
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in {"completed", "failed", "not_applicable"}:
        raise ContractError("terminal status is invalid")
    if status == "completed" and int(exit_code) != 0:
        raise ContractError("completed terminal must have exit_code 0")
    payload = {
        "schema_version": TERMINAL_SCHEMA,
        "run_id": str(run_id),
        "action": str(action),
        "status": status,
        "exit_code": int(exit_code),
        "execution_manifest_sha256": str(execution_manifest_sha256),
        "produced_artifacts": [dict(item) for item in produced_artifacts],
        "scientific_payload": None if scientific_payload is None else dict(scientific_payload),
        "failure": None if failure is None else dict(failure),
    }
    _finite_tree(payload)
    return {**payload, "terminal_id": domain_id("csvl-v1/terminal", payload)}


def _metric(metrics: Mapping[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if value is None:
        return None
    return _finite(value, name)


def decide_gate_a(
    score_artifact: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    tier: str,
) -> dict[str, Any]:
    """Apply the frozen conjunction without converting missing labels to failure data."""

    if score_artifact.get("schema_version") != SCORE_SCHEMA:
        raise SchemaError("wrong Gate A score schema")
    if score_artifact.get("evidence_type") != "human_reference":
        return {
            "tier": tier,
            "status": "not_evaluable",
            "reason": "genuine_human_reference_unavailable",
            "criteria": {},
        }
    if score_artifact.get("labels_complete") is not True:
        return {
            "tier": tier,
            "status": "not_evaluable",
            "reason": "human_reference_incomplete",
            "criteria": {},
        }
    metrics = score_artifact.get("metrics", {})
    thresholds = config["gate_a"][tier]
    bindings = {
        "ordering_accuracy": ("ordering_accuracy_minimum", "minimum"),
        "ordering_auroc": ("ordering_auroc_minimum", "minimum"),
        "ordering_coverage": ("ordering_coverage_minimum", "minimum"),
        "event_f1": ("event_f1_minimum", "minimum"),
        "event_recall": ("event_recall_minimum", "minimum"),
        "boundary_f1_delta": ("boundary_f1_delta_minimum", "minimum"),
        "region_iou_delta": ("region_iou_delta_minimum", "minimum"),
        "cross_view_inconsistency_relative_reduction": ("cross_view_inconsistency_relative_reduction_minimum", "minimum"),
        "temporal_inconsistency_relative_reduction": ("temporal_inconsistency_relative_reduction_minimum", "minimum"),
    }
    criteria = {}
    missing = []
    for metric_name, (threshold_name, direction) in bindings.items():
        value = _metric(metrics, metric_name)
        threshold = float(thresholds[threshold_name])
        if value is None:
            missing.append(metric_name)
            criteria[metric_name] = {"status": "not_evaluable", "value": None, "threshold": threshold}
        else:
            passed = value >= threshold
            criteria[metric_name] = {"status": "pass" if passed else "fail", "value": value, "threshold": threshold, "direction": direction}
    for metric_name in ("ordering_ece", "transition_ece"):
        value = _metric(metrics, metric_name)
        threshold = float(thresholds["ece_maximum"])
        if value is None:
            missing.append(metric_name)
            criteria[metric_name] = {"status": "not_evaluable", "value": None, "threshold": threshold}
        else:
            passed = value <= threshold
            criteria[metric_name] = {"status": "pass" if passed else "fail", "value": value, "threshold": threshold, "direction": "maximum"}
    if tier == "engineering":
        name = "evaluable_track_fraction"
        value = _metric(metrics, name)
        threshold = float(thresholds["evaluable_track_fraction_minimum"])
        if value is None:
            missing.append(name)
            criteria[name] = {"status": "not_evaluable", "value": None, "threshold": threshold}
        else:
            criteria[name] = {"status": "pass" if value >= threshold else "fail", "value": value, "threshold": threshold, "direction": "minimum"}
    else:
        name = "no_represented_event_family_missed"
        value = metrics.get(name)
        if value is None:
            missing.append(name)
            criteria[name] = {"status": "not_evaluable", "value": None, "required": True}
        else:
            criteria[name] = {"status": "pass" if value is True else "fail", "value": bool(value), "required": True}
    if missing:
        status, reason = "not_evaluable", "missing_required_metrics"
    elif all(item["status"] == "pass" for item in criteria.values()):
        status, reason = "pass", "all_frozen_criteria_pass"
    else:
        status, reason = "fail", "one_or_more_frozen_criteria_fail"
    payload = {"tier": tier, "status": status, "reason": reason, "criteria": criteria, "missing": sorted(missing)}
    return {**payload, "decision_id": domain_id("csvl-v1/gate-a-decision", payload)}


def artifact_reference(path: str | Path, schema: str, producer_run_id: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ProvenanceError(f"expected artifact is missing: {source}")
    return {"path": str(path), "schema": schema, "producer_run_id": producer_run_id, "sha256": sha256_file(source)}


__all__ = [
    "SCORE_SCHEMA",
    "TERMINAL_SCHEMA",
    "artifact_reference",
    "decide_gate_a",
    "load_run_entry",
    "load_schema_bundle",
    "terminal_manifest",
    "validate_execution_manifest",
    "validate_named_artifact",
    "validate_schema",
    "write_json_atomic",
]
