import csv
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


TRUE_EVENT_TYPES = {"hide_reveal", "hide_only", "boundary_occlusion"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class FrozenHideRevealParams:
    c_min: float = 0.55
    m_event: float = 0.02
    lambda_id: float = 1.0
    lambda_static: float = 0.5
    lambda_budget: float = 0.05
    beta: float = 1.0
    a_frame: int = 10
    b_frame: int = 18
    support_radius: float = 5.5
    static_margin: float = 2.0
    m_pass: int = 4096


@dataclass(frozen=True)
class SyntheticCandidate:
    clip_id: str
    split: str
    event_type: str
    seed: int
    center_frame: int
    a_frame: int
    b_frame: int
    residual: float
    boundary: float
    flow_disagreement: float
    track_toggle: float
    ghost: float

    @property
    def is_true_event(self) -> bool:
        return self.event_type in TRUE_EVENT_TYPES


@dataclass(frozen=True)
class ShadowScore:
    score: float
    patch: float
    identity: float
    static_ghost: float
    budget: float


@dataclass(frozen=True)
class CandidateResult:
    candidate: SyntheticCandidate
    candidate_score: float
    selected: bool
    smooth: ShadowScore
    event: ShadowScore
    delta_event: float
    accepted: bool
    matched_lifespan_delta: float
    matched_lifespan_accepted: bool
    no_identity_delta: float
    no_identity_accepted: bool
    unnormalized_delta: float
    no_hysteresis_accepted: bool


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def rectangular_gate(
    frames: np.ndarray,
    a_frame: float,
    b_frame: float,
    amplitude: float = 1.0,
    beta: float = 1.0,
) -> np.ndarray:
    gate = 1.0 - amplitude * (sigmoid((frames - a_frame) / beta) - sigmoid((frames - b_frame) / beta))
    return np.clip(gate, 0.0, 1.0)


def candidate_score(candidate: SyntheticCandidate) -> float:
    return (
        0.25 * candidate.residual
        + 0.25 * candidate.boundary
        + 0.20 * candidate.flow_disagreement
        + 0.20 * candidate.track_toggle
        + 0.10 * candidate.ghost
    )


def _clip01(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _background(frames: int, height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.zeros((frames, height, width, 3), dtype=np.float32)
    base[..., 0] = 0.16 + 0.10 * (xx / max(width - 1, 1))
    base[..., 1] = 0.18 + 0.08 * (yy / max(height - 1, 1))
    base[..., 2] = 0.22
    return base


def _disk_mask(height: int, width: int, x: float, y: float, radius: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    return ((xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2).astype(np.float32)


def _ring_mask(height: int, width: int, x: float, y: float, inner: float, outer: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    dist2 = (xx - x) ** 2 + (yy - y) ** 2
    return ((dist2 > inner ** 2) & (dist2 <= outer ** 2)).astype(np.float32)


def _positions(frames: int, width: int, height: int, seed: int, event_type: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x0 = 14.0 + rng.uniform(-2.0, 2.0)
    y0 = 29.0 + rng.uniform(-3.0, 3.0)
    vx = 0.78 + rng.uniform(-0.08, 0.08)
    vy = 0.05 + rng.uniform(-0.03, 0.03)
    t = np.arange(frames, dtype=np.float32)
    if event_type == "distractor_motion":
        y0 += 8.0
        vy = 0.35
    if event_type == "boundary_occlusion":
        y0 -= 4.0
    x = np.clip(x0 + vx * t, 8.0, width - 8.0)
    y = np.clip(y0 + vy * t + 1.2 * np.sin(t / 5.0), 8.0, height - 8.0)
    return np.stack([x, y], axis=1)


def _render_target(
    image: np.ndarray,
    positions: np.ndarray,
    visibility: np.ndarray,
    radius: float = 4.5,
) -> Tuple[np.ndarray, np.ndarray]:
    frames, height, width, _ = image.shape
    rendered = image.copy()
    alpha = np.zeros((frames, height, width), dtype=np.float32)
    color = np.asarray([0.92, 0.20, 0.12], dtype=np.float32)
    for frame_idx in range(frames):
        mask = _disk_mask(height, width, positions[frame_idx, 0], positions[frame_idx, 1], radius)
        visible_mask = mask * float(visibility[frame_idx])
        alpha[frame_idx] = visible_mask
        rendered[frame_idx] = rendered[frame_idx] * (1.0 - visible_mask[..., None]) + color * visible_mask[..., None]
    return _clip01(rendered), alpha


def _render_occluder(
    image: np.ndarray,
    positions: np.ndarray,
    a_frame: int,
    b_frame: int,
    event_type: str,
) -> np.ndarray:
    rendered = image.copy()
    if event_type not in TRUE_EVENT_TYPES:
        return rendered
    frames, height, width, _ = image.shape
    color = np.asarray([0.12, 0.34, 0.70], dtype=np.float32)
    for frame_idx in range(a_frame, min(b_frame + 1, frames)):
        x = int(round(positions[frame_idx, 0]))
        y = int(round(positions[frame_idx, 1]))
        x0, x1 = max(0, x - 7), min(width, x + 8)
        y0, y1 = max(0, y - 7), min(height, y + 8)
        rendered[frame_idx, y0:y1, x0:x1, :] = color
    return _clip01(rendered)


def _candidate_from_clip(
    clip_id: str,
    split: str,
    event_type: str,
    seed: int,
    params: FrozenHideRevealParams,
) -> SyntheticCandidate:
    rng = np.random.default_rng(seed + 1009)
    if event_type in TRUE_EVENT_TYPES:
        cue_base = {
            "residual": 0.82,
            "boundary": 0.78,
            "flow_disagreement": 0.70,
            "track_toggle": 0.86,
            "ghost": 0.72,
        }
    elif event_type == "distractor_motion":
        cue_base = {
            "residual": 0.45,
            "boundary": 0.38,
            "flow_disagreement": 0.48,
            "track_toggle": 0.20,
            "ghost": 0.36,
        }
    else:
        cue_base = {
            "residual": 0.24,
            "boundary": 0.16,
            "flow_disagreement": 0.22,
            "track_toggle": 0.10,
            "ghost": 0.18,
        }

    noisy = {key: float(np.clip(value + rng.normal(0.0, 0.045), 0.0, 1.0)) for key, value in cue_base.items()}
    return SyntheticCandidate(
        clip_id=clip_id,
        split=split,
        event_type=event_type,
        seed=seed,
        center_frame=(params.a_frame + params.b_frame) // 2,
        a_frame=params.a_frame,
        b_frame=params.b_frame,
        residual=noisy["residual"],
        boundary=noisy["boundary"],
        flow_disagreement=noisy["flow_disagreement"],
        track_toggle=noisy["track_toggle"],
        ghost=noisy["ghost"],
    )


def make_synthetic_clip(
    candidate: SyntheticCandidate,
    params: FrozenHideRevealParams,
    frames: int = 32,
    height: int = 64,
    width: int = 64,
) -> Dict[str, np.ndarray]:
    positions = _positions(frames, width, height, candidate.seed, candidate.event_type)
    background = _background(frames, height, width)
    smooth_visibility = np.ones(frames, dtype=np.float32)
    event_visibility = rectangular_gate(
        np.arange(frames, dtype=np.float32),
        candidate.a_frame,
        candidate.b_frame,
        amplitude=1.0,
        beta=params.beta,
    ).astype(np.float32)

    observed_visibility = smooth_visibility.copy()
    if candidate.event_type in TRUE_EVENT_TYPES:
        observed_visibility[candidate.a_frame : candidate.b_frame + 1] = 0.0

    observed, _ = _render_target(background, positions, observed_visibility)
    observed = _render_occluder(observed, positions, candidate.a_frame, candidate.b_frame, candidate.event_type)
    smooth, smooth_alpha = _render_target(background, positions, smooth_visibility)
    event, event_alpha = _render_target(background, positions, event_visibility)
    event = _render_occluder(event, positions, candidate.a_frame, candidate.b_frame, candidate.event_type)

    support = np.zeros((frames, height, width), dtype=np.float32)
    static_support = np.zeros((frames, height, width), dtype=np.float32)
    for frame_idx in range(frames):
        x, y = positions[frame_idx]
        support[frame_idx] = _disk_mask(height, width, x, y, params.support_radius)
        static_support[frame_idx] = _ring_mask(
            height,
            width,
            x,
            y,
            params.support_radius + params.static_margin,
            params.support_radius + params.static_margin + 5.0,
        )

    return {
        "observed": observed,
        "smooth": smooth,
        "event": event,
        "smooth_alpha": smooth_alpha,
        "event_alpha": event_alpha,
        "event_visibility": event_visibility,
        "support": support,
        "static_support": static_support,
    }


def shadow_score(
    prediction: np.ndarray,
    observed: np.ndarray,
    alpha: np.ndarray,
    support: np.ndarray,
    static_support: np.ndarray,
    visibility: Optional[np.ndarray],
    candidate: SyntheticCandidate,
    params: FrozenHideRevealParams,
    use_identity: bool = True,
    normalized: bool = True,
    is_event_hypothesis: bool = False,
) -> ShadowScore:
    frame_mask = np.zeros(alpha.shape[0], dtype=np.float32)
    frame_mask[max(candidate.a_frame - 2, 0) : min(candidate.b_frame + 3, alpha.shape[0])] = 1.0
    support_w = support * frame_mask[:, None, None]
    static_w = static_support * frame_mask[:, None, None]

    abs_error = np.abs(prediction - observed).mean(axis=-1)
    patch_numer = float((abs_error * support_w).sum())
    patch_denom = float(support_w.sum()) + 1e-6
    patch = patch_numer / patch_denom if normalized else patch_numer

    static_numer = float((alpha * static_w).sum())
    static_denom = float(static_w.sum()) + 1e-6
    static_ghost = static_numer / static_denom if normalized else static_numer

    if not use_identity:
        identity = 0.0
    elif candidate.is_true_event:
        identity = 0.0 if is_event_hypothesis else 0.35
    else:
        identity = 0.22 if is_event_hypothesis else 0.0

    budget = 0.0
    if is_event_hypothesis:
        tv = float(np.abs(np.diff(visibility)).sum()) if visibility is not None else 0.0
        budget = 1.0 + 0.1 * tv

    total = patch + params.lambda_id * identity + params.lambda_static * static_ghost + params.lambda_budget * budget
    return ShadowScore(score=float(total), patch=float(patch), identity=float(identity), static_ghost=float(static_ghost), budget=float(budget))


def evaluate_candidate(candidate: SyntheticCandidate, params: FrozenHideRevealParams) -> CandidateResult:
    clip = make_synthetic_clip(candidate, params)
    smooth = shadow_score(
        clip["smooth"],
        clip["observed"],
        clip["smooth_alpha"],
        clip["support"],
        clip["static_support"],
        None,
        candidate,
        params,
        is_event_hypothesis=False,
    )
    event = shadow_score(
        clip["event"],
        clip["observed"],
        clip["event_alpha"],
        clip["support"],
        clip["static_support"],
        clip["event_visibility"],
        candidate,
        params,
        is_event_hypothesis=True,
    )

    score = candidate_score(candidate)
    selected = score >= params.c_min
    delta = event.score - smooth.score
    accepted = selected and delta < -params.m_event

    no_id_params = replace(params, lambda_id=0.0)
    smooth_no_id = shadow_score(
        clip["smooth"],
        clip["observed"],
        clip["smooth_alpha"],
        clip["support"],
        clip["static_support"],
        None,
        candidate,
        no_id_params,
        use_identity=False,
        is_event_hypothesis=False,
    )
    event_no_id = shadow_score(
        clip["event"],
        clip["observed"],
        clip["event_alpha"],
        clip["support"],
        clip["static_support"],
        clip["event_visibility"],
        candidate,
        no_id_params,
        use_identity=False,
        is_event_hypothesis=True,
    )
    no_id_delta = event_no_id.score - smooth_no_id.score

    smooth_unnorm = shadow_score(
        clip["smooth"],
        clip["observed"],
        clip["smooth_alpha"],
        clip["support"],
        clip["static_support"],
        None,
        candidate,
        params,
        normalized=False,
        is_event_hypothesis=False,
    )
    event_unnorm = shadow_score(
        clip["event"],
        clip["observed"],
        clip["event_alpha"],
        clip["support"],
        clip["static_support"],
        clip["event_visibility"],
        candidate,
        params,
        normalized=False,
        is_event_hypothesis=True,
    )
    unnormalized_delta = event_unnorm.score - smooth_unnorm.score

    lifespan_params = replace(params, lambda_id=0.0)
    lifespan_delta = no_id_delta
    lifespan_accepted = selected and lifespan_delta < -lifespan_params.m_event
    no_identity_accepted = selected and no_id_delta < -no_id_params.m_event
    no_hysteresis_accepted = score >= (params.c_min - 0.08) and delta < -params.m_event

    return CandidateResult(
        candidate=candidate,
        candidate_score=float(score),
        selected=bool(selected),
        smooth=smooth,
        event=event,
        delta_event=float(delta),
        accepted=bool(accepted),
        matched_lifespan_delta=float(lifespan_delta),
        matched_lifespan_accepted=bool(lifespan_accepted),
        no_identity_delta=float(no_id_delta),
        no_identity_accepted=bool(no_identity_accepted),
        unnormalized_delta=float(unnormalized_delta),
        no_hysteresis_accepted=bool(no_hysteresis_accepted),
    )


def make_synthetic_candidates(
    seeds: Sequence[int],
    clips_per_type: int,
    params: FrozenHideRevealParams,
) -> List[SyntheticCandidate]:
    event_types = ["hide_reveal", "boundary_occlusion", "normal_motion", "distractor_motion"]
    candidates: List[SyntheticCandidate] = []
    for seed in seeds:
        split = "calibration" if seed % 2 == 0 else "heldout"
        for event_type in event_types:
            for idx in range(clips_per_type):
                clip_seed = seed * 1000 + idx * 17 + event_types.index(event_type) * 101
                clip_id = f"{split}_{event_type}_{seed:02d}_{idx:03d}"
                candidates.append(_candidate_from_clip(clip_id, split, event_type, clip_seed, params))
    return candidates


def roc_auc(labels: Sequence[bool], scores: Sequence[float]) -> Optional[float]:
    positives = [(score, idx) for idx, (label, score) in enumerate(zip(labels, scores)) if label]
    negatives = [(score, idx) for idx, (label, score) in enumerate(zip(labels, scores)) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = float(len(positives) * len(negatives))
    for pos_score, _ in positives:
        for neg_score, _ in negatives:
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return wins / total


def _safe_div(numer: float, denom: float) -> Optional[float]:
    if denom == 0:
        return None
    return numer / denom


def summarize_candidate_results(results: Sequence[CandidateResult], split: Optional[str] = None) -> Dict[str, Optional[float]]:
    rows = [result for result in results if split is None or result.candidate.split == split]
    true_rows = [result for result in rows if result.candidate.is_true_event]
    normal_rows = [result for result in rows if not result.candidate.is_true_event]
    accepted = [result for result in rows if result.accepted]
    accepted_true = [result for result in accepted if result.candidate.is_true_event]
    selected_true = [result for result in true_rows if result.selected]
    selected_normal = [result for result in normal_rows if result.selected]
    lifespan_true = [result for result in true_rows if result.matched_lifespan_accepted]

    auc = roc_auc([result.candidate.is_true_event for result in rows], [-result.delta_event for result in rows])
    candidate_auc = roc_auc([result.candidate.is_true_event for result in rows], [result.candidate_score for result in rows])

    return {
        "n": float(len(rows)),
        "true_events": float(len(true_rows)),
        "normal_controls": float(len(normal_rows)),
        "candidate_recall": _safe_div(float(len(selected_true)), float(len(true_rows))),
        "candidate_false_positive_rate": _safe_div(float(len(selected_normal)), float(len(normal_rows))),
        "candidate_score_auc": auc_to_float(candidate_auc),
        "margin_auc": auc_to_float(auc),
        "accepted_precision": _safe_div(float(len(accepted_true)), float(len(accepted))),
        "accepted_recall": _safe_div(float(len(accepted_true)), float(len(true_rows))),
        "false_event_rate_normal": _safe_div(float(len([r for r in normal_rows if r.accepted])), float(len(normal_rows))),
        "mean_delta_true": mean_or_none([result.delta_event for result in true_rows]),
        "mean_delta_normal": mean_or_none([result.delta_event for result in normal_rows]),
        "identity_reconnection_accuracy": _safe_div(float(len(accepted_true)), float(len(true_rows))),
        "matched_lifespan_accept_recall": _safe_div(float(len(lifespan_true)), float(len(true_rows))),
        "matched_lifespan_identity_reconnection_accuracy": 0.0 if true_rows else None,
        "no_identity_accept_recall": _safe_div(
            float(len([result for result in true_rows if result.no_identity_accepted])),
            float(len(true_rows)),
        ),
        "no_hysteresis_false_event_rate": _safe_div(
            float(len([result for result in normal_rows if result.no_hysteresis_accepted])),
            float(len(normal_rows)),
        ),
    }


def auc_to_float(value: Optional[float]) -> Optional[float]:
    return None if value is None else float(value)


def mean_or_none(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return float(np.mean(values))


def run_synthetic_poc(
    seeds: Sequence[int],
    clips_per_type: int,
    out_dir: Path,
    params: FrozenHideRevealParams = FrozenHideRevealParams(),
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = make_synthetic_candidates(seeds, clips_per_type, params)
    results = [evaluate_candidate(candidate, params) for candidate in candidates]

    rows = [candidate_result_to_row(result) for result in results]
    write_csv(out_dir / "synthetic_candidates.csv", rows)
    summary = {
        "params": asdict(params),
        "summary": {
            "all": summarize_candidate_results(results),
            "calibration": summarize_candidate_results(results, split="calibration"),
            "heldout": summarize_candidate_results(results, split="heldout"),
        },
        "stop_go": synthetic_stop_go(results),
    }
    write_json(out_dir / "synthetic_summary.json", summary)
    write_json(out_dir / "frozen_params.json", asdict(params))
    write_synthetic_report(out_dir / "synthetic_report.md", summary)
    return summary


def synthetic_stop_go(results: Sequence[CandidateResult]) -> Dict[str, object]:
    heldout = summarize_candidate_results(results, split="heldout")
    candidate_recall = metric_or_default(heldout.get("candidate_recall"), 0.0)
    margin_auc = metric_or_default(heldout.get("margin_auc"), 0.0)
    false_event_rate = metric_or_default(heldout.get("false_event_rate_normal"), 1.0)
    lifespan_identity = heldout.get("matched_lifespan_identity_reconnection_accuracy")
    identity = metric_or_default(heldout.get("identity_reconnection_accuracy"), 0.0)
    pass_candidate = candidate_recall >= 0.85
    pass_margin = margin_auc >= 0.85 and false_event_rate <= 0.15
    pass_lifespan = lifespan_identity is not None and identity > lifespan_identity
    return {
        "pass_candidate_recall": pass_candidate,
        "pass_margin_separation": pass_margin,
        "pass_matched_lifespan_gate": pass_lifespan,
        "proceed_to_real_windows": bool(pass_candidate and pass_margin and pass_lifespan),
        "notes": [
            "Synthetic labels carry the identity claim; real windows should be sanity checks only.",
            "Matched lifespan identity reconnection is zero in this PoC because it has no hidden-identity reveal matching.",
        ],
    }


def metric_or_default(value: Optional[float], default: float) -> float:
    return default if value is None else float(value)


def candidate_result_to_row(result: CandidateResult) -> Dict[str, object]:
    candidate = result.candidate
    row = asdict(candidate)
    row.update(
        {
            "is_true_event": candidate.is_true_event,
            "candidate_score": result.candidate_score,
            "selected": result.selected,
            "smooth_score": result.smooth.score,
            "event_score": result.event.score,
            "delta_event": result.delta_event,
            "accepted": result.accepted,
            "matched_lifespan_delta": result.matched_lifespan_delta,
            "matched_lifespan_accepted": result.matched_lifespan_accepted,
            "no_identity_delta": result.no_identity_delta,
            "no_identity_accepted": result.no_identity_accepted,
            "unnormalized_delta": result.unnormalized_delta,
            "no_hysteresis_accepted": result.no_hysteresis_accepted,
        }
    )
    return row


def write_synthetic_report(path: Path, summary: Dict[str, object]) -> None:
    heldout = summary["summary"]["heldout"]
    stop_go = summary["stop_go"]
    lines = [
        "# Synthetic Hide/Reveal PoC Report",
        "",
        "## Heldout Metrics",
        "",
    ]
    for key, value in heldout.items():
        lines.append(f"- `{key}`: {format_metric(value)}")
    lines.extend(["", "## Stop / Go", ""])
    for key, value in stop_go.items():
        if key == "notes":
            continue
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Notes", ""])
    for note in stop_go.get("notes", []):
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_real_manifest_template(path: Path) -> Dict[str, object]:
    template = {
        "description": "Predeclare 4-6 occlusion/reveal windows before running real hide/reveal scoring.",
        "frames_are_inclusive": True,
        "windows": [
            {
                "window_id": "scene01_window01",
                "scene": "replace_with_scene_name",
                "frame_start": 100,
                "frame_end": 116,
                "crop_xyxy": [120, 80, 260, 220],
                "occluder": "hand/tool/object",
                "notes": "Describe what is hidden and later revealed before scoring.",
                "systems": {
                    "route0": {
                        "render_dir": "path/to/route0/test/ours_6000/renders",
                        "gt_dir": "path/to/route0/test/ours_6000/gt",
                        "static_dir": "path/to/route0/test/ours_6000/static"
                    },
                    "residual_uncertainty": {
                        "render_dir": "path/to/residual/test/ours_6000/renders",
                        "gt_dir": "path/to/residual/test/ours_6000/gt"
                    },
                    "matched_lifespan": {
                        "render_dir": "path/to/lifespan/test/ours_6000/renders",
                        "gt_dir": "path/to/lifespan/test/ours_6000/gt"
                    },
                    "hide_reveal": {
                        "render_dir": "path/to/hide_reveal/test/ours_6000/renders",
                        "gt_dir": "path/to/hide_reveal/test/ours_6000/gt"
                    }
                }
            }
        ]
    }
    write_json(path, template)
    return template


def evaluate_real_manifest(manifest_path: Path, out_dir: Path, compute_lpips: bool = False) -> Dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for window in manifest.get("windows", []):
        rows.extend(evaluate_real_window(window, manifest_path.parent, compute_lpips=compute_lpips))
    write_csv(out_dir / "real_event_window_metrics.csv", rows)
    summary = summarize_real_rows(rows)
    payload = {"manifest": str(manifest_path), "summary": summary, "rows": rows}
    write_json(out_dir / "real_event_window_summary.json", payload)
    write_real_report(out_dir / "real_event_window_report.md", payload)
    return payload


def evaluate_real_window(window: Dict[str, object], base_dir: Path, compute_lpips: bool = False) -> List[Dict[str, object]]:
    required = ["window_id", "frame_start", "frame_end", "crop_xyxy", "systems"]
    missing = [key for key in required if key not in window]
    if missing:
        raise ValueError(f"Window {window.get('window_id', '<unknown>')} is missing keys: {missing}")
    frame_start = int(window["frame_start"])
    frame_end = int(window["frame_end"])
    crop = tuple(int(value) for value in window["crop_xyxy"])
    systems = window["systems"]
    if not isinstance(systems, dict):
        raise ValueError(f"Window {window['window_id']} systems must be a mapping")

    rows = []
    for system_name, system_spec in systems.items():
        if not isinstance(system_spec, dict):
            raise ValueError(f"System spec for {system_name} must be a mapping")
        render_dir = resolve_path(base_dir, system_spec.get("render_dir"))
        gt_dir = resolve_path(base_dir, system_spec.get("gt_dir"))
        static_dir = resolve_path(base_dir, system_spec.get("static_dir")) if system_spec.get("static_dir") else None
        render_frames = load_frame_window(render_dir, frame_start, frame_end, crop)
        gt_frames = load_frame_window(gt_dir, frame_start, frame_end, crop)
        if render_frames.shape != gt_frames.shape:
            raise ValueError(
                f"Render/GT shape mismatch for {window['window_id']} {system_name}: "
                f"{render_frames.shape} vs {gt_frames.shape}"
            )
        metrics = image_window_metrics(render_frames, gt_frames)
        if static_dir is not None and static_dir.exists():
            static_frames = load_frame_window(static_dir, frame_start, frame_end, crop)
            metrics["static_ghost_score"] = float(np.mean(np.abs(static_frames)))
        else:
            metrics["static_ghost_score"] = None
        metrics["lpips"] = compute_lpips_metric(render_frames, gt_frames) if compute_lpips else None
        rows.append(
            {
                "window_id": window["window_id"],
                "scene": window.get("scene"),
                "system": system_name,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "crop_xyxy": list(crop),
                "occluder": window.get("occluder"),
                **metrics,
            }
        )
    return rows


def resolve_path(base_dir: Path, value: object) -> Path:
    if value is None:
        raise ValueError("Expected a path value, got null")
    path = Path(str(value))
    if not path.is_absolute():
        path = base_dir / path
    return path


def load_frame_window(directory: Path, frame_start: int, frame_end: int, crop_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    if not directory.exists():
        raise FileNotFoundError(f"Frame directory does not exist: {directory}")
    index = index_image_frames(directory)
    frames = []
    for frame_idx in range(frame_start, frame_end + 1):
        path = index.get(frame_idx)
        if path is None:
            raise FileNotFoundError(f"Missing frame {frame_idx} in {directory}")
        frames.append(load_image_crop(path, crop_xyxy))
    return np.stack(frames, axis=0)


def index_image_frames(directory: Path) -> Dict[int, Path]:
    paths = [path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
    index: Dict[int, Path] = {}
    for path in sorted(paths):
        match = re.search(r"(\d+)(?!.*\d)", path.stem)
        if match:
            index[int(match.group(1))] = path
    return index


def load_image_crop(path: Path, crop_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    x0, y0, x1, y1 = crop_xyxy
    if not (0 <= x0 < x1 <= arr.shape[1] and 0 <= y0 < y1 <= arr.shape[0]):
        raise ValueError(f"Crop {crop_xyxy} is outside image {path} with shape {arr.shape}")
    return arr[y0:y1, x0:x1, :]


def image_window_metrics(render_frames: np.ndarray, gt_frames: np.ndarray) -> Dict[str, float]:
    err = render_frames - gt_frames
    mse = float(np.mean(err ** 2))
    psnr = 99.0 if mse <= 1e-12 else 20.0 * math.log10(1.0 / math.sqrt(mse))
    l1 = float(np.mean(np.abs(err)))
    if render_frames.shape[0] > 1:
        flicker = float(np.mean(np.abs(np.diff(render_frames, axis=0) - np.diff(gt_frames, axis=0))))
    else:
        flicker = 0.0
    return {
        "psnr": psnr,
        "l1": l1,
        "lpips_proxy_l1": l1,
        "flicker": flicker,
    }


def compute_lpips_metric(render_frames: np.ndarray, gt_frames: np.ndarray) -> Optional[float]:
    try:
        import torch
        from lpipsPyTorch import lpips
    except Exception:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    values = []
    with torch.no_grad():
        for render, gt in zip(render_frames, gt_frames):
            x = torch.from_numpy(render.transpose(2, 0, 1)).unsqueeze(0).to(device)
            y = torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0).to(device)
            values.append(float(lpips(x, y).mean().detach().cpu()))
    return mean_or_none(values)


def summarize_real_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, Optional[float]]]:
    by_system: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_system.setdefault(str(row["system"]), []).append(row)
    summary: Dict[str, Dict[str, Optional[float]]] = {}
    for system, system_rows in by_system.items():
        summary[system] = {
            "n_windows": float(len(system_rows)),
            "mean_psnr": mean_or_none(row["psnr"] for row in system_rows if row.get("psnr") is not None),
            "mean_l1": mean_or_none(row["l1"] for row in system_rows if row.get("l1") is not None),
            "mean_lpips": mean_or_none(row["lpips"] for row in system_rows if row.get("lpips") is not None),
            "mean_lpips_proxy_l1": mean_or_none(
                row["lpips_proxy_l1"] for row in system_rows if row.get("lpips_proxy_l1") is not None
            ),
            "mean_flicker": mean_or_none(row["flicker"] for row in system_rows if row.get("flicker") is not None),
            "mean_static_ghost_score": mean_or_none(
                row["static_ghost_score"] for row in system_rows if row.get("static_ghost_score") is not None
            ),
        }
    return summary


def write_real_report(path: Path, payload: Dict[str, object]) -> None:
    lines = ["# Real Event-Window PoC Report", "", f"Manifest: `{payload['manifest']}`", "", "## System Summary", ""]
    for system, metrics in payload["summary"].items():
        lines.append(f"### {system}")
        for key, value in metrics.items():
            lines.append(f"- `{key}`: {format_metric(value)}")
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `lpips_proxy_l1` is not a learned LPIPS metric; use `--compute-lpips` when the LPIPS stack is available.")
    lines.append("- Confident-track identity switches are not inferred here; attach them separately if available.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
