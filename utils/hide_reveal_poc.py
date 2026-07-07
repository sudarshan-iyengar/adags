import ast
import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image


TRUE_EVENT_TYPES = {"hide_reveal", "hide_only", "boundary_occlusion", "identity_confuser"}
REVEAL_EVENT_TYPES = {"hide_reveal", "boundary_occlusion", "identity_confuser"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
LPIPS_UNAVAILABLE_REASON: Optional[str] = None


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

    @property
    def requires_identity_reconnection(self) -> bool:
        return self.event_type in REVEAL_EVENT_TYPES


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
    matched_lifespan_identity_reconnected: bool
    no_identity_delta: float
    no_identity_accepted: bool
    no_identity_reconnected: bool
    unnormalized_delta: float
    unnormalized_accepted: bool
    no_hysteresis_accepted: bool
    identity_reconnected: bool


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
    if event_type == "identity_confuser":
        y0 += 2.0
        vx += 0.10
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
    unnormalized_accepted = selected and unnormalized_delta < -params.m_event
    no_hysteresis_accepted = score >= (params.c_min - 0.08) and delta < -params.m_event
    identity_reconnected = accepted and candidate.requires_identity_reconnection
    matched_lifespan_identity_reconnected = False
    no_identity_reconnected = False

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
        matched_lifespan_identity_reconnected=bool(matched_lifespan_identity_reconnected),
        no_identity_delta=float(no_id_delta),
        no_identity_accepted=bool(no_identity_accepted),
        no_identity_reconnected=bool(no_identity_reconnected),
        unnormalized_delta=float(unnormalized_delta),
        unnormalized_accepted=bool(unnormalized_accepted),
        no_hysteresis_accepted=bool(no_hysteresis_accepted),
        identity_reconnected=bool(identity_reconnected),
    )


def make_synthetic_candidates(
    seeds: Sequence[int],
    clips_per_type: int,
    params: FrozenHideRevealParams,
) -> List[SyntheticCandidate]:
    event_types = ["hide_reveal", "boundary_occlusion", "identity_confuser", "normal_motion", "distractor_motion"]
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
    reveal_rows = [result for result in rows if result.candidate.requires_identity_reconnection]
    normal_rows = [result for result in rows if not result.candidate.is_true_event]
    accepted = [result for result in rows if result.accepted]
    accepted_true = [result for result in accepted if result.candidate.is_true_event]
    selected_true = [result for result in true_rows if result.selected]
    selected_normal = [result for result in normal_rows if result.selected]
    lifespan_true = [result for result in true_rows if result.matched_lifespan_accepted]
    full_reconnected = [result for result in reveal_rows if result.identity_reconnected]
    lifespan_reconnected = [result for result in reveal_rows if result.matched_lifespan_identity_reconnected]
    no_identity_reconnected = [result for result in reveal_rows if result.no_identity_reconnected]

    auc = roc_auc([result.candidate.is_true_event for result in rows], [-result.delta_event for result in rows])
    candidate_auc = roc_auc([result.candidate.is_true_event for result in rows], [result.candidate_score for result in rows])

    return {
        "n": float(len(rows)),
        "true_events": float(len(true_rows)),
        "reveal_events": float(len(reveal_rows)),
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
        "identity_reconnection_accuracy": _safe_div(float(len(full_reconnected)), float(len(reveal_rows))),
        "matched_lifespan_accept_recall": _safe_div(float(len(lifespan_true)), float(len(true_rows))),
        "matched_lifespan_identity_reconnection_accuracy": _safe_div(
            float(len(lifespan_reconnected)),
            float(len(reveal_rows)),
        ),
        "no_identity_accept_recall": _safe_div(
            float(len([result for result in true_rows if result.no_identity_accepted])),
            float(len(true_rows)),
        ),
        "no_identity_identity_reconnection_accuracy": _safe_div(
            float(len(no_identity_reconnected)),
            float(len(reveal_rows)),
        ),
        "unnormalized_accept_recall": _safe_div(
            float(len([result for result in true_rows if result.unnormalized_accepted])),
            float(len(true_rows)),
        ),
        "unnormalized_false_event_rate": _safe_div(
            float(len([result for result in normal_rows if result.unnormalized_accepted])),
            float(len(normal_rows)),
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
    no_identity_identity = heldout.get("no_identity_identity_reconnection_accuracy")
    identity = metric_or_default(heldout.get("identity_reconnection_accuracy"), 0.0)
    pass_candidate = candidate_recall >= 0.85
    pass_margin = margin_auc >= 0.85 and false_event_rate <= 0.15
    pass_lifespan = lifespan_identity is not None and identity > lifespan_identity
    pass_no_identity = no_identity_identity is not None and identity > no_identity_identity
    return {
        "pass_candidate_recall": pass_candidate,
        "pass_margin_separation": pass_margin,
        "pass_matched_lifespan_gate": pass_lifespan,
        "pass_no_identity_deletion": pass_no_identity,
        "proceed_to_real_windows": bool(pass_candidate and pass_margin and pass_lifespan and pass_no_identity),
        "notes": [
            "Synthetic labels carry the identity claim; real windows should be sanity checks only.",
            "Matched lifespan can accept the same patch event, but gets no identity-reconnection credit because it has no hidden-identity reveal matching.",
            "The no-identity deletion can accept patch events, but should not pass the identity reconnection gate.",
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
            "matched_lifespan_identity_reconnected": result.matched_lifespan_identity_reconnected,
            "no_identity_delta": result.no_identity_delta,
            "no_identity_accepted": result.no_identity_accepted,
            "no_identity_reconnected": result.no_identity_reconnected,
            "unnormalized_delta": result.unnormalized_delta,
            "unnormalized_accepted": result.unnormalized_accepted,
            "no_hysteresis_accepted": result.no_hysteresis_accepted,
            "identity_reconnected": result.identity_reconnected,
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


@dataclass(frozen=True)
class EvalFrameRun:
    scene: str
    eval_dir: Path
    render_dir: Path
    gt_dir: Path
    static_dir: Optional[Path]


def discover_eval_frame_runs(search_roots: Sequence[Path], max_depth: int = 5) -> List[EvalFrameRun]:
    runs: List[EvalFrameRun] = []
    seen: Set[str] = set()
    for root in search_roots:
        root = root.expanduser()
        if not root.exists():
            raise FileNotFoundError(f"Eval root does not exist: {root}")
        root = root.resolve()
        for eval_dir in iter_eval_dirs(root, max_depth=max_depth):
            key = str(eval_dir.resolve())
            if key in seen:
                continue
            seen.add(key)
            runs.append(
                EvalFrameRun(
                    scene=infer_scene_name(eval_dir),
                    eval_dir=eval_dir,
                    render_dir=eval_dir / "renders",
                    gt_dir=eval_dir / "gt",
                    static_dir=(eval_dir / "static") if (eval_dir / "static").is_dir() else None,
                )
            )
    return sorted(runs, key=lambda run: (run.scene, str(run.eval_dir)))


def iter_eval_dirs(root: Path, max_depth: int = 5) -> Iterable[Path]:
    if is_eval_dir(root):
        yield root
        return
    stack: List[Tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > max_depth:
            continue
        if is_eval_dir(current):
            yield current
            continue
        if depth == max_depth:
            continue
        try:
            children = [child for child in current.iterdir() if child.is_dir()]
        except OSError:
            continue
        stack.extend((child, depth + 1) for child in reversed(children))


def is_eval_dir(path: Path) -> bool:
    return (path / "renders").is_dir() and (path / "gt").is_dir()


def infer_scene_name(eval_dir: Path) -> str:
    parent = eval_dir.parent
    if eval_dir.name.startswith("ours_"):
        if parent.name.lower() == "test" and parent.parent != parent:
            return parent.parent.name
        return parent.name
    if parent.name.lower() == "test" and parent.parent != parent:
        return parent.parent.name
    return parent.name or eval_dir.name


def parse_crop_xyxy(value: object) -> Tuple[int, int, int, int]:
    if isinstance(value, str):
        pieces = [piece.strip() for piece in re.split(r"[, ]+", value.strip()) if piece.strip()]
    elif isinstance(value, Sequence):
        pieces = list(value)
    else:
        raise ValueError(f"Crop must be a 4-value list or comma string, got {value!r}")
    if len(pieces) != 4:
        raise ValueError(f"Crop must have four values [x0, y0, x1, y1], got {value!r}")
    crop = tuple(int(piece) for piece in pieces)
    x0, y0, x1, y1 = crop
    if not (0 <= x0 < x1 and 0 <= y0 < y1):
        raise ValueError(f"Invalid crop_xyxy {crop}; expected 0 <= x0 < x1 and 0 <= y0 < y1")
    return crop


def image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def first_indexed_frame(index: Dict[int, Path]) -> Optional[Path]:
    if not index:
        return None
    return index[min(index.keys())]


def common_frame_numbers(run: EvalFrameRun) -> List[int]:
    render_index = index_image_frames(run.render_dir)
    gt_index = index_image_frames(run.gt_dir)
    return sorted(set(render_index) & set(gt_index))


def contiguous_segments(frames: Sequence[int]) -> List[List[int]]:
    if not frames:
        return []
    segments: List[List[int]] = [[int(frames[0])]]
    for frame in frames[1:]:
        frame = int(frame)
        if frame == segments[-1][-1] + 1:
            segments[-1].append(frame)
        else:
            segments.append([frame])
    return segments


def sample_frame_windows(frames: Sequence[int], num_windows: int, window_length: int) -> List[Tuple[int, int]]:
    if num_windows <= 0:
        raise ValueError("--num-windows must be positive")
    if window_length <= 0:
        raise ValueError("--window-length must be positive")
    starts: List[int] = []
    for segment in contiguous_segments(frames):
        if len(segment) < window_length:
            continue
        starts.extend(segment[: len(segment) - window_length + 1])
    if not starts:
        return []
    if len(starts) <= num_windows:
        chosen = starts
    else:
        idxs = np.linspace(0, len(starts) - 1, num_windows)
        chosen = [starts[int(round(idx))] for idx in idxs]
    deduped: List[int] = []
    for start in chosen:
        if start not in deduped:
            deduped.append(start)
    return [(start, start + window_length - 1) for start in deduped]


def run_system_spec(run: EvalFrameRun) -> Dict[str, str]:
    spec = {
        "render_dir": str(run.render_dir.resolve()),
        "gt_dir": str(run.gt_dir.resolve()),
    }
    if run.static_dir is not None:
        spec["static_dir"] = str(run.static_dir.resolve())
    return spec


def manifest_payload(windows: Sequence[Dict[str, object]], source: str = "real-manifest-from-eval") -> Dict[str, object]:
    return {
        "description": "Predeclared real occlusion/reveal windows for hide/reveal PoC scoring.",
        "frames_are_inclusive": True,
        "generated_by": source,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "windows": list(windows),
    }


def load_window_specs(json_path: Optional[Path] = None, csv_path: Optional[Path] = None) -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []
    if json_path is not None:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "windows" in data:
                raw_windows = data["windows"]
            else:
                raw_windows = [data]
        elif isinstance(data, list):
            raw_windows = data
        else:
            raise ValueError(f"Window JSON must be a manifest, object, or list: {json_path}")
        if not isinstance(raw_windows, list):
            raise ValueError(f"Window JSON `windows` must be a list: {json_path}")
        specs.extend(dict(window) for window in raw_windows)
    if csv_path is not None:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            specs.extend(csv_row_to_window_spec(row) for row in reader)
    return specs


def csv_row_to_window_spec(row: Dict[str, str]) -> Dict[str, object]:
    spec: Dict[str, object] = {
        key: value
        for key, value in row.items()
        if value is not None and str(value).strip() != "" and not key.endswith("_dir")
    }
    if "frame_start" in spec:
        spec["frame_start"] = int(spec["frame_start"])
    if "frame_end" in spec:
        spec["frame_end"] = int(spec["frame_end"])
    if "crop_xyxy" in spec:
        spec["crop_xyxy"] = list(parse_crop_xyxy(spec["crop_xyxy"]))
    elif all(key in row and row[key] for key in ("x0", "y0", "x1", "y1")):
        spec["crop_xyxy"] = [int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"])]

    systems: Dict[str, Dict[str, str]] = {}
    for system_name in sorted({key[:-11] for key in row if key.endswith("_render_dir") and key != "render_dir"}):
        render_dir = row.get(f"{system_name}_render_dir")
        gt_dir = row.get(f"{system_name}_gt_dir")
        if render_dir and gt_dir:
            systems[system_name] = {"render_dir": render_dir, "gt_dir": gt_dir}
            static_dir = row.get(f"{system_name}_static_dir")
            if static_dir:
                systems[system_name]["static_dir"] = static_dir
    if row.get("render_dir") and row.get("gt_dir"):
        systems.setdefault("route0", {"render_dir": row["render_dir"], "gt_dir": row["gt_dir"]})
        if row.get("static_dir"):
            systems["route0"]["static_dir"] = row["static_dir"]
    if systems:
        spec["systems"] = systems
    return spec


def direct_window_spec(
    scene: Optional[str],
    frame_start: Optional[int],
    frame_end: Optional[int],
    crop_xyxy: Optional[Sequence[int]],
    occluder: str,
    notes: str,
) -> Optional[Dict[str, object]]:
    if frame_start is None and frame_end is None:
        return None
    if frame_start is None or frame_end is None:
        raise ValueError("--frame-start and --frame-end must be provided together for a direct CLI window")
    spec: Dict[str, object] = {
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "occluder": occluder,
        "notes": notes,
    }
    if scene:
        spec["scene"] = scene
    if crop_xyxy is not None:
        spec["crop_xyxy"] = list(parse_crop_xyxy(crop_xyxy))
    return spec


def create_real_manifest_from_eval(
    search_roots: Sequence[Path],
    out_path: Path,
    system_name: str = "route0",
    scene: Optional[str] = None,
    num_windows: int = 6,
    window_length: int = 16,
    crop_xyxy: Optional[Sequence[int]] = None,
    occluder: str = "TBD_PREDECLARE",
    notes: str = "Review and freeze this candidate window before scoring.",
    window_specs: Optional[Sequence[Dict[str, object]]] = None,
    max_depth: int = 5,
) -> Dict[str, object]:
    runs = discover_eval_frame_runs(search_roots, max_depth=max_depth)
    if not runs:
        roots = ", ".join(str(root) for root in search_roots)
        raise FileNotFoundError(f"No eval folders with renders/ and gt/ found under: {roots}")
    if scene:
        runs = [replace_eval_scene(run, scene) for run in runs]
    run_by_scene = make_run_lookup(runs)
    windows = (
        windows_from_user_specs(window_specs or [], run_by_scene, runs, system_name, crop_xyxy, occluder, notes)
        if window_specs
        else windows_from_discovered_runs(runs, system_name, num_windows, window_length, crop_xyxy, occluder, notes)
    )
    if not windows:
        raise ValueError("No manifest windows could be created. Check frame indices and window length.")
    payload = manifest_payload(windows)
    write_json(out_path, payload)
    return {
        "manifest": payload,
        "out_path": str(out_path),
        "eval_runs": [eval_run_to_row(run) for run in runs],
    }


def replace_eval_scene(run: EvalFrameRun, scene: str) -> EvalFrameRun:
    return EvalFrameRun(
        scene=scene,
        eval_dir=run.eval_dir,
        render_dir=run.render_dir,
        gt_dir=run.gt_dir,
        static_dir=run.static_dir,
    )


def make_run_lookup(runs: Sequence[EvalFrameRun]) -> Dict[str, EvalFrameRun]:
    lookup: Dict[str, EvalFrameRun] = {}
    for run in runs:
        lookup.setdefault(run.scene, run)
        lookup.setdefault(str(run.eval_dir), run)
        lookup.setdefault(run.eval_dir.name, run)
    return lookup


def eval_run_to_row(run: EvalFrameRun) -> Dict[str, object]:
    return {
        "scene": run.scene,
        "eval_dir": str(run.eval_dir),
        "render_dir": str(run.render_dir),
        "gt_dir": str(run.gt_dir),
        "static_dir": str(run.static_dir) if run.static_dir is not None else None,
        "n_common_frames": len(common_frame_numbers(run)),
    }


def windows_from_discovered_runs(
    runs: Sequence[EvalFrameRun],
    system_name: str,
    num_windows: int,
    window_length: int,
    crop_xyxy: Optional[Sequence[int]],
    occluder: str,
    notes: str,
) -> List[Dict[str, object]]:
    windows: List[Dict[str, object]] = []
    remaining = num_windows
    for run in runs:
        if remaining <= 0:
            break
        frames = common_frame_numbers(run)
        sampled = sample_frame_windows(frames, remaining, window_length)
        if not sampled:
            continue
        crop = list(resolve_default_crop(run, crop_xyxy))
        for frame_start, frame_end in sampled:
            window_idx = len(windows) + 1
            windows.append(
                {
                    "window_id": f"{run.scene}_window{window_idx:02d}",
                    "scene": run.scene,
                    "frame_start": int(frame_start),
                    "frame_end": int(frame_end),
                    "crop_xyxy": crop,
                    "occluder": occluder,
                    "notes": notes,
                    "systems": {system_name: run_system_spec(run)},
                }
            )
            remaining -= 1
            if remaining <= 0:
                break
    return windows


def windows_from_user_specs(
    specs: Sequence[Dict[str, object]],
    run_by_scene: Dict[str, EvalFrameRun],
    runs: Sequence[EvalFrameRun],
    system_name: str,
    fallback_crop: Optional[Sequence[int]],
    fallback_occluder: str,
    fallback_notes: str,
) -> List[Dict[str, object]]:
    windows: List[Dict[str, object]] = []
    for idx, spec in enumerate(specs, start=1):
        window = dict(spec)
        if "frame_start" not in window or "frame_end" not in window:
            raise ValueError(f"Window spec {idx} must include frame_start and frame_end")
        scene = str(window.get("scene") or "")
        run = run_by_scene.get(scene) if scene else None
        if run is None and len(runs) == 1:
            run = runs[0]
            scene = scene or run.scene
        if run is None and "systems" not in window:
            known = ", ".join(sorted({run.scene for run in runs}))
            raise ValueError(f"Window spec {idx} has no matching scene. Known scenes: {known}")
        if "crop_xyxy" in window:
            crop = list(parse_crop_xyxy(window["crop_xyxy"]))
        elif fallback_crop is not None:
            crop = list(parse_crop_xyxy(fallback_crop))
        elif run is not None:
            crop = list(resolve_default_crop(run, None))
        else:
            raise ValueError(f"Window spec {idx} needs crop_xyxy when no eval run is matched")
        window["window_id"] = window.get("window_id") or f"{scene or 'scene'}_window{idx:02d}"
        window["scene"] = scene or window.get("scene") or "unknown_scene"
        window["frame_start"] = int(window["frame_start"])
        window["frame_end"] = int(window["frame_end"])
        window["crop_xyxy"] = crop
        window["occluder"] = window.get("occluder") or fallback_occluder
        window["notes"] = window.get("notes") or fallback_notes
        if "systems" not in window:
            if run is None:
                raise ValueError(f"Window spec {idx} needs systems or a matching eval run")
            window["systems"] = {system_name: run_system_spec(run)}
        windows.append(window)
    return windows


def resolve_default_crop(run: EvalFrameRun, crop_xyxy: Optional[Sequence[int]]) -> Tuple[int, int, int, int]:
    if crop_xyxy is not None:
        return parse_crop_xyxy(crop_xyxy)
    first = first_indexed_frame(index_image_frames(run.render_dir))
    if first is None:
        raise FileNotFoundError(f"No indexed render frames found in {run.render_dir}")
    width, height = image_size(first)
    return (0, 0, width, height)


def validate_real_manifest(
    manifest_path: Path,
    require_systems: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require_systems = list(require_systems or [])
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be a JSON object: {manifest_path}")
    windows = manifest.get("windows")
    if not isinstance(windows, list):
        errors.append("Manifest must contain a `windows` list.")
        windows = []
    if len(windows) < 4:
        warnings.append(f"R009 expected 4-6 windows; manifest has {len(windows)}.")
    if len(windows) > 6:
        warnings.append(f"R009 expected 4-6 windows; manifest has {len(windows)}.")

    window_reports = []
    for idx, window in enumerate(windows, start=1):
        report = validate_real_window_spec(window, manifest_path.parent, require_systems, idx)
        window_reports.append(report)
        errors.extend(f"{report['window_id']}: {message}" for message in report["errors"])
        warnings.extend(f"{report['window_id']}: {message}" for message in report["warnings"])
    return {
        "manifest": str(manifest_path),
        "ok": not errors,
        "n_windows": len(windows),
        "errors": errors,
        "warnings": warnings,
        "windows": window_reports,
    }


def validate_real_window_spec(
    window: object,
    base_dir: Path,
    require_systems: Sequence[str],
    fallback_idx: int,
) -> Dict[str, object]:
    window_id = f"window{fallback_idx:02d}"
    errors: List[str] = []
    warnings: List[str] = []
    system_reports: Dict[str, object] = {}
    if not isinstance(window, dict):
        return {
            "window_id": window_id,
            "ok": False,
            "errors": ["Window must be a JSON object."],
            "warnings": [],
            "systems": {},
        }
    window_id = str(window.get("window_id") or window_id)
    for key in ("scene", "frame_start", "frame_end", "crop_xyxy", "systems"):
        if key not in window:
            errors.append(f"Missing required key `{key}`.")
    try:
        frame_start = int(window.get("frame_start"))
        frame_end = int(window.get("frame_end"))
        if frame_end < frame_start:
            errors.append("frame_end must be >= frame_start.")
    except Exception:
        frame_start = 0
        frame_end = -1
        errors.append("frame_start and frame_end must be integers.")
    try:
        crop = parse_crop_xyxy(window.get("crop_xyxy"))
    except Exception as exc:
        crop = (0, 0, 1, 1)
        errors.append(str(exc))

    systems = window.get("systems")
    if not isinstance(systems, dict):
        systems = {}
        errors.append("systems must be a mapping.")
    for system_name in require_systems:
        if system_name not in systems:
            errors.append(f"Missing required system `{system_name}`.")
    for system_name, system_spec in systems.items():
        if not isinstance(system_spec, dict):
            system_reports[str(system_name)] = {
                "ok": False,
                "errors": ["System spec must be a mapping."],
                "warnings": [],
            }
            errors.append(f"System `{system_name}` spec must be a mapping.")
            continue
        report = validate_real_system_spec(str(system_name), system_spec, base_dir, frame_start, frame_end, crop)
        system_reports[str(system_name)] = report
        errors.extend(f"{system_name}: {message}" for message in report["errors"])
        warnings.extend(f"{system_name}: {message}" for message in report["warnings"])

    return {
        "window_id": window_id,
        "scene": window.get("scene"),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "systems": system_reports,
    }


def validate_real_system_spec(
    system_name: str,
    system_spec: Dict[str, object],
    base_dir: Path,
    frame_start: int,
    frame_end: int,
    crop_xyxy: Tuple[int, int, int, int],
) -> Dict[str, object]:
    errors: List[str] = []
    warnings: List[str] = []
    paths: Dict[str, str] = {}
    for key in ("render_dir", "gt_dir"):
        if not system_spec.get(key):
            errors.append(f"Missing `{key}`.")
            continue
        path = resolve_path(base_dir, system_spec[key])
        paths[key] = str(path)
        if not path.is_dir():
            errors.append(f"{key} does not exist or is not a directory: {path}")
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings, "paths": paths}

    render_dir = resolve_path(base_dir, system_spec["render_dir"])
    gt_dir = resolve_path(base_dir, system_spec["gt_dir"])
    render_index = index_image_frames(render_dir)
    gt_index = index_image_frames(gt_dir)
    required_frames = list(range(frame_start, frame_end + 1))
    missing_render = [frame for frame in required_frames if frame not in render_index]
    missing_gt = [frame for frame in required_frames if frame not in gt_index]
    if missing_render:
        errors.append(f"Missing render frames: {summarize_missing_frames(missing_render)}")
    if missing_gt:
        errors.append(f"Missing gt frames: {summarize_missing_frames(missing_gt)}")
    check_path = render_index.get(frame_start) or first_indexed_frame(render_index)
    if check_path is not None:
        width, height = image_size(check_path)
        x0, y0, x1, y1 = crop_xyxy
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            errors.append(f"Crop {crop_xyxy} is outside render frame bounds {(width, height)}.")
    else:
        errors.append(f"No indexed render frames found in {render_dir}.")

    static_value = system_spec.get("static_dir")
    if static_value:
        static_dir = resolve_path(base_dir, static_value)
        paths["static_dir"] = str(static_dir)
        if static_dir.is_dir():
            static_index = index_image_frames(static_dir)
            missing_static = [frame for frame in required_frames if frame not in static_index]
            if missing_static:
                warnings.append(f"Missing optional static frames: {summarize_missing_frames(missing_static)}")
        else:
            warnings.append(f"Optional static_dir does not exist or is not a directory: {static_dir}")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "paths": paths,
        "n_frames": max(0, frame_end - frame_start + 1),
    }


def derive_real_poc_render_folders(
    manifest_path: Path,
    out_dir: Path,
    route0_eval_dir: Optional[Path] = None,
    route0_system: str = "route0",
    hide_reveal_strength: float = 1.0,
    matched_lifespan_strength: float = 0.35,
    event_beta: float = 1.0,
    feather_px: int = 8,
    overwrite: bool = False,
) -> Dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be a JSON object: {manifest_path}")
    windows = manifest.get("windows")
    if not isinstance(windows, list):
        raise ValueError(f"Manifest must contain a `windows` list: {manifest_path}")
    if not windows:
        raise ValueError("Manifest has no windows to derive")

    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = manifest_path.parent
    override_spec = route0_eval_system_spec(route0_eval_dir) if route0_eval_dir is not None else None
    groups: Dict[str, Dict[str, object]] = {}
    augmented_windows: List[Dict[str, object]] = []
    metadata_windows: List[Dict[str, object]] = []

    for idx, raw_window in enumerate(windows, start=1):
        if not isinstance(raw_window, dict):
            raise ValueError(f"Window {idx} must be a JSON object")
        window = dict(raw_window)
        window_id = str(window.get("window_id") or f"window{idx:02d}")
        scene = str(window.get("scene") or "unknown_scene")
        frame_start = int(window["frame_start"])
        frame_end = int(window["frame_end"])
        if frame_end < frame_start:
            raise ValueError(f"{window_id}: frame_end must be >= frame_start")
        crop = parse_crop_xyxy(window["crop_xyxy"])
        raw_systems = window.get("systems")
        systems = dict(raw_systems) if isinstance(raw_systems, dict) else {}
        route0_spec = override_spec or systems.get(route0_system)
        if not isinstance(route0_spec, dict):
            raise ValueError(f"{window_id}: missing `{route0_system}` system and no --route0-eval override was given")
        source_spec = absolute_system_spec(route0_spec, base_dir)
        check = validate_real_system_spec(route0_system, source_spec, base_dir, frame_start, frame_end, crop)
        if not check["ok"]:
            joined = "; ".join(str(error) for error in check["errors"])
            raise ValueError(f"{window_id}: source route0 validation failed: {joined}")

        group_id = source_group_id(scene, source_spec["render_dir"], source_spec["gt_dir"])
        group_root = out_dir / "derived_renders"
        hide_render_dir = group_root / "hide_reveal" / group_id / "renders"
        lifespan_render_dir = group_root / "matched_lifespan" / group_id / "renders"
        group = groups.setdefault(
            group_id,
            {
                "scene": scene,
                "source": source_spec,
                "hide_render_dir": hide_render_dir,
                "lifespan_render_dir": lifespan_render_dir,
                "windows": [],
            },
        )
        group["windows"].append(
            {
                "window_id": window_id,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "crop_xyxy": crop,
            }
        )

        systems[route0_system] = source_spec
        systems["matched_lifespan"] = derived_system_spec(lifespan_render_dir, source_spec)
        systems["hide_reveal"] = derived_system_spec(hide_render_dir, source_spec)
        window["window_id"] = window_id
        window["scene"] = scene
        window["frame_start"] = frame_start
        window["frame_end"] = frame_end
        window["crop_xyxy"] = list(crop)
        window["systems"] = systems
        augmented_windows.append(window)
        metadata_windows.append(
            {
                "window_id": window_id,
                "scene": scene,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "crop_xyxy": list(crop),
                "source_render_dir": source_spec["render_dir"],
                "source_gt_dir": source_spec["gt_dir"],
                "matched_lifespan_render_dir": str(lifespan_render_dir.resolve()),
                "hide_reveal_render_dir": str(hide_render_dir.resolve()),
            }
        )

    group_reports = []
    for group_id, group in groups.items():
        report = write_derived_group_renders(
            group_id=group_id,
            source_spec=group["source"],
            windows=group["windows"],
            hide_render_dir=group["hide_render_dir"],
            lifespan_render_dir=group["lifespan_render_dir"],
            hide_reveal_strength=hide_reveal_strength,
            matched_lifespan_strength=matched_lifespan_strength,
            event_beta=event_beta,
            feather_px=feather_px,
            overwrite=overwrite,
        )
        group_reports.append(report)

    augmented_manifest = dict(manifest)
    augmented_manifest["generated_by"] = "derive-real-renders"
    augmented_manifest["derived_poc_outputs"] = {
        "is_trained_model_output": False,
        "description": (
            "R012/R013 proof-of-concept derived image folders. hide_reveal composites GT crops into "
            "predeclared event windows as a local counterfactual target; matched_lifespan uses the same "
            "temporal/crop budget but a route0 temporal reference without identity-aware reveal matching."
        ),
    }
    augmented_manifest["windows"] = augmented_windows
    manifest_out = out_dir / "derived_real_windows_manifest.json"
    write_json(manifest_out, augmented_manifest)

    metadata = {
        "generated_by": "derive-real-renders",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_manifest": str(manifest_path.resolve()),
        "output_manifest": str(manifest_out.resolve()),
        "route0_system": route0_system,
        "route0_eval_override": str(route0_eval_dir.resolve()) if route0_eval_dir is not None else None,
        "is_trained_model_output": False,
        "parameters": {
            "hide_reveal_strength": float(hide_reveal_strength),
            "matched_lifespan_strength": float(matched_lifespan_strength),
            "event_beta": float(event_beta),
            "feather_px": int(feather_px),
        },
        "limitations": [
            "Derived image-level PoC composite; no GaussianModel state was trained or checkpointed.",
            "hide_reveal uses GT only inside predeclared event crops to test the maximum possible local artifact effect.",
            "matched_lifespan uses the same windows and crop budget but no identity-aware hidden/reveal target.",
        ],
        "windows": metadata_windows,
        "groups": group_reports,
    }
    metadata_out = out_dir / "derived_poc_metadata.json"
    write_json(metadata_out, metadata)
    validation = validate_real_manifest(manifest_out, require_systems=[route0_system, "matched_lifespan", "hide_reveal"])
    write_json(out_dir / "derived_real_windows_validation.json", validation)
    return {
        "manifest": augmented_manifest,
        "manifest_path": str(manifest_out),
        "metadata": metadata,
        "metadata_path": str(metadata_out),
        "validation": validation,
    }


def route0_eval_system_spec(eval_dir: Path) -> Dict[str, str]:
    eval_dir = eval_dir.expanduser().resolve()
    render_dir = eval_dir / "renders"
    gt_dir = eval_dir / "gt"
    if not render_dir.is_dir() or not gt_dir.is_dir():
        raise FileNotFoundError(f"Expected renders/ and gt/ under route0 eval dir: {eval_dir}")
    spec = {"render_dir": str(render_dir), "gt_dir": str(gt_dir)}
    static_dir = eval_dir / "static"
    if static_dir.is_dir():
        spec["static_dir"] = str(static_dir)
    return spec


def absolute_system_spec(system_spec: Dict[str, object], base_dir: Path) -> Dict[str, str]:
    spec = {
        "render_dir": str(resolve_path(base_dir, system_spec["render_dir"]).resolve()),
        "gt_dir": str(resolve_path(base_dir, system_spec["gt_dir"]).resolve()),
    }
    if system_spec.get("static_dir"):
        spec["static_dir"] = str(resolve_path(base_dir, system_spec["static_dir"]).resolve())
    return spec


def derived_system_spec(render_dir: Path, source_spec: Dict[str, str]) -> Dict[str, str]:
    spec = {
        "render_dir": str(render_dir.resolve()),
        "gt_dir": source_spec["gt_dir"],
    }
    if source_spec.get("static_dir"):
        spec["static_dir"] = source_spec["static_dir"]
    return spec


def source_group_id(scene: str, render_dir: str, gt_dir: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", scene).strip("_") or "scene"
    digest = hashlib.sha1(f"{render_dir}|{gt_dir}".encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{digest}"


def infer_run_dir_from_eval_render_dir(render_dir: Path) -> Path:
    parts = list(render_dir.parts)
    if len(parts) >= 3 and parts[-1] == "renders" and parts[-3] == "test":
        return Path(*parts[:-3])
    if len(parts) >= 2 and parts[-1] == "renders":
        return render_dir.parents[1]
    raise ValueError(f"Could not infer run dir from render_dir={render_dir}")


def infer_checkpoint_from_render_dir(render_dir: Path) -> Tuple[Path, int]:
    run_dir = infer_run_dir_from_eval_render_dir(render_dir)
    iteration = None
    if render_dir.parent.name.startswith("ours_"):
        try:
            iteration = int(render_dir.parent.name.split("_", 1)[1])
        except ValueError:
            iteration = None
    if iteration is None:
        iteration = 6000
    checkpoint = run_dir / f"chkpnt{iteration}.pth"
    return checkpoint, iteration


def parse_simple_yaml_scalar(value: str) -> Any:
    value = value.split("#", 1)[0].strip()
    if not value:
        return ""
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value.strip("\"'")


def load_simple_yaml_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except ImportError:
        pass

    data: Dict[str, Any] = {}
    current_section: Optional[str] = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        if line.startswith(" "):
            if current_section is None or not isinstance(data.get(current_section), dict):
                continue
            key, value = line.strip().split(":", 1)
            data[current_section][key.strip()] = parse_simple_yaml_scalar(value)
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if value.strip():
            data[key] = parse_simple_yaml_scalar(value)
            current_section = None
        else:
            data[key] = {}
            current_section = key
    return data


def load_run_family_config(run_dir: Path) -> Tuple[Optional[Path], Dict[str, Any]]:
    config_dir = Path.cwd() / "configs" / "n3v"
    candidates = []
    if run_dir.parent.name:
        candidates.append(config_dir / f"{run_dir.parent.name}.yaml")
    if config_dir.exists():
        for config_path in sorted(config_dir.glob("*.yaml")):
            if run_dir.name.endswith(f"_{config_path.stem}"):
                candidates.append(config_path)
    for config_path in candidates:
        if config_path.exists():
            return config_path, load_simple_yaml_config(config_path)
    return None, {}


def apply_run_config_defaults(args: object, run_dir: Path) -> Optional[Path]:
    config_path, config = load_run_family_config(run_dir)
    if not config:
        return config_path
    for key, value in config.items():
        if isinstance(value, dict):
            continue
        if not hasattr(args, key):
            setattr(args, key, value)
    for section in ("ModelParams", "PipelineParams"):
        section_values = config.get(section, {})
        if not isinstance(section_values, dict):
            continue
        for key, value in section_values.items():
            if not hasattr(args, key):
                setattr(args, key, value)
    return config_path


def infer_gaussian_dim_from_checkpoint_args(model_args: object) -> Optional[int]:
    if not isinstance(model_args, (tuple, list)):
        return None
    if len(model_args) == 12:
        return 3
    if len(model_args) >= 29:
        return 4
    return None


def load_run_cfg_args(run_dir: Path, scene: str) -> object:
    from argparse import Namespace

    cfg_path = run_dir / "cfg_args"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing cfg_args for checkpoint-backed render stage: {cfg_path}")
    args = eval(cfg_path.read_text(encoding="utf-8"), {"Namespace": Namespace})
    config_path = apply_run_config_defaults(args, run_dir)
    args.hide_reveal_config_path = str(config_path) if config_path else None
    args.model_path = str(run_dir)
    if not getattr(args, "source_path", ""):
        args.source_path = str(Path("/leonardo_work/EUHPC_D21_034/proj_adags/data/n3v") / scene)
    args.eval = True
    return args


def merge_baseline_systems(
    window: Dict[str, object],
    baseline_windows: Sequence[Dict[str, object]],
    system_names: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    window_id = str(window["window_id"])
    merged = {}
    for baseline in baseline_windows:
        if str(baseline.get("window_id")) != window_id:
            continue
        systems = baseline.get("systems", {})
        if not isinstance(systems, dict):
            continue
        for system_name in system_names:
            if system_name in systems:
                merged[system_name] = systems[system_name]
    return merged


def gray_image(path: Path, target_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if target_hw is not None and arr.shape != tuple(target_hw):
        height, width = target_hw
        pil = Image.fromarray(np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8), mode="L")
        arr = np.asarray(pil.resize((int(width), int(height)), Image.BILINEAR), dtype=np.float32) / 255.0
    return arr.astype(np.float32)


def indexed_gray(index: Dict[int, Path], frame_idx: int, target_hw: Tuple[int, int]) -> np.ndarray:
    path = index.get(int(frame_idx))
    if path is None:
        return np.zeros(target_hw, dtype=np.float32)
    return gray_image(path, target_hw=target_hw)


def edge_map(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return mask.astype(np.float32)
    dx = np.zeros_like(mask, dtype=np.float32)
    dy = np.zeros_like(mask, dtype=np.float32)
    dx[:, 1:] = np.abs(mask[:, 1:] - mask[:, :-1])
    dy[1:, :] = np.abs(mask[1:, :] - mask[:-1, :])
    return np.clip(dx + dy, 0.0, 1.0)


def crop_iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax0, ay0, ax1, ay1 = [int(v) for v in a]
    bx0, by0, bx1, by1 = [int(v) for v in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def temporal_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(int(a_end), int(b_end)) - max(int(a_start), int(b_start)) + 1)
    union = max(int(a_end), int(b_end)) - min(int(a_start), int(b_start)) + 1
    return float(inter / union) if union > 0 else 0.0


def nms_candidate_rows(
    rows: Sequence[Dict[str, object]],
    max_candidates: int,
    crop_iou_threshold: float,
    temporal_iou_threshold: float,
) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = []
    for row in sorted(rows, key=lambda item: float(item["score"]), reverse=True):
        suppress = False
        for kept in selected:
            if str(row["scene"]) != str(kept["scene"]):
                continue
            if crop_iou(row["crop_xyxy"], kept["crop_xyxy"]) < crop_iou_threshold:
                continue
            if temporal_iou(
                int(row["frame_start"]),
                int(row["frame_end"]),
                int(kept["frame_start"]),
                int(kept["frame_end"]),
            ) < temporal_iou_threshold:
                continue
            suppress = True
            break
        if not suppress:
            selected.append(dict(row))
        if len(selected) >= max_candidates:
            break
    return selected


def route0_spec_from_scene_source(scene_source: Dict[str, object]) -> Dict[str, str]:
    eval_dir = Path(str(scene_source["route0_eval_dir"]))
    spec = {
        "render_dir": str((eval_dir / "renders").resolve()),
        "gt_dir": str((eval_dir / "gt").resolve()),
    }
    static_dir = eval_dir / "static"
    dynamic_dir = eval_dir / "dynamic"
    if static_dir.is_dir() or str(scene_source.get("route0_eval_dir", "")).startswith("/"):
        spec["static_dir"] = str(static_dir.resolve())
    if dynamic_dir.is_dir() or str(scene_source.get("route0_eval_dir", "")).startswith("/"):
        spec["dynamic_dir"] = str(dynamic_dir.resolve())
    return spec


def score_maps_for_scene(
    scene_source: Dict[str, object],
    frame_ids: Sequence[int],
    route0_system: str,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Tuple[int, int], Dict[str, object]]:
    eval_dir = Path(str(scene_source["route0_eval_dir"]))
    render_dir = eval_dir / "renders"
    static_dir = eval_dir / "static"
    dynamic_dir = eval_dir / "dynamic"
    mask_dir = Path(str(scene_source.get("mask_dir", ""))) if scene_source.get("mask_dir") else None
    if not render_dir.is_dir():
        raise FileNotFoundError(f"Route0 render directory does not exist: {render_dir}")

    render_index = index_image_frames(render_dir)
    static_index = index_image_frames(static_dir) if static_dir.is_dir() else {}
    dynamic_index = index_image_frames(dynamic_dir) if dynamic_dir.is_dir() else {}
    mask_index = index_image_frames(mask_dir) if mask_dir is not None and mask_dir.is_dir() else {}

    first_path = first_indexed_frame(render_index)
    if first_path is None:
        raise FileNotFoundError(f"No route0 render frames found in {render_dir}")
    width, height = image_size(first_path)
    target_hw = (height, width)
    maps: Dict[int, Dict[str, np.ndarray]] = {}
    prev_render: Optional[np.ndarray] = None
    used_frames = []
    for frame_idx in frame_ids:
        render_path = render_index.get(int(frame_idx))
        if render_path is None:
            continue
        render_gray = gray_image(render_path, target_hw=target_hw)
        dynamic_gray = indexed_gray(dynamic_index, int(frame_idx), target_hw)
        static_gray = indexed_gray(static_index, int(frame_idx), target_hw)
        static_delta = np.abs(render_gray - static_gray).astype(np.float32) if static_index else np.zeros(target_hw, dtype=np.float32)
        mask_gray = indexed_gray(mask_index, int(frame_idx), target_hw)
        motion_support = mask_gray if mask_index else np.ones(target_hw, dtype=np.float32)
        mask_boundary = edge_map(mask_gray)
        if prev_render is None:
            flicker_gray = np.zeros(target_hw, dtype=np.float32)
        else:
            flicker_gray = np.abs(render_gray - prev_render).astype(np.float32)
        prev_render = render_gray
        dynamic_motion = (dynamic_gray * motion_support).astype(np.float32)
        static_delta_motion = (static_delta * motion_support).astype(np.float32)
        flicker_motion = (flicker_gray * np.maximum(motion_support, mask_boundary)).astype(np.float32)
        score = (
            0.35 * dynamic_motion
            + 0.25 * static_delta_motion
            + 0.25 * motion_support
            + 0.10 * mask_boundary
            + 0.05 * flicker_motion
        )
        maps[int(frame_idx)] = {
            "score": np.clip(score, 0.0, 1.0).astype(np.float32),
            "dynamic_motion": dynamic_motion,
            "static_delta_motion": static_delta_motion,
            "motion_mask": motion_support.astype(np.float32),
            "mask_boundary": mask_boundary,
            "flicker_motion": flicker_motion,
        }
        used_frames.append(int(frame_idx))

    metadata = {
        "route0_system": route0_system,
        "route0_eval_dir": str(eval_dir),
        "render_dir": str(render_dir),
        "static_dir": str(static_dir),
        "dynamic_dir": str(dynamic_dir),
        "mask_dir": str(mask_dir) if mask_dir is not None else None,
        "n_render_frames_indexed": len(render_index),
        "n_static_frames_indexed": len(static_index),
        "n_dynamic_frames_indexed": len(dynamic_index),
        "n_mask_frames_indexed": len(mask_index),
        "n_scored_frames": len(used_frames),
    }
    return maps, target_hw, metadata


def tile_rows_for_scene(
    scene: str,
    maps: Dict[int, Dict[str, np.ndarray]],
    target_hw: Tuple[int, int],
    window_length: int,
    temporal_stride: int,
    tile_size: int,
    tile_stride: int,
) -> List[Dict[str, object]]:
    frame_ids = sorted(maps)
    if len(frame_ids) < window_length:
        return []
    height, width = target_hw
    rows: List[Dict[str, object]] = []
    min_frame, max_frame = min(frame_ids), max(frame_ids)
    frame_set = set(frame_ids)
    window_length = max(1, int(window_length))
    temporal_stride = max(1, int(temporal_stride))
    tile_size = max(1, int(tile_size))
    tile_stride = max(1, int(tile_stride))
    tile_width = min(width, tile_size)
    tile_height = min(height, tile_size)
    max_x = max(0, width - tile_width)
    max_y = max(0, height - tile_height)
    x_starts = list(range(0, max_x + 1, max(1, tile_stride)))
    y_starts = list(range(0, max_y + 1, max(1, tile_stride)))
    if x_starts[-1] != max_x:
        x_starts.append(max_x)
    if y_starts[-1] != max_y:
        y_starts.append(max_y)

    for frame_start in range(min_frame, max_frame - window_length + 2, max(1, temporal_stride)):
        frames = list(range(frame_start, frame_start + window_length))
        if any(frame not in frame_set for frame in frames):
            continue
        frame_end = frames[-1]
        for y0 in y_starts:
            y1 = min(height, y0 + tile_height)
            for x0 in x_starts:
                x1 = min(width, x0 + tile_width)
                score_values = []
                term_values = {
                    "dynamic_motion": [],
                    "static_delta_motion": [],
                    "motion_mask": [],
                    "mask_boundary": [],
                    "flicker_motion": [],
                }
                for frame_idx in frames:
                    frame_maps = maps[frame_idx]
                    score_values.append(float(frame_maps["score"][y0:y1, x0:x1].mean()))
                    for key in term_values:
                        term_values[key].append(float(frame_maps[key][y0:y1, x0:x1].mean()))
                score_mean = float(np.mean(score_values))
                score_peak = float(np.max(score_values))
                score = 0.7 * score_mean + 0.3 * score_peak
                rows.append(
                    {
                        "scene": scene,
                        "frame_start": int(frame_start),
                        "frame_end": int(frame_end),
                        "crop_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                        "score": float(score),
                        "score_mean": score_mean,
                        "score_peak": score_peak,
                        "dynamic_motion_mean": float(np.mean(term_values["dynamic_motion"])),
                        "static_delta_motion_mean": float(np.mean(term_values["static_delta_motion"])),
                        "motion_mask_mean": float(np.mean(term_values["motion_mask"])),
                        "mask_boundary_mean": float(np.mean(term_values["mask_boundary"])),
                        "flicker_motion_mean": float(np.mean(term_values["flicker_motion"])),
                    }
                )
    return rows


def discover_nonoracle_event_candidates(
    manifest_path: Path,
    out_dir: Path,
    route0_system: str = "route0",
    window_length: int = 16,
    temporal_stride: int = 4,
    tile_size: int = 160,
    tile_stride: int = 80,
    top_k_per_scene: int = 8,
    crop_iou_threshold: float = 0.5,
    temporal_iou_threshold: float = 0.5,
) -> Dict[str, object]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene_sources = manifest.get("scene_sources")
    if not isinstance(scene_sources, dict):
        raise ValueError(f"Manifest lacks scene_sources needed for non-oracle discovery: {manifest_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    scene_reports: List[Dict[str, object]] = []
    selected_windows: List[Dict[str, object]] = []
    for scene, raw_scene_source in scene_sources.items():
        if not isinstance(raw_scene_source, dict):
            continue
        frame_range = raw_scene_source.get("frame_range", [0, 299])
        frame_start, frame_end = int(frame_range[0]), int(frame_range[1])
        frame_ids = list(range(frame_start, frame_end + 1))
        maps, target_hw, scene_metadata = score_maps_for_scene(raw_scene_source, frame_ids, route0_system)
        rows = tile_rows_for_scene(
            scene=str(scene),
            maps=maps,
            target_hw=target_hw,
            window_length=window_length,
            temporal_stride=temporal_stride,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        selected = nms_candidate_rows(rows, top_k_per_scene, crop_iou_threshold, temporal_iou_threshold)
        route0_spec = route0_spec_from_scene_source(raw_scene_source)
        for rank, row in enumerate(selected, start=1):
            candidate_id = (
                f"{scene}_nonoracle_{rank:02d}_"
                f"{int(row['frame_start']):03d}_{int(row['frame_end']):03d}_"
                f"{'_'.join(str(v) for v in row['crop_xyxy'])}"
            )
            window = {
                "window_id": candidate_id,
                "scene": str(scene),
                "frame_start": int(row["frame_start"]),
                "frame_end": int(row["frame_end"]),
                "crop_xyxy": [int(v) for v in row["crop_xyxy"]],
                "occluder": "NONORACLE_CANDIDATE",
                "notes": (
                    "Automatically discovered from motion-supported route0 dynamic output, "
                    "route0-vs-static deltas, motion masks, motion-mask boundaries, and route0 render flicker; "
                    "no GT residual and no frozen event-crop labels used."
                ),
                "candidate_score": float(row["score"]),
                "candidate_terms": {
                    "score_mean": float(row["score_mean"]),
                    "score_peak": float(row["score_peak"]),
                    "dynamic_motion_mean": float(row["dynamic_motion_mean"]),
                    "static_delta_motion_mean": float(row["static_delta_motion_mean"]),
                    "motion_mask_mean": float(row["motion_mask_mean"]),
                    "mask_boundary_mean": float(row["mask_boundary_mean"]),
                    "flicker_motion_mean": float(row["flicker_motion_mean"]),
                },
                "systems": {route0_system: route0_spec},
            }
            selected_windows.append(window)
            all_rows.append({"candidate_id": candidate_id, **row})
        scene_reports.append(
            {
                "scene": str(scene),
                "n_raw_candidates": len(rows),
                "n_selected_candidates": len(selected),
                "image_size_hw": [int(target_hw[0]), int(target_hw[1])],
                **scene_metadata,
            }
        )

    candidate_manifest = {
        "description": "Non-oracle event-support candidates discovered without frozen event-crop labels.",
        "frames_are_inclusive": True,
        "generated_by": "nonoracle-candidates",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_manifest": str(manifest_path),
        "route0_system": route0_system,
        "uses_gt_residual": False,
        "uses_frozen_window_labels": False,
        "selection_parameters": {
            "window_length": int(window_length),
            "temporal_stride": int(temporal_stride),
            "tile_size": int(tile_size),
            "tile_stride": int(tile_stride),
            "top_k_per_scene": int(top_k_per_scene),
            "crop_iou_threshold": float(crop_iou_threshold),
            "temporal_iou_threshold": float(temporal_iou_threshold),
            "score_weights": {
                "motion_supported_dynamic_render": 0.35,
                "motion_supported_static_render_delta": 0.25,
                "motion_mask_interior": 0.25,
                "motion_mask_boundary": 0.10,
                "motion_supported_route0_render_flicker": 0.05,
            },
        },
        "windows": selected_windows,
    }
    manifest_out = out_dir / "nonoracle_candidate_manifest.json"
    metadata = {
        "candidate_manifest": str(manifest_out.resolve()),
        "source_manifest": str(manifest_path),
        "scene_reports": scene_reports,
        "n_candidates": len(selected_windows),
        "limitations": [
            "This is candidate discovery only; it does not prove a Gaussian method improves the frozen windows.",
            "Scores do not use GT residual or frozen crop labels, but they can still select easy motion rather than the target event-crop failures.",
            "The candidate manifest includes gt_dir only so downstream evaluators can score renders; gt_dir is not used for candidate scoring.",
        ],
    }
    write_json(manifest_out, candidate_manifest)
    write_json(out_dir / "nonoracle_candidate_metadata.json", metadata)
    write_csv(out_dir / "nonoracle_candidate_components.csv", all_rows)
    validation = validate_real_manifest(manifest_out, require_systems=[route0_system])
    write_json(out_dir / "nonoracle_candidate_validation.json", validation)
    write_nonoracle_candidate_report(out_dir / "nonoracle_candidate_report.md", candidate_manifest, metadata, validation)
    return {
        "manifest": candidate_manifest,
        "manifest_path": str(manifest_out),
        "metadata": metadata,
        "metadata_path": str(out_dir / "nonoracle_candidate_metadata.json"),
        "validation": validation,
    }


def write_nonoracle_candidate_report(
    path: Path,
    candidate_manifest: Dict[str, object],
    metadata: Dict[str, object],
    validation: Dict[str, object],
) -> None:
    lines = [
        "# Non-Oracle Event Candidate Discovery",
        "",
        f"Generated: {candidate_manifest.get('generated_at_utc')}",
        "",
        "## Scientific Guardrails",
        "",
        f"- Uses GT residual: `{candidate_manifest.get('uses_gt_residual')}`",
        f"- Uses frozen event-crop labels: `{candidate_manifest.get('uses_frozen_window_labels')}`",
        "- Candidate scores use motion-supported route0 dynamic output, route0-vs-static render deltas, motion masks, motion-mask boundaries, and route0 render flicker.",
        "- The generated candidate crops are method inputs; the frozen R009 crops remain evaluation-only.",
        "",
        "## Parameters",
        "",
    ]
    params = candidate_manifest.get("selection_parameters", {})
    if isinstance(params, dict):
        for key, value in params.items():
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Scene Summary", "", "| Scene | Raw candidates | Selected | Scored frames | Indexed masks |", "| --- | ---: | ---: | ---: | ---: |"])
    for report in metadata.get("scene_reports", []):
        lines.append(
            "| {scene} | {raw} | {selected} | {frames} | {masks} |".format(
                scene=report.get("scene"),
                raw=report.get("n_raw_candidates"),
                selected=report.get("n_selected_candidates"),
                frames=report.get("n_scored_frames"),
                masks=report.get("n_mask_frames_indexed"),
            )
        )
    lines.extend(["", "## Selected Candidates", "", "| Candidate | Scene | Frames | Crop | Score |", "| --- | --- | --- | --- | ---: |"])
    for window in candidate_manifest.get("windows", []):
        lines.append(
            "| `{wid}` | `{scene}` | {start}-{end} | `{crop}` | {score:.6f} |".format(
                wid=window.get("window_id"),
                scene=window.get("scene"),
                start=window.get("frame_start"),
                end=window.get("frame_end"),
                crop=window.get("crop_xyxy"),
                score=float(window.get("candidate_score", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- validation_ok: `{validation.get('ok')}`",
            f"- validation_errors: `{len(validation.get('errors', []))}`",
            f"- validation_warnings: `{len(validation.get('warnings', []))}`",
            "",
            "## Outputs",
            "",
            "- `nonoracle_candidate_manifest.json`",
            "- `nonoracle_candidate_metadata.json`",
            "- `nonoracle_candidate_components.csv`",
            "- `nonoracle_candidate_validation.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_actual_hide_reveal_real_windows(
    manifest_path: Path,
    out_dir: Path,
    residual_manifest_path: Optional[Path] = None,
    matched_manifest_path: Optional[Path] = None,
    route0_system: str = "route0",
    actual_system: str = "actual_hide_reveal",
    opacity_attenuation: float = 0.95,
    dynamic_probability_min: Optional[float] = 0.55,
    event_beta: float = 1.0,
    overwrite: bool = False,
    run_eval: bool = True,
    eval_out_dir: Optional[Path] = None,
    compute_lpips: bool = False,
) -> Dict[str, object]:
    import torch
    from argparse import Namespace

    from gaussian_renderer import render
    from scene import Scene
    from scene.gaussian_model import GaussianModel
    from utils.render_utils import save_img_u8

    manifest_path = manifest_path.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("windows"):
        raise ValueError(f"Manifest has no windows: {manifest_path}")

    residual_manifest = (
        json.loads(residual_manifest_path.read_text(encoding="utf-8")) if residual_manifest_path else {"windows": []}
    )
    matched_manifest = (
        json.loads(matched_manifest_path.read_text(encoding="utf-8")) if matched_manifest_path else {"windows": []}
    )

    groups: Dict[str, Dict[str, object]] = {}
    augmented_windows = []
    metadata_windows = []
    for window in manifest["windows"]:
        window_id = str(window["window_id"])
        scene = str(window.get("scene", "scene"))
        systems = window.get("systems", {})
        if route0_system not in systems:
            raise ValueError(f"Window {window_id} is missing required system {route0_system}")
        source_spec = systems[route0_system]
        source_render_dir = resolve_path(manifest_path.parent, source_spec["render_dir"])
        source_gt_dir = resolve_path(manifest_path.parent, source_spec["gt_dir"])
        checkpoint, iteration = infer_checkpoint_from_render_dir(source_render_dir)
        run_dir = infer_run_dir_from_eval_render_dir(source_render_dir)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found for {window_id}: {checkpoint}")
        group_id = source_group_id(scene, str(source_render_dir), str(source_gt_dir))
        group_root = out_dir / "actual_renders" / actual_system / group_id
        render_dir = group_root / "renders"
        static_dir = group_root / "static"
        dynamic_dir = group_root / "dynamic"
        groups.setdefault(
            group_id,
            {
                "group_id": group_id,
                "scene": scene,
                "source_spec": absolute_system_spec(source_spec, manifest_path.parent),
                "source_render_dir": str(source_render_dir),
                "source_gt_dir": str(source_gt_dir),
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "checkpoint_iteration": iteration,
                "render_dir": render_dir,
                "static_dir": static_dir,
                "dynamic_dir": dynamic_dir,
                "windows": [],
                "frame_ids": set(),
            },
        )
        groups[group_id]["windows"].append(window)
        groups[group_id]["frame_ids"].update(range(int(window["frame_start"]), int(window["frame_end"]) + 1))

        augmented = dict(window)
        augmented_systems = {route0_system: source_spec}
        augmented_systems.update(
            merge_baseline_systems(window, residual_manifest.get("windows", []), ["residual_uncertainty"])
        )
        augmented_systems.update(
            merge_baseline_systems(window, matched_manifest.get("windows", []), ["matched_lifespan"])
        )
        augmented_systems[actual_system] = {
            "render_dir": str(render_dir.resolve()),
            "gt_dir": str(source_gt_dir.resolve()),
            "static_dir": str(static_dir.resolve()),
            "dynamic_dir": str(dynamic_dir.resolve()),
        }
        augmented["systems"] = augmented_systems
        augmented_windows.append(augmented)
        metadata_windows.append(
            {
                "window_id": window_id,
                "scene": scene,
                "frame_start": int(window["frame_start"]),
                "frame_end": int(window["frame_end"]),
                "crop_xyxy": [int(value) for value in window["crop_xyxy"]],
                "actual_render_dir": str(render_dir.resolve()),
                "checkpoint": str(checkpoint),
                "checkpoint_iteration": int(iteration),
            }
        )

    group_reports = []
    for group_id, group in groups.items():
        render_dir = Path(group["render_dir"])
        static_dir = Path(group["static_dir"])
        dynamic_dir = Path(group["dynamic_dir"])
        for directory in (render_dir, static_dir, dynamic_dir):
            ensure_render_dir_writable(directory, overwrite)

        args = load_run_cfg_args(Path(group["run_dir"]), str(group["scene"]))
        model_params, loaded_iteration = torch.load(str(group["checkpoint"]))
        checkpoint_gaussian_dim = infer_gaussian_dim_from_checkpoint_args(model_params)
        configured_gaussian_dim = getattr(args, "gaussian_dim", None)
        gaussian_dim = int(checkpoint_gaussian_dim or configured_gaussian_dim or 3)
        if configured_gaussian_dim is not None and checkpoint_gaussian_dim is not None:
            if int(configured_gaussian_dim) != int(checkpoint_gaussian_dim):
                raise ValueError(
                    f"Checkpoint {group['checkpoint']} is gaussian_dim={checkpoint_gaussian_dim}, "
                    f"but run config says gaussian_dim={configured_gaussian_dim}"
                )
        pipe = Namespace(
            convert_SHs_python=getattr(args, "convert_SHs_python", False),
            compute_cov3D_python=getattr(args, "compute_cov3D_python", False),
            debug=getattr(args, "debug", False),
            env_map_res=getattr(args, "env_map_res", 0),
            eval_shfs_4d=getattr(args, "eval_shfs_4d", False),
            opa_threshold=getattr(args, "opa_threshold", 0.05),
        )
        bg_color = [1, 1, 1] if getattr(args, "white_background", False) else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        gaussians = GaussianModel(
            getattr(args, "sh_degree", 3),
            gaussian_dim=gaussian_dim,
            time_duration=getattr(args, "time_duration", [-0.5, 0.5]),
            rot_4d=getattr(args, "rot_4d", False),
            force_sh_3d=getattr(args, "force_sh_3d", False),
            sh_degree_t=2 if getattr(pipe, "eval_shfs_4d", False) else 0,
        )
        scene_obj = Scene(
            args,
            gaussians,
            shuffle=False,
            num_pts=getattr(args, "num_pts", 100000),
            num_pts_ratio=getattr(args, "num_pts_ratio", 1.0),
            time_duration=getattr(args, "time_duration", [-0.5, 0.5]),
        )
        gaussians.restore(model_params, None)
        gaussians.hide_reveal_runtime_events = [
            {
                "window_id": str(window["window_id"]),
                "frame_start": int(window["frame_start"]),
                "frame_end": int(window["frame_end"]),
                "crop_xyxy": [int(value) for value in window["crop_xyxy"]],
                "opacity_attenuation": float(opacity_attenuation),
                "dynamic_probability_min": dynamic_probability_min,
                "event_beta": float(event_beta),
            }
            for window in group["windows"]
        ]

        test_cameras = scene_obj.getTestCameras()
        frame_stats = []
        for frame_idx in sorted(group["frame_ids"]):
            if frame_idx < 0 or frame_idx >= len(test_cameras):
                raise IndexError(f"Frame {frame_idx} is outside test camera range 0..{len(test_cameras)-1}")
            gt_image, camera = test_cameras[frame_idx]
            camera = camera.cuda()
            setattr(camera, "hide_reveal_frame_idx", int(frame_idx))
            with torch.no_grad():
                render_pkg = render(camera, gaussians, pipe, background)
            rgb = torch.clamp(render_pkg["render"], 0, 1).permute(1, 2, 0).detach().cpu().numpy()
            static = torch.clamp(render_pkg["render_3d"], 0, 1).permute(1, 2, 0).detach().cpu().numpy()
            dynamic = torch.clamp(render_pkg["render_4d"], 0, 1).permute(1, 2, 0).detach().cpu().numpy()
            save_img_u8(rgb, render_dir / f"{frame_idx:05d}.png")
            save_img_u8(static, static_dir / f"static_{frame_idx:05d}.png")
            save_img_u8(dynamic, dynamic_dir / f"dynamic_{frame_idx:05d}.png")
            frame_stats.extend(render_pkg.get("hide_reveal_gate_stats", []))
            del render_pkg, rgb, static, dynamic, gt_image, camera
            torch.cuda.empty_cache()

        group_reports.append(
            {
                "group_id": group_id,
                "scene": group["scene"],
                "checkpoint": group["checkpoint"],
                "checkpoint_iteration": int(loaded_iteration),
                "run_config_path": getattr(args, "hide_reveal_config_path", None),
                "gaussian_dim": int(gaussian_dim),
                "render_dir": str(render_dir.resolve()),
                "static_dir": str(static_dir.resolve()),
                "dynamic_dir": str(dynamic_dir.resolve()),
                "n_windows": len(group["windows"]),
                "n_frames_written": len(group["frame_ids"]),
                "gate_stats": frame_stats,
            }
        )

    augmented_manifest = dict(manifest)
    augmented_manifest["generated_by"] = "actual-real-renders"
    augmented_manifest["actual_hide_reveal_outputs"] = {
        "system": actual_system,
        "is_checkpoint_backed_inference": True,
        "uses_gaussian_renderer": True,
        "uses_gt_pixels_in_render": False,
        "newly_trained_checkpoint": False,
        "description": (
            "R017 checkpoint-backed runtime opacity-gate render. Dynamic Gaussians projected inside "
            "predeclared event crops are attenuated during the frozen event windows; no GT pixels are "
            "composited into the output."
        ),
    }
    augmented_manifest["windows"] = augmented_windows
    manifest_out = out_dir / "actual_real_windows_manifest.json"
    write_json(manifest_out, augmented_manifest)

    metadata = {
        "generated_by": "actual-real-renders",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_manifest": str(manifest_path),
        "output_manifest": str(manifest_out.resolve()),
        "route0_system": route0_system,
        "actual_system": actual_system,
        "is_checkpoint_backed_inference": True,
        "uses_gaussian_renderer": True,
        "uses_gt_pixels_in_render": False,
        "newly_trained_checkpoint": False,
        "parameters": {
            "opacity_attenuation": float(opacity_attenuation),
            "dynamic_probability_min": None if dynamic_probability_min is None else float(dynamic_probability_min),
            "event_beta": float(event_beta),
        },
        "limitations": [
            "Runtime inference gate on existing route0 checkpoints; no new Gaussian state was trained.",
            "Candidate support is the predeclared R009 crop projected onto currently visible dynamic Gaussians.",
            "This tests whether a checkpoint-backed event opacity gate helps real windows without using GT crop composites.",
        ],
        "windows": metadata_windows,
        "groups": group_reports,
    }
    metadata_out = out_dir / "actual_render_metadata.json"
    write_json(metadata_out, metadata)

    required_systems = [route0_system, actual_system]
    if residual_manifest_path is not None:
        required_systems.append("residual_uncertainty")
    if matched_manifest_path is not None:
        required_systems.append("matched_lifespan")
    validation = validate_real_manifest(manifest_out, require_systems=required_systems)
    write_json(out_dir / "actual_real_windows_validation.json", validation)
    if not validation["ok"]:
        return {
            "manifest": augmented_manifest,
            "manifest_path": str(manifest_out),
            "metadata": metadata,
            "metadata_path": str(metadata_out),
            "validation": validation,
        }

    eval_payload = None
    if run_eval:
        eval_payload = evaluate_real_manifest(
            manifest_out,
            eval_out_dir or out_dir / "eval",
            compute_lpips=compute_lpips,
        )
    return {
        "manifest": augmented_manifest,
        "manifest_path": str(manifest_out),
        "metadata": metadata,
        "metadata_path": str(metadata_out),
        "validation": validation,
        "eval": eval_payload,
    }


def write_derived_group_renders(
    group_id: str,
    source_spec: Dict[str, str],
    windows: Sequence[Dict[str, object]],
    hide_render_dir: Path,
    lifespan_render_dir: Path,
    hide_reveal_strength: float,
    matched_lifespan_strength: float,
    event_beta: float,
    feather_px: int,
    overwrite: bool,
) -> Dict[str, object]:
    render_dir = Path(source_spec["render_dir"])
    gt_dir = Path(source_spec["gt_dir"])
    render_index = index_image_frames(render_dir)
    gt_index = index_image_frames(gt_dir)
    for output_dir in (hide_render_dir, lifespan_render_dir):
        ensure_render_dir_writable(output_dir, overwrite)

    frame_ids = sorted(
        {
            frame_idx
            for window in windows
            for frame_idx in range(int(window["frame_start"]), int(window["frame_end"]) + 1)
        }
    )
    image_cache: Dict[Path, np.ndarray] = {}
    lifespan_targets = {
        str(window["window_id"]): lifespan_reference_crop(render_index, window, image_cache)
        for window in windows
    }
    frames_written = 0
    for frame_idx in frame_ids:
        source_path = render_index.get(frame_idx)
        gt_path = gt_index.get(frame_idx)
        if source_path is None:
            raise FileNotFoundError(f"Missing route0 frame {frame_idx} in {render_dir}")
        if gt_path is None:
            raise FileNotFoundError(f"Missing GT frame {frame_idx} in {gt_dir}")
        route0_image = load_full_image(source_path)
        gt_image = load_full_image(gt_path)
        hide_image = route0_image.copy()
        lifespan_image = route0_image.copy()
        for window in windows:
            frame_start = int(window["frame_start"])
            frame_end = int(window["frame_end"])
            if frame_idx < frame_start or frame_idx > frame_end:
                continue
            crop = tuple(int(value) for value in window["crop_xyxy"])
            temporal = event_window_mix(frame_idx, frame_start, frame_end, event_beta)
            hide_image = composite_crop(
                hide_image,
                gt_image,
                crop,
                mix=hide_reveal_strength * temporal,
                feather_px=feather_px,
            )
            reference_crop = lifespan_targets[str(window["window_id"])]
            lifespan_image = composite_crop_with_crop_target(
                lifespan_image,
                reference_crop,
                crop,
                mix=matched_lifespan_strength * temporal,
                feather_px=feather_px,
            )
        save_full_image(hide_render_dir / source_path.name, hide_image)
        save_full_image(lifespan_render_dir / source_path.name, lifespan_image)
        frames_written += 1

    return {
        "group_id": group_id,
        "source_render_dir": str(render_dir),
        "source_gt_dir": str(gt_dir),
        "hide_reveal_render_dir": str(hide_render_dir.resolve()),
        "matched_lifespan_render_dir": str(lifespan_render_dir.resolve()),
        "n_windows": len(windows),
        "n_frames_written": frames_written,
    }


def ensure_render_dir_writable(render_dir: Path, overwrite: bool) -> None:
    if render_dir.exists() and any(render_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output render dir is not empty; use --overwrite to replace: {render_dir}")
    render_dir.mkdir(parents=True, exist_ok=True)


def load_full_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def save_full_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def event_window_mix(frame_idx: int, frame_start: int, frame_end: int, beta: float) -> float:
    beta = max(float(beta), 1e-6)
    value = 1.0 - float(rectangular_gate(np.asarray([float(frame_idx)]), frame_start, frame_end, beta=beta)[0])
    return float(np.clip(value, 0.0, 1.0))


def composite_crop(
    image: np.ndarray,
    target_image: np.ndarray,
    crop_xyxy: Tuple[int, int, int, int],
    mix: float,
    feather_px: int,
) -> np.ndarray:
    x0, y0, x1, y1 = crop_xyxy
    target_crop = target_image[y0:y1, x0:x1, :]
    return composite_crop_with_crop_target(image, target_crop, crop_xyxy, mix, feather_px)


def composite_crop_with_crop_target(
    image: np.ndarray,
    target_crop: np.ndarray,
    crop_xyxy: Tuple[int, int, int, int],
    mix: float,
    feather_px: int,
) -> np.ndarray:
    mix = float(np.clip(mix, 0.0, 1.0))
    if mix <= 0.0:
        return image
    x0, y0, x1, y1 = crop_xyxy
    source_crop = image[y0:y1, x0:x1, :]
    if source_crop.shape != target_crop.shape:
        raise ValueError(f"Target crop shape {target_crop.shape} does not match source crop shape {source_crop.shape}")
    mask = feather_mask(source_crop.shape[0], source_crop.shape[1], feather_px)[:, :, None] * mix
    image[y0:y1, x0:x1, :] = source_crop * (1.0 - mask) + target_crop * mask
    return image


def feather_mask(height: int, width: int, feather_px: int) -> np.ndarray:
    if feather_px <= 0:
        return np.ones((height, width), dtype=np.float32)
    yy, xx = np.mgrid[0:height, 0:width]
    edge_distance = np.minimum.reduce([xx + 1, yy + 1, width - xx, height - yy]).astype(np.float32)
    return np.clip(edge_distance / float(max(feather_px, 1)), 0.0, 1.0)


def lifespan_reference_crop(
    render_index: Dict[int, Path],
    window: Dict[str, object],
    image_cache: Dict[Path, np.ndarray],
) -> np.ndarray:
    frame_start = int(window["frame_start"])
    frame_end = int(window["frame_end"])
    crop = tuple(int(value) for value in window["crop_xyxy"])
    before = max((frame for frame in render_index if frame < frame_start), default=None)
    after = min((frame for frame in render_index if frame > frame_end), default=None)
    crops = []
    for frame_idx in (before, after):
        if frame_idx is None:
            continue
        path = render_index[frame_idx]
        if path not in image_cache:
            image_cache[path] = load_full_image(path)
        x0, y0, x1, y1 = crop
        crops.append(image_cache[path][y0:y1, x0:x1, :])
    if not crops:
        first_path = render_index[frame_start]
        if first_path not in image_cache:
            image_cache[first_path] = load_full_image(first_path)
        x0, y0, x1, y1 = crop
        return image_cache[first_path][y0:y1, x0:x1, :].copy()
    return np.mean(np.stack(crops, axis=0), axis=0).astype(np.float32)


def summarize_missing_frames(frames: Sequence[int], limit: int = 8) -> str:
    shown = ", ".join(str(frame) for frame in frames[:limit])
    if len(frames) > limit:
        shown += f", ... ({len(frames)} total)"
    return shown


def evaluate_real_manifest(manifest_path: Path, out_dir: Path, compute_lpips: bool = False) -> Dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for window in manifest.get("windows", []):
        rows.extend(evaluate_real_window(window, manifest_path.parent, compute_lpips=compute_lpips))
    write_csv(out_dir / "real_event_window_metrics.csv", rows)
    summary = summarize_real_rows(rows)
    payload = {
        "manifest": str(manifest_path),
        "compute_lpips_requested": bool(compute_lpips),
        "lpips_unavailable_reason": LPIPS_UNAVAILABLE_REASON,
        "summary": summary,
        "rows": rows,
    }
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
    global LPIPS_UNAVAILABLE_REASON
    if LPIPS_UNAVAILABLE_REASON is not None:
        return None
    try:
        import torch
        from lpipsPyTorch import lpips
    except Exception as exc:
        LPIPS_UNAVAILABLE_REASON = repr(exc)
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    values = []
    try:
        with torch.no_grad():
            for render, gt in zip(render_frames, gt_frames):
                x = torch.from_numpy(render.transpose(2, 0, 1)).unsqueeze(0).to(device)
                y = torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0).to(device)
                values.append(float(lpips(x, y).mean().detach().cpu()))
    except Exception as exc:
        LPIPS_UNAVAILABLE_REASON = repr(exc)
        return None
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
