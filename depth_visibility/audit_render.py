"""Rendering for the Phase 0 blinded visual forensic audit (workstream A).

Consumes the per-case JSON + NPZ written by scripts/run_phase0_audit_extract.py
and produces one contact sheet PNG plus one synchronized GIF clip per case.
Sheets are self-contained: RGB before/during/after with the projected
primitive marked, evidence depth and uncertainty crops as consumed at
decision time, the primitive-vs-evidence depth timeline with margin band,
the inferred-state strip, cross-view witness counts, and the neutral
admission/rejection text. No provenance appears on any output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from PIL import Image  # noqa: E402

STATE_COLORS = {0: "#bdbdbd", 1: "#2e7d32", 2: "#c62828", 3: "#ef6c00"}
STATE_NAMES = {0: "not evaluable", 1: "near-surface", 2: "behind", 3: "in-front"}


def load_case(root: Path, blind_id: str):
    with open(root / "cases" / f"{blind_id}.json") as handle:
        case = json.load(handle)
    arrays = np.load(root / "cases" / f"{blind_id}.npz")
    return case, arrays


def rgb_crop(case, frame_pos):
    """RGB crop for one series position: real image or synthetic stack."""
    series = case["series"]
    if case["rgb_root"] is None:
        return None
    frame = series["frame"][frame_pos]
    path = Path(case["rgb_root"]) / f"{case['camera']}_{frame:04d}.png"
    if not path.exists():
        return None
    sx, sy = case["rgb_scale"]
    cx, cy = case["crop_center_depth_res"]
    half = case["crop_half"]
    with Image.open(path) as img:
        box = (int((cx - half) * sx), int((cy - half) * sy),
               int((cx + half) * sx), int((cy + half) * sy))
        box = (max(box[0], 0), max(box[1], 0),
               min(box[2], img.width), min(box[3], img.height))
        return np.asarray(img.crop(box).convert("RGB"))


def marker_position(case, frame_pos, image_shape):
    series = case["series"]
    px, py = series["px"][frame_pos], series["py"][frame_pos]
    if px < 0:
        return None
    cx, cy = case["crop_center_depth_res"]
    half = case["crop_half"]
    height, width = image_shape[:2]
    mx = (px - (cx - half)) / (2 * half) * width
    my = (py - (cy - half)) / (2 * half) * height
    if 0 <= mx < width and 0 <= my < height:
        return mx, my
    return None


def pick_key_positions(case):
    """pre / onset / mid / post positions within the series."""
    states = case["series"]["state"]
    n = len(states)
    behind = [i for i, s in enumerate(states) if s == 2]
    if behind:
        onset = behind[0]
        mid = behind[len(behind) // 2]
        pre = max(0, onset - 2)
        post = min(n - 1, behind[-1] + 2)
    else:
        pre, onset, mid, post = 0, n // 3, 2 * n // 3, n - 1
    return [pre, onset, mid, post]


def render_case(root: Path, blind_id: str, fps: int = 5) -> None:
    root = Path(root)
    case, arrays = load_case(root, blind_id)
    series = case["series"]
    frames = series["frame"]
    n = len(frames)
    if n == 0:
        return
    key_positions = pick_key_positions(case)
    d_stack = arrays["d"].astype(np.float32)
    finite = d_stack[np.isfinite(d_stack)]
    vmin, vmax = (np.percentile(finite, 2), np.percentile(finite, 98)) if finite.size else (0, 1)
    rgb_stack = arrays["rgb"] if "rgb" in arrays.files else None

    fig = plt.figure(figsize=(14, 10.5))
    grid = gridspec.GridSpec(4, 4, height_ratios=[3, 3, 2.4, 0.8], hspace=0.35)

    labels = ["pre", "onset", "mid", "post"]
    for col, pos in enumerate(key_positions):
        ax = fig.add_subplot(grid[0, col])
        img = rgb_stack[pos] if rgb_stack is not None else rgb_crop(case, pos)
        if img is not None:
            ax.imshow(img)
            marker = marker_position(case, pos, img.shape)
            if marker:
                ax.plot(marker[0], marker[1], "o", ms=14, mfc="none", mec="red", mew=2.5)
        else:
            ax.text(0.5, 0.5, "RGB unavailable", ha="center", va="center")
        ax.set_title(f"RGB {labels[col]} (f{frames[pos]})", fontsize=9)
        ax.axis("off")

        ax = fig.add_subplot(grid[1, col])
        ax.imshow(d_stack[pos], cmap="viridis", vmin=vmin, vmax=vmax)
        marker = marker_position(case, pos, d_stack[pos].shape)
        if marker:
            ax.plot(marker[0], marker[1], "o", ms=14, mfc="none", mec="red", mew=2.5)
        evidence_frame = series["evidence_frame"][pos]
        ax.set_title(f"evidence depth (f{evidence_frame})", fontsize=9)
        ax.axis("off")

    ax = fig.add_subplot(grid[2, :])
    frame_arr = np.array(frames, dtype=float)
    z = np.array(series["z"], dtype=float)
    d = np.array(series["d"], dtype=float)
    margin = np.array(series["margin"], dtype=float)
    ax.plot(frame_arr, z, "-", color="#1f77b4", lw=2, label="primitive depth")
    ax.plot(frame_arr, d, "-", color="#7b1fa2", lw=1.6, label="evidence depth at pixel")
    ax.fill_between(frame_arr, d - margin, d + margin, color="#7b1fa2", alpha=0.15,
                    label="near-surface margin")
    ax.set_xlabel("frame")
    ax.set_ylabel("depth")
    ax2 = ax.twinx()
    ax2.bar(frame_arr, series["witness"], width=0.8, color="#2e7d32", alpha=0.25)
    ax2.set_ylabel("witness cameras", color="#2e7d32")
    ax2.set_ylim(0, max(max(series["witness"], default=1), 1) * 2.5)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("primitive vs evidence depth; green bars: cameras seeing the "
                 "primitive near-surface elsewhere", fontsize=10)

    ax = fig.add_subplot(grid[3, :])
    for i, s in enumerate(series["state"]):
        ax.axvspan(frame_arr[i] - 0.5, frame_arr[i] + 0.5, color=STATE_COLORS[int(s)])
    ax.set_xlim(frame_arr[0] - 0.5, frame_arr[-1] + 0.5)
    ax.set_yticks([])
    ax.set_xlabel("inferred state per frame")
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATE_COLORS[k]) for k in sorted(STATE_NAMES)]
    ax.legend(handles, [STATE_NAMES[k] for k in sorted(STATE_NAMES)],
              loc="upper right", fontsize=7, ncol=4)

    gap_values = [g for g in series["gap_ratio"] if np.isfinite(g)]
    gap_note = (f"signed gap/margin: min {min(gap_values):.2f}, max {max(gap_values):.2f}"
                if gap_values else "gap: n/a")
    fig.suptitle(
        f"{blind_id} — camera {case['camera']} — frames {frames[0]}-{frames[-1]}\n"
        f"decision: {case['decision_text']} — {gap_note}",
        fontsize=11,
    )
    fig.savefig(root / "sheets" / f"{blind_id}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Synchronized clip: RGB | evidence depth per frame.
    clip_frames = []
    cmap = plt.get_cmap("viridis")
    for pos in range(n):
        img = rgb_stack[pos] if rgb_stack is not None else rgb_crop(case, pos)
        depth = d_stack[pos]
        norm = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0, 1)
        depth_rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
        target_h = 192
        def resize(a):
            pil = Image.fromarray(a)
            w = int(pil.width * target_h / pil.height)
            return np.asarray(pil.resize((w, target_h)))
        left = resize(img) if img is not None else np.zeros((target_h, target_h, 3), np.uint8)
        right = resize(depth_rgb)
        canvas = np.concatenate([left, np.full((target_h, 4, 3), 255, np.uint8), right], axis=1)
        pil = Image.fromarray(canvas)
        clip_frames.append(pil)
    if clip_frames:
        clip_frames[0].save(
            root / "clips" / f"{blind_id}.gif", save_all=True,
            append_images=clip_frames[1:], duration=int(1000 / fps), loop=0,
        )
