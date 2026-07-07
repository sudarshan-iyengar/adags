#!/usr/bin/env python3
"""Build crop-strip comparison panels from a hide/reveal real-window manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from PIL import Image, ImageDraw, ImageFont


DEFAULT_SYSTEMS = ("gt", "route0", "event_candidate_refine", "hide_reveal")


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def frame_samples(start: int, end: int, count: int) -> List[int]:
    if count <= 1:
        return [start]
    if end <= start:
        return [start]
    values = [round(start + (end - start) * idx / (count - 1)) for idx in range(count)]
    return sorted(dict.fromkeys(int(v) for v in values))


def image_path_for(window: Dict[str, object], system: str, frame: int) -> Path:
    systems = window["systems"]
    if system == "gt":
        route0 = systems["route0"]
        return Path(route0["gt_dir"]) / f"{frame:05d}.png"
    spec = systems[system]
    return Path(spec["render_dir"]) / f"{frame:05d}.png"


def crop_image(path: Path, crop_xyxy: Sequence[int]) -> Image.Image:
    with Image.open(path) as src:
        image = src.convert("RGB")
    return image.crop(tuple(int(v) for v in crop_xyxy))


def fit_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, xy: Sequence[int], text: str, font: ImageFont.ImageFont) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 4
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(255, 255, 255))
    draw.text((x, y), text, fill=(20, 20, 20), font=font)


def build_panel(
    window: Dict[str, object],
    systems: Sequence[str],
    frames: Sequence[int],
    tile_padding: int,
    label_width: int,
    header_height: int,
) -> Image.Image:
    crop_xyxy = window["crop_xyxy"]
    first = crop_image(image_path_for(window, systems[0], frames[0]), crop_xyxy)
    tile_w, tile_h = first.size
    first.close()

    panel_w = label_width + len(frames) * (tile_w + tile_padding) + tile_padding
    panel_h = header_height + len(systems) * (tile_h + tile_padding) + tile_padding
    panel = Image.new("RGB", (panel_w, panel_h), (245, 245, 245))
    draw = ImageDraw.Draw(panel)
    label_font = fit_font(16)
    small_font = fit_font(13)

    draw_label(draw, (tile_padding, 6), str(window["window_id"]), small_font)
    for col, frame in enumerate(frames):
        x = label_width + tile_padding + col * (tile_w + tile_padding)
        draw_label(draw, (x, 6), f"f{frame}", small_font)

    for row, system in enumerate(systems):
        y = header_height + tile_padding + row * (tile_h + tile_padding)
        draw_label(draw, (tile_padding, y + max(0, tile_h // 2 - 9)), system, label_font)
        for col, frame in enumerate(frames):
            x = label_width + tile_padding + col * (tile_w + tile_padding)
            image = crop_image(image_path_for(window, system, frame), crop_xyxy)
            panel.paste(image, (x, y))
            image.close()

    return panel


def available_systems(window: Dict[str, object], requested: Iterable[str]) -> List[str]:
    systems = window["systems"]
    out: List[str] = []
    for system in requested:
        if system == "gt" or system in systems:
            out.append(system)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--systems", nargs="+", default=list(DEFAULT_SYSTEMS))
    parser.add_argument("--frames-per-window", type=int, default=4)
    parser.add_argument("--quality", type=int, default=92)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    out_dir = args.out_dir
    strips_dir = out_dir / "crop_strips"
    strips_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for window in manifest["windows"]:
        frames = frame_samples(int(window["frame_start"]), int(window["frame_end"]), args.frames_per_window)
        systems = available_systems(window, args.systems)
        panel = build_panel(window, systems, frames, tile_padding=8, label_width=230, header_height=38)
        out_path = strips_dir / f"{window['window_id']}.jpg"
        panel.save(out_path, quality=args.quality)
        panel.close()
        entries.append(
            {
                "window_id": window["window_id"],
                "scene": window["scene"],
                "frame_start": window["frame_start"],
                "frame_end": window["frame_end"],
                "crop_xyxy": window["crop_xyxy"],
                "frames": frames,
                "systems": systems,
                "path": str(out_path),
            }
        )

    payload = {
        "source_manifest": str(args.manifest),
        "systems_requested": args.systems,
        "frames_per_window": args.frames_per_window,
        "windows": entries,
    }
    with (out_dir / "crop_strip_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    lines = [
        "# R025 Event-Candidate Refine Crop Strips",
        "",
        f"Source manifest: `{args.manifest}`",
        "",
        "Rows compare GT, route0, the checkpoint-backed `event_candidate_refine` method, and the derived oracle `hide_reveal` upper bound.",
        "",
    ]
    for entry in entries:
        lines.append(
            f"- `{entry['window_id']}` frames {entry['frames']}, crop `{entry['crop_xyxy']}`: "
            f"`{entry['path']}`"
        )
    (out_dir / "crop_strip_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(entries)} crop strips to {strips_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
