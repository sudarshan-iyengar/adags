"""Text-parameterized deterministic fixtures for Slice A correctness tests."""

from __future__ import annotations

from typing import Any

import numpy as np


def analytic_camera(
    camera_id: str,
    center_x: float = 0.0,
    *,
    width: int = 32,
    height: int = 24,
) -> dict[str, Any]:
    w2c = np.eye(4, dtype=np.float64)
    w2c[0, 3] = -float(center_x)
    return {
        "camera_id": camera_id,
        "K": np.array(
            [
                [20.0, 0.0, (width - 1) / 2],
                [0.0, 20.0, (height - 1) / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        "w2c": w2c,
        "width": width,
        "height": height,
        "center": np.array([center_x, 0.0, 0.0], dtype=np.float64),
    }


def constant_flow(height: int, width: int, dx: float, dy: float) -> np.ndarray:
    if height <= 0 or width <= 0:
        raise ValueError("flow fixture dimensions must be positive")
    flow = np.empty((height, width, 2), dtype=np.float64)
    flow[..., 0] = dx
    flow[..., 1] = dy
    return flow


def flow_manifest(
    *,
    camera: str = "cam01",
    source_frame: int | None = None,
    target_frame: int | None = None,
    height: int = 24,
    width: int = 32,
    direction: str = "forward_t_to_t_plus_1",
) -> dict[str, Any]:
    if direction == "forward_t_to_t_plus_1":
        source_frame = 0 if source_frame is None else int(source_frame)
        target_frame = source_frame + 1 if target_frame is None else int(target_frame)
    elif direction == "backward_t_plus_1_to_t":
        source_frame = 1 if source_frame is None else int(source_frame)
        target_frame = source_frame - 1 if target_frame is None else int(target_frame)
    else:
        source_frame = 0 if source_frame is None else int(source_frame)
        target_frame = 1 if target_frame is None else int(target_frame)
    return {
        "source_camera": camera,
        "target_camera": camera,
        "source_image": f"{camera}/{source_frame:04d}.png",
        "target_image": f"{camera}/{target_frame:04d}.png",
        "source_frame": source_frame,
        "target_frame": target_frame,
        "direction": direction,
        "dt": float(target_frame - source_frame),
        "height": height,
        "width": width,
        "units": "pixels",
        "pixel_centers": "integer",
        "sampling": "bilinear_align_corners_false",
        "validity_semantics": "true_means_sample_is_valid",
        "occlusion_semantics": "true_means_not_occluded",
        "generator_revision": "fixture-v1",
        "source_hashes": ["0" * 64, "1" * 64],
        "array_hash": "2" * 64,
    }


def two_plane_track_pixels(
    *, revealed: bool = False, sign_error: bool = False
) -> tuple[dict, dict]:
    """Return overlapping 4x4 layers and an independent rear witness."""

    front_z, rear_z = (2.0, 4.0) if not sign_error else (-2.0, -4.0)
    pixels = [(y, x) for y in range(8, 12) for x in range(10, 14)]
    front = {
        pixel: {
            "z": front_z,
            "sigma_z": 0.01,
            "risk": 0.1,
            "patch_id": "front-patch",
            "physical_ancestry": ("cam01", "cam02", "cam03"),
        }
        for pixel in pixels
    }
    rear_pixels = pixels if not revealed else [(y, x + 5) for y, x in pixels]
    rear = {
        pixel: {
            "z": rear_z,
            "sigma_z": 0.02,
            "risk": 0.2,
            "patch_id": "rear-patch",
            "physical_ancestry": ("cam04", "cam05", "cam06"),
        }
        for pixel in rear_pixels
    }
    witnesses = {"rear": {"cam07"}}
    return {"front": front, "rear": rear}, witnesses


def temporal_patch_candidates(*, split: bool = False) -> list[dict[str, Any]]:
    base = {
        "cost": 0.1,
        "risk": 0.1,
        "flow_manifests": ["forward-hash", "backward-hash"],
        "match_tuple": [0.1, 0.1, 0.1, 0.1],
        "r_scene": 1.0,
        "centroid_distance": 0.01,
        "rgb_l2": 0.10,
        "camera_node_match_counts": {"cam01": 3, "cam02": 3},
        "valid_flow_cameras": ["cam01", "cam02"],
    }
    output = [
        {
            **base,
            "source_patch_id": "p0",
            "destination_patch_id": "p1",
            "candidate_id": "e0",
        }
    ]
    if split:
        output.append(
            {
                **base,
                "cost": 0.2,
                "source_patch_id": "p0",
                "destination_patch_id": "p2",
                "candidate_id": "e1",
            }
        )
    return output


def planar_fused_points() -> list[dict[str, Any]]:
    """Three connected patch members with unequal color-confidence weights."""

    coordinates = [(0.001, 0.001, 2.0), (0.004, 0.001, 2.0), (0.001, 0.004, 2.0)]
    colors = [(0.10, 0.10, 0.10), (0.15, 0.15, 0.15), (0.20, 0.20, 0.20)]
    weights = [1.0, 10.0, 1.0]
    return [
        {
            "fused_id": f"f{index}",
            "scene": "fixture",
            "frame": 0,
            "scored_target": "cam00",
            "world_point_array": np.array(xyz, dtype=np.float64),
            "normal": np.array([0.0, 0.0, -1.0], dtype=np.float64),
            "linear_rgb": np.array(color, dtype=np.float64),
            "patch_color_weight": weight,
            "risk": 0.1 + 0.1 * index,
            "physical_ancestry": ["cam01", "cam02", f"cam{index + 3:02d}"],
        }
        for index, (xyz, color, weight) in enumerate(zip(coordinates, colors, weights, strict=True))
    ]


__all__ = [
    "analytic_camera",
    "constant_flow",
    "flow_manifest",
    "planar_fused_points",
    "temporal_patch_candidates",
    "two_plane_track_pixels",
]
