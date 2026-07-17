"""Deterministic image sampling operators for CSVL-ISR Slice A."""

from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """Convert IEC 61966-2-1 sRGB values in [0, 1] to linear RGB."""

    values = np.asarray(image, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError("sRGB input must have a final RGB dimension")
    if not np.all(np.isfinite(values)):
        raise ValueError("sRGB input must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("sRGB input must be in [0, 1]")
    return np.where(
        values <= 0.04045,
        values / 12.92,
        np.power((values + 0.055) / 1.055, 2.4),
    )


def linear_gray(linear_rgb: np.ndarray) -> np.ndarray:
    """Return the contract-pinned linear-RGB luminance."""

    values = np.asarray(linear_rgb, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError("linear RGB input must have a final RGB dimension")
    if not np.all(np.isfinite(values)):
        raise ValueError("linear RGB input must be finite")
    return np.tensordot(
        values,
        np.array([0.2126, 0.7152, 0.0722], dtype=np.float64),
        axes=([-1], [0]),
    )


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Compute 3x3 Sobel magnitude with OpenCV BORDER_REFLECT_101."""

    values = np.asarray(gray, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("gray image must be a finite 2D array")
    dx = cv2.Sobel(
        values,
        cv2.CV_64F,
        1,
        0,
        ksize=3,
        borderType=cv2.BORDER_REFLECT_101,
    )
    dy = cv2.Sobel(
        values,
        cv2.CV_64F,
        0,
        1,
        ksize=3,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return np.hypot(dx, dy)


def regular_samples(height: int, width: int, stride: int = 8) -> list[tuple[int, int]]:
    """Generate the grid in scientific order, beginning at native (0, 0)."""

    if height <= 0 or width <= 0 or stride <= 0:
        raise ValueError("height, width, and stride must be positive")
    return [
        (y, x)
        for y in range(0, height, stride)
        for x in range(0, width, stride)
    ]


def salient_samples(
    magnitude: np.ndarray,
    grid: Iterable[tuple[int, int]],
    *,
    minimum_separation: float = 8.0,
    fraction_of_grid: float = 0.25,
) -> list[tuple[int, int]]:
    """Choose stable high-gradient additions and remove exact grid duplicates.

    The 8-pixel greedy separation applies among salient additions. Applying it
    against the stride-8 base grid would suppress nearly the entire admitted
    salient population, whereas the contract separately states exact
    de-duplication against the grid.
    """

    values = np.asarray(magnitude, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("magnitude must be a finite 2D array")
    if (
        not math.isfinite(minimum_separation)
        or minimum_separation < 0
        or not math.isfinite(fraction_of_grid)
        or not 0.0 <= fraction_of_grid <= 1.0
    ):
        raise ValueError("invalid salient sampling parameters")
    grid_list = [(int(y), int(x)) for y, x in grid]
    if len(grid_list) != len(set(grid_list)):
        raise ValueError("regular grid contains duplicates")
    if any(
        y < 0 or x < 0 or y >= values.shape[0] or x >= values.shape[1]
        for y, x in grid_list
    ):
        raise ValueError("regular grid lies outside the magnitude image")
    grid_set = set(grid_list)
    cap = math.floor(fraction_of_grid * len(grid_list))
    candidates = [
        (-float(values[y, x]), y, x)
        for y in range(values.shape[0])
        for x in range(values.shape[1])
        if (y, x) not in grid_set
    ]
    candidates.sort()
    selected: list[tuple[int, int]] = []
    radius2 = float(minimum_separation) ** 2
    for _, y, x in candidates:
        if len(selected) >= cap:
            break
        if all((y - oy) ** 2 + (x - ox) ** 2 >= radius2 for oy, ox in selected):
            selected.append((y, x))
    return selected


def ordered_samples(
    magnitude: np.ndarray,
    *,
    stride: int = 8,
    minimum_separation: float = 8.0,
    fraction_of_grid: float = 0.25,
) -> list[tuple[int, int]]:
    """Return base grid followed by exact-de-duplicated salient additions."""

    grid = regular_samples(magnitude.shape[0], magnitude.shape[1], stride)
    return grid + salient_samples(
        magnitude,
        grid,
        minimum_separation=minimum_separation,
        fraction_of_grid=fraction_of_grid,
    )


def extract_patch(image: np.ndarray, y: int, x: int, radius: int = 2) -> np.ndarray:
    """Extract a reflected square patch centered on an integer pixel."""

    values = np.asarray(image)
    if values.ndim not in (2, 3) or radius < 0:
        raise ValueError("image must be 2D/3D and radius nonnegative")
    if not (0 <= y < values.shape[0] and 0 <= x < values.shape[1]):
        raise ValueError("patch center is outside the image")
    if radius > 0 and (values.shape[0] < 2 or values.shape[1] < 2):
        raise ValueError("reflect padding requires both image dimensions >=2")
    pad = ((radius, radius), (radius, radius)) + (
        ((0, 0),) if values.ndim == 3 else ()
    )
    padded = np.pad(values, pad, mode="reflect")
    return np.array(
        padded[y : y + 2 * radius + 1, x : x + 2 * radius + 1],
        copy=True,
    )


__all__ = [
    "extract_patch",
    "linear_gray",
    "ordered_samples",
    "regular_samples",
    "salient_samples",
    "sobel_magnitude",
    "srgb_to_linear",
]
