"""DiVa-360's OWN metric conventions, reproduced exactly.

Authority: `utils/benchmark.py` in the official DiVa-360 repository
(brown-ivl/DiVa360, branch `main`), read directly. That file is the
evaluator behind the published numbers, and it does three things this
repository's training loop does not:

    psnr = cv.PSNR(gt, pred)                                  # uint8, R=255
    ssim = structural_similarity(gt, pred, channel_axis=2)    # uint8, skimage DEFAULTS
    lpips_net = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
    image_pred = image_pred * 2 - 1;  image_gt = image_gt * 2 - 1

and averages per image, not pooled over pixels.

`research-wiki/operations/diva360-protocol-parity-audit.md` records that
this repository instead used float-domain PSNR, the 3DGS 11x11 Gaussian
SSIM, and AlexNet LPIPS. Two of those three are materially different
conventions, not merely different code, so a number produced under them
cannot be placed beside a published DiVa-360 row.

WHY SSIM IS REIMPLEMENTED RATHER THAN IMPORTED. `scikit-image` is NOT
installed in the Apollo image (probed directly: `skimage MISSING
ModuleNotFoundError`), while `cv2` and `torchmetrics` are. Rather than
let the score depend on whether an optional package happens to be
present, `ssim_skimage_default` reproduces skimage's DEFAULTS exactly and
is pinned against the real `skimage` by
`tests/test_diva360_official_metrics.py`, which SKIPS where skimage is
absent and runs where it is. The defaults being reproduced, all verified
in scikit-image's own source rather than assumed:

  * `gaussian_weights=False` -> a UNIFORM 7x7 window, not the Gaussian
    window 3DGS uses. `win_size=None` resolves to 7 ("backwards
    compatibility").
  * `data_range=None` on uint8 input resolves to 255 via `dtype_range`.
  * `K1=0.01`, `K2=0.03`, `use_sample_covariance=True`, so the
    covariance normalizer is `NP/(NP-1) = 49/48` with `NP = 7**2`.
  * `channel_axis=2` -> SSIM is computed per channel over the two
    spatial axes and the three scalars are averaged.
  * the SSIM map is cropped by `pad = (win_size-1)//2 = 3` on every side
    before the mean.

That crop is what makes the reimplementation exact without scipy: for a
7-tap filter the cropped region is precisely the region whose window
lies wholly inside the image, so skimage's boundary padding never
reaches any surviving pixel and a plain valid-region box mean is
identical to `uniform_filter` followed by the crop.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SSIM_C1",
    "SSIM_C2",
    "SSIM_WIN_SIZE",
    "psnr_official",
    "ssim_skimage_default",
    "to_uint8_hwc",
]

SSIM_WIN_SIZE = 7
_SSIM_K1 = 0.01
_SSIM_K2 = 0.03
_SSIM_DATA_RANGE = 255.0
SSIM_C1 = (_SSIM_K1 * _SSIM_DATA_RANGE) ** 2
SSIM_C2 = (_SSIM_K2 * _SSIM_DATA_RANGE) ** 2


def to_uint8_hwc(image) -> np.ndarray:
    """A float image in [0,1] as the uint8 HWC array a PNG would hold.

    Accepts a torch tensor or ndarray, CHW or HWC, with or without a
    leading batch of 1.

    The rounding rule is `round(x*255)` implemented as `x*255 + 0.5`
    truncated, which is what `torchvision.utils.save_image` does when a
    Gaussian-splatting pipeline writes its prediction PNGs. It is chosen
    deliberately over numpy's banker's rounding: DiVa-360's evaluator
    reads PNG FILES for both the ground truth and the prediction, so the
    faithful pipeline is to quantize exactly as the PNG write would and
    derive every metric from the quantized data.
    """
    array = image
    if hasattr(array, "detach"):
        array = array.detach().to("cpu").float().numpy()
    array = np.asarray(array)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"expected a single image, got batch {array.shape[0]}")
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"expected a 3D image, got shape {array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    array = array[:, :, :3]
    scaled = np.asarray(array, dtype=np.float64) * 255.0 + 0.5
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def psnr_official(gt_u8: np.ndarray, pred_u8: np.ndarray) -> float:
    """`cv.PSNR(gt, pred)` on uint8 with R=255, for ONE image.

    OpenCV's `PSNR(src1, src2, R=255)` is `10*log10(R^2/MSE)` with the
    MSE taken over every element of both arrays. `cv2` is used when it is
    importable so the published convention is executed rather than
    imitated; the fallback computes the identical expression in float64
    and exists only so this module is testable where cv2 is not
    installed. An exactly-equal pair gives infinity in OpenCV; that is
    preserved rather than being capped at some finite decibel value.
    """
    if gt_u8.shape != pred_u8.shape:
        raise ValueError(f"shape mismatch {gt_u8.shape} vs {pred_u8.shape}")
    try:
        import cv2
    except ImportError:
        diff = gt_u8.astype(np.float64) - pred_u8.astype(np.float64)
        mse = float(np.mean(diff * diff))
        if mse == 0.0:
            return float("inf")
        return float(10.0 * np.log10((255.0 ** 2) / mse))
    return float(cv2.PSNR(gt_u8, pred_u8))


def _box_mean_valid(plane: np.ndarray, size: int) -> np.ndarray:
    """Mean over every fully-interior `size x size` window.

    Shape (H, W) -> (H-size+1, W-size+1), which is exactly the region
    skimage keeps after its `pad = (size-1)//2` crop.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    rows = sliding_window_view(plane, size, axis=0).sum(axis=-1)
    both = sliding_window_view(rows, size, axis=1).sum(axis=-1)
    return both / float(size * size)


def _ssim_plane(x: np.ndarray, y: np.ndarray, size: int) -> float:
    n_points = float(size * size)
    cov_norm = n_points / (n_points - 1.0)  # use_sample_covariance=True

    ux = _box_mean_valid(x, size)
    uy = _box_mean_valid(y, size)
    uxx = _box_mean_valid(x * x, size)
    uyy = _box_mean_valid(y * y, size)
    uxy = _box_mean_valid(x * y, size)

    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    a1 = 2.0 * ux * uy + SSIM_C1
    a2 = 2.0 * vxy + SSIM_C2
    b1 = ux * ux + uy * uy + SSIM_C1
    b2 = vx + vy + SSIM_C2
    return float(np.mean((a1 * a2) / (b1 * b2), dtype=np.float64))


def ssim_skimage_default(
    gt_u8: np.ndarray, pred_u8: np.ndarray, win_size: int = SSIM_WIN_SIZE
) -> float:
    """`structural_similarity(gt, pred, channel_axis=2)` on uint8 HWC.

    Reproduces scikit-image's defaults exactly; see the module docstring
    for the list and `tests/test_diva360_official_metrics.py` for the
    parity pin against the real implementation.
    """
    if gt_u8.shape != pred_u8.shape:
        raise ValueError(f"shape mismatch {gt_u8.shape} vs {pred_u8.shape}")
    if gt_u8.ndim != 3:
        raise ValueError(f"expected HWC, got shape {gt_u8.shape}")
    if min(gt_u8.shape[0], gt_u8.shape[1]) < win_size:
        raise ValueError(
            f"image {gt_u8.shape[:2]} is smaller than the {win_size}x{win_size} window"
        )
    x = gt_u8.astype(np.float64)
    y = pred_u8.astype(np.float64)
    per_channel = [
        _ssim_plane(x[:, :, c], y[:, :, c], win_size) for c in range(x.shape[2])
    ]
    return float(np.mean(per_channel, dtype=np.float64))
