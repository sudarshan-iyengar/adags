"""Independent reference for alpha-composited colour and its VJP.

Written from the compositing EQUATION alone, not from the CUDA source, so
that it can serve as an oracle for `diff-gaussian-rasterization`. Nothing
here reads a rasterizer buffer, a kernel, or a launch configuration.

The equation, per pixel, over contributors i ordered front to back:

    T_0        = 1
    w_i        = a_i * T_{i-1}
    T_i        = T_{i-1} * (1 - a_i)
    Colour[c]  = sum_i  c_i[c] * w_i  +  T_N * bg[c]

The trailing term is the ASYMMETRY with flow, whose image has no
background at all: whatever transmittance survives the whole stack shows
the background through it. Every gradient below follows from that.

Differentiating the same equation by hand:

    d Colour[c] / d c_i[c] = w_i

    d Colour[c] / d a_i    = c_i[c] * T_{i-1}
                             - (1 / (1 - a_i))
                               * ( sum_{k > i} c_k[c] * w_k  +  T_N * bg[c] )

Raising a_i attenuates everything BEHIND i by the common factor
(1 - a_i) -- and the background is behind everything, so it sits inside
the same bracket rather than forming a term of its own. Whether an
implementation writes that bracket as one quantity or splits the
background out is a free choice; counting it in BOTH places is not, and
the corresponding regression test in `tests/test_colour_background_vjp.py`
exists because the kernel once did.

Everything is dense and readable on purpose. This is an oracle: its
value is independence and obviousness, not speed.
"""

from __future__ import annotations

import torch


def composite_colour(alphas, colours, background, order=None):
    """Alpha-composite per-primitive colours over a background.

    Args:
        alphas:     (N, H, W) per-primitive per-pixel alpha in [0, 1].
        colours:    (N, C) per-primitive colour, constant over the
                    primitive's footprint (this is what the rasterizer
                    composites: one colour per Gaussian, from its SH).
        background: (C,) background colour shown through the surviving
                    transmittance.
        order:      optional sequence of N indices, front to back.
                    Defaults to 0, 1, ..., N-1.

    Returns:
        (image, transmittance) where image is (C, H, W) and transmittance
        is (H, W), the T remaining after all contributors.
    """
    alphas, colours, background, order = _check(alphas, colours, background, order)
    channels = colours.shape[1]
    _, height, width = alphas.shape

    transmittance = torch.ones((height, width), dtype=alphas.dtype, device=alphas.device)
    image = torch.zeros((channels, height, width), dtype=alphas.dtype, device=alphas.device)

    for i in order:
        alpha = alphas[i]
        weight = alpha * transmittance
        for channel in range(channels):
            image[channel] = image[channel] + colours[i, channel] * weight
        transmittance = transmittance * (1.0 - alpha)

    for channel in range(channels):
        image[channel] = image[channel] + transmittance * background[channel]

    return image, transmittance


def colour_vjp(alphas, colours, background, dL_dimage, order=None):
    """Gradients of a scalar loss w.r.t. the colours and the alphas.

    Args:
        alphas:     (N, H, W) as in `composite_colour`.
        colours:    (N, C) as in `composite_colour`.
        background: (C,) as in `composite_colour`.
        dL_dimage:  (C, H, W) upstream gradient of the loss w.r.t. the
                    composited image.
        order:      optional front-to-back index order.

    Returns:
        (dL_dcolours, dL_dalphas):
            dL_dcolours is (N, C)
            dL_dalphas  is (N, H, W)
    """
    alphas, colours, background, order = _check(alphas, colours, background, order)
    channels = colours.shape[1]
    dL_dimage = torch.as_tensor(dL_dimage, dtype=alphas.dtype, device=alphas.device)
    if dL_dimage.shape[0] != channels or dL_dimage.shape[-2:] != alphas.shape[-2:]:
        raise ValueError(
            f"dL_dimage must be ({channels}, H, W) matching alphas, got "
            f"{tuple(dL_dimage.shape)} vs {tuple(alphas.shape)}"
        )

    count = alphas.shape[0]
    dL_dcolours = torch.zeros((count, channels), dtype=alphas.dtype, device=alphas.device)
    dL_dalphas = torch.zeros_like(alphas)

    # Forward sweep: the transmittance IN FRONT of each contributor, and
    # the weight it ends up carrying.
    transmittance_before = []
    weights = []
    transmittance = torch.ones(alphas.shape[-2:], dtype=alphas.dtype, device=alphas.device)
    for i in order:
        transmittance_before.append(transmittance)
        weights.append(alphas[i] * transmittance)
        transmittance = transmittance * (1.0 - alphas[i])

    # The suffix starts at the background rather than at zero: T_N * bg is
    # what lies behind the last contributor. Walking forward toward the
    # camera then accumulates the ordinary contributor terms on top, so
    # `suffix` is always the complete remainder behind primitive i.
    suffix = torch.stack(
        [transmittance * background[channel] for channel in range(channels)]
    )
    for position in range(count - 1, -1, -1):
        i = order[position]
        one_minus = 1.0 - alphas[i]
        weight = weights[position]
        t_before = transmittance_before[position]
        for channel in range(channels):
            upstream = dL_dimage[channel]
            dL_dcolours[i, channel] = dL_dcolours[i, channel] + (weight * upstream).sum()
            dL_dalphas[i] = dL_dalphas[i] + (
                colours[i, channel] * t_before - suffix[channel] / one_minus
            ) * upstream
        for channel in range(channels):
            suffix[channel] = suffix[channel] + colours[i, channel] * weight

    return dL_dcolours, dL_dalphas


def _check(alphas, colours, background, order):
    alphas = torch.as_tensor(alphas)
    if alphas.dim() != 3:
        raise ValueError(f"alphas must be (N, H, W), got {tuple(alphas.shape)}")
    colours = torch.as_tensor(colours, dtype=alphas.dtype, device=alphas.device)
    if colours.dim() != 2:
        raise ValueError(f"colours must be (N, C), got {tuple(colours.shape)}")
    if colours.shape[0] != alphas.shape[0]:
        raise ValueError("alphas and colours disagree on the number of primitives")
    background = torch.as_tensor(background, dtype=alphas.dtype, device=alphas.device).reshape(-1)
    if background.shape[0] != colours.shape[1]:
        raise ValueError(
            f"background must have {colours.shape[1]} channels, got {background.shape[0]}"
        )
    if order is None:
        order = list(range(alphas.shape[0]))
    else:
        order = [int(i) for i in order]
        if sorted(order) != list(range(alphas.shape[0])):
            raise ValueError("order must be a permutation of range(N)")
    return alphas, colours, background, order
