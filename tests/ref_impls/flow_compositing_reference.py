"""Independent reference for alpha-composited 2D flow and its VJP.

Written from the compositing EQUATION alone, not from the CUDA source, so
that it can serve as an oracle for `diff-gaussian-rasterization`. Nothing
here reads a rasterizer buffer, a kernel, or a launch configuration.

The equation, per pixel, over contributors i ordered front to back:

    T_0      = 1
    w_i      = a_i * T_{i-1}
    T_i      = T_{i-1} * (1 - a_i)
    Flow[c]  = sum_i  f_i[c] * w_i           for c in {0, 1}

Note the deliberate ASYMMETRY with colour: the flow image has NO
background term. A pixel whose transmittance never closes keeps the
partial sum as-is; no `T_final * bg` is added. Every gradient below
follows from that fact and nothing else.

Differentiating the same equation by hand:

    d Flow[c] / d f_i[c] = w_i

    d Flow[c] / d a_i    = f_i[c] * T_{i-1}
                           - (1 / (1 - a_i)) * sum_{k > i} f_k[c] * w_k

The second term is the suffix: raising a_i attenuates every contributor
BEHIND i by the common factor (1 - a_i), so each of their weights loses
w_k / (1 - a_i). This term survives even when f_i is exactly zero, which
is why a primitive that carries no flow of its own (a static Gaussian)
still receives a flow-mediated opacity gradient.

Everything is dense and readable on purpose. This is an oracle: its
value is independence and obviousness, not speed.
"""

from __future__ import annotations

import torch


def composite_flow(alphas, flows, order=None):
    """Alpha-composite per-primitive flow vectors into a flow image.

    Args:
        alphas: (N, H, W) per-primitive per-pixel alpha in [0, 1].
        flows:  (N, 2) per-primitive flow vector, constant over the
                primitive's footprint (this is what the rasterizer
                composites: one 2-vector per Gaussian).
        order:  optional sequence of N indices, front to back. Defaults
                to 0, 1, ..., N-1, i.e. `alphas` already front-to-back.

    Returns:
        (flow_image, transmittance) where flow_image is (2, H, W) and
        transmittance is (H, W), the T remaining after all contributors.
    """
    alphas, flows, order = _check(alphas, flows, order)
    _, height, width = alphas.shape

    transmittance = torch.ones((height, width), dtype=alphas.dtype, device=alphas.device)
    flow_image = torch.zeros((2, height, width), dtype=alphas.dtype, device=alphas.device)

    for i in order:
        alpha = alphas[i]
        weight = alpha * transmittance
        for channel in range(2):
            flow_image[channel] = flow_image[channel] + flows[i, channel] * weight
        transmittance = transmittance * (1.0 - alpha)

    return flow_image, transmittance


def flow_vjp(alphas, flows, dL_dflow_image, order=None):
    """Gradients of a scalar loss w.r.t. the flow vectors and the alphas.

    Args:
        alphas:         (N, H, W) as in `composite_flow`.
        flows:          (N, 2) as in `composite_flow`.
        dL_dflow_image: (2, H, W) upstream gradient of the loss w.r.t.
                        the composited flow image.
        order:          optional front-to-back index order.

    Returns:
        (dL_dflows, dL_dalphas):
            dL_dflows  is (N, 2)
            dL_dalphas is (N, H, W)
    """
    alphas, flows, order = _check(alphas, flows, order)
    dL_dflow_image = torch.as_tensor(dL_dflow_image, dtype=alphas.dtype, device=alphas.device)
    if dL_dflow_image.shape[0] != 2 or dL_dflow_image.shape[-2:] != alphas.shape[-2:]:
        raise ValueError(
            "dL_dflow_image must be (2, H, W) matching alphas, got "
            f"{tuple(dL_dflow_image.shape)} vs {tuple(alphas.shape)}"
        )

    count = alphas.shape[0]
    dL_dflows = torch.zeros((count, 2), dtype=alphas.dtype, device=alphas.device)
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

    # Backward sweep: accumulate the suffix sum_{k > i} f_k * w_k as we
    # walk from the far end toward the camera.
    suffix = torch.zeros((2,) + tuple(alphas.shape[-2:]), dtype=alphas.dtype, device=alphas.device)
    for position in range(count - 1, -1, -1):
        i = order[position]
        alpha = alphas[i]
        weight = weights[position]
        t_before = transmittance_before[position]
        one_minus = 1.0 - alpha
        for channel in range(2):
            upstream = dL_dflow_image[channel]
            dL_dflows[i, channel] = dL_dflows[i, channel] + (weight * upstream).sum()
            dL_dalphas[i] = dL_dalphas[i] + (
                flows[i, channel] * t_before - suffix[channel] / one_minus
            ) * upstream
        for channel in range(2):
            suffix[channel] = suffix[channel] + flows[i, channel] * weight

    return dL_dflows, dL_dalphas


def dL_dopacity_from_dL_dalpha(dL_dalphas, alphas, opacities):
    """Chain per-pixel d L / d alpha down to the per-primitive opacity.

    The splatting model is alpha_i(x) = o_i * G_i(x) with G_i the
    normalised Gaussian footprint, so d alpha_i(x) / d o_i = G_i(x) =
    alpha_i(x) / o_i. Valid only where alpha is NOT clamped; callers must
    keep o_i * G_i below the renderer's 0.99 ceiling for this to hold.
    """
    opacities = torch.as_tensor(opacities, dtype=alphas.dtype, device=alphas.device).reshape(-1)
    if opacities.shape[0] != alphas.shape[0]:
        raise ValueError("one opacity per primitive is required")
    if bool((opacities <= 0).any()):
        raise ValueError("opacities must be strictly positive to divide out G")
    per_primitive = (dL_dalphas * alphas / opacities.reshape(-1, 1, 1)).sum(dim=(1, 2))
    return per_primitive


def flow_loss_l1(flow_image, target, mask=None, eps=1e-6):
    """Mask-normalised L1 mirroring the training loss's reduction.

    Kept here so a test can drive the oracle and the renderer with the
    same scalar without importing the trainer.
    """
    if mask is None:
        return (flow_image - target).abs().mean()
    while mask.dim() < flow_image.dim():
        mask = mask.unsqueeze(0)
    denom = mask.sum().clamp_min(eps) * flow_image.shape[-3]
    return ((flow_image - target).abs() * mask).sum() / denom


def _check(alphas, flows, order):
    alphas = torch.as_tensor(alphas)
    if alphas.dim() != 3:
        raise ValueError(f"alphas must be (N, H, W), got {tuple(alphas.shape)}")
    flows = torch.as_tensor(flows, dtype=alphas.dtype, device=alphas.device)
    if flows.dim() != 2 or flows.shape[1] != 2:
        raise ValueError(f"flows must be (N, 2), got {tuple(flows.shape)}")
    if flows.shape[0] != alphas.shape[0]:
        raise ValueError("alphas and flows disagree on the number of primitives")
    if order is None:
        order = list(range(alphas.shape[0]))
    else:
        order = [int(i) for i in order]
        if sorted(order) != list(range(alphas.shape[0])):
            raise ValueError("order must be a permutation of range(N)")
    return alphas, flows, order
