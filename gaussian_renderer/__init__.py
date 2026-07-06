#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch.nn import functional as F
import math
from .diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh, eval_shfs_4d
from utils.motion_prior_utils import project_points_to_screen
from collections import defaultdict


def _sigmoid_scalar(x):
    return 1.0 / (1.0 + math.exp(-x))


def _event_window_weight(frame_idx, frame_start, frame_end, beta):
    beta = max(float(beta), 1e-6)
    value = _sigmoid_scalar((float(frame_idx) - float(frame_start)) / beta)
    value -= _sigmoid_scalar((float(frame_idx) - float(frame_end)) / beta)
    return max(0.0, min(1.0, value))


def _apply_runtime_hide_reveal_gate(opacity, means3D, dynamic_probability, viewpoint_camera, pc):
    events = getattr(pc, "hide_reveal_runtime_events", None)
    frame_idx = getattr(viewpoint_camera, "hide_reveal_frame_idx", None)
    if not events or frame_idx is None or opacity is None or means3D.numel() == 0:
        return opacity, []

    screen_xy, valid = project_points_to_screen(means3D.detach(), viewpoint_camera)
    valid = valid.squeeze(-1)
    gate_scale = torch.ones_like(opacity)
    stats = []
    for event in events:
        frame_start = int(event["frame_start"])
        frame_end = int(event["frame_end"])
        if int(frame_idx) < frame_start or int(frame_idx) > frame_end:
            continue

        x0, y0, x1, y1 = [float(v) for v in event["crop_xyxy"]]
        support = (
            valid
            & (screen_xy[:, 0] >= x0)
            & (screen_xy[:, 0] < x1)
            & (screen_xy[:, 1] >= y0)
            & (screen_xy[:, 1] < y1)
        )
        min_dynamic_probability = event.get("dynamic_probability_min", None)
        if min_dynamic_probability is not None and dynamic_probability.numel() == opacity.numel():
            support = support & (dynamic_probability.reshape(-1) >= float(min_dynamic_probability))

        temporal_weight = _event_window_weight(
            frame_idx,
            frame_start,
            frame_end,
            event.get("event_beta", 1.0),
        )
        strength = max(0.0, min(1.0, float(event.get("opacity_attenuation", 0.95))))
        scale_value = max(0.0, min(1.0, 1.0 - strength * temporal_weight))
        if support.any():
            event_scale = torch.full_like(gate_scale, scale_value)
            gate_scale = torch.where(support[:, None], torch.minimum(gate_scale, event_scale), gate_scale)

        stats.append(
            {
                "window_id": str(event.get("window_id", "")),
                "frame_idx": int(frame_idx),
                "selected_gaussians": int(support.sum().detach().cpu()),
                "temporal_weight": float(temporal_weight),
                "opacity_scale": float(scale_value),
            }
        )
    return opacity * gate_scale, stats


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color if not pipe.env_map_res else torch.zeros(3, device="cuda"),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        sh_degree_t=pc.active_sh_degree_t,
        campos=viewpoint_camera.camera_center,
        timestamp=viewpoint_camera.timestamp,
        time_duration=pc.time_duration[1]-pc.time_duration[0],
        rot_4d=pc.rot_4d,
        gaussian_dim=pc.gaussian_dim,
        force_sh_3d=pc.force_sh_3d,
        prefiltered=False,
        opa_threshold=pipe.opa_threshold,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_dynamic_xyz(viewpoint_camera.timestamp)
    means2D = screenspace_points
    base_opacity = pc.get_opacity
    dynamic_probability = pc.get_dynamic_probability
    static_probability = pc.get_static_probability
    use_soft_routing = (
        pc.gaussian_dim == 4
        and pc.enable_soft_routing
        and dynamic_probability.numel() > 0
    )
    opacity = base_opacity * dynamic_probability

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    scales_t = None
    rotations = None
    rotations_r = None
    ts = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        if pc.rot_4d:
            cov3D_precomp, delta_mean = pc.get_current_covariance_and_mean_offset(scaling_modifier, viewpoint_camera.timestamp)
            means3D = means3D + delta_mean
        else:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        if pc.gaussian_dim == 4:
            marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp)
            opacity = opacity * marginal_t
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        if pc.gaussian_dim == 4:
            scales_t = pc.get_scaling_t
            ts = pc.get_t
            if pc.rot_4d:
                rotations_r = pc.get_rotation_r

            marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp)
            opacity = opacity * marginal_t

    opacity, runtime_hide_reveal_stats = _apply_runtime_hide_reveal_gate(
        opacity,
        means3D,
        dynamic_probability,
        viewpoint_camera,
        pc,
    )

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, pc.get_max_sh_channels)
            if pipe.compute_cov3D_python:
                dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)).detach()
            else:
                _, delta_mean = pc.get_current_covariance_and_mean_offset(scaling_modifier, viewpoint_camera.timestamp)
                dir_pp = ((means3D + delta_mean) - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)).detach()
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            if pc.gaussian_dim == 3 or pc.force_sh_3d:
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            elif pc.gaussian_dim == 4:
                dir_t = (pc.get_t - viewpoint_camera.timestamp).detach()
                sh2rgb = eval_shfs_4d(pc.active_sh_degree, pc.active_sh_degree_t, shs_view, dir_pp_normalized, dir_t, pc.time_duration[1] - pc.time_duration[0])
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
            if pc.gaussian_dim == 4 and ts is None:
                ts = pc.get_t
    else:
        colors_precomp = override_color

    if pc.gaussian_dim == 4 and getattr(pc, "enable_rendered_flow", False):
        means3D_next = pc.get_dynamic_xyz(viewpoint_camera.timestamp + getattr(pc, "motion_track_dt", 1.0 / 30.0))
        screen_now, valid_now = project_points_to_screen(means3D, viewpoint_camera)
        screen_next, valid_next = project_points_to_screen(means3D_next, viewpoint_camera)
        flow_2d = torch.where((valid_now & valid_next).expand_as(screen_now), screen_next - screen_now, torch.zeros_like(screen_now))
    else:
        flow_2d = torch.zeros_like(pc.get_xyz[:,:2])

    # Prefilter
    if pipe.compute_cov3D_python and pc.gaussian_dim == 4:
        mask = marginal_t[:,0] > 0.05
        if means2D is not None:
            means2D = means2D[mask]
        if means3D is not None:
            means3D = means3D[mask]
        if ts is not None:
            ts = ts[mask]
        if shs is not None:
            shs = shs[mask]
        if colors_precomp is not None:
            colors_precomp = colors_precomp[mask]
        if opacity is not None:
            opacity = opacity[mask]
        if scales is not None:
            scales = scales[mask]
        if scales_t is not None:
            scales_t = scales_t[mask]
        if rotations is not None:
            rotations = rotations[mask]
        if rotations_r is not None:
            rotations_r = rotations_r[mask]
        if cov3D_precomp is not None:
            cov3D_precomp = cov3D_precomp[mask]
        if flow_2d is not None:
            flow_2d = flow_2d[mask]

    # Rasterize visible Gaussians to image, obtain their radii (on screen).

    if use_soft_routing:
        means3D_static = pc.get_xyz
        opacity_static = base_opacity * static_probability
        sh_static = pc.get_features
        scales_static = pc.get_scaling
        rotations_static = pc.get_rotation
        hard_static_count = 0
    else:
        means3D_static = pc.get_static_xyz
        opacity_static = pc.get_static_opacity
        sh_static = pc.get_static_features
        scales_static = pc.get_static_scaling
        rotations_static = pc.get_static_rotation
        hard_static_count = means3D_static.shape[0]

    screenspace_points_static = torch.zeros_like(means3D_static, dtype=means3D_static.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points_static.retain_grad()
    except:
        pass
    means2D_static = screenspace_points_static

    rendered_image, radii, depth, alpha, flow, covs_com, radii_static, color_4d, color_3d, invdepth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        flow_2d = flow_2d,
        opacities = opacity,
        ts = ts,
        scales = scales,
        scales_t = scales_t,
        rotations = rotations,
        rotations_r = rotations_r,
        cov3D_precomp = cov3D_precomp,
        means3D_static = means3D_static,
        means2D_static = means2D_static,
        shs_static = sh_static,
        opacities_static = opacity_static,
        scales_static = scales_static,
        rotations_static = rotations_static)


    if pipe.env_map_res:
        assert pc.env_map is not None
        R = 60
        rays_o, rays_d = viewpoint_camera.get_rays()
        delta = ((rays_o*rays_d).sum(-1))**2 - (rays_d**2).sum(-1)*((rays_o**2).sum(-1)-R**2)
        assert (delta > 0).all()
        t_inter = -(rays_o*rays_d).sum(-1)+torch.sqrt(delta)/(rays_d**2).sum(-1)
        xyz_inter = rays_o + rays_d * t_inter.unsqueeze(-1)
        tu = torch.atan2(xyz_inter[...,1:2], xyz_inter[...,0:1]) / (2 * torch.pi) + 0.5 # theta
        tv = torch.acos(xyz_inter[...,2:3] / R) / torch.pi
        texcoord = torch.cat([tu, tv], dim=-1) * 2 - 1
        bg_color_from_envmap = F.grid_sample(pc.env_map[None], texcoord[None])[0] # 3,H,W
        # mask2 = (0 < xyz_inter[...,0]) & (xyz_inter[...,1] > 0) # & (xyz_inter[...,2] > -19)
        rendered_image = rendered_image + (1 - alpha) * bg_color_from_envmap # * mask2[None]

    if pipe.compute_cov3D_python and pc.gaussian_dim == 4:
        radii_all = radii.new_zeros(mask.shape)
        radii_all[mask] = radii
    else:
        radii_all = radii

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii_all > 0,
            "radii": radii_all,
            "depth": depth,
            "alpha": alpha,
            "flow": flow,
            "radii_static": radii_static,
            "visibility_filter_static": radii_static > 0,
            "render_4d": color_4d,
            "render_3d": color_3d,
            "viewspace_points_static": screenspace_points_static,
            "soft_routing": use_soft_routing,
            "hard_static_count": hard_static_count,
            "hide_reveal_gate_stats": runtime_hide_reveal_stats,
            }
