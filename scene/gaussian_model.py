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
import numpy as np
from utils.general_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation,
    build_rotation_4d,
    build_scaling_rotation_4d,
    rotation_matrix_to_rotation_3d,
)
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.sh_utils import sh_channels_4d
import torch.nn.functional as F

from depth_visibility.capacity import CapacityBank, capacity_budget as capacity_budget_from_bank
from depth_visibility.errors import ContractError


def _as_row_bool_mask(mask, row_count, where, device=None):
    """Fail-closed (N,) bool view of a per-primitive mask.

    Rows appended after the mask was captured (densification clones/splits) are
    never protected, so a shorter mask is padded with ``False``. A longer mask
    means the caller lost row alignment and is a hard error.
    """

    if mask is None:
        return None
    values = torch.as_tensor(mask).reshape(-1)
    if values.dtype != torch.bool:
        values = values.to(torch.bool)
    if device is not None:
        values = values.to(device)
    row_count = int(row_count)
    have = int(values.shape[0])
    if have == row_count:
        return values
    if have < row_count:
        pad = torch.zeros(row_count - have, dtype=torch.bool, device=values.device)
        return torch.cat([values, pad], dim=0)
    raise ContractError(
        f"{where}: protection mask has {have} rows but the Gaussian bank has {row_count}"
    )


def _as_row_denominator(values, row_count, where, device=None):
    """Fail-closed (N, 1) float view of a per-primitive densification denominator."""

    tensor = torch.as_tensor(values).reshape(-1, 1)
    if int(tensor.shape[0]) != int(row_count):
        raise ContractError(
            f"{where}: densification denominator has {int(tensor.shape[0])} rows but the "
            f"Gaussian bank has {int(row_count)}"
        )
    if device is not None:
        tensor = tensor.to(device)
    return tensor.to(torch.float32)


class GaussianModel:

    def setup_functions(self):
        def safe_normalize(x):
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            return F.normalize(x, p=2, dim=-1, eps=1e-6)

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L.transpose(1, 2) @ L
            symm = strip_symmetric(actual_covariance)
            return symm

        def build_covariance_from_scaling_rotation_4d(scaling, scaling_modifier, rotation_l, rotation_r, dt=0.0):
            L = build_scaling_rotation_4d(scaling_modifier * scaling, rotation_l, rotation_r)
            actual_covariance = L @ L.transpose(1, 2)
            cov_11 = actual_covariance[:,:3,:3]
            cov_12 = actual_covariance[:,0:3,3:4]
            cov_t = actual_covariance[:,3:4,3:4]
            current_covariance = cov_11 - cov_12 @ cov_12.transpose(1, 2) / cov_t
            symm = strip_symmetric(current_covariance)

            if isinstance(dt, torch.Tensor) and dt.shape[1] > 1:
                # [num_pts, num_time, 3, 1]
                mean_offset = (cov_12.squeeze(-1) / cov_t.squeeze(-1))[:, None, :] * dt[..., None]
                mean_offset = mean_offset[..., None]
            else:
                mean_offset = cov_12.squeeze(-1) / cov_t.squeeze(-1) * dt
            return symm, mean_offset.squeeze(-1)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        if not self.rot_4d:
            self.covariance_activation = build_covariance_from_scaling_rotation
        else:
            self.covariance_activation = build_covariance_from_scaling_rotation_4d

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = safe_normalize


    def __init__(self, sh_degree : int, gaussian_dim : int = 3, time_duration: list = [-0.5, 0.5], rot_4d: bool = False, force_sh_3d: bool = False, sh_degree_t : int = 0):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.gaussian_dim = gaussian_dim
        self._t = torch.empty(0)
        self._scaling_t = torch.empty(0)
        self.time_duration = time_duration
        self.rot_4d = rot_4d
        self._rotation_r = torch.empty(0)
        self.force_sh_3d = force_sh_3d
        self.t_gradient_accum = torch.empty(0)
        if self.rot_4d or self.force_sh_3d:
            assert self.gaussian_dim == 4
        self.env_map = torch.empty(0)

        self.active_sh_degree_t = 0
        self.max_sh_degree_t = sh_degree_t

        # Static set
        self.static_xyz = torch.empty(0, device="cuda")
        self.static_features_dc = torch.empty(0, device="cuda")
        self.static_features_rest = torch.empty(0, device="cuda")
        self.static_scaling = torch.empty(0, device="cuda")
        self.static_rotation = torch.empty(0, device="cuda")
        self.static_opacity = torch.empty(0, device="cuda")
        self.static_max_radii2D = torch.empty(0)
        self.static_denom = torch.empty(0)
        self.static_xyz_gradient_accum = torch.empty(0)

        # Gating: monotonic logistic on log sigma_t
        self.logit_a = None  # nn.Parameter, created in training_setup for 4D
        self.logit_b = None
        self.differentiable_s = None
        self.gate_logits = None  # store logits if needed
        self._staticness_score = torch.empty(0)
        # instrumentation for debugging conversion
        self.num_static_candidates_last = 0
        self.num_converted_last = 0

        # Reversible static/dynamic routing and simple polynomial motion.
        self.enable_soft_routing = True
        self.route_logit_init = 4.0
        self._route_logit = torch.empty(0)
        self.motion_model = "poly"
        self.motion_poly_order = 2
        self._motion_v = torch.empty(0)
        self._motion_a = torch.empty(0)
        self.motion_lora_rank = 8
        self.motion_lora_anchors = 16
        self.motion_lora_init_scale = 0.01
        self._motion_lora_coeff = torch.empty(0)
        self._motion_lora_basis = None
        self.motion_scaffold_enable = False
        self.motion_scaffold_count = 512
        self.motion_scaffold_rank = 8
        self.motion_scaffold_anchors = 32
        self.motion_scaffold_knn = 4
        self.motion_scaffold_init_scale = 0.01
        self.motion_scaffold_weight_temp = 0.05
        self.motion_scaffold_chunk = 65536
        self.motion_track_dt = 1.0 / 30.0
        self._capacity_stable_ids = torch.empty(0, dtype=torch.long)
        self._capacity_generation = torch.empty(0, dtype=torch.long)
        self._capacity_last_reassigned = torch.empty(0, dtype=torch.long)
        self._capacity_next_stable_id = 0
        self._capacity_cumulative_point_iterations = 0
        self.enable_rendered_flow = False
        self.enable_motion_aware_densify = False
        self.motion_aware_densify_boost = 1.0
        self._motion_scaffold_node_xyz = torch.empty(0)
        self._motion_scaffold_coeff = torch.empty(0)
        self._motion_scaffold_basis = None
        self._motion_scaffold_attach_idx = torch.empty(0, dtype=torch.long)
        self._motion_scaffold_attach_w = torch.empty(0)

        # CSVL-VPL v2: set by the trainer to a scene.lifecycle.LifecycleManager.
        # Stays None for every lane that does not enable the lifecycle.
        self.lifecycle = None
        self._pending_lifecycle_state = None

        # EL-GS: set by the trainer to an elgs.runtime.ElgsRuntime; the
        # per-row family-id column ties rows to presence programs
        # (-1 = unassigned). Stays None for non-EL-GS lanes.
        self.elgs_runtime = None
        self._elgs_family_ids = torch.empty(0, dtype=torch.long)
        self._pending_elgs_state = None
        self._elgs_checkpoint_extras = None

        self.setup_functions()

    def get_elgs_presence(self, timestamp):
        """(N, 1) winner-lookup presence multiplier (elgs.runtime).

        _elgs_presence_override (family_id -> IntervalState, or None)
        supports counterfactual candidate renders: the live state is
        never mutated by scoring (frozen-functional snapshot).
        """
        if self.elgs_runtime is None:
            raise RuntimeError("EL-GS runtime is not attached")
        override = getattr(self, "_elgs_presence_override", None)
        return self.elgs_runtime.presence_multiplier(
            self._elgs_family_ids, float(timestamp), overrides=override
        ).to(self._opacity.device, self._opacity.dtype)

    def _capture_elgs_state(self):
        """Build the schema-versioned elgs_state for capture(), or None."""
        if self.elgs_runtime is None or self._elgs_checkpoint_extras is None:
            return None
        from elgs.state_io import build_elgs_state

        extras = self._elgs_checkpoint_extras
        self.elgs_runtime.flush_to_registry()
        payload = build_elgs_state(
            self.elgs_runtime.registry,
            extras["binding"],
            extras["ledger"],
            extras["search_cost"],
            sampler=extras["sampler"],
            slot_grid=extras.get("slot_grid", {}),
            confirmation_refs=extras.get("confirmation_refs", {}),
            moment_reset_log=extras.get("moment_reset_log", []),
            round_bookkeeping=extras.get("round_bookkeeping", {}),
            rng=extras.get("rng"),
        )
        payload_families = self._elgs_family_ids.detach().cpu().tolist()
        payload["round_bookkeeping"]["row_family_ids"] = payload_families
        return payload

    def capture(self):
        """Serialize state for checkpointing."""
        if self.gaussian_dim == 3:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
        elif self.gaussian_dim == 4:
            # Save gate parameters explicitly (if they exist)
            gate_params = None
            if self.logit_a is not None and self.logit_b is not None:
                gate_params = {
                    "logit_a": self.logit_a.detach().cpu(),
                    "logit_b": self.logit_b.detach().cpu(),
                }
            routing_motion_params = {
                "route_logit": self._route_logit,
                "motion_v": self._motion_v,
                "motion_a": self._motion_a,
                "motion_lora_coeff": self._motion_lora_coeff,
                "motion_lora_basis": self._motion_lora_basis,
                "motion_scaffold_node_xyz": self._motion_scaffold_node_xyz,
                "motion_scaffold_coeff": self._motion_scaffold_coeff,
                "motion_scaffold_basis": self._motion_scaffold_basis,
                "motion_scaffold_attach_idx": self._motion_scaffold_attach_idx,
                "motion_scaffold_attach_w": self._motion_scaffold_attach_w,
                "enable_soft_routing": self.enable_soft_routing,
                "route_logit_init": self.route_logit_init,
                "motion_model": self.motion_model,
                "motion_poly_order": self.motion_poly_order,
                "motion_lora_rank": self.motion_lora_rank,
                "motion_lora_anchors": self.motion_lora_anchors,
                "motion_lora_init_scale": self.motion_lora_init_scale,
                "motion_scaffold_enable": self.motion_scaffold_enable,
                "motion_scaffold_count": self.motion_scaffold_count,
                "motion_scaffold_rank": self.motion_scaffold_rank,
                "motion_scaffold_anchors": self.motion_scaffold_anchors,
                "motion_scaffold_knn": self.motion_scaffold_knn,
                "motion_scaffold_init_scale": self.motion_scaffold_init_scale,
                "motion_scaffold_weight_temp": self.motion_scaffold_weight_temp,
                "motion_track_dt": self.motion_track_dt,
                "capacity_state": self._capture_capacity_state(),
                "lifecycle_state": (
                    self.lifecycle.state_dict()
                    if getattr(self, "lifecycle", None) is not None
                    else None
                ),
                "elgs_state": self._capture_elgs_state(),
            }

            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.t_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
                self._t,
                self._scaling_t,
                self._rotation_r,
                self.rot_4d,
                self.env_map,
                self.active_sh_degree_t,
                self.static_xyz,
                self.static_features_dc,
                self.static_features_rest,
                self.static_scaling,
                self.static_rotation,
                self.static_opacity,
                self.static_max_radii2D,
                self.static_xyz_gradient_accum,
                self.static_denom,
                gate_params,
                routing_motion_params,
            )

    def restore(self, model_args, training_args):
        """Restore from checkpoint + re-init optimizer."""
        if self.gaussian_dim == 3:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
            ) = model_args
            gate_params = None
            routing_motion_params = None
        elif self.gaussian_dim == 4:
            routing_motion_params = None
            base_args = model_args
            if len(model_args) == 30:
                base_args = model_args[:-1]
                routing_motion_params = model_args[-1]
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                xyz_gradient_accum,
                t_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self._t,
                self._scaling_t,
                self._rotation_r,
                self.rot_4d,
                self.env_map,
                self.active_sh_degree_t,
                self.static_xyz,
                self.static_features_dc,
                self.static_features_rest,
                self.static_scaling,
                self.static_rotation,
                self.static_opacity,
                self.static_max_radii2D,
                self.static_xyz_gradient_accum,
                self.static_denom,
                gate_params,
            ) = base_args

            if routing_motion_params is not None:
                self._route_logit = routing_motion_params.get(
                    "route_logit", torch.empty(0, device=self._xyz.device)
                )
                self._motion_v = routing_motion_params.get(
                    "motion_v", torch.empty(0, device=self._xyz.device)
                )
                self._motion_a = routing_motion_params.get(
                    "motion_a", torch.empty(0, device=self._xyz.device)
                )
                self._motion_lora_coeff = routing_motion_params.get(
                    "motion_lora_coeff", torch.empty(0, device=self._xyz.device)
                )
                self._motion_lora_basis = routing_motion_params.get(
                    "motion_lora_basis", None
                )
                self._motion_scaffold_node_xyz = routing_motion_params.get(
                    "motion_scaffold_node_xyz", torch.empty(0, device=self._xyz.device)
                )
                self._motion_scaffold_coeff = routing_motion_params.get(
                    "motion_scaffold_coeff", torch.empty(0, device=self._xyz.device)
                )
                self._motion_scaffold_basis = routing_motion_params.get(
                    "motion_scaffold_basis", None
                )
                self._motion_scaffold_attach_idx = routing_motion_params.get(
                    "motion_scaffold_attach_idx", torch.empty(0, dtype=torch.long, device=self._xyz.device)
                )
                self._motion_scaffold_attach_w = routing_motion_params.get(
                    "motion_scaffold_attach_w", torch.empty(0, device=self._xyz.device)
                )
                self.enable_soft_routing = routing_motion_params.get(
                    "enable_soft_routing", self.enable_soft_routing
                )
                self.route_logit_init = routing_motion_params.get(
                    "route_logit_init", self.route_logit_init
                )
                self.motion_model = routing_motion_params.get(
                    "motion_model", self.motion_model
                )
                self.motion_poly_order = routing_motion_params.get(
                    "motion_poly_order", self.motion_poly_order
                )
                self.motion_lora_rank = routing_motion_params.get(
                    "motion_lora_rank", self.motion_lora_rank
                )
                self.motion_lora_anchors = routing_motion_params.get(
                    "motion_lora_anchors", self.motion_lora_anchors
                )
                self.motion_lora_init_scale = routing_motion_params.get(
                    "motion_lora_init_scale", self.motion_lora_init_scale
                )
                self.motion_scaffold_enable = routing_motion_params.get(
                    "motion_scaffold_enable", self.motion_scaffold_enable
                )
                self.motion_scaffold_count = routing_motion_params.get(
                    "motion_scaffold_count", self.motion_scaffold_count
                )
                self.motion_scaffold_rank = routing_motion_params.get(
                    "motion_scaffold_rank", self.motion_scaffold_rank
                )
                self.motion_scaffold_anchors = routing_motion_params.get(
                    "motion_scaffold_anchors", self.motion_scaffold_anchors
                )
                self.motion_scaffold_knn = routing_motion_params.get(
                    "motion_scaffold_knn", self.motion_scaffold_knn
                )
                self.motion_scaffold_init_scale = routing_motion_params.get(
                    "motion_scaffold_init_scale", self.motion_scaffold_init_scale
                )
                self.motion_scaffold_weight_temp = routing_motion_params.get(
                    "motion_scaffold_weight_temp", self.motion_scaffold_weight_temp
                )
                self.motion_track_dt = routing_motion_params.get(
                    "motion_track_dt", self.motion_track_dt
                )
                self._restore_capacity_state(routing_motion_params.get("capacity_state"))
                # The lifecycle manager does not exist yet at restore time; stash
                # the payload so the trainer can hand it to the manager it builds
                # immediately after. Tolerant when the checkpoint predates v2.
                self._pending_lifecycle_state = routing_motion_params.get("lifecycle_state")
                # EL-GS: same stash pattern. A MISSING elgs_state is a
                # baseline checkpoint (substrate initializes fresh); a
                # PRESENT payload is validated by the trainer's setup
                # via elgs.state_io.load_elgs_state (present-but-invalid
                # is an explicit rejection there, never a silent skip).
                self._pending_elgs_state = routing_motion_params.get("elgs_state")

        if training_args is not None:
            self.training_setup(training_args)

            # Restore optimizer + accumulators
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)

            if self.gaussian_dim == 4:
                self.t_gradient_accum = t_gradient_accum

                # Restore gate params if present
                if gate_params is not None:
                    if self.logit_a is None:
                        self.logit_a = nn.Parameter(
                            gate_params["logit_a"].to("cuda"), requires_grad=True
                        )
                    else:
                        self.logit_a.data = gate_params["logit_a"].to("cuda")
                    if self.logit_b is None:
                        self.logit_b = nn.Parameter(
                            gate_params["logit_b"].to("cuda"), requires_grad=True
                        )
                    else:
                        self.logit_b.data = gate_params["logit_b"].to("cuda")


    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_scaling_t(self):
        return self.scaling_activation(self._scaling_t)

    @property
    def get_scaling_xyzt(self):
        return self.scaling_activation(torch.cat([self._scaling, self._scaling_t], dim = 1))

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_rotation_r(self):
        return self.rotation_activation(self._rotation_r)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_t(self):
        return self._t

    @property
    def get_xyzt(self):
        return torch.cat([self._xyz, self._t], dim = 1)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_static_xyz(self):
        return self.static_xyz

    @property
    def get_static_features(self):
        if self.static_features_dc.numel() == 0:
            return self.static_features_dc
        features_dc = self.static_features_dc
        features_rest = self.static_features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_static_opacity(self):
        return self.opacity_activation(self.static_opacity)

    @property
    def get_static_scaling(self):
        return self.scaling_activation(self.static_scaling)

    @property
    def get_static_rotation(self):
        if len(self.static_rotation) == 0:
            return self.static_rotation
        return self.rotation_activation(self.static_rotation)

    @property
    def get_route_logit(self):
        return self._route_logit

    @property
    def get_dynamic_probability(self):
        if (
                not self.enable_soft_routing
                or self._route_logit.numel() == 0
                or self._route_logit.shape[0] != self._xyz.shape[0]
        ):
            return torch.ones((self._xyz.shape[0], 1), device=self._xyz.device)
        return torch.sigmoid(self._route_logit)

    @property
    def get_static_probability(self):
        return 1.0 - self.get_dynamic_probability

    @property
    def get_motion_v(self):
        return self._motion_v

    @property
    def get_motion_a(self):
        return self._motion_a

    @property
    def get_motion_lora_coeff(self):
        return self._motion_lora_coeff

    @property
    def get_motion_lora_basis(self):
        return self._motion_lora_basis

    @property
    def get_motion_scaffold_node_xyz(self):
        return self._motion_scaffold_node_xyz

    @property
    def get_motion_scaffold_coeff(self):
        return self._motion_scaffold_coeff

    @property
    def get_motion_scaffold_basis(self):
        return self._motion_scaffold_basis

    @property
    def get_motion_scaffold_attach_idx(self):
        return self._motion_scaffold_attach_idx

    @property
    def get_motion_scaffold_attach_w(self):
        return self._motion_scaffold_attach_w

    def get_polynomial_motion_offset(self, timestamp):
        if (
                self.gaussian_dim != 4
                or self.motion_model != "poly"
                or self._motion_v.numel() == 0
                or self._motion_v.shape[0] != self._xyz.shape[0]
        ):
            return torch.zeros_like(self._xyz)

        dt = timestamp - self.get_t
        offset = self._motion_v * dt
        if self.motion_poly_order >= 2 and self._motion_a.numel() > 0:
            offset = offset + self._motion_a * dt * dt
        return offset

    def _sample_lora_basis(self, relative_dt):
        if self._motion_lora_basis is None:
            return None
        if self._motion_lora_basis.numel() == 0:
            return None

        basis = self._motion_lora_basis
        num_anchors = basis.shape[1]
        if num_anchors < 2:
            return basis[:, 0:1, :].permute(1, 0, 2).expand(relative_dt.shape[0], -1, -1)

        duration = max(float(self.time_duration[1] - self.time_duration[0]), 1e-6)
        u = ((relative_dt / duration) + 1.0) * 0.5
        u = u.clamp(0.0, 1.0).squeeze(-1)
        pos = u * float(num_anchors - 1)
        lo = torch.floor(pos).long().clamp(0, num_anchors - 1)
        hi = (lo + 1).clamp(0, num_anchors - 1)
        w = (pos - lo.float()).view(1, -1, 1)

        basis_lo = basis[:, lo, :]
        basis_hi = basis[:, hi, :]
        sampled = basis_lo * (1.0 - w) + basis_hi * w
        return sampled.permute(1, 0, 2)

    def get_lora_motion_offset(self, timestamp):
        if (
                self.gaussian_dim != 4
                or self.motion_model != "lora"
                or self._motion_lora_coeff.numel() == 0
                or self._motion_lora_coeff.shape[0] != self._xyz.shape[0]
                or self._motion_lora_basis is None
        ):
            return torch.zeros_like(self._xyz)

        relative_dt = timestamp - self.get_t
        basis_at_dt = self._sample_lora_basis(relative_dt)
        basis_at_center = self._sample_lora_basis(torch.zeros_like(relative_dt))
        if basis_at_dt is None or basis_at_center is None:
            return torch.zeros_like(self._xyz)

        centered_basis = basis_at_dt - basis_at_center
        return torch.einsum("nr,nrd->nd", self._motion_lora_coeff, centered_basis)

    def _sample_scaffold_basis(self, relative_dt):
        if self._motion_scaffold_basis is None:
            return None
        if self._motion_scaffold_basis.numel() == 0:
            return None

        basis = self._motion_scaffold_basis
        num_anchors = basis.shape[1]
        if num_anchors < 2:
            return torch.zeros(
                (relative_dt.shape[0], self.motion_scaffold_rank, 3),
                device=relative_dt.device,
                dtype=relative_dt.dtype,
            )

        dt_min = float(self.time_duration[0] - self.time_duration[1])
        dt_max = float(self.time_duration[1] - self.time_duration[0])
        dt_norm = (relative_dt - dt_min) / max(dt_max - dt_min, 1e-6)
        dt_norm = dt_norm.clamp(0.0, 1.0)
        anchor_pos = dt_norm * (num_anchors - 1)
        idx0 = torch.floor(anchor_pos).long().clamp(0, num_anchors - 1)
        idx1 = (idx0 + 1).clamp(0, num_anchors - 1)
        frac = (anchor_pos - idx0.float()).view(-1, 1, 1)

        basis0 = basis[:, idx0.view(-1), :].permute(1, 0, 2)
        basis1 = basis[:, idx1.view(-1), :].permute(1, 0, 2)
        return basis0 * (1.0 - frac) + basis1 * frac

    def get_scaffold_motion_offset(self, timestamp):
        if (
                self.gaussian_dim != 4
                or not self.motion_scaffold_enable
                or self._motion_scaffold_coeff.numel() == 0
                or self._motion_scaffold_basis is None
                or self._motion_scaffold_attach_idx.numel() == 0
                or self._motion_scaffold_attach_w.numel() == 0
                or self._motion_scaffold_attach_idx.shape[0] != self._xyz.shape[0]
        ):
            return torch.zeros_like(self._xyz)

        relative_dt = timestamp - self.get_t
        basis_at_dt = self._sample_scaffold_basis(relative_dt)
        basis_at_center = self._sample_scaffold_basis(torch.zeros_like(relative_dt))
        if basis_at_dt is None or basis_at_center is None:
            return torch.zeros_like(self._xyz)

        centered_basis = basis_at_dt - basis_at_center
        idx = self._motion_scaffold_attach_idx.long().clamp(0, max(self._motion_scaffold_coeff.shape[0] - 1, 0))
        weights = self._motion_scaffold_attach_w.to(centered_basis.dtype)
        coeff = self._motion_scaffold_coeff[idx.reshape(-1)].reshape(
            idx.shape[0], idx.shape[1], self.motion_scaffold_rank
        )
        attached_offsets = torch.einsum("nkr,nrd->nkd", coeff, centered_basis)
        return (attached_offsets * weights[..., None]).sum(dim=1)

    def get_motion_offset(self, timestamp):
        if self.gaussian_dim != 4:
            return torch.zeros_like(self._xyz)
        if self.motion_model == "poly" and not self.rot_4d:
            return self.get_polynomial_motion_offset(timestamp)
        if self.motion_model == "lora" and not self.rot_4d:
            return self.get_lora_motion_offset(timestamp) + self.get_scaffold_motion_offset(timestamp)
        return torch.zeros_like(self._xyz)

    def get_scaffold_smoothness_loss(self):
        if (
                not self.motion_scaffold_enable
                or self._motion_scaffold_coeff.numel() == 0
                or self._motion_scaffold_node_xyz.numel() == 0
                or self._motion_scaffold_coeff.shape[0] < 2
        ):
            return torch.zeros((), device=self._xyz.device)
        with torch.no_grad():
            k = min(4, self._motion_scaffold_node_xyz.shape[0])
            dist = torch.cdist(self._motion_scaffold_node_xyz, self._motion_scaffold_node_xyz)
            _, idx = torch.topk(dist, k=k, largest=False, dim=1)
            idx = idx[:, 1:] if k > 1 else idx
        if idx.numel() == 0:
            return torch.zeros((), device=self._xyz.device)
        neighbor_coeff = self._motion_scaffold_coeff[idx]
        center_coeff = self._motion_scaffold_coeff[:, None, :]
        return (neighbor_coeff - center_coeff).pow(2).mean()

    def get_scaffold_reg_loss(self):
        if not self.motion_scaffold_enable:
            return torch.zeros((), device=self._xyz.device)
        loss = torch.zeros((), device=self._xyz.device)
        if self._motion_scaffold_coeff.numel() > 0:
            loss = loss + self._motion_scaffold_coeff.pow(2).mean()
        if self._motion_scaffold_basis is not None and self._motion_scaffold_basis.numel() > 0:
            loss = loss + self._motion_scaffold_basis.pow(2).mean()
        return loss

    def get_dynamic_xyz(self, timestamp):
        if self.gaussian_dim == 4 and self.motion_model == "poly" and not self.rot_4d:
            return self._xyz + self.get_polynomial_motion_offset(timestamp)
        if self.gaussian_dim == 4 and self.motion_model == "lora" and not self.rot_4d:
            return self._xyz + self.get_lora_motion_offset(timestamp) + self.get_scaffold_motion_offset(timestamp)
        return self._xyz

    @property
    def get_max_sh_channels(self):
        if self.gaussian_dim == 3 or self.force_sh_3d:
            return (self.max_sh_degree+1)**2
        elif self.gaussian_dim == 4 and self.max_sh_degree_t == 0:
            return sh_channels_4d[self.max_sh_degree]
        elif self.gaussian_dim == 4 and self.max_sh_degree_t > 0:
            return (self.max_sh_degree+1)**2 * (self.max_sh_degree_t + 1)

    # ----------------- Covariance / motion -----------------

    def get_cov_t(self, scaling_modifier = 1):
        if self.rot_4d:
            L = build_scaling_rotation_4d(
                scaling_modifier * self.get_scaling_xyzt, self._rotation, self._rotation_r
            )
            actual_covariance = L @ L.transpose(1, 2)
            return actual_covariance[:,3,3].unsqueeze(1)
        else:
            return self.get_scaling_t * scaling_modifier

    def get_marginal_t(self, timestamp, scaling_modifier = 1): # Standard
        sigma = self.get_cov_t(scaling_modifier)
        return torch.exp(-0.5*(self.get_t-timestamp)**2/sigma) # / torch.sqrt(2*torch.pi*sigma)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def get_current_covariance_and_mean_offset(self, scaling_modifier=1, timestamp=0.0):
        if self.rot_4d:
            return self.covariance_activation(
                self.get_scaling_xyzt,
                scaling_modifier,
                self._rotation,
                self._rotation_r,
                dt = timestamp - self.get_t
            )
        else:
            covariance = self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
            mean_offset = torch.zeros_like(self._xyz, device=self._xyz.device)
            return covariance, mean_offset

    def compute_motion_magnitude(self, dt=0.1, scaling_modifier=1.0):
        """
        Approximate per-Gaussian motion magnitude using the 4D covariance-induced mean offsets.
        Only meaningful when rot_4d=True and gaussian_dim==4.
        """
        if self.gaussian_dim != 4 or not self.rot_4d:
            return None
        if self._t.numel() == 0:
            return None

        # Use the same mechanism as rigid loss: evaluate mean offset at t+dt
        _, mean_offset = self.get_current_covariance_and_mean_offset(
            scaling_modifier=scaling_modifier,
            timestamp=self.get_t + dt,
        )
        if mean_offset is None or mean_offset.numel() == 0:
            return None

        # mean_offset: [N,3]
        motion_mag = torch.norm(mean_offset, dim=-1, keepdim=True)
        motion_mag = torch.nan_to_num(motion_mag)
        return motion_mag

    # ----------------- Gating (log-sigma_t monotonic) -----------------

    def compute_differentiable_staticness(self):
        """
        Compute s in [0,1] as a monotonic function of log sigma_t.

        Larger temporal scale (sigma_t) ⇒ larger s (more static).
        No per-point MLP; only two shared params logit_a, logit_b.
        """
        if (
                self.gaussian_dim == 4
                and self._scaling_t.numel() > 0
                and self.logit_a is not None
                and self.logit_b is not None
        ):
            # _scaling_t is in log-scale because scaling_activation=exp.
            log_sigma_t = self._scaling_t.detach()  # [N,1]
            logits = self.logit_a * log_sigma_t + self.logit_b  # [N,1]
            self.gate_logits = logits
            self.differentiable_s = torch.sigmoid(logits)
        else:
            if self._xyz.numel() > 0:
                self.gate_logits = None
                self.differentiable_s = torch.zeros(
                    (self._xyz.shape[0], 1), device="cuda"
                )
            else:
                self.gate_logits = None
                self.differentiable_s = None

    @torch.no_grad()
    def update_staticness_score(self):
        if self.differentiable_s is not None:
            self._staticness_score = self.differentiable_s.detach()

    # ----------------- IO helpers -----------------

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))

        if self.active_sh_degree > 0:
            for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        if self.active_sh_degree != 0:
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        # TODO: may need to add empty shs for SIBR_viewer?
        if self.active_sh_degree > 0:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # ----------------- SH scheduling -----------------

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
        elif self.max_sh_degree_t and self.active_sh_degree_t < self.max_sh_degree_t:
            self.active_sh_degree_t += 1

    # ----------------- Initialization -----------------

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, self.get_max_sh_channels)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        if self.gaussian_dim == 4:
            if pcd.time is None:
                fused_times = (torch.rand(fused_point_cloud.shape[0], 1, device="cuda") * 1.2 - 0.1) * (self.time_duration[1] - self.time_duration[0]) + self.time_duration[0]
            else:
                fused_times = torch.from_numpy(pcd.time).cuda().float()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        if self.gaussian_dim == 4:
            # dist_t = torch.clamp_min(distCUDA2(fused_times.repeat(1,3)), 1e-10)[...,None]
            dist_t = torch.zeros_like(fused_times, device="cuda") + (self.time_duration[1] - self.time_duration[0]) / 5
            scales_t = torch.log(torch.sqrt(dist_t))
            if self.rot_4d:
                rots_r = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
                rots_r[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.gaussian_dim == 4:
            self._t = nn.Parameter(fused_times.requires_grad_(True))
            self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
            if self.rot_4d:
                self._rotation_r = nn.Parameter(rots_r.requires_grad_(True))
            self._route_logit = nn.Parameter(
                torch.full(
                    (self.get_xyz.shape[0], 1),
                    self.route_logit_init,
                    device="cuda",
                ).requires_grad_(True)
            )
            self._motion_v = nn.Parameter(
                torch.zeros((self.get_xyz.shape[0], 3), device="cuda").requires_grad_(True)
            )
            self._motion_a = nn.Parameter(
                torch.zeros((self.get_xyz.shape[0], 3), device="cuda").requires_grad_(True)
            )
            self._motion_lora_coeff = nn.Parameter(
                torch.zeros((self.get_xyz.shape[0], self.motion_lora_rank), device="cuda").requires_grad_(True)
            )
            self._ensure_capacity_state(created_iteration=0)

    def create_from_pth(self, path, spatial_lr_scale):
        assert self.gaussian_dim == 4 and self.rot_4d
        self.spatial_lr_scale = spatial_lr_scale
        init_4d_gaussian = torch.load(path)
        fused_point_cloud = init_4d_gaussian['xyz'].cuda()
        features_dc = init_4d_gaussian['features_dc'].cuda()
        features_rest = init_4d_gaussian['features_rest'].cuda()
        fused_times = init_4d_gaussian['t'].cuda()
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        scales = init_4d_gaussian['scaling'].cuda()
        rots = init_4d_gaussian['rotation'].cuda()
        scales_t = init_4d_gaussian['scaling_t'].cuda()
        rots_r = init_4d_gaussian['rotation_r'].cuda()
        opacities = init_4d_gaussian['opacity'].cuda()

        if init_4d_gaussian['static_xyz'] is not None:
            static_xyz = init_4d_gaussian['static_xyz'].cuda()
            static_features_dc = init_4d_gaussian['static_features_dc'].cuda()
            static_features_rest = init_4d_gaussian['static_features_rest'].cuda()
            static_scaling = init_4d_gaussian['static_scaling'].cuda()
            static_rotation = init_4d_gaussian['static_rotation'].cuda()
            static_opacity = init_4d_gaussian['static_opacity'].cuda()

            self.static_xyz = nn.Parameter(static_xyz.requires_grad_(True))
            self.static_features_dc = nn.Parameter(static_features_dc.requires_grad_(True))
            self.static_features_rest = nn.Parameter(static_features_rest.requires_grad_(True))
            self.static_scaling = nn.Parameter(static_scaling.requires_grad_(True))
            self.static_rotation = nn.Parameter(static_rotation.requires_grad_(True))
            self.static_opacity = nn.Parameter(static_opacity.requires_grad_(True))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.transpose(1, 2).requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest.transpose(1, 2).requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self._t = nn.Parameter(fused_times.requires_grad_(True))
        self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
        self._rotation_r = nn.Parameter(rots_r.requires_grad_(True))
        self._route_logit = nn.Parameter(
            init_4d_gaussian.get(
                "route_logit",
                torch.full(
                    (self.get_xyz.shape[0], 1),
                    self.route_logit_init,
                    device="cuda",
                ),
            ).cuda().requires_grad_(True)
        )
        self._motion_v = nn.Parameter(
            init_4d_gaussian.get(
                "motion_v",
                torch.zeros((self.get_xyz.shape[0], 3), device="cuda"),
            ).cuda().requires_grad_(True)
        )
        self._motion_a = nn.Parameter(
            init_4d_gaussian.get(
                "motion_a",
                torch.zeros((self.get_xyz.shape[0], 3), device="cuda"),
            ).cuda().requires_grad_(True)
        )
        self._motion_lora_coeff = nn.Parameter(
            init_4d_gaussian.get(
                "motion_lora_coeff",
                torch.zeros((self.get_xyz.shape[0], self.motion_lora_rank), device="cuda"),
            ).cuda().requires_grad_(True)
        )
        motion_lora_basis = init_4d_gaussian.get("motion_lora_basis", None)
        if motion_lora_basis is not None:
            self._motion_lora_basis = nn.Parameter(motion_lora_basis.cuda().requires_grad_(True))
        self._motion_scaffold_node_xyz = init_4d_gaussian.get(
            "motion_scaffold_node_xyz", torch.empty(0, device="cuda")
        ).cuda()
        self._motion_scaffold_attach_idx = init_4d_gaussian.get(
            "motion_scaffold_attach_idx", torch.empty(0, dtype=torch.long, device="cuda")
        ).cuda().long()
        self._motion_scaffold_attach_w = init_4d_gaussian.get(
            "motion_scaffold_attach_w", torch.empty(0, device="cuda")
        ).cuda()
        motion_scaffold_coeff = init_4d_gaussian.get("motion_scaffold_coeff", None)
        if motion_scaffold_coeff is not None:
            self._motion_scaffold_coeff = nn.Parameter(motion_scaffold_coeff.cuda().requires_grad_(True))
        motion_scaffold_basis = init_4d_gaussian.get("motion_scaffold_basis", None)
        if motion_scaffold_basis is not None:
            self._motion_scaffold_basis = nn.Parameter(motion_scaffold_basis.cuda().requires_grad_(True))
        self._ensure_capacity_state(created_iteration=0)

    # ----------------- Training setup -----------------

    def _init_motion_scaffold_basis(self, device):
        anchor_grid = torch.linspace(-1.0, 1.0, self.motion_scaffold_anchors, device=device)
        basis = torch.zeros((self.motion_scaffold_rank, self.motion_scaffold_anchors, 3), device=device)
        for rank_idx in range(self.motion_scaffold_rank):
            freq = rank_idx // 6 + 1
            phase = torch.sin(anchor_grid * torch.pi * freq) if rank_idx % 2 == 0 else torch.cos(anchor_grid * torch.pi * freq) - 1.0
            axis = rank_idx % 3
            basis[rank_idx, :, axis] = phase * self.motion_scaffold_init_scale
        return basis

    @torch.no_grad()
    def _init_motion_scaffold_nodes(self, device):
        n_points = self.get_xyz.shape[0]
        if n_points == 0 or self.motion_scaffold_count <= 0:
            return torch.empty((0, 3), device=device)
        count = min(int(self.motion_scaffold_count), n_points)
        sample_idx = torch.linspace(0, n_points - 1, count, device=device).long()
        return self.get_xyz.detach()[sample_idx].clone()

    @torch.no_grad()
    def _compute_scaffold_attachments(self, xyz):
        if (
                xyz.numel() == 0
                or self._motion_scaffold_node_xyz.numel() == 0
                or self.motion_scaffold_knn <= 0
        ):
            return (
                torch.empty((xyz.shape[0], 0), dtype=torch.long, device=xyz.device),
                torch.empty((xyz.shape[0], 0), dtype=xyz.dtype, device=xyz.device),
            )

        node_xyz = self._motion_scaffold_node_xyz.to(xyz.device, xyz.dtype)
        k = min(int(self.motion_scaffold_knn), node_xyz.shape[0])
        idx_chunks, weight_chunks = [], []
        chunk = max(int(self.motion_scaffold_chunk), 1)
        for start in range(0, xyz.shape[0], chunk):
            cur = xyz[start:start + chunk]
            dist = torch.cdist(cur, node_xyz)
            vals, idx = torch.topk(dist, k=k, largest=False, dim=1)
            weights = torch.softmax(-vals / max(float(self.motion_scaffold_weight_temp), 1e-6), dim=1)
            idx_chunks.append(idx.long())
            weight_chunks.append(weights.to(xyz.dtype))
        return torch.cat(idx_chunks, dim=0), torch.cat(weight_chunks, dim=0)

    @torch.no_grad()
    def _rebuild_motion_scaffold_attachments(self):
        idx, weights = self._compute_scaffold_attachments(self.get_xyz.detach())
        self._motion_scaffold_attach_idx = idx
        self._motion_scaffold_attach_w = weights

    def _ensure_route_and_motion_tensors(self, route_logit_init=None):
        n_points = self.get_xyz.shape[0]
        device = self.get_xyz.device
        init = self.route_logit_init if route_logit_init is None else route_logit_init

        if self._route_logit.numel() == 0 or self._route_logit.shape[0] != n_points:
            self._route_logit = nn.Parameter(
                torch.full((n_points, 1), init, device=device).requires_grad_(True)
            )
        elif not isinstance(self._route_logit, nn.Parameter):
            self._route_logit = nn.Parameter(self._route_logit.to(device).requires_grad_(True))

        if self._motion_v.numel() == 0 or self._motion_v.shape[0] != n_points:
            self._motion_v = nn.Parameter(
                torch.zeros((n_points, 3), device=device).requires_grad_(True)
            )
        elif not isinstance(self._motion_v, nn.Parameter):
            self._motion_v = nn.Parameter(self._motion_v.to(device).requires_grad_(True))

        if self._motion_a.numel() == 0 or self._motion_a.shape[0] != n_points:
            self._motion_a = nn.Parameter(
                torch.zeros((n_points, 3), device=device).requires_grad_(True)
            )
        elif not isinstance(self._motion_a, nn.Parameter):
            self._motion_a = nn.Parameter(self._motion_a.to(device).requires_grad_(True))

        if self._motion_lora_coeff.numel() == 0 or self._motion_lora_coeff.shape[0] != n_points or self._motion_lora_coeff.shape[1] != self.motion_lora_rank:
            self._motion_lora_coeff = nn.Parameter(
                torch.zeros((n_points, self.motion_lora_rank), device=device).requires_grad_(True)
            )
        elif not isinstance(self._motion_lora_coeff, nn.Parameter):
            self._motion_lora_coeff = nn.Parameter(self._motion_lora_coeff.to(device).requires_grad_(True))

        need_basis = (
                self._motion_lora_basis is None
                or self._motion_lora_basis.numel() == 0
                or self._motion_lora_basis.shape[0] != self.motion_lora_rank
                or self._motion_lora_basis.shape[1] != self.motion_lora_anchors
        )
        if need_basis:
            anchor_grid = torch.linspace(-1.0, 1.0, self.motion_lora_anchors, device=device)
            basis = torch.zeros((self.motion_lora_rank, self.motion_lora_anchors, 3), device=device)
            for rank_idx in range(self.motion_lora_rank):
                freq = rank_idx // 6 + 1
                phase = torch.sin(anchor_grid * torch.pi * freq) if rank_idx % 2 == 0 else torch.cos(anchor_grid * torch.pi * freq) - 1.0
                axis = rank_idx % 3
                basis[rank_idx, :, axis] = phase * self.motion_lora_init_scale
            self._motion_lora_basis = nn.Parameter(basis.requires_grad_(True))
        elif not isinstance(self._motion_lora_basis, nn.Parameter):
            self._motion_lora_basis = nn.Parameter(self._motion_lora_basis.to(device).requires_grad_(True))

        if self.motion_scaffold_enable:
            node_count = min(max(int(self.motion_scaffold_count), 0), max(n_points, 0))
            need_nodes = (
                    self._motion_scaffold_node_xyz.numel() == 0
                    or self._motion_scaffold_node_xyz.shape[0] != node_count
                    or self._motion_scaffold_node_xyz.shape[1] != 3
            )
            if need_nodes:
                self._motion_scaffold_node_xyz = self._init_motion_scaffold_nodes(device)
            else:
                self._motion_scaffold_node_xyz = self._motion_scaffold_node_xyz.to(device)

            if (
                    self._motion_scaffold_coeff.numel() == 0
                    or self._motion_scaffold_coeff.shape[0] != self._motion_scaffold_node_xyz.shape[0]
                    or self._motion_scaffold_coeff.shape[1] != self.motion_scaffold_rank
            ):
                self._motion_scaffold_coeff = nn.Parameter(
                    torch.zeros(
                        (self._motion_scaffold_node_xyz.shape[0], self.motion_scaffold_rank),
                        device=device,
                    ).requires_grad_(True)
                )
            elif not isinstance(self._motion_scaffold_coeff, nn.Parameter):
                self._motion_scaffold_coeff = nn.Parameter(self._motion_scaffold_coeff.to(device).requires_grad_(True))

            need_scaffold_basis = (
                    self._motion_scaffold_basis is None
                    or self._motion_scaffold_basis.numel() == 0
                    or self._motion_scaffold_basis.shape[0] != self.motion_scaffold_rank
                    or self._motion_scaffold_basis.shape[1] != self.motion_scaffold_anchors
            )
            if need_scaffold_basis:
                self._motion_scaffold_basis = nn.Parameter(
                    self._init_motion_scaffold_basis(device).requires_grad_(True)
                )
            elif not isinstance(self._motion_scaffold_basis, nn.Parameter):
                self._motion_scaffold_basis = nn.Parameter(self._motion_scaffold_basis.to(device).requires_grad_(True))

            expected_knn = min(int(self.motion_scaffold_knn), self._motion_scaffold_node_xyz.shape[0])
            if (
                    self._motion_scaffold_attach_idx.numel() == 0
                    or self._motion_scaffold_attach_idx.shape[0] != n_points
                    or self._motion_scaffold_attach_idx.shape[1] != expected_knn
            ):
                self._rebuild_motion_scaffold_attachments()
            else:
                self._motion_scaffold_attach_idx = self._motion_scaffold_attach_idx.to(device).long()
                self._motion_scaffold_attach_w = self._motion_scaffold_attach_w.to(device)


    def _capacity_device(self):
        return self.get_xyz.device if torch.is_tensor(self.get_xyz) else torch.device("cpu")

    def _capture_capacity_state(self):
        self._ensure_capacity_state()
        return {
            "schema_version": "phase9-capacity-state-v1",
            "stable_ids": self._capacity_stable_ids.detach().cpu(),
            "generation": self._capacity_generation.detach().cpu(),
            "last_reassigned": self._capacity_last_reassigned.detach().cpu(),
            "next_stable_id": int(self._capacity_next_stable_id),
            "cumulative_point_iterations": int(self._capacity_cumulative_point_iterations),
        }

    def _restore_capacity_state(self, state):
        if not state:
            self._ensure_capacity_state(created_iteration=0)
            return
        if state.get("schema_version") != "phase9-capacity-state-v1":
            raise ContractError("unsupported capacity state schema")
        n_points = int(self.get_xyz.shape[0])
        device = self._capacity_device()
        stable_ids = torch.as_tensor(state["stable_ids"], dtype=torch.long, device=device).reshape(-1)
        generation = torch.as_tensor(state["generation"], dtype=torch.long, device=device).reshape(-1)
        last_reassigned = torch.as_tensor(state["last_reassigned"], dtype=torch.long, device=device).reshape(-1)
        if stable_ids.shape[0] != n_points or generation.shape[0] != n_points or last_reassigned.shape[0] != n_points:
            raise ContractError("capacity state row count does not match Gaussian rows")
        if stable_ids.numel() and torch.unique(stable_ids).numel() != stable_ids.numel():
            raise ContractError("capacity stable IDs must be unique")
        self._capacity_stable_ids = stable_ids
        self._capacity_generation = generation
        self._capacity_last_reassigned = last_reassigned
        next_id = int(state.get("next_stable_id", 0))
        minimum_next = int(stable_ids.max().item()) + 1 if stable_ids.numel() else 0
        self._capacity_next_stable_id = max(next_id, minimum_next)
        self._capacity_cumulative_point_iterations = int(state.get("cumulative_point_iterations", 0))

    def _ensure_capacity_state(self, created_iteration=0):
        n_points = int(self.get_xyz.shape[0])
        device = self._capacity_device()
        valid = (
            self._capacity_stable_ids.shape[0] == n_points
            and self._capacity_generation.shape[0] == n_points
            and self._capacity_last_reassigned.shape[0] == n_points
        )
        if not valid:
            self._capacity_stable_ids = torch.arange(n_points, dtype=torch.long, device=device)
            self._capacity_generation = torch.full((n_points,), int(created_iteration), dtype=torch.long, device=device)
            self._capacity_last_reassigned = torch.zeros(n_points, dtype=torch.long, device=device)
            self._capacity_next_stable_id = n_points
            return
        self._capacity_stable_ids = self._capacity_stable_ids.to(device=device, dtype=torch.long)
        self._capacity_generation = self._capacity_generation.to(device=device, dtype=torch.long)
        self._capacity_last_reassigned = self._capacity_last_reassigned.to(device=device, dtype=torch.long)
        minimum_next = int(self._capacity_stable_ids.max().item()) + 1 if n_points else 0
        self._capacity_next_stable_id = max(int(self._capacity_next_stable_id), minimum_next)

    def _append_capacity_rows(self, row_count, created_iteration):
        row_count = int(row_count)
        if self.gaussian_dim != 4 or row_count <= 0:
            return
        current_count = int(self.get_xyz.shape[0])
        old_count = int(self._capacity_stable_ids.shape[0])
        if current_count != old_count + row_count:
            raise ContractError("capacity append does not match Gaussian row growth")
        device = self._capacity_device()
        start = int(self._capacity_next_stable_id)
        new_ids = torch.arange(start, start + row_count, dtype=torch.long, device=device)
        self._capacity_stable_ids = torch.cat([self._capacity_stable_ids.to(device), new_ids], dim=0)
        self._capacity_generation = torch.cat(
            [
                self._capacity_generation.to(device),
                torch.full((row_count,), int(created_iteration or 0), dtype=torch.long, device=device),
            ],
            dim=0,
        )
        self._capacity_last_reassigned = torch.cat(
            [self._capacity_last_reassigned.to(device), torch.zeros(row_count, dtype=torch.long, device=device)],
            dim=0,
        )
        self._capacity_next_stable_id = start + row_count

    def _prune_capacity_state(self, valid_points_mask):
        if self.gaussian_dim != 4:
            return
        mask = torch.as_tensor(valid_points_mask, dtype=torch.bool, device=self._capacity_device()).reshape(-1)
        if mask.shape[0] != self._capacity_stable_ids.shape[0]:
            raise ContractError("capacity prune mask does not match stable ID rows")
        self._capacity_stable_ids = self._capacity_stable_ids[mask]
        self._capacity_generation = self._capacity_generation[mask]
        self._capacity_last_reassigned = self._capacity_last_reassigned[mask]

    def build_capacity_bank(self):
        if self.gaussian_dim != 4:
            raise ContractError("Slice B capacity bank requires a 4D Gaussian model")
        self._ensure_capacity_state()
        parameters = {
            "_xyz": self._xyz,
            "_features_dc": self._features_dc,
            "_features_rest": self._features_rest,
            "_scaling": self._scaling,
            "_rotation": self._rotation,
            "_opacity": self._opacity,
            "_t": self._t,
            "_scaling_t": self._scaling_t,
            "_route_logit": self._route_logit,
            "_motion_v": self._motion_v,
            "_motion_a": self._motion_a,
            "_motion_lora_coeff": self._motion_lora_coeff,
            "_staticness_score": self._staticness_score,
        }
        parameters = {
            name: value for name, value in parameters.items()
            if isinstance(value, nn.Parameter) and value.shape[:1] == self._xyz.shape[:1]
        }
        accumulators = {
            "xyz_gradient_accum": self.xyz_gradient_accum,
            "t_gradient_accum": self.t_gradient_accum,
            "denom": self.denom,
            "max_radii2D": self.max_radii2D,
        }
        accumulators = {
            name: value for name, value in accumulators.items()
            if torch.is_tensor(value) and value.shape[:1] == self._xyz.shape[:1]
        }
        return CapacityBank(
            parameters=parameters,
            accumulators=accumulators,
            stable_ids=self._capacity_stable_ids,
            generation=self._capacity_generation,
            last_reassigned=self._capacity_last_reassigned,
            hard_static_count=int(self.get_static_xyz.shape[0]),
        )

    def capacity_budget_summary(self):
        return capacity_budget_from_bank(self.build_capacity_bank())

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.enable_soft_routing = getattr(training_args, "enable_soft_routing", True)
        self.route_logit_init = getattr(training_args, "route_logit_init", 4.0)
        self.motion_model = getattr(training_args, "motion_model", "poly")
        self.motion_poly_order = getattr(training_args, "motion_poly_order", 2)
        self.motion_lora_rank = getattr(training_args, "motion_lora_rank", 8)
        self.motion_lora_anchors = getattr(training_args, "motion_lora_anchors", 16)
        self.motion_lora_init_scale = getattr(training_args, "motion_lora_init_scale", 0.01)
        self.motion_scaffold_enable = getattr(training_args, "motion_scaffold_enable", False)
        self.motion_scaffold_count = getattr(training_args, "motion_scaffold_count", 512)
        self.motion_scaffold_rank = getattr(training_args, "motion_scaffold_rank", 8)
        self.motion_scaffold_anchors = getattr(training_args, "motion_scaffold_anchors", 32)
        self.motion_scaffold_knn = getattr(training_args, "motion_scaffold_knn", 4)
        self.motion_scaffold_init_scale = getattr(training_args, "motion_scaffold_init_scale", 0.01)
        self.motion_scaffold_weight_temp = getattr(training_args, "motion_scaffold_weight_temp", 0.05)
        self.motion_scaffold_chunk = getattr(training_args, "motion_scaffold_chunk", 65536)
        self.motion_track_dt = getattr(training_args, "motion_track_dt", 1.0 / 30.0)
        self.enable_rendered_flow = getattr(training_args, "enable_rendered_flow", False)
        self.enable_motion_aware_densify = getattr(training_args, "enable_motion_aware_densify", False)
        self.motion_aware_densify_boost = getattr(training_args, "motion_aware_densify_boost", 1.0)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        if self.gaussian_dim == 4:
            self._ensure_route_and_motion_tensors(self.route_logit_init)
            self._ensure_capacity_state(created_iteration=0)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self.gaussian_dim == 4: # TODO: tune time_lr_scale
            if training_args.position_t_lr_init < 0:
                training_args.position_t_lr_init = training_args.position_lr_init
            self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            l.append({'params': [self._t], 'lr': training_args.position_t_lr_init * self.spatial_lr_scale, "name": "t"})
            l.append({'params': [self._scaling_t], 'lr': training_args.scaling_lr, "name": "scaling_t"})
            route_lr = training_args.route_lr if getattr(training_args, "route_lr", -1.0) >= 0 else training_args.feature_lr
            motion_lr = training_args.motion_lr_init if getattr(training_args, "motion_lr_init", -1.0) >= 0 else training_args.position_lr_init * self.spatial_lr_scale
            motion_lora_coeff_lr = training_args.motion_lora_coeff_lr if getattr(training_args, "motion_lora_coeff_lr", -1.0) >= 0 else motion_lr
            motion_lora_basis_lr = training_args.motion_lora_basis_lr if getattr(training_args, "motion_lora_basis_lr", -1.0) >= 0 else motion_lr
            motion_scaffold_coeff_lr = training_args.motion_scaffold_coeff_lr if getattr(training_args, "motion_scaffold_coeff_lr", -1.0) >= 0 else motion_lr
            motion_scaffold_basis_lr = training_args.motion_scaffold_basis_lr if getattr(training_args, "motion_scaffold_basis_lr", -1.0) >= 0 else motion_lr
            if self.enable_soft_routing:
                l.append({'params': [self._route_logit], 'lr': route_lr, "name": "route_logit"})
            if self.motion_model == "poly":
                l.append({'params': [self._motion_v], 'lr': motion_lr, "name": "motion_v"})
                l.append({'params': [self._motion_a], 'lr': motion_lr, "name": "motion_a"})
            elif self.motion_model == "lora":
                l.append({'params': [self._motion_lora_coeff], 'lr': motion_lora_coeff_lr, "name": "motion_lora_coeff"})
                l.append({'params': [self._motion_lora_basis], 'lr': motion_lora_basis_lr, "name": "motion_lora_basis"})
            if self.motion_scaffold_enable:
                l.append({'params': [self._motion_scaffold_coeff], 'lr': motion_scaffold_coeff_lr, "name": "motion_scaffold_coeff"})
                l.append({'params': [self._motion_scaffold_basis], 'lr': motion_scaffold_basis_lr, "name": "motion_scaffold_basis"})
            if self.rot_4d:
                l.append({'params': [self._rotation_r], 'lr': training_args.rotation_lr, "name": "rotation_r"})

                # Static sets (if any)
            if self.static_xyz.numel() > 0:
                l.append({'params': [self.static_xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "static_xyz"}),
                l.append({'params': [self.static_features_dc], 'lr': training_args.feature_lr, "name": "static_f_dc"}),
                l.append({'params': [self.static_features_rest], 'lr': training_args.feature_lr / 20.0, "name": "static_f_rest"}),
                l.append({'params': [self.static_opacity], 'lr': training_args.opacity_lr, "name": "static_opacity"}),
                l.append({'params': [self.static_scaling], 'lr': training_args.scaling_lr, "name": "static_scaling"}),
                l.append({'params': [self.static_rotation], 'lr': training_args.rotation_lr, "name": "static_rotation"})

            # Gate params: monotonic logistic on log sigma_t
            if self.logit_a is None:
                self.logit_a = nn.Parameter(
                    torch.tensor(5.0, device="cuda"), requires_grad=True
                )
            if self.logit_b is None:
                self.logit_b = nn.Parameter(
                    torch.tensor(-2.0, device="cuda"), requires_grad=True
                )
            l.append({'params': [self.logit_a, self.logit_b],'lr': training_args.feature_lr,'name': "gate_params",})

            self._staticness_score = torch.zeros(
                (self.get_xyz.shape[0], 1), device="cuda"
            )

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    # ----------------- Opacity / optimizer utils -----------------

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

        if len(self.static_opacity) != 0:
            static_opacities_new = inverse_sigmoid(torch.min(self.get_static_opacity, torch.ones_like(self.get_static_opacity)*0.01))
            optimizable_tensors = self.replace_tensor_to_optimizer(static_opacities_new, "static_opacity")
            self.static_opacity = optimizable_tensors["static_opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask, static=False):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            name = group["name"]

            # Skip global / non-per-point groups
            if name in ("gate_mlp", "gate_params", "motion_lora_basis", "motion_scaffold_coeff", "motion_scaffold_basis"):
                continue

            if static and "static" not in name:
                continue
            elif not static and "static" in name:
                continue

            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                old_param = group["params"][0]
                del self.optimizer.state[old_param]

                new_param = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                group["params"][0] = new_param
                self.optimizer.state[new_param] = stored_state

                optimizable_tensors[name] = new_param
            else:
                new_param = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                group["params"][0] = new_param
                optimizable_tensors[name] = new_param
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        capacity_valid_points_mask = None
        if self.gaussian_dim == 4:
            self._ensure_capacity_state(created_iteration=0)
            capacity_valid_points_mask = valid_points_mask.detach().clone()
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if self.gaussian_dim == 4:
            self._t = optimizable_tensors['t']
            self._scaling_t = optimizable_tensors['scaling_t']
            if "route_logit" in optimizable_tensors:
                self._route_logit = optimizable_tensors["route_logit"]
            if "motion_v" in optimizable_tensors:
                self._motion_v = optimizable_tensors["motion_v"]
            if "motion_a" in optimizable_tensors:
                self._motion_a = optimizable_tensors["motion_a"]
            if "motion_lora_coeff" in optimizable_tensors:
                self._motion_lora_coeff = optimizable_tensors["motion_lora_coeff"]
            if self._motion_scaffold_attach_idx.numel() > 0:
                self._motion_scaffold_attach_idx = self._motion_scaffold_attach_idx[valid_points_mask]
            if self._motion_scaffold_attach_w.numel() > 0:
                self._motion_scaffold_attach_w = self._motion_scaffold_attach_w[valid_points_mask]
            if self.rot_4d:
                self._rotation_r = optimizable_tensors['rotation_r']
            self.t_gradient_accum = self.t_gradient_accum[valid_points_mask]
            if self._staticness_score.numel() > 0:
                self._staticness_score = self._staticness_score[valid_points_mask]
            self._prune_capacity_state(capacity_valid_points_mask)

        # CSVL-VPL v2 row-alignment hook: slice lifecycle state with the surviving
        # rows. Runs for every prune path (densify prune, split parent removal,
        # dynamic->static conversion).
        lifecycle = getattr(self, "lifecycle", None)
        if lifecycle is not None:
            lifecycle.on_rows_pruned(valid_points_mask)

        # EL-GS row alignment: slice the family-id column and report the
        # per-family removals to the registry (auditable row counts).
        if getattr(self, "elgs_runtime", None) is not None:
            removed = self._elgs_family_ids[~valid_points_mask.to(self._elgs_family_ids.device)]
            self._elgs_family_ids = self._elgs_family_ids[valid_points_mask.to(self._elgs_family_ids.device)]
            for family_id, count in zip(*[t.tolist() for t in removed.unique(return_counts=True)]):
                if family_id >= 0:
                    self.elgs_runtime.registry.on_rows_pruned(int(family_id), int(count))

    def prune_static_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask, static=True)

        self.static_xyz = optimizable_tensors["static_xyz"]
        self.static_features_dc = optimizable_tensors["static_f_dc"]
        self.static_features_rest = optimizable_tensors["static_f_rest"]
        self.static_opacity = optimizable_tensors["static_opacity"]
        self.static_scaling = optimizable_tensors["static_scaling"]
        self.static_rotation = optimizable_tensors["static_rotation"]

        self.static_xyz_gradient_accum = self.static_xyz_gradient_accum[valid_points_mask]
        self.static_denom = self.static_denom[valid_points_mask]
        self.static_max_radii2D = self.static_max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            name = group["name"]
            if name in ("gate_mlp", "gate_params", "motion_lora_basis", "motion_scaffold_coeff", "motion_scaffold_basis"):
                # Global params; no concatenation
                continue

            assert (
                    len(group["params"]) == 1
            ), f"Group {name} has more than one param"

            try:
                extension_tensor = tensors_dict[name]
            except KeyError:
                continue

            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (
                        stored_state["exp_avg"],
                        torch.zeros_like(extension_tensor),
                    ),
                    dim=0,
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (
                        stored_state["exp_avg_sq"],
                        torch.zeros_like(extension_tensor),
                    ),
                    dim=0,
                )

                old_param = group["params"][0]
                del self.optimizer.state[old_param]

                new_param = nn.Parameter(
                    torch.cat(
                        (old_param, extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                group["params"][0] = new_param
                self.optimizer.state[new_param] = stored_state

                optimizable_tensors[name] = new_param
            else:
                new_param = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                group["params"][0] = new_param
                optimizable_tensors[name] = new_param
        return optimizable_tensors

    # ----------------- Densification -----------------

    def _limit_densify_selection(self, selected_pts_mask, scores, max_selected):
        if max_selected is None or max_selected < 0:
            return selected_pts_mask
        max_selected = int(max_selected)
        if max_selected <= 0:
            return torch.zeros_like(selected_pts_mask, dtype=torch.bool)

        selected_count = int(selected_pts_mask.sum().item())
        if selected_count <= max_selected:
            return selected_pts_mask

        selected_idx = selected_pts_mask.nonzero(as_tuple=False).squeeze(-1)
        selected_scores = scores.reshape(-1)[selected_idx]
        keep_local = torch.topk(selected_scores, k=max_selected, largest=True).indices
        limited = torch.zeros_like(selected_pts_mask, dtype=torch.bool)
        limited[selected_idx[keep_local]] = True
        return limited

    def densification_postfix(
            self,
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_t,
            new_scaling_t,
            new_rotation_r,
            new_route_logit=None,
            new_motion_v=None,
            new_motion_a=None,
            new_motion_lora_coeff=None,
            new_motion_scaffold_attach_idx=None,
            new_motion_scaffold_attach_w=None,
            created_iteration=None,
            elgs_source_indices=None,
    ):
        if self.gaussian_dim == 4:
            self._ensure_capacity_state(created_iteration=0)
        old_motion_scaffold_attach_idx = self._motion_scaffold_attach_idx
        old_motion_scaffold_attach_w = self._motion_scaffold_attach_w
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        if self.gaussian_dim == 4:
            d["t"] = new_t
            d["scaling_t"] = new_scaling_t
            if self.enable_soft_routing and new_xyz.shape[0] > 0:
                d["route_logit"] = (
                    new_route_logit
                    if new_route_logit is not None
                    else self._route_logit.new_full((new_xyz.shape[0], 1), self.route_logit_init)
                )
            if self.motion_model == "poly" and new_xyz.shape[0] > 0:
                d["motion_v"] = (
                    new_motion_v
                    if new_motion_v is not None
                    else self._motion_v.new_zeros((new_xyz.shape[0], 3))
                )
                d["motion_a"] = (
                    new_motion_a
                    if new_motion_a is not None
                    else self._motion_a.new_zeros((new_xyz.shape[0], 3))
                )
            if self.motion_model == "lora" and new_xyz.shape[0] > 0:
                d["motion_lora_coeff"] = (
                    new_motion_lora_coeff
                    if new_motion_lora_coeff is not None
                    else self._motion_lora_coeff.new_zeros((new_xyz.shape[0], self.motion_lora_rank))
                )
            if self.rot_4d:
                d["rotation_r"] = new_rotation_r

        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.gaussian_dim == 4:
            self._t = optimizable_tensors["t"]
            self._scaling_t = optimizable_tensors["scaling_t"]
            if "route_logit" in optimizable_tensors:
                self._route_logit = optimizable_tensors["route_logit"]
            if "motion_v" in optimizable_tensors:
                self._motion_v = optimizable_tensors["motion_v"]
            if "motion_a" in optimizable_tensors:
                self._motion_a = optimizable_tensors["motion_a"]
            if "motion_lora_coeff" in optimizable_tensors:
                self._motion_lora_coeff = optimizable_tensors["motion_lora_coeff"]
            if self.motion_scaffold_enable and new_xyz.shape[0] > 0:
                if new_motion_scaffold_attach_idx is None or new_motion_scaffold_attach_w is None:
                    new_motion_scaffold_attach_idx, new_motion_scaffold_attach_w = self._compute_scaffold_attachments(new_xyz.detach())
                if old_motion_scaffold_attach_idx.numel() == 0:
                    old_motion_scaffold_attach_idx, old_motion_scaffold_attach_w = self._compute_scaffold_attachments(
                        self._xyz[: self._xyz.shape[0] - new_xyz.shape[0]].detach()
                    )
                self._motion_scaffold_attach_idx = torch.cat(
                    [old_motion_scaffold_attach_idx.to(new_xyz.device).long(), new_motion_scaffold_attach_idx.to(new_xyz.device).long()],
                    dim=0,
                )
                self._motion_scaffold_attach_w = torch.cat(
                    [old_motion_scaffold_attach_w.to(new_xyz.device), new_motion_scaffold_attach_w.to(new_xyz.device)],
                    dim=0,
                )
            if self.rot_4d:
                self._rotation_r = optimizable_tensors["rotation_r"]
            self.t_gradient_accum = torch.zeros(
                (self.get_xyz.shape[0], 1), device="cuda"
            )
            # Extend staticness score for new points (init 0)
            if self._staticness_score.numel() > 0:
                self._staticness_score = torch.cat(
                    [
                        self._staticness_score,
                        torch.zeros(
                            (new_xyz.shape[0], 1), device="cuda"
                        ),
                    ],
                    dim=0,
                )
            else:
                self._staticness_score = torch.zeros(
                    (self.get_xyz.shape[0], 1), device="cuda"
                )

        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device="cuda"
        )
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if self.gaussian_dim == 4:
            self._append_capacity_rows(new_xyz.shape[0], created_iteration)

        # CSVL-VPL v2 row-alignment hook: extend lifecycle state for the appended
        # rows and restart the exposure denominator at the densify round boundary
        # (mirrors the xyz_gradient_accum/denom reset above).
        lifecycle = getattr(self, "lifecycle", None)
        if lifecycle is not None:
            lifecycle.on_rows_added(int(new_xyz.shape[0]), iteration=created_iteration)

        # EL-GS row alignment: appended rows inherit the SOURCE rows'
        # family ids (topology-inheriting clone/split — family-level
        # tying; substrate). Rows appended outside a family context
        # (e.g. lifecycle birth targets) carry -1 = unassigned.
        if getattr(self, "elgs_runtime", None) is not None:
            if elgs_source_indices is not None:
                inherited = self._elgs_family_ids[
                    elgs_source_indices.to(self._elgs_family_ids.device)
                ]
            else:
                inherited = torch.full(
                    (int(new_xyz.shape[0]),), -1,
                    dtype=self._elgs_family_ids.dtype,
                    device=self._elgs_family_ids.device,
                )
            self._elgs_family_ids = torch.cat([self._elgs_family_ids, inherited], dim=0)
            for family_id, count in zip(*[t.tolist() for t in inherited.unique(return_counts=True)]):
                if family_id >= 0:
                    self.elgs_runtime.registry.on_rows_added(int(family_id), int(count))

    def densification_postfix_static(
            self,
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
    ):
        # --- Decide if we need to register static params in the optimizer ---
        need_register = False

        # Case 1: no static points yet
        if self.static_xyz.numel() == 0:
            need_register = True

        # Case 2: defensive check – maybe called before training_setup saw statics
        for g in self.optimizer.param_groups:
            if g.get("name", None) == "static_xyz":
                need_register = False
                break

        if need_register:
            # Initialize empty static tensors with correct ranks
            self.static_xyz = torch.empty((0, new_xyz.shape[1]), device="cuda", dtype=torch.float32, requires_grad=True)
            self.static_features_dc = torch.empty((0, new_features_dc.shape[1], new_features_dc.shape[2]),
                                                  device="cuda", dtype=torch.float32, requires_grad=True)
            self.static_features_rest = torch.empty((0, new_features_rest.shape[1], new_features_rest.shape[2]),
                                                    device="cuda", dtype=torch.float32, requires_grad=True)
            self.static_opacity = torch.empty((0, new_opacities.shape[1]),
                                              device="cuda", dtype=torch.float32, requires_grad=True)
            self.static_scaling = torch.empty((0, new_scaling.shape[1]),
                                              device="cuda", dtype=torch.float32, requires_grad=True)
            self.static_rotation = torch.empty((0, new_rotation.shape[1]),
                                               device="cuda", dtype=torch.float32, requires_grad=True)

            base_lr = self.optimizer.param_groups[0]["lr"]
            self.optimizer.add_param_group({"params": [self.static_xyz],          "lr": base_lr, "name": "static_xyz"})
            self.optimizer.add_param_group({"params": [self.static_features_dc],  "lr": base_lr, "name": "static_f_dc"})
            self.optimizer.add_param_group({"params": [self.static_features_rest],"lr": base_lr, "name": "static_f_rest"})
            self.optimizer.add_param_group({"params": [self.static_opacity],      "lr": base_lr, "name": "static_opacity"})
            self.optimizer.add_param_group({"params": [self.static_scaling],      "lr": base_lr, "name": "static_scaling"})
            self.optimizer.add_param_group({"params": [self.static_rotation],     "lr": base_lr, "name": "static_rotation"})

        # --- Build dictionary for concatenation ---
        d = {
            "static_xyz":      new_xyz,
            "static_f_dc":     new_features_dc,
            "static_f_rest":   new_features_rest,
            "static_opacity":  new_opacities,
            "static_scaling":  new_scaling,
            "static_rotation": new_rotation,
        }

        # --- Merge new tensors into optimizer-managed set ---
        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        # Small sanity check to fail loudly if something is off
        for key in ["static_xyz", "static_f_dc", "static_f_rest",
                    "static_opacity", "static_scaling", "static_rotation"]:
            assert key in optimizable_tensors, f"{key} not found in optimizable_tensors (optimizer groups misconfigured)"

        # --- Re-bind updated references ---
        self.static_xyz = optimizable_tensors["static_xyz"]
        self.static_features_dc = optimizable_tensors["static_f_dc"]
        self.static_features_rest = optimizable_tensors["static_f_rest"]
        self.static_opacity = optimizable_tensors["static_opacity"]
        self.static_scaling = optimizable_tensors["static_scaling"]
        self.static_rotation = optimizable_tensors["static_rotation"]

        # --- Reset accumulators for static points ---
        n_static = self.static_xyz.shape[0]
        self.static_max_radii2D = torch.zeros((n_static,), device="cuda")
        self.static_denom = torch.zeros((n_static, 1), device="cuda")
        self.static_xyz_gradient_accum = torch.zeros((n_static, 1), device="cuda")



    # split/clone methods unchanged (just using helpers above)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, max_new_points=None, iteration=None,
                          protect_mask=None):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = padded_grad >= grad_threshold
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
            )
        # CSVL-VPL v2 protection veto: a split destroys its parent, so persistently
        # occluded rows must never be selected.
        split_protect_mask = _as_row_bool_mask(
            protect_mask, n_init_points, "densify_and_split", device=selected_pts_mask.device
        )
        if split_protect_mask is not None:
            selected_pts_mask = torch.logical_and(selected_pts_mask, ~split_protect_mask)
        selected_pts_mask = self._limit_densify_selection(selected_pts_mask, padded_grad, max_new_points)
        selected_count = int(selected_pts_mask.sum().item())
        if selected_count == 0:
            return 0

        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(
            N, 1, 1
        )
        new_features_rest = self._features_rest[selected_pts_mask].repeat(
            N, 1, 1
        )
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        if not self.rot_4d:
            stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(
                self._rotation[selected_pts_mask]
            ).repeat(N, 1, 1)
            new_xyz = (
                    torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
                    + self.get_xyz[selected_pts_mask].repeat(N, 1)
            )
            new_t = None
            new_scaling_t = None
            new_rotation_r = None
            new_route_logit = None
            new_motion_v = None
            new_motion_a = None
            new_motion_lora_coeff = None
            new_motion_scaffold_attach_idx = None
            new_motion_scaffold_attach_w = None
            if self.gaussian_dim == 4:
                stds_t = self.get_scaling_t[selected_pts_mask].repeat(N, 1)
                means_t = torch.zeros((stds_t.size(0), 1), device="cuda")
                samples_t = torch.normal(mean=means_t, std=stds_t)
                new_t = (
                        samples_t
                        + self.get_t[selected_pts_mask].repeat(N, 1)
                )
                new_scaling_t = self.scaling_inverse_activation(
                    self.get_scaling_t[selected_pts_mask].repeat(N, 1)
                    / (0.8 * N)
                )
                if self.enable_soft_routing:
                    new_route_logit = self._route_logit[selected_pts_mask].repeat(N, 1)
                if self.motion_model == "poly":
                    new_motion_v = self._motion_v[selected_pts_mask].repeat(N, 1)
                    new_motion_a = self._motion_a[selected_pts_mask].repeat(N, 1)
                elif self.motion_model == "lora":
                    new_motion_lora_coeff = self._motion_lora_coeff[selected_pts_mask].repeat(N, 1)
                if self.motion_scaffold_enable and self._motion_scaffold_attach_idx.numel() > 0:
                    new_motion_scaffold_attach_idx = self._motion_scaffold_attach_idx[selected_pts_mask].repeat(N, 1)
                    new_motion_scaffold_attach_w = self._motion_scaffold_attach_w[selected_pts_mask].repeat(N, 1)
        else:
            stds = self.get_scaling_xyzt[selected_pts_mask].repeat(N, 1)
            means = torch.zeros((stds.size(0), 4), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation_4d(
                self._rotation[selected_pts_mask],
                self._rotation_r[selected_pts_mask],
            ).repeat(N, 1, 1)
            new_xyzt = (
                    torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
                    + self.get_xyzt[selected_pts_mask].repeat(N, 1)
            )

            new_xyz = new_xyzt[..., 0:3]
            new_scaling_t = self.scaling_inverse_activation(
                self.get_scaling_t[selected_pts_mask].repeat(N, 1)
                / (0.8 * N)
            )
            new_t = new_xyzt[..., 3:4]
            new_rotation_r = self._rotation_r[selected_pts_mask].repeat(
                N, 1
            )
            new_route_logit = None
            new_motion_v = None
            new_motion_a = None
            new_motion_lora_coeff = None
            new_motion_scaffold_attach_idx = None
            new_motion_scaffold_attach_w = None
            if self.enable_soft_routing:
                new_route_logit = self._route_logit[selected_pts_mask].repeat(N, 1)
            if self.motion_model == "poly":
                new_motion_v = self._motion_v[selected_pts_mask].repeat(N, 1)
                new_motion_a = self._motion_a[selected_pts_mask].repeat(N, 1)
            elif self.motion_model == "lora":
                new_motion_lora_coeff = self._motion_lora_coeff[selected_pts_mask].repeat(N, 1)
            if self.motion_scaffold_enable and self._motion_scaffold_attach_idx.numel() > 0:
                new_motion_scaffold_attach_idx = self._motion_scaffold_attach_idx[selected_pts_mask].repeat(N, 1)
                new_motion_scaffold_attach_w = self._motion_scaffold_attach_w[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_t,
            new_scaling_t,
            new_rotation_r,
            new_route_logit,
            new_motion_v,
            new_motion_a,
            new_motion_lora_coeff,
            new_motion_scaffold_attach_idx,
            new_motion_scaffold_attach_w,
            created_iteration=iteration,
            elgs_source_indices=torch.where(selected_pts_mask)[0].repeat(N),
        )

        parent_removal_mask = selected_pts_mask
        if split_protect_mask is not None:
            # Redundant by construction (the veto was applied before selection);
            # kept explicit so the parent-removal path can never delete a
            # protected row if the selection logic changes.
            parent_removal_mask = torch.logical_and(parent_removal_mask, ~split_protect_mask)
        prune_filter = torch.cat(
            (
                parent_removal_mask,
                torch.zeros(
                    N * selected_pts_mask.sum(),
                    device="cuda",
                    dtype=bool,
                    ),
            )
        )
        self.prune_points(prune_filter)
        return selected_count

    def densify_and_clone(self, grads, grad_threshold, scene_extent, max_new_points=None, iteration=None):
        grad_norm = torch.norm(grads, dim=-1)
        selected_pts_mask = grad_norm >= grad_threshold
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
            )
        selected_pts_mask = self._limit_densify_selection(selected_pts_mask, grad_norm, max_new_points)
        selected_count = int(selected_pts_mask.sum().item())
        if selected_count == 0:
            return 0

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_t = None
        new_scaling_t = None
        new_rotation_r = None
        new_route_logit = None
        new_motion_v = None
        new_motion_a = None
        new_motion_lora_coeff = None
        new_motion_scaffold_attach_idx = None
        new_motion_scaffold_attach_w = None
        if self.gaussian_dim == 4:
            new_t = self._t[selected_pts_mask]
            new_scaling_t = self._scaling_t[selected_pts_mask]
            if self.enable_soft_routing:
                new_route_logit = self._route_logit[selected_pts_mask]
            if self.motion_model == "poly":
                new_motion_v = self._motion_v[selected_pts_mask]
                new_motion_a = self._motion_a[selected_pts_mask]
            elif self.motion_model == "lora":
                new_motion_lora_coeff = self._motion_lora_coeff[selected_pts_mask]
            if self.motion_scaffold_enable and self._motion_scaffold_attach_idx.numel() > 0:
                new_motion_scaffold_attach_idx = self._motion_scaffold_attach_idx[selected_pts_mask]
                new_motion_scaffold_attach_w = self._motion_scaffold_attach_w[selected_pts_mask]
            if self.rot_4d:
                new_rotation_r = self._rotation_r[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_t,
            new_scaling_t,
            new_rotation_r,
            new_route_logit,
            new_motion_v,
            new_motion_a,
            new_motion_lora_coeff,
            new_motion_scaffold_attach_idx,
            new_motion_scaffold_attach_w,
            created_iteration=iteration,
            elgs_source_indices=torch.where(selected_pts_mask)[0],
        )
        return selected_count

    def densify_and_split_static(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_static_xyz.shape[0]
        if n_init_points == 0:
            return
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = padded_grad >= grad_threshold
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_static_scaling, dim=1).values
            > self.percent_dense * scene_extent,
            )

        new_scaling = self.scaling_inverse_activation(
            self.get_static_scaling[selected_pts_mask].repeat(N, 1)
            / (0.8 * N)
        )
        new_rotation = self.static_rotation[selected_pts_mask].repeat(
            N, 1
        )
        new_features_dc = self.static_features_dc[
            selected_pts_mask
        ].repeat(N, 1, 1)
        new_features_rest = self.static_features_rest[
            selected_pts_mask
        ].repeat(N, 1, 1)
        new_opacity = self.static_opacity[selected_pts_mask].repeat(N, 1)

        stds = self.get_static_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(
            self.static_rotation[selected_pts_mask]
        ).repeat(N, 1, 1)
        new_xyz = (
                torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
                + self.get_static_xyz[selected_pts_mask].repeat(N, 1)
        )

        self.densification_postfix_static(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(
                    N * selected_pts_mask.sum(),
                    device="cuda",
                    dtype=bool,
                    ),
            )
        )
        self.prune_static_points(prune_filter)

    def densify_and_clone_static(self, grads, grad_threshold, scene_extent):
        n_init_points = self.get_static_xyz.shape[0]
        if n_init_points == 0:
            return
        selected_pts_mask = torch.norm(grads, dim=-1) >= grad_threshold
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_static_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
            )

        new_xyz = self.static_xyz[selected_pts_mask]
        new_features_dc = self.static_features_dc[selected_pts_mask]
        new_features_rest = self.static_features_rest[selected_pts_mask]
        new_opacities = self.static_opacity[selected_pts_mask]
        new_scaling = self.static_scaling[selected_pts_mask]
        new_rotation = self.static_rotation[selected_pts_mask]

        self.densification_postfix_static(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

    def densify_and_prune(
            self,
            max_grad,
            min_opacity,
            extent,
            max_screen_size,
            max_grad_t=None,
            static_conversion_threshold=0.99,
            gate_activation_iter=None,
            gate_warmup_until_iter=None,
            iteration=None,
            enable_hard_static_conversion=False,
            max_total_points=-1,
            *,
            exposure_denominator=None,
            protect_mask=None,
    ):
        # CSVL-VPL v2: `exposure_denominator` replaces `denom` in the densification
        # gradient normalization and `protect_mask` vetoes destructive operations
        # (prune, split parent removal) on persistently occluded rows. Both are
        # None for every lane that does not enable the lifecycle, which leaves the
        # base behaviour bit-identical.
        def _current_protect_mask(expected_rows, where):
            if protect_mask is None:
                return None
            lifecycle = getattr(self, "lifecycle", None)
            # The lifecycle manager keeps its state row-aligned through the
            # densification_postfix/prune_points hooks, so re-reading it is the
            # only correct way to track the mask across intra-call row changes.
            source = protect_mask if lifecycle is None else lifecycle.protected_persistent()
            return _as_row_bool_mask(source, expected_rows, where, device=self._xyz.device)

        # 1) Snapshot temporal gradients BEFORE we touch the point set
        avg_t_grad_snapshot = None
        if self.gaussian_dim == 4 and self.t_gradient_accum.numel() > 0:
            avg_t_grad_snapshot = self.t_gradient_accum / self.denom.clamp_min(1.0)
            avg_t_grad_snapshot = torch.abs(avg_t_grad_snapshot)
            avg_t_grad_snapshot = torch.nan_to_num(avg_t_grad_snapshot)

        # 2) Gated dynamic → static conversion (before densify/prune)
        if (
                enable_hard_static_conversion
                and
                static_conversion_threshold is not None
                and iteration is not None
                and gate_activation_iter is not None
                and gate_warmup_until_iter is not None
                and iteration >= max(gate_activation_iter, gate_warmup_until_iter)
                and self._staticness_score.numel() > 0
        ):
            if avg_t_grad_snapshot is not None:
                with torch.no_grad():
                    if avg_t_grad_snapshot.numel() > 0:
                        scale = torch.quantile(avg_t_grad_snapshot, 0.9).clamp_min(1e-6)
                    else:
                        scale = torch.tensor(1.0, device=self._xyz.device)

                    dynamicness = (avg_t_grad_snapshot / scale).clamp(0.0, 1.0)
                    # High dynamicness -> reduce staticness score before thresholding
                    self._staticness_score = self._staticness_score * (1.0 - dynamicness)

            # thresholding with t-grad veto
            self.dynamic2static(
                static_conversion_threshold,
                t_grad_quantile=0.5,
                avg_t_grad_snapshot=avg_t_grad_snapshot,
            )

        # 3) NOW compute grads for densification on the updated dynamic set
        densify_denominator = self.denom
        if exposure_denominator is not None:
            densify_denominator = _as_row_denominator(
                exposure_denominator,
                self.xyz_gradient_accum.shape[0],
                "densify_and_prune exposure denominator",
                device=self.xyz_gradient_accum.device,
            )
            # Mean-preserving renormalization: exposure weights are <= 1, so a
            # raw exposure denominator would inflate every densification score
            # and turn the mechanism into a one-sided capacity amplifier,
            # breaking the capacity-matched control lanes. Rescale so the
            # total denominator mass over observed rows equals the baseline
            # denominator mass: exposure then REALLOCATES priority toward
            # under-exposed primitives without changing the average score.
            # Under all-ones weights the scale is exactly 1 and the baseline
            # is recovered bit-for-bit up to float rounding.
            observed = (self.denom > 0).reshape(-1)
            if observed.any():
                base_mass = self.denom.reshape(-1)[observed].sum()
                exposure_mass = densify_denominator.reshape(-1)[observed].sum().clamp_min(1e-12)
                densify_denominator = densify_denominator * (base_mass / exposure_mass)
        grads = self.xyz_gradient_accum / densify_denominator.clamp_min(1.0)
        grads[grads.isnan()] = 0.0

        # CSVL-VPL v2 protection veto on the densification score. A persistently
        # occluded row is frozen (its gradient rows are zeroed every iteration),
        # so it must not spend densification budget either. This also removes the
        # exposure floor artifact: `exposure.clamp(min=1)` turns "no exposed
        # observation this round" into a denominator of one, which would turn the
        # raw accumulated sum of a fully hidden row into a large densification
        # score. Only active when protection is on (protect_mask is not None).
        densify_protect_mask = _current_protect_mask(
            int(grads.shape[0]), "densify_and_prune densification-score veto"
        )
        if densify_protect_mask is not None:
            grads[densify_protect_mask] = 0.0

        # 4) Densify dynamic gaussians
        remaining_new_points = None
        if max_total_points is not None and max_total_points >= 0:
            remaining_new_points = max(0, int(max_total_points) - int(self.get_xyz.shape[0]))

        cloned_points = self.densify_and_clone(
            grads,
            max_grad,
            extent,
            max_new_points=remaining_new_points,
            iteration=iteration,
        )
        if remaining_new_points is not None:
            remaining_new_points = max(0, remaining_new_points - cloned_points)
        self.densify_and_split(
            grads,
            max_grad,
            extent,
            max_new_points=remaining_new_points,
            iteration=iteration,
            protect_mask=_current_protect_mask(
                int(self.get_xyz.shape[0]), "densify_and_prune split veto"
            ),
        )

        # 5) Densify static gaussians (unchanged)
        if len(self.static_xyz) != 0:
            static_grads = self.static_xyz_gradient_accum / self.static_denom.clamp_min(1.0)
            static_grads[static_grads.isnan()] = 0.0
            self.densify_and_clone_static(static_grads, max_grad, extent)
            self.densify_and_split_static(static_grads, max_grad, extent)

        # 6) Prune dynamic
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        prune_protect_mask = _current_protect_mask(
            int(self.get_xyz.shape[0]), "densify_and_prune prune veto"
        )
        if prune_protect_mask is not None:
            prune_mask = torch.logical_and(prune_mask, ~prune_protect_mask)
        self.prune_points(prune_mask)

        # 7) Prune static
        if len(self.static_xyz) != 0:
            prune_static_mask = (self.get_static_opacity < min_opacity).squeeze()
            if max_screen_size:
                big_points_vs_static = self.static_max_radii2D > max_screen_size
                big_points_ws_static = self.get_static_scaling.max(dim=1).values > 0.1 * extent
                prune_static_mask = torch.logical_or(
                    torch.logical_or(prune_static_mask, big_points_vs_static),
                    big_points_ws_static,
                )
            self.prune_static_points(prune_static_mask)

        torch.cuda.empty_cache()

    # ----------------- Densification stats accumulation -----------------

    def _apply_motion_aware_densify_boost(self, grad, dynamic_weight, update_filter):
        if (
                not self.enable_motion_aware_densify
                or dynamic_weight is None
                or dynamic_weight.numel() == 0
        ):
            return grad
        weight = dynamic_weight.to(grad.device, grad.dtype)
        if weight.dim() == 1:
            weight = weight.unsqueeze(-1)
        return grad * (1.0 + float(self.motion_aware_densify_boost) * weight[update_filter].clamp(0.0, 1.0))

    def add_densification_stats(
            self, viewspace_point_tensor, update_filter, avg_t_grad=None, dynamic_weight=None
    ):
        grad = torch.norm(
            viewspace_point_tensor.grad[update_filter, :2],
            dim=-1,
            keepdim=True,
        )
        grad = self._apply_motion_aware_densify_boost(grad, dynamic_weight, update_filter)
        self.xyz_gradient_accum[update_filter] += grad
        self.denom[update_filter] += 1
        if self.gaussian_dim == 4 and avg_t_grad is not None:
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter]

    def add_densification_stats_grad(
            self, viewspace_point_grad, update_filter, avg_t_grad=None, dynamic_weight=None
    ):
        grad = viewspace_point_grad[update_filter]
        grad = self._apply_motion_aware_densify_boost(grad, dynamic_weight, update_filter)
        self.xyz_gradient_accum[update_filter] += grad
        self.denom[update_filter] += 1
        if self.gaussian_dim == 4 and avg_t_grad is not None:
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter]

    def add_densification_stats_grad_static(
            self, viewspace_point_grad, update_filter
    ):
        self.static_xyz_gradient_accum[update_filter] += (
            viewspace_point_grad[update_filter]
        )
        self.static_denom[update_filter] += 1

    # ----------------- Dynamic → Static -----------------
    def dynamic2static(self, conversion_threshold, t_grad_quantile=0.5, avg_t_grad_snapshot=None):
        """
        Move 'static-enough' Gaussians to the static set.

        conversion_threshold: threshold on staticness_score (s).
        t_grad_quantile: among candidates with s>threshold, only convert those
                         whose avg t-grad is below this quantile (0.5 = median).
        """
        if self._staticness_score.numel() == 0:
            self.num_static_candidates_last = 0
            self.num_converted_last = 0
            return
        device = self._xyz.device
        # 1) σ_t-based static candidates (same as before)
        candidates_mask = (self._staticness_score > conversion_threshold).squeeze().to(device)
        if candidates_mask.dim() == 0:
            candidates_mask = candidates_mask.unsqueeze(0)

        num_candidates = int(candidates_mask.sum().item())
        self.num_static_candidates_last = num_candidates

        if num_candidates == 0:
            self.num_converted_last = 0
            return

        # 2) Optional t-grad veto: only keep low-motion candidates
        final_mask = candidates_mask
        if (
                self.gaussian_dim == 4
                and avg_t_grad_snapshot is not None
                and t_grad_quantile is not None
        ):
            avg_t_grad = avg_t_grad_snapshot

            candidate_vals = avg_t_grad[candidates_mask]
            if candidate_vals.numel() > 0:
                # e.g. 0.5 ⇒ median t-grad among candidates
                thr = torch.quantile(candidate_vals.detach(), t_grad_quantile)
                low_motion = avg_t_grad <= thr
                final_mask = candidates_mask & low_motion.squeeze()

        num_converted = int(final_mask.sum().item())
        self.num_converted_last = num_converted

        if num_converted == 0:
            return

        # Below is same as old code except final_mask is used instead of static_mask

        # static_mask = (
        #         self._staticness_score > conversion_threshold
        # ).squeeze().to(self._xyz.device)
        # if static_mask.sum() == 0:
        #     return
        static_mask = final_mask

        new_xyz = self._xyz[static_mask]
        new_features_dc = self._features_dc[static_mask]
        new_features_rest = self._features_rest[static_mask]
        new_opacities = self._opacity[static_mask]
        new_scaling = self._scaling[static_mask]

        if self.rot_4d:
            r_4d = build_rotation_4d(
                self._rotation[static_mask],
                self._rotation_r[static_mask],
            )
            R = r_4d[:, :3, :3]
            R = torch.nan_to_num(R)
            U, _, Vh = torch.linalg.svd(R)
            R_clean = U @ Vh
            det = torch.det(R_clean)
            if (det < 0).any():
                flip = (det < 0).nonzero(as_tuple=False).squeeze(-1)
                U[flip, :, -1] *= -1
                R_clean = U @ Vh
            new_rotation = rotation_matrix_to_rotation_3d(R_clean)
        else:
            new_rotation = self._rotation[static_mask]

        # remove from dynamic cloud
        self.prune_points(static_mask)

        # append into static cloud
        self.densification_postfix_static(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )
