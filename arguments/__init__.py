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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.extension = ".png"
        self.num_extra_pts = 0
        self.loaded_pth = ""
        self.frame_ratio = 1
        self.dataloader = False
        self.from3dgs = ""
        self.start_timestamp = 0
        self.end_timestamp = -1
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.env_map_res = 0
        self.env_optimize_until = 1000000000
        self.env_optimize_from = 0
        self.eval_shfs_4d = False
        self.opa_threshold = 0.05
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_t_lr_init = -1.0
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.thresh_opa_prune = 0.005
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.densify_grad_t_threshold = 0.0002 / 40
        self.densify_until_num_points = -1
        self.final_prune_from_iter = -1
        self.sh_increase_interval = 1000
        self.lambda_opa_mask = 0.0
        self.lambda_rigid = 0.0
        self.lambda_motion = 0.0

        self.enable_hard_static_conversion = False
        self.static_conversion_threshold = 0.99
        self.lambda_gate_sparsity = 0.0

        self.gate_activation_iter = 500
        self.gate_warmup_until_iter = 1500

        self.lambda_sparsity = 0.0
        self.lambda_motion_gate = 0.0
        self.motion_gate_quantile = 0.8

        self.enable_soft_routing = True
        self.route_logit_init = 4.0
        self.route_lr = -1.0

        self.motion_model = "poly"
        self.motion_poly_order = 2
        self.motion_lr_init = -1.0
        self.motion_reg_lambda = 0.0
        self.motion_lora_rank = 8
        self.motion_lora_anchors = 16
        self.motion_lora_init_scale = 0.01
        self.motion_lora_coeff_lr = -1.0
        self.motion_lora_basis_lr = -1.0

        self.motion_prior_root = ""
        self.dynamic_mask_from_residual = False
        self.dynamic_mask_residual_quantile = 0.85
        self.dynamic_mask_dilate = 2
        self.event_candidate_manifest = ""
        self.event_candidate_scene = ""
        self.event_candidate_dilate = 0
        self.event_boundary_support_manifest = ""
        self.event_boundary_scene = ""
        self.event_boundary_dilate = 0
        self.event_boundary_replace_dynamic_mask = False
        self.event_boundary_frame_fallback = False
        self.visibility_event_manifest = ""
        self.visibility_event_scene = ""
        self.visibility_event_opacity_attenuation = 0.85
        self.visibility_event_dynamic_probability_min = 0.55
        self.visibility_event_beta = 1.0
        self.visibility_event_start_iter = 0
        self.visibility_event_warmup_iters = 0
        self.slice_b_capacity_mode = "disabled"
        self.slice_b_capacity_iteration = 5001
        self.slice_b_capacity_k = 0
        self.slice_b_capacity_seed = 0
        self.slice_b_capacity_sidecar = ""
        self.lambda_dynamic_roi = 0.0
        self.lambda_static_exclusion = 0.0
        self.lambda_track_flow = 0.0
        self.lambda_scaffold_smooth = 0.0
        self.lambda_scaffold_reg = 0.0

        self.motion_scaffold_enable = False
        self.motion_scaffold_count = 512
        self.motion_scaffold_rank = 8
        self.motion_scaffold_anchors = 32
        self.motion_scaffold_knn = 4
        self.motion_scaffold_init_scale = 0.01
        self.motion_scaffold_weight_temp = 0.05
        self.motion_scaffold_chunk = 65536
        self.motion_scaffold_coeff_lr = -1.0
        self.motion_scaffold_basis_lr = -1.0
        self.motion_track_dt = 0.0333333333
        self.enable_rendered_flow = False
        self.enable_motion_aware_densify = False
        self.motion_aware_densify_boost = 1.0
        self.motion_aware_densify_use_residual = True

        self.blur_until_iter = 0       # Set > 0 to enable blurring
        self.blur_start_sigma = 8.0    # Initial sigma for the blur kernel
        self.histogram_log_interval = 1

        # ---------------- CSVL-VPL v2 visibility-conditioned lifecycle ----------------
        # Master switch plus the evidence-runtime knobs (EvidenceRuntime, Component 2).
        self.lifecycle_enable = False
        self.lifecycle_evidence_dir = ""
        self.lifecycle_time_shift = 0
        self.lifecycle_tau_rel = 0.03
        self.lifecycle_kappa = 2.5
        self.lifecycle_k_gap = 3.0
        self.lifecycle_sigma_abstain = 0.20
        self.lifecycle_evidence_preload = True

        # LifecycleConfig mirror: every field below is read by
        # scene.lifecycle.LifecycleConfig.from_namespace(opt, prefix="lifecycle_")
        # and MUST keep the `lifecycle_<field name>` spelling.
        self.lifecycle_evidence_mode = "e1"
        self.lifecycle_protection = False
        self.lifecycle_exposure = False
        self.lifecycle_birth = False
        self.lifecycle_birth_mode = "e2"
        self.lifecycle_occ_exposure_weight = 0.0
        self.lifecycle_weak_exposure_weight = 0.5
        self.lifecycle_protect_ema_beta = 0.98
        self.lifecycle_protect_ema_threshold = 0.6
        self.lifecycle_birth_interval = 500
        self.lifecycle_birth_k = 256
        self.lifecycle_birth_warmup = 1500
        self.lifecycle_birth_until = 5500
        self.lifecycle_birth_min_deficit = 64
        self.lifecycle_deficit_cell_rel = 0.01
        self.lifecycle_birth_view_checks = 3
        self.lifecycle_birth_view_min_near = 2
        self.lifecycle_backproject_stride = 6
        self.lifecycle_backproject_sigma_max = 0.12
        self.lifecycle_ledger_interval = 100

        # ---------------- CCR B1 observation-born packet birth ----------------
        # scene.packet_birth.PacketBirthConfig.from_namespace(opt,
        # prefix="packet_birth_") reads every field below and MUST keep the
        # `packet_birth_<field name>` spelling. False leaves the substrate
        # bit-identical (ccr-method-2026-08-20 §3.1). packet_birth_until_iter
        # -1 means "densify_until_iter".
        self.packet_birth_enable = False
        self.packet_birth_interval = 500
        self.packet_birth_fraction = 0.005
        self.packet_birth_t_sigma_frames = 1.5
        self.packet_birth_from_iter = 1000
        self.packet_birth_until_iter = -1

        # ---------------- EL-GS (evidence-lineage) ----------------
        # Category-1 structural constants and the schedule anchors are
        # NOT tunable here: they load from the hash-recorded prereg
        # files under elgs_prereg_dir (preregistration integrity —
        # research-wiki/operations/elgs-m0-m1-implementation-plan §7).
        # The keys below are the master switch, artifact locations,
        # and category-3 development hyperparameters (tunable only on
        # declared dev scenes; -1.0 sentinel = unset, and the
        # submission wrapper refuses evidence-bearing runs while any
        # required prereg entry is unfrozen).
        self.elgs_enable = False
        self.elgs_prereg_dir = "configs/elgs"
        self.elgs_tracks_dir = ""
        self.elgs_smoke_schedule = False
        # Structural-search master switch, independent of elgs_enable.
        # False keeps the lineage substrate fully live (seeding, the
        # per-family a-logit optimizer group, the presence multiplier
        # and therefore ordinary photometric training and
        # densification) while running NO structural round and
        # proposing NO candidate, so a K=1 presence-only cell needs
        # neither track nor evidence artifacts. True is today's
        # behaviour exactly.
        # SET IT FROM YAML, NEVER FROM THE CLI: ParamGroup renders every
        # bool as `action="store_true"`, so a default-True flag cannot be
        # turned off on the command line at all (the same is true of
        # enable_soft_routing, motion_aware_densify_use_residual and
        # lifecycle_evidence_preload -- a pre-existing repo-wide property,
        # not one this flag introduces). `--elgs_rounds_enabled false`
        # would leave rounds ON and produce a scientifically wrong cell
        # rather than a failed one. The YAML route is exercised by
        # tests/test_elgs_configs.py::RoundsFlagPlumbingTests.
        self.elgs_rounds_enabled = True
        self.elgs_beta = -1.0
        self.elgs_kappa = -1.0
        self.elgs_chi = -1.0
        self.elgs_mu = -1.0
        self.elgs_tau_b = -1.0
        self.elgs_k_se = -1.0
        self.elgs_lambda_u = -1.0
        self.elgs_c_cap = -1
        self.elgs_candidate_cap = -1
        self.elgs_confirmation_samples = -1
        self.elgs_a_lr = -1.0
        self.elgs_r_site = -1.0
        self.elgs_binding_threshold = -1.0
        # SMOKE-ONLY evaluation bound (0 = disabled = full-report
        # semantics). NOT a production report cap: it truncates what
        # the evidence estimator sees, so elgs.evidence_stack
        # .resolve_smoke_report_cap refuses it unless the run both
        # sets elgs_smoke_schedule and is stamped evidence_bearing
        # false.
        self.elgs_smoke_max_reports_per_window = 0
        # MATCHED-COMPARISON PARITY, not an EL-GS feature: build the
        # stratified reserved confirmation pool and drop it from the
        # training data WITHOUT enabling any other EL-GS state, so a
        # temporal-substrate control trains on the same ~75% of the same
        # units an EL-GS cell trains on. A no-op when elgs_enable is set
        # (that path already drops them). See
        # research-wiki/operations/elgs-matched-triple-readiness-2026-08-18.md §2.
        self.elgs_reserved_parity = False
        # EXPLORATORY oracle-episode supply: path to a JSON file that gives a
        # spatial region and the interior absence gaps of a fixed K >= 2
        # episode program. Families seeded from cells that intersect the
        # region get that program; every other family keeps the preregistered
        # K=1 spanning one. Empty (the default) leaves seeding exactly as
        # preregistered. A str because ParamGroup renders bool as
        # `store_true` (so a bool could never be turned off from the CLI) and
        # neither int nor float can carry the boundaries.
        self.elgs_oracle_episodes = ""
        # EXPLORATORY localized presence: when true (YAML-only, like the
        # other EL-GS bools), EL-GS presence applies ONLY to rows whose
        # family carries a K >= 2 episodic program, membership is
        # PER-PRIMITIVE (only rows inside the oracle region join the
        # family; every other row stays unassigned at -1), presence
        # multiplies the TOTAL routed contribution (dynamic AND static
        # branches), and every non-gated row keeps the ordinary learnable
        # temporal marginal. Requires elgs_oracle_episodes. False leaves
        # the global-replacement semantics of the LRV3 A1/A2 cells
        # byte-identical.
        self.elgs_local_presence = False
        # Routing pins zero the route-logit gradient of K > 1 families'
        # rows (substrate). The corrected localized cell disables them so
        # the gated rows' static/dynamic mixture can learn like every
        # other row's. True preserves the historical behaviour.
        self.elgs_routing_pins_enabled = True

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
