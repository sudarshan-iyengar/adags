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

import os
import random
import torch
from torch import nn
from utils.loss_utils import l1_loss, ssim, msssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, knn
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, easy_cmap
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import make_grid
import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader

from utils.mesh_utils import GaussianExtractor
from utils.render_utils import generate_path, create_videos

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def identity_collate(x):
    return x

def validation(dataset, opt, pipe,checkpoint, gaussian_dim, time_duration, rot_4d, force_sh_3d,
               num_pts, num_pts_ratio):
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=gaussian_dim, time_duration=time_duration,
                              rot_4d=rot_4d, force_sh_3d=force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)

    assert checkpoint, "No checkpoint provided for validation"
    scene = Scene(dataset, gaussians, shuffle=False,num_pts=num_pts, num_pts_ratio=num_pts_ratio, time_duration=time_duration)

    (model_params, first_iter) = torch.load(checkpoint)
    train_dir = os.path.join(dataset.model_path, 'train', "ours_{}".format(first_iter))
    test_dir = os.path.join(dataset.model_path, 'test', "ours_{}".format(first_iter))
    gaussians.restore(model_params, None)
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)

    #########   1. Validation and Rendering ############

    print("export rendered testing images ...")
    os.makedirs(test_dir, exist_ok=True)
    gaussExtractor.reconstruction(scene.getTestCameras(),test_dir,stage = "validation")
    gaussExtractor.export_image(test_dir,mode = "validation")

    # #########    2. Render Trajectory       ############

    # print("rendering trajectory ...")
    # traj_dir = os.path.join(test_dir, 'traj')
    # os.makedirs(traj_dir, exist_ok=True)
    # n_fames = 480
    # cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
    # gaussExtractor.reconstruction(cam_traj, test_dir,stage = "trajectory")
    # gaussExtractor.export_image(traj_dir,mode = "trajectory")
    # create_videos( base_dir =traj_dir,
    #                input_dir=traj_dir,
    #                out_name='render_traj',
    #                num_frames=n_fames)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint, debug_from,
             gaussian_dim, time_duration, num_pts, num_pts_ratio, rot_4d, force_sh_3d, batch_size):

    if dataset.frame_ratio > 1:
        time_duration = [time_duration[0] / dataset.frame_ratio,  time_duration[1] / dataset.frame_ratio]

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=gaussian_dim, time_duration=time_duration, rot_4d=rot_4d, force_sh_3d=force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    scene = Scene(dataset, gaussians, num_pts=num_pts, num_pts_ratio=num_pts_ratio, time_duration=time_duration)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    best_psnr = 0.0
    ema_loss_for_log = 0.0
    ema_l1loss_for_log = 0.0
    ema_ssimloss_for_log = 0.0
    lambda_all = [key for key in opt.__dict__.keys() if key.startswith('lambda') and key!='lambda_dssim']
    for lambda_name in lambda_all:
        vars()[f"ema_{lambda_name.replace('lambda_','')}_for_log"] = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    if pipe.env_map_res:
        env_map = nn.Parameter(torch.zeros((3,pipe.env_map_res, pipe.env_map_res),dtype=torch.float, device="cuda").requires_grad_(True))
        env_map_optimizer = torch.optim.Adam([env_map], lr=opt.feature_lr, eps=1e-15)
    else:
        env_map = None

    gaussians.env_map = env_map

    training_dataset = scene.getTrainCameras()
    # training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=12 if dataset.dataloader else 0, collate_fn=lambda x: x, drop_last=True)
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12 if dataset.dataloader else 0,
        collate_fn=identity_collate,
        drop_last=True
    )
    iteration = first_iter

    # Initialize placeholders for new gate losses (for logging continuity)
    Lsparsity = torch.tensor(0.0, device="cuda")
    Lmotion_gate = torch.tensor(0.0, device="cuda")

    while iteration < opt.iterations + 1:
        for batch_data in training_dataloader:
            iteration += 1
            if iteration > opt.iterations:
                break

            iter_start.record()
            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % opt.sh_increase_interval == 0:
                gaussians.oneupSHdegree()

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            # Compute gate inputs / scores (but do not add any legacy gate loss here)
            if iteration >= opt.gate_activation_iter:
                gaussians.compute_differentiable_staticness()
            else:
                if gaussians._xyz.shape[0] > 0:
                    gaussians.differentiable_s = torch.zeros((gaussians._xyz.shape[0], 1), device="cuda", requires_grad=False)

            total_loss = 0.0

            batch_point_grad = []
            batch_visibility_filter = []
            batch_radii = []

            batch_point_grad_static = []
            batch_visibility_filter_static = []
            batch_radii_static = []

            # --------- Inner micro-batch loop (reconstruction + other regularizers) ---------
            for batch_idx in range(batch_size):
                gt_image, viewpoint_cam = batch_data[batch_idx]
                gt_image = gt_image.cuda()
                viewpoint_cam = viewpoint_cam.cuda()

                render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                depth = render_pkg["depth"]
                alpha = render_pkg["alpha"]
                viewspace_point_tensor_static = render_pkg["viewspace_points_static"]
                visibility_filter_static = render_pkg["visibility_filter_static"]
                radii_static = render_pkg["radii_static"]

                # Reconstruction Loss
                Ll1 = l1_loss(image, gt_image)
                Lssim = 1.0 - ssim(image, gt_image)
                loss_recon = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim
                loss = loss_recon

                ###### opa mask Loss ######
                if opt.lambda_opa_mask > 0:
                    o = alpha.clamp(1e-6, 1-1e-6)
                    sky = 1 - viewpoint_cam.gt_alpha_mask
                    Lopa_mask = (- sky * torch.log(1 - o)).mean()
                    lambda_opa_mask = opt.lambda_opa_mask
                    loss = loss + lambda_opa_mask * Lopa_mask
                ###### opa mask Loss ######

                ###### rigid loss ######
                if opt.lambda_rigid > 0:
                    k = 20
                    # cur_time = viewpoint_cam.timestamp
                    # _, delta_mean = gaussians.get_current_covariance_and_mean_offset(1.0, cur_time)
                    xyz_mean = gaussians.get_xyz
                    xyz_cur =  xyz_mean #  + delta_mean
                    idx, dist = knn(xyz_cur[None].contiguous().detach(),
                                    xyz_cur[None].contiguous().detach(),
                                    k)
                    _, velocity = gaussians.get_current_covariance_and_mean_offset(1.0, gaussians.get_t + 0.1)
                    weight = torch.exp(-100 * dist)
                    # cur_marginal_t = gaussians.get_marginal_t(cur_time).detach().squeeze(-1)
                    # marginal_weights = cur_marginal_t[idx] * cur_marginal_t[None,:,None]
                    # weight *= marginal_weights

                    # mean_t, cov_t = gaussians.get_t, gaussians.get_cov_t(scaling_modifier=1)
                    # mean_t_nn, cov_t_nn = mean_t[idx], cov_t[idx]
                    # weight *= torch.exp(-0.5*(mean_t[None, :, None]-mean_t_nn)**2/cov_t[None, :, None]/cov_t_nn*(cov_t[None, :, None]+cov_t_nn)).squeeze(-1).detach()
                    vel_dist = torch.norm(velocity[idx] - velocity[None, :, None], p=2, dim=-1)
                    Lrigid = (weight * vel_dist).sum() / k / xyz_cur.shape[0]
                    loss = loss + opt.lambda_rigid * Lrigid
                ########################

                # ---- NOTE: Step 1 removes legacy motion loss from total training loss ----
                # Keep lambda_motion = 0 for Step 1 (see config). We still compute motion later
                # (once per iteration) for Lmotion_gate, but do NOT add a global Lmotion here.

                total_loss += loss.item()
                (loss / batch_size).backward(retain_graph=True)

                batch_point_grad.append(torch.norm(viewspace_point_tensor.grad[:,:2], dim=-1))
                batch_radii.append(radii)
                batch_visibility_filter.append(visibility_filter)

                static = False
                if len(viewspace_point_tensor_static) > 0:
                    static = True
                if static:
                    batch_point_grad_static.append(torch.norm(viewspace_point_tensor_static.grad[:,:2], dim=-1))
                    batch_radii_static.append(radii_static)
                    batch_visibility_filter_static.append(visibility_filter_static)

            # --------- End inner micro-batch loop ---------

            # Aggregate per-pixel grads across micro-batches (unchanged)
            if batch_size > 1:
                visibility_count = torch.stack(batch_visibility_filter,1).sum(1)
                visibility_filter = visibility_count > 0
                radii = torch.stack(batch_radii,1).max(1)[0]

                batch_viewspace_point_grad = torch.stack(batch_point_grad,1).sum(1)
                batch_viewspace_point_grad[visibility_filter] = batch_viewspace_point_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                batch_viewspace_point_grad = batch_viewspace_point_grad.unsqueeze(1)

                if static:
                    visibility_count_static = torch.stack(batch_visibility_filter_static,1).sum(1)
                    visibility_filter_static = visibility_count_static > 0
                    radii_static = torch.stack(batch_radii_static,1).max(1)[0]

                    batch_viewspace_point_grad_static = torch.stack(batch_point_grad_static,1).sum(1)
                    batch_viewspace_point_grad_static[visibility_filter_static] = batch_viewspace_point_grad_static[visibility_filter_static] * batch_size / visibility_count_static[visibility_filter_static]
                    batch_viewspace_point_grad_static = batch_viewspace_point_grad_static.unsqueeze(1)

                if gaussians.gaussian_dim == 4:
                    batch_t_grad = gaussians._t.grad.clone()[:,0].detach()
                    batch_t_grad[visibility_filter] = batch_t_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                    batch_t_grad = batch_t_grad.unsqueeze(1)

            else:
                if gaussians.gaussian_dim == 4:
                    batch_t_grad = gaussians._t.grad.clone().detach()

            # ---------------------- Step 1: Dueling gate losses (once per iteration) ----------------------
            # Defaults (in case of early iterations or empty set)
            Lsparsity = torch.tensor(0.0, device="cuda")
            Lmotion_gate = torch.tensor(0.0, device="cuda")

            if iteration > opt.gate_activation_iter and gaussians._xyz.shape[0] > 0:
                s = gaussians.differentiable_s
                if s is not None and s.numel() > 0:
                    # Annealed sparsity
                    if opt.gate_warmup_until_iter > opt.gate_activation_iter:
                        annealing = min(1.0, (iteration - opt.gate_activation_iter) / (opt.gate_warmup_until_iter - opt.gate_activation_iter))
                    else:
                        annealing = 1.0
                    lambda_sparsity_eff = opt.lambda_sparsity * annealing
                    Lsparsity = lambda_sparsity_eff * (1.0 - s).mean()

                    # Motion magnitude for each Gaussian (normalized), detached for Step 1
                    with torch.no_grad():
                        _, velocity = gaussians.get_current_covariance_and_mean_offset(1.0, gaussians.get_t + 0.1)  # [N,3]
                        motion_mag = velocity.norm(p=2, dim=1, keepdim=True)                                   # [N,1]
                        scale = torch.quantile(motion_mag, getattr(opt, "motion_gate_quantile", 0.8)).clamp_min(1e-6)
                        Lmotion_per_point = motion_mag / scale

                    Lmotion_gate = opt.lambda_motion_gate * (s * Lmotion_per_point.detach()).mean()

                    # Backprop these gate-only losses (after micro-batch backward)
                    total_loss += (Lsparsity + Lmotion_gate).item()
                    (Lsparsity + Lmotion_gate).backward()

            iter_end.record()

            # ---------------------- Logging ----------------------
            # Build loss_dict with available pieces (avoid referencing undefined names)
            loss_dict = {"Ll1": Ll1,
                         "Lssim": Lssim}

            if 'Lrigid' in locals():
                loss_dict["Lrigid"] = Lrigid
            # Step 1: add new terms
            loss_dict["Lsparsity"] = Lsparsity
            loss_dict["Lmotion_gate"] = Lmotion_gate

            with torch.no_grad():
                psnr_for_log = psnr(image, gt_image).mean().double()
                # Progress bar
                ema_loss_for_log = 0.4 * total_loss + 0.6 * ema_loss_for_log
                ema_l1loss_for_log = 0.4 * Ll1.item() + 0.6 * ema_l1loss_for_log
                ema_ssimloss_for_log = 0.4 * Lssim.item() + 0.6 * ema_ssimloss_for_log

                for lambda_name in lambda_all:
                    if opt.__dict__[lambda_name] > 0:
                        loss_key = f"L{lambda_name.replace('lambda_', '')}"
                        if loss_key in locals():
                            ema = vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"]
                            vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"] = 0.4 * vars()[loss_key].item() + 0.6 * ema
                            loss_dict[loss_key] = vars()[loss_key]

                if iteration % 10 == 0:
                    postfix = {"Loss": f"{ema_loss_for_log:.{7}f}",
                               "PSNR": f"{psnr_for_log:.{2}f}",
                               "Ll1": f"{ema_l1loss_for_log:.{4}f}",
                               "Lssim": f"{ema_ssimloss_for_log:.{4}f}",
                               "points": scene.gaussians.get_xyz.shape[0],
                               "static": scene.gaussians.get_static_xyz.shape[0]}

                    for lambda_name in lambda_all:
                        if opt.__dict__[lambda_name] > 0:
                            ema_loss = vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"]
                            postfix[lambda_name.replace("lambda_", "L")] = f"{ema_loss:.{4}f}"

                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                test_psnr = training_report(tb_writer, iteration, Ll1, total_loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), loss_dict)
                if (iteration in testing_iterations):
                    if test_psnr is None:
                        test_psnr = 0.0 # or 0.0 if you prefer
                    if test_psnr >= best_psnr:
                        best_psnr = test_psnr
                        print("\n[ITER {}] Saving best checkpoint".format(iteration))
                        torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt_best.pth")

                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if iteration < opt.densify_until_iter and (opt.densify_until_num_points < 0 or gaussians.get_xyz.shape[0] < opt.densify_until_num_points):
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    if static:
                        gaussians.static_max_radii2D[visibility_filter_static] = torch.max(gaussians.static_max_radii2D[visibility_filter_static], radii_static[visibility_filter_static])
                    if batch_size == 1:
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, batch_t_grad if gaussians.gaussian_dim == 4 else None)
                    else:
                        gaussians.add_densification_stats_grad(batch_viewspace_point_grad, visibility_filter, batch_t_grad if gaussians.gaussian_dim == 4 else None)
                        if static:
                            gaussians.add_densification_stats_grad_static(batch_viewspace_point_grad_static, visibility_filter_static)

                    if iteration > opt.densify_from_iter:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        if iteration % opt.densification_interval == 0:
                            gaussians.update_staticness_score()
                            gaussians.densify_and_prune(
                                max_grad=opt.densify_grad_threshold,
                                min_opacity=opt.thresh_opa_prune,
                                extent=scene.cameras_extent,
                                max_screen_size=size_threshold,
                                max_grad_t=opt.densify_grad_t_threshold,
                                static_conversion_threshold=opt.static_conversion_threshold,
                                gate_activation_iter=opt.gate_activation_iter,
                                gate_warmup_until_iter=opt.gate_warmup_until_iter,
                                iteration=iteration,
                            )
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    if pipe.env_map_res and iteration < pipe.env_optimize_until:
                        env_map_optimizer.step()
                        env_map_optimizer.zero_grad(set_to_none = True)



def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs, loss_dict=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)

        gaussians = scene.gaussians
        opt = getattr(scene, 'opt', None)  # if you pass opt through Scene

        # ---- POINT COUNTS ----
        total_points = gaussians.get_xyz.shape[0]
        static_points = gaussians.get_static_xyz.shape[0] if hasattr(gaussians, 'get_static_xyz') else 0
        dynamic_points = total_points - static_points

        tb_writer.add_scalar('points/total', total_points, iteration)
        tb_writer.add_scalar('points/static', static_points, iteration)
        tb_writer.add_scalar('points/dynamic', dynamic_points, iteration)

        if hasattr(gaussians, '_staticness_score') and opt is not None:
            conversion_rate = (gaussians._staticness_score > opt.static_conversion_threshold).float().mean().item() * 100
            tb_writer.add_scalar('points/static_conversion_rate', conversion_rate, iteration)

        # ---- GATE SCALARS ----
        if hasattr(gaussians, 'differentiable_s') and gaussians.differentiable_s is not None and gaussians.differentiable_s.numel() > 0:
            s = gaussians.differentiable_s.detach()

            tb_writer.add_scalar('gate/scalars/mean_s', s.mean().item(), iteration)
            tb_writer.add_scalar('gate/scalars/median_s', s.median().item(), iteration)
            tb_writer.add_scalar('gate/scalars/min_s', s.min().item(), iteration)
            tb_writer.add_scalar('gate/scalars/max_s', s.max().item(), iteration)
            tb_writer.add_scalar('gate/scalars/percent_s>0.9', (s > 0.9).float().mean().item() * 100, iteration)
            tb_writer.add_scalar('gate/scalars/percent_s<0.1', (s < 0.1).float().mean().item() * 100, iteration)

            tb_writer.add_histogram('gate/hist/s_distribution', s, iteration, bins=50)

            # Optional: check how gating affects timestamps
            if hasattr(gaussians, 'get_t'):
                ts = gaussians.get_t.detach()
                tb_writer.add_histogram('gate/ts/timestamps_after_gating', ts, iteration, bins=50)
        # ---------------------------------------------------------------------------------

        if loss_dict is not None:
            if "Lrigid" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/rigid_loss', loss_dict['Lrigid'].item(), iteration)
            if "Ldepth" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/depth_loss', loss_dict['Ldepth'].item(), iteration)
            if "Ltv" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/tv_loss', loss_dict['Ltv'].item(), iteration)
            if "Lopa" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/opa_loss', loss_dict['Lopa'].item(), iteration)
            if "Lptsopa" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/pts_opa_loss', loss_dict['Lptsopa'].item(), iteration)
            if "Lsmooth" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/smooth_loss', loss_dict['Lsmooth'].item(), iteration)
            if "Llaplacian" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/laplacian_loss', loss_dict['Llaplacian'].item(), iteration)
            if "Lsparsity" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/gate_sparsity_loss', loss_dict['Lsparsity'].item(), iteration)
            if "Lmotion_gate" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/motion_gate_loss', loss_dict['Lmotion_gate'].item(), iteration)

        tb_writer.add_scalar('gpu/memory_allocated_MB', torch.cuda.memory_allocated() / 1e6, iteration)
        tb_writer.add_scalar('gpu/memory_reserved_MB', torch.cuda.memory_reserved() / 1e6, iteration)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2_000, 4_000, 6_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2_000, 3_000, 4_000, 5_000, 6_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument('--num_pts', type=int, default=100_000)
    parser.add_argument('--num_pts_ratio', type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")
    parser.add_argument("--val", action="store_true", default=False)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    cfg = OmegaConf.load(args.config)

    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])
    for k in cfg.keys():
        recursive_merge(k, cfg)

    if args.exhaust_test:
        args.test_iterations = args.test_iterations + [i for i in range(0,args.iterations,500)]

    setup_seed(args.seed)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if args.val == False:
        training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.start_checkpoint, args.debug_from,
                 args.gaussian_dim, args.time_duration, args.num_pts, args.num_pts_ratio, args.rot_4d, args.force_sh_3d, args.batch_size)

    else:
        validation(lp.extract(args), op.extract(args), pp.extract(args),args.start_checkpoint,args.gaussian_dim,
                   args.time_duration,args.rot_4d, args.force_sh_3d, args.num_pts, args.num_pts_ratio)


    print("\nComplete.")
