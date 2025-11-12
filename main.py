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

import os, sys, uuid, random
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader

from gaussian_renderer import render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import l1_loss, ssim, msssim
from utils.image_utils import psnr, easy_cmap
from utils.general_utils import safe_state, knn
from utils.mesh_utils import GaussianExtractor
from utils.render_utils import generate_path, create_videos

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def identity_collate(x):
    return x


def validation(dataset, opt, pipe, checkpoint, gaussian_dim, time_duration, rot_4d, force_sh_3d, num_pts, num_pts_ratio):
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=gaussian_dim, time_duration=time_duration, rot_4d=rot_4d, force_sh_3d=force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    assert checkpoint, "No checkpoint provided for validation"
    scene = Scene(dataset, gaussians, shuffle=False, num_pts=num_pts, num_pts_ratio=num_pts_ratio, time_duration=time_duration)

    (model_params, first_iter) = torch.load(checkpoint)
    train_dir = os.path.join(dataset.model_path, 'train', f"ours_{first_iter}")
    test_dir = os.path.join(dataset.model_path, 'test', f"ours_{first_iter}")
    gaussians.restore(model_params, None)
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)

    print("export rendered testing images ...")
    os.makedirs(test_dir, exist_ok=True)
    gaussExtractor.reconstruction(scene.getTestCameras(), test_dir, stage="validation")
    gaussExtractor.export_image(test_dir, mode="validation")


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint, debug_from,
             gaussian_dim, time_duration, num_pts, num_pts_ratio, rot_4d, force_sh_3d, batch_size):

    if dataset.frame_ratio > 1:
        time_duration = [time_duration[0] / dataset.frame_ratio, time_duration[1] / dataset.frame_ratio]

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

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    best_psnr = 0.0
    ema_loss_for_log = 0.0
    ema_l1loss_for_log = 0.0
    ema_ssimloss_for_log = 0.0

    lambda_all = [key for key in opt.__dict__.keys() if key.startswith('lambda') and key != 'lambda_dssim']
    for lambda_name in lambda_all:
        vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"] = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    if pipe.env_map_res:
        env_map = nn.Parameter(torch.zeros((3, pipe.env_map_res, pipe.env_map_res), dtype=torch.float, device="cuda").requires_grad_(True))
        env_map_optimizer = torch.optim.Adam([env_map], lr=opt.feature_lr, eps=1e-15)
    else:
        env_map = None
        env_map_optimizer = None

    gaussians.env_map = env_map

    training_dataset = scene.getTrainCameras()
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True,
                                     num_workers=0 if dataset.dataloader else 0,
                                     collate_fn=identity_collate, drop_last=True)
    iteration = first_iter

    # placeholders for gate losses
    device = "cuda"
    Lsparsity = torch.tensor(0.0, device=device)
    Lmotion_gate = torch.tensor(0.0, device=device)

    motion_gate_quantile = getattr(opt, "motion_gate_quantile", 0.8)

    while iteration < opt.iterations + 1:
        for batch_data in training_dataloader:
            iteration += 1
            if iteration > opt.iterations:
                break

            iter_start.record()
            gaussians.update_learning_rate(iteration)

            if iteration % opt.sh_increase_interval == 0:
                gaussians.oneupSHdegree()

            if (iteration - 1) == debug_from:
                pipe.debug = True

            # compute gate scores (no gate loss yet)
            if iteration >= opt.gate_activation_iter:
                gaussians.compute_differentiable_staticness()
            else:
                if gaussians._xyz.shape[0] > 0:
                    gaussians.differentiable_s = torch.zeros((gaussians._xyz.shape[0], 1), device=device, requires_grad=False)

            total_loss = 0.0

            batch_point_grad, batch_visibility_filter, batch_radii = [], [], []
            batch_point_grad_static, batch_visibility_filter_static, batch_radii_static = [], [], []

            static = False

            # ================= inner micro-batch loop =================
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

                # opa mask loss
                if opt.lambda_opa_mask > 0:
                    o = alpha.clamp(1e-6, 1 - 1e-6)
                    sky = 1 - viewpoint_cam.gt_alpha_mask
                    Lopa_mask = (-sky * torch.log(1 - o)).mean()
                    loss = loss + opt.lambda_opa_mask * Lopa_mask

                # rigid loss
                if opt.lambda_rigid > 0 and gaussians.gaussian_dim == 4 and gaussians.get_xyz.shape[0] > 0:
                    k = 20
                    xyz_cur = gaussians.get_xyz
                    idx, dist = knn(xyz_cur[None].contiguous().detach(), xyz_cur[None].contiguous().detach(), k)
                    _, velocity = gaussians.get_current_covariance_and_mean_offset(1.0, gaussians.get_t + 0.1)
                    weight = torch.exp(-100 * dist)
                    vel_dist = torch.norm(velocity[idx] - velocity[None, :, None], p=2, dim=-1)
                    Lrigid = (weight * vel_dist).sum() / k / xyz_cur.shape[0]
                    loss = loss + opt.lambda_rigid * Lrigid

                total_loss += loss.item()
                (loss / batch_size).backward(retain_graph=True)

                batch_point_grad.append(torch.norm(viewspace_point_tensor.grad[:, :2], dim=-1))
                batch_radii.append(radii)
                batch_visibility_filter.append(visibility_filter)

                if len(viewspace_point_tensor_static) > 0:
                    static = True
                    batch_point_grad_static.append(torch.norm(viewspace_point_tensor_static.grad[:, :2], dim=-1))
                    batch_radii_static.append(radii_static)
                    batch_visibility_filter_static.append(visibility_filter_static)

            # ================= aggregate grads over micro-batches =================
            if batch_size > 1:
                visibility_count = torch.stack(batch_visibility_filter, 1).sum(1)
                visibility_filter = visibility_count > 0
                radii = torch.stack(batch_radii, 1).max(1)[0]

                batch_viewspace_point_grad = torch.stack(batch_point_grad, 1).sum(1)
                batch_viewspace_point_grad[visibility_filter] = batch_viewspace_point_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                batch_viewspace_point_grad = batch_viewspace_point_grad.unsqueeze(1)

                if static:
                    visibility_count_static = torch.stack(batch_visibility_filter_static, 1).sum(1)
                    visibility_filter_static = visibility_count_static > 0
                    radii_static = torch.stack(batch_radii_static, 1).max(1)[0]

                    batch_viewspace_point_grad_static = torch.stack(batch_point_grad_static, 1).sum(1)
                    batch_viewspace_point_grad_static[visibility_filter_static] = batch_viewspace_point_grad_static[visibility_filter_static] * batch_size / visibility_count_static[visibility_filter_static]
                    batch_viewspace_point_grad_static = batch_viewspace_point_grad_static.unsqueeze(1)

                if gaussians.gaussian_dim == 4:
                    batch_t_grad = gaussians._t.grad.clone()[:, 0].detach()
                    batch_t_grad[visibility_filter] = batch_t_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                    batch_t_grad = batch_t_grad.unsqueeze(1)
            else:
                visibility_filter = batch_visibility_filter[0]
                radii = batch_radii[0]
                batch_viewspace_point_grad = batch_point_grad[0].unsqueeze(1)
                if static:
                    visibility_filter_static = batch_visibility_filter_static[0]
                    radii_static = batch_radii_static[0]
                    batch_viewspace_point_grad_static = batch_point_grad_static[0].unsqueeze(1)
                if gaussians.gaussian_dim == 4:
                    batch_t_grad = gaussians._t.grad.clone().detach()

            # ================= gate losses (monotonic logistic on log σ_t) =================
            Lsparsity = torch.tensor(0.0, device=device)
            Lmotion_gate = torch.tensor(0.0, device=device)

            if iteration > opt.gate_activation_iter and gaussians._xyz.shape[0] > 0:
                s = gaussians.differentiable_s
                if s is not None and s.numel() > 0:
                    # annealed sparsity toward static (s→1)
                    if opt.gate_warmup_until_iter > opt.gate_activation_iter:
                        anneal = min(1.0, (iteration - opt.gate_activation_iter) / (opt.gate_warmup_until_iter - opt.gate_activation_iter))
                    else:
                        anneal = 1.0
                    lambda_sparsity_eff = opt.lambda_sparsity * anneal
                    Lsparsity = lambda_sparsity_eff * (1.0 - s).mean()

                    # motion magnitude from covariance-induced mean offset (no grad into motion)
                    with torch.no_grad():
                        _, velocity = gaussians.get_current_covariance_and_mean_offset(1.0, gaussians.get_t + 0.1)
                        motion_mag = velocity.norm(p=2, dim=1, keepdim=True)
                        motion_mag = torch.nan_to_num(motion_mag)
                        if motion_mag.numel() > 0:
                            scale = torch.quantile(motion_mag, motion_gate_quantile).clamp_min(1e-6)
                        else:
                            scale = torch.tensor(1.0, device=device)
                        Lmotion_per_point = motion_mag / scale

                    # penalize s for high-motion gaussians
                    Lmotion_gate = opt.lambda_motion_gate * (s * Lmotion_per_point.detach()).mean()

                    gate_loss = Lsparsity + Lmotion_gate
                    if gate_loss.requires_grad and gate_loss.item() != 0.0:
                        total_loss += gate_loss.item()
                        gate_loss.backward()

            iter_end.record()

            # ================= logging dictionary =================
            loss_dict = {"Ll1": Ll1, "Lssim": Lssim, "Lsparsity": Lsparsity, "Lmotion_gate": Lmotion_gate}
            if 'Lrigid' in locals(): loss_dict["Lrigid"] = Lrigid

            with torch.no_grad():
                psnr_for_log = psnr(image, gt_image).mean().double()

                ema_loss_for_log = 0.4 * total_loss + 0.6 * ema_loss_for_log
                ema_l1loss_for_log = 0.4 * Ll1.item() + 0.6 * ema_l1loss_for_log
                ema_ssimloss_for_log = 0.4 * Lssim.item() + 0.6 * ema_ssimloss_for_log

                for lambda_name in lambda_all:
                    if opt.__dict__[lambda_name] > 0:
                        loss_key = f"L{lambda_name.replace('lambda_', '')}"
                        if loss_key in locals():
                            ema_name = f"ema_{lambda_name.replace('lambda_', '')}_for_log"
                            ema_val = vars()[ema_name]
                            vars()[ema_name] = 0.4 * vars()[loss_key].item() + 0.6 * ema_val
                            loss_dict[loss_key] = vars()[loss_key]

                if iteration % 10 == 0:
                    postfix = {
                        "Loss": f"{ema_loss_for_log:.7f}",
                        "PSNR": f"{psnr_for_log:.2f}",
                        "Ll1": f"{ema_l1loss_for_log:.4f}",
                        "Lssim": f"{ema_ssimloss_for_log:.4f}",
                        "points": scene.gaussians.get_xyz.shape[0],
                        "static": scene.gaussians.get_static_xyz.shape[0]
                    }
                    for lambda_name in lambda_all:
                        if opt.__dict__[lambda_name] > 0:
                            key = lambda_name.replace("lambda_", "L")
                            ema_name = f"ema_{lambda_name.replace('lambda_', '')}_for_log"
                            postfix[key] = f"{vars()[ema_name]:.4f}"
                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                test_psnr = training_report(tb_writer, iteration, Ll1, total_loss, l1_loss, iter_start.elapsed_time(iter_end),
                                            testing_iterations, scene, render, (pipe, background), loss_dict)

                if iteration in testing_iterations:
                    if test_psnr is None: test_psnr = 0.0
                    if test_psnr >= best_psnr:
                        best_psnr = test_psnr
                        print(f"\n[ITER {iteration}] Saving best checkpoint")
                        torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt_best.pth")

                if iteration in saving_iterations:
                    print(f"\n[ITER {iteration}] Saving Gaussians")
                    scene.save(iteration)

                # ================= densification =================
                if iteration < opt.densify_until_iter and (opt.densify_until_num_points < 0 or gaussians.get_xyz.shape[0] < opt.densify_until_num_points):
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    if static:
                        gaussians.static_max_radii2D[visibility_filter_static] = torch.max(gaussians.static_max_radii2D[visibility_filter_static], radii_static[visibility_filter_static])

                    if batch_size == 1:
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter,
                                                          batch_t_grad if gaussians.gaussian_dim == 4 else None)
                    else:
                        gaussians.add_densification_stats_grad(batch_viewspace_point_grad, visibility_filter,
                                                               batch_t_grad if gaussians.gaussian_dim == 4 else None)
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

                # ================= optimizer step =================
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    if pipe.env_map_res and iteration < pipe.env_optimize_until and env_map_optimizer is not None:
                        env_map_optimizer.step()
                        env_map_optimizer.zero_grad(set_to_none=True)


def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv('OAR_JOB_ID') if os.getenv('OAR_JOB_ID') else str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
        return None


def training_report(tb_writer, iteration, Ll1, loss, l1_loss_fn, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs, loss_dict=None):
    test_psnr = None

    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)

        gaussians = scene.gaussians
        opt = getattr(scene, 'opt', None)

        total_points = gaussians.get_xyz.shape[0]
        static_points = gaussians.get_static_xyz.shape[0] if hasattr(gaussians, 'get_static_xyz') else 0
        dynamic_points = total_points - static_points

        tb_writer.add_scalar('points/total', total_points, iteration)
        tb_writer.add_scalar('points/static', static_points, iteration)
        tb_writer.add_scalar('points/dynamic', dynamic_points, iteration)

        if hasattr(gaussians, '_staticness_score') and opt is not None and gaussians._staticness_score.numel() > 0:
            conversion_rate = (gaussians._staticness_score > opt.static_conversion_threshold).float().mean().item() * 100
            tb_writer.add_scalar('points/static_conversion_rate', conversion_rate, iteration)

        if hasattr(gaussians, 'differentiable_s') and gaussians.differentiable_s is not None and gaussians.differentiable_s.numel() > 0:
            s = gaussians.differentiable_s.detach()
            tb_writer.add_scalar('gate/scalars/mean_s', s.mean().item(), iteration)
            tb_writer.add_scalar('gate/scalars/median_s', s.median().item(), iteration)
            tb_writer.add_scalar('gate/scalars/min_s', s.min().item(), iteration)
            tb_writer.add_scalar('gate/scalars/max_s', s.max().item(), iteration)
            tb_writer.add_scalar('gate/scalars/percent_s>0.9', (s > 0.9).float().mean().item() * 100, iteration)
            tb_writer.add_scalar('gate/scalars/percent_s<0.1', (s < 0.1).float().mean().item() * 100, iteration)
            tb_writer.add_histogram('gate/hist/s_distribution', s, iteration, bins=50)
            if hasattr(gaussians, 'get_t') and gaussians.get_t.numel() > 0:
                ts = gaussians.get_t.detach()
                tb_writer.add_histogram('gate/ts/timestamps_after_gating', ts, iteration, bins=50)

        if loss_dict is not None:
            if "Lrigid" in loss_dict: tb_writer.add_scalar('train_loss_patches/rigid_loss', loss_dict['Lrigid'].item(), iteration)
            if "Ldepth" in loss_dict: tb_writer.add_scalar('train_loss_patches/depth_loss', loss_dict['Ldepth'].item(), iteration)
            if "Ltv" in loss_dict: tb_writer.add_scalar('train_loss_patches/tv_loss', loss_dict['Ltv'].item(), iteration)
            if "Lopa" in loss_dict: tb_writer.add_scalar('train_loss_patches/opa_loss', loss_dict['Lopa'].item(), iteration)
            if "Lptsopa" in loss_dict: tb_writer.add_scalar('train_loss_patches/pts_opa_loss', loss_dict['Lptsopa'].item(), iteration)
            if "Lsmooth" in loss_dict: tb_writer.add_scalar('train_loss_patches/smooth_loss', loss_dict['Lsmooth'].item(), iteration)
            if "Llaplacian" in loss_dict: tb_writer.add_scalar('train_loss_patches/laplacian_loss', loss_dict['Llaplacian'].item(), iteration)
            if "Lsparsity" in loss_dict: tb_writer.add_scalar('train_loss_patches/gate_sparsity_loss', loss_dict['Lsparsity'].item(), iteration)
            if "Lmotion_gate" in loss_dict: tb_writer.add_scalar('train_loss_patches/motion_gate_loss', loss_dict['Lmotion_gate'].item(), iteration)

        tb_writer.add_scalar('gpu/memory_allocated_MB', torch.cuda.memory_allocated() / 1e6, iteration)
        tb_writer.add_scalar('gpu/memory_reserved_MB', torch.cuda.memory_reserved() / 1e6, iteration)

    # simple evaluation on test set when requested
    if iteration in testing_iterations:
        (pipe, background) = renderArgs
        test_cams = scene.getTestCameras()
        if len(test_cams) > 0:
            psnrs = []
            with torch.no_grad():
                for data in test_cams:
                    # Unpack in case dataset returns (gt_image, cam)
                    if isinstance(data, (list, tuple)) and len(data) == 2:
                        gt_image, cam = data
                    else:
                        gt_image, cam = None, data

                    cam = cam.cuda()
                    render_out = renderFunc(cam, scene.gaussians, pipe, background)
                    pred = render_out["render"]

                    if gt_image is not None: gt = gt_image.cuda()
                    elif hasattr(cam, "original_image"): gt = cam.original_image.cuda()
                    elif hasattr(cam, "gt_image"): gt = cam.gt_image.cuda()
                    else:
                        raise ValueError("No ground truth image found for test camera.")

                    psnrs.append(psnr(pred, gt).mean().item())

                    # for cam in test_cams:
                    # cam = cam.cuda()
                    # render_out = renderFunc(cam, scene.gaussians, pipe, background)
                    # pred = render_out["render"]
                    # gt = cam.original_image.cuda() if hasattr(cam, "original_image") else cam.gt_image.cuda()
                    # psnrs.append(psnr(pred, gt).mean().item())
            if psnrs:
                test_psnr = float(np.mean(psnrs))
                if tb_writer: tb_writer.add_scalar('test/psnr', test_psnr, iteration)
                return test_psnr
    return None


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--config", type=str)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2_000, 4_000, 6_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2_000, 3_000, 4_000, 5_000, 6_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)

    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
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
            for key1 in host[key].keys(): recursive_merge(key1, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])

    for k in cfg.keys(): recursive_merge(k, cfg)

    if args.exhaust_test:
        args.test_iterations = args.test_iterations + [i for i in range(0, args.iterations, 500)]

    setup_seed(args.seed)
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    if not args.val:
        training(lp.extract(args), op.extract(args), pp.extract(args),
                 args.test_iterations, args.save_iterations, args.start_checkpoint, args.debug_from,
                 args.gaussian_dim, args.time_duration, args.num_pts, args.num_pts_ratio,
                 args.rot_4d, args.force_sh_3d, args.batch_size)
    else:
        validation(lp.extract(args), op.extract(args), pp.extract(args),
                   args.start_checkpoint, args.gaussian_dim, args.time_duration,
                   args.rot_4d, args.force_sh_3d, args.num_pts, args.num_pts_ratio)

    print("\nComplete.")
