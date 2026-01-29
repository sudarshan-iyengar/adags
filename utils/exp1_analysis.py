import os
import json
import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import torch
from torchvision.utils import save_image

# Optional: if you already have easy_cmap, use it. If not present, Exp1 still works without colormap images.
try:
    from utils.image_utils import easy_cmap
    EASY_CMAP_AVAILABLE = True
except Exception:
    EASY_CMAP_AVAILABLE = False


@dataclass
class Exp1Config:
    out_dir: str                 # relative to model_path
    num_views: int               # max number of test views to analyze
    view_indices: Optional[List[int]]  # explicit indices if provided
    top_err_q: float             # quantile for ghost mask (e.g. 0.95)
    marginal_thr: float          # marginal_t threshold for "active-in-time" (e.g. 0.05)
    use_alpha_mask: bool         # if True and gt_alpha_mask exists, restrict error quantile to FG region
    save_images: bool            # whether to dump debug PNGs
    save_jsonl: bool             # whether to dump per-view JSONL
    save_summary_json: bool      # whether to dump summary JSON


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@torch.no_grad()
def _project_to_pixels_try_both(xyz: torch.Tensor, cam) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Robust projection: tries both row-vector and column-vector multiplication conventions.
    Returns:
      u, v in pixel coords (float32), shape [N]
    Assumes cam.full_proj_transform is [4,4].
    """
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    device = xyz.device
    dtype = xyz.dtype
    N = xyz.shape[0]
    ones = torch.ones((N, 1), device=device, dtype=dtype)
    xyz_h = torch.cat([xyz, ones], dim=1)  # [N,4]
    M = cam.full_proj_transform  # [4,4]

    clip_a = xyz_h @ M
    clip_b = (M @ xyz_h.T).T

    def clip_to_uv(clip: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w = clip[:, 3].clamp_min(1e-8)
        ndc = clip[:, :3] / w.unsqueeze(-1)  # [N,3]
        u = (ndc[:, 0] + 1.0) * 0.5 * float(cam.image_width)
        v = (1.0 - (ndc[:, 1] + 1.0) * 0.5) * float(cam.image_height)
        return u, v, ndc

    u1, v1, ndc1 = clip_to_uv(clip_a)
    u2, v2, ndc2 = clip_to_uv(clip_b)

    # Choose convention that yields more points plausibly inside extended NDC bounds
    score1 = ((ndc1.abs() < 1.5).all(dim=1)).float().mean()
    score2 = ((ndc2.abs() < 1.5).all(dim=1)).float().mean()

    if score1 >= score2:
        return u1.float(), v1.float()
    return u2.float(), v2.float()


@torch.no_grad()
def _compute_error_and_ghost_mask(pred: torch.Tensor, gt: torch.Tensor, cam, top_err_q: float, use_alpha_mask: bool):
    """
    pred, gt: [3,H,W] in [0,1]
    returns:
      err: [H,W] float
      ghost_mask: [H,W] bool
      fg_mask (or None): [H,W] bool or None
      thr: float
    """
    assert pred.ndim == 3 and gt.ndim == 3
    err = (pred - gt).abs().mean(dim=0)  # [H,W]

    fg_mask = None
    if use_alpha_mask and hasattr(cam, "gt_alpha_mask") and cam.gt_alpha_mask is not None:
        # Expect gt_alpha_mask broadcastable to [H,W]
        fg_mask = (cam.gt_alpha_mask > 0.5)
        # Ensure shape [H,W]
        if fg_mask.ndim == 3:
            fg_mask = fg_mask.squeeze(0)
        if fg_mask.shape != err.shape:
            # If mismatch, disable alpha masking deterministically
            fg_mask = None

    if fg_mask is None:
        thr = torch.quantile(err.flatten(), top_err_q)
        ghost_mask = (err >= thr)
    else:
        fg_err = err[fg_mask]
        if fg_err.numel() < 10:
            thr = torch.quantile(err.flatten(), top_err_q)
            ghost_mask = (err >= thr)
        else:
            thr = torch.quantile(fg_err, top_err_q)
            ghost_mask = (err >= thr) & fg_mask

    return err, ghost_mask, fg_mask, float(thr.item())


@torch.no_grad()
def _temporal_std_and_marginal(gaussians, timestamp: float, eps: float = 1e-8):
    """
    Uses your GaussianModel definitions:
      cov_t = gaussians.get_cov_t()
      marginal_t = gaussians.get_marginal_t(timestamp)
    Interprets cov_t as the variance used in exp(-0.5*dt^2/cov_t).
    """
    cov_t = gaussians.get_cov_t().squeeze(-1).clamp_min(eps)   # [N]
    std_t = torch.sqrt(cov_t)                                  # [N]
    marginal = gaussians.get_marginal_t(float(timestamp)).squeeze(-1)  # [N]
    return std_t, marginal, cov_t


@torch.no_grad()
def _make_std_t_override_colors(std_t: torch.Tensor, marginal: torch.Tensor, active_mask: torch.Tensor, eps=1e-8):
    """
    Deterministic visualization ramp:
      low std_t -> blue, high std_t -> red, then multiply by marginal to suppress irrelevant Gaussians.
    Returns [N,3] colors in [0,1].
    """
    vals = std_t[active_mask]
    if vals.numel() < 10:
        vals = std_t

    lo = torch.quantile(vals, 0.05)
    hi = torch.quantile(vals, 0.95)
    x = ((std_t - lo) / (hi - lo + eps)).clamp(0.0, 1.0)

    colors = torch.stack([x, torch.zeros_like(x), 1.0 - x], dim=-1)  # [N,3]
    colors = colors * marginal.clamp(0.0, 1.0).unsqueeze(-1)
    return colors


@torch.no_grad()
def _pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    """
    x,y: [N] float
    """
    x = x.float()
    y = y.float()
    x = (x - x.mean()) / (x.std() + eps)
    y = (y - y.mean()) / (y.std() + eps)
    return float((x * y).mean().item())


@torch.no_grad()
def analyze_single_view_exp1(
    gaussians,
    cam,
    pipe,
    background: torch.Tensor,
    gt_image: torch.Tensor,
    render_fn,
    cfg: Exp1Config,
) -> Dict[str, Any]:
    """
    Returns a dict of all per-view metrics.
    Also optionally saves images if cfg.save_images is True.
    """
    # Render standard
    pkg = render_fn(cam, gaussians, pipe, background)
    pred = torch.clamp(pkg["render"], 0.0, 1.0)  # [3,H,W]
    gt = torch.clamp(gt_image, 0.0, 1.0)

    # Error + ghost mask
    err, ghost_mask, fg_mask, thr = _compute_error_and_ghost_mask(pred, gt, cam, cfg.top_err_q, cfg.use_alpha_mask)

    # Dynamic visibility from rasterizer
    visibility_filter = pkg["visibility_filter"]  # [N] bool for dynamic set

    # Temporal stats for dynamic Gaussians (requires gaussian_dim==4)
    std_t, marginal, cov_t = _temporal_std_and_marginal(gaussians, float(cam.timestamp))
    active_dyn = visibility_filter & (marginal > cfg.marginal_thr)

    # Project dynamic centers to pixels and sample ghost/error at centers
    u, v = _project_to_pixels_try_both(gaussians.get_xyz, cam)
    ui = u.round().long()
    vi = v.round().long()
    H, W = err.shape
    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

    valid_dyn = in_bounds & active_dyn
    idx_dyn = torch.where(valid_dyn)[0]

    center_err_dyn = torch.zeros_like(std_t)
    center_ghost_dyn = torch.zeros_like(active_dyn, dtype=torch.bool)
    center_err_dyn[idx_dyn] = err[vi[idx_dyn], ui[idx_dyn]]
    center_ghost_dyn[idx_dyn] = ghost_mask[vi[idx_dyn], ui[idx_dyn]]

    # Weight each Gaussian by opacity*marginal (deterministic, no extra heuristics)
    opacity = gaussians.get_opacity.squeeze(-1)  # [N]
    weight_dyn = (opacity * marginal).clamp_min(0.0)

    # Split indices
    ghost_dyn_idx = idx_dyn[center_ghost_dyn[idx_dyn]]
    nonghost_dyn_idx = idx_dyn[~center_ghost_dyn[idx_dyn]]

    # Summaries
    def summarize(x: torch.Tensor) -> Dict[str, float]:
        if x.numel() == 0:
            return {"n": 0, "mean": float("nan"), "median": float("nan"), "p90": float("nan"), "p99": float("nan")}
        return {
            "n": int(x.numel()),
            "mean": float(x.mean().item()),
            "median": float(x.median().item()),
            "p90": float(torch.quantile(x, 0.90).item()),
            "p99": float(torch.quantile(x, 0.99).item()),
        }

    std_active = std_t[idx_dyn]
    err_active = center_err_dyn[idx_dyn]
    corr_std_err = _pearson_corr(std_active, err_active) if idx_dyn.numel() >= 50 else float("nan")

    std_ghost = std_t[ghost_dyn_idx]
    std_nonghost = std_t[nonghost_dyn_idx]

    # "Domination" statistic: fraction above active p90
    dom_thr = torch.quantile(std_active, 0.90) if std_active.numel() > 0 else torch.tensor(float("nan"), device=std_t.device)
    frac_high_std_in_ghost = float((std_ghost > dom_thr).float().mean().item()) if std_ghost.numel() else float("nan")
    frac_high_std_in_nonghost = float((std_nonghost > dom_thr).float().mean().item()) if std_nonghost.numel() else float("nan")

    # Weight-based domination
    w_ghost = weight_dyn[ghost_dyn_idx].sum().item() if ghost_dyn_idx.numel() else 0.0
    w_nonghost = weight_dyn[nonghost_dyn_idx].sum().item() if nonghost_dyn_idx.numel() else 0.0
    w_total = (weight_dyn[idx_dyn].sum().item() if idx_dyn.numel() else 0.0) + 1e-12
    frac_weight_ghost = float(w_ghost / w_total)

    # Static analysis (if present in your renderer output)
    static_counts = {}
    if "visibility_filter_static" in pkg and hasattr(gaussians, "get_static_xyz"):
        vis_s = pkg["visibility_filter_static"]  # [Ns] bool
        if vis_s.numel() > 0:
            u_s, v_s = _project_to_pixels_try_both(gaussians.get_static_xyz, cam)
            ui_s = u_s.round().long()
            vi_s = v_s.round().long()
            inb_s = (ui_s >= 0) & (ui_s < W) & (vi_s >= 0) & (vi_s < H)
            valid_s = inb_s & vis_s
            idx_s = torch.where(valid_s)[0]
            ghost_s = ghost_mask[vi_s[idx_s], ui_s[idx_s]]
            static_counts = {
                "static_visible": int(idx_s.numel()),
                "static_in_ghost": int(ghost_s.sum().item()),
                "static_frac_in_ghost": float(ghost_s.float().mean().item()) if idx_s.numel() else float("nan"),
            }
        else:
            static_counts = {"static_visible": 0, "static_in_ghost": 0, "static_frac_in_ghost": float("nan")}

    # Optional debug images
    debug_paths = {}
    if cfg.save_images:
        # standard dumps
        debug_paths["gt"] = "gt.png"
        debug_paths["pred"] = "pred.png"
        debug_paths["err"] = "err.png"
        debug_paths["ghost_overlay"] = "ghost_overlay.png"
        debug_paths["stdt_render"] = "stdt_render.png"

    out = {
        "image_name": getattr(cam, "image_name", "unknown"),
        "timestamp": float(cam.timestamp),
        "H": int(pred.shape[1]),
        "W": int(pred.shape[2]),
        "ghost_q": cfg.top_err_q,
        "ghost_thr": thr,
        "use_alpha_mask": cfg.use_alpha_mask and (fg_mask is not None),
        "num_dyn_visible_active": int(idx_dyn.numel()),
        "num_dyn_ghost": int(ghost_dyn_idx.numel()),
        "num_dyn_nonghost": int(nonghost_dyn_idx.numel()),
        "std_t_active": summarize(std_active),
        "std_t_ghost": summarize(std_ghost),
        "std_t_nonghost": summarize(std_nonghost),
        "corr_std_t__err_center": corr_std_err,
        "dom_thr_std_t_p90_active": float(dom_thr.item()) if torch.isfinite(dom_thr) else float("nan"),
        "frac_high_std_in_ghost": frac_high_std_in_ghost,
        "frac_high_std_in_nonghost": frac_high_std_in_nonghost,
        "frac_weight_ghost": frac_weight_ghost,
        "static": static_counts,
    }

    # Return images to caller to save (caller controls directory names)
    out["_images"] = {
        "gt": gt,
        "pred": pred,
        "err": err,
        "ghost_mask": ghost_mask,
        "std_t": std_t,
        "marginal": marginal,
        "active_dyn": active_dyn,
    }
    return out


@torch.no_grad()
def save_exp1_images(
    out_dir: str,
    view_result: Dict[str, Any],
    gaussians,
    cam,
    pipe,
    background: torch.Tensor,
    render_fn,
):
    """
    Saves deterministic PNGs:
      - gt.png
      - pred.png
      - err.png (colormap if easy_cmap exists; otherwise grayscale)
      - ghost_overlay.png
      - stdt_render.png (render with override_color based on std_t)
    """
    imgs = view_result["_images"]
    gt = imgs["gt"]
    pred = imgs["pred"]
    err = imgs["err"]
    ghost_mask = imgs["ghost_mask"]
    std_t = imgs["std_t"]
    marginal = imgs["marginal"]
    active_dyn = imgs["active_dyn"]

    _mkdir(out_dir)
    save_image(gt, os.path.join(out_dir, "gt.png"))
    save_image(pred, os.path.join(out_dir, "pred.png"))

    # Error image
    err_norm = (err - err.min()) / (err.max() - err.min() + 1e-8)
    if EASY_CMAP_AVAILABLE:
        err_rgb = easy_cmap(err_norm)  # should return [3,H,W]
        save_image(err_rgb, os.path.join(out_dir, "err.png"))
    else:
        save_image(err_norm.unsqueeze(0).repeat(3, 1, 1), os.path.join(out_dir, "err.png"))

    # Ghost overlay on pred
    overlay = pred.clone()
    # Red highlight for ghost pixels
    overlay[0, ghost_mask] = 1.0
    overlay[1, ghost_mask] = overlay[1, ghost_mask] * 0.2
    overlay[2, ghost_mask] = overlay[2, ghost_mask] * 0.2
    save_image(overlay, os.path.join(out_dir, "ghost_overlay.png"))

    # std_t render via override_color
    colors = _make_std_t_override_colors(std_t, marginal, active_dyn)
    stdt_pkg = render_fn(cam, gaussians, pipe, background, override_color=colors)
    stdt_img = torch.clamp(stdt_pkg["render"], 0.0, 1.0)
    save_image(stdt_img, os.path.join(out_dir, "stdt_render.png"))


@torch.no_grad()
def run_exp1_over_testset(
    scene,
    gaussians,
    pipe,
    background: torch.Tensor,
    render_fn,
    cfg: Exp1Config,
) -> Dict[str, Any]:
    """
    Runs Exp1 on a subset of test cameras.
    Writes:
      - per-view JSONL (optional)
      - summary.json (optional)
      - per-view images (optional)
    Returns summary dict.
    """
    exp1_root = os.path.join(scene.model_path, cfg.out_dir)
    images_root = os.path.join(exp1_root, "views")
    _mkdir(exp1_root)
    _mkdir(images_root)

    # Select cameras
    test_ds = scene.getTestCameras()
    test_len = len(test_ds)

    if cfg.view_indices is not None and len(cfg.view_indices) > 0:
        indices = [i for i in cfg.view_indices if 0 <= i < test_len]
    else:
        indices = list(range(min(cfg.num_views, test_len)))

    jsonl_path = os.path.join(exp1_root, "views.jsonl")
    jsonl_f = open(jsonl_path, "w") if cfg.save_jsonl else None

    results = []
    for i, idx in enumerate(indices):
        gt_image, cam = test_ds[idx]
        gt_image = gt_image.cuda()
        cam = cam.cuda()

        view_res = analyze_single_view_exp1(
            gaussians=gaussians,
            cam=cam,
            pipe=pipe,
            background=background,
            gt_image=gt_image,
            render_fn=render_fn,
            cfg=cfg,
        )

        # Save per-view images
        if cfg.save_images:
            name = view_res["image_name"]
            safe_name = str(name).replace("/", "_")
            view_dir = os.path.join(images_root, f"{idx:04d}_{safe_name}")
            save_exp1_images(view_dir, view_res, gaussians, cam, pipe, background, render_fn)

        # Strip tensors before writing JSON
        view_res.pop("_images", None)
        results.append(view_res)

        if jsonl_f is not None:
            jsonl_f.write(json.dumps(view_res) + "\n")
            jsonl_f.flush()

    if jsonl_f is not None:
        jsonl_f.close()

    # Aggregate summary deterministically (simple mean over finite values)
    def mean_over(key: str) -> float:
        vals = []
        for r in results:
            v = r.get(key, float("nan"))
            if isinstance(v, (int, float)) and math.isfinite(v):
                vals.append(float(v))
        return float(sum(vals) / max(1, len(vals))) if vals else float("nan")

    summary = {
        "num_views": int(len(results)),
        "mean_corr_std_t__err_center": mean_over("corr_std_t__err_center"),
        "mean_frac_high_std_in_ghost": mean_over("frac_high_std_in_ghost"),
        "mean_frac_high_std_in_nonghost": mean_over("frac_high_std_in_nonghost"),
        "mean_frac_weight_ghost": mean_over("frac_weight_ghost"),
        "views_jsonl": "views.jsonl" if cfg.save_jsonl else None,
        "views_dir": "views" if cfg.save_images else None,
    }

    if cfg.save_summary_json:
        with open(os.path.join(exp1_root, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    return summary
