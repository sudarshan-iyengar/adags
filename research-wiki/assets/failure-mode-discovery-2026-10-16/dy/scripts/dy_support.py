"""fmd dy Pilot 1 (D1): training-only reliability evidence for per-Gaussian temporal support (see patch_d1.py).

compute_reach_samples(s2d, cams, scf): moving BootsTAPIR tracks, up to K_REF visible reference frames each (depth
valid), lifted to 3D with the training depth/camera, attached to the nearest scaffold node at t0, warped by the
scaffold to every frame, projected into the training camera; residual vs the observed track (pixels).
Reach = largest R in GRID with median residual in every band (prev R, R] <= max(2 px, 2 * r0), r0 = median residual
over 0 < |dt| <= 2. Bands without observations neither pass nor fail. Returns pos (S, T, 3; positions at every frame,
half precision) and reach (S). Writes a summary next to the run log (DY_LOG env, optional).
"""
import os, json, logging
import numpy as np, torch

GRID = [2, 4, 8, 16, 32, 64, 128]
K_REF = 6
S_MAX = 40000
FLOOR_PX = 2.0
FACTOR = 2.0


@torch.no_grad()
def compute_reach_samples(s2d, cams, scf, seed=0):
    dev = s2d.track.device
    T, H, W = s2d.T, s2d.H, s2d.W
    trk = s2d.track[..., :2].long()            # T,N,2 pixel (x, y)
    vis = s2d.track_mask.bool()                # T,N
    dyn = s2d.dynamic_track_mask.bool()        # N
    homo = s2d.homo_map.to(dev)                # H,W,2 (short side in [-1, 1])
    px_per_unit = min(H, W) / 2.0
    jid = torch.nonzero(dyn).flatten()
    g = torch.Generator(device="cpu").manual_seed(seed)
    rows_j, rows_t = [], []
    for j in jid.tolist():
        tv = torch.nonzero(vis[:, j]).flatten()
        if len(tv) == 0: continue
        x, y = trk[tv, j, 0].clamp(0, W - 1), trk[tv, j, 1].clamp(0, H - 1)
        ok = s2d.dep_mask[tv, y, x].bool()
        tv = tv[ok]
        if len(tv) == 0: continue
        pick = tv[torch.linspace(0, len(tv) - 1, min(K_REF, len(tv))).round().long().unique()]
        rows_j.append(torch.full_like(pick, j)); rows_t.append(pick)
    J = torch.cat(rows_j); T0 = torch.cat(rows_t)
    if len(J) > S_MAX:
        sel = torch.randperm(len(J), generator=g)[:S_MAX].to(J.device); J, T0 = J[sel], T0[sel]
    S = len(J)
    x0, y0 = trk[T0, J, 0].clamp(0, W - 1), trk[T0, J, 1].clamp(0, H - 1)
    uv0 = homo[y0, x0]; d0 = s2d.dep[T0, y0, x0]
    pc = cams.backproject(uv0, d0)
    pw = torch.zeros_like(pc)
    for t in torch.unique(T0).tolist():
        m = T0 == t; pw[m] = cams.trans_pts_to_world(t, pc[m])
    attach = scf.identify_nearest_node_id(pw, T0.int()).long()
    pos = torch.zeros(S, T, 3, dtype=torch.float16, device=dev)
    res = torch.full((S, T), float("nan"), device=dev)
    for t in range(T):
        mu, _ = scf.warp(attach_node_ind=attach, query_xyz=pw, query_tid=T0, target_tid=t, dyn_o_flag=True)
        pos[:, t] = mu.half()
        uv = cams.project(cams.trans_pts_to_cam(t, mu))
        v = vis[t, J]
        if v.any():
            ox, oy = trk[t, J, 0].clamp(0, W - 1), trk[t, J, 1].clamp(0, H - 1)
            r = (uv - homo[oy, ox]).norm(dim=-1) * px_per_unit
            res[v, t] = r[v]
    dt = (torch.arange(T, device=dev)[None] - T0[:, None]).abs().float()   # S,T
    def band_med(lo, hi):
        m = (dt > lo) & (dt <= hi) & ~torch.isnan(res)
        x = torch.where(m, res, torch.full_like(res, float("nan")))
        return torch.nanmedian(x, dim=1).values, m.any(1)
    r0, has0 = band_med(0, 2)
    thr = torch.clamp(FACTOR * torch.nan_to_num(r0, nan=FLOOR_PX), min=FLOOR_PX)
    reach = torch.full((S,), float(GRID[0]), device=dev)
    alive = torch.ones(S, dtype=torch.bool, device=dev)
    prev = 0
    for R in GRID + [T]:
        med, has = band_med(prev, R)
        fail = has & (med > thr)
        alive &= ~fail
        reach = torch.where(alive, torch.full_like(reach, float(R)), reach)
        prev = R
    reach = torch.where(reach >= T, torch.full_like(reach, float(T)), reach)
    vals, cnts = torch.unique(reach, return_counts=True)
    summ = {"S": S, "tracks": int(len(jid)), "r0_median_px": float(torch.nanmedian(r0)),
            "thr_median_px": float(thr.median()), "reach_hist": {str(int(v)): int(c) for v, c in zip(vals.tolist(), cnts.tolist())},
            "res_median_by_band": {}}
    prev = 0
    for R in GRID + [T]:
        med, has = band_med(prev, R)
        summ["res_median_by_band"][f"({prev},{R}]"] = float(torch.nanmedian(med[has])) if has.any() else None
        prev = R
    logging.info(f"fmd dy D1 reach samples: {json.dumps(summ)}")
    print("FMD_DY_REACH", json.dumps(summ), flush=True)
    return {"pos": pos, "reach": reach}
