"""fmd dy Pilot 1 (D1, fusion reach): patch a COPY of MoSca (repo/mosca_d1) with per-Gaussian temporal support.

Environment variables (read once when a DynSCFGaussian is built; stored as buffers so a reloaded model renders the same):
  DY_MODE = off      published all-frame fusion (variant 1)
          = window   hard support |t - ref_time| <= DY_R for every foreground Gaussian (variants 2, 3)
          = soft     all-frame fusion, opacity x exp(-max(0, |t-ref| - R_i) / R_i), R_i from the reliability rule
                     (variant 4: reliability WEIGHTING, the USplat4D-style control)
          = reach    hard support |t - ref_time| <= R_i, R_i from the reliability rule (variant 5)
          = shuffle  as reach, but R_i permuted at random within each batch of new Gaussians (variant 6, placebo)
  DY_R    window half-width in frames (variants 2, 3)
  DY_SEED seed for the shuffle
Reliability rule (ONE fixed rule, training-only evidence, computed before the photometric fit):
  every moving BootsTAPIR track is sampled at up to 6 visible reference frames t0 with valid depth; the point is lifted
  with the training depth and camera, attached to its nearest scaffold node at t0, warped with MoSca's own scaffold
  to every frame t, projected into training camera t and compared with the track's observed position where visible.
  Short-range error r0 = median residual over 0 < |t - t0| <= 2. Reach = the largest R in {2,4,8,16,32,64,128,all}
  such that the median residual in every distance band (previous R, R] stays <= max(2 px, 2 * r0)
  ("support extends while the warp error stays within twice its short-range value").
  A new Gaussian at frame tid gets the median reach of its 8 nearest samples (sample positions warped to tid);
  clones/splits inherit the parent's value; pruning filters it like ref_time. Static Gaussians are untouched.
Support acts like MoSca's own nn_fusion option: by masking (or weighting) the Gaussian's opacity at frame t, in
training and in rendering.
"""
import re, sys
root = sys.argv[1]
p = f"{root}/lib_mosca/dynamic_gs.py"; s = open(p).read(); n0 = len(s)

def sub(old, new, count=1):
    global s
    c = s.count(old)
    assert c == count, (old[:80], c, count)
    s = s.replace(old, new)

# 1. buffers
sub('''        self.register_buffer("ref_time", torch.zeros(init_N).long())  # N
        # * init states''',
'''        self.register_buffer("ref_time", torch.zeros(init_N).long())  # N
        # * fmd dy D1: per-Gaussian temporal support (see lib_mosca/dy_support.py)
        import os as _os
        _m = {"off": 0, "window": 1, "reach": 2, "shuffle": 2, "soft": 3}[_os.environ.get("DY_MODE", "off")]
        self.register_buffer("support_R", torch.zeros(init_N).float())
        self.register_buffer("dy_mode", torch.tensor(_m))
        self.register_buffer("dy_R", torch.tensor(float(_os.environ.get("DY_R", "-1"))))
        self._dy_shuffle = _os.environ.get("DY_MODE", "off") == "shuffle"
        self._dy_gen = torch.Generator().manual_seed(int(_os.environ.get("DY_SEED", "0")))
        self._dy_samples = None
        # * init states''')
# 2. creation sites (append_new_gs and the node-growing append): identical text, twice
sub('''        self.ref_time = torch.cat(
            [self.ref_time, torch.ones_like(attach_ind[0, :, 0]) * tid], dim=0
        )
''', '''        self.ref_time = torch.cat(
            [self.ref_time, torch.ones_like(attach_ind[0, :, 0]) * tid], dim=0
        )
        self.support_R = torch.cat([self.support_R, self._dy_assign(mu_w, tid)], dim=0)
''', count=2)
# 3. clone by index
sub('''        self.ref_time = torch.cat(
            [self.ref_time, self.ref_time.clone()[new_gs_ind]], dim=0
        )
''', '''        self.ref_time = torch.cat(
            [self.ref_time, self.ref_time.clone()[new_gs_ind]], dim=0
        )
        self.support_R = torch.cat([self.support_R, self.support_R.clone()[new_gs_ind]], dim=0)
''')
# 4. densify clone / split
sub('''        new_ref_time = self.ref_time[selected_pts_mask]
''', '''        new_ref_time = self.ref_time[selected_pts_mask]
        new_support_R = self.support_R[selected_pts_mask]
''')
sub('''        new_ref_time = self.ref_time[selected_pts_mask].repeat(N)
''', '''        new_ref_time = self.ref_time[selected_pts_mask].repeat(N)
        new_support_R = self.support_R[selected_pts_mask].repeat(N)
''')
sub('''        self.ref_time = torch.cat([self.ref_time, new_ref_time], dim=0)
''', '''        self.ref_time = torch.cat([self.ref_time, new_ref_time], dim=0)
        self.support_R = torch.cat([self.support_R, new_support_R], dim=0)
''', count=2)
# 5. prune
sub('''        self.ref_time = self.ref_time[valid_points_mask]
''', '''        self.ref_time = self.ref_time[valid_points_mask]
        self.support_R = self.support_R[valid_points_mask]
''')
# 6. apply in forward (after opacity is read, before nn_fusion)
sub('''        s = self.get_s
        o = self.get_o
        sph_dim = 3 * sph_order2nfeat(active_sph_order)''', '''        s = self.get_s
        o = self.get_o
        o = self._dy_apply(o, t)  # fmd dy D1
        sph_dim = 3 * sph_order2nfeat(active_sph_order)''')
# 7. methods
sub('''    def local_gs_warp(''', '''    @torch.no_grad()
    def _dy_assign(self, mu_w, tid):
        n = len(mu_w)
        if int(self.dy_mode) in (0, 1) or self._dy_samples is None:
            return torch.full((n,), float(self.T), device=mu_w.device)
        sm = self._dy_samples
        pos = sm["pos"][:, int(tid)].float()
        _, idx, _ = knn_points(mu_w[None].float(), pos[None], K=min(8, len(pos)))
        r = sm["reach"][idx[0]].float().median(dim=1).values
        if self._dy_shuffle:
            r = r[torch.randperm(n, generator=self._dy_gen).to(r.device)]
        return r.to(mu_w.device)

    def _dy_apply(self, o, t):
        m = int(self.dy_mode)
        if m == 0:
            return o
        d = (self.ref_time.float() - float(t)).abs()
        if m == 1:
            w = (d <= self.dy_R).float()
        elif m == 2:
            w = (d <= self.support_R).float()
        else:
            R = self.support_R.clamp(min=1.0)
            w = torch.exp(-(d - R).clamp(min=0.0) / R)
        return o * w[:, None]

    def local_gs_warp(''')
open(p, "w").write(s); print("dynamic_gs.py patched", n0, "->", len(s))

# 8. compute the reliability samples before the leaves are appended
p2 = f"{root}/lib_mosca/photo_recon.py"; s = open(p2).read()
sub('''        d_model.to(device)

        # * Init the leaves''', '''        d_model.to(device)
        if int(d_model.dy_mode) in (2, 3):  # fmd dy D1: training-only reliability evidence
            from lib_mosca.dy_support import compute_reach_samples
            d_model._dy_samples = compute_reach_samples(s2d, cams, scf)

        # * Init the leaves''')
open(p2, "w").write(s); print("photo_recon.py patched")
