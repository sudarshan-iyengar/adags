"""Oracle O4b (plan page section 30): a learned per-TRAINING-camera colour transform (FreeTimeGS++-style).

Each training camera c gets a 3x3 matrix A_c (initialised to identity) and an offset b_c (initialised to zero),
applied to the render before the photometric loss, so that colour differences between training cameras are absorbed
here instead of in the scene. A small L2 pull towards identity keeps it from drifting. Training only: test renders
(render.py) never import this module, so held-out views are rendered with the identity transform.
"""
import torch


class CamColour:
    def __init__(self, n=64, lr=1e-3, reg=1e-2):
        self.I = torch.eye(3, device="cuda")
        self.A = self.I.repeat(n, 1, 1).clone().requires_grad_(True)
        self.b = torch.zeros(n, 3, device="cuda", requires_grad=True)
        self.opt = torch.optim.Adam([self.A, self.b], lr=lr)
        self.reg = reg
        self.ids = {}

    def _i(self, key):
        return self.ids.setdefault(int(key), len(self.ids))

    def apply(self, img, key):
        """img: 3xHxW render of training camera `key`."""
        i = self._i(key)
        return torch.einsum("ij,jhw->ihw", self.A[i], img) + self.b[i][:, None, None]

    def reg_loss(self):
        return self.reg * (((self.A - self.I) ** 2).sum() + (self.b ** 2).sum())

    def step(self):
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

    def state(self):
        return {"ids": dict(self.ids), "A": self.A.detach().cpu().tolist(), "b": self.b.detach().cpu().tolist()}
