"""Signed Hessian ridge features: separate dark-line and bright-line channels.

Motivated by the profile measurement in RESULTS.md: SF sit in an intensity DIP
(median dip depth +0.054, 79.8% positive) while GB are predominantly
BRIGHT-centred (median -0.036, only 27.0% positive).  `ridge.py` keeps only
max(lambda_max, 0), which fires on dark lines and on the dark flanks bracketing a
bright boundary, so it conflates the two classes.  Splitting the Hessian spectrum
into signed halves separates them.

Two output channels, from the grayscale image in [0,1] BEFORE ImageNet
normalisation and BEFORE replication to 3 channels:

    0  dark_ridge    max over sigma of sigma^2 * max( lambda_max, 0)   -> SF
    1  bright_ridge  max over sigma of sigma^2 * max(-lambda_min, 0)   -> GB

Each normalised by its OWN per-image 99th percentile and clipped to [0,1].
No orientation channels.

Scales sigma in {2, 3, 4} px.  sigma = 1 is dropped: the measured image noise is
comparable to the structure contrast at that scale, so sigma=1 mostly amplifies
noise.

Kernels live in register_buffer (not nn.Parameter) and forward runs under
torch.no_grad(): no trainable parameters, no gradient into the kernels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SIGMAS = (2.0, 3.0, 4.0)


def _odd(n: int) -> int:
    n = max(3, int(round(n)))
    return n if n % 2 == 1 else n + 1


def _gauss_derivs(sigma: float):
    K = _odd(6 * sigma)
    h = K // 2
    x = torch.arange(-h, h + 1, dtype=torch.float64)
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    g1 = -(x / sigma ** 2) * g
    g2 = ((x ** 2 - sigma ** 2) / sigma ** 4) * g
    return g.float(), (g1 - g1.mean()).float(), (g2 - g2.mean()).float(), K


class SignedRidgeFeatures(nn.Module):
    """Grayscale [B,1,H,W] in [0,1] -> [B,2,H,W]: (dark_ridge, bright_ridge)."""

    def __init__(self, sigmas=SIGMAS):
        super().__init__()
        self.sigmas = tuple(sigmas)
        self.ksizes = []
        for i, s in enumerate(self.sigmas):
            g, g1, g2, K = _gauss_derivs(s)
            self.register_buffer(f"g{i}", g.view(1, 1, 1, K), persistent=False)
            self.register_buffer(f"d1_{i}", g1.view(1, 1, 1, K), persistent=False)
            self.register_buffer(f"d2_{i}", g2.view(1, 1, 1, K), persistent=False)
            self.ksizes.append(K)

    @staticmethod
    def _sep(x, kx, ky):
        K = kx.shape[-1]
        h = K // 2
        x = F.pad(x, (h, h, 0, 0), mode="reflect")
        x = F.conv2d(x, kx)
        x = F.pad(x, (0, 0, h, h), mode="reflect")
        return F.conv2d(x, ky.transpose(2, 3))

    @staticmethod
    def _norm99(t):
        b = t.shape[0]
        q = torch.quantile(t.reshape(b, -1).float(), 0.99, dim=1)
        return (t / q.clamp_min(1e-12).view(b, 1, 1, 1)).clamp(0.0, 1.0)

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[1] != 1:
            raise ValueError(f"SignedRidgeFeatures expects 1 channel, got {img.shape[1]}")
        x = img.float()

        dark, bright = [], []
        for i, s in enumerate(self.sigmas):
            g = getattr(self, f"g{i}")
            d1 = getattr(self, f"d1_{i}")
            d2 = getattr(self, f"d2_{i}")
            Ixx = self._sep(x, d2, g)
            Iyy = self._sep(x, g, d2)
            Ixy = self._sep(x, d1, d1)
            tr = Ixx + Iyy
            dsc = torch.sqrt(((Ixx - Iyy) * 0.5) ** 2 + Ixy ** 2 + 1e-12)
            lmax = tr * 0.5 + dsc
            lmin = tr * 0.5 - dsc
            dark.append((s ** 2) * lmax.clamp_min(0.0))
            bright.append((s ** 2) * (-lmin).clamp_min(0.0))

        d = torch.cat(dark, dim=1).max(dim=1, keepdim=True)[0]
        b = torch.cat(bright, dim=1).max(dim=1, keepdim=True)[0]
        return torch.cat([self._norm99(d), self._norm99(b)], dim=1).to(img.dtype)
