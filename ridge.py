"""Fixed Hessian dark-line ridge features for Stage 2.  No learned parameters.

Motivated by the measurement in RESULTS.md: annotated SF are same-polarity DARK
lines (89.5% darker than local background, median contrast -0.089 on [0,1]), not
an oscillating fringe pattern.  A ridge detector matches that evidence; a Gabor
quadrature bank does not.

Three output channels, from the grayscale image in [0,1] BEFORE ImageNet
normalisation and BEFORE replication to 3 channels:

    0  ridge_strength  max over sigma of R_sigma, divided by the image's 99th
                       percentile, clipped to [0,1]
    1  cos(2 theta*)
    2  sin(2 theta*)

For each sigma in {1, 2, 4}: Ixx, Iyy, Ixy from separable Gaussian-derivative
kernels (size ~6 sigma, odd, reflect padding), Hessian eigenvalues
lambda_max >= lambda_min, and

    R_sigma = sigma^2 * max(lambda_max, 0)

which fires on dark lines over a bright background only (a dark line is a local
intensity minimum across its width, so the across-line second derivative is
positive).  theta* is the LINE direction -- the lambda_min eigenvector -- taken
at the sigma giving the maximum response.

Kernels live in register_buffer (not nn.Parameter) and forward runs under
torch.no_grad(), so the extractor has no trainable parameters and no gradient
flows into it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SIGMAS = (1.0, 2.0, 4.0)


def _odd(n: int) -> int:
    n = max(3, int(round(n)))
    return n if n % 2 == 1 else n + 1


def _gauss_derivs(sigma: float):
    """(g, g', g'') sampled on an odd grid of length ~6 sigma."""
    K = _odd(6 * sigma)
    h = K // 2
    x = torch.arange(-h, h + 1, dtype=torch.float64)
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()                                   # unit area
    g1 = -(x / sigma ** 2) * g
    g2 = ((x ** 2 - sigma ** 2) / sigma ** 4) * g
    g1 = g1 - g1.mean()
    g2 = g2 - g2.mean()                               # zero-mean derivative kernels
    return g.float(), g1.float(), g2.float(), K


class RidgeFeatures(nn.Module):
    """Grayscale [B,1,H,W] in [0,1] -> [B,3,H,W] dark-line ridge features."""

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
        """Separable conv: kx along x (dim 3), ky along y (dim 2).  Reflect padded."""
        K = kx.shape[-1]
        h = K // 2
        x = F.pad(x, (h, h, 0, 0), mode="reflect")
        x = F.conv2d(x, kx)
        x = F.pad(x, (0, 0, h, h), mode="reflect")
        return F.conv2d(x, ky.transpose(2, 3))

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[1] != 1:
            raise ValueError(f"RidgeFeatures expects 1 channel, got {img.shape[1]}")
        x = img.float()

        Rs, C2, S2 = [], [], []
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
            Rs.append((s ** 2) * lmax.clamp_min(0.0))

            # principal axis of lambda_max; the LINE runs perpendicular to it
            th_max = 0.5 * torch.atan2(2 * Ixy, Ixx - Iyy)
            th_line = th_max + math.pi / 2
            C2.append(torch.cos(2 * th_line))
            S2.append(torch.sin(2 * th_line))

        R = torch.cat(Rs, dim=1)                       # [B, n_sigma, H, W]
        strength, idx = R.max(dim=1, keepdim=True)
        cos2 = torch.cat(C2, dim=1).gather(1, idx)
        sin2 = torch.cat(S2, dim=1).gather(1, idx)

        b = strength.shape[0]
        q = torch.quantile(strength.reshape(b, -1).float(), 0.99, dim=1)
        strength = (strength / q.clamp_min(1e-12).view(b, 1, 1, 1)).clamp(0.0, 1.0)

        return torch.cat([strength, cos2, sin2], dim=1).to(img.dtype)
