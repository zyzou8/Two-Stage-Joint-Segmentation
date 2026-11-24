"""Fixed Gabor fringe features for Stage 2.  No learned parameters.

Three output channels, computed from the grayscale image in [0,1] BEFORE
ImageNet normalisation and BEFORE replication to 3 channels:

    0  fringe_energy  max over (theta, lambda) of the quadrature magnitude,
                      divided by the image's 99th percentile, clipped to [0,1]
    1  cos(2 theta*)
    2  sin(2 theta*)

theta* is the orientation of the maximum response, taken after the max over
lambda.  Orientation channels are plain cos/sin, not energy-weighted.

Bank: 8 orientations {0, 22.5, ..., 157.5} x lambda {16, 24, 36} px,
quadrature (even/odd) pair each, sigma = 0.56*lambda, odd kernel size ~6*sigma
(55, 81, 121), zero-mean kernels, reflect padding.

Kernels live in register_buffer (not nn.Parameter) and forward runs under
torch.no_grad(), so the extractor contributes no trainable parameters and no
gradient flows into it.

IMPLEMENTATION CHOICE worth flagging: kernels are L2-normalised after
zero-meaning.  The spec fixes zero-mean but not scale.  Without L2
normalisation the response grows with kernel size, so `max over lambda` would
always select lambda=36 and the multi-scale bank would be inert.  L2
normalisation makes the three scales directly comparable, which is what the
max requires.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LAMBDAS = (16.0, 24.0, 36.0)
N_THETA = 8
SIGMA_FACTOR = 0.56


def _odd(n: int) -> int:
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


def _gabor_bank(lmbda: float, n_theta: int = N_THETA):
    """[2*n_theta, 1, K, K] -- first n_theta are even (cos), rest odd (sin)."""
    sigma = SIGMA_FACTOR * lmbda
    K = _odd(6 * sigma)
    h = K // 2
    ax = torch.arange(-h, h + 1, dtype=torch.float64)
    y, x = torch.meshgrid(ax, ax, indexing="ij")

    even, odd = [], []
    for i in range(n_theta):
        th = math.pi * i / n_theta
        xp = x * math.cos(th) + y * math.sin(th)
        yp = -x * math.sin(th) + y * math.cos(th)
        env = torch.exp(-(xp ** 2 + yp ** 2) / (2 * sigma ** 2))
        e = env * torch.cos(2 * math.pi * xp / lmbda)
        o = env * torch.sin(2 * math.pi * xp / lmbda)
        e = e - e.mean()
        o = o - o.mean()                       # zero-mean
        e = e / e.norm().clamp_min(1e-12)
        o = o / o.norm().clamp_min(1e-12)      # L2 -> comparable across lambda
        even.append(e)
        odd.append(o)
    w = torch.stack(even + odd).unsqueeze(1).float()
    return w, K


class FringeFeatures(nn.Module):
    """Grayscale [B,1,H,W] in [0,1] -> [B,3,H,W] fringe features."""

    def __init__(self, lambdas=LAMBDAS, n_theta: int = N_THETA):
        super().__init__()
        self.lambdas = tuple(lambdas)
        self.n_theta = n_theta
        self.ksizes = []
        for i, lam in enumerate(self.lambdas):
            w, K = _gabor_bank(lam, n_theta)
            self.register_buffer(f"k{i}", w, persistent=False)
            self.ksizes.append(K)
        thetas = torch.arange(n_theta, dtype=torch.float32) * (math.pi / n_theta)
        self.register_buffer("cos2t", torch.cos(2 * thetas), persistent=False)
        self.register_buffer("sin2t", torch.sin(2 * thetas), persistent=False)

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[1] != 1:
            raise ValueError(f"FringeFeatures expects 1 channel, got {img.shape[1]}")
        x = img.float()

        mags = []
        for i, K in enumerate(self.ksizes):
            w = getattr(self, f"k{i}")
            h = K // 2
            xp = F.pad(x, (h, h, h, h), mode="reflect")
            r = F.conv2d(xp, w)                       # [B, 2*n_theta, H, W]
            e, o = r[:, :self.n_theta], r[:, self.n_theta:]
            mags.append(torch.sqrt(e * e + o * o + 1e-12))
        m = torch.stack(mags, 0).amax(0)              # max over lambda -> [B,n_theta,H,W]

        energy, idx = m.max(dim=1)                    # max over theta -> [B,H,W]
        cos2 = self.cos2t[idx]
        sin2 = self.sin2t[idx]

        # per-image 99th-percentile normalisation
        b = energy.shape[0]
        q = torch.quantile(energy.reshape(b, -1).float(), 0.99, dim=1)
        energy = (energy / q.clamp_min(1e-12).view(b, 1, 1)).clamp(0.0, 1.0)

        return torch.stack([energy, cos2, sin2], dim=1).to(img.dtype)
