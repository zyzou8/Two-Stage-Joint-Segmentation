"""Soft Dice on probabilities.

Loss = Dice(gb_prob, gb_mask) + Dice(sf_prob, sf_mask).  No other terms --
the spec is explicit about this.
"""

from __future__ import annotations

import torch

import config


def soft_dice_loss(prob: torch.Tensor, target: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
    """1 - Dice, computed per-sample on probabilities and then averaged.

    `prob` is already sigmoid output -- no thresholding anywhere, per spec.
    Averaging per-sample rather than pooling the whole batch keeps a
    near-empty mask in one image from being swamped by a dense one in another.
    """
    if prob.shape != target.shape:
        raise ValueError(f"shape mismatch: prob {tuple(prob.shape)} vs "
                         f"target {tuple(target.shape)}")

    dims = tuple(range(1, prob.ndim))          # everything but batch
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def batch_dice_loss(prob: torch.Tensor, target: torch.Tensor,
                    eps: float = 1e-6) -> torch.Tensor:
    """1 - Dice pooled over the whole batch.

    With random crops some windows contain no SF at all.  Per-sample Dice on an
    empty target is degenerate (it rewards predicting nothing and the gradient
    is dominated by those crops), so intersections and sums are accumulated
    across every pixel of every sample before the ratio is taken.
    """
    if prob.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(prob.shape)} vs {tuple(target.shape)}")
    inter = (prob * target).sum()
    denom = prob.sum() + target.sum()
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


_GK = {}


def _gauss_kernel(sigma: float, device, dtype):
    """Cached 1-D Gaussian kernel, truncated at 3 sigma."""
    key = (round(sigma, 4), device, dtype)
    if key not in _GK:
        r = max(1, int(round(3 * sigma)))
        x = torch.arange(-r, r + 1, device=device, dtype=dtype)
        k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        _GK[key] = (k / k.sum()).view(1, 1, 1, -1)
    return _GK[key]


def soft_target(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """clip(GaussianBlur(mask, sigma) / 0.5, 0, 1).

    Widens the SF target by roughly one sigma so a prediction that is correct
    to within a pixel or two is not penalised as hard as one that misses
    entirely.  The ~6 px fault width and the 4 px matching tolerance used at
    evaluation make a 1 px slack reasonable.  Evaluation still uses the
    original binary mask.
    """
    k = _gauss_kernel(sigma, mask.device, mask.dtype)
    r = k.shape[-1] // 2
    x = torch.nn.functional.pad(mask, (r, r, 0, 0), mode="reflect")
    x = torch.nn.functional.conv2d(x, k)
    x = torch.nn.functional.pad(x, (0, 0, r, r), mode="reflect")
    x = torch.nn.functional.conv2d(x, k.transpose(2, 3))
    return (x / 0.5).clamp(0.0, 1.0)


def dice_bce_loss(prob: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    """Batch Dice + BCE against a soft target, equal weights.

    BCE is forced to float32: autocast refuses binary_cross_entropy in fp16.
    """
    d = batch_dice_loss(prob, soft)
    with torch.cuda.amp.autocast(enabled=False):
        p = prob.float().clamp(1e-6, 1 - 1e-6)
        b = torch.nn.functional.binary_cross_entropy(p, soft.float())
    return d + b


def joint_loss(gb_prob: torch.Tensor, gb_target: torch.Tensor,
               sf_prob: torch.Tensor, sf_target: torch.Tensor):
    """Returns (total, gb_component, sf_component) for logging."""
    f = batch_dice_loss if config.BATCH_DICE else soft_dice_loss
    gb = f(gb_prob, gb_target)
    if config.SOFT_SF_SIGMA:
        sf = dice_bce_loss(sf_prob, soft_target(sf_target, config.SOFT_SF_SIGMA))
    else:
        sf = f(sf_prob, sf_target)
    return gb + sf, gb, sf
