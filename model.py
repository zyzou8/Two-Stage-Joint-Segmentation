"""EffUNet in PyTorch, and the two-stage GB -> SF wrapper.

Architecture mirrors the earlier Keras EffUNet of the SF-only pipeline
(`encoder_name='efficientnetb0'`, `decoder_name='unet'`).  That model is a
reference only -- nothing here imports Keras.

Encoder tap points, verified against torchvision efficientnet_b0:

    features[1]   16 ch   1/2     <- Keras block2a_expand_activation
    features[2]   24 ch   1/4     <- Keras block3a_expand_activation
    features[3]   40 ch   1/8     <- Keras block4a_expand_activation
    features[5]  112 ch   1/16    <- Keras block6a_expand_activation
    features[8] 1280 ch   1/32    <- Keras top_activation

The stem is stride 2, so the decoder needs one extra upsample at the end to get
back to full resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b7, EfficientNet_B7_Weights,
)

import config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Encoder registry: builder, ImageNet weights, and tap points as
# (index into .features, output channels), fine -> coarse.
# Both variants tap the same block indices; only the widths differ.
# Verified empirically against torchvision, not assumed.
ENCODERS = {
    "efficientnet_b0": (
        efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1,
        ((1, 16), (2, 24), (3, 40), (5, 112), (8, 1280)),
    ),
    "efficientnet_b7": (
        efficientnet_b7, EfficientNet_B7_Weights.IMAGENET1K_V1,
        ((1, 32), (2, 48), (3, 80), (5, 224), (8, 2560)),
    ),
}


def _norm(ch: int) -> nn.Module:
    """GroupNorm, as models.py uses -- far more stable than BatchNorm at the
    batch sizes 1024x1024 forces you into."""
    return nn.GroupNorm(config.NORM_GROUPS, ch)


class ConvBlock(nn.Module):
    """Two 3x3 conv + norm + relu, then spatial dropout.  Mirrors conv_block()."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            _norm(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _norm(out_ch), nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.body(x))


class UpBlock(nn.Module):
    """Bilinear x2 then 3x3 conv + norm + relu.  Mirrors upsample_block()."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm = _norm(out_ch)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return F.relu(self.norm(self.conv(x)), inplace=True)


class EffUNet(nn.Module):
    """EfficientNet-B0 encoder + U-Net decoder, single sigmoid output.

    Parameters
    ----------
    extra_channels : int
        Planes appended to the (RGB-replicated) image.  0 for stage 1,
        1 for stage 2's `gb_prob`.

    The grayscale image is replicated to 3 channels and ImageNet-normalised so
    the pretrained stem sees the statistics it was trained on.  Extra planes are
    appended raw -- they are already in [0,1] -- and their stem weights are
    zero-initialised, so at initialisation the network behaves exactly as if
    they were absent and then learns to use them.  This is what lets stage 2
    take `gb_prob` without discarding ImageNet weights; the Keras original
    could not (INSPECTION.md 3.3).
    """

    def __init__(self, extra_channels: int = 0, pretrained: bool | None = None,
                 decoder_filters=config.DECODER_FILTERS, dropout: float = config.DROPOUT,
                 encoder: str | None = None, cond_norm: str | None = None,
                 deep_cond: bool | None = None):
        super().__init__()
        self.extra_channels = extra_channels
        self.cond_norm = config.COND_NORM if cond_norm is None else cond_norm
        deep = config.DEEP_COND if deep_cond is None else deep_cond
        self.deep_cond = bool(deep) and extra_channels > 0

        # resolved at call time, not import time, so config stays patchable
        if pretrained is None:
            pretrained = config.PRETRAINED
        if encoder is None:
            encoder = config.ENCODER
        if encoder not in ENCODERS:
            raise ValueError(f"unknown encoder '{encoder}'; have {sorted(ENCODERS)}")

        builder, w, taps = ENCODERS[encoder]
        self.encoder_name = encoder
        self.taps = taps
        net = builder(weights=w if pretrained else None)
        self.features = net.features

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        if extra_channels > 0:
            self._inflate_stem(extra_channels)

        skip_ch = [c for _, c in taps]            # b0: [16, 24, 40, 112, 1280]
        f = list(decoder_filters)                 # [512, 256, 128, 64, 32]

        # coarse -> fine: 1/32 -> 1/16 -> 1/8 -> 1/4 -> 1/2
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        in_ch = skip_ch[-1]
        # with deep conditioning each decoder stage also receives gb_prob,
        # bilinearly resized to that stage's resolution
        cond_in = extra_channels if self.deep_cond else 0
        for i, skip in enumerate(reversed(skip_ch[:-1])):   # 112, 40, 24, 16
            self.ups.append(UpBlock(in_ch, f[i]))
            self.decs.append(ConvBlock(f[i] + skip + cond_in, f[i], dropout))
            in_ch = f[i]

        # stem was stride 2 -- one more upsample to reach full resolution
        self.final_up = UpBlock(in_ch, f[-1])
        self.head = nn.Conv2d(f[-1], 1, 1)

    def _inflate_stem(self, extra: int) -> None:
        """Widen the stem conv from 3 to 3+extra input channels.

        Pretrained RGB filters are copied verbatim; the new columns start at
        zero so they contribute nothing until trained.
        """
        old = self.features[0][0]
        new = nn.Conv2d(3 + extra, old.out_channels,
                        kernel_size=old.kernel_size, stride=old.stride,
                        padding=old.padding, bias=old.bias is not None)
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            new.weight[:, 3:] = 0.0
            if old.bias is not None:
                new.bias.copy_(old.bias)
        self.features[0][0] = new

    def forward(self, img: torch.Tensor, extra: torch.Tensor | None = None):
        """img: [B,1,H,W] in [0,1].  extra: [B,extra_channels,H,W] or None.

        Returns sigmoid probabilities [B,1,H,W].
        """
        if img.shape[1] != 1:
            raise ValueError(f"expected 1-channel image, got {img.shape[1]}")
        x = img.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std

        cond = None
        if self.extra_channels > 0:
            if extra is None or extra.shape[1] != self.extra_channels:
                raise ValueError(
                    f"model built with extra_channels={self.extra_channels}, "
                    f"got {None if extra is None else extra.shape[1]}")
            cond = extra
            if self.cond_norm == "center":
                # match the image channels' scale: [0,1] -> [-1,1]
                cond = (cond - 0.5) / 0.5
            elif self.cond_norm != "raw":
                raise ValueError(f"unknown cond_norm '{self.cond_norm}'")
            x = torch.cat([x, cond], dim=1)
        elif extra is not None:
            raise ValueError("model has extra_channels=0 but `extra` was passed")

        skips = []
        tap_idx = {i for i, _ in self.taps}
        for i, blk in enumerate(self.features):
            x = blk(x)
            if i in tap_idx:
                skips.append(x)

        x = skips[-1]
        for up, dec, skip in zip(self.ups, self.decs, reversed(skips[:-1])):
            x = up(x)
            parts = [x, skip]
            if self.deep_cond:
                parts.append(F.interpolate(cond, size=x.shape[-2:],
                                           mode="bilinear", align_corners=False))
            x = dec(torch.cat(parts, dim=1))

        x = self.final_up(x)
        return torch.sigmoid(self.head(x))


class TwoStage(nn.Module):
    """Stage 1: image -> gb_prob.  Stage 2: (image, gb_prob) -> sf_prob.

    `gb_for_stage2` selects what stage 2 is conditioned on.  Training mixes
    ground truth and prediction per step (teacher forcing, decision 3); at test
    time it is always the prediction.

    No thresholding between stages, and no detach -- the SF loss gradient flows
    back into stage 1, which is what makes the two stages coupled rather than
    merely chained.
    """

    def __init__(self, pretrained: bool | None = None, encoder: str | None = None,
                 cond_norm: str | None = None, deep_cond: bool | None = None):
        super().__init__()
        self.stage1 = EffUNet(extra_channels=0, pretrained=pretrained, encoder=encoder)
        self.stage2 = EffUNet(extra_channels=1, pretrained=pretrained, encoder=encoder,
                              cond_norm=cond_norm, deep_cond=deep_cond)

    def forward(self, img: torch.Tensor, gb_target: torch.Tensor | None = None,
                p_teacher: float = 0.0):
        gb_prob = self.stage1(img)

        if self.training and gb_target is not None and p_teacher > 0.0:
            # one Bernoulli draw per sample, not per pixel
            b = img.shape[0]
            use_gt = (torch.rand(b, 1, 1, 1, device=img.device) < p_teacher).float()
            cond = use_gt * gb_target + (1.0 - use_gt) * gb_prob
        else:
            cond = gb_prob

        sf_prob = self.stage2(img, cond)
        return gb_prob, sf_prob


class SharedTwoStage(nn.Module):
    """`joint_shared`: ONE Eff-UNet, run twice.  This is the architecture the
    paper describes.

        "A single Eff-UNet ... is run twice.  Its input always has four
         channels, the three image channels and a GB channel, which is empty in
         Stage 1 and holds B in Stage 2."

    Pass 1   net(image, 0)      -> gb_prob      (GB channel empty)
    Pass 2   net(image, cond)   -> sf_prob      (GB channel holds B)

    The GB channel is therefore also the TASK SWITCH: the same weights and the
    same single sigmoid head produce boundaries when the channel is empty and
    planar defects when it is filled.  Because the weights are shared, the PD
    loss back-propagates into the one network through both passes, which is the
    dashed arrow in the pipeline figure.

    Contrast `TwoStage`, which allocates a separate 18.4M-parameter network per
    stage (36.8M total).  This one is 18.4M and ties them.

    Note on initialisation: `EffUNet` zero-initialises the stem column for the
    extra channel, so at step 0 the two passes are bit-identical and the task
    switch is inert.  It becomes active as soon as that column takes a gradient.
    The behaviour is asserted in `selfcheck_shared()`.
    """

    def __init__(self, pretrained: bool | None = None, encoder: str | None = None,
                 cond_norm: str | None = None, deep_cond: bool | None = None):
        super().__init__()
        self.net = EffUNet(extra_channels=1, pretrained=pretrained, encoder=encoder,
                           cond_norm=cond_norm, deep_cond=deep_cond)

    def forward(self, img: torch.Tensor, gb_target: torch.Tensor | None = None,
                p_teacher: float = 0.0):
        b, _, h, w = img.shape
        empty = img.new_zeros(b, 1, h, w)
        gb_prob = self.net(img, empty)                 # stage 1: GB channel empty

        if self.training and gb_target is not None and p_teacher > 0.0:
            use_gt = (torch.rand(b, 1, 1, 1, device=img.device) < p_teacher).float()
            cond = use_gt * gb_target + (1.0 - use_gt) * gb_prob
        else:
            cond = gb_prob

        sf_prob = self.net(img, cond)                  # stage 2: GB channel holds B
        return gb_prob, sf_prob


class SingleStage(nn.Module):
    """Baseline: image -> one mask.  Used for the `Single GB` and `Single SF`
    rows of the results table (decision 5)."""

    def __init__(self, pretrained: bool | None = None, encoder: str | None = None):
        super().__init__()
        self.net = EffUNet(extra_channels=0, pretrained=pretrained, encoder=encoder)

    def forward(self, img: torch.Tensor):
        return self.net(img)


class TwoStageFringe(nn.Module):
    """`joint_fringe`: identical to TwoStage except Stage 2 also receives 3 fixed
    Gabor fringe channels.

    Stage 2 input = concat(image 3ch ImageNet-normalised, gb_cond 1ch, fringe 3ch)
    = 7 channels.  The stem is inflated 3 -> 7; pretrained RGB columns are copied
    and ALL four extra columns (gb_prob + 3 fringe) are zero-initialised, so at
    init this is numerically identical to `joint`.

    Fringe features are computed inside forward() from the already-augmented
    batch, so they always match the image geometry.  They depend on the image
    only -- identical under either teacher-forcing branch.
    """

    def __init__(self, pretrained: bool | None = None, encoder: str | None = None):
        super().__init__()
        from fringe import FringeFeatures
        self.stage1 = EffUNet(extra_channels=0, pretrained=pretrained, encoder=encoder)
        # extra = 1 (gb_cond) + 3 (fringe); cond_norm='raw' and deep_cond=False to
        # match the `joint` baseline in runs/ exactly
        self.stage2 = EffUNet(extra_channels=4, pretrained=pretrained, encoder=encoder,
                              cond_norm="raw", deep_cond=False)
        self.fringe = FringeFeatures()

    def forward(self, img: torch.Tensor, gb_target: torch.Tensor | None = None,
                p_teacher: float = 0.0):
        gb_prob = self.stage1(img)

        if self.training and gb_target is not None and p_teacher > 0.0:
            b = img.shape[0]
            use_gt = (torch.rand(b, 1, 1, 1, device=img.device) < p_teacher).float()
            cond = use_gt * gb_target + (1.0 - use_gt) * gb_prob
        else:
            cond = gb_prob

        fr = self.fringe(img)                      # [B,3,H,W], no grad, no params
        sf_prob = self.stage2(img, torch.cat([cond, fr], dim=1))
        return gb_prob, sf_prob


class TwoStageRidge(nn.Module):
    """`joint_ridge`: identical to TwoStageFringe except the 3 extra Stage-2
    channels come from a fixed Hessian dark-line ridge filter instead of a Gabor
    bank.  Stage 2 input = image(3) + gb_cond(1) + ridge(3) = 7 channels."""

    def __init__(self, pretrained: bool | None = None, encoder: str | None = None):
        super().__init__()
        from ridge import RidgeFeatures
        self.stage1 = EffUNet(extra_channels=0, pretrained=pretrained, encoder=encoder)
        self.stage2 = EffUNet(extra_channels=4, pretrained=pretrained, encoder=encoder,
                              cond_norm="raw", deep_cond=False)
        self.ridge = RidgeFeatures()

    def forward(self, img: torch.Tensor, gb_target: torch.Tensor | None = None,
                p_teacher: float = 0.0):
        gb_prob = self.stage1(img)
        if self.training and gb_target is not None and p_teacher > 0.0:
            b = img.shape[0]
            use_gt = (torch.rand(b, 1, 1, 1, device=img.device) < p_teacher).float()
            cond = use_gt * gb_target + (1.0 - use_gt) * gb_prob
        else:
            cond = gb_prob
        rg = self.ridge(img)
        sf_prob = self.stage2(img, torch.cat([cond, rg], dim=1))
        return gb_prob, sf_prob


class TwoStageSigned(nn.Module):
    """`joint_signed`: Stage 2 gets gb_cond plus TWO signed ridge channels
    (dark_ridge for SF-like dips, bright_ridge for GB-like bright lines).
    Stage 2 input = image(3) + gb_cond(1) + dark(1) + bright(1) = 6 channels."""

    def __init__(self, pretrained: bool | None = None, encoder: str | None = None):
        super().__init__()
        from signed_ridge import SignedRidgeFeatures
        self.stage1 = EffUNet(extra_channels=0, pretrained=pretrained, encoder=encoder)
        self.stage2 = EffUNet(extra_channels=3, pretrained=pretrained, encoder=encoder,
                              cond_norm="raw", deep_cond=False)
        self.sridge = SignedRidgeFeatures()

    def forward(self, img: torch.Tensor, gb_target: torch.Tensor | None = None,
                p_teacher: float = 0.0):
        gb_prob = self.stage1(img)
        if self.training and gb_target is not None and p_teacher > 0.0:
            b = img.shape[0]
            use_gt = (torch.rand(b, 1, 1, 1, device=img.device) < p_teacher).float()
            cond = use_gt * gb_target + (1.0 - use_gt) * gb_prob
        else:
            cond = gb_prob
        sr = self.sridge(img)
        sf_prob = self.stage2(img, torch.cat([cond, sr], dim=1))
        return gb_prob, sf_prob
