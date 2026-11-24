"""Joint augmentation for (image, gb_mask, sf_mask).

Ported from the earlier Keras SF-only pipeline, with two deliberate changes
from that reference:

1.  Every geometric op is applied to BOTH masks with the same sampled
    parameters, so image / gb_mask / sf_mask stay registered.  The reference
    handled a single mask.

2.  Output range is [0, 1] float32, not 0-255.  The reference targeted Keras
    EfficientNet, which carries its own Rescaling layer and wants raw 0-255;
    torchvision's EfficientNet wants [0,1] plus ImageNet mean/std, which
    model.py applies.

The masks arrive already binarised by data.py (threshold 127) -- do NOT
re-binarise with `> 0.5` the way the reference pipeline did.  These masks are
0-255 on disk, so `> 0.5` would keep every anti-aliasing pixel and inflate the
GB target from ~13% to ~20% of the image.

Pure numpy + scipy, no albumentations.  order=1 for the image, order=0 for
masks, so masks stay strictly binary.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    map_coordinates,
    gaussian_filter,
    rotate as nd_rotate,
    zoom as nd_zoom,
)


# --------------------------------------------------------------------------
# intensity normalisation -- applied to every sample, train and eval
# --------------------------------------------------------------------------

def percentile_normalize(img: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    """Clip to percentiles, rescale to [0, 1] float32.

    More robust than min/max: a handful of hot pixels will not squash the
    histogram.  TEM images vary a lot in absolute intensity between sessions.
    """
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [low, high])
    if hi - lo < 1e-6:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


# --------------------------------------------------------------------------
# geometric ops -- image + both masks, one shared set of parameters
# --------------------------------------------------------------------------

def random_d4(img, gb, sf, rng):
    """One of the 8 dihedral transforms (4 rotations x optional flip).

    The only augmentation that is exact -- no resampling, no interpolation.
    With cropping disabled (decision 8) this carries most of the geometric
    variety, which is why it runs at p=1.0.
    """
    k = int(rng.integers(0, 4))
    img, gb, sf = np.rot90(img, k), np.rot90(gb, k), np.rot90(sf, k)
    if rng.random() < 0.5:
        img, gb, sf = np.fliplr(img), np.fliplr(gb), np.fliplr(sf)
    return (np.ascontiguousarray(img),
            np.ascontiguousarray(gb),
            np.ascontiguousarray(sf))


def random_rotate(img, gb, sf, max_deg, rng):
    angle = float(rng.uniform(-max_deg, max_deg))
    kw = dict(reshape=False, mode="reflect")
    img = nd_rotate(img, angle, order=1, **kw)
    gb = nd_rotate(gb, angle, order=0, **kw)
    sf = nd_rotate(sf, angle, order=0, **kw)
    return img, gb, sf


def _fit(arr, h, w, order_pad="reflect"):
    """Centre-crop or reflect-pad `arr` back to (h, w)."""
    ah, aw = arr.shape[:2]
    if ah >= h and aw >= w:
        top, left = (ah - h) // 2, (aw - w) // 2
        return arr[top:top + h, left:left + w]
    ph, pw = max(0, h - ah), max(0, w - aw)
    pad = ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2))
    out = np.pad(arr, pad, mode=order_pad)
    return out[:h, :w]


def random_scale(img, gb, sf, scale_range, rng):
    """Isotropic zoom, then crop/pad back to the original shape."""
    s = float(rng.uniform(*scale_range))
    h, w = img.shape[:2]
    img = _fit(nd_zoom(img, s, order=1), h, w)
    gb = _fit(nd_zoom(gb, s, order=0), h, w)
    sf = _fit(nd_zoom(sf, s, order=0), h, w)
    return img, gb, sf


def elastic_deform(img, gb, sf, alpha, sigma, rng):
    """Smooth random displacement field, shared by image and both masks."""
    shape = img.shape[:2]
    dx = gaussian_filter(rng.uniform(-1, 1, shape), sigma, mode="reflect") * alpha
    dy = gaussian_filter(rng.uniform(-1, 1, shape), sigma, mode="reflect") * alpha
    yy, xx = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    coords = np.stack([yy + dy, xx + dx])
    img = map_coordinates(img, coords, order=1, mode="reflect")
    gb = map_coordinates(gb, coords, order=0, mode="reflect")
    sf = map_coordinates(sf, coords, order=0, mode="reflect")
    return img, gb, sf


# --------------------------------------------------------------------------
# intensity / noise ops -- image only, masks untouched
# --------------------------------------------------------------------------

def random_brightness_contrast(img, b_lim, c_lim, rng):
    b = float(rng.uniform(-b_lim, b_lim))
    c = 1.0 + float(rng.uniform(-c_lim, c_lim))
    mean = img.mean()
    return (img - mean) * c + mean + b


def random_gamma(img, gamma_range, rng):
    g = float(rng.uniform(*gamma_range))
    return np.power(np.clip(img, 0.0, 1.0), g)


def gaussian_noise(img, sigma_range, rng):
    sigma = float(rng.uniform(*sigma_range))
    return img + rng.normal(0.0, sigma, img.shape)


def poisson_noise(img, scale_range, rng):
    """Shot noise -- the dominant noise source in low-dose TEM.

    `scale` is effective counts at full scale; lower = noisier.
    """
    scale = float(rng.uniform(*scale_range))
    x = np.clip(img, 0.0, 1.0) * scale
    return rng.poisson(x).astype(np.float32) / max(scale, 1e-6)


def random_blur(img, sigma_range, rng):
    """Stands in for defocus spread / slight drift."""
    sigma = float(rng.uniform(*sigma_range))
    if sigma < 1e-3:
        return img
    return gaussian_filter(img, sigma, mode="reflect")


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------

DEFAULT_AUG = {
    "hrtem_safe": False,          # BF-TEM, so resampling ops are allowed

    "p_d4": 1.0,
    "p_rotate": 0.3,
    "rotate_max_deg": 15.0,
    "p_scale": 0.3,
    "scale_range": (0.85, 1.20),
    "p_elastic": 0.25,
    "elastic_alpha": 12.0,
    "elastic_sigma": 10.0,

    "p_brightness_contrast": 0.5,
    "brightness_limit": 0.20,
    "contrast_limit": 0.25,
    "p_gamma": 0.4,
    "gamma_range": (0.7, 1.4),

    "p_gaussian_noise": 0.3,
    "gaussian_sigma_range": (0.01, 0.06),
    "p_poisson_noise": 0.3,
    "poisson_scale_range": (40.0, 300.0),
    "p_blur": 0.25,
    "blur_sigma_range": (0.0, 1.2),
}


class JointAugmenter:
    """Callable augmentation over (image, gb_mask, sf_mask).

    Parameters
    ----------
    cfg : dict | None
        Overrides merged on top of DEFAULT_AUG.
    train : bool
        If False, only percentile normalisation is applied -- no geometry, no
        noise.  Test images are never augmented (spec).
    seed : int
        Each fold / worker should get its own.
    """

    def __init__(self, cfg: dict | None = None, train: bool = True, seed: int = 0,
                 crop: int | None = None):
        self.cfg = {**DEFAULT_AUG, **(cfg or {})}
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.crop = crop            # random window size, applied after geometry

    def __call__(self, img: np.ndarray, gb: np.ndarray, sf: np.ndarray):
        c, rng = self.cfg, self.rng

        img = percentile_normalize(img)
        gb = gb.astype(np.float32)
        sf = sf.astype(np.float32)

        if not self.train:
            return (img[None].astype(np.float32),
                    gb[None].astype(np.float32),
                    sf[None].astype(np.float32))

        # -------- geometry (shared parameters across image and both masks) --
        if rng.random() < c["p_d4"]:
            img, gb, sf = random_d4(img, gb, sf, rng)

        if not c["hrtem_safe"]:
            if rng.random() < c["p_rotate"]:
                img, gb, sf = random_rotate(img, gb, sf, c["rotate_max_deg"], rng)
            if rng.random() < c["p_scale"]:
                img, gb, sf = random_scale(img, gb, sf, c["scale_range"], rng)
            if rng.random() < c["p_elastic"]:
                img, gb, sf = elastic_deform(
                    img, gb, sf, c["elastic_alpha"], c["elastic_sigma"], rng)

        # -------- random crop, after geometry, before photometry ------------
        # Same window for image and both masks.  Uniform position, no
        # SF- or GB-biased sampling.
        if self.crop:
            k = self.crop
            H, W = img.shape[:2]
            if H < k or W < k:
                raise ValueError(f"crop {k} larger than image {H}x{W}")
            top = int(rng.integers(0, H - k + 1))
            left = int(rng.integers(0, W - k + 1))
            img = img[top:top + k, left:left + k]
            gb = gb[top:top + k, left:left + k]
            sf = sf[top:top + k, left:left + k]

        # -------- intensity (image only) ------------------------------------
        if rng.random() < c["p_brightness_contrast"]:
            img = random_brightness_contrast(
                img, c["brightness_limit"], c["contrast_limit"], rng)
        if rng.random() < c["p_gamma"]:
            img = random_gamma(img, c["gamma_range"], rng)

        # -------- blur then noise: optics first, detector second ------------
        if rng.random() < c["p_blur"]:
            img = random_blur(img, c["blur_sigma_range"], rng)
        if rng.random() < c["p_poisson_noise"]:
            img = poisson_noise(img, c["poisson_scale_range"], rng)
        if rng.random() < c["p_gaussian_noise"]:
            img = gaussian_noise(img, c["gaussian_sigma_range"], rng)

        img = np.clip(img, 0.0, 1.0)

        # order=0 resampling keeps masks binary, but guard against float drift
        gb = (gb > 0.5).astype(np.float32)
        sf = (sf > 0.5).astype(np.float32)

        return (img[None].astype(np.float32), gb[None], sf[None])
