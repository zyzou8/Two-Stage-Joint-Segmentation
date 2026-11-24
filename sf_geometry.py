"""
Geometry helpers for the grain-conditioned stacking-fault (SF) model (model.md).

Everything here is numpy/scipy/skimage only, so the evaluation protocol can be
run and unit-tested without TensorFlow.

Conventions
-----------
* Annotation colours (GB_SF_train_21): red = background, green = GB only,
  blue = SF only, cyan = GB and SF.  GB = G>127, SF = B>127.
* Coordinates: x = column, y = row (y down).  Orientation theta is in degrees,
  180-periodic, measured in image coordinates: theta = atan2(dy, dx) mod 180.
* Orientation is handled in double-angle space, o = (cos 2theta, sin 2theta).
  Averaging raw angles is wrong (1 deg and 179 deg are 2 deg apart).

Every threshold is a module constant so it is stated in one place.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import polygon as _polygon

try:
    from imageio.v3 import imread
except ImportError:  # pragma: no cover
    from imageio import imread

# ---- thresholds (all stated in the report) --------------------------------
MIN_GRAIN_PX = 500        # grains smaller than this are fragments (label 0)
SF_MIN_PX = 30            # SF components smaller than this are dropped
ANCHOR_TOL = 6.0          # endpoint is "on the GB" if within this many px
PU_MARGIN = 3.0           # extra half-width of the chord-extension ray (px)
MIN_ELONGATION = 2.0      # std-ratio major/minor below which theta is not trusted
MATCH_MIN_COV = 0.25      # fraction of an annotated SF a prediction must cover to match
MATCH_DILATE = 2          # px, tolerance when testing coverage


# --------------------------------------------------------------------------
# orientation utilities (double-angle space)
# --------------------------------------------------------------------------

def double_angle(theta_deg):
    t = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    return np.stack([np.cos(2 * t), np.sin(2 * t)], axis=-1)


def theta_from_double_angle(v):
    """(..., 2) double-angle vectors -> theta in [0, 180)."""
    v = np.asarray(v, dtype=np.float64)
    return (np.rad2deg(0.5 * np.arctan2(v[..., 1], v[..., 0]))) % 180.0


def orientation_error_deg(a, b):
    """Smallest angle between two undirected orientations, in [0, 90] degrees."""
    d = (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64) + 90.0) % 180.0 - 90.0
    return np.abs(d)


def d4_double_angle(o, k: int, flip: bool):
    """Transform double-angle vectors under np.rot90(.., k) followed by optional np.fliplr.

    rot90 k=1 maps a direction (dx, dy) -> (dy, -dx), i.e. theta -> theta - 90,
    which is o -> -o in double-angle space; fliplr maps theta -> 180 - theta,
    i.e. sin2theta -> -sin2theta.  Verified against rasterised lines in the tests.
    """
    o = np.asarray(o, dtype=np.float64).copy()
    if k % 2:
        o = -o
    if flip:
        o[..., 1] = -o[..., 1]
    return o


# --------------------------------------------------------------------------
# I/O and grains
# --------------------------------------------------------------------------

def read_annotation(path):
    """Return (gb, sf) boolean masks from a 4-colour RGB annotation PNG."""
    a = np.asarray(imread(path))
    if a.ndim != 3 or a.shape[-1] < 3:
        raise ValueError(f"{path}: expected an RGB annotation, got shape {a.shape}")
    return a[..., 1] > 127, a[..., 2] > 127


def label_grains(gb: np.ndarray, min_area: int = MIN_GRAIN_PX):
    """4-connected labelling of non-GB pixels.  Fragments < min_area get label 0.

    Returns (lab int32, area per label (index 0 unused), touches_border bool per label).
    """
    lab0, n = ndi.label(~gb, ndi.generate_binary_structure(2, 1))
    area = np.bincount(lab0.ravel(), minlength=n + 1)
    keep = area >= min_area
    keep[0] = False
    new = np.zeros(n + 1, np.int32)
    new[keep] = np.arange(1, keep.sum() + 1)
    lab = new[lab0]
    areas = np.concatenate([[0], area[keep]])
    border = np.zeros(len(areas), bool)
    for e in (lab[0], lab[-1], lab[:, 0], lab[:, -1]):
        border[np.unique(e)] = True
    border[0] = False
    return lab, areas, border


# --------------------------------------------------------------------------
# one band (or predicted component) -> oriented segment
# --------------------------------------------------------------------------

def fit_band(xs: np.ndarray, ys: np.ndarray) -> dict:
    """Total-least-squares major axis of a pixel set -> oriented segment.

    Returns centre, unit axis (ux, uy; ux>0 or ux==0,uy>0), theta (deg, [0,180)),
    length along the axis, half-width by area, endpoints (x1,y1),(x2,y2),
    elongation (std ratio) and whether theta is trusted (>= MIN_ELONGATION).
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    n = len(xs)
    cx, cy = xs.mean(), ys.mean()
    if n > 2:
        ev, evec = np.linalg.eigh(np.cov(np.vstack([xs, ys])))
    else:
        ev, evec = np.array([1e-9, 1.0]), np.eye(2)
    u = evec[:, 1]
    if u[0] < 0 or (u[0] == 0 and u[1] < 0):
        u = -u
    ux, uy = float(u[0]), float(u[1])
    t = (xs - cx) * ux + (ys - cy) * uy
    t0, t1 = float(t.min()), float(t.max())
    length = t1 - t0 + 1.0
    elong = float(np.sqrt(max(ev[1], 1e-12) / max(ev[0], 1e-12)))
    return dict(
        cx=float(cx), cy=float(cy), ux=ux, uy=uy,
        theta=float(np.rad2deg(np.arctan2(uy, ux)) % 180.0),
        length=float(length), half_width=float(n / length / 2.0), n_pix=int(n),
        x1=float(cx + t0 * ux), y1=float(cy + t0 * uy),
        x2=float(cx + t1 * ux), y2=float(cy + t1 * uy),
        elongation=elong, valid=bool(elong >= MIN_ELONGATION),
    )


def _march_len(lab, g, x0, y0, ux, uy, tmax=1500.0, step=0.5) -> float:
    """Distance travelled from (x0,y0) along (ux,uy) while staying in grain g."""
    h, w = lab.shape
    ts = np.arange(0.0, tmax, step)
    xs, ys = x0 + ux * ts, y0 + uy * ts
    inb = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
    ok = np.zeros(len(ts), bool)
    ok[inb] = lab[np.rint(ys[inb]).astype(int), np.rint(xs[inb]).astype(int)] == g
    bad = np.flatnonzero(~ok)
    k = bad[0] if len(bad) else len(ts)
    return float(ts[k - 1]) if k > 0 else 0.0


def components(mask: np.ndarray, min_px: int = SF_MIN_PX):
    """8-connected components of a boolean mask, dropping those < min_px.

    Returns (label image int32 with only kept components renumbered 1..n, list of
    (ys, xs) coordinate arrays).
    """
    lab0, n = ndi.label(mask, np.ones((3, 3), int))
    if n == 0:
        return np.zeros(mask.shape, np.int32), []
    cnt = np.bincount(lab0.ravel(), minlength=n + 1)
    keep = np.flatnonzero(cnt >= min_px)
    keep = keep[keep > 0]
    remap = np.zeros(n + 1, np.int32)
    remap[keep] = np.arange(1, len(keep) + 1)
    lab = remap[lab0]
    coords = ndi.find_objects(lab0)
    out = []
    for new_id, old_id in enumerate(keep, start=1):
        sl = coords[old_id - 1]
        sub = lab0[sl] == old_id
        yy, xx = np.nonzero(sub)
        out.append((yy + sl[0].start, xx + sl[1].start))
    return lab, out


def majority_grain(lab: np.ndarray, ys, xs) -> int:
    g = lab[ys, xs]
    g = g[g > 0]
    return int(np.bincount(g).argmax()) if g.size else 0


# --------------------------------------------------------------------------
# GT scene: grains, bands, chord-extension (PU) region
# --------------------------------------------------------------------------

NEAR_EXT_RANGE = (5.0, 20.0)   # px beyond an annotated end used by the off-mask validity test


def chord_extension_region(lab, bands, sf_mask, margin: float = PU_MARGIN):
    """Pixels inside a band's own grain, along its fitted direction, beyond the
    annotated ends, within (half_width + margin) px of the fitted line, and not
    annotated SF.  This is where the true SF is expected to continue but the
    annotation stopped: unlabelled, not negative (model.md, central design
    consequence).  The ray stops at the first exit from the grain.
    """
    h, w = lab.shape
    region = np.zeros((h, w), bool)
    for b in bands:
        g = b["grain"]
        if g <= 0 or not b["valid"]:
            continue
        wd = max(b["half_width"], 1.0) + margin
        for sgn, (ex, ey) in ((+1.0, (b["x2"], b["y2"])), (-1.0, (b["x1"], b["y1"]))):
            dx, dy = sgn * b["ux"], sgn * b["uy"]
            ext = _march_len(lab, g, ex, ey, dx, dy)
            if ext < 1.0:
                continue
            nx, ny = -dy, dx
            cx0, cy0 = ex, ey
            cx1, cy1 = ex + dx * (ext + 1.0), ey + dy * (ext + 1.0)
            xs = [cx0 + nx * wd, cx1 + nx * wd, cx1 - nx * wd, cx0 - nx * wd]
            ys = [cy0 + ny * wd, cy1 + ny * wd, cy1 - ny * wd, cy0 - ny * wd]
            rr, cc = _polygon(ys, xs, shape=(h, w))
            region[rr, cc] = True
            t = (cc - ex) * dx + (rr - ey) * dy          # distance beyond the annotated end
            ok = (t >= NEAR_EXT_RANGE[0]) & (t <= NEAR_EXT_RANGE[1]) & (lab[rr, cc] == g) & ~sf_mask[rr, cc]
            b.setdefault("_near", []).append((rr[ok].astype(np.int32), cc[ok].astype(np.int32)))
    return region & (lab > 0) & ~sf_mask


def build_scene(gb: np.ndarray, sf: np.ndarray, margin: float = PU_MARGIN) -> dict:
    """Everything the protocol and the training targets need from one annotation."""
    lab, areas, border = label_grains(gb)
    sf_lab, comps = components(sf)
    dist_gb = ndi.distance_transform_edt(~gb).astype(np.float32)
    bands = []
    for i, (ys, xs) in enumerate(comps, start=1):
        b = fit_band(xs, ys)
        b["id"] = i
        b["grain"] = majority_grain(lab, ys, xs)
        b["ys"], b["xs"] = ys.astype(np.int32), xs.astype(np.int32)
        bands.append(b)
    for b in bands:
        d1 = dist_gb[int(np.clip(round(b["y1"]), 0, gb.shape[0] - 1)), int(np.clip(round(b["x1"]), 0, gb.shape[1] - 1))]
        d2 = dist_gb[int(np.clip(round(b["y2"]), 0, gb.shape[0] - 1)), int(np.clip(round(b["x2"]), 0, gb.shape[1] - 1))]
        b["d_end1"], b["d_end2"] = float(d1), float(d2)
    ext = chord_extension_region(lab, bands, sf, margin)
    for b in bands:      # pixels 5-20 px beyond the annotated ends, inside the SF's own grain
        parts = b.pop("_near", [])
        b["near_ys"] = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, np.int32)
        b["near_xs"] = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, np.int32)
    return dict(gb=gb, sf=sf, lab=lab, grain_area=areas, grain_border=border,
                sf_lab=sf_lab, bands=bands, dist_gb=dist_gb, ext=ext)


def edge_distance(shape, x, y) -> float:
    h, w = shape
    return float(min(x, y, w - 1 - x, h - 1 - y))


def end_anchored(dist_ref_gb, shape, x, y, tol: float = ANCHOR_TOL) -> bool:
    """Endpoint counts as boundary-anchored if within tol px of the reference GB or
    of the image border (an SF cut by the field of view is not a truncated SF)."""
    xi = int(np.clip(round(x), 0, shape[1] - 1))
    yi = int(np.clip(round(y), 0, shape[0] - 1))
    return bool(dist_ref_gb[yi, xi] <= tol or edge_distance(shape, x, y) <= tol)


def is_chord(band: dict, dist_ref_gb, shape, tol: float = ANCHOR_TOL) -> bool:
    return (end_anchored(dist_ref_gb, shape, band["x1"], band["y1"], tol)
            and end_anchored(dist_ref_gb, shape, band["x2"], band["y2"], tol))


def image_stems(root: str, images_dir="images"):
    return sorted((p.stem for p in (Path(root) / images_dir).glob("*.png")), key=lambda s: (len(s), s))
