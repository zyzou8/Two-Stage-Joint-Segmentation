"""SF object-level P/R/F1 under the project's tolerance-based instance protocol.

This replaces the IoU>=0.5 connected-component rule in metrics.py for the results
table, so the numbers are comparable to the existing Eff-UNet / DINOv3 rows.

Definitions are imported from `sf_geometry.py` (fit_band, components,
orientation_error_deg) rather than reimplemented, so component extraction and
segment fitting are bit-identical to the established SF-only pipeline that
module came from.

Protocol (as specified, including the stated deviation from the original
`instance_match`):

  objects   8-connected components, >= SF_MIN_PX (30) px, on BOTH sides.
            GT   = SF annotation > 127
            pred = SF probability >= 0.5
            Each component -> a straight segment by total least squares.
            No valid/grain filtering on either side -- the original
            `evaluate_instances` filtered GT to `valid and grain > 0` but not
            predictions, which made precision unattainable at 1.0.

  match     BOTH must hold:
            position    >= INST_MIN_COV (0.25) of one segment's length lies
                        within INST_TOL_PX (4) px of the other, tested in each
                        direction separately
            orientation |theta_gt - theta_pred| <= INST_TOL_DEG (15), 180-periodic
            The orientation test is SKIPPED when either object is not clearly
            elongated (elongation < MIN_ELONGATION = 2.0), so a roundish blob
            matches on position alone.  The original required a trusted angle on
            both sides.

  scoring   Not one-to-one.  Recall = share of GT objects matched by >= 1
            prediction; precision = share of predictions matching >= 1 GT.
            Counts POOL over the images of a fold, then mean +/- std over folds.
            Splitting and merging are therefore not penalised.
"""

from __future__ import annotations

import numpy as np

from sf_geometry import (
    components, fit_band, orientation_error_deg, SF_MIN_PX, MIN_ELONGATION,
)

INST_TOL_PX = 4.0
INST_TOL_DEG = 15.0
INST_MIN_COV = 0.25


def fits(mask: np.ndarray, min_px: int = SF_MIN_PX) -> list[dict]:
    """Boolean mask -> list of fitted segments, one per kept component."""
    _, comps = components(mask.astype(bool), min_px=min_px)
    return [fit_band(xs, ys) for ys, xs in comps]


def _seg_samples(f: dict, step: float = 1.0) -> np.ndarray:
    a = np.array([f["x1"], f["y1"]], float)
    b = np.array([f["x2"], f["y2"]], float)
    n = max(2, int(np.ceil(np.hypot(*(b - a)) / step)) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return a[None] * (1 - t) + b[None] * t


def _pts_to_seg_dist(pts: np.ndarray, f: dict) -> np.ndarray:
    a = np.array([f["x1"], f["y1"]], float)
    b = np.array([f["x2"], f["y2"]], float)
    ab = b - a
    den = float((ab ** 2).sum())
    if den < 1e-9:
        return np.hypot(*(pts - a).T)
    t = np.clip(((pts - a) @ ab) / den, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.hypot(*(pts - proj).T)


def match_counts(gt_fits: list[dict], pred_fits: list[dict],
                 tol_px: float = INST_TOL_PX, tol_deg: float = INST_TOL_DEG,
                 min_cov: float = INST_MIN_COV) -> tuple[int, int, int, int]:
    """Returns (n_gt_matched, n_gt, n_pred_matched, n_pred) for ONE image."""
    ng, npd = len(gt_fits), len(pred_fits)
    if ng == 0 and npd == 0:
        return 0, 0, 0, 0
    if ng == 0:
        return 0, 0, 0, npd
    if npd == 0:
        return 0, ng, 0, 0

    gs = [_seg_samples(f) for f in gt_fits]
    ps = [_seg_samples(f) for f in pred_fits]

    cov_g = np.zeros((ng, npd))   # share of GT length near the prediction
    cov_p = np.zeros((ng, npd))   # share of prediction length near the GT
    ang_ok = np.zeros((ng, npd), bool)

    for i, gfit in enumerate(gt_fits):
        for j, pfit in enumerate(pred_fits):
            cov_g[i, j] = float((_pts_to_seg_dist(gs[i], pfit) <= tol_px).mean())
            cov_p[i, j] = float((_pts_to_seg_dist(ps[j], gfit) <= tol_px).mean())
            # orientation is only meaningful for clearly elongated objects
            if not (gfit["valid"] and pfit["valid"]):
                ang_ok[i, j] = True
            else:
                ang_ok[i, j] = bool(
                    orientation_error_deg(gfit["theta"], pfit["theta"]) <= tol_deg)

    gt_hit = ((cov_g >= min_cov) & ang_ok).any(axis=1)
    pred_hit = ((cov_p >= min_cov) & ang_ok).any(axis=0)
    return int(gt_hit.sum()), ng, int(pred_hit.sum()), npd


def prf(counts: list[tuple[int, int, int, int]]) -> tuple[float, float, float]:
    """Pool per-image counts, then compute P, R, F1."""
    a = np.array(counts, dtype=float).reshape(-1, 4)
    gt_hit, n_gt, pred_hit, n_pred = a.sum(axis=0)
    r = gt_hit / n_gt if n_gt else float("nan")
    p = pred_hit / n_pred if n_pred else float("nan")
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return float(p), float(r), float(f1)
