"""All metrics under test, computed one way for every comparison in the analysis.

GB: Dice, clDice, ARE, VI (and the merge/split terms separately).
SF: Dice, object F1 under IoU >= 0.5 (greedy one-to-one), object F1 under the project's
    tolerance protocol (sf_instance.py, with d and theta exposed so Step 4 can sweep them).

Partitions use targets.grain_labels so the metric and the target quantity are computed on
exactly the same objects -- the point main.tex makes about evaluating on equal terms.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.metrics import adapted_rand_error, variation_of_information
from skimage.morphology import skeletonize

from partition import (border_closure, grain_match_f1,
                       gt_interior_mask, partition_metrics)
from sf_instance import INST_MIN_COV, fits, match_counts, prf
from targets import binarize, fault_labels, grain_labels


def dice(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0


def cldice(pred, gt):
    """Shit et al. 2021: harmonic mean of skeleton precision and sensitivity."""
    p, g = np.asarray(pred, bool), np.asarray(gt, bool)
    if not p.any() or not g.any():
        return 0.0
    sp, sg = skeletonize(p), skeletonize(g)
    tprec = (sp & g).sum() / sp.sum() if sp.sum() else 0.0
    tsens = (sg & p).sum() / sg.sum() if sg.sum() else 0.0
    return float(2 * tprec * tsens / (tprec + tsens)) if (tprec + tsens) else 0.0


def gb_metrics(gb_pred, gb_gt, fair=True, exclude_border=False):
    """Pixel metrics plus the FAIR partition metrics (boundary pixels assigned to the
    nearest grain first -- see partition.py).  `*_raw` keeps the old label-0 version so
    the change is auditable rather than silent."""
    p, g = np.asarray(gb_pred, bool), np.asarray(gb_gt, bool)
    out = {"gb_dice": dice(p, g), "gb_cldice": cldice(p, g)}
    # PRIMARY: fair (boundary pixels assigned to nearest grain) AND interior-only
    # (border grains excluded, defined on GT).  The two other versions are kept beside it
    # so every change is auditable and the supplement can report them.
    out.update(partition_metrics(p, g, fair=True, exclude_border=True))
    out.update(grain_match_f1(p, g, fair=True))
    out.update(border_closure(p, g))
    inc = partition_metrics(p, g, fair=True, exclude_border=False)
    out.update({"gb_are_allgrains": inc["gb_are"], "gb_vi_allgrains": inc["gb_vi"],
                "gb_vi_merge_allgrains": inc["gb_vi_merge"],
                "n_grains_pred_allgrains": inc["n_grains_pred"]})
    raw = partition_metrics(p, g, fair=False)
    out.update({"gb_are_raw": raw["gb_are"], "gb_vi_raw": raw["gb_vi"]})
    return out


def _iou_f1(pred_lab, gt_lab):
    """Greedy one-to-one matching on component IoU at threshold 0.5."""
    npd, ngt = int(pred_lab.max()), int(gt_lab.max())
    if npd == 0 or ngt == 0:
        return 0.0 if (npd or ngt) else 1.0
    pairs = []
    for pid, sl in enumerate(ndi.find_objects(pred_lab), start=1):
        if sl is None:
            continue
        m = pred_lab[sl] == pid
        for gid in np.unique(gt_lab[sl][m]):
            if gid == 0:
                continue
            inter = int((m & (gt_lab[sl] == gid)).sum())
            union = int((pred_lab == pid).sum() + (gt_lab == gid).sum() - inter)
            if union and inter / union >= 0.5:
                pairs.append((inter / union, pid, int(gid)))
    pairs.sort(reverse=True)
    up, ug, tp = set(), set(), 0
    for _, pid, gid in pairs:
        if pid in up or gid in ug:
            continue
        up.add(pid); ug.add(gid); tp += 1
    p = tp / npd
    r = tp / ngt
    return float(2 * p * r / (p + r)) if (p + r) else 0.0


def sf_metrics(sf_pred, sf_gt, tol_px=4.0, tol_deg=15.0):
    p, g = np.asarray(sf_pred, bool), np.asarray(sf_gt, bool)
    lp, lg = fault_labels(p), fault_labels(g)
    counts = match_counts(fits(g), fits(p), tol_px=tol_px, tol_deg=tol_deg,
                          min_cov=INST_MIN_COV)
    P, R, F = prf([counts])
    return {"sf_dice": dice(p, g), "sf_f1_iou": _iou_f1(lp, lg),
            "sf_p_tol": P, "sf_r_tol": R, "sf_f1_tol": F,
            "sf_counts": [int(c) for c in counts]}


def interior_mask(gb_gt):
    return gt_interior_mask(np.asarray(gb_gt, bool))


def all_metrics(gb_pred, sf_pred, gb_gt, sf_gt, is_pred=True, tol_px=4.0, tol_deg=15.0):
    gbp, sfp = binarize(gb_pred, is_pred), binarize(sf_pred, is_pred)
    gbg, sfg = binarize(gb_gt, False), binarize(sf_gt, False)
    out = {}
    if gbp is not None:
        out.update(gb_metrics(gbp, gbg))
    if sfp is not None:
        out.update(sf_metrics(sfp, sfg, tol_px, tol_deg))
    return out
