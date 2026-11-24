"""PRIMARY / SECONDARY scoring -- the one definition every v2 script uses.

PRIMARY   boundary pixels assigned to the nearest grain (no label-0 segment), and every
          quantity restricted to pixels of GT grains that do not touch the frame.
SECONDARY identical except border grains are kept (the "include-border" supplement).

The interior region is defined on the GROUND TRUTH only (partition.gt_interior_mask), so a
prediction can never choose the region it is scored on.

Keys are returned flat: primary keys carry no suffix, secondary keys carry `_inc`
("include border").  The identity check in perturb_v2.py pins every one of them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from metrics_all import cldice, dice, _iou_f1
from partition import (fill_to_nearest, grain_labels as _plabels,
                       gt_interior_mask, partition_metrics)
from sf_instance import INST_MIN_COV, fits, match_counts, prf
from targets import fault_labels, image_targets

FOUR = ndi.generate_binary_structure(2, 1)


def interior_mask(gb_gt):
    """Pixels of GT grains that do not touch the image frame."""
    return gt_interior_mask(np.asarray(gb_gt, bool))


# --------------------------------------------------------------- grain matching F1

def _renumber(lab):
    """Compress labels to 1..n over the values actually present (0 stays 0)."""
    vals = np.unique(lab)
    vals = vals[vals > 0]
    remap = np.zeros(int(lab.max()) + 1, np.int32)
    remap[vals] = np.arange(1, len(vals) + 1)
    return remap[lab], len(vals)


def grain_f1(gb_pred, gb_gt, mask=None, iou_thr=0.5):
    """One-to-one grain matching at IoU >= thr, optionally inside `mask`.

    Labels are renumbered AFTER masking, so a grain that lies entirely outside the region
    is not counted as a miss -- without this the precision/recall denominators would still
    include every excluded grain.
    """
    gt = fill_to_nearest(_plabels(np.asarray(gb_gt, bool)))
    pred = fill_to_nearest(_plabels(np.asarray(gb_pred, bool)))
    if mask is not None:
        m = np.asarray(mask, bool)
        gt, pred = np.where(m, gt, 0), np.where(m, pred, 0)
    gt, ng = _renumber(gt)
    pred, npd = _renumber(pred)
    if ng == 0 or npd == 0:
        return {"grain_f1": 0.0, "grain_prec": 0.0, "grain_rec": 0.0}
    a = gt.ravel().astype(np.int64) * (npd + 1) + pred.ravel().astype(np.int64)
    inter = np.bincount(a, minlength=(ng + 1) * (npd + 1)).reshape(ng + 1, npd + 1)
    agt, apd = inter.sum(1, keepdims=True), inter.sum(0, keepdims=True)
    union = agt + apd - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    iou[0, :] = 0.0
    iou[:, 0] = 0.0
    pairs = [(iou[i, j], i, j) for i, j in zip(*np.nonzero(iou >= iou_thr))]
    pairs.sort(reverse=True)
    ug, up, tp = set(), set(), 0
    for _, i, j in pairs:
        if i in ug or j in up:
            continue
        ug.add(i)
        up.add(j)
        tp += 1
    p, r = tp / npd, tp / ng
    return {"grain_f1": float(2 * p * r / (p + r)) if (p + r) else 0.0,
            "grain_prec": float(p), "grain_rec": float(r)}


# --------------------------------------------------------------- GB / SF scoring

def gb_scores(gb_pred, gb_gt, mask=None, exclude_border=True):
    """Dice, clDice, ARE, VI (+merge/split), grain counts and grain F1 for ONE mode."""
    p, g = np.asarray(gb_pred, bool), np.asarray(gb_gt, bool)
    pm = (p & mask, g & mask) if mask is not None else (p, g)
    out = {"gb_dice": dice(*pm), "gb_cldice": cldice(*pm)}
    pmet = partition_metrics(p, g, fair=True, exclude_border=exclude_border)
    # The partition counts segments of the FILLED partition; targets.image_targets counts
    # grains as raw connected components of NOT(boundary).  They agree on ground truth but
    # not necessarily on a prediction, and both keys were called `n_grains_*`, so one
    # silently overwrote the other.  Keep the partition's under its own name.
    pmet["n_seg_pred"] = pmet.pop("n_grains_pred")
    pmet["n_seg_gt"] = pmet.pop("n_grains_gt")
    out.update(pmet)
    out.update(grain_f1(p, g, mask=mask))
    return out


def sf_scores(sf_pred, sf_gt, mask=None, tol_px=4.0, tol_deg=15.0):
    p, g = np.asarray(sf_pred, bool), np.asarray(sf_gt, bool)
    if mask is not None:
        m = np.asarray(mask, bool)
        p, g = p & m, g & m
    counts = match_counts(fits(g), fits(p), tol_px=tol_px, tol_deg=tol_deg,
                          min_cov=INST_MIN_COV)
    P, R, F = prf([counts])
    # `sf_counts_tol` is the raw (gt_hit, n_gt, pred_hit, n_pred) for THIS image.
    # The paper pools these within a fold before computing F1, so a per-image F1
    # cannot simply be averaged; the caller needs the counts to do it correctly.
    return {"sf_dice": dice(p, g), "sf_f1_iou": _iou_f1(fault_labels(p), fault_labels(g)),
            "sf_p_tol": P, "sf_r_tol": R, "sf_f1_tol": F,
            "sf_counts_tol": [int(c) for c in counts]}


def score_all(gb_pred, sf_pred, gb_gt, sf_gt):
    """PRIMARY (interior-only) + SECONDARY (`_inc`) for every metric and every target.

    Everything is boolean in, boolean out -- binarization is the caller's job, so the
    perturbation study and the gallery cannot disagree about a threshold.
    """
    gb_pred = np.asarray(gb_pred, bool)
    sf_pred = np.asarray(sf_pred, bool)
    gb_gt = np.asarray(gb_gt, bool)
    sf_gt = np.asarray(sf_gt, bool)
    m = interior_mask(gb_gt)

    out = {}
    out.update(gb_scores(gb_pred, gb_gt, mask=m, exclude_border=True))
    out.update(sf_scores(sf_pred, sf_gt, mask=m))
    out.update(image_targets(gb_pred, sf_pred, gb_gt, sf_gt, is_pred=False, mask=m))
    out["interior_frac"] = float(m.mean())

    sec = {}
    sec.update(gb_scores(gb_pred, gb_gt, mask=None, exclude_border=False))
    sec.update(sf_scores(sf_pred, sf_gt, mask=None))
    sec.update(image_targets(gb_pred, sf_pred, gb_gt, sf_gt, is_pred=False, mask=None))
    out.update({f"{k}_inc": v for k, v in sec.items()})
    return out


# --------------------------------------------------------------- identity check

IDENTITY_SPEC = [
    ("gb_dice", 1.0), ("gb_cldice", 1.0), ("gb_are", 0.0), ("gb_vi", 0.0),
    ("gb_vi_merge", 0.0), ("gb_vi_split", 0.0), ("grain_f1", 1.0),
    ("sf_f1_tol", 1.0), ("sf_f1_iou", 1.0), ("sf_dice", 1.0),
    ("grain_count_err", 0.0), ("grain_size_w1", 0.0),
    ("fault_count_err", 0.0), ("fault_density_err", 0.0),
]
TOL = 1e-9


def check_identity(gb_gt, sf_gt, label=""):
    """GT scored against itself must be exactly perfect in BOTH modes.  Returns the row."""
    s = score_all(gb_gt, sf_gt, gb_gt, sf_gt)
    bad = []
    for key, want in IDENTITY_SPEC:
        for suffix, mode in (("", "primary"), ("_inc", "secondary")):
            got = s[key + suffix]
            if not np.isfinite(got) or abs(got - want) > TOL:
                bad.append(f"{label}{key}[{mode}] = {got!r}, expected {want}")
    if bad:
        raise AssertionError("IDENTITY CHECK FAILED:\n  " + "\n  ".join(bad))
    return s
