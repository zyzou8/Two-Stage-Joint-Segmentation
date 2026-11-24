"""Evaluation metrics -- exactly the columns the spec asks for, nothing else.

    Single GB    Dice, clDice, ARE, VI
    Single SF    Dice; object-level P, R, F1
    Joint GB+SF  GB Dice, ARE; SF Dice, F1

Grains are connected components of NOT(gb_mask), border-touching grains
included (spec: the image border is not a boundary, so partial grains at the
edge are valid grains).

NOTE on VI: skimage's variation_of_information returns (vi_split, vi_merge).
The spec says report the sum.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.metrics import adapted_rand_error, variation_of_information
from skimage.morphology import skeletonize

import config


# --------------------------------------------------------------------------
# pixel-level
# --------------------------------------------------------------------------

def dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """Binary Dice.  Both inputs boolean."""
    pred, target = pred.astype(bool), target.astype(bool)
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0                      # both empty: perfect agreement
    return float((2.0 * (pred & target).sum() + eps) / (denom + eps))


def cl_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """Centreline Dice -- the topology-aware score for thin structures.

        Tprec = |skel(pred)  & target| / |skel(pred)|
        Tsens = |skel(target) & pred|  / |skel(target)|
        clDice = harmonic mean

    Rewards getting the boundary *network* connected, not just the pixel
    overlap, which is why it matters for GB.
    """
    pred, target = pred.astype(bool), target.astype(bool)
    if not pred.any() or not target.any():
        return 0.0 if (pred.any() or target.any()) else 1.0

    sp, st = skeletonize(pred), skeletonize(target)
    if sp.sum() == 0 or st.sum() == 0:
        return 0.0

    tprec = (sp & target).sum() / (sp.sum() + eps)
    tsens = (st & pred).sum() / (st.sum() + eps)
    if tprec + tsens < eps:
        return 0.0
    return float(2.0 * tprec * tsens / (tprec + tsens))


# --------------------------------------------------------------------------
# grain-level (GB)
# --------------------------------------------------------------------------

def grain_labels(gb_mask: np.ndarray) -> np.ndarray:
    """Label map of grains = connected components of NOT(gb_mask).

    Border-touching components are kept: a partial grain at the image edge is a
    valid grain (spec, Domain notes).
    """
    structure = ndimage.generate_binary_structure(2, config.GRAIN_CONNECTIVITY)
    labels, _ = ndimage.label(~gb_mask.astype(bool), structure=structure)
    return labels


def are_vi(gb_pred: np.ndarray, gb_true: np.ndarray) -> tuple[float, float]:
    """(ARE, VI) between the grain segmentations induced by the two GB masks.

    ARE is adapted Rand *error* -- lower is better.
    VI is the sum of the split and merge components, per spec.
    """
    lt, lp = grain_labels(gb_true), grain_labels(gb_pred)
    are, _, _ = adapted_rand_error(lt, lp)
    vi_split, vi_merge = variation_of_information(lt, lp)
    return float(are), float(vi_split + vi_merge)


# --------------------------------------------------------------------------
# object-level (SF)
# --------------------------------------------------------------------------

def sf_objects(mask: np.ndarray) -> np.ndarray:
    """Label SF connected components.

    """
    structure = ndimage.generate_binary_structure(2, config.SF_CONNECTIVITY)
    labels, n = ndimage.label(mask.astype(bool), structure=structure)
    if config.MIN_SF_OBJECT_PX > 0 and n > 0:
        sizes = np.bincount(labels.ravel())
        too_small = np.where(sizes < config.MIN_SF_OBJECT_PX)[0]
        labels[np.isin(labels, too_small[too_small > 0])] = 0
        labels, _ = ndimage.label(labels > 0, structure=structure)
    return labels


def object_prf(sf_pred: np.ndarray, sf_true: np.ndarray,
               iou_thr: float = config.SF_MATCH_IOU) -> tuple[float, float, float]:
    """Object-level precision, recall, F1 with greedy one-to-one IoU matching.

    Matches are taken in descending IoU order, each predicted and each true
    object used at most once, and only pairs with IoU >= iou_thr count.
    """
    lp, lt = sf_objects(sf_pred), sf_objects(sf_true)
    n_pred, n_true = int(lp.max()), int(lt.max())

    if n_pred == 0 and n_true == 0:
        return 1.0, 1.0, 1.0
    if n_pred == 0:
        return 0.0, 0.0, 0.0
    if n_true == 0:
        return 0.0, 0.0, 0.0

    # joint histogram of (pred label, true label) over overlapping pixels only
    both = (lp > 0) & (lt > 0)
    pairs = (lp[both].astype(np.int64) * (n_true + 1) + lt[both].astype(np.int64))
    counts = np.bincount(pairs, minlength=(n_pred + 1) * (n_true + 1))
    inter = counts.reshape(n_pred + 1, n_true + 1)[1:, 1:]

    area_p = np.bincount(lp.ravel(), minlength=n_pred + 1)[1:]
    area_t = np.bincount(lt.ravel(), minlength=n_true + 1)[1:]
    union = area_p[:, None] + area_t[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)

    tp = 0
    used_p, used_t = set(), set()
    pi, ti = np.where(iou >= iou_thr)
    for k in np.argsort(-iou[pi, ti]):
        a, b = int(pi[k]), int(ti[k])
        if a in used_p or b in used_t:
            continue
        used_p.add(a)
        used_t.add(b)
        tp += 1

    precision = tp / n_pred
    recall = tp / n_true
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def gb_metrics(gb_pred: np.ndarray, gb_true: np.ndarray) -> dict:
    are, vi = are_vi(gb_pred, gb_true)
    return {
        "gb_dice": dice(gb_pred, gb_true),
        "gb_cldice": cl_dice(gb_pred, gb_true),
        "gb_are": are,
        "gb_vi": vi,
    }


def sf_metrics(sf_pred: np.ndarray, sf_true: np.ndarray) -> dict:
    p, r, f1 = object_prf(sf_pred, sf_true)
    return {
        "sf_dice": dice(sf_pred, sf_true),
        "sf_precision": p,
        "sf_recall": r,
        "sf_f1": f1,
    }
