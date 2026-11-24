"""Fair partition metrics -- the single definition every script must use.

THE PROBLEM.  `grain_labels` gives boundary pixels label 0.  skimage's
`adapted_rand_error(gt, pred, ignore_labels=(0,))` drops label 0 from the GROUND TRUTH
only; the prediction keeps it, so every predicted boundary pixel lands in one segment that
spans the image.  That segment's size inflates sum(t_j^2) and pushes ARE up, and it inflates
VI badly -- measured 25-68% overstatement per prediction, varying with how thick the
predicted boundary happens to be.  A model is then penalised for boundary thickness rather
than for getting the partition wrong.

THE FIX.  Assign every boundary pixel, in GT and in prediction, to its nearest grain before
scoring, so the two partitions tile the image completely and no label 0 survives.  This is
the standard choice in EM connectomics.  `assert_complete` enforces it.

BORDER GRAINS are a separate issue and are NOT handled by the fix above.  A model that fails
to close the boundary at the image edge merges every border grain into one huge ring, which
dominates an area-weighted metric: on image 21 this alone moves ARE from 0.134 (interior
only) to 0.624 (all grains).  `exclude_border` makes that choice explicit and reportable; it
defaults to False, matching the paper's statement that border grains are kept as partial
grains.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.metrics import adapted_rand_error, variation_of_information

FOUR = ndi.generate_binary_structure(2, 1)


def grain_labels(boundary):
    """4-connected components of the non-boundary pixels; boundary pixels are label 0."""
    lab, _ = ndi.label(~np.asarray(boundary, bool), FOUR)
    return lab


def fill_to_nearest(lab):
    """Assign every label-0 pixel to its nearest labelled grain (exact Euclidean).

    `distance_transform_edt(..., return_indices=True)` gives, for each background pixel, the
    coordinates of the closest foreground pixel, so this is a true nearest-grain assignment
    rather than a fixed-radius dilation that can leave holes on thick boundaries.
    """
    lab = np.asarray(lab)
    holes = lab == 0
    if not holes.any():
        return lab.copy()
    if not (~holes).any():
        return np.ones_like(lab)
    _, (iy, ix) = ndi.distance_transform_edt(holes, return_indices=True)
    return lab[iy, ix]


def assert_complete(lab, name="partition"):
    if (np.asarray(lab) == 0).any():
        raise AssertionError(f"{name} still has label-0 pixels after nearest-grain filling")


def frame_ids(lab):
    return set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}


def gt_interior_mask(gb_gt):
    """Pixels belonging to GT grains that do not touch the frame.

    Defined on the GROUND TRUTH only.  Deriving it from predicted segments would let a
    prediction choose the region it is scored on: a model that leaks at the frame would
    have its own leak excluded from the metric.
    """
    gt = fill_to_nearest(grain_labels(gb_gt))
    return ~np.isin(gt, list(frame_ids(gt)))


def border_closure(gb_pred, gb_gt):
    """How badly does the prediction fuse the GT border grains?

    Reported separately from ARE because a single unclosed frame merges every border grain
    into one ring, which then dominates any area-weighted partition score.
    """
    gt = fill_to_nearest(grain_labels(gb_gt))
    pred = fill_to_nearest(grain_labels(gb_pred))
    gt_border = frame_ids(gt)
    if not gt_border:
        return {"border_grains_gt": 0, "border_grains_fused": 0, "border_fused_frac": 0.0,
                "largest_border_seg_frac": 0.0}
    pb = frame_ids(pred)
    if not pb:
        return {"border_grains_gt": len(gt_border), "border_grains_fused": 0,
                "border_fused_frac": 0.0, "largest_border_seg_frac": 0.0}
    sizes = {j: int((pred == j).sum()) for j in pb}
    big = max(sizes, key=sizes.get)
    swallowed = len(set(np.unique(gt[pred == big])) & gt_border)
    return {"border_grains_gt": len(gt_border), "border_grains_fused": int(swallowed),
            "border_fused_frac": float(swallowed / len(gt_border)),
            "largest_border_seg_frac": float(sizes[big] / pred.size)}


def partition_metrics(gb_pred, gb_gt, fair=True, exclude_border=False):
    """ARE and VI (with merge/split terms) between the two induced grain partitions.

    fair=True            boundary pixels assigned to the nearest grain first (recommended)
    exclude_border=True  additionally drop grains touching the image frame
    """
    gt, pred = grain_labels(gb_gt), grain_labels(gb_pred)
    if fair:
        gt, pred = fill_to_nearest(gt), fill_to_nearest(pred)
        assert_complete(gt, "gt"); assert_complete(pred, "pred")
    if exclude_border:
        keep = gt_interior_mask(gb_gt)          # GT-defined, never prediction-defined
        gt, pred = np.where(keep, gt, 0), np.where(keep, pred, 0)
    ign = () if (fair and not exclude_border) else (0,)
    are, prec, rec = adapted_rand_error(gt, pred, ignore_labels=ign)
    vi_split, vi_merge = variation_of_information(gt, pred, ignore_labels=ign)
    return {"gb_are": float(are), "gb_are_prec": float(prec), "gb_are_rec": float(rec),
            "gb_vi": float(vi_split + vi_merge),
            "gb_vi_merge": float(vi_merge), "gb_vi_split": float(vi_split),
            "n_grains_pred": int(len(np.unique(pred)) - (1 if (pred == 0).any() else 0)),
            "n_grains_gt": int(len(np.unique(gt)) - (1 if (gt == 0).any() else 0))}


# ------------------------------------------------------------------ grain-matching F1

def grain_match_f1(gb_pred, gb_gt, iou_thr=0.5, fair=True):
    """Fraction of GT grains matched one-to-one by a predicted grain at IoU >= thr.

    A prediction can have exactly the right grain count and still score badly here, by
    splitting one grain and merging two others -- which is why this is the target that the
    grain count cannot trivially satisfy.
    """
    gt, pred = grain_labels(gb_gt), grain_labels(gb_pred)
    if fair:
        gt, pred = fill_to_nearest(gt), fill_to_nearest(pred)
    ng, npd = int(gt.max()), int(pred.max())
    if ng == 0 or npd == 0:
        return {"grain_f1": 0.0, "grain_prec": 0.0, "grain_rec": 0.0}
    a = gt.ravel().astype(np.int64) * (npd + 1) + pred.ravel().astype(np.int64)
    inter = np.bincount(a, minlength=(ng + 1) * (npd + 1)).reshape(ng + 1, npd + 1)
    agt, apd = inter.sum(1, keepdims=True), inter.sum(0, keepdims=True)
    union = agt + apd - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    iou[0, :] = 0.0; iou[:, 0] = 0.0
    pairs = [(iou[i, j], i, j) for i, j in zip(*np.nonzero(iou >= iou_thr))]
    pairs.sort(reverse=True)
    ug, up, tp = set(), set(), 0
    for _, i, j in pairs:
        if i in ug or j in up:
            continue
        ug.add(i); up.add(j); tp += 1
    p, r = tp / npd, tp / ng
    return {"grain_f1": float(2 * p * r / (p + r)) if (p + r) else 0.0,
            "grain_prec": float(p), "grain_rec": float(r)}


# ----------------------------------------------------------------------------- unit test

def _test():
    rng = np.random.default_rng(0)
    gb = np.zeros((200, 200), bool)
    gb[::50, :] = True; gb[:, ::50] = True          # grid of 16 grains, 1-px boundary
    thick = ndi.binary_dilation(gb, FOUR, iterations=3)

    for name, lab in (("gt", grain_labels(gb)), ("thick", grain_labels(thick))):
        f = fill_to_nearest(lab)
        assert_complete(f, name)
        assert set(np.unique(f)) == set(np.unique(lab)) - {0}, f"{name}: labels changed"
    print("  fill_to_nearest leaves no label-0 and invents no labels      OK")

    same = partition_metrics(gb, gb, fair=True)
    assert abs(same["gb_are"]) < 1e-12 and abs(same["gb_vi"]) < 1e-12, same
    print(f"  identical masks -> ARE {same['gb_are']:.1e}, VI {same['gb_vi']:.1e}   OK")

    raw = partition_metrics(thick, gb, fair=False)
    fair = partition_metrics(thick, gb, fair=True)
    print(f"  thicker boundary, same partition: raw ARE {raw['gb_are']:.3f} VI "
          f"{raw['gb_vi']:.3f}  ->  fair ARE {fair['gb_are']:.3f} VI {fair['gb_vi']:.3f}")
    assert fair["gb_are"] < raw["gb_are"] and fair["gb_vi"] < raw["gb_vi"], \
        "fair metrics should not punish boundary thickness"
    assert fair["gb_vi"] < 0.05, f"dilating a correct boundary should be ~free, got {fair}"
    print("  dilation no longer penalised                                 OK")

    f1 = grain_match_f1(gb, gb)
    assert abs(f1["grain_f1"] - 1.0) < 1e-12, f1
    merged = gb.copy(); merged[50, :50] = False        # fuse two grains
    f1m = grain_match_f1(merged, gb)
    print(f"  grain F1: identical {f1['grain_f1']:.3f}, one merge {f1m['grain_f1']:.3f}  OK")
    assert f1m["grain_f1"] < f1["grain_f1"]
    print("\nALL PARTITION UNIT TESTS PASS")


if __name__ == "__main__":
    _test()
