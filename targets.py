"""Step 1 -- the target quantities a materials scientist actually extracts.

ONE extraction, applied identically to predictions and to ground truth (main.tex
"Target quantities").  Nothing here is tuned on anything.

    grains  4-connected components of NOT gb_mask, border grains KEPT, no area filter
    faults  8-connected components of sf_mask with >= 30 px

Per image:
    grain-count error   |Ng_pred - Ng_gt| / Ng_gt
    grain-size W1       1-D Wasserstein distance between log10 grain areas (px)
    fault-count error   |Nf_pred - Nf_gt| / Nf_gt
    fault-density error each fault assigned to the GT grain holding the majority of its
                        pixels (pixels on the GT boundary ignored); mean over GT grains of
                        |n_pred_faults - n_gt_faults|

`self_check()` asserts all four are exactly 0 for GT vs GT.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import wasserstein_distance

GT_THRESHOLD = 127          # data.py binarizes the anti-aliased masks here
PRED_THRESHOLD = 0.5
SF_MIN_PX = 30
FOUR = ndi.generate_binary_structure(2, 1)
EIGHT = np.ones((3, 3), int)


def binarize(a, is_pred):
    a = np.asarray(a)
    if a.dtype == bool:
        return a
    return a >= (PRED_THRESHOLD if is_pred else GT_THRESHOLD)


def grain_labels(gb_mask):
    """4-connected components of the non-boundary pixels; border grains kept."""
    lab, _ = ndi.label(~np.asarray(gb_mask, bool), FOUR)
    return lab


def fault_labels(sf_mask, min_px=SF_MIN_PX):
    """8-connected fault components of at least `min_px` pixels, renumbered 1..n."""
    lab, n = ndi.label(np.asarray(sf_mask, bool), EIGHT)
    if n == 0:
        return lab
    sizes = np.bincount(lab.ravel())
    keep = np.flatnonzero(sizes >= min_px)
    keep = keep[keep > 0]
    remap = np.zeros(n + 1, np.int32)
    remap[keep] = np.arange(1, len(keep) + 1)
    return remap[lab]


def grain_areas(lab):
    a = np.bincount(lab.ravel())
    return a[1:][a[1:] > 0].astype(float) if a.size > 1 else np.zeros(0)


def faults_per_grain(fault_lab, gt_grain_lab):
    """Count faults per GT grain, assigning each fault to the grain holding most of it.

    Pixels on the GT boundary (grain label 0) are ignored in the vote; a fault lying
    entirely on the boundary is unassigned and counted nowhere.
    """
    n_g = int(gt_grain_lab.max())
    out = np.zeros(n_g + 1, int)
    n_f = int(fault_lab.max())
    if n_f == 0:
        return out[1:]
    objs = ndi.find_objects(fault_lab)
    for fid, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        m = fault_lab[sl] == fid
        g = gt_grain_lab[sl][m]
        g = g[g > 0]                       # ignore boundary pixels
        if g.size:
            out[np.bincount(g).argmax()] += 1
    return out[1:]


def image_targets(gb_pred, sf_pred, gb_gt, sf_gt, is_pred=True, mask=None):
    """All four target-quantity errors for one image.

    `mask` restricts every quantity to a region defined on the GROUND TRUTH (the interior
    mask from partition.gt_interior_mask).  Grains and faults outside it are dropped from
    both the prediction and the reference, so the comparison stays like-for-like.
    """
    gbp, sfp = binarize(gb_pred, is_pred), binarize(sf_pred, is_pred)
    gbg, sfg = binarize(gb_gt, False), binarize(sf_gt, False)

    lp, lg = grain_labels(gbp), grain_labels(gbg)
    if mask is not None:
        m = np.asarray(mask, bool)
        lp, lg = np.where(m, lp, 0), np.where(m, lg, 0)
        sfp, sfg = sfp & m, sfg & m
    # Count the labels that SURVIVE, not the largest label value.  `ndi.label` numbers in
    # raster order, so once `mask` zeroes part of the image the largest surviving label is
    # no longer the number of grains -- it is merely the index of the last one.  Without
    # `mask` the two agree (labels are 1..N), which is why this never bit the unmasked
    # analyses; with the interior mask it does, and it bites hardest exactly where a
    # perturbation renumbers the labels, e.g. opening the frame fuses every border grain
    # into label 1 and shifts the rest down.  `grain_areas` already drops empty labels.
    ap, ag = grain_areas(lp), grain_areas(lg)
    np_, ng_ = int(ap.size), int(ag.size)

    fp, fg = fault_labels(sfp), fault_labels(sfg)
    nfp, nfg = int(fp.max()), int(fg.max())

    dens_p = faults_per_grain(fp, lg)      # both assigned to GT grains, per the spec
    dens_g = faults_per_grain(fg, lg)

    return {
        "n_grains_pred": np_, "n_grains_gt": ng_,
        "grain_count_err": abs(np_ - ng_) / ng_ if ng_ else float("nan"),
        "grain_size_w1": (float(wasserstein_distance(np.log10(ap), np.log10(ag)))
                          if ap.size and ag.size else float("nan")),
        "n_faults_pred": nfp, "n_faults_gt": nfg,
        "fault_count_err": abs(nfp - nfg) / nfg if nfg else float("nan"),
        "fault_density_err": (float(np.mean(np.abs(dens_p - dens_g)))
                              if dens_g.size else float("nan")),
    }


def self_check(gb_gt, sf_gt):
    """GT scored against itself must give exactly zero on all four errors."""
    t = image_targets(gb_gt, sf_gt, gb_gt, sf_gt, is_pred=False)
    for k in ("grain_count_err", "grain_size_w1", "fault_count_err", "fault_density_err"):
        assert t[k] == 0.0, f"{k} = {t[k]!r}, expected exactly 0.0"
    return t


if __name__ == "__main__":
    import config
    from PIL import Image

    def g(p):
        a = np.array(Image.open(p))
        return a[..., 0] if a.ndim == 3 else a

    print("GT-vs-GT self-check on all 21 images (all four errors must be exactly 0):")
    for i in range(1, 22):
        gb = g(Path(config.GB_MASK_DIR) / f"{i}.png")
        sf = g(Path(config.SF_MASK_DIR) / f"{i}.png")
        t = self_check(gb, sf)
        print(f"  img {i:>2}: grains={t['n_grains_gt']:>4} faults={t['n_faults_gt']:>3}  "
              f"all errors 0.0  OK")
    print("\nUNIT TEST PASSED -- extraction is identical for GT and predictions")
