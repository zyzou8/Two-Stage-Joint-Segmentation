"""The paper's results table, over true 7-fold cross-validation.

PROTOCOL (paper, "Evaluation protocol").  The 21 images are split into seven
folds of three.  Each model is trained on six folds and evaluated on the
held-out fold, so every image is evaluated exactly once by a model that never
saw it.  Hyperparameters and the 0.5 threshold are fixed in advance, only
final-epoch weights are used, and the held-out images are never used for any
choice.

AGGREGATION, exactly as the paper states it:
  * every metric except tolerance F1  -> computed per image, averaged over all 21
  * tolerance F1                      -> counts POOLED within each fold, then
                                         averaged over the seven folds

  The second rule is why this script cannot just average `sf_f1_tol`: an F1 of
  pooled counts is not the mean of per-image F1s.  `primary.sf_scores` returns
  `sf_counts_tol` per image for exactly this reason.

SCORING uses `primary.score_all` -- boundary pixels assigned to the nearest
grain, every column restricted to GT grains that do not touch the frame.  The
root `metrics.py` is NOT used here; it keeps border grains and leaves a label-0
segment, which overstates ARE/VI.

INFERENCE uses 8x D4 TTA and threshold 0.5, per the paper's Setup.

    PYTHONNOUSERSITE=1 python tab_results_cv.py --model joint_shared --gpu 2
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import config
from data import folds, list_image_names, load_sample
from infer import load_model, predict
from primary import score_all
from sf_instance import prf

warnings.filterwarnings("ignore")          # skimage divide warnings on empty masks

# the 11 columns of tab:results, in the paper's order
COLS = [("gb_dice", "GB Dice"), ("gb_cldice", "GB clDice"),
        ("gb_are", "GB ARE"), ("gb_vi", "GB VI"),
        ("grain_count_err", "E_count"), ("grain_size_w1", "W1"),
        ("sf_dice", "PD Dice"), ("sf_f1_iou", "PD F1 (IoU)"),
        ("sf_f1_tol", "PD F1 (tol)"),
        ("fault_count_err", "N_PD err"), ("fault_density_err", "E_dens")]

POOLED = {"sf_f1_tol"}                     # fold-pooled, not image-averaged


def load_gt(stem: str):
    def g(p):
        a = np.array(Image.open(p))
        return a[..., 0] if a.ndim == 3 else a
    return (g(config.GB_MASK_DIR / stem) >= config.MASK_THRESHOLD,
            g(config.SF_MASK_DIR / stem) >= config.MASK_THRESHOLD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="joint_shared",
                    help="joint_shared | joint | sf/gb for the single-task row")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="paper_outputs/tab_results_cv.json")
    a = ap.parse_args()

    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    names = list_image_names()
    assert len(names) == 21, f"expected 21 images, found {len(names)}"

    per_image, per_fold_counts, covered = [], [], []

    for fold, (_, test_names) in enumerate(folds(names)):
        model = load_model(a.model, fold, device, at=None)   # at=None -> final epoch
        fold_counts = []
        for stem in test_names:
            image, _, _ = load_sample(stem)
            gb_prob, sf_prob = predict(model, a.model, image, device)
            gb_gt, sf_gt = load_gt(stem)
            s = score_all(gb_prob >= config.PRED_THRESHOLD,
                          sf_prob >= config.PRED_THRESHOLD, gb_gt, sf_gt)
            s["_image"], s["_fold"] = stem, fold
            per_image.append(s)
            fold_counts.append(s["sf_counts_tol"])
            covered.append(stem)
        per_fold_counts.append(fold_counts)
        del model
        torch.cuda.empty_cache()
        print(f"  fold {fold}: {', '.join(test_names)}", flush=True)

    # every image exactly once -- the claim the caption makes
    assert sorted(covered) == sorted(names), "images not covered exactly once"
    assert len(per_image) == 21, f"{len(per_image)} rows, expected 21"

    mean, std = {}, {}
    for k, _ in COLS:
        if k in POOLED:
            # pool counts inside each fold, F1 per fold, then mean over folds
            f1s = [prf(c)[2] for c in per_fold_counts]
            mean[k], std[k] = float(np.mean(f1s)), float(np.std(f1s))
        else:
            v = [p[k] for p in per_image]
            mean[k], std[k] = float(np.nanmean(v)), float(np.nanstd(v))

    res = {
        "model": a.model,
        "protocol": ("true 7-fold CV, 21 images, each evaluated once by a model "
                     "trained on the other six folds"),
        "encoder": config.ENCODER, "epochs": config.EPOCHS,
        "crop": config.CROP_SIZE, "tta": bool(config.TTA),
        "checkpoint": "final epoch",
        "spread_is": ("std over the 21 images; for PD F1 (tol) std over the 7 "
                      "fold-pooled values"),
        "mean": mean, "std": std,
        "per_image": [{k: p[k] for k, _ in COLS} | {"image": p["_image"],
                                                    "fold": p["_fold"]}
                      for p in per_image],
    }

    print(f"\n{a.model}  ({config.ENCODER}, {config.EPOCHS} ep, "
          f"crop={config.CROP_SIZE}, TTA={bool(config.TTA)})")
    for k, lab in COLS:
        print(f"  {lab:14s} {mean[k]:.3f} +- {std[k]:.3f}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(f"\n[write] {out}")

    print("\nLaTeX row:")
    print(f"    {a.model} & " + " & ".join(f"{mean[k]:.3f}" for k, _ in COLS) + r" \\")


if __name__ == "__main__":
    main()
