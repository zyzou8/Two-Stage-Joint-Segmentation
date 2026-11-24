"""Inference: write gb_mask.png and sf_mask.png per image.

    python infer.py --model joint

Each fold's checkpoint predicts only that fold's 3 held-out test images, so
every image is predicted exactly once by a model that never saw it.

Test time uses the PREDICTED gb_prob to condition stage 2 -- never the ground
truth (spec).  TwoStage.forward with p_teacher=0 does this.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

import config
from data import folds, list_image_names, load_sample
from augment import percentile_normalize
from model import (TwoStage, SingleStage, TwoStageFringe, TwoStageRidge,
                   TwoStageSigned, SharedTwoStage)


def load_model(kind: str, fold: int, device: torch.device,
               at: int | None = None) -> torch.nn.Module:
    """`at` selects a milestone snapshot; None uses the final checkpoint."""
    name = "checkpoint.pt" if at is None else f"checkpoint_e{at}.pt"
    ckpt = config.RUN_DIR / kind / f"fold{fold}" / name
    if not ckpt.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt} -- run train.py first")
    state = torch.load(ckpt, map_location=device)
    weights = state["ema"] if (config.EMA_DECAY and "ema" in state) else state["model"]
    # pretrained=False: weights come from the checkpoint, no need to fetch ImageNet
    if kind == "joint_shared":
        model = SharedTwoStage(pretrained=False)
    elif kind == "joint_fringe":
        model = TwoStageFringe(pretrained=False)
    elif kind == "joint_ridge":
        model = TwoStageRidge(pretrained=False)
    elif kind == "joint_signed":
        model = TwoStageSigned(pretrained=False)
    else:
        model = (TwoStage(pretrained=False) if kind == "joint"
                 else SingleStage(pretrained=False))
    model.load_state_dict(weights)
    return model.to(device).eval()


JOINTLIKE = ("joint", "joint_shared", "joint_fringe", "joint_ridge", "joint_signed")


def _d4(x, k, flip):
    x = torch.rot90(x, k, dims=(2, 3))
    return torch.flip(x, dims=(3,)) if flip else x


def _d4_inv(y, k, flip):
    if flip:
        y = torch.flip(y, dims=(3,))
    return torch.rot90(y, -k, dims=(2, 3))


@torch.no_grad()
def _forward(model, kind, x):
    if kind in JOINTLIKE:
        gb, sf = model(x, gb_target=None, p_teacher=0.0)
        return gb, sf
    out = model(x)
    return (out, None) if kind == "gb" else (None, out)


@torch.no_grad()
def predict(model: torch.nn.Module, kind: str, img_u8: np.ndarray,
            device: torch.device) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Returns (gb_prob, sf_prob) as float arrays; either may be None.

    With config.TTA the FULL pipeline (stage 1 -> stage 2 on the predicted
    gb_prob) runs on each of the 8 D4 transforms; each output is mapped back
    and the 8 probability maps are averaged BEFORE thresholding.
    """
    x = torch.from_numpy(percentile_normalize(img_u8)[None, None]).to(device)

    if not config.TTA:
        gb, sf = _forward(model, kind, x)
        return (None if gb is None else gb[0, 0].float().cpu().numpy(),
                None if sf is None else sf[0, 0].float().cpu().numpy())

    acc_gb, acc_sf, n = None, None, 0
    for k in range(4):
        for flip in (False, True):
            gb, sf = _forward(model, kind, _d4(x, k, flip))
            if gb is not None:
                g = _d4_inv(gb, k, flip).float()
                acc_gb = g if acc_gb is None else acc_gb + g
            if sf is not None:
                s_ = _d4_inv(sf, k, flip).float()
                acc_sf = s_ if acc_sf is None else acc_sf + s_
            n += 1
    return (None if acc_gb is None else (acc_gb / n)[0, 0].cpu().numpy(),
            None if acc_sf is None else (acc_sf / n)[0, 0].cpu().numpy())


def save_mask(prob: np.ndarray, path) -> None:
    """Write a binary mask as 0/255 uint8 PNG."""
    binary = (prob >= config.PRED_THRESHOLD).astype(np.uint8) * 255
    Image.fromarray(binary, mode="L").save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("joint", "joint_shared", "gb", "sf", "joint_fringe",
                             "joint_ridge", "joint_signed"), default="joint")
    ap.add_argument("--save-prob", action="store_true",
                    help="also write raw probabilities as .npy")
    ap.add_argument("--at", type=int, default=None,
                    help="milestone epoch to evaluate; default = final checkpoint")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = folds(list_image_names())
    sub = "predictions" if args.at is None else f"predictions_e{args.at}"

    for fold, (_, test_names) in enumerate(split):
        model = load_model(args.model, fold, device, at=args.at)
        out_dir = config.RUN_DIR / args.model / f"fold{fold}" / sub
        out_dir.mkdir(parents=True, exist_ok=True)

        for name in test_names:
            img, _, _ = load_sample(name)
            gb, sf = predict(model, args.model, img, device)
            stem = name.rsplit(".", 1)[0]
            d = out_dir / stem
            d.mkdir(exist_ok=True)

            if gb is not None:
                save_mask(gb, d / "gb_mask.png")
                if args.save_prob:
                    np.save(d / "gb_prob.npy", gb.astype(np.float32))
            if sf is not None:
                save_mask(sf, d / "sf_mask.png")
                if args.save_prob:
                    np.save(d / "sf_prob.npy", sf.astype(np.float32))
            print(f"  fold{fold} {name} -> {d}")

    print(f"\ndone: {config.RUN_DIR / args.model}")


if __name__ == "__main__":
    main()
