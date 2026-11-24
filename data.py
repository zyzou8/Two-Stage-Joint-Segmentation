"""Dataset and 7-fold split.

Pairing rule (spec): the same filename in the GB and SF folders is the same
source image.  Verified 1:1 with no orphans in either direction
(INSPECTION.md 2.1).

The raw image is read from SF_train_rs/images because it is byte-identical to
GB_train_21_relabeled/images (verified for all 21) and the GB side is a
symlink farm.  See config.IMAGE_DIR.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import config
from augment import JointAugmenter


def _stem_key(name: str) -> int:
    """Sort 1.png .. 21.png numerically, not lexically."""
    return int(Path(name).stem)


def list_image_names() -> list[str]:
    """Filenames present in all three folders, numerically sorted.

    Raises if the folders disagree -- a silent intersection would quietly drop
    images and change the fold layout.
    """
    img = {p.name for p in config.IMAGE_DIR.iterdir() if p.suffix == ".png"}
    gb = {p.name for p in config.GB_MASK_DIR.iterdir() if p.suffix == ".png"}
    sf = {p.name for p in config.SF_MASK_DIR.iterdir() if p.suffix == ".png"}

    if not (img == gb == sf):
        raise RuntimeError(
            "image / gb / sf folders do not contain the same filenames.\n"
            f"  image-only: {sorted(img - gb - sf)}\n"
            f"  gb-only   : {sorted(gb - img - sf)}\n"
            f"  sf-only   : {sorted(sf - img - gb)}"
        )
    return sorted(img, key=_stem_key)


def folds(names: list[str] | None = None) -> list[tuple[list[str], list[str]]]:
    """7 contiguous folds of 3 test images each (decision 6).

    Returns [(train_names, test_names), ...].  Every image is tested exactly
    once; 21 / 3 divides evenly so there is no remainder fold.
    """
    names = names or list_image_names()

    if config.HOLDOUT:
        # single fixed split; returned as one "fold" so train/infer/eval reuse
        # the cross-validation machinery unchanged
        tr = [f"{i}.png" for i in config.HOLDOUT_TRAIN]
        va = [f"{i}.png" for i in config.HOLDOUT_VAL]
        missing = [n for n in tr + va if n not in set(names)]
        if missing:
            raise RuntimeError(f"holdout split names not on disk: {missing}")
        if set(tr) & set(va):
            raise RuntimeError(f"train/val overlap: {sorted(set(tr) & set(va))}")
        if len(tr) + len(va) != len(names):
            raise RuntimeError(
                f"holdout split covers {len(tr) + len(va)} of {len(names)} images")
        return [(tr, va)]

    if len(names) != config.N_FOLDS * config.TEST_PER_FOLD:
        raise RuntimeError(
            f"expected {config.N_FOLDS * config.TEST_PER_FOLD} images "
            f"({config.N_FOLDS} folds x {config.TEST_PER_FOLD}), found {len(names)}"
        )
    out = []
    for f in range(config.N_FOLDS):
        lo = f * config.TEST_PER_FOLD
        test = names[lo:lo + config.TEST_PER_FOLD]
        train = [n for n in names if n not in set(test)]
        out.append((train, test))
    return out


def _load_gray(path: Path) -> np.ndarray:
    """Read a PNG as a single-channel uint8 array.

    GB annotations are mode 'L'; SF annotations and all images are RGB with
    R=G=B (INSPECTION.md 2.2).  Taking channel 0 of an RGB is therefore
    lossless, and `convert('L')` would needlessly re-weight identical channels.
    """
    a = np.array(Image.open(path))
    if a.ndim == 3:
        a = a[..., 0]
    return a


def load_sample(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (image_uint8, gb_binary_float, sf_binary_float) for one filename.

    Masks are binarised here, once, at config.MASK_THRESHOLD -- never again
    downstream.  Both annotation sets are anti-aliased, so this threshold moves
    the GB target area by more than 2x (INSPECTION.md 2.4).
    """
    img = _load_gray(config.IMAGE_DIR / name)
    gb = _load_gray(config.GB_MASK_DIR / name)
    sf = _load_gray(config.SF_MASK_DIR / name)

    gb = (gb >= config.MASK_THRESHOLD).astype(np.float32)
    sf = (sf >= config.MASK_THRESHOLD).astype(np.float32)
    return img, gb, sf


class SegDataset(Dataset):
    """One item = (image[1,H,W], gb[1,H,W], sf[1,H,W]) float32.

    Images are loaded once into memory -- 21 x 1024 x 1024 uint8 is ~22 MB, so
    there is no reason to touch disk every epoch.
    """

    def __init__(self, names: list[str], train: bool, seed: int = 0,
                 aug_cfg: dict | None = None):
        self.names = list(names)
        self.train = train
        crop = config.CROP_SIZE if train else None
        self.aug = JointAugmenter(aug_cfg, train=train, seed=seed, crop=crop)
        self.cache = [load_sample(n) for n in self.names]
        # in crop mode each image contributes CROPS_PER_IMAGE windows per epoch
        self.reps = config.CROPS_PER_IMAGE if (train and crop) else 1

    def __len__(self) -> int:
        return len(self.names) * self.reps

    def __getitem__(self, i: int):
        img, gb, sf = self.cache[i // self.reps]
        return self.aug(img, gb, sf)
