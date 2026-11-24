"""Single source of truth for every hyperparameter.

Nothing in this pipeline reads a hyperparameter from anywhere else.  Every
switch below can be overridden by an environment variable, so experiments never
require editing this file.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

# Point GB_DATA_ROOT at your dataset; defaults to ./data inside the repo.
# Expected layout is documented in the README under "Dataset layout".
DATA_ROOT = Path(
    os.environ.get("GB_DATA_ROOT", Path(__file__).resolve().parent / "data")
)

# The raw image is byte-identical in the GB and SF folders; it is read from the
# SF side because the GB side is stored as a symlink farm.
IMAGE_DIR = DATA_ROOT / "SF_train_rs" / "images"
GB_MASK_DIR = DATA_ROOT / "GB_train_21_relabeled" / "annotations"
SF_MASK_DIR = DATA_ROOT / "SF_train_rs" / "annotations"

# decision 7 -- applied to the ground-truth label, not the prediction
MASK_THRESHOLD = 127

IMAGE_SIZE = 1024          # decision 8: full resolution, no cropping

# --- random-crop training ------------------------------------------------
# PAPER DEFAULT: random 512x512 crops, eight per image per epoch.  Set
# GB_CROP=0 to train on full 1024x1024 images instead.
# Crops are taken AFTER full-image normalisation and geometric augmentation, so
# normalisation statistics match test time and crops carry no border artifacts.
# With crops enabled some windows contain no SF at all, so the Dice losses are
# computed at BATCH level (pooled intersections/sums) instead of per sample.
_crop_env = os.environ.get("GB_CROP")
CROP_SIZE = (int(_crop_env) or None) if _crop_env is not None else 512
CROPS_PER_IMAGE = 8        # 18 train images x 8 = 144 crops/epoch
BATCH_DICE = CROP_SIZE is not None

# --------------------------------------------------------------------------
# cross-validation -- decision 6
# --------------------------------------------------------------------------

N_FOLDS = 7                # 21 images / 3 test each, no remainder
TEST_PER_FOLD = 3
# No validation holdout: checkpoint by fixed epoch budget instead.

# --------------------------------------------------------------------------
# single fixed split (GB_HOLDOUT=1) -- no cross-validation
# --------------------------------------------------------------------------
# 14 train / 7 held out.  Used for the convergence study: with a held-out set
# that is never trained on, the epoch curve can be read directly.
HOLDOUT = os.environ.get("GB_HOLDOUT", "") == "1"
HOLDOUT_TRAIN = [1, 2, 3, 4, 7, 8, 10, 11, 12, 14, 15, 18, 19, 20]
HOLDOUT_VAL = [5, 6, 9, 13, 16, 17, 21]

# --------------------------------------------------------------------------
# model -- decision 2
# --------------------------------------------------------------------------

ENCODER = os.environ.get("GB_ENCODER", "efficientnet_b7")   # paper default

# --- stage-2 conditioning -------------------------------------------------
# "raw"    : gb_prob fed as-is, [0,1], mean ~0.17 std ~0.34
# "center" : (gb_prob - 0.5) / 0.5  -> [-1,1], zero-centred, std ~0.67
#
# Measured: the ImageNet-normalised image channels have std ~0.83, so raw
# gb_prob enters at 2.5x smaller scale and is not zero-centred.  A conv's
# response -- and its gradient -- scales with input magnitude, so the raw
# channel is handicapped before training starts.
COND_NORM = os.environ.get("GB_COND_NORM", "raw")

# Inject gb_prob at every decoder stage (resized to that resolution) instead of
# only at the stem.  At the stem it must survive the whole encoder; at the
# decoder skips it reaches the SF head directly, at high spatial resolution.
DEEP_COND = os.environ.get("GB_DEEP_COND", "") == "1"
PRETRAINED = True
DECODER_FILTERS = (512, 256, 128, 64, 32)   # mirrors models.py DEFAULT_MODEL_CFG
NORM_GROUPS = 8                             # models.py uses GroupNorm(groups=8)
DROPOUT = 0.10

# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

EPOCHS = 100               # paper default; final-epoch weights, no early stopping

# Checkpoints are also written at these epochs so test metrics can be reported as
# a curve rather than at one selected epoch.  Reporting the whole curve avoids
# choosing an epoch on test data, which would bias every number -- and bias the
# longer-running model most.
MILESTONES = (40, 80, 120, 160, 200, 240)
BATCH_SIZE = 8 if CROP_SIZE else 2   # 512 crops -> 8; full 1024 -> 2
LR = 3e-4
WEIGHT_DECAY = 1e-4        # models.py L2_regularization_strength
AMP = True
NUM_WORKERS = 4
SEED = 0

# --- Recipe B switches ---------------------------------------------------
# EMA of weights (decay), tolerance-aware SF loss (Gaussian sigma in px),
# and D4 test-time augmentation.  All three default OFF so Recipe A and every
# earlier run are untouched.
EMA_DECAY = float(os.environ["GB_EMA"]) if os.environ.get("GB_EMA") else None
SOFT_SF_SIGMA = float(os.environ["GB_SOFT_SF"]) if os.environ.get("GB_SOFT_SF") else None
# PAPER DEFAULT: average over eight orientations (4 rotations x 2 flips) at
# inference, then threshold at 0.5.  Set GB_TTA=0 to disable.
TTA = os.environ.get("GB_TTA", "1") == "1"

# teacher forcing -- decision 3
# p = P(feed ground-truth gb_mask to stage 2); linear 1 -> 0 then held at 0.
TF_START = 1.0
TF_END = 0.0
TF_DECAY_FRACTION = 0.60   # fraction of EPOCHS over which p reaches TF_END


def teacher_forcing_p(epoch: int, total_epochs: int = EPOCHS) -> float:
    """Probability of using the ground-truth GB mask at `epoch` (0-indexed)."""
    span = max(1.0, TF_DECAY_FRACTION * total_epochs)
    frac = min(1.0, epoch / span)
    return TF_START + (TF_END - TF_START) * frac


# --------------------------------------------------------------------------
# evaluation -- decision 4
# --------------------------------------------------------------------------

PRED_THRESHOLD = 0.5       # applied to predicted probabilities at eval time
SF_MATCH_IOU = 0.5         # object-level P/R/F1 matching criterion
SF_CONNECTIVITY = 2        # 8-connectivity for SF connected components
GRAIN_CONNECTIVITY = 1     # 4-connectivity for grains = CC of NOT(gb)
MIN_SF_OBJECT_PX = 0       # 0 = count every component, however small

# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

# GB_RUN_DIR lets a sweep write to its own directory instead of overwriting a
# previous run's checkpoints and predictions.
RUN_DIR = Path(os.environ.get("GB_RUN_DIR", PROJECT_ROOT / "runs"))
