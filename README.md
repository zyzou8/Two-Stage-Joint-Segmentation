# Two-Stage Joint Segmentation — Grain Boundaries → Planar Defects

PyTorch implementation of a two-stage pipeline that jointly segments grain
boundaries (GBs) and intragrain planar defects (PDs) in bright-field TEM
micrographs of FAPbI₃ perovskite, and converts the predictions into **per-grain PD
densities**.

GBs and PDs are coupled: PDs terminate on GBs, PD density is only defined per
grain, and a sharp PD can be mistaken for a GB. So rather than segment either in
isolation, an Eff-UNet is **run twice** — Stage 1 predicts the GB probability
map, Stage 2 predicts PDs conditioned on it. Stage 2's input carries the
predicted GB map as a fourth channel, and because nothing is detached, the PD
loss back-propagates into Stage 1.

Each stage gets its **own** network (`--model joint`) — this is the recommended
configuration. A weight-shared variant that runs a single network twice
(`--model joint_shared`) is also implemented, but measures substantially worse on
this dataset.

```
  image ──► Stage 1 (Eff-UNet B7) ──► gb_prob ──┐
    │                                            ▼
    └──────────────────────────► Stage 2 (Eff-UNet B7) ──► pd_mask
                                        ▲             │
                                        └── PD loss ──┘
                      (gb_prob is a 4th input channel, zero-initialised;
                       no detach, so the PD loss also trains Stage 1)
```

Alongside the segmentation, the repo implements the **object-level evaluation**
the paper argues for: VI and ARE for grains, tolerance-based F1 for PDs, each
validated against the quantities a microscopist actually measures.

---

## Install

```bash
git clone https://github.com/zyzou8/Two-Stage-Joint-Segmentation.git
cd Two-Stage-Joint-Segmentation
pip install -r requirements.txt
```

Needs a CUDA GPU for training. The paper recipe (B7, 512² crops, batch 8) uses
~30 GB; drop to `GB_ENCODER=efficientnet_b0` or a smaller `GB_CROP` for less.
Inference runs on CPU.

> **numpy must stay below 2.0.** numpy 2.x breaks the `scipy` / `scikit-image`
> builds this pipeline depends on, and `skimage.metrics` is required for the
> ARE/VI partition metrics. If you have a stray user-site numpy, run with
> `PYTHONNOUSERSITE=1`.

## Dataset layout

Point `GB_DATA_ROOT` at your data; it defaults to `./data`:

```
$GB_DATA_ROOT/
├── SF_train_rs/
│   ├── images/           1.png … 21.png   grayscale micrographs, 1024×1024
│   └── annotations/      1.png … 21.png   planar-defect labels
└── GB_train_21_relabeled/
    └── annotations/      1.png … 21.png   grain-boundary labels
```

```bash
export GB_DATA_ROOT=/path/to/data
```

Images are read from the SF folder because it is byte-identical to the GB folder
(verified across all 21). Both annotation sets are **anti-aliased 8-bit PNGs**,
not binary — see Stage 1 of the pipeline below, where that matters a great deal.

The micrographs themselves are not redistributed here. The 1024×1024 `_rs`
(resized) volumes were produced upstream of this repository; this code consumes
them and does not regenerate them.

---

## The pipeline

### 1 · Preprocessing — `data.py`, `augment.py`

| step | where | detail |
|---|---|---|
| Mask binarisation | `data.py` | **threshold 127**, applied once at load and never again |
| Intensity normalisation | `augment.py` | per-image percentile (0.5–99.5) clip, rescaled to `[0,1]` float32 |
| Geometric augmentation | `augment.py` | D4 flips/rotations, free rotation, scaling, elastic warp — applied to `(image, gb, sf)` with **one shared set of sampled parameters** so all three stay registered |
| Intensity augmentation | `augment.py` | brightness/contrast, gamma, Gaussian noise, **Poisson shot noise** (the dominant noise source in low-dose TEM), blur — image only, masks untouched |
| Interpolation order | `augment.py` | `order=1` for the image, `order=0` for masks, so masks stay strictly binary |
| Splitting | `data.py` | 7-fold CV, 18 train / 3 test, every image tested exactly once |

**Binarisation at 127 is load-bearing, not a detail.** The annotations are
anti-aliased, so the threshold swings the GB target area from ~20 % (`≥1`) down to
~13 % (`≥128`). An earlier Keras version of this pipeline applied `mask > 0.5` to
0–255 data, which silently keeps *every* faint anti-aliasing pixel. This port
does not; if you adapt the loader, keep the threshold explicit.

**Random-crop training is the default** (`GB_CROP=512`, eight crops per image per
epoch), matching the paper: more and more varied samples per epoch than full
images, with the model applied to the full 1024² image at test time. Crops are
taken **after** full-image normalisation and geometric augmentation, so crop
statistics match test time and carry no border artifacts. Because some crops
contain no PD at all, Dice is pooled at batch level (`BATCH_DICE`) rather than per
sample — a per-sample Dice would be undefined on an empty crop.

### 2 · Metrics — `primary.py`, `partition.py`, `targets.py`, `sf_instance.py`

Defined before training, because *how* objects are matched changes the
conclusion more than any architectural choice here.

**`primary.py` is the scorer.** Call `score_all(gb_pred, sf_pred, gb_gt, sf_gt)`
for every reported column at once. Two conventions it enforces, both required to
reproduce the paper:

1. **Boundary pixels are assigned to the nearest grain**, in the prediction *and*
   the ground truth, so the two partitions tile the image and no label-0 segment
   survives. Without this, skimage's `adapted_rand_error` drops label 0 from the
   ground truth only, leaving every predicted boundary pixel in one
   image-spanning segment — a measured **25–68 % overstatement of ARE and VI**
   that penalises boundary thickness rather than a wrong partition.
2. **ARE, VI and the target errors are restricted to ground-truth grains that do
   not touch the image frame**, because a clipped grain's true size is unknown.
   The interior region is defined on the ground truth alone, so a prediction can
   never influence the region it is scored on. Pixel metrics use the whole image.

> `metrics.py` is the **superseded** scorer: it keeps border grains and leaves the
> label-0 segment in place. It is retained only to reproduce the earlier B0 run
> and must not be used for paper numbers.

- **Pixel level** — Dice, clDice (centreline Dice, for thin structures).
- **Partition level** — ARE (adapted Rand error) and VI (variation of
  information), on grains = 4-connected components of `NOT(gb)`, split into merge
  and split terms.
- **Target quantities** (`targets.py`) — what a microscopist actually measures,
  extracted identically from prediction and ground truth: grain-count error
  `|N_pred − N_gt| / N_gt`, grain-size `W₁` (1-D Wasserstein distance between
  log₁₀ grain areas), PD-count error, and PD-density error.
- **PD object level** — `sf_instance.py` implements a **tolerance-based instance
  protocol**, not IoU ≥ 0.5:
  - objects are 8-connected components ≥ 30 px, each fit to a straight segment by
    total least squares;
  - a match requires ≥ 25 % of one segment's length within **4 px** of the other
    *and* orientation agreement within **15°** (the angle test is skipped for
    objects that are not clearly elongated);
  - matching is not one-to-one, so splits and merges are not penalised;
  - counts pool over a fold's images, then mean ± std across folds.

Why this protocol: planar defects are 1–2 px wide, so a **single-pixel offset
roughly halves IoU** and visually-correct predictions score zero. Switching from
IoU ≥ 0.5 to the tolerance rule roughly doubled measured object F1 in earlier
runs. Sanity check: ground truth scored against itself gives exactly
**P = R = F1 = 1.000**, and `primary.score_all` passes a full identity check on
all 21 annotations (Dice/clDice/F1 = 1, ARE/VI/E_count/W₁/E_dens = 0).

`sf_geometry.py` is vendored unchanged from the earlier SF-only pipeline so that
component extraction and segment fitting are bit-identical to previously
published numbers.

### 3 · Training — `model.py`, `losses.py`, `train.py`

```bash
python train.py --model joint          # RECOMMENDED: a network per stage
python train.py --model joint_shared   # variant: ONE network run twice (weaker)
python train.py --model gb             # Stage 1 alone
python train.py --model sf             # PD alone, no conditioning

python train.py --model joint --fold 0 --epochs 5    # quick smoke run
```

**Architecture** — EfficientNet-B7 encoder + U-Net decoder, GroupNorm(8),
decoder filters `(512, 256, 128, 64, 32)`, ImageNet-pretrained.

`joint_shared` is **one** Eff-UNet run twice. Its
input always has four channels — three image channels plus a GB channel that is
**empty in Stage 1** and **holds `gb_prob` in Stage 2**. The same weights and the
same sigmoid head produce boundaries on the first pass and planar defects on the
second, so the GB channel doubles as the task switch. Because the weights are
tied, the PD loss back-propagates into the single network through both passes.
Total: **18.4 M parameters**.

`joint` allocates a **separate network per stage** (**36.8 M** total): Stage 1
takes 3 channels, Stage 2 takes 4. This is the configuration that performs, and
the one to use for results.

**Stage-2 conditioning** — the grayscale image is replicated to RGB and
ImageNet-normalised so the pretrained stem sees its native statistics, with
`gb_prob` appended as a **4th channel whose stem weights are zero-initialised**.
At initialisation Stage 2 is therefore provably identical to Stage 1, and only
then learns to use the conditioning.

**Teacher forcing** — `p = P(feed ground-truth GB instead of gb_prob)` decays
linearly 1.0 → 0.0 over the first 60 % of epochs, then stays at 0. Test time is
always `p = 0`. Note the built-in tension: on a ground-truth step the
conditioning is a constant, so there is **no gradient path to Stage 1 at all**.
Teacher forcing and end-to-end coupling are mutually exclusive — under the default
schedule ~70 % of training steps actually carry SF gradient into Stage 1.

**Loss** — soft Dice, `Dice(gb) + Dice(sf)`, unweighted. No `detach` anywhere.

Key defaults (all in `config.py`): 80 epochs, **AdamW** at `lr 3e-4`,
`weight_decay 1e-4`, cosine-annealed, batch 2 at 1024², AMP on, seed 0.

### 4 · Evaluation — `infer.py`, `infer_dir.py`, `eval.py`

```bash
python infer.py --model joint      # per-fold masks for the labelled test images
python eval.py                     # full results table -> runs/results.json
python eval.py --curve             # metrics at every milestone epoch
```

To apply a trained model to **new, unlabelled** micrographs — which is what the
pipeline exists to do:

```bash
python infer_dir.py --images /path/to/micrographs \
    --ckpt runs/joint/fold0/checkpoint_best.pt \
    --out preds/my_sample --tta
```

This writes `gb_mask.png` / `sf_mask.png` per image plus a `summary.csv` of the
downstream measurements: grain count, equivalent-diameter distribution, faults
per grain, and SF orientation. Those statistics, not the masks, are the actual
scientific output.

Outputs land in `runs/<model>/fold<N>/` as `checkpoint.pt`, `history.json`, and
`predictions/<image>/{gb,sf}_mask.png`.

---

## Reproducing the results table

Reported numbers live in the paper, not here. To regenerate them:

```bash
python tab_results_cv.py --model joint --gpu 0          # recommended
python tab_results_cv.py --model joint_shared --gpu 0   # shared variant
```

**Protocol.** True 7-fold cross-validation: the 21 images are split into seven
folds of three, each model trains on six folds for 100 epochs and is evaluated on
the held-out fold, so **every image is evaluated exactly once by a model that
never saw it** (the script asserts this). Hyperparameters and the 0.5 threshold
are fixed in advance, only final-epoch weights are used, and held-out images are
never used for any choice. Inference averages over eight orientations (4
rotations × 2 flips).

**Aggregation.** Every metric except tolerance F1 is computed per image and
averaged over all 21. Tolerance F1 pools its match counts *within* each fold and
then averages over the seven folds — an F1 of pooled counts is not the mean of
per-image F1s, which is why `primary.sf_scores` returns the raw counts.

Each run writes `paper_outputs/*.json` with the per-image breakdown, and prints a
ready-to-paste LaTeX row.

### Caveats

- **21 images, 7 folds.** The dataset is small, so small differences between
  configurations are fragile.
- **One seed.** `train.py` varies the seed by fold, not by run, so any spread you
  compute is image-to-image variability, not seed variability.
- **No validation split.** The 7×3 CV holds out three images per fold for
  *testing*, with nothing left for model selection, so the 100-epoch budget is
  fixed in advance rather than tuned. All loss curves are training loss.
- Training far past the fixed budget lowers training loss without improving
  held-out performance, which is why the budget is fixed in advance and
  final-epoch weights are used.
- `joint_shared` ties Stage 1 and Stage 2 into one 18.4 M-parameter network, half
  the capacity of `joint`. Differences between the two confound architecture with
  capacity.

---

## Configuration

Everything lives in `config.py`; no hyperparameter is read anywhere else. Common
overrides are environment variables, so experiments never require editing code:

| variable | default | effect |
|---|---|---|
| `GB_DATA_ROOT` | `./data` | dataset location |
| `GB_ENCODER` | `efficientnet_b7` | paper default; `efficientnet_b0` is cheaper |
| `GB_CROP` | `512` | paper default; `GB_CROP=0` trains on full 1024² images |
| `GB_HOLDOUT` | *(off)* | `1` = single 14/7 split instead of 7-fold CV |
| `GB_RUN_DIR` | `./runs` | output directory, so sweeps don't overwrite |
| `GB_TTA` | `1` (on) | 8× D4 test-time augmentation; `GB_TTA=0` disables |
| `GB_EMA` | *(off)* | EMA decay on weights, e.g. `0.999` |
| `GB_COND_NORM` | `raw` | `center` = feed `gb_prob` as `[-1,1]` |
| `GB_DEEP_COND` | *(off)* | `1` = inject `gb_prob` at every decoder stage |

## Files

| file | role |
|---|---|
| `config.py` | every hyperparameter and path |
| **preprocessing** | |
| `data.py` | loading, binarisation at 127, 7-fold split |
| `augment.py` | joint augmentation of `(image, gb, sf)`; percentile normalisation |
| **metrics** | |
| `primary.py` | **the scorer** — every reported column, interior-only, nearest-grain fill |
| `partition.py` | grain partition, ARE/VI, boundary-pixel assignment |
| `targets.py` | target quantities: grain-count error, size `W₁`, PD-density error |
| `metrics_all.py` | Dice, clDice, IoU-matched F1 |
| `sf_instance.py` | PD object P/R/F1, tolerance protocol |
| `sf_geometry.py` | vendored component extraction and segment fitting |
| `metrics.py` | *superseded* scorer; earlier runs only — see the warning above |
| **training** | |
| `model.py` | `EffUNet`, `SharedTwoStage` (paper), `TwoStage`, `SingleStage` |
| `losses.py` | soft Dice, batch-pooled Dice, tolerance-aware PD loss |
| `train.py` | training entry point |
| `fringe.py`, `ridge.py`, `signed_ridge.py` | fixed feature banks for the Stage-2 ablations |
| **evaluation** | |
| `tab_results_cv.py` | **the paper's table** — 7-fold CV, D4 TTA, `primary.py` scoring |
| `infer.py` | per-fold inference on labelled test images |
| `infer_dir.py` | inference on an unlabelled directory + statistics CSV |
| `eval.py` | earlier results table (uses the superseded scorer) |

## A note on naming

The paper calls the intragrain features **planar defects (PDs)**, covering
stacking faults and twin boundaries as one class. The code predates that choice
and calls them **SF** throughout — `sf_mask`, `sf_prob`, `SF_MIN_PX`,
`sf_instance.py`. Read `sf` as `PD` everywhere; the two refer to the same channel
and the same annotations.

## License

MIT — see [LICENSE](LICENSE).
