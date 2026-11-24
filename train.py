"""7-fold cross-validated training.

    python train.py --model joint       # two-stage GB -> SF
    python train.py --model gb          # Single GB baseline
    python train.py --model sf          # Single SF baseline
    python train.py --model joint --fold 0      # one fold only

Always run with PYTHONNOUSERSITE=1 -- the user site-packages numpy 2.0.2 breaks
scipy/skimage/sklearn.  See INSPECTION.md 5.

Checkpointing is by fixed epoch budget, not validation (decision 6): with 3
test images per fold there is no validation holdout to select on.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from augment import percentile_normalize
from data import SegDataset, folds, list_image_names, load_sample
from losses import soft_dice_loss, batch_dice_loss, joint_loss, dice_bce_loss, soft_target
from metrics import dice
from model import (TwoStage, SingleStage, TwoStageFringe, TwoStageRidge,
                   TwoStageSigned, SharedTwoStage)
from sf_instance import fits, match_counts, prf

MODELS = ("joint", "joint_shared", "gb", "sf", "joint_fringe", "joint_ridge",
          "joint_signed")
JOINTLIKE = ("joint", "joint_shared", "joint_fringe", "joint_ridge", "joint_signed")


@torch.no_grad()
def validate(model, kind: str, names: list[str], device: torch.device) -> dict:
    """Validation metrics on held-out images.

    The selection criterion is SF object F1 under the project's tolerance
    protocol -- the same quantity the SF-only pipeline selects on, so
    best-epoch choice is made on a comparable signal.  For the `gb` row there
    is no SF output, so GB Dice is used instead.
    """
    model.eval()
    counts, sf_d, gb_d = [], [], []
    for n in names:
        img, gbt, sft = load_sample(n)
        x = torch.from_numpy(percentile_normalize(img)[None, None]).to(device)
        if kind in JOINTLIKE:
            gbp, sfp = model(x, gb_target=None, p_teacher=0.0)
        elif kind == "sf":
            gbp, sfp = None, model(x)
        else:
            gbp, sfp = model(x), None

        if sfp is not None:
            p = sfp[0, 0].float().cpu().numpy() >= config.PRED_THRESHOLD
            g = sft.astype(bool)
            counts.append(match_counts(fits(g), fits(p)))
            sf_d.append(dice(p, g))
        if gbp is not None:
            p = gbp[0, 0].float().cpu().numpy() >= config.PRED_THRESHOLD
            gb_d.append(dice(p, gbt.astype(bool)))

    model.train()
    out = {}
    if counts:
        pr, rc, f1 = prf(counts)
        out.update(val_sf_p=round(pr, 4), val_sf_r=round(rc, 4),
                   val_sf_f1=round(f1, 4), val_sf_dice=round(float(np.mean(sf_d)), 4))
    if gb_d:
        out["val_gb_dice"] = round(float(np.mean(gb_d)), 4)
    out["score"] = out.get("val_sf_f1", out.get("val_gb_dice", 0.0))
    return out


class EMA:
    """Exponential moving average of parameters.

    Buffers (BatchNorm running stats, ImageNet mean/std, fixed kernels) are
    copied from the live model rather than averaged.
    """

    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.param_names = {n for n, _ in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                continue
            if k in self.param_names:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                self.shadow[k].copy_(v.detach().float())   # buffers: copy

    def state_dict(self, model):
        out = {}
        for k, v in model.state_dict().items():
            out[k] = self.shadow[k].to(v.dtype) if k in self.shadow else v.clone()
        return out


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build(kind: str) -> torch.nn.Module:
    if kind == "joint_shared":
        return SharedTwoStage()
    if kind == "joint_fringe":
        return TwoStageFringe()
    if kind == "joint_ridge":
        return TwoStageRidge()
    if kind == "joint_signed":
        return TwoStageSigned()
    return TwoStage() if kind == "joint" else SingleStage()


def train_fold(kind: str, fold: int, train_names: list[str], device: torch.device,
               epochs: int = config.EPOCHS, val_names: list[str] | None = None,
               patience: int | None = None) -> dict:
    """`val_names` enables best-epoch selection + early stopping on that set."""
    set_seed(config.SEED + fold)

    ds = SegDataset(train_names, train=True, seed=config.SEED + fold)
    dl = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True,
                    num_workers=config.NUM_WORKERS, drop_last=False,
                    pin_memory=True, persistent_workers=config.NUM_WORKERS > 0)

    model = build(kind).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.LR,
                            weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP and device.type == "cuda")
    ema = EMA(model, config.EMA_DECAY) if config.EMA_DECAY else None

    out = config.RUN_DIR / kind / f"fold{fold}"
    out.mkdir(parents=True, exist_ok=True)

    history = []
    best = {"score": -1.0, "epoch": -1}
    for epoch in range(epochs):
        model.train()
        p = config.teacher_forcing_p(epoch, epochs) if kind in JOINTLIKE else 0.0
        agg, n = {}, 0
        t0 = time.time()

        for img, gb, sf in dl:
            img, gb, sf = img.to(device), gb.to(device), sf.to(device)
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=config.AMP and device.type == "cuda"):
                if kind in JOINTLIKE:
                    gb_prob, sf_prob = model(img, gb_target=gb, p_teacher=p)
                    # SF gradient flows into stage 1 -- no detach, by design
                    loss, l_gb, l_sf = joint_loss(gb_prob, gb, sf_prob, sf)
                    parts = {"loss": loss, "gb": l_gb, "sf": l_sf}
                else:
                    target = gb if kind == "gb" else sf
                    prob = model(img)
                    if kind == "sf" and config.SOFT_SF_SIGMA:
                        loss = dice_bce_loss(prob, soft_target(target, config.SOFT_SF_SIGMA))
                    elif config.BATCH_DICE:
                        loss = batch_dice_loss(prob, target)
                    else:
                        loss = soft_dice_loss(prob, target)
                    parts = {"loss": loss}

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)

            b = img.shape[0]
            n += b
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + float(v) * b

        sched.step()
        rec = {"epoch": epoch, "p_teacher": round(p, 4),
               "lr": round(sched.get_last_lr()[0], 8),
               "secs": round(time.time() - t0, 1),
               **{k: round(v / max(n, 1), 5) for k, v in agg.items()}}
        if val_names:
            v = validate(model, kind, val_names, device)
            rec.update(v)
            if v["score"] > best["score"] + 1e-6:
                best = {"score": v["score"], "epoch": epoch, **v}
                torch.save({"model": model.state_dict(), "kind": kind, "fold": fold,
                            "epochs": epoch + 1, "train_names": train_names,
                            "val": v, "encoder": config.ENCODER},
                           out / "checkpoint_best.pt")

        history.append(rec)
        if epoch % 5 == 0 or epoch == epochs - 1:
            msg = "  ".join(f"{k}={rec[k]}" for k in rec if k not in ("lr",))
            print(f"[{kind} fold{fold}] {msg}", flush=True)

        if patience and best["epoch"] >= 0 and epoch - best["epoch"] >= patience:
            print(f"[{kind} fold{fold}] early stop at epoch {epoch} "
                  f"(best e{best['epoch']} score={best['score']:.4f})", flush=True)
            break

        # milestone snapshot: lets eval report a curve instead of one epoch
        done = epoch + 1
        if done in config.MILESTONES and done < epochs:
            torch.save({"model": model.state_dict(), "kind": kind, "fold": fold,
                        "epochs": done, "train_names": train_names},
                       out / f"checkpoint_e{done}.pt")

    final = {"model": model.state_dict(), "kind": kind, "fold": fold,
             "epochs": epochs, "train_names": train_names,
             "encoder": config.ENCODER}
    if ema is not None:
        final["ema"] = ema.state_dict(model)
    torch.save(final, out / "checkpoint.pt")
    if epochs in config.MILESTONES:
        torch.save({"model": model.state_dict(), "kind": kind, "fold": fold,
                    "epochs": epochs, "train_names": train_names},
                   out / f"checkpoint_e{epochs}.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    if best["epoch"] >= 0:
        print(f"[{kind} fold{fold}] BEST epoch {best['epoch']}: "
              + "  ".join(f"{k}={v}" for k, v in best.items() if k != "score"), flush=True)
    return {"fold": fold, "final": history[-1], "best": best}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="joint")
    ap.add_argument("--fold", type=int, default=None, help="run one fold only")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--select", action="store_true",
                    help="keep the best-validation-F1 epoch (uses the held-out set)")
    ap.add_argument("--patience", type=int, default=30,
                    help="early-stop patience when --select is on")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = list_image_names()
    split = folds(names)
    print(f"{len(names)} images, {len(split)} folds, device={device}")

    targets = [args.fold] if args.fold is not None else range(len(split))
    summary = []
    for f in targets:
        train_names, test_names = split[f]
        print(f"\n=== fold {f}: {len(train_names)} train, "
              f"test={test_names} ===", flush=True)
        summary.append(train_fold(
            args.model, f, train_names, device, args.epochs,
            val_names=test_names if args.select else None,
            patience=args.patience if args.select else None))

    # fold-suffixed when running a single fold, so parallel per-fold jobs on
    # different GPUs do not clobber each other's summary
    stem = "train_summary" if args.fold is None else f"train_summary_fold{args.fold}"
    path = config.RUN_DIR / args.model / f"{stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
