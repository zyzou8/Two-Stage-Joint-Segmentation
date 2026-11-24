"""Evaluation: produce every column of the results table.

    python eval.py                     # final checkpoints, all model rows
    python eval.py --model joint
    python eval.py --at 120            # milestone snapshot
    python eval.py --curve             # every milestone, as a curve

SF object P/R/F1 use the project's tolerance protocol (sf_instance.py), with
counts pooled over a fold's images and mean +/- std taken over folds -- matching
`summarize_instance_records` in the original metrics.py.  Pixel metrics are
averaged per image, then over folds.

Reporting a curve (--curve) avoids selecting an epoch on test data, which would
bias every number and bias the longer-running model most.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from PIL import Image

import config
from data import folds, list_image_names, load_sample
from metrics import gb_metrics, dice
from sf_instance import fits, match_counts, prf

ROWS = {
    "gb": ("Single GB", ("gb_dice", "gb_cldice", "gb_are", "gb_vi")),
    "sf": ("Single SF", ("sf_dice", "sf_precision", "sf_recall", "sf_f1")),
    "joint": ("Joint GB+SF", ("gb_dice", "gb_cldice", "gb_are", "gb_vi",
                              "sf_dice", "sf_precision", "sf_recall", "sf_f1")),
    "joint_fringe": ("Joint GB+SF + fringe", ("gb_dice", "gb_cldice", "gb_are", "gb_vi",
                                              "sf_dice", "sf_precision", "sf_recall", "sf_f1")),
    "joint_ridge": ("Joint GB+SF + ridge", ("gb_dice", "gb_cldice", "gb_are", "gb_vi",
                                            "sf_dice", "sf_precision", "sf_recall", "sf_f1")),
    "joint_signed": ("Joint GB+SF + signed ridge", ("gb_dice", "gb_cldice", "gb_are", "gb_vi",
                                                    "sf_dice", "sf_precision", "sf_recall", "sf_f1")),
}
LOWER_BETTER = {"gb_are", "gb_vi"}


def _read_mask(path):
    if not path.exists():
        return None
    a = np.array(Image.open(path))
    if a.ndim == 3:
        a = a[..., 0]
    return a >= 128


def evaluate_model(kind: str, at: int | None = None) -> dict:
    split = folds(list_image_names())
    sub = "predictions" if at is None else f"predictions_e{at}"
    per_image, per_fold = [], []

    for fold, (_, test_names) in enumerate(split):
        pred_dir = config.RUN_DIR / kind / f"fold{fold}" / sub
        pix, counts = [], []

        for name in test_names:
            stem = name.rsplit(".", 1)[0]
            _, gb_true, sf_true = load_sample(name)
            gb_true, sf_true = gb_true.astype(bool), sf_true.astype(bool)

            row = {"fold": fold, "image": name}
            gb_pred = _read_mask(pred_dir / stem / "gb_mask.png")
            sf_pred = _read_mask(pred_dir / stem / "sf_mask.png")

            if gb_pred is not None:
                row.update(gb_metrics(gb_pred, gb_true))
            if sf_pred is not None:
                row["sf_dice"] = dice(sf_pred, sf_true)
                # object counts pool over the fold -- do not average per image
                counts.append(match_counts(fits(sf_true), fits(sf_pred)))

            if len(row) == 2:
                raise FileNotFoundError(
                    f"no predictions for {name} under {pred_dir} -- run infer.py")
            pix.append(row)
            per_image.append(row)

        keys = [k for k in pix[0] if k not in ("fold", "image")]
        rec = {"fold": fold, **{k: float(np.mean([r[k] for r in pix])) for k in keys}}
        if counts:
            p, r, f1 = prf(counts)
            rec.update(sf_precision=p, sf_recall=r, sf_f1=f1)
        per_fold.append(rec)

    keys = [k for k in per_fold[0] if k != "fold"]
    summary = {k: {"mean": float(np.mean([f[k] for f in per_fold])),
                   "std": float(np.std([f[k] for f in per_fold]))} for k in keys}
    return {"model": kind, "at": at, "per_image": per_image,
            "per_fold": per_fold, "summary": summary}


def format_table(results: dict[str, dict]) -> str:
    lines = []
    for kind, (label, cols) in ROWS.items():
        if kind not in results:
            continue
        s = results[kind]["summary"]
        lines.append(f"\n{label}")
        lines.append("-" * len(label))
        for c in cols:
            if c not in s:
                continue
            arrow = " (lower better)" if c in LOWER_BETTER else ""
            lines.append(f"  {c:<14} {s[c]['mean']:.4f} +/- {s[c]['std']:.4f}{arrow}")
    return "\n".join(lines)


def format_curve(curve: dict) -> str:
    """curve[epoch][model] -> summary.  One block per metric."""
    epochs = sorted(curve)
    models = ["gb", "sf", "joint"]
    out = []
    for metric in ("gb_dice", "gb_cldice", "gb_are", "gb_vi",
                   "sf_dice", "sf_precision", "sf_recall", "sf_f1"):
        rows = []
        for m in models:
            vals = [curve[e].get(m, {}).get(metric) for e in epochs]
            if any(v is not None for v in vals):
                rows.append((m, vals))
        if not rows:
            continue
        arrow = " (lower better)" if metric in LOWER_BETTER else ""
        out.append(f"\n{metric}{arrow}")
        out.append("  model   " + "".join(f"{('e'+str(e)):>12}" for e in epochs))
        for m, vals in rows:
            out.append(f"  {m:<8}" + "".join(
                f"{(f'{v:.4f}' if v is not None else '--'):>12}" for v in vals))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=tuple(ROWS), default=None)
    ap.add_argument("--at", type=int, default=None,
                    help="milestone epoch; default = final checkpoint")
    ap.add_argument("--curve", action="store_true",
                    help="evaluate every milestone in config.MILESTONES")
    args = ap.parse_args()

    kinds = [args.model] if args.model else list(ROWS)

    if args.curve:
        curve = {}
        for e in config.MILESTONES:
            got = {}
            for k in kinds:
                try:
                    got[k] = evaluate_model(k, at=e)["summary"]
                    got[k] = {m: v["mean"] for m, v in got[k].items()}
                except FileNotFoundError:
                    pass
            if got:
                curve[e] = got
        if not curve:
            raise SystemExit("no milestone predictions found -- "
                             "run infer.py --at <epoch> first")
        out = config.RUN_DIR / "results_curve.json"
        out.write_text(json.dumps(curve, indent=2))
        print(format_curve(curve))
        print(f"\nwrote {out}")
        return

    results = {}
    for k in kinds:
        if not (config.RUN_DIR / k).exists():
            print(f"skipping '{k}': no run directory")
            continue
        results[k] = evaluate_model(k, at=args.at)

    if not results:
        raise SystemExit("nothing to evaluate -- run train.py and infer.py first")

    config.RUN_DIR.mkdir(parents=True, exist_ok=True)
    stem = "results" if args.model is None else f"results_{args.model}"
    if args.at is not None:
        stem += f"_e{args.at}"
    out = config.RUN_DIR / f"{stem}.json"
    out.write_text(json.dumps(results, indent=2))

    print(format_table(results))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
