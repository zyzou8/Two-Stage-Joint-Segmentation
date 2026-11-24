"""Run a trained two-stage model over a directory of unlabelled micrographs.

`infer.py` only predicts each CV fold's held-out images from the labelled set.  This script
points the same model at an arbitrary folder -- the condition series (`US5min_rs`, `US20min_rs`,
`Control_rs`), which is the measurement the pipeline exists to make.

    python infer_dir.py --images .../data/US5min_rs \
        --ckpt runs_crop/joint/fold0/checkpoint_best.pt --out preds/US5min_rs --tta

Outputs per image: gb_mask.png, sf_mask.png (0/255), optional *_prob.npy, and a per-image row
in summary.csv with the grain and fault statistics -- grain count, equivalent-diameter
distribution, faults per grain, SF orientation.  Those statistics, not the masks, are the
deliverable.

The encoder is read from the checkpoint, so no environment variable has to match the training
run.  Inference is at the image's native resolution; the crop-trained models were evaluated the
same way (`runs_crop/logs/CVF.log`: "full=1024 whole image, crop=512 random windows" describes
training, not evaluation).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config                                            # noqa: E402
from augment import percentile_normalize                 # noqa: E402
from model import SingleStage, TwoStage, TwoStageFringe, TwoStageRidge, TwoStageSigned  # noqa: E402

KINDS = {"joint": TwoStage, "gb": SingleStage, "sf": SingleStage,
         "joint_fringe": TwoStageFringe, "joint_ridge": TwoStageRidge,
         "joint_signed": TwoStageSigned}
MIN_GRAIN_PX = 25


def load_gray(path: Path) -> np.ndarray:
    a = np.array(Image.open(path))
    return a[..., 0] if a.ndim == 3 else a


def load_model(ckpt_path: Path, device):
    state = torch.load(ckpt_path, map_location=device)
    kind = state.get("kind", "joint")
    enc = state.get("encoder", config.ENCODER)
    weights = state["ema"] if "ema" in state else state["model"]
    cls = KINDS[kind]
    model = cls(pretrained=False, encoder=enc) if cls is not SingleStage else \
        SingleStage(pretrained=False, encoder=enc)
    model.load_state_dict(weights)
    print(f"[model] {ckpt_path}  kind={kind}  encoder={enc}  "
          f"weights={'ema' if 'ema' in state else 'raw'}  "
          f"trained_on={len(state.get('train_names', []))} images")
    return model.to(device).eval(), kind


def _d4(x, k, flip):
    x = torch.rot90(x, k, dims=(2, 3))
    return torch.flip(x, dims=(3,)) if flip else x


def _d4_inv(y, k, flip):
    if flip:
        y = torch.flip(y, dims=(3,))
    return torch.rot90(y, -k, dims=(2, 3))


@torch.no_grad()
def _fwd(model, kind, x):
    if kind.startswith("joint"):
        return model(x, gb_target=None, p_teacher=0.0)
    out = model(x)
    return (out, None) if kind == "gb" else (None, out)


@torch.no_grad()
def predict(model, kind, img_u8, device, tta: bool):
    x = torch.from_numpy(percentile_normalize(img_u8)[None, None]).to(device)
    if not tta:
        gb, sf = _fwd(model, kind, x)
        return (None if gb is None else gb[0, 0].float().cpu().numpy(),
                None if sf is None else sf[0, 0].float().cpu().numpy())
    agb = asf = None
    n = 0
    for k in range(4):
        for flip in (False, True):
            gb, sf = _fwd(model, kind, _d4(x, k, flip))
            if gb is not None:
                g = _d4_inv(gb, k, flip).float()
                agb = g if agb is None else agb + g
            if sf is not None:
                s = _d4_inv(sf, k, flip).float()
                asf = s if asf is None else asf + s
            n += 1
    return (None if agb is None else (agb / n)[0, 0].cpu().numpy(),
            None if asf is None else (asf / n)[0, 0].cpu().numpy())


def stats(gb_prob, sf_prob, gb_thr, sf_thr, px_per_nm=None):
    """Grain and fault statistics -- the actual measurement."""
    import sf_geometry as G
    out = {}
    if gb_prob is not None:
        lab, areas, border = G.label_grains(gb_prob >= gb_thr, MIN_GRAIN_PX)
        n = int(lab.max())
        a = areas[1:][areas[1:] > 0] if len(areas) > 1 else np.array([])
        d = 2.0 * np.sqrt(a / np.pi) if a.size else np.array([])
        interior = ~border[1:] if len(border) > 1 else np.array([], bool)
        out.update(n_grains=n,
                   n_grains_interior=int(interior.sum()) if interior.size else 0,
                   grain_area_mean=float(a.mean()) if a.size else float("nan"),
                   eqdiam_mean_px=float(d.mean()) if d.size else float("nan"),
                   eqdiam_median_px=float(np.median(d)) if d.size else float("nan"),
                   eqdiam_p10_px=float(np.percentile(d, 10)) if d.size else float("nan"),
                   eqdiam_p90_px=float(np.percentile(d, 90)) if d.size else float("nan"),
                   eqdiam_std_px=float(d.std(ddof=1)) if d.size > 1 else float("nan"),
                   gb_area_frac=float((gb_prob >= gb_thr).mean()))
        if px_per_nm:
            for k in ("eqdiam_mean_px", "eqdiam_median_px"):
                out[k.replace("_px", "_nm")] = out[k] / px_per_nm
    if sf_prob is not None:
        sfb = sf_prob >= sf_thr
        _, bands = G.components(sfb, G.SF_MIN_PX)      # [(ys, xs), ...], 8-connected, >= 30 px
        out.update(sf_area_frac=float(sfb.mean()), n_sf=len(bands))
        if gb_prob is not None and out.get("n_grains"):
            out["sf_per_grain"] = len(bands) / out["n_grains"]
        thetas, lengths = [], []
        for ys, xs in bands:
            fb = G.fit_band(xs, ys)                    # theta already in degrees, [0, 180)
            lengths.append(fb["length"])
            if fb["valid"]:                            # elongation >= MIN_ELONGATION
                thetas.append(fb["theta"])
        if lengths:
            out["sf_length_mean_px"] = float(np.mean(lengths))
            out["sf_length_median_px"] = float(np.median(lengths))
        out["n_sf_oriented"] = len(thetas)
        if thetas:
            # double-angle mean: orientation is 180-periodic, so naive averaging is wrong
            t2 = np.radians(2 * np.asarray(thetas))
            out["sf_orient_concentration_R"] = float(
                np.hypot(np.cos(t2).mean(), np.sin(t2).mean()))
            out["sf_orient_mean_deg"] = float(
                (np.degrees(np.arctan2(np.sin(t2).mean(), np.cos(t2).mean())) / 2) % 180)
    return out


def save_overlay(img_u8, gb, sf, gb_thr, sf_thr, path, title=None):
    """Green GB / blue SF burned onto the micrograph, saved at native resolution."""
    rgb = np.repeat(img_u8[..., None], 3, axis=2).astype(np.uint8)
    if sf is not None:
        rgb[sf >= sf_thr] = (0, 102, 255)
    if gb is not None:
        rgb[gb >= gb_thr] = (0, 255, 0)      # GB drawn last: boundaries win where they overlap
    Image.fromarray(rgb, "RGB").save(path)


def save_check_figure(rows, img_dir, out_dir):
    """Two columns: original micrograph | overlay.  Three images picked by grain-count rank
    (fewest / median / most), a fixed rule rather than by eye."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rs = sorted(rows, key=lambda r: r.get("n_grains", 0))
    picks = [("fewest grains", rs[0]), ("median", rs[len(rs) // 2]), ("most grains", rs[-1])]
    fig, ax = plt.subplots(3, 2, figsize=(10, 15))
    for i, (tag, r) in enumerate(picks):
        stem = r["image"].rsplit(".", 1)[0]
        orig = load_gray(img_dir / r["image"])
        ov = np.array(Image.open(out_dir / stem / "overlay.png"))
        ax[i, 0].imshow(orig, cmap="gray")
        ax[i, 0].set_title(f"{tag} — {stem[:24]}\noriginal", fontsize=10)
        ax[i, 1].imshow(ov)
        ax[i, 1].set_title(f"overlay — GB green / SF blue\n"
                           f"grains {r.get('n_grains','-')}   SF {r.get('n_sf','-')}", fontsize=10)
        for j in (0, 1):
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    fig.suptitle(f"{img_dir.name} — images chosen by grain-count rank", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "overlay_check.png", dpi=120)
    plt.close(fig)


def save_contact_sheet(rows, img_dir, out_dir, gb_thr, sf_thr, ncol=6):
    """One thumbnail per image, ordered by grain count -- a whole-series sanity check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rs = sorted(rows, key=lambda r: r.get("n_grains", 0))
    nrow = int(np.ceil(len(rs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.45 * nrow))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, r in zip(np.ravel(axes), rs):
        stem = r["image"].rsplit(".", 1)[0]
        ov = np.array(Image.open(out_dir / stem / "overlay.png").resize((320, 320)))
        ax.imshow(ov)
        ax.set_title(f"{stem[:11]}\ngrains {r.get('n_grains','-')}  SF {r.get('n_sf','-')}",
                     fontsize=7)
    fig.suptitle(f"{img_dir.name} — GB green / SF blue, ordered by grain count", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "contact_sheet.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="directory of PNG/TIF micrographs")
    ap.add_argument("--ckpt", default="runs_crop/joint/fold0/checkpoint_best.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tta", action="store_true", help="8x D4 test-time augmentation")
    ap.add_argument("--gb-thr", type=float, default=config.PRED_THRESHOLD)
    ap.add_argument("--sf-thr", type=float, default=config.PRED_THRESHOLD)
    ap.add_argument("--save-prob", action="store_true")
    ap.add_argument("--no-overlay", action="store_true",
                    help="skip the per-image overlay.png and the contact sheet")
    ap.add_argument("--px-per-nm", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--figures-only", action="store_true",
                    help="rebuild overlay_check/contact_sheet from an existing --out, no inference")
    a = ap.parse_args()

    img_dir = Path(a.images)
    files = sorted([p for p in img_dir.iterdir()
                    if p.suffix.lower() in (".png", ".tif", ".tiff", ".jpg")])
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise SystemExit(f"no images in {img_dir}")
    out_dir = Path(a.out or (HERE / "preds" / img_dir.name))
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.figures_only:
        rows = [{k: (float(v) if k not in ("image",) and v not in ("", None) else v)
                 for k, v in r.items()}
                for r in csv.DictReader(open(out_dir / "summary.csv"))]
        save_check_figure(rows, img_dir, out_dir)
        save_contact_sheet(rows, img_dir, out_dir, a.gb_thr, a.sf_thr)
        print(f"[write] {out_dir/'overlay_check.png'}\n[write] {out_dir/'contact_sheet.png'}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, kind = load_model(Path(a.ckpt), device)
    print(f"[data] {len(files)} images from {img_dir}")
    print(f"[out ] {out_dir}   tta={a.tta}  gb_thr={a.gb_thr}  sf_thr={a.sf_thr}\n")

    rows, t0 = [], time.time()
    for i, p in enumerate(files, 1):
        img = load_gray(p)
        gb, sf = predict(model, kind, img, device, a.tta)
        d = out_dir / p.stem
        d.mkdir(exist_ok=True)
        if gb is not None:
            Image.fromarray(((gb >= a.gb_thr) * 255).astype(np.uint8), "L").save(d / "gb_mask.png")
        if sf is not None:
            Image.fromarray(((sf >= a.sf_thr) * 255).astype(np.uint8), "L").save(d / "sf_mask.png")
        if a.save_prob:
            if gb is not None:
                np.save(d / "gb_prob.npy", gb.astype(np.float16))
            if sf is not None:
                np.save(d / "sf_prob.npy", sf.astype(np.float16))
        if not a.no_overlay:
            save_overlay(img, gb, sf, a.gb_thr, a.sf_thr, d / "overlay.png")
        r = {"image": p.name, "height": img.shape[0], "width": img.shape[1]}
        r.update(stats(gb, sf, a.gb_thr, a.sf_thr, a.px_per_nm))
        rows.append(r)
        print(f"  [{i}/{len(files)}] {p.name[:46]:<46} "
              f"grains={r.get('n_grains','-'):>4}  eqdiam={r.get('eqdiam_mean_px',float('nan')):6.1f}px  "
              f"SF={r.get('n_sf','-'):>3}", flush=True)

    keys = sorted({k for r in rows for k in r})
    with open(out_dir / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image"] + [k for k in keys if k != "image"])
        w.writeheader()
        w.writerows(rows)
    num = [k for k in keys if k != "image" and
           all(isinstance(r.get(k), (int, float)) and np.isfinite(r.get(k, np.nan)) for r in rows)]
    agg = {k: {"mean": float(np.mean([r[k] for r in rows])),
               "std": float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0}
           for k in num}
    json.dump({"images": len(rows), "checkpoint": str(a.ckpt), "tta": a.tta,
               "gb_thr": a.gb_thr, "sf_thr": a.sf_thr, "aggregate": agg},
              open(out_dir / "aggregate.json", "w"), indent=1)
    if not a.no_overlay:
        save_check_figure(rows, img_dir, out_dir)
        save_contact_sheet(rows, img_dir, out_dir, a.gb_thr, a.sf_thr)
        print(f"[write] {out_dir/'contact_sheet.png'}  (+ overlay.png per image)")
    print(f"\n[write] {out_dir/'summary.csv'}  ({len(rows)} rows, {time.time()-t0:.0f}s)")
    for k in ("n_grains", "eqdiam_mean_px", "n_sf", "sf_per_grain", "sf_orient_concentration_R"):
        if k in agg:
            print(f"   {k:<28} {agg[k]['mean']:.3f} +- {agg[k]['std']:.3f}")


if __name__ == "__main__":
    main()
