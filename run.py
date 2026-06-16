#!/usr/bin/env python3
"""A3 Self-Supervised Learning — training & evaluation CLI.

Reproduces every command from the assignment's Submission section, e.g.::

    python run.py --model dino   --epochs 50 --train
    python run.py --model mae    --epochs 50 --train
    python run.py --model dino   --weights saved/dino.pt        --evaluate --linear
    python run.py --model mae    --weights saved/mae_encoder.pt --evaluate --linear
    python run.py --model dino --no-centering --epochs 50 --train
    python run.py --model dino --n-local 0    --epochs 50 --train
    python run.py --model mae  --mask-ratio 0.25 --epochs 50 --train

Every ``--train`` / ``--evaluate`` call records its metrics into
``results/results.json`` (per-epoch losses, time/epoch, DINO center-norm, MAE recon
loss, linear-eval accuracy). After all models are trained/evaluated, ``--figures``
regenerates every plot from those checkpoints + metrics. SimCLR is also supported
(``--model simclr``) for the Exercise-3 comparison.

The end-to-end reproduction (all models + ablations + evaluation + figures) is wired
up in ``scripts.sh`` (bash) and ``scripts.ps1`` (PowerShell).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from src import data, pipeline, train
from src.evaluate import linear_eval
from src.figures import generate_all_figures
from src.utils import FIGURES_DIR, RESULTS_DIR, SAVED_DIR, get_device, set_seed

RESULTS_PATH = RESULTS_DIR / "results.json"


# --- results.json persistence ----------------------------------------------
def load_results(path=RESULTS_PATH):
    if Path(path).exists():
        try:
            return json.loads(Path(path).read_text())
        except Exception:
            return {}
    return {}


def save_results(results, path=RESULTS_PATH):
    Path(path).write_text(json.dumps(results, indent=2))


def persist(results, tag, data_dict):
    """Write ``data_dict`` under ``tag``; ``a/b`` means nested results[a][b]."""
    if "/" in tag:
        parent, child = tag.split("/", 1)
        results.setdefault(parent, {}).setdefault(child, {}).update(data_dict)
    else:
        results.setdefault(tag, {}).update(data_dict)


def train_tag(args):
    if args.model == "simclr":
        return "simclr"
    if args.model == "dino":
        if args.no_centering:
            return "dino_no_centering"
        if args.n_local == 0:
            return "dino_no_local"
        return "dino_default"
    # mae: 0.75 with a long schedule is the "main" model; short runs are ablations
    if abs(args.mask_ratio - 0.75) < 1e-6 and args.epochs >= 20:
        return "mae_075_main"
    return f"mae_ablation/{args.mask_ratio:.2f}"


def eval_tag(args):
    """Derive the results.json key for an --evaluate call from the weights filename."""
    stem = Path(args.weights).stem if args.weights else args.model
    m = re.match(r"mae_(\d{3})_encoder", stem)
    if m:
        return f"mae_ablation/{int(m.group(1)) / 100:.2f}"
    return {
        "simclr": "simclr",
        "dino": "dino_default", "dino_default": "dino_default",
        "dino_no_centering": "dino_no_centering", "dino_no_local": "dino_no_local",
        "mae_encoder": "mae_075_main", "mae_full": "mae_075_main",
    }.get(stem, args.model)


# --- actions -----------------------------------------------------------------
def do_train(args, device):
    set_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rpath = Path(args.results_file)
    tag = train_tag(args)

    if args.model == "simclr":
        loader = data.simclr_loader(batch_size=args.batch_size or 256, num_workers=args.num_workers)
        model, hist = train.train_simclr(loader, epochs=args.epochs, lr=args.lr or 3e-4,
                                         device=device, amp=args.amp)
        ckpt = out_dir / "simclr.pt"
        pipeline.save_simclr(model, ckpt)

    elif args.model == "dino":
        loader = data.dino_loader(batch_size=args.batch_size or 64, n_local=args.n_local,
                                  num_workers=args.num_workers)
        bundle, hist = train.train_dino(loader, epochs=args.epochs, lr=args.lr or 5e-4,
                                        use_centering=not args.no_centering,
                                        device=device, amp=args.amp)
        ckpt = out_dir / f"{tag}.pt"
        pipeline.save_dino(bundle, ckpt)
        if tag == "dino_default":
            pipeline.save_dino(bundle, out_dir / "dino.pt")  # canonical assignment name

    elif args.model == "mae":
        loader = data.mae_loader(batch_size=args.batch_size or 128, num_workers=args.num_workers)
        model, hist = train.train_mae(loader, epochs=args.epochs, lr=args.lr or 1.5e-4,
                                      mask_ratio=args.mask_ratio, device=device, amp=args.amp)
        rr = f"{int(args.mask_ratio * 100):03d}"
        ckpt = out_dir / f"mae_{rr}_encoder.pt"
        pipeline.save_mae(model, ckpt, full_path=out_dir / f"mae_{rr}_full.pt")
        if tag == "mae_075_main":  # canonical assignment names
            pipeline.save_mae(model, out_dir / "mae_encoder.pt", full_path=out_dir / "mae_full.pt")

    # persist metrics
    results = load_results(rpath)
    record = {"epochs": args.epochs, "losses": hist["losses"],
              "epoch_times": hist["epoch_times"], "time_per_epoch": hist["time_per_epoch"]}
    if "center_norms" in hist:
        record["center_norms"] = hist["center_norms"]
    if "recon_loss" in hist:
        record["recon_loss"] = hist["recon_loss"]
    persist(results, tag, record)
    results.setdefault("meta", {}).update(
        {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
         "torch": torch.__version__, "seed": args.seed, "amp_bf16": args.amp})
    save_results(results, rpath)

    print(f"\n[{tag}] saved -> {ckpt}")
    print(f"avg time/epoch: {hist['time_per_epoch']:.1f}s | total: {hist['total_time']/60:.1f} min "
          f"| metrics -> {rpath}")
    return ckpt


def do_evaluate(args, device):
    set_seed(args.seed)
    weights = args.weights or str({"simclr": SAVED_DIR / "simclr.pt", "dino": SAVED_DIR / "dino.pt",
                                   "mae": SAVED_DIR / "mae_encoder.pt"}[args.model])
    tag = eval_tag(args)
    print(f"Loading weights: {weights}  (results key: {tag})")

    if args.model == "simclr":
        model = pipeline.load_simclr(weights, device)
        feat_fn, trl, tel = pipeline.simclr_features(model), *data.eval_loaders(num_workers=args.num_workers)
    elif args.model == "dino":
        vit = pipeline.load_dino(weights, device)
        feat_fn, trl, tel = pipeline.dino_features(vit), *data.eval_loaders(num_workers=args.num_workers)
    elif args.model == "mae":
        model = pipeline.load_mae_encoder(weights, device)
        feat_fn, trl, tel = pipeline.mae_features(model), *data.eval_loaders(num_workers=args.num_workers, mae=True)

    acc, _, _ = linear_eval(feat_fn, pipeline.FEAT_DIM[args.model], trl, tel, device)
    print(f"\n{args.model.upper()} Linear Eval Test Accuracy: {acc:.2f}%")

    rpath = Path(args.results_file)
    results = load_results(rpath)
    persist(results, tag, {"linear_acc": acc})
    save_results(results, rpath)
    print(f"recorded linear_acc -> {rpath} [{tag}]")
    return acc


def do_figures(args, device):
    rpath = Path(args.results_file)
    results = load_results(rpath)
    if not results:
        raise SystemExit(f"No metrics found at {rpath}; train the models first.")
    fig_dir = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    generate_all_figures(results, device, Path(args.out_dir), fig_dir, num_workers=args.num_workers)


def build_parser():
    p = argparse.ArgumentParser(description="A3 SSL — SimCLR / DINO / MAE")
    p.add_argument("--model", choices=["simclr", "dino", "mae"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--train", action="store_true", help="train the model")
    p.add_argument("--evaluate", action="store_true", help="run evaluation")
    p.add_argument("--linear", action="store_true", help="linear-probe evaluation")
    p.add_argument("--figures", action="store_true", help="regenerate all figures from results.json + checkpoints")
    p.add_argument("--weights", type=str, default=None, help="checkpoint path for --evaluate")
    p.add_argument("--no-centering", action="store_true", help="DINO: disable centering (collapse ablation)")
    p.add_argument("--n-local", type=int, default=4, help="DINO: number of local crops")
    p.add_argument("--mask-ratio", type=float, default=0.75, help="MAE: masking ratio")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out-dir", type=str, default=str(SAVED_DIR))
    p.add_argument("--fig-dir", type=str, default=str(FIGURES_DIR))
    p.add_argument("--results-file", type=str, default=str(RESULTS_PATH))
    amp = p.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true", help="bfloat16 mixed precision (default)")
    amp.add_argument("--no-amp", dest="amp", action="store_false", help="full fp32")
    p.set_defaults(amp=True)
    return p


def main():
    args = build_parser().parse_args()
    device = get_device()
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}) | AMP: {args.amp}")
    if not (args.train or args.evaluate or args.figures):
        raise SystemExit("Nothing to do: pass --train, --evaluate, and/or --figures")
    if (args.train or args.evaluate) and not args.model:
        raise SystemExit("--model is required for --train / --evaluate")
    if args.train:
        do_train(args, device)
    if args.evaluate:
        do_evaluate(args, device)
    if args.figures:
        do_figures(args, device)


if __name__ == "__main__":
    main()
