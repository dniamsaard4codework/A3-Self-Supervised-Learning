"""Regenerate every figure from persisted state (checkpoints + results.json).

Called by ``run.py --figures`` (the final step of scripts.sh / scripts.ps1). Keeps
all figure orchestration in the package; needs only the saved checkpoints and the
metrics in results.json — no training state in memory.
"""
from __future__ import annotations

import torch

from . import pipeline, visualize
from .data import eval_loaders


@torch.no_grad()
def _emb(feature_fn, device, num_workers, mae=False):
    # MAE was trained/normalized with MAE_MEAN/MAE_STD, so its test features must be
    # extracted under the same normalization (mae=True); SimCLR/DINO use CIFAR stats.
    _, tel = eval_loaders(num_workers=num_workers, mae=mae)
    feats, labels = [], []
    for imgs, y in tel:
        feats.append(feature_fn(imgs.to(device)).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def generate_all_figures(results, device, saved_dir, fig_dir, num_workers=2, n_tsne=2000):
    """Produce all README/notebook figures from saved checkpoints + ``results``."""
    def fig(name):
        return str(fig_dir / name)

    def has(key):
        return key in results and results[key].get("losses")

    # --- per-model loss curves (+ time/epoch bars) ---
    specs = [("simclr", "SimCLR", "seagreen", "NT-Xent Loss"),
             ("dino_default", "DINO", "darkorange", "Cross-Entropy"),
             ("mae_075_main", "MAE (mask=0.75)", "steelblue", "MSE (masked patches)")]
    fname = {"simclr": "simclr_loss.png", "dino_default": "dino_loss.png", "mae_075_main": "mae_loss.png"}
    for key, title, color, ylabel in specs:
        if has(key):
            r = results[key]
            losses = r["losses"]
            times = r.get("epoch_times") or [r.get("time_per_epoch", 0)] * len(losses)
            visualize.plot_loss_curve(losses, times, title, color=color, ylabel=ylabel,
                                      save_path=fig(fname[key]), show=False)

    # --- DINO variant loss overlay ---
    overlay = {label: results[tag]["losses"]
               for tag, label in [("dino_default", "default"),
                                   ("dino_no_centering", "no centering"),
                                   ("dino_no_local", "no local crops")]
               if has(tag)}
    if overlay:
        visualize.plot_loss_overlay(overlay, "DINO variants — training loss",
                                    ylabel="Cross-Entropy",
                                    save_path=fig("dino_variants_loss.png"), show=False)

    # --- DINO center.norm() (Exercise 1a) ---
    cn = results.get("dino_default", {}).get("center_norms")
    if cn:
        visualize.plot_dino_center_norm(cn, save_path=fig("dino_center_norm.png"), show=False)

    # --- MAE mask-ratio summary (Exercise 2) ---
    abl = results.get("mae_ablation", {})
    ratios = [r for r in (0.25, 0.50, 0.75) if f"{r:.2f}" in abl]
    if ratios:
        visualize.plot_mae_maskratio(
            ratios,
            [abl[f"{r:.2f}"].get("recon_loss") for r in ratios],
            [abl[f"{r:.2f}"].get("linear_acc") for r in ratios],
            save_path=fig("mae_maskratio.png"), show=False)

    # --- linear-eval accuracy bars ---
    bar = {}
    for tag, name in [("simclr", "SimCLR"), ("dino_default", "DINO (default)"),
                      ("dino_no_centering", "DINO (no centering)"), ("dino_no_local", "DINO (no local crops)"),
                      ("mae_075_main", "MAE (0.75)")]:
        a = results.get(tag, {}).get("linear_acc")
        if isinstance(a, (int, float)):
            bar[name] = a
    if bar:
        visualize.plot_linear_eval_bars(bar, save_path=fig("linear_eval_bars.png"), show=False)

    # --- DINO attention maps (load student ViT) ---
    dino_ckpt = saved_dir / "dino.pt"
    if dino_ckpt.exists():
        vit = pipeline.load_dino(str(dino_ckpt), device)
        visualize.plot_dino_attention(vit, device, n_images=10,
                                      save_path=fig("dino_attention.png"), show=False)

    # --- MAE reconstruction grid (load full model) ---
    mae_ckpt = saved_dir / "mae_full.pt"
    if mae_ckpt.exists():
        mae = pipeline.load_mae_full(str(mae_ckpt), device)
        visualize.plot_mae_reconstruction(mae, device, n_show=6,
                                          save_path=fig("mae_reconstruction.png"), show=False)

    # --- t-SNE comparison (recompute test embeddings from checkpoints) ---
    embeddings = {}
    if (saved_dir / "simclr.pt").exists():
        m = pipeline.load_simclr(str(saved_dir / "simclr.pt"), device)
        embeddings["SimCLR (ResNet-18)"] = _emb(pipeline.simclr_features(m), device, num_workers)
    if dino_ckpt.exists():
        vit = pipeline.load_dino(str(dino_ckpt), device)
        embeddings["DINO (ViT-Tiny)"] = _emb(pipeline.dino_features(vit), device, num_workers)
    if mae_ckpt.exists():
        mae = pipeline.load_mae_full(str(mae_ckpt), device)
        embeddings["MAE (ViT)"] = _emb(pipeline.mae_features(mae), device, num_workers, mae=True)
    if len(embeddings) >= 2:
        visualize.plot_tsne(embeddings, n_points=n_tsne,
                            save_path=fig("tsne_comparison.png"), show=False)

    print(f"Figures written to {fig_dir}")
