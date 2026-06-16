"""All visualizations, extended from lab cells 7/17/21/29/33/35.

Every function optionally saves a PNG (for the README) and/or shows inline (for the
notebook). Figures: per-model loss/time, DINO center-norm, MAE reconstruction grid,
DINO attention maps (10 images x all heads), t-SNE comparison, plus summary bars.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.manifold import TSNE

from .data import mae_test_tf
from .models import unpatchify
from .utils import CIFAR_MEAN, CIFAR_STD, CLASSES, DATA_DIR, EVAL_TF, MAE_MEAN, MAE_STD


def _finish(fig, save_path, show):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# Loss / time curves
# =============================================================================
def plot_loss_curve(losses, epoch_times, title, color="steelblue", ylabel="Loss",
                    save_path=None, show=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
    ax1.plot(range(1, len(losses) + 1), losses, marker="o", color=color)
    ax1.set_title(f"{title} Training Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel(ylabel); ax1.grid(True)
    ax2.bar(range(1, len(epoch_times) + 1), epoch_times, color=color)
    ax2.set_title(f"{title} Time per Epoch")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Seconds"); ax2.grid(True, axis="y")
    fig.tight_layout()
    _finish(fig, save_path, show)


def plot_loss_overlay(curves: dict, title, ylabel="Loss", save_path=None, show=True):
    """Overlay several loss curves. ``curves`` maps label -> list of per-epoch losses."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, losses in curves.items():
        ax.plot(range(1, len(losses) + 1), losses, marker="o", ms=3, label=label)
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    _finish(fig, save_path, show)


def plot_dino_center_norm(center_norms, save_path=None, show=True):
    """Exercise 1a: DINO ``center.norm()`` across training epochs."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(center_norms) + 1), center_norms, marker="o", color="darkorange")
    ax.set_title("DINO center.norm() across training epochs (Exercise 1a)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("||center||"); ax.grid(True)
    fig.tight_layout()
    _finish(fig, save_path, show)


# =============================================================================
# MAE reconstruction grid (lab cell 33)
# =============================================================================
def plot_mae_reconstruction(mae_model, device, n_show=6, mask_ratio=0.75, seed=42,
                            save_path=None, show=True):
    prev_ratio = mae_model.encoder.mask_ratio
    mae_model.encoder.mask_ratio = mask_ratio
    mae_model.eval()

    g = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(str(DATA_DIR), train=False, transform=mae_test_tf, download=True),
        batch_size=n_show, shuffle=True, generator=g,
    )
    imgs_viz, _ = next(iter(loader))
    imgs_viz = imgs_viz.to(device)
    with torch.no_grad():
        loss_viz, pred, mask = mae_model(imgs_viz)

    p = mae_model.patch_size
    h_g = w_g = 32 // p
    pred_imgs = unpatchify(pred.float().cpu(), p, h_g, w_g)

    mean_t = torch.tensor(MAE_MEAN).view(3, 1, 1)
    std_t = torch.tensor(MAE_STD).view(3, 1, 1)
    orig_np = (imgs_viz.cpu() * std_t + mean_t).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    pred_np = (pred_imgs * std_t + mean_t).clamp(0, 1).permute(0, 2, 3, 1).numpy()

    mask_exp = mask.cpu().view(-1, h_g, w_g).unsqueeze(1)
    mask_exp = mask_exp.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)
    mask_np = mask_exp.expand(-1, 3, -1, -1).permute(0, 2, 3, 1).numpy()
    masked_np = orig_np.copy()
    masked_np[mask_np.astype(bool)] = 0.5

    fig, axes = plt.subplots(3, n_show, figsize=(2 * n_show, 6))
    for row, (imgs_row, title) in enumerate(zip(
            [orig_np, masked_np, pred_np],
            ["Original", f"Masked ({int(mask_ratio*100)}%)", "Reconstructed"])):
        axes[row, 0].set_ylabel(title, fontsize=11)
        for col in range(n_show):
            axes[row, col].imshow(imgs_row[col])
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    fig.suptitle(f"MAE Reconstruction (CIFAR-10) — recon loss {loss_viz.item():.4f}",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    _finish(fig, save_path, show)
    mae_model.encoder.mask_ratio = prev_ratio
    return float(loss_viz.item())


# =============================================================================
# DINO attention maps (lab cell 21) — 10 images x all heads
# =============================================================================
def plot_dino_attention(student_vit, device, n_images=10, seed=42, save_path=None, show=True):
    student_vit.eval()
    img_mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    img_std = torch.tensor(CIFAR_STD).view(3, 1, 1)

    attentions = {}
    attn_module = student_vit.blocks[-1].attn
    original_forward = attn_module.forward

    def patched_forward(x, **kwargs):
        B, N, C = x.shape
        qkv = attn_module.qkv(x).reshape(
            B, N, 3, attn_module.num_heads, C // attn_module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn_w = (q @ k.transpose(-2, -1)) * attn_module.scale
        attn_w = attn_w.softmax(dim=-1)
        attentions["last"] = attn_w.detach()
        attn_w = attn_module.attn_drop(attn_w)
        x = (attn_w @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_module.proj(x)
        x = attn_module.proj_drop(x)
        return x

    attn_module.forward = patched_forward
    try:
        g = torch.Generator().manual_seed(seed)
        raw_test = torchvision.datasets.CIFAR10(str(DATA_DIR), train=False, transform=EVAL_TF, download=True)
        img_loader = torch.utils.data.DataLoader(raw_test, batch_size=1, shuffle=True, generator=g)

        n_heads = student_vit.blocks[-1].attn.num_heads
        patch_h = patch_w = 32 // 4

        fig, axes = plt.subplots(n_images, n_heads + 1, figsize=(2 * (n_heads + 1), 2 * n_images))
        sample_iter = iter(img_loader)
        for row in range(n_images):
            img_tensor, label = next(sample_iter)
            img_tensor = img_tensor.to(device)
            with torch.no_grad():
                _ = student_vit(img_tensor)
            attn = attentions["last"]
            cls_attn = attn[0, :, 0, 1:]  # (n_heads, n_patches)
            img_disp = torch.clamp(img_tensor[0].cpu() * img_std + img_mean, 0, 1).permute(1, 2, 0).numpy()
            axes[row][0].imshow(img_disp)
            axes[row][0].set_title(CLASSES[label.item()], fontsize=9)
            axes[row][0].axis("off")
            for h in range(n_heads):
                head_map = cls_attn[h].reshape(patch_h, patch_w).cpu().numpy()
                head_map = (head_map - head_map.min()) / (head_map.max() - head_map.min() + 1e-8)
                head_up = np.array(Image.fromarray((head_map * 255).astype(np.uint8)).resize((32, 32)))
                axes[row][h + 1].imshow(img_disp, alpha=0.4)
                axes[row][h + 1].imshow(head_up, cmap="hot", alpha=0.7, vmin=0, vmax=255)
                if row == 0:
                    axes[row][h + 1].set_title(f"Head {h + 1}", fontsize=9)
                axes[row][h + 1].axis("off")
        fig.suptitle("DINO Self-Attention Maps: [CLS] token -> patches\n"
                     "Emergent object focus — no segmentation labels used", fontsize=12, y=1.005)
        fig.tight_layout()
        _finish(fig, save_path, show)
    finally:
        attn_module.forward = original_forward


# =============================================================================
# t-SNE comparison (lab cell 35)
# =============================================================================
def plot_tsne(embeddings: dict, n_points=2000, seed=42, save_path=None, show=True):
    """``embeddings`` maps name -> (emb_tensor, label_tensor)."""
    names = list(embeddings.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(7 * len(names), 6))
    if len(names) == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    rng = np.random.default_rng(seed)
    for ax, name in zip(axes, names):
        emb, lbls = embeddings[name]
        emb, lbls = emb.numpy(), lbls.numpy()
        n = min(n_points, len(emb))
        idx = rng.choice(len(emb), n, replace=False)
        proj = TSNE(n_components=2, random_state=seed, perplexity=30, init="pca").fit_transform(emb[idx])
        for c in range(10):
            m = lbls[idx] == c
            ax.scatter(proj[m, 0], proj[m, 1], c=[colors[c]], label=CLASSES[c], alpha=0.6, s=10)
        ax.set_title(name, fontsize=12); ax.axis("off")
        ax.legend(fontsize=7, markerscale=2)
    fig.suptitle("t-SNE: Learned Representations on CIFAR-10 (no labels used in training)", fontsize=13)
    fig.tight_layout()
    _finish(fig, save_path, show)


# =============================================================================
# Summary bars (extras)
# =============================================================================
def plot_linear_eval_bars(acc_dict: dict, save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(9, 4))
    names = list(acc_dict.keys())
    vals = [acc_dict[n] for n in names]
    x = np.arange(len(names))
    bars = ax.bar(x, vals, color=plt.cm.viridis(np.linspace(0.15, 0.85, len(names))))
    ax.axhline(10, ls="--", color="red", lw=1, label="Random (10%)")
    ax.set_ylabel("Linear Eval Test Acc (%)")
    ax.set_title("Linear Evaluation Accuracy by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _finish(fig, save_path, show)


def plot_mae_maskratio(ratios, recon_losses, accs, save_path=None, show=True):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ratios, recon_losses, "o-", color="steelblue", label="Recon loss")
    ax1.set_xlabel("Mask ratio"); ax1.set_ylabel("Recon loss", color="steelblue")
    ax2 = ax1.twinx()
    ax2.plot(ratios, accs, "s-", color="darkorange", label="Linear eval acc")
    ax2.set_ylabel("Linear eval acc (%)", color="darkorange")
    ax1.set_title("MAE: reconstruction loss vs linear-eval accuracy by mask ratio")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    _finish(fig, save_path, show)
