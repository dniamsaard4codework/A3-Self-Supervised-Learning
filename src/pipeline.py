"""High-level helpers used by run.py (training, evaluation, figures).

Checkpoint save/load (using the lab's file conventions: ``dino.pt`` holds
student_vit + student_head; ``mae_encoder.pt`` holds the encoder only) and the
frozen feature extractors used for linear evaluation / t-SNE.
"""
from __future__ import annotations

import torch

from .models import MAE, SimCLR, build_dino_model


# --- Save --------------------------------------------------------------------
def save_simclr(model, path):
    torch.save(model.state_dict(), path)


def save_dino(bundle, path):
    torch.save({"student_vit": bundle["student_vit"].state_dict(),
                "student_head": bundle["student_head"].state_dict()}, path)


def save_mae(model, encoder_path, full_path=None):
    """Encoder-only at ``encoder_path`` (assignment convention for --evaluate);
    optionally the full model at ``full_path`` so reconstruction viz can reload."""
    torch.save(model.encoder.state_dict(), encoder_path)
    if full_path is not None:
        torch.save(model.state_dict(), full_path)


# --- Load --------------------------------------------------------------------
def load_simclr(path, device):
    model = SimCLR().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_dino(path, device, out_dim=256):
    student_vit, _ = build_dino_model(out_dim=out_dim)
    student_vit = student_vit.to(device)
    ckpt = torch.load(path, map_location=device)
    student_vit.load_state_dict(ckpt["student_vit"])
    student_vit.eval()
    for p in student_vit.parameters():
        p.requires_grad = False
    return student_vit


def load_mae_encoder(path, device, mask_ratio=0.75):
    model = MAE(mask_ratio=mask_ratio, norm_pix_loss=True).to(device)
    model.encoder.load_state_dict(torch.load(path, map_location=device))
    model.encoder.eval()
    for p in model.encoder.parameters():
        p.requires_grad = False
    return model


def load_mae_full(path, device, mask_ratio=0.75):
    model = MAE(mask_ratio=mask_ratio, norm_pix_loss=True).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# --- Frozen feature extractors (for linear eval / t-SNE) ---------------------
def simclr_features(model):
    return lambda imgs: torch.flatten(model.encoder(imgs), 1)  # dim 512


def dino_features(student_vit):
    return lambda imgs: student_vit(imgs)  # dim = embed_dim (192)


def mae_features(mae_model):
    mae_model.encoder.mask_ratio = 0.0  # see all patches at eval time (lab cell 31)

    def fn(imgs):
        x_vis, _, _ = mae_model.encoder(imgs)
        return x_vis.mean(dim=1)  # global average pool over patch tokens (dim 192)

    return fn


FEAT_DIM = {"simclr": 512, "dino": 192, "mae": 192}
