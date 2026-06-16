"""Training loops for SimCLR, DINO, and MAE.

Each function mirrors the corresponding lab cell (6, 16, 28), adds optional
bfloat16 AMP, and returns a history dict (per-epoch losses, epoch times, and for
DINO the per-epoch ``center.norm()``) so callers can build tables and plots.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .models import DINOLoss, MAE, NTXentLoss, SimCLR, build_dino_model
from .utils import amp_autocast


# =============================================================================
# SimCLR (lab cell 6)
# =============================================================================
def train_simclr(loader, epochs=30, lr=3e-4, weight_decay=1e-4, temperature=0.5,
                 device="cuda", amp=True, verbose=True, max_batches=None):
    model = SimCLR().to(device)
    criterion = NTXentLoss(temperature=temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses, epoch_times = [], []
    total_start = time.time()
    for epoch in range(epochs):
        model.train()
        ep, t0 = [], time.time()
        it = tqdm(loader, desc=f"SimCLR {epoch+1}/{epochs}", disable=not verbose)
        for b, (x_i, x_j, _) in enumerate(it):
            if max_batches and b >= max_batches:
                break
            x_i, x_j = x_i.to(device), x_j.to(device)
            with amp_autocast(torch.device(device), amp):
                z_i, z_j, _, _ = model(x_i, x_j)
                loss = criterion(z_i, z_j)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ep.append(loss.item())
        epoch_times.append(time.time() - t0)
        losses.append(float(np.mean(ep)))
        if verbose:
            print(f"Epoch {epoch+1:02d} | Loss: {losses[-1]:.4f} | Time: {epoch_times[-1]:.1f}s")
    total_time = time.time() - total_start
    return model, {"losses": losses, "epoch_times": epoch_times,
                   "time_per_epoch": float(np.mean(epoch_times)), "total_time": total_time}


# =============================================================================
# DINO (lab cell 16)
# =============================================================================
def train_dino(loader, epochs=50, lr=5e-4, weight_decay=0.04, out_dim=256,
               ema_m=0.996, use_centering=True, device="cuda", amp=True, verbose=True,
               max_batches=None):
    student_vit, student_head = build_dino_model(out_dim=out_dim)
    teacher_vit, teacher_head = build_dino_model(out_dim=out_dim)
    student_vit, student_head = student_vit.to(device), student_head.to(device)
    teacher_vit, teacher_head = teacher_vit.to(device), teacher_head.to(device)
    teacher_vit.load_state_dict(student_vit.state_dict())
    teacher_head.load_state_dict(student_head.state_dict())
    for p in teacher_vit.parameters():
        p.requires_grad = False
    for p in teacher_head.parameters():
        p.requires_grad = False

    loss_fn = DINOLoss(out_dim=out_dim, use_centering=use_centering).to(device)
    optimizer = torch.optim.AdamW(
        list(student_vit.parameters()) + list(student_head.parameters()),
        lr=lr, weight_decay=weight_decay,
    )

    losses, center_norms, epoch_times = [], [], []
    total_start = time.time()
    for epoch in range(epochs):
        student_vit.train(); student_head.train()
        ep, t0 = [], time.time()
        it = tqdm(loader, desc=f"DINO {epoch+1}/{epochs}", disable=not verbose)
        for b, (crops, _) in enumerate(it):
            if max_batches and b >= max_batches:
                break
            crops = [c.to(device) for c in crops]
            with amp_autocast(torch.device(device), amp):
                student_out = [student_head(student_vit(c)) for c in crops]
                with torch.no_grad():
                    teacher_out = [teacher_head(teacher_vit(crops[0])),
                                   teacher_head(teacher_vit(crops[1]))]
                loss = loss_fn([s.float() for s in student_out],
                               [t.float() for t in teacher_out])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            with torch.no_grad():
                for s_p, t_p in zip(student_vit.parameters(), teacher_vit.parameters()):
                    t_p.data = ema_m * t_p.data + (1 - ema_m) * s_p.data
                for s_p, t_p in zip(student_head.parameters(), teacher_head.parameters()):
                    t_p.data = ema_m * t_p.data + (1 - ema_m) * s_p.data
            ep.append(loss.item())
        epoch_times.append(time.time() - t0)
        losses.append(float(np.mean(ep)))
        center_norms.append(float(loss_fn.center.norm().item()))
        if verbose:
            print(f"Epoch {epoch+1:02d} | Loss: {losses[-1]:.4f} | "
                  f"Center norm: {center_norms[-1]:.4f} | Time: {epoch_times[-1]:.1f}s")
    total_time = time.time() - total_start
    bundle = {"student_vit": student_vit, "student_head": student_head,
              "teacher_vit": teacher_vit, "teacher_head": teacher_head, "loss_fn": loss_fn}
    history = {"losses": losses, "center_norms": center_norms, "epoch_times": epoch_times,
               "time_per_epoch": float(np.mean(epoch_times)), "total_time": total_time}
    return bundle, history


# =============================================================================
# MAE (lab cell 28)
# =============================================================================
def train_mae(loader, epochs=50, lr=1.5e-4, weight_decay=0.05, mask_ratio=0.75,
              device="cuda", amp=True, verbose=True, max_batches=None):
    model = MAE(mask_ratio=mask_ratio, norm_pix_loss=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                                  betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    losses, epoch_times = [], []
    model.train()
    total_start = time.time()
    for epoch in range(epochs):
        ep, t0 = [], time.time()
        it = tqdm(loader, desc=f"MAE(m={mask_ratio}) {epoch+1}/{epochs}", disable=not verbose)
        for b, (imgs, _) in enumerate(it):
            if max_batches and b >= max_batches:
                break
            imgs = imgs.to(device)
            with amp_autocast(torch.device(device), amp):
                loss, _, _ = model(imgs)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep.append(loss.item())
        scheduler.step()
        epoch_times.append(time.time() - t0)
        losses.append(float(np.mean(ep)))
        if verbose:
            print(f"Epoch {epoch+1:02d} | Recon Loss: {losses[-1]:.4f} | Time: {epoch_times[-1]:.1f}s")
    total_time = time.time() - total_start
    return model, {"losses": losses, "epoch_times": epoch_times, "recon_loss": losses[-1],
                   "time_per_epoch": float(np.mean(epoch_times)), "total_time": total_time}
