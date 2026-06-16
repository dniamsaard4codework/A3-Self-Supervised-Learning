"""DINO: self-distillation on a ViT-Tiny backbone (lab cells 14, 15).

Adds one ablation knob over the lab code: ``DINOLoss(use_centering=...)`` lets us
disable the centering subtraction for the collapse ablation (Exercise 1). The
running center buffer is still updated either way, so ``center.norm()`` can always
be logged for the Exercise 1a plot.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    """Projection head: MLP -> L2 normalize -> weight-normed linear (lab cell 14)."""

    def __init__(self, in_dim: int = 192, hidden_dim: int = 512, out_dim: int = 256, n_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, out_dim, bias=False))
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.weight_norm(nn.Linear(out_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


def build_dino_model(out_dim: int = 256):
    """Create a CIFAR-adapted ViT-Tiny backbone + DINO head pair."""
    vit = timm.create_model(
        "vit_tiny_patch16_224", pretrained=False,
        img_size=32, patch_size=4, num_classes=0,
    )
    head = DINOHead(in_dim=vit.embed_dim, out_dim=out_dim)
    return vit, head


class DINOLoss(nn.Module):
    """Cross-entropy between centered/sharpened teacher and student distributions."""

    def __init__(self, out_dim: int = 256, teacher_temp: float = 0.04,
                 student_temp: float = 0.1, center_momentum: float = 0.9,
                 use_centering: bool = True):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.use_centering = use_centering
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out):
        # student_out: list of (N, out_dim) for all crops (global + local).
        # teacher_out: list of (N, out_dim) for the global crops only (idx 0, 1).
        s_probs = [F.log_softmax(s / self.student_temp, dim=-1) for s in student_out]

        if self.use_centering:
            t_probs = [F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach()
                       for t in teacher_out]
        else:
            # Ablation: no centering -> nothing stops one dim from dominating.
            t_probs = [F.softmax(t / self.teacher_temp, dim=-1).detach()
                       for t in teacher_out]

        total_loss = 0.0
        n_loss_terms = 0
        for t_idx, t_prob in enumerate(t_probs):
            for s_idx, s_log_prob in enumerate(s_probs):
                if s_idx == t_idx:  # skip matching student/teacher global crop
                    continue
                loss = -(t_prob * s_log_prob).sum(dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1

        total_loss /= n_loss_terms
        # Always track the center so center.norm() is plottable (Exercise 1a).
        self.update_center(torch.stack(teacher_out).mean(dim=0))
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_mean):
        self.center = self.center * self.center_momentum + teacher_mean * (1 - self.center_momentum)
