"""SimCLR: contrastive learning with a ResNet-18 encoder (lab cell 5).

The encoder is a CIFAR-adapted ResNet-18 (3x3 stem, no maxpool). Loss is computed
on the projector output ``z``; downstream linear eval uses the encoder output ``h``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross-entropy loss (NT-Xent)."""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        N = z_i.shape[0]
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.mm(z, z.T) / self.temperature
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, float("-inf"))
        labels = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
        return F.cross_entropy(sim, labels)


class SimCLR(nn.Module):
    """ResNet-18 encoder (CIFAR stem) + 2-layer MLP projector."""

    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.projector = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128)
        )

    def forward(self, x_i, x_j):
        h_i = torch.flatten(self.encoder(x_i), 1)
        h_j = torch.flatten(self.encoder(x_j), 1)
        return self.projector(h_i), self.projector(h_j), h_i, h_j

    @torch.no_grad()
    def features(self, x):
        """Frozen-encoder features ``h`` used for linear evaluation (dim 512)."""
        return torch.flatten(self.encoder(x), 1)
