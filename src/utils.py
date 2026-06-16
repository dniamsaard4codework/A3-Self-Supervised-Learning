"""Shared utilities: seeding, device, paths, CIFAR-10 constants, AMP context.

All constants mirror the lab note exactly so that checkpoints and metrics are
reproducible across ``run.py`` (and the shell scripts that drive it) and the notebook.
"""
from __future__ import annotations

import contextlib
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms

# --- Reproducibility ---------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and Torch (CPU + CUDA) for reproducible runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Paths (repo-root relative, independent of the current working dir) ------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SAVED_DIR = ROOT / "saved"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

for _d in (DATA_DIR, SAVED_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --- CIFAR-10 constants ------------------------------------------------------
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# SimCLR / DINO / linear-eval normalization (lab cells 2, 5, 13).
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

# MAE uses a slightly different std (lab cell 28).
MAE_MEAN = [0.4914, 0.4822, 0.4465]
MAE_STD = [0.247, 0.243, 0.261]

# Transform used for frozen-encoder linear evaluation (lab cell 2).
EVAL_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])


# --- Mixed precision ---------------------------------------------------------
def amp_autocast(device: torch.device, enabled: bool):
    """Return a bfloat16 autocast context on CUDA when ``enabled``, else a no-op.

    bfloat16 (not fp16) is used because it needs no GradScaler and its dynamic
    range avoids the instabilities of fp16 with DINO's sharp (tau=0.04) teacher
    softmax. Blackwell tensor cores run bf16 at full speed.
    """
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def count_params(*modules: torch.nn.Module) -> int:
    return sum(p.numel() for m in modules for p in m.parameters())
