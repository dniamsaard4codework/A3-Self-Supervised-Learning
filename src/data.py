"""Datasets, augmentations, and dataloaders for SimCLR / DINO / MAE + linear eval.

Lifted verbatim from the lab note (cells 5, 13, 28, 31) so that downstream
behaviour and results match the class material exactly.
"""
from __future__ import annotations

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from .utils import CIFAR_MEAN, CIFAR_STD, DATA_DIR, EVAL_TF, MAE_MEAN, MAE_STD

_NORM = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)


# =============================================================================
# SimCLR (lab cell 5)
# =============================================================================
class SimCLRAugmentation:
    """Returns two independently augmented views of the same image."""

    def __init__(self, image_size: int = 32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            _NORM,
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class CIFAR10SSL(Dataset):
    """CIFAR-10 returning (view_i, view_j, label) for SimCLR."""

    def __init__(self, root=DATA_DIR, train: bool = True):
        self.dataset = torchvision.datasets.CIFAR10(root=str(root), train=train, download=True)
        self.augment = SimCLRAugmentation()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        x_i, x_j = self.augment(img)
        return x_i, x_j, label


# =============================================================================
# DINO multi-crop (lab cell 13)
# =============================================================================
class DINOAugmentation:
    """2 global crops (scale 0.4-1.0) + ``n_local`` local crops (scale 0.05-0.4).

    Teacher only sees the 2 global crops; the student sees all of them.
    """

    def __init__(self, image_size: int = 32, n_local: int = 4):
        normalize = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        flip_jitter = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
            *flip_jitter,
            transforms.ToTensor(), normalize,
        ])
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.05, 0.4)),
            *flip_jitter,
            transforms.ToTensor(), normalize,
        ])
        self.n_local = n_local

    def __call__(self, img):
        global1 = self.global_transform(img)
        global2 = self.global_transform(img)
        locals_ = [self.local_transform(img) for _ in range(self.n_local)]
        return [global1, global2] + locals_  # teacher uses [0, 1]; student uses all


class CIFAR10DINO(Dataset):
    def __init__(self, root=DATA_DIR, train: bool = True, n_local: int = 4):
        self.dataset = torchvision.datasets.CIFAR10(root=str(root), train=train, download=True)
        self.augment = DINOAugmentation(n_local=n_local)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return self.augment(img), label


def dino_collate(batch):
    """Stack a batch of variable-count crop lists into a list of (N, C, H, W)."""
    crops_list, labels = zip(*batch)
    n_views = len(crops_list[0])
    stacked = [torch.stack([crops_list[i][v] for i in range(len(crops_list))])
               for v in range(n_views)]
    return stacked, torch.tensor(labels)


# =============================================================================
# MAE transforms (lab cells 28, 31)
# =============================================================================
mae_train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MAE_MEAN, MAE_STD),
])
mae_test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MAE_MEAN, MAE_STD),
])
# Stronger augmentation for MAE *linear eval* (lab cell 31).
mae_clf_train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MAE_MEAN, MAE_STD),
])


# =============================================================================
# Dataloader factories
# =============================================================================
def _loader_kwargs(num_workers):
    """persistent_workers avoids re-spawning workers each epoch (costly on Windows)."""
    kw = dict(num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return kw


def simclr_loader(batch_size=256, num_workers=2):
    return DataLoader(CIFAR10SSL(), batch_size=batch_size, shuffle=True,
                      drop_last=True, **_loader_kwargs(num_workers))


def dino_loader(batch_size=64, n_local=4, num_workers=2):
    ds = CIFAR10DINO(n_local=n_local)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                      collate_fn=dino_collate, **_loader_kwargs(num_workers))


def mae_loader(batch_size=128, num_workers=2):
    ds = torchvision.datasets.CIFAR10(str(DATA_DIR), train=True, transform=mae_train_tf, download=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                      **_loader_kwargs(num_workers))


def eval_loaders(batch_size=256, num_workers=2, mae=False):
    """Labeled CIFAR-10 train/test loaders for frozen-encoder linear evaluation."""
    if mae:
        train_tf, test_tf = mae_clf_train_tf, mae_test_tf
    else:
        train_tf = test_tf = EVAL_TF
    train_ds = torchvision.datasets.CIFAR10(str(DATA_DIR), train=True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR10(str(DATA_DIR), train=False, download=True, transform=test_tf)
    trl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **_loader_kwargs(num_workers))
    tel = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **_loader_kwargs(num_workers))
    return trl, tel
