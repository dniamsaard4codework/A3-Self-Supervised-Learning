"""Frozen-encoder linear evaluation (lab cells 9, 19, 31).

Freeze the encoder, train a single ``nn.Linear(feat_dim, 10)`` on CIFAR-10 labels,
report test accuracy, and return test-set embeddings/labels for t-SNE.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def _collect(feature_fn, loader, device):
    feats, labels = [], []
    for imgs, lbl in loader:
        imgs = imgs.to(device)
        feats.append(feature_fn(imgs).cpu())
        labels.append(lbl)
    return torch.cat(feats), torch.cat(labels)


def linear_eval(feature_fn, feat_dim, train_loader, test_loader, device,
                epochs=10, lr=1e-3, verbose=True, max_batches=None):
    """Train a linear probe on frozen features; return (test_acc%, emb, labels).

    ``feature_fn(imgs)`` must return ``(N, feat_dim)`` features with grads disabled.
    """
    clf = nn.Linear(feat_dim, 10).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)

    for epoch in range(epochs):
        clf.train(); correct = total = 0
        it = tqdm(train_loader, desc=f"Linear Eval {epoch+1}/{epochs}", disable=not verbose)
        for b, (imgs, labels) in enumerate(it):
            if max_batches and b >= max_batches:
                break
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                h = feature_fn(imgs)
            logits = clf(h)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += (logits.argmax(1) == labels).sum().item(); total += labels.size(0)
        if verbose:
            print(f"  Train Acc: {correct/total*100:.2f}%")

    clf.eval(); correct = total = 0
    embeddings, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            h = feature_fn(imgs)
            correct += (clf(h).argmax(1) == labels).sum().item(); total += labels.size(0)
            embeddings.append(h.cpu()); all_labels.append(labels.cpu())
    test_acc = correct / total * 100
    if verbose:
        print(f"Linear Eval Test Accuracy: {test_acc:.2f}%")
    return test_acc, torch.cat(embeddings), torch.cat(all_labels)
