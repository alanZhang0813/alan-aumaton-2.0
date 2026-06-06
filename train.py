"""Train the policy network by behavioral cloning on your moves.

Reads the cached dataset, splits into train/val, optimises cross-entropy, reports
top-1 / top-3 move accuracy on the held-out split (the honest "how like me" metric),
and saves the best model to ``models/policy_<user>.pt`` alongside a small meta.json.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import PolicyNet

MODELS_DIR = "models"


def _accuracy(logits, targets, k=3):
    top1 = (logits.argmax(dim=1) == targets).float().mean().item()
    topk = logits.topk(k, dim=1).indices
    topk_hit = (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()
    return top1, topk_hit


def train(
    X,
    y,
    username,
    epochs=20,
    batch_size=256,
    lr=1e-3,
    val_frac=0.1,
    device=None,
    seed=42,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} | {len(y)} examples")

    # Deterministic shuffle + split.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    n_val = max(1, int(len(y) * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def loader(idx, shuffle):
        ds = TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(y[idx]))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = loader(train_idx, True)
    val_loader = loader(val_idx, False)

    model = PolicyNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"policy_{username.lower()}.pt")
    best_top1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(yb)
        train_loss = running / len(train_idx)

        # Validation.
        model.eval()
        v_loss, v_top1, v_top3, n = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                v_loss += criterion(logits, yb).item() * len(yb)
                t1, t3 = _accuracy(logits, yb)
                v_top1 += t1 * len(yb)
                v_top3 += t3 * len(yb)
                n += len(yb)
        print(
            f"Epoch {epoch:2d} | train_loss {train_loss:.3f} | "
            f"val_loss {v_loss / n:.3f} | val_top1 {v_top1 / n:.1%} | "
            f"val_top3 {v_top3 / n:.1%}"
        )

        if v_top1 / n > best_top1:
            best_top1 = v_top1 / n
            torch.save(model.state_dict(), model_path)
            with open(
                os.path.join(MODELS_DIR, f"meta_{username.lower()}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "username": username,
                        "epoch": epoch,
                        "val_top1": best_top1,
                        "num_examples": int(len(y)),
                    },
                    f,
                    indent=2,
                )

    print(f"Best val top-1 {best_top1:.1%} | saved -> {model_path}")
    return model_path
