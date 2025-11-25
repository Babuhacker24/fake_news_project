#!/usr/bin/env python3
"""
Deeper neural network classifier using precomputed embeddings,
with the SAME architecture as nn_tf_idf_2 (3 hidden layers + 1 logit).

Inputs (defaults):
  X: <repo>/artifacts/embeddings_minilm.npy
  y: <repo>/artifacts/preprocessed_news.csv (column: 'label' as 0/1)

Run:
  python src/nn_embeddings_2.py
  # or override
  python src/nn_embeddings_2.py --X artifacts/embeddings_minilm.npy --labels-csv artifacts/preprocessed_news.csv
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------- dataset ----------
class EmbeddingsDataset(Dataset):
    def __init__(self, X, y):
        # X: dense numpy array (N, D)
        self.X = torch.tensor(X, dtype=torch.float32)
        # y will be used as float for BCEWithLogitsLoss
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------- model (same as nn_tf_idf_2, but for dense embeddings) ----------
class DeepNN(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 1),  # single logit (same as nn_tf_idf_2)
        )

    def forward(self, x):
        # returns logits shape (N,)
        return self.net(x).squeeze(1)


# ---------- metrics ----------
@dataclass
class Metrics:
    acc: float
    f1: float
    report: str
    cm: np.ndarray


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Metrics:
    model.eval()
    ys, probs = [], []

    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)

        logits = model(Xb)
        p = torch.sigmoid(logits).cpu().numpy()

        ys.append(yb.cpu().numpy())
        probs.append(p)

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(probs)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    return Metrics(acc=acc, f1=f1, report=report, cm=cm)


# ---------- main ----------
def main(X_path: str, labels_csv: str):
    repo = Path(__file__).resolve().parents[1]

    # Load embeddings
    X = np.load(repo / X_path)
    # Load labels and normalize to 0/1 ints, then float for BCE
    df = pd.read_csv(repo / labels_csv)
    y_raw = df["label"]
    # in your data they’re already 0/1, but this keeps it robust:
    y = (
        y_raw.astype(str)
        .str.lower()
        .str.strip()
        .replace({"true": 1, "false": 0, "real": 1, "fake": 0, "1": 1, "0": 0})
        .astype(int)
        .values
    ).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_ds = EmbeddingsDataset(X_train, y_train)
    test_ds = EmbeddingsDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepNN(input_dim=X.shape[1], dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    EPOCHS = 20

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0

        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                total_correct += (preds == yb).sum().item()
                total_n += yb.size(0)
                total_loss += loss.item() * yb.size(0)

        avg_loss = total_loss / total_n
        train_acc = total_correct / total_n
        print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={avg_loss:.4f} | train_acc={train_acc:.4f}")

    # ----- Evaluate (same style as nn_tf_idf_2) -----
    train_metrics = evaluate(model, train_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    print("\n== Train ==")
    print(f"Accuracy: {train_metrics.acc:.4f} | F1: {train_metrics.f1:.4f}")
    print(train_metrics.report)
    print("Confusion matrix:\n", train_metrics.cm)

    print("\n== Test ==")
    print(f"Accuracy: {test_metrics.acc:.4f} | F1: {test_metrics.f1:.4f}")
    print(test_metrics.report)
    print("Confusion matrix:\n", test_metrics.cm)

    # ----- Save -----
    artifacts = repo / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifacts / "nn_embeddings_2_model.pt")
    print("\nSaved: artifacts/nn_embeddings_2_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--X", default="artifacts/embeddings_minilm.npy")
    parser.add_argument("--labels-csv", default="artifacts/preprocessed_news.csv")
    args = parser.parse_args()
    main(args.X, args.labels_csv)
