#!/usr/bin/env python3
"""
Train a small neural network on *precomputed embeddings*.

Inputs (defaults):
  X: <repo>/artifacts/embeddings_minilm.npy
  y: <repo>/artifacts/preprocessed_news.csv (column: 'label' as 0/1)

Run:
  python src/nn_embeddings.py
  # or override
  python src/nn_embeddings.py --X artifacts/embeddings_minilm.npy --labels-csv artifacts/preprocessed_news.csv
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
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------- model ----------
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.2),

    nn.Linear(128, 2)
)


    def forward(self, x):
        return self.net(x)


# ---------- metrics ----------
@dataclass
class Metrics:
    acc: float
    f1: float
    report: str
    cm: np.ndarray


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Metrics:
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            logits = model(Xb)
            preds = logits.argmax(dim=1)

            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return Metrics(acc=acc, f1=f1, report=report, cm=cm)


# ---------- main ----------
def main(X_path: str, labels_csv: str):
    repo = Path(__file__).resolve().parents[1]

    X = np.load(repo / X_path)
    y = pd.read_csv(repo / labels_csv)["label"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_ds = EmbeddingsDataset(X_train, y_train)
    test_ds = EmbeddingsDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleNN(input_dim=X.shape[1])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 20

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}: Train loss = {total_loss:.4f}")

    # ----- Evaluate (same style as nn_tf_idf) -----
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
    torch.save(model.state_dict(), artifacts / "nn_embeddings_model.pt")
    print("\nSaved: artifacts/nn_embeddings_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--X", default="artifacts/embeddings_minilm.npy")
    parser.add_argument("--labels-csv", default="artifacts/preprocessed_news.csv")
    args = parser.parse_args()
    main(args.X, args.labels_csv)
