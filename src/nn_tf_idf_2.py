#!/usr/bin/env python3
"""
Deeper neural net on precomputed TF-IDF features.

This is a variant of nn_tf_idf.py with more hidden layers.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils import Bunch


# ---------- repo ----------
def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / ".git").exists():
            return p
    return cur.parents[1]


# ---------- dataset wrapper ----------
class CSRRowDataset(Dataset):
    """Wrap a CSR matrix + labels; converts each row to dense *per sample*."""
    def __init__(self, X_csr: sparse.csr_matrix, y: np.ndarray, indices: np.ndarray):
        self.X = X_csr
        self.y = y
        self.idx = indices

    def __len__(self):
        return self.idx.shape[0]

    def __getitem__(self, i):
        r = self.idx[i]
        x_dense = self.X.getrow(r).toarray().astype(np.float32).squeeze(0)  # (D,)
        y_val = np.float32(self.y[r])
        return torch.from_numpy(x_dense), torch.tensor(y_val)


# ---------- deeper model ----------
class MLP(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # logits


# ---------- train/eval ----------
def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    tot_loss, tot_ok, tot_n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        with torch.no_grad():
            preds = (torch.sigmoid(logits) >= 0.5).long()
            tot_ok += (preds.squeeze() == yb.long()).sum().item()
            tot_n += xb.size(0)
            tot_loss += loss.item() * xb.size(0)
    return tot_loss / tot_n, tot_ok / tot_n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        prob = torch.sigmoid(logits).cpu().numpy()
        ys.append(yb.numpy())
        ps.append(prob)
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    y_pred = (y_prob >= 0.5).astype(int)
    return Bunch(
        acc=accuracy_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred),
        report=classification_report(y_true, y_pred, digits=4),
        cm=confusion_matrix(y_true, y_pred),
    )


def main():
    print(">>> Starting nn_tf_idf_2 experiment (deeper MLP)")

    ap = argparse.ArgumentParser()
    ap.add_argument("--X", default=None, help="Path to CSR .npz (default: artifacts/X_tfidf.npz)")
    ap.add_argument("--labels-csv", default=None, help="CSV with 'label' (default: artifacts/preprocessed_news.csv)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    repo = find_repo_root(Path(__file__).resolve())
    X_path = Path(args.X) if args.X else repo / "artifacts" / "X_tfidf.npz"
    y_csv  = Path(args.labels_csv) if args.labels_csv else repo / "artifacts" / "preprocessed_news.csv"

    if not X_path.exists():
        raise FileNotFoundError(f"TF-IDF matrix not found: {X_path}")
    if not y_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {y_csv}")

    # Load features
    X = sparse.load_npz(X_path).astype(np.float32).tocsr()

    # Load + normalize labels
    df = pd.read_csv(y_csv)
    if "label" not in df.columns:
        raise ValueError(f"'label' column not found in {y_csv}; columns: {list(df.columns)}")
    y_raw = df["label"].astype(str).str.lower().str.strip()
    y = y_raw.replace({"true": 1, "false": 0, "real": 1, "fake": 0, "1": 1, "0": 0}).astype(int).values

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Row mismatch: X {X.shape} vs y {y.shape}")

    # Split indices
    idx_all = np.arange(X.shape[0])
    idx_train, idx_test, _, _ = train_test_split(
        idx_all, y, test_size=0.2, random_state=args.seed, stratify=y, shuffle=True
    )

    in_dim = X.shape[1]
    print(f"[TF-IDF] X shape={X.shape}, epochs={args.epochs}, layers=4 (512, 256, 128, 1)")

    # DataLoaders
    train_ds = CSRRowDataset(X, y, idx_train)
    test_ds  = CSRRowDataset(X, y, idx_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # Train
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, loss_fn, device)
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | train_acc={tr_acc:.4f}")

    # Evaluate
    train_metrics = evaluate(model, train_loader, device)
    test_metrics  = evaluate(model, test_loader, device)

    print("\n== Train ==")
    print(f"Accuracy: {train_metrics.acc:.4f} | F1: {train_metrics.f1:.4f}")
    print(train_metrics.report)
    print("Confusion matrix:\n", train_metrics.cm)

    print("\n== Test ==")
    print(f"Accuracy: {test_metrics.acc:.4f} | F1: {test_metrics.f1:.4f}")
    print(test_metrics.report)
    print("Confusion matrix:\n", test_metrics.cm)

    # Save
    artifacts = repo / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifacts / "nn_tfidf_2_model.pt")
    print("\nSaved: artifacts/nn_tfidf_2_model.pt")


if __name__ == "__main__":
    main()
