#!/usr/bin/env python3
"""
Logistic Regression using sentence-transformer embeddings.

Run:
  python src/logreg_embeddings.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    f1_score
)
import matplotlib.pyplot as plt
import pickle


def main():
    repo = Path(__file__).resolve().parents[1]

    # Load embeddings
    X = np.load(repo / "artifacts" / "embeddings.npy")

    # Load labels
    df = pd.read_csv(repo / "artifacts" / "preprocessed_news.csv")
    y = df["label"].astype(int).values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Logistic Regression
    clf = LogisticRegression(
        max_iter=3000,
        n_jobs=-1,
        solver="lbfgs"
    )
    clf.fit(X_train, y_train)

    # Default threshold metrics
    y_pred = clf.predict(X_test)
    print("\n=== Logistic Regression on Embeddings (threshold = 0.5) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model
    with open(repo / "artifacts" / "logreg_embeddings.pkl", "wb") as f:
        pickle.dump(clf, f)

    # ROC curve
    y_scores = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # Best threshold - Youden’s J
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_threshold = thresholds[best_idx]

    print("\n=== ROC-Optimal Threshold ===")
    print("Best threshold:", best_threshold)
    print(f"TPR={tpr[best_idx]:.4f}  FPR={fpr[best_idx]:.4f}")

    # Predict with ROC threshold
    y_pred_opt = (y_scores >= best_threshold).astype(int)
    print("\n=== Metrics at ROC-Optimal Threshold ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_opt))
    print("F1:", f1_score(y_test, y_pred_opt))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred_opt))

    # Plot ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.scatter(fpr[best_idx], tpr[best_idx], color="red", s=60, label="Optimal threshold")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Logistic Regression (Embeddings)")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

