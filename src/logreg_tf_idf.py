#!/usr/bin/env python3
"""
Logistic Regression classifier using precomputed TF-IDF features.

Run:
  python src/logreg_tfidf.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import pickle


def main():
    repo = Path(__file__).resolve().parents[1]

    # Load TF-IDF matrix
    X_path = repo / "artifacts" / "X_tfidf.npz"
    X = sparse.load_npz(X_path)

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

    # Predict
    y_pred = clf.predict(X_test)

    # Metrics
    print("\n=== Logistic Regression on TF-IDF ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model
    model_path = repo / "artifacts" / "logreg_tfidf.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nSaved model to: {model_path}")

    # ============================
    # ROC CURVE (NOW IN CORRECT PLACE)
    # ============================
    y_scores = clf.predict_proba(X_test)[:, 1]  # probability of class 1

    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"LogReg (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guessing")

    plt.title("ROC Curve – Logistic Regression (TF-IDF)", fontsize=14)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
