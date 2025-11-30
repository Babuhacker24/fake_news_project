#!/usr/bin/env python3
"""
Logistic Regression classifier using LLM embeddings (Sentence-Transformers).

Outputs:
  - Train & test metrics at default threshold (0.5)
  - ROC curve + AUC
  - Youden's J optimal threshold
  - Manual threshold testing
  - Threshold curves (TPR, FPR, specificity)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
    f1_score
)
import pickle


# ============================================================
# Utility Functions
# ============================================================
def evaluate_threshold(y_true, y_scores, threshold):
    """Return predictions + metrics at a given threshold."""
    y_pred = (y_scores >= threshold).astype(int)

    return {
        "threshold": threshold,
        "pred": y_pred,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "cm": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred, digits=4)
    }


def print_metrics(title, metrics):
    """Beautiful, consistent formatted printing."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"Threshold: {metrics['threshold']:.4f}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print("\nClassification Report:\n", metrics["report"])
    print("Confusion Matrix:\n", metrics["cm"], "\n")


# ============================================================
# Main Script
# ============================================================
def main():
    repo = Path(__file__).resolve().parents[1]

    # ============================================================
    # LOAD EMBEDDINGS & LABELS
    # ============================================================
    X = np.load(repo / "artifacts" / "embeddings_minilm.npy")
    y = np.load(repo / "artifacts" / "embeddings_minilm_labels.npy")

    # IMPORTANT: Flip labels if they were inverted during labeling
    y = 1 - y  # ensures: 1 = fake, 0 = real (or whichever convention you use)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ============================================================
    # TRAIN LOGISTIC REGRESSION
    # ============================================================
    clf = LogisticRegression(max_iter=3000, n_jobs=-1, solver="lbfgs")
    clf.fit(X_train, y_train)

    # ============================================================
    # TRAINING PERFORMANCE
    # ============================================================
    y_train_pred = clf.predict(X_train)

    print("\n=== Logistic Regression on Embeddings (TRAINING SET) ===")
    print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"F1 Score: {f1_score(y_train, y_train_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_train, y_train_pred, digits=4))
    print("\nConfusion Matrix:\n", confusion_matrix(y_train, y_train_pred))

    # ============================================================
    # TEST PERFORMANCE — DEFAULT THRESHOLD 0.5
    # ============================================================
    print("\n=== Logistic Regression on Embeddings (TEST SET — Threshold = 0.5) ===")
    y_pred_default = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_default):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred_default):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred_default, digits=4))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_default))

    # Save model
    model_path = repo / "artifacts" / "logreg_embeddings.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    print(f"\nSaved model to: {model_path}")

    # ============================================================
    # ROC CURVE + THRESHOLDS
    # ============================================================
    y_scores = clf.predict_proba(X_test)[:, 1]  # probability of class "1"

    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # Youden's J index
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_thr_youden = thresholds[best_idx]

    # Evaluate at Youden threshold
    youden_metrics = evaluate_threshold(y_test, y_scores, best_thr_youden)
    print_metrics("ROC-Optimal Threshold (Youden’s J)", youden_metrics)

    # ============================================================
    # MANUAL TRIAL-AND-ERROR THRESHOLD
    # ============================================================
    manual_threshold = 0.4550
    manual_metrics = evaluate_threshold(y_test, y_scores, manual_threshold)
    print_metrics("Manual Threshold Test", manual_metrics)

    # ============================================================
    # SENSITIVITY & SPECIFICITY CURVES
    # ============================================================
    sensitivities = tpr
    specificities = 1 - fpr  # TNR

    # Sensitivity vs Threshold
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, sensitivities, label="TPR (Sensitivity)")
    plt.plot(thresholds, specificities, label="TNR (Specificity)")
    plt.axvline(best_thr_youden, color="red", ls="--", label="Youden Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Sensitivity & Specificity vs Threshold")
    plt.grid(True)
    plt.legend()
    plt.show()

    # ROC Curve
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.scatter(fpr[best_idx], tpr[best_idx], color="red", s=60, label="Youden Threshold")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Logistic Regression (Embeddings)")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

