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
    auc,
    recall_score
)
import matplotlib.pyplot as plt
import pickle


def main():
    repo = Path(__file__).resolve().parents[1]

    # ============================
    # LOAD FEATURES & LABELS
    # ============================
    X_path = repo / "artifacts" / "X_tfidf.npz"
    X = sparse.load_npz(X_path)

    df = pd.read_csv(repo / "artifacts" / "preprocessed_news.csv")
    y = df["label"].astype(int).values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ============================
    # TRAIN MODEL
    # ============================
    clf = LogisticRegression(
        max_iter=3000,
        n_jobs=-1,
        solver="lbfgs"
    )
    clf.fit(X_train, y_train)

    # ============================
    # DEFAULT THRESHOLD (0.5)
    # ============================
    y_pred_default = clf.predict(X_test)

    print("\n=== Logistic Regression on TF-IDF (Default Threshold = 0.5) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_default))
    print("F1 Score:", f1_score(y_test, y_pred_default))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_default))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_default))

    # Save model
    model_path = repo / "artifacts" / "logreg_tfidf.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nSaved model to: {model_path}")

    # ============================
    # ROC COMPUTATION
    # ============================
    y_scores = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # ============================
    # ROC-OPTIMAL THRESHOLD (YOUDEN'S J)
    # ============================
    youden_J = tpr - fpr
    best_idx = np.argmax(youden_J)
    roc_best_threshold = thresholds[best_idx]

    print("\n=== ROC-Optimal Threshold (Youden's J) ===")
    print(f"Best threshold from ROC: {roc_best_threshold:.4f}")
    print(f"TPR at this threshold: {tpr[best_idx]:.4f}")
    print(f"FPR at this threshold: {fpr[best_idx]:.4f}")

    # ============================
    # LOOP OVER THRESHOLDS FOR SENSITIVITY & SPECIFICITY
    # ============================
    sensitivities = []
    specificities = []

    for t in thresholds:
        y_pred_thr = (y_scores >= t).astype(int)

        TP = np.sum((y_test == 0) & (y_pred_thr == 0))  # fake detected
        FN = np.sum((y_test == 0) & (y_pred_thr == 1))  # fake missed
        TN = np.sum((y_test == 1) & (y_pred_thr == 1))  # real kept
        FP = np.sum((y_test == 1) & (y_pred_thr == 0))  # real misclassified as fake

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  # TPR
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0  # TNR

        sensitivities.append(sensitivity)
        specificities.append(specificity)

    # ============================
    # THRESHOLD BASED ON PRIORITY (FAKE NEWS RECALL)
    # ============================
    best_idx_sens = np.argmax(sensitivities)
    best_threshold_sensitivity = thresholds[best_idx_sens]

    print("\n=== Sensitivity-Maximizing Threshold (Minimizing Type II Errors) ===")
    print(f"Threshold maximizing sensitivity (fake news recall): {best_threshold_sensitivity:.4f}")
    print(f"Sensitivity at this threshold: {sensitivities[best_idx_sens]:.4f}")

    # Predict using sensitivity-optimal threshold
    y_pred_sensitivity = (y_scores >= best_threshold_sensitivity).astype(int)

    print("\n=== Performance Using Sensitivity-Optimized Threshold ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_sensitivity))
    print("F1 Score:", f1_score(y_test, y_pred_sensitivity))
    print("Recall (Fake News):", sensitivities[best_idx_sens])
    print("\nClassification Report:\n", classification_report(y_test, y_pred_sensitivity))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_sensitivity))

    # ============================
    # PLOTS: SENSITIVITY & SPECIFICITY
    # ============================

    # Sensitivity vs Threshold
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, sensitivities, label="Sensitivity (TPR)", color="blue")
    plt.xlabel("Threshold")
    plt.ylabel("Sensitivity")
    plt.title("Sensitivity vs Threshold")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Specificity vs Threshold
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, specificities, label="Specificity (TNR)", color="green")
    plt.xlabel("Threshold")
    plt.ylabel("Specificity")
    plt.title("Specificity vs Threshold")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Combined Sensitivity + Specificity
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, sensitivities, label="Sensitivity (TPR)", color="blue")
    plt.plot(thresholds, specificities, label="Specificity (TNR)", color="green")
    plt.axvline(best_threshold_sensitivity, color="red", linestyle="--",
                label=f"Chosen Threshold = {best_threshold_sensitivity:.4f}")
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Sensitivity and Specificity vs Threshold")
    plt.grid(True)
    plt.legend()
    plt.show()

    # ============================
    # ROC CURVE (WITH THRESHOLD POINT)
    # ============================
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"LogReg (AUC = {roc_auc:.4f})")
    plt.scatter(fpr[best_idx], tpr[best_idx], color="red", s=60,
                label="ROC-Optimal Threshold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Logistic Regression (TF-IDF)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

