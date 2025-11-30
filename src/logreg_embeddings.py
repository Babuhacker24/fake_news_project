#!/usr/bin/env python3
"""
Logistic Regression classifier using LLM embeddings (Sentence-Transformers).

Outputs:
  - Default-threshold metrics (0.5)
  - ROC curve + AUC
  - Youden's J optimal threshold
  - Sensitivity/specificity curves
  - Threshold-optimized metrics
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
    f1_score,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import pickle


def main():
    repo = Path(__file__).resolve().parents[1]

    # ============================================================
    # LOAD EMBEDDINGS & LABELS
    # ============================================================
    X = np.load(repo / "artifacts" / "embeddings_minilm.npy")
    y = np.load(repo / "artifacts" / "embeddings_minilm_labels.npy")
    y = 1 - y

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ============================================================
    # TRAIN LOGISTIC REGRESSION
    # ============================================================
    clf = LogisticRegression(
        max_iter=3000,
        n_jobs=-1,
        solver="lbfgs"
    )
    clf.fit(X_train, y_train)
  
    # ============================================================
    # TRAINING PERFORMANCE
    # ============================================================
    y_train_pred = clf.predict(X_train)
    
    print("\n=== Logistic Regression on Embeddings (TRAINING SET) ===")
    print("Accuracy: {:.4f}".format(accuracy_score(y_train, y_train_pred)))
    print("F1 Score: {:.4f}".format(f1_score(y_train, y_train_pred)))
    print("\nClassification Report:\n", classification_report(y_train, y_train_pred, digits=4))
    print("\nConfusion Matrix:\n", confusion_matrix(y_train, y_train_pred))


    # ============================================================
    # DEFAULT THRESHOLD METRICS (0.5)
    # ============================================================
    y_pred_default = clf.predict(X_test)

    print("\n=== Logistic Regression on Embeddings (Threshold = 0.5) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_default))
    print("F1 Score:", f1_score(y_test, y_pred_default))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_default,digits=4))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_default))

    # Save model
    model_path = repo / "artifacts" / "logreg_embeddings.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nSaved model to: {model_path}")

    # ============================================================
    # ROC CURVE
    # ============================================================
    y_scores = clf.predict_proba(X_test)[:, 1]   # probability of class 1 (real)
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # ============================================================
    # YOUDEN'S J STATISTIC (BEST ROC THRESHOLD)
    # ============================================================
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_threshold_roc = thresholds[best_idx]

    print("\n=== ROC-Optimal Threshold (Youden's J) ===")
    print(f"Best threshold: {best_threshold_roc:.4f}")
    print(f"TPR = {tpr[best_idx]:.4f}")
    print(f"FPR = {fpr[best_idx]:.4f}")

    # Predictions at ROC-optimal threshold
    y_pred_roc = (y_scores >= best_threshold_roc).astype(int)
    print("\n=== Metrics at ROC-Optimal Threshold ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_roc))
    print("F1 Score:", f1_score(y_test, y_pred_roc))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_roc))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_roc,digits=4))

    # ============================================================
    # MANUAL THRESHOLD TESTING (TRIAL & ERROR)
    # ============================================================
    
    # >>> Change this value to test any threshold you want <<<
    manual_threshold = 0.4550
    
    print(f"\n=== Manual Threshold Test at {manual_threshold:.4f} ===")
    
    # Apply threshold manually
    y_pred_manual = (y_scores >= manual_threshold).astype(int)
    
    # Compute metrics
    acc_manual = accuracy_score(y_test, y_pred_manual)
    f1_manual = f1_score(y_test, y_pred_manual)
    cm_manual = confusion_matrix(y_test, y_pred_manual)
    
    print(f"Accuracy: {acc_manual:.4f}")
    print(f"F1 Score: {f1_manual:.4f}")
    
    print("\nClassification Report:\n",
          classification_report(y_test, y_pred_manual, digits=4))
    
    print("\nConfusion Matrix:\n", cm_manual)

    # ============================================================
    # PLOTS
    # ============================================================

    # Sensitivity vs Threshold
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, sensitivities, label="Sensitivity (TPR)", color="blue")
    plt.xlabel("Threshold")
    plt.ylabel("Sensitivity")
    plt.title("Sensitivity vs Threshold (Embeddings)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Specificity vs Threshold
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, specificities, label="Specificity (TNR)", color="green")
    plt.xlabel("Threshold")
    plt.ylabel("Specificity")
    plt.title("Specificity vs Threshold (Embeddings)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Combined sensitivity & specificity
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, sensitivities, label="Sensitivity (TPR)", color="blue")
    plt.plot(thresholds, specificities, label="Specificity (TNR)", color="green")
    plt.axvline(best_threshold_sensitivity, color="red", linestyle="--",
                label=f"Chosen Threshold = {best_threshold_sensitivity:.4f}")
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Sensitivity & Specificity vs Threshold (Embeddings)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.scatter(fpr[best_idx], tpr[best_idx], color="red", s=60,
                label="ROC-Optimal Threshold")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Logistic Regression (Embeddings)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
