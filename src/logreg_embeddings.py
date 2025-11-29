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
    # LOOP OVER THRESHOLDS → SENSITIVITY & SPECIFICITY
    # ============================================================
    sensitivities = []
    specificities = []

    for t in thresholds:
        y_pred_thr = (y_scores >= t).astype(int)

        TP = np.sum((y_test == 0) & (y_pred_thr == 0))
        FN = np.sum((y_test == 0) & (y_pred_thr == 1))
        TN = np.sum((y_test == 1) & (y_pred_thr == 1))
        FP = np.sum((y_test == 1) & (y_pred_thr == 0))

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        sensitivities.append(sensitivity)
        specificities.append(specificity)

    # ============================================================
    # THRESHOLD THAT MAXIMIZES SENSITIVITY (FAKE NEWS PRIORITY)
    # ============================================================
    best_idx_sens = np.argmax(sensitivities)
    best_threshold_sensitivity = thresholds[best_idx_sens]

    print("\n=== Sensitivity-Maximizing Threshold (Fake News Priority) ===")
    print(f"Best threshold for sensitivity: {best_threshold_sensitivity:.4f}")
    print(f"Sensitivity at this threshold: {sensitivities[best_idx_sens]:.4f}")

    y_pred_sens = (y_scores >= best_threshold_sensitivity).astype(int)

    print("\n=== Metrics at Sensitivity-Optimized Threshold ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_sens))
    print("F1 Score:", f1_score(y_test, y_pred_sens))
    print("Recall (Fake News):", sensitivities[best_idx_sens])
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_sens))

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

    # ============================================================
    # THRESHOLD THAT MINIMIZES WEIGHTED ERROR COST
    # ============================================================
    
    C_FN = 5   # cost of false negative (missing fake news)
    C_FP = 1   # cost of false positive (flagging true news)
    
    costs = []
    FN_list = []
    FP_list = []
    
    for t in thresholds:
        y_pred_thr = (y_scores >= t).astype(int)
    
        # Confusion matrix components
        TP = np.sum((y_test == 1) & (y_pred_thr == 1))
        FN = np.sum((y_test == 1) & (y_pred_thr == 0))
        FP = np.sum((y_test == 0) & (y_pred_thr == 1))
        TN = np.sum((y_test == 0) & (y_pred_thr == 0))
    
        total_cost = C_FN * FN + C_FP * FP
    
        costs.append(total_cost)
        FN_list.append(FN)
        FP_list.append(FP)
    
    best_idx_cost = np.argmin(costs)
    best_threshold_cost = thresholds[best_idx_cost]
    
    print("\n=== Threshold that Minimizes Weighted Error Cost ===")
    print(f"Cost_FN = {C_FN}, Cost_FP = {C_FP}")
    print(f"Best threshold: {best_threshold_cost:.4f}")
    print(f"FN at this threshold: {FN_list[best_idx_cost]}")
    print(f"FP at this threshold: {FP_list[best_idx_cost]}")
    print(f"Total cost: {costs[best_idx_cost]}")
    
    # Predict using weighted-cost optimal threshold
    y_pred_cost = (y_scores >= best_threshold_cost).astype(int)
    
    print("\n=== Performance with Weighted-Cost Threshold ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_cost))
    print("F1 Score:", f1_score(y_test, y_pred_cost))
    print("\nClassification Report:\n",
          classification_report(y_test, y_pred_cost, digits=4))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_cost))


if __name__ == "__main__":
    main()
