#!/usr/bin/env python3
"""
Logistic Regression classifier using precomputed TF-IDF features.

Run:a
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
    y = 1 - y


    # ============================================================
    # TRAIN / TEST SPLIT
    # ============================================================
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
    
    print("\n=== Logistic Regression on TF-IDF (TRAINING SET) ===")
    print("Accuracy:", accuracy_score(y_train, y_train_pred))
    print("F1 Score:", f1_score(y_train, y_train_pred))
    print("\nClassification Report:\n", classification_report(y_train, y_train_pred, digits=4))
    print("\nConfusion Matrix:\n", confusion_matrix(y_train, y_train_pred))


    # ============================================================
    # TEST PERFORMANCE (DEFAULT THRESHOLD = 0.5)
    # ============================================================
    y_pred_default = clf.predict(X_test)

    print("\n=== Logistic Regression on TF-IDF (TEST SET, Threshold = 0.5) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_default))
    print("F1 Score:", f1_score(y_test, y_pred_default))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_default,digits=4))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_default))

    # Save model
    model_path = repo / "artifacts" / "logreg_tfidf.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nSaved model to: {model_path}")

    # ============================================================
    # ROC CURVE (TEST SET)
    # ============================================================
    y_scores = clf.predict_proba(X_test)[:, 1]  # probability of class 1

    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # --- Youden's J Threshold ---
    youden_J = tpr - fpr
    best_idx = np.argmax(youden_J)
    best_threshold = thresholds[best_idx]

    print(f"\nOptimal threshold (Youden's J): {best_threshold:.4f}")
    print(f"TPR at optimal threshold: {tpr[best_idx]:.4f}")
    print(f"FPR at optimal threshold: {fpr[best_idx]:.4f}")

    # Predict using Youden threshold
    y_pred_optimal = (y_scores >= best_threshold).astype(int)

    print("\n=== Performance Using Optimal Threshold (TEST SET) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_optimal))
    print("F1 Score:", f1_score(y_test, y_pred_optimal))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_optimal,digits=4))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_optimal))

    # ============================================================
    # MANUAL THRESHOLD TESTING (TRIAL & ERROR)
    # ============================================================
    
    # >>> Change this value to test any threshold you want <<<
    manual_threshold = 0.5550
    
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
    # PLOT ROC CURVE
    # ============================================================
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"LogReg (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guessing")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Logistic Regression (TF-IDF)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()


