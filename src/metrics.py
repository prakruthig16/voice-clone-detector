"""Evaluation metrics, including the Equal Error Rate (EER) used in anti-spoofing."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, roc_curve,
)


def equal_error_rate(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """
    EER = the operating point where false-accept rate == false-reject rate.
    Lower is better. Returns (eer, threshold). y_score is P(fake).
    """
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    return eer, float(thr[idx])


def evaluate(y_true, y_pred, y_score) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    eer, eer_thr = equal_error_rate(np.asarray(y_true), np.asarray(y_score))
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_fake": p,
        "recall_fake": r,
        "f1_fake": f1,
        "roc_auc": auc,
        "eer": eer,
        "eer_threshold": eer_thr,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def format_report(m: dict) -> str:
    cm = m["confusion_matrix"]
    return (
        "\n==================== Evaluation ====================\n"
        f"Accuracy        : {m['accuracy']:.4f}\n"
        f"Precision (fake): {m['precision_fake']:.4f}\n"
        f"Recall (fake)   : {m['recall_fake']:.4f}\n"
        f"F1 (fake)       : {m['f1_fake']:.4f}\n"
        f"ROC-AUC         : {m['roc_auc']:.4f}\n"
        f"EER             : {m['eer']:.4f}  (thr={m['eer_threshold']:.3f})\n"
        "Confusion matrix (rows=true, cols=pred):\n"
        f"            pred_real  pred_fake\n"
        f"  real   {cm[0][0]:9d}  {cm[0][1]:9d}\n"
        f"  fake   {cm[1][0]:9d}  {cm[1][1]:9d}\n"
        "====================================================\n"
    )
