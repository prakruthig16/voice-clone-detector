
"""
Improved training pipeline for AI Voice Clone Detector.

Usage:
    python scripts/train.py
    python scripts/train.py --model gb
    python scripts/train.py --model all
    python scripts/train.py --no-cache
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
)

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.dataset import build_dataset
from src.models import build_model
from src.metrics import evaluate, format_report


# --------------------------------------------------
# PLOT 1: CONFUSION MATRIX
# --------------------------------------------------

def plot_confusion(cm, output):

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1], config.CLASS_NAMES)
    ax.set_yticks([0, 1], config.CLASS_NAMES)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i][j]),
                ha="center",
                va="center",
                fontsize=14
            )

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


# --------------------------------------------------
# PLOT 2: ROC CURVE
# --------------------------------------------------

def plot_roc(y_true, y_score, output):

    fpr, tpr, _ = roc_curve(y_true, y_score)

    score = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        fpr, tpr,
        label=f"ROC-AUC = {score:.4f}"
    )

    ax.plot([0, 1], [0, 1], "--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")

    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


# --------------------------------------------------
# PLOT 3: PRECISION-RECALL CURVE
# --------------------------------------------------

def plot_precision_recall(y_true, y_score, output):

    precision, recall, _ = precision_recall_curve(
        y_true, y_score
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(recall, precision)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")

    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


# --------------------------------------------------
# PLOT 4: LEARNING CURVE
# --------------------------------------------------

def plot_learning_curve(model, X, y, cv, output):

    sizes, train_scores, validation_scores = learning_curve(
        model,
        X,
        y,
        cv=cv,
        scoring="f1",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    validation_mean = validation_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        sizes,
        train_mean,
        marker="o",
        label="Training F1"
    )

    ax.plot(
        sizes,
        validation_mean,
        marker="o",
        label="Validation F1"
    )

    ax.set_xlabel("Training samples")
    ax.set_ylabel("F1 Score")
    ax.set_title("Model Learning Curve")

    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="all",
        choices=["gb", "rf", "svm", "all"]
    )

    parser.add_argument(
        "--no-cache",
        action="store_true"
    )

    parser.add_argument(
        "--cv",
        type=int,
        default=5
    )

    args = parser.parse_args()

    # Load dataset

    X, y, paths = build_dataset(
        use_cache=not args.no_cache
    )

    counts = np.bincount(y, minlength=2)

    if np.min(counts) < 2:
        raise ValueError(
            "At least two samples per class are required."
        )

    print("\nDataset information")
    print("-------------------")
    print("Total samples:", len(y))
    print("REAL samples:", counts[0])
    print("FAKE samples:", counts[1])
    print("Features:", X.shape[1])

    # Separate final test set

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE
    )

    # Select suitable cross-validation folds

    folds = min(args.cv, int(np.min(np.bincount(y_train))))

    if folds < 2:
        raise ValueError("Insufficient samples for cross-validation.")

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=config.RANDOM_STATE
    )

    model_names = (
        ["gb", "rf", "svm"]
        if args.model == "all"
        else [args.model]
    )

    results = {}

    best_name = None
    best_score = -1
    best_model = None

    print("\nModel comparison")
    print("----------------")

    for name in model_names:

        print(f"\nEvaluating {name.upper()}...")

        model = build_model(name)

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1",
            n_jobs=-1
        )

        mean_score = float(scores.mean())
        std_score = float(scores.std())

        results[name] = {
            "cv_f1_mean": mean_score,
            "cv_f1_std": std_score
        }

        print(
            f"CV F1: {mean_score:.4f} "
            f"+/- {std_score:.4f}"
        )

        if mean_score > best_score:

            best_score = mean_score
            best_name = name
            best_model = model

    # Fit selected model

    print(f"\nBest model: {best_name.upper()}")

    best_model.fit(X_train, y_train)

    # Final test evaluation

    y_pred = best_model.predict(X_test)

    y_score = best_model.predict_proba(X_test)[
        :, config.LABEL_FAKE
    ]

    metrics = evaluate(
        y_test,
        y_pred,
        y_score
    )

    print(format_report(metrics))

    # Save plots

    config.DOCS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plot_confusion(
        metrics["confusion_matrix"],
        config.DOCS_DIR / "confusion_matrix.png"
    )

    plot_roc(
        y_test,
        y_score,
        config.DOCS_DIR / "roc_curve.png"
    )

    plot_precision_recall(
        y_test,
        y_score,
        config.DOCS_DIR / "precision_recall.png"
    )

    plot_learning_curve(
        build_model(best_name),
        X_train,
        y_train,
        cv,
        config.DOCS_DIR / "learning_curve.png"
    )

    # Save model

    config.MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "pipeline": best_model,
            "model_kind": best_name,
            "metrics": metrics,
            "eer_threshold": metrics["eer_threshold"],
            "cv_results": results,
        },
        config.MODEL_PATH
    )

    # Save evaluation report

    report = {
        "best_model": best_name,
        "test_metrics": metrics,
        "model_comparison": results,
        "test_samples": len(y_test),
        "feature_count": X.shape[1]
    }

    with open(
        config.DOCS_DIR / "training_report.json",
        "w"
    ) as file:

        json.dump(report, file, indent=4)

    print("\nTraining completed successfully!")

    print("Best model:", best_name)
    print("Saved model:", config.MODEL_PATH)
    print("Saved graphs:", config.DOCS_DIR)


if __name__ == "__main__":
    main()
