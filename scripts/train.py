"""
Train the voice-clone / impersonation detector.

Usage:
    python scripts/train.py                # GradientBoosting, cached features
    python scripts/train.py --model rf     # RandomForest
    python scripts/train.py --no-cache     # force feature re-extraction
    python scripts/train.py --cv 5         # add 5-fold cross-validation

Outputs:
    models/detector.joblib   trained pipeline + metadata
    docs/confusion_matrix.png, docs/roc_curve.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve

# Make the project root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.dataset import build_dataset
from src.models import build_model
from src.metrics import evaluate, format_report


def plot_confusion(cm, out_path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], config.CLASS_NAMES)
    ax.set_yticks([0, 1], config.CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="black", fontsize=14)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_roc(y_true, y_score, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label="detector")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve"); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gb", choices=["gb", "rf", "svm"])
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--cv", type=int, default=0, help="k-fold CV (0=off)")
    args = ap.parse_args()

    X, y, _ = build_dataset(use_cache=not args.no_cache)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_STATE)

    model = build_model(args.model)

    if args.cv and args.cv > 1:
        scores = cross_val_score(model, X, y, cv=args.cv, scoring="f1")
        print(f"{args.cv}-fold CV F1: {scores.mean():.4f} +/- {scores.std():.4f}")

    print(f"Training {args.model} on {X_tr.shape[0]} clips "
          f"({X_tr.shape[1]} features)...")
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_score = model.predict_proba(X_te)[:, config.LABEL_FAKE]
    m = evaluate(y_te, y_pred, y_score)
    print(format_report(m))

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "model_kind": args.model,
                 "metrics": m, "eer_threshold": m["eer_threshold"]},
                config.MODEL_PATH)
    print(f"Saved model -> {config.MODEL_PATH}")

    plot_confusion(m["confusion_matrix"], config.DOCS_DIR / "confusion_matrix.png")
    plot_roc(y_te, y_score, config.DOCS_DIR / "roc_curve.png")
    print(f"Saved plots -> {config.DOCS_DIR}")


if __name__ == "__main__":
    main()
