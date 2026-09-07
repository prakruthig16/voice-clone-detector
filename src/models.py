"""
Classifier definitions.

Primary model is a scikit-learn pipeline: StandardScaler -> classifier.
Two ready choices:
  - "gb"  : GradientBoosting (strong default on tabular acoustic features)
  - "rf"  : RandomForest (fast, robust baseline)
  - "svm" : RBF SVM with probability estimates

An optional PyTorch CNN on log-mel spectrograms lives in cnn.py and is only
imported if torch is installed.
"""
from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

import config


def build_model(kind: str = "gb") -> Pipeline:
    kind = kind.lower()
    if kind == "gb":
        clf = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.9, random_state=config.RANDOM_STATE,
        )
    elif kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=400, max_depth=None, n_jobs=-1,
            class_weight="balanced", random_state=config.RANDOM_STATE,
        )
    elif kind == "svm":
        clf = SVC(
            C=10.0, gamma="scale", kernel="rbf", probability=True,
            class_weight="balanced", random_state=config.RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model kind: {kind!r} (use gb|rf|svm)")

    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
