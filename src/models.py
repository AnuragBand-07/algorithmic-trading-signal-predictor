"""Modeling layer.

Trains and tunes three classifiers -- Logistic Regression (baseline), Random
Forest and SVM -- and compares them honestly:

* Chronological train/test split (NEVER shuffled) so the model never trains on
  the future to predict the past.
* Every model lives inside a ``Pipeline`` with its scaler, so the scaler is fit
  only on training folds -> no data leakage into the test set.
* Hyper-parameters are tuned with ``GridSearchCV`` using ``TimeSeriesSplit``
  (expanding window) rather than random k-fold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config
from .features import FEATURE_COLUMNS


@dataclass
class ModelResult:
    name: str
    estimator: Pipeline
    best_params: dict
    cv_accuracy: float          # best mean CV score during tuning
    test_accuracy: float        # accuracy on the held-out test window
    predictions: np.ndarray     # predicted labels on the test window
    report: str                 # sklearn classification report (test)


def _model_grids(random_state: int = config.RANDOM_STATE):
    """Return {name: (pipeline, param_grid)} for the three classifiers."""
    return {
        "Logistic Regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "Random Forest": (
            Pipeline([
                ("scaler", StandardScaler()),  # harmless for trees, keeps API uniform
                ("clf", RandomForestClassifier(random_state=random_state, n_jobs=-1)),
            ]),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [3, 5, 7],
                "clf__min_samples_leaf": [5, 20, 50],
            },
        ),
        "SVM": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", random_state=random_state)),
            ]),
            {
                "clf__C": [0.1, 1.0, 10.0],
                "clf__gamma": ["scale", 0.01, 0.1],
            },
        ),
    }


def train_test_split_time(features: pd.DataFrame, test_window: int = config.TEST_WINDOW):
    """Chronological split: last ``test_window`` rows are the out-of-sample test."""
    X = features[FEATURE_COLUMNS]
    y = features["y"]
    split = len(features) - test_window
    if split <= 0:
        raise ValueError("Not enough data for the requested test window.")
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    return X_train, X_test, y_train, y_test


def train_all(features: pd.DataFrame) -> dict[str, ModelResult]:
    """Tune and evaluate every classifier; return results keyed by name."""
    X_train, X_test, y_train, y_test = train_test_split_time(features)
    tscv = TimeSeriesSplit(n_splits=config.N_CV_SPLITS)

    results: dict[str, ModelResult] = {}
    for name, (pipe, grid) in _model_grids().items():
        search = GridSearchCV(
            pipe, grid, cv=tscv, scoring="accuracy", n_jobs=-1, refit=True,
        )
        search.fit(X_train, y_train)

        preds = search.predict(X_test)
        results[name] = ModelResult(
            name=name,
            estimator=search.best_estimator_,
            best_params=search.best_params_,
            cv_accuracy=float(search.best_score_),
            test_accuracy=float(accuracy_score(y_test, preds)),
            predictions=preds,
            report=classification_report(y_test, preds, digits=3, zero_division=0),
        )
    return results


def baseline_accuracy(features: pd.DataFrame, test_window: int = config.TEST_WINDOW) -> float:
    """Naive 'always predict the majority class' accuracy on the test window.

    A model is only meaningful if it beats this base rate, not just 50%.
    """
    _, _, y_train, y_test = train_test_split_time(features, test_window)
    majority = int(y_train.mode().iloc[0])
    return float((y_test == majority).mean())
