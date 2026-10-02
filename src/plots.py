"""Visualization helpers. All figures are saved to the artifacts directory."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend, safe for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from . import config
from .backtest import BacktestResult
from .features import FEATURE_COLUMNS

sns.set_theme(style="whitegrid")


def plot_equity_curves(bt: BacktestResult, model_name: str, ticker: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bt.equity_strategy.index, bt.equity_strategy.values,
            label=f"ML strategy ({model_name})", linewidth=2)
    ax.plot(bt.equity_hold.index, bt.equity_hold.values,
            label="Buy & hold", linewidth=2, linestyle="--")
    ax.set_title(f"Cumulative returns over test window -- {ticker}")
    ax.set_ylabel("Growth of 1 unit")
    ax.set_xlabel("Date")
    ax.legend()
    fig.tight_layout()
    path = config.ARTIFACT_DIR / "equity_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_confusion(y_true, y_pred, model_name: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Down", "Up"], cmap="Blues", ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Confusion matrix -- {model_name}")
    fig.tight_layout()
    path = config.ARTIFACT_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_feature_importance(estimator, model_name: str):
    """Feature importance for tree models (skips models without it)."""
    clf = estimator.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return None
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLUMNS)
    importances = importances.sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot.barh(ax=ax, color="steelblue")
    ax.set_title(f"Feature importance -- {model_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    path = config.ARTIFACT_DIR / "feature_importance.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_model_comparison(results: dict, baseline: float):
    names = list(results.keys())
    cv = [results[n].cv_accuracy for n in names]
    test = [results[n].test_accuracy for n in names]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, cv, width, label="CV accuracy")
    ax.bar(x + width / 2, test, width, label="Test accuracy")
    ax.axhline(baseline, color="red", linestyle="--",
               label=f"Majority baseline ({baseline:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.4, max(max(cv), max(test), baseline) + 0.05)
    ax.set_title("Model comparison")
    ax.legend()
    fig.tight_layout()
    path = config.ARTIFACT_DIR / "model_comparison.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
