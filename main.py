"""End-to-end pipeline: data -> features -> models -> backtest -> report.

Run:
    python -m main                      # default ticker (Reliance)
    python -m main --ticker ^NSEI       # NIFTY 50 index
    python -m main --refresh            # ignore cache, re-download
"""
from __future__ import annotations

import argparse

import joblib

from src import config
from src.backtest import run_backtest
from src.data import load_prices
from src.features import build_features
from src.models import baseline_accuracy, train_all, train_test_split_time
from src.plots import (
    plot_confusion,
    plot_equity_curves,
    plot_feature_importance,
    plot_model_comparison,
)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Algorithmic Trading Signal Predictor")
    parser.add_argument("--ticker", default=config.TICKER,
                        help="Ticker (e.g. RELIANCE.NS, ^NSEI)")
    parser.add_argument("--refresh", action="store_true", help="Force re-download")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  ALGORITHMIC TRADING SIGNAL PREDICTOR  --  {args.ticker}")
    print("=" * 70)

    # 1. Data ---------------------------------------------------------------
    prices = load_prices(ticker=args.ticker, force_refresh=args.refresh)
    print(f"\n[1] Data       : {len(prices)} rows "
          f"({prices.index[0].date()} -> {prices.index[-1].date()})")

    # 2. Features -----------------------------------------------------------
    feats = build_features(prices)
    up_rate = feats["y"].mean()
    print(f"[2] Features   : {len(feats)} usable rows, "
          f"{len(feats.columns) - 1} engineered cols")
    print(f"    Up-day rate (full sample) : {up_rate:.3f}")

    # 3. Models -------------------------------------------------------------
    baseline = baseline_accuracy(feats)
    print(f"\n[3] Training 3 classifiers (GridSearchCV + TimeSeriesSplit)...")
    results = train_all(feats)
    print(f"    Majority-class baseline (test window) : {baseline:.3f}\n")
    print(f"    {'Model':<22}{'CV acc':>9}{'Test acc':>10}  Best params")
    print("    " + "-" * 62)
    for name, r in results.items():
        print(f"    {name:<22}{r.cv_accuracy:>9.3f}{r.test_accuracy:>10.3f}  "
              f"{r.best_params}")

    # Select the best model by CROSS-VALIDATION accuracy, never by test
    # accuracy -- choosing on the test set would leak it into model selection.
    # The test window is then reported as an untouched out-of-sample estimate.
    best_name = max(results, key=lambda n: results[n].cv_accuracy)
    best = results[best_name]
    print(f"\n    >> Best model (by CV): {best_name} "
          f"(CV {best.cv_accuracy:.3f} | test {best.test_accuracy:.3f} "
          f"| baseline {baseline:.3f})")
    print("\n    Classification report (test window):")
    print("    " + best.report.replace("\n", "\n    "))

    # 4. Backtest -----------------------------------------------------------
    _, X_test, _, y_test = train_test_split_time(feats)
    test_frame = feats.loc[X_test.index]
    bt = run_backtest(test_frame, best.predictions)

    print(f"[4] Backtest over {len(test_frame)} trading days "
          f"(transaction cost {config.TRANSACTION_COST * 100:.1f}%/trade):")
    print(f"    {'':<26}{'Strategy':>12}{'Buy & hold':>14}")
    print("    " + "-" * 52)
    print(f"    {'Total return':<26}{_fmt_pct(bt.total_return_strategy):>12}"
          f"{_fmt_pct(bt.total_return_hold):>14}")
    print(f"    {'Sharpe ratio (annual)':<26}{bt.sharpe_strategy:>12.2f}"
          f"{bt.sharpe_hold:>14.2f}")
    print(f"    {'Max drawdown':<26}{_fmt_pct(bt.max_drawdown_strategy):>12}"
          f"{_fmt_pct(bt.max_drawdown_hold):>14}")
    print(f"    {'Number of trades':<26}{bt.n_trades:>12}")

    # 5. Artifacts ----------------------------------------------------------
    model_path = config.ARTIFACT_DIR / "best_model.joblib"
    joblib.dump({"model": best.estimator, "features": list(X_test.columns),
                 "ticker": args.ticker, "name": best_name}, model_path)

    p1 = plot_model_comparison(results, baseline)
    p2 = plot_confusion(y_test, best.predictions, best_name)
    p3 = plot_equity_curves(bt, best_name, args.ticker)
    p4 = plot_feature_importance(best.estimator, best_name)

    print(f"\n[5] Saved artifacts:")
    print(f"    model  -> {model_path}")
    for p in [p1, p2, p3, p4]:
        if p:
            print(f"    figure -> {p}")
    print("\nDone.")


if __name__ == "__main__":
    main()
