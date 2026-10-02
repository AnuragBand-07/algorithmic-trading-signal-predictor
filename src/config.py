"""Central configuration for the Algorithmic Trading Signal Predictor.

Keeping all knobs in one place makes the pipeline reproducible and easy to
audit -- important for a project whose whole point is *honest* evaluation.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_cache"
ARTIFACT_DIR = ROOT / "artifacts"
DATA_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# Default ticker. NIFTY 50 index = "^NSEI", Reliance = "RELIANCE.NS".
TICKER = "RELIANCE.NS"
# How much history to pull. More history => more training data.
PERIOD = "10y"
INTERVAL = "1d"

# ---------------------------------------------------------------------------
# Backtest / evaluation
# ---------------------------------------------------------------------------
# Size of the chronological, out-of-sample test window (one trading year).
TEST_WINDOW = 252
# Trading days per year, used to annualise the Sharpe ratio.
TRADING_DAYS = 252
# Per-trade transaction cost (fraction). 0.001 = 0.1% (a realistic retail cost).
# Applied whenever the position changes, so a leakage-free strategy is not
# flattered by ignoring frictions.
TRANSACTION_COST = 0.001
# Annual risk-free rate used in the Sharpe ratio (India ~ 6-7%).
RISK_FREE_RATE = 0.06

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
# Number of expanding-window splits for TimeSeriesSplit during GridSearchCV.
N_CV_SPLITS = 5

RANDOM_STATE = 42
