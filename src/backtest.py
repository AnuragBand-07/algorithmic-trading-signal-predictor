"""Vectorized backtesting engine.

Turns a model's directional predictions into a simulated long/flat strategy and
compares it against buy-and-hold on a *risk-adjusted* basis -- because accuracy
alone is misleading in finance.

Signal alignment (the #1 source of fake profits):
    ``future_return`` at row ``t`` is the realised return from close[t] ->
    close[t+1]. The model's ``signal`` at row ``t`` is decided using ONLY
    information available at close[t]. Multiplying them (same row) therefore
    trades strictly on past information -- no look-ahead. This is exactly
    equivalent to ``signal.shift(1) * next_day_return`` but avoids an
    error-prone extra shift.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class BacktestResult:
    equity_strategy: pd.Series      # cumulative growth of 1 unit (strategy)
    equity_hold: pd.Series          # cumulative growth of 1 unit (buy & hold)
    total_return_strategy: float
    total_return_hold: float
    sharpe_strategy: float
    sharpe_hold: float
    max_drawdown_strategy: float
    max_drawdown_hold: float
    n_trades: int


def _sharpe(daily_returns: pd.Series) -> float:
    """Annualised Sharpe ratio of a daily return series (excess of risk-free)."""
    daily_rf = config.RISK_FREE_RATE / config.TRADING_DAYS
    excess = daily_returns - daily_rf
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(config.TRADING_DAYS))


def _max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline of an equity curve (negative number)."""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def run_backtest(
    test_frame: pd.DataFrame,
    signal: np.ndarray,
    transaction_cost: float = config.TRANSACTION_COST,
) -> BacktestResult:
    """Backtest a long/flat strategy over the test window.

    Parameters
    ----------
    test_frame : DataFrame aligned to the test window, must contain
                 ``future_return`` (close[t] -> close[t+1] return).
    signal     : array of 0/1 predictions aligned row-for-row with ``test_frame``
                 (1 = hold the stock for the next day, 0 = stay in cash).
    """
    idx = test_frame.index
    signal = pd.Series(signal, index=idx).astype(float)
    fwd_return = test_frame["future_return"].astype(float)

    # Transaction cost charged whenever the position changes (incl. day 1 entry).
    position_change = signal.diff().abs().fillna(signal.abs())
    costs = position_change * transaction_cost

    strat_daily = signal * fwd_return - costs
    hold_daily = fwd_return  # always invested

    equity_strategy = (1.0 + strat_daily).cumprod()
    equity_hold = (1.0 + hold_daily).cumprod()

    return BacktestResult(
        equity_strategy=equity_strategy,
        equity_hold=equity_hold,
        total_return_strategy=float(equity_strategy.iloc[-1] - 1.0),
        total_return_hold=float(equity_hold.iloc[-1] - 1.0),
        sharpe_strategy=_sharpe(strat_daily),
        sharpe_hold=_sharpe(hold_daily),
        max_drawdown_strategy=_max_drawdown(equity_strategy),
        max_drawdown_hold=_max_drawdown(equity_hold),
        n_trades=int(position_change.gt(0).sum()),
    )
