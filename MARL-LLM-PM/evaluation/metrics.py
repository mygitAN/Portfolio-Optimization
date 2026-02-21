"""
Performance and risk metrics.

All metrics follow standard definitions used in the portfolio finance
literature. Results are annualised assuming 252 trading days per year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class PerformanceReport:
    annual_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    total_return: float
    avg_turnover: float | None = None

    def as_dict(self) -> dict[str, float]:
        return {
            "annual_return":  self.annual_return,
            "volatility":     self.volatility,
            "sharpe_ratio":   self.sharpe_ratio,
            "sortino_ratio":  self.sortino_ratio,
            "max_drawdown":   self.max_drawdown,
            "calmar_ratio":   self.calmar_ratio,
            "total_return":   self.total_return,
            "avg_turnover":   self.avg_turnover or float("nan"),
        }

    def print(self, title: str = "") -> None:
        header = f"  {title}  " if title else ""
        print(f"\n{'=' * 50}")
        if header:
            print(header.center(50))
            print("=" * 50)
        for k, v in self.as_dict().items():
            print(f"  {k:20s}: {v:+.4f}")
        print("=" * 50)


def compute_metrics(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> PerformanceReport:
    """
    Compute standard portfolio performance metrics.

    Args:
        returns: daily portfolio returns series.
        turnover: daily portfolio turnover series (optional).
        rf: risk-free rate (annualised, default 0).
        periods_per_year: trading days per year.
    """
    r = returns.dropna()
    n = len(r)

    ann_factor = periods_per_year
    daily_rf = rf / ann_factor

    excess = r - daily_rf
    mean_excess = excess.mean()
    std_r = r.std()

    annual_return = (1 + r.mean()) ** ann_factor - 1
    volatility = std_r * np.sqrt(ann_factor)

    sharpe = (mean_excess / (std_r + 1e-10)) * np.sqrt(ann_factor)

    downside = r[r < daily_rf]
    downside_std = downside.std() if len(downside) > 0 else 1e-10
    sortino = (mean_excess / (downside_std + 1e-10)) * np.sqrt(ann_factor)

    cum = (1 + r).cumprod()
    drawdown = cum / cum.cummax() - 1
    max_dd = float(drawdown.min())

    calmar = annual_return / abs(max_dd) if max_dd != 0 else float("nan")
    total_return = float(cum.iloc[-1] - 1) if n > 0 else 0.0

    avg_to = float(turnover.mean()) if turnover is not None else None

    return PerformanceReport(
        annual_return=annual_return,
        volatility=volatility,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        total_return=total_return,
        avg_turnover=avg_to,
    )


def regime_conditional_metrics(
    returns: pd.Series,
    regimes: pd.Series,
) -> dict[str, PerformanceReport]:
    """
    Compute per-regime performance metrics.

    Args:
        returns: daily portfolio returns.
        regimes: daily regime labels (must share index with returns).

    Returns:
        dict mapping regime label → PerformanceReport.
    """
    aligned = pd.concat([returns, regimes], axis=1, join="inner")
    aligned.columns = ["returns", "regime"]
    results = {}
    for label, group in aligned.groupby("regime"):
        results[str(label)] = compute_metrics(group["returns"])
    return results
