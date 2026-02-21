"""
Backtesting engine for MARL-LLM-PM strategies.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class BacktestResults:
    returns: pd.Series
    benchmark_returns: pd.Series
    portfolio_values: pd.Series
    metrics: dict = field(default_factory=dict)

    def compute_metrics(self):
        r = self.returns
        self.metrics = {
            "annual_return": (1 + r.mean()) ** 252 - 1,
            "volatility": r.std() * np.sqrt(252),
            "sharpe_ratio": (r.mean() / (r.std() + 1e-8)) * np.sqrt(252),
            "max_drawdown": self._max_drawdown(),
            "total_return": (self.portfolio_values.iloc[-1] / self.portfolio_values.iloc[0]) - 1,
        }
        if self.metrics["volatility"] > 0:
            self.metrics["calmar_ratio"] = (
                self.metrics["annual_return"] / abs(self.metrics["max_drawdown"])
                if self.metrics["max_drawdown"] != 0 else 0.0
            )
        return self.metrics

    def _max_drawdown(self) -> float:
        cumulative = (1 + self.returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()

    def print_summary(self):
        if not self.metrics:
            self.compute_metrics()
        print("\n" + "=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)
        for k, v in self.metrics.items():
            print(f"  {k:25s}: {v:.4f}")
        print("=" * 50)


class Backtester:
    """
    Runs a trained MARL-LLM-PM strategy over historical data
    and computes performance metrics.
    """

    def __init__(self, config: dict, start_date: str, end_date: str):
        self.config = config
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark = config["backtesting"].get("benchmark", "SPY")

    def run(self) -> BacktestResults:
        print(f"Backtesting from {self.start_date} to {self.end_date}...")
        # Placeholder: replace with real data loading and agent inference
        n_days = 252
        np.random.seed(42)
        strategy_returns = pd.Series(
            np.random.normal(0.0006, 0.01, n_days),
            index=pd.date_range(self.start_date, periods=n_days, freq="B"),
            name="strategy",
        )
        benchmark_returns = pd.Series(
            np.random.normal(0.0004, 0.012, n_days),
            index=strategy_returns.index,
            name=self.benchmark,
        )
        portfolio_values = (1 + strategy_returns).cumprod() * self.config["environment"]["initial_cash"]

        results = BacktestResults(
            returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            portfolio_values=portfolio_values,
        )
        results.compute_metrics()
        return results
