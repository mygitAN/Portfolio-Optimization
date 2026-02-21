"""
Backtesting engine for MARL-LLM-PM.

Runs strategy agents + meta-allocator over historical data within the
walk-forward protocol defined in evaluation/walk_forward.py.

Ablation modes:
  "baseline"    — MARL only, no LLM, no constraints
  "constrained" — MARL + constraints, no LLM
  "llm_only"    — MARL + LLM, no constraints
  "full"        — MARL + LLM + constraints  (proposed system)

Benchmarks run alongside every ablation:
  - 1/N equal weight
  - Volatility-scaled momentum
  - Heuristic rotation (MetaAllocator.heuristic_rotation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from environment.strategy_env import PortfolioStrategyEnv, STRATEGIES
from agents.meta_allocator import MetaAllocator
from llm.regime_interpreter import RegimeInterpreter
from evaluation.metrics import compute_metrics, regime_conditional_metrics, PerformanceReport
from evaluation.walk_forward import DataSplit


AblationMode = Literal["baseline", "constrained", "llm_only", "full"]


@dataclass
class AblationResult:
    mode: str
    split_name: str
    performance: PerformanceReport
    weights_history: pd.DataFrame
    regime_history: pd.Series
    constraint_binding_history: pd.Series
    returns_history: pd.Series
    per_regime_perf: dict[str, PerformanceReport] = field(default_factory=dict)


class Backtester:
    """
    Runs the full evaluation pipeline over a given data split.

    Args:
        strategy_returns: pd.DataFrame (dates x strategies), columns = STRATEGIES.
        allocator: MetaAllocator instance (with agents loaded).
        interpreter: RegimeInterpreter instance.
        config: full experiment config dict.
    """

    def __init__(
        self,
        strategy_returns: pd.DataFrame,
        allocator: MetaAllocator,
        interpreter: RegimeInterpreter,
        config: dict,
    ):
        self.returns = strategy_returns
        self.allocator = allocator
        self.interpreter = interpreter
        self.config = config
        self.env_cfg = config["environment"]
        self.constraint_cfg = config["constraints"]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_split(
        self,
        split: DataSplit,
        mode: AblationMode = "full",
    ) -> AblationResult:
        """Run backtesting over a single data split in the given ablation mode."""
        use_llm = mode in ("llm_only", "full")
        use_constraints = mode in ("constrained", "full")

        data = self.returns.loc[split.start: split.end]
        dates = data.index

        weights_hist = pd.DataFrame(index=dates, columns=STRATEGIES, dtype=float)
        regime_hist = pd.Series(index=dates, dtype=str)
        constraint_hist = pd.Series(index=dates, dtype=bool)
        port_returns = pd.Series(index=dates, dtype=float)

        prev_weights = np.ones(3, dtype=np.float32) / 3
        lookback = min(self.env_cfg.get("lookback_window", 60), len(data))

        for i, date in enumerate(dates):
            start_i = max(0, i - lookback)
            window_returns = data.iloc[start_i: i + 1].values   # (window, 3)

            # Regime classification
            if use_llm and i >= 20:
                label, _ = self.interpreter.classify(
                    window_returns, date=str(date.date())
                )
            else:
                label = RegimeInterpreter._fallback_classifier(window_returns)
            regime_hist.iloc[i] = label

            # Build observation
            obs = self._build_obs(window_returns, prev_weights, label)

            # Meta-allocator action (raw logits)
            raw_logits = self.allocator.step_signal_as_action(
                obs, regime=label if use_llm else None
            )

            # Constraint enforcement
            if use_constraints:
                weights = self._apply_constraints(raw_logits)
                max_w = self.constraint_cfg.get("max_weight", 0.70)
                constraint_active = bool(np.any(weights >= max_w - 1e-4))
            else:
                e = np.exp(raw_logits - raw_logits.max())
                weights = (e / e.sum()).astype(np.float32)
                constraint_active = False

            constraint_hist.iloc[i] = constraint_active
            weights_hist.iloc[i] = weights

            # Realised portfolio return minus transaction cost
            step_ret = float(np.dot(weights, data.iloc[i].values))
            turnover = float(np.abs(weights - prev_weights).sum())
            tc = turnover * self.env_cfg.get("transaction_cost", 0.001)
            port_returns.iloc[i] = step_ret - tc
            prev_weights = weights

        perf = compute_metrics(port_returns)
        per_regime = regime_conditional_metrics(port_returns, regime_hist)

        return AblationResult(
            mode=mode,
            split_name=split.name,
            performance=perf,
            weights_history=weights_hist,
            regime_history=regime_hist,
            constraint_binding_history=constraint_hist,
            returns_history=port_returns,
            per_regime_perf=per_regime,
        )

    def run_benchmarks(self, split: DataSplit) -> dict[str, PerformanceReport]:
        """Run the three benchmarks over the split."""
        data = self.returns.loc[split.start: split.end]
        tc = self.env_cfg.get("transaction_cost", 0.001)
        results = {}

        # 1/N benchmark
        equal_w = np.ones(3) / 3
        eq_returns = (data * equal_w).sum(axis=1)
        results["1/N"] = compute_metrics(eq_returns)

        # Volatility-scaled momentum
        mom_vol = data["momentum"].rolling(20).std().bfill()
        target_vol = 0.15 / np.sqrt(252)
        mom_scale = (target_vol / (mom_vol + 1e-8)).clip(0, 2)
        vol_scaled_ret = data["momentum"] * mom_scale
        results["vol_scaled_momentum"] = compute_metrics(vol_scaled_ret)

        # Heuristic rotation
        lookback = self.env_cfg.get("lookback_window", 60)
        prev_w = np.ones(3) / 3
        hr_returns = pd.Series(index=data.index, dtype=float)
        for i, date in enumerate(data.index):
            start_i = max(0, i - lookback)
            window = data.iloc[start_i: i + 1].values
            label = RegimeInterpreter._fallback_classifier(window)
            w = MetaAllocator.heuristic_rotation(label)
            step_ret = float(np.dot(w, data.iloc[i].values))
            turnover = float(np.abs(w - prev_w).sum())
            hr_returns.iloc[i] = step_ret - turnover * tc
            prev_w = w
        results["heuristic_rotation"] = compute_metrics(hr_returns)

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_obs(
        self,
        window_returns: np.ndarray,
        prev_weights: np.ndarray,
        regime_label: str,
    ) -> np.ndarray:
        features = []
        for s in range(3):
            r = window_returns[:, s]
            mean_r = r.mean() if len(r) > 0 else 0.0
            std_r = r.std() + 1e-8 if len(r) > 0 else 1e-8
            sharpe = mean_r / std_r * np.sqrt(252)
            cum = (1 + r).cumprod()
            max_dd = float((cum / cum.cummax() - 1).min()) if len(cum) > 0 else 0.0
            features.extend([mean_r, std_r, sharpe, max_dd])

        regime_idx = PortfolioStrategyEnv.REGIME_TO_IDX.get(regime_label, 0)
        regime_onehot = np.zeros(5, dtype=np.float32)
        regime_onehot[regime_idx] = 1.0

        return np.array(
            features + list(prev_weights) + list(regime_onehot) + [0.0],
            dtype=np.float32,
        )

    def _apply_constraints(self, logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - logits.max())
        w = e / e.sum()
        max_w = self.constraint_cfg.get("max_weight", 0.70)
        min_w = self.constraint_cfg.get("min_weight", 0.0)
        w = np.clip(w, min_w, max_w)
        return (w / w.sum()).astype(np.float32)
