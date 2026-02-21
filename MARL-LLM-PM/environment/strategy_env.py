"""
Strategy-level portfolio allocation environment.

Each action is a weight vector over equity investment strategies
(Momentum, Value, Quality). The environment tracks cumulative
strategy returns and rewards the agent based on risk-adjusted performance.

This is a Gymnasium-compatible MDP. The constraint wrapper
(constraint_wrapper.py) sits on top of this environment and is
applied separately so that constrained vs unconstrained modes can
be compared directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Any


# Strategy identifiers — order is fixed throughout the codebase.
STRATEGIES = ["momentum", "value", "quality"]
N_STRATEGIES = len(STRATEGIES)


class PortfolioStrategyEnv(gym.Env):
    """
    Multi-strategy equity allocation environment.

    State s_t:
        - lookback-window of strategy return features (rolling mean,
          rolling std, rolling Sharpe-like, max drawdown over window)
        - current portfolio weights
        - one-hot or continuous regime encoding (5 regimes)
        - steps remaining (normalised)

    Action a_t:
        Raw weight logits over N_STRATEGIES strategies.
        Converted to simplex via softmax inside the env.
        The constraint wrapper clips / masks before this step.

    Reward r_t:
        Risk-adjusted step return minus transaction cost:
            r_t = portfolio_return_t - λ * portfolio_vol_t - tc * turnover_t

    Args:
        strategy_returns: pd.DataFrame of shape (T, N_STRATEGIES),
            daily returns for each strategy, index = dates.
        config: experiment config dict (environment section).
    """

    metadata = {"render_modes": ["human"]}

    # Fixed regime vocabulary (must match RegimeInterpreter)
    REGIMES = [
        "TRENDING-LOWVOL",
        "STRESS-DRAWDOWN",
        "RECOVERY",
        "SIDEWAYS-HIGHCORR",
        "RISK-OFF-DEFENSIVE",
    ]
    N_REGIMES = len(REGIMES)
    REGIME_TO_IDX = {r: i for i, r in enumerate(REGIMES)}

    def __init__(self, strategy_returns: pd.DataFrame, config: dict):
        super().__init__()
        assert list(strategy_returns.columns) == STRATEGIES, (
            f"strategy_returns columns must be {STRATEGIES}"
        )
        self.returns = strategy_returns.values.astype(np.float32)   # (T, 3)
        self.dates = strategy_returns.index
        self.T = len(self.returns)

        cfg = config
        self.lookback = cfg.get("lookback_window", 60)
        self.initial_cash = cfg.get("initial_cash", 1_000_000)
        self.transaction_cost = cfg.get("transaction_cost", 0.001)
        self.risk_aversion = cfg.get("risk_aversion", 1.0)

        # Features per strategy: [mean_ret, std_ret, sharpe, max_dd]
        n_strategy_features = N_STRATEGIES * 4
        # Current weights + regime one-hot + steps_remaining
        n_other = N_STRATEGIES + self.N_REGIMES + 1

        obs_dim = n_strategy_features + n_other
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Raw logits — softmax applied internally
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(N_STRATEGIES,), dtype=np.float32
        )

        self._regime_idx: int = 0   # updated externally by the wrapper/runner
        self._reset_state()

    # ------------------------------------------------------------------
    # External interface for regime injection
    # ------------------------------------------------------------------

    def set_regime(self, regime_label: str) -> None:
        """Called by the runner after each LLM regime update."""
        self._regime_idx = self.REGIME_TO_IDX.get(regime_label, 0)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Convert logits to weights via softmax
        weights = self._softmax(action)

        # Realised strategy returns at current step
        step_returns = self.returns[self._t]               # (3,)
        portfolio_return = float(np.dot(weights, step_returns))

        # Volatility penalty (rolling std of portfolio returns)
        if len(self._portfolio_returns_history) >= 5:
            port_vol = float(np.std(self._portfolio_returns_history[-20:]))
        else:
            port_vol = 0.0

        # Transaction cost: L1 turnover
        turnover = float(np.abs(weights - self._weights).sum())
        tc = turnover * self.transaction_cost

        # Risk-adjusted reward
        reward = portfolio_return - self.risk_aversion * port_vol - tc

        self._portfolio_returns_history.append(portfolio_return)
        self._portfolio_value *= 1.0 + portfolio_return - tc
        self._weights = weights.copy()
        self._t += 1

        terminated = (
            self._portfolio_value < self.initial_cash * 0.5
            or self._t >= self.T - 1
        )
        truncated = False

        obs = self._get_obs()
        info = {
            "date": str(self.dates[self._t - 1]),
            "weights": weights.tolist(),
            "portfolio_return": portfolio_return,
            "portfolio_value": self._portfolio_value,
            "turnover": turnover,
            "regime": self.REGIMES[self._regime_idx],
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        w = dict(zip(STRATEGIES, self._weights))
        print(
            f"t={self._t:4d} | "
            f"Value=${self._portfolio_value:,.0f} | "
            f"Weights={w} | "
            f"Regime={self.REGIMES[self._regime_idx]}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self._t = self.lookback           # start after enough history
        self._weights = np.ones(N_STRATEGIES, dtype=np.float32) / N_STRATEGIES
        self._portfolio_value = float(self.initial_cash)
        self._portfolio_returns_history: list[float] = []
        self._regime_idx = 0

    def _get_obs(self) -> np.ndarray:
        window = self.returns[self._t - self.lookback: self._t]  # (lookback, 3)

        features = []
        for s in range(N_STRATEGIES):
            r = window[:, s]
            mean_r = r.mean()
            std_r = r.std() + 1e-8
            sharpe = mean_r / std_r * np.sqrt(252)
            cum = (1 + r).cumprod()
            max_dd = float((cum / cum.cummax() - 1).min())
            features.extend([mean_r, std_r, sharpe, max_dd])

        regime_onehot = np.zeros(self.N_REGIMES, dtype=np.float32)
        regime_onehot[self._regime_idx] = 1.0

        steps_remaining = (self.T - 1 - self._t) / max(self.T, 1)

        obs = np.array(
            features + self._weights.tolist() + regime_onehot.tolist() + [steps_remaining],
            dtype=np.float32,
        )
        return obs

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()
