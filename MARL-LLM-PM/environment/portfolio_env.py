"""
Multi-asset portfolio trading environment compatible with Gymnasium.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any


class PortfolioEnv(gym.Env):
    """
    A multi-asset portfolio management environment.

    Observation: [prices, returns, technical_indicators, llm_features]
    Action: portfolio weights for each asset (sum to 1)
    Reward: risk-adjusted return (Sharpe-like) over the step
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.assets = self._flatten_assets(config["assets"])
        self.n_assets = len(self.assets)
        self.lookback = config.get("lookback_window", 60)
        self.initial_cash = config.get("initial_cash", 1_000_000)
        self.transaction_cost = config.get("transaction_cost", 0.001)

        # Observation: lookback * n_assets price features + cash position
        obs_dim = self.lookback * self.n_assets + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action: portfolio weights (will be softmax-normalized)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        self._reset_state()

    def _flatten_assets(self, asset_groups: list) -> list:
        tickers = []
        for group in asset_groups:
            tickers.extend(group["tickers"])
        return tickers

    def _reset_state(self):
        self.portfolio_value = self.initial_cash
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.step_count = 0
        self.price_history = None

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._reset_state()
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Normalize weights
        weights = np.clip(action, 0, 1)
        weights = weights / (weights.sum() + 1e-8)

        # Simulate price change (placeholder — real data loaded by Backtester)
        price_change = self.np_random.normal(0.0005, 0.01, self.n_assets)
        returns = np.dot(weights, price_change)

        # Transaction costs
        weight_diff = np.abs(weights - self.weights).sum()
        cost = weight_diff * self.transaction_cost

        net_return = returns - cost
        self.portfolio_value *= 1 + net_return
        self.weights = weights.copy()
        self.step_count += 1

        reward = net_return
        terminated = self.portfolio_value < self.initial_cash * 0.5
        truncated = False

        obs = self._get_obs()
        info = {
            "portfolio_value": self.portfolio_value,
            "weights": weights,
            "net_return": net_return,
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        # Placeholder observation — real implementation uses price_history
        obs = np.zeros(self.lookback * self.n_assets + 1, dtype=np.float32)
        obs[-1] = self.portfolio_value / self.initial_cash
        return obs

    def render(self):
        print(f"Step {self.step_count} | Portfolio: ${self.portfolio_value:,.2f} | Weights: {self.weights}")
