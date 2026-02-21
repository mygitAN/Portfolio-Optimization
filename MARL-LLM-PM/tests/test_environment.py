"""
Tests for the portfolio environment.
"""

import numpy as np
import pytest

from environment.portfolio_env import PortfolioEnv


@pytest.fixture
def default_config():
    return {
        "assets": [
            {"type": "equity", "tickers": ["SPY", "QQQ"]},
            {"type": "bond", "tickers": ["TLT"]},
        ],
        "initial_cash": 100_000,
        "transaction_cost": 0.001,
        "lookback_window": 10,
    }


def test_env_reset(default_config):
    env = PortfolioEnv(default_config)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)


def test_env_step_valid_action(default_config):
    env = PortfolioEnv(default_config)
    env.reset()
    action = np.array([0.5, 0.3, 0.2])
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert "portfolio_value" in info


def test_env_weights_sum_to_one(default_config):
    env = PortfolioEnv(default_config)
    env.reset()
    action = np.array([1.0, 2.0, 3.0])  # un-normalized
    _, _, _, _, info = env.step(action)
    assert abs(info["weights"].sum() - 1.0) < 1e-6


def test_env_n_assets(default_config):
    env = PortfolioEnv(default_config)
    assert env.n_assets == 3  # SPY, QQQ, TLT
