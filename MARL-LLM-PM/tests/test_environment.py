"""
Tests for the strategy allocation environment and constraint wrapper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from environment.strategy_env import PortfolioStrategyEnv, STRATEGIES, N_STRATEGIES
from environment.constraint_wrapper import ConstraintWrapper


def _make_returns(n: int = 200) -> pd.DataFrame:
    np.random.seed(0)
    data = np.random.normal(0.0003, 0.01, (n, N_STRATEGIES))
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(data, index=dates, columns=STRATEGIES)


@pytest.fixture
def base_env():
    returns = _make_returns()
    config = {"initial_cash": 100_000, "transaction_cost": 0.001, "lookback_window": 20}
    return PortfolioStrategyEnv(returns, config)


@pytest.fixture
def constrained_env(base_env):
    cfg = {"enabled": True, "mechanism": "projection", "max_weight": 0.70, "min_weight": 0.0}
    return ConstraintWrapper(base_env, cfg)


# ------------------------------------------------------------------
# Environment tests
# ------------------------------------------------------------------

def test_observation_space_shape(base_env):
    obs, _ = base_env.reset()
    assert obs.shape == base_env.observation_space.shape


def test_step_returns_valid_obs(base_env):
    base_env.reset()
    action = np.array([1.0, 0.5, -0.5])   # raw logits
    obs, reward, terminated, truncated, info = base_env.step(action)
    assert obs.shape == base_env.observation_space.shape
    assert isinstance(reward, float)


def test_weights_sum_to_one(base_env):
    base_env.reset()
    obs, _, _, _, info = base_env.step(np.array([2.0, 1.0, 0.5]))
    weights = np.array(info["weights"])
    assert abs(weights.sum() - 1.0) < 1e-5


def test_weights_nonnegative(base_env):
    base_env.reset()
    _, _, _, _, info = base_env.step(np.array([-5.0, 2.0, 1.0]))
    assert all(w >= -1e-7 for w in info["weights"])


def test_regime_injection(base_env):
    base_env.reset()
    base_env.set_regime("STRESS-DRAWDOWN")
    assert base_env._regime_idx == PortfolioStrategyEnv.REGIME_TO_IDX["STRESS-DRAWDOWN"]


def test_regime_in_obs(base_env):
    base_env.reset()
    base_env.set_regime("RECOVERY")
    obs = base_env._get_obs()
    regime_start = N_STRATEGIES * 4  # after strategy features
    regime_vec = obs[regime_start: regime_start + 5]
    recovery_idx = PortfolioStrategyEnv.REGIME_TO_IDX["RECOVERY"]
    assert regime_vec[recovery_idx] == 1.0


def test_n_strategies(base_env):
    assert base_env.n_assets == N_STRATEGIES


# ------------------------------------------------------------------
# Constraint wrapper tests
# ------------------------------------------------------------------

def test_constraint_wrapper_projection(constrained_env):
    constrained_env.reset()
    # Force an extreme action that would violate max_weight=0.70
    action = np.array([10.0, 0.1, 0.1])   # would give ~99% to momentum
    _, _, _, _, info = constrained_env.step(action)
    weights = np.array(info["weights"])
    assert all(w <= 0.70 + 1e-5 for w in weights), f"Constraint violated: {weights}"
    assert abs(weights.sum() - 1.0) < 1e-5


def test_unconstrained_wrapper_passes_through():
    returns = _make_returns()
    base_env = PortfolioStrategyEnv(
        returns, {"initial_cash": 10_000, "transaction_cost": 0, "lookback_window": 20}
    )
    cfg = {"enabled": False, "mechanism": "projection", "max_weight": 0.50, "min_weight": 0.0}
    wrapped = ConstraintWrapper(base_env, cfg)
    wrapped.reset()
    action = np.array([10.0, 0.1, 0.1])
    _, _, _, _, info = wrapped.step(action)
    # Unconstrained: weight can exceed 0.50
    weights = np.array(info["weights"])
    assert weights.max() > 0.50


def test_penalty_mode():
    returns = _make_returns()
    base_env = PortfolioStrategyEnv(
        returns, {"initial_cash": 10_000, "transaction_cost": 0, "lookback_window": 20}
    )
    cfg = {"enabled": True, "mechanism": "penalty", "max_weight": 0.50,
           "min_weight": 0.0, "penalty_coeff": 5.0}
    wrapped = ConstraintWrapper(base_env, cfg)
    wrapped.reset()
    action = np.array([10.0, 0.1, 0.1])
    _, reward, _, _, info = wrapped.step(action)
    assert "constraint_violation" in info
