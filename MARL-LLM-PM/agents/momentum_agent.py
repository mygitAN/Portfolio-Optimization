"""
Momentum strategy agent.

Economic rationale: exploits persistence in asset returns based on
investor under-reaction and trend continuation (Jegadeesh & Titman, 1993).

Strong regimes: TRENDING-LOWVOL, sustained directional markets.
Crash-prone regimes: STRESS-DRAWDOWN, bear-to-bull transitions.

The agent's preference signal is intentionally conservative during
STRESS-DRAWDOWN and RECOVERY regimes (momentum crash risk).
"""

from __future__ import annotations

import numpy as np

from agents.base_agent import BaseStrategyAgent
from environment.strategy_env import STRATEGIES


class MomentumAgent(BaseStrategyAgent):
    """
    Rule-based momentum preference signal (Phase 1 baseline).

    The preference signal is the rolling Sharpe of the momentum
    strategy extracted from the observation, scaled by a regime
    multiplier to reflect momentum crash risk.

    In Phase 2 this will be replaced by a learned PPO policy.
    """

    strategy_name = "momentum"

    # Regime multipliers: reduce momentum conviction during crash-prone regimes
    REGIME_SCALE: dict[str, float] = {
        "TRENDING-LOWVOL":      1.0,
        "STRESS-DRAWDOWN":      0.20,   # momentum crash risk — strongly muted
        "RECOVERY":             0.40,   # reversal risk — cautious
        "SIDEWAYS-HIGHCORR":    0.70,
        "RISK-OFF-DEFENSIVE":   0.50,
    }

    # Index of momentum strategy in STRATEGIES list
    _STRAT_IDX = STRATEGIES.index("momentum")

    # Observation layout (must match PortfolioStrategyEnv._get_obs):
    #   [mean_r, std_r, sharpe, max_dd] * 3  then weights (3) then regime (5) then step
    _SHARPE_OFFSET = _STRAT_IDX * 4 + 2  # sharpe is 3rd feature per strategy

    def preference_signal(
        self, observation: np.ndarray, regime: str | None = None
    ) -> float:
        raw_sharpe = float(observation[self._SHARPE_OFFSET])
        scale = self.REGIME_SCALE.get(regime or "TRENDING-LOWVOL", 0.7)
        return raw_sharpe * scale

    def update(self, experience: dict) -> dict:
        # Placeholder — PPO learning loop will be wired here.
        return {"loss": 0.0}

    def save(self, path: str) -> None:
        pass  # no learned params yet

    def load(self, path: str) -> None:
        pass
