"""
Quality strategy agent.

Economic rationale: selects firms with strong balance sheets, high
profitability, and earnings stability (Novy-Marx, 2013).

Strong regimes: RISK-OFF-DEFENSIVE, high-uncertainty / stress environments.
Underperforms: strong risk-on rallies, speculative expansions.
"""

from __future__ import annotations

import numpy as np

from agents.base_agent import BaseStrategyAgent
from environment.strategy_env import STRATEGIES


class QualityAgent(BaseStrategyAgent):
    """
    Rule-based quality preference signal (Phase 1 baseline).

    Quality is the natural defensive complement to momentum and value.
    It is boosted in risk-off and stress regimes and muted during
    aggressive risk-on environments.
    """

    strategy_name = "quality"

    REGIME_SCALE: dict[str, float] = {
        "TRENDING-LOWVOL":      0.70,
        "STRESS-DRAWDOWN":      1.0,    # quality shines in stress
        "RECOVERY":             0.65,   # lags as cyclicals outperform
        "SIDEWAYS-HIGHCORR":    0.80,
        "RISK-OFF-DEFENSIVE":   1.0,    # peak quality environment
    }

    _STRAT_IDX = STRATEGIES.index("quality")
    _SHARPE_OFFSET = _STRAT_IDX * 4 + 2

    def preference_signal(
        self, observation: np.ndarray, regime: str | None = None
    ) -> float:
        raw_sharpe = float(observation[self._SHARPE_OFFSET])
        scale = self.REGIME_SCALE.get(regime or "RISK-OFF-DEFENSIVE", 0.80)
        return raw_sharpe * scale

    def update(self, experience: dict) -> dict:
        return {"loss": 0.0}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
