"""
Value strategy agent.

Economic rationale: targets assets priced below fundamental value,
exploiting mean reversion and mispricing (Lakonishok et al., 1994).

Strong regimes: RECOVERY, rising rate / repricing environments.
Underperforms: growth-led, prolonged low-rate / QE environments.
"""

from __future__ import annotations

import numpy as np

from agents.base_agent import BaseStrategyAgent
from environment.strategy_env import STRATEGIES


class ValueAgent(BaseStrategyAgent):
    """
    Rule-based value preference signal (Phase 1 baseline).

    Value strategies tend to recover from drawdowns during economic
    rebounds and benefit from mean-reversion dynamics. The preference
    signal is boosted in RECOVERY regimes and dampened when risk-off
    sentiment dominates (value traps).
    """

    strategy_name = "value"

    REGIME_SCALE: dict[str, float] = {
        "TRENDING-LOWVOL":      0.80,
        "STRESS-DRAWDOWN":      0.60,   # value cheap but no catalyst yet
        "RECOVERY":             1.0,    # strongest environment for value
        "SIDEWAYS-HIGHCORR":    0.75,
        "RISK-OFF-DEFENSIVE":   0.55,
    }

    _STRAT_IDX = STRATEGIES.index("value")
    _SHARPE_OFFSET = _STRAT_IDX * 4 + 2

    def preference_signal(
        self, observation: np.ndarray, regime: str | None = None
    ) -> float:
        raw_sharpe = float(observation[self._SHARPE_OFFSET])
        scale = self.REGIME_SCALE.get(regime or "RECOVERY", 0.75)
        return raw_sharpe * scale

    def update(self, experience: dict) -> dict:
        return {"loss": 0.0}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
