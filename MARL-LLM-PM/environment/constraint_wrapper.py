"""
Mandate-style constraint wrapper for the strategy allocation environment.

Implements the CMDP constraint layer described in the thesis:
  - Concentration cap: w_i <= max_weight for each strategy i
  - Full investment: sum(w) = 1
  - Non-negativity: w_i >= 0

Two enforcement mechanisms are available (configured via config):
  1. "projection"  — project the softmax output onto the feasible set
  2. "penalty"     — add a constraint violation term to the reward

The wrapper is designed so that unconstrained vs constrained modes
produce directly comparable observations and reward scales, allowing
clean ablation comparisons.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environment.strategy_env import PortfolioStrategyEnv, N_STRATEGIES


class ConstraintWrapper(gym.Wrapper):
    """
    Wraps PortfolioStrategyEnv with mandate-style concentration constraints.

    Args:
        env: the base PortfolioStrategyEnv instance.
        config: constraint config dict, e.g.:
            {
              "enabled": true,
              "mechanism": "projection",   # or "penalty"
              "max_weight": 0.70,
              "min_weight": 0.05,
              "penalty_coeff": 5.0
            }
    """

    def __init__(self, env: PortfolioStrategyEnv, config: dict):
        super().__init__(env)
        self.cfg = config
        self.enabled: bool = config.get("enabled", True)
        self.mechanism: str = config.get("mechanism", "projection")
        self.max_weight: float = config.get("max_weight", 0.70)
        self.min_weight: float = config.get("min_weight", 0.0)
        self.penalty_coeff: float = config.get("penalty_coeff", 5.0)

        assert self.max_weight >= 1.0 / N_STRATEGIES, (
            "max_weight must be >= 1/N_STRATEGIES to allow a feasible solution"
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray):
        if not self.enabled:
            return self.env.step(action)

        if self.mechanism == "projection":
            obs, reward, terminated, truncated, info = self.env.step(action)
            # Project weights post-hoc (env already converted logits → softmax)
            constrained_weights = self._project_simplex(
                np.array(info["weights"]), self.min_weight, self.max_weight
            )
            info["weights"] = constrained_weights.tolist()
            info["constraint_active"] = self._any_constraint_active(constrained_weights)
        else:
            # "penalty" mode: let env step freely, penalise violations
            obs, reward, terminated, truncated, info = self.env.step(action)
            weights = np.array(info["weights"])
            violation = self._constraint_violation(weights)
            reward -= self.penalty_coeff * violation
            info["constraint_violation"] = float(violation)
            info["constraint_active"] = violation > 0.0

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Constraint utilities
    # ------------------------------------------------------------------

    def _project_simplex(
        self, w: np.ndarray, lo: float, hi: float
    ) -> np.ndarray:
        """
        Project w onto the simplex intersected with [lo, hi]^n.
        Uses iterative clipping + re-normalisation (Duchi et al. 2008 variant).
        """
        w = np.clip(w, lo, hi)
        n = len(w)
        for _ in range(n):
            total = w.sum()
            if abs(total - 1.0) < 1e-8:
                break
            excess = total - 1.0
            # Distribute excess proportionally from unconstrained dims
            free = (w > lo + 1e-9) & (w < hi - 1e-9)
            if free.sum() == 0:
                break
            w[free] -= excess / free.sum()
            w = np.clip(w, lo, hi)
        w = w / w.sum()
        return w.astype(np.float32)

    def _constraint_violation(self, w: np.ndarray) -> float:
        """L1 magnitude of constraint violations."""
        upper = np.maximum(0.0, w - self.max_weight)
        lower = np.maximum(0.0, self.min_weight - w)
        return float((upper + lower).sum())

    def _any_constraint_active(self, w: np.ndarray) -> bool:
        return bool(
            np.any(w >= self.max_weight - 1e-4)
            or np.any(w <= self.min_weight + 1e-4)
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        base_env: PortfolioStrategyEnv,
        constraint_config: dict,
        enabled: bool,
    ) -> "ConstraintWrapper":
        cfg = dict(constraint_config)
        cfg["enabled"] = enabled
        return cls(base_env, cfg)
