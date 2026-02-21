"""
MetaAllocator: hierarchical portfolio manager.

Collects preference signals from all strategy agents and translates
them into a final portfolio weight allocation. This mirrors the
institutional structure described in the thesis:

    Strategy agents (teams) → preference signals
    MetaAllocator (PM)      → final weights

Two modes:
  1. "signal_softmax": softmax over raw agent signals (rule-based baseline)
  2. "learned":        PPO-trained policy that conditions on (signals, obs,
                       regime) to produce final weights (Phase 2)

The MetaAllocator is the primary unit trained by the MARL loop.
Individual strategy agents may also be trained (centralised-training,
decentralised-execution) or kept as fixed heuristic signal generators.
"""

from __future__ import annotations

import numpy as np
from typing import Literal

from agents.base_agent import BaseStrategyAgent
from environment.strategy_env import STRATEGIES, N_STRATEGIES


class MetaAllocator:
    """
    Aggregates strategy agent signals into a portfolio weight vector.

    Args:
        agents: list of BaseStrategyAgent instances (one per strategy,
                in STRATEGIES order).
        config: meta_allocator section of experiment config.
    """

    def __init__(self, agents: list[BaseStrategyAgent], config: dict):
        assert len(agents) == N_STRATEGIES, (
            f"Expected {N_STRATEGIES} agents, one per strategy in {STRATEGIES}"
        )
        self.agents = agents
        self.config = config
        self.mode: str = config.get("mode", "signal_softmax")

    # ------------------------------------------------------------------
    # Core allocation
    # ------------------------------------------------------------------

    def allocate(
        self,
        observation: np.ndarray,
        regime: str | None = None,
    ) -> np.ndarray:
        """
        Produce final strategy weights given current observation + regime.

        Returns:
            np.ndarray of shape (N_STRATEGIES,) summing to 1, >= 0.
        """
        signals = np.array(
            [a.preference_signal(observation, regime) for a in self.agents],
            dtype=np.float32,
        )

        if self.mode == "signal_softmax":
            return self._softmax(signals)
        elif self.mode == "equal_signal":
            # Ignores signal magnitude — for ablation comparison
            return np.ones(N_STRATEGIES, dtype=np.float32) / N_STRATEGIES
        elif self.mode == "learned":
            # Placeholder: learned policy would call a neural network here
            # and return its weight output.
            return self._softmax(signals)
        else:
            raise ValueError(f"Unknown meta_allocator mode: {self.mode}")

    def step_signal_as_action(
        self, observation: np.ndarray, regime: str | None = None
    ) -> np.ndarray:
        """
        Return raw logits (before softmax) for use as environment action.
        The environment applies softmax internally.
        """
        signals = np.array(
            [a.preference_signal(observation, regime) for a in self.agents],
            dtype=np.float32,
        )
        return signals

    # ------------------------------------------------------------------
    # Benchmarks
    # ------------------------------------------------------------------

    @staticmethod
    def equal_weight() -> np.ndarray:
        """1/N benchmark."""
        return np.ones(N_STRATEGIES, dtype=np.float32) / N_STRATEGIES

    @staticmethod
    def heuristic_rotation(regime: str) -> np.ndarray:
        """
        Rule-based regime rotation benchmark.

        Assigns high weight to the strategy best suited for the given
        regime (hard-coded heuristic, not learned).
        """
        mapping: dict[str, np.ndarray] = {
            "TRENDING-LOWVOL":     np.array([0.60, 0.20, 0.20]),  # momentum
            "STRESS-DRAWDOWN":     np.array([0.10, 0.20, 0.70]),  # quality
            "RECOVERY":            np.array([0.25, 0.60, 0.15]),  # value
            "SIDEWAYS-HIGHCORR":   np.array([0.33, 0.33, 0.34]),  # equal
            "RISK-OFF-DEFENSIVE":  np.array([0.15, 0.15, 0.70]),  # quality
        }
        w = mapping.get(regime, np.ones(N_STRATEGIES) / N_STRATEGIES)
        return w.astype(np.float32)

    # ------------------------------------------------------------------
    # Training interface
    # ------------------------------------------------------------------

    def update(self, experiences: list[dict]) -> dict:
        """
        Update all agents and the meta-policy from a batch of experiences.
        Each experience dict: {obs, action, reward, next_obs, done, regime}.
        """
        diagnostics: dict = {}
        for agent in self.agents:
            for exp in experiences:
                d = agent.update(exp)
                diagnostics[agent.strategy_name] = d
        return diagnostics

    def save_all(self, checkpoint_dir: str) -> None:
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        for agent in self.agents:
            agent.save(f"{checkpoint_dir}/{agent.strategy_name}.pt")

    def load_all(self, checkpoint_dir: str) -> None:
        for agent in self.agents:
            agent.load(f"{checkpoint_dir}/{agent.strategy_name}.pt")

    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return (e / e.sum()).astype(np.float32)
