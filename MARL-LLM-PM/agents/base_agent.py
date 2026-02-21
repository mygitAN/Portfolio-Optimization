"""
Abstract base class for all strategy agents.

Each concrete agent represents one equity investment strategy
(Momentum, Value, or Quality). Agents emit a *preference signal*
(unnormalised weight logit for their strategy sleeve) which the
MetaAllocator combines into a final portfolio weight vector.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BaseStrategyAgent(ABC):
    """
    Interface for a single-strategy RL agent.

    A strategy agent's role is analogous to a strategy team in an
    institutional asset manager: it evaluates conditions relevant to
    its own strategy and signals its preferred allocation intensity.
    The MetaAllocator acts as the portfolio manager that translates
    these signals into final weights.
    """

    strategy_name: str = "base"

    def __init__(self, agent_id: str, config: dict):
        self.agent_id = agent_id
        self.config = config

    @abstractmethod
    def preference_signal(
        self,
        observation: np.ndarray,
        regime: str | None = None,
    ) -> float:
        """
        Emit a scalar preference signal for this strategy.

        A higher value indicates stronger conviction that this strategy
        sleeve should receive a larger allocation.

        Args:
            observation: Current environment observation vector.
            regime: Current regime label from the LLM interpreter
                    (or None if LLM is disabled).

        Returns:
            Scalar float preference signal (unnormalised).
        """

    @abstractmethod
    def update(self, experience: dict) -> dict:
        """
        Update the agent's policy from a single transition experience.

        Args:
            experience: dict with keys {obs, action, reward, next_obs, done}.

        Returns:
            dict of training diagnostics.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent checkpoint to path."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent checkpoint from path."""
