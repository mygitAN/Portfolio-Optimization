"""
Base agent interface for all MARL portfolio agents.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for portfolio management agents."""

    def __init__(self, agent_id: str, asset_universe: list[str], config: dict):
        self.agent_id = agent_id
        self.asset_universe = asset_universe
        self.config = config

    @abstractmethod
    def act(self, observation: np.ndarray, llm_context: dict | None = None) -> np.ndarray:
        """
        Produce portfolio weight recommendations for the agent's asset universe.

        Args:
            observation: Current market observation vector.
            llm_context: Optional LLM-derived sentiment/macro signals.

        Returns:
            np.ndarray of weights (one per asset in universe, sum to 1).
        """

    @abstractmethod
    def learn(self, experiences: list[dict]) -> dict:
        """
        Update agent policy from a batch of experiences.

        Returns:
            dict of training metrics (loss, entropy, etc.)
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist agent checkpoint."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore agent from checkpoint."""
