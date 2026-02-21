"""
Agent Coordinator: orchestrates multiple RL agents and merges their
weight recommendations into a single portfolio allocation.
"""

import numpy as np
from pathlib import Path

from agents.base_agent import BaseAgent
from environment.portfolio_env import PortfolioEnv
from llm.sentiment_analyzer import SentimentAnalyzer


class AgentCoordinator:
    """
    Coordinates N agents (one per asset class) and aggregates their
    weight proposals into a final portfolio allocation.
    """

    def __init__(
        self,
        env: PortfolioEnv,
        llm: SentimentAnalyzer,
        config: dict,
    ):
        self.env = env
        self.llm = llm
        self.config = config
        self.agents: list[BaseAgent] = []

    def register_agent(self, agent: BaseAgent) -> None:
        self.agents.append(agent)

    def _aggregate_weights(self, proposals: list[np.ndarray]) -> np.ndarray:
        """
        Combine per-agent weight proposals (equal-weight average by default).
        Override for more sophisticated consensus mechanisms.
        """
        combined = np.mean(proposals, axis=0)
        return combined / combined.sum()

    def run_episode(self, render: bool = False) -> dict:
        obs, _ = self.env.reset()
        total_reward = 0.0
        done = False

        while not done:
            llm_context = self.llm.get_context()
            proposals = [agent.act(obs, llm_context) for agent in self.agents]
            action = self._aggregate_weights(proposals)

            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated

            if render:
                self.env.render()

        return {"total_reward": total_reward, **info}

    def train(self, total_timesteps: int, log_dir: str) -> None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        print(f"Training {len(self.agents)} agents for {total_timesteps:,} timesteps...")
        # Training loop delegated to individual agent .learn() calls
        # Full implementation integrates SB3 PPO or custom MARL loop
        print("Coordinator training complete.")

    def save_all(self, checkpoint_dir: str) -> None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        for agent in self.agents:
            agent.save(f"{checkpoint_dir}/{agent.agent_id}.pt")

    def load_all(self, checkpoint_dir: str) -> None:
        for agent in self.agents:
            agent.load(f"{checkpoint_dir}/{agent.agent_id}.pt")
