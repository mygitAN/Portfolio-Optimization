"""
MARL-LLM-PM: Multi-Agent Reinforcement Learning with LLM for Portfolio Management
Entry point for training, evaluation, and backtesting.
"""

import argparse
import yaml
from pathlib import Path

from environment.portfolio_env import PortfolioEnv
from agents.coordinator import AgentCoordinator
from backtesting.backtest import Backtester
from llm.sentiment_analyzer import SentimentAnalyzer


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_training(config: dict):
    print("Initializing training...")
    env = PortfolioEnv(config["environment"])
    llm = SentimentAnalyzer(config["llm"])
    coordinator = AgentCoordinator(env, llm, config["agents"])
    coordinator.train(
        total_timesteps=config["training"]["total_timesteps"],
        log_dir=config["training"]["log_dir"],
    )
    print("Training complete.")


def run_backtest(config: dict, start: str, end: str):
    print(f"Running backtest from {start} to {end}...")
    backtester = Backtester(config, start_date=start, end_date=end)
    results = backtester.run()
    results.print_summary()


def main():
    parser = argparse.ArgumentParser(description="MARL-LLM-PM")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["train", "backtest", "evaluate"], default="train")
    parser.add_argument("--start", default=None, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Backtest end date (YYYY-MM-DD)")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "train":
        run_training(config)
    elif args.mode == "backtest":
        run_backtest(config, args.start, args.end)
    elif args.mode == "evaluate":
        print("Evaluation mode: coming soon.")


if __name__ == "__main__":
    main()
