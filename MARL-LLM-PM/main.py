"""
MARL-LLM-PM: LLM-assisted MARL for equity strategy rotation.
CLI entry point for training, backtesting, and ablation runs.

Usage:
  python main.py --mode train   --ablation full
  python main.py --mode backtest --ablation constrained
  python main.py --mode ablation            # run all four ablations
  python main.py --mode held_out            # FINAL evaluation (run once only)
"""

from __future__ import annotations

import argparse
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from agents.momentum_agent import MomentumAgent
from agents.value_agent import ValueAgent
from agents.quality_agent import QualityAgent
from agents.meta_allocator import MetaAllocator
from llm.regime_interpreter import RegimeInterpreter
from backtesting.backtest import Backtester, AblationMode
from evaluation.walk_forward import WalkForwardManager
from evaluation.attribution import AttributionAnalyzer


ABLATION_MODES: list[AblationMode] = ["baseline", "constrained", "llm_only", "full"]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_allocator(config: dict) -> MetaAllocator:
    """Construct strategy agents and meta-allocator from config."""
    agent_cfg = config["agents"]
    agents = [
        MomentumAgent("momentum_agent", agent_cfg),
        ValueAgent("value_agent", agent_cfg),
        QualityAgent("quality_agent", agent_cfg),
    ]
    return MetaAllocator(agents, config["meta_allocator"])


def load_strategy_returns(config: dict) -> pd.DataFrame:
    """
    Load or generate strategy return series.
    Columns: [momentum, value, quality], index: DatetimeIndex.
    """
    data_cfg = config["data"]
    strategies = data_cfg["strategies"]

    if data_cfg["mode"] == "csv":
        df = pd.read_csv(data_cfg["csv_path"], index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        return df[strategies]

    # Synthetic data (development placeholder)
    # Replace with real JSE/index data once available.
    print("[data] Using synthetic strategy returns (development mode).")
    np.random.seed(42)
    n_days = 2520  # ~10 years

    # Regime-aware synthetic returns with momentum crash episodes
    base_dates = pd.bdate_range("2015-01-01", periods=n_days)
    corr = np.array([[1.0, -0.2, 0.1],
                     [-0.2, 1.0, 0.3],
                     [0.1, 0.3, 1.0]])
    L = np.linalg.cholesky(corr)
    raw = np.random.normal(0, 1, (n_days, 3)) @ L.T
    # Annualised vols: momentum 18%, value 14%, quality 10%
    vols = np.array([0.18, 0.14, 0.10]) / np.sqrt(252)
    # Drift: momentum 8%, value 5%, quality 6%
    drift = np.array([0.08, 0.05, 0.06]) / 252
    returns = raw * vols + drift

    # Inject momentum crash episodes (2 episodes)
    for crash_start in [500, 1800]:
        returns[crash_start: crash_start + 20, 0] -= 0.035  # momentum crash
        returns[crash_start: crash_start + 20, 2] += 0.010  # quality defensive

    df = pd.DataFrame(returns, index=base_dates, columns=strategies)
    return df


def run_backtest(
    config: dict,
    mode: AblationMode,
    split_name: str = "Walk-forward 1",
    allow_held_out: bool = False,
) -> None:
    strategy_returns = load_strategy_returns(config)
    allocator = build_allocator(config)
    interpreter = RegimeInterpreter(config["llm"])

    wf_manager = WalkForwardManager(strategy_returns.index, config["walk_forward"])
    schedule = wf_manager.build_schedule()
    schedule.print_schedule()

    backtester = Backtester(strategy_returns, allocator, interpreter, config)

    # Select split
    if split_name == "held_out":
        split = schedule.get_held_out(allow=allow_held_out)
    else:
        all_wf = {w.name: w for w in schedule.walk_forward_windows}
        split = all_wf.get(split_name, schedule.walk_forward_windows[0])

    print(f"\nRunning ablation '{mode}' on {split}")
    result = backtester.run_split(split, mode=mode)
    result.performance.print(title=f"{mode.upper()} | {split.name}")

    # Benchmarks
    print("\n--- Benchmarks ---")
    benchmarks = backtester.run_benchmarks(split)
    for name, perf in benchmarks.items():
        perf.print(title=name)

    # Attribution consistency (RQ3)
    eval_cfg = config.get("evaluation", {})
    if eval_cfg.get("attribution", {}).get("enabled", True):
        _run_attribution(result, config)


def run_all_ablations(config: dict) -> None:
    strategy_returns = load_strategy_returns(config)
    allocator = build_allocator(config)
    interpreter = RegimeInterpreter(config["llm"])

    wf_manager = WalkForwardManager(strategy_returns.index, config["walk_forward"])
    schedule = wf_manager.build_schedule()
    schedule.print_schedule()

    backtester = Backtester(strategy_returns, allocator, interpreter, config)

    for mode in ABLATION_MODES:
        for wf_split in schedule.walk_forward_windows:
            print(f"\n{'=' * 60}")
            print(f"Ablation: {mode.upper()} | {wf_split}")
            result = backtester.run_split(wf_split, mode=mode)
            result.performance.print(title=f"{mode} | {wf_split.name}")


def _run_attribution(result, config: dict) -> None:
    eval_cfg = config.get("evaluation", {}).get("attribution", {})
    threshold = eval_cfg.get("dominance_threshold", 0.50)
    n_bg = eval_cfg.get("background_n", 50)

    # Build a simple predict_fn wrapper for the heuristic allocator
    def predict_fn(X: np.ndarray) -> np.ndarray:
        out = []
        for row in X:
            regime_onehot = row[15:20]
            regime_idx = int(regime_onehot.argmax()) if regime_onehot.sum() > 0 else 0
            from environment.strategy_env import PortfolioStrategyEnv
            regime_label = PortfolioStrategyEnv.REGIMES[regime_idx]
            w = MetaAllocator.heuristic_rotation(regime_label)
            out.append(w)
        return np.array(out)

    feature_names = (
        [f"{s}_{f}" for s in ["momentum", "value", "quality"] for f in ["mean_ret", "std_ret", "sharpe", "max_dd"]]
        + ["weight_momentum", "weight_value", "weight_quality"]
        + [f"regime_{r}" for r in ["TRENDING-LOWVOL", "STRESS-DRAWDOWN", "RECOVERY", "SIDEWAYS-HIGHCORR", "RISK-OFF-DEFENSIVE"]]
        + ["steps_remaining"]
    )

    analyzer = AttributionAnalyzer(predict_fn, feature_names, dominance_threshold=threshold)

    # Build inputs DataFrame (weights + regime serve as proxy observations here)
    obs_df = result.weights_history.copy()
    attr_result = analyzer._fallback_heuristic(
        result.weights_history,
        result.regime_history,
        result.constraint_binding_history,
    )

    if config.get("evaluation", {}).get("regime_responsiveness", True):
        attr_result.regime_responsiveness = analyzer.compute_regime_responsiveness(
            result.weights_history, result.regime_history
        )

    attr_result.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="MARL-LLM-PM")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--mode",
        choices=["train", "backtest", "ablation", "held_out"],
        default="backtest",
    )
    parser.add_argument(
        "--ablation",
        choices=ABLATION_MODES,
        default="full",
        help="Ablation mode (ignored for --mode ablation)",
    )
    parser.add_argument(
        "--split",
        default="Walk-forward 1",
        help="Walk-forward window name (e.g. 'Walk-forward 2')",
    )
    parser.add_argument(
        "--allow-held-out",
        action="store_true",
        help="Unlock the sealed held-out set. Use only for final evaluation.",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    if args.mode == "train":
        print("Training mode: PPO learning loop — to be implemented in Phase 2.")
        print("Phase 1 uses rule-based signal agents (no training required).")

    elif args.mode == "backtest":
        run_backtest(config, args.ablation, args.split)

    elif args.mode == "ablation":
        run_all_ablations(config)

    elif args.mode == "held_out":
        if not args.allow_held_out:
            print(
                "ERROR: Held-out evaluation requires --allow-held-out flag.\n"
                "Only use this after ALL design decisions are frozen.\n"
                "This evaluation should be run exactly once."
            )
            sys.exit(1)
        run_backtest(config, args.ablation, split_name="held_out", allow_held_out=True)


if __name__ == "__main__":
    main()
