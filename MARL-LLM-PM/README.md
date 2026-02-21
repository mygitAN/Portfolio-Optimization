# MARL-LLM-PM

**Multi-Agent Reinforcement Learning with Large Language Models for Portfolio Management**

## Overview

MARL-LLM-PM is a research framework that combines Multi-Agent Reinforcement Learning (MARL) with Large Language Models (LLMs) for intelligent and adaptive portfolio management.

Traditional portfolio management relies on static optimization (e.g., Markowitz Efficient Frontier). This framework augments RL-based trading agents with LLM-driven market reasoning to enable dynamic, context-aware allocation strategies.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  MARL Environment                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Agent 1  │  │ Agent 2  │  │    Agent N       │  │
│  │(Equities)│  │ (Bonds)  │  │  (Alternatives)  │  │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────┘  │
│       │              │                 │             │
│  ┌────▼──────────────▼─────────────────▼──────────┐ │
│  │              LLM Reasoning Layer               │ │
│  │  (Market Sentiment · News · Macro Analysis)   │ │
│  └────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

## Features

- **Multi-Agent RL**: Independent agents per asset class with cooperative/competitive dynamics
- **LLM Integration**: Claude-powered market sentiment analysis and macro reasoning
- **Dynamic Allocation**: Real-time portfolio rebalancing based on agent consensus
- **Backtesting**: Full historical simulation with Sharpe, Sortino, and max-drawdown metrics
- **Risk Management**: Per-agent risk budgets and portfolio-level constraints

## Project Structure

```
MARL-LLM-PM/
├── agents/              # RL agent implementations
│   ├── base_agent.py
│   ├── equity_agent.py
│   ├── bond_agent.py
│   └── coordinator.py
├── environment/         # Multi-asset trading environment
│   ├── portfolio_env.py
│   └── market_simulator.py
├── llm/                 # LLM integration layer
│   ├── sentiment_analyzer.py
│   └── macro_reasoner.py
├── data/                # Data loaders and processors
│   ├── market_data.py
│   └── news_fetcher.py
├── backtesting/         # Backtesting engine
│   └── backtest.py
├── configs/             # Experiment configurations
│   └── default.yaml
├── notebooks/           # Research notebooks
├── tests/               # Unit and integration tests
├── requirements.txt
└── main.py
```

## Getting Started

### Prerequisites

- Python 3.10+
- Anthropic API key (for LLM features)

### Installation

```bash
git clone https://github.com/mygitAN/MARL-LLM-PM.git
cd MARL-LLM-PM
pip install -r requirements.txt
```

### Configuration

```bash
cp configs/default.yaml configs/local.yaml
# Edit local.yaml with your API keys and preferences
```

### Run Training

```bash
python main.py --config configs/local.yaml --mode train
```

### Run Backtesting

```bash
python main.py --config configs/local.yaml --mode backtest --start 2020-01-01 --end 2024-12-31
```

## Research Background

This project builds on:
- **MARL for Finance**: Multi-agent cooperation in portfolio optimization
- **LLM-augmented RL**: Incorporating language model reasoning into policy learning
- **Risk-aware RL**: Reward shaping for Sharpe/CVaR optimization

## License

MIT
