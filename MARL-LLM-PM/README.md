# MARL-LLM-PM

**Designing an LLM–MARL Allocation Engine Within an Experimental Framework for Equity Strategy Rotation with Regulatory Constraint Adaptability**

> Abulele Nxitywa · Supervisor: Prof Gwetu · January 2026

---

## Overview

MARL-LLM-PM is the experimental codebase for the above thesis. It implements a modular, simulation-based framework that integrates Multi-Agent Reinforcement Learning (MARL) with a Large Language Model (LLM) bounded strictly to an interpretive role, within a constrained equity strategy allocation environment.

The portfolio is **not** a collection of asset classes. Each MARL agent represents a distinct **equity investment strategy**:

| Agent | Strategy | Economic Driver |
|---|---|---|
| `MomentumAgent` | Momentum | Persistence / under-reaction |
| `ValueAgent` | Value | Mean-reversion / mispricing |
| `QualityAgent` | Quality | Defensiveness / balance-sheet strength |

A higher-level `MetaAllocator` learns to dynamically weight these strategy agents in response to changing market regimes, operating inside a constraint wrapper that enforces mandate-style limits (e.g., no single strategy may exceed 70% weight).

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 0: Data & Features                                       │
│  Market returns · Strategy stats · Macro indicators            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  Layer 3: LLM Regime Interpreter          (interpretive only)   │
│  Input: structured numeric prompt                               │
│  Output: regime_label ∈ {TRENDING-LOWVOL, STRESS-DRAWDOWN,     │
│          RECOVERY, SIDEWAYS-HIGHCORR, RISK-OFF-DEFENSIVE}       │
│  + short explanation grounded in numeric inputs only            │
│  ✗ No allocations · ✗ No reward modification · ✗ No forecasts  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ regime features
┌───────────────────────────▼─────────────────────────────────────┐
│  Layer 1: Strategy Environment           (PortfolioStrategyEnv) │
│  State s_t · Action a_t (strategy weights) · Reward r_t         │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Layer 4: Constraint Wrapper                             │   │
│  │  Action masks / penalty terms                            │   │
│  │  Modes: UNCONSTRAINED | MANDATE-CONSTRAINED              │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  Layer 2: MARL                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Momentum   │  │    Value    │  │   Quality   │  Agents     │
│  │   Agent     │  │   Agent     │  │   Agent     │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         └────────────────▼────────────────┘                     │
│                   MetaAllocator                                 │
│                 (final strategy weights)                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  Training & Backtesting                                         │
│  Walk-forward splits · Regime-based tests                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  Evaluation & Reporting                                         │
│  Performance · Stability · Attribution consistency (SHAP)       │
│  Ablations: LLM off/on · Constraints off/on                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Research Questions

1. **Regime sensitivity** — Does MARL improve regime sensitivity for equity strategy rotation vs benchmarks (1/N, vol-scaled, heuristic rotations) under comparable transaction costs?

2. **Constraint impact** — How do mandate-style constraints affect learning stability, allocation behaviour, and out-of-sample performance vs the unconstrained baseline?

3. **LLM interpretability value** — Does a bounded LLM regime interpreter improve *attribution consistency* and regime responsiveness vs a numeric-only baseline?
   - **Attribution consistency**: fraction of rebalance dates where a single dominant source (strategy signal, constraint-binding event, or regime label change) accounts for >50% of the weight change (SHAP-based).

---

## LLM Regime Interpreter — Design Constraints

The LLM functions **exclusively** as an interpretive component:

- **Input**: structured numeric prompt (trailing returns, realised vol, drawdowns, dispersion, correlation)
- **Output**: one of exactly five fixed regime labels + a short explanation grounded only in the numeric inputs
- **Regime vocabulary** (fixed, closed — no additions during experiment):
  - `TRENDING-LOWVOL`
  - `STRESS-DRAWDOWN`
  - `RECOVERY`
  - `SIDEWAYS-HIGHCORR`
  - `RISK-OFF-DEFENSIVE`
- Any output outside this vocabulary is **rejected** and replaced by a volatility-quantile fallback classifier
- The LLM does **not** generate allocations, modify rewards, access news/text, or enforce constraints

---

## Project Structure

```
MARL-LLM-PM/
├── agents/
│   ├── base_agent.py          # Abstract agent interface
│   ├── momentum_agent.py      # Momentum strategy agent
│   ├── value_agent.py         # Value strategy agent
│   ├── quality_agent.py       # Quality strategy agent
│   └── meta_allocator.py      # Hierarchical meta-allocator (final weights)
├── environment/
│   ├── strategy_env.py        # Gymnasium-compatible strategy allocation env
│   └── constraint_wrapper.py  # CMDP constraint layer (masks + penalties)
├── llm/
│   └── regime_interpreter.py  # Bounded LLM regime classifier (5-label vocab)
├── data/
│   └── market_data.py         # Data loader (JSE / synthetic)
├── evaluation/
│   ├── walk_forward.py        # Walk-forward split manager (60/20/20)
│   ├── attribution.py         # SHAP-based attribution consistency
│   └── metrics.py             # Sharpe, Sortino, max-drawdown, Calmar
├── backtesting/
│   └── backtest.py            # Backtesting engine
├── configs/
│   └── default.yaml           # Full experiment configuration
├── notebooks/                 # Research & analysis notebooks
├── tests/                     # pytest suite
├── requirements.txt
└── main.py                    # CLI entry point
```

---

## Evaluation Protocol

| Split | Proportion | Purpose |
|---|---|---|
| Training | 60% | Train MARL agents from scratch |
| Validation | 20% | Hyperparameter selection / early stopping |
| Walk-forward test | 20% (6-month rolls) | Sequential out-of-sample evaluation |
| **Held-out** | **Last 12 months** | **Evaluated exactly once — sealed** |

> **Integrity commitment**: the held-out set is accessed only after the full pipeline (architecture, hyperparameters, reward spec, constraint mechanism, LLM prompt template) is frozen. No adjustments are made after first evaluation.

---

## Ablation Modes

| Mode | LLM | Constraints | Purpose |
|---|---|---|---|
| `baseline` | off | off | Pure MARL, no LLM, no constraints |
| `constrained` | off | on | Constraint impact only |
| `llm_only` | on | off | LLM value without constraints |
| `full` | on | on | Full proposed system |

---

## Benchmarks

- `1/N` equal-weight across strategies
- Volatility-scaled momentum baseline
- Heuristic rotation (rule-based regime switch)

---

## Getting Started

```bash
git clone https://github.com/mygitAN/MARL-LLM-PM.git
cd MARL-LLM-PM
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...

# Train full system
python main.py --config configs/default.yaml --mode train --ablation full

# Backtest a specific ablation
python main.py --config configs/default.yaml --mode backtest --ablation constrained

# Run all ablations sequentially
python main.py --config configs/default.yaml --mode ablation
```

---

## License

MIT — academic research use. Not financial advice.
