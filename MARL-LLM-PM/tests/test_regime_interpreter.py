"""
Tests for the bounded LLM regime interpreter.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm.regime_interpreter import RegimeInterpreter, REGIME_VOCAB


def _synthetic_returns(n: int = 60) -> np.ndarray:
    np.random.seed(1)
    return np.random.normal(0.0003, 0.01, (n, 3))


def test_fallback_always_in_vocab():
    """Fallback classifier must always return a label within the fixed vocabulary."""
    for seed in range(20):
        np.random.seed(seed)
        returns = np.random.normal(0, 0.02, (30, 3))
        label = RegimeInterpreter._fallback_classifier(returns)
        assert label in REGIME_VOCAB, f"Fallback returned unknown label: {label!r}"


def test_fallback_stress_regime():
    """Large drawdown should trigger STRESS-DRAWDOWN."""
    # Simulate crash: sustained negative returns
    returns = np.full((30, 3), -0.04)
    label = RegimeInterpreter._fallback_classifier(returns)
    assert label == "STRESS-DRAWDOWN"


def test_fallback_trending():
    """Low vol, positive returns should be TRENDING-LOWVOL."""
    returns = np.full((30, 3), 0.003)   # consistent small positive, near-zero vol
    label = RegimeInterpreter._fallback_classifier(returns)
    assert label == "TRENDING-LOWVOL"


def test_classify_disabled_llm():
    """When LLM is disabled, classify uses the fallback and returns valid label."""
    config = {"enabled": False, "model": "claude-opus-4-6", "max_tokens": 256, "temperature": 0.1}
    interpreter = RegimeInterpreter(config)
    returns = _synthetic_returns()
    label, explanation = interpreter.classify(returns, date="2024-01-01")
    assert label in REGIME_VOCAB
    assert isinstance(explanation, str)


def test_classify_caches_result():
    """Same date should return cached result without re-querying."""
    config = {"enabled": False, "model": "claude-opus-4-6", "max_tokens": 256, "temperature": 0.1}
    interpreter = RegimeInterpreter(config)
    returns = _synthetic_returns()
    result1 = interpreter.classify(returns, date="2024-06-01")
    result2 = interpreter.classify(returns, date="2024-06-01")
    assert result1 == result2


def test_regime_to_onehot():
    config = {"enabled": False}
    interpreter = RegimeInterpreter(config)
    for label in REGIME_VOCAB:
        vec = interpreter.regime_to_onehot(label)
        assert vec.sum() == 1.0
        assert vec.shape == (5,)


def test_invalid_label_rejected():
    """Parsing a response with an invalid label should fall back gracefully."""
    config = {"enabled": False}
    interpreter = RegimeInterpreter(config)
    returns = _synthetic_returns()
    bad_raw = '{"regime_label": "BULL-MARKET", "explanation": "made up label"}'
    label, explanation = interpreter._parse_and_validate(bad_raw, returns)
    assert label in REGIME_VOCAB   # must fall back to a valid label
