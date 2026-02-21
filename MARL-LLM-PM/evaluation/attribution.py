"""
SHAP-based attribution consistency.

Implements the interpretability metric from Research Question 3:

  Attribution consistency = fraction of rebalance dates at which a single
  dominant attribution source (strategy signal, constraint-binding event,
  or regime label change) accounts for >50% of the weight change,
  measured using SHAP values computed over the meta-allocator's inputs.

  A higher score indicates allocation decisions are more attributable
  to identifiable, inspectable causes.

Regime responsiveness:
  Correlation between LLM-generated regime transitions and allocation
  weight shifts. Compared between LLM-on and numeric-only conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class AttributionResult:
    consistency_score: float          # fraction of dates with dominant source > 50%
    dominant_source_counts: dict[str, int]  # per-source count
    n_rebalance_dates: int
    regime_responsiveness: float | None = None   # correlation metric

    def print(self) -> None:
        print("\n=== Attribution Consistency ===")
        print(f"  Consistency score:    {self.consistency_score:.4f}")
        print(f"  Rebalance dates:      {self.n_rebalance_dates}")
        print("  Dominant sources:")
        for src, cnt in self.dominant_source_counts.items():
            pct = cnt / max(self.n_rebalance_dates, 1) * 100
            print(f"    {src:30s}: {cnt:4d}  ({pct:.1f}%)")
        if self.regime_responsiveness is not None:
            print(f"  Regime responsiveness: {self.regime_responsiveness:.4f}")
        print("=" * 34)


class AttributionAnalyzer:
    """
    Computes SHAP-based attribution for MetaAllocator decisions.

    Args:
        predict_fn: callable that maps input_array (n_samples, n_features)
                    → weight_array (n_samples, n_strategies). This wraps
                    the meta-allocator's forward pass.
        feature_names: list of input feature names (must align with columns
                       of the observation matrix passed to compute()).
        dominance_threshold: fraction above which a single source is deemed
                             dominant (default 0.50, per thesis spec).
    """

    SOURCE_STRATEGY_SIGNAL = "strategy_signal"
    SOURCE_CONSTRAINT      = "constraint_binding"
    SOURCE_REGIME_CHANGE   = "regime_label_change"
    SOURCE_MIXED           = "mixed"

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        feature_names: list[str],
        dominance_threshold: float = 0.50,
    ):
        self.predict_fn = predict_fn
        self.feature_names = feature_names
        self.dominance_threshold = dominance_threshold

    def compute(
        self,
        observations: pd.DataFrame,
        weights_history: pd.DataFrame,
        regime_history: pd.Series,
        constraint_binding_history: pd.Series,
        background_n: int = 50,
    ) -> AttributionResult:
        """
        Compute attribution consistency over all rebalance dates.

        Args:
            observations: DataFrame (dates × features) of meta-allocator inputs.
            weights_history: DataFrame (dates × strategies) of realised weights.
            regime_history: Series (dates) of regime labels.
            constraint_binding_history: Series (dates) bool — was any
                constraint active at this date?
            background_n: number of background samples for SHAP kernel explainer.

        Returns:
            AttributionResult with consistency score and source breakdown.
        """
        try:
            import shap
        except ImportError:
            return self._fallback_heuristic(
                weights_history, regime_history, constraint_binding_history
            )

        X = observations.values
        background = X[np.random.choice(len(X), min(background_n, len(X)), replace=False)]
        explainer = shap.KernelExplainer(self.predict_fn, background)

        dominant_counts = {
            self.SOURCE_STRATEGY_SIGNAL: 0,
            self.SOURCE_CONSTRAINT:      0,
            self.SOURCE_REGIME_CHANGE:   0,
            self.SOURCE_MIXED:           0,
        }
        n_dominant = 0
        dates = observations.index

        # Identify feature groups
        strategy_feature_mask = self._get_strategy_feature_mask()
        regime_feature_mask = self._get_regime_feature_mask()

        for i in range(1, len(dates)):
            date = dates[i]
            weight_change = float(
                np.abs(weights_history.iloc[i].values - weights_history.iloc[i - 1].values).sum()
            )
            if weight_change < 1e-4:
                continue   # not a meaningful rebalance event

            shap_vals = explainer.shap_values(X[i: i + 1], silent=True)
            # shap_vals: list of (1, n_features) per output, or (1, n_features) if single output
            if isinstance(shap_vals, list):
                abs_shap = np.abs(np.array(shap_vals)).mean(axis=0).squeeze()
            else:
                abs_shap = np.abs(shap_vals).squeeze()

            total_shap = abs_shap.sum() + 1e-10

            # Attribute to sources
            signal_share = abs_shap[strategy_feature_mask].sum() / total_shap
            regime_share = abs_shap[regime_feature_mask].sum() / total_shap

            regime_changed = (
                i > 0 and regime_history.iloc[i] != regime_history.iloc[i - 1]
            )
            constraint_active = bool(constraint_binding_history.get(date, False))

            # Determine dominant source
            source = self._dominant_source(
                signal_share,
                regime_share,
                constraint_active,
                regime_changed,
            )

            dominant_counts[source] += 1
            if source != self.SOURCE_MIXED:
                n_dominant += 1

        n_total = sum(dominant_counts.values())
        consistency = n_dominant / max(n_total, 1)

        return AttributionResult(
            consistency_score=consistency,
            dominant_source_counts=dominant_counts,
            n_rebalance_dates=n_total,
        )

    def compute_regime_responsiveness(
        self,
        weights_history: pd.DataFrame,
        regime_history: pd.Series,
    ) -> float:
        """
        Pearson correlation between regime transitions and weight shifts.

        A regime transition = 1 when regime label changes, else 0.
        Weight shift = sum of absolute weight changes.
        """
        regimes = regime_history.reindex(weights_history.index)
        transitions = (regimes != regimes.shift(1)).astype(float)

        weight_changes = weights_history.diff().abs().sum(axis=1)

        aligned = pd.concat([transitions, weight_changes], axis=1).dropna()
        if len(aligned) < 5:
            return float("nan")

        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        return float(corr)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_strategy_feature_mask(self) -> np.ndarray:
        """Mask for strategy-signal features (mean_ret, std, sharpe, dd × 3)."""
        mask = np.zeros(len(self.feature_names), dtype=bool)
        for i, name in enumerate(self.feature_names):
            if any(s in name for s in ["momentum", "value", "quality", "sharpe", "mean_ret", "std_ret", "max_dd"]):
                mask[i] = True
        return mask

    def _get_regime_feature_mask(self) -> np.ndarray:
        """Mask for regime one-hot features."""
        mask = np.zeros(len(self.feature_names), dtype=bool)
        for i, name in enumerate(self.feature_names):
            if "regime" in name or any(r in name for r in [
                "TRENDING", "STRESS", "RECOVERY", "SIDEWAYS", "RISK-OFF"
            ]):
                mask[i] = True
        return mask

    def _dominant_source(
        self,
        signal_share: float,
        regime_share: float,
        constraint_active: bool,
        regime_changed: bool,
    ) -> str:
        t = self.dominance_threshold
        if signal_share >= t:
            return self.SOURCE_STRATEGY_SIGNAL
        if regime_changed and regime_share >= t:
            return self.SOURCE_REGIME_CHANGE
        if constraint_active and (1 - signal_share - regime_share) >= t:
            return self.SOURCE_CONSTRAINT
        return self.SOURCE_MIXED

    def _fallback_heuristic(
        self,
        weights_history: pd.DataFrame,
        regime_history: pd.Series,
        constraint_binding_history: pd.Series,
    ) -> AttributionResult:
        """
        Heuristic attribution when shap is not installed.
        Uses regime transitions and constraint flags as proxies.
        """
        dominant_counts = {
            self.SOURCE_STRATEGY_SIGNAL: 0,
            self.SOURCE_CONSTRAINT:      0,
            self.SOURCE_REGIME_CHANGE:   0,
            self.SOURCE_MIXED:           0,
        }
        dates = weights_history.index
        n_dominant = 0

        for i in range(1, len(dates)):
            date = dates[i]
            weight_change = float(
                np.abs(weights_history.iloc[i].values - weights_history.iloc[i - 1].values).sum()
            )
            if weight_change < 1e-4:
                continue

            regime_changed = regime_history.iloc[i] != regime_history.iloc[i - 1]
            constraint_active = bool(constraint_binding_history.get(date, False))

            if regime_changed:
                dominant_counts[self.SOURCE_REGIME_CHANGE] += 1
                n_dominant += 1
            elif constraint_active:
                dominant_counts[self.SOURCE_CONSTRAINT] += 1
                n_dominant += 1
            else:
                dominant_counts[self.SOURCE_STRATEGY_SIGNAL] += 1
                n_dominant += 1

            dominant_counts[self.SOURCE_MIXED] = (
                sum(dominant_counts.values()) - n_dominant
            )

        n_total = sum(dominant_counts.values())
        consistency = n_dominant / max(n_total, 1)
        return AttributionResult(
            consistency_score=consistency,
            dominant_source_counts=dominant_counts,
            n_rebalance_dates=n_total,
        )
