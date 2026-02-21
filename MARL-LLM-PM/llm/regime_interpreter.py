"""
Bounded LLM regime interpreter.

Design constraints (from thesis specification):
  - Input:  structured numeric prompt ONLY — no unstructured text, news, or
            external information sources.
  - Output: one of exactly five fixed regime labels + a short explanation
            grounded only in the numeric inputs.
  - Regime vocabulary is FIXED AND CLOSED. Any output outside this
    vocabulary is rejected and replaced by the quantitative fallback.
  - The LLM does NOT generate allocations, modify rewards, access forecasts,
    or enforce constraints.
  - The regime label (+ optional explanation embedding) is added to the
    environment state as a feature, conditioning the strategy agents and
    meta-allocator.

Fallback classifier:
  When LLM output fails validation, a rule-based volatility-quantile
  classifier assigns a regime from the same fixed vocabulary, ensuring
  the state always contains a valid regime feature.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

import numpy as np


# ---------------------------------------------------------------------------
# Fixed regime vocabulary (must match PortfolioStrategyEnv.REGIMES)
# ---------------------------------------------------------------------------
REGIME_VOCAB = frozenset([
    "TRENDING-LOWVOL",
    "STRESS-DRAWDOWN",
    "RECOVERY",
    "SIDEWAYS-HIGHCORR",
    "RISK-OFF-DEFENSIVE",
])

_SYSTEM_PROMPT = """\
You are a quantitative market analyst. You will be given a structured summary \
of recent market statistics. Based ONLY on the numbers provided — no external \
knowledge, no forecasts, no opinions — classify the current market regime.

You MUST respond with valid JSON in exactly this format:
{
  "regime_label": "<ONE OF THE FIVE LABELS>",
  "explanation": "<one sentence referencing specific numbers from the input>"
}

The regime_label MUST be exactly one of:
  TRENDING-LOWVOL
  STRESS-DRAWDOWN
  RECOVERY
  SIDEWAYS-HIGHCORR
  RISK-OFF-DEFENSIVE

Do not output any text outside the JSON object.
"""

_USER_TEMPLATE = """\
Market statistics summary ({date}):

Strategy trailing returns (20-day):
  Momentum:  {mom_ret_20:.4f}
  Value:     {val_ret_20:.4f}
  Quality:   {qua_ret_20:.4f}

Strategy trailing returns (60-day):
  Momentum:  {mom_ret_60:.4f}
  Value:     {val_ret_60:.4f}
  Quality:   {qua_ret_60:.4f}

Realised volatility (20-day annualised):
  Momentum:  {mom_vol:.4f}
  Value:     {val_vol:.4f}
  Quality:   {qua_vol:.4f}

Max drawdown from peak (rolling 60-day):
  Momentum:  {mom_dd:.4f}
  Value:     {val_dd:.4f}
  Quality:   {qua_dd:.4f}

Strategy return dispersion (cross-sectional std, 20-day): {dispersion:.4f}
Average pairwise correlation (20-day): {avg_corr:.4f}

Classify the current market regime using only the numbers above.
"""


class RegimeInterpreter:
    """
    Bounded LLM regime classifier.

    Args:
        config: llm section of experiment config, e.g.:
            {
              "enabled": true,
              "model": "claude-opus-4-6",
              "max_tokens": 256,
              "temperature": 0.1,
              "update_frequency": "weekly"   # or "monthly" / integer days
            }
    """

    def __init__(self, config: dict):
        self.config = config
        self.enabled: bool = config.get("enabled", True)
        self.model: str = config.get("model", "claude-opus-4-6")
        self.max_tokens: int = config.get("max_tokens", 256)
        self.temperature: float = config.get("temperature", 0.1)
        self._cache: dict[str, tuple[str, str]] = {}  # date → (label, explanation)
        self._client = None  # lazy-init to avoid import errors when LLM disabled

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def classify(
        self,
        strategy_returns: np.ndarray,
        date: str | None = None,
    ) -> tuple[str, str]:
        """
        Classify the current market regime.

        Args:
            strategy_returns: np.ndarray of shape (T, 3) — recent daily returns
                              for [momentum, value, quality].
            date: date string for caching (YYYY-MM-DD). Defaults to today.

        Returns:
            (regime_label, explanation) — both from the fixed vocabulary.
        """
        date = date or datetime.today().strftime("%Y-%m-%d")
        if date in self._cache:
            return self._cache[date]

        if not self.enabled:
            label = self._fallback_classifier(strategy_returns)
            result = (label, "LLM disabled — quantitative fallback applied.")
        else:
            prompt_vars = self._build_prompt_vars(strategy_returns, date)
            label, explanation = self._query_llm(prompt_vars)
            result = (label, explanation)

        self._cache[date] = result
        return result

    def regime_to_onehot(self, label: str) -> np.ndarray:
        """Convert regime label to a 5-dim one-hot feature vector."""
        vocab_list = sorted(REGIME_VOCAB)
        vec = np.zeros(len(vocab_list), dtype=np.float32)
        if label in vocab_list:
            vec[vocab_list.index(label)] = 1.0
        return vec

    # ------------------------------------------------------------------
    # LLM query
    # ------------------------------------------------------------------

    def _query_llm(self, prompt_vars: dict) -> tuple[str, str]:
        try:
            import anthropic
            if self._client is None:
                self._client = anthropic.Anthropic()

            user_msg = _USER_TEMPLATE.format(**prompt_vars)
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()
            return self._parse_and_validate(raw, prompt_vars["returns_array"])
        except Exception as e:
            fallback = self._fallback_classifier(prompt_vars["returns_array"])
            return fallback, f"LLM error ({e}) — quantitative fallback applied."

    def _parse_and_validate(
        self, raw: str, returns_array: np.ndarray
    ) -> tuple[str, str]:
        """
        Parse JSON response and validate regime label is in fixed vocabulary.
        On any failure, fall back to quantitative classifier.
        """
        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        try:
            parsed = json.loads(raw)
            label = parsed.get("regime_label", "").strip().upper()
            explanation = parsed.get("explanation", "").strip()
            if label in REGIME_VOCAB:
                return label, explanation
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback
        fallback = self._fallback_classifier(returns_array)
        return fallback, "LLM output failed validation — quantitative fallback applied."

    # ------------------------------------------------------------------
    # Quantitative fallback classifier
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_classifier(strategy_returns: np.ndarray) -> str:
        """
        Rule-based regime classifier using volatility quantiles and
        return momentum. Mirrors the five-label vocabulary.

        Uses the most recent 20 days (or all available if < 20).
        """
        window = strategy_returns[-20:] if len(strategy_returns) >= 20 else strategy_returns
        if len(window) == 0:
            return "TRENDING-LOWVOL"

        # Annualised cross-strategy average volatility
        avg_vol = float(window.std(axis=0).mean() * np.sqrt(252))

        # Average 20-day return across strategies
        avg_ret = float(window.mean(axis=0).mean() * 20)

        # Max drawdown proxy: worst single-strategy 20-day cumulative return
        cum = (1 + window).cumprod(axis=0)
        dd = float((cum / cum.cummax(axis=0) - 1).min())

        # Average pairwise correlation
        if window.shape[1] > 1 and window.shape[0] > 2:
            corr_mat = np.corrcoef(window.T)
            mask = ~np.eye(corr_mat.shape[0], dtype=bool)
            avg_corr = float(corr_mat[mask].mean())
        else:
            avg_corr = 0.0

        # Classification rules (prioritised)
        if dd < -0.10:
            return "STRESS-DRAWDOWN"
        if avg_vol > 0.25 and avg_ret < 0.0:
            return "RISK-OFF-DEFENSIVE"
        if avg_ret > 0.03 and avg_vol < 0.15:
            return "TRENDING-LOWVOL"
        if avg_ret > 0.01 and dd > -0.05:
            return "RECOVERY"
        if avg_corr > 0.70:
            return "SIDEWAYS-HIGHCORR"
        return "TRENDING-LOWVOL"

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt_vars(
        strategy_returns: np.ndarray, date: str
    ) -> dict:
        """
        Extract numeric features from strategy_returns for the prompt.
        strategy_returns: (T, 3) array — columns: [momentum, value, quality]
        """
        r = strategy_returns
        w20 = r[-20:] if len(r) >= 20 else r
        w60 = r[-60:] if len(r) >= 60 else r

        def _ret(arr, col): return float(arr[:, col].sum())
        def _vol(arr, col): return float(arr[:, col].std() * np.sqrt(252))
        def _dd(arr, col):
            cum = (1 + arr[:, col]).cumprod()
            return float((cum / cum.cummax() - 1).min())

        cross_std = float(w20.mean(axis=0).std())
        corr = np.corrcoef(w20.T) if w20.shape[0] > 2 else np.eye(3)
        mask = ~np.eye(3, dtype=bool)
        avg_corr = float(corr[mask].mean()) if w20.shape[0] > 2 else 0.0

        return {
            "date": date,
            "mom_ret_20": _ret(w20, 0), "val_ret_20": _ret(w20, 1), "qua_ret_20": _ret(w20, 2),
            "mom_ret_60": _ret(w60, 0), "val_ret_60": _ret(w60, 1), "qua_ret_60": _ret(w60, 2),
            "mom_vol": _vol(w20, 0),    "val_vol": _vol(w20, 1),    "qua_vol": _vol(w20, 2),
            "mom_dd":  _dd(w60, 0),     "val_dd":  _dd(w60, 1),     "qua_dd":  _dd(w60, 2),
            "dispersion": cross_std,
            "avg_corr": avg_corr,
            "returns_array": r,         # passed to fallback if needed
        }
