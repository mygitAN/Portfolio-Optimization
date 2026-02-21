"""
LLM-powered market sentiment and macro reasoning using Claude.
"""

import anthropic
from datetime import datetime


SENTIMENT_PROMPT = """You are a quantitative finance analyst. Analyze current market conditions and provide:
1. Overall market sentiment (bullish/neutral/bearish) with confidence 0-1
2. Key macro themes affecting portfolio allocation
3. Asset class outlook (equities, bonds, alternatives)

Respond in JSON format:
{
  "sentiment": "bullish|neutral|bearish",
  "confidence": 0.0-1.0,
  "equity_outlook": "positive|neutral|negative",
  "bond_outlook": "positive|neutral|negative",
  "alternative_outlook": "positive|neutral|negative",
  "key_themes": ["theme1", "theme2"],
  "summary": "brief narrative"
}

Today's date: {date}
"""


class SentimentAnalyzer:
    """
    Queries Claude to generate market sentiment signals for RL agents.
    Results are cached per trading day to minimize API calls.
    """

    def __init__(self, config: dict):
        self.config = config
        self.client = anthropic.Anthropic()
        self.model = config.get("model", "claude-opus-4-6")
        self.max_tokens = config.get("max_tokens", 1024)
        self.temperature = config.get("temperature", 0.3)
        self._cache: dict = {}

    def get_context(self, date: str | None = None) -> dict:
        """
        Fetch LLM-derived sentiment context. Cached per date.

        Returns:
            dict with sentiment signals ready to be consumed by agents.
        """
        date = date or datetime.today().strftime("%Y-%m-%d")
        if date in self._cache:
            return self._cache[date]

        context = self._query_llm(date)
        self._cache[date] = context
        return context

    def _query_llm(self, date: str) -> dict:
        prompt = SENTIMENT_PROMPT.format(date=date)
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            import json
            raw = message.content[0].text
            # Strip markdown code fences if present
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            # Return neutral context on failure
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "equity_outlook": "neutral",
                "bond_outlook": "neutral",
                "alternative_outlook": "neutral",
                "key_themes": [],
                "summary": f"LLM unavailable: {e}",
            }

    def sentiment_to_vector(self, context: dict) -> list[float]:
        """Convert LLM context dict to a numeric feature vector for agents."""
        sentiment_map = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
        outlook_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        return [
            sentiment_map.get(context.get("sentiment", "neutral"), 0.0),
            context.get("confidence", 0.5),
            outlook_map.get(context.get("equity_outlook", "neutral"), 0.0),
            outlook_map.get(context.get("bond_outlook", "neutral"), 0.0),
            outlook_map.get(context.get("alternative_outlook", "neutral"), 0.0),
        ]
