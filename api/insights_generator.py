"""
TacticsAI Insights Generator
Uses Claude API to generate tactical insights from match predictions.
"""

import os
import time
from functools import lru_cache
from typing import List, Dict, Optional
from anthropic import Anthropic
from loguru import logger

from api.models import Insight


class InsightsGenerator:
    """
    Generates tactical insights using Claude API.
    Falls back to rule-based insights if API unavailable.
    """

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = None

        if self.api_key:
            try:
                self.client = Anthropic(api_key=self.api_key)
                logger.info("InsightsGenerator initialized with Claude API")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
                self.client = None
        else:
            logger.warning("ANTHROPIC_API_KEY not set. Using fallback insights.")

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Min 1 second between requests

        # Cache for insights (lru_cache on method level)
        self._cache = {}
        self.cache_ttl = 3600  # 1 hour cache

    def generate_insights(
        self,
        team_a: str,
        formation_a: str,
        team_b: str,
        formation_b: str,
        dominance_score: float,
        confidence: float,
        battle_zones: List[Dict],
    ) -> List[Insight]:
        """
        Generate tactical insights for a matchup prediction.

        Uses Claude API if available, otherwise falls back to rules.
        Implements caching and rate limiting.

        Args:
            team_a: First team name
            formation_a: First team's formation
            team_b: Second team name
            formation_b: Second team's formation
            dominance_score: Predicted dominance (0-100)
            confidence: Model confidence (0-1)
            battle_zones: List of battle zone dicts

        Returns:
            List of Insight objects
        """
        # Check cache first
        cache_key = f"{team_a}:{formation_a}:{team_b}:{formation_b}:{dominance_score:.0f}"
        cached = self._get_cached_insights(cache_key)
        if cached:
            logger.debug(f"Returning cached insights for {cache_key}")
            return cached

        # Generate insights
        if self.client:
            insights = self._generate_llm_insights(
                team_a,
                formation_a,
                team_b,
                formation_b,
                dominance_score,
                confidence,
                battle_zones,
            )
        else:
            insights = self._generate_fallback_insights(
                team_a, formation_a, team_b, formation_b, dominance_score
            )

        # Cache results
        self._cache_insights(cache_key, insights)

        return insights

    def _get_cached_insights(self, cache_key: str) -> Optional[List[Insight]]:
        """Get cached insights if still valid."""
        if cache_key in self._cache:
            cached_time, cached_insights = self._cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_insights
            else:
                # Expired, remove from cache
                del self._cache[cache_key]
        return None

    def _cache_insights(self, cache_key: str, insights: List[Insight]):
        """Cache insights with timestamp."""
        self._cache[cache_key] = (time.time(), insights)

        # Clean old cache entries (keep max 100 entries)
        if len(self._cache) > 100:
            # Remove oldest entries
            sorted_cache = sorted(self._cache.items(), key=lambda x: x[1][0])
            for key, _ in sorted_cache[:20]:  # Remove oldest 20
                del self._cache[key]

    def _rate_limit(self):
        """Implement rate limiting for API calls."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _generate_llm_insights(
        self,
        team_a: str,
        formation_a: str,
        team_b: str,
        formation_b: str,
        dominance_score: float,
        confidence: float,
        battle_zones: List[Dict],
    ) -> List[Insight]:
        """Generate insights using Claude API with retry logic."""

        # Format battle zones for context
        zones_summary = "\n".join(
            [
                f"- {z['zone']}: {z['advantage']} has advantage ({z['margin']:.0f}%)"
                for z in battle_zones[:5]  # Top 5 zones
            ]
        )

        # Determine tactical advantage team
        advantage_team = team_a if dominance_score > 50 else team_b
        advantage_pct = (
            dominance_score if dominance_score > 50 else (100 - dominance_score)
        )

        prompt = f"""You are a professional football tactical analyst. Based on the following match prediction data, generate exactly 4 tactical insights. Each insight should be 1-2 sentences, specific, and actionable.

**Match:** {team_a} ({formation_a}) vs {team_b} ({formation_b})

**Prediction:**
- Overall: {advantage_team} has tactical advantage ({advantage_pct:.1f}% dominance)
- Model confidence: {confidence:.0%}

**Key Battle Zones:**
{zones_summary}

Generate 4 insights covering:
1. A formation-specific tactical observation (how the formations match up)
2. A zone-based insight (where the key battles will be)
3. A potential vulnerability or opportunity for the underdog
4. A key factor that could swing the match

Format each insight as:
[CATEGORY]: insight text

Categories should be one of: formation, tactical, historical, key_factor

Be specific to these teams and formations. Avoid generic statements."""

        # Retry logic with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Rate limit
                self._rate_limit()

                logger.info(f"Generating insights via Claude API (attempt {attempt + 1})")

                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )

                # Parse response into insights
                insights = self._parse_llm_response(response.content[0].text)

                if insights:
                    logger.info(f"Generated {len(insights)} insights via Claude API")
                    return insights
                else:
                    logger.warning("Claude API returned no valid insights")
                    raise ValueError("No valid insights parsed")

            except Exception as e:
                logger.error(f"Claude API error (attempt {attempt + 1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Final attempt failed, use fallback
                    logger.warning("All Claude API attempts failed, using fallback")
                    return self._generate_fallback_insights(
                        team_a, formation_a, team_b, formation_b, dominance_score
                    )

        # Should not reach here, but just in case
        return self._generate_fallback_insights(
            team_a, formation_a, team_b, formation_b, dominance_score
        )

    def _parse_llm_response(self, response_text: str) -> List[Insight]:
        """Parse Claude's response into Insight objects."""
        insights = []

        for line in response_text.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Parse [CATEGORY]: text format
            if ":" in line and "[" in line:
                try:
                    # Find category between brackets
                    start = line.find("[")
                    end = line.find("]")

                    if start >= 0 and end > start:
                        category = line[start + 1 : end].lower().strip()
                        text = line[end + 1 :].strip()

                        # Remove leading colon if present
                        if text.startswith(":"):
                            text = text[1:].strip()

                        # Validate category
                        valid_categories = ["formation", "tactical", "historical", "key_factor"]
                        if category not in valid_categories:
                            category = "tactical"

                        if len(text) > 10:
                            insights.append(Insight(text=text, category=category))
                except Exception as e:
                    logger.debug(f"Failed to parse line: {line} - {e}")
                    continue
            elif len(line) > 20 and not line.startswith("-"):
                # Fallback: treat as tactical insight
                insights.append(Insight(text=line, category="tactical"))

        return insights[:5]  # Max 5 insights

    def _generate_fallback_insights(
        self,
        team_a: str,
        formation_a: str,
        team_b: str,
        formation_b: str,
        dominance_score: float,
    ) -> List[Insight]:
        """Fallback rule-based insights when API unavailable."""
        logger.debug("Generating fallback rule-based insights")
        insights = []

        # Define formation characteristics
        wide_formations = {"4-3-3", "3-4-3"}
        narrow_formations = {"4-2-3-1", "4-4-2"}
        three_at_back = {"3-5-2", "3-4-3", "5-3-2"}
        four_at_back = {"4-3-3", "4-4-2", "4-2-3-1"}
        midfield_heavy = {"3-5-2", "4-3-3", "4-2-3-1"}
        two_striker = {"4-4-2", "3-5-2", "5-3-2"}
        wing_back_formations = {"3-5-2", "5-3-2", "3-4-3"}
        winger_formations = {"4-3-3", "3-4-3"}
        defensive_formations = {"5-3-2", "4-4-2"}
        attacking_formations = {"4-3-3", "3-4-3", "4-2-3-1"}

        # Formation matchup insights
        if formation_a in wide_formations and formation_b in narrow_formations:
            insights.append(
                Insight(
                    text=f"{team_a}'s {formation_a} provides width that could stretch {team_b}'s narrow {formation_b}",
                    category="formation",
                )
            )
        elif formation_b in wide_formations and formation_a in narrow_formations:
            insights.append(
                Insight(
                    text=f"{team_b}'s wide {formation_b} may exploit the flanks against {team_a}'s {formation_a}",
                    category="formation",
                )
            )

        # Defensive structure
        if formation_a in three_at_back and formation_b in four_at_back:
            insights.append(
                Insight(
                    text=f"{team_a}'s back three offers numerical advantage in build-up against {team_b}'s front line",
                    category="formation",
                )
            )

        # Midfield battle
        if formation_a in midfield_heavy and formation_b in midfield_heavy:
            insights.append(
                Insight(
                    text=f"Both teams' formations prioritize midfield control - expect a congested central battle",
                    category="formation",
                )
            )

        # Two striker systems
        if formation_a in two_striker and formation_b not in two_striker:
            insights.append(
                Insight(
                    text=f"{team_a}'s dual striker system in {formation_a} creates 2v2 situations against the center backs",
                    category="formation",
                )
            )

        # Dominance-based insights
        if dominance_score > 60:
            insights.append(
                Insight(
                    text=f"{team_a}'s tactical setup appears well-suited to exploit {team_b}'s formation weaknesses",
                    category="tactical",
                )
            )
        elif dominance_score < 40:
            insights.append(
                Insight(
                    text=f"{team_b}'s {formation_b} configuration provides structural advantages in this matchup",
                    category="tactical",
                )
            )
        else:
            insights.append(
                Insight(
                    text=f"This matchup appears tactically balanced - individual quality may be decisive",
                    category="tactical",
                )
            )

        # Wing-back vs winger matchups
        if formation_a in wing_back_formations and formation_b in winger_formations:
            insights.append(
                Insight(
                    text=f"{team_a}'s wing-backs will need to track {team_b}'s wide forwards, creating 1v1 duels",
                    category="tactical",
                )
            )

        # Counter-attack potential
        if formation_a in defensive_formations and formation_b in attacking_formations:
            insights.append(
                Insight(
                    text=f"{team_a}'s compact {formation_a} could invite pressure and create counter-attack opportunities",
                    category="tactical",
                )
            )

        return insights[:5]  # Max 5 insights
