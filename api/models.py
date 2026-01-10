"""
TacticsAI API Models
Pydantic v2 models for request/response validation.
"""

import uuid
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# Import configuration for validation
import sys
from pathlib import Path

# Ensure backend is importable
backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from data.config import PREMIER_LEAGUE_TEAMS, FORMATIONS, list_teams, list_formations


# =============================================================================
# Constants
# =============================================================================

VALID_TEAMS = set(PREMIER_LEAGUE_TEAMS.keys())
VALID_FORMATIONS = set(FORMATIONS.keys())

# Zone names for battle zone predictions (3x3 grid)
ZONE_NAMES = [
    "left_defense",
    "central_defense",
    "right_defense",
    "left_midfield",
    "central_midfield",
    "right_midfield",
    "left_attack",
    "central_attack",
    "right_attack",
]


# =============================================================================
# Request Models
# =============================================================================

class TeamFormation(BaseModel):
    """
    A team's formation configuration.
    
    Attributes:
        team: Premier League team name (must be valid)
        formation: Tactical formation (e.g., "4-3-3", "4-4-2")
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "team": "Arsenal",
                    "formation": "4-3-3"
                },
                {
                    "team": "Manchester City",
                    "formation": "4-2-3-1"
                }
            ]
        }
    )
    
    team: str = Field(
        ...,
        description="Premier League team name",
        examples=["Arsenal", "Liverpool", "Manchester City"],
    )
    formation: str = Field(
        ...,
        description="Tactical formation",
        examples=["4-3-3", "4-4-2", "4-2-3-1"],
    )
    
    @field_validator("team")
    @classmethod
    def validate_team(cls, v: str) -> str:
        """Validate team name is in Premier League."""
        # Try exact match first
        if v in VALID_TEAMS:
            return v
        
        # Try case-insensitive match
        v_lower = v.lower()
        for team in VALID_TEAMS:
            if team.lower() == v_lower:
                return team
        
        # Try partial match
        for team in VALID_TEAMS:
            if v_lower in team.lower() or team.lower() in v_lower:
                return team
        
        valid_teams_str = ", ".join(sorted(VALID_TEAMS))
        raise ValueError(
            f"Invalid team: '{v}'. Must be one of: {valid_teams_str}"
        )
    
    @field_validator("formation")
    @classmethod
    def validate_formation(cls, v: str) -> str:
        """Validate formation is supported."""
        # Normalize format (handle variations like "4-3-3" or "433")
        normalized = v.replace(" ", "").upper()
        
        # Check exact match
        if v in VALID_FORMATIONS:
            return v
        
        # Try with hyphens
        if len(normalized) >= 3 and normalized.isdigit():
            # Convert "433" to "4-3-3"
            hyphenated = "-".join(normalized)
            if hyphenated in VALID_FORMATIONS:
                return hyphenated
        
        valid_formations_str = ", ".join(sorted(VALID_FORMATIONS))
        raise ValueError(
            f"Invalid formation: '{v}'. Must be one of: {valid_formations_str}"
        )


class PredictionRequest(BaseModel):
    """
    Request for a tactical matchup prediction.
    
    Attributes:
        team_a: First team's configuration (typically home team)
        team_b: Second team's configuration (typically away team)
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "team_a": {
                        "team": "Arsenal",
                        "formation": "4-3-3"
                    },
                    "team_b": {
                        "team": "Chelsea",
                        "formation": "4-2-3-1"
                    }
                },
                {
                    "team_a": {
                        "team": "Liverpool",
                        "formation": "4-3-3"
                    },
                    "team_b": {
                        "team": "Manchester City",
                        "formation": "4-3-3"
                    }
                }
            ]
        }
    )
    
    team_a: TeamFormation = Field(
        ...,
        description="First team (home team) configuration",
    )
    team_b: TeamFormation = Field(
        ...,
        description="Second team (away team) configuration",
    )
    
    @model_validator(mode="after")
    def validate_different_teams(self) -> "PredictionRequest":
        """Ensure teams are different."""
        if self.team_a.team == self.team_b.team:
            raise ValueError("Teams must be different. Cannot predict a team vs itself.")
        return self


# =============================================================================
# Response Models
# =============================================================================

class BattleZone(BaseModel):
    """
    Analysis of a pitch zone in the tactical matchup.
    
    Attributes:
        zone: Name of the pitch zone (e.g., "central_midfield")
        advantage: Name of the team with tactical advantage in this zone
        margin: Advantage margin as percentage (0-100)
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "zone": "central_midfield",
                    "advantage": "Arsenal",
                    "margin": 65.5
                },
                {
                    "zone": "right_attack",
                    "advantage": "Chelsea",
                    "margin": 52.3
                }
            ]
        }
    )
    
    zone: str = Field(
        ...,
        description="Pitch zone identifier",
        examples=["left_defense", "central_midfield", "right_attack"],
    )
    advantage: str = Field(
        ...,
        description="Team with advantage in this zone",
        examples=["Arsenal", "Liverpool"],
    )
    margin: float = Field(
        ...,
        ge=0,
        le=100,
        description="Advantage margin (0-100 percentage)",
        examples=[65.5, 52.3, 78.2],
    )
    
    @field_validator("zone")
    @classmethod
    def validate_zone(cls, v: str) -> str:
        """Validate zone name."""
        v_normalized = v.lower().replace(" ", "_")
        if v_normalized in ZONE_NAMES:
            return v_normalized
        
        # Allow the original value if it's descriptive
        return v


class Insight(BaseModel):
    """
    Tactical insight from the prediction analysis.
    
    Attributes:
        text: Human-readable insight text
        category: Category of the insight
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "Arsenal's 4-3-3 provides width that could exploit Chelsea's narrow midfield",
                    "category": "formation"
                },
                {
                    "text": "Historical data shows 4-3-3 has a 58% win rate against 4-2-3-1",
                    "category": "historical"
                },
                {
                    "text": "The high defensive line creates space for counter-attacks",
                    "category": "tactical"
                }
            ]
        }
    )
    
    text: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Insight text explaining a tactical observation",
    )
    category: Literal["formation", "tactical", "historical"] = Field(
        ...,
        description="Category of insight",
    )


class PredictionResponse(BaseModel):
    """
    Complete prediction response for a tactical matchup.
    
    Attributes:
        dominance_score: Overall dominance score (0-100, >50 favors team_a)
        confidence: Model confidence in the prediction (0-1)
        favored_team: Name of the team predicted to have tactical advantage
        battle_zones: List of zone-by-zone analysis
        insights: List of tactical insights
        request_id: Unique identifier for this prediction request
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "dominance_score": 62.5,
                    "confidence": 0.78,
                    "favored_team": "Arsenal",
                    "battle_zones": [
                        {
                            "zone": "central_midfield",
                            "advantage": "Arsenal",
                            "margin": 65.5
                        },
                        {
                            "zone": "right_attack",
                            "advantage": "Arsenal",
                            "margin": 58.2
                        },
                        {
                            "zone": "left_defense",
                            "advantage": "Chelsea",
                            "margin": 52.1
                        }
                    ],
                    "insights": [
                        {
                            "text": "Arsenal's 4-3-3 provides numerical superiority in midfield against Chelsea's 4-2-3-1",
                            "category": "formation"
                        },
                        {
                            "text": "Wide forwards create overloads on the flanks",
                            "category": "tactical"
                        }
                    ],
                    "request_id": "550e8400-e29b-41d4-a716-446655440000"
                }
            ]
        }
    )
    
    dominance_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall dominance score (0-100). >50 favors team_a, <50 favors team_b",
        examples=[62.5, 45.3, 78.9],
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Model confidence in prediction (0-1)",
        examples=[0.78, 0.85, 0.62],
    )
    favored_team: str = Field(
        ...,
        description="Name of the team with predicted tactical advantage",
        examples=["Arsenal", "Manchester City"],
    )
    battle_zones: List[BattleZone] = Field(
        default_factory=list,
        description="Zone-by-zone analysis of tactical advantages",
    )
    insights: List[Insight] = Field(
        default_factory=list,
        description="List of tactical insights",
    )
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this request",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    
    @classmethod
    def create(
        cls,
        dominance_score: float,
        confidence: float,
        team_a_name: str,
        team_b_name: str,
        battle_zones: Optional[List[BattleZone]] = None,
        insights: Optional[List[Insight]] = None,
    ) -> "PredictionResponse":
        """
        Factory method to create a prediction response.
        
        Args:
            dominance_score: Raw dominance score (0-100)
            confidence: Model confidence (0-1)
            team_a_name: Name of team A
            team_b_name: Name of team B
            battle_zones: Optional list of battle zones
            insights: Optional list of insights
            
        Returns:
            PredictionResponse instance
        """
        # Determine favored team
        if dominance_score >= 50:
            favored_team = team_a_name
        else:
            favored_team = team_b_name
        
        return cls(
            dominance_score=dominance_score,
            confidence=confidence,
            favored_team=favored_team,
            battle_zones=battle_zones or [],
            insights=insights or [],
        )


# =============================================================================
# Health & Status Models
# =============================================================================

class HealthResponse(BaseModel):
    """
    API health check response.
    
    Attributes:
        status: Overall API status ("healthy", "degraded", "unhealthy")
        model_loaded: Whether the ML model is loaded and ready
        version: API version string
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "healthy",
                    "model_loaded": True,
                    "version": "1.0.0"
                },
                {
                    "status": "degraded",
                    "model_loaded": False,
                    "version": "1.0.0"
                }
            ]
        }
    )
    
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="Overall API health status",
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the prediction model is loaded",
    )
    version: str = Field(
        ...,
        description="API version",
        examples=["1.0.0", "1.1.0"],
    )


class ErrorResponse(BaseModel):
    """
    Standard error response.
    
    Attributes:
        error: Error type/code
        message: Human-readable error message
        details: Optional additional details
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error": "validation_error",
                    "message": "Invalid team name: 'Arsenall'",
                    "details": {"field": "team_a.team", "valid_options": ["Arsenal", "..."]}
                },
                {
                    "error": "model_error",
                    "message": "Model not loaded. Please try again later.",
                    "details": None
                }
            ]
        }
    )
    
    error: str = Field(
        ...,
        description="Error type or code",
        examples=["validation_error", "model_error", "internal_error"],
    )
    message: str = Field(
        ...,
        description="Human-readable error message",
    )
    details: Optional[dict] = Field(
        default=None,
        description="Additional error details",
    )


# =============================================================================
# Utility Models
# =============================================================================

class TeamInfo(BaseModel):
    """Information about a supported team."""
    
    name: str = Field(..., description="Team name")
    formations_available: List[str] = Field(
        default_factory=list_formations,
        description="Available formations",
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "Arsenal",
                    "formations_available": ["4-3-3", "4-4-2", "4-2-3-1", "3-5-2", "5-3-2", "3-4-3"]
                }
            ]
        }
    )


class TeamsListResponse(BaseModel):
    """Response containing list of supported teams."""
    
    teams: List[str] = Field(
        ...,
        description="List of supported Premier League teams",
    )
    count: int = Field(
        ...,
        description="Number of teams",
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "teams": ["Arsenal", "Aston Villa", "Bournemouth", "..."],
                    "count": 20
                }
            ]
        }
    )


class FormationsListResponse(BaseModel):
    """Response containing list of supported formations."""
    
    formations: List[str] = Field(
        ...,
        description="List of supported tactical formations",
    )
    count: int = Field(
        ...,
        description="Number of formations",
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "formations": ["4-3-3", "4-4-2", "4-2-3-1", "3-5-2", "5-3-2", "3-4-3"],
                    "count": 6
                }
            ]
        }
    )

