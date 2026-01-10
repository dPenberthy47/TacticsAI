# TacticsAI API Module

from .models import (
    BattleZone,
    ErrorResponse,
    FormationsListResponse,
    HealthResponse,
    Insight,
    PredictionRequest,
    PredictionResponse,
    TeamFormation,
    TeamInfo,
    TeamsListResponse,
    ZONE_NAMES,
)
from .prediction_service import (
    FormationGraphBuilder,
    PredictionService,
    get_prediction_service,
    get_cached_prediction_service,
    reset_prediction_service,
)
from .main import app, API_VERSION, API_TITLE

__all__ = [
    # FastAPI app
    "app",
    "API_VERSION",
    "API_TITLE",
    # Request models
    "TeamFormation",
    "PredictionRequest",
    # Response models
    "BattleZone",
    "Insight",
    "PredictionResponse",
    "HealthResponse",
    "ErrorResponse",
    # Utility models
    "TeamInfo",
    "TeamsListResponse",
    "FormationsListResponse",
    # Constants
    "ZONE_NAMES",
    # Service
    "FormationGraphBuilder",
    "PredictionService",
    "get_prediction_service",
    "get_cached_prediction_service",
    "reset_prediction_service",
]
