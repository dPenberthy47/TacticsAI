"""
TacticsAI FastAPI Application
Football tactics prediction API powered by Graph Neural Networks.
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import List

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import ValidationError

# Import configuration
from data.config import (
    FORMATIONS,
    PREMIER_LEAGUE_TEAMS,
    list_formations,
    list_teams,
)

# Import API models
from api.models import (
    ErrorResponse,
    FormationsListResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    TeamsListResponse,
)

# Import prediction service
from api.prediction_service import (
    PredictionService,
    get_prediction_service,
)


# =============================================================================
# App Configuration
# =============================================================================

API_VERSION = "1.0.0"
API_TITLE = "TacticsAI API"
API_DESCRIPTION = """
# TacticsAI - Football Tactics Prediction Engine 🏈⚽

A Graph Neural Network powered API for predicting tactical matchup outcomes
in football (soccer) matches.

## Features

- **Tactical Prediction**: Analyze formation matchups between Premier League teams
- **Battle Zone Analysis**: Understand which areas of the pitch favor which team
- **Tactical Insights**: Get AI-generated insights about the matchup

## How It Works

1. Select two teams and their formations
2. Our GNN model analyzes the tactical graphs
3. Receive dominance predictions, battle zones, and insights

## Formations Supported

- 4-3-3, 4-4-2, 4-2-3-1
- 3-5-2, 5-3-2, 3-4-3

## Data Sources

- StatsBomb Open Data (primary)
- FBref statistics
- TransferMarkt valuations

---
Built with ❤️ using PyTorch Geometric and FastAPI
"""


# =============================================================================
# Lifespan Context Manager
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info("=" * 60)
    logger.info(f"Starting {API_TITLE} v{API_VERSION}")
    logger.info("=" * 60)
    
    # Pre-load prediction service
    try:
        service = get_prediction_service()
        status = service.get_status()
        
        logger.info(f"Model loaded: {status['model_loaded']}")
        logger.info(f"Device: {status['device']}")
        logger.info(f"Formations: {len(status['formations_available'])}")
        logger.info("Prediction service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize prediction service: {e}")
    
    logger.info("=" * 60)
    logger.info("API ready to accept requests")
    logger.info("=" * 60)
    
    yield  # App runs here
    
    # Shutdown
    logger.info("Shutting down TacticsAI API...")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# =============================================================================
# CORS Middleware
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",  # Vite default
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://*.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request Logging Middleware
# =============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # Log request
    logger.info(f"[{request_id}] {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration_ms = (time.time() - start_time) * 1000
    
    # Log response
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"-> {response.status_code} ({duration_ms:.1f}ms)"
    )
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    
    return response


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors."""
    logger.warning(f"Validation error: {exc}")
    
    # Extract error details
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        })
    
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="validation_error",
            message="Request validation failed",
            details={"errors": errors},
        ).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP error {exc.status_code}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="http_error",
            message=str(exc.detail),
            details=None,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.exception(f"Unexpected error: {exc}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            message="An unexpected error occurred. Please try again later.",
            details=None,
        ).model_dump(),
    )


# =============================================================================
# Endpoints
# =============================================================================

@app.get(
    "/",
    summary="Root",
    description="Welcome endpoint with API information",
    tags=["General"],
)
async def root():
    """Root endpoint with API information."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check API health and model status",
    tags=["General"],
)
async def health_check(
    service: PredictionService = Depends(get_prediction_service),
) -> HealthResponse:
    """
    Health check endpoint.
    
    Returns the API status, whether the ML model is loaded,
    and the current API version.
    """
    model_loaded = service.model_loaded
    
    if model_loaded:
        status = "healthy"
    else:
        status = "degraded"  # Running but without model
    
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        version=API_VERSION,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Matchup",
    description="Generate tactical prediction for a team matchup",
    tags=["Predictions"],
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
)
async def predict(
    request: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    """
    Generate a tactical prediction for a matchup.
    
    Analyzes the formations of both teams and predicts:
    - **Dominance Score**: Which team has the tactical advantage (0-100)
    - **Confidence**: How confident the model is in the prediction (0-1)
    - **Battle Zones**: Zone-by-zone analysis of pitch control
    - **Insights**: AI-generated tactical observations
    
    Example request:
    ```json
    {
        "team_a": {"team": "Arsenal", "formation": "4-3-3"},
        "team_b": {"team": "Chelsea", "formation": "4-2-3-1"}
    }
    ```
    """
    try:
        response = service.predict(
            team_a=request.team_a.team,
            formation_a=request.team_a.formation,
            team_b=request.team_b.team,
            formation_b=request.team_b.formation,
        )
        
        logger.info(
            f"Prediction: {request.team_a.team} vs {request.team_b.team} "
            f"-> {response.favored_team} ({response.dominance_score:.1f}%)"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@app.get(
    "/teams",
    response_model=TeamsListResponse,
    summary="List Teams",
    description="Get all supported Premier League teams",
    tags=["Reference Data"],
)
async def get_teams() -> TeamsListResponse:
    """
    Get list of all supported Premier League teams.
    
    Returns the 20 current Premier League teams that can be
    used in prediction requests.
    """
    teams = list_teams()
    
    return TeamsListResponse(
        teams=sorted(teams),
        count=len(teams),
    )


@app.get(
    "/formations",
    response_model=FormationsListResponse,
    summary="List Formations",
    description="Get all supported tactical formations",
    tags=["Reference Data"],
)
async def get_formations() -> FormationsListResponse:
    """
    Get list of all supported tactical formations.
    
    Returns the available formations with their tactical setup.
    """
    formations = list_formations()
    
    return FormationsListResponse(
        formations=sorted(formations),
        count=len(formations),
    )


@app.get(
    "/formations/{formation_name}",
    summary="Get Formation Details",
    description="Get detailed position data for a specific formation",
    tags=["Reference Data"],
    responses={
        200: {"description": "Formation details"},
        404: {"model": ErrorResponse, "description": "Formation not found"},
    },
)
async def get_formation_details(formation_name: str):
    """
    Get detailed position data for a specific formation.
    
    Returns the (x, y) coordinates for each player position
    in the specified formation.
    """
    if formation_name not in FORMATIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Formation '{formation_name}' not found. Available: {list_formations()}",
        )
    
    formation = FORMATIONS[formation_name]
    
    return {
        "name": formation.name,
        "positions": [
            {
                "name": pos.name,
                "x": pos.x,
                "y": pos.y,
            }
            for pos in formation.positions
        ],
        "coordinates": formation.to_coordinates(),
    }


@app.get(
    "/teams/{team_name}",
    summary="Get Team Details",
    description="Get details for a specific team",
    tags=["Reference Data"],
    responses={
        200: {"description": "Team details"},
        404: {"model": ErrorResponse, "description": "Team not found"},
    },
)
async def get_team_details(team_name: str):
    """
    Get details for a specific Premier League team.
    
    Returns the team's identifiers across different data sources.
    """
    # Case-insensitive lookup
    team_mapping = None
    actual_name = None
    
    for name, mapping in PREMIER_LEAGUE_TEAMS.items():
        if name.lower() == team_name.lower():
            team_mapping = mapping
            actual_name = name
            break
    
    if team_mapping is None:
        raise HTTPException(
            status_code=404,
            detail=f"Team '{team_name}' not found. Available: {list_teams()}",
        )
    
    return {
        "name": actual_name,
        "identifiers": {
            "fbref_id": team_mapping.fbref_id,
            "fbref_slug": team_mapping.fbref_slug,
            "statsbomb_id": team_mapping.statsbomb_id,
            "understat_id": team_mapping.understat_id,
            "transfermarkt_id": team_mapping.transfermarkt_id,
        },
        "available_formations": list_formations(),
    }


# =============================================================================
# Run with Uvicorn (for development)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

