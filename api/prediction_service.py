"""
TacticsAI Prediction Service
Handles model inference and prediction generation.
"""

import random
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger

# Import data utilities
from data.config import FORMATIONS, Formation, PREMIER_LEAGUE_TEAMS
from data.dataset import create_tactical_graph, formation_to_graph

# Import models
from models.gnn import TacticsGNN, SimpleTacticsGNN
from models.trainer import get_device

# Import API models
from api.models import (
    BattleZone,
    Insight,
    PredictionResponse,
    ZONE_NAMES,
)

# Import insights generator
from api.insights_generator import InsightsGenerator


# =============================================================================
# Formation Graph Builder
# =============================================================================

class FormationGraphBuilder:
    """
    Builds tactical graphs from formation configurations.
    
    Converts formation names and team configurations into
    PyTorch Geometric Data objects for model inference.
    """
    
    def __init__(self, formations: Dict[str, Formation] = None):
        """
        Initialize the builder.
        
        Args:
            formations: Dict of formation name to Formation objects
        """
        self.formations = formations or FORMATIONS
        logger.debug(f"FormationGraphBuilder initialized with {len(self.formations)} formations")
    
    def build(self, formation_name: str, team_name: Optional[str] = None):
        """
        Build a tactical graph from a formation name.
        
        Args:
            formation_name: Name of the formation (e.g., "4-3-3")
            team_name: Optional team name for feature augmentation
            
        Returns:
            PyG Data object representing the formation
        """
        if formation_name not in self.formations:
            logger.warning(f"Unknown formation '{formation_name}', using 4-4-2")
            formation_name = "4-4-2"
        
        formation = self.formations[formation_name]
        graph = formation_to_graph(formation)
        
        # Add team-specific features if available
        if team_name and team_name in PREMIER_LEAGUE_TEAMS:
            graph = self._augment_with_team_features(graph, team_name)
        
        return graph
    
    def _augment_with_team_features(self, graph, team_name: str):
        """
        Augment graph with team-specific features.
        
        Currently a placeholder - could add team playing style metrics.
        """
        # Future: Add team-specific features like avg possession, pressing intensity
        return graph
    
    def get_formation_positions(self, formation_name: str) -> List[Tuple[float, float]]:
        """Get player positions for a formation."""
        if formation_name not in self.formations:
            return []
        return self.formations[formation_name].to_coordinates()


# =============================================================================
# Prediction Service
# =============================================================================

class PredictionService:
    """
    Service for generating tactical predictions.
    
    Handles model loading, graph building, and inference.
    Falls back to mock predictions if model is not available.
    
    Usage:
        service = PredictionService()
        result = service.predict("Arsenal", "4-3-3", "Chelsea", "4-2-3-1")
    """
    
    # Default model path
    DEFAULT_MODEL_PATH = "backend/models/saved/tacticsai.pt"
    
    def __init__(self, model_path: str = None):
        """
        Initialize the prediction service.

        Args:
            model_path: Path to trained model checkpoint
        """
        self.model_path = Path(model_path or self.DEFAULT_MODEL_PATH)
        self.device = get_device()
        self.model: Optional[torch.nn.Module] = None
        self.model_loaded = False

        # Initialize graph builder
        self.graph_builder = FormationGraphBuilder(FORMATIONS)

        # Store current matchup formations for fallback zone generation
        self.current_formation_a: Optional[str] = None
        self.current_formation_b: Optional[str] = None

        # Initialize insights generator
        self.insights_generator = InsightsGenerator()

        # Try to load model
        self._load_model()
    
    def _load_model(self) -> bool:
        """
        Load the trained model from disk.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not self.model_path.exists():
            logger.warning(f"Model not found at {self.model_path}. Using mock predictions.")
            self.model_loaded = False
            return False
        
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Determine model type from checkpoint
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            
            # Check if it's a simple or full model based on layer names
            is_simple = any("gat1" in k for k in state_dict.keys()) and "gat_layers" not in str(state_dict.keys())

            if is_simple:
                self.model = SimpleTacticsGNN(
                    node_features=12,
                    hidden_dim=32,
                    edge_features=6,
                    use_edge_features=True
                )
            else:
                self.model = TacticsGNN(
                    node_features=12,
                    hidden_dim=64,
                    edge_features=6,
                    use_edge_features=True
                )
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
            return False
    
    def predict(
        self,
        team_a: str,
        formation_a: str,
        team_b: str,
        formation_b: str,
    ) -> PredictionResponse:
        """
        Generate a tactical prediction for a matchup.
        
        Args:
            team_a: First team name
            formation_a: First team's formation
            team_b: Second team name
            formation_b: Second team's formation
            
        Returns:
            PredictionResponse with dominance score, battle zones, and insights
        """
        logger.info(f"Predicting: {team_a} ({formation_a}) vs {team_b} ({formation_b})")

        # Store formations for fallback zone generation
        self.current_formation_a = formation_a
        self.current_formation_b = formation_b

        # Build graphs
        graph_a = self.graph_builder.build(formation_a, team_a)
        graph_b = self.graph_builder.build(formation_b, team_b)
        
        # Run inference or mock
        if self.model_loaded and self.model is not None:
            result = self._run_inference(graph_a, graph_b, team_a, team_b)
        else:
            result = self._mock_prediction(team_a, formation_a, team_b, formation_b)

        # Map battle zones
        battle_zones = self._map_battle_zones(
            result.get("battle_zones_raw"),
            team_a, team_b,
            result["dominance_score"],
            attention_weights=result.get("attention_weights"),
        )

        # Generate insights using LLM
        insights = self.insights_generator.generate_insights(
            team_a=team_a,
            formation_a=formation_a,
            team_b=team_b,
            formation_b=formation_b,
            dominance_score=result["dominance_score"],
            confidence=result["confidence"],
            battle_zones=[z.model_dump() for z in battle_zones],
        )
        
        # Build response
        return PredictionResponse.create(
            dominance_score=result["dominance_score"],
            confidence=result["confidence"],
            team_a_name=team_a,
            team_b_name=team_b,
            battle_zones=battle_zones,
            insights=insights,
        )
    
    @torch.no_grad()
    def _run_inference(
        self,
        graph_a,
        graph_b,
        team_a: str,
        team_b: str,
    ) -> Dict:
        """
        Run model inference on the graphs.
        
        Args:
            graph_a: First team's tactical graph
            graph_b: Second team's tactical graph
            team_a: First team name
            team_b: Second team name
            
        Returns:
            Dict with prediction results
        """
        # Move to device
        graph_a = graph_a.to(self.device)
        graph_b = graph_b.to(self.device)
        
        # Add batch dimension for single sample
        graph_a.batch = torch.zeros(graph_a.x.size(0), dtype=torch.long, device=self.device)
        graph_b.batch = torch.zeros(graph_b.x.size(0), dtype=torch.long, device=self.device)
        
        # Forward pass
        output = self.model(graph_a, graph_b)
        
        # Extract predictions
        dominance = output["dominance"]
        if dominance.dim() > 0:
            dominance = dominance.item()
        
        confidence = output.get("confidence")
        if confidence is not None:
            if confidence.dim() > 0:
                confidence = confidence.item()
        else:
            confidence = 0.75  # Default confidence
        
        # Battle zones are now attention-derived (already 0-1 range)
        battle_zones_raw = output.get("battle_zones")
        if battle_zones_raw is not None:
            if isinstance(battle_zones_raw, torch.Tensor):
                battle_zones_raw = battle_zones_raw.cpu().numpy()

        # Get attention weights for interpretability
        attention_a = output.get("attention_a")
        attention_b = output.get("attention_b")
        attention_weights = None
        if attention_a is not None and attention_b is not None:
            attention_weights = {
                "team_a": attention_a.cpu().numpy() if isinstance(attention_a, torch.Tensor) else attention_a,
                "team_b": attention_b.cpu().numpy() if isinstance(attention_b, torch.Tensor) else attention_b,
            }
        
        # Convert dominance to percentage
        dominance_score = dominance * 100
        
        return {
            "dominance_score": dominance_score,
            "confidence": confidence,
            "battle_zones_raw": battle_zones_raw,
            "attention_weights": attention_weights,
        }
    
    def _mock_prediction(
        self,
        team_a: str,
        formation_a: str,
        team_b: str,
        formation_b: str,
    ) -> Dict:
        """
        Generate mock prediction when model is not available.
        
        Uses simple heuristics based on formation matchups.
        """
        logger.debug("Generating mock prediction")
        
        # Simple formation advantage rules
        formation_scores = {
            "4-3-3": 0.55,   # Slightly attacking
            "4-4-2": 0.50,   # Balanced
            "4-2-3-1": 0.52, # Slightly attacking
            "3-5-2": 0.48,   # Midfield focus
            "5-3-2": 0.45,   # Defensive
            "3-4-3": 0.58,   # Very attacking
        }
        
        score_a = formation_scores.get(formation_a, 0.50)
        score_b = formation_scores.get(formation_b, 0.50)
        
        # Calculate relative advantage
        base_dominance = 50 + (score_a - score_b) * 50
        
        # Add some randomness
        noise = random.gauss(0, 5)
        dominance_score = max(20, min(80, base_dominance + noise))

        # Generate formation-based battle zones (NOT random)
        battle_zones_raw = self._generate_formation_based_zones(
            formation_a, formation_b, dominance_score
        )

        return {
            "dominance_score": dominance_score,
            "confidence": 0.3,  # Low confidence for mock
            "battle_zones_raw": battle_zones_raw,
            "attention_weights": None,  # No attention in mock mode
        }
    
    def _map_battle_zones(
        self,
        zones_raw: Optional[List[float]],
        team_a: str,
        team_b: str,
        dominance_score: float,
        attention_weights: Optional[Dict] = None,
    ) -> List[BattleZone]:
        """
        Map model-derived zone predictions to BattleZone objects.

        If model output available, use attention-derived zones.
        If not, generate heuristic zones based on formation analysis (not random!).

        Args:
            zones_raw: Attention-derived tensor values (9 zones) or None
            team_a: First team name
            team_b: Second team name
            dominance_score: Overall dominance score
            attention_weights: Optional attention weights for additional context

        Returns:
            List of BattleZone objects
        """
        # Zone names (3x3 grid, but we report 8 main zones)
        zone_mapping = [
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
        
        battle_zones = []

        if zones_raw is None or len(zones_raw) < 9:
            # Fallback: formation-based heuristic (NOT random)
            zones_raw = self._generate_formation_based_zones(
                self.current_formation_a,
                self.current_formation_b,
                dominance_score
            )
        
        # Ensure we have 9 values
        zones_raw = list(zones_raw)[:9]
        while len(zones_raw) < 9:
            zones_raw.append(0.5)
        
        for i, zone_name in enumerate(zone_mapping):
            value = zones_raw[i]
            
            # Convert to margin (0-100)
            # Value > 0.5 means team_a has advantage
            if value >= 0.5:
                advantage = team_a
                margin = (value - 0.5) * 200  # Scale to 0-100
            else:
                advantage = team_b
                margin = (0.5 - value) * 200
            
            # Ensure margin is at least 50 (slight advantage)
            margin = max(50.1, min(99.9, 50 + margin))
            
            battle_zones.append(BattleZone(
                zone=zone_name,
                advantage=advantage,
                margin=round(margin, 1),
            ))
        
        return battle_zones
    
    def _generate_formation_based_zones(
        self,
        formation_a: Optional[str],
        formation_b: Optional[str],
        dominance_score: float
    ) -> List[float]:
        """
        Generate zone values based on formation characteristics.

        NOT random - uses tactical knowledge about formation strengths.
        Each formation has typical strength profiles across the 3x3 pitch grid.

        Args:
            formation_a: First team's formation
            formation_b: Second team's formation
            dominance_score: Overall dominance score (0-100)

        Returns:
            List of 9 float values (0-1) representing zone control
        """
        # Formation zone strengths (which zones each formation is strong in)
        # Grid layout: [0=left_def, 1=center_def, 2=right_def,
        #               3=left_mid, 4=center_mid, 5=right_mid,
        #               6=left_att, 7=center_att, 8=right_att]
        formation_strengths = {
            "4-3-3": [0.4, 0.5, 0.4, 0.5, 0.6, 0.5, 0.7, 0.6, 0.7],  # Strong on wings in attack
            "4-4-2": [0.5, 0.6, 0.5, 0.6, 0.5, 0.6, 0.5, 0.6, 0.5],  # Balanced, strong midfield width
            "4-2-3-1": [0.5, 0.6, 0.5, 0.4, 0.7, 0.4, 0.6, 0.7, 0.6],  # Strong central midfield/attack
            "3-5-2": [0.5, 0.7, 0.5, 0.6, 0.7, 0.6, 0.5, 0.6, 0.5],  # Strong central areas
            "5-3-2": [0.6, 0.7, 0.6, 0.5, 0.6, 0.5, 0.4, 0.5, 0.4],  # Strong defense
            "3-4-3": [0.4, 0.5, 0.4, 0.5, 0.5, 0.5, 0.7, 0.6, 0.7],  # Strong wide attack
        }

        # Get formation strengths (default to balanced if unknown)
        str_a = formation_strengths.get(formation_a, [0.5] * 9)
        str_b = formation_strengths.get(formation_b, [0.5] * 9)

        zones = []
        for i in range(9):
            # Compare formation strengths at this zone
            total_strength = str_a[i] + str_b[i]
            if total_strength > 0:
                base = str_a[i] / total_strength
            else:
                base = 0.5

            # Adjust by overall dominance (70% formation, 30% dominance)
            dominance_factor = dominance_score / 100
            adjusted = base * 0.7 + dominance_factor * 0.3

            # Clamp to reasonable range
            adjusted = max(0.2, min(0.8, adjusted))
            zones.append(adjusted)

        return zones
    
    @property
    def is_ready(self) -> bool:
        """Check if service is ready for predictions."""
        return True  # Always ready (uses mock if no model)
    
    def get_status(self) -> Dict:
        """Get service status."""
        return {
            "model_loaded": self.model_loaded,
            "model_path": str(self.model_path),
            "device": str(self.device),
            "formations_available": list(self.graph_builder.formations.keys()),
        }


# =============================================================================
# FastAPI Dependency (Singleton Pattern)
# =============================================================================

_prediction_service: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """
    Get the singleton PredictionService instance.
    
    Use this as a FastAPI dependency:
    
        @app.post("/predict")
        def predict(
            request: PredictionRequest,
            service: PredictionService = Depends(get_prediction_service)
        ):
            return service.predict(...)
    """
    global _prediction_service
    
    if _prediction_service is None:
        logger.info("Initializing PredictionService singleton")
        _prediction_service = PredictionService()
    
    return _prediction_service


def reset_prediction_service() -> None:
    """Reset the singleton (useful for testing)."""
    global _prediction_service
    _prediction_service = None


@lru_cache(maxsize=1)
def get_cached_prediction_service() -> PredictionService:
    """
    Alternative: LRU cached singleton.
    
    Use this if you prefer functools caching over global state.
    """
    return PredictionService()

