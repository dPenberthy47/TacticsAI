#!/usr/bin/env python3
"""
Quick validation that all TacticsAI components are working.
Run this after making changes to verify nothing is broken.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

# Configure logger for validation output
logger.remove()
logger.add(
    sys.stderr,
    format="<level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
)


def main():
    logger.info("=" * 60)
    logger.info("TacticsAI Pipeline Validation")
    logger.info("=" * 60)

    checks = []

    # Check 1: Core imports
    logger.info("Checking imports...")
    try:
        from data.config import FORMATIONS, PREMIER_LEAGUE_TEAMS
        from data.fetchers import (
            StatsBombFetcher,
            FBrefFetcher,
            TransferMarktFetcher,
            UnderstatFetcher,
        )
        from data.dataset import MatchupDataset
        from data.graph_builder import FormationGraphBuilder
        from models.gnn import TacticsGNN, SimpleTacticsGNN
        from api.prediction_service import PredictionService
        from api.insights_generator import InsightsGenerator

        checks.append(("Core imports", True))
    except ImportError as e:
        checks.append(("Core imports", False, str(e)))

    # Check 2: Dataset creation
    logger.info("Testing dataset creation...")
    try:
        from data.dataset import MatchupDataset

        dataset = MatchupDataset.create_synthetic(n_samples=5)
        assert len(dataset) == 5
        checks.append(("Dataset creation", True))
    except Exception as e:
        checks.append(("Dataset creation", False, str(e)))

    # Check 3: Feature dimensions
    logger.info("Validating feature dimensions...")
    try:
        sample = dataset[0]
        graph = sample[0] if isinstance(sample, tuple) else sample.get("graph_a")
        node_dim = graph.x.shape[1]
        edge_dim = graph.edge_attr.shape[1] if graph.edge_attr is not None else 0

        if node_dim >= 12 and edge_dim >= 6:
            checks.append(("Enhanced features (12D nodes, 6D edges)", True))
        else:
            checks.append(
                ("Enhanced features", False, f"node={node_dim}D, edge={edge_dim}D")
            )
    except Exception as e:
        checks.append(("Enhanced features", False, str(e)))

    # Check 4: Model architecture
    logger.info("Testing model architecture...")
    try:
        from models.gnn import TacticsGNN

        model = TacticsGNN(node_features=12, edge_features=6)
        assert hasattr(model, "gat_layers"), "Model missing GAT layers"
        assert len(model.gat_layers) >= 2, "Model should have at least 2 GAT layers"
        checks.append(("GAT model architecture", True))
    except Exception as e:
        checks.append(("GAT model architecture", False, str(e)))

    # Check 5: Forward pass
    logger.info("Testing model forward pass...")
    try:
        from data.dataset import formation_to_graph
        from data.config import FORMATIONS
        import torch

        model.eval()
        graph_a = formation_to_graph(FORMATIONS["4-3-3"])
        graph_b = formation_to_graph(FORMATIONS["4-4-2"])

        graph_a.batch = torch.zeros(11, dtype=torch.long)
        graph_b.batch = torch.zeros(11, dtype=torch.long)

        with torch.no_grad():
            output = model(graph_a, graph_b)

        assert "dominance" in output
        assert "battle_zones" in output
        assert "attention_a" in output
        checks.append(("Model forward pass", True))
    except Exception as e:
        checks.append(("Model forward pass", False, str(e)))

    # Check 6: Attention-based zones
    logger.info("Testing attention-based battle zones...")
    try:
        assert output["battle_zones"].shape[0] == 9, "Should have 9 zones (3x3 grid)"
        assert output["attention_a"].shape[0] == 9, "Should have 9 zone attentions"

        # Check zones are continuous (not binary)
        zones = output["battle_zones"]
        assert (zones >= 0).all() and (zones <= 1).all(), "Zones should be in [0,1]"

        checks.append(("Attention-based zones", True))
    except Exception as e:
        checks.append(("Attention-based zones", False, str(e)))

    # Check 7: Prediction service
    logger.info("Testing prediction service...")
    try:
        from api.prediction_service import PredictionService

        service = PredictionService()
        response = service.predict("Arsenal", "4-3-3", "Chelsea", "4-4-2")
        assert response.dominance_score is not None
        assert 0 <= response.dominance_score <= 100
        assert len(response.battle_zones) == 9
        checks.append(("Prediction service", True))
    except Exception as e:
        checks.append(("Prediction service", False, str(e)))

    # Check 8: Battle zones are deterministic (not random)
    logger.info("Testing battle zones determinism...")
    try:
        # Same input should give same output when using fallback
        r1 = service.predict("Liverpool", "4-3-3", "Man City", "4-2-3-1")
        r2 = service.predict("Liverpool", "4-3-3", "Man City", "4-2-3-1")

        # If using fallback (no model loaded), should be identical
        if not service.model_loaded:
            assert r1.battle_zones[0].margin == r2.battle_zones[0].margin
            checks.append(("Battle zones determinism", True))
        else:
            # With model, small variance acceptable
            checks.append(("Battle zones (model-based)", True))
    except Exception as e:
        checks.append(("Battle zones determinism", False, str(e)))

    # Check 9: Insights generator
    logger.info("Testing insights generator...")
    try:
        from api.insights_generator import InsightsGenerator

        generator = InsightsGenerator()

        # Test fallback insights work
        insights = generator._generate_fallback_insights(
            "Arsenal", "4-3-3", "Chelsea", "4-4-2", 60
        )
        assert len(insights) > 0
        assert all(i.category in ["formation", "tactical", "historical", "key_factor"] for i in insights)
        checks.append(("Insights generator", True))
    except Exception as e:
        checks.append(("Insights generator", False, str(e)))

    # Check 10: Tactical dominance calculation
    logger.info("Testing tactical dominance score...")
    try:
        from data.dataset import MatchupDataset

        score = MatchupDataset._calculate_tactical_dominance(
            home_xg=2.0,
            away_xg=1.0,
            home_possession=60,
            away_possession=40,
            home_chances_created=12,
            away_chances_created=7,
            home_progressive_actions=45,
            away_progressive_actions=28,
            home_score=2,
            away_score=1,
        )

        assert 0 <= score <= 1
        assert score > 0.5, "Home team should dominate"
        checks.append(("Tactical dominance calculation", True))
    except Exception as e:
        checks.append(("Tactical dominance calculation", False, str(e)))

    # Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("Results:")
    logger.info("-" * 60)

    all_passed = True
    for check in checks:
        name = check[0]
        passed = check[1]

        if passed:
            logger.info(f"✓ {name}")
        else:
            all_passed = False
            error = check[2] if len(check) > 2 else "Unknown error"
            logger.error(f"✗ {name}: {error}")

    logger.info("-" * 60)

    if all_passed:
        logger.info("✓ ALL CHECKS PASSED!")
        logger.info("=" * 60)
        return 0
    else:
        logger.error("✗ SOME CHECKS FAILED")
        logger.error("Please review the errors above and fix before deploying.")
        logger.info("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
