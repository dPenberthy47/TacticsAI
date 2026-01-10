"""
TacticsAI Integration Tests
Validates the full pipeline from data fetching to prediction.
"""

import pytest
import torch
import pandas as pd
from pathlib import Path
import sys

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


# =============================================================================
# Test imports work
# =============================================================================


def test_imports():
    """Test all modules import correctly."""
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

    assert len(FORMATIONS) >= 6
    assert len(PREMIER_LEAGUE_TEAMS) == 20


# =============================================================================
# Test data sources
# =============================================================================


class TestDataSources:
    """Test all 4 data sources are functional."""

    def test_statsbomb_fetcher(self):
        """Test StatsBomb fetcher can load data."""
        from data.fetchers import StatsBombFetcher

        fetcher = StatsBombFetcher()
        matches = fetcher.fetch_matches(competition_id=2, season_id=27, max_matches=5)

        assert len(matches) > 0
        assert "home_team" in matches.columns
        assert "away_team" in matches.columns

    def test_fbref_fetcher_initialization(self):
        """Test FBref fetcher initializes without error."""
        from data.fetchers import FBrefFetcher

        fetcher = FBrefFetcher()
        # FBref scraping may not work in all environments
        # Just check it initializes
        assert fetcher is not None

    def test_understat_fetcher_initialization(self):
        """Test Understat fetcher initializes without error."""
        from data.fetchers import UnderstatFetcher

        fetcher = UnderstatFetcher()
        assert fetcher is not None

    def test_transfermarkt_fetcher_initialization(self):
        """Test TransferMarkt fetcher initializes without error."""
        from data.fetchers import TransferMarktFetcher

        fetcher = TransferMarktFetcher()
        assert fetcher is not None

    def test_create_fetcher_factory(self):
        """Test fetcher factory method."""
        from data.fetchers import create_fetcher

        statsbomb = create_fetcher("statsbomb")
        fbref = create_fetcher("fbref")
        understat = create_fetcher("understat")
        transfermarkt = create_fetcher("transfermarkt")

        assert statsbomb is not None
        assert fbref is not None
        assert understat is not None
        assert transfermarkt is not None


# =============================================================================
# Test enhanced features
# =============================================================================


class TestEnhancedFeatures:
    """Test enhanced node and edge features."""

    def test_node_features_dimension(self):
        """Test synthetic dataset has 12D node features."""
        from data.dataset import MatchupDataset

        dataset = MatchupDataset.create_synthetic(n_samples=10)
        sample = dataset[0]
        graph = sample[0] if isinstance(sample, tuple) else sample.get("graph_a")

        assert graph.x.shape[0] == 11  # 11 players
        assert graph.x.shape[1] == 12  # 12 features (enhanced)

    def test_edge_features_dimension(self):
        """Test synthetic dataset has 6D edge features."""
        from data.dataset import MatchupDataset

        dataset = MatchupDataset.create_synthetic(n_samples=10)
        sample = dataset[0]
        graph = sample[0] if isinstance(sample, tuple) else sample.get("graph_a")

        assert graph.edge_attr is not None
        assert graph.edge_attr.shape[1] == 6  # 6 edge features (enhanced)

    def test_tactical_dominance_score(self):
        """Test tactical dominance calculation produces valid scores."""
        from data.dataset import MatchupDataset

        # Test calculation directly
        score = MatchupDataset._calculate_tactical_dominance(
            home_xg=2.5,
            away_xg=1.0,
            home_possession=60,
            away_possession=40,
            home_chances_created=15,
            away_chances_created=8,
            home_progressive_actions=50,
            away_progressive_actions=30,
            home_score=2,
            away_score=1,
        )

        assert 0 <= score <= 1
        assert score > 0.5  # Home team clearly dominated

    def test_tactical_dominance_balanced(self):
        """Test balanced matchup produces score near 0.5."""
        from data.dataset import MatchupDataset

        score = MatchupDataset._calculate_tactical_dominance(
            home_xg=1.5,
            away_xg=1.5,
            home_possession=50,
            away_possession=50,
            home_chances_created=10,
            away_chances_created=10,
            home_progressive_actions=40,
            away_progressive_actions=40,
            home_score=1,
            away_score=1,
        )

        assert 0.4 <= score <= 0.6  # Should be balanced

    def test_extract_match_stats_from_events(self):
        """Test match stats extraction from events."""
        from data.dataset import MatchupDataset
        import pandas as pd

        # Create mock events
        events = pd.DataFrame(
            {
                "team": ["Arsenal", "Arsenal", "Chelsea", "Chelsea", "Arsenal"],
                "type": ["Pass", "Shot", "Pass", "Shot", "Carry"],
                "location": [[30, 40], [80, 50], [40, 40], [85, 45], [50, 40]],
                "pass": [
                    {"end_location": [50, 40]},
                    None,
                    {"end_location": [60, 50]},
                    None,
                    None,
                ],
                "carry": [None, None, None, None, {"end_location": [70, 40]}],
            }
        )

        stats = MatchupDataset._extract_match_stats_from_events(events, "Arsenal")

        assert "chances_created" in stats
        assert "progressive_actions" in stats
        assert "possession" in stats
        assert stats["shots"] >= 1
        assert 0 <= stats["possession"] <= 100


# =============================================================================
# Test model architecture
# =============================================================================


class TestModelArchitecture:
    """Test GNN architecture matches PRD requirements."""

    def test_gat_layers(self):
        """Test model uses GAT layers (not GCN)."""
        from models.gnn import TacticsGNN

        model = TacticsGNN(node_features=12, edge_features=6)

        # Check model has GAT layers
        assert hasattr(model, "gat_layers")
        assert len(model.gat_layers) >= 2  # PRD says 2-3 GAT layers

    def test_attention_weights_extraction(self):
        """Test attention weights can be extracted."""
        from models.gnn import TacticsGNN
        from data.dataset import formation_to_graph
        from data.config import FORMATIONS

        model = TacticsGNN(node_features=12, edge_features=6)
        model.eval()

        graph = formation_to_graph(FORMATIONS["4-3-3"])

        # Add batch dimension
        graph.batch = torch.zeros(11, dtype=torch.long)

        # Should be able to get attention weights
        attention = model.get_attention_weights(graph)

        assert isinstance(attention, list)
        assert len(attention) > 0

    def test_zone_attention(self):
        """Test zone attention aggregation."""
        from models.gnn import TacticsGNN
        from data.dataset import formation_to_graph
        from data.config import FORMATIONS

        model = TacticsGNN(node_features=12, edge_features=6)
        model.eval()

        graph = formation_to_graph(FORMATIONS["4-3-3"])
        graph.batch = torch.zeros(11, dtype=torch.long)

        zone_attention = model.get_zone_attention(graph)

        assert zone_attention.shape[0] == 9  # 3x3 grid
        assert torch.allclose(zone_attention.sum(), torch.tensor(1.0), atol=1e-5)  # Should sum to 1

    def test_forward_pass(self):
        """Test forward pass produces valid outputs."""
        from models.gnn import TacticsGNN
        from data.dataset import formation_to_graph
        from data.config import FORMATIONS

        model = TacticsGNN(node_features=12, edge_features=6)
        model.eval()

        # Create sample graphs
        graph_a = formation_to_graph(FORMATIONS["4-3-3"])
        graph_b = formation_to_graph(FORMATIONS["4-4-2"])

        # Add batch dimension
        graph_a.batch = torch.zeros(11, dtype=torch.long)
        graph_b.batch = torch.zeros(11, dtype=torch.long)

        with torch.no_grad():
            output = model(graph_a, graph_b)

        assert "dominance" in output
        assert "confidence" in output
        assert "battle_zones" in output
        assert "attention_a" in output
        assert "attention_b" in output

        # Check output ranges
        assert 0 <= output["dominance"].item() <= 1
        assert 0 <= output["confidence"].item() <= 1
        assert output["battle_zones"].shape[0] == 9

    def test_simple_model(self):
        """Test SimpleTacticsGNN also works."""
        from models.gnn import SimpleTacticsGNN
        from data.dataset import formation_to_graph
        from data.config import FORMATIONS

        model = SimpleTacticsGNN(node_features=12, edge_features=6)
        model.eval()

        graph_a = formation_to_graph(FORMATIONS["4-3-3"])
        graph_b = formation_to_graph(FORMATIONS["4-4-2"])

        graph_a.batch = torch.zeros(11, dtype=torch.long)
        graph_b.batch = torch.zeros(11, dtype=torch.long)

        with torch.no_grad():
            output = model(graph_a, graph_b)

        assert "dominance" in output
        assert 0 <= output["dominance"].item() <= 1


# =============================================================================
# Test prediction service
# =============================================================================


class TestPredictionService:
    """Test end-to-end prediction."""

    def test_prediction_response_format(self):
        """Test prediction returns properly formatted response."""
        from api.prediction_service import PredictionService

        service = PredictionService()
        response = service.predict(
            team_a="Arsenal",
            formation_a="4-3-3",
            team_b="Chelsea",
            formation_b="4-2-3-1",
        )

        assert 0 <= response.dominance_score <= 100
        assert 0 <= response.confidence <= 1
        assert response.favored_team in ["Arsenal", "Chelsea"]
        assert len(response.battle_zones) == 9
        assert len(response.insights) >= 1

    def test_battle_zones_deterministic(self):
        """Battle zones should be deterministic for same input when using fallback."""
        from api.prediction_service import PredictionService

        service = PredictionService()

        # Run same prediction twice
        r1 = service.predict("Liverpool", "4-3-3", "Man City", "4-2-3-1")
        r2 = service.predict("Liverpool", "4-3-3", "Man City", "4-2-3-1")

        # Battle zones should be identical when using formation-based fallback
        # (no model loaded means deterministic formation-based generation)
        if not service.model_loaded:
            for z1, z2 in zip(r1.battle_zones, r2.battle_zones):
                assert z1.zone == z2.zone
                assert z1.advantage == z2.advantage
                # Should be exactly the same (deterministic)
                assert z1.margin == z2.margin

    def test_formation_based_zones(self):
        """Test formation-based zone generation is not random."""
        from api.prediction_service import PredictionService

        service = PredictionService()

        # Generate zones for same formations multiple times
        zones1 = service._generate_formation_based_zones("4-3-3", "4-4-2", 60)
        zones2 = service._generate_formation_based_zones("4-3-3", "4-4-2", 60)

        # Should be identical (deterministic)
        assert zones1 == zones2

    def test_different_formations_different_zones(self):
        """Test different formations produce different zone patterns."""
        from api.prediction_service import PredictionService

        service = PredictionService()

        zones_433 = service._generate_formation_based_zones("4-3-3", "4-4-2", 50)
        zones_532 = service._generate_formation_based_zones("5-3-2", "4-4-2", 50)

        # Different formations should have different zone patterns
        assert zones_433 != zones_532


# =============================================================================
# Test insights generation
# =============================================================================


class TestInsightsGeneration:
    """Test LLM-powered insights generation."""

    def test_insights_generator_initialization(self):
        """Test insights generator initializes."""
        from api.insights_generator import InsightsGenerator

        generator = InsightsGenerator()
        assert generator is not None

    def test_fallback_insights(self):
        """Test rule-based fallback insights work."""
        from api.insights_generator import InsightsGenerator

        generator = InsightsGenerator()

        # Force fallback (no API key)
        insights = generator._generate_fallback_insights(
            team_a="Arsenal",
            formation_a="4-3-3",
            team_b="Chelsea",
            formation_b="4-2-3-1",
            dominance_score=65,
        )

        assert len(insights) >= 1
        assert len(insights) <= 5
        for insight in insights:
            assert insight.category in ["formation", "tactical", "historical", "key_factor"]
            assert len(insight.text) > 10

    def test_insights_caching(self):
        """Test insights are cached."""
        from api.insights_generator import InsightsGenerator

        generator = InsightsGenerator()

        # Same request should be cached
        cache_key = "Arsenal:4-3-3:Chelsea:4-4-2:60"

        # Generate once
        insights1 = generator._generate_fallback_insights(
            "Arsenal", "4-3-3", "Chelsea", "4-4-2", 60
        )
        generator._cache_insights(cache_key, insights1)

        # Should get cached version
        cached = generator._get_cached_insights(cache_key)
        assert cached is not None
        assert len(cached) == len(insights1)


# =============================================================================
# Test dataset creation
# =============================================================================


class TestDatasetCreation:
    """Test dataset creation from different sources."""

    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        from data.dataset import MatchupDataset

        dataset = MatchupDataset.create_synthetic(n_samples=50)

        assert len(dataset) == 50

        # Check first sample
        sample = dataset[0]
        assert len(sample) == 3  # graph_a, graph_b, label
        graph_a, graph_b, label = sample

        assert graph_a.x.shape == (11, 12)
        assert graph_b.x.shape == (11, 12)
        assert 0 <= label <= 1

    def test_statsbomb_dataset(self):
        """Test StatsBomb dataset creation."""
        from data.dataset import MatchupDataset

        dataset = MatchupDataset.from_statsbomb(
            competition_id=2, season_id=27, max_matches=10
        )

        # Should have at least a few matches
        assert len(dataset) > 0

    def test_dataset_split(self):
        """Test dataset train/val splitting."""
        from data.dataset import MatchupDataset

        dataset = MatchupDataset.create_synthetic(n_samples=100)
        train, val = dataset.split(val_ratio=0.2, shuffle=True, seed=42)

        assert len(train) == 80
        assert len(val) == 20

        # Check they're different
        train_sample = train[0]
        val_sample = val[0]
        # Should not be the exact same reference
        assert train_sample is not val_sample


# =============================================================================
# Test graph builder
# =============================================================================


class TestGraphBuilder:
    """Test graph builder with enhanced features."""

    def test_graph_builder_initialization(self):
        """Test graph builder initializes."""
        from data.graph_builder import FormationGraphBuilder

        builder = FormationGraphBuilder()
        assert builder is not None

    def test_build_graph_basic(self):
        """Test building basic graph from formation."""
        from data.graph_builder import FormationGraphBuilder

        builder = FormationGraphBuilder()
        graph = builder.build_graph(team="Arsenal", formation="4-3-3")

        assert graph.x.shape[0] == 11  # 11 players
        assert graph.x.shape[1] == 12  # 12 features
        assert graph.edge_attr.shape[1] == 6  # 6 edge features

    def test_node_features_content(self):
        """Test node features contain expected data."""
        from data.graph_builder import FormationGraphBuilder

        builder = FormationGraphBuilder()
        graph = builder.build_graph(team="Arsenal", formation="4-3-3")

        # Check position features (first 2 dims)
        positions = graph.x[:, :2]
        assert positions.min() >= 0
        assert positions.max() <= 1

        # Check position type one-hot (dims 2-6)
        pos_types = graph.x[:, 2:6]
        assert (pos_types.sum(dim=1) == 1).all()  # Each row sums to 1

    def test_edge_features_content(self):
        """Test edge features contain expected data."""
        from data.graph_builder import FormationGraphBuilder

        builder = FormationGraphBuilder()
        graph = builder.build_graph(team="Arsenal", formation="4-3-3")

        # Check edge features are in valid ranges
        edge_feats = graph.edge_attr

        # Distance (normalized)
        distances = edge_feats[:, 0]
        assert distances.min() >= 0
        assert distances.max() <= 1

        # Same line (binary)
        same_line = edge_feats[:, 1]
        assert ((same_line == 0) | (same_line == 1)).all()


# =============================================================================
# Test trainer
# =============================================================================


class TestTrainer:
    """Test training infrastructure."""

    def test_trainer_initialization(self):
        """Test trainer initializes with model and data."""
        from models.gnn import SimpleTacticsGNN
        from models.trainer import TacticsTrainer, create_paired_dataloader
        from data.dataset import MatchupDataset

        # Create small dataset
        dataset = MatchupDataset.create_synthetic(n_samples=20)
        train, val = dataset.split(val_ratio=0.2)

        train_loader = create_paired_dataloader(train, batch_size=4)
        val_loader = create_paired_dataloader(val, batch_size=4)

        model = SimpleTacticsGNN(node_features=12, edge_features=6)
        trainer = TacticsTrainer(model, train_loader, val_loader)

        assert trainer is not None
        assert trainer.model is not None

    def test_loss_computation(self):
        """Test multi-task loss computation."""
        from models.trainer import TacticsTrainer, TrainerConfig
        from models.gnn import SimpleTacticsGNN
        from data.dataset import MatchupDataset

        dataset = MatchupDataset.create_synthetic(n_samples=10)
        train, val = dataset.split(val_ratio=0.2)

        from models.trainer import create_paired_dataloader

        train_loader = create_paired_dataloader(train, batch_size=4)
        val_loader = create_paired_dataloader(val, batch_size=4)

        model = SimpleTacticsGNN(node_features=12, edge_features=6)
        config = TrainerConfig(epochs=1)
        trainer = TacticsTrainer(model, train_loader, val_loader, config)

        # Test loss computation
        predictions = {
            "dominance": torch.tensor([0.6, 0.4]),
            "confidence": torch.tensor([0.8, 0.7]),
        }
        targets = {"dominance": torch.tensor([0.7, 0.3])}

        loss, components = trainer.compute_total_loss(predictions, targets)

        assert loss.item() > 0
        assert "dominance" in components
        assert "confidence" in components


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
