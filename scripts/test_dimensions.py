#!/usr/bin/env python3
"""
Quick test to verify dimension fix for 12D nodes and 6D edges.
This script tests the core graph creation functions without requiring all dependencies.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_create_tactical_graph():
    """Test create_tactical_graph produces 12D nodes and 6D edges."""
    print("Testing create_tactical_graph()...")

    from data.dataset import create_tactical_graph

    # Create sample positions (11 players)
    positions = [
        (10, 10),   # GK
        (20, 30),   # DEF
        (30, 30),   # DEF
        (40, 30),   # DEF
        (50, 30),   # DEF
        (30, 50),   # MID
        (40, 50),   # MID
        (50, 50),   # MID
        (30, 70),   # FWD
        (40, 70),   # FWD
        (50, 70),   # FWD
    ]

    graph = create_tactical_graph(positions)

    # Check dimensions
    node_dim = graph.x.shape[1]
    edge_dim = graph.edge_attr.shape[1]

    print(f"  Node features: {node_dim}D (expected: 12D)")
    print(f"  Edge features: {edge_dim}D (expected: 6D)")
    print(f"  Nodes: {graph.x.shape[0]} (expected: 11)")

    assert node_dim == 12, f"Expected 12D node features, got {node_dim}D"
    assert edge_dim == 6, f"Expected 6D edge features, got {edge_dim}D"
    assert graph.x.shape[0] == 11, f"Expected 11 nodes, got {graph.x.shape[0]}"

    print("  ✓ create_tactical_graph() dimensions correct!\n")
    return True


def test_formation_to_graph():
    """Test formation_to_graph produces 12D nodes and 6D edges."""
    print("Testing formation_to_graph()...")

    from data.dataset import formation_to_graph
    from data.config import FORMATIONS

    # Test with 4-3-3 formation
    formation = FORMATIONS["4-3-3"]
    graph = formation_to_graph(formation)

    # Check dimensions
    node_dim = graph.x.shape[1]
    edge_dim = graph.edge_attr.shape[1]

    print(f"  Node features: {node_dim}D (expected: 12D)")
    print(f"  Edge features: {edge_dim}D (expected: 6D)")
    print(f"  Nodes: {graph.x.shape[0]} (expected: 11)")

    assert node_dim == 12, f"Expected 12D node features, got {node_dim}D"
    assert edge_dim == 6, f"Expected 6D edge features, got {edge_dim}D"
    assert graph.x.shape[0] == 11, f"Expected 11 nodes, got {graph.x.shape[0]}"

    print("  ✓ formation_to_graph() dimensions correct!\n")
    return True


def test_synthetic_dataset():
    """Test synthetic dataset produces correct dimensions."""
    print("Testing synthetic dataset...")

    from data.dataset import MatchupDataset

    # Create small synthetic dataset
    dataset = MatchupDataset.create_synthetic(n_samples=5)

    print(f"  Dataset size: {len(dataset)}")

    # Check first sample
    sample = dataset[0]
    graph_a = sample[0] if isinstance(sample, tuple) else sample.get('graph_a')
    graph_b = sample[1] if isinstance(sample, tuple) else sample.get('graph_b')

    # Check dimensions
    node_dim_a = graph_a.x.shape[1]
    edge_dim_a = graph_a.edge_attr.shape[1]
    node_dim_b = graph_b.x.shape[1]
    edge_dim_b = graph_b.edge_attr.shape[1]

    print(f"  Graph A: {node_dim_a}D nodes, {edge_dim_a}D edges")
    print(f"  Graph B: {node_dim_b}D nodes, {edge_dim_b}D edges")

    assert node_dim_a == 12, f"Expected 12D node features for graph_a, got {node_dim_a}D"
    assert edge_dim_a == 6, f"Expected 6D edge features for graph_a, got {edge_dim_a}D"
    assert node_dim_b == 12, f"Expected 12D node features for graph_b, got {node_dim_b}D"
    assert edge_dim_b == 6, f"Expected 6D edge features for graph_b, got {edge_dim_b}D"

    print("  ✓ Synthetic dataset dimensions correct!\n")
    return True


def main():
    print("=" * 60)
    print("TacticsAI Dimension Fix Verification")
    print("=" * 60)
    print()

    try:
        # Run tests
        test_create_tactical_graph()
        test_formation_to_graph()
        test_synthetic_dataset()

        print("=" * 60)
        print("✓ ALL DIMENSION CHECKS PASSED!")
        print("=" * 60)
        print()
        print("The dimension mismatch has been fixed:")
        print("  - Node features: 12D ✓")
        print("  - Edge features: 6D ✓")
        print("  - All graph creation functions consistent ✓")
        print()
        return 0

    except AssertionError as e:
        print("=" * 60)
        print("✗ DIMENSION CHECK FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        return 1
    except Exception as e:
        print("=" * 60)
        print("✗ TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
