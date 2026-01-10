#!/usr/bin/env python3
"""
TacticsAI Dataset Check Script

This script validates that the dataset module is working correctly
by creating a small dataset from StatsBomb data and checking its structure.
"""

import sys
from pathlib import Path

# Add the data module to the path (assuming this script is in the project root)
# Adjust the path based on your actual project structure
sys.path.insert(0, str(Path(__file__).parent))

# Try importing with proper error handling
try:
    from data.dataset import MatchupDataset
    from data.graph_builder import FormationGraphBuilder
    from data.config import list_formations
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nMake sure:")
    print("  1. You have a 'data' directory with __init__.py")
    print("  2. All required files are in the 'data' directory:")
    print("     - dataset.py")
    print("     - graph_builder.py")
    print("     - config.py")
    print("     - fetchers.py")
    print("     - api_client.py")
    print("     - __init__.py")
    print("  3. Required packages are installed:")
    print("     - torch")
    print("     - torch_geometric")
    print("     - pandas")
    print("     - numpy")
    print("     - statsbombpy")
    print("     - loguru")
    sys.exit(1)

def main():
    """Run the dataset check."""
    print("\n" + "=" * 70)
    print("TacticsAI Dataset Check")
    print("=" * 70)
    
    # List available formations
    print("\n📋 Available formations:")
    formations = list_formations()
    for formation in formations:
        print(f"   - {formation}")
    
    # Create dataset from StatsBomb
    print("\n📊 Creating dataset from StatsBomb...")
    try:
        dataset = MatchupDataset.from_statsbomb(max_matches=10)
        print(f"✓ Dataset created successfully")
    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check dataset size
    print(f"\n📈 Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("⚠️  Warning: Dataset is empty. This might be because:")
        print("   - StatsBomb data is not available")
        print("   - Network connection issues")
        print("   - API rate limiting")
        return
    
    # Get first sample
    print("\n🔍 Examining first sample...")
    try:
        sample = dataset[0]
        print(f"✓ Sample retrieved successfully")
    except Exception as e:
        print(f"✗ Failed to retrieve sample: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check sample structure
    print(f"\n📦 Sample keys: {list(sample.keys())}")
    
    # Check Graph A
    print(f"\n🏠 Graph A (Home Team):")
    print(f"   Team: {sample['metadata']['home_team']}")
    print(f"   Formation: {sample['metadata']['home_formation']}")
    print(f"   Nodes: {sample['graph_a'].num_nodes}")
    print(f"   Edges: {sample['graph_a'].num_edges}")
    print(f"   Node features shape: {sample['graph_a'].x.shape}")
    print(f"   Edge features shape: {sample['graph_a'].edge_attr.shape}")
    
    # Validate Graph A
    assert sample['graph_a'].num_nodes == 11, "Graph A should have 11 nodes"
    assert sample['graph_a'].x.shape[1] == 12, "Node features should have 12 dimensions (enhanced)"
    assert sample['graph_a'].edge_attr.shape[1] == 6, "Edge features should have 6 dimensions (enhanced)"
    print("   ✓ Graph A structure validated (12D nodes, 6D edges)")
    
    # Check Graph B
    print(f"\n✈️  Graph B (Away Team):")
    print(f"   Team: {sample['metadata']['away_team']}")
    print(f"   Formation: {sample['metadata']['away_formation']}")
    print(f"   Nodes: {sample['graph_b'].num_nodes}")
    print(f"   Edges: {sample['graph_b'].num_edges}")
    print(f"   Node features shape: {sample['graph_b'].x.shape}")
    print(f"   Edge features shape: {sample['graph_b'].edge_attr.shape}")
    
    # Validate Graph B
    assert sample['graph_b'].num_nodes == 11, "Graph B should have 11 nodes"
    assert sample['graph_b'].x.shape[1] == 12, "Node features should have 12 dimensions (enhanced)"
    assert sample['graph_b'].edge_attr.shape[1] == 6, "Edge features should have 6 dimensions (enhanced)"
    print("   ✓ Graph B structure validated (12D nodes, 6D edges)")
    
    # Check label
    if sample['label'] is not None:
        label_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        print(f"\n🎯 Match Result: {label_map[sample['label']]}")
        print(f"   Score: {sample['metadata']['home_team']} {sample['metadata']['home_score']} - "
              f"{sample['metadata']['away_score']} {sample['metadata']['away_team']}")
    
    # Get dataset statistics
    print("\n📊 Dataset Statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"      {k}: {v}")
        else:
            print(f"   {key}: {value}")
    
    # Test iteration
    print("\n🔄 Testing dataset iteration...")
    try:
        for i, sample in enumerate(dataset):
            if i >= 3:  # Just test first 3
                break
            print(f"   Sample {i}: {sample['metadata']['home_team']} vs {sample['metadata']['away_team']}")
        print("   ✓ Dataset iteration works")
    except Exception as e:
        print(f"   ✗ Iteration failed: {e}")
    
    print("\n" + "=" * 70)
    print("✅ All checks passed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
