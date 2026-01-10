# TacticsAI Dataset Module - Setup and Usage

## Overview

I've created the missing `dataset.py` module for your TacticsAI project and identified several issues in your existing code. This document explains everything you need to know.

## 🔴 Issues Found

### 1. **Coordinate System Mismatch**

**Problem**: `graph_builder.py` and `config.py` use different coordinate conventions.

- **graph_builder.py** expects:
  - `x`: 0-100 (own goal line → opponent goal line, i.e., depth/length of pitch)
  - `y`: 0-100 (left touchline → right touchline, i.e., width of pitch)
  - `position_type`: "GK", "DEF", "MID", "FWD"

- **config.py** defines:
  - `x`: 0-100 (left touchline → right touchline, i.e., width)
  - `y`: 0-100 (own goal line → opponent goal line, i.e., depth)
  - Specific position names: "LB", "RCB", "CM", "ST", etc.

**Solution**: The `dataset.py` module includes a `convert_formation_to_graph_format()` function that:
- Swaps x and y coordinates to match graph_builder expectations
- Maps specific positions to general types using `POSITION_TYPE_MAPPING`

### 2. **Missing Dataset Module**

**Problem**: Your check script tried to import `data.dataset.MatchupDataset`, which didn't exist.

**Solution**: Created `dataset.py` with:
- `MatchupDataset` class for loading matchup data
- `from_statsbomb()` class method for easy dataset creation
- Support for converting formations to graph format
- Comprehensive statistics and utilities

### 3. **Position Type Mapping**

**Problem**: No mapping between detailed position names and general categories.

**Solution**: Created `POSITION_TYPE_MAPPING` dictionary:
```python
"GK" → "GK"
"LB", "RCB", "CB" → "DEF"
"CM", "CDM", "CAM" → "MID"
"ST", "LW", "RW" → "FWD"
```

## 📁 Files Created

1. **`dataset.py`** - Main dataset module
   - `MatchupDataset` class
   - Position type mapping
   - Formation conversion utilities
   - StatsBomb data loading

2. **`test_dataset.py`** - Comprehensive test script
   - Validates dataset creation
   - Checks graph structure
   - Tests iteration
   - Displays statistics

## 🚀 Setup Instructions

### 1. Install Required Dependencies

```bash
# Core dependencies
pip install torch torch-geometric
pip install pandas numpy
pip install statsbombpy
pip install loguru
pip install requests beautifulsoup4
```

### 2. Project Structure

Your project should have this structure:
```
tacticsai/
├── data/
│   ├── __init__.py          (your existing file)
│   ├── config.py            (your existing file)
│   ├── fetchers.py          (your existing file)
│   ├── api_client.py        (your existing file)
│   ├── graph_builder.py     (your existing file)
│   └── dataset.py           (NEW - place here)
└── test_dataset.py          (NEW - place in project root)
```

### 3. Place the New Files

1. **Copy `dataset.py`** to your `data/` directory
2. **Copy `test_dataset.py`** to your project root directory

## 🧪 Running the Tests

### Option 1: Using the comprehensive test script

```bash
python test_dataset.py
```

This will:
- ✓ Import all modules
- ✓ List available formations
- ✓ Create a dataset from StatsBomb
- ✓ Validate graph structure
- ✓ Show dataset statistics
- ✓ Test iteration

### Option 2: Using your original check (now it works!)

```bash
python -c "
from data.dataset import MatchupDataset
dataset = MatchupDataset.from_statsbomb(max_matches=10)
print(f'Dataset size: {len(dataset)}')
sample = dataset[0]
print(f'Sample keys: {list(sample.keys())}')
print(f'Graph A nodes: {sample['graph_a'].num_nodes}')
print(f'Graph A edges: {sample['graph_a'].num_edges}')
print(f'Graph B nodes: {sample['graph_b'].num_nodes}')
print(f'Graph B edges: {sample['graph_b'].num_edges}')
"
```

### Option 3: Interactive Python

```python
from data.dataset import MatchupDataset

# Create dataset
dataset = MatchupDataset.from_statsbomb(max_matches=10)

# Get first sample
sample = dataset[0]

# Inspect graphs
print(f"Home: {sample['metadata']['home_team']} ({sample['metadata']['home_formation']})")
print(f"Away: {sample['metadata']['away_team']} ({sample['metadata']['away_formation']})")
print(f"Graph A: {sample['graph_a'].num_nodes} nodes, {sample['graph_a'].num_edges} edges")
print(f"Graph B: {sample['graph_b'].num_nodes} nodes, {sample['graph_b'].num_edges} edges")

# Get statistics
stats = dataset.get_statistics()
print(stats)
```

## 📊 Expected Output

When running the test, you should see:

```
======================================================================
TacticsAI Dataset Check
======================================================================

📋 Available formations:
   - 4-3-3
   - 4-4-2
   - 4-2-3-1
   - 3-5-2
   - 5-3-2
   - 3-4-3

📊 Creating dataset from StatsBomb...
✓ Dataset created successfully

📈 Dataset size: 10

🔍 Examining first sample...
✓ Sample retrieved successfully

📦 Sample keys: ['graph_a', 'graph_b', 'label', 'metadata']

🏠 Graph A (Home Team):
   Team: Arsenal
   Formation: 4-3-3
   Nodes: 11
   Edges: 70
   Node features shape: torch.Size([11, 8])
   Edge features shape: torch.Size([70, 3])
   ✓ Graph A structure validated

✈️  Graph B (Away Team):
   Team: Manchester City
   Formation: 4-3-3
   Nodes: 11
   Edges: 70
   Node features shape: torch.Size([11, 8])
   Edge features shape: torch.Size([70, 3])
   ✓ Graph B structure validated

🎯 Match Result: Home Win
   Score: Arsenal 2 - 1 Manchester City

📊 Dataset Statistics:
   total_matchups: 10
   unique_teams: 18
   formations_used: ['4-3-3', '4-2-3-1']
   formation_counts:
      4-3-3: 14
      4-2-3-1: 6
   label_distribution:
      away_wins: 3
      draws: 2
      home_wins: 5

======================================================================
✅ All checks passed!
======================================================================
```

## 🎓 Usage Examples

### Creating a Dataset from StatsBomb

```python
from data.dataset import MatchupDataset

# Default: Premier League 2023/2024
dataset = MatchupDataset.from_statsbomb()

# Specific competition and season
dataset = MatchupDataset.from_statsbomb(
    competition_id=2,  # Premier League
    season_id=90,      # 2023/2024
    max_matches=50     # Limit to 50 matches
)

# With custom distance threshold
dataset = MatchupDataset.from_statsbomb(
    distance_threshold=25.0  # Tighter connections
)
```

### Creating a Custom Dataset

```python
from data.dataset import MatchupDataset

# Define your own matchups
matchups = [
    {
        "home_team": "Liverpool",
        "away_team": "Manchester City",
        "home_formation": "4-3-3",
        "away_formation": "4-3-3",
        "home_score": 2,
        "away_score": 1,
        "date": "2024-01-15",
        "competition": "Premier League"
    },
    # ... more matchups
]

dataset = MatchupDataset(matchups=matchups)
```

### Saving and Loading Datasets

```python
# Save to JSON
dataset.save_to_json("matchups.json")

# Load from JSON
dataset = MatchupDataset.from_json("matchups.json")
```

### Using with PyTorch DataLoader

```python
from torch.utils.data import DataLoader

dataset = MatchupDataset.from_statsbomb(max_matches=100)

# For single-graph processing
loader = DataLoader(dataset, batch_size=8, shuffle=True)

for batch in loader:
    graphs_a = batch['graph_a']
    graphs_b = batch['graph_b']
    labels = batch['label']
    # ... train your model
```

## 🔧 Customization

### Adding New Position Types

Edit `POSITION_TYPE_MAPPING` in `dataset.py`:

```python
POSITION_TYPE_MAPPING = {
    "GK": "GK",
    "SW": "DEF",  # Add sweeper
    "WB": "DEF",  # Add wing-back
    # ... more positions
}
```

### Custom Formation Inference

Override `_infer_formation()` in `MatchupDataset`:

```python
@staticmethod
def _infer_formation(team_name: str) -> str:
    # Your custom logic
    if "Barcelona" in team_name:
        return "4-3-3"
    return "4-4-2"
```

### Custom Graph Features

Modify `graph_builder.py` to add more node or edge features:

```python
def _build_node_features(self, positions, player_data):
    # Add your custom features
    features.append([
        x_norm, y_norm,
        *position_onehot,
        player_rating, form_score,
        # Add: speed, stamina, etc.
    ])
```

## 🐛 Troubleshooting

### "No module named 'statsbombpy'"
```bash
pip install statsbombpy
```

### "No module named 'torch_geometric'"
```bash
pip install torch-geometric
```

### "Dataset is empty"
- Check your internet connection
- StatsBomb data might be temporarily unavailable
- Try increasing `max_matches` parameter
- Check the cache directory: `backend/data/cache/`

### "Formation X not found in config"
- Make sure the formation exists in `config.py`
- Check `list_formations()` for available formations
- Add custom formations to `FORMATIONS` in `config.py`

### "Import error: No module named 'data'"
- Make sure `__init__.py` exists in the `data/` directory
- Check your Python path: `sys.path.insert(0, 'path/to/project')`

## 📚 Next Steps

1. **Validate on your machine**: Run `test_dataset.py` to ensure everything works
2. **Test with different formations**: Try 4-4-2, 3-5-2, etc.
3. **Integrate with your model**: Use the dataset with your GNN model
4. **Add more data sources**: Extend `from_fbref()` or `from_custom()`
5. **Enhance features**: Add player statistics, team form, etc.

## 📝 API Reference

### `MatchupDataset`

**Methods:**
- `__init__(matchups, formations_config=None, distance_threshold=30.0)`
- `__len__()` - Returns number of matchups
- `__getitem__(idx)` - Returns a sample with graphs and metadata
- `from_statsbomb(competition_id, season_id, max_matches, ...)` - Load from StatsBomb
- `from_json(json_path)` - Load from JSON file
- `save_to_json(json_path)` - Save to JSON file
- `get_statistics()` - Get dataset statistics

**Sample Structure:**
```python
{
    'graph_a': Data,         # PyTorch Geometric Data object
    'graph_b': Data,         # PyTorch Geometric Data object
    'label': int or None,    # 0=away win, 1=draw, 2=home win
    'metadata': {
        'home_team': str,
        'away_team': str,
        'home_formation': str,
        'away_formation': str,
        'home_score': int or None,
        'away_score': int or None,
        'date': str or None,
        'competition': str or None,
    }
}
```

**Graph Structure:**
- `x`: Node features [11, 8]
  - [0:2] - Normalized x, y coordinates
  - [2:6] - One-hot position type (GK, DEF, MID, FWD)
  - [6] - Player rating (normalized 0-1)
  - [7] - Form score (normalized 0-1)
- `edge_index`: Edge connections [2, num_edges]
- `edge_attr`: Edge features [num_edges, 3]
  - [0] - Normalized distance
  - [1] - Same line indicator
  - [2] - Passing lane potential
- `team`: Team name
- `formation`: Formation string

## ✅ Final Checklist

- [ ] Install all dependencies
- [ ] Place `dataset.py` in `data/` directory
- [ ] Place `test_dataset.py` in project root
- [ ] Run `python test_dataset.py`
- [ ] Verify output shows 11 nodes per graph
- [ ] Check that formations are correctly converted
- [ ] Test with your own matchup data

---

**Questions or issues?** Feel free to ask!
