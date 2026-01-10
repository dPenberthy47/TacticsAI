"""
Graph Builder for TacticsAI GNN

This module converts football formation data into graph structures suitable for
Graph Neural Networks using PyTorch Geometric.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union
import torch
from torch_geometric.data import Data


# =============================================================================
# Position Type Mapping
# =============================================================================

# Maps specific position names (LB, RCB, CM, ST, etc.) to general types (DEF, MID, FWD, GK)
POSITION_TYPE_MAP = {
    # Goalkeepers
    "GK": "GK",

    # Defenders
    "LB": "DEF",
    "LCB": "DEF",
    "CB": "DEF",
    "RCB": "DEF",
    "RB": "DEF",
    "LWB": "DEF",
    "RWB": "DEF",

    # Midfielders
    "LM": "MID",
    "LCM": "MID",
    "CM": "MID",
    "CDM": "MID",
    "LCDM": "MID",
    "RCDM": "MID",
    "RCM": "MID",
    "RM": "MID",
    "LAM": "MID",
    "CAM": "MID",
    "RAM": "MID",

    # Forwards
    "LW": "FWD",
    "ST": "FWD",
    "LST": "FWD",
    "RST": "FWD",
    "RW": "FWD",
}


class FormationGraphBuilder:
    """
    Builds graph representations of football formations for GNN processing.

    Converts formation configurations and player data into PyTorch Geometric
    Data objects with nodes representing players and edges representing
    tactical relationships.

    Attributes:
        formations_config (dict): Configuration containing formation positions
        distance_threshold (float): Maximum distance for connecting nodes
    """

    def __init__(self, formations_config: Union[dict, object], distance_threshold: float = 30.0):
        """
        Initialize the FormationGraphBuilder.

        Args:
            formations_config: Dictionary or object mapping formations to player positions.
                Can be either:
                1. Dict format: {"4-3-3": {"positions": [{"x": 5, "y": 50, "position_type": "GK"}, ...]}}
                2. Dataclass format from config.py: Dict[str, Formation] where Formation has .positions attribute
            distance_threshold (float): Distance threshold for connecting nearby players
        """
        self.formations_config = formations_config
        self.distance_threshold = distance_threshold

        # Position type to one-hot mapping
        self.position_types = ["GK", "DEF", "MID", "FWD"]
        self.position_to_idx = {pos: idx for idx, pos in enumerate(self.position_types)}

    def build_graph(
        self,
        team: str,
        formation: str,
        player_data: Optional[pd.DataFrame] = None,
        team_stats: Optional[Dict] = None,
        player_values: Optional[pd.DataFrame] = None,
        passing_network: Optional[Dict] = None,
    ) -> Data:
        """
        Build a graph representation for a team's formation.

        Args:
            team (str): Team name
            formation (str): Formation string (e.g., "4-3-3")
            player_data (pd.DataFrame, optional): Player statistics with columns:
                - 'rating': Player rating (0-100)
                - 'form': Form score (0-100)
            team_stats (dict, optional): Team-level statistics from FBref:
                - 'xg': Expected goals per game
                - 'xga': Expected goals against per game
                - 'possession': Average possession percentage
                - 'wins', 'draws', 'losses': Recent form
            player_values (pd.DataFrame, optional): Player market values from TransferMarkt:
                - 'market_value_eur': Market value in euros
            passing_network (dict, optional): Passing network from StatsBomb events:
                - Keys: (passer_position, receiver_position) tuples
                - Values: dict with pass_count, pass_success_rate, avg_pass_distance, progressive_pass_count

        Returns:
            Data: PyTorch Geometric Data object with:
                - x: Node features [11, 12]
                - edge_index: Edge connections [2, num_edges]
                - edge_attr: Edge features [num_edges, 6]
                - team: Team name
                - formation: Formation string
        """
        # Get formation configuration
        if formation not in self.formations_config:
            raise ValueError(f"Formation {formation} not found in config")

        formation_config = self.formations_config[formation]
        positions = self._get_positions(formation_config)

        if len(positions) != 11:
            raise ValueError(f"Formation must have exactly 11 positions, got {len(positions)}")

        # Build node features with enriched data
        node_features = self._build_node_features(
            positions,
            player_data,
            team_stats,
            player_values
        )

        # Build edges and edge features with passing network
        edge_index, edge_attr = self._build_edges(positions, passing_network)

        # Create PyTorch Geometric Data object
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            team=team,
            formation=formation
        )

        return data

    def _get_positions(self, formation_config: Union[Dict, object]) -> List[Dict]:
        """
        Extract positions from formation config (handles both dict and dataclass formats).

        Args:
            formation_config: Either a dict with "positions" key or a Formation dataclass

        Returns:
            List of position dictionaries with "x", "y", "position_type" keys
        """
        # Check if it's a dict with "positions" key
        if isinstance(formation_config, dict) and "positions" in formation_config:
            positions = formation_config["positions"]
            # Convert each position if needed
            return [self._convert_position(pos) for pos in positions]

        # Check if it's a dataclass/object with positions attribute (from config.py)
        if hasattr(formation_config, "positions"):
            positions = formation_config.positions
            # Convert Position dataclasses to dict format
            return [self._convert_position(pos) for pos in positions]

        raise ValueError(f"Unknown formation config format: {type(formation_config)}")

    def _convert_position(self, pos: Union[Dict, object]) -> Dict:
        """
        Convert a position to the standardized dictionary format.

        Handles both:
        1. Dict format: {"x": float, "y": float, "position_type": str}
        2. Position dataclass from config.py: Position(name, x, y)

        Note: config.py uses x for width and y for depth, but graph_builder
        expects x for depth and y for width, so we swap them.

        Args:
            pos: Position dict or Position dataclass

        Returns:
            Dictionary with "x", "y", "position_type" keys
        """
        # If already a dict with the right keys, validate and return
        if isinstance(pos, dict):
            if "position_type" in pos:
                # Already in the right format
                return pos
            elif "name" in pos:
                # Dict from converted dataclass, needs position_type mapping
                position_type = POSITION_TYPE_MAP.get(pos["name"], "MID")
                # Note: if this dict came from config.py conversion, x and y might already be swapped
                # Check if it has both x and y
                if "x" in pos and "y" in pos:
                    return {
                        "x": pos["x"],
                        "y": pos["y"],
                        "position_type": position_type,
                        "name": pos.get("name", "")
                    }

        # If it's a dataclass/object with name, x, y attributes (from config.py)
        if hasattr(pos, "name") and hasattr(pos, "x") and hasattr(pos, "y"):
            # Map position name to type
            position_type = POSITION_TYPE_MAP.get(pos.name, "MID")

            # IMPORTANT: config.py uses x for width (0-100) and y for depth (0-100)
            # but graph_builder expects x for depth and y for width, so we swap
            return {
                "x": pos.y,  # config.py y (depth) -> graph x
                "y": pos.x,  # config.py x (width) -> graph y
                "position_type": position_type,
                "name": pos.name
            }

        # Fallback: assume it's already in the right format
        return pos

    def _build_node_features(
        self,
        positions: List[Dict],
        player_data: Optional[pd.DataFrame],
        team_stats: Optional[Dict],
        player_values: Optional[pd.DataFrame],
    ) -> np.ndarray:
        """
        Build node feature matrix for all players with enriched data.

        Args:
            positions (list): List of position dictionaries with x, y, position_type
            player_data (pd.DataFrame, optional): Player statistics
            team_stats (dict, optional): Team-level statistics
            player_values (pd.DataFrame, optional): Player market values

        Returns:
            np.ndarray: Node features array of shape [11, 12]

        Feature vector (12 dimensions):
        [0-1]: Normalized x, y position
        [2-5]: Position type one-hot (GK, DEF, MID, FWD)
        [6]: Player quality score (from market value, normalized 0-1)
        [7]: Team form score (from recent results, normalized 0-1)
        [8]: Team attacking strength (from xG per game, normalized 0-1)
        [9]: Team defensive strength (from xGA per game, normalized 0-1)
        [10]: Team possession tendency (from avg possession, normalized 0-1)
        [11]: Team pressing intensity (from PPDA if available, else 0.5)
        """
        # Calculate team-level features once (same for all players)
        team_form = self._calculate_team_form(team_stats)
        team_attack = self._calculate_attacking_strength(team_stats)
        team_defense = self._calculate_defensive_strength(team_stats)
        team_possession = self._calculate_possession_tendency(team_stats)
        team_pressing = self._calculate_pressing_intensity(team_stats)

        node_features = []

        for idx, pos in enumerate(positions):
            # Normalize positions (assuming field is 100x100)
            x_norm = pos["x"] / 100.0
            y_norm = pos["y"] / 100.0

            # One-hot encode position type
            position_type = pos["position_type"]
            position_onehot = np.zeros(4)
            if position_type in self.position_to_idx:
                position_onehot[self.position_to_idx[position_type]] = 1.0

            # Get player quality from market value
            player_quality = 0.5  # Default
            if player_values is not None and idx < len(player_values):
                market_value = player_values.iloc[idx].get("market_value_eur", 0)
                player_quality = self._normalize_market_value(market_value)

            # Combine all features: [x, y, pos_onehot(4), quality, form, attack, defense, possession, pressing]
            features = np.concatenate([
                [x_norm, y_norm],           # Position (2)
                position_onehot,             # Position type (4)
                [player_quality],            # Player quality (1)
                [team_form],                 # Team form (1)
                [team_attack],               # Attacking strength (1)
                [team_defense],              # Defensive strength (1)
                [team_possession],           # Possession tendency (1)
                [team_pressing],             # Pressing intensity (1)
            ])

            node_features.append(features)

        return np.array(node_features, dtype=np.float32)

    def _build_edges(
        self,
        positions: List[Dict],
        passing_network: Optional[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build edge index and edge features based on tactical relationships and passing data.

        Applies four rules:
        1. Connect adjacent positions (same line)
        2. Connect players within distance threshold
        3. Connect GK to all defenders
        4. Connect strikers to attacking midfielders

        Args:
            positions (list): List of position dictionaries
            passing_network (dict, optional): Passing network data from events

        Returns:
            tuple: (edge_index [2, num_edges], edge_attr [num_edges, 6])
        """
        edges = []
        edge_features = []

        num_players = len(positions)

        # Extract position arrays for distance calculations
        coords = np.array([[pos["x"], pos["y"]] for pos in positions])

        # Calculate all pairwise distances
        distances = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)

        # Warn if no passing network data
        if passing_network is None:
            import warnings
            warnings.warn(
                "No passing network data provided. Falling back to geometric heuristics for edge features.",
                UserWarning
            )

        for i in range(num_players):
            for j in range(i + 1, num_players):  # Only upper triangle to avoid duplicates
                should_connect = False

                # Rule 1: Connect adjacent positions (same line)
                if self._are_adjacent(positions[i], positions[j]):
                    should_connect = True

                # Rule 2: Connect players within distance threshold
                if distances[i, j] <= self.distance_threshold:
                    should_connect = True

                # Rule 3: Connect GK to all defenders
                if self._is_gk_to_defender(positions[i], positions[j]):
                    should_connect = True

                # Rule 4: Connect strikers to attacking midfielders
                if self._is_striker_to_attacking_mid(positions[i], positions[j]):
                    should_connect = True

                if should_connect:
                    # Add bidirectional edges
                    edges.append([i, j])
                    edges.append([j, i])

                    # Calculate edge features (now 6D with passing data)
                    edge_feat_ij = self._calculate_edge_features(
                        positions[i], positions[j], distances[i, j],
                        i, j, passing_network
                    )
                    edge_feat_ji = self._calculate_edge_features(
                        positions[j], positions[i], distances[i, j],
                        j, i, passing_network
                    )

                    edge_features.append(edge_feat_ij)
                    edge_features.append(edge_feat_ji)  # Different features for each direction

        # Convert to numpy arrays
        if len(edges) == 0:
            # If no edges, create empty arrays with correct shape
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, 6), dtype=np.float32)  # 6D features now
        else:
            edge_index = np.array(edges, dtype=np.int64).T
            edge_attr = np.array(edge_features, dtype=np.float32)

        return edge_index, edge_attr

    def _are_adjacent(self, pos1: Dict, pos2: Dict) -> bool:
        """
        Check if two positions are adjacent (on same defensive/midfield/attacking line).

        Args:
            pos1, pos2 (dict): Position dictionaries

        Returns:
            bool: True if positions are adjacent
        """
        # Adjacent if same position type and close in y-coordinate
        if pos1["position_type"] == pos2["position_type"]:
            y_diff = abs(pos1["y"] - pos2["y"])
            return y_diff < 30  # Adjacent if within 30 units in width
        return False

    def _is_gk_to_defender(self, pos1: Dict, pos2: Dict) -> bool:
        """Check if connection is between GK and a defender."""
        types = {pos1["position_type"], pos2["position_type"]}
        return types == {"GK", "DEF"}

    def _is_striker_to_attacking_mid(self, pos1: Dict, pos2: Dict) -> bool:
        """Check if connection is between a striker and attacking midfielder."""
        types = {pos1["position_type"], pos2["position_type"]}
        # FWD to MID connection where MID is advanced (x > 50)
        if types == {"FWD", "MID"}:
            mid_pos = pos1 if pos1["position_type"] == "MID" else pos2
            return mid_pos["x"] > 50  # Attacking midfielder in advanced position
        return False

    def _calculate_edge_features(
        self,
        pos1: Dict,
        pos2: Dict,
        distance: float,
        idx1: int,
        idx2: int,
        passing_network: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Calculate edge features for a connection between two players.

        Features (6 dimensions):
        [0]: Normalized Euclidean distance
        [1]: Same tactical line indicator
        [2]: Pass frequency between positions (from passing_network, normalized 0-1)
        [3]: Pass success rate between positions (from passing_network, 0-1)
        [4]: Is progressive passing lane (1.0 if >50% of passes are progressive, else 0.0)
        [5]: Defensive coverage indicator (not implemented in this method, returns 0.0)

        Args:
            pos1, pos2 (dict): Position dictionaries
            distance (float): Euclidean distance between positions
            idx1, idx2 (int): Position indices
            passing_network (dict, optional): Passing network data

        Returns:
            np.ndarray: Edge features [6]
        """
        # Feature 0: Normalized distance (max field distance ~141 for 100x100 field)
        distance_norm = distance / 141.0

        # Feature 1: Same line indicator
        same_line = 1.0 if pos1["position_type"] == pos2["position_type"] else 0.0

        # Features 2-4: From passing network if available
        if passing_network is not None:
            # Try to get passing data between these position types
            pos1_type = pos1.get("position_type", "")
            pos2_type = pos2.get("position_type", "")

            # Look up passing data (using position types as keys)
            pass_key = (pos1_type, pos2_type)
            pass_data = passing_network.get(pass_key, {})

            # Feature 2: Pass frequency (normalized by max passes)
            pass_count = pass_data.get("pass_count", 0)
            max_passes = 50  # Typical max for a position pair in a match
            pass_frequency = min(pass_count / max_passes, 1.0)

            # Feature 3: Pass success rate
            pass_success_rate = pass_data.get("pass_success_rate", 0.0)

            # Feature 4: Progressive passing lane
            progressive_count = pass_data.get("progressive_pass_count", 0)
            if pass_count > 0:
                progressive_ratio = progressive_count / pass_count
                is_progressive = 1.0 if progressive_ratio > 0.5 else 0.0
            else:
                is_progressive = 0.0

        else:
            # Fallback to geometric heuristics
            # Pass frequency heuristic: closer = more passes
            pass_frequency = max(0.0, 1.0 - distance_norm)

            # Pass success rate heuristic: short passes more successful
            pass_success_rate = 0.8 if distance_norm < 0.3 else 0.6

            # Progressive heuristic: forward passes (increasing x)
            is_forward = 1.0 if pos2["x"] > pos1["x"] else 0.0
            is_progressive = is_forward if distance_norm > 0.2 else 0.0

        # Feature 5: Defensive coverage (placeholder, would need opponent formation)
        defensive_coverage = 0.0

        return np.array([
            distance_norm,
            same_line,
            pass_frequency,
            pass_success_rate,
            is_progressive,
            defensive_coverage
        ], dtype=np.float32)

    def _normalize_market_value(self, value: float) -> float:
        """
        Normalize market value to 0-1 scale.

        Uses a logarithmic scale to handle the wide range of values.
        Max reference: ~200M euros for top players.

        Args:
            value (float): Market value in euros

        Returns:
            float: Normalized value (0-1)
        """
        if value <= 0:
            return 0.1  # Minimum quality for any player

        # Use log scale: log(1 + value/1M) / log(1 + 200)
        # This gives better distribution across price ranges
        normalized = np.log1p(value / 1_000_000) / np.log1p(200)

        # Clip to [0.1, 1.0] range
        return float(np.clip(normalized, 0.1, 1.0))

    def _calculate_team_form(self, team_stats: Optional[Dict]) -> float:
        """
        Calculate team form score from recent results.

        Form is calculated from win/draw/loss record.

        Args:
            team_stats (dict, optional): Team statistics

        Returns:
            float: Form score (0-1), where 1.0 is best
        """
        if team_stats is None:
            return 0.5  # Neutral default

        try:
            wins = float(team_stats.get("wins", 0))
            draws = float(team_stats.get("draws", 0))
            losses = float(team_stats.get("losses", 0))

            total_matches = wins + draws + losses
            if total_matches == 0:
                return 0.5

            # Form = (wins * 3 + draws * 1) / (total_matches * 3)
            points = (wins * 3 + draws * 1)
            max_points = total_matches * 3
            form = points / max_points

            return float(np.clip(form, 0.0, 1.0))

        except (KeyError, ValueError, TypeError):
            return 0.5  # Fallback

    def _calculate_attacking_strength(self, team_stats: Optional[Dict]) -> float:
        """
        Calculate attacking strength from xG per game.

        Args:
            team_stats (dict, optional): Team statistics

        Returns:
            float: Attacking strength (0-1)
        """
        if team_stats is None:
            return 0.5  # Neutral default

        try:
            # Try to get xG data
            xg = float(team_stats.get("xg", team_stats.get("xG", 0)))
            matches = float(team_stats.get("matches_played", 1))

            if matches == 0:
                return 0.5

            xg_per_game = xg / matches

            # Normalize: 0 xG = 0.0, 2.5+ xG = 1.0
            # Most teams average 1.0-1.5 xG per game
            normalized = xg_per_game / 2.5

            return float(np.clip(normalized, 0.0, 1.0))

        except (KeyError, ValueError, TypeError):
            return 0.5  # Fallback

    def _calculate_defensive_strength(self, team_stats: Optional[Dict]) -> float:
        """
        Calculate defensive strength from xGA per game.

        Lower xGA = better defense = higher score.

        Args:
            team_stats (dict, optional): Team statistics

        Returns:
            float: Defensive strength (0-1), where 1.0 is best
        """
        if team_stats is None:
            return 0.5  # Neutral default

        try:
            # Try to get xGA data
            xga = float(team_stats.get("xga", team_stats.get("xGA", 0)))
            matches = float(team_stats.get("matches_played", 1))

            if matches == 0:
                return 0.5

            xga_per_game = xga / matches

            # Normalize: 0 xGA = 1.0, 2.5+ xGA = 0.0
            # Lower xGA is better, so invert
            normalized = 1.0 - (xga_per_game / 2.5)

            return float(np.clip(normalized, 0.0, 1.0))

        except (KeyError, ValueError, TypeError):
            return 0.5  # Fallback

    def _calculate_possession_tendency(self, team_stats: Optional[Dict]) -> float:
        """
        Calculate possession tendency from average possession.

        Args:
            team_stats (dict, optional): Team statistics

        Returns:
            float: Possession tendency (0-1)
        """
        if team_stats is None:
            return 0.5  # Neutral default

        try:
            possession = float(team_stats.get("possession", 50.0))

            # Possession is already a percentage (0-100)
            # Normalize to 0-1
            normalized = possession / 100.0

            return float(np.clip(normalized, 0.0, 1.0))

        except (KeyError, ValueError, TypeError):
            return 0.5  # Fallback

    def _calculate_pressing_intensity(self, team_stats: Optional[Dict]) -> float:
        """
        Calculate pressing intensity from PPDA (Passes Per Defensive Action).

        Lower PPDA = more intense pressing = higher score.

        Args:
            team_stats (dict, optional): Team statistics

        Returns:
            float: Pressing intensity (0-1), where 1.0 is most intense
        """
        if team_stats is None:
            return 0.5  # Neutral default

        try:
            # PPDA (Passes Per Defensive Action): lower = more pressing
            ppda = float(team_stats.get("ppda", team_stats.get("PPDA", 0)))

            if ppda == 0:
                return 0.5

            # Normalize: PPDA of 5 = high press (1.0), PPDA of 15 = low press (0.0)
            # Most teams range from 8-12 PPDA
            normalized = 1.0 - ((ppda - 5) / 10)

            return float(np.clip(normalized, 0.0, 1.0))

        except (KeyError, ValueError, TypeError):
            return 0.5  # Fallback

    @staticmethod
    def _build_passing_network_from_events(
        events: pd.DataFrame,
        team_name: str
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Build passing network from StatsBomb event data.

        Args:
            events (pd.DataFrame): StatsBomb events DataFrame
            team_name (str): Name of the team to analyze

        Returns:
            Dict mapping (passer_position, receiver_position) to:
            {
                'pass_count': int,
                'pass_success_rate': float (0-1),
                'avg_pass_distance': float,
                'progressive_pass_count': int
            }
        """
        # Filter for this team's passes
        team_events = events[events["team"] == team_name]
        pass_events = team_events[team_events["type"] == "Pass"]

        if pass_events.empty:
            return {}

        # Build passing network
        passing_network = {}

        for _, pass_event in pass_events.iterrows():
            try:
                # Get passer and receiver positions
                passer_position = pass_event.get("position", "Unknown")
                receiver_position = pass_event.get("pass_recipient", "Unknown")

                # Skip if positions not available
                if passer_position == "Unknown" or receiver_position == "Unknown":
                    continue

                # Map detailed positions to general types (GK, DEF, MID, FWD)
                passer_type = FormationGraphBuilder._map_position_to_type(passer_position)
                receiver_type = FormationGraphBuilder._map_position_to_type(receiver_position)

                key = (passer_type, receiver_type)

                # Initialize if not exists
                if key not in passing_network:
                    passing_network[key] = {
                        'pass_count': 0,
                        'successful_passes': 0,
                        'total_distance': 0.0,
                        'progressive_pass_count': 0
                    }

                # Increment pass count
                passing_network[key]['pass_count'] += 1

                # Check if pass was successful
                outcome = pass_event.get("pass_outcome")
                if pd.isna(outcome) or outcome == "Complete":
                    passing_network[key]['successful_passes'] += 1

                # Calculate pass distance
                start_loc = pass_event.get("location", [0, 0])
                end_loc = pass_event.get("pass_end_location", [0, 0])

                if isinstance(start_loc, (list, tuple)) and isinstance(end_loc, (list, tuple)):
                    distance = np.sqrt((end_loc[0] - start_loc[0])**2 + (end_loc[1] - start_loc[1])**2)
                    passing_network[key]['total_distance'] += distance

                    # Progressive pass: moves ball significantly forward (>10 units in x direction)
                    if end_loc[0] > start_loc[0] + 10:
                        passing_network[key]['progressive_pass_count'] += 1

            except Exception as e:
                continue

        # Calculate success rates and averages
        for key, data in passing_network.items():
            if data['pass_count'] > 0:
                data['pass_success_rate'] = data['successful_passes'] / data['pass_count']
                data['avg_pass_distance'] = data['total_distance'] / data['pass_count']
            else:
                data['pass_success_rate'] = 0.0
                data['avg_pass_distance'] = 0.0

            # Remove intermediate fields
            del data['successful_passes']
            del data['total_distance']

        return passing_network

    @staticmethod
    def _map_position_to_type(position: str) -> str:
        """
        Map detailed position name to general type (GK, DEF, MID, FWD).

        Args:
            position (str): Detailed position name (e.g., "Left Center Back")

        Returns:
            str: General position type
        """
        position = position.lower()

        if "goalkeeper" in position or "gk" in position:
            return "GK"
        elif any(word in position for word in ["back", "defender", "wing back"]):
            return "DEF"
        elif any(word in position for word in ["forward", "striker", "wing"]) and "back" not in position:
            return "FWD"
        else:
            return "MID"

    @staticmethod
    def _get_marking_relationships(
        formation_a: str,
        formation_b: str
    ) -> List[Tuple[int, int]]:
        """
        Return typical marking assignments between two formations.

        This is a simplified heuristic. In reality, marking relationships
        are dynamic and depend on game state.

        Args:
            formation_a (str): Formation string (e.g., "4-3-3")
            formation_b (str): Formation string (e.g., "4-4-2")

        Returns:
            List of (position_idx_a, position_idx_b) tuples representing
            typical marking assignments
        """
        # Simplified marking relationships
        # In a real implementation, this would be much more sophisticated

        marking_pairs = []

        # GK typically doesn't mark anyone (except in set pieces)
        # Defenders typically mark forwards
        # Midfielders mark midfielders
        # Forwards press defenders

        # This is a placeholder - would need formation-specific logic
        # For now, return empty list (defensive coverage feature disabled)

        return marking_pairs


def test_graph_builder():
    """
    Test function to demonstrate graph building with enriched features.
    """
    # Sample formation configuration
    formations_config = {
        "4-3-3": {
            "positions": [
                {"x": 5, "y": 50, "position_type": "GK"},      # Goalkeeper
                {"x": 20, "y": 20, "position_type": "DEF"},    # Left Back
                {"x": 20, "y": 40, "position_type": "DEF"},    # Center Back 1
                {"x": 20, "y": 60, "position_type": "DEF"},    # Center Back 2
                {"x": 20, "y": 80, "position_type": "DEF"},    # Right Back
                {"x": 45, "y": 30, "position_type": "MID"},    # Defensive Mid
                {"x": 55, "y": 20, "position_type": "MID"},    # Left Mid
                {"x": 55, "y": 80, "position_type": "MID"},    # Right Mid
                {"x": 80, "y": 20, "position_type": "FWD"},    # Left Winger
                {"x": 85, "y": 50, "position_type": "FWD"},    # Striker
                {"x": 80, "y": 80, "position_type": "FWD"},    # Right Winger
            ]
        }
    }

    # Sample player data (not used with new feature system, but kept for compatibility)
    player_data = pd.DataFrame({
        "rating": [85, 82, 88, 87, 83, 86, 84, 85, 90, 89, 88],
        "form": [75, 70, 80, 78, 72, 85, 80, 75, 88, 90, 85]
    })

    # Sample team statistics (from FBref)
    team_stats = {
        "wins": 25,
        "draws": 8,
        "losses": 5,
        "matches_played": 38,
        "xg": 85.4,        # Total xG for season
        "xga": 42.1,       # Total xGA for season
        "possession": 61.5, # Average possession %
        "ppda": 8.2,       # Passes Per Defensive Action (lower = more pressing)
    }

    # Sample player market values (from TransferMarkt)
    player_values = pd.DataFrame({
        "player": ["Alisson", "Robertson", "Van Dijk", "Konate", "Alexander-Arnold",
                   "Fabinho", "Henderson", "Thiago", "Diaz", "Salah", "Nunez"],
        "market_value_eur": [
            35_000_000,  # GK: Alisson
            70_000_000,  # LB: Robertson
            50_000_000,  # CB: Van Dijk
            45_000_000,  # CB: Konate
            80_000_000,  # RB: Alexander-Arnold
            40_000_000,  # DM: Fabinho
            15_000_000,  # CM: Henderson
            25_000_000,  # CM: Thiago
            75_000_000,  # LW: Diaz
            55_000_000,  # ST: Salah
            70_000_000,  # RW: Nunez
        ]
    })

    # Sample passing network (from StatsBomb events)
    # This would normally be built from actual event data using _build_passing_network_from_events()
    passing_network = {
        ("GK", "DEF"): {
            "pass_count": 35,
            "pass_success_rate": 0.94,
            "avg_pass_distance": 25.3,
            "progressive_pass_count": 8
        },
        ("DEF", "DEF"): {
            "pass_count": 48,
            "pass_success_rate": 0.92,
            "avg_pass_distance": 18.5,
            "progressive_pass_count": 5
        },
        ("DEF", "MID"): {
            "pass_count": 67,
            "pass_success_rate": 0.85,
            "avg_pass_distance": 22.7,
            "progressive_pass_count": 42
        },
        ("MID", "MID"): {
            "pass_count": 125,
            "pass_success_rate": 0.88,
            "avg_pass_distance": 15.2,
            "progressive_pass_count": 28
        },
        ("MID", "FWD"): {
            "pass_count": 89,
            "pass_success_rate": 0.76,
            "avg_pass_distance": 19.8,
            "progressive_pass_count": 67
        },
        ("FWD", "FWD"): {
            "pass_count": 34,
            "pass_success_rate": 0.71,
            "avg_pass_distance": 12.4,
            "progressive_pass_count": 15
        },
        ("FWD", "MID"): {
            "pass_count": 21,
            "pass_success_rate": 0.67,
            "avg_pass_distance": 16.3,
            "progressive_pass_count": 3
        },
    }

    # Build graph with enriched data
    builder = FormationGraphBuilder(formations_config)
    graph = builder.build_graph(
        team="Liverpool",
        formation="4-3-3",
        player_data=player_data,
        team_stats=team_stats,
        player_values=player_values,
        passing_network=passing_network
    )

    # Print graph information
    print("=" * 80)
    print("TacticsAI Graph Builder Test - Enriched Features + Passing Network")
    print("=" * 80)
    print(f"Team: {graph.team}")
    print(f"Formation: {graph.formation}")
    print(f"Node features shape: {graph.x.shape}")
    print(f"Edge index shape: {graph.edge_index.shape}")
    print(f"Edge attributes shape: {graph.edge_attr.shape}")
    print(f"Number of nodes: {graph.x.shape[0]}")
    print(f"Number of edges: {graph.edge_index.shape[1]}")
    print(f"Node feature dimension: {graph.x.shape[1]} (upgraded from 8 to 12)")
    print(f"Edge feature dimension: {graph.edge_attr.shape[1]} (upgraded from 3 to 6)")

    print("\n" + "-" * 80)
    print("Feature Vector Breakdown (12 dimensions):")
    print("-" * 80)
    print("[0-1]:   Position (x, y normalized)")
    print("[2-5]:   Position type one-hot (GK, DEF, MID, FWD)")
    print("[6]:     Player quality (from market value)")
    print("[7]:     Team form (from W/D/L record)")
    print("[8]:     Team attacking strength (from xG)")
    print("[9]:     Team defensive strength (from xGA)")
    print("[10]:    Team possession tendency")
    print("[11]:    Team pressing intensity (from PPDA)")

    print("\n" + "-" * 80)
    print("Sample Node Features:")
    print("-" * 80)

    # Show GK features (first player)
    gk_features = graph.x[0]
    print(f"\nGoalkeeper (Alisson, €35M):")
    print(f"  Position: ({gk_features[0]:.3f}, {gk_features[1]:.3f})")
    print(f"  Type: GK={gk_features[2]:.0f}, DEF={gk_features[3]:.0f}, MID={gk_features[4]:.0f}, FWD={gk_features[5]:.0f}")
    print(f"  Quality: {gk_features[6]:.3f}")
    print(f"  Team form: {gk_features[7]:.3f}")
    print(f"  Attacking: {gk_features[8]:.3f}, Defensive: {gk_features[9]:.3f}")
    print(f"  Possession: {gk_features[10]:.3f}, Pressing: {gk_features[11]:.3f}")

    # Show striker features
    striker_features = graph.x[9]  # Salah
    print(f"\nStriker (Salah, €55M):")
    print(f"  Position: ({striker_features[0]:.3f}, {striker_features[1]:.3f})")
    print(f"  Type: GK={striker_features[2]:.0f}, DEF={striker_features[3]:.0f}, MID={striker_features[4]:.0f}, FWD={striker_features[5]:.0f}")
    print(f"  Quality: {striker_features[6]:.3f}")
    print(f"  Team form: {striker_features[7]:.3f}")
    print(f"  Attacking: {striker_features[8]:.3f}, Defensive: {striker_features[9]:.3f}")
    print(f"  Possession: {striker_features[10]:.3f}, Pressing: {striker_features[11]:.3f}")

    print("\n" + "-" * 80)
    print("Team Statistics Summary:")
    print("-" * 80)
    print(f"  Form: {team_stats['wins']}W-{team_stats['draws']}D-{team_stats['losses']}L")
    print(f"  xG/game: {team_stats['xg']/team_stats['matches_played']:.2f}")
    print(f"  xGA/game: {team_stats['xga']/team_stats['matches_played']:.2f}")
    print(f"  Possession: {team_stats['possession']:.1f}%")
    print(f"  PPDA: {team_stats['ppda']:.1f} (lower = more pressing)")
    print(f"  Squad value: €{player_values['market_value_eur'].sum()/1_000_000:.1f}M")

    print("\n" + "-" * 80)
    print("Sample Edge:")
    print("-" * 80)
    print(f"From node {graph.edge_index[0, 0].item()} to node {graph.edge_index[1, 0].item()}")
    print(f"Edge features: {graph.edge_attr[0]}")
    print(f"  Distance: {graph.edge_attr[0][0]:.3f}")
    print(f"  Same line: {graph.edge_attr[0][1]:.0f}")
    print(f"  Passing lane: {graph.edge_attr[0][2]:.3f}")

    print("\n" + "=" * 80)
    print("✓ Graph built successfully with enriched features!")
    print("=" * 80)

    return graph


if __name__ == "__main__":
    test_graph_builder()
