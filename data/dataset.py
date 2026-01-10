"""
TacticsAI Dataset
PyTorch Geometric dataset for football tactical matchups.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

try:
    from torch_geometric.data import Data, Dataset
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    Dataset = object  # Fallback for type hints

from .config import FORMATIONS, Formation, PREMIER_LEAGUE_TEAMS
from .fetchers import (
    StatsBombFetcher,
    FBrefFetcher,
    TransferMarktFetcher,
    UnderstatFetcher,
)


# =============================================================================
# Graph Construction
# =============================================================================

def create_tactical_graph(
    positions: List[Tuple[float, float]],
    features: Optional[np.ndarray] = None,
    connection_threshold: float = 35.0,
) -> Data:
    """
    Create a PyG Data object from player positions.

    Nodes: Players with positional and tactical features (12D)
    Edges: Connections between nearby players with enhanced features (6D)

    Args:
        positions: List of (x, y) coordinates for 11 players
        features: Optional node features array (11, 12)
        connection_threshold: Max distance for edge creation

    Returns:
        PyG Data object with enhanced 12D node features and 6D edge features
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("torch_geometric required for graph creation")

    num_players = len(positions)
    positions = np.array(positions)

    # Create node features (12D to match enhanced format)
    if features is None:
        # Enhanced features: [x_norm, y_norm, pos_onehot(4), quality, form, attack, defense, possession, pressing]
        features = np.zeros((num_players, 12), dtype=np.float32)

        for i, (x, y) in enumerate(positions):
            # Normalized position (0-1)
            features[i, 0] = x / 100.0
            features[i, 1] = y / 100.0

            # Position type one-hot encoding (4D: DEF, MID, FWD, GK)
            # Simple heuristic based on y-coordinate
            if y < 20:  # Goalkeeper
                features[i, 2:6] = [0, 0, 0, 1]
            elif y < 40:  # Defender
                features[i, 2:6] = [1, 0, 0, 0]
            elif y < 70:  # Midfielder
                features[i, 2:6] = [0, 1, 0, 0]
            else:  # Forward
                features[i, 2:6] = [0, 0, 1, 0]

            # Team statistics (default values when no real data available)
            features[i, 6] = 0.5  # quality
            features[i, 7] = 0.5  # form
            features[i, 8] = 0.5  # attack
            features[i, 9] = 0.5  # defense
            features[i, 10] = 0.5  # possession
            features[i, 11] = 0.5  # pressing

    # Create edges based on distance with enhanced 6D features
    edge_index = []
    edge_attr = []

    for i in range(num_players):
        for j in range(i + 1, num_players):
            dist = np.linalg.norm(positions[i] - positions[j])

            if dist < connection_threshold:
                # Add bidirectional edges
                edge_index.append([i, j])
                edge_index.append([j, i])

                # Enhanced edge features (6D):
                # [distance_norm, same_line, pass_frequency, pass_success, is_progressive, defensive_coverage]
                distance_norm = dist / 100.0  # Normalize to 0-1
                same_line = 1.0 if abs(positions[i][1] - positions[j][1]) < 15 else 0.0
                pass_frequency = 0.5  # Default value (no passing data)
                pass_success = 0.7  # Default value
                is_progressive = 0.0  # Default (not progressive)
                defensive_coverage = 0.0  # Default

                edge_feats = [
                    distance_norm,
                    same_line,
                    pass_frequency,
                    pass_success,
                    is_progressive,
                    defensive_coverage,
                ]

                edge_attr.extend([edge_feats, edge_feats])  # For both directions

    # Convert to tensors
    x = torch.tensor(features, dtype=torch.float32)

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    else:
        # Fully connected fallback if no edges created
        edge_index = []
        edge_attr = []
        for i in range(num_players):
            for j in range(num_players):
                if i != j:
                    edge_index.append([i, j])
                    # Default 6D edge features
                    edge_attr.append([0.5, 0.0, 0.5, 0.7, 0.0, 0.0])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def formation_to_graph(formation: Formation) -> Data:
    """
    Convert a Formation object to a tactical graph.

    Uses FormationGraphBuilder for consistent 12D node and 6D edge features.

    Args:
        formation: Formation object with player positions

    Returns:
        PyG Data object with enhanced features
    """
    from .graph_builder import FormationGraphBuilder

    builder = FormationGraphBuilder(FORMATIONS)

    # Use FormationGraphBuilder for consistent feature dimensions
    # This ensures 12D node features and 6D edge features
    return builder.build_graph(
        team="Default",
        formation=formation.name,
        player_data=None,
        team_stats=None,
        player_values=None,
        passing_network=None,
    )


# =============================================================================
# Matchup Dataset
# =============================================================================

class MatchupDataset(Dataset):
    """
    Dataset of tactical matchups for training.
    
    Each sample is a tuple of (graph_a, graph_b, label) where:
    - graph_a: First team's tactical formation as a graph
    - graph_b: Second team's tactical formation as a graph
    - label: Outcome (dominance score based on match result)
    
    Labels:
    - 1.0 = Team A won
    - 0.5 = Draw
    - 0.0 = Team B won
    """
    
    def __init__(
        self,
        matchups: List[Dict],
        transform=None,
        pre_transform=None,
    ):
        """
        Initialize the dataset.
        
        Args:
            matchups: List of matchup dicts with 'graph_a', 'graph_b', 'label'
            transform: Optional transform to apply
            pre_transform: Optional pre-transform to apply
        """
        super().__init__(None, transform, pre_transform)
        self.matchups = matchups
        logger.info(f"MatchupDataset initialized with {len(matchups)} samples")
    
    def len(self) -> int:
        return len(self.matchups)
    
    def get(self, idx: int) -> Tuple[Data, Data, torch.Tensor]:
        """Get a single matchup sample."""
        matchup = self.matchups[idx]
        
        graph_a = matchup["graph_a"]
        graph_b = matchup["graph_b"]
        label = torch.tensor(matchup["label"], dtype=torch.float32)
        
        # Optional battle zones
        if "battle_zones" in matchup:
            battle_zones = torch.tensor(matchup["battle_zones"], dtype=torch.float32)
            return graph_a, graph_b, (label, battle_zones)
        
        return graph_a, graph_b, label
    
    @classmethod
    def from_statsbomb(
        cls,
        competition_id: int = 2,
        season_id: int = 27,
        max_matches: Optional[int] = None,
        cache_dir: str = "backend/data/cache",
    ) -> "MatchupDataset":
        """
        Create dataset from StatsBomb open data.
        
        Args:
            competition_id: StatsBomb competition ID (default: Premier League)
            season_id: StatsBomb season ID (default: 27 = 2015/2016)
            max_matches: Maximum number of matches to load (None = all)
            cache_dir: Cache directory path
            
        Returns:
            MatchupDataset instance
        """
        logger.warning("Using StatsBomb-only data source. For enriched data with xG, team stats, "
                       "and player values, use from_multi_source() instead.")
        logger.info(f"Loading StatsBomb data (competition={competition_id}, season={season_id})")

        fetcher = StatsBombFetcher(cache_dir=cache_dir)
        
        # Fetch matches
        matches_df = fetcher.fetch_matches(competition_id, season_id)
        
        if matches_df.empty:
            logger.warning("No matches found, creating synthetic dataset")
            return cls.create_synthetic(n_samples=100)
        
        # Limit matches if specified
        if max_matches is not None:
            matches_df = matches_df.head(max_matches)
        
        logger.info(f"Processing {len(matches_df)} matches")
        
        matchups = []
        
        for _, match in matches_df.iterrows():
            try:
                matchup = cls._process_match(match, fetcher)
                if matchup is not None:
                    matchups.append(matchup)
            except Exception as e:
                logger.debug(f"Error processing match: {e}")
                continue
        
        if not matchups:
            logger.warning("No valid matchups created, falling back to synthetic data")
            return cls.create_synthetic(n_samples=100)
        
        logger.info(f"Created {len(matchups)} matchups from StatsBomb data")
        return cls(matchups)

    @classmethod
    def from_multi_source(
        cls,
        season: str = "2015-2016",
        max_matches: Optional[int] = None,
        cache_dir: str = "backend/data/cache",
    ) -> "MatchupDataset":
        """
        Create dataset by integrating all 4 data sources.

        This method enriches match data with:
        - Match events and lineups (StatsBomb)
        - xG, xGA, shot data (Understat)
        - Team form, possession stats (FBref)
        - Player market values (TransferMarkt)

        Args:
            season: Season string (e.g., "2015-2016" for StatsBomb season 27)
            max_matches: Maximum number of matches to load (None = all)
            cache_dir: Cache directory path

        Returns:
            MatchupDataset instance with enriched data
        """
        logger.info(f"Creating multi-source dataset for season {season}")

        # Map season to StatsBomb season_id
        season_mapping = {
            "2015-2016": 27,
            "2003-2004": 44,
        }
        statsbomb_season = season_mapping.get(season, 27)

        # Initialize all fetchers
        try:
            sb_fetcher = StatsBombFetcher(cache_dir=cache_dir)
            logger.info("✓ StatsBomb fetcher initialized")
        except Exception as e:
            logger.error(f"Failed to initialize StatsBomb: {e}")
            return cls.create_synthetic(n_samples=100)

        try:
            understat_fetcher = UnderstatFetcher(cache_dir=cache_dir)
            logger.info("✓ Understat fetcher initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Understat: {e}")
            understat_fetcher = None

        try:
            fbref_fetcher = FBrefFetcher(cache_dir=cache_dir)
            logger.info("✓ FBref fetcher initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize FBref: {e}")
            fbref_fetcher = None

        try:
            tm_fetcher = TransferMarktFetcher(cache_dir=cache_dir)
            logger.info("✓ TransferMarkt fetcher initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize TransferMarkt: {e}")
            tm_fetcher = None

        # Fetch StatsBomb matches (primary source)
        matches_df = sb_fetcher.fetch_matches(competition_id=2, season_id=statsbomb_season)

        if matches_df.empty:
            logger.warning("No StatsBomb matches found, creating synthetic dataset")
            return cls.create_synthetic(n_samples=100)

        # Limit matches if specified
        if max_matches is not None:
            matches_df = matches_df.head(max_matches)

        logger.info(f"Processing {len(matches_df)} matches with multi-source enrichment")

        # Fetch supplementary data (once per dataset)
        understat_season = season.split("-")[0]  # "2015-2016" -> "2015"

        # Get Understat league matches
        understat_matches = None
        if understat_fetcher:
            try:
                understat_matches = understat_fetcher.fetch_league_matches(season=understat_season)
                logger.info(f"✓ Fetched {len(understat_matches)} Understat matches")
            except Exception as e:
                logger.warning(f"Failed to fetch Understat data: {e}")

        # Get FBref league table for team stats
        fbref_league_table = None
        if fbref_fetcher:
            try:
                fbref_league_table = fbref_fetcher.fetch_league_table(season=season)
                logger.info(f"✓ Fetched FBref league table")
            except Exception as e:
                logger.warning(f"Failed to fetch FBref data: {e}")

        # Process each match with enrichment
        matchups = []
        for idx, match in matches_df.iterrows():
            try:
                # Get enrichment data for this match
                enrichment = cls._enrich_match_data(
                    match=match,
                    understat_matches=understat_matches,
                    fbref_league_table=fbref_league_table,
                    tm_fetcher=tm_fetcher,
                )

                # Process match with enrichment
                matchup = cls._process_match(
                    match=match,
                    fetcher=sb_fetcher,
                    enrichment=enrichment,
                )

                if matchup is not None:
                    matchups.append(matchup)
                    if (idx + 1) % 10 == 0:
                        logger.debug(f"Processed {idx + 1}/{len(matches_df)} matches")

            except Exception as e:
                logger.debug(f"Error processing match {idx}: {e}")
                continue

        if not matchups:
            logger.warning("No valid matchups created, falling back to synthetic data")
            return cls.create_synthetic(n_samples=100)

        logger.info(f"✓ Created {len(matchups)} enriched matchups from multi-source data")
        return cls(matchups)

    @classmethod
    def _enrich_match_data(
        cls,
        match: pd.Series,
        understat_matches: Optional[pd.DataFrame],
        fbref_league_table: Optional[pd.DataFrame],
        tm_fetcher: Optional[TransferMarktFetcher],
    ) -> Dict:
        """
        Enrich a match with data from multiple sources.

        Args:
            match: StatsBomb match data
            understat_matches: DataFrame of Understat matches
            fbref_league_table: DataFrame of FBref league standings
            tm_fetcher: TransferMarkt fetcher instance

        Returns:
            Dictionary with enrichment data:
            {
                'xg_data': {...},  # Understat xG data
                'home_team_stats': {...},  # FBref home team stats
                'away_team_stats': {...},  # FBref away team stats
                'home_player_values': pd.DataFrame,  # TransferMarkt values
                'away_player_values': pd.DataFrame,
            }
        """
        enrichment = {
            'xg_data': None,
            'home_team_stats': None,
            'away_team_stats': None,
            'home_player_values': None,
            'away_player_values': None,
        }

        home_team = str(match.get("home_team", ""))
        away_team = str(match.get("away_team", ""))
        match_date = match.get("match_date", "")

        # 1. Match with Understat data
        if understat_matches is not None and not understat_matches.empty:
            try:
                understat_match = cls._match_understat_to_statsbomb(
                    statsbomb_match=match,
                    understat_matches=understat_matches,
                )
                if understat_match is not None:
                    enrichment['xg_data'] = {
                        'home_xg': understat_match.get('home_xg', 0.0),
                        'away_xg': understat_match.get('away_xg', 0.0),
                        'home_xga': understat_match.get('home_xg', 0.0),  # Away xG = Home xGA
                        'away_xga': understat_match.get('away_xg', 0.0),  # Home xG = Away xGA
                    }
            except Exception as e:
                logger.debug(f"Failed to match Understat data: {e}")

        # 2. Get FBref team stats
        if fbref_league_table is not None and not fbref_league_table.empty:
            try:
                # Match teams by name (handle variations)
                home_stats = cls._find_team_in_table(home_team, fbref_league_table)
                away_stats = cls._find_team_in_table(away_team, fbref_league_table)

                enrichment['home_team_stats'] = home_stats
                enrichment['away_team_stats'] = away_stats
            except Exception as e:
                logger.debug(f"Failed to get FBref team stats: {e}")

        # 3. Get TransferMarkt player values (cached, so not too expensive)
        if tm_fetcher is not None:
            try:
                # Try to match team names to config
                home_config_name = cls._normalize_team_name(home_team)
                away_config_name = cls._normalize_team_name(away_team)

                if home_config_name:
                    enrichment['home_player_values'] = tm_fetcher.fetch_squad_values(home_config_name)
                if away_config_name:
                    enrichment['away_player_values'] = tm_fetcher.fetch_squad_values(away_config_name)
            except Exception as e:
                logger.debug(f"Failed to get TransferMarkt data: {e}")

        return enrichment

    @classmethod
    def _match_understat_to_statsbomb(
        cls,
        statsbomb_match: pd.Series,
        understat_matches: pd.DataFrame,
    ) -> Optional[Dict]:
        """
        Match a StatsBomb match to an Understat match.

        Matches by date and team names (handling variations).

        Args:
            statsbomb_match: StatsBomb match data
            understat_matches: DataFrame of Understat matches

        Returns:
            Understat match data dict or None if no match found
        """
        sb_date = str(statsbomb_match.get("match_date", ""))
        sb_home = str(statsbomb_match.get("home_team", "")).lower()
        sb_away = str(statsbomb_match.get("away_team", "")).lower()

        # Try to match by date first (most reliable)
        if sb_date and not understat_matches.empty:
            # Convert dates to comparable format
            understat_matches['date_str'] = pd.to_datetime(
                understat_matches['date'], errors='coerce'
            ).dt.strftime('%Y-%m-%d')
            sb_date_str = pd.to_datetime(sb_date, errors='coerce').strftime('%Y-%m-%d')

            # Filter by date
            date_matches = understat_matches[understat_matches['date_str'] == sb_date_str]

            if not date_matches.empty:
                # Then match by team names (with fuzzy matching)
                for _, us_match in date_matches.iterrows():
                    us_home = str(us_match.get('home_team', '')).lower()
                    us_away = str(us_match.get('away_team', '')).lower()

                    # Check if team names match (fuzzy)
                    home_match = (
                        sb_home in us_home or us_home in sb_home or
                        cls._fuzzy_team_match(sb_home, us_home)
                    )
                    away_match = (
                        sb_away in us_away or us_away in sb_away or
                        cls._fuzzy_team_match(sb_away, us_away)
                    )

                    if home_match and away_match:
                        return us_match.to_dict()

        return None

    @staticmethod
    def _fuzzy_team_match(name1: str, name2: str) -> bool:
        """
        Fuzzy match team names (handles variations like 'Man City' vs 'Manchester City').

        Args:
            name1: First team name
            name2: Second team name

        Returns:
            True if names likely refer to same team
        """
        # Common abbreviations and variations
        variations = {
            'man city': ['manchester city', 'man. city'],
            'man united': ['manchester united', 'man. united'],
            'spurs': ['tottenham', 'tottenham hotspur'],
            'newcastle': ['newcastle united'],
            'west ham': ['west ham united'],
            'wolves': ['wolverhampton', 'wolverhampton wanderers'],
        }

        name1 = name1.lower().strip()
        name2 = name2.lower().strip()

        # Check variations
        for key, alts in variations.items():
            if (name1 == key or name1 in alts) and (name2 == key or name2 in alts):
                return True

        return False

    @staticmethod
    def _find_team_in_table(team_name: str, table: pd.DataFrame) -> Optional[Dict]:
        """
        Find a team in FBref league table by name.

        Args:
            team_name: Team name to search for
            table: FBref league table DataFrame

        Returns:
            Team stats dict or None
        """
        if table is None or table.empty:
            return None

        team_name = team_name.lower()

        # Try exact match first
        for col in ['team', 'squad', 'team_name']:
            if col in table.columns:
                for idx, row in table.iterrows():
                    if team_name in str(row[col]).lower():
                        return row.to_dict()

        return None

    @staticmethod
    def _calculate_tactical_dominance(
        home_xg: float,
        away_xg: float,
        home_possession: float,  # 0-100
        away_possession: float,
        home_chances_created: int,
        away_chances_created: int,
        home_progressive_actions: int,  # progressive passes + carries
        away_progressive_actions: int,
        home_score: int,
        away_score: int,
    ) -> float:
        """
        Calculate composite Tactical Dominance Score (0-1 scale).

        Formula (from PRD):
        - 40% weight: xG differential
        - 20% weight: Possession in attacking third (approximated by possession * xG)
        - 20% weight: Chances created differential
        - 20% weight: Progressive actions differential

        Returns value 0-1 where:
        - >0.5 means home team dominated tactically
        - <0.5 means away team dominated
        - 0.5 means balanced

        Args:
            home_xg: Home team expected goals
            away_xg: Away team expected goals
            home_possession: Home possession percentage (0-100)
            away_possession: Away possession percentage (0-100)
            home_chances_created: Home team chances (shots + key passes)
            away_chances_created: Away team chances
            home_progressive_actions: Home progressive passes + carries
            away_progressive_actions: Away progressive passes + carries
            home_score: Home team goals scored
            away_score: Away team goals scored

        Returns:
            Dominance score (0-1), where >0.5 = home advantage
        """
        # Normalize each component to -1 to 1 scale (home advantage to away advantage)

        # xG differential (most important - 40% weight)
        xg_diff = (home_xg - away_xg) / max(home_xg + away_xg, 0.1)

        # Possession-weighted attacking presence (20% weight)
        home_attacking_presence = (home_possession / 100) * home_xg
        away_attacking_presence = (away_possession / 100) * away_xg
        possession_diff = (home_attacking_presence - away_attacking_presence) / max(
            home_attacking_presence + away_attacking_presence, 0.1
        )

        # Chances created (20% weight)
        chances_diff = (home_chances_created - away_chances_created) / max(
            home_chances_created + away_chances_created, 1
        )

        # Progressive actions (20% weight)
        prog_diff = (home_progressive_actions - away_progressive_actions) / max(
            home_progressive_actions + away_progressive_actions, 1
        )

        # Weighted combination
        raw_score = (
            0.4 * xg_diff + 0.2 * possession_diff + 0.2 * chances_diff + 0.2 * prog_diff
        )

        # Convert from [-1, 1] to [0, 1]
        dominance = (raw_score + 1) / 2

        return float(max(0.0, min(1.0, dominance)))

    @staticmethod
    def _extract_match_stats_from_events(events: pd.DataFrame, team_name: str) -> Dict:
        """
        Extract tactical stats from StatsBomb events for a specific team.

        Returns:
        - chances_created: shots + key passes
        - progressive_passes: passes that move ball >10m toward goal
        - progressive_carries: carries that move ball >10m toward goal
        - possession_estimate: % of events by this team

        Args:
            events: StatsBomb events DataFrame
            team_name: Team name to extract stats for

        Returns:
            Dictionary with tactical stats:
            {
                'chances_created': int,
                'progressive_actions': int,  # progressive_passes + progressive_carries
                'possession': float,  # estimated possession % (0-100)
                'shots': int,
                'key_passes': int,
            }
        """
        if events is None or events.empty:
            return {
                'chances_created': 0,
                'progressive_actions': 0,
                'possession': 50.0,
                'shots': 0,
                'key_passes': 0,
            }

        # Filter events for this team
        team_events = events[events['team'] == team_name]
        total_events = len(events)
        team_event_count = len(team_events)

        # Estimate possession (% of total events)
        possession = (team_event_count / total_events * 100) if total_events > 0 else 50.0

        # Count shots
        shots = len(team_events[team_events['type'] == 'Shot'])

        # Count key passes (passes that lead to shots)
        # In StatsBomb, check if pass has 'shot_assist' or 'goal_assist' in pass outcome
        key_passes = 0
        pass_events = team_events[team_events['type'] == 'Pass']
        for _, event in pass_events.iterrows():
            pass_data = event.get('pass', {})
            if isinstance(pass_data, dict):
                # Check for assist-related fields
                if (pass_data.get('shot_assist') or
                    pass_data.get('goal_assist') or
                    pass_data.get('assisted_shot_id')):
                    key_passes += 1

        # Chances created = shots + key passes
        chances_created = shots + key_passes

        # Count progressive passes (moves ball >10m toward goal)
        progressive_passes = 0
        for _, event in pass_events.iterrows():
            try:
                start_loc = event.get('location', [0, 0])
                pass_data = event.get('pass', {})
                if isinstance(pass_data, dict):
                    end_loc = pass_data.get('end_location', [0, 0])

                    if isinstance(start_loc, (list, tuple)) and isinstance(end_loc, (list, tuple)):
                        if len(start_loc) >= 2 and len(end_loc) >= 2:
                            # Calculate distance moved toward goal (x direction)
                            # StatsBomb pitch: 120x80, goal at x=120
                            x_progress = end_loc[0] - start_loc[0]

                            # Progressive if moves >10 units forward
                            if x_progress > 10:
                                progressive_passes += 1
            except (TypeError, IndexError, KeyError):
                continue

        # Count progressive carries
        progressive_carries = 0
        carry_events = team_events[team_events['type'] == 'Carry']
        for _, event in carry_events.iterrows():
            try:
                start_loc = event.get('location', [0, 0])
                carry_data = event.get('carry', {})
                if isinstance(carry_data, dict):
                    end_loc = carry_data.get('end_location', [0, 0])

                    if isinstance(start_loc, (list, tuple)) and isinstance(end_loc, (list, tuple)):
                        if len(start_loc) >= 2 and len(end_loc) >= 2:
                            # Calculate distance moved toward goal
                            x_progress = end_loc[0] - start_loc[0]

                            # Progressive if moves >10 units forward
                            if x_progress > 10:
                                progressive_carries += 1
            except (TypeError, IndexError, KeyError):
                continue

        # Total progressive actions
        progressive_actions = progressive_passes + progressive_carries

        return {
            'chances_created': chances_created,
            'progressive_actions': progressive_actions,
            'possession': possession,
            'shots': shots,
            'key_passes': key_passes,
        }

    @staticmethod
    def _normalize_team_name(team_name: str) -> Optional[str]:
        """
        Normalize a team name to match PREMIER_LEAGUE_TEAMS config keys.

        Args:
            team_name: Team name from data source

        Returns:
            Normalized team name matching config, or None
        """
        team_name = team_name.lower().strip()

        # Mapping of common variations to config keys
        name_map = {
            'arsenal': 'Arsenal',
            'aston villa': 'Aston Villa',
            'bournemouth': 'Bournemouth',
            'brentford': 'Brentford',
            'brighton': 'Brighton',
            'chelsea': 'Chelsea',
            'crystal palace': 'Crystal Palace',
            'everton': 'Everton',
            'fulham': 'Fulham',
            'ipswich': 'Ipswich Town',
            'leicester': 'Leicester City',
            'liverpool': 'Liverpool',
            'man city': 'Manchester City',
            'manchester city': 'Manchester City',
            'man united': 'Manchester United',
            'manchester united': 'Manchester United',
            'newcastle': 'Newcastle United',
            'nottingham forest': 'Nottingham Forest',
            'southampton': 'Southampton',
            'spurs': 'Tottenham',
            'tottenham': 'Tottenham',
            'west ham': 'West Ham',
            'wolves': 'Wolves',
            'wolverhampton': 'Wolves',
        }

        # Check if name is in map
        for key, value in name_map.items():
            if key in team_name:
                return value

        return None

    @classmethod
    def _process_match(
        cls,
        match: pd.Series,
        fetcher: StatsBombFetcher,
        enrichment: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Process a single match into a matchup dict.

        Args:
            match: Match data row from StatsBomb
            fetcher: StatsBombFetcher instance
            enrichment: Optional enrichment data from other sources (xG, team stats, player values)

        Returns:
            Matchup dict or None if processing fails
        """
        try:
            match_id = match.get("match_id")
            if match_id is None:
                return None
            
            # Get lineups
            lineups = fetcher.fetch_lineups(match_id)
            
            if not lineups or len(lineups) < 2:
                # Use default formation if lineups unavailable
                return cls._create_matchup_from_result(match)
            
            # Extract team names
            home_team = match.get("home_team", {})
            away_team = match.get("away_team", {})
            
            if isinstance(home_team, dict):
                home_name = home_team.get("home_team_name", "Home")
            else:
                home_name = str(home_team)
            
            if isinstance(away_team, dict):
                away_name = away_team.get("away_team_name", "Away")
            else:
                away_name = str(away_team)
            
            # Get formations from lineups or use defaults
            home_lineup = lineups.get(home_name)
            away_lineup = lineups.get(away_name)
            
            # Create graphs
            if home_lineup is not None and not home_lineup.empty:
                graph_a = cls._lineup_to_graph(home_lineup)
            else:
                graph_a = formation_to_graph(FORMATIONS["4-3-3"])
            
            if away_lineup is not None and not away_lineup.empty:
                graph_b = cls._lineup_to_graph(away_lineup)
            else:
                graph_b = formation_to_graph(FORMATIONS["4-4-2"])
            
            # Get match scores
            home_score = match.get("home_score", 0)
            away_score = match.get("away_score", 0)

            # Try to calculate proper tactical dominance score
            # This requires: xG, possession, chances, progressive actions
            data_quality = "result_based"  # Default
            label = 0.5  # Default balanced

            try:
                # Fetch events for tactical stats extraction
                events = fetcher.fetch_events(match_id)

                # Extract stats from events for both teams
                home_stats = cls._extract_match_stats_from_events(events, home_name)
                away_stats = cls._extract_match_stats_from_events(events, away_name)

                # Get xG from enrichment (Understat) or fallback to event-based approximation
                if enrichment and enrichment.get('xg_data'):
                    home_xg = float(enrichment['xg_data'].get('home_xg', 0.0))
                    away_xg = float(enrichment['xg_data'].get('away_xg', 0.0))
                else:
                    # Fallback: approximate xG from shots (rough estimate: 0.1 xG per shot)
                    home_xg = home_stats['shots'] * 0.1
                    away_xg = away_stats['shots'] * 0.1

                # Calculate tactical dominance score
                label = cls._calculate_tactical_dominance(
                    home_xg=home_xg,
                    away_xg=away_xg,
                    home_possession=home_stats['possession'],
                    away_possession=away_stats['possession'],
                    home_chances_created=home_stats['chances_created'],
                    away_chances_created=away_stats['chances_created'],
                    home_progressive_actions=home_stats['progressive_actions'],
                    away_progressive_actions=away_stats['progressive_actions'],
                    home_score=home_score,
                    away_score=away_score,
                )

                # Mark data quality based on whether we had xG data
                if enrichment and enrichment.get('xg_data'):
                    data_quality = "high_quality"  # Full xG + events
                else:
                    data_quality = "medium_quality"  # Events only, approximate xG

            except Exception as e:
                # Fallback to result-based label if dominance calculation fails
                logger.debug(f"Failed to calculate tactical dominance: {e}")
                if home_score > away_score:
                    label = 1.0  # Home team won
                elif home_score < away_score:
                    label = 0.0  # Away team won
                else:
                    label = 0.5  # Draw
                data_quality = "result_based"
            
            # Create battle zones (simplified)
            battle_zones = cls._calculate_battle_zones(graph_a, graph_b)

            # Build metadata with enrichment if available
            metadata = {
                "match_id": match_id,
                "home_team": home_name,
                "away_team": away_name,
                "score": f"{home_score}-{away_score}",
                "data_quality": data_quality,  # Track label quality: high_quality, medium_quality, result_based
            }

            # Add enrichment data to metadata if provided
            if enrichment is not None:
                if enrichment.get('xg_data'):
                    metadata['xg_data'] = enrichment['xg_data']
                if enrichment.get('home_team_stats'):
                    metadata['home_team_stats'] = enrichment['home_team_stats']
                if enrichment.get('away_team_stats'):
                    metadata['away_team_stats'] = enrichment['away_team_stats']
                if enrichment.get('home_player_values') is not None:
                    metadata['home_player_values'] = enrichment['home_player_values']
                if enrichment.get('away_player_values') is not None:
                    metadata['away_player_values'] = enrichment['away_player_values']

            return {
                "graph_a": graph_a,
                "graph_b": graph_b,
                "label": label,
                "battle_zones": battle_zones,
                "metadata": metadata,
            }
            
        except Exception as e:
            logger.debug(f"Match processing error: {e}")
            return None
    
    @classmethod
    def _create_matchup_from_result(cls, match: pd.Series) -> Dict:
        """Create matchup using default formations when lineup data unavailable."""
        # Use different default formations based on match characteristics
        formations_list = list(FORMATIONS.values())

        graph_a = formation_to_graph(random.choice(formations_list))
        graph_b = formation_to_graph(random.choice(formations_list))

        # Calculate label (result-based only)
        home_score = match.get("home_score", 0)
        away_score = match.get("away_score", 0)

        if home_score > away_score:
            label = 1.0
        elif home_score < away_score:
            label = 0.0
        else:
            label = 0.5

        return {
            "graph_a": graph_a,
            "graph_b": graph_b,
            "label": label,
            "battle_zones": cls._calculate_battle_zones(graph_a, graph_b),
            "metadata": {
                "data_quality": "result_based",  # No lineup or event data available
            },
        }
    
    @classmethod
    def _lineup_to_graph(cls, lineup: pd.DataFrame) -> Data:
        """Convert StatsBomb lineup to tactical graph."""
        # Extract positions from lineup
        positions = []
        
        if "positions" in lineup.columns:
            for _, player in lineup.iterrows():
                pos_data = player.get("positions", [])
                if pos_data and len(pos_data) > 0:
                    # Use first position
                    pos = pos_data[0] if isinstance(pos_data, list) else pos_data
                    if isinstance(pos, dict):
                        # Map position names to coordinates
                        x, y = cls._position_name_to_coords(pos.get("position", ""))
                        positions.append((x, y))
        
        # If we don't have enough positions, use default formation
        if len(positions) < 11:
            return formation_to_graph(FORMATIONS["4-3-3"])
        
        return create_tactical_graph(positions[:11])
    
    @staticmethod
    def _position_name_to_coords(position_name: str) -> Tuple[float, float]:
        """Map position name to approximate (x, y) coordinates."""
        position_coords = {
            "Goalkeeper": (50, 5),
            "Right Back": (85, 25),
            "Right Center Back": (65, 20),
            "Center Back": (50, 20),
            "Left Center Back": (35, 20),
            "Left Back": (15, 25),
            "Right Wing Back": (90, 40),
            "Left Wing Back": (10, 40),
            "Right Defensive Midfield": (65, 35),
            "Center Defensive Midfield": (50, 35),
            "Left Defensive Midfield": (35, 35),
            "Right Midfield": (85, 50),
            "Right Center Midfield": (65, 45),
            "Center Midfield": (50, 45),
            "Left Center Midfield": (35, 45),
            "Left Midfield": (15, 50),
            "Right Attacking Midfield": (70, 60),
            "Center Attacking Midfield": (50, 60),
            "Left Attacking Midfield": (30, 60),
            "Right Wing": (85, 70),
            "Left Wing": (15, 70),
            "Right Center Forward": (60, 80),
            "Center Forward": (50, 80),
            "Left Center Forward": (40, 80),
            "Striker": (50, 85),
        }
        
        # Try to find a match
        for name, coords in position_coords.items():
            if name.lower() in position_name.lower():
                return coords
        
        # Default to center midfield
        return (50, 45)
    
    @staticmethod
    def _calculate_battle_zones(graph_a: Data, graph_b: Data) -> List[float]:
        """
        Calculate battle zone indicators (3x3 grid).
        
        A zone is "contested" if both teams have players nearby.
        """
        zones = [0.0] * 9  # 3x3 grid
        
        # Get player positions from graphs
        pos_a = graph_a.x[:, 2:4].numpy()  # Raw x, y
        pos_b = graph_b.x[:, 2:4].numpy()
        
        # Check each zone
        for zone_idx in range(9):
            zone_x = (zone_idx % 3) * 33.33
            zone_y = (zone_idx // 3) * 33.33
            
            # Count players in zone for each team
            a_in_zone = sum(
                1 for x, y in pos_a 
                if zone_x <= x < zone_x + 33.33 and zone_y <= y < zone_y + 33.33
            )
            b_in_zone = sum(
                1 for x, y in pos_b
                if zone_x <= x < zone_x + 33.33 and zone_y <= y < zone_y + 33.33
            )
            
            # Zone is contested if both teams present
            if a_in_zone > 0 and b_in_zone > 0:
                zones[zone_idx] = 1.0
        
        return zones
    
    @classmethod
    def create_synthetic(cls, n_samples: int = 1000) -> "MatchupDataset":
        """
        Create a synthetic dataset for testing/development.

        Generates realistic tactical dominance scores based on formation matchup tendencies.

        Args:
            n_samples: Number of samples to generate

        Returns:
            MatchupDataset with synthetic matchups
        """
        logger.info(f"Creating synthetic dataset with {n_samples} samples")

        formations_list = list(FORMATIONS.values())
        matchups = []

        # Formation tactical profiles (typical characteristics)
        formation_profiles = {
            "4-3-3": {"attack": 0.75, "defense": 0.55, "possession": 0.65, "pressing": 0.70},
            "4-4-2": {"attack": 0.60, "defense": 0.70, "possession": 0.50, "pressing": 0.60},
            "3-4-3": {"attack": 0.80, "defense": 0.50, "possession": 0.60, "pressing": 0.75},
            "4-2-3-1": {"attack": 0.70, "defense": 0.60, "possession": 0.70, "pressing": 0.65},
            "3-5-2": {"attack": 0.65, "defense": 0.65, "possession": 0.60, "pressing": 0.55},
            "5-3-2": {"attack": 0.50, "defense": 0.80, "possession": 0.45, "pressing": 0.50},
        }

        for i in range(n_samples):
            # Random formations
            form_a = random.choice(formations_list)
            form_b = random.choice(formations_list)

            graph_a = formation_to_graph(form_a)
            graph_b = formation_to_graph(form_b)

            # Get formation profiles (default to balanced if not in profiles)
            profile_a = formation_profiles.get(
                form_a.name, {"attack": 0.6, "defense": 0.6, "possession": 0.5, "pressing": 0.6}
            )
            profile_b = formation_profiles.get(
                form_b.name, {"attack": 0.6, "defense": 0.6, "possession": 0.5, "pressing": 0.6}
            )

            # Generate synthetic match stats based on formation profiles
            # Add randomness to simulate match variance
            home_xg = profile_a["attack"] * 2.0 + np.random.normal(0, 0.3)  # 0-2 xG range
            away_xg = profile_b["attack"] * 2.0 + np.random.normal(0, 0.3)

            home_possession = profile_a["possession"] * 100  # Convert to 0-100
            away_possession = 100 - home_possession  # Possession sums to 100

            # Chances created correlates with attack strength
            home_chances = int(profile_a["attack"] * 15 + np.random.normal(0, 3))
            away_chances = int(profile_b["attack"] * 15 + np.random.normal(0, 3))

            # Progressive actions correlate with pressing + possession
            home_progressive = int(
                (profile_a["pressing"] + profile_a["possession"]) * 25 + np.random.normal(0, 5)
            )
            away_progressive = int(
                (profile_b["pressing"] + profile_b["possession"]) * 25 + np.random.normal(0, 5)
            )

            # Generate scores (simplified)
            home_score = int(max(0, home_xg + np.random.normal(0, 0.5)))
            away_score = int(max(0, away_xg + np.random.normal(0, 0.5)))

            # Calculate realistic tactical dominance score
            label = cls._calculate_tactical_dominance(
                home_xg=max(0, home_xg),
                away_xg=max(0, away_xg),
                home_possession=home_possession,
                away_possession=away_possession,
                home_chances_created=max(0, home_chances),
                away_chances_created=max(0, away_chances),
                home_progressive_actions=max(0, home_progressive),
                away_progressive_actions=max(0, away_progressive),
                home_score=home_score,
                away_score=away_score,
            )

            battle_zones = cls._calculate_battle_zones(graph_a, graph_b)

            matchups.append({
                "graph_a": graph_a,
                "graph_b": graph_b,
                "label": label,
                "battle_zones": battle_zones,
                "metadata": {
                    "data_quality": "synthetic",
                    "home_formation": form_a.name,
                    "away_formation": form_b.name,
                },
            })

        return cls(matchups)
    
    def split(
        self,
        val_ratio: float = 0.2,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Tuple["MatchupDataset", "MatchupDataset"]:
        """
        Split dataset into train and validation sets.
        
        Args:
            val_ratio: Fraction of data for validation
            shuffle: Whether to shuffle before splitting
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        indices = list(range(len(self.matchups)))
        
        if shuffle:
            random.seed(seed)
            random.shuffle(indices)
        
        split_idx = int(len(indices) * (1 - val_ratio))
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_matchups = [self.matchups[i] for i in train_indices]
        val_matchups = [self.matchups[i] for i in val_indices]
        
        logger.info(f"Split: {len(train_matchups)} train, {len(val_matchups)} val")
        
        return (
            MatchupDataset(train_matchups),
            MatchupDataset(val_matchups),
        )
