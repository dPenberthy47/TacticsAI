"""
TacticsAI Data Fetchers
Fetcher classes for retrieving football data from various sources.
"""

import json
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger

# StatsBomb library for free open data
try:
    from statsbombpy import sb
    STATSBOMB_AVAILABLE = True
except ImportError:
    STATSBOMB_AVAILABLE = False
    logger.warning("statsbombpy not installed. StatsBombFetcher will be unavailable.")

from .config import PREMIER_LEAGUE_TEAMS, API_URLS


# =============================================================================
# Base Fetcher
# =============================================================================

class BaseFetcher(ABC):
    """
    Abstract base class for all data fetchers.
    
    Provides caching functionality with 24-hour expiration and
    common utilities for data retrieval.
    """
    
    # Cache expiration time (24 hours)
    CACHE_EXPIRY_HOURS = 24
    
    def __init__(self, cache_dir: str = "backend/data/cache"):
        """
        Initialize the fetcher.
        
        Args:
            cache_dir: Directory for caching fetched data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized {self.__class__.__name__} with cache at {self.cache_dir}")
    
    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """
        Fetch data from the source.
        
        Returns:
            DataFrame containing the fetched data
        """
        pass
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache entry."""
        # Sanitize cache key for filesystem
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in cache_key)
        return self.cache_dir / f"{self.__class__.__name__}_{safe_key}.parquet"
    
    def _get_cache_meta_path(self, cache_key: str) -> Path:
        """Get the metadata file path for a cache entry."""
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in cache_key)
        return self.cache_dir / f"{self.__class__.__name__}_{safe_key}.meta.json"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if cached data exists and is not expired.
        
        Args:
            cache_key: Unique identifier for the cached data
            
        Returns:
            True if valid cache exists, False otherwise
        """
        cache_path = self._get_cache_path(cache_key)
        meta_path = self._get_cache_meta_path(cache_key)
        
        if not cache_path.exists() or not meta_path.exists():
            return False
        
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            
            cached_at = datetime.fromisoformat(meta["cached_at"])
            expiry = cached_at + timedelta(hours=self.CACHE_EXPIRY_HOURS)
            
            if datetime.now() > expiry:
                logger.debug(f"Cache expired for {cache_key}")
                return False
            
            return True
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Invalid cache metadata: {e}")
            return False
    
    def _get_cached(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Retrieve cached data if available and not expired.
        
        Args:
            cache_key: Unique identifier for the cached data
            
        Returns:
            Cached DataFrame if available, None otherwise
        """
        if not self._is_cache_valid(cache_key):
            return None
        
        cache_path = self._get_cache_path(cache_key)
        
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"Cache hit for {cache_key} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.warning(f"Failed to read cache: {e}")
            return None
    
    def _save_cache(self, cache_key: str, data: pd.DataFrame) -> None:
        """
        Save data to cache.
        
        Args:
            cache_key: Unique identifier for the cached data
            data: DataFrame to cache
        """
        cache_path = self._get_cache_path(cache_key)
        meta_path = self._get_cache_meta_path(cache_key)
        
        try:
            # Save data as parquet (efficient binary format)
            data.to_parquet(cache_path, index=False)
            
            # Save metadata
            meta = {
                "cached_at": datetime.now().isoformat(),
                "rows": len(data),
                "columns": list(data.columns),
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            
            logger.debug(f"Cached {len(data)} rows for {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _empty_dataframe(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Return an empty DataFrame with optional column structure."""
        if columns:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame()


# =============================================================================
# StatsBomb Fetcher (FREE DATA - Primary Source)
# =============================================================================

class StatsBombFetcher(BaseFetcher):
    """
    Fetcher for StatsBomb open data.

    StatsBomb provides free, high-quality match event data for selected
    competitions. This is the primary data source for TacticsAI.

    Default competition: Premier League (competition_id=2)
    Default season: 2015/2016 (season_id=27)

    Note: Free StatsBomb Premier League data only includes seasons 27 (2015/16) and 44 (2003/04).
    """
    
    # Premier League competition ID in StatsBomb
    PREMIER_LEAGUE_ID = 2
    
    # Available season IDs for FREE Premier League data from StatsBomb
    # Note: Only these seasons are available in the free open data
    SEASON_IDS = {
        "2015/2016": 27,
        "2003/2004": 44,  # The "Invincibles" season
    }
    
    def __init__(self, cache_dir: str = "backend/data/cache"):
        super().__init__(cache_dir)
        
        if not STATSBOMB_AVAILABLE:
            logger.error("statsbombpy is not installed. Please install with: pip install statsbombpy")
    
    def fetch(self) -> pd.DataFrame:
        """Fetch default Premier League matches."""
        return self.fetch_matches()
    
    def fetch_competitions(self) -> pd.DataFrame:
        """
        Fetch all available competitions from StatsBomb.
        
        Returns:
            DataFrame with competition details
        """
        cache_key = "competitions"
        
        # Check cache
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        if not STATSBOMB_AVAILABLE:
            logger.error("statsbombpy not available")
            return self._empty_dataframe(["competition_id", "competition_name", "country_name"])
        
        try:
            logger.info("Fetching StatsBomb competitions...")
            df = sb.competitions()
            
            if df is not None and not df.empty:
                self._save_cache(cache_key, df)
                logger.info(f"Fetched {len(df)} competitions")
                return df
            
            return self._empty_dataframe()
        except Exception as e:
            logger.error(f"Failed to fetch competitions: {e}")
            return self._empty_dataframe()
    
    def fetch_matches(
        self,
        competition_id: int = PREMIER_LEAGUE_ID,
        season_id: int = 27,
    ) -> pd.DataFrame:
        """
        Fetch matches for a competition and season.

        Args:
            competition_id: StatsBomb competition ID (default: Premier League)
            season_id: StatsBomb season ID (default: 27 = 2015/2016)
            
        Returns:
            DataFrame with match details
        """
        cache_key = f"matches_{competition_id}_{season_id}"
        
        # Check cache
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        if not STATSBOMB_AVAILABLE:
            logger.error("statsbombpy not available")
            return self._empty_dataframe([
                "match_id", "match_date", "home_team", "away_team",
                "home_score", "away_score", "competition", "season"
            ])
        
        try:
            logger.info(f"Fetching matches for competition={competition_id}, season={season_id}...")
            df = sb.matches(competition_id=competition_id, season_id=season_id)
            
            if df is not None and not df.empty:
                self._save_cache(cache_key, df)
                logger.info(f"Fetched {len(df)} matches")
                return df
            
            logger.warning(f"No matches found for competition={competition_id}, season={season_id}")
            return self._empty_dataframe()
        except Exception as e:
            logger.error(f"Failed to fetch matches: {e}")
            return self._empty_dataframe()
    
    def fetch_lineups(self, match_id: int) -> Dict[str, pd.DataFrame]:
        """
        Fetch lineups for a specific match.
        
        Args:
            match_id: StatsBomb match ID
            
        Returns:
            Dict with team names as keys and lineup DataFrames as values
        """
        cache_key = f"lineups_{match_id}"
        
        # Check cache (we'll cache as JSON since it's a dict of DataFrames)
        cache_path = self.cache_dir / f"{self.__class__.__name__}_{cache_key}.json"
        
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    cached_data = json.load(f)
                
                # Check expiry
                cached_at = datetime.fromisoformat(cached_data["cached_at"])
                if datetime.now() < cached_at + timedelta(hours=self.CACHE_EXPIRY_HOURS):
                    logger.info(f"Cache hit for lineups {match_id}")
                    return {
                        team: pd.DataFrame(data) 
                        for team, data in cached_data["lineups"].items()
                    }
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        if not STATSBOMB_AVAILABLE:
            logger.error("statsbombpy not available")
            return {}
        
        try:
            logger.info(f"Fetching lineups for match {match_id}...")
            lineups = sb.lineups(match_id=match_id)
            
            if lineups:
                # Cache the result
                try:
                    cache_data = {
                        "cached_at": datetime.now().isoformat(),
                        "lineups": {
                            team: df.to_dict(orient="records") 
                            for team, df in lineups.items()
                        }
                    }
                    with open(cache_path, "w") as f:
                        json.dump(cache_data, f)
                except Exception as e:
                    logger.warning(f"Failed to cache lineups: {e}")
                
                logger.info(f"Fetched lineups for {len(lineups)} teams")
                return lineups
            
            return {}
        except Exception as e:
            logger.error(f"Failed to fetch lineups: {e}")
            return {}
    
    def fetch_events(self, match_id: int) -> pd.DataFrame:
        """
        Fetch all events (passes, shots, tackles, etc.) for a match.
        
        Args:
            match_id: StatsBomb match ID
            
        Returns:
            DataFrame with all match events
        """
        cache_key = f"events_{match_id}"
        
        # Check cache
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        if not STATSBOMB_AVAILABLE:
            logger.error("statsbombpy not available")
            return self._empty_dataframe([
                "id", "type", "player", "team", "minute", "second",
                "location", "pass_end_location", "shot_outcome"
            ])
        
        try:
            logger.info(f"Fetching events for match {match_id}...")
            df = sb.events(match_id=match_id)
            
            if df is not None and not df.empty:
                self._save_cache(cache_key, df)
                logger.info(f"Fetched {len(df)} events")
                return df
            
            return self._empty_dataframe()
        except Exception as e:
            logger.error(f"Failed to fetch events: {e}")
            return self._empty_dataframe()
    
    def fetch_player_match_stats(self, match_id: int) -> pd.DataFrame:
        """
        Aggregate player statistics from match events.
        
        Args:
            match_id: StatsBomb match ID
            
        Returns:
            DataFrame with player-level statistics
        """
        events = self.fetch_events(match_id)
        
        if events.empty:
            return self._empty_dataframe([
                "player", "team", "passes", "pass_accuracy", 
                "shots", "tackles", "interceptions"
            ])
        
        try:
            # Aggregate stats by player
            stats = []
            
            for player in events["player"].dropna().unique():
                player_events = events[events["player"] == player]
                team = player_events["team"].iloc[0] if not player_events.empty else None
                
                # Count event types
                passes = player_events[player_events["type"] == "Pass"]
                shots = player_events[player_events["type"] == "Shot"]
                
                pass_complete = passes[passes.get("pass_outcome", pd.Series()).isna()].shape[0] if "pass_outcome" in passes.columns else len(passes)
                pass_total = len(passes)
                
                stats.append({
                    "player": player,
                    "team": team,
                    "passes": pass_total,
                    "pass_accuracy": (pass_complete / pass_total * 100) if pass_total > 0 else 0,
                    "shots": len(shots),
                    "tackles": len(player_events[player_events["type"] == "Tackle"]),
                    "interceptions": len(player_events[player_events["type"] == "Interception"]),
                })
            
            return pd.DataFrame(stats)
        except Exception as e:
            logger.error(f"Failed to aggregate player stats: {e}")
            return self._empty_dataframe()


# =============================================================================
# FBref Fetcher
# =============================================================================

class FBrefFetcher(BaseFetcher):
    """
    Fetcher for FBref team and player statistics.
    
    Scrapes data from fbref.com with proper rate limiting.
    """
    
    BASE_URL = "https://fbref.com/en"
    RATE_LIMIT_SECONDS = 2.0
    
    # Standard columns for team stats
    TEAM_STATS_COLUMNS = [
        "team", "matches_played", "wins", "draws", "losses",
        "goals_for", "goals_against", "goal_difference",
        "points", "possession", "xg", "xga", "xg_diff"
    ]
    
    def __init__(self, cache_dir: str = "backend/data/cache"):
        super().__init__(cache_dir)
        self._last_request_time: float = 0.0
        
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "TacticsAI/1.0 (Football Analytics Research)",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        })
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        if self._last_request_time > 0:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.RATE_LIMIT_SECONDS:
                sleep_time = self.RATE_LIMIT_SECONDS - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        self._last_request_time = time.time()
    
    def fetch(self) -> pd.DataFrame:
        """Fetch Premier League standings table."""
        return self.fetch_league_table()
    
    def fetch_league_table(self, season: str = "2023-2024") -> pd.DataFrame:
        """
        Fetch Premier League standings/table.
        
        Args:
            season: Season string (e.g., "2023-2024")
            
        Returns:
            DataFrame with league standings
        """
        cache_key = f"league_table_{season}"
        
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        try:
            self._rate_limit()
            
            url = f"{self.BASE_URL}/comps/9/Premier-League-Stats"
            logger.info(f"Fetching FBref league table from {url}")
            
            response = self._session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Find the league standings table
            table = soup.find("table", {"id": "results2024-202591_overall"})
            
            if table is None:
                # Try alternative table ID patterns
                table = soup.find("table", class_="stats_table")
            
            if table is None:
                logger.warning("Could not find standings table")
                return self._empty_dataframe(self.TEAM_STATS_COLUMNS)
            
            # Parse table
            df = self._parse_html_table(table)
            
            if not df.empty:
                self._save_cache(cache_key, df)
                logger.info(f"Fetched league table with {len(df)} teams")
            
            return df
            
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            return self._empty_dataframe(self.TEAM_STATS_COLUMNS)
        except Exception as e:
            logger.error(f"Failed to fetch league table: {e}")
            return self._empty_dataframe(self.TEAM_STATS_COLUMNS)
    
    def fetch_team_stats(self, team_name: str) -> pd.DataFrame:
        """
        Fetch detailed statistics for a specific team.
        
        Args:
            team_name: Name of the team (must match PREMIER_LEAGUE_TEAMS keys)
            
        Returns:
            DataFrame with team statistics
        """
        # Get team mapping
        team_mapping = PREMIER_LEAGUE_TEAMS.get(team_name)
        
        if team_mapping is None:
            logger.warning(f"Unknown team: {team_name}")
            return self._empty_dataframe([
                "stat", "value", "per_90", "percentile"
            ])
        
        cache_key = f"team_stats_{team_mapping.fbref_slug}"
        
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        try:
            self._rate_limit()
            
            url = f"{self.BASE_URL}/squads/{team_mapping.fbref_id}/{team_mapping.fbref_slug}-Stats"
            logger.info(f"Fetching team stats for {team_name} from {url}")
            
            response = self._session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Look for standard stats table
            stats_table = soup.find("table", {"id": "stats_standard_9"})
            
            if stats_table is None:
                stats_table = soup.find("table", {"id": lambda x: x and "stats_standard" in x})
            
            if stats_table is None:
                logger.warning(f"Could not find stats table for {team_name}")
                return self._empty_dataframe()
            
            df = self._parse_html_table(stats_table)
            
            if not df.empty:
                df["team"] = team_name
                self._save_cache(cache_key, df)
                logger.info(f"Fetched stats for {team_name}: {len(df)} rows")
            
            return df
            
        except requests.RequestException as e:
            logger.error(f"Request failed for {team_name}: {e}")
            return self._empty_dataframe()
        except Exception as e:
            logger.error(f"Failed to fetch team stats: {e}")
            return self._empty_dataframe()
    
    def _parse_html_table(self, table) -> pd.DataFrame:
        """
        Parse an HTML table element into a DataFrame.
        
        Args:
            table: BeautifulSoup table element
            
        Returns:
            Parsed DataFrame
        """
        try:
            # Get headers
            headers = []
            header_row = table.find("thead")
            
            if header_row:
                # Get the last row in thead (actual column names)
                header_rows = header_row.find_all("tr")
                if header_rows:
                    last_header = header_rows[-1]
                    for th in last_header.find_all(["th", "td"]):
                        text = th.get_text(strip=True)
                        # Handle data-stat attribute for cleaner names
                        stat_name = th.get("data-stat", text)
                        headers.append(stat_name if stat_name else text)
            
            # Get data rows
            rows = []
            tbody = table.find("tbody")
            
            if tbody:
                for tr in tbody.find_all("tr"):
                    # Skip header rows within tbody
                    if tr.get("class") and "thead" in tr.get("class", []):
                        continue
                    
                    row = []
                    for td in tr.find_all(["th", "td"]):
                        text = td.get_text(strip=True)
                        row.append(text)
                    
                    if row and len(row) == len(headers):
                        rows.append(row)
            
            if not rows:
                return pd.DataFrame()
            
            df = pd.DataFrame(rows, columns=headers if headers else None)
            
            # Clean up column names
            df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
            
            return df
            
        except Exception as e:
            logger.error(f"Table parsing error: {e}")
            return pd.DataFrame()


# =============================================================================
# TransferMarkt Fetcher
# =============================================================================

class TransferMarktFetcher(BaseFetcher):
    """
    Fetcher for TransferMarkt player market values.
    
    Note: TransferMarkt has aggressive anti-scraping measures.
    This fetcher includes fallback mock data for reliability.
    """
    
    BASE_URL = "https://www.transfermarkt.com"
    RATE_LIMIT_SECONDS = 3.0
    
    def __init__(self, cache_dir: str = "backend/data/cache"):
        super().__init__(cache_dir)
        self._last_request_time: float = 0.0
        
        self._session = requests.Session()
        # TransferMarkt requires specific headers to avoid blocks
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        })
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        if self._last_request_time > 0:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.RATE_LIMIT_SECONDS:
                sleep_time = self.RATE_LIMIT_SECONDS - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        self._last_request_time = time.time()
    
    def fetch(self) -> pd.DataFrame:
        """Fetch squad values for all Premier League teams."""
        all_values = []
        
        for team_name in list(PREMIER_LEAGUE_TEAMS.keys())[:5]:  # Limit for demo
            df = self.fetch_squad_values(team_name)
            if not df.empty:
                all_values.append(df)
        
        if all_values:
            return pd.concat(all_values, ignore_index=True)
        return self._empty_dataframe()
    
    def fetch_squad_values(self, team_name: str) -> pd.DataFrame:
        """
        Fetch squad market values for a team.
        
        Args:
            team_name: Name of the team (must match PREMIER_LEAGUE_TEAMS keys)
            
        Returns:
            DataFrame with player names and market values
        """
        team_mapping = PREMIER_LEAGUE_TEAMS.get(team_name)
        
        if team_mapping is None:
            logger.warning(f"Unknown team: {team_name}")
            return self._empty_dataframe([
                "player", "position", "age", "nationality", "market_value_eur", "team"
            ])
        
        cache_key = f"squad_values_{team_mapping.transfermarkt_slug}"
        
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        # Try to scrape real data first
        df = self._scrape_squad_values(team_name, team_mapping)
        
        # If scraping fails, use mock data
        if df.empty:
            logger.warning(f"Scraping failed for {team_name}, using mock data")
            df = self._get_mock_squad_values(team_name)
        
        if not df.empty:
            self._save_cache(cache_key, df)
        
        return df
    
    def _scrape_squad_values(self, team_name: str, team_mapping) -> pd.DataFrame:
        """
        Attempt to scrape real squad values from TransferMarkt.
        
        Args:
            team_name: Team name
            team_mapping: TeamMapping object with TransferMarkt IDs
            
        Returns:
            DataFrame with squad values or empty DataFrame on failure
        """
        try:
            self._rate_limit()
            
            url = f"{self.BASE_URL}/{team_mapping.transfermarkt_slug}/kader/verein/{team_mapping.transfermarkt_id}"
            logger.info(f"Attempting to fetch squad values for {team_name} from {url}")
            
            response = self._session.get(url, timeout=30)
            
            # Check for anti-scraping blocks
            if response.status_code == 403:
                logger.warning("TransferMarkt blocked request (403)")
                return self._empty_dataframe()
            
            if response.status_code == 429:
                logger.warning("TransferMarkt rate limit hit (429)")
                return self._empty_dataframe()
            
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Look for player table
            player_table = soup.find("table", class_="items")
            
            if player_table is None:
                logger.warning("Could not find player table")
                return self._empty_dataframe()
            
            players = []
            rows = player_table.find_all("tr", class_=["odd", "even"])
            
            for row in rows:
                try:
                    # Player name
                    name_cell = row.find("td", class_="hauptlink")
                    name = name_cell.get_text(strip=True) if name_cell else "Unknown"
                    
                    # Position
                    pos_cell = row.find_all("td")
                    position = ""
                    for td in pos_cell:
                        if td.get_text(strip=True) in ["Goalkeeper", "Defender", "Midfielder", "Forward", 
                                                        "Centre-Back", "Left-Back", "Right-Back",
                                                        "Defensive Midfield", "Central Midfield",
                                                        "Attacking Midfield", "Left Winger", "Right Winger",
                                                        "Centre-Forward"]:
                            position = td.get_text(strip=True)
                            break
                    
                    # Market value
                    value_cell = row.find("td", class_="rechts hauptlink")
                    value_text = value_cell.get_text(strip=True) if value_cell else "€0"
                    market_value = self._parse_market_value(value_text)
                    
                    players.append({
                        "player": name,
                        "position": position,
                        "market_value_eur": market_value,
                        "team": team_name,
                    })
                except Exception as e:
                    logger.debug(f"Error parsing row: {e}")
                    continue
            
            if players:
                df = pd.DataFrame(players)
                logger.info(f"Scraped {len(df)} players for {team_name}")
                return df
            
            return self._empty_dataframe()
            
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            return self._empty_dataframe()
        except Exception as e:
            logger.error(f"Scraping error: {e}")
            return self._empty_dataframe()
    
    def _parse_market_value(self, value_str: str) -> int:
        """
        Parse TransferMarkt market value string to integer.
        
        Args:
            value_str: Value string like "€50.00m" or "€800k"
            
        Returns:
            Integer value in EUR
        """
        try:
            value_str = value_str.replace("€", "").replace(",", ".").strip()
            
            if "m" in value_str.lower():
                return int(float(value_str.lower().replace("m", "")) * 1_000_000)
            elif "k" in value_str.lower():
                return int(float(value_str.lower().replace("k", "")) * 1_000)
            else:
                return int(float(value_str))
        except (ValueError, AttributeError):
            return 0
    
    def _get_mock_squad_values(self, team_name: str) -> pd.DataFrame:
        """
        Generate mock squad values data for fallback.
        
        Args:
            team_name: Name of the team
            
        Returns:
            DataFrame with mock player data
        """
        # Mock data for major Premier League teams
        mock_squads = {
            "Manchester City": [
                ("Erling Haaland", "Centre-Forward", 25, "Norway", 180_000_000),
                ("Kevin De Bruyne", "Central Midfield", 33, "Belgium", 45_000_000),
                ("Phil Foden", "Attacking Midfield", 24, "England", 150_000_000),
                ("Rodri", "Defensive Midfield", 28, "Spain", 120_000_000),
                ("Bernardo Silva", "Attacking Midfield", 30, "Portugal", 70_000_000),
            ],
            "Arsenal": [
                ("Bukayo Saka", "Right Winger", 23, "England", 140_000_000),
                ("Martin Ødegaard", "Attacking Midfield", 26, "Norway", 110_000_000),
                ("Declan Rice", "Defensive Midfield", 26, "England", 100_000_000),
                ("William Saliba", "Centre-Back", 23, "France", 90_000_000),
                ("Gabriel Martinelli", "Left Winger", 23, "Brazil", 75_000_000),
            ],
            "Liverpool": [
                ("Mohamed Salah", "Right Winger", 32, "Egypt", 60_000_000),
                ("Virgil van Dijk", "Centre-Back", 33, "Netherlands", 30_000_000),
                ("Trent Alexander-Arnold", "Right-Back", 26, "England", 70_000_000),
                ("Alisson", "Goalkeeper", 32, "Brazil", 35_000_000),
                ("Darwin Núñez", "Centre-Forward", 25, "Uruguay", 70_000_000),
            ],
            "Chelsea": [
                ("Cole Palmer", "Attacking Midfield", 22, "England", 100_000_000),
                ("Enzo Fernández", "Central Midfield", 24, "Argentina", 75_000_000),
                ("Moisés Caicedo", "Defensive Midfield", 23, "Ecuador", 80_000_000),
                ("Nicolas Jackson", "Centre-Forward", 23, "Senegal", 55_000_000),
                ("Reece James", "Right-Back", 25, "England", 50_000_000),
            ],
            "Manchester United": [
                ("Marcus Rashford", "Left Winger", 27, "England", 60_000_000),
                ("Bruno Fernandes", "Attacking Midfield", 30, "Portugal", 55_000_000),
                ("Kobbie Mainoo", "Central Midfield", 19, "England", 60_000_000),
                ("Alejandro Garnacho", "Left Winger", 20, "Argentina", 55_000_000),
                ("Rasmus Højlund", "Centre-Forward", 21, "Denmark", 55_000_000),
            ],
        }
        
        # Default mock data for other teams
        default_squad = [
            ("Player 1", "Centre-Forward", 25, "England", 30_000_000),
            ("Player 2", "Central Midfield", 27, "England", 25_000_000),
            ("Player 3", "Centre-Back", 26, "England", 20_000_000),
            ("Player 4", "Goalkeeper", 28, "England", 15_000_000),
            ("Player 5", "Right Winger", 24, "England", 22_000_000),
        ]
        
        squad_data = mock_squads.get(team_name, default_squad)
        
        players = []
        for name, position, age, nationality, value in squad_data:
            players.append({
                "player": name,
                "position": position,
                "age": age,
                "nationality": nationality,
                "market_value_eur": value,
                "team": team_name,
            })
        
        logger.info(f"Generated mock data for {team_name}: {len(players)} players")
        return pd.DataFrame(players)


# =============================================================================
# Understat Fetcher
# =============================================================================

class UnderstatFetcher(BaseFetcher):
    """
    Fetcher for Understat expected goals (xG) and shot data.

    Understat provides detailed xG data for matches, teams, and individual shots.
    The site uses JavaScript rendering, so we parse embedded JSON from script tags.

    URL patterns:
    - Match: https://understat.com/match/{match_id}
    - Team: https://understat.com/team/{team_slug}/{season}
    - League: https://understat.com/league/EPL/{season}
    """

    BASE_URL = "https://understat.com"
    RATE_LIMIT_SECONDS = 2.0

    def __init__(self, cache_dir: str = "backend/data/cache"):
        super().__init__(cache_dir)
        self._last_request_time: float = 0.0

        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        })

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        if self._last_request_time > 0:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.RATE_LIMIT_SECONDS:
                sleep_time = self.RATE_LIMIT_SECONDS - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        self._last_request_time = time.time()

    def fetch(self) -> pd.DataFrame:
        """Fetch default league matches for current season."""
        return self.fetch_league_matches(season="2023")

    def _extract_json_from_script(self, html: str, var_name: str) -> Optional[Dict]:
        """
        Extract JSON data from embedded JavaScript variables in HTML.

        Understat embeds data as JavaScript variables like:
        var matchesData = JSON.parse('[{...}]');

        Args:
            html: HTML content
            var_name: JavaScript variable name to extract (e.g., "matchesData")

        Returns:
            Parsed JSON data or None if not found
        """
        import re
        import json

        # Pattern to match: var varName = JSON.parse('...');
        pattern = rf"var\s+{var_name}\s*=\s*JSON\.parse\('(.+?)'\)"

        match = re.search(pattern, html, re.DOTALL)
        if not match:
            # Try alternative pattern without JSON.parse
            pattern = rf"var\s+{var_name}\s*=\s*'(.+?)'"
            match = re.search(pattern, html, re.DOTALL)

        if not match:
            logger.warning(f"Could not find {var_name} in HTML")
            return None

        try:
            # Extract and decode the JSON string
            json_str = match.group(1)
            # Unescape the string
            json_str = json_str.encode().decode('unicode_escape')
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {var_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting {var_name}: {e}")
            return None

    def fetch_match_xg(self, match_id: int) -> Optional[Dict]:
        """
        Fetch xG data for a specific match.

        Args:
            match_id: Understat match ID

        Returns:
            Dictionary with match xG data:
            {
                'match_id': int,
                'home_team': str,
                'away_team': str,
                'home_xg': float,
                'away_xg': float,
                'home_xga': float,
                'away_xga': float,
                'home_score': int,
                'away_score': int,
                'date': str,
            }
        """
        cache_key = f"match_xg_{match_id}"

        # Check cache
        cache_path = self.cache_dir / f"{self.__class__.__name__}_{cache_key}.json"
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    cached = json.load(f)
                cached_at = datetime.fromisoformat(cached["cached_at"])
                if datetime.now() < cached_at + timedelta(hours=self.CACHE_EXPIRY_HOURS):
                    logger.info(f"Cache hit for match {match_id}")
                    return cached["data"]
            except Exception as e:
                logger.warning(f"Cache read error: {e}")

        try:
            self._rate_limit()

            url = f"{self.BASE_URL}/match/{match_id}"
            logger.info(f"Fetching match xG from {url}")

            response = self._session.get(url, timeout=30)
            response.raise_for_status()

            # Extract match info from page
            soup = BeautifulSoup(response.text, "html.parser")

            # Get team names and scores from page title or headers
            match_info = self._parse_match_page(soup, response.text)

            if match_info:
                # Cache the result
                try:
                    cache_data = {
                        "cached_at": datetime.now().isoformat(),
                        "data": match_info
                    }
                    with open(cache_path, "w") as f:
                        json.dump(cache_data, f)
                except Exception as e:
                    logger.warning(f"Failed to cache match data: {e}")

                return match_info

            return None

        except requests.RequestException as e:
            logger.error(f"Request failed for match {match_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch match xG: {e}")
            return None

    def _parse_match_page(self, soup: BeautifulSoup, html: str) -> Optional[Dict]:
        """
        Parse match page to extract xG data.

        Args:
            soup: BeautifulSoup object of the page
            html: Raw HTML content

        Returns:
            Dictionary with match data or None
        """
        try:
            # Extract match info from script tags
            match_info_json = self._extract_json_from_script(html, "match_info")

            if match_info_json:
                # Extract relevant fields
                home_team = match_info_json.get("h", {})
                away_team = match_info_json.get("a", {})

                return {
                    "match_id": match_info_json.get("id"),
                    "home_team": home_team.get("title", ""),
                    "away_team": away_team.get("title", ""),
                    "home_xg": float(home_team.get("xG", 0)),
                    "away_xg": float(away_team.get("xG", 0)),
                    "home_xga": float(away_team.get("xG", 0)),  # Away xG = Home xGA
                    "away_xga": float(home_team.get("xG", 0)),  # Home xG = Away xGA
                    "home_score": int(home_team.get("goals", 0)),
                    "away_score": int(away_team.get("goals", 0)),
                    "date": match_info_json.get("datetime", ""),
                }

            # Fallback: try to parse from page structure
            return self._parse_match_from_dom(soup)

        except Exception as e:
            logger.error(f"Error parsing match page: {e}")
            return None

    def _parse_match_from_dom(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Fallback method to parse match data from DOM structure."""
        try:
            # Look for team names in headers
            team_headers = soup.find_all("div", class_="team-name")
            if len(team_headers) >= 2:
                home_team = team_headers[0].get_text(strip=True)
                away_team = team_headers[1].get_text(strip=True)

                # Look for xG values
                xg_elements = soup.find_all("div", class_="xg")
                if len(xg_elements) >= 2:
                    home_xg = float(xg_elements[0].get_text(strip=True))
                    away_xg = float(xg_elements[1].get_text(strip=True))

                    return {
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_xg": home_xg,
                        "away_xg": away_xg,
                        "home_xga": away_xg,
                        "away_xga": home_xg,
                    }

            return None
        except Exception as e:
            logger.debug(f"DOM parsing failed: {e}")
            return None

    def fetch_match_shots(self, match_id: int) -> pd.DataFrame:
        """
        Fetch detailed shot data with xG for a specific match.

        Args:
            match_id: Understat match ID

        Returns:
            DataFrame with shot data:
            - player: Player name
            - team: Team name
            - minute: Match minute
            - xg: Expected goals value for the shot
            - result: Shot result (Goal, SavedShot, MissedShots, etc.)
            - situation: Play situation (OpenPlay, FromCorner, SetPiece, etc.)
            - x: Shot x coordinate
            - y: Shot y coordinate
        """
        cache_key = f"match_shots_{match_id}"

        # Check cache
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            self._rate_limit()

            url = f"{self.BASE_URL}/match/{match_id}"
            logger.info(f"Fetching shot data from {url}")

            response = self._session.get(url, timeout=30)
            response.raise_for_status()

            # Extract shots data from embedded JSON
            shots_data = self._extract_json_from_script(response.text, "shotsData")

            if not shots_data:
                logger.warning(f"No shots data found for match {match_id}")
                return self._empty_dataframe([
                    "player", "team", "minute", "xg", "result",
                    "situation", "x", "y"
                ])

            # Parse shots data
            shots = []
            for team_name, team_shots in shots_data.items():
                for shot in team_shots:
                    shots.append({
                        "player": shot.get("player", ""),
                        "team": team_name,
                        "minute": int(shot.get("minute", 0)),
                        "xg": float(shot.get("xG", 0)),
                        "result": shot.get("result", ""),
                        "situation": shot.get("situation", ""),
                        "x": float(shot.get("X", 0)),
                        "y": float(shot.get("Y", 0)),
                        "lastAction": shot.get("lastAction", ""),
                    })

            df = pd.DataFrame(shots)

            if not df.empty:
                self._save_cache(cache_key, df)
                logger.info(f"Fetched {len(df)} shots for match {match_id}")

            return df

        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            return self._empty_dataframe()
        except Exception as e:
            logger.error(f"Failed to fetch shot data: {e}")
            return self._empty_dataframe()

    def fetch_team_season_stats(self, team_name: str, season: str = "2023") -> pd.DataFrame:
        """
        Fetch season xG statistics for a specific team.

        Args:
            team_name: Team name (must match PREMIER_LEAGUE_TEAMS keys)
            season: Season year (e.g., "2023" for 2023/24 season)

        Returns:
            DataFrame with team's match-by-match xG data for the season
        """
        # Get team mapping
        team_mapping = PREMIER_LEAGUE_TEAMS.get(team_name)

        if team_mapping is None:
            logger.warning(f"Unknown team: {team_name}")
            return self._empty_dataframe([
                "date", "opponent", "home", "goals", "xG", "xGA", "result"
            ])

        cache_key = f"team_season_{team_mapping.understat_slug}_{season}"

        # Check cache
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            self._rate_limit()

            url = f"{self.BASE_URL}/team/{team_mapping.understat_slug}/{season}"
            logger.info(f"Fetching season stats for {team_name} from {url}")

            response = self._session.get(url, timeout=30)
            response.raise_for_status()

            # Extract matches data from embedded JSON
            matches_data = self._extract_json_from_script(response.text, "datesData")

            if not matches_data:
                logger.warning(f"No season data found for {team_name}")
                return self._empty_dataframe()

            # Parse matches data
            matches = []
            for match in matches_data:
                is_home = match.get("side", "h") == "h"

                matches.append({
                    "date": match.get("datetime", ""),
                    "opponent": match.get("title", ""),
                    "home": is_home,
                    "goals": int(match.get("goals", 0)),
                    "xG": float(match.get("xG", 0)),
                    "xGA": float(match.get("xGA", 0)),
                    "result": match.get("result", ""),
                    "match_id": match.get("id", ""),
                })

            df = pd.DataFrame(matches)

            if not df.empty:
                self._save_cache(cache_key, df)
                logger.info(f"Fetched {len(df)} matches for {team_name}")

            return df

        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            return self._empty_dataframe()
        except Exception as e:
            logger.error(f"Failed to fetch team season stats: {e}")
            return self._empty_dataframe()

    def fetch_league_matches(self, season: str = "2023") -> pd.DataFrame:
        """
        Fetch all Premier League matches for a season with xG data.

        Args:
            season: Season year (e.g., "2023" for 2023/24 season)

        Returns:
            DataFrame with all league matches and xG data
        """
        cache_key = f"league_matches_{season}"

        # Check cache
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            self._rate_limit()

            # EPL is the league code for Premier League
            url = f"{self.BASE_URL}/league/EPL/{season}"
            logger.info(f"Fetching league matches from {url}")

            response = self._session.get(url, timeout=30)
            response.raise_for_status()

            # Extract matches data from embedded JSON
            matches_data = self._extract_json_from_script(response.text, "datesData")

            if not matches_data:
                logger.warning(f"No league data found for season {season}")
                return self._empty_dataframe([
                    "match_id", "date", "home_team", "away_team",
                    "home_xg", "away_xg", "home_score", "away_score"
                ])

            # Parse matches data
            matches = []
            for match in matches_data:
                matches.append({
                    "match_id": match.get("id", ""),
                    "date": match.get("datetime", ""),
                    "home_team": match.get("h", {}).get("title", ""),
                    "away_team": match.get("a", {}).get("title", ""),
                    "home_xg": float(match.get("xG", {}).get("h", 0)),
                    "away_xg": float(match.get("xG", {}).get("a", 0)),
                    "home_score": int(match.get("goals", {}).get("h", 0)),
                    "away_score": int(match.get("goals", {}).get("a", 0)),
                })

            df = pd.DataFrame(matches)

            if not df.empty:
                self._save_cache(cache_key, df)
                logger.info(f"Fetched {len(df)} league matches for season {season}")

            return df

        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            return self._empty_dataframe()
        except Exception as e:
            logger.error(f"Failed to fetch league matches: {e}")
            return self._empty_dataframe()


# =============================================================================
# Convenience Factory Function
# =============================================================================

def create_fetcher(source: str, cache_dir: str = "backend/data/cache") -> BaseFetcher:
    """
    Factory function to create a fetcher by source name.

    Args:
        source: One of "statsbomb", "fbref", "transfermarkt", "understat"
        cache_dir: Cache directory path

    Returns:
        Appropriate fetcher instance

    Raises:
        ValueError: If unknown source specified
    """
    fetchers = {
        "statsbomb": StatsBombFetcher,
        "fbref": FBrefFetcher,
        "transfermarkt": TransferMarktFetcher,
        "understat": UnderstatFetcher,
    }

    if source.lower() not in fetchers:
        raise ValueError(f"Unknown source: {source}. Available: {list(fetchers.keys())}")

    return fetchers[source.lower()](cache_dir=cache_dir)

