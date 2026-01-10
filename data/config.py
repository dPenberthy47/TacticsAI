"""
TacticsAI Configuration
Football data API endpoints, team mappings, and formation definitions.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


# =============================================================================
# API Configuration
# =============================================================================

@dataclass(frozen=True)
class APIConfig:
    """API endpoint configuration."""
    base_url: str
    rate_limit_seconds: float = 1.0


API_URLS = {
    "fbref": APIConfig(
        base_url="https://fbref.com/en",
        rate_limit_seconds=3.0,  # FBref is strict about rate limiting
    ),
    "statsbomb": APIConfig(
        base_url="https://raw.githubusercontent.com/statsbomb/open-data/master/data",
        rate_limit_seconds=0.5,
    ),
    "understat": APIConfig(
        base_url="https://understat.com",
        rate_limit_seconds=2.0,
    ),
    "transfermarkt": APIConfig(
        base_url="https://www.transfermarkt.com",
        rate_limit_seconds=3.0,
    ),
}


# =============================================================================
# Premier League Teams (2024-25 Season)
# =============================================================================

@dataclass(frozen=True)
class TeamMapping:
    """Team identifiers across different data sources."""
    fbref_id: str
    fbref_slug: str
    statsbomb_id: int
    understat_id: int
    understat_slug: str
    transfermarkt_id: int
    transfermarkt_slug: str


PREMIER_LEAGUE_TEAMS: Dict[str, TeamMapping] = {
    "Arsenal": TeamMapping(
        fbref_id="18bb7c10",
        fbref_slug="Arsenal",
        statsbomb_id=1,
        understat_id=83,
        understat_slug="Arsenal",
        transfermarkt_id=11,
        transfermarkt_slug="fc-arsenal",
    ),
    "Aston Villa": TeamMapping(
        fbref_id="8602292d",
        fbref_slug="Aston-Villa",
        statsbomb_id=2,
        understat_id=71,
        understat_slug="Aston_Villa",
        transfermarkt_id=405,
        transfermarkt_slug="aston-villa",
    ),
    "Bournemouth": TeamMapping(
        fbref_id="4ba7cbea",
        fbref_slug="Bournemouth",
        statsbomb_id=127,
        understat_id=89,
        understat_slug="Bournemouth",
        transfermarkt_id=989,
        transfermarkt_slug="afc-bournemouth",
    ),
    "Brentford": TeamMapping(
        fbref_id="cd051869",
        fbref_slug="Brentford",
        statsbomb_id=160,
        understat_id=218,
        understat_slug="Brentford",
        transfermarkt_id=1148,
        transfermarkt_slug="fc-brentford",
    ),
    "Brighton": TeamMapping(
        fbref_id="d07537b9",
        fbref_slug="Brighton-and-Hove-Albion",
        statsbomb_id=30,
        understat_id=220,
        understat_slug="Brighton",
        transfermarkt_id=1237,
        transfermarkt_slug="brighton-hove-albion",
    ),
    "Chelsea": TeamMapping(
        fbref_id="cff3d9bb",
        fbref_slug="Chelsea",
        statsbomb_id=4,
        understat_id=80,
        understat_slug="Chelsea",
        transfermarkt_id=631,
        transfermarkt_slug="fc-chelsea",
    ),
    "Crystal Palace": TeamMapping(
        fbref_id="47c64c55",
        fbref_slug="Crystal-Palace",
        statsbomb_id=6,
        understat_id=78,
        understat_slug="Crystal_Palace",
        transfermarkt_id=873,
        transfermarkt_slug="crystal-palace",
    ),
    "Everton": TeamMapping(
        fbref_id="d3fd31cc",
        fbref_slug="Everton",
        statsbomb_id=7,
        understat_id=72,
        understat_slug="Everton",
        transfermarkt_id=29,
        transfermarkt_slug="fc-everton",
    ),
    "Fulham": TeamMapping(
        fbref_id="fd962109",
        fbref_slug="Fulham",
        statsbomb_id=8,
        understat_id=228,
        understat_slug="Fulham",
        transfermarkt_id=931,
        transfermarkt_slug="fc-fulham",
    ),
    "Ipswich Town": TeamMapping(
        fbref_id="b74092de",
        fbref_slug="Ipswich-Town",
        statsbomb_id=163,
        understat_id=268,
        understat_slug="Ipswich",
        transfermarkt_id=677,
        transfermarkt_slug="ipswich-town",
    ),
    "Leicester City": TeamMapping(
        fbref_id="a2d435b3",
        fbref_slug="Leicester-City",
        statsbomb_id=9,
        understat_id=75,
        understat_slug="Leicester",
        transfermarkt_id=1003,
        transfermarkt_slug="leicester-city",
    ),
    "Liverpool": TeamMapping(
        fbref_id="822bd0ba",
        fbref_slug="Liverpool",
        statsbomb_id=10,
        understat_id=87,
        understat_slug="Liverpool",
        transfermarkt_id=31,
        transfermarkt_slug="fc-liverpool",
    ),
    "Manchester City": TeamMapping(
        fbref_id="b8fd03ef",
        fbref_slug="Manchester-City",
        statsbomb_id=11,
        understat_id=88,
        understat_slug="Manchester_City",
        transfermarkt_id=281,
        transfermarkt_slug="manchester-city",
    ),
    "Manchester United": TeamMapping(
        fbref_id="19538871",
        fbref_slug="Manchester-United",
        statsbomb_id=12,
        understat_id=89,
        understat_slug="Manchester_United",
        transfermarkt_id=985,
        transfermarkt_slug="manchester-united",
    ),
    "Newcastle United": TeamMapping(
        fbref_id="b2b47a98",
        fbref_slug="Newcastle-United",
        statsbomb_id=13,
        understat_id=86,
        understat_slug="Newcastle_United",
        transfermarkt_id=762,
        transfermarkt_slug="newcastle-united",
    ),
    "Nottingham Forest": TeamMapping(
        fbref_id="e4a775cb",
        fbref_slug="Nottingham-Forest",
        statsbomb_id=14,
        understat_id=232,
        understat_slug="Nottingham_Forest",
        transfermarkt_id=703,
        transfermarkt_slug="nottingham-forest",
    ),
    "Southampton": TeamMapping(
        fbref_id="33c895d4",
        fbref_slug="Southampton",
        statsbomb_id=18,
        understat_id=74,
        understat_slug="Southampton",
        transfermarkt_id=180,
        transfermarkt_slug="fc-southampton",
    ),
    "Tottenham": TeamMapping(
        fbref_id="361ca564",
        fbref_slug="Tottenham-Hotspur",
        statsbomb_id=19,
        understat_id=82,
        understat_slug="Tottenham",
        transfermarkt_id=148,
        transfermarkt_slug="tottenham-hotspur",
    ),
    "West Ham": TeamMapping(
        fbref_id="7c21e445",
        fbref_slug="West-Ham-United",
        statsbomb_id=21,
        understat_id=81,
        understat_slug="West_Ham",
        transfermarkt_id=379,
        transfermarkt_slug="west-ham-united",
    ),
    "Wolves": TeamMapping(
        fbref_id="8cec06e1",
        fbref_slug="Wolverhampton-Wanderers",
        statsbomb_id=22,
        understat_id=229,
        understat_slug="Wolverhampton_Wanderers",
        transfermarkt_id=543,
        transfermarkt_slug="wolverhampton-wanderers",
    ),
}


# =============================================================================
# Formation Definitions
# =============================================================================
# Coordinates are (x, y) on a normalized 0-100 pitch
# x: 0 = left touchline, 100 = right touchline
# y: 0 = own goal line, 100 = opponent goal line

@dataclass(frozen=True)
class Position:
    """Player position on the pitch."""
    name: str
    x: float
    y: float


@dataclass(frozen=True)
class Formation:
    """Formation definition with player positions."""
    name: str
    positions: Tuple[Position, ...]
    
    def to_coordinates(self) -> List[Tuple[float, float]]:
        """Return list of (x, y) coordinate tuples."""
        return [(p.x, p.y) for p in self.positions]
    
    def to_dict(self) -> Dict[str, Tuple[float, float]]:
        """Return dict mapping position names to coordinates."""
        return {p.name: (p.x, p.y) for p in self.positions}


FORMATIONS: Dict[str, Formation] = {
    "4-3-3": Formation(
        name="4-3-3",
        positions=(
            Position("GK", 50, 5),
            Position("LB", 15, 25),
            Position("LCB", 35, 20),
            Position("RCB", 65, 20),
            Position("RB", 85, 25),
            Position("LCM", 30, 45),
            Position("CM", 50, 40),
            Position("RCM", 70, 45),
            Position("LW", 15, 70),
            Position("ST", 50, 82),
            Position("RW", 85, 70),
        ),
    ),
    "4-4-2": Formation(
        name="4-4-2",
        positions=(
            Position("GK", 50, 5),
            Position("LB", 15, 25),
            Position("LCB", 35, 20),
            Position("RCB", 65, 20),
            Position("RB", 85, 25),
            Position("LM", 15, 50),
            Position("LCM", 38, 45),
            Position("RCM", 62, 45),
            Position("RM", 85, 50),
            Position("LST", 38, 75),
            Position("RST", 62, 75),
        ),
    ),
    "4-2-3-1": Formation(
        name="4-2-3-1",
        positions=(
            Position("GK", 50, 5),
            Position("LB", 15, 25),
            Position("LCB", 35, 20),
            Position("RCB", 65, 20),
            Position("RB", 85, 25),
            Position("LCDM", 38, 38),
            Position("RCDM", 62, 38),
            Position("LAM", 20, 58),
            Position("CAM", 50, 55),
            Position("RAM", 80, 58),
            Position("ST", 50, 80),
        ),
    ),
    "3-5-2": Formation(
        name="3-5-2",
        positions=(
            Position("GK", 50, 5),
            Position("LCB", 28, 20),
            Position("CB", 50, 18),
            Position("RCB", 72, 20),
            Position("LWB", 10, 45),
            Position("LCM", 35, 42),
            Position("CM", 50, 38),
            Position("RCM", 65, 42),
            Position("RWB", 90, 45),
            Position("LST", 38, 72),
            Position("RST", 62, 72),
        ),
    ),
    "5-3-2": Formation(
        name="5-3-2",
        positions=(
            Position("GK", 50, 5),
            Position("LWB", 10, 30),
            Position("LCB", 30, 20),
            Position("CB", 50, 18),
            Position("RCB", 70, 20),
            Position("RWB", 90, 30),
            Position("LCM", 32, 48),
            Position("CM", 50, 44),
            Position("RCM", 68, 48),
            Position("LST", 38, 72),
            Position("RST", 62, 72),
        ),
    ),
    "3-4-3": Formation(
        name="3-4-3",
        positions=(
            Position("GK", 50, 5),
            Position("LCB", 28, 20),
            Position("CB", 50, 18),
            Position("RCB", 72, 20),
            Position("LM", 12, 48),
            Position("LCM", 38, 42),
            Position("RCM", 62, 42),
            Position("RM", 88, 48),
            Position("LW", 22, 72),
            Position("ST", 50, 78),
            Position("RW", 78, 72),
        ),
    ),
}


# =============================================================================
# Utility Functions
# =============================================================================

def get_team_by_name(name: str) -> TeamMapping | None:
    """Get team mapping by name (case-insensitive, partial match)."""
    name_lower = name.lower()
    for team_name, mapping in PREMIER_LEAGUE_TEAMS.items():
        if name_lower in team_name.lower():
            return mapping
    return None


def get_formation(name: str) -> Formation | None:
    """Get formation by name."""
    return FORMATIONS.get(name)


def list_formations() -> List[str]:
    """List all available formation names."""
    return list(FORMATIONS.keys())


def list_teams() -> List[str]:
    """List all Premier League team names."""
    return list(PREMIER_LEAGUE_TEAMS.keys())

