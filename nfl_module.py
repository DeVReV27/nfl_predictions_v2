from __future__ import annotations
import pandas as pd
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
import datetime as _dt

# External dependency
try:
    import nfl_data_py as nfl
except Exception as e:  # pragma: no cover
    nfl = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

CURRENT_YEAR = _dt.datetime.now().year

class NFLDataError(RuntimeError):
    pass

def _check_lib():
    if nfl is None:
        raise NFLDataError(
            "nfl_data_py is not installed or failed to load. "
            "Install with `pip install nfl_data_py`."
        )

@lru_cache(maxsize=1)
def team_meta() -> pd.DataFrame:
    """Return NFL team metadata (colors, logos, names, abbreviations)."""
    _check_lib()
    meta = nfl.import_team_desc().copy()
    # normalize column names we rely on
    cols = {c.lower(): c for c in meta.columns}
    # expected columns include: team_abbr, team_name, team_nick, team_color, team_color2, team_logo_espn
    return meta

def list_teams() -> pd.DataFrame:
    df = team_meta()[['team_abbr','team_name','team_nick','team_color','team_color2','team_logo_espn']]
    df = df.rename(columns={'team_nick':'nickname'}).sort_values('team_name')
    return df

@lru_cache(maxsize=8)
def rosters(season: int) -> pd.DataFrame:
    """Seasonal rosters."""
    _check_lib()
    try:
        r = nfl.import_seasonal_rosters([season])
    except Exception:
        r = nfl.import_weekly_rosters([season]).drop_duplicates(subset=['player_id'])
    # Keep essentials
    keep = [c for c in ['player_id','player_name','recent_team','position','status','esb_id','gsis_id','height','weight','birth_date'] if c in r.columns]
    return r[keep].rename(columns={'recent_team':'team_abbr','player_name':'name'})

@lru_cache(maxsize=8)
def weekly_stats(season: int) -> pd.DataFrame:
    """Weekly player stats for given season (pre, reg, post included)."""
    _check_lib()
    cols = None
    try:
        cols = nfl.see_weekly_cols()
    except Exception:
        cols = None
    df = nfl.import_weekly_data([season], columns=cols, downcast=True)
    # harmonize key columns likely needed by UI
    # many columns exist; we will map safely with get
    rename_map = {
        'pass_yards':'passing_yards', 'pass_td':'passing_tds', 'pass_touchdowns':'passing_tds',
        'rush_yards':'rushing_yards','rush_td':'rushing_tds','rush_touchdowns':'rushing_tds',
        'rec_yards':'receiving_yards','rec_td':'receiving_tds','receptions':'receptions',
        'sacks':'sacks','sack':'sacks',
        'fgm':'field_goals_made','xpm':'extra_points_made','extra_points_made':'extra_points_made',
        'field_goals_made':'field_goals_made'
    }
    for a,b in rename_map.items():
        if a in df.columns and b not in df.columns:
            df[b] = df[a]
    return df

def team_roster(season: int, team_abbr: str) -> pd.DataFrame:
    r = rosters(season)
    return r[r['team_abbr'].eq(team_abbr)].sort_values(['position','name'])

def player_game_stats(season: int, team_abbr: str, player_name_or_id: str) -> pd.DataFrame:
    ws = weekly_stats(season)
    # match on player_id or name (case-insensitive)
    m = ws[(ws.get('recent_team', ws.get('team', ''))==team_abbr) & (
        ws.get('player_id','').astype(str).str.fullmatch(str(player_name_or_id), na=False) |
        ws.get('player_name','').str.contains(str(player_name_or_id), case=False, na=False)
    )]
    return m.sort_values(['week','game_id'])

def team_aggregates(season: int, team_abbr: str) -> pd.DataFrame:
    ws = weekly_stats(season)
    key = ws.get('recent_team', ws.get('team',''))
    team_df = ws[key.eq(team_abbr)]
    # Aggregate by player and stat
    agg_cols = [c for c in ['passing_yards','passing_tds','rushing_yards','rushing_tds','receiving_yards','receiving_tds','receptions','sacks','field_goals_made','extra_points_made'] if c in team_df.columns]
    grouped = (team_df
               .groupby(['player_id','player_name','position'], dropna=False)[agg_cols]
               .sum(numeric_only=True)
               .reset_index()
               .sort_values(['position', 'player_name']))
    return grouped

def schedule(season: int) -> pd.DataFrame:
    _check_lib()
    s = nfl.import_schedules([season])
    return s

def season_weeks_present(season: int) -> List[int]:
    df = weekly_stats(season)
    return sorted(df['week'].dropna().astype(int).unique().tolist())

def safe_stat_columns() -> List[str]:
    return ['passing_yards','rushing_yards','receiving_yards','field_goals_made','extra_points_made','passing_tds','rushing_tds','receptions','sacks']
