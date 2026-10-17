"""Loading the nflverse play-by-play data.

Terms used everywhere in the package:
    pos   = yardline_100 = yards to the opponent's end zone (1..99).
            pos = 75 is your own 25, pos = 50 is midfield, pos = 5 is 5 yards from scoring.
    togo  = yards to go for a 1st down.
"""
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
SEASONS = range(2018, 2026)
URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"

COLUMNS = [
    "season", "season_type", "week", "game_id", "posteam", "qtr", "half_seconds_remaining",
    "score_differential", "down", "ydstogo", "yardline_100", "play_type", "yards_gained",
    "first_down", "touchdown", "interception", "fumble_lost", "fourth_down_converted",
    "field_goal_result", "kick_distance", "extra_point_result", "touchback", "return_yards",
    "ep",
]


def download(seasons=SEASONS):
    """Download the raw play-by-play files into data/ (~20 MB per season)."""
    DATA_DIR.mkdir(exist_ok=True)
    for season in seasons:
        target = DATA_DIR / f"pbp_{season}.csv.gz"
        if not target.exists():
            print(f"downloading {season} ...")
            urlretrieve(URL.format(season=season), target)


def load_pbp(seasons=SEASONS, season_type="REG"):
    """Plays with only the columns we use: season_type "REG" (regular season, used to estimate
    the model) or "POST" (playoffs, only used for the comparison).  Cached in data/pbp_all.pkl."""
    cached = DATA_DIR / "pbp_all.pkl"
    if cached.exists():
        pbp = pd.read_pickle(cached)
    else:
        download(seasons)
        frames = [pd.read_csv(DATA_DIR / f"pbp_{s}.csv.gz", usecols=lambda c: c in COLUMNS,
                              low_memory=False) for s in seasons]
        pbp = pd.concat(frames, ignore_index=True)
        pbp.to_pickle(cached)
    return pbp[pbp["season_type"] == season_type].reset_index(drop=True)


def neutral(pbp):
    """Plays where the clock matters little: quarters 1-3, not the last 2 minutes of the half."""
    end_of_half = (pbp["qtr"] == 2) & (pbp["half_seconds_remaining"] <= 120)
    return (pbp["qtr"] <= 3) & ~end_of_half


def fourth_down_decisions(pbp):
    """Real 4th-down decisions in neutral situations, with action = go / fg / punt."""
    d = pbp[neutral(pbp) & (pbp["down"] == 4)
            & pbp["play_type"].isin(["run", "pass", "punt", "field_goal"])].copy()
    d["action"] = d["play_type"].map({"run": "go", "pass": "go", "punt": "punt",
                                      "field_goal": "fg"})
    d["togo"] = d["ydstogo"].astype(int)
    d["pos"] = d["yardline_100"].astype(int)
    return d
