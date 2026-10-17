"""
01_load_data.py
Les play-by-play gögn frá nflverse (2018-2025), heldur aðeins deildarkeppni og
þeim dálkum sem við notum, og vistar í data/pbp_small.pkl.

Gögnin (play_by_play_YYYY.csv.gz) eru sótt af
https://github.com/nflverse/nflverse-data/releases/tag/pbp
og vistuð sem data/pbp_YYYY.csv.gz.  Þetta eru sömu gögn og
nflreadpy.load_pbp() skilar.
"""
import pandas as pd

from common import DATA_DIR, SEASONS

COLUMNS = [
    # leikur og klukka
    "season", "season_type", "game_id", "qtr", "half_seconds_remaining",
    # ástand fyrir kerfið
    "posteam", "down", "ydstogo", "yardline_100",
    # kerfið og útkoma þess
    "play_type", "yards_gained", "first_down", "touchdown",
    "fourth_down_converted", "fourth_down_failed",
    "third_down_converted", "third_down_failed",
    "interception", "fumble_lost", "safety", "penalty",
    # spörk
    "field_goal_result", "kick_distance", "extra_point_result",
    "punt_blocked", "touchback", "return_yards",
    "kickoff_attempt", "punt_attempt", "field_goal_attempt",
    # sóknir
    "drive", "fixed_drive", "fixed_drive_result", "drive_start_yard_line",
    "qb_kneel", "qb_spike", "wp",
]


def main():
    frames = []
    for season in SEASONS:
        df = pd.read_csv(DATA_DIR / f"pbp_{season}.csv.gz",
                         usecols=lambda c: c in COLUMNS, low_memory=False)
        frames.append(df)
        print(f"{season}: {len(df):,} kerfi")

    pbp = pd.concat(frames, ignore_index=True)
    pbp = pbp[pbp["season_type"] == "REG"]
    pbp.to_pickle(DATA_DIR / "pbp_small.pkl")
    print(f"Samtals: {len(pbp):,} kerfi í deildarkeppni")


if __name__ == "__main__":
    main()
