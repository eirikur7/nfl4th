"""
02_estimate.py
Metur allar líkur sem hermirinn þarf úr play-by-play gögnunum og vistar þær
í model.pkl.  Hver liður er metinn í sínu falli:

  estimate_plays()         dreifing yarda í hverju kerfi og líkur á töpuðum bolta,
                           skilyrt á (down, yards to go, svæði)
  estimate_conversion()    p(d) = P(1st down | d yards to go) á 3rd/4th down
  estimate_field_goals()   pFG(x) = P(FG heppnast | x yarda spark), logistic regression
  estimate_punts()         vallarstaða mótherja eftir punt, skilyrt á vallarstöðu
  estimate_kickoffs()      upphafsstaða sóknar eftir kickoff (2024-25 reglur)
  estimate_extra_point()   líkur á heppnuðu aukastigi
  estimate_coaches()       hegðun þjálfara: P(go / fg / punt | yards to go, vallarstaða)
  estimate_drive_outcomes() hvernig sóknir enda í raun (til að staðfesta hermi)

Síun: aðeins 1.-3. leikhluti og ekki síðustu 2 mín. fyrri hálfleiks, svo
klukka og staða í leik stjórni ekki ákvörðunum (hermirinn hefur enga klukku).
"""
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from common import (COACH_POS_EDGES, COACH_TOGO_EDGES, DATA_DIR, MODEL_FILE,
                    down_group, field_zone, togo_bucket)

MIN_PLAYS_PER_CELL = 80     # færri kerfi í hólfi -> nota hólf án svæðisskiptingar
PUNT_BUCKETS = range(30, 100, 5)


def neutral_situations(pbp):
    """True fyrir kerfi þar sem klukkan skiptir litlu máli."""
    end_of_first_half = (pbp["qtr"] == 2) & (pbp["half_seconds_remaining"] <= 120)
    return (pbp["qtr"] <= 3) & ~end_of_first_half


def fit_logistic(x, y):
    """ML-mat á b0, b1 í  P(y=1 | x) = 1 / (1 + exp(-(b0 + b1 x)))."""
    X = np.column_stack([np.ones_like(x, dtype=float), x])

    def negative_log_likelihood(b):
        z = X @ b
        return np.sum(np.logaddexp(0, z)) - np.sum(y * z)

    return minimize(negative_log_likelihood, np.zeros(2), method="BFGS").x


# ------------------------------------------------------------------ kerfi
def estimate_plays(pbp):
    """Yarda-dreifing og líkur á töpuðum bolta fyrir run/pass kerfi.

    Lykill: (down_group, togo_bucket, field_zone).  Að auki er til lykill með
    field_zone = -1 (öll svæði saman) sem varaleið fyrir fámenn hólf.
    """
    plays = pbp[neutral_situations(pbp) & pbp["play_type"].isin(["run", "pass"])
                & pbp["down"].notna()].copy()
    plays["dgroup"] = plays["down"].map(down_group)
    plays["tbucket"] = plays["ydstogo"].map(togo_bucket)
    plays["zone"] = plays["yardline_100"].map(field_zone)
    plays["turnover"] = (plays["interception"] == 1) | (plays["fumble_lost"] == 1)

    yards, turnover_prob = {}, {}

    def add(key, group):
        yards[key] = group.loc[~group["turnover"], "yards_gained"].to_numpy(dtype=int)
        turnover_prob[key] = group["turnover"].mean()

    for (dg, tb, zone), group in plays.groupby(["dgroup", "tbucket", "zone"]):
        if len(group) >= MIN_PLAYS_PER_CELL:
            add((dg, tb, zone), group)
    for (dg, tb), group in plays.groupby(["dgroup", "tbucket"]):
        add((dg, tb, -1), group)

    print(f"Kerfi notuð í yarda-dreifingu: {len(plays):,}")
    return {"play_yards": yards, "play_turnover_prob": turnover_prob}


def estimate_conversion(pbp):
    """Hlutfall 3rd/4th down kerfa þar sem 1st down (eða TD) næst, eftir togo."""
    plays = pbp[neutral_situations(pbp) & pbp["play_type"].isin(["run", "pass"])
                & (pbp["down"] >= 3) & (pbp["ydstogo"] <= 20)].copy()
    plays["success"] = (plays["first_down"] == 1) | (plays["touchdown"] == 1)
    both = plays.groupby("ydstogo")["success"].agg(["mean", "count"])
    fourth = plays[plays["down"] == 4].groupby("ydstogo")["success"].agg(["mean", "count"])
    print("\np(d) á 3rd/4th down:")
    print(both.head(10).round(3).to_string())
    return {"conversion_3rd_4th": both, "conversion_4th": fourth}


def estimate_field_goals(pbp):
    """pFG(x) með logistic regression, og bilið milli línu og sparkstaðar."""
    fg = pbp[(pbp["play_type"] == "field_goal") & pbp["kick_distance"].notna()]
    made = (fg["field_goal_result"] == "made").to_numpy(float)
    distance = fg["kick_distance"].to_numpy(float)
    b0, b1 = fit_logistic(distance, made)

    by_distance = (pd.DataFrame({"made": made, "bin": distance // 5 * 5})
                   .groupby("bin")["made"].agg(["mean", "count"]))
    offset = int(round((fg["kick_distance"] - fg["yardline_100"]).median()))

    print(f"\npFG(x) = 1 / (1 + exp(-({b0:.3f} + {b1:.4f} x)))   ({len(fg):,} spörk)")
    for x in [30, 40, 50, 55, 60]:
        print(f"  pFG({x}) = {1 / (1 + np.exp(-(b0 + b1 * x))):.3f}")
    print(f"  sparklengd = pos + {offset}")
    return {"fg_logit": (b0, b1), "fg_by_distance": by_distance, "fg_kick_offset": offset}


def estimate_punts(pbp):
    """Fyrir hvert 5 yarda bil af vallarstöðu: vallarstaða mótherja eftir punt."""
    punts = pbp[pbp["play_type"] == "punt"]
    landing_pos = punts["yardline_100"] - punts["kick_distance"].fillna(0)
    opp_pos = 100 - landing_pos - punts["return_yards"].fillna(0)
    opp_pos = np.where(punts["touchback"] == 1, 80, opp_pos)
    opp_pos = pd.Series(np.clip(opp_pos, 1, 99), index=punts.index)

    next_pos = {}
    for b in PUNT_BUCKETS:
        in_bucket = (punts["yardline_100"] > b - 5) & (punts["yardline_100"] <= b)
        if in_bucket.sum() < 50:     # of fá punt: víkka bilið
            in_bucket = (punts["yardline_100"] > b - 10) & (punts["yardline_100"] <= b + 5)
        next_pos[b] = opp_pos[in_bucket].to_numpy(dtype=int)

    net = (punts["yardline_100"] - (100 - opp_pos)).mean()
    print(f"\nPunt: meðalnettó = {net:.1f} yardar")
    return {"punt_next_pos": next_pos}


def estimate_kickoffs(pbp):
    """Vallarstaða í fyrsta kerfi eftir kickoff (nýju reglurnar frá 2024)."""
    pbp = pbp.reset_index(drop=True)
    after_kickoff = pbp.index[pbp["kickoff_attempt"] == 1] + 1
    after_kickoff = after_kickoff[after_kickoff < len(pbp)]
    nxt = pbp.loc[after_kickoff]
    nxt = nxt[(nxt["down"] == 1) & nxt["season"].isin([2024, 2025])]
    start = nxt["yardline_100"].dropna().to_numpy(dtype=int)
    print(f"Kickoff: sókn byrjar að meðaltali á eigin {100 - start.mean():.1f} yarda línu")
    return {"kickoff_start_pos": start}


def estimate_extra_point(pbp):
    xp = pbp[pbp["play_type"] == "extra_point"]
    prob = (xp["extra_point_result"] == "good").mean()
    print(f"P(aukastig heppnast) = {prob:.3f}")
    return {"extra_point_prob": prob}


def estimate_coaches(pbp):
    """P(go / fg / punt) á 4th down í hverju (togo-bili, vallarstöðubili)."""
    fourth = pbp[neutral_situations(pbp) & (pbp["down"] == 4)
                 & pbp["play_type"].isin(["run", "pass", "punt", "field_goal"])].copy()
    fourth["action"] = fourth["play_type"].map(
        {"run": "go", "pass": "go", "punt": "punt", "field_goal": "fg"})
    fourth["togo_cell"] = pd.cut(fourth["ydstogo"], COACH_TOGO_EDGES, labels=False)
    fourth["pos_cell"] = pd.cut(fourth["yardline_100"], COACH_POS_EDGES, labels=False)

    counts = (fourth.groupby(["togo_cell", "pos_cell"])["action"].value_counts()
              .unstack(fill_value=0).reindex(columns=["go", "fg", "punt"], fill_value=0))
    # Laplace-jöfnun (+0.5) svo engar líkur verði nákvæmlega 0 í fámennum hólfum
    probs = (counts + 0.5).div((counts + 0.5).sum(axis=1), axis=0)

    go_by_season = fourth.groupby("season")["action"].apply(lambda s: (s == "go").mean())
    print(f"\n4th down kerfi (1.-3. leikhluti): {len(fourth):,}")
    print("Hlutfall 4th down þar sem reynt er, eftir tímabili:")
    print(go_by_season.round(3).to_string())
    return {"coach_probs": probs, "coach_counts": counts,
            "go_rate_by_season": go_by_season, "n_fourth_downs": len(fourth)}


def estimate_drive_outcomes(pbp):
    """Hlutfall sókna sem enda á TD, FG, punti o.s.frv. (1.-3. leikhluti).

    Sóknir sem enda á „End of half“ eru ekki taldar, því hermirinn hefur enga klukku.
    """
    first_play = pbp.dropna(subset=["fixed_drive_result"]).groupby(
        ["game_id", "fixed_drive"]).first()
    first_play = first_play[first_play["qtr"] <= 3]
    names = {"Touchdown": "TD", "Field goal": "FG", "Punt": "punt",
             "Turnover": "turnover", "Opp touchdown": "turnover",
             "Turnover on downs": "downs", "Missed field goal": "FG miss",
             "Safety": "safety"}
    result = first_play["fixed_drive_result"].map(names).dropna()
    shares = result.value_counts(normalize=True).to_dict()
    print("\nÚtkomur sókna í gögnum:", {k: round(v, 3) for k, v in shares.items()})
    return {"drive_outcomes_data": shares}


def main():
    pbp = pd.read_pickle(DATA_DIR / "pbp_small.pkl").reset_index(drop=True)
    model = {}
    for estimate in [estimate_plays, estimate_conversion, estimate_field_goals,
                     estimate_punts, estimate_kickoffs, estimate_extra_point,
                     estimate_coaches, estimate_drive_outcomes]:
        model.update(estimate(pbp))

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    print(f"\nVistað: {MODEL_FILE.name}")


if __name__ == "__main__":
    main()
