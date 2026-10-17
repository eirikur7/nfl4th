"""Estimating the probabilities the model needs from play-by-play data.

    play outcomes   distribution of yards gained + P(turnover), per (down, togo, field zone)
    field goals     pFG(x) = P(FG good | kick distance x), logistic regression
    punts           where the opponent starts after a punt, per field-position bucket
    extra point     P(XP good)
    coaches         P(go / fg / punt | togo, pos) on 4th down

estimate(pbp, rng) re-estimates everything from a bootstrap resample of the data when an
rng is given.  That is how parameter uncertainty is carried into the results.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .data import neutral

MIN_PLAYS_PER_CELL = 80
PUNT_BUCKETS = range(30, 100, 5)
COACH_TOGO_EDGES = [0, 1, 2, 3, 5, 7, 10, 15, 99]      # cells (a, b]: 1, 2, 3, 4-5, 6-7, ...
COACH_POS_EDGES = list(range(0, 101, 10))


# ------------------------------------------------------------------ cells
def togo_bucket(togo):
    """1..10 separately, 11-15 -> 11, 16+ -> 12."""
    return np.where(togo <= 10, togo, np.where(togo <= 15, 11, 12)).astype(int)


def field_zone(pos):
    """0 = red zone (pos <= 20), 1 = elsewhere."""
    return (np.asarray(pos) > 20).astype(int)


def down_group(down):
    """3rd and 4th down share one distribution: 4th-down attempts are rare and
    3rd down is the same situation (one play to gain the yards)."""
    return np.minimum(np.asarray(down), 3).astype(int)


@dataclass
class Params:
    play_yards: dict        # (down_group, togo_bucket, zone) -> (yards values, probabilities)
    turnover: dict          # same key -> P(interception or lost fumble)
    fg_logit: tuple         # (b0, b1)
    fg_offset: int          # kick distance = pos + fg_offset
    punt_next_pos: dict     # bucket -> (opponent pos values, probabilities)
    xp: float
    coach: pd.DataFrame     # index (togo_cell, pos_cell), columns go / fg / punt

    def play_cell(self, down, togo, pos):
        key = (int(down_group(down)), int(togo_bucket(togo)), int(field_zone(pos)))
        return key if key in self.play_yards else (key[0], key[1], -1)

    def p_fg(self, pos):
        b0, b1 = self.fg_logit
        return 1 / (1 + np.exp(-(b0 + b1 * (np.asarray(pos) + self.fg_offset))))


def _resample(df, rng):
    return df if rng is None else df.iloc[rng.integers(0, len(df), len(df))]


def _pmf(values):
    v, c = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return v, c / c.sum()


# ------------------------------------------------------------------ pieces
def run_pass_plays(pbp):
    plays = pbp[neutral(pbp) & pbp["play_type"].isin(["run", "pass"]) & pbp["down"].notna()].copy()
    plays["dgroup"] = down_group(plays["down"])
    plays["tbucket"] = togo_bucket(plays["ydstogo"].astype(int))
    plays["zone"] = field_zone(plays["yardline_100"])
    plays["turnover"] = (plays["interception"] == 1) | (plays["fumble_lost"] == 1)
    return plays


def estimate_plays(plays):
    yards, turnover = {}, {}

    def add(key, g):
        yards[key] = _pmf(g.loc[~g["turnover"], "yards_gained"])
        turnover[key] = g["turnover"].mean()

    for key, g in plays.groupby(["dgroup", "tbucket", "zone"]):
        if len(g) >= MIN_PLAYS_PER_CELL:
            add(tuple(int(k) for k in key), g)
    for (dg, tb), g in plays.groupby(["dgroup", "tbucket"]):      # fallback: zones pooled
        add((int(dg), int(tb), -1), g)
    return yards, turnover


def field_goals(pbp):
    return pbp[(pbp["play_type"] == "field_goal") & pbp["kick_distance"].notna()]


def fit_logistic(x, y):
    """Maximum likelihood for P(y=1 | x) = 1 / (1 + exp(-(b0 + b1 x)))."""
    X = np.column_stack([np.ones_like(x, dtype=float), x])

    def nll(b):
        z = X @ b
        return np.sum(np.logaddexp(0, z)) - np.sum(y * z)

    return tuple(minimize(nll, np.zeros(2), method="BFGS").x)


def punts(pbp):
    p = pbp[pbp["play_type"] == "punt"].copy()
    landing = p["yardline_100"] - p["kick_distance"].fillna(0)
    opp = 100 - landing - p["return_yards"].fillna(0)
    p["opp_pos"] = np.clip(np.where(p["touchback"] == 1, 80, opp), 1, 99).astype(int)
    return p


def estimate_punts(p):
    out = {}
    for b in PUNT_BUCKETS:
        sel = (p["yardline_100"] > b - 5) & (p["yardline_100"] <= b)
        if sel.sum() < 50:
            sel = (p["yardline_100"] > b - 10) & (p["yardline_100"] <= b + 5)
        out[b] = _pmf(p.loc[sel, "opp_pos"])
    return out


def coach_cells(decisions):
    d = decisions.copy()
    d["togo_cell"] = pd.cut(d["togo"], COACH_TOGO_EDGES, labels=False)
    d["pos_cell"] = pd.cut(d["pos"], COACH_POS_EDGES, labels=False)
    return d


def estimate_coaches(decisions):
    d = coach_cells(decisions)
    counts = (d.groupby(["togo_cell", "pos_cell"])["action"].value_counts().unstack(fill_value=0)
              .reindex(columns=["go", "fg", "punt"], fill_value=0))
    return (counts + 0.5).div((counts + 0.5).sum(axis=1), axis=0)    # Laplace smoothing


# ------------------------------------------------------------------ all
def estimate(pbp, decisions, rng=None):
    """All parameters.  With an rng, every data set is bootstrap-resampled first."""
    plays = _resample(run_pass_plays(pbp), rng)
    yards, turnover = estimate_plays(plays)

    fg = _resample(field_goals(pbp), rng)
    b = fit_logistic(fg["kick_distance"].to_numpy(float),
                     (fg["field_goal_result"] == "made").to_numpy(float))
    offset = int(round((fg["kick_distance"] - fg["yardline_100"]).median()))

    xp = pbp[pbp["play_type"] == "extra_point"]
    return Params(
        play_yards=yards, turnover=turnover, fg_logit=b, fg_offset=offset,
        punt_next_pos=estimate_punts(_resample(punts(pbp), rng)),
        xp=(_resample(xp, rng)["extra_point_result"] == "good").mean(),
        coach=estimate_coaches(_resample(decisions, rng)),
    )
