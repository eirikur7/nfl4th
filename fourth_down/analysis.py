"""Higher-level analyses used by the notebook.

    setup()               load data, estimate parameters, build and solve the chain
    bootstrap()           Monte Carlo over parameter uncertainty -> P(go is best) per state (cached)
    score_decisions()     every real 4th-down decision valued with the model: EP lost vs best
    compare_rules()       simple decision rules vs the coaches, in expected points per game
    go_thresholds()       the handbook condensed: go for it if togo <= k in each field zone
    situations()          the "fun / unexpected" situations, with all numbers side by side
"""
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data, estimate, markov

FG_RULE_MAX_POS = 37          # simple rules kick a FG from pos <= 37 (55-yard kick), else punt


@dataclass
class Setup:
    pbp: pd.DataFrame
    decisions: pd.DataFrame
    params: estimate.Params
    chain: markov.Chain
    coach_policy: np.ndarray
    ep: np.ndarray              # EP of every state when everyone plays like the coaches
    handbook: pd.DataFrame


def setup():
    pbp = data.load_pbp()
    decisions = data.fourth_down_decisions(pbp)
    params = estimate.estimate(pbp, decisions)
    chain = markov.Chain(params)
    policy = chain.coach_policy()
    ep = chain.solve(policy)
    return Setup(pbp, decisions, params, chain, policy, ep, chain.handbook(ep))


# ------------------------------------------------------------------ bootstrap
def bootstrap(s, n_boot=200, seed=0, recompute=False):
    """Re-estimate every probability from a bootstrap resample of the data, re-solve the chain
    and recompute the handbook, n_boot times.  Cached in cache/bootstrap.npz."""
    path = data.CACHE_DIR / "bootstrap.npz"
    if path.exists() and not recompute:
        return dict(np.load(path))
    rng = np.random.default_rng(seed)
    go_minus_kick, best, p_conv, ep_first = [], [], [], []
    first = s.chain.index(1, np.minimum(10, np.arange(1, 100)), np.arange(1, 100))
    start = time.time()
    for b in range(n_boot):
        params = estimate.estimate(s.pbp, s.decisions, rng)
        chain = markov.Chain(params)
        ep = chain.solve(chain.coach_policy())
        hb = chain.handbook(ep)
        go_minus_kick.append(hb["go_minus_kick"].to_numpy())
        best.append(hb["best"].map({"go": 0, "fg": 1, "punt": 2}).to_numpy())
        p_conv.append(hb["p_conv"].to_numpy())
        ep_first.append(ep[first])
        if b % 20 == 0:
            print(f"bootstrap {b}/{n_boot}  ({time.time() - start:.0f} s)")
    out = {"go_minus_kick": np.array(go_minus_kick), "best": np.array(best),
           "p_conv": np.array(p_conv), "ep_first": np.array(ep_first)}
    data.CACHE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def add_bootstrap(hb, boot):
    """P(go is best), P(each action is best) and a 95% interval for go_minus_kick."""
    hb = hb.copy()
    for k, a in enumerate(markov.ACTIONS):
        hb[f"p_best_{a}"] = (boot["best"] == k).mean(axis=0)
    hb["gmk_low"], hb["gmk_high"] = np.percentile(boot["go_minus_kick"], [2.5, 97.5], axis=0)
    return hb


# ------------------------------------------------------------------ real decisions
def team_games(pbp):
    return pbp.groupby(["season", "game_id", "posteam"]).ngroups


def score_decisions(s, decisions=None):
    """Value every real (neutral) 4th-down decision with the model.

    ep_lost = Q(best action) - Q(action the coach chose), in expected points.
    By default the regular-season decisions; pass others (e.g. playoffs) to score those.
    """
    d = (s.decisions if decisions is None else decisions).copy()
    k = np.searchsorted(s.chain.fourth, s.chain.index(4, d["togo"].to_numpy(), d["pos"].to_numpy()))
    hb = s.handbook.iloc[k].reset_index(drop=True)
    d = d.reset_index(drop=True)
    for col in ["q_go", "q_fg", "q_punt", "best", "gain", "p_conv", "breakeven"]:
        d[col] = hb[col].to_numpy()
    d["p_go_coach"] = s.coach_policy[k, 0]      # P(go) for a regular-season coach in this spot
    q_chosen = d[["q_go", "q_fg", "q_punt"]].to_numpy()[np.arange(len(d)),
                                                          d["action"].map({"go": 0, "fg": 1, "punt": 2})]
    d["q_chosen"] = q_chosen
    d["q_best"] = d[["q_go", "q_fg", "q_punt"]].max(axis=1)
    d["ep_lost"] = (d["q_best"] - d["q_chosen"]).fillna(0)
    d["agrees"] = d["action"] == d["best"]
    return d


def compare_rules(scored, n_team_games):
    """Expected points per game gained over what the coaches actually did, for simple rules.

    Every rule is evaluated on the same real 4th-down situations (law of total expectation:
    average over the situations teams actually face).  When a rule does not go for it, it
    kicks a FG from pos <= 37 (a 55-yard kick) and punts otherwise.
    """
    d = scored
    kick = np.where(d["pos"] <= FG_RULE_MAX_POS, d["q_fg"], d["q_punt"])
    kick = np.where(np.isnan(kick), d["q_go"], kick)

    def per_game(q):
        return (np.nansum(q - d["q_chosen"]) / n_team_games)

    rules = {"Coaches (what they did)": d["q_chosen"],
             "Never go for it": kick,
             "Always go for it": d["q_go"]}
    for x in [1, 2, 3, 4, 5, 6, 7, 8, 10]:
        rules[f"Go if togo ≤ {x}"] = np.where(d["togo"] <= x, d["q_go"], kick)
    rules["Handbook (best action every time)"] = d["q_best"]
    out = pd.DataFrame({"rule": list(rules),
                        "go_rate": [np.nan] * len(rules),
                        "ep_per_game_vs_coaches": [per_game(np.asarray(q)) for q in rules.values()]})
    go = {"Coaches (what they did)": (d["action"] == "go").mean(), "Never go for it": 0.0,
          "Always go for it": 1.0, "Handbook (best action every time)": (d["best"] == "go").mean()}
    for x in [1, 2, 3, 4, 5, 6, 7, 8, 10]:
        go[f"Go if togo ≤ {x}"] = (d["togo"] <= x).mean()
    out["go_rate"] = out["rule"].map(go)
    return out


def go_thresholds(hb, zones=range(0, 100, 10)):
    """For each 10-yard zone: go for it with up to k yards to go, where k is the largest togo
    such that going for it is best for every togo <= k in that whole zone."""
    rows = []
    for z in zones:
        part = hb[(hb["pos"] > z) & (hb["pos"] <= z + 10)]
        k = 0
        for t in range(1, 31):
            cell = part[part["togo"] == t]
            if len(cell) and (cell["best"] == "go").all():
                k = t
            else:
                break
        kick = part[(part["togo"] > k) & (part["best"] != "go")]["best"].mode()
        rows.append({"zone (yards to end zone)": f"{z + 1}-{z + 10}",
                     "go for it if togo ≤": k,
                     "otherwise": kick.iloc[0] if len(kick) else "-"})
    return pd.DataFrame(rows)


def costliest(scored, n_team_games, top=12):
    """The situations where coaches lose the most expected points in total."""
    d = scored.copy()
    d["togo "] = np.where(d["togo"] <= 5, d["togo"].astype(str), np.where(d["togo"] <= 10, "6-10", "11+"))
    d["pos zone"] = ((d["pos"] - 1) // 10 * 10 + 1).astype(str) + "-" + ((d["pos"] - 1) // 10 * 10 + 10).astype(str)
    g = d.groupby(["togo ", "pos zone"]).agg(
        plays=("ep_lost", "size"),
        coaches_go=("action", lambda a: (a == "go").mean()),
        handbook=("best", lambda b: b.mode().iloc[0]),
        ep_lost_total=("ep_lost", "sum"))
    g["ep_lost_per_100_games"] = 100 * g["ep_lost_total"] / n_team_games
    g["ep_lost_per_play"] = g["ep_lost_total"] / g["plays"]
    return g.sort_values("ep_lost_total", ascending=False).head(top).drop(columns="ep_lost_total")


def teams(scored, pbp):
    """EP lost per game and go rate for every team-season."""
    games = pbp.dropna(subset=["posteam"]).drop_duplicates(["game_id", "posteam"]).groupby(["season", "posteam"]).size()
    g = scored.groupby(["season", "posteam"]).agg(
        go_rate=("action", lambda a: (a == "go").mean()),
        agree=("agrees", "mean"), ep_lost=("ep_lost", "sum"))
    g["ep_lost_per_game"] = g["ep_lost"] / games.reindex(g.index)
    return g.drop(columns="ep_lost").sort_values("ep_lost_per_game")


# ------------------------------------------------------------------ playoffs
def _per_team_game(scored, pbp):
    """EP lost in every team-game (0 for team-games without a neutral 4th down)."""
    games = pbp.dropna(subset=["posteam"]).drop_duplicates(["game_id", "posteam"])
    lost = scored.groupby(["game_id", "posteam"])["ep_lost"].sum()
    return lost.reindex(pd.MultiIndex.from_frame(games[["game_id", "posteam"]]), fill_value=0)


def playoffs_vs_regular(s, scored_reg, scored_post, pbp_post):
    """Are coaches more cautious in the playoffs?  Three comparisons:

    1. Go rate adjusted for the situation.  Expected number of attempts if every playoff coach
       went for it as often as a regular-season coach in the same spot (law of total probability):
       E = sum of P(go | situation).  Var = sum p(1 - p) (sum of independent Bernoullis).
       z = (observed - expected) / sqrt(Var).
    2. Go rate in clear go-for-it spots (handbook says go and it gains >= 0.5 EP),
       two-proportion z-test.
    3. EP lost per team-game, two-sample z-test (CLT on the per-game totals).
    """
    from scipy.stats import norm

    def pval(z):
        return 2 * (1 - norm.cdf(abs(z)))

    rows = []
    for name, d in [("regular season", scored_reg), ("playoffs", scored_post)]:
        go = (d["action"] == "go").to_numpy()
        p = d["p_go_coach"].to_numpy()
        z = (go.sum() - p.sum()) / np.sqrt(np.sum(p * (1 - p)))
        rows.append({"games": name, "4th downs": len(d), "went for it": go.sum(),
                     "expected (regular-season coach)": p.sum(),
                     "actual / expected": go.sum() / p.sum(), "z": z, "p-value": pval(z)})
    situation = pd.DataFrame(rows).set_index("games")

    clear = [d[(d["best"] == "go") & (d["gain"] >= 0.5)] for d in (scored_reg, scored_post)]
    rates = [(c["action"] == "go").mean() for c in clear]
    n = [len(c) for c in clear]
    pooled = (rates[0] * n[0] + rates[1] * n[1]) / sum(n)
    z = (rates[1] - rates[0]) / np.sqrt(pooled * (1 - pooled) * (1 / n[0] + 1 / n[1]))
    clear_spots = pd.DataFrame({
        "clear go spots": n, "coaches went for it": rates,
        "95% CI ±": [1.96 * np.sqrt(r * (1 - r) / k) for r, k in zip(rates, n)]},
        index=["regular season", "playoffs"])
    clear_spots["z (playoffs vs regular)"] = [np.nan, z]
    clear_spots["p-value"] = [np.nan, pval(z)]

    lost = [_per_team_game(scored_reg, s.pbp), _per_team_game(scored_post, pbp_post)]
    means = [x.mean() for x in lost]
    ses = [x.std() / np.sqrt(len(x)) for x in lost]
    z = (means[1] - means[0]) / np.sqrt(ses[0] ** 2 + ses[1] ** 2)
    ep = pd.DataFrame({"team-games": [len(x) for x in lost], "EP lost per game": means,
                       "95% CI ±": [1.96 * e for e in ses]}, index=["regular season", "playoffs"])
    ep["z (playoffs vs regular)"] = [np.nan, z]
    ep["p-value"] = [np.nan, pval(z)]
    return situation, clear_spots, ep


# ------------------------------------------------------------------ fun situations
SITUATIONS = [
    ("4th & 1 on your own 20", 1, 80),
    ("4th & 1 on your own 5", 1, 95),
    ("4th & 2 on your own 30", 2, 70),
    ("4th & 4 at your own 45", 4, 55),
    ("4th & 10 at midfield", 10, 50),
    ("No man's land: 4th & 5 at the opponent's 36", 5, 36),
    ("Long FG or punt: 4th & 8 at the opponent's 40", 8, 40),
    ("4th & 2 at the opponent's 30", 2, 30),
    ("4th & 15 at the opponent's 35", 15, 35),
    ("4th & goal from the 1", 1, 1),
    ("4th & goal from the 3", 3, 3),
    ("4th & 6 at the opponent's 8", 6, 8),
]


def situations(s, hb, scored, items=SITUATIONS):
    rows = []
    for name, togo, pos in items:
        r = hb[(hb["togo"] == togo) & (hb["pos"] == pos)].iloc[0]
        real = scored[(scored["togo"] == togo) & (scored["pos"].between(pos - 2, pos + 2))]
        rows.append({
            "situation": name, "togo": togo, "pos": pos,
            "EP go": r["q_go"], "EP fg": r["q_fg"], "EP punt": r["q_punt"],
            "best": r["best"], "EP gained vs 2nd best": r["gain"],
            "P(convert)": r["p_conv"], "break-even P": r["breakeven"],
            "P(FG good)": r["p_fg"],
            "P(go is best), bootstrap": r.get("p_best_go", np.nan),
            "coaches go %": (real["action"] == "go").mean() if len(real) else np.nan,
            "n real plays (±2 yd)": len(real),
        })
    return pd.DataFrame(rows)
