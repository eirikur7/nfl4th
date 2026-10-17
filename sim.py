"""
sim.py
Markov-keðju hermir fyrir NFL sóknir og leiki.

Ástand sóknar: (down, yards to go, vallarstaða)
  vallarstaða pos = yardline_100 = fjarlægð að endamarki mótherja (1..99)

Hver sókn er runa ástanda sem endar í einu af gleypiástöndunum:
  TD, FG (heppnað), misheppnað FG, punt, tapaður bolti, tapað á downs, safety.

Leikur: liðin skiptast á, 12 sóknir hvort.  Engin klukka, jöfn lið,
engar 2-point tilraunir, engin framlenging (jafntefli telst hálfur sigur).
"""
import math
import pickle
import random
from pathlib import Path

ROOT = Path(__file__).parent
with open(ROOT / "model.pkl", "rb") as f:
    M = pickle.load(f)

# Breyta numpy fylkjum í python lista (hraðara að draga úr með random)
YARDS = {k: v.tolist() for k, v in M["yards"].items()}
TO_P = dict(M["to_prob"])
PUNT = {k: v.tolist() for k, v in M["punt"].items()}
KICKOFF = M["kickoff"].tolist()
B0, B1 = M["fg_coef"]
FG_OFFSET = M["fg_offset"]
XP = M["xp_prob"]
COACH = M["coach"]["table"]
TOGO_EDGES = M["coach"]["togo_edges"]
POS_EDGES = M["coach"]["pos_edges"]

FG_RANGE = 37          # reynum FG ef pos <= 37  (spark <= 55 yardar)
DRIVES_PER_TEAM = 12

# Þjálfaratafla sem uppflettitafla: (tcut, pcut) -> (P(go), P(go)+P(fg))
COACH_LU = {}
for (tc, pc), row in COACH.iterrows():
    # FG er ómögulegt utan ~63 yarda (pos > 45); jöfnunin gaf þar smá FG-líkur
    fg = row["fg"] if POS_EDGES[int(pc)] < 45 else 0.0
    COACH_LU[(int(tc), int(pc))] = (row["go"], row["go"] + fg)


def p_fg(pos):
    x = pos + FG_OFFSET
    return 1.0 / (1.0 + math.exp(-(B0 + B1 * x)))


def _tb(t):
    return t if t <= 10 else (11 if t <= 15 else 12)


def _cut(v, edges):
    for i in range(len(edges) - 1):
        if edges[i] < v <= edges[i + 1]:
            return i
    return len(edges) - 2


def sample_play(down, togo, pos, rng):
    """Skilar (tapaður_bolti, yardar)."""
    dg = 3 if down >= 3 else down
    tb = _tb(togo)
    key = (dg, tb, 0 if pos <= 20 else 1)
    if key not in YARDS:
        key = (dg, tb, -1)
    if rng.random() < TO_P[key]:
        return True, 0
    ys = YARDS[key]
    return False, ys[int(rng.random() * len(ys))]


# ------------------------------------------------------------------ stefnur
# Stefna = fall (togo, pos, rng) -> "go" | "fg" | "punt"

def kick(pos):
    return "fg" if pos <= FG_RANGE else "punt"


def coach(togo, pos, rng):
    p_go, p_go_fg = COACH_LU[(_cut(togo, TOGO_EDGES), _cut(pos, POS_EDGES))]
    u = rng.random()
    if u < p_go:
        return "go"
    return "fg" if u < p_go_fg else "punt"


def never(togo, pos, rng):
    return kick(pos)


def always(togo, pos, rng):
    return "go"


def prob_q(q):
    def f(togo, pos, rng):
        return "go" if rng.random() < q else kick(pos)
    f.__name__ = f"q={q:.2f}"
    return f


def togo_le(x):
    def f(togo, pos, rng):
        return "go" if togo <= x else kick(pos)
    f.__name__ = f"togo<={x}"
    return f


# ------------------------------------------------------------------ sókn
def drive(pos, policy, rng, stats=None):
    """Hermir eina sókn sem byrjar á 1st & 10 á pos.
    Skilar (stig_sóknar, stig_varnar, upphafsstaða mótherja, útkoma)."""
    down, togo = 1, min(10, pos)
    while True:
        if down == 4:
            action = policy(togo, pos, rng)
            if stats is not None:
                stats[action] = stats.get(action, 0) + 1
            if action == "fg":
                if rng.random() < p_fg(pos):
                    return 3, 0, KICKOFF[int(rng.random() * len(KICKOFF))], "FG"
                return 0, 0, max(1, min(100 - (pos + 7), 80)), "FG miss"
            if action == "punt":
                b = min(95, max(30, 5 * math.ceil(pos / 5)))
                arr = PUNT[b]
                return 0, 0, arr[int(rng.random() * len(arr))], "punt"
        turnover, y = sample_play(down, togo, pos, rng)
        if turnover:
            return 0, 0, max(1, min(99, 100 - pos)), "turnover"
        pos -= y
        if pos <= 0:                                   # touchdown
            pts = 6 + (1 if rng.random() < XP else 0)
            return pts, 0, KICKOFF[int(rng.random() * len(KICKOFF))], "TD"
        if pos >= 100:                                 # safety
            return 0, 2, 60, "safety"
        if y >= togo:                                  # 1st down
            down, togo = 1, min(10, pos)
        else:
            togo -= y
            if down == 4:                              # tapað á downs
                return 0, 0, 100 - pos, "downs"
            down += 1


# ------------------------------------------------------------------ leikur
def game(policy_a, policy_b, rng, stats=None):
    """Lið A (policy_a) gegn liði B (policy_b).  Skilar (stig A, stig B)."""
    score = [0, 0]
    pols = (policy_a, policy_b)
    team = 0 if rng.random() < 0.5 else 1             # hver byrjar með boltann
    pos = KICKOFF[int(rng.random() * len(KICKOFF))]
    for _ in range(2 * DRIVES_PER_TEAM):
        st = stats if team == 0 else None
        off, deff, pos, _ = drive(pos, pols[team], rng, st)
        score[team] += off
        score[1 - team] += deff
        team = 1 - team
    return score[0], score[1]


def simulate(policy_a, n, seed=0, policy_b=coach):
    """Hermir n leiki.  Skilar dict með sigrum, jafnteflum, p̂ og öryggisbili."""
    rng = random.Random(seed)
    w = t = 0
    pf = pa = 0
    stats = {}
    for _ in range(n):
        a, b = game(policy_a, policy_b, rng, stats)
        pf += a
        pa += b
        if a > b:
            w += 1
        elif a == b:
            t += 1
    p = (w + 0.5 * t) / n
    se = math.sqrt(p * (1 - p) / n)
    n4 = sum(stats.values())
    return {
        "n": n, "wins": w, "ties": t, "p": p, "se": se,
        "ci": (p - 1.96 * se, p + 1.96 * se),
        "pts_for": pf / n, "pts_against": pa / n,
        "go_rate": stats.get("go", 0) / n4 if n4 else float("nan"),
        "fourth_downs_per_game": n4 / n,
    }
