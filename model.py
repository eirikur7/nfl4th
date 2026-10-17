"""
model.py
Líkurnar sem 02_estimate.py mat (model.pkl), á formi sem hermirinn notar.
Hvert fall hér dregur eina slembiútkomu úr metinni dreifingu.

  run_play(down, togo, pos, rng)  -> (tapaður_bolti, yardar)
  field_goal_prob(pos)            -> pFG fyrir spark frá pos
  punt_next_pos(pos, rng)         -> vallarstaða mótherja eftir punt
  kickoff_start_pos(rng)          -> vallarstaða í byrjun sóknar eftir kickoff
  coach_decision(togo, pos, rng)  -> "go" / "fg" / "punt" eins og þjálfarar velja
"""
import math
import pickle

from common import (COACH_POS_EDGES, COACH_TOGO_EDGES, MODEL_FILE, down_group,
                    field_zone, interval_index, togo_bucket)

with open(MODEL_FILE, "rb") as f:
    ESTIMATES = pickle.load(f)

# numpy fylki -> python listar, því það er hraðara að draga eitt stak í einu úr lista
_PLAY_YARDS = {k: v.tolist() for k, v in ESTIMATES["play_yards"].items()}
_TURNOVER_PROB = dict(ESTIMATES["play_turnover_prob"])
_PUNT_NEXT_POS = {k: v.tolist() for k, v in ESTIMATES["punt_next_pos"].items()}
_KICKOFF_START = ESTIMATES["kickoff_start_pos"].tolist()
_FG_B0, _FG_B1 = ESTIMATES["fg_logit"]
_FG_KICK_OFFSET = ESTIMATES["fg_kick_offset"]
EXTRA_POINT_PROB = ESTIMATES["extra_point_prob"]


def _draw(values, rng):
    """Dregur eitt stak af handahófi úr lista (raundreifing gagnanna)."""
    return values[int(rng.random() * len(values))]


def _play_cell(down, togo, pos):
    """(P(tapaður bolti), yarda-dreifing) fyrir hólfið sem ástandið lendir í."""
    key = (down_group(down), togo_bucket(togo), field_zone(pos))
    if key not in _PLAY_YARDS:                 # of fá gögn: öll svæði saman
        key = (key[0], key[1], -1)
    return _TURNOVER_PROB[key], _PLAY_YARDS[key]


# Uppflettitafla fyrir öll möguleg ástönd, reiknuð einu sinni (hermirinn er mun hraðari svona)
_PLAY_TABLE = {(down, togo, pos): _play_cell(down, togo, pos)
               for down in range(1, 5) for togo in range(1, 100) for pos in range(1, 100)}


def run_play(down, togo, pos, rng):
    """Eitt run/pass kerfi.  Skilar (tapaður_bolti, yardar sem náðust)."""
    turnover_prob, yards = _PLAY_TABLE[(down, togo, pos)]
    if rng.random() < turnover_prob:
        return True, 0
    return False, _draw(yards, rng)


def field_goal_prob(pos):
    """pFG(x) þar sem x = pos + 18 er lengd sparksins."""
    x = pos + _FG_KICK_OFFSET
    return 1.0 / (1.0 + math.exp(-(_FG_B0 + _FG_B1 * x)))


def punt_next_pos(pos, rng):
    """Vallarstaða mótherja eftir punt frá pos (dregið úr puntum frá svipuðum stað)."""
    bucket = min(95, max(30, 5 * math.ceil(pos / 5)))
    return _draw(_PUNT_NEXT_POS[bucket], rng)


def kickoff_start_pos(rng):
    return _draw(_KICKOFF_START, rng)


# Þjálfaratafla: (togo-hólf, pos-hólf) -> (P(go), P(go) + P(fg)).  P(punt) er afgangurinn.
_COACH_CELLS = {}
for (togo_cell, pos_cell), row in ESTIMATES["coach_probs"].iterrows():
    # Lengra en ~63 yarda spark (pos > 45) er ekki reynt; jöfnunin gaf þar smá FG-líkur
    p_fg = row["fg"] if COACH_POS_EDGES[int(pos_cell)] < 45 else 0.0
    _COACH_CELLS[(int(togo_cell), int(pos_cell))] = (row["go"], row["go"] + p_fg)



def _coach_cell(togo, pos):
    togo_cell = interval_index(togo, COACH_TOGO_EDGES)
    pos_cell = interval_index(pos, COACH_POS_EDGES)
    # Sum hólf koma aldrei fyrir í gögnunum (t.d. 4th & 1 á eigin 5 yarda línu):
    # þá er notað næsta vallarstöðuhólf með sama togo-hólf.
    nearest = min((pc for tc, pc in _COACH_CELLS if tc == togo_cell),
                  key=lambda pc: abs(pc - pos_cell))
    return _COACH_CELLS[(togo_cell, nearest)]


# Sama tafla, en flett beint upp eftir (togo, pos)
_COACH_TABLE = {(togo, pos): _coach_cell(togo, pos)
                for togo in range(1, 100) for pos in range(1, 100)}


def coach_decision(togo, pos, rng):
    """Dregur ákvörðun á 4th down eins og þjálfarar taka hana (1.-3. leikhluti)."""
    p_go, p_go_or_fg = _COACH_TABLE[(togo, pos)]
    u = rng.random()
    if u < p_go:
        return "go"
    return "fg" if u < p_go_or_fg else "punt"
