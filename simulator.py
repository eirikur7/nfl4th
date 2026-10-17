"""
simulator.py
Markov-keðju hermir fyrir NFL sóknir og leiki.

Sókn: runa ástanda (down, togo, pos) sem byrjar á 1st & 10 og endar í einu
gleypiástandi: TD, FG, misheppnað FG, punt, tapaður bolti, tapað á downs, safety.

Leikur: liðin skiptast á, 12 sóknir hvort.  Engin klukka, jöfn lið, engar
2-point tilraunir, engin framlenging (jafntefli telst hálfur sigur).
"""
import math
import random
from collections import Counter
from dataclasses import dataclass

import model
import strategies
from common import DRIVES_PER_TEAM


@dataclass(frozen=True)
class Situation:
    """Það sem stefna sér þegar hún tekur ákvörðun á 4th down."""
    togo: int
    pos: int
    score_diff: int      # eigin stig mínus stig mótherja
    drives_left: int     # sóknir eftir, að þessari meðtalinni


@dataclass
class DriveResult:
    points: int          # stig sóknarliðsins
    opp_points: int      # stig varnarliðsins (aðeins safety)
    next_pos: int        # hvar mótherjinn byrjar næstu sókn
    outcome: str         # "TD", "FG", "FG miss", "punt", "turnover", "downs", "safety"


def play_drive(pos, strategy, rng, score_diff=0, drives_left=DRIVES_PER_TEAM, decisions=None):
    """Hermir eina sókn sem byrjar á 1st & 10 á pos.

    decisions: ef Counter er gefinn er hver 4th down ákvörðun talin í hann.
    """
    down, togo = 1, min(10, pos)
    while True:
        if down == 4:
            action = strategy(Situation(togo, pos, score_diff, drives_left), rng)
            if decisions is not None:
                decisions[action] += 1
            if action == "fg":
                if rng.random() < model.field_goal_prob(pos):
                    return DriveResult(3, 0, model.kickoff_start_pos(rng), "FG")
                # misheppnað FG: mótherjinn fær boltann þar sem sparkað var (pos + 7)
                return DriveResult(0, 0, max(1, min(100 - (pos + 7), 80)), "FG miss")
            if action == "punt":
                return DriveResult(0, 0, model.punt_next_pos(pos, rng), "punt")
            # annars "go": spilað eins og venjulegt kerfi hér fyrir neðan

        turnover, yards = model.run_play(down, togo, pos, rng)
        if turnover:
            return DriveResult(0, 0, max(1, min(99, 100 - pos)), "turnover")

        pos -= yards
        if pos <= 0:
            points = 6 + (rng.random() < model.EXTRA_POINT_PROB)
            return DriveResult(points, 0, model.kickoff_start_pos(rng), "TD")
        if pos >= 100:
            return DriveResult(0, 2, 60, "safety")

        if yards >= togo:                  # 1st down
            down, togo = 1, min(10, pos)
        elif down == 4:                    # tapað á downs
            return DriveResult(0, 0, 100 - pos, "downs")
        else:
            down, togo = down + 1, togo - yards


def play_game(strategy_a, strategy_b, rng, decisions_a=None):
    """Lið A gegn liði B.  Skilar (stig A, stig B)."""
    score = [0, 0]
    drives_left = [DRIVES_PER_TEAM, DRIVES_PER_TEAM]
    strategy = (strategy_a, strategy_b)
    team = rng.randrange(2)                # hvort liðið byrjar með boltann
    pos = model.kickoff_start_pos(rng)
    for _ in range(2 * DRIVES_PER_TEAM):
        other = 1 - team
        result = play_drive(pos, strategy[team], rng,
                            score_diff=score[team] - score[other],
                            drives_left=drives_left[team],
                            decisions=decisions_a if team == 0 else None)
        score[team] += result.points
        score[other] += result.opp_points
        drives_left[team] -= 1
        pos = result.next_pos
        team = other
    return score[0], score[1]


def simulate(strategy, n_games, seed, opponent=strategies.coach):
    """Hermir n_games leiki af stefnu gegn mótherja sem fylgir stefnu þjálfara.

    Sigurlíkur  p̂ = (W + ½T) / N,  95% öryggisbil  p̂ ± 1.96 · sqrt(p̂(1-p̂)/N).
    """
    rng = random.Random(seed)
    wins = ties = points_for = points_against = 0
    decisions = Counter()
    for _ in range(n_games):
        a, b = play_game(strategy, opponent, rng, decisions)
        points_for += a
        points_against += b
        wins += a > b
        ties += a == b

    p = (wins + 0.5 * ties) / n_games
    se = math.sqrt(p * (1 - p) / n_games)
    n_fourth = sum(decisions.values())
    return {
        "n": n_games, "wins": wins, "ties": ties, "p": p, "se": se,
        "ci_low": p - 1.96 * se, "ci_high": p + 1.96 * se,
        "points_for": points_for / n_games, "points_against": points_against / n_games,
        "go_rate": decisions["go"] / n_fourth if n_fourth else math.nan,
        "fourth_downs_per_game": n_fourth / n_games,
    }
