"""Monte Carlo simulation of the same Markov chain, play by play.

Used to check the exact solution in markov.py: by the law of large numbers the average of
simulated next scores converges to EP, and by the CLT the error is about 1.96 * s / sqrt(n).
"""
import bisect
import math
import random

import numpy as np

from .markov import ACTIONS


class Simulator:
    def __init__(self, chain, policy, seed=0):
        self.chain, self.P = chain, chain.params
        self.rng = random.Random(seed)
        self.policy_cdf = np.cumsum(policy, axis=1)
        self._cdf = {}

    def _draw(self, key, values, probs):
        """Draw from a discrete distribution by inverting its CDF."""
        if key not in self._cdf:
            self._cdf[key] = (list(values), list(np.cumsum(probs)))
        vals, cdf = self._cdf[key]
        return vals[min(bisect.bisect(cdf, self.rng.random()), len(vals) - 1)]

    def next_score(self, down, togo, pos):
        """Simulate until someone scores.  Returns points from the view of the starting team."""
        P, rng, sign = self.P, self.rng, 1
        while True:
            if down == 4:
                k = np.searchsorted(self.chain.fourth, self.chain.index(4, togo, pos))
                action = ACTIONS[min(bisect.bisect(self.policy_cdf[k], rng.random()), 2)]
                if action == "fg":
                    if rng.random() < P.p_fg(pos):
                        return sign * 3
                    down, togo, pos, sign = 1, *_first(min(80, max(1, 100 - (pos + 7)))), -sign
                    continue
                if action == "punt":
                    bucket = min(95, max(30, 5 * math.ceil(pos / 5)))
                    opp = self._draw(("punt", bucket), *P.punt_next_pos[bucket])
                    down, togo, pos, sign = 1, *_first(opp), -sign
                    continue
            cell = P.play_cell(down, togo, pos)
            if rng.random() < P.turnover[cell]:
                down, togo, pos, sign = 1, *_first(100 - pos), -sign
                continue
            y = self._draw(cell, *P.play_yards[cell])
            new = pos - y
            if new <= 0:
                return sign * (6 + (rng.random() < P.xp))
            if new >= 100:
                return -sign * 2
            if y >= togo:
                down, togo, pos = 1, *_first(new)
            elif down < 4:
                down, togo, pos = down + 1, min(togo - y, 30), new
            else:
                down, togo, pos, sign = 1, *_first(100 - new), -sign


def _first(pos):
    return min(10, pos), pos
