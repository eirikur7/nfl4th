"""The game as an absorbing Markov chain, solved exactly.

State: (down, togo, pos) for the team with the ball.  The chain runs until the NEXT SCORE,
which is the absorbing event: +7 / +3 / +2 for the team with the ball, or the same against it.

Expected points EP(s) = E[value of the next score | state s], seen from the team with the ball.
When possession changes the opponent faces state s', and by symmetry (equal teams) that is worth
-EP(s') to us.  So by the law of total expectation

    EP(s) = sum over outcomes o of P(o | s) * value(o)
          = r(s) + sum_s' A(s, s') EP(s')

where r(s) is the expected points scored directly from s, and A(s, s') is +P(s -> s') if we keep
the ball and -P(s -> s') if the opponent gets it.  That is one linear system (I - A) EP = r.

On 4th down the transition depends on the action (go / fg / punt).  For each 4th-down state we
compute the action value Q(a) = E[EP after taking a], and the best action is the largest Q.
"""
import math

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .estimate import COACH_POS_EDGES, COACH_TOGO_EDGES, PUNT_BUCKETS

TOGO_MAX = 30          # togo above 30 is treated as 30 (very rare)
FG_MAX_POS = 45        # longest field goal considered: 45 + 18 = 63 yards
PUNT_MIN_POS = 30      # no punts from inside the opponent's 30 (no data, never sensible)
ACTIONS = ["go", "fg", "punt"]


def _first_down(pos):
    """Opponent/own 1st & 10 (or 1st & goal) at pos."""
    return np.ones_like(pos), np.minimum(10, pos), pos


class Chain:
    def __init__(self, params):
        self.params = params
        # all states (down, togo, pos) with togo <= pos
        d, t, p = [], [], []
        for down in range(1, 5):
            for pos in range(1, 100):
                for togo in range(1, min(pos, TOGO_MAX) + 1):
                    d.append(down), t.append(togo), p.append(pos)
        self.down, self.togo, self.pos = map(np.array, (d, t, p))
        self.n = len(self.down)
        self._index = np.full((5, TOGO_MAX + 1, 100), -1)
        self._index[self.down, self.togo, self.pos] = np.arange(self.n)
        self.fourth = np.flatnonzero(self.down == 4)          # the 4th-down states
        self._build()

    def index(self, down, togo, pos):
        return self._index[down, np.minimum(togo, TOGO_MAX), pos]

    # ------------------------------------------------------------ transitions
    def _plays(self, states):
        """Run/pass play from each state in `states`.

        Returns (rows, cols, vals, reward) as arrays of signed transitions, plus per-state
        P(conversion) and the conversion/failure split used for break-even probabilities.
        """
        P = self.params
        rows, cols, vals = [], [], []
        reward = np.zeros(len(states))
        conv = {"rows": [], "cols": [], "vals": [], "reward": np.zeros(len(states))}
        fail = {"rows": [], "cols": [], "vals": [], "reward": np.zeros(len(states))}
        p_conv = np.zeros(len(states))
        local = np.arange(len(states))
        d, t, p = self.down[states], self.togo[states], self.pos[states]
        cells = [P.play_cell(*x) for x in zip(d, t, p)]
        for cell in set(cells):
            m = np.array([c == cell for c in cells])
            ys, probs = P.play_yards[cell]
            q = P.turnover[cell]
            i = np.repeat(local[m], len(ys))
            y = np.tile(ys, m.sum())
            pr = np.tile(probs, m.sum()) * (1 - q)
            dd, tt, pp = d[i], t[i], p[i]
            new = pp - y

            td = new <= 0
            safety = new >= 100
            first = ~td & ~safety & (y >= tt)
            short = ~td & ~safety & (y < tt)
            keep = short & (dd < 4)
            downs = short & (dd == 4)

            np.add.at(reward, i[td], pr[td] * (6 + P.xp))
            np.add.at(reward, i[safety], -2 * pr[safety])
            np.add.at(conv["reward"], i[td], pr[td] * (6 + P.xp))
            np.add.at(p_conv, i[td | first], pr[td | first])

            # 1st down: same team, (1, min(10, new), new)
            c = self.index(*_first_down(new[first]))
            rows.append(i[first]), cols.append(c), vals.append(pr[first])
            conv["rows"].append(i[first]), conv["cols"].append(c), conv["vals"].append(pr[first])
            # next down: same team
            c = self.index(dd[keep] + 1, tt[keep] - y[keep], new[keep])
            rows.append(i[keep]), cols.append(c), vals.append(pr[keep])
            # turnover on downs: opponent's 1st down at 100 - new, value negated
            c = self.index(*_first_down(100 - new[downs]))
            rows.append(i[downs]), cols.append(c), vals.append(-pr[downs])
            fail["rows"].append(i[downs]), fail["cols"].append(c), fail["vals"].append(-pr[downs])

            # interception / lost fumble at the line of scrimmage
            j = local[m]
            c = self.index(*_first_down(100 - p[j]))
            rows.append(j), cols.append(c), vals.append(-np.full(len(j), q))
            fail["rows"].append(j), fail["cols"].append(c), fail["vals"].append(-np.full(len(j), q))

        def mat(r, c, v):
            return sparse.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))),
                                     shape=(len(states), self.n))
        return (mat(rows, cols, vals), reward, p_conv,
                (mat(conv["rows"], conv["cols"], conv["vals"]), conv["reward"]),
                (mat(fail["rows"], fail["cols"], fail["vals"]), fail["reward"]))

    def _field_goal(self, states):
        P = self.params
        p = self.pos[states]
        make = np.where(p <= FG_MAX_POS, P.p_fg(p), 0.0)
        # miss: opponent gets the ball at the kick spot (pos + 7), or at their 20
        opp = np.clip(100 - (p + 7), 1, 80)
        A = sparse.csr_matrix((-(1 - make), (np.arange(len(states)), self.index(*_first_down(opp)))),
                              shape=(len(states), self.n))
        return A, 3 * make, make

    def _punt(self, states):
        P = self.params
        rows, cols, vals = [], [], []
        for k, pos in enumerate(self.pos[states]):
            bucket = min(95, max(PUNT_BUCKETS[0], 5 * math.ceil(pos / 5)))
            opp, probs = P.punt_next_pos[bucket]
            rows.append(np.full(len(opp), k)), cols.append(self.index(*_first_down(opp)))
            vals.append(-probs)
        A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(len(states), self.n))
        return A, np.zeros(len(states))

    def _build(self):
        early = np.flatnonzero(self.down < 4)
        A_early, r_early, *_ = self._plays(early)
        self._early = (early, A_early, r_early)

        f = self.fourth
        A_go, r_go, self.p_conv, self._conv, self._fail = self._plays(f)
        A_fg, r_fg, self.p_fg = self._field_goal(f)
        A_punt, r_punt = self._punt(f)
        self._actions = {"go": (A_go, r_go), "fg": (A_fg, r_fg), "punt": (A_punt, r_punt)}
        self.allowed = np.column_stack([np.ones(len(f), bool),
                                        self.pos[f] <= FG_MAX_POS,
                                        self.pos[f] >= PUNT_MIN_POS])

    # ------------------------------------------------------------ solving
    def solve(self, policy):
        """EP for every state when 4th downs are played with `policy`
        (array n_fourth x 3 of probabilities for go / fg / punt)."""
        early, A_early, r_early = self._early
        A4 = sum(sparse.diags(policy[:, k]) @ self._actions[a][0] for k, a in enumerate(ACTIONS))
        r4 = sum(policy[:, k] * self._actions[a][1] for k, a in enumerate(ACTIONS))
        order = np.concatenate([early, self.fourth])
        A = sparse.vstack([A_early, A4]).tocsr()
        r = np.concatenate([r_early, r4])
        # rows are in `order`; put them back in state order
        perm = sparse.csr_matrix((np.ones(self.n), (order, np.arange(self.n))), shape=(self.n,) * 2)
        A, r = perm @ A, perm @ r
        return spsolve((sparse.identity(self.n) - A).tocsc(), r)

    def q_values(self, ep):
        """Value of each 4th-down action given EP for all states (NaN if not allowed)."""
        q = np.column_stack([self._actions[a][1] + self._actions[a][0] @ ep for a in ACTIONS])
        return np.where(self.allowed, q, np.nan)

    # ------------------------------------------------------------ policies
    def coach_policy(self):
        """What coaches actually do, from the estimated table.  Actions that are not allowed
        here (very long FG, punt inside the 30) get their probability moved to an allowed one."""
        f = self.fourth
        tc = np.searchsorted(COACH_TOGO_EDGES, self.togo[f], side="left") - 1
        pc = np.searchsorted(COACH_POS_EDGES, self.pos[f], side="left") - 1
        table = self.params.coach
        probs = np.zeros((len(f), 3))
        for k, (a, b) in enumerate(zip(tc, pc)):
            if (a, b) in table.index:
                probs[k] = table.loc[(a, b)].to_numpy()
            else:                     # cell never seen in the data: nearest pos cell
                cells = table.loc[a]
                probs[k] = cells.iloc[np.argmin(np.abs(cells.index - b))].to_numpy()
        probs[~self.allowed[:, 1], 2] += probs[~self.allowed[:, 1], 1]
        probs[~self.allowed[:, 1], 1] = 0
        no_punt = ~self.allowed[:, 2]
        moved = probs[no_punt, 2]
        probs[no_punt, 2] = 0
        probs[no_punt, 1] += np.where(self.allowed[no_punt, 1], moved, 0)
        probs[no_punt, 0] += np.where(self.allowed[no_punt, 1], 0, moved)
        return probs / probs.sum(axis=1, keepdims=True)

    def greedy_policy(self, q):
        best = np.nanargmax(q, axis=1)
        return np.eye(3)[best]

    def optimal(self, max_iter=20):
        """Policy iteration: both teams play the best 4th-down action everywhere."""
        policy = self.coach_policy()
        for _ in range(max_iter):
            ep = self.solve(policy)
            new = self.greedy_policy(self.q_values(ep))
            if np.array_equal(new, policy):
                break
            policy = new
        return policy, ep

    # ------------------------------------------------------------ the handbook
    def handbook(self, ep):
        """One row per 4th-down state: action values, best action and success probabilities.

        q_*          expected points after each action (given everyone plays like ep's policy)
        best         action with the highest expected points
        gain         expected points of the best action minus the second best
        p_conv       P(conversion) if going for it (incl. touchdowns)
        p_fg         P(field goal good)
        breakeven    P(conversion) at which going for it equals the best kick:
                     Q_go = p * E[EP | convert] + (1 - p) * E[EP | fail]  =  Q_kick
        """
        q = self.q_values(ep)
        f = self.fourth
        hb = pd.DataFrame({"togo": self.togo[f], "pos": self.pos[f],
                           "q_go": q[:, 0], "q_fg": q[:, 1], "q_punt": q[:, 2]})
        hb["best"] = np.array(ACTIONS)[np.nanargmax(q, axis=1)]
        s = np.sort(np.nan_to_num(q, nan=-np.inf), axis=1)
        hb["gain"] = s[:, -1] - s[:, -2]
        hb["p_conv"] = self.p_conv
        hb["p_fg"] = np.where(self.allowed[:, 1], self.p_fg, np.nan)

        with np.errstate(invalid="ignore", divide="ignore"):    # P(convert) = 0 in a few extreme states
            ev_conv = (self._conv[1] + self._conv[0] @ ep) / self.p_conv
            ev_fail = (self._fail[1] + self._fail[0] @ ep) / (1 - self.p_conv)
            kick = np.nanmax(q[:, 1:], axis=1)
            hb["breakeven"] = np.clip((kick - ev_fail) / (ev_conv - ev_fail), 0, 1)
        hb["go_minus_kick"] = q[:, 0] - kick
        return hb
