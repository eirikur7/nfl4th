# 4th down: go for it, kick, or punt?

Applied Probability (T-811, Reykjavik University) group project.  For every 4th-down situation
(yards to go × field position) we find the call with the highest **expected points**, and the
**probabilities** behind it, from NFL play-by-play data 2018–2025 (nflverse).

**Open [`analysis.ipynb`](analysis.ipynb)**: all plots and tables, with short explanations.

## Method

1. **Probabilities from data:** yards gained per play, P(1st down | yards to go), P(FG good | distance)
   (logistic regression), punt distances, and what coaches actually do on 4th down.
2. **Markov chain:** state = (down, yards to go, field position). The chain runs until the next score
   (absorbing states).  Expected points for every state come from one linear system, (I − A)·EP = r.
3. **Handbook:** on 4th down, compare EP after going for it, a field goal and a punt.  Also the
   **break-even conversion probability**: go for it when P(convert) is above it.
4. **Monte Carlo:** play-by-play simulation of the chain to check the exact solution (LLN/CLT), and a
   bootstrap over the data (200 re-estimations) for P(going for it is the best call).
5. **Coaches vs the handbook:** every real 4th-down decision valued with the model, and simple rules
   ("go for it if togo ≤ X") compared in expected points per game.
6. **Side note, playoffs vs regular season:** the same handbook applied to playoff 4th downs, to test
   whether coaches are more cautious in the playoffs (situation-adjusted go rate, z-tests).

## Setup

```bash
conda create -n nfl4th python numpy pandas scipy matplotlib notebook    # or: pip install -r requirements.txt
conda activate nfl4th
jupyter notebook analysis.ipynb
```

The first run downloads the data (~150 MB) into `data/` and caches it.  The bootstrap result is saved in
`cache/bootstrap.npz` (committed), so the notebook runs in about a minute.  Set `RECOMPUTE = True` at the
top of the notebook to redo it (~6 min).

## Code (`fourth_down/`)

| File | Purpose |
|---|---|
| `data.py` | Download and load the play-by-play data; the real 4th-down decisions |
| `estimate.py` | Estimate all probabilities (optionally from a bootstrap resample) |
| `markov.py` | The Markov chain: build transitions, solve for EP, action values, handbook, policy iteration |
| `simulate.py` | Monte Carlo simulation of the same chain, to check the exact solution |
| `analysis.py` | Bootstrap, scoring real decisions, simple rules, situations table |
| `plots.py` | All plots used in the notebook |

## Assumptions

- No clock or score: EP is the right measure in early-game situations, so the data is quarters 1–3
  (not the last 2 minutes of the half).
- 3rd and 4th down share one play distribution (4th-down attempts are rare).
- The handbook values each 4th down assuming the rest of the game is played like today's coaches;
  the notebook also shows the version where both teams play the handbook (policy iteration).
- Field goals up to 63 yards; no punts from inside the opponent's 30.
