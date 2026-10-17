"""Plots for the notebook.  Field position runs left to right like a drive:
own end zone on the left (pos 99), opponent's end zone on the right (pos 1)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

from .data import neutral

GO, FG, PUNT = "#2a78d6", "#1baf7a", "#eb6834"
GRAY, INK, INK2, GRID = "#8a8984", "#0b0b0b", "#52514e", "#e6e5e1"
ACTION_COLORS = {"go": GO, "fg": FG, "punt": PUNT}
DIVERGING = LinearSegmentedColormap.from_list("kick_go", [PUNT, "#f1f0ec", GO])

plt.rcParams.update({
    "figure.dpi": 110, "axes.axisbelow": True, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "font.size": 10.5,
})


def _field_axis(ax, label=True):
    ticks = [99, 90, 80, 70, 60, 50, 40, 30, 20, 10, 1]
    ax.set_xticks(ticks, ["own 1" if t == 99 else ("50" if t == 50 else
                          (f"own {100 - t}" if t > 50 else f"opp {t}")) for t in ticks],
                  fontsize=8.5)
    ax.set_xlim(100, 0)
    if label:
        ax.set_xlabel("field position  (← own end zone · opponent's end zone →)")


def _grid(hb, column, max_togo=15):
    g = hb[hb["togo"] <= max_togo].pivot(index="togo", columns="pos", values=column)
    return g.reindex(index=range(1, max_togo + 1), columns=range(1, 100))


def _heatmap(ax, grid, **kw):
    """Grid with togo rows and pos columns, drawn on a field axis."""
    return ax.imshow(grid.to_numpy(), origin="lower", aspect="auto",
                     extent=(0.5, 99.5, 0.5, grid.shape[0] + 0.5), **kw)


# ------------------------------------------------------------------ data
def conversion_rates(pbp, model_hb, max_togo=15):
    """P(1st down | togo): 3rd+4th down data, 4th down data (95% CI), and the model."""
    plays = pbp[neutral(pbp) & pbp["play_type"].isin(["run", "pass"]) & (pbp["down"] >= 3)
                & (pbp["ydstogo"] <= max_togo)]
    success = (plays["first_down"] == 1) | (plays["touchdown"] == 1)
    rows = []
    for label, sel in [("3rd + 4th down", plays["down"] >= 3), ("4th down only", plays["down"] == 4)]:
        g = success[sel].groupby(plays.loc[sel, "ydstogo"]).agg(["mean", "count"])
        g["ci"] = 1.96 * np.sqrt(g["mean"] * (1 - g["mean"]) / g["count"])
        rows.append((label, g))
    model = model_hb[model_hb["pos"] == 50].set_index("togo")["p_conv"].loc[1:max_togo]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for (label, g), color, off in zip(rows, [GO, FG], [-0.12, 0.12]):
        ax.errorbar(g.index + off, g["mean"], yerr=g["ci"], fmt="o", color=color, ms=4,
                    capsize=2, label=f"data: {label} (95% CI)")
    ax.plot(model.index, model.values, color=PUNT, lw=2, label="model (at midfield)")
    ax.set(xlabel="yards to go", ylabel="P(1st down)", ylim=(0, 0.85),
           title="Probability of converting, p(d)")
    ax.legend()
    table = pd.DataFrame({f"{label} rate": g["mean"] for label, g in rows}).join(
        pd.DataFrame({f"{label} n": g["count"] for label, g in rows}))
    return fig, table.loc[1:10].round(3)


def field_goal_curve(pbp, params):
    fg = pbp[(pbp["play_type"] == "field_goal") & pbp["kick_distance"].notna()]
    made = fg["field_goal_result"] == "made"
    g = made.groupby((fg["kick_distance"] // 3) * 3 + 1).agg(["mean", "count"])
    g = g[g["count"] >= 20]
    x = np.linspace(18, 68, 200)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(x, params.p_fg(x - params.fg_offset), color=GO, lw=2, label="logistic fit")
    ax.scatter(g.index, g["mean"], s=np.clip(g["count"] / 6, 10, 150), color=PUNT, zorder=3,
               label="data (3-yard bins, size ~ attempts)")
    ax.set(xlabel="kick distance (yards)", ylabel="P(field goal good)", ylim=(0, 1.02),
           title="Field goal success, p_FG(x)")
    ax.legend(loc="lower left")
    return fig


def punt_distance(params):
    fig, ax = plt.subplots(figsize=(8, 4))
    for pos, color in [(80, GO), (60, FG), (45, PUNT)]:
        opp, probs = params.punt_next_pos[pos]
        net = pos - (100 - opp)
        order = np.argsort(net)
        ax.plot(net[order], np.cumsum(probs[order]), color=color, lw=2,
                label=f"punt from pos {pos} (mean net {np.sum(net * probs):.0f} yd)")
    ax.set(xlabel="net punt yards", ylabel="cumulative probability", title="Punts: net yards")
    ax.legend()
    return fig


def coach_go_rates(decisions):
    d = decisions.copy()
    d["tb"] = np.minimum(d["togo"], 10)
    d["pb"] = (d["pos"] - 1) // 10
    d["go"] = (d["action"] == "go").astype(float)
    g = d.pivot_table(index="tb", columns="pb", values="go", aggfunc="mean")
    n = d.pivot_table(index="tb", columns="pb", values="go", aggfunc="size")
    g = g.where(n >= 15)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4), gridspec_kw={"width_ratios": [2.2, 1]},
                             layout="constrained")
    ax = axes[0]
    im = ax.imshow(g.to_numpy(), origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=1)
    for i, j in zip(*np.where(~np.isnan(g.to_numpy()))):
        v = g.to_numpy()[i, j]
        ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=8,
                color="white" if v > 0.55 else INK)
    ax.set_xticks(range(g.shape[1]), [f"{10 * c + 1}-{10 * c + 10}" for c in g.columns], fontsize=8)
    ax.set_yticks(range(g.shape[0]), [str(t) if t < 10 else "10+" for t in g.index])
    ax.set(xlabel="yards to the end zone (pos)", ylabel="yards to go",
           title="Coaches: % of 4th downs where they go for it")
    ax.grid(False)
    ax.invert_xaxis()
    fig.colorbar(im, ax=ax, fraction=0.03)
    by_season = d.groupby("season")["action"].apply(lambda a: (a == "go").mean())
    axes[1].bar(by_season.index, 100 * by_season.values, color=GO)
    axes[1].set(title="Go rate by season", ylabel="% of 4th downs")
    axes[1].grid(axis="x", visible=False)
    return fig


# ------------------------------------------------------------------ Markov chain
def ep_curve(s, boot):
    pos = np.arange(1, 100)
    ep = s.ep[s.chain.index(1, np.minimum(10, pos), pos)]
    n = neutral(s.pbp)
    real = (s.pbp[n & (s.pbp["down"] == 1) & (s.pbp["ydstogo"] == np.minimum(10, s.pbp["yardline_100"]))]
            .groupby("yardline_100")["ep"].mean())
    lo, hi = np.percentile(boot["ep_first"], [2.5, 97.5], axis=0)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.fill_between(pos, lo, hi, color=GO, alpha=0.25, lw=0, label="95% bootstrap band")
    ax.plot(pos, ep, color=GO, lw=2, label="our Markov chain")
    ax.plot(real.index, real.values, color=PUNT, lw=1.2, ls="--", label="nflverse EP model")
    ax.axhline(0, color=GRAY, lw=0.8)
    _field_axis(ax)
    ax.set(ylabel="expected points", title="Expected points of 1st & 10")
    ax.legend()
    return fig


def mc_convergence(samples, exact, title):
    n = np.arange(1, len(samples) + 1)
    mean = np.cumsum(samples) / n
    sd = np.std(samples)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.fill_between(n, exact - 1.96 * sd / np.sqrt(n), exact + 1.96 * sd / np.sqrt(n),
                    color=GO, alpha=0.18, lw=0, label="exact ± 1.96·s/√n")
    ax.plot(n, mean, color=PUNT, lw=1.3, label="Monte Carlo running mean")
    ax.axhline(exact, color=GO, lw=1.5, label="exact (linear system)")
    ax.set_xscale("log")
    ax.set(xlabel="number of simulated drives-until-score", ylabel="expected points",
           title=title, ylim=(exact - 2, exact + 2))
    ax.legend()
    return fig


# ------------------------------------------------------------------ handbook
def handbook_map(hb, max_togo=15, title="The handbook: best call on 4th down"):
    codes = hb["best"].map({"go": 0, "fg": 1, "punt": 2}).to_frame().assign(
        togo=hb["togo"], pos=hb["pos"])
    grid = _grid(codes, "best", max_togo)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    _heatmap(ax, grid, cmap=ListedColormap([GO, FG, PUNT]), vmin=0, vmax=2,
             interpolation="nearest")
    _field_axis(ax)
    ax.set_yticks(range(1, max_togo + 1))
    ax.set(ylabel="yards to go", title=title)
    ax.grid(False)
    ax.legend(handles=[Patch(color=c, label=a) for a, c in
                       [("go for it", GO), ("field goal", FG), ("punt", PUNT)]],
              loc="upper left", ncol=3, bbox_to_anchor=(0, -0.13))
    return fig


def value_map(hb, column="go_minus_kick", max_togo=15, title=None, label=None, lim=2.0):
    grid = _grid(hb, column, max_togo)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    im = _heatmap(ax, grid, cmap=DIVERGING, vmin=-lim, vmax=lim, interpolation="nearest")
    _field_axis(ax)
    ax.set_yticks(range(1, max_togo + 1))
    ax.set(ylabel="yards to go", title=title or column)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03, label=label or column)
    return fig


def probability_map(hb, column, max_togo=15, title=None, label=None):
    grid = _grid(hb, column, max_togo)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    im = _heatmap(ax, grid, cmap=DIVERGING, vmin=0, vmax=1, interpolation="nearest")
    cs = ax.contour(np.arange(1, 100), np.arange(1, max_togo + 1), grid.to_numpy(),
                    levels=[0.5, 0.9], colors=INK, linewidths=[1.2, 0.8], linestyles=["-", "--"])
    ax.clabel(cs, fmt={0.5: "50%", 0.9: "90%"}, fontsize=8)
    _field_axis(ax)
    ax.set_yticks(range(1, max_togo + 1))
    ax.set(ylabel="yards to go", title=title or column)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03, label=label or column)
    return fig


def breakeven_lines(hb, positions=(80, 50, 30), max_togo=10):
    """Break-even conversion probability vs the actual one, for a few field positions."""
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    colors = [PUNT, FG, GO]
    for pos, c in zip(positions, colors):
        h = hb[(hb["pos"] == pos) & (hb["togo"] <= max_togo)]
        name = f"own {100 - pos}" if pos > 50 else ("midfield" if pos == 50 else f"opp {pos}")
        ax.plot(h["togo"], h["breakeven"], color=c, lw=1.8, ls="--")
        ax.text(max_togo + 0.15, h["breakeven"].iloc[-1], name, color=c, fontsize=9, va="center")
    p = hb[(hb["pos"] == 50) & (hb["togo"] <= max_togo)]
    ax.plot(p["togo"], p["p_conv"], color=INK, lw=2.5, label="actual P(convert)")
    ax.plot([], [], color=GRAY, ls="--", label="break-even P(convert), by field position")
    ax.set(xlabel="yards to go", ylabel="probability", ylim=(0, 1), xlim=(0.8, max_togo + 1.2),
           title="Go for it when the black line is above the dashed one")
    ax.legend(loc="upper right")
    return fig


# ------------------------------------------------------------------ coaches vs handbook
def ep_lost_by_season(scored, pbp):
    games = pbp.dropna(subset=["posteam"]).drop_duplicates(["game_id", "posteam"]).groupby("season").size()
    lost = scored.groupby("season")["ep_lost"].sum() / games
    agree = scored.groupby("season")["agrees"].mean()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(lost.index, lost.values, color=PUNT)
    for x, v, a in zip(lost.index, lost.values, agree.values):
        ax.text(x, v + 0.01, f"{v:.2f}\n({100 * a:.0f}% agree)", ha="center", fontsize=8.5)
    ax.set(ylabel="EP lost per team-game", title="Expected points coaches leave on the table",
           ylim=(0, lost.max() * 1.3))
    ax.grid(axis="x", visible=False)
    return fig


def playoff_go_rates(scored_reg, scored_post):
    """Go rate in the regular season vs the playoffs, by what the handbook says."""
    def group(d):
        return np.select([(d["best"] == "go") & (d["gain"] >= 0.5), d["best"] == "go"],
                         ["handbook: clearly go\n(gain ≥ 0.5 EP)", "handbook: go, but close\n(gain < 0.5 EP)"],
                         "handbook: kick")
    order = ["handbook: clearly go\n(gain ≥ 0.5 EP)", "handbook: go, but close\n(gain < 0.5 EP)",
             "handbook: kick"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for k, (name, d, color) in enumerate([("regular season", scored_reg, GO),
                                          ("playoffs", scored_post, PUNT)]):
        g = (d["action"] == "go").groupby(group(d)).agg(["mean", "count"]).reindex(order)
        ci = 1.96 * np.sqrt(g["mean"] * (1 - g["mean"]) / g["count"])
        x = np.arange(len(order)) + (k - 0.5) * 0.36
        ax.bar(x, 100 * g["mean"], 0.34, yerr=100 * ci, capsize=3, color=color, label=name)
        for xi, m, c, n in zip(x, g["mean"], ci, g["count"]):
            ax.text(xi, 100 * (m + c) + 2, f"{100 * m:.0f}%\nn={n}", ha="center", fontsize=8)
    ax.set_xticks(range(len(order)), order)
    ax.set(ylabel="% of 4th downs where coaches went for it", ylim=(0, 100),
           title="Do coaches get more cautious in the playoffs?")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    return fig


def rules_bar(rules):
    r = rules.sort_values("ep_per_game_vs_coaches")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    colors = [GO if v > 0 else (GRAY if v == 0 else PUNT) for v in r["ep_per_game_vs_coaches"]]
    ax.barh(r["rule"], r["ep_per_game_vs_coaches"], color=colors)
    for y, (v, g) in enumerate(zip(r["ep_per_game_vs_coaches"], r["go_rate"])):
        ax.text(v + (0.03 if v >= 0 else -0.03), y, f"{v:+.2f}  ({100 * g:.0f}% go)",
                va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.axvline(0, color=INK2, lw=1)
    lim = r["ep_per_game_vs_coaches"].abs().max() * 1.45
    ax.set(xlim=(-lim, lim), xlabel="expected points per game vs what coaches did",
           title="Simple rules vs the coaches")
    ax.grid(axis="y", visible=False)
    return fig
