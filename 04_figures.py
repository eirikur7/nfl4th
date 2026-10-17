"""
04_figures.py
Öll gröf, vistuð í figures/.  Les model.pkl og results/ (keyra 03_experiments.py fyrst).

Niðurstöður:
  1_samanburdur_stefna.png     allar 11 stefnurnar, munur á sigurlíkum miðað við þjálfara
  2_sigurlikur_arasargirni.png sigurlíkur sem fall af því hversu oft er reynt
  3_hitakort_X_Y.png           stefna 6: sigurlíkur fyrir hverja samsetningu X og Y
  4_undir_seint.png            stefna 9: áhrif þess hversu snemma er byrjað að reyna
Líkanið og staðfesting:
  5_p_d.png, 6_pFG.png, 7_thjalfarar.png, 8_throun_eftir_arum.png, 9_utkomur_sokna.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import strategies as st
from common import (COACH_POS_LABELS, COACH_TOGO_LABELS, FG_MAX_POS, FIGURES_DIR,
                    RESULTS_DIR)
from model import ESTIMATES

MAIN = pd.read_csv(RESULTS_DIR / "main.csv")
SWEEPS = pd.read_csv(RESULTS_DIR / "sweeps.csv")
VALIDATION = json.loads((RESULTS_DIR / "validation.json").read_text(encoding="utf-8"))
COACH = MAIN[MAIN["strategy"] == 1].iloc[0]

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
# Frávik frá þjálfurum: appelsínugult = verra, grátt = eins, blátt = betra
DIVERGING = LinearSegmentedColormap.from_list("diff", [ORANGE, "#efeeea", BLUE])

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 12, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=200)
    plt.close(fig)
    print("  ", name)


def subtitle(ax, text):
    ax.text(0, 1.01, text, transform=ax.transAxes, fontsize=10.5, color=INK2, va="bottom")


def pp(x):
    """Brot -> prósentustig."""
    return 100 * x


def main_label(row):
    """T.d. '4. Reyna ef togo ≤ 6' (stikinn settur inn í nafnið)."""
    name = st.NAMES[row["strategy"]]
    if row["strategy"] == 3:
        name = name.replace("q", f"q = {row['q']:.1f}")
    if row["strategy"] in (4, 6):
        name = name.replace("X", str(int(row["max_togo"])))
    if row["strategy"] == 6:
        name = name.replace("Y", str(int(row["max_pos"])))
    if row["strategy"] == 9:
        name += f" (≤ {int(row['max_drives_left'])})"
    return f"{row['strategy']}. {name}"


# ---------------------------------------------------- 1. samanburður allra stefna
def fig_comparison():
    d = MAIN.sort_values("p").reset_index(drop=True)
    diff = pp(d["p"] - COACH["p"])
    err = pp(1.96 * d["se"])
    colors = [GRAY if r["strategy"] == 1 or r["p_value"] >= 0.05
              else (BLUE if r["diff_vs_coach"] > 0 else ORANGE) for _, r in d.iterrows()]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    y = np.arange(len(d))
    ax.barh(y, diff, color=colors, height=0.62)
    ax.errorbar(diff, y, xerr=err, fmt="none", ecolor=INK, capsize=3, lw=1)
    ax.axvline(0, color=INK2, lw=1)
    for yi, dv, e, (_, r) in zip(y, diff, err, d.iterrows()):
        text = "viðmið" if r["strategy"] == 1 else f"{dv:+.1f} pp"
        x = dv + e + 0.25 if dv >= 0 else dv - e - 0.25
        ax.text(x, yi, f"{text}   ({100 * r['go_rate']:.0f}% reynt)" if dv >= 0 else
                f"({100 * r['go_rate']:.0f}% reynt)   {text}",
                va="center", ha="left" if dv >= 0 else "right", fontsize=9.5, color=INK)
    ax.set_yticks(y, [main_label(r) for _, r in d.iterrows()], fontsize=10.5)
    lim = max(abs(diff).max() + err.max(), 1) + 5
    ax.set_xlim(-lim, lim)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Breyting á sigurlíkum miðað við þjálfara (prósentustig, 95% öryggisbil)")
    ax.set_title("Hvaða 4th down stefna vinnur flesta leiki?", pad=22)
    subtitle(ax, f"Blátt = marktækt betra en þjálfarar, appelsínugult = marktækt verra.  "
                 f"Þjálfarar vinna {COACH['p']:.1%} gegn sjálfum sér.  N = {int(COACH['n']):,} leikir á stefnu.")
    save(fig, "1_samanburdur_stefna.png")


# --------------------------------------- 2. sigurlíkur sem fall af árásargirni
def fig_aggressiveness():
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.axhline(pp(COACH["p"]), color=GRAY, lw=1, ls="--", zorder=1)
    ax.text(101, pp(COACH["p"]) + 0.2, "þjálfarar", color=INK2, fontsize=9.5, ha="right", va="bottom")

    for number, col, label in [(3, ORANGE, "3. Reyna með líkum q (óháð stöðu)"),
                               (4, BLUE, "4. Reyna ef togo ≤ X")]:
        d = SWEEPS[SWEEPS["strategy"] == number].sort_values("go_rate")
        x = 100 * d["go_rate"]
        ax.fill_between(x, pp(d["ci_low"]), pp(d["ci_high"]), color=col, alpha=0.18, lw=0)
        ax.plot(x, pp(d["p"]), color=col, lw=2, marker="o", ms=6, label=label, zorder=3)
        if number == 4:
            for _, r in d[d["max_togo"].isin([1, 2, 4, 6, 10, 15])].iterrows():
                ax.annotate(f"X={int(r['max_togo'])}", (100 * r["go_rate"], pp(r["p"])),
                            textcoords="offset points", xytext=(0, 9), ha="center",
                            fontsize=9, color=INK2)

    # Aðrar stefnur sem stakir punktar, merktir með númeri
    others = MAIN[~MAIN["strategy"].isin([3, 4])]
    ax.errorbar(100 * others["go_rate"], pp(others["p"]), yerr=pp(1.96 * others["se"]),
                fmt="D", ms=7, color=INK, capsize=2, zorder=5, label="Aðrar stefnur (númer)")
    for _, r in others.iterrows():
        ax.annotate(str(r["strategy"]), (100 * r["go_rate"], pp(r["p"])),
                    textcoords="offset points", xytext=(7, 5), fontsize=10.5,
                    fontweight="bold", color=INK)

    ax.set_xlabel("Hlutfall 4th down þar sem liðið reynir (%)")
    ax.set_ylabel("Sigurlíkur (%)")
    ax.set_xlim(-4, 104)
    ax.set_title("Það skiptir máli HVAR liðið reynir, ekki bara hversu oft", pad=22)
    subtitle(ax, "Að reyna af handahófi (appelsínugult) er alltaf verra en að velja stuttu tilraunirnar (blátt).")
    ax.legend(loc="lower center", fontsize=9.5)
    save(fig, "2_sigurlikur_arasargirni.png")


# --------------------------------------------- 3. hitakort fyrir stefnu 6
def fig_heatmap_xy():
    d = SWEEPS[SWEEPS["strategy"] == 6]
    grid = d.pivot(index="max_togo", columns="max_pos", values="p")
    diff = pp(grid - COACH["p"]).to_numpy()
    lim = np.nanmax(np.abs(diff))

    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    im = ax.imshow(diff, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    best = np.unravel_index(np.nanargmax(diff), diff.shape)
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            ax.text(j, i, f"{diff[i, j]:+.1f}", ha="center", va="center", fontsize=9.5,
                    color="white" if abs(diff[i, j]) > 0.6 * lim else INK,
                    fontweight="bold" if (i, j) == best else "normal")
    ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, lw=2.2, ec=INK))

    xlabels = ["hvar sem er" if y == 99 else str(y) for y in grid.columns]
    ylabels = ["hvað sem er" if x == 99 else str(x) for x in grid.index]
    ax.set_xticks(range(len(xlabels)), xlabels, fontsize=10)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=10)
    ax.set_xlabel("Y: reyna aðeins ef liðið er innan Y yarda frá marki")
    ax.set_ylabel("X: reyna aðeins ef yards to go ≤ X")
    ax.grid(False)
    ax.set_title("Stefna 6: skiptir staðsetning á vellinum máli?", pad=22)
    subtitle(ax, "Breyting á sigurlíkum miðað við þjálfara (prósentustig).  "
                 "Aftasti dálkur = stefna 4 (staðsetning hunsuð).  Rammi = besta samsetning.")
    fig.colorbar(im, ax=ax, fraction=0.03, label="prósentustig")
    save(fig, "3_hitakort_X_Y.png")


# ------------------------------------------------ 4. stefna 9 eftir K
def fig_trailing():
    d = SWEEPS[SWEEPS["strategy"] == 9].sort_values("max_drives_left")
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.axhline(0, color=GRAY, lw=1, ls="--")
    k = d["max_drives_left"].astype(int)
    diff = pp(d["p"] - COACH["p"])
    ax.errorbar(k, diff, yerr=pp(1.96 * d["se"]), color=BLUE, lw=2, marker="o", ms=7, capsize=3)
    ax.set_xticks(k)
    ax.set_xlabel("K: liðið reynir alltaf ef það er undir og á K eða færri sóknir eftir")
    ax.set_ylabel("Breyting á sigurlíkum (pp)")
    ax.set_title("Stefna 9: reyna oftar þegar liðið er undir", pad=22)
    subtitle(ax, "Miðað við þjálfara.  Nema field goal jafni eða komi liðinu yfir.  "
                 f"Hvert lið fær 12 sóknir; K = 12 þýðir allan leikinn.")
    save(fig, "4_undir_seint.png")


# --------------------------------------------- 5. p(d): gögn vs hermir
def fig_conversion():
    v = pd.DataFrame(VALIDATION["conversion"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(v["d"], v["data_3rd_4th"], color=BLUE, lw=2, marker="o", label="Gögn: 3rd + 4th down")
    ax.plot(v["d"], v["data_4th"], color=AQUA, lw=0, marker="s", ms=7, label="Gögn: aðeins 4th down")
    ax.plot(v["d"], v["sim"], color=ORANGE, lw=2, ls="--", marker="^", label="Hermir")
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("Yards to go (d)")
    ax.set_ylabel("P(1st down | d)")
    ax.set_title("Líkur á 1st down, p(d)")
    ax.legend()
    save(fig, "5_p_d.png")


# ------------------------------------------------------------- 6. pFG(x)
def fig_field_goal():
    b0, b1 = ESTIMATES["fg_logit"]
    xs = np.linspace(18, 65, 200)
    emp = ESTIMATES["fg_by_distance"]
    emp = emp[(emp["count"] >= 30) & (emp.index >= 15) & (emp.index <= 65)]
    max_kick = FG_MAX_POS + ESTIMATES["fg_kick_offset"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, 1 / (1 + np.exp(-(b0 + b1 * xs))), color=BLUE, lw=2,
            label=f"Logistic: 1 / (1 + e^-({b0:.2f} {b1:+.3f}x))")
    ax.scatter(emp.index + 2.5, emp["mean"], s=np.clip(emp["count"] / 8, 15, 200), color=ORANGE,
               zorder=3, label="Gögn (5 yarda bil, stærð ~ fjöldi)")
    ax.axvline(max_kick, color=GRAY, ls="--", lw=1)
    ax.text(max_kick + 0.5, 0.15, f"Lengsta FG í\nhermi ({max_kick} yd)", color=INK2, fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Lengd sparks x (yardar)")
    ax.set_ylabel("P(FG heppnast)")
    ax.set_title("Líkur á heppnuðu field goal, pFG(x)")
    ax.legend(loc="lower left")
    save(fig, "6_pFG.png")


# --------------------------------------- 7. hegðun þjálfara: hitakort
def fig_coaches():
    go = ESTIMATES["coach_probs"]["go"].unstack()
    n = ESTIMATES["coach_counts"].sum(axis=1).unstack()
    shape = dict(index=range(len(COACH_TOGO_LABELS)), columns=range(len(COACH_POS_LABELS)))
    arr = go.reindex(**shape).to_numpy()
    arr = np.where(np.nan_to_num(n.reindex(**shape).to_numpy()) >= 15, arr, np.nan)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(arr, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if not np.isnan(arr[i, j]):
                ax.text(j, i, f"{100 * arr[i, j]:.0f}", ha="center", va="center", fontsize=9,
                        color="white" if arr[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(COACH_POS_LABELS)), COACH_POS_LABELS, fontsize=9)
    ax.set_yticks(range(len(COACH_TOGO_LABELS)), COACH_TOGO_LABELS)
    ax.set_xlabel("Fjarlægð að endamarki mótherja (yardar)")
    ax.set_ylabel("Yards to go")
    ax.grid(False)
    ax.set_title("Hversu oft reyna þjálfarar á 4th down? (%)", pad=22)
    subtitle(ax, f"{ESTIMATES['n_fourth_downs']:,} 4th down kerfi í 1.-3. leikhluta 2018-2025.  "
                 "Auð hólf: færri en 15 kerfi.")
    fig.colorbar(im, ax=ax, fraction=0.03, label="P(reyna)")
    save(fig, "7_thjalfarar.png")


# ------------------------------------------- 8. þróun eftir tímabilum
def fig_seasons():
    g = ESTIMATES["go_rate_by_season"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(g.index, 100 * g.values, color=BLUE, width=0.6)
    for season, v in g.items():
        ax.text(season, 100 * v + 0.3, f"{100 * v:.1f}%", ha="center", fontsize=10, color=INK)
    ax.set_xticks(g.index)
    ax.set_ylabel("% af 4th down þar sem reynt er")
    ax.grid(axis="x", visible=False)
    ax.set_title("Þjálfarar verða árásargjarnari (1.-3. leikhluti)")
    save(fig, "8_throun_eftir_arum.png")


# --------------------------------------- 9. útkomur sókna: hermir vs gögn
def fig_drive_outcomes():
    names = {"punt": "Punt", "TD": "TD", "FG": "FG", "turnover": "Tapaður bolti",
             "downs": "Tapað á downs", "FG miss": "Misheppnað FG"}
    data = VALIDATION["drives"]["data"]
    sim = VALIDATION["drives"]["sim"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - 0.19, [data.get(k, 0) for k in names], 0.36, color=BLUE,
           label="NFL gögn 2018-2025 (1.-3. leikhluti)")
    ax.bar(x + 0.19, [sim.get(k, 0) for k in names], 0.36, color=ORANGE,
           label="Hermir (þjálfarastefna)")
    ax.set_xticks(x, list(names.values()))
    ax.set_ylabel("Hlutfall sókna")
    ax.grid(axis="x", visible=False)
    ax.legend()
    ax.set_title("Staðfesting: hvernig enda sóknir?")
    save(fig, "9_utkomur_sokna.png")


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    for make in [fig_comparison, fig_aggressiveness, fig_heatmap_xy, fig_trailing,
                 fig_conversion, fig_field_goal, fig_coaches, fig_seasons, fig_drive_outcomes]:
        make()
    print("Gröf vistuð í figures/")


if __name__ == "__main__":
    main()
