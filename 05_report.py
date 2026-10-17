"""
05_report.py
Skrifar results/NIDURSTODUR.md: allar niðurstöður í læsilegum töflum, með
tilvísunum í gröfin.  Les results/ (keyra 03_experiments.py fyrst).
"""
import json

import pandas as pd

import strategies as st
from common import RESULTS_DIR

MAIN = pd.read_csv(RESULTS_DIR / "main.csv")
SWEEPS = pd.read_csv(RESULTS_DIR / "sweeps.csv")
VALIDATION = json.loads((RESULTS_DIR / "validation.json").read_text(encoding="utf-8"))
COACH = MAIN[MAIN["strategy"] == 1].iloc[0]

DESCRIPTIONS = {
    1: "Það sem NFL þjálfarar gera í raun (metið úr gögnum)",
    2: "Alltaf punt eða field goal",
    3: "Reyna með líkum q, óháð stöðu",
    4: "Reyna ef yards to go ≤ X, hvar sem er á vellinum",
    5: "Reyna ef komið er yfir miðlínu, sama hve langt er í 1st down",
    6: "Reyna ef yards to go ≤ X og liðið er innan Y yarda frá marki",
    7: "Reyna alltaf 35-45 yörðum frá marki, annars eins og þjálfarar",
    8: "Reyna alltaf innan 5 yarda frá marki, annars eins og þjálfarar",
    9: "Reyna alltaf ef liðið er undir og á ≤ K sóknir eftir, annars eins og þjálfarar",
    10: "Innan FG-færis: FG ef það gefur > 8 stiga forystu eða minnkar mun í ≤ 8, annars reyna",
    11: "Alltaf reyna",
}


# ---------------------------------------------------------------- snið
def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def pp(x):
    return f"{100 * x:+.1f} pp"


def ci(row):
    return f"{100 * row['ci_low']:.1f}–{100 * row['ci_high']:.1f}%"


def p_value(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def verdict(row):
    if row["strategy"] == 1:
        return "viðmið"
    if row["p_value"] >= 0.05:
        return "enginn marktækur munur"
    return "**betri**" if row["diff_vs_coach"] > 0 else "verri"


def params_text(row):
    parts = []
    if row["strategy"] == 3:
        parts.append(f"q = {row['q']:.1f}")
    if row["strategy"] in (4, 6):
        parts.append(f"X = {int(row['max_togo'])}")
    if row["strategy"] == 6:
        parts.append(f"Y = {int(row['max_pos'])}")
    if row["strategy"] == 9:
        parts.append(f"K = {int(row['max_drives_left'])}")
    return ", ".join(parts)


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def sweep_rows(number, param, fmt):
    d = SWEEPS[SWEEPS["strategy"] == number].sort_values(param)
    return [[fmt(r[param]), pct(r["go_rate"], 0), pct(r["p"]), pp(r["p"] - COACH["p"])]
            for _, r in d.iterrows()]


# ---------------------------------------------------------------- kaflar
def section_main():
    d = MAIN.sort_values("p", ascending=False).reset_index(drop=True)
    rows = []
    for rank, r in d.iterrows():
        name = f"{r['strategy']}. {st.NAMES[r['strategy']]}"
        if params_text(r):
            name += f" ({params_text(r)})"
        rows.append([rank + 1, name, pct(r["go_rate"], 0), pct(r["p"]), ci(r),
                     "—" if r["strategy"] == 1 else pp(r["diff_vs_coach"]),
                     "—" if r["strategy"] == 1 else p_value(r["p_value"]), verdict(r)])
    return f"""## 1. Allar stefnurnar bornar saman

Hver stefna spilar {int(COACH['n']):,} leiki gegn liði sem fylgir stefnu þjálfara.
Raðað frá bestu til verstu.  Fyrir stefnur með stika (3, 4, 6, 9) er notað besta
gildið úr skönnun (kafli 3-5), hermt aftur með nýjum slembitölum.

![Samanburður stefna](../figures/1_samanburdur_stefna.png)

{table(["Röð", "Stefna", "Reynt á 4th down", "Sigurlíkur", "95% öryggisbil",
        "Munur vs þjálfara", "p-gildi", "Niðurstaða"], rows)}

**Lýsing stefna**

{table(["Nr.", "Stefna"], [[k, v] for k, v in DESCRIPTIONS.items()])}

*Reynt á 4th down* = hlutfall 4th down þar sem liðið reyndi.
*pp* = prósentustig.  *p-gildi* úr tveggja hlutfalla z-prófi, H0: sömu sigurlíkur og þjálfarar.
"""


def section_aggressiveness():
    tests = VALIDATION["tests"]["always_vs_never"]
    return f"""## 2. Skiptir máli hversu oft er reynt, eða hvar?

![Sigurlíkur sem fall af árásargirni](../figures/2_sigurlikur_arasargirni.png)

* Alltaf reyna vs aldrei reyna: munur {pp(tests['diff'])} (z = {tests['z']:.1f}, p {p_value(tests['p_value'])}).
* Bláa línan (stefna 4) liggur alls staðar ofan við appelsínugulu línuna (stefna 3):
  sama hlutfall tilrauna gefur hærri sigurlíkur ef reynt er þegar stutt er í 1st down.

### Stefna 3: reyna með líkum q

{table(["q", "Reynt", "Sigurlíkur", "Munur vs þjálfara"], sweep_rows(3, "q", lambda v: f"{v:.1f}"))}

### Stefna 4: reyna ef yards to go ≤ X

{table(["X", "Reynt", "Sigurlíkur", "Munur vs þjálfara"], sweep_rows(4, "max_togo", lambda v: int(v)))}

Hvert gildi: {VALIDATION['n_sweep']:,} leikir, 95% öryggisbil ± ~{100 * 1.96 * SWEEPS['se'].mean():.1f} pp.
"""


def section_location():
    d = SWEEPS[SWEEPS["strategy"] == 6]
    grid = d.pivot(index="max_togo", columns="max_pos", values="p")
    col_name = {99: "hvar sem er"}
    header = ["X \\ Y"] + [col_name.get(y, str(y)) for y in grid.columns]
    rows = [[("hvað sem er" if x == 99 else x)] + [pp(v - COACH["p"]) for v in grid.loc[x]]
            for x in grid.index]
    test = VALIDATION["tests"]["combined_vs_togo_only"]
    s4 = MAIN[MAIN["strategy"] == 4].iloc[0]
    s6 = MAIN[MAIN["strategy"] == 6].iloc[0]
    significant = "marktækur" if test["p_value"] < 0.05 else "ekki marktækur"
    return f"""## 3. Skiptir staðsetning á vellinum máli? (stefna 6)

Hvert hólf: reyna ef yards to go ≤ X **og** liðið er innan Y yarda frá marki, annars sparka.
Tölurnar eru breyting á sigurlíkum miðað við þjálfara.  Dálkurinn „hvar sem er“ er stefna 4
(staðsetning hunsuð) og línan „hvað sem er“ með Y = 50 er stefna 5 (yfir miðlínu).

![Hitakort X og Y](../figures/3_hitakort_X_Y.png)

{table(header, rows)}

**Svar:** besta samsetningin (X = {int(s6['max_togo'])}, Y = {int(s6['max_pos'])}) gefur
{pct(s6['p'])} en besta reglan sem hunsar staðsetningu (X = {int(s4['max_togo'])}) gefur
{pct(s4['p'])}.  Munurinn er {pp(test['diff'])} (z = {test['z']:.2f}, p = {p_value(test['p_value'])}),
þ.e. {significant} munur (báðar hermdar aftur með {int(s4['n']):,} leikjum).
"""


def section_game_state():
    by_number = MAIN.set_index("strategy")
    rows = [[n, st.NAMES[n], pct(by_number.loc[n, "go_rate"], 0), pct(by_number.loc[n, "p"]),
             pp(by_number.loc[n, "diff_vs_coach"]), p_value(by_number.loc[n, "p_value"])]
            for n in (7, 8, 9, 10)]
    return f"""## 4. Breytingar á hegðun þjálfara (stefnur 7-10)

Þessar stefnur fylgja þjálfurum nema í þeim aðstæðum sem þær fjalla um, svo munurinn
mælir áhrif þessarar einu breytingar.

{table(["Nr.", "Stefna", "Reynt", "Sigurlíkur", "Munur vs þjálfara", "p-gildi"], rows)}

### Stefna 9: hversu snemma á að byrja að reyna þegar liðið er undir?

K = fjöldi sókna sem liðið á eftir (af 12).  K = 12 þýðir: alltaf þegar liðið er undir.

![Stefna 9](../figures/4_undir_seint.png)

{table(["K", "Reynt", "Sigurlíkur", "Munur vs þjálfara"], sweep_rows(9, "max_drives_left", lambda v: int(v)))}
"""


def section_validation():
    t = VALIDATION["tests"]["coach_vs_coach"]
    conv = [[c["d"], pct(c["sim"]), pct(c["data_3rd_4th"]),
             "—" if pd.isna(c["data_4th"]) else pct(c["data_4th"]), c["n_4th"]]
            for c in VALIDATION["conversion"]]
    names = {"TD": "TD", "FG": "FG", "punt": "Punt", "turnover": "Tapaður bolti",
             "downs": "Tapað á downs", "FG miss": "Misheppnað FG", "safety": "Safety"}
    sim, data = VALIDATION["drives"]["sim"], VALIDATION["drives"]["data"]
    drives = [[label, pct(sim.get(k, 0)), pct(data.get(k, 0))] for k, label in names.items()]
    return f"""## 5. Staðfesting á líkaninu

* **Þjálfari gegn þjálfara** á að gefa 50%: p̂ = {pct(t['p'], 2)}, z = {t['z']:.2f}, p = {p_value(t['p_value'])}.
* **Líkur á 1st down á 4th & d** (hermt frá miðlínu vs gögn):

{table(["d", "Hermir", "Gögn 3rd+4th", "Gögn aðeins 4th", "Fjöldi 4th down"], conv)}

* **Hvernig sóknir enda** (hermt með þjálfarastefnu vs gögn, 1.-3. leikhluti).
  Hermd stig á sókn: {VALIDATION['drives']['points_per_drive_sim']:.2f}.

{table(["Útkoma", "Hermir", "Gögn"], drives)}

Sjá einnig figures/5_p_d.png, 6_pFG.png og 9_utkomur_sokna.png.
"""


def main():
    text = f"""# Niðurstöður: hversu oft borgar sig að reyna við 4th down?

*Skráin er búin til sjálfkrafa af 05_report.py úr results/.*

**Hvernig á að lesa tölurnar.**  Sigurlíkur p̂ = (sigrar + ½ · jafntefli) / N, þar sem N er
fjöldi hermdra leikja gegn liði sem fylgir stefnu þjálfara.  Jafn góð stefna og þjálfarar
fær því um 50%.  95% öryggisbil: p̂ ± 1.96 · √(p̂(1−p̂)/N).  „Munur vs þjálfara“ er í
prósentustigum (pp): +5 pp þýðir t.d. 55% sigurlíkur í stað 50%.

{section_main()}
{section_aggressiveness()}
{section_location()}
{section_game_state()}
{section_validation()}"""
    out = RESULTS_DIR / "NIDURSTODUR.md"
    out.write_text(text, encoding="utf-8")
    print(f"Vistað: {out}")


if __name__ == "__main__":
    main()
