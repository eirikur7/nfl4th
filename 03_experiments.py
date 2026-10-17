"""
03_experiments.py
Monte Carlo hermun á öllum stefnunum.  Lið A notar stefnuna sem er prófuð,
mótherjinn (lið B) fylgir alltaf stefnu 1 (hegðun þjálfara).

Skref:
  1. Skönnun: stefnur með stika eru hermdar fyrir mörg gildi (q, X, Y, K), N_SWEEP leikir hvert.
  2. Aðalsamanburður: allar 11 stefnurnar hermdar með N_MAIN leikjum.  Fyrir stefnur með
     stika er notað besta gildið úr skönnuninni, hermt upp á nýtt með nýjum slembitölum
     (annars væri besta gildið ofmetið, því það var valið sem hæsta af mörgum).
  3. Tilgátupróf: tveggja hlutfalla z-próf, H0: p_stefna = p_þjálfari.
  4. Staðfesting á líkani: hermt p(d) og útkomur sókna borið saman við gögn.

Úttak (results/):
  sweeps.csv      allar keyrslur í skönnun (ein lína á stefnu og stika)
  main.csv        aðalsamanburður, ein lína á stefnu, með z-prófi gegn þjálfurum
  validation.json staðfesting á líkani og önnur próf
"""
import json
import math
import random
from collections import Counter
from functools import partial
from multiprocessing import Pool

import pandas as pd
from scipy.stats import norm

import model
import simulator
import strategies as st
from common import RESULTS_DIR

N_MAIN = 200_000     # leikir í aðalsamanburði
N_SWEEP = 30_000     # leikir fyrir hvert stikagildi í skönnun

# Gildi sem eru skönnuð.  max_pos = 99 þýðir „hvar sem er á vellinum“.
Q_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
X_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
HEATMAP_X = [1, 2, 3, 4, 5, 6, 8, 10, 99]
HEATMAP_Y = [10, 20, 30, 40, 50, 60, 70, 80, 99]
K_VALUES = [1, 2, 3, 4, 6, 12]


# ---------------------------------------------------------------- keyrslur
def experiment(number, strategy, n, **params):
    """Lýsing á einni keyrslu: stefna nr. `number` með stikum `params`."""
    label = st.NAMES[number]
    if params:
        label += " (" + ", ".join(f"{k}={v}" for k, v in params.items()) + ")"
    return {"strategy": number, "label": label, "n": n, "params": params,
            "policy": partial(strategy, **params) if params else strategy}


def run_one(task):
    """Keyrt í sér ferli.  Hver keyrsla fær sitt eigið fræ (seed)."""
    exp, seed = task
    result = simulator.simulate(exp["policy"], exp["n"], seed)
    return {"strategy": exp["strategy"], "label": exp["label"], **exp["params"], **result}


def run_all(experiments, first_seed):
    tasks = [(exp, first_seed + i) for i, exp in enumerate(experiments)]
    with Pool() as pool:
        rows = []
        for row in pool.imap(run_one, tasks):
            rows.append(row)
            print(f"  {row['label']:<45} p̂ = {row['p']:.3f} ± {1.96 * row['se']:.3f}"
                  f"   reynt á {100 * row['go_rate']:3.0f}% af 4th down")
    return pd.DataFrame(rows)


def sweep_experiments():
    return (
        [experiment(3, st.random_q, N_SWEEP, q=q) for q in Q_VALUES]
        + [experiment(4, st.short_yardage, N_SWEEP, max_togo=x) for x in X_VALUES]
        + [experiment(6, st.short_and_close, N_SWEEP, max_togo=x, max_pos=y)
           for x in HEATMAP_X for y in HEATMAP_Y]
        + [experiment(9, st.trailing_late, N_SWEEP, max_drives_left=k) for k in K_VALUES]
    )


def best_params(sweeps, number, param_names):
    """Stikagildin sem gáfu hæstar sigurlíkur í skönnun fyrir stefnu `number`."""
    best = sweeps[sweeps["strategy"] == number].sort_values("p").iloc[-1]
    return {name: float(best[name]) if name == "q" else int(best[name]) for name in param_names}


def main_experiments(sweeps):
    return [
        experiment(1, st.coach, N_MAIN),
        experiment(2, st.never, N_MAIN),
        experiment(3, st.random_q, N_MAIN, **best_params(sweeps, 3, ["q"])),
        experiment(4, st.short_yardage, N_MAIN, **best_params(sweeps, 4, ["max_togo"])),
        experiment(5, st.past_midfield, N_MAIN),
        experiment(6, st.short_and_close, N_MAIN,
                   **best_params(sweeps, 6, ["max_togo", "max_pos"])),
        experiment(7, st.no_mans_land, N_MAIN),
        experiment(8, st.goal_line, N_MAIN),
        experiment(9, st.trailing_late, N_MAIN, **best_params(sweeps, 9, ["max_drives_left"])),
        experiment(10, st.score_aware_fg, N_MAIN),
        experiment(11, st.always, N_MAIN),
    ]


# ---------------------------------------------------------------- tilgátupróf
def z_test(a, b):
    """Tveggja hlutfalla z-próf (pooled), H0: p_a = p_b.  Skilar (z, p-gildi)."""
    pooled = (a["p"] * a["n"] + b["p"] * b["n"]) / (a["n"] + b["n"])
    se = math.sqrt(pooled * (1 - pooled) * (1 / a["n"] + 1 / b["n"]))
    z = (a["p"] - b["p"]) / se
    return z, 2 * (1 - norm.cdf(abs(z)))


def add_tests_vs_coach(main):
    coach = main[main["strategy"] == 1].iloc[0]
    tests = [z_test(row, coach) for _, row in main.iterrows()]
    main["diff_vs_coach"] = main["p"] - coach["p"]
    main["z"] = [z for z, _ in tests]
    main["p_value"] = [p for _, p in tests]
    main.loc[main["strategy"] == 1, ["diff_vs_coach", "z", "p_value"]] = math.nan
    return main


def other_tests(main):
    by_number = main.set_index("strategy")
    coach = by_number.loc[1]
    z_coach = (coach["p"] - 0.5) / coach["se"]
    z_always_never, p_always_never = z_test(by_number.loc[11], by_number.loc[2])
    z_pos, p_pos = z_test(by_number.loc[6], by_number.loc[4])
    return {
        "coach_vs_coach": {"p": coach["p"], "z": z_coach,
                           "p_value": 2 * (1 - norm.cdf(abs(z_coach)))},
        "always_vs_never": {"diff": by_number.loc[11, "p"] - by_number.loc[2, "p"],
                            "z": z_always_never, "p_value": p_always_never},
        "combined_vs_togo_only": {"diff": by_number.loc[6, "p"] - by_number.loc[4, "p"],
                                  "z": z_pos, "p_value": p_pos},
    }


# ---------------------------------------------------------------- staðfesting
def validate_conversion(n=40_000):
    """Hermt P(1st down | 4th & d) frá miðjum velli borið saman við gögnin."""
    rng = random.Random(7)
    data_both = model.ESTIMATES["conversion_3rd_4th"]
    data_4th = model.ESTIMATES["conversion_4th"]
    rows = []
    for d in range(1, 11):
        successes = 0
        for _ in range(n):
            turnover, yards = model.run_play(4, d, 50, rng)
            successes += (not turnover) and yards >= d
        rows.append({"d": d, "sim": successes / n,
                     "data_3rd_4th": data_both.loc[d, "mean"],
                     "data_4th": data_4th["mean"].get(d, math.nan),
                     "n_4th": int(data_4th["count"].get(d, 0))})
    return rows


def validate_drives(n=200_000):
    """Hvernig sóknir enda í hermi (þjálfarastefna) borið saman við gögnin."""
    rng = random.Random(11)
    outcomes = Counter()
    points = 0
    for _ in range(n):
        result = simulator.play_drive(model.kickoff_start_pos(rng), st.coach, rng)
        outcomes[result.outcome] += 1
        points += result.points
    return {"sim": {k: v / n for k, v in outcomes.items()},
            "data": model.ESTIMATES["drive_outcomes_data"],
            "points_per_drive_sim": points / n}


# ---------------------------------------------------------------- keyrsla
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"=== 1. Skönnun ({N_SWEEP:,} leikir á hvert gildi) ===")
    sweeps = run_all(sweep_experiments(), first_seed=1000)
    sweeps.to_csv(RESULTS_DIR / "sweeps.csv", index=False)

    print(f"\n=== 2. Aðalsamanburður ({N_MAIN:,} leikir á stefnu) ===")
    main_df = add_tests_vs_coach(run_all(main_experiments(sweeps), first_seed=1))
    main_df.to_csv(RESULTS_DIR / "main.csv", index=False)

    print("\n=== 3-4. Tilgátupróf og staðfesting ===")
    validation = {"tests": other_tests(main_df),
                  "conversion": validate_conversion(),
                  "drives": validate_drives(),
                  "n_main": N_MAIN, "n_sweep": N_SWEEP}
    with open(RESULTS_DIR / "validation.json", "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2, ensure_ascii=False)
    print(json.dumps(validation["tests"], indent=2))
    print("\nVistað í results/.  Keyrðu 04_figures.py og 05_report.py næst.")


if __name__ == "__main__":
    main()
