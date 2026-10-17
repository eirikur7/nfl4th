"""
strategies.py
4th down stefnurnar sem bornar eru saman (númerin fylgja verkefnislýsingunni).

Stefna er fall  strategy(s, rng) -> "go" | "fg" | "punt"  þar sem s er
Situation (sjá simulator.py) með reitunum
  s.togo         yards to go
  s.pos          fjarlægð að endamarki mótherja
  s.score_diff   eigin stig mínus stig mótherja
  s.drives_left  sóknir sem liðið á eftir, að þessari meðtalinni (1 = síðasta)

Þegar lið reynir ekki tekur það field goal ef það er innan seilingar
(pos <= 37, spark <= 55 yd), annars punt.  Það gerir kick(s).

Stefnur 2-6 og 11 eru hreinar reglur: reyna ef skilyrðið er uppfyllt, annars sparka.
Stefnur 7-10 breyta hegðun þjálfara (stefnu 1) aðeins í þeim aðstæðum sem þær
fjalla um, og fylgja þjálfurunum annars.  Þannig mæla þær áhrif þessarar einu breytingar.

Stefnur með stika (q, X, Y, ...) eru búnar til með functools.partial, t.d.
partial(short_yardage, max_togo=4).
"""
from model import coach_decision
from common import FG_MAX_POS

GO, FG, PUNT = "go", "fg", "punt"


def kick(s):
    return FG if s.pos <= FG_MAX_POS else PUNT


def go_if(condition, s):
    return GO if condition else kick(s)


# 1. Raunveruleg hegðun þjálfara, metin úr gögnum (viðmið)
def coach(s, rng):
    return coach_decision(s.togo, s.pos, rng)


# 2. Aldrei reyna: alltaf punt eða field goal
def never(s, rng):
    return kick(s)


# 3. Reyna með líkum q, óháð stöðu
def random_q(s, rng, q):
    return go_if(rng.random() < q, s)


# 4. Reyna ef yards to go <= X, hvar sem er á vellinum
def short_yardage(s, rng, max_togo):
    return go_if(s.togo <= max_togo, s)


# 5. Reyna ef komið er yfir miðlínu, sama hversu margir yardar eru eftir
def past_midfield(s, rng):
    return go_if(s.pos < 50, s)


# 6. Sameinuð: reyna ef yards to go <= X og liðið er innan Y yarda frá marki
def short_and_close(s, rng, max_togo, max_pos):
    return go_if(s.togo <= max_togo and s.pos <= max_pos, s)


# 7. „Einskismannsland“: reyna alltaf 35-45 yarda frá marki, annars eins og þjálfarar
def no_mans_land(s, rng):
    return GO if 35 <= s.pos <= 45 else coach(s, rng)


# 8. Nálægt marki: reyna alltaf innan 5 yarda frá marki, annars eins og þjálfarar
def goal_line(s, rng):
    return GO if s.pos <= 5 else coach(s, rng)


# 9. Eftir stöðu: reyna alltaf þegar liðið er undir og á <= K sóknir eftir,
#    nema field goal jafni leikinn eða komi liðinu yfir.  Annars eins og þjálfarar.
def trailing_late(s, rng, max_drives_left):
    fg_ties_or_leads = s.pos <= FG_MAX_POS and s.score_diff + 3 >= 0
    if s.score_diff < 0 and s.drives_left <= max_drives_left and not fg_ties_or_leads:
        return GO
    return coach(s, rng)


# 10. Field goal eftir stöðu: innan FG-færis er sparkað ef FG
#     (a) kemur liðinu meira en 8 stigum yfir (tveggja sókna forysta), eða
#     (b) minnkar mun úr meira en 8 stigum í 8 eða minna (verður „one possession game“),
#     annars reynt.  Utan FG-færis eins og þjálfarar.
def score_aware_fg(s, rng):
    if s.pos > FG_MAX_POS:
        return coach(s, rng)
    after_fg = s.score_diff + 3
    two_score_lead = after_fg > 8
    back_to_one_score = s.score_diff < -8 and after_fg >= -8
    return FG if (two_score_lead or back_to_one_score) else GO


# 11. Alltaf reyna
def always(s, rng):
    return GO


NAMES = {
    1: "Þjálfarar (viðmið)",
    2: "Aldrei reyna",
    3: "Reyna með líkum q",
    4: "Reyna ef togo ≤ X",
    5: "Reyna yfir miðlínu",
    6: "Reyna ef togo ≤ X og pos ≤ Y",
    7: "Einskismannsland (35-45)",
    8: "Nálægt marki (≤ 5 yd)",
    9: "Undir og fáar sóknir eftir",
    10: "Field goal eftir stöðu",
    11: "Alltaf reyna",
}
