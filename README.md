# Hversu oft borgar sig að reyna við 4th down?

Verkefni í Hagnýtri líkindafræði.  Markov-keðju hermir fyrir NFL sóknir og Monte Carlo
samanburður á 11 mismunandi 4th down stefnum, metinn úr play-by-play gögnum
NFL 2018–2025 (nflverse, ~372 þúsund kerfi í deildarkeppni).

**Niðurstöðurnar eru í [results/NIDURSTODUR.md](results/NIDURSTODUR.md)** (töflur og gröf).

## Keyrsla

```bash
pip install pandas numpy scipy matplotlib
# sækja play_by_play_YYYY.csv.gz af https://github.com/nflverse/nflverse-data/releases/tag/pbp
# og vista sem data/pbp_YYYY.csv.gz  (2018-2025)
python 01_load_data.py     # les gögnin                        (~30 s)
python 02_estimate.py      # metur líkur -> model.pkl          (~10 s)
python 03_experiments.py   # Monte Carlo hermun -> results/    (~10-15 mín)
python 04_figures.py       # gröf -> figures/
python 05_report.py        # töflur -> results/NIDURSTODUR.md
```

`model.pkl` fylgir með, svo það má byrja á skrefi 3 án þess að sækja gögnin.

## Skrárnar

Gögn → líkur → hermir → niðurstöður:

| Skrá | Hlutverk |
|---|---|
| `common.py` | Slóðir, fastar og hólfaskiptingar sem fleiri skrár nota. Útskýrir `pos` og `togo`. |
| `01_load_data.py` | Les gögnin og heldur aðeins nauðsynlegum dálkum |
| `02_estimate.py` | Metur allar líkur, eitt fall á hvern lið (yardar, p(d), pFG, punt, kickoff, þjálfarar) |
| `model.py` | Dregur slembiútkomur úr metnu líkunum: eitt kerfi, FG, punt, kickoff, ákvörðun þjálfara |
| `strategies.py` | Stefnurnar 11, númeraðar eins og í verkefnislýsingunni |
| `simulator.py` | Sókn (Markov-keðja), leikur (12 sóknir á lið) og `simulate()` sem metur sigurlíkur |
| `03_experiments.py` | Keyrir allar stefnurnar, tilgátupróf og staðfestingu á líkaninu |
| `04_figures.py` | Gröf |
| `05_report.py` | Skrifar niðurstöðurnar sem læsilegar töflur |

## Líkanið

* **Ástand sóknar:** (down, yards to go, `pos`), þar sem `pos` = fjarlægð að endamarki mótherja.
  Sókn endar í einu gleypiástandi: TD, FG, misheppnað FG, punt, tapaður bolti, tapað á downs, safety.
* **Kerfi:** yardar dregnir úr raunverulegri dreifingu kerfa með sama down, sama yards-to-go hólfi
  og svæði (red zone / annað).  3rd og 4th down deila dreifingu (meiri gögn).
  Líkur á töpuðum bolta metnar í sama hólfi.
* **pFG(x)** = 1 / (1 + e^-(6.18 − 0.105x)), logistic regression (ML) á 8.241 spörkum.
* **Punt:** vallarstaða mótherja dregin úr raunverulegum puntum frá svipaðri vallarstöðu (meðalnettó 41 yd).
* **Þjálfarar (stefna 1):** P(go / FG / punt | yards to go, vallarstaða) úr 18.921 4th down kerfum í 1.–3. leikhluta.
* **Ef ekki reynt:** FG ef spark ≤ 55 yd (`pos` ≤ 37), annars punt.
* **Leikur:** 12 sóknir á lið.  Stefnurnar sjá stöðuna í leiknum (`score_diff`) og hversu margar sóknir
  eru eftir (`drives_left`), sem kemur í stað klukku.
* **Einföldun:** engin klukka, jafn sterk lið, engar 2-point tilraunir, engin framlenging (jafntefli = ½ sigur).

## Stefnurnar

Stefnur 2–6 og 11 eru hreinar reglur: reyna ef skilyrðið gildir, annars sparka.
Stefnur 7–10 eru **breytingar á hegðun þjálfara**: þær fylgja stefnu 1 nema í þeim aðstæðum sem
þær fjalla um, svo munurinn mælir áhrif þessarar einu breytingar.

| Nr. | Stefna | Stikar skannaðir |
|---|---|---|
| 1 | Raunveruleg hegðun þjálfara (viðmið) | |
| 2 | Aldrei reyna | |
| 3 | Reyna með líkum q | q = 0, 0.1, …, 1 |
| 4 | Reyna ef yards to go ≤ X | X = 1–15 |
| 5 | Reyna ef komið er yfir miðlínu (`pos` < 50) | |
| 6 | Reyna ef yards to go ≤ X og `pos` ≤ Y | 9 × 9 hitakort |
| 7 | Einskismannsland: alltaf reyna á `pos` 35–45 | |
| 8 | Nálægt marki: alltaf reyna á `pos` ≤ 5 | |
| 9 | Alltaf reyna ef undir og ≤ K sóknir eftir (nema FG jafni eða komi liðinu yfir) | K = 1, 2, 3, 4, 6, 12 |
| 10 | Innan FG-færis: FG ef það gefur > 8 stiga forystu eða minnkar mun úr > 8 í ≤ 8, annars reyna | |
| 11 | Alltaf reyna | |

Fyrir stefnur með stika er besta gildið úr skönnun hermt aftur með nýjum slembitölum og
200.000 leikjum, svo það sé ekki ofmetið (besta af mörgum keyrslum er alltaf heppið).

## Helstu niðurstöður

RESULTS_PLACEHOLDER

## Fyrirvarar

* Engin klukka: í raun breytast ákvarðanir eftir tíma, og mótherjinn bregst ekki við árásargjarnri stefnu.
* 4th down tilraunir eru metnar með 3rd down gögnum; vörnin gæti spilað öðruvísi á 4th down.
* Hlutfallsleg stærð munar er traustari en nákvæm gildi sigurlíkanna.
* Jafntefli teljast ½ sigur; þá er Bernoulli-dreifnin örlítið ofmetin, svo öryggisbilin eru íhaldssöm.
