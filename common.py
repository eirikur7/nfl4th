"""
common.py
Slóðir, fastar og hólfaskiptingar sem fleiri en ein skrá nota.

Hugtök sem eru notuð í öllum skránum:
  pos   = yardline_100 = fjarlægð (yardar) að endamarki mótherja, 1..99.
          pos = 75 er eigin 25 yarda lína, pos = 50 er miðlína, pos = 5 er
          5 yördum frá því að skora.
  togo  = yards to go, yardar sem vantar upp á 1st down.
"""
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"            # hrá gögn frá nflverse (01_load_data.py)
MODEL_FILE = ROOT / "model.pkl"     # metnar líkur (02_estimate.py)
RESULTS_DIR = ROOT / "results"      # niðurstöður hermunar (03_experiments.py)
FIGURES_DIR = ROOT / "figures"      # gröf (04_figures.py)

SEASONS = range(2018, 2026)

# Leikurinn
DRIVES_PER_TEAM = 12    # hvort lið fær 12 sóknir í hverjum leik

# Field goal er reynt ef pos <= 37, þ.e. spark <= 55 yardar
# (sparkað er ~18 yördum aftar en boltinn: 37 + 18 = 55).
FG_MAX_POS = 37


# ---------------------------------------------------------------- hólfaskipting
# Yarda-dreifingin er metin sér fyrir hvert hólf (down, togo-hólf, svæði).

def togo_bucket(togo):
    """togo 1..10 fá hvert sitt hólf, 11-15 -> hólf 11, 16+ -> hólf 12."""
    if togo <= 10:
        return int(togo)
    return 11 if togo <= 15 else 12


def field_zone(pos):
    """0 = red zone (innan 20 yarda frá marki), 1 = annars staðar."""
    return 0 if pos <= 20 else 1


def down_group(down):
    """3rd og 4th down deila einni dreifingu (4th down tilraunir eru fáar)."""
    return 3 if down >= 3 else int(down)


# Þjálfaratafla: P(go / fg / punt) er metin í grófari hólfum.
# Bil eru hálfopin (a, b]; t.d. eru togo-hólfin 1, 2, 3, 4-5, 6-7, 8-10, 11-15, 16+.
COACH_TOGO_EDGES = [0, 1, 2, 3, 5, 7, 10, 15, 99]
COACH_POS_EDGES = list(range(0, 101, 10))
COACH_TOGO_LABELS = ["1", "2", "3", "4-5", "6-7", "8-10", "11-15", "16+"]
COACH_POS_LABELS = [f"{a + 1}-{b}" for a, b in zip(COACH_POS_EDGES, COACH_POS_EDGES[1:])]


def interval_index(value, edges):
    """Númer bilsins (edges[i], edges[i+1]] sem value lendir í."""
    for i in range(len(edges) - 1):
        if edges[i] < value <= edges[i + 1]:
            return i
    return len(edges) - 2
