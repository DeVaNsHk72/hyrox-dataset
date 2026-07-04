"""Shared helpers + dark visual theme for the Hyrox analysis suite.

Every figure follows the same recipe:
    fig, ax = plt.subplots(...)
    ...plot...
    hc.title(ax, "Main claim", "How to read this chart")      # single-axes
    hc.sup(fig, "Main claim", "How to read this figure")      # multi-axes
    hc.save(fig, "NN_name", takeaway="One-sentence insight.")
"""

import textwrap
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
RAW_PARQUET = DATASET / "hyrox_results.parquet"
CLEAN_PARQUET = DATASET / "hyrox_clean.parquet"

OUT = ROOT / "analysis" / "output"
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- theme
# Clean white theme: dark slate text, saturated accents that read on white.
BG = "#ffffff"
GRID = "#e2e8f0"
TEXT = "#000000"
DIM = "#000000"

ACCENT = "#0284c7"  # blue
VIOLET = "#7c3aed"
PINK = "#db2777"
GREEN = "#059669"
AMBER = "#d97706"
RED = "#dc2626"
CYCLE = [ACCENT, PINK, GREEN, AMBER, VIOLET, RED,
         "#0891b2", "#65a30d", "#ea580c", "#c026d3"]

SEX_COLORS = {"M": ACCENT, "W": PINK}
SEX_NAMES = {"M": "Men", "W": "Women"}

# ---------------------------------------------------------------- data
STATIONS = [
    ("ski_erg_seconds", "SkiErg"),
    ("sled_push_seconds", "Sled Push"),
    ("sled_pull_seconds", "Sled Pull"),
    ("burpee_broad_jump_seconds", "Burpee Jumps"),
    ("row_seconds", "Row"),
    ("farmers_carry_seconds", "Farmers Carry"),
    ("sandbag_lunges_seconds", "Lunges"),
    ("wall_balls_seconds", "Wall Balls"),
]
STATION_COLS = [c for c, _ in STATIONS]
STATION_LABELS = {c: l for c, l in STATIONS}
# Longer names where space allows (tables, horizontal bars).
STATION_LABELS_FULL = {
    "ski_erg_seconds": "1km SkiErg",
    "sled_push_seconds": "50m Sled Push",
    "sled_pull_seconds": "50m Sled Pull",
    "burpee_broad_jump_seconds": "80m Burpee Broad Jump",
    "row_seconds": "1km Row",
    "farmers_carry_seconds": "200m Farmers Carry",
    "sandbag_lunges_seconds": "100m Sandbag Lunges",
    "wall_balls_seconds": "Wall Balls",
}

RUN_COLS = [f"run_{i}_seconds" for i in range(1, 9)]

AGE_GROUPS = [
    "16-24", "25-29", "30-34", "35-39", "40-44", "45-49",
    "50-54", "55-59", "60-64", "65-69", "70-74", "75-79",
]

DIV_ORDER = ["OPEN", "PRO", "DOUBLES", "PRO_DOUBLES", "RELAY", "ELITE",
             "ELITE_DOUBLES", "ADAPTIVE", "YOUNGSTARS", "GORUCK", "OTHER"]
DIV_NAMES = {
    "OPEN": "Open", "PRO": "Pro", "DOUBLES": "Doubles",
    "PRO_DOUBLES": "Pro Doubles", "RELAY": "Relay", "ELITE": "Elite",
    "ELITE_DOUBLES": "Elite Doubles", "ADAPTIVE": "Adaptive",
    "YOUNGSTARS": "Youngstars", "GORUCK": "Goruck", "OTHER": "Other",
}

# Season n spans roughly (2016+n)/(2017+n); S3 was the COVID season.
SEASON_YEARS = {s: f"'{16 + s:02d}/{17 + s:02d}" for s in range(1, 10)}


def load(clean: bool = True) -> pl.LazyFrame:
    """Lazy scan of the deduplicated, enriched dataset."""
    return pl.scan_parquet(CLEAN_PARQUET if clean else RAW_PARQUET)


def singles(lf: pl.LazyFrame, pro: bool = True) -> pl.LazyFrame:
    """Individual (non-team) races with plausible complete timing."""
    divs = ["OPEN", "PRO"] if pro else ["OPEN"]
    return lf.filter(pl.col("division_canonical").is_in(divs) & pl.col("is_clean_race"))


# ---------------------------------------------------------------- formatting
def fmt_hms(seconds: float, _pos=None) -> str:
    if seconds is None:
        return ""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def fmt_k(v: float, _pos=None) -> str:
    return f"{v / 1000:.0f}k" if abs(v) >= 1000 else f"{v:.0f}"


HMS = FuncFormatter(fmt_hms)
KFMT = FuncFormatter(fmt_k)


# ---------------------------------------------------------------- style
def style() -> None:
    available = {f.name for f in fm.fontManager.ttflist}
    family = next((f for f in ["Inter", "SF Pro Display", "Helvetica Neue", "Arial"]
                   if f in available), "sans-serif")
    plt.rcParams.update({
        "font.family": family,
        "figure.dpi": 110,
        "savefig.dpi": 170,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.35,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": TEXT,
        "axes.labelcolor": DIM,
        "axes.labelsize": 10,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.55,
        "xtick.color": DIM,
        "ytick.color": DIM,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=CYCLE),
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": TEXT,
        "font.size": 10,
    })


def title(ax: plt.Axes, main: str, sub: str | None = None) -> None:
    """Bold left-aligned headline + dim explainer line.

    The subtitle is offset in points (not axes fraction) so the spacing is
    identical on every figure size and can never collide with the headline.
    """
    if sub:
        ax.set_title(main, loc="left", fontsize=13, fontweight="bold",
                     color=TEXT, pad=28)
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 9), textcoords="offset points",
                    fontsize=9.5, color=DIM, va="bottom", ha="left")
    else:
        ax.set_title(main, loc="left", fontsize=13, fontweight="bold",
                     color=TEXT, pad=12)


def sup(fig: plt.Figure, main: str, sub: str | None = None) -> None:
    """Figure-level headline for multi-panel figures. Call AFTER plotting."""
    fig.tight_layout(rect=(0, 0, 1, 0.90 if sub else 0.94))
    fig.suptitle(main, x=0.02, y=0.99, ha="left", va="top",
                 fontsize=14, fontweight="bold", color=TEXT)
    if sub:
        fig.text(0.02, 0.945, sub, ha="left", va="top", fontsize=9.5, color=DIM)


def season_axis(ax: plt.Axes, first: int = 1, last: int = 8) -> None:
    """Season ticks labelled S1..S9 with their years underneath."""
    seasons = list(range(first, last + 1))
    ax.set_xticks(seasons)
    ax.set_xticklabels([f"S{s}\n{SEASON_YEARS[s]}" for s in seasons], fontsize=8)
    ax.set_xlabel("")


def partial_season_note(ax: plt.Axes, season: int = 9) -> None:
    """Mark a season that is still in progress (data incomplete)."""
    import matplotlib.transforms as mtransforms
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(season, 0.97, "season in\nprogress", transform=trans, ha="center",
            va="top", fontsize=7.5, color=DIM, style="italic")


def covid_note(ax: plt.Axes, x: float = 3, y_frac: float = 0.5) -> None:
    ax.annotate("COVID\n(S3 cancelled)", xy=(x, ax.get_ylim()[0]),
                xytext=(x, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * y_frac),
                fontsize=8, color=DIM, ha="center", style="italic")


def save(fig: plt.Figure, name: str, takeaway: str | None = None) -> None:
    """Save figure; optional takeaway renders as a caption strip below the chart."""
    if takeaway:
        width = max(70, int(fig.get_figwidth() * 12))
        wrapped = "\n".join(textwrap.wrap("»  " + takeaway, width=width,
                                          subsequent_indent="    "))
        fig.text(0.02, -0.015, wrapped, ha="left", va="top",
                 fontsize=10, color=ACCENT, fontweight="medium")
    path = FIG / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  fig: {path.name}")


def table(df: pl.DataFrame, name: str) -> None:
    path = TAB / f"{name}.csv"
    df.write_csv(path)
    print(f"  tab: {path.name}")
