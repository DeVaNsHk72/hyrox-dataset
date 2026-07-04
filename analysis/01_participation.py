"""Participation & demographics: growth, geography, divisions, sex, age, nationality, world map."""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import hyrox_common as hc

# City → country mapping for Hyrox venues
CITY_COUNTRY = {
    "Abu Dhabi": "UAE", "Acapulco": "Mexico", "Amsterdam": "Netherlands",
    "Anaheim": "USA", "Atlanta": "USA", "Austin": "USA",
    "Barcelona": "Spain", "Berlin": "Germany", "Bilbao": "Spain",
    "Birmingham": "UK", "Bologna": "Italy", "Boston": "USA",
    "Brisbane": "Australia", "Brussels": "Belgium", "Budapest": "Hungary",
    "Buenos Aires": "Argentina", "Charlotte": "USA", "Chicago": "USA",
    "Cologne": "Germany", "Copenhagen": "Denmark", "Dallas": "USA",
    "Delhi": "India", "Denver": "USA", "Detroit": "USA",
    "Doha": "Qatar", "Dubai": "UAE", "Dublin": "Ireland",
    "Düsseldorf": "Germany", "Edinburgh": "UK", "Frankfurt": "Germany",
    "Gdańsk": "Poland", "Glasgow": "UK", "Gothenburg": "Sweden",
    "Guangzhou": "China", "Hamburg": "Germany", "Hannover": "Germany",
    "Helsinki": "Finland", "Hong Kong": "China", "Houston": "USA",
    "Hyrox": "Germany", "Indianapolis": "USA", "Istanbul": "Turkey",
    "Jacksonville": "USA", "Johannesburg": "South Africa",
    "Karlsruhe": "Germany", "Las Vegas": "USA", "Leipzig": "Germany",
    "Lisbon": "Portugal", "Liverpool": "UK", "London": "UK",
    "Los Angeles": "USA", "Lyon": "France", "Madrid": "Spain",
    "Malaga": "Spain", "Manchester": "UK", "Marseille": "France",
    "Melbourne": "Australia", "Mexico City": "Mexico", "Miami": "USA",
    "Milan": "Italy", "Minneapolis": "USA", "Monterrey": "Mexico",
    "Munich": "Germany", "Nashville": "USA", "New York": "USA",
    "Nice": "France", "Nuremberg": "Germany", "Orlando": "USA",
    "Oslo": "Norway", "Paris": "France", "Philadelphia": "USA",
    "Phoenix": "USA", "Portland": "USA", "Prague": "Czech Republic",
    "Rimini": "Italy", "Rio de Janeiro": "Brazil", "Riyadh": "Saudi Arabia",
    "Rome": "Italy", "Rotterdam": "Netherlands", "Salt Lake City": "USA",
    "San Diego": "USA", "San Francisco": "USA", "São Paulo": "Brazil",
    "Seattle": "USA", "Seoul": "South Korea", "Shanghai": "China",
    "Singapore": "Singapore", "Stockholm": "Sweden", "Stuttgart": "Germany",
    "Sydney": "Australia", "Taipei": "Taiwan", "Tampa": "USA",
    "Tel Aviv": "Israel", "Tokyo": "Japan", "Toronto": "Canada",
    "Toulouse": "France", "Turin": "Italy", "Valencia": "Spain",
    "Vancouver": "Canada", "Vienna": "Austria", "Warsaw": "Poland",
    "Washington DC": "USA", "Zurich": "Switzerland",
}

NATIONALITY_NAMES = {
    "GER": "Germany", "USA": "United States", "GBR": "United Kingdom",
    "NED": "Netherlands", "AUS": "Australia", "FRA": "France",
    "AUT": "Austria", "SUI": "Switzerland", "ITA": "Italy",
    "ESP": "Spain", "CAN": "Canada", "IRL": "Ireland",
    "BEL": "Belgium", "SWE": "Sweden", "DEN": "Denmark",
    "NOR": "Norway", "POL": "Poland", "BRA": "Brazil",
    "MEX": "Mexico", "JPN": "Japan", "KOR": "South Korea",
    "RSA": "South Africa", "ISR": "Israel", "POR": "Portugal",
    "CZE": "Czech Republic", "FIN": "Finland", "HUN": "Hungary",
    "NZL": "New Zealand", "SGP": "Singapore", "HKG": "Hong Kong",
    "IND": "India", "UAE": "UAE", "ARG": "Argentina",
    "COL": "Colombia", "CHI": "Chile", "TUR": "Turkey",
    "GRE": "Greece", "ROU": "Romania", "CHN": "China",
}


def _get_country(city: str) -> str:
    base = city.split(" - ")[0].strip()
    for prefix in ["APAC Championship ", "EMEA ", "NA Championship ",
                   "NA Champs ", "APAC Champs "]:
        if base.startswith(prefix):
            base = base.replace(prefix, "").strip()
    return CITY_COUNTRY.get(base, "Unknown")


def _nat_name(code: str) -> str:
    return NATIONALITY_NAMES.get(code, code)


def participants_per_year(lf: pl.LazyFrame) -> None:
    df = (
        lf.filter((pl.col("season") < 9) & pl.col("year").is_not_null())
        .group_by("year")
        .agg(
            pl.len().alias("participants"),
            pl.col("athlete_key").n_unique().alias("unique_athletes"),
        )
        .sort("year")
        .collect()
    )
    fig, ax = plt.subplots(figsize=(10.0, 8.0))
    x = df["year"].to_numpy()
    bars = ax.bar(x, df["participants"], color=hc.ACCENT, width=0.68)
    ax.bar_label(bars, labels=[hc.fmt_k(v) for v in df["participants"].to_list()],
                 fontsize=9, padding=3, color=hc.TEXT)
    ax.plot(x, df["unique_athletes"], marker="o", color=hc.AMBER, lw=2,
            markersize=5, label="of which unique athletes")
    ax.yaxis.set_major_formatter(hc.KFMT)
    ax.set_ylabel("Race results")
    ax.set_xticks(x)
    ax.legend(loc="upper left")
    hc.title(ax, "Hyrox has roughly doubled every year since COVID",
             "Bars = race results per calendar year · amber line = unique athletes behind them (S1–S8, S9 in progress excluded)")
    hc.save(fig, "01_participants_per_year",
            takeaway=f"From {hc.fmt_k(df['participants'][0])} results in {df['year'][0]} to "
                     f"{hc.fmt_k(df['participants'][-1])} in {df['year'][-1]}: "
                     "the 2020 COVID gap barely dented the growth curve.")
    hc.table(df, "01_participants_per_year")


def season_growth(lf: pl.LazyFrame) -> None:
    df = (
        lf.filter(pl.col("season") < 9)
        .group_by("season", "division_canonical").len()
        .collect()
        .pivot(on="division_canonical", index="season", values="len")
        .sort("season")
        .fill_null(0)
    )
    main_divs = ["OPEN", "DOUBLES", "PRO", "PRO_DOUBLES"]
    other = [d for d in df.columns if d not in main_divs + ["season"]]
    df = df.with_columns(pl.sum_horizontal([pl.col(c) for c in other]).alias("OTHER_ALL"))

    colors = {"OPEN": hc.ACCENT, "DOUBLES": hc.GREEN, "PRO": hc.VIOLET,
              "PRO_DOUBLES": hc.PINK, "OTHER_ALL": "#94a3b8"}
    names = {"OPEN": "Open (solo)", "DOUBLES": "Doubles", "PRO": "Pro (solo)",
             "PRO_DOUBLES": "Pro Doubles", "OTHER_ALL": "Relay & other"}

    fig, ax = plt.subplots(figsize=(11.0, 8.7))
    bottom = np.zeros(len(df))
    for d in ["OPEN", "DOUBLES", "PRO", "PRO_DOUBLES", "OTHER_ALL"]:
        vals = df[d].to_numpy()
        ax.bar(df["season"], vals, bottom=bottom, label=names[d], color=colors[d],
               width=0.72)
        bottom += vals
    for s, tot in zip(df["season"], bottom):
        ax.text(s, tot + bottom.max() * 0.015, hc.fmt_k(tot), ha="center",
                fontsize=9, color=hc.TEXT, fontweight="bold")
    ax.text(3, bottom.max() * 0.08, "COVID", ha="center", fontsize=8.5,
            color=hc.DIM, style="italic")
    hc.season_axis(ax)
    ax.yaxis.set_major_formatter(hc.KFMT)
    ax.set_ylabel("Race results")
    ax.legend(loc="upper left", ncols=2)
    ax.set_ylim(0, bottom.max() * 1.09)
    hc.title(ax, "Doubles has overtaken solo racing",
             "Race results per season, stacked by division · label on top = season total")
    share_d = (df["DOUBLES"][-1] + df["PRO_DOUBLES"][-1]) / bottom[-1] * 100
    dbl8 = df["DOUBLES"][-1]
    open8 = df["OPEN"][-1]
    hc.save(fig, "01_participation_by_season_division",
            takeaway=f"In Season 8, Doubles ({hc.fmt_k(dbl8)}) passed Open ({hc.fmt_k(open8)}), and "
                     f"partnered formats now make up {share_d:.0f}% of all results.")

    summary = (
        lf.filter(pl.col("season") < 9)
        .group_by("season")
        .agg(
            pl.len().alias("results"),
            pl.col("athlete_key").n_unique().alias("unique_athletes"),
            pl.col("city").n_unique().alias("cities"),
            pl.struct("city", "event_id").n_unique().alias("race_days"),
            (pl.col("sex") == "W").mean().round(3).alias("share_women"),
        )
        .sort("season")
        .collect()
    )
    hc.table(summary, "01_season_summary")


def events_and_cities(lf: pl.LazyFrame) -> None:
    df = (
        lf.filter(pl.col("season") < 9)
        .group_by("season")
        .agg(pl.col("city").n_unique().alias("cities"))
        .sort("season")
        .collect()
    )
    fig, ax = plt.subplots(figsize=(9.0, 7.2))
    ax.plot(df["season"], df["cities"], marker="o", color=hc.ACCENT, lw=2.2,
            markersize=6)
    for s, c in zip(df["season"], df["cities"]):
        ax.annotate(str(c), (s, c), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color=hc.TEXT)
    hc.season_axis(ax)
    ax.set_ylabel("Host cities")
    ax.set_ylim(0, df["cities"].max() * 1.18)
    hc.title(ax, "The race calendar exploded after COVID",
             "Number of distinct host cities per season")
    hc.save(fig, "01_cities_per_season",
            takeaway=f"From {df['cities'][0]} cities in Season 1 to {df['cities'][-1]} in Season 8.")

    top = (
        lf.filter(pl.col("season") < 9)
        .group_by("city").len().sort("len", descending=True).head(25).collect()
    )
    cities_with_country = [f"{c}  ·  {_get_country(c)}" for c in top["city"].to_list()]
    fig, ax = plt.subplots(figsize=(10.0, 12.3))
    bars = ax.barh(cities_with_country[::-1], top["len"].to_list()[::-1], color=hc.ACCENT)
    bars[-1].set_color(hc.AMBER)  # highlight #1
    ax.bar_label(bars, labels=[hc.fmt_k(v) for v in top["len"].to_list()[::-1]],
                 fontsize=8.5, padding=4, color=hc.TEXT)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlim(0, top["len"].max() * 1.1)
    ax.xaxis.set_major_formatter(hc.KFMT)
    ax.grid(axis="y", visible=False)
    hc.title(ax, "Where Hyrox lives: the 25 biggest host cities",
             "All-time race results per city, all divisions")
    hc.save(fig, "01_top_cities",
            takeaway=f"{top['city'][0]} is the capital of Hyrox with {hc.fmt_k(top['len'][0])} "
                     f"all-time results: ahead of {top['city'][1]} and {top['city'][2]}.")
    hc.table(top, "01_top_cities")


def sex_split(lf: pl.LazyFrame) -> None:
    df = (
        lf.filter(pl.col("sex").is_in(["M", "W"]) & (pl.col("season") < 9))
        .group_by("season", "sex").len()
        .collect()
        .pivot(on="sex", index="season", values="len")
        .sort("season")
        .with_columns((pl.col("W") / (pl.col("M") + pl.col("W"))).alias("share_w"))
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 6.5))
    ax1.plot(df["season"], df["M"], marker="o", label="Men", color=hc.SEX_COLORS["M"], lw=2)
    ax1.plot(df["season"], df["W"], marker="o", label="Women", color=hc.SEX_COLORS["W"], lw=2)
    ax1.yaxis.set_major_formatter(hc.KFMT)
    ax1.set_ylabel("Race results")
    ax1.legend(loc="upper left")
    hc.season_axis(ax1)
    hc.title(ax1, "Results by sex", "Race results per season")

    ax2.plot(df["season"], df["share_w"] * 100, marker="o", color=hc.SEX_COLORS["W"], lw=2)
    for s, v in zip(df["season"], df["share_w"]):
        ax2.annotate(f"{v * 100:.0f}%", (s, v * 100), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8.5, color=hc.TEXT)
    ax2.set_ylim(0, 60)
    ax2.axhline(50, color=hc.DIM, lw=1, ls="--")
    ax2.text(1.1, 51.5, "parity", fontsize=8, color=hc.DIM)
    hc.season_axis(ax2)
    hc.title(ax2, "Share of women", "% of all results")
    fig.tight_layout()
    hc.save(fig, "01_sex_split",
            takeaway=f"Women's share has climbed to {df['share_w'][-1] * 100:.0f}% and keeps rising: "
                     "Hyrox is closing in on a 50/50 field.")


def nationalities(lf: pl.LazyFrame) -> None:
    nat = lf.filter(pl.col("nationality").is_not_null() & (pl.col("season") < 9))
    top = nat.group_by("nationality").len().sort("len", descending=True).head(20).collect()
    labels = [_nat_name(c) for c in top["nationality"].to_list()]
    fig, ax = plt.subplots(figsize=(10.0, 10.9))
    bars = ax.barh(labels[::-1], top["len"].to_list()[::-1], color=hc.VIOLET)
    bars[-1].set_color(hc.AMBER)
    ax.bar_label(bars, labels=[hc.fmt_k(v) for v in top["len"].to_list()[::-1]],
                 fontsize=8.5, padding=4, color=hc.TEXT)
    ax.set_xlim(0, top["len"].max() * 1.1)
    ax.xaxis.set_major_formatter(hc.KFMT)
    ax.grid(axis="y", visible=False)
    hc.title(ax, "The 20 biggest Hyrox nations",
             "All-time race results by athlete nationality (where recorded)")
    hc.save(fig, "01_top_nationalities",
            takeaway=f"{_nat_name(top['nationality'][0])} leads with {hc.fmt_k(top['len'][0])} results, "
                     f"followed by {_nat_name(top['nationality'][1])} and {_nat_name(top['nationality'][2])}.")
    hc.table(top, "01_top_nationalities")

    top8 = top["nationality"].head(8).to_list()
    trend = (
        nat.filter(pl.col("nationality").is_in(top8))
        .group_by("season", "nationality").len()
        .collect()
        .pivot(on="nationality", index="season", values="len")
        .sort("season")
        .fill_null(0)
    )
    fig, ax = plt.subplots(figsize=(11.0, 8.7))
    for i, n in enumerate(top8):
        color = hc.CYCLE[i % len(hc.CYCLE)]
        ax.plot(trend["season"], trend[n], marker="o", color=color, lw=1.8, markersize=4)
        ax.annotate(_nat_name(n), (trend["season"][-1], trend[n][-1]),
                    textcoords="offset points", xytext=(8, 0), fontsize=8.5,
                    color=color, va="center", fontweight="bold")
    hc.season_axis(ax)
    ax.yaxis.set_major_formatter(hc.KFMT)
    ax.set_ylabel("Race results")
    ax.set_xlim(0.5, 9.6)
    hc.title(ax, "Germany started it: the UK and US now drive the boom",
             "Results per season for the 8 biggest nations · labels at line ends")
    hc.save(fig, "01_nationality_growth")


def age_distribution(lf: pl.LazyFrame) -> None:
    df = (
        lf.filter(pl.col("age_group").is_in(hc.AGE_GROUPS) & pl.col("sex").is_in(["M", "W"])
                  & (pl.col("season") < 9))
        .group_by("age_group", "sex").len()
        .collect()
        .pivot(on="sex", index="age_group", values="len")
        .sort("age_group")
        .fill_null(0)
    )
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(11.0, 8.0))
    ax.bar(x - 0.2, df["M"], width=0.4, label="Men", color=hc.SEX_COLORS["M"])
    ax.bar(x + 0.2, df["W"], width=0.4, label="Women", color=hc.SEX_COLORS["W"])
    ax.set_xticks(x, df["age_group"], rotation=0, fontsize=8.5)
    ax.yaxis.set_major_formatter(hc.KFMT)
    ax.set_ylabel("Race results")
    ax.set_xlabel("Age group")
    ax.legend()
    peak = df.with_columns((pl.col("M") + pl.col("W")).alias("t")).sort("t", descending=True)
    hc.title(ax, "Hyrox peaks at 30–34: but the 40+ field is huge",
             "All-time results by age group and sex (standard age brackets only)")
    hc.save(fig, "01_age_distribution",
            takeaway=f"The {peak['age_group'][0]} bracket is the biggest, and athletes 40+ make up "
                     f"{df.filter(pl.col('age_group').is_in(['40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79'])).select(pl.sum_horizontal('M','W').sum()).item() / df.select(pl.sum_horizontal('M','W').sum()).item() * 100:.0f}% of the field.")


def division_growth_rate(lf: pl.LazyFrame) -> None:
    df = (
        lf.filter(pl.col("season") < 9)
        .group_by("season", "division_canonical").len()
        .collect()
        .pivot(on="division_canonical", index="season", values="len")
        .sort("season")
        .fill_null(0)
    )
    divs = [d for d in ["OPEN", "PRO", "DOUBLES", "PRO_DOUBLES"] if d in df.columns]
    colors = {"OPEN": hc.ACCENT, "DOUBLES": hc.GREEN, "PRO": hc.VIOLET, "PRO_DOUBLES": hc.PINK}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 7.2))
    for d in divs:
        vals = df[d].to_numpy().astype(float)
        ax1.plot(df["season"], vals, marker="o", label=hc.DIV_NAMES[d],
                 color=colors[d], lw=2, markersize=4)
    ax1.set_ylabel("Race results")
    ax1.yaxis.set_major_formatter(hc.KFMT)
    ax1.legend(loc="upper left")
    hc.season_axis(ax1)
    hc.title(ax1, "Division size", "Results per season")

    # Growth rates only from S5 onward: the COVID collapse and rebound (S3-S4)
    # produce meaningless four-digit percentages that distort the whole panel.
    for d in divs:
        vals = df[d].to_numpy().astype(float)
        seasons = df["season"].to_numpy()
        growth, valid = [], []
        for i in range(1, len(vals)):
            if seasons[i] >= 5 and vals[i - 1] > 100:
                growth.append((vals[i] - vals[i - 1]) / vals[i - 1] * 100)
                valid.append(seasons[i])
        if growth:
            ax2.plot(valid, growth, marker="o", label=hc.DIV_NAMES[d],
                     color=colors[d], lw=2, markersize=4)
    ax2.axhline(0, color=hc.DIM, lw=1)
    ax2.set_ylabel("Growth vs previous season (%)")
    hc.season_axis(ax2, first=5)
    hc.title(ax2, "Season-over-season growth", "From S5 onward; the COVID rebound (S3-S4) is excluded as it distorts the scale")
    hc.sup(fig, "Every division is still growing: Doubles fastest of all",
           "Left: absolute size per season · Right: relative growth rate")
    hc.save(fig, "01_division_growth_rate")


AGE_BANDS = {
    "16-24": "16-24", "25-29": "25-34", "30-34": "25-34",
    "35-39": "35-44", "40-44": "35-44",
    "45-49": "45+", "50-54": "45+", "55-59": "45+", "60-64": "45+",
    "65-69": "45+", "70-74": "45+", "75-79": "45+",
}
BAND_ORDER = ["16-24", "25-34", "35-44", "45+"]
BAND_COLORS = {"16-24": hc.GREEN, "25-34": hc.ACCENT, "35-44": hc.VIOLET, "45+": hc.AMBER}


def age_growth_by_season(lf: pl.LazyFrame) -> None:
    """How the age mix of the field shifts as new (often younger) markets join."""
    df = (
        lf.filter(pl.col("age_group").is_in(hc.AGE_GROUPS)
                  & pl.col("season").is_between(4, 8))
        .with_columns(pl.col("age_group").replace_strict(AGE_BANDS).alias("band"))
        .group_by("season", "band").len()
        .collect()
        .pivot(on="band", index="season", values="len")
        .sort("season")
        .fill_null(0)
    )
    totals = df.select(pl.sum_horizontal(BAND_ORDER)).to_series()
    fig, ax = plt.subplots(figsize=(10.5, 8.4))
    for band in BAND_ORDER:
        share = (df[band] / totals * 100).to_numpy()
        ax.plot(df["season"], share, marker="o", color=BAND_COLORS[band],
                lw=2.2, markersize=5)
        ax.annotate(band, (df["season"][-1], share[-1]),
                    textcoords="offset points", xytext=(10, 0), va="center",
                    fontsize=9.5, color=BAND_COLORS[band], fontweight="bold")
    hc.season_axis(ax, first=4)
    ax.set_xlim(3.6, 8.9)
    ax.set_ylabel("Share of age-known results (%)")
    hc.title(ax, "The age mix is stable, with a slight tilt toward the young",
             "Share of each season's results by age band (standard brackets only, S4-S8)")
    hc.save(fig, "01_age_growth_by_season",
            takeaway="25-34 remains the core of the sport in every season; the newer, "
                     "younger markets have not yet shifted the global mix dramatically.")
    hc.table(df, "01_age_band_by_season")


EMERGING = [("MEX", "Mexico"), ("CHN", "China"), ("IND", "India"),
            ("RSA", "South Africa"), ("BRA", "Brazil")]


def emerging_markets(lf: pl.LazyFrame) -> None:
    df = (
        lf.filter(pl.col("nationality").is_in([c for c, _ in EMERGING])
                  & pl.col("season").is_between(5, 8))
        .group_by("season", "nationality").len()
        .collect()
        .pivot(on="nationality", index="season", values="len")
        .sort("season")
        .fill_null(0)
    )
    fig, ax = plt.subplots(figsize=(10.5, 8.7))
    # Stagger end labels vertically: on a log scale India (5,861) and South
    # Africa (5,456) land almost on top of each other, so nudge labels apart
    # in point-space based on their rank order at the final season.
    finals = sorted(EMERGING, key=lambda cn: df[cn[0]][-1], reverse=True)
    offsets = {}
    prev_log = None
    shift = 0.0
    for code, _ in finals:
        import math
        cur_log = math.log10(max(df[code][-1], 1))
        if prev_log is not None and prev_log - cur_log < 0.12:
            shift -= 11  # push this label further down than the one above
        else:
            shift = 0.0
        offsets[code] = shift
        prev_log = cur_log
    for i, (code, name) in enumerate(EMERGING):
        ax.plot(df["season"], df[code], marker="o", color=hc.CYCLE[i],
                lw=2.2, markersize=5)
        ax.annotate(name, (df["season"][-1], df[code][-1]),
                    textcoords="offset points", xytext=(10, offsets[code]),
                    va="center", fontsize=9.5, color=hc.CYCLE[i],
                    fontweight="bold")
    ax.set_yscale("log")
    hc.season_axis(ax, first=5)
    ax.set_xlim(4.6, 9.1)
    ax.set_ylabel("Results per season (log scale)")
    hc.title(ax, "The next wave: emerging Hyrox nations",
             "Results by athlete nationality per season, log scale · note the 10x-per-season slopes")
    hc.save(fig, "01_emerging_markets",
            takeaway="India went from 100 results in S6 to 5,861 in S8, China from 65 to 8,263. "
                     "On these slopes the next Hyrox superpowers are already visible.")
    hc.table(df, "01_emerging_markets")


def city_season_heatmap(lf: pl.LazyFrame) -> None:
    top_cities = (
        lf.filter(pl.col("season") < 9)
        .group_by("city").len().sort("len", descending=True).head(20).collect()
    )["city"].to_list()
    df = (
        lf.filter(pl.col("city").is_in(top_cities) & (pl.col("season") < 9))
        .group_by("city", "season").len()
        .collect()
        .pivot(on="season", index="city", values="len")
        .fill_null(0)
    )
    order = [c for c in top_cities if c in df["city"].to_list()]
    df = df.with_columns(pl.col("city").cast(pl.Enum(order))).sort("city")
    seasons = sorted([c for c in df.columns if c != "city"], key=int)
    cities = [f"{c}  ·  {_get_country(c)}" for c in df["city"].to_list()]
    mat = df.select(seasons).to_numpy()
    fig, ax = plt.subplots(figsize=(11.0, 11.6))
    im = ax.imshow(mat, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(seasons)),
                  [f"S{s}\n{hc.SEASON_YEARS[int(s)]}" for s in seasons], fontsize=8)
    ax.set_yticks(range(len(cities)), cities, fontsize=8.5)
    ax.grid(visible=False)
    for i in range(len(cities)):
        for j in range(len(seasons)):
            v = mat[i, j]
            if v > 0:
                ax.text(j, i, hc.fmt_k(v), ha="center", va="center", fontsize=6.5,
                        color="white" if v < mat.max() * 0.6 else "black")
    cb = fig.colorbar(im, shrink=0.7)
    cb.set_label("Results", color=hc.DIM)
    cb.ax.yaxis.set_tick_params(color=hc.DIM)
    plt.setp(cb.ax.get_yticklabels(), color=hc.DIM)
    hc.title(ax, "When each big city joined the calendar",
             "Results per season for the top 20 cities · dark = none/few, bright = thousands")
    hc.save(fig, "01_city_season_heatmap",
            takeaway="Only Hamburg and London predate COVID at scale: most of today's biggest venues "
                     "joined in Seasons 5–7.")


def world_map(lf: pl.LazyFrame) -> None:
    import plotly.express as px
    import pycountry

    cities = (
        lf.filter(pl.col("season") < 9)
        .group_by("city").len().sort("len", descending=True).collect()
    )
    country_df: dict[str, int] = {}
    for r in cities.iter_rows(named=True):
        c = _get_country(r["city"])
        country_df[c] = country_df.get(c, 0) + r["len"]

    iso_map = {}
    for c in country_df:
        try:
            iso_map[c] = pycountry.countries.search_fuzzy(c)[0].alpha_3
        except LookupError:
            pass

    data = [{"country": c, "iso": iso_map[c], "results": t}
            for c, t in country_df.items() if c in iso_map]
    pdf = pl.DataFrame(data)
    fig = px.choropleth(
        pdf.to_pandas(),
        locations="iso",
        color="results",
        hover_name="country",
        color_continuous_scale=["#e0f2fe", "#0284c7", "#d97706"],
    )
    fig.update_layout(
        title=dict(text="<b>Hyrox around the world</b><br>"
                        "<span style='font-size:12px;color:#64748b'>All-time race results by host country</span>",
                   font=dict(size=17, color="#1e293b"), x=0.02),
        geo=dict(showframe=False, showcoastlines=False, showland=True,
                 landcolor="#e2e8f0", bgcolor="rgba(0,0,0,0)",
                 projection_type="natural earth"),
        paper_bgcolor="#ffffff",
        font=dict(size=11, color="#1e293b"),
        coloraxis_colorbar=dict(title="Results", tickfont=dict(color="#64748b")),
        margin=dict(l=0, r=0, t=70, b=0),
        width=1150, height=560,
    )
    fig.write_image(str(hc.FIG / "01_world_map.png"), scale=2)
    print("  fig: 01_world_map.png")


def main() -> None:
    hc.style()
    lf = hc.load()
    print("[01] participation & demographics")
    participants_per_year(lf)
    season_growth(lf)
    events_and_cities(lf)
    sex_split(lf)
    nationalities(lf)
    age_distribution(lf)
    age_growth_by_season(lf)
    emerging_markets(lf)
    division_growth_rate(lf)
    city_season_heatmap(lf)
    world_map(lf)


if __name__ == "__main__":
    main()
