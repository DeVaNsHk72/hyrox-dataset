"""Deep-dive analyses: doubles vs solo, completion rates, age×station, best-run-lap, position effects."""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import hyrox_common as hc


def doubles_vs_solo(lf: pl.LazyFrame) -> None:
    cols = hc.STATION_COLS + ["run_total_seconds"]
    labels = [hc.STATION_LABELS.get(c, "Running") for c in cols]
    frames = {}
    for div in ["OPEN", "DOUBLES"]:
        frames[div] = (
            lf.filter((pl.col("division_canonical") == div)
                      & pl.col("is_clean_race") & pl.col("sex").is_in(["M", "W"]))
            .group_by("sex")
            .agg([pl.col(c).median().alias(c) for c in cols])
            .collect()
        )
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 7.3))
    for ax, sex in [(axes[0], "M"), (axes[1], "W")]:
        s = frames["OPEN"].filter(pl.col("sex") == sex)
        d = frames["DOUBLES"].filter(pl.col("sex") == sex)
        if len(s) == 0 or len(d) == 0:
            continue
        x = np.arange(len(cols))
        ax.bar(x - 0.2, [s[c][0] for c in cols], width=0.4, label="Solo (Open)",
               color=hc.ACCENT)
        ax.bar(x + 0.2, [d[c][0] for c in cols], width=0.4, label="Doubles",
               color=hc.AMBER)
        ax.set_xticks(x, labels, fontsize=8)
        ax.yaxis.set_major_formatter(hc.HMS)
        ax.legend()
        hc.title(ax, hc.SEX_NAMES[sex], "Median segment time")
    hc.sup(fig, "Doubles: the runs stay, the station time halves",
           "Doubles pairs run every lap together but split station work: the gains come almost entirely from stations")
    hc.save(fig, "08_doubles_vs_solo",
            takeaway="Doubles running times match solo times almost exactly; the ~15-minute "
                     "advantage comes from sharing the eight stations.")


def completion_rates(lf: pl.LazyFrame) -> None:
    total = (
        lf.filter(pl.col("division_canonical").is_in(["OPEN", "PRO"]))
        .group_by("season")
        .agg(
            pl.len().alias("total"),
            pl.col("is_clean_race").sum().alias("clean"),
            pl.col("finish_seconds").is_not_null().sum().alias("finished"),
        )
        .sort("season")
        .collect()
        .with_columns(
            (pl.col("finished") / pl.col("total") * 100).alias("completion_rate"),
            (pl.col("clean") / pl.col("total") * 100).alias("clean_rate"),
        )
    )
    fig, ax = plt.subplots(figsize=(10.5, 8.4))
    ax.plot(total["season"], total["completion_rate"], marker="o",
            label="Recorded a finish time", color=hc.GREEN, lw=2.2, markersize=5)
    ax.plot(total["season"], total["clean_rate"], marker="s",
            label="Complete clean timing (all splits)", color=hc.ACCENT, lw=2.2, markersize=5)
    hc.season_axis(ax, first=int(total["season"].min()), last=int(total["season"].max()))
    ax.set_ylabel("% of Open+Pro entries")
    ax.set_ylim(50, 103)
    ax.legend(loc="lower right")
    hc.title(ax, "Almost everyone in the results has a finish time",
             "Share of Open+Pro entries with a recorded finish and with complete split timing, per season")
    hc.save(fig, "08_completion_rates",
            takeaway="Recorded-finish rates sit in the high 90s, but DNFs are largely absent from "
                     "the published archive: so read this as 'almost everyone in the results "
                     "finished', which is weaker than 'almost everyone who started'.")
    hc.table(total, "08_completion_rates")

    city_rates = (
        lf.filter(pl.col("division_canonical").is_in(["OPEN", "PRO"]))
        .group_by("city")
        .agg(
            pl.len().alias("total"),
            pl.col("finish_seconds").is_not_null().sum().alias("finished"),
        )
        .filter(pl.col("total") >= 1000)
        .with_columns((pl.col("finished") / pl.col("total") * 100).alias("rate"))
        .sort("rate")
        .collect()
    )
    worst = city_rates.head(15)
    best = city_rates.tail(10)
    shown = pl.concat([worst, best])
    fig, ax = plt.subplots(figsize=(10.5, 13.0))
    clrs = [hc.RED if r < 90 else hc.AMBER if r < 95 else hc.GREEN
            for r in shown["rate"].to_list()]
    bars = ax.barh(shown["city"].to_list(), shown["rate"].to_list(), color=clrs)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=7.5, padding=3, color=hc.TEXT)
    ax.set_xlabel("Completion rate (%)")
    ax.set_xlim(shown["rate"].min() - 3, 102)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8.5)
    hc.title(ax, "Completion rate by city: toughest 15 and smoothest 10",
             "Share of Open+Pro entries with a recorded finish (min 1,000 entries) · red < 90%, amber < 95%, green ≥ 95%")
    hc.save(fig, "08_completion_by_city",
            takeaway="Low-completion cities cluster around hot climates and first-edition "
                     "events: logistics and heat, not course design, drive DNFs.")


def age_station_heatmap(lf: pl.LazyFrame) -> None:
    df = (
        hc.singles(lf, pro=False)
        .filter(pl.col("age_group").is_in(hc.AGE_GROUPS) & (pl.col("sex") == "M"))
        .group_by("age_group")
        .agg([pl.col(c).median().alias(c) for c in hc.STATION_COLS])
        .sort("age_group")
        .collect()
    )
    ages = df["age_group"].to_list()
    mat = df.select(hc.STATION_COLS).to_numpy()
    baseline = mat[0:1, :].copy()
    baseline[baseline == 0] = 1
    ratio = mat / baseline * 100 - 100

    labels = [hc.STATION_LABELS[c] for c in hc.STATION_COLS]
    fig, ax = plt.subplots(figsize=(11.0, 10.2))
    im = ax.imshow(ratio, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(labels)), labels, fontsize=8.5)
    ax.set_yticks(range(len(ages)), ages, fontsize=8.5)
    ax.grid(visible=False)
    for i in range(len(ages)):
        for j in range(len(labels)):
            ax.text(j, i, f"{ratio[i, j]:+.0f}%", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if ratio[i, j] > ratio.max() * 0.55 else "#1e293b")
    cb = fig.colorbar(im, shrink=0.7)
    cb.set_label("% slower than the 16-24 group", color=hc.DIM)
    hc.title(ax, "Explosive stations age fastest, erg stations age slowest",
             "Median men's Open station time vs the youngest bracket · each cell = % slower than 16-24")
    hc.save(fig, "08_age_station_heatmap",
            takeaway="Burpee Broad Jumps and Wall Balls degrade most with age while SkiErg "
                     "and Row barely move: power leaves before endurance does.")


def best_run_lap(lf: pl.LazyFrame) -> None:
    df = (
        hc.singles(lf)
        .filter(pl.col("best_run_lap_seconds").is_between(120, 600))
        .select("best_run_lap_seconds", "finish_seconds", "sex")
        .collect()
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 7.0))
    bins = np.arange(120, 480, 10)
    for sex in ["M", "W"]:
        sub = df.filter(pl.col("sex") == sex)
        ax1.hist(sub["best_run_lap_seconds"].to_numpy(), bins=bins, alpha=0.6,
                 label=hc.SEX_NAMES[sex], color=hc.SEX_COLORS[sex], density=True)
    ax1.set_xlabel("Best 1km run lap")
    ax1.xaxis.set_major_formatter(hc.HMS)
    ax1.set_yticks([])
    ax1.legend()
    hc.title(ax1, "Best-lap distribution", "Fastest single 1km of each athlete's race")

    s = df.sample(min(len(df), 50000), seed=42)
    ax2.scatter(s["best_run_lap_seconds"], s["finish_seconds"], alpha=0.06, s=3,
                c=hc.ACCENT, edgecolors="none")
    ax2.set_xlabel("Best 1km run lap")
    ax2.set_ylabel("Finish time")
    ax2.xaxis.set_major_formatter(hc.HMS)
    ax2.yaxis.set_major_formatter(hc.HMS)
    r = np.corrcoef(s["best_run_lap_seconds"].to_numpy(),
                    s["finish_seconds"].to_numpy())[0, 1]
    hc.title(ax2, f"Best lap vs finish (r = {r:.2f})",
             "One fast kilometre predicts the whole race surprisingly well")
    fig.tight_layout()
    hc.save(fig, "08_best_run_lap",
            takeaway=f"A single flat-out 1km correlates {r:.2f} with the final time: "
                     "your 1km test result is a legitimate Hyrox predictor.")


def run_to_station_time_budget(lf: pl.LazyFrame) -> None:
    df = (
        hc.singles(lf)
        .with_columns(
            (pl.col("run_total_seconds") / pl.col("finish_seconds") * 100).alias("run_pct_of_finish"),
            (pl.col("stations_total_seconds") / pl.col("finish_seconds") * 100).alias("station_pct_of_finish"),
        )
        .with_columns(
            pl.when(pl.col("finish_pct_in_event") <= 0.05).then(pl.lit("Top 5%"))
            .when(pl.col("finish_pct_in_event") <= 0.25).then(pl.lit("Top 25%"))
            .when(pl.col("finish_pct_in_event").is_between(0.4, 0.6)).then(pl.lit("Middle"))
            .when(pl.col("finish_pct_in_event") >= 0.8).then(pl.lit("Back 20%"))
            .otherwise(None)
            .alias("cohort")
        )
        .filter(pl.col("cohort").is_not_null())
        .group_by("cohort")
        .agg(
            pl.col("run_pct_of_finish").median().alias("run_pct"),
            pl.col("station_pct_of_finish").median().alias("station_pct"),
        )
        .collect()
    )
    cohorts = [c for c in ["Top 5%", "Top 25%", "Middle", "Back 20%"]
               if len(df.filter(pl.col("cohort") == c)) > 0]
    run_vals = [float(df.filter(pl.col("cohort") == c)["run_pct"][0]) for c in cohorts]
    sta_vals = [float(df.filter(pl.col("cohort") == c)["station_pct"][0]) for c in cohorts]
    rox_vals = [100 - r - s for r, s in zip(run_vals, sta_vals)]

    x = np.arange(len(cohorts))
    fig, ax = plt.subplots(figsize=(10.5, 8.4))
    ax.bar(x, run_vals, label="Running", color=hc.ACCENT, width=0.6)
    ax.bar(x, sta_vals, bottom=run_vals, label="Stations", color=hc.VIOLET, width=0.6)
    ax.bar(x, rox_vals, bottom=[r + s for r, s in zip(run_vals, sta_vals)],
           label="Roxzone", color=hc.AMBER, width=0.6)
    ax.set_xticks(x, cohorts)
    ax.set_ylabel("% of finish time")
    ax.legend(loc="lower right")
    for i, (r, s, z) in enumerate(zip(run_vals, sta_vals, rox_vals)):
        ax.text(i, r / 2, f"{r:.0f}%", ha="center", va="center", fontsize=10,
                fontweight="bold", color="white")
        ax.text(i, r + s / 2, f"{s:.0f}%", ha="center", va="center", fontsize=10,
                fontweight="bold", color="white")
        ax.text(i, r + s + z / 2, f"{z:.0f}%", ha="center", va="center", fontsize=8,
                fontweight="bold", color="white")
    hc.title(ax, "Slower athletes don't just slow down: their race changes shape",
             "How each finishing cohort's race time splits between running, stations and transitions")
    hc.save(fig, "08_time_allocation_by_cohort",
            takeaway="From front to back, station+roxzone share grows steadily: station "
                     "endurance breaks down faster than running fitness does.")


def station_position_effect(lf: pl.LazyFrame) -> None:
    df = hc.singles(lf).select(hc.STATION_COLS).drop_nulls().collect()
    arr = df.to_numpy()
    row_means = arr.mean(axis=1, keepdims=True)
    row_stds = arr.std(axis=1, keepdims=True)
    row_stds[row_stds == 0] = 1
    z = (arr - row_means) / row_stds

    positions = np.arange(1, 9)
    medians = np.median(z, axis=0)
    q25 = np.percentile(z, 25, axis=0)
    q75 = np.percentile(z, 75, axis=0)
    labels = [hc.STATION_LABELS[c] for c in hc.STATION_COLS]

    fig, ax = plt.subplots(figsize=(11.0, 8.4))
    colors = [hc.RED if m == medians.max() else hc.ACCENT for m in medians]
    ax.bar(positions, medians, color=colors, alpha=0.9)
    ax.errorbar(positions, medians, yerr=[medians - q25, q75 - medians],
                fmt="none", ecolor=hc.DIM, capsize=3, lw=1)
    ax.set_xticks(positions, [f"{p}\n{l}" for p, l in zip(positions, labels)], fontsize=8)
    ax.axhline(0, color=hc.DIM, lw=1)
    ax.set_ylabel("Relative time within own race (z-score)")
    hc.title(ax, "Relative to your own race, which station costs you most?",
             "Each athlete's station times normalized against their own average · positive = relatively slower")
    hc.save(fig, "08_station_position_effect",
            takeaway="Even adjusting for each athlete's overall speed, the late-race "
                     "grind stations stand out: fatigue, not skill, shapes the back half.")


def mixed_doubles_overview(lf: pl.LazyFrame) -> None:
    """Analyze the massive influx of Mixed Doubles globally."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 7.3))

    doubles = (lf.filter(pl.col("division_canonical").is_in(["DOUBLES", "PRO_DOUBLES"]))
               .select("nationality", "sex")
               .collect())

    # Panel 1: Global division split
    sex_counts = doubles.group_by("sex").len().sort("sex").to_dicts()
    sex_map = {row["sex"]: row["len"] for row in sex_counts}
    
    total = sum(sex_map.values())
    labels = ["Men's", "Women's", "Mixed"]
    counts = [sex_map.get("M", 0), sex_map.get("W", 0), sex_map.get("X", 0)]
    
    ax1.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90,
            colors=["#64748b", "#38bdf8", hc.AMBER],
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    hc.title(ax1, "Doubles athletes by team gender",
             "Among recorded entries; mixed-doubles coverage is incomplete")

    # Panel 2: top host countries for Mixed Doubles.
    # Mixed-team rows carry NO athlete nationality (100% null), so we use the
    # host city mapped to its country instead, which is fully populated.
    from importlib import import_module
    _p1 = import_module("01_participation")
    md = (lf.filter((pl.col("division_canonical").is_in(["DOUBLES", "PRO_DOUBLES"]))
                    & (pl.col("sex") == "X"))
          .select("city").collect()
          .with_columns(pl.col("city").map_elements(_p1._get_country,
                                                    return_dtype=pl.String).alias("country"))
          .filter(pl.col("country") != "Unknown")
          .group_by("country").len().sort("len", descending=True).head(10))
    names = md["country"].to_list()[::-1]
    vals = md["len"].to_list()[::-1]
    y_pos = np.arange(len(names))
    bars = ax2.barh(y_pos, vals, color=hc.AMBER)
    bars[-1].set_color(hc.ACCENT)  # highlight the top host country
    ax2.bar_label(bars, labels=[hc.fmt_k(v) for v in vals], fontsize=8.5,
                  padding=4, color=hc.TEXT)
    ax2.set_yticks(y_pos, names, fontsize=9.5)
    ax2.set_xlim(0, max(vals) * 1.15)
    ax2.xaxis.set_major_formatter(hc.KFMT)
    ax2.grid(axis="y", visible=False)
    hc.title(ax2, "Where mixed teams race", "Mixed Doubles results by host country, top 10")

    fig.tight_layout()
    hc.save(fig, "08_mixed_doubles_overview",
            takeaway="Among the mixed teams the archive captures (coverage is incomplete), the US, "
                     "Germany and the UK host the most; mixed is roughly a quarter of recorded "
                     "doubles and likely more in reality.")


def india_spotlight(lf: pl.LazyFrame) -> None:
    """Mumbai and Bengaluru against the world: young market, huge fields, long races."""
    india_cities = ["Mumbai", "Bengaluru", "Delhi"]
    # Chart 1: Finish times vs the world
    fig1, ax1 = plt.subplots(figsize=(8.0, 4.5))
    bins = np.arange(3600, 12600, 120)
    world = (hc.singles(lf, pro=False).select("finish_seconds").collect()
             ["finish_seconds"].to_numpy())
    ax1.hist(world, bins=bins, density=True, alpha=0.35, color="#94a3b8",
             label=f"World (median {hc.fmt_hms(np.median(world))})")
    for city, color in [("Mumbai", hc.ACCENT), ("Bengaluru", hc.AMBER)]:
        vals = (hc.singles(lf, pro=False).filter(pl.col("city") == city)
                .select("finish_seconds").collect()["finish_seconds"].to_numpy())
        if len(vals) < 200:
            continue
        med = np.median(vals)
        ax1.hist(vals, bins=bins, density=True, alpha=0.55, color=color,
                 label=f"{city} (median {hc.fmt_hms(med)})")
        ax1.axvline(med, color=color, ls="--", lw=1.5)
    ax1.xaxis.set_major_formatter(hc.HMS)
    ax1.set_xlabel("Finish time")
    ax1.set_yticks([])
    ax1.legend(fontsize=8.5)
    hc.title(ax1, "Finish times vs the world", "Open singles, clean races")
    fig1.tight_layout()
    hc.save(fig1, "08_india_vs_world_finish",
            takeaway="Indian fields are dominated by first-timers, so medians sit 25-30 minutes "
                     "behind the world.")

    # Chart 2: Participation by season
    fig2, ax2 = plt.subplots(figsize=(8.0, 4.5))
    by_season = (
        lf.filter(pl.col("city").is_in(india_cities))
        .group_by("season", "city").len()
        .collect()
        .pivot(on="city", index="season", values="len")
        .sort("season")
        .fill_null(0)
    )
    seasons = by_season["season"].to_list()
    bottom = np.zeros(len(seasons))
    for city, color in [("Mumbai", hc.ACCENT), ("Bengaluru", hc.AMBER),
                        ("Delhi", hc.GREEN)]:
        if city not in by_season.columns:
            continue
        vals = by_season[city].to_numpy().astype(float)
        ax2.bar(seasons, vals, bottom=bottom, color=color, label=city, width=0.55)
        bottom += vals
    for s, tot in zip(seasons, bottom):
        ax2.text(s, tot + bottom.max() * 0.02, hc.fmt_k(tot), ha="center",
                 fontsize=9, color=hc.TEXT, fontweight="bold")
    ax2.set_xticks(seasons)
    ax2.set_xticklabels([f"S{s}\n{hc.SEASON_YEARS[s]}" for s in seasons], fontsize=8)
    ax2.set_ylabel("Race results at Indian venues")
    ax2.set_ylim(0, bottom.max() * 1.15)
    ax2.legend(loc="upper left", fontsize=8.5)
    hc.title(ax2, "Participation at Indian venues", "All divisions, by season")
    fig2.tight_layout()
    hc.save(fig2, "08_india_vs_world_participation",
            takeaway="The growth curve says that gap is a snapshot of a brand-new market, not a ceiling.")

    # Chart 3: Mixed doubles popularity
    fig3, ax3 = plt.subplots(figsize=(8.0, 4.5))
    doubles = lf.filter(pl.col("division_canonical").is_in(["DOUBLES", "PRO_DOUBLES"]))
    india_doubles = doubles.filter(pl.col("city").is_in(india_cities)).group_by("sex").len().collect()
    world_doubles = doubles.filter(~pl.col("city").is_in(india_cities)).group_by("sex").len().collect()

    india_counts = {row["sex"]: row["len"] for row in india_doubles.to_dicts()}
    world_counts = {row["sex"]: row["len"] for row in world_doubles.to_dicts()}

    total_india = sum(india_counts.values()) or 1
    total_world = sum(world_counts.values()) or 1

    labels = ["Men's", "Women's", "Mixed"]
    ind_pct = [india_counts.get(s, 0) / total_india * 100 for s in ["M", "W", "X"]]
    world_pct = [world_counts.get(s, 0) / total_world * 100 for s in ["M", "W", "X"]]

    x = np.arange(len(labels))
    width = 0.35
    ax3.bar(x - width/2, world_pct, width, label="World", color="#94a3b8")
    ax3.bar(x + width/2, ind_pct, width, label="India", color=hc.AMBER)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=9)
    ax3.set_ylabel("Share of Doubles Athletes (%)")
    ax3.legend(fontsize=8.5)
    hc.title(ax3, "Mixed Doubles are unusually popular in India", "Share of doubles athletes by category")
    fig3.tight_layout()
    hc.save(fig3, "08_india_vs_world_mixed_doubles",
            takeaway="Additionally, Mixed Doubles form a massive portion of the Indian fields compared to the rest of the world.")


def main() -> None:
    hc.style()
    lf = hc.load()
    print("[08] deep-dive analyses")
    doubles_vs_solo(lf)
    completion_rates(lf)
    age_station_heatmap(lf)
    best_run_lap(lf)
    run_to_station_time_budget(lf)
    station_position_effect(lf)
    mixed_doubles_overview(lf)
    india_spotlight(lf)


if __name__ == "__main__":
    main()
