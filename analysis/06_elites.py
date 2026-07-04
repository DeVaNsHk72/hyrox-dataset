"""The sharp end: winning times, all-time bests, elite trends, station records, split breakdowns."""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import hyrox_common as hc

INDIV = ["OPEN", "PRO", "ELITE"]


def winning_times(lf: pl.LazyFrame) -> None:
    df = (
        lf.filter(
            pl.col("division_canonical").is_in(["OPEN", "PRO"])
            & pl.col("is_clean_race")
            & pl.col("sex").is_in(["M", "W"])
        )
        .group_by("season", "city", "event_id", "division_canonical", "sex")
        .agg(pl.col("finish_seconds").min().alias("win"))
        .group_by("season", "division_canonical", "sex")
        .agg(pl.col("win").median().alias("median_winning_time"), pl.len().alias("events"))
        .filter(pl.col("events") >= 5)
        .sort("season")
        .collect()
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.0), sharey=True)
    for ax, d in zip(axes, ["OPEN", "PRO"]):
        for sex in ["M", "W"]:
            sub = df.filter((pl.col("division_canonical") == d) & (pl.col("sex") == sex))
            ax.plot(sub["season"], sub["median_winning_time"], marker="o",
                    label=hc.SEX_NAMES[sex], color=hc.SEX_COLORS[sex], lw=2, markersize=4.5)
        sub_all = df.filter(pl.col("division_canonical") == d)
        hc.season_axis(ax, first=int(sub_all["season"].min()), last=int(sub_all["season"].max()))
        if int(sub_all["season"].max()) == 9:
            hc.partial_season_note(ax)
        hc.title(ax, hc.DIV_NAMES[d], "Median event-winning time per season")
    axes[0].yaxis.set_major_formatter(hc.HMS)
    axes[0].legend()
    fig.tight_layout()
    hc.save(fig, "06_winning_time_trend",
            takeaway="While the median athlete slowed (growth effect), event winners got faster "
                     "season after season: the sharp end is professionalizing.")
    hc.table(df, "06_winning_times_by_season")


def alltime_lists(lf: pl.LazyFrame) -> None:
    base = lf.filter(
        pl.col("division_canonical").is_in(INDIV)
        & pl.col("is_clean_race")
        & pl.col("sex").is_in(["M", "W"])
    )
    for sex, name in [("M", "men"), ("W", "women")]:
        top = (
            base.filter(pl.col("sex") == sex)
            .sort("finish_seconds")
            .head(500)
            .select("name", "nationality", "season", "year", "city",
                    "division_canonical", "finish_seconds")
            .collect()
            .unique(subset=["name"], keep="first", maintain_order=True)
            .head(25)
            .with_columns(
                pl.col("finish_seconds").map_elements(hc.fmt_hms, return_dtype=pl.String)
                .alias("finish")
            )
        )
        hc.table(top, f"06_alltime_top25_{name}")

    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    for sex in ["M", "W"]:
        seasons = (
            base.filter(pl.col("sex") == sex)
            .sort("finish_seconds").head(200).select("season").collect()
        )["season"].to_numpy()
        counts = np.bincount(seasons, minlength=10)[1:]
        off = -0.2 if sex == "M" else 0.2
        ax.bar(np.arange(1, 10) + off, counts, width=0.4, label=hc.SEX_NAMES[sex],
               color=hc.SEX_COLORS[sex])
    ax.set_xticks(range(1, 10))
    ax.set_xticklabels([f"S{s}\n{hc.SEASON_YEARS[s]}" for s in range(1, 10)], fontsize=8)
    ax.set_ylabel("All-time top-200 performances")
    ax.legend(loc="upper left")
    hc.title(ax, "Almost every all-time-great performance is recent",
             "Season in which each of the 200 fastest individual times ever was set")
    hc.save(fig, "06_top200_by_season",
            takeaway="The record book is being rewritten right now: the vast majority of "
                     "all-time top-200 times were set in the last two seasons.")


def station_bests(lf: pl.LazyFrame) -> None:
    base = lf.filter(
        pl.col("division_canonical").is_in(INDIV)
        & pl.col("is_clean_race")
        & pl.col("sex").is_in(["M", "W"])
    )
    rows = []
    for col in hc.STATION_COLS + ["run_total_seconds", "best_run_lap_seconds"]:
        label = hc.STATION_LABELS_FULL.get(
            col, {"run_total_seconds": "Running total",
                  "best_run_lap_seconds": "Best run lap"}.get(col, col))
        for sex in ["M", "W"]:
            # 99.9th percentile: with ~10^5 rows per group, anything tighter is
            # still dominated by chip-timing glitches (e.g. 19-second burpee legs).
            q = base.filter(pl.col("sex") == sex).select(
                pl.col(col).quantile(0.001).alias("best")
            ).collect()["best"][0]
            rows.append({"segment": label, "sex": sex,
                         "best_plausible": hc.fmt_hms(q), "seconds": q})
    hc.table(pl.DataFrame(rows), "06_station_bests")


def elite_field_gap(lf: pl.LazyFrame) -> None:
    df = (
        hc.singles(lf)
        .group_by("season", "sex")
        .agg(
            pl.col("finish_seconds").quantile(0.01).alias("p1"),
            pl.col("finish_seconds").median().alias("p50"),
            pl.len().alias("n"),
        )
        .filter((pl.col("n") >= 1000) & pl.col("sex").is_in(["M", "W"]))
        .sort("season")
        .collect()
    )
    fig, ax = plt.subplots(figsize=(10.5, 8.4))
    for sex in ["M", "W"]:
        sub = df.filter(pl.col("sex") == sex)
        ax.plot(sub["season"], (sub["p50"] - sub["p1"]) / 60, marker="o",
                label=hc.SEX_NAMES[sex], color=hc.SEX_COLORS[sex], lw=2.2, markersize=5)
    hc.season_axis(ax, first=int(df["season"].min()), last=int(df["season"].max()))
    if int(df["season"].max()) == 9:
        hc.partial_season_note(ax)
    ax.set_ylabel("Median minus 1st-percentile finish (minutes)")
    ax.legend()
    hc.title(ax, "The gap between elite and average keeps widening",
             "Minutes separating the top 1% from the median athlete, per season · singles")
    hc.save(fig, "06_elite_field_gap",
            takeaway="Both ends of the sport are pulling apart: elites train like pros while "
                     "each season's newcomers start further back.")


def sub_hour_club(lf: pl.LazyFrame) -> None:
    df = (
        lf.filter(
            pl.col("division_canonical").is_in(INDIV)
            & pl.col("is_clean_race") & pl.col("sex").is_in(["M", "W"])
        )
        .group_by("season", "sex")
        .agg(
            (pl.col("finish_seconds") < 3600).sum().alias("sub60"),
            (pl.col("finish_seconds") < 4200).sum().alias("sub70"),
            pl.len().alias("n"),
        )
        .sort("season")
        .collect()
        .with_columns(
            (pl.col("sub60") / pl.col("n") * 100).alias("pct_sub60"),
            (pl.col("sub70") / pl.col("n") * 100).alias("pct_sub70"),
        )
    )
    fig, ax = plt.subplots(figsize=(10.5, 8.4))
    for sex, label, col in [("M", "Men under 60 min", "pct_sub60"),
                            ("W", "Women under 70 min", "pct_sub70")]:
        sub = df.filter(pl.col("sex") == sex)
        ax.plot(sub["season"], sub[col], marker="o", label=label,
                color=hc.SEX_COLORS[sex], lw=2.2, markersize=5)
    hc.season_axis(ax, first=int(df["season"].min()), last=int(df["season"].max()))
    if int(df["season"].max()) == 9:
        hc.partial_season_note(ax)
    ax.set_ylabel("% of individual finishers")
    ax.legend()
    hc.title(ax, "Benchmark times: rarer than you'd think",
             "Share of individual finishers beating the classic benchmarks, per season")
    hc.save(fig, "06_benchmark_share",
            takeaway="Even as elites get faster, the benchmark share of the total field shrinks: "
                     "the flood of newcomers dilutes the percentages every season.")
    hc.table(df, "06_benchmark_share")


def elite_time_budget(lf: pl.LazyFrame) -> None:
    """Share of race time per segment: top-10 all-time men vs the median field."""
    base = hc.singles(lf).filter(pl.col("sex") == "M")
    top10 = base.sort("finish_seconds").head(10).collect()
    median_field = base.select(
        [pl.col(c).median().alias(c) for c in hc.STATION_COLS]
        + [pl.col("run_total_seconds").median().alias("run_total_seconds"),
           pl.col("roxzone_filled_seconds").median().alias("roxzone_filled_seconds")]
    ).collect()

    cols = ["run_total_seconds"] + hc.STATION_COLS + ["roxzone_filled_seconds"]
    labels = ["Running"] + [hc.STATION_LABELS[c] for c in hc.STATION_COLS] + ["Roxzone"]
    colors = ([hc.ACCENT] + [hc.VIOLET, hc.PINK, hc.GREEN, hc.AMBER,
                             "#0891b2", "#65a30d", "#ea580c", "#c026d3"] + [hc.RED])

    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(13.0, 6.2))
    rows = [("Top 10 men all-time", top10), ("Median men's field", median_field)]
    for y, (name, data) in enumerate(rows):
        vals = np.array([float(data[c].mean()) for c in cols])
        pcts = vals / vals.sum() * 100
        left = 0.0
        # Only the percentage goes inside the segment; names live in the legend.
        for pct, color in zip(pcts, colors):
            ax.barh(y, pct, left=left, color=color, height=0.55)
            if pct >= 3:
                ax.text(left + pct / 2, y, f"{pct:.0f}%", ha="center", va="center",
                        fontsize=8.5, color="#ffffff", fontweight="bold")
            left += pct
        ax.text(101, y, name, va="center", fontsize=10, color=hc.TEXT, fontweight="bold")
    ax.set_yticks([])
    ax.set_xlim(0, 130)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of race time (%)")
    ax.grid(axis="y", visible=False)
    handles = [Patch(facecolor=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncols=5, fontsize=8.5, frameon=False)
    hc.title(ax, "Elites and the field spend their race almost identically",
             "Each bar = 100% of race time, split by segment · elites are faster everywhere, not differently balanced")
    hc.save(fig, "06_elite_time_budget",
            takeaway="The time distribution barely changes from world-class to mid-pack: "
                     "Hyrox rewards raising your whole profile, not re-balancing it.")


def fastest_station_with_names(lf: pl.LazyFrame) -> None:
    base = lf.filter(
        pl.col("division_canonical").is_in(INDIV)
        & pl.col("is_clean_race")
        & pl.col("sex").is_in(["M", "W"])
    )
    rows = []
    for col in hc.STATION_COLS:
        label = hc.STATION_LABELS_FULL[col]
        for sex in ["M", "W"]:
            best = (
                base.filter(pl.col("sex") == sex)
                .sort(col)
                .head(5)
                .select("name", "nationality", "season", "city", pl.col(col).alias("time_seconds"))
                .collect()
            )
            for i, r in enumerate(best.iter_rows(named=True)):
                rows.append({
                    "station": label, "sex": sex, "rank": i + 1,
                    "name": r["name"], "nationality": r["nationality"],
                    "season": r["season"], "city": r["city"],
                    "time": hc.fmt_hms(r["time_seconds"]),
                })
    hc.table(pl.DataFrame(rows), "06_fastest_station_holders")


def main() -> None:
    hc.style()
    lf = hc.load()
    print("[06] elites & records")
    winning_times(lf)
    alltime_lists(lf)
    station_bests(lf)
    elite_field_gap(lf)
    sub_hour_club(lf)
    elite_time_budget(lf)
    fastest_station_with_names(lf)


if __name__ == "__main__":
    main()
