"""Athlete careers: repeat racing, improvement, nationality performance, retention, PB rates."""

import matplotlib.pyplot as plt
import polars as pl

import hyrox_common as hc

NAT_NAMES = {
    "GER": "Germany", "USA": "United States", "GBR": "United Kingdom",
    "NED": "Netherlands", "AUS": "Australia", "FRA": "France",
    "AUT": "Austria", "SUI": "Switzerland", "ITA": "Italy",
    "ESP": "Spain", "CAN": "Canada", "IRL": "Ireland",
    "BEL": "Belgium", "SWE": "Sweden", "DEN": "Denmark",
    "NOR": "Norway", "POL": "Poland", "BRA": "Brazil",
    "MEX": "Mexico", "JPN": "Japan", "KOR": "South Korea",
    "RSA": "South Africa", "ISR": "Israel", "POR": "Portugal",
    "CZE": "Czechia", "FIN": "Finland", "HUN": "Hungary",
    "NZL": "New Zealand", "SGP": "Singapore", "HKG": "Hong Kong",
}


def races_per_athlete(lf: pl.LazyFrame) -> None:
    counts = (
        hc.singles(lf)
        .filter(pl.col("athlete_key").is_not_null())
        .group_by("athlete_key").len()
        .collect()
    )
    dist = counts["len"].value_counts().sort("len")
    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    shown = dist.filter(pl.col("len") <= 15)
    bars = ax.bar(shown["len"], shown["count"], color=hc.ACCENT)
    bars[0].set_color(hc.AMBER)
    ax.set_yscale("log")
    ax.set_xlabel("Singles races per athlete (all seasons)")
    ax.set_ylabel("Athletes (log scale)")
    ax.bar_label(bars, labels=[hc.fmt_k(v) for v in shown["count"].to_list()],
                 fontsize=8, padding=3, color=hc.TEXT)
    one = int(dist.filter(pl.col("len") == 1)["count"][0])
    multi = int(dist.filter(pl.col("len") > 1)["count"].sum())
    hc.title(ax, "Most people race Hyrox exactly once",
             "Number of singles races per unique athlete · log scale")
    hc.save(fig, "07_races_per_athlete",
            takeaway=f"{one:,} athletes have raced once vs {multi:,} repeaters: cracking "
                     "one-timer conversion is the sport's biggest growth lever.")
    hc.table(dist, "07_races_per_athlete")


def improvement_curve(lf: pl.LazyFrame) -> None:
    base = (
        hc.singles(lf)
        .filter(pl.col("athlete_key").is_not_null())
        .select("athlete_key", "season", "finish_seconds")
        .collect()
        .sort("athlete_key", "season")
        .with_columns(pl.col("finish_seconds").cum_count().over("athlete_key").alias("race_no"))
    )
    firsts = base.filter(pl.col("race_no") == 1).select(
        "athlete_key", pl.col("finish_seconds").alias("first_time")
    )
    joined = (
        base.join(firsts, on="athlete_key")
        .with_columns(((pl.col("finish_seconds") - pl.col("first_time")) / 60).alias("delta_min"))
        .filter(pl.col("race_no") <= 8)
    )
    agg = (
        joined.group_by("race_no")
        .agg(
            pl.col("delta_min").median().alias("med"),
            pl.col("delta_min").quantile(0.25).alias("q25"),
            pl.col("delta_min").quantile(0.75).alias("q75"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= 500)
        .sort("race_no")
    )
    fig, ax = plt.subplots(figsize=(10.5, 8.4))
    ax.plot(agg["race_no"], agg["med"], marker="o", color=hc.GREEN, lw=2.4, markersize=5)
    ax.fill_between(agg["race_no"], agg["q25"], agg["q75"], color=hc.GREEN, alpha=0.15,
                    label="middle 50% of athletes")
    ax.axhline(0, color=hc.DIM, lw=1)
    for r, m in zip(agg["race_no"], agg["med"]):
        ax.annotate(f"{m:+.1f}", (r, m), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8.5, color=hc.TEXT)
    ax.set_xlabel("Race number in career")
    ax.set_ylabel("Minutes vs first race (negative = faster)")
    ax.legend(loc="upper right")
    hc.title(ax, "Experience pays: fast, then it plateaus",
             "Median change in finish time relative to each athlete's first Hyrox")
    best = float(agg["med"].min())
    hc.save(fig, "07_improvement_curve",
            takeaway=f"The typical repeater is ~{abs(best):.0f} minutes faster than their debut "
                     "within a few races; most of the gain lands by race 3-4.")
    hc.table(agg, "07_improvement_by_race_number")


def most_active(lf: pl.LazyFrame) -> None:
    top = (
        hc.singles(lf)
        .filter(pl.col("athlete_key").is_not_null() & pl.col("name").is_not_null())
        .group_by("athlete_key")
        .agg(
            pl.col("name").first(),
            pl.col("nationality").drop_nulls().first(),
            pl.len().alias("races"),
            pl.col("finish_seconds").min().alias("pb_seconds"),
            pl.col("season").min().alias("first_season"),
            pl.col("season").max().alias("last_season"),
        )
        .sort("races", descending=True)
        .head(30)
        .collect()
        .with_columns(
            pl.col("pb_seconds").map_elements(hc.fmt_hms, return_dtype=pl.String).alias("pb")
        )
        .drop("athlete_key", "pb_seconds")
    )
    hc.table(top, "07_most_active_athletes")


def nationality_performance(lf: pl.LazyFrame) -> None:
    df = (
        hc.singles(lf, pro=False)
        .filter(pl.col("nationality").is_not_null() & pl.col("sex").is_in(["M", "W"]))
        .group_by("nationality", "sex")
        .agg(pl.col("finish_seconds").median().alias("med"), pl.len().alias("n"))
        .filter(pl.col("n") >= 2000)
        .collect()
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 8.5))
    for ax, sex in [(axes[0], "M"), (axes[1], "W")]:
        sub = df.filter(pl.col("sex") == sex).sort("med").head(15)
        names = [NAT_NAMES.get(n, n) for n in sub["nationality"].to_list()]
        bars = ax.barh(names[::-1], sub["med"].to_list()[::-1], color=hc.SEX_COLORS[sex])
        bars[-1].set_color(hc.AMBER)
        ax.bar_label(bars, labels=[hc.fmt_hms(v) for v in sub["med"].to_list()[::-1]],
                     fontsize=8, padding=4, color=hc.TEXT)
        ax.xaxis.set_major_formatter(hc.HMS)
        lo = min(sub["med"]) - 420
        ax.set_xlim(lo, max(sub["med"]) + 300)
        ax.grid(axis="y", visible=False)
        hc.title(ax, hc.SEX_NAMES[sex], "Median Open-singles finish · min 2,000 results")
    hc.sup(fig, "The fastest Hyrox nations",
           "Countries ranked by their median athlete: amber = fastest nation")
    hc.save(fig, "07_nationality_performance",
            takeaway="Smaller endurance-sport nations tend to send faster-than-average fields: "
                     "big markets bring volume, small ones bring specialists.")
    hc.table(df.sort("sex", "med"), "07_nationality_medians")


def retention(lf: pl.LazyFrame) -> None:
    base = (
        hc.singles(lf)
        .filter(pl.col("athlete_key").is_not_null())
        .select("athlete_key", "season")
        .unique()
        .collect()
    )
    rows = []
    # S9 has barely started, so S8->S9 retention would be meaningless.
    for s in range(1, 8):
        cur = set(base.filter(pl.col("season") == s)["athlete_key"].to_list())
        nxt = set(base.filter(pl.col("season") == s + 1)["athlete_key"].to_list())
        if len(cur) >= 500 and len(nxt) >= 500:
            rows.append({"season": s, "athletes": len(cur),
                         "returned_next": len(cur & nxt),
                         "retention_pct": len(cur & nxt) / len(cur) * 100})
    ret = pl.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10.0, 8.0))
    ax.plot(ret["season"], ret["retention_pct"], marker="o", color=hc.VIOLET,
            lw=2.4, markersize=5)
    for s, v in zip(ret["season"], ret["retention_pct"]):
        ax.annotate(f"{v:.0f}%", (s, v), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color=hc.TEXT)
    hc.season_axis(ax, first=int(ret["season"].min()), last=int(ret["season"].max()))
    ax.set_ylabel("% racing singles again next season")
    ax.set_ylim(0, 50)
    hc.title(ax, "Around four in ten athletes come back the next season",
             "Share of each season's singles athletes who raced singles again the following season")
    last = float(ret["retention_pct"][-1])
    hc.save(fig, "07_retention",
            takeaway=f"Retention has settled near {last:.0f}%: strong for an event this brutal, "
                     "yet growth still comes overwhelmingly from first-timers.")
    hc.table(ret, "07_retention")


def career_trajectory_pct(lf: pl.LazyFrame) -> None:
    base = (
        hc.singles(lf)
        .filter(pl.col("athlete_key").is_not_null())
        .select("athlete_key", "season", "finish_pct_in_event")
        .collect()
        .sort("athlete_key", "season")
        .with_columns(pl.col("finish_pct_in_event").cum_count().over("athlete_key").alias("race_no"))
    )
    firsts = base.filter(pl.col("race_no") == 1).select(
        "athlete_key", pl.col("finish_pct_in_event").alias("first_pct")
    )
    joined = (
        base.join(firsts, on="athlete_key")
        .with_columns(
            ((pl.col("first_pct") - pl.col("finish_pct_in_event")) * 100).alias("pct_improvement")
        )
        .filter(pl.col("race_no") <= 8)
    )
    agg = (
        joined.group_by("race_no")
        .agg(
            pl.col("pct_improvement").median().alias("med"),
            pl.col("pct_improvement").quantile(0.25).alias("q25"),
            pl.col("pct_improvement").quantile(0.75).alias("q75"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= 500)
        .sort("race_no")
    )
    fig, ax = plt.subplots(figsize=(10.5, 8.4))
    ax.plot(agg["race_no"], agg["med"], marker="o", color=hc.ACCENT, lw=2.4, markersize=5)
    ax.fill_between(agg["race_no"], agg["q25"], agg["q75"], color=hc.ACCENT, alpha=0.15,
                    label="middle 50% of athletes")
    ax.axhline(0, color=hc.DIM, lw=1)
    ax.set_xlabel("Race number in career")
    ax.set_ylabel("Percentile places gained vs first race")
    ax.legend(loc="upper left")
    hc.title(ax, "Repeaters climb the field, not just the clock",
             "Median improvement in within-event percentile relative to each athlete's debut")
    hc.save(fig, "07_career_traj_pct",
            takeaway="Field position keeps improving with every race even after raw times "
                     "plateau: experience converts to racecraft.")


def pb_rate(lf: pl.LazyFrame) -> None:
    base = (
        hc.singles(lf)
        .filter(pl.col("athlete_key").is_not_null())
        .select("athlete_key", "season", "finish_seconds")
        .collect()
        .sort("athlete_key", "season")
        .with_columns(pl.col("finish_seconds").cum_count().over("athlete_key").alias("race_no"))
    )
    base = base.with_columns(
        pl.col("finish_seconds").cum_min().over("athlete_key").alias("cum_best")
    )
    base = base.with_columns(
        ((pl.col("finish_seconds") <= pl.col("cum_best")) & (pl.col("race_no") > 1)).alias("is_pb")
    )
    agg = (
        base.filter(pl.col("race_no") > 1)
        .group_by("race_no")
        .agg(pl.col("is_pb").mean().alias("pb_rate"), pl.len().alias("n"))
        .filter(pl.col("n") >= 500)
        .sort("race_no")
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(10.0, 8.0))
    bars = ax.bar(agg["race_no"], agg["pb_rate"] * 100, color=hc.GREEN)
    ax.bar_label(bars, fmt="%.0f%%", fontsize=9.5, color=hc.TEXT)
    ax.set_xlabel("Race number in career")
    ax.set_ylabel("% who set a personal best")
    ax.set_xticks(agg["race_no"].to_list())
    hc.title(ax, "Personal bests get harder to find with every race",
             "PB = personal best, an athlete's fastest Hyrox finish so far · "
             "share of athletes beating their own record at each career race")
    hc.save(fig, "07_pb_rate",
            takeaway="Over half of second races are PBs; by race 8 the majority aren't: "
                     "early gains are cheap, later ones must be earned.")
    hc.table(agg, "07_pb_rate")


def nationality_diversity(lf: pl.LazyFrame) -> None:
    df = (
        lf.filter(pl.col("nationality").is_not_null() & (pl.col("season") < 9))
        .group_by("season")
        .agg(pl.col("nationality").n_unique().alias("unique_nats"), pl.len().alias("n"))
        .sort("season")
        .collect()
    )
    fig, ax = plt.subplots(figsize=(10.0, 8.0))
    ax.plot(df["season"], df["unique_nats"], marker="o", color=hc.VIOLET, lw=2.4,
            markersize=5)
    for s, v in zip(df["season"], df["unique_nats"]):
        ax.annotate(str(v), (s, v), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color=hc.TEXT)
    hc.season_axis(ax, first=int(df["season"].min()), last=int(df["season"].max()))
    ax.set_ylabel("Countries represented")
    ax.set_ylim(0, df["unique_nats"].max() * 1.2)
    hc.title(ax, "From a German event to a global sport",
             "Unique athlete nationalities appearing in results each season")
    hc.save(fig, "07_nationality_diversity",
            takeaway=f"Athletes from {df['unique_nats'][-1]} countries raced in Season 8: "
                     "Hyrox now has genuinely global reach.")


def main() -> None:
    hc.style()
    lf = hc.load()
    print("[07] athlete careers")
    races_per_athlete(lf)
    improvement_curve(lf)
    most_active(lf)
    nationality_performance(lf)
    retention(lf)
    career_trajectory_pct(lf)
    pb_rate(lf)
    nationality_diversity(lf)


if __name__ == "__main__":
    main()
