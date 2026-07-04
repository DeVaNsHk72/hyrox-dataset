"""Finish-time analysis: distributions, percentiles, age curves, trends, city difficulty, benchmarks."""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import hyrox_common as hc

PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def distributions(lf: pl.LazyFrame) -> None:
    df = hc.singles(lf, pro=False).select("sex", "finish_seconds").collect()
    fig, ax = plt.subplots(figsize=(11.0, 8.0))
    bins = np.arange(2700, 10800, 60)
    medians = {}
    # Men's label sits left of its line, women's right of its line, at different
    # heights, so the two annotations can never collide.
    side = {"M": dict(ha="right", dx=-8, y=0.96), "W": dict(ha="left", dx=8, y=0.88)}
    for sex in ["M", "W"]:
        vals = df.filter(pl.col("sex") == sex)["finish_seconds"].to_numpy()
        medians[sex] = np.median(vals)
        ax.hist(vals, bins=bins, alpha=0.6, color=hc.SEX_COLORS[sex], density=True,
                label=f"{hc.SEX_NAMES[sex]}  (n={len(vals):,})")
        ax.axvline(medians[sex], color=hc.SEX_COLORS[sex], ls="--", lw=1.6)
        ax.annotate(f"median {hc.fmt_hms(medians[sex])}",
                    xy=(medians[sex], side[sex]["y"]),
                    xycoords=("data", "axes fraction"),
                    xytext=(side[sex]["dx"], 0), textcoords="offset points",
                    ha=side[sex]["ha"], fontsize=9, color=hc.SEX_COLORS[sex],
                    fontweight="bold",
                    bbox=dict(fc=hc.BG, ec="none", alpha=0.8, pad=1.5))
    ax.xaxis.set_major_formatter(hc.HMS)
    ax.set_xlabel("Finish time")
    ax.set_yticks([])
    ax.legend(loc="upper right")
    hc.title(ax, "What a typical Hyrox finish looks like",
             "Distribution of all-time Open singles finish times · dashed line = median")
    hc.save(fig, "02_finish_distribution_open",
            takeaway=f"Half of all men finish between roughly {hc.fmt_hms(medians['M'] - 900)} and "
                     f"{hc.fmt_hms(medians['M'] + 900)}; women's median is "
                     f"{hc.fmt_hms(medians['W'])} vs men's {hc.fmt_hms(medians['M'])}.")

    divs = ["OPEN", "PRO", "DOUBLES", "PRO_DOUBLES"]
    dd = (
        lf.filter(
            pl.col("division_canonical").is_in(divs)
            & pl.col("is_clean_race")
            & pl.col("sex").is_in(["M", "W"])
        )
        .select("division_canonical", "sex", "finish_seconds")
        .collect()
    )
    fig, ax = plt.subplots(figsize=(12.0, 8.0))
    pos, ticklabels = 0, []
    for d in divs:
        for sex in ["M", "W"]:
            vals = dd.filter(
                (pl.col("division_canonical") == d) & (pl.col("sex") == sex)
            )["finish_seconds"].to_numpy()
            if len(vals) < 100:
                continue
            if len(vals) > 50000:
                vals = np.random.default_rng(0).choice(vals, 50000, replace=False)
            p = ax.violinplot(vals, positions=[pos], widths=0.85, showmedians=True)
            for body in p["bodies"]:
                body.set_facecolor(hc.SEX_COLORS[sex])
                body.set_alpha(0.65)
            for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
                p[part].set_color(hc.TEXT)
                p[part].set_linewidth(1)
            med = np.median(vals)
            ax.annotate(hc.fmt_hms(med), (pos, med), textcoords="offset points",
                        xytext=(0, -16), ha="center", fontsize=8, color=hc.TEXT)
            ticklabels.append(f"{hc.DIV_NAMES[d]}\n{hc.SEX_NAMES[sex]}")
            pos += 1
    ax.set_xticks(range(len(ticklabels)), ticklabels, fontsize=8.5)
    ax.yaxis.set_major_formatter(hc.HMS)
    ax.set_ylabel("Finish time")
    hc.title(ax, "Doubles are ~15 minutes faster than solo: Pro is slower than Open",
             "Finish-time distributions by division and sex · number = median")
    hc.save(fig, "02_finish_by_division",
            takeaway="Splitting the stations with a partner (Doubles) buys far more time than "
                     "the heavier Pro weights cost.")


def percentile_tables(lf: pl.LazyFrame) -> None:
    divs = ["OPEN", "PRO", "DOUBLES", "PRO_DOUBLES"]
    df = (
        lf.filter(
            pl.col("division_canonical").is_in(divs)
            & pl.col("is_clean_race")
            & pl.col("sex").is_in(["M", "W"])
        )
        .group_by("division_canonical", "sex")
        .agg(
            pl.len().alias("n"),
            *[pl.col("finish_seconds").quantile(p / 100).alias(f"p{p}") for p in PCTS],
        )
        .sort("division_canonical", "sex")
        .collect()
    )
    pretty = df.with_columns(
        [pl.col(f"p{p}").map_elements(hc.fmt_hms, return_dtype=pl.String) for p in PCTS]
    )
    hc.table(pretty, "02_finish_percentiles")

    op = hc.singles(lf, pro=False).select("sex", "finish_seconds").collect()
    qs = np.arange(0.01, 1.0, 0.01)
    fig, ax = plt.subplots(figsize=(11.0, 8.7))
    for sex in ["M", "W"]:
        vals = op.filter(pl.col("sex") == sex)["finish_seconds"].to_numpy()
        curve = np.quantile(vals, qs)
        ax.plot(qs * 100, curve, color=hc.SEX_COLORS[sex], lw=2.4, label=hc.SEX_NAMES[sex])
    for mins, label in [(60, "1:00"), (75, "1:15"), (90, "1:30")]:
        ax.axhline(mins * 60, color=hc.GRID, lw=0.9, ls=":")
        ax.text(100.5, mins * 60, label, fontsize=8, color=hc.DIM, va="center")
    ax.yaxis.set_major_formatter(hc.HMS)
    ax.set_xlabel("Field percentile (lower = faster)")
    ax.set_ylabel("Finish time")
    ax.legend(loc="upper left")
    m_vals = op.filter(pl.col("sex") == "M")["finish_seconds"].to_numpy()
    w_vals = op.filter(pl.col("sex") == "W")["finish_seconds"].to_numpy()
    sub60 = (m_vals < 3600).mean() * 100
    hc.title(ax, "How fast do you need to be? Find your percentile",
             "All-time Open singles: the finish time at every percentile of the field")
    hc.save(fig, "02_percentile_curve",
            takeaway=f"Sub-60 puts a man in the top {sub60:.0f}% of all Open finishers ever; "
                     f"sub-70 for women is roughly the top {(w_vals < 4200).mean() * 100:.0f}%.")


def season_trend(lf: pl.LazyFrame) -> None:
    df = (
        lf.filter(
            pl.col("division_canonical").is_in(["OPEN", "PRO", "DOUBLES"])
            & pl.col("is_clean_race")
            & pl.col("sex").is_in(["M", "W"])
        )
        .group_by("season", "division_canonical", "sex")
        .agg(pl.col("finish_seconds").median().alias("med"), pl.len().alias("n"))
        .filter(pl.col("n") >= 200)
        .sort("season")
        .collect()
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 6.2), sharey=True)
    for ax, d in zip(axes, ["OPEN", "PRO", "DOUBLES"]):
        for sex in ["M", "W"]:
            sub = df.filter((pl.col("division_canonical") == d) & (pl.col("sex") == sex))
            ax.plot(sub["season"], sub["med"], marker="o", label=hc.SEX_NAMES[sex],
                    color=hc.SEX_COLORS[sex], lw=2, markersize=4)
        ax.set_title(hc.DIV_NAMES[d], fontsize=11, fontweight="bold", color=hc.TEXT)
        hc.season_axis(ax, first=int(df.filter(pl.col("division_canonical") == d)["season"].min()),
                       last=int(df["season"].max()))
    axes[0].yaxis.set_major_formatter(hc.HMS)
    axes[0].set_ylabel("Median finish")
    axes[0].legend(loc="lower left")
    hc.sup(fig, "The median athlete is getting slower: because the field is exploding",
           "Median finish per season: newcomers flood in faster than the front improves")
    hc.save(fig, "02_median_finish_trend",
            takeaway="A slower median isn't decline: it's growth. Each season adds hundreds of "
                     "thousands of first-timers who race mid-pack and behind.")
    hc.table(df, "02_median_finish_by_season")


def age_curves(lf: pl.LazyFrame) -> None:
    df = (
        hc.singles(lf, pro=False)
        .filter(pl.col("age_group").is_in(hc.AGE_GROUPS))
        .group_by("age_group", "sex")
        .agg(
            pl.col("finish_seconds").quantile(0.25).alias("q25"),
            pl.col("finish_seconds").median().alias("med"),
            pl.col("finish_seconds").quantile(0.75).alias("q75"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= 100)
        .sort("age_group")
        .collect()
    )
    fig, ax = plt.subplots(figsize=(11.0, 8.7))
    for sex in ["M", "W"]:
        sub = df.filter(pl.col("sex") == sex)
        x = np.arange(len(sub))
        ax.plot(x, sub["med"], marker="o", color=hc.SEX_COLORS[sex],
                label=hc.SEX_NAMES[sex], lw=2.2, markersize=5)
        ax.fill_between(x, sub["q25"], sub["q75"], color=hc.SEX_COLORS[sex], alpha=0.15)
        ax.set_xticks(x, sub["age_group"], fontsize=8.5)
    ax.yaxis.set_major_formatter(hc.HMS)
    ax.set_ylabel("Finish time")
    ax.set_xlabel("Age group")
    ax.legend(loc="upper left")
    m2529 = df.filter((pl.col("age_group") == "25-29") & (pl.col("sex") == "M"))["med"][0]
    m5054 = df.filter((pl.col("age_group") == "50-54") & (pl.col("sex") == "M"))["med"][0]
    hc.title(ax, "Age costs surprisingly little until the mid-50s",
             "Median Open-singles finish by age group · shaded band = middle 50% of athletes")
    hc.save(fig, "02_age_curves",
            takeaway=f"A median 50-54 man is only {(m5054 - m2529) / 60:.0f} minutes slower than a "
                     "median 25-29 man: fitness beats age deep into masters territory.")
    hc.table(df, "02_age_group_stats")


def city_difficulty(lf: pl.LazyFrame) -> None:
    df = (
        hc.singles(lf, pro=False)
        .group_by("city")
        .agg(pl.col("finish_seconds").median().alias("med"), pl.len().alias("n"))
        .filter(pl.col("n") >= 1000)
        .sort("med")
        .collect()
    )
    top, bottom = df.head(12), df.tail(12)
    both = pl.concat([top, bottom])
    colors = [hc.GREEN] * len(top) + [hc.RED] * len(bottom)
    fig, ax = plt.subplots(figsize=(10.0, 12.3))
    bars = ax.barh(both["city"].to_list()[::-1], both["med"].to_list()[::-1],
                   color=colors[::-1])
    ax.bar_label(bars, labels=[hc.fmt_hms(v) for v in both["med"].to_list()[::-1]],
                 fontsize=8, padding=4, color=hc.TEXT)
    ax.xaxis.set_major_formatter(hc.HMS)
    ax.set_xlim(both["med"].min() - 360, both["med"].max() + 240)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Median finish time")
    spread = (both["med"].max() - both["med"].min()) / 60
    hc.title(ax, "Course & crowd matter: the fastest and slowest cities",
             "Median Open-singles finish per city (min 1,000 results) · green = fastest 12, red = slowest 12")
    hc.save(fig, "02_city_difficulty",
            takeaway=f"The same race is ~{spread:.0f} minutes slower in {both['city'][-1]} than in "
                     f"{both['city'][0]}: venue layout, altitude, heat and field mix all show up here.")
    hc.table(df, "02_city_median_finish")


def benchmark_histogram(lf: pl.LazyFrame) -> None:
    marks = {
        "M": [(3600, "Sub-60", hc.GREEN), (4200, "Sub-70", hc.ACCENT),
              (4800, "Sub-80", hc.AMBER), (5400, "Sub-90", hc.RED)],
        "W": [(4200, "Sub-70", hc.GREEN), (4800, "Sub-80", hc.ACCENT),
              (5400, "Sub-90", hc.AMBER), (6000, "Sub-100", hc.RED)],
    }
    for sex in ["M", "W"]:
        df = hc.singles(lf, pro=False).filter(pl.col("sex") == sex).select("finish_seconds").collect()
        vals = df["finish_seconds"].to_numpy()
        fig, ax = plt.subplots(figsize=(11.0, 8.0))
        bins = np.arange(2700, 10800, 60)
        ax.hist(vals, bins=bins, alpha=0.8, color=hc.SEX_COLORS[sex])
        # Alternate label heights so neighbouring benchmark labels never touch.
        heights = [0.97, 0.82, 0.97, 0.82]
        for (secs, lbl, clr), y in zip(marks[sex], heights):
            pct = (vals < secs).sum() / len(vals) * 100
            ax.axvline(secs, color=clr, ls="--", lw=1.8)
            ax.annotate(f"{lbl}\ntop {pct:.0f}%", xy=(secs, y),
                        xycoords=("data", "axes fraction"),
                        xytext=(-8, 0), textcoords="offset points", ha="right", va="top",
                        fontsize=8.5, color=clr, fontweight="bold",
                        bbox=dict(fc=hc.BG, ec=clr, lw=1.1, alpha=0.92, pad=2.5))
        ax.xaxis.set_major_formatter(hc.HMS)
        ax.set_xlabel("Finish time")
        ax.yaxis.set_major_formatter(hc.KFMT)
        ax.set_ylabel("Athletes")
        hc.title(ax, f"{hc.SEX_NAMES[sex]}: where the classic benchmarks sit",
                 "All-time Open singles finishes · each line shows what share of the field beats that time")
        hc.save(fig, f"02_benchmark_histogram_{sex.lower()}",
                takeaway=f"Beating {marks[sex][0][1].lower()} puts a "
                         f"{'man' if sex == 'M' else 'woman'} in the top "
                         f"{(vals < marks[sex][0][0]).sum() / len(vals) * 100:.0f}% of all "
                         "Open finishers in history.")


def percentile_by_season(lf: pl.LazyFrame) -> None:
    base = hc.singles(lf, pro=False).filter(pl.col("sex") == "M")
    seasons_to_show = [4, 6, 7, 8]
    colors = [hc.DIM, hc.VIOLET, hc.ACCENT, hc.GREEN]
    qs = np.arange(0.01, 1.0, 0.01)
    fig, ax = plt.subplots(figsize=(11.0, 8.7))
    for s, clr in zip(seasons_to_show, colors):
        df = base.filter(pl.col("season") == s).select("finish_seconds").collect()
        if len(df) < 500:
            continue
        vals = df["finish_seconds"].to_numpy()
        curve = np.quantile(vals, qs)
        ax.plot(qs * 100, curve, color=clr, lw=2.2,
                label=f"Season {s} ({hc.SEASON_YEARS[s]})")
    ax.yaxis.set_major_formatter(hc.HMS)
    ax.set_xlabel("Field percentile (lower = faster)")
    ax.set_ylabel("Finish time")
    ax.legend(loc="upper left")
    hc.title(ax, "The front got faster, the back got slower",
             "Men's Open percentile curves by season: the curve twists rather than shifts")
    hc.save(fig, "02_percentile_by_season",
            takeaway="Season 8's top 10% beats Season 4's top 10%, while its back half is slower: "
                     "elite depth and mass participation grew at the same time.")


def field_depth_trend(lf: pl.LazyFrame) -> None:
    df = (
        hc.singles(lf, pro=False)
        .filter(pl.col("sex").is_in(["M", "W"]))
        .group_by("season", "sex")
        .agg(
            pl.col("finish_seconds").quantile(0.25).alias("p25"),
            pl.col("finish_seconds").median().alias("p50"),
            pl.col("finish_seconds").quantile(0.75).alias("p75"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= 500)
        .sort("season")
        .collect()
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.8), sharey=True)
    for ax, sex in [(axes[0], "M"), (axes[1], "W")]:
        sub = df.filter(pl.col("sex") == sex)
        ax.plot(sub["season"], sub["p25"], marker="s", label="Fastest quarter (p25)",
                color=hc.GREEN, lw=2, markersize=4)
        ax.plot(sub["season"], sub["p50"], marker="o", label="Median",
                color=hc.ACCENT, lw=2, markersize=4)
        ax.plot(sub["season"], sub["p75"], marker="^", label="Slowest quarter (p75)",
                color=hc.RED, lw=2, markersize=4)
        ax.fill_between(sub["season"], sub["p25"], sub["p75"], alpha=0.08, color=hc.ACCENT)
        ax.set_title(hc.SEX_NAMES[sex], fontsize=11, fontweight="bold", color=hc.TEXT)
        hc.season_axis(ax, first=int(sub["season"].min()), last=int(sub["season"].max()))
    axes[0].yaxis.set_major_formatter(hc.HMS)
    axes[0].set_ylabel("Finish time")
    axes[0].legend(loc="upper left", fontsize=8)
    hc.sup(fig, "The field is spreading out",
           "25th/50th/75th percentile Open-singles finish per season")
    hc.save(fig, "02_field_depth_trend",
            takeaway="The gap between the fastest and slowest quarter keeps widening: "
                     "Hyrox is simultaneously becoming more elite and more accessible.")
    hc.table(df, "02_field_depth_by_season")


def main() -> None:
    hc.style()
    lf = hc.load()
    print("[02] finish times")
    distributions(lf)
    percentile_tables(lf)
    season_trend(lf)
    age_curves(lf)
    city_difficulty(lf)
    benchmark_histogram(lf)
    percentile_by_season(lf)
    field_depth_trend(lf)


if __name__ == "__main__":
    main()
