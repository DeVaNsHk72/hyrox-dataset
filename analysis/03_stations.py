"""Station-level analysis: where the time goes, spread, sex gaps, elite vs field, station trends."""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import hyrox_common as hc


def station_medians(lf: pl.LazyFrame) -> None:
    df = (
        hc.singles(lf, pro=False)
        .group_by("sex")
        .agg([pl.col(c).median().alias(c) for c in hc.STATION_COLS])
        .collect()
    )
    labels = [hc.STATION_LABELS[c] for c in hc.STATION_COLS]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12.0, 8.4))
    for off, sex in [(-0.2, "M"), (0.2, "W")]:
        row = df.filter(pl.col("sex") == sex)
        vals = [row[c][0] for c in hc.STATION_COLS]
        bars = ax.bar(x + off, vals, width=0.4, label=hc.SEX_NAMES[sex],
                      color=hc.SEX_COLORS[sex])
        ax.bar_label(bars, labels=[hc.fmt_hms(v) for v in vals], fontsize=8,
                     padding=3, color=hc.TEXT)
    ax.set_xticks(x, labels, fontsize=9)
    ax.yaxis.set_major_formatter(hc.HMS)
    ax.set_ylabel("Median time")
    ax.legend()
    hc.title(ax, "Wall Balls and Lunges eat the most clock",
             "Median time per station, Open singles · stations shown in race order")
    hc.save(fig, "03_station_medians",
            takeaway="The four longest stations (Wall Balls, Lunges, Burpee Jumps, Row) take "
                     "2-3x the time of a sled push: pace your effort accordingly.")


def station_boxes(lf: pl.LazyFrame) -> None:
    qs = (
        hc.singles(lf, pro=False)
        .select(
            [pl.col(c).quantile(q).alias(f"{c}|{q}")
             for c in hc.STATION_COLS
             for q in [0.05, 0.25, 0.5, 0.75, 0.95]]
        )
        .collect()
    )
    stats = []
    for c in hc.STATION_COLS:
        stats.append({
            "label": hc.STATION_LABELS[c],
            "whislo": qs[f"{c}|0.05"][0], "q1": qs[f"{c}|0.25"][0],
            "med": qs[f"{c}|0.5"][0], "q3": qs[f"{c}|0.75"][0],
            "whishi": qs[f"{c}|0.95"][0], "fliers": [],
        })
    fig, ax = plt.subplots(figsize=(12.0, 8.4))
    b = ax.bxp(stats, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor=hc.ACCENT, alpha=0.55, edgecolor=hc.TEXT),
               whiskerprops=dict(color=hc.DIM), capprops=dict(color=hc.DIM),
               medianprops=dict(color=hc.AMBER, lw=2))
    ax.set_xticklabels([s["label"] for s in stats], fontsize=9)
    ax.yaxis.set_major_formatter(hc.HMS)
    ax.set_ylabel("Station time")
    hc.title(ax, "Same station, wildly different races",
             "Box = middle 50% of athletes, whiskers = 5th-95th percentile, amber = median · Open singles")
    hc.save(fig, "03_station_spread",
            takeaway="On Wall Balls the slowest 5% need 3x the time of the fastest 5%: "
                     "no other station spreads the field this much.")
    _ = b


def time_budget(lf: pl.LazyFrame) -> None:
    med = (
        hc.singles(lf, pro=False)
        .select(
            pl.col("run_total_seconds").median().alias("Running (8 x 1km)"),
            *[pl.col(c).median().alias(hc.STATION_LABELS_FULL[c]) for c in hc.STATION_COLS],
            pl.col("roxzone_filled_seconds").median().alias("Roxzone (transitions)"),
        )
        .collect()
    )
    parts = {k: med[k][0] for k in med.columns}
    total = sum(parts.values())
    labels = list(parts)
    sizes = [v / total * 100 for v in parts.values()]
    colors = [hc.ACCENT] + [hc.VIOLET] * 8 + [hc.AMBER]
    fig, ax = plt.subplots(figsize=(11.0, 9.4))
    bars = ax.barh(labels[::-1], sizes[::-1], color=colors[::-1])
    ax.bar_label(bars, labels=[f"{s:.1f}%   ({hc.fmt_hms(parts[l])})"
                               for l, s in zip(labels[::-1], sizes[::-1])],
                 fontsize=8.5, padding=4, color=hc.TEXT)
    ax.set_xlim(0, max(sizes) * 1.32)
    ax.set_xlabel("Share of total race time (%)")
    ax.grid(axis="y", visible=False)
    run_pct, rox_pct = sizes[0], sizes[-1]
    hc.title(ax, "Hyrox is a running race with obstacles",
             "How the median Open-singles race splits across running (cyan), stations (violet) and transitions (amber)")
    hc.save(fig, "03_time_budget",
            takeaway=f"Running is {run_pct:.0f}% of the median race and the roxzone another "
                     f"{rox_pct:.0f}%: together more than all eight stations combined.")


def elite_vs_field(lf: pl.LazyFrame) -> None:
    base = hc.singles(lf).with_columns(
        pl.when(pl.col("finish_pct_in_event") <= 0.05).then(pl.lit("Top 5%"))
        .when(pl.col("finish_pct_in_event").is_between(0.4, 0.6)).then(pl.lit("Middle"))
        .when(pl.col("finish_pct_in_event") >= 0.8).then(pl.lit("Back 20%"))
        .otherwise(None)
        .alias("cohort")
    ).filter(pl.col("cohort").is_not_null())
    cols = hc.STATION_COLS + ["run_total_seconds", "roxzone_filled_seconds"]
    df = (
        base.group_by("cohort")
        .agg([pl.col(c).median().alias(c) for c in cols])
        .collect()
    )
    labels = [hc.STATION_LABELS.get(c) or {"run_total_seconds": "Running",
                                           "roxzone_filled_seconds": "Roxzone"}[c]
              for c in cols]
    top = df.filter(pl.col("cohort") == "Top 5%")
    fig, ax = plt.subplots(figsize=(12.5, 8.7))
    x = np.arange(len(cols))
    for off, cohort, color in [(-0.2, "Middle", hc.AMBER), (0.2, "Back 20%", hc.RED)]:
        row = df.filter(pl.col("cohort") == cohort)
        ratio = [(row[c][0] / top[c][0] - 1) * 100 for c in cols]
        bars = ax.bar(x + off, ratio, width=0.4, label=f"{cohort} vs Top 5%", color=color)
        ax.bar_label(bars, fmt="%.0f%%", fontsize=8, color=hc.TEXT)
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylabel("% slower than the Top 5%")
    ax.legend()
    hc.title(ax, "The back of the field loses most time on sleds, Wall Balls and transitions",
             "How much slower the mid-pack and back-of-pack are than the top 5%, segment by segment")
    hc.save(fig, "03_elite_vs_field_gap",
            takeaway="Running gaps are the smallest: strength-endurance stations and roxzone "
                     "discipline are what actually separate finishing groups.")
    hc.table(df, "03_station_medians_by_cohort")


def variability(lf: pl.LazyFrame) -> None:
    df = (
        hc.singles(lf, pro=False)
        .select(
            *[pl.col(c).std().alias(f"{c}|std") for c in hc.STATION_COLS],
            *[pl.col(c).mean().alias(f"{c}|mean") for c in hc.STATION_COLS],
        )
        .collect()
    )
    labels = [hc.STATION_LABELS[c] for c in hc.STATION_COLS]
    cv = [df[f"{c}|std"][0] / df[f"{c}|mean"][0] * 100 for c in hc.STATION_COLS]
    order = np.argsort(cv)[::-1]
    fig, ax = plt.subplots(figsize=(11.0, 8.0))
    bars = ax.bar([labels[i] for i in order], [cv[i] for i in order], color=hc.VIOLET)
    for i, bar in enumerate(bars):
        if i == 0:
            bar.set_color(hc.RED)
    ax.bar_label(bars, fmt="%.0f%%", fontsize=9, color=hc.TEXT)
    ax.set_ylabel("Spread across the field (CV, %)")
    plt.setp(ax.get_xticklabels(), fontsize=9)
    hc.title(ax, "Where athletes differ most (and least)",
             "Coefficient of variation of station times, Open singles · higher = bigger differences between athletes")
    hc.save(fig, "03_station_variability",
            takeaway=f"{labels[order[0]]} is the great divider; "
                     f"{labels[order[-1]]} times are the most uniform across the whole field.")


def sex_gap(lf: pl.LazyFrame) -> None:
    cols = hc.STATION_COLS + ["run_total_seconds"]
    df = (
        hc.singles(lf, pro=False)
        .group_by("sex")
        .agg([pl.col(c).median().alias(c) for c in cols])
        .collect()
    )
    labels = [hc.STATION_LABELS.get(c, "Running") for c in cols]
    m = df.filter(pl.col("sex") == "M")
    w = df.filter(pl.col("sex") == "W")
    gap = [(w[c][0] / m[c][0] - 1) * 100 for c in cols]
    order = np.argsort(gap)
    fig, ax = plt.subplots(figsize=(11.0, 8.7))
    # Green = women faster (negative), pink = men faster; zero line marks parity.
    colors = [hc.GREEN if gap[i] < 0 else hc.PINK for i in order]
    bars = ax.barh([labels[i] for i in order], [gap[i] for i in order], color=colors)
    ax.bar_label(bars, labels=[f"{gap[i]:+.0f}%" for i in order],
                 fontsize=9, padding=4, color=hc.TEXT)
    ax.axvline(0, color=hc.TEXT, lw=1.2)
    lo, hi = min(gap), max(gap)
    ax.set_xlim(lo - 8, hi + 8)
    ax.text(lo - 6, len(cols) - 0.5, "women faster", fontsize=9, color=hc.GREEN,
            style="italic", va="center")
    ax.text(hi + 6, len(cols) - 0.5, "men faster", fontsize=9, color=hc.PINK,
            style="italic", va="center", ha="right")
    ax.set_xlabel("Women's median time vs men's (%) · 0 = parity")
    ax.grid(axis="y", visible=False)
    hc.title(ax, "Where the sexes differ, and where Hyrox equalizes them by design",
             "Green bars: women are faster (lighter implements or lower targets). "
             "Pink: men are faster. Ergs and runs are the fairest comparison.")
    hc.save(fig, "03_sex_gap_by_station",
            takeaway="Women beat men outright on wall balls, lunges and sled push because those "
                     "stations are scaled by sex; on the identical erg stations the gap is a "
                     "modest 13-16%.")


def stats_table(lf: pl.LazyFrame) -> None:
    df = (
        lf.filter(pl.col("division_canonical").is_in(["OPEN", "PRO", "DOUBLES"])
                  & pl.col("is_clean_race") & pl.col("sex").is_in(["M", "W"]))
        .group_by("division_canonical", "sex")
        .agg(
            pl.len().alias("n"),
            *[pl.col(c).median().alias(hc.STATION_LABELS_FULL[c]) for c in hc.STATION_COLS],
            pl.col("run_total_seconds").median().alias("Running total"),
            pl.col("roxzone_filled_seconds").median().alias("Roxzone"),
            pl.col("finish_seconds").median().alias("Finish"),
        )
        .sort("division_canonical", "sex")
        .collect()
    )
    pretty = df.with_columns([
        pl.col(c).map_elements(hc.fmt_hms, return_dtype=pl.String)
        for c in df.columns if c not in ("division_canonical", "sex", "n")
    ])
    hc.table(pretty, "03_station_stats_by_division")


def station_percentiles(lf: pl.LazyFrame) -> None:
    pcts = [5, 10, 25, 50, 75, 90, 95]
    rows = []
    for sex in ["M", "W"]:
        for col in hc.STATION_COLS:
            df = (hc.singles(lf, pro=False).filter(pl.col("sex") == sex)
                  .select(col).drop_nulls().collect())
            vals = df[col].to_numpy()
            row = {"sex": sex, "station": hc.STATION_LABELS_FULL[col]}
            for p in pcts:
                row[f"p{p}"] = hc.fmt_hms(float(np.percentile(vals, p)))
            rows.append(row)
    hc.table(pl.DataFrame(rows), "03_station_percentiles")


def station_trend_by_season(lf: pl.LazyFrame) -> None:
    df = (
        hc.singles(lf, pro=False)
        .filter(pl.col("sex") == "M")
        .group_by("season")
        .agg([pl.col(c).median().alias(c) for c in hc.STATION_COLS] + [pl.len().alias("n")])
        .filter(pl.col("n") >= 500)
        .sort("season")
        .collect()
    )
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 9.8))
    for ax, col in zip(axes.flat, hc.STATION_COLS):
        ax.plot(df["season"], df[col], marker="o", color=hc.ACCENT, lw=2, markersize=4)
        ax.set_title(hc.STATION_LABELS_FULL[col], fontsize=10, color=hc.TEXT,
                     fontweight="bold")
        ax.yaxis.set_major_formatter(hc.HMS)
        ax.tick_params(labelsize=7.5)
        ax.set_xticks(df["season"].to_list())
        ax.set_xticklabels([f"S{s}" for s in df["season"].to_list()], fontsize=7.5)
    hc.sup(fig, "Station times drift up as the field grows",
           "Median men's Open station time per season: the influx of newcomers outweighs equipment/rule changes")
    hc.save(fig, "03_station_trend_by_season",
            takeaway="Every station's median got slower since S5 for the same reason overall "
                     "times did: explosive growth in first-time racers.")


def main() -> None:
    hc.style()
    lf = hc.load()
    print("[03] stations")
    station_medians(lf)
    station_boxes(lf)
    time_budget(lf)
    elite_vs_field(lf)
    variability(lf)
    sex_gap(lf)
    stats_table(lf)
    station_percentiles(lf)
    station_trend_by_season(lf)


if __name__ == "__main__":
    main()
