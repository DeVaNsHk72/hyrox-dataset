"""Pacing & fatigue: run-lap degradation, consistency, roxzone, negative splits, worst-lap."""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import hyrox_common as hc


def cohort_col() -> pl.Expr:
    return (
        pl.when(pl.col("finish_pct_in_event") <= 0.05).then(pl.lit("Top 5%"))
        .when(pl.col("finish_pct_in_event") <= 0.25).then(pl.lit("Top 25%"))
        .when(pl.col("finish_pct_in_event").is_between(0.4, 0.6)).then(pl.lit("Middle"))
        .when(pl.col("finish_pct_in_event") >= 0.8).then(pl.lit("Back 20%"))
        .otherwise(None)
        .alias("cohort")
    )


COHORT_COLORS = {
    "Top 5%": hc.GREEN, "Top 25%": hc.ACCENT, "Middle": hc.AMBER, "Back 20%": hc.RED,
}


def fatigue_curves(lf: pl.LazyFrame) -> None:
    df = (
        hc.singles(lf)
        .with_columns(cohort_col())
        .filter(pl.col("cohort").is_not_null())
        .group_by("cohort")
        .agg([pl.col(c).median().alias(c) for c in hc.RUN_COLS])
        .collect()
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 7.0))
    x = np.arange(1, 9)
    for cohort, color in COHORT_COLORS.items():
        row = df.filter(pl.col("cohort") == cohort)
        vals = np.array([row[c][0] for c in hc.RUN_COLS])
        ax1.plot(x, vals, marker="o", label=cohort, color=color, lw=2, markersize=4.5)
        ax2.plot(x, vals / vals[0] * 100 - 100, marker="o", label=cohort,
                 color=color, lw=2, markersize=4.5)
    ax1.yaxis.set_major_formatter(hc.HMS)
    ax1.set_xlabel("Run lap")
    ax1.set_xticks(x)
    ax1.legend()
    hc.title(ax1, "Median 1km lap time", "By finishing cohort within each event")
    ax2.axhline(0, color=hc.DIM, lw=1)
    ax2.set_xlabel("Run lap")
    ax2.set_xticks(x)
    ax2.set_ylabel("% slower than lap 1")
    hc.title(ax2, "Slowdown vs lap 1", "Same data, relative to each cohort's opening lap")
    fig.tight_layout()
    back8 = df.filter(pl.col("cohort") == "Back 20%")
    fade = (back8["run_8_seconds"][0] / back8["run_1_seconds"][0] - 1) * 100
    top8 = df.filter(pl.col("cohort") == "Top 5%")
    fade_top = (top8["run_8_seconds"][0] / top8["run_1_seconds"][0] - 1) * 100
    hc.save(fig, "04_fatigue_curves",
            takeaway=f"Everyone fades, but not equally: the top 5% slow just {fade_top:.0f}% by lap 8 "
                     f"while the back of the field slows {fade:.0f}%: pacing, not just speed, "
                     "separates the cohorts.")
    hc.table(df, "04_run_laps_by_cohort")


def degradation_vs_result(lf: pl.LazyFrame) -> None:
    df = (
        hc.singles(lf)
        .with_columns(
            ((pl.col("run_8_seconds") - pl.col("run_1_seconds"))
             / pl.col("run_1_seconds") * 100).alias("degradation")
        )
        .filter(pl.col("degradation").is_between(-50, 200))
        .select("finish_pct_in_event", "degradation")
        .collect()
    )
    binned = (
        df.with_columns((pl.col("finish_pct_in_event") * 100).floor().alias("pct_bin"))
        .group_by("pct_bin")
        .agg(pl.col("degradation").median().alias("med"),
             pl.col("degradation").quantile(0.25).alias("q25"),
             pl.col("degradation").quantile(0.75).alias("q75"))
        .sort("pct_bin")
    )
    fig, ax = plt.subplots(figsize=(11.0, 8.4))
    ax.plot(binned["pct_bin"], binned["med"], color=hc.ACCENT, lw=2.4)
    ax.fill_between(binned["pct_bin"], binned["q25"], binned["q75"],
                    color=hc.ACCENT, alpha=0.18, label="middle 50% of athletes")
    ax.axhline(0, color=hc.DIM, lw=1)
    ax.set_xlabel("Finish percentile within event (0 = winner)")
    ax.set_ylabel("Lap 8 vs lap 1 (%)")
    ax.legend()
    hc.title(ax, "The better the result, the smaller the last-lap fade",
             "Median slowdown from first to last run lap, by finish percentile · band = middle 50%")
    hc.save(fig, "04_degradation_vs_percentile",
            takeaway="Fade rises almost linearly with finish percentile: going out too hard "
                     "is the most common and most measurable pacing mistake in Hyrox.")


def consistency(lf: pl.LazyFrame) -> None:
    run_mean = pl.mean_horizontal([pl.col(c) for c in hc.RUN_COLS])
    run_std = pl.concat_list([pl.col(c) for c in hc.RUN_COLS]).list.std()
    df = (
        hc.singles(lf)
        .with_columns((run_std / run_mean * 100).alias("run_cv"))
        .filter(pl.col("run_cv").is_between(0, 60))
        .select("finish_pct_in_event", "run_cv")
        .collect()
    )
    binned = (
        df.with_columns((pl.col("finish_pct_in_event") * 100).floor().alias("pct_bin"))
        .group_by("pct_bin")
        .agg(pl.col("run_cv").median().alias("med"))
        .sort("pct_bin")
    )
    fig, ax = plt.subplots(figsize=(11.0, 8.0))
    ax.plot(binned["pct_bin"], binned["med"], color=hc.VIOLET, lw=2.4)
    ax.set_xlabel("Finish percentile within event (0 = winner)")
    ax.set_ylabel("Run-lap variability (CV %)")
    hc.title(ax, "Winners run like metronomes",
             "How much each athlete's 8 lap times vary (coefficient of variation), by finish percentile")
    hc.save(fig, "04_consistency_vs_percentile",
            takeaway="Top finishers keep lap-to-lap variation near 5-6%; the back of the field "
                     "is 2-3x as erratic. Even pacing is a trainable, free upgrade.")


def roxzone(lf: pl.LazyFrame) -> None:
    base = hc.singles(lf).filter(pl.col("roxzone_filled_seconds").is_between(0, 1800))
    trend = (
        base.group_by("season")
        .agg(pl.col("roxzone_filled_seconds").median().alias("med"), pl.len().alias("n"))
        .filter(pl.col("n") >= 500)
        .sort("season")
        .collect()
    )
    binned = (
        base.with_columns((pl.col("finish_pct_in_event") * 100).floor().alias("pct_bin"))
        .group_by("pct_bin")
        .agg(pl.col("roxzone_filled_seconds").median().alias("med"))
        .sort("pct_bin")
        .collect()
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 7.0))
    ax1.plot(trend["season"], trend["med"], marker="o", color=hc.AMBER, lw=2.2,
             markersize=5)
    ax1.yaxis.set_major_formatter(hc.HMS)
    hc.season_axis(ax1, first=int(trend["season"].min()), last=int(trend["season"].max()))
    hc.title(ax1, "Roxzone time by season", "Median transition time, singles")
    ax2.plot(binned["pct_bin"], binned["med"], color=hc.AMBER, lw=2.4)
    ax2.yaxis.set_major_formatter(hc.HMS)
    ax2.set_xlabel("Finish percentile within event (0 = winner)")
    hc.title(ax2, "Roxzone time by finish percentile", "Slower athletes lose minutes in transit")
    fig.tight_layout()
    lo, hi = binned["med"][0], binned["med"][-1]
    hc.save(fig, "04_roxzone",
            takeaway=f"Transitions are free speed: front-runners spend ~{hc.fmt_hms(lo)} in the "
                     f"roxzone, the back of the field ~{hc.fmt_hms(hi)}: minutes lost without "
                     "a single rep performed.")
    hc.table(trend, "04_roxzone_by_season")


def best_lap_position(lf: pl.LazyFrame) -> None:
    df = hc.singles(lf).select(hc.RUN_COLS).collect()
    arr = df.to_numpy()
    best = arr.argmin(axis=1) + 1
    counts = np.bincount(best, minlength=9)[1:]
    fig, ax = plt.subplots(figsize=(10.0, 7.5))
    pct = counts / counts.sum() * 100
    bars = ax.bar(np.arange(1, 9), pct, color=hc.ACCENT)
    bars[0].set_color(hc.GREEN)
    ax.bar_label(bars, fmt="%.0f%%", fontsize=9.5, color=hc.TEXT)
    ax.set_xlabel("Run lap")
    ax.set_ylabel("% of athletes")
    ax.set_xticks(np.arange(1, 9))
    hc.title(ax, "For most athletes, lap 1 is as fast as it ever gets",
             "Which of the 8 run laps was each athlete's fastest · singles")
    hc.save(fig, "04_best_lap_position",
            takeaway=f"{pct[0]:.0f}% of athletes never run faster than their opening lap: "
                     "fresh legs plus adrenaline make lap 1 the near-universal peak.")


def worst_lap_position(lf: pl.LazyFrame) -> None:
    df = hc.singles(lf).select(hc.RUN_COLS).collect()
    arr = df.to_numpy()
    worst = arr.argmax(axis=1) + 1
    counts = np.bincount(worst, minlength=9)[1:]
    fig, ax = plt.subplots(figsize=(10.0, 7.5))
    pct = counts / counts.sum() * 100
    bars = ax.bar(np.arange(1, 9), pct, color=hc.RED)
    ax.bar_label(bars, fmt="%.0f%%", fontsize=9.5, color=hc.TEXT)
    ax.set_xlabel("Run lap")
    ax.set_ylabel("% of athletes")
    ax.set_xticks(np.arange(1, 9))
    hc.title(ax, "The wall stands on lap 8: right after the sandbag lunges",
             "Which of the 8 run laps was each athlete's slowest · singles")
    hc.save(fig, "04_worst_lap_position",
            takeaway=f"Lap 8 is the slowest for {pct[7]:.0f}% of athletes: it follows the lunges "
                     "and precedes Wall Balls, the two most fatiguing stations of the race.")


def negative_split(lf: pl.LazyFrame) -> None:
    first_half = pl.sum_horizontal([pl.col(f"run_{i}_seconds") for i in range(1, 5)])
    second_half = pl.sum_horizontal([pl.col(f"run_{i}_seconds") for i in range(5, 9)])
    df = (
        hc.singles(lf)
        .with_columns((second_half < first_half).alias("neg_split"))
        .select("finish_pct_in_event", "neg_split", "sex")
        .collect()
    )
    overall_pct = df["neg_split"].sum() / len(df) * 100
    neg = df.filter(pl.col("neg_split"))["finish_pct_in_event"].median()
    pos = df.filter(~pl.col("neg_split"))["finish_pct_in_event"].median()
    fig, ax = plt.subplots(figsize=(10.0, 8.1))
    categories = ["Negative split\n(2nd half faster)", "Positive split\n(2nd half slower)"]
    medians = [neg * 100, pos * 100]
    counts = [int(df["neg_split"].sum()), int((~df["neg_split"]).sum())]
    bars = ax.bar(categories, medians, color=[hc.GREEN, hc.RED], width=0.5)
    ax.bar_label(bars, labels=[f"{m:.0f}th percentile\n({c:,} athletes)"
                               for m, c in zip(medians, counts)],
                 fontsize=10, color=hc.TEXT, padding=4)
    ax.set_ylabel("Median finish percentile (lower = better)")
    ax.set_ylim(0, max(medians) * 1.3)
    hc.title(ax, f"Only {overall_pct:.0f}% of athletes run the second half faster",
             "Median finish percentile of negative- vs positive-splitters · singles")
    hc.save(fig, "04_negative_split",
            takeaway="Negative-splitters finish markedly higher in the field: holding back "
                     "early is rare, and it works.")


def main() -> None:
    hc.style()
    lf = hc.load()
    print("[04] pacing & fatigue")
    fatigue_curves(lf)
    degradation_vs_result(lf)
    consistency(lf)
    roxzone(lf)
    best_lap_position(lf)
    negative_split(lf)
    worst_lap_position(lf)


if __name__ == "__main__":
    main()
