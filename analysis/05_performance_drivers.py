"""What drives a Hyrox result: correlations, variance, archetypes, ratios, consistency."""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import hyrox_common as hc

SEGMENTS = hc.STATION_COLS + ["run_total_seconds", "roxzone_filled_seconds"]
SEG_LABELS = {
    **hc.STATION_LABELS,
    "run_total_seconds": "Running",
    "roxzone_filled_seconds": "Roxzone",
    "finish_seconds": "FINISH",
}


def sample(lf: pl.LazyFrame, cols: list[str], n: int = 300_000) -> pl.DataFrame:
    df = hc.singles(lf).select(cols).drop_nulls().collect()
    if len(df) > n:
        df = df.sample(n, seed=42)
    return df


def correlation_heatmap(lf: pl.LazyFrame) -> None:
    cols = SEGMENTS + ["finish_seconds"]
    df = sample(lf, cols)
    mat = np.corrcoef(df.to_numpy().T)
    labels = [SEG_LABELS[c] for c in cols]
    fig, ax = plt.subplots(figsize=(10.0, 12.3))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.grid(visible=False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=7.5, color="white" if abs(mat[i, j]) > 0.55 else "#1e293b")
    cb = fig.colorbar(im, shrink=0.8)
    cb.ax.yaxis.set_tick_params(color=hc.DIM)
    plt.setp(cb.ax.get_yticklabels(), color=hc.DIM)
    hc.title(ax, "Everything correlates: Hyrox punishes weaknesses everywhere",
             "Pearson correlation between segment times, Open+Pro singles · 1.0 = move in lockstep")
    hc.save(fig, "05_correlation_heatmap",
            takeaway="No segment is independent: if you're slow anywhere, you're likely slow "
                     "everywhere. The bottom row shows what predicts the finish best.")


def finish_correlations(lf: pl.LazyFrame) -> None:
    cols = SEGMENTS + ["finish_seconds"]
    df = sample(lf, cols)
    fin = df["finish_seconds"].to_numpy()
    corrs = {SEG_LABELS[c]: np.corrcoef(df[c].to_numpy(), fin)[0, 1] for c in SEGMENTS}
    items = sorted(corrs.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(10.5, 8.7))
    colors = [hc.GREEN if v == max(corrs.values()) else hc.ACCENT for _, v in items]
    bars = ax.barh([k for k, _ in items], [v for _, v in items], color=colors)
    ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=4, color=hc.TEXT)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Correlation with finish time")
    ax.grid(axis="y", visible=False)
    hc.title(ax, "Running is the single best predictor of your result",
             "Correlation between each segment's time and the final finish time · Open+Pro singles")
    hc.save(fig, "05_finish_correlations",
            takeaway="Total running time correlates ~0.95 with the finish: if you can only "
                     "train one thing for a better Hyrox time, train running endurance.")
    hc.table(
        pl.DataFrame({"segment": [k for k, _ in items], "r_with_finish": [v for _, v in items]}),
        "05_finish_correlations",
    )


def variance_contribution(lf: pl.LazyFrame) -> None:
    df = sample(lf, SEGMENTS)
    stds = {SEG_LABELS[c]: float(np.std(df[c].to_numpy())) for c in SEGMENTS}
    items = sorted(stds.items(), key=lambda kv: kv[1], reverse=True)
    fig, ax = plt.subplots(figsize=(11.0, 8.4))
    colors = [hc.AMBER if i == 0 else hc.VIOLET for i in range(len(items))]
    bars = ax.bar([k for k, _ in items], [v for _, v in items], color=colors)
    ax.bar_label(bars, labels=[hc.fmt_hms(v) for _, v in items], fontsize=8.5,
                 color=hc.TEXT)
    ax.set_ylabel("Std deviation across the field")
    ax.yaxis.set_major_formatter(hc.HMS)
    plt.setp(ax.get_xticklabels(), fontsize=9)
    hc.title(ax, "Where races are won and lost, in raw minutes",
             "Standard deviation of each segment across all athletes: bigger spread = more time up for grabs")
    hc.save(fig, "05_variance_by_segment",
            takeaway="Running spreads athletes by the most total time by far; among stations, "
                     "Wall Balls and Lunges offer the biggest gains over rivals.")


def archetypes(lf: pl.LazyFrame) -> None:
    df = (
        hc.singles(lf)
        .select("sex", "run_total_seconds", "stations_total_seconds", "finish_pct_in_event")
        .drop_nulls()
        .collect()
    )
    df = df.with_columns([
        ((pl.col(c) - pl.col(c).mean().over("sex")) / pl.col(c).std().over("sex")).alias(f"{c}_z")
        for c in ["run_total_seconds", "stations_total_seconds"]
    ])
    s = df.sample(min(len(df), 150_000), seed=1)
    fig, ax = plt.subplots(figsize=(10.5, 12.3))
    hb = ax.hexbin(s["run_total_seconds_z"], s["stations_total_seconds_z"],
                   C=s["finish_pct_in_event"] * 100, gridsize=40, cmap="RdYlGn_r",
                   extent=(-3, 3, -3, 3), mincnt=15)
    cb = fig.colorbar(hb, shrink=0.8)
    cb.set_label("Mean finish percentile (lower = better)", fontsize=9, color=hc.DIM)
    cb.ax.yaxis.set_tick_params(color=hc.DIM)
    plt.setp(cb.ax.get_yticklabels(), color=hc.DIM)
    ax.axhline(0, color=hc.TEXT, lw=0.8, alpha=0.4)
    ax.axvline(0, color=hc.TEXT, lw=0.8, alpha=0.4)
    ax.grid(visible=False)
    ax.set_xlabel("Running z-score   (left = faster runner, right = slower)")
    ax.set_ylabel("Stations z-score   (lower = stronger at stations)")
    bbox = dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor=hc.GRID, alpha=0.92)
    ax.text(-2.2, -2.5, "FAST + STRONG\nbest finishers", fontsize=9.5, fontweight="bold",
            color=hc.GREEN, ha="center", bbox=bbox)
    ax.text(2.2, -2.5, "STRONG\nbut slow runner", fontsize=9.5, fontweight="bold",
            color=hc.AMBER, ha="center", bbox=bbox)
    ax.text(-2.2, 2.5, "RUNNER\nbut weak stations", fontsize=9.5, fontweight="bold",
            color=hc.AMBER, ha="center", bbox=bbox)
    ax.text(2.2, 2.5, "DEVELOPING\nneeds both", fontsize=9.5, fontweight="bold",
            color=hc.RED, ha="center", bbox=bbox)
    hc.title(ax, "Athlete archetypes: hybrids beat specialists",
             "Every athlete placed by running speed vs station strength · color = average result")
    hc.save(fig, "05_archetypes",
            takeaway="The color gradient runs diagonally: a balanced 'pretty good at both' athlete "
                     "beats a one-dimensional specialist of the same total ability.")

    imb = (
        df.with_columns(
            (pl.col("run_total_seconds_z") - pl.col("stations_total_seconds_z"))
            .abs().alias("imbalance")
        )
        .with_columns(pl.col("imbalance").round(1))
        .group_by("imbalance")
        .agg(pl.col("finish_pct_in_event").mean().alias("mean_pct"), pl.len().alias("n"))
        .filter((pl.col("n") >= 200) & (pl.col("imbalance") <= 3))
        .sort("imbalance")
    )
    fig, ax = plt.subplots(figsize=(10.0, 8.0))
    ax.plot(imb["imbalance"], imb["mean_pct"] * 100, marker="o", color=hc.RED,
            lw=2.2, markersize=5)
    ax.set_xlabel("Imbalance between running and station ability (|z difference|)")
    ax.set_ylabel("Mean finish percentile (lower = better)")
    hc.title(ax, "The specialist tax",
             "Average result vs how lopsided an athlete's run-vs-strength profile is")
    hc.save(fig, "05_imbalance_penalty",
            takeaway="Every unit of imbalance costs field position: being one-dimensional "
                     "is measurably penalized in a hybrid sport.")


def run_station_ratio(lf: pl.LazyFrame) -> None:
    df = (
        hc.singles(lf)
        .with_columns(
            (pl.col("run_total_seconds")
             / (pl.col("run_total_seconds") + pl.col("stations_total_seconds")) * 100)
            .alias("run_pct")
        )
        .filter(pl.col("run_pct").is_between(20, 80))
        .select("finish_pct_in_event", "run_pct", "sex")
        .collect()
    )
    fig, ax = plt.subplots(figsize=(11.0, 8.4))
    for sex in ["M", "W"]:
        sub = df.filter(pl.col("sex") == sex)
        binned = (
            sub.with_columns((pl.col("finish_pct_in_event") * 100).floor().alias("pct_bin"))
            .group_by("pct_bin").agg(pl.col("run_pct").median().alias("med"))
            .sort("pct_bin")
        )
        ax.plot(binned["pct_bin"], binned["med"], color=hc.SEX_COLORS[sex], lw=2.4,
                label=hc.SEX_NAMES[sex])
    ax.set_xlabel("Finish percentile within event (0 = winner)")
    ax.set_ylabel("Running as % of work time")
    ax.legend()
    hc.title(ax, "Faster athletes spend a bigger share of their race running",
             "Median run-time share of (run + station) time, by finish percentile")
    hc.save(fig, "05_run_station_ratio",
            takeaway="Top athletes' station work is so fast that running dominates their race; "
                     "for the back of the field, stations swallow the clock.")


def consistency_score(lf: pl.LazyFrame) -> None:
    all_cols = hc.RUN_COLS + hc.STATION_COLS
    df = (
        hc.singles(lf)
        .select(all_cols + ["finish_pct_in_event"])
        .drop_nulls()
        .collect()
    )
    arr = df.select(all_cols).to_numpy()
    means = arr.mean(axis=0, keepdims=True)
    stds = arr.std(axis=0, keepdims=True)
    stds[stds == 0] = 1
    z = (arr - means) / stds
    row_std = z.std(axis=1)
    result = pl.DataFrame({
        "consistency": row_std,
        "finish_pct": df["finish_pct_in_event"].to_numpy(),
    })
    binned = (
        result.with_columns((pl.col("finish_pct") * 100).floor().alias("pct_bin"))
        .group_by("pct_bin").agg(pl.col("consistency").median().alias("med"))
        .sort("pct_bin")
    )
    fig, ax = plt.subplots(figsize=(11.0, 8.0))
    ax.plot(binned["pct_bin"], binned["med"], color=hc.VIOLET, lw=2.4)
    ax.set_xlabel("Finish percentile within event (0 = winner)")
    ax.set_ylabel("Profile spread (std of 16 segment z-scores)")
    hc.title(ax, "Balanced profiles finish first",
             "How uneven each athlete's 16 segments are relative to the field, by finish percentile")
    hc.save(fig, "05_consistency_score",
            takeaway="The best athletes are uniformly good: their 16 segment times sit at "
                     "nearly the same percentile, with no glaring weak link.")


def main() -> None:
    hc.style()
    lf = hc.load()
    print("[05] performance drivers")
    correlation_heatmap(lf)
    finish_correlations(lf)
    variance_contribution(lf)
    archetypes(lf)
    run_station_ratio(lf)
    consistency_score(lf)


if __name__ == "__main__":
    main()
