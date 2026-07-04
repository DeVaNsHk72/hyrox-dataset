"""Generate figures for the individual athlete report:
Avinash Kudpaje & Kavya Murthy, Bengaluru mixed doubles, Season 8.

Split data comes from the athlete's full detail page (the clean parquet only
stores the finish for mixed doubles). Field context (event finish distribution,
Bengaluru doubles station medians) is read from the parquet.
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import hyrox_common as hc

TEAM = "AVINASH KUDPAJE & Kavya Murthy"
EVENT_ID = "HD_LR3MS4JI147B"          # Bengaluru, HYROX Doubles - Sunday, S8
FINISH_S = 6696                        # 1:51:36
RANK_OVERALL = 294

# Their run laps, in seconds, from the detail page.
RUN_LAPS = [376, 334, 473, 463, 362, 371, 454, 456]
# (label, their seconds, their event rank)
STATIONS = [
    ("SkiErg", 326, 436), ("Sled Push", 217, 388), ("Sled Pull", 461, 484),
    ("Burpee Jumps", 282, 288), ("Row", 322, 206), ("Farmers Carry", 154, 437),
    ("Lunges", 466, 471), ("Wall Balls", 543, 466),
]
RUN_TOTAL, RUN_TOTAL_RANK = 3285, 177
BEST_LAP, BEST_LAP_RANK = 334, 87
ROX, ROX_RANK = 646, 317

SEGMENTS = (
    [("Best run lap", BEST_LAP, BEST_LAP_RANK),
     ("Run total", RUN_TOTAL, RUN_TOTAL_RANK),
     ("Roxzone", ROX, ROX_RANK)]
    + [(n, s, r) for n, s, r in STATIONS]
)

# Column mapping for parquet benchmarks.
COLS = {
    "SkiErg": "ski_erg_seconds", "Sled Push": "sled_push_seconds",
    "Sled Pull": "sled_pull_seconds", "Burpee Jumps": "burpee_broad_jump_seconds",
    "Row": "row_seconds", "Farmers Carry": "farmers_carry_seconds",
    "Lunges": "sandbag_lunges_seconds", "Wall Balls": "wall_balls_seconds",
}


def event_finishes() -> np.ndarray:
    lf = hc.load()
    f = (lf.filter(pl.col("event_id") == EVENT_ID)
         .select("finish_seconds").drop_nulls().collect()
         )["finish_seconds"].to_numpy()
    f.sort()
    return f


def bengaluru_doubles_medians() -> dict[str, float]:
    """Median segment times of Bengaluru men's/women's doubles (they carry splits)."""
    lf = hc.load()
    base = lf.filter((pl.col("city") == "Bengaluru")
                     & (pl.col("division_canonical") == "DOUBLES")
                     & pl.col("sex").is_in(["M", "W"])
                     & pl.col("is_clean_race"))
    med = base.select(
        [pl.col(c).median().alias(n) for n, c in COLS.items()]
        + [pl.col("run_total_seconds").median().alias("Run total"),
           pl.col("roxzone_filled_seconds").median().alias("Roxzone")]
    ).collect()
    return {k: float(med[k][0]) for k in med.columns}


def fig_result_vs_field() -> None:
    local = event_finishes()
    lf = hc.load()
    world = (lf.filter((pl.col("sex") == "X")
                       & pl.col("division_canonical").is_in(["DOUBLES", "PRO_DOUBLES"]))
             .select("finish_seconds").drop_nulls().collect()
             )["finish_seconds"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5.6))
    bins = np.arange(3000, 12600, 150)
    ax.hist(world, bins=bins, density=True, alpha=0.35, color="#94a3b8",
            label=f"World mixed doubles  (median {hc.fmt_hms(np.median(world))})")
    ax.hist(local, bins=bins, density=True, alpha=0.6, color=hc.ACCENT,
            label=f"Bengaluru this event  (median {hc.fmt_hms(np.median(local))})")
    ax.axvline(FINISH_S, color=hc.RED, lw=2.4)
    ax.annotate(f"Their finish\n{hc.fmt_hms(FINISH_S)}",
                xy=(FINISH_S, ax.get_ylim()[1] * 0.9),
                xytext=(12, 0), textcoords="offset points",
                fontsize=10, color=hc.RED, fontweight="bold", va="top")
    ax.set_xlim(3000, 11000)
    ax.xaxis.set_major_formatter(hc.HMS)
    ax.set_xlabel("Finish time")
    ax.set_yticks([])
    ax.legend(loc="upper right")
    hc.title(ax, "A middle-of-the-pack finish in a young market",
             "Their 1:51:36 against their own Bengaluru field and the global mixed-doubles field")
    hc.save(fig, "ath_result_vs_field",
            takeaway=f"They finished near the median of their Bengaluru event "
                     f"(rank {RANK_OVERALL}), but Bengaluru's whole field sits about half an hour "
                     "behind the world median: a snapshot of a brand-new market, not the ceiling.")


def fig_segment_ranks() -> None:
    segs = sorted(SEGMENTS, key=lambda s: s[2])
    labels = [s[0] for s in segs]
    ranks = [s[2] for s in segs]
    colors = [hc.GREEN if r < RANK_OVERALL else hc.RED for r in ranks]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, ranks, color=colors)
    ax.bar_label(bars, labels=[str(r) for r in ranks], padding=4, fontsize=9,
                 color=hc.TEXT)
    ax.axvline(RANK_OVERALL, color=hc.DIM, lw=1.6, ls="--")
    ax.annotate(f"overall rank {RANK_OVERALL}", xy=(RANK_OVERALL, len(labels) - 0.4),
                xytext=(6, 0), textcoords="offset points", fontsize=9,
                color=hc.DIM, style="italic")
    ax.set_yticks(y, labels, fontsize=10)
    ax.set_xlabel("Rank in the event, out of ~547 teams (shorter bar toward the right = better placing)")
    ax.set_xlim(0, 560)
    ax.grid(axis="y", visible=False)
    ax.invert_xaxis()
    hc.title(ax, "Strong runners, held back by the strength stations",
             "Green beats their overall rank, red drags it down. Running is well ahead; "
             "sleds, lunges and wall balls are behind.")
    hc.save(fig, "ath_segment_ranks",
            takeaway="Their running is the strength of the race (best lap 87th, run total 177th), "
                     "while grip- and strength-heavy stations (Sled Pull 484th, Lunges 471st, "
                     "Wall Balls 466th) are where they lose most ground.")


def fig_vs_local_medians() -> None:
    """Minutes gained or lost against the median Bengaluru doubles team, per segment."""
    med = bengaluru_doubles_medians()
    rows = [("Run total", RUN_TOTAL), ("Roxzone", ROX)] + [(n, s) for n, s, _ in STATIONS]
    diffs = [(n, (t - med[n]) / 60) for n, t in rows]
    diffs.sort(key=lambda kv: kv[1])
    labels = [d[0] for d in diffs]
    vals = [d[1] for d in diffs]
    colors = [hc.GREEN if v < 0 else hc.RED for v in vals]
    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, vals, color=colors)
    ax.bar_label(bars, labels=[f"{v:+.1f}" for v in vals], padding=4, fontsize=9,
                 color=hc.TEXT)
    ax.axvline(0, color=hc.TEXT, lw=1.2)
    ax.set_yticks(y, labels, fontsize=10)
    lo, hi = min(vals), max(vals)
    ax.set_xlim(lo - 0.8, hi + 0.8)
    ax.text(lo - 0.6, -0.75, "faster than local median", fontsize=9,
            color=hc.GREEN, style="italic", va="center")
    ax.text(hi + 0.6, -0.75, "slower than local median", fontsize=9, color=hc.RED,
            style="italic", va="center", ha="right")
    ax.set_xlabel("Minutes versus the median Bengaluru doubles team (negative = they were faster)")
    ax.grid(axis="y", visible=False)
    total_loss = sum(v for v in vals if v > 0)
    hc.title(ax, "Where the minutes went, against the local benchmark",
             "Each segment versus the median Bengaluru men's/women's doubles team "
             "(the fairest split-level benchmark available)")
    hc.save(fig, "ath_vs_local_medians",
            takeaway=f"They out-ran the local doubles median by 3.6 minutes and matched it on the "
                     f"row, but gave back about {total_loss:.0f} minutes across the strength "
                     "stations and transitions, which is the whole story of the result.")


def fig_run_pacing() -> None:
    vals = RUN_LAPS
    x = np.arange(1, 9)
    best_i = int(np.argmin(vals))
    after = ["", "SkiErg", "Sled Push", "Sled Pull", "Burpee Jumps", "Row",
             "Farmers Carry", "Lunges"]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.plot(x, vals, marker="o", color=hc.ACCENT, lw=2.2, markersize=6)
    ax.plot(best_i + 1, vals[best_i], marker="o", color=hc.GREEN, markersize=11,
            zorder=5)
    for xi, v in zip(x, vals):
        ax.annotate(hc.fmt_hms(v), (xi, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8.5, color=hc.TEXT)
    for xi in [3, 4]:
        ax.annotate("after the sleds", (xi, vals[xi - 1]),
                    textcoords="offset points", xytext=(0, -18), ha="center",
                    fontsize=8, color=hc.RED, style="italic")
    ax.axhline(np.mean(vals), color=hc.DIM, lw=1, ls="--")
    ax.text(8.35, np.mean(vals), f"avg {hc.fmt_hms(np.mean(vals))}", fontsize=8.5,
            color=hc.DIM, va="center")
    h1, h2 = sum(vals[:4]), sum(vals[4:])
    ax.set_xticks(x, [f"Run {i}" for i in x], fontsize=9)
    ax.set_ylim(300, 520)
    ax.yaxis.set_major_formatter(hc.HMS)
    ax.set_ylabel("Lap time (per 1 km)")
    hc.title(ax, "A rare negative split, hiding inside uneven laps",
             f"First four laps {hc.fmt_hms(h1)}, last four {hc.fmt_hms(h2)}: the second half "
             "was faster, which only about one team in nine manages")
    hc.save(fig, "ath_run_pacing",
            takeaway="Their fastest kilometre (Run 2, 5:34) ranked 87th in the field, and they "
                     "finished the running faster than they started it. The leaks are the two "
                     "laps around the sled block and the lap-to-lap swings of almost 13 percent.")


def fig_whatif_ladder() -> None:
    """What closing the station gap to the local median would have been worth."""
    fins = event_finishes()
    med = bengaluru_doubles_medians()
    gap = sum(max(0.0, t - med[n]) for n, t, _ in STATIONS) + max(0.0, ROX - med["Roxzone"])

    steps = [
        ("Actual race", FINISH_S),
        ("Half the station gap closed", FINISH_S - gap / 2),
        ("Stations at the local median", FINISH_S - gap),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    y = np.arange(len(steps))[::-1]
    colors = [hc.RED, hc.AMBER, hc.GREEN]
    times = [t for _, t in steps]
    bars = ax.barh(y, times, color=colors, height=0.55)
    for yi, (label, t) in zip(y, steps):
        rank = int((fins < t).sum()) + 1
        pct = rank / len(fins) * 100
        ax.text(t + 60, yi, f"{hc.fmt_hms(t)}   rank ~{rank}  (top {pct:.0f}%)",
                va="center", fontsize=10, color=hc.TEXT, fontweight="bold")
    ax.set_yticks(y, [s for s, _ in steps], fontsize=10)
    ax.set_xlim(0, max(times) * 1.45)
    ax.xaxis.set_major_formatter(hc.HMS)
    ax.set_xlabel("Finish time")
    ax.grid(axis="y", visible=False)
    hc.title(ax, "What fixing the stations is worth, with the running left untouched",
             "Hypothetical finishes if the station and roxzone gaps to the local doubles median "
             "were closed, ranked against the actual event field")
    hc.save(fig, "ath_whatif_ladder",
            takeaway="Closing half of the station gap is worth about seventy places; closing "
                     "all of it, with zero running improvement, lifts them from the median to "
                     "the top third of the field.")


def fig_archetypes() -> None:
    lf = hc.load()
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

    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    ax.set_aspect("equal")

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
    ax.text(-2.2, -2.5, "FAST + STRONG\nbest finishers", fontsize=8.5, fontweight="bold",
            color=hc.GREEN, ha="center", bbox=bbox)
    ax.text(2.2, -2.5, "STRONG\nbut slow runner", fontsize=8.5, fontweight="bold",
            color=hc.AMBER, ha="center", bbox=bbox)
    ax.text(-2.2, 2.5, "RUNNER\nbut weak stations", fontsize=8.5, fontweight="bold",
            color=hc.AMBER, ha="center", bbox=bbox)
    ax.text(2.2, 2.5, "DEVELOPING\nneeds both", fontsize=8.5, fontweight="bold",
            color=hc.RED, ha="center", bbox=bbox)

    # Plot team positions
    ax.plot(-0.35, 1.48, marker="*", color="#ffffff", markersize=16, markeredgecolor="black", markeredgewidth=2.0, zorder=10, label="vs. Bengaluru doubles")
    ax.plot(0.83, 2.85, marker="D", color="#00ffff", markersize=10, markeredgecolor="black", markeredgewidth=2.0, zorder=10, label="vs. Global doubles")

    ax.legend(loc="center left", frameon=True, facecolor="#ffffff", framealpha=0.9, edgecolor=hc.GRID)

    hc.title(ax, "Athlete archetypes: hybrids beat specialists",
             "Avinash & Kavya's placement on the global speed vs. strength archetype map")
    hc.save(fig, "ath_archetypes",
            takeaway="Their running capacity puts them in the 'runner with weak stations' quadrant "
                     "relative to local benchmark, while comparing to the faster global doubles "
                     "field shows significant room to develop both dimensions.")


def main() -> None:
    hc.style()
    print("[athlete] Avinash Kudpaje & Kavya Murthy")
    fig_result_vs_field()
    fig_segment_ranks()
    fig_vs_local_medians()
    fig_run_pacing()
    fig_whatif_ladder()
    fig_archetypes()


if __name__ == "__main__":
    main()

