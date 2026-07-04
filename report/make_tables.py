"""Generate booktabs LaTeX table fragments from the analysis output CSVs.

Run with:  ../analysis/.venv/bin/python make_tables.py
Writes fragments into report/tables/ which report.tex \\input's.
"""

import re
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TABDIR = ROOT / "analysis" / "output" / "tables"
OUT = HERE / "tables"
OUT.mkdir(exist_ok=True)

LATEX_ESC = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def esc(s) -> str:
    return "".join(LATEX_ESC.get(ch, ch) for ch in str(s))


def hms(v) -> str:
    s = int(round(float(v)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def write(name: str, body: str) -> None:
    (OUT / f"{name}.tex").write_text(body, encoding="utf-8")
    print(f"  {name}.tex")


def tabular(colspec: str, header: list[str], rows: list[list[str]]) -> str:
    # Align each header with its column: r-columns get right-aligned headers,
    # c-columns centred, so headers sit over their numbers instead of floating left.
    cells = []
    for align, h in zip(colspec, header):
        cell = f"\\textbf{{{h}}}"
        if align in "rc":
            cell = f"\\multicolumn{{1}}{{{align}}}{{{cell}}}"
        cells.append(cell)
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
             " & ".join(cells) + r" \\", "\\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def season_summary() -> None:
    lf = pl.scan_parquet(ROOT / "dataset" / "hyrox_clean.parquet")
    df = (
        lf.group_by("season")
        .agg(
            pl.len().alias("results"),
            pl.col("athlete_key").n_unique().alias("athletes"),
            pl.col("city").n_unique().alias("cities"),
            pl.col("nationality").drop_nulls().n_unique().alias("countries"),
            # Women's share among individually-sexed athletes (M/W). Mixed
            # doubles teams carry sex "X" and are excluded, so this matches the
            # sex-split chart and the prose.
            (
                (pl.col("sex") == "W").sum()
                / (pl.col("sex").is_in(["M", "W"])).sum() * 100
            ).round(1).alias("women"),
        )
        .sort("season")
        .collect()
    )
    rows = []
    for r in df.iter_rows(named=True):
        note = " (COVID)" if r["season"] == 3 else (" (in progress)" if r["season"] == 9 else "")
        rows.append([f"S{r['season']}{note}", f"{r['results']:,}", f"{r['athletes']:,}",
                     str(r["cities"]), str(r["countries"]), f"{r['women']:.0f}\\%"])
    write("season_summary", tabular(
        "lrrrrr",
        ["Season", "Results", "Athletes (est.)", "Host cities", "Nationalities", "Women"],
        rows))


def finish_percentiles() -> None:
    df = pl.read_csv(TABDIR / "02_finish_percentiles.csv")
    rows = []
    for r in df.iter_rows(named=True):
        div = r["division_canonical"].replace("_", " ").title().replace("Pro Doubles", "Pro Dbls")
        rows.append([div, "Men" if r["sex"] == "M" else "Women", f"{r['n']:,}",
                     r["p5"], r["p25"], r["p50"], r["p75"], r["p95"]])
    write("finish_percentiles", tabular(
        "llrrrrrr",
        ["Division", "Sex", "n", "p5", "p25", "Median", "p75", "p95"],
        rows))


def station_stats() -> None:
    df = pl.read_csv(TABDIR / "03_station_stats_by_division.csv")
    segs = [c for c in df.columns if c not in ("division_canonical", "sex", "n")]
    combos = [("OPEN", "M"), ("OPEN", "W"), ("PRO", "M"), ("PRO", "W"),
              ("DOUBLES", "M"), ("DOUBLES", "W")]
    rows = []
    for seg in segs:
        row = [esc(seg)]
        for d, s in combos:
            sub = df.filter((pl.col("division_canonical") == d) & (pl.col("sex") == s))
            row.append(sub[seg][0] if len(sub) else "--")
        rows.append(row)
    write("station_stats", tabular(
        "lrrrrrr",
        ["Segment", "Open M", "Open W", "Pro M", "Pro W", "Dbls M", "Dbls W"],
        rows))


def alltime_top10() -> None:
    for sex in ["men", "women"]:
        df = pl.read_csv(TABDIR / f"06_alltime_top25_{sex}.csv").head(10)
        rows = []
        for i, r in enumerate(df.iter_rows(named=True), 1):
            rows.append([str(i), esc(r["name"]), esc(r["nationality"] or "--"),
                         f"S{r['season']}", esc(r["city"]),
                         r["division_canonical"].title(), r["finish"]])
        write(f"top10_{sex}", tabular(
            "rlllllr",
            ["\\#", "Athlete", "Nat", "Season", "City", "Division", "Time"],
            rows))


def station_bests() -> None:
    df = pl.read_csv(TABDIR / "06_station_bests.csv")
    segs = df["segment"].unique(maintain_order=True).to_list()
    rows = []
    for seg in segs:
        m = df.filter((pl.col("segment") == seg) & (pl.col("sex") == "M"))["best_plausible"][0]
        w = df.filter((pl.col("segment") == seg) & (pl.col("sex") == "W"))["best_plausible"][0]
        rows.append([esc(seg), m, w])
    write("station_bests", tabular(
        "lrr", ["Segment", "Best men", "Best women"], rows))


def run_laps_cohort() -> None:
    df = pl.read_csv(TABDIR / "04_run_laps_by_cohort.csv")
    order = ["Top 5%", "Top 25%", "Middle", "Back 20%"]
    rows = []
    for c in order:
        sub = df.filter(pl.col("cohort") == c)
        if len(sub) == 0:
            continue
        row = [esc(c)]
        for i in range(1, 9):
            row.append(hms(sub[f"run_{i}_seconds"][0]))
        rows.append(row)
    write("run_laps_cohort", tabular(
        "lrrrrrrrr",
        ["Cohort"] + [f"Lap {i}" for i in range(1, 9)],
        rows))


def improvement() -> None:
    df = pl.read_csv(TABDIR / "07_improvement_by_race_number.csv")
    rows = []
    for r in df.iter_rows(named=True):
        rows.append([str(r["race_no"]), f"{r['med']:+.1f}", f"{r['q25']:+.1f}",
                     f"{r['q75']:+.1f}", f"{r['n']:,}"])
    write("improvement", tabular(
        "crrrr",
        ["Race \\#", "Median $\\Delta$ (min)", "q25", "q75", "Athletes"],
        rows))


def retention() -> None:
    df = pl.read_csv(TABDIR / "07_retention.csv")
    rows = [[f"S{r['season']} $\\to$ S{r['season'] + 1}", f"{r['athletes']:,}",
             f"{r['returned_next']:,}", f"{r['retention_pct']:.1f}\\%"]
            for r in df.iter_rows(named=True)]
    write("retention", tabular(
        "lrrr", ["Transition", "Athletes", "Returned", "Retention"], rows))


def most_active() -> None:
    df = pl.read_csv(TABDIR / "07_most_active_athletes.csv").head(10)
    rows = []
    for r in df.iter_rows(named=True):
        rows.append([esc(r["name"]), esc(r["nationality"] or "--"), str(r["races"]),
                     r["pb"], f"S{r['first_season']}--S{r['last_season']}"])
    write("most_active", tabular(
        "llrrl", ["Athlete", "Nat", "Races", "PB", "Active"], rows))


def main() -> None:
    print("writing LaTeX fragments to report/tables/")
    season_summary()
    finish_percentiles()
    station_stats()
    alltime_top10()
    station_bests()
    run_laps_cohort()
    improvement()
    retention()
    most_active()


if __name__ == "__main__":
    main()
