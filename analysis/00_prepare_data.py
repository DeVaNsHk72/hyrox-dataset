"""Build the cleaned, deduplicated analysis parquet from the raw results.

Raw data quirks handled here:
- '-' / en-dash null markers (handled at CSV->parquet conversion).
- 'X - Overall' divisions fully duplicate the day-level rows of the same
  event -> dedupe on (season, city, idp, finish_seconds).
- 127 raw division strings -> canonical categories.
"""

import re

import polars as pl

from hyrox_common import CLEAN_PARQUET, RAW_PARQUET, STATION_COLS

DAY_SUFFIX = (
    r"\s*-\s*(Overall|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
    r"|Week I{1,2}|(Mon|Tue|Wed|Thu|Fri|Sat|Sun), \d{2} \w{3})$"
)


def canonical_division() -> pl.Expr:
    base = pl.col("division").str.replace(DAY_SUFFIX, "").str.to_uppercase()
    return (
        pl.when(base.str.starts_with("HYROX PRO DOUBLES")).then(pl.lit("PRO_DOUBLES"))
        .when(base.str.starts_with("HYROX PRO")).then(pl.lit("PRO"))
        .when(base.str.starts_with("HYROX GORUCK DOUBLES")).then(pl.lit("DOUBLES"))
        .when(base.str.starts_with("HYROX DOUBLES")).then(pl.lit("DOUBLES"))
        .when(base.str.contains("ELITE") & base.str.contains("DOUBLES"))
        .then(pl.lit("ELITE_DOUBLES"))
        .when(base.str.starts_with("HYROX ELITE")).then(pl.lit("ELITE"))
        .when(base.str.contains("ELITE")).then(pl.lit("ELITE"))
        .when(base.str.contains("RELAY|COMPANY CHALLENGE")).then(pl.lit("RELAY"))
        .when(base.str.starts_with("HYROX ADAPTIVE")).then(pl.lit("ADAPTIVE"))
        .when(base.str.starts_with("HYROX YOUNGSTARS")).then(pl.lit("YOUNGSTARS"))
        .when(base.str.starts_with("HYROX GORUCK")).then(pl.lit("GORUCK"))
        .when(base == "HYROX").then(pl.lit("OPEN"))
        .otherwise(pl.lit("OTHER"))
        .alias("division_canonical")
    )


def main() -> None:
    lf = pl.scan_parquet(RAW_PARQUET)

    # Dedup key collides across listings of the same entry (e.g. a day listing
    # that carries splits and an "Overall"/team listing that does not). Prefer
    # the copy WITH split times so we never discard a complete record in favour
    # of a split-less duplicate. Sort split-having rows first, then keep first.
    lf = (
        lf.with_columns(pl.col("run_1_seconds").is_not_null().alias("_has_splits"))
        .sort("_has_splits", descending=True)
        .unique(subset=["season", "city", "idp", "finish_seconds"],
                keep="first", maintain_order=True)
        .drop("_has_splits")
    )
    lf = lf.with_columns(
        pl.col("season").cast(pl.Int32, strict=False),
        pl.col("year").cast(pl.Int32, strict=False),
        canonical_division()
    )

    # idp is unique per event entry, NOT a stable athlete id: career tracking
    # must use a name-based key instead (imperfect: same-name athletes merge).
    lf = lf.with_columns(
        pl.when(pl.col("name").is_not_null())
        .then(pl.col("name").str.to_lowercase().str.strip_chars() + "|" + pl.col("sex"))
        .otherwise(None)
        .alias("athlete_key")
    )

    stations_total = pl.sum_horizontal([pl.col(c) for c in STATION_COLS])
    lf = lf.with_columns(
        stations_total.alias("stations_total_seconds"),
        (stations_total + pl.col("run_total_seconds")).alias("work_seconds"),
    )
    # Roxzone: prefer the recorded value, else derive from finish - work.
    lf = lf.with_columns(
        pl.coalesce(
            pl.col("roxzone_seconds"),
            pl.col("finish_seconds") - pl.col("work_seconds"),
        ).alias("roxzone_filled_seconds")
    )

    # A "clean race": plausible finish, all stations and runs timed, sane splits.
    all_segments_ok = pl.all_horizontal(
        [pl.col(c).is_between(15, 3600) for c in STATION_COLS]
        + [pl.col(f"run_{i}_seconds").is_between(60, 3600) for i in range(1, 9)]
    )
    lf = lf.with_columns(
        (
            pl.col("finish_seconds").is_between(1800, 14400)
            & all_segments_ok
            & pl.col("roxzone_filled_seconds").is_between(0, 3600)
        )
        .fill_null(False)
        .alias("is_clean_race")
    )

    # Field-relative percentile within one event+division+sex (0 = winner).
    grp = ["season", "city", "event_id", "division_canonical", "sex"]
    lf = lf.with_columns(
        (
            (pl.col("finish_seconds").rank("average").over(grp) - 1)
            / (pl.len().over(grp) - 1).clip(lower_bound=1)
        ).alias("finish_pct_in_event")
    )

    lf.sink_parquet(CLEAN_PARQUET)

    check = pl.scan_parquet(CLEAN_PARQUET)
    n = check.select(pl.len()).collect().item()
    clean = check.filter(pl.col("is_clean_race")).select(pl.len()).collect().item()
    print(f"wrote {CLEAN_PARQUET.name}: {n:,} unique results, {clean:,} clean-race rows")
    print(
        check.group_by("division_canonical").len().sort("len", descending=True).collect()
    )


if __name__ == "__main__":
    main()
