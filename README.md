# HYROX Results — Dataset, Scraper & Analysis

The most complete open dataset of HYROX race results: **1.48M+ individual race
results** (2018–2026, all 9 seasons) with **full per-station splits, per-station
place, and granular Roxzone timing** — scraped from
[results.hyrox.com](https://results.hyrox.com).

This repo contains the scraper that built it, the analysis pipeline, and the
generated report. **The dataset files themselves are hosted separately** (see
[Getting the data](#getting-the-data)) since they're too large for git.

## Dataset at a glance

| | |
|---|---|
| Unique race results | **1,478,643** |
| Unique competitor IDs | ~1,024,000 |
| Seasons covered | 1–9 (2018 → 2026, in progress) |
| Host cities | 145+ |
| Nationalities | 220+ |
| Columns | 68 (identity, result, all 8 runs + 8 stations × time & place, Roxzone, `*_seconds` numeric companions) |

Per-season breakdown (from `analysis/output/tables/01_season_summary.csv`):

| Season | Results | Cities | Women |
|---|---:|---:|---:|
| S1 (2018) | 5,254 | 9 | 33% |
| S2 | 7,646 | 12 | 35% |
| S3 (2020, COVID) | 986 | 4 | 35% |
| S4 | 20,399 | 23 | 37% |
| S5 | 45,070 | 35 | 36% |
| S6 | 111,652 | 50 | 37% |
| S7 | 302,791 | 66 | 39% |
| S8 (2025) | 679,500 | 102 | 42% |
| S9 (2026, in progress) | 8,113 | 2 | 45% |

## Known limitation

**Mixed-category results (Doubles/Relay teams registered as "Mixed") are
under-represented in the dataset above.** `results.hyrox.com` filters athletes
by `M` / `W` / **`X` (Mixed)**, and the scraper originally only queried `M` and
`W`. This has since been **fixed** (`hyrox_scraper.py` now loops all three), but
the published numbers above were collected before the fix, and a full
Mixed-category backfill is in progress. If you need Mixed Doubles/Relay results
specifically, re-run the scraper with the current code, or check back for an
updated release.

## What's in this repo

| Path | What it is |
|---|---|
| `hyrox_scraper.py` | The scraper — full per-athlete splits, resumable, rate-limited |
| `dist_worker.py` / `seed_queue.py` | Distributed scraping via a Redis pull-queue (many workers, no manual sharding) |
| `hyrox_colab.ipynb` | Turnkey Google Colab worker notebook |
| `DISTRIBUTED.md` / `DISTRIBUTED_QUEUE.md` | Distributed-scraping setup guides |
| `DATA_DICTIONARY.md` | Every column in the dataset, explained |
| `analysis/` | Full analysis pipeline (see below) — participation, finish times, station performance, pacing, elites, athlete-level reports |
| `report/` | Generated LaTeX/PDF report (`report.pdf`) built from the analysis output |

## Analysis pipeline

`analysis/` runs as a numbered sequence over the combined dataset:

| Script | Covers |
|---|---|
| `00_prepare_data.py` | Load + clean the combined dataset |
| `01_participation.py` | Growth, geography, nationality, age/gender mix over time |
| `02_finish_times.py` | Finish-time distributions & percentiles by division/season |
| `03_stations.py` | Per-station medians, sex gap, elite-vs-field gap |
| `04_pacing.py` | Run 1→8 fatigue curves, Roxzone trends, negative-splitting |
| `05_performance_drivers.py` | What predicts finish time; archetypes; correlations |
| `06_elites.py` | All-time top 25s, winning-time trends, station records |
| `07_athletes.py` | Multi-race athletes: retention, PB rate, career trajectories |
| `08_deep_dive.py` | Extras: completion rates, India-vs-world, doubles-vs-solo |

Each writes tables to `analysis/output/tables/` and figures to
`analysis/output/figures/`. `report/build_report.py` (+ `report/report.tex`)
assembles the final PDF at `report/report.pdf`.

### A few findings already in the data

- **Roxzone (transition) time has been shrinking** as the sport matures — season
  median dropped from ~470s (S1) to ~400s (S6) even as fields ballooned 15×.
- **Year-over-year retention is ~38–40%** — roughly 2 in 5 athletes who race one
  season come back the next.
- **Top nationalities by results:** GBR (110.5k), USA (72.1k), GER (56.0k),
  FRA (56.0k), NED (40.8k).
- Full station-level medians by division/sex are in
  `analysis/output/tables/03_station_stats_by_division.csv`.

### Reproducing the analysis

```bash
cd analysis
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy matplotlib pyarrow
python 00_prepare_data.py      # point this at your local hyrox_results.parquet
python 01_participation.py
...                            # 02 through 08
python build_report.py         # -> ../report/report.pdf (requires a LaTeX install)
```

## Getting the data

The combined dataset (`hyrox_results.parquet/.csv`, `hyrox_granular_timing.parquet`,
`DATA_DICTIONARY.md`) is **not** committed here — it's several GB and git isn't
built for that. It's published at: **[add Kaggle/HuggingFace link here once
uploaded]**. Until then, reproduce it yourself:

```bash
pip install requests beautifulsoup4 lxml playwright pandas pyarrow
playwright install chromium

python hyrox_scraper.py --season 8       # scrape one season (resumable)
python hyrox_scraper.py                  # all seasons (1-9)
python hyrox_scraper.py --combine        # merge -> hyrox_full.*
```

See `DISTRIBUTED.md` / `DISTRIBUTED_QUEUE.md` for running many workers in
parallel (Colab, cloud VMs, a Redis-backed queue) — a full re-scrape from
scratch is a multi-day job on one machine.

## Notes on use

- Data is sourced from publicly displayed race results. It includes personal
  names — treat it like any public race-results dataset and follow HYROX's
  terms and applicable privacy rules when redistributing.
- Please keep the scraper's rate limiting intact; don't hammer the source site.
- Not affiliated with or endorsed by HYROX.
