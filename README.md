# HYROX Results Dataset

Open dataset of HYROX race results with **full per-station splits**, scraped from
[results.hyrox.com](https://results.hyrox.com). One row per athlete per
event / division / gender.

## What's inside

For every athlete: identity (name, nationality, age group), result (overall &
age-group rank, net + gross finish time), and the complete split breakdown —
**time _and_ place for all 8 runs and all 8 workout stations** (SkiErg, Sled Push,
Sled Pull, Burpee Broad Jumps, Row, Farmers Carry, Sandbag Lunges, Wall Balls),
plus Roxzone, Run Total and Best Run Lap. Every `*_time` has a numeric
`*_seconds` companion for analysis, and the raw In/Out timing table is preserved
in `granular_timing_json`.

See [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) for every column.

## Files

| File | Description |
| --- | --- |
| `hyrox_full.csv` / `.parquet` / `.jsonl` | The combined dataset (Parquet recommended for analysis) |
| `DATA_DICTIONARY.md` | Column-by-column reference |
| `by_event/` | Per-event source CSVs (the resumable scrape units) |
| `manifest.json` | Scrape progress checkpoint |
| `hyrox_scraper.py` | The scraper that produced it |

## Reproducing / updating

```bash
pip install requests beautifulsoup4 lxml playwright pandas pyarrow
playwright install chromium

python hyrox_scraper.py --season 8          # scrape one season (resumable)
python hyrox_scraper.py                      # all seasons (1-9)
python hyrox_scraper.py --combine            # merge by_event/ -> hyrox_full.*
```

The scrape is **resumable** — stop any time and re-run; completed
`(season, event, gender)` combos are skipped via `manifest.json`.

### Speed tuning

Throughput ≈ `sessions × rate` requests/sec. Detail-page fetches (the bottleneck)
are run concurrently across several independent cookie sessions:

```bash
python hyrox_scraper.py --sessions 4 --rate 10 --workers 12   # ~30 req/s
```

- `--sessions K` — independent cookie sessions; multiplies throughput (the site
  throttles per-session, so this scales where a single session can't).
- `--rate R` — per-session requests/sec cap.
- `--workers W` — threads per session.

Short bursts hit ~30/s, but **sustained** high rate on a single IP gets that IP
rate-blocked (HTTP 403 on everything; the headless seed then returns 0 cookies).
The adaptive limiter eases off on 403s and recovers, but the real fix for speed
is **not** cranking one connection — it's spreading the work across **many IPs**,
each running the gentle defaults. See [DISTRIBUTED.md](DISTRIBUTED.md) for the
Google Colab setup (`--make-shards` / `--events-file`). Defaults are deliberately
gentle (`--sessions 1 --rate 6`).

Useful flags: `--list-only` (enumerate events), `--max-events N` /
`--limit-athletes N` (sampling), `--no-splits` (list pages only).

## How it works

The site blocks plain HTTP clients (HTTP 403), so the scraper drives a headless
browser (Playwright) to clear the anti-bot challenge and harvest cookies — once
per session — then does the bulk fetching with fast `requests`, re-seeding
cookies automatically and backing off on 403/429.

## Notes on use

- Data is sourced from publicly displayed race results. It includes personal
  names — treat it like any public race-results dataset and follow HYROX's terms
  and applicable privacy rules when redistributing.
- Please keep the rate limiting intact; don't hammer the source site.
- Not affiliated with or endorsed by HYROX.
