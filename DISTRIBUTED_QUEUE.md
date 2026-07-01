# Distributed scrape with a pull-queue (Upstash Redis)

No more hard-coded `SHARD=`. A central queue hands out tiny tasks; every worker
grabs the next one, scrapes it, marks it done, and grabs another. Crashed workers'
tasks auto-reclaim. Add or remove workers anytime — they self-coordinate.

```
        ┌──────────────── Upstash Redis (the queue) ────────────────┐
        │  pending → [t0][t1][t2]…   processing(ZSET+lease)   done   │
        └───────────────────────────────────────────────────────────┘
           ▲ claim         ▲ claim          ▲ claim         ▲ claim
        worker1         worker2          worker3     …    workerN
       (Colab)         (Colab)          (laptop)         (Oracle)
```

## 0. One-time: make the queue (free)

1. Sign up at **upstash.com** → create a **Redis** database (free tier).
2. Copy the **`rediss://…` connect URL** (Upstash console → "Redis Connect" → redis-py).
3. That string is your `REDIS_URL`. Keep it secret-ish (it's write access to your queue).

## 1. Seed the queue (once, from your Mac — it has `shards/`)

```bash
cd hyrox_dataset
pip install redis
export REDIS_URL="rediss://default:<pw>@<host>:6379"

# (optional) skip events already scraped: drop every node's manifest.json into
# collected_manifests/ first, then:
python seed_queue.py --seed --chunk 4 --done-manifests "collected_manifests/*.json"
# or just seed everything:
python seed_queue.py --seed --chunk 4

python seed_queue.py --status      # watch progress anytime
```

## 2. Run workers (identical everywhere)

Every machine runs the **same** command — no shard number:

```bash
pip install redis requests beautifulsoup4 lxml playwright pandas pyarrow
playwright install chromium
export REDIS_URL="rediss://default:<pw>@<host>:6379"
python dist_worker.py --out ./data_dist          # local/Oracle: writes ./data_dist/worker_*/by_event
```

### On Colab (one cell — set it once, run it in every account/tab)

```python
!pip -q install redis requests beautifulsoup4 lxml pandas pyarrow playwright
!python -m playwright install chromium
from google.colab import drive; drive.mount('/content/drive')
import os; os.environ["REDIS_URL"] = "rediss://default:<pw>@<host>:6379"
# hyrox_scraper.py + dist_worker.py must be in this Drive folder:
%cd /content/drive/MyDrive/hyrox
!python dist_worker.py --out /content/drive/MyDrive/hyrox/dist
```

Each worker writes to its own `dist/worker_<host>-<pid>/` subfolder, so any number
of workers can share the one Drive folder without stepping on each other.

## 3. Collect the result

When `--status` shows `pending: 0` and `processing: 0`:

```bash
python hyrox_scraper.py --combine --out /content/drive/MyDrive/hyrox/dist
# → hyrox_full.csv / .parquet / .jsonl + DATA_DICTIONARY.md   (dedups automatically)
```

## Please keep the fleet sane

Every worker is polite (~2.5/s), but **aggregate** hits one small site. ~15–25
workers finishes the whole ~2–3M-row dataset in roughly a day and stays
defensible. Don't run 100 — that's ~250 req/s on one server, which is abusive and
will get IP ranges banned. More workers ≠ "free"; the site pays for it.

## How it's resilient

- **Atomic claim** (Lua `LPOP`+`ZADD`): two workers never get the same task.
- **Lease + heartbeat**: a working worker keeps its task; a dead one's task
  reappears in `pending` after the lease lapses and another worker takes it.
- **Local resume too**: each worker also keeps its own `manifest.json`.
- **Idempotent output**: `--combine` de-dupes by `(event_id, idp, sex)`, so any
  double-scraped task is harmless.
