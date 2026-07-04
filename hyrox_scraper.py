"""
HYROX Results Scraper (v2)
==========================
Collects EVERY participant with FULL per-station splits from results.hyrox.com,
across all seasons / cities / divisions / genders.

What it captures per athlete
----------------------------
- Identity:   season, event_id, city, year, division, sex, idp, bib, name,
              nationality, age_group
- Result:     rank_overall, rank_age_group, finish_time (net), overall_time (gross)
- Splits:     time + per-station PLACE for all 8 runs and all 8 workout stations
              (SkiErg, Sled Push, Sled Pull, Burpees, Row, Farmers, Lunges, Wall Balls),
              plus Roxzone total, Run Total, Best Run Lap.
- Granular:   the full "In/Out" timing table (time-of-day, cumulative, diff per
              segment) preserved as JSON in `granular_timing_json`.

How it gets past the anti-bot wall
----------------------------------
The site returns HTTP 403 to plain HTTP clients. We drive a real headless browser
ONCE (Playwright) to clear the challenge and harvest cookies, then do the bulk
fetching with fast `requests`. Cookies are automatically re-seeded on 403/expiry.

Resumable
---------
Every (season, event_id, sex) combo is written to its own CSV under
`<out>/by_event/` and recorded in `<out>/manifest.json`. Re-running skips combos
already completed, so a multi-day scrape can be stopped and resumed freely.

Usage
-----
    python hyrox_scraper.py --season 8                 # one season, everything
    python hyrox_scraper.py                            # ALL seasons (1-9)
    python hyrox_scraper.py --season 8 --list-only     # enumerate events, no fetch
    python hyrox_scraper.py --season 8 --max-events 1 --limit-athletes 25   # smoke test
    python hyrox_scraper.py --combine                  # merge by_event/ -> full dataset
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hyrox")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

SEASONS = {n: f"https://results.hyrox.com/season-{n}/" for n in range(1, 10)}

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

PAGE_SIZE = 100          # list rows per page
RESEED_EVERY = 4000      # proactively refresh cookies after this many requests
MAX_RETRIES = 4          # with exponential backoff on 403/429/503

# If the IP does get blocked, the penalty is TIME-based (~100s observed):
# hammering keeps returning 403. After N consecutive blocks, pause to let it
# expire, then resume at base rate. At the gentle default rate this rarely fires.
BLOCK_COOLDOWN_AFTER = 12     # consecutive blocks before a full pause
COOLDOWN_SECONDS = 150        # length of that pause (~matches the server penalty)

# Defaults (overridable via CLI). Aggregate throughput ≈ SESSIONS × RATE req/s.
# Kept GENTLE on purpose: this site blocks sustained high rate per IP. For real
# speed, distribute across IPs (Colab nodes), each running these gentle defaults.
DEFAULT_SESSIONS = 1     # independent cookie sessions
DEFAULT_RATE = 2.5       # per-session requests/second cap (sustainable on one IP)
DEFAULT_WORKERS = 4      # threads per session


class RateLimiter:
    """Global throttle: ensures <= `rate` request starts/second across all
    threads, while requests run concurrently. Adaptive: slows down on blocks
    and recovers slowly (AIMD), so it settles at the server's sustainable rate."""

    FLOOR = 2.0   # max seconds/request (i.e. never slower than 0.5 req/s)

    def __init__(self, rate: float):
        self.base = 1.0 / rate if rate > 0 else 0.0
        self.min_interval = self.base
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait_for = self._next - now
            start = max(now, self._next)
            self._next = start + self.min_interval
        if wait_for > 0:
            time.sleep(wait_for)

    def slow_down(self, factor: float = 1.5) -> None:
        with self._lock:
            self.min_interval = min(max(self.min_interval, 0.02) * factor, self.FLOOR)

    def speed_up(self, factor: float = 0.97) -> None:
        with self._lock:
            self.min_interval = max(self.min_interval * factor, self.base)

    def reset(self) -> None:
        with self._lock:
            self.min_interval = self.base

    @property
    def rate(self) -> float:
        return 1.0 / self.min_interval if self.min_interval > 0 else 0.0

# --------------------------------------------------------------------------- #
# Split classification — robust to label variations across seasons
# --------------------------------------------------------------------------- #

STATION_COLUMNS = [
    "run_1", "ski_erg", "run_2", "sled_push", "run_3", "sled_pull",
    "run_4", "burpee_broad_jump", "run_5", "row", "run_6", "farmers_carry",
    "run_7", "sandbag_lunges", "run_8", "wall_balls",
    "roxzone", "run_total", "best_run_lap",
]


def classify_split(label: str) -> str | None:
    """Map a (possibly season-varying) split label to a canonical column name."""
    s = label.lower().strip()
    # running laps: "Running 1", "Run 1", "Lauf 1"
    m = re.search(r"(?:running|run|lauf)\s*(\d)", s)
    if m and "total" not in s and "best" not in s:
        return f"run_{m.group(1)}"
    if "best" in s and ("run" in s or "lap" in s or "lauf" in s):
        return "best_run_lap"
    if ("run" in s or "lauf" in s) and "total" in s:
        return "run_total"
    if "roxzone" in s or "rox zone" in s:
        return "roxzone"
    if "ski" in s:
        return "ski_erg"
    if "sled" in s and "push" in s:
        return "sled_push"
    if "sled" in s and "pull" in s:
        return "sled_pull"
    if "burpee" in s:
        return "burpee_broad_jump"
    if "row" in s:
        return "row"
    if "farmer" in s:
        return "farmers_carry"
    if "lunge" in s:
        return "sandbag_lunges"
    if "wall" in s:
        return "wall_balls"
    return None


# --------------------------------------------------------------------------- #
# HTTP client with browser-seeded cookies + auto-reseed
# --------------------------------------------------------------------------- #

class HyroxClient:
    """Thread-safe requests.Session primed with cookies harvested from a real
    browser, with a global rate limiter and stampede-free cookie re-seeding."""

    def __init__(self, seed_url: str, rate: float = DEFAULT_RATE, workers: int = DEFAULT_WORKERS):
        self.seed_url = seed_url
        self.limiter = RateLimiter(rate)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        # size the connection pool to the concurrency level
        adapter = HTTPAdapter(pool_connections=workers * 2, pool_maxsize=workers * 2)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self._req_count = 0
        self._blocks = 0
        self._consec_blocks = 0
        self._cooldown_until = 0.0
        self._last_block_log = 0.0
        self._count_lock = threading.Lock()
        self._seed_lock = threading.Lock()
        self._last_seed = 0.0
        self.seed()

    def seed(self) -> None:
        """Use Playwright to clear the anti-bot challenge and copy its cookies.
        Stampede-free: concurrent callers within a few seconds reseed only once."""
        with self._seed_lock:
            if time.monotonic() - self._last_seed < 8:
                return  # another thread just reseeded
            from playwright.sync_api import sync_playwright
            log.info("Seeding cookies via headless browser ...")
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                ctx = browser.new_context(user_agent=UA)
                page = ctx.new_page()
                page.goto(self.seed_url, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(2500)
                for c in ctx.cookies():
                    self.session.cookies.set(c["name"], c["value"],
                                             domain=c.get("domain"), path=c.get("path", "/"))
                browser.close()
            self._last_seed = time.monotonic()
            log.info("  cookies seeded (%d)", len(self.session.cookies))

    def get(self, url: str, params: dict | None = None) -> str | None:
        """Thread-safe GET. On a 403/429/503 block we BACK OFF and slow the rate
        limiter — we do NOT reseed (the cookie isn't the problem, the request
        rate is; reseeding mid-block stalls every thread on a browser launch).
        Cookies are only refreshed proactively or as a last resort."""
        with self._count_lock:
            self._req_count += 1
            do_reseed = self._req_count % RESEED_EVERY == 0
        if do_reseed:
            self.seed()

        blocked = False
        for attempt in range(MAX_RETRIES):
            self._await_cooldown()           # honor any active global pause
            self.limiter.wait()
            try:
                r = self.session.get(url, params=params, timeout=30)
            except requests.RequestException:
                time.sleep(min(2 ** attempt, 10) + random.random())
                continue
            if r.status_code == 200 and len(r.text) > 500:
                with self._count_lock:
                    self._consec_blocks = 0  # success clears the streak
                self.limiter.speed_up()      # drift back toward base rate
                return r.text
            if r.status_code in (403, 429, 503):
                blocked = True
                self._note_block(r.status_code)
                self.limiter.slow_down()
                time.sleep(min(2 ** attempt, 10) + random.random())
                continue
            time.sleep(0.5)                   # 200-but-short / 404 / other
        if blocked:                           # exhausted retries on a block: maybe cookie died
            self.seed()
        return None

    def _await_cooldown(self) -> None:
        with self._count_lock:
            wait = self._cooldown_until - time.monotonic()
        if wait > 0:
            time.sleep(wait)

    def _note_block(self, code: int) -> None:
        """Count blocks; after a run of them, pause for a real cooldown so the
        IP's time-based penalty can expire (hammering at 0.5/s never clears it)."""
        trigger = False
        with self._count_lock:
            self._blocks += 1
            self._consec_blocks += 1
            now = time.monotonic()
            if now - self._last_block_log > 5:
                self._last_block_log = now
                log.warning("HTTP %d block (#%d, streak %d) — easing to ~%.1f req/s",
                            code, self._blocks, self._consec_blocks, self.limiter.rate)
            if self._consec_blocks >= BLOCK_COOLDOWN_AFTER and now >= self._cooldown_until:
                self._cooldown_until = now + COOLDOWN_SECONDS
                self._consec_blocks = 0
                trigger = True
        if trigger:
            log.warning("Persistent blocking — pausing %ds to let the penalty clear, "
                        "then resuming at base rate", COOLDOWN_SECONDS)
            self.limiter.reset()


# --------------------------------------------------------------------------- #
# Event enumeration
# --------------------------------------------------------------------------- #

@dataclass
class EventCombo:
    season: int
    city: str             # event_main_group, e.g. "2026 Stockholm"
    event_id: str         # e.g. "HPRO_LR3MS4JI163A"
    division: str         # e.g. "HYROX PRO - Friday"


def get_cities(client: HyroxClient, season_url: str) -> list[str]:
    html = client.get(season_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    sel = soup.find("select", {"name": "event_main_group"})
    if not sel:
        return []
    return [o["value"] for o in sel.find_all("option") if o.get("value")]


def get_divisions(client: HyroxClient, season_url: str, city: str) -> list[tuple[str, str]]:
    """Return [(event_id, division_name)] for a given city."""
    html = client.get(urljoin(season_url, "index.php"),
                      {"pid": "start", "event_main_group": city})
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    sel = soup.find("select", {"name": "event"})
    if not sel:
        return []
    out = []
    for o in sel.find_all("option"):
        v = o.get("value")
        if v:
            out.append((v, o.get_text(strip=True)))
    return out


def enumerate_events(client: HyroxClient, season: int, season_url: str) -> list[EventCombo]:
    cities = get_cities(client, season_url)
    log.info("Season %d: %d cities", season, len(cities))
    combos: list[EventCombo] = []
    seen: set[str] = set()
    for city in cities:
        for event_id, division in get_divisions(client, season_url, city):
            if event_id in seen:
                continue
            seen.add(event_id)
            combos.append(EventCombo(season, city, event_id, division))
    log.info("Season %d: %d unique events", season, len(combos))
    return combos


# --------------------------------------------------------------------------- #
# List page parsing
# --------------------------------------------------------------------------- #

def _field_text(div) -> str:
    for lbl in div.find_all(class_="list-label"):
        lbl.decompose()
    return div.get_text(strip=True)


def parse_list_page(html: str) -> tuple[list[dict], int]:
    soup = BeautifulSoup(html, "lxml")
    total = 0
    info = soup.find(class_="str_num")
    if info:
        m = re.search(r"(\d[\d.,]*)", info.get_text())
        if m:
            total = int(re.sub(r"[.,]", "", m.group(1)))

    rows = []
    for item in soup.find_all(class_="list-group-item"):
        cls = " ".join(item.get("class", []))
        if "list-group-header" in cls:
            continue
        link = item.find("a", href=re.compile(r"idp=", re.I))
        if not link:
            continue
        m = re.search(r"idp=([A-Za-z0-9]+)", link["href"])
        if not m or m.group(1).lower() in ("ranking", "list", "start"):
            continue
        row = {"idp": m.group(1), "name": link.get_text(strip=True)}
        for div in item.find_all(class_="list-field"):
            c = " ".join(div.get("class", []))
            if "place-primary" in c:
                row["rank_overall"] = _field_text(div)
            elif "place-secondary" in c:
                row["rank_age_group"] = _field_text(div)
            elif "type-nation_flag" in c:
                abbr = div.find(class_="nation__abbr")
                img = div.find("img")
                row["nationality"] = (abbr.get_text(strip=True) if abbr
                                      else (img["alt"] if img and img.get("alt") else _field_text(div)))
            elif "type-age_class" in c:
                row["age_group"] = _field_text(div)
            elif "type-bib" in c or "type-start_no" in c:
                row["bib"] = _field_text(div)
            elif "type-time" in c:
                row["finish_time"] = _field_text(div)
        rows.append(row)
    return rows, total


def iter_list_rows(client: HyroxClient, season_url: str, ev: EventCombo, sex: str) -> Iterator[dict]:
    page, total, fetched = 1, None, 0
    while True:
        html = client.get(urljoin(season_url, "index.php"), {
            "event_main_group": ev.city, "pid": "list", "pidp": "ranking_nav",
            "event": ev.event_id, "ranking": "time_finish_netto",
            "search[sex]": sex, "search[age_class]": "%",
            "num_results": PAGE_SIZE, "page": page,
        })
        if not html:
            break
        rows, count = parse_list_page(html)
        if total is None:
            total = count
        if not rows:
            break
        for r in rows:
            yield r
        fetched += len(rows)
        if (total and fetched >= total) or len(rows) < PAGE_SIZE:
            break
        page += 1
        if page > 500:
            log.warning("  page cap hit for %s/%s", ev.event_id, sex)
            break


# --------------------------------------------------------------------------- #
# Detail page parsing (the part the old scraper got wrong)
# --------------------------------------------------------------------------- #

def parse_detail(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    out: dict = {}

    # --- metadata table(s): Name / Nat / Race / Division / Rank / Overall Time
    for tr in soup.find_all("tr"):
        th = tr.find("th", class_="desc")
        td = tr.find("td")
        if not th or not td:
            continue
        key = th.get_text(strip=True).lower()
        val = td.get_text(strip=True)
        if key == "overall time":
            out["overall_time"] = val
        elif key.startswith("rank"):
            out.setdefault("rank_overall_detail", val)

    # --- workout summary table: Split | Time | Place
    summary_table = None
    for t in soup.find_all("table"):
        head = t.find("tr")
        if head and re.search(r"\bsplit\b", head.get_text(" ", strip=True), re.I) \
                and re.search(r"\bplace\b", head.get_text(" ", strip=True), re.I):
            summary_table = t
            break

    if summary_table:
        for tr in summary_table.find_all("tr"):
            th = tr.find("th", class_="desc") or tr.find("th")
            tds = tr.find_all("td")
            if not th or not tds:
                continue
            col = classify_split(th.get_text(strip=True))
            if not col:
                continue
            t_time = tds[0].get_text(strip=True)
            t_place = tds[1].get_text(strip=True) if len(tds) > 1 else ""
            if t_time and t_time != "–":
                out[f"{col}_time"] = t_time
            if t_place and t_place != "–":
                out[f"{col}_place"] = t_place

    # --- granular In/Out table: Split | Time Of Day | Time | Diff  -> JSON
    granular = []
    for t in soup.find_all("table"):
        head = t.find("tr")
        if head and re.search(r"time\s*of\s*day", head.get_text(" ", strip=True), re.I):
            for tr in t.find_all("tr")[1:]:
                th = tr.find("th")
                tds = tr.find_all("td")
                if not th or not tds:
                    continue
                cells = [c.get_text(strip=True) for c in tds]
                granular.append({
                    "label": th.get_text(strip=True),
                    "time_of_day": cells[0] if len(cells) > 0 else "",
                    "cumulative": cells[1] if len(cells) > 1 else "",
                    "diff": cells[2] if len(cells) > 2 else "",
                })
            break
    if granular:
        out["granular_timing_json"] = json.dumps(granular, ensure_ascii=False)

    return out


def fetch_detail(client: HyroxClient, season_url: str, idp: str, event_id: str, sex: str) -> dict:
    html = client.get(urljoin(season_url, "index.php"), {
        "content": "detail", "fpid": "list", "pid": "list", "idp": idp,
        "lang": "EN_CAP", "event": event_id, "pidp": "ranking_nav",
        "ranking": "time_finish_netto", "search[sex]": sex,
        "search[age_class]": "%", "search_event": event_id,
    })
    return parse_detail(html) if html else {}


# --------------------------------------------------------------------------- #
# Orchestration with checkpoint/resume
# --------------------------------------------------------------------------- #

CITY_RE = re.compile(r"^\s*(\d{4})\s+(.*)$")


def split_city_year(city: str) -> tuple[str, str]:
    m = CITY_RE.match(city)
    return (m.group(1), m.group(2).strip()) if m else ("", city.strip())


@dataclass
class Manifest:
    path: Path
    done: dict = field(default_factory=dict)   # "season|event_id|sex" -> count

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        if path.exists():
            data = json.loads(path.read_text())
            return cls(path, data.get("done", {}))
        return cls(path)

    def key(self, ev: EventCombo, sex: str) -> str:
        return f"{ev.season}|{ev.event_id}|{sex}"

    def is_done(self, ev: EventCombo, sex: str) -> bool:
        return self.key(ev, sex) in self.done

    def mark(self, ev: EventCombo, sex: str, n: int) -> None:
        self.done[self.key(ev, sex)] = n
        self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps({"done": self.done}, indent=2))


def write_event_csv(rows: list[dict], path: Path) -> None:
    import csv
    if not rows:
        return
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def scrape_event(clients: list[HyroxClient], season_url: str, ev: EventCombo, out_dir: Path,
                 manifest: Manifest, fetch_splits: bool, limit_athletes: int | None,
                 workers: int = DEFAULT_WORKERS) -> int:
    year, city_name = split_city_year(ev.city)
    k = len(clients)
    n_total = 0
    for sex in ("M", "W", "X"):
        if manifest.is_done(ev, sex):
            continue
        rows: list[dict] = []
        for r in iter_list_rows(clients[0], season_url, ev, sex):
            r.update({
                "season": ev.season, "event_id": ev.event_id, "city": city_name,
                "year": year, "division": ev.division, "sex": sex,
            })
            rows.append(r)
            if limit_athletes and len(rows) >= limit_athletes:
                break

        if fetch_splits and rows:
            # Detail fetches are the bottleneck. Spread them across all
            # independent sessions (round-robin); each session has its own
            # RateLimiter, so aggregate ≈ sessions × per-session rate.
            n = len(rows)
            prog = {"n": 0}
            plock = threading.Lock()

            def _enrich(item: tuple[int, dict]) -> None:
                i, row = item
                c = clients[i % k]
                row.update(fetch_detail(c, season_url, row["idp"], ev.event_id, sex))
                with plock:                       # heartbeat so big events show life
                    prog["n"] += 1
                    if prog["n"] % 100 == 0 or prog["n"] == n:
                        log.info("      %s [%s]: %d/%d fetched", ev.event_id, sex, prog["n"], n)

            with ThreadPoolExecutor(max_workers=k * workers) as pool:
                list(pool.map(_enrich, enumerate(rows)))

        if rows:
            safe = re.sub(r"[^\w]", "_", f"s{ev.season}_{ev.city}_{ev.event_id}_{sex}")[:120]
            write_event_csv(rows, out_dir / f"{safe}.csv")
        manifest.mark(ev, sex, len(rows))
        log.info("    %s / %s [%s]: %d athletes", ev.city, ev.division, sex, len(rows))
        n_total += len(rows)
    return n_total


def load_events_file(path: str) -> dict[int, list[EventCombo]]:
    """Load a shard file ([{season,city,event_id,division}, ...]) grouped by season."""
    data = json.loads(Path(path).read_text())
    plan: dict[int, list[EventCombo]] = {}
    for d in data:
        ev = EventCombo(int(d["season"]), d["city"], d["event_id"], d["division"])
        plan.setdefault(ev.season, []).append(ev)
    return plan


def scrape(seasons: list[int], out_dir: Path, fetch_splits: bool,
           list_only: bool, max_events: int | None, limit_athletes: int | None,
           workers: int = DEFAULT_WORKERS, rate: float = DEFAULT_RATE,
           sessions: int = DEFAULT_SESSIONS, events_file: str | None = None) -> None:
    by_event = out_dir / "by_event"
    by_event.mkdir(parents=True, exist_ok=True)
    manifest = Manifest.load(out_dir / "manifest.json")

    # Work plan: either an explicit shard file, or per-season enumeration.
    plan = load_events_file(events_file) if events_file else None
    season_list = sorted(plan) if plan is not None else seasons
    if plan is not None:
        log.info("Shard mode: %d events across seasons %s",
                 sum(len(v) for v in plan.values()), season_list)

    grand_total = 0
    for season in season_list:
        season_url = SEASONS[season]
        log.info("Starting %d session(s) @ %.1f req/s each (~%.0f/s aggregate)",
                 sessions, rate, sessions * rate)
        clients = [HyroxClient(season_url, rate=rate, workers=workers) for _ in range(sessions)]
        combos = plan[season] if plan is not None else enumerate_events(clients[0], season, season_url)
        if max_events:
            combos = combos[:max_events]
        if list_only:
            for ev in combos:
                print(f"  S{season} | {ev.city} | {ev.division} | {ev.event_id}")
            continue
        for j, ev in enumerate(combos, 1):
            log.info("  [%d/%d] %s / %s", j, len(combos), ev.city, ev.division)
            try:
                grand_total += scrape_event(clients, season_url, ev, by_event,
                                            manifest, fetch_splits, limit_athletes, workers)
            except Exception as e:
                log.error("    event failed (%s): %s", ev.event_id, e, exc_info=True)
    if not list_only:
        log.info("=== Scrape pass complete. New athletes this run: %d ===", grand_total)
        log.info("Run with --combine to build the merged dataset.")


def make_shards(seasons: list[int], out_dir: Path, n: int,
                rate: float, workers: int) -> None:
    """Enumerate all events across seasons and split them into N shard files
    for distributed scraping (one shard per machine/Colab node)."""
    all_combos: list[dict] = []
    for season in seasons:
        url = SEASONS[season]
        client = HyroxClient(url, rate=rate, workers=workers)
        for ev in enumerate_events(client, season, url):
            all_combos.append({"season": ev.season, "city": ev.city,
                               "event_id": ev.event_id, "division": ev.division})
    random.shuffle(all_combos)   # mix big/small events evenly across shards
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    buckets: list[list[dict]] = [[] for _ in range(n)]
    for i, c in enumerate(all_combos):
        buckets[i % n].append(c)
    for i, b in enumerate(buckets):
        (shards_dir / f"shard_{i:02d}.json").write_text(json.dumps(b, ensure_ascii=False))
    log.info("Wrote %d events into %d shards → %s", len(all_combos), n, shards_dir)
    log.info("Each node runs: python hyrox_scraper.py --events-file shards/shard_XX.json "
             "--out node_XX --sessions 1 --rate 6")


# --------------------------------------------------------------------------- #
# Combine + derived columns + outputs
# --------------------------------------------------------------------------- #

def to_seconds(v) -> float | None:
    if not isinstance(v, str) or not v or v == "–":
        return None
    v = v.strip()
    # formats: HH:MM:SS, MM:SS, MM:SS.ss
    parts = v.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        return float(v)
    except ValueError:
        return None


def combine(out_dir: Path) -> None:
    import pandas as pd
    # Gather from this dir AND any nested node_*/ dirs (distributed runs).
    files = sorted(set(out_dir.rglob("by_event/*.csv")))
    if not files:
        log.error("No per-event CSVs found under %s", out_dir)
        return
    log.info("Combining %d event files ...", len(files))
    frames = [pd.read_csv(f, dtype=str, keep_default_na=False) for f in files]
    df = pd.concat(frames, ignore_index=True)

    # de-duplicate identical athlete rows
    df.drop_duplicates(subset=["event_id", "idp", "sex"], inplace=True)

    # derived *_seconds for every *_time column (e.g. ski_erg_time -> ski_erg_seconds)
    for c in [c for c in df.columns if c.endswith("_time")]:
        df[c[:-len("_time")] + "_seconds"] = df[c].map(to_seconds)

    # tidy column order
    lead = ["season", "year", "city", "division", "event_id", "sex",
            "idp", "bib", "name", "nationality", "age_group",
            "rank_overall", "rank_age_group", "finish_time", "overall_time"]
    lead = [c for c in lead if c in df.columns]
    rest = [c for c in df.columns if c not in lead]
    df = df[lead + rest]

    csv_path = out_dir / "hyrox_full.csv"
    pq_path = out_dir / "hyrox_full.parquet"
    jsonl_path = out_dir / "hyrox_full.jsonl"
    df.to_csv(csv_path, index=False)
    df.to_parquet(pq_path, index=False)
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

    log.info("Wrote %d rows", len(df))
    log.info("  %s", csv_path)
    log.info("  %s", pq_path)
    log.info("  %s", jsonl_path)
    write_data_dictionary(out_dir, df)


def write_data_dictionary(out_dir: Path, df) -> None:
    lines = [
        "# HYROX Results — Data Dictionary", "",
        f"Rows: **{len(df):,}**  |  Columns: **{len(df.columns)}**", "",
        "One row per athlete per event/division/gender. Times are `HH:MM:SS` "
        "(or `MM:SS.ss`); every `*_time` has a numeric `*_seconds` companion.", "",
        "| Column | Meaning |", "| --- | --- |",
    ]
    desc = {
        "season": "HYROX season number (from the season-N results URL)",
        "year": "Calendar year parsed from the event name",
        "city": "Host city / race location",
        "division": "Division + race day (e.g. HYROX PRO - Friday)",
        "event_id": "Site event identifier (division + race slot)",
        "sex": "M, W, or X",
        "idp": "Site athlete/result id (unique within an event)",
        "bib": "Bib / start number (when published)",
        "name": "Athlete name (Last, First)",
        "nationality": "3-letter nationality code",
        "age_group": "Age class (e.g. 30-34)",
        "rank_overall": "Finishing place within division+gender",
        "rank_age_group": "Finishing place within age group",
        "finish_time": "Net finish time (start mat to finish)",
        "overall_time": "Gross/official overall time",
        "granular_timing_json": "Full In/Out timing table (time-of-day, cumulative, diff) as JSON",
    }
    for col in STATION_COLUMNS:
        pretty = col.replace("_", " ").title()
        desc[f"{col}_time"] = f"{pretty} split time"
        desc[f"{col}_place"] = f"Place within the {pretty} split"
    for c in df.columns:
        if c in desc:
            lines.append(f"| `{c}` | {desc[c]} |")
        elif c.endswith("_seconds"):
            lines.append(f"| `{c}` | Numeric seconds for `{c[:-8]}` / `{c[:-8]}_time` |")
        else:
            lines.append(f"| `{c}` | — |")
    (out_dir / "DATA_DICTIONARY.md").write_text("\n".join(lines))
    log.info("  %s", out_dir / "DATA_DICTIONARY.md")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="HYROX results scraper (v2, full splits)")
    ap.add_argument("--season", type=int, nargs="+", help="seasons (default: all 1-9)")
    ap.add_argument("--out", default="./data_v2", help="output directory")
    ap.add_argument("--list-only", action="store_true", help="enumerate events only")
    ap.add_argument("--no-splits", action="store_true", help="skip per-athlete detail pages")
    ap.add_argument("--max-events", type=int, help="cap events per season (testing)")
    ap.add_argument("--limit-athletes", type=int, help="cap athletes per event/sex (testing)")
    ap.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS,
                    help=f"independent cookie sessions, multiplies throughput (default {DEFAULT_SESSIONS})")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"threads per session (default {DEFAULT_WORKERS})")
    ap.add_argument("--rate", type=float, default=DEFAULT_RATE,
                    help=f"per-session requests/sec cap (default {DEFAULT_RATE}); lower if blocked")
    ap.add_argument("--events-file", help="scrape only the events in this shard JSON (distributed mode)")
    ap.add_argument("--make-shards", type=int, metavar="N",
                    help="enumerate all events and split into N shard files (for distributed runs)")
    ap.add_argument("--combine", action="store_true", help="merge by_event/ (incl. node_*/ ) into final dataset")
    args = ap.parse_args()

    out_dir = Path(args.out)
    seasons = args.season or sorted(SEASONS)

    if args.combine:
        combine(out_dir)
        return
    if args.make_shards:
        make_shards(seasons, out_dir, args.make_shards, rate=args.rate, workers=args.workers)
        return

    scrape(seasons, out_dir,
           fetch_splits=not args.no_splits,
           list_only=args.list_only,
           max_events=args.max_events,
           limit_athletes=args.limit_athletes,
           workers=args.workers,
           rate=args.rate,
           sessions=args.sessions,
           events_file=args.events_file)


if __name__ == "__main__":
    main()
