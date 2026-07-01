"""
Universal distributed worker for the HYROX scrape.

Runs the SAME loop everywhere (Colab / Kaggle / laptop / Oracle Cloud):
  claim next task (atomic) -> scrape its events -> mark done -> repeat.
No hard-coded shard numbers. Crashed workers' tasks auto-reclaim via a lease.

Setup (same on every machine):
  pip install redis requests beautifulsoup4 lxml playwright && playwright install chromium
  export REDIS_URL="rediss://default:<pw>@<host>:6379"     # from Upstash
  python dist_worker.py --out ./data_dist                  # results land in ./data_dist/by_event

Point --out at a shared Google Drive folder on Colab so all results collect in one place.
Keep --rate gentle (default 2.5). Be a good citizen: don't run a huge fleet at once.
"""
import argparse, json, os, socket, sys, threading, time
from pathlib import Path

import redis

from hyrox_scraper import HyroxClient, EventCombo, Manifest, scrape_event, SEASONS

LEASE = 5400          # seconds a claimed task is "owned" before it can be reclaimed
HEARTBEAT = 120       # re-extend the lease this often while working

# atomic claim: pop from pending, register in processing with a lease score
CLAIM_LUA = """
local t = redis.call('LPOP', KEYS[1])
if not t then return false end
redis.call('ZADD', KEYS[2], ARGV[1], t)
return t
"""
# reclaim: move lease-expired processing tasks back to pending
REAP_LUA = """
local expired = redis.call('ZRANGEBYSCORE', KEYS[1], 0, ARGV[1])
for _, t in ipairs(expired) do
  redis.call('ZREM', KEYS[1], t)
  redis.call('RPUSH', KEYS[2], t)
end
return #expired
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data_dist", help="output dir (use a shared Drive path on Colab)")
    ap.add_argument("--rate", type=float, default=2.5, help="per-worker requests/sec")
    ap.add_argument("--workers", type=int, default=4, help="threads per worker")
    args = ap.parse_args()

    url = os.environ.get("REDIS_URL")
    if not url:
        sys.exit("Set REDIS_URL (from the Upstash console).")
    r = redis.from_url(url, decode_responses=True)
    claim = r.register_script(CLAIM_LUA)
    reap = r.register_script(REAP_LUA)

    wid = f"{socket.gethostname()}-{os.getpid()}"
    # Namespace each worker under --out so many workers can share ONE Drive
    # folder without clobbering each other's manifest. combine() rglobs it all.
    work_dir = Path(args.out) / f"worker_{wid}"
    by_event = work_dir / "by_event"
    by_event.mkdir(parents=True, exist_ok=True)
    manifest = Manifest.load(work_dir / "manifest.json")
    season_clients: dict[int, list] = {}

    def clients_for(season: int) -> list:
        if season not in season_clients:
            season_clients[season] = [HyroxClient(SEASONS[season], rate=args.rate, workers=args.workers)]
        return season_clients[season]

    print(f"[{wid}] worker up → {work_dir}  (rate {args.rate}/s)")

    while True:
        now = time.time()
        task_json = claim(keys=["hyrox:pending", "hyrox:processing"], args=[now + LEASE])
        if task_json is None:
            moved = reap(keys=["hyrox:processing", "hyrox:pending"], args=[now])
            if moved:
                print(f"[{wid}] reclaimed {moved} stale task(s)")
                continue
            if r.zcard("hyrox:processing") == 0 and r.llen("hyrox:pending") == 0:
                print(f"[{wid}] queue empty — all done. exiting.")
                break
            time.sleep(15)
            continue

        task = json.loads(task_json)
        stop = threading.Event()

        def heartbeat():                       # keep the lease fresh while we work
            while not stop.wait(HEARTBEAT):
                r.zadd("hyrox:processing", {task_json: time.time() + LEASE})
        hb = threading.Thread(target=heartbeat, daemon=True)
        hb.start()

        print(f"[{wid}] claim {task['id']} ({len(task['events'])} events)")
        try:
            got = 0
            for e in task["events"]:
                ev = EventCombo(int(e["season"]), e["city"], e["event_id"], e["division"])
                got += scrape_event(clients_for(ev.season), SEASONS[ev.season], ev,
                                    by_event, manifest, fetch_splits=True,
                                    limit_athletes=None, workers=args.workers)
            stop.set()
            pipe = r.pipeline()
            pipe.zrem("hyrox:processing", task_json)
            pipe.sadd("hyrox:done", task["id"])
            pipe.incrby("hyrox:athletes", got)
            pipe.execute()
            print(f"[{wid}] done {task['id']} +{got} | pending={r.llen('hyrox:pending')} "
                  f"done={r.scard('hyrox:done')}/{r.get('hyrox:total_tasks')}")
        except Exception as ex:
            stop.set()
            print(f"[{wid}] FAILED {task['id']}: {ex} — leaving for reclaim")
            time.sleep(5)


if __name__ == "__main__":
    main()
