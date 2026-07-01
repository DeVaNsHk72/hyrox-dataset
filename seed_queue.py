"""
Seed / inspect the distributed work queue (Upstash Redis).

The queue coordinates WHICH events each worker scrapes — atomically, so no two
workers grab the same task and crashed workers' tasks get reclaimed.

Keys:
  hyrox:pending     LIST  of task JSON  {"id","events":[{season,city,event_id,division}...]}
  hyrox:processing  ZSET  member=task JSON, score=lease-expiry epoch
  hyrox:done        SET   of completed task ids
  hyrox:athletes    INT   running athlete counter

Setup:
  pip install redis
  export REDIS_URL="rediss://default:<password>@<host>:6379"   # from Upstash console

Usage:
  python seed_queue.py --seed                 # push all events as micro-shards
  python seed_queue.py --seed --chunk 4 \
         --done-manifests "collected_manifests/*.json"   # skip already-done events
  python seed_queue.py --status               # show queue depth + progress
  python seed_queue.py --reset                # wipe the queue (careful)
"""
import argparse, glob, json, os, sys
from collections import defaultdict

import redis


def client() -> redis.Redis:
    url = os.environ.get("REDIS_URL")
    if not url:
        sys.exit("Set REDIS_URL (rediss://default:<pw>@<host>:6379) from the Upstash console.")
    return redis.from_url(url, decode_responses=True)


def load_all_events() -> list[dict]:
    """All unique events, taken from the already-generated shard files (no network)."""
    events, seen = [], set()
    for f in sorted(glob.glob("data_v2/shards/shard_*.json")):
        for e in json.load(open(f)):
            key = (e["season"], e["event_id"])
            if key not in seen:
                seen.add(key)
                events.append(e)
    return events


def done_events(pattern: str) -> set:
    """Events finished (BOTH sexes) according to any collected manifest.json files."""
    sexes = defaultdict(set)
    for mf in glob.glob(pattern):
        for k in json.load(open(mf)).get("done", {}):
            s, eid, sex = k.split("|")
            sexes[(int(s), eid)].add(sex)
    return {k for k, v in sexes.items() if "M" in v and "W" in v}


def seed(chunk: int, done_pattern: str | None) -> None:
    r = client()
    events = load_all_events()
    skip = done_events(done_pattern) if done_pattern else set()
    todo = [e for e in events if (int(e["season"]), e["event_id"]) not in skip]
    print(f"{len(events)} total events | {len(skip)} already done | {len(todo)} to queue")

    pipe = r.pipeline()
    n_tasks = 0
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        task = {"id": f"t{i//chunk:04d}", "events": batch}
        pipe.rpush("hyrox:pending", json.dumps(task, ensure_ascii=False))
        n_tasks += 1
    pipe.set("hyrox:total_tasks", n_tasks)
    pipe.execute()
    print(f"Seeded {n_tasks} micro-shards (~{chunk} events each) into hyrox:pending")


def status() -> None:
    r = client()
    print(f"pending:    {r.llen('hyrox:pending')}")
    print(f"processing: {r.zcard('hyrox:processing')}  (in-flight / leased)")
    print(f"done:       {r.scard('hyrox:done')} / {r.get('hyrox:total_tasks')} tasks")
    print(f"athletes:   {int(r.get('hyrox:athletes') or 0):,}")


def reset() -> None:
    r = client()
    r.delete("hyrox:pending", "hyrox:processing", "hyrox:done",
             "hyrox:athletes", "hyrox:total_tasks")
    print("queue cleared")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--chunk", type=int, default=4, help="events per micro-shard")
    ap.add_argument("--done-manifests", help="glob of manifest.json files to skip (already scraped)")
    args = ap.parse_args()
    if args.reset:
        reset()
    if args.seed:
        seed(args.chunk, args.done_manifests)
    if args.status:
        status()
    if not any([args.seed, args.status, args.reset]):
        ap.print_help()


if __name__ == "__main__":
    main()
