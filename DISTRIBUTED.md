# Distributed scraping across Google Colab

Run the scrape on several Colab nodes in parallel. Each Colab runtime has its
**own IP**, so each gets an independent rate-limit budget — this is how you go
fast *without* hammering (and getting blocked on) a single connection.

The golden rule: **many gentle nodes, not few aggressive ones.** Each node runs
1 session at ~6 req/s. Speed comes from running many of them.

## 1. Generate the shards (once, locally — needs the source IP un-blocked)

```bash
cd hyrox_dataset
python hyrox_scraper.py --make-shards 8 --out ./data_v2
# -> writes data_v2/shards/shard_00.json ... shard_07.json
```

Pick the shard count = how many Colab nodes you'll run. More shards = each node
finishes sooner (and survives Colab's idle/12h limits more easily):

At a gentle ~2.5 req/s per node (and ~470k–1M total athletes), rough per-node time:

| Nodes | Per-node runtime (rough) |
|------:|--------------------------|
| 4     | ~13–26 h (will need a re-run past Colab's 12 h cap) |
| 8     | ~6.5–13 h |
| 16    | ~3–6 h |

More nodes = each finishes comfortably inside Colab's 12 h limit. With your 7
accounts + local = 8 nodes, expect roughly half a day per node; all run in
parallel, and re-running any node just continues where it left off.

## 2. Upload to Google Drive (once)

Create `MyDrive/hyrox/` and put inside it:
- `hyrox_scraper.py`
- the whole `shards/` folder

## 3. Run the nodes

Open `hyrox_colab.ipynb` in Colab. For **each** node:
1. Set `SHARD` (cell 3) to a different number: `0, 1, 2, …, N-1`.
2. Runtime → Run all.

Run them in parallel — different browser tabs, or different Google accounts for
more simultaneous runtimes. Each writes to its own `MyDrive/hyrox/node_XX/`
(resumable: re-run if it disconnects).

> You can also mix in your **local** machine as one node:
> `python hyrox_scraper.py --events-file data_v2/shards/shard_07.json --out data_v2/node_07`

## 4. Merge everything

Once all nodes finish, run once (cell 5, or locally):

```bash
python hyrox_scraper.py --combine --out /content/drive/MyDrive/hyrox   # on Colab
# or locally, after downloading the node_*/ folders:
python hyrox_scraper.py --combine --out ./data_v2
```

`--combine` recursively gathers every `node_*/by_event/*.csv`, de-dupes by
`(event_id, idp, sex)`, derives the `*_seconds` columns, and writes
`hyrox_full.csv/.parquet/.jsonl` + `DATA_DICTIONARY.md`.

## Notes / etiquette

- Keep per-node settings gentle (`--sessions 1 --rate 2.5`). This site blocks
  sustained high rates **per IP**; ~2.5/s is sustainable. If a node still trips a
  block, the scraper auto-pauses ~7 min to let the penalty clear, then resumes.
  The point of going distributed is to be polite *per IP* while still finishing
  quickly across many IPs.
- Colab free tier disconnects when idle and caps runtime (~12 h). The scrape is
  resumable, so just re-run a node's cell to continue its shard.
- Colab's terms aren't really meant for long unattended jobs; expect occasional
  runtime reclamation. The shard + resume design tolerates it.
