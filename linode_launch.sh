#!/bin/bash
# Linode / Akamai per-worker bootstrap. Runs identically on all 50 boxes.
# Requires REDIS_URL to be set (StackScript UDF or exported before running).
set -e
exec > >(tee -a /var/log/hyrox-worker.log) 2>&1
echo "=== hyrox worker booting $(date) on $(hostname) ==="

# 1. deps
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv git curl
python3 -m venv /opt/hyrox
source /opt/hyrox/bin/activate
pip install -q redis requests beautifulsoup4 lxml pandas pyarrow playwright
playwright install --with-deps chromium

# 2. get the code — public GitHub, no auth needed
cd /workspace 2>/dev/null || { mkdir -p /workspace && cd /workspace; }
git clone --depth 1 https://github.com/DeVaNsHk72/hyrox-dataset.git hyrox
cd hyrox

# 3. random stagger so 50 boxes don't hit the site in the same second
sleep $((RANDOM % 300))

# 4. run the worker, auto-restart if it crashes
while true; do
  python3 dist_worker.py --out /workspace/data --rate 2.0 --workers 4 || true
  echo "$(date) worker exited — restarting in 30s"
  sleep 30
done
