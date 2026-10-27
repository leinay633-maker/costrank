#!/usr/bin/env bash
# Run several configs against one snapshot with at most PARALLEL runs at a time (Linux, the cloud container).
# A full-A S4 run peaks near 19 GB, so keep PARALLEL x that inside the container's memory.
#
# usage: bash scripts/run_grid.sh DATA RUNS_ROOT LOG_DIR PARALLEL WORKERS CONFIG [CONFIG ...]
# e.g.:  nohup bash scripts/run_grid.sh /data/costrank/data/bao-alla-dev-1 /data/costrank/runs/v04 \
#            /data/costrank/logs/v04 2 8 configs/v04-s4p4.json configs/v04-s2p2.json ... > grid.log 2>&1 &
# Each run writes its own manifest (complete or failed, evidence retained); this script only
# queues them and prints one status line per config.
set -euo pipefail
if [ "$#" -lt 6 ]; then
  sed -n '2,10p' "$0"
  exit 2
fi
export DATA=$1 RUNS=$2 LOGS=$3 WORKERS=$5
PARALLEL=$4
shift 5
mkdir -p "$RUNS" "$LOGS"
run_one() {
  local cfg=$1 name
  name=$(basename "$cfg" .json)
  echo "$(date -Is) start $name"
  if .venv/bin/python -m costrank.cli run --config "$cfg" --data "$DATA" --allow-proxy --output "$RUNS" \
      --workers "$WORKERS" > "$LOGS/$name.log" 2>&1 < /dev/null; then
    echo "$(date -Is) ok $name $(tail -n 1 "$LOGS/$name.log")"
  else
    echo "$(date -Is) FAILED $name (see $LOGS/$name.log)"
  fi
}
export -f run_one
printf '%s\n' "$@" | xargs -P "$PARALLEL" -I{} bash -c 'run_one "$1"' _ {}
