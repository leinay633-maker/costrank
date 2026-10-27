#!/usr/bin/env bash
# Forward paper trading on the cloud container (RESEARCH section 10, docs/PAPER.md), whose clock is Beijing time.
# A run fetches, then records every forward session not yet recorded, in order, with the code of the
# directory it sits in; a failed gate sets that session's raw files aside so the next run fetches them
# again. `retry` does nothing once today's session is recorded. `schedule` stays running and makes the
# run at 20:30 and the retry at 23:30 on weekdays (a slot passed while it was down runs as soon as it
# is back the same day), because the cloud container runs no cron daemon; it exits after the last session
# within the reviewed trading rules. Nothing runs before the protocol's forward start, and one run at
# a time holds the lock.
set -u
CODE=$(cd "$(dirname "$0")/.." && pwd)
SELF=$CODE/scripts/$(basename "$0")
BASE=${COSTRANK_BASE:-/data/costrank}
ROOT=${COSTRANK_ROOT:-$BASE/data/bao-alla-live}
FREEZE=${COSTRANK_FREEZE:-$BASE/protocol/v04-freeze-3.json}
PROTOCOL=${COSTRANK_PROTOCOL:-$BASE/protocol/forward.json}
OUT=${COSTRANK_PAPER:-$BASE/paper}
STATE=${COSTRANK_SCHEDULE_STATE:-$BASE/logs/paper-schedule}
cd "$CODE" || exit 1
source "$BASE/env.sh"
read -r start end < <(.venv/bin/python -c "import json, sys; from costrank.config import RULES_REVIEWED_THROUGH as r; \
print(json.load(open(sys.argv[1]))['forward_start'], r)" "$PROTOCOL")
[[ -n "${start:-}" && -n "${end:-}" ]] || { echo "$(date -Is) cannot read $PROTOCOL"; exit 1; }
mkdir -p "$STATE"

if [[ "${1:-}" == schedule ]]; then
  exec 9> "$STATE/schedule.lock"
  flock -n 9 || { echo "$(date -Is) a schedule is already running"; exit 1; }
  echo "== $(date -Is) schedule $SELF pid $$, through $end"
  while [[ ! "$(date +%F)" > "$end" ]]; do
    today=$(date +%F)
    if (( $(date +%u) <= 5 )); then
      for slot in "${COSTRANK_RUN_AT:-2030} run" "${COSTRANK_RETRY_AT:-2330} retry"; do
        read -r at mode <<< "$slot"
        if [[ ! "$(date +%H%M)" < "$at" && ! -e "$STATE/$today-$mode" ]]; then
          touch "$STATE/$today-$mode"
          bash "$SELF" "$mode"
        fi
      done
    fi
    sleep $((60 - 10#$(date +%S)))
  done
  echo "== $(date -Is) schedule done"
  exit 0
fi

today=$(date +%F)
[[ "$today" < "$start" || "$today" > "$end" ]] && exit 0
[[ "${1:-}" == retry && -e "$OUT/ledger/daily/$today.csv" ]] && exit 0
exec 8> "$STATE/run.lock"
flock -n 8 || { echo "$(date -Is) another run holds $STATE/run.lock"; exit 1; }
echo "== $(date -Is) ${1:-run} $CODE"
.venv/bin/python -m costrank.cli daily --root "$ROOT" --freeze "$FREEZE" --protocol "$PROTOCOL" --output "$OUT"
echo "exit=$?"
