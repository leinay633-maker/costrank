#!/usr/bin/env bash
# Keeps the formal paper-trading schedule alive on the cloud container (docs/PAPER.md). The container runs no
# cron daemon and nothing starts the schedule again after a container restart, so a scheduled task on
# the user's PC pipes this script over ssh twice a day:  ssh ... 'bash -s' < paper_watchdog.sh
# A running schedule is left alone. Otherwise the schedule is started again from the code directory it
# last ran from (it then runs a missed slot of the day at once, and earlier sessions in order) and the
# restart is alerted on WeCom. Nothing is done after the forward period ends.
set -u
BASE=${COSTRANK_BASE:-/data/costrank}
cd "$BASE" || exit 1
[[ "$(date +%F)" > "2026-12-31" ]] && { echo "$(date -Is) forward period over"; exit 0; }
if pgrep -f "[p]aper_daily.sh schedule" > /dev/null; then
  echo "$(date -Is) schedule alive"
  exit 0
fi
code=$(grep -o "schedule $BASE/v04-[0-9a-f]*/" logs/paper.log | tail -n 1 | sed "s|schedule $BASE/||; s|/$||")
[[ -n "$code" && -x "$code/.venv/bin/python" ]] || { echo "$(date -Is) cannot find the schedule's code directory"; exit 1; }
if ! flock -n logs/paper-schedule/schedule.lock true; then
  echo "$(date -Is) schedule.lock is held without a schedule process; trying again next time"
  exit 1
fi
COSTRANK_WECOM_WEBHOOK=$(cat secrets/wecom-webhook) setsid nohup bash "$code/scripts/paper_daily.sh" schedule >> logs/paper.log 2>&1 < /dev/null &
sleep 3
pid=$(pgrep -f "[p]aper_daily.sh schedule" | head -n 1)
echo "$(date -Is) restarted the schedule from $code, pid ${pid:-none}"
cd "$code" || exit 1
COSTRANK_WECOM_WEBHOOK=$(cat ../secrets/wecom-webhook) .venv/bin/python -c "
import sys
from pathlib import Path
from costrank.live import send_alert
send_alert(Path(sys.argv[1]), 'WARN', f'watchdog: the schedule was not running (container restart?); started again from {sys.argv[2]}, pid {sys.argv[3]}')
" "$BASE/paper" "$code" "${pid:-none}"
