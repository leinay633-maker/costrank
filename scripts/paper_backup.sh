#!/usr/bin/env bash
# Writes a tar.gz of the forward paper-trading record to stdout (docs/PAPER.md), so the scheduled task
# on the user's PC keeps a copy off the container:  ssh ... 'bash -s' < paper_backup.sh > record-<time>.tar.gz
# Included: the official outputs except the daily snapshots (rebuilt from the raw cache), the protocol
# files, the schedule log and slot markers, and every raw-cache file written since the forward cache was
# extended on 2026-10-15 (the forward period's inputs).
set -euo pipefail
cd "${COSTRANK_BASE:-/data/costrank}"
{
  find paper -path paper/snapshots -prune -o -type f -print
  find protocol logs/paper-schedule -type f
  echo logs/paper.log
  find data/bao-alla-live -type f -newermt "2026-10-15 06:00"
} | tar -czf - -T -
