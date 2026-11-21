#!/usr/bin/env bash
# Runs the remaining experiment stages one after another (laptop has 8 GB RAM: never in parallel).
cd "$(dirname "$0")/.."
log() { echo "$(date '+%H:%M:%S') $*"; }

log "waiting for LFW eval + engagement stage (task btssiz6dg) and YOLO fine-tune ..."
until [ -f reports/experiments/engagement/engagement.json ] && [ -f models/yolo/yolov8n-classroom.pt ]; do sleep 30; done
log "prerequisites ready"

for s in sequence detection seating pipeline_latency; do
  log "=== stage $s ==="
  python -m smartattend stage "$s" > "logs/stage_$s.log" 2>&1 || { log "stage $s FAILED"; tail -20 "logs/stage_$s.log"; }
  tail -3 "logs/stage_$s.log" | cut -d'|' -f4-
done

log "=== demo ==="
python scripts/make_demo.py > logs/demo.log 2>&1 || { log "demo FAILED"; tail -20 logs/demo.log; }
tail -2 logs/demo.log | cut -d'|' -f4-
log "ALL DONE"
