#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-}"
REPORT_NAME="${2:-}"
MSG="${3:-Add run analysis report}"

if [ -z "$RUN_DIR" ]; then
  RUN_DIR=$(find ./runs -maxdepth 1 -mindepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
fi
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
  echo "Usage: bash commands/publish_run_report.sh ./runs/<run_dir> [report_name] [commit_msg]"
  exit 1
fi
if [ -z "$REPORT_NAME" ]; then
  REPORT_NAME=$(basename "$RUN_DIR")
fi

python tools/analyze_run.py "$RUN_DIR"

DEST="reports/$REPORT_NAME"
mkdir -p "$DEST"
cp "$RUN_DIR/metrics.csv" "$DEST/" 2>/dev/null || true
cp "$RUN_DIR/final_report.json" "$DEST/" 2>/dev/null || true
cp "$RUN_DIR/auto_summary.json" "$DEST/" 2>/dev/null || true
cp "$RUN_DIR/AUTO_SUMMARY.md" "$DEST/" 2>/dev/null || true
latest_analysis=$(ls "$RUN_DIR"/analysis_epoch_*.json 2>/dev/null | sort | tail -n 1 || true)
latest_events=$(ls "$RUN_DIR"/events_epoch_*.jsonl 2>/dev/null | sort | tail -n 1 || true)
if [ -n "$latest_analysis" ]; then cp "$latest_analysis" "$DEST/latest_analysis.json"; fi
if [ -n "$latest_events" ]; then cp "$latest_events" "$DEST/latest_events.jsonl"; fi

file_count=$(find "$DEST" -type f | wc -l)
if [ "$file_count" -eq 0 ]; then
  echo "ERROR: no report files were copied to $DEST"
  echo "Check that $RUN_DIR contains metrics.csv, analysis_epoch_*.json, events_epoch_*.jsonl, or AUTO_SUMMARY.md"
  exit 2
fi

echo "Report files prepared in $DEST:"
find "$DEST" -maxdepth 1 -type f -printf '  %p\n' | sort

git pull --rebase origin main
# Force-add only report/source text files.  We never add checkpoints because .gitignore excludes them and this list is explicit.
git add -f "$DEST"
git add tools README.md ARCHITECTURE.md DEVELOPMENT_LOG.md commands experiments || true

echo "Staged files:"
git diff --cached --name-only

if git diff --cached --quiet; then
  echo "Nothing to commit. If you expected a report commit, run: git status --ignored --short"
else
  git commit -m "$MSG"
  git push origin main
fi

echo "Published report: $DEST"
