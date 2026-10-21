#!/bin/bash
# PazBrain nightly snapshot: one self-contained git bundle of the wiki repo (all refs),
# verified before it replaces anything. Runs from cron at 21:30, 30 min before
# ~/synology-backup.sh, which rsyncs /home/jeffpaz (and so this backup dir) to the NAS.
# Restore: git clone /opt/pazbrain/backup/pazbrain-YYYY-MM-DD.bundle wiki
set -euo pipefail

WIKI=/opt/pazbrain/wiki
DEST=/opt/pazbrain/backup
KEEP_DAYS=14
OUT="$DEST/pazbrain-$(date +%F).bundle"
TMP="$OUT.tmp"

echo "=== pazbrain snapshot $(date -Is) ==="
mkdir -p "$DEST"
trap 'rm -f "$TMP"' EXIT

git -C "$WIKI" bundle create "$TMP" --all
git -C "$WIKI" bundle verify "$TMP"
mv -f "$TMP" "$OUT"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1), HEAD $(git -C "$WIKI" rev-parse --short HEAD))"

find "$DEST" -maxdepth 1 -name 'pazbrain-*.bundle' -type f -mtime +"$KEEP_DAYS" -print -delete
echo "=== done ==="
