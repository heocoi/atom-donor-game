#!/usr/bin/env bash
# Phase 0 invasion-robustness sweep: condition C (Atom Hybrid) only.
# Vary initial defector fraction; 3 reps each. Logs SWEEP lines to data/sweep_results.txt.
set -euo pipefail
cd "$(dirname "$0")"

BIN=./target/release/donor_world
TICKS=${1:-200}
AGENTS=${2:-20}
OUT=data/sweep_results.txt
FRACS=(0.20 0.33 0.50 0.67)
REPS=3

echo "# invasion-robustness sweep $(date '+%Y-%m-%d %H:%M')" > "$OUT"
echo "# ticks=$TICKS agents=$AGENTS reps=$REPS fracs=${FRACS[*]}" >> "$OUT"

for f in "${FRACS[@]}"; do
  for r in $(seq 1 $REPS); do
    echo ">>> frac=$f rep=$r"
    "$BIN" "$TICKS" "$AGENTS" "$f" c "$r" 2>&1 | grep "^SWEEP" | tee -a "$OUT"
  done
done

echo "=== sweep done -> $OUT ==="
