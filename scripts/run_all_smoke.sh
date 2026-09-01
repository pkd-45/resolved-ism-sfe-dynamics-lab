#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-outputs/smoke}"
rm -rf "$OUT"
mkdir -p "$OUT"

python scripts/run_synthetic_smoke.py --outdir "$OUT"

python scripts/analyze_resolved_table.py   --csv "$OUT/synthetic_resolved_regions.csv"   --radius-col radius_kpc   --vrot-col vrot   --sigma-col sigma_gas   --predictors radius_kpc   --n-boot 300   --outdir "$OUT/analysis"

pytest -q
ruff check src tests scripts

echo "ALL_SMOKE=PASS"
