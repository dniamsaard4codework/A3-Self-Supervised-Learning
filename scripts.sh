#!/usr/bin/env bash
# =============================================================================
# A3 Self-Supervised Learning — full reproduction (bash / Linux / macOS / Git Bash)
#
# Runs the complete experiment matrix through the assignment's run.py CLI:
#   SimCLR (comparison)  +  DINO default / no-centering / no-local  +
#   MAE main (0.75) + MAE masking ablation {0.25, 0.50, 0.75}, then all figures.
#
# Each call records metrics into results/results.json; `run.py --figures` then
# regenerates every plot from the saved checkpoints + metrics.
#
#   bash scripts.sh                 # ~3 h on an RTX 5060 Ti
#   PY=python NW=4 bash scripts.sh  # override interpreter / dataloader workers
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-uv run python}"   # interpreter (e.g. PY="python" or PY=".venv/Scripts/python.exe")
NW="${NW:-8}"               # DataLoader workers

echo "==> SimCLR (comparison baseline, 30 epochs)"
$PY run.py --model simclr --epochs 30 --train --num-workers "$NW"
$PY run.py --model simclr --weights saved/simclr.pt --evaluate --linear --num-workers "$NW"

echo "==> DINO default (50 epochs)"
$PY run.py --model dino --epochs 50 --train --num-workers "$NW"
$PY run.py --model dino --weights saved/dino.pt --evaluate --linear --num-workers "$NW"

echo "==> DINO ablation: no centering (collapse) (50 epochs)"
$PY run.py --model dino --no-centering --epochs 50 --train --num-workers "$NW"
$PY run.py --model dino --weights saved/dino_no_centering.pt --evaluate --linear --num-workers "$NW"

echo "==> DINO ablation: no local crops (50 epochs)"
$PY run.py --model dino --n-local 0 --epochs 50 --train --num-workers "$NW"
$PY run.py --model dino --weights saved/dino_no_local.pt --evaluate --linear --num-workers "$NW"

echo "==> MAE main (mask 0.75, 50 epochs)"
$PY run.py --model mae --epochs 50 --train --num-workers "$NW"
$PY run.py --model mae --weights saved/mae_encoder.pt --evaluate --linear --num-workers "$NW"

echo "==> MAE masking ablation (50 epochs each; mask=0.75 reuses the main run above)"
$PY run.py --model mae --mask-ratio 0.25 --epochs 50 --train --num-workers "$NW"
$PY run.py --model mae --mask-ratio 0.25 --epochs 50 --weights saved/mae_025_encoder.pt --evaluate --linear --num-workers "$NW"
$PY run.py --model mae --mask-ratio 0.50 --epochs 50 --train --num-workers "$NW"
$PY run.py --model mae --mask-ratio 0.50 --epochs 50 --weights saved/mae_050_encoder.pt --evaluate --linear --num-workers "$NW"

echo "==> Generating all figures"
$PY run.py --figures --num-workers "$NW"

echo ""
echo "==> Done. Metrics -> results/results.json | Figures -> figures/"
