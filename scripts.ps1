# =============================================================================
# A3 Self-Supervised Learning - full reproduction (Windows PowerShell)
#
# Runs the complete experiment matrix through the assignment's run.py CLI:
#   SimCLR (comparison)  +  DINO default / no-centering / no-local  +
#   MAE main (0.75) + MAE masking ablation {0.25, 0.50, 0.75}, then all figures.
#
# Each call records metrics into results/results.json; `run.py --figures` then
# regenerates every plot from the saved checkpoints + metrics.
#
#   powershell -ExecutionPolicy Bypass -File scripts.ps1     # ~3 h on an RTX 5060 Ti
# Edit $NW (DataLoader workers) or the Run function (interpreter) below if needed.
# =============================================================================
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$NW = 8                                   # DataLoader workers
function Run { uv run python run.py @args }   # interpreter: edit to "python run.py @args" if not using uv

Write-Host "==> SimCLR (comparison baseline, 30 epochs)"
Run --model simclr --epochs 30 --train --num-workers $NW
Run --model simclr --weights saved/simclr.pt --evaluate --linear --num-workers $NW

Write-Host "==> DINO default (50 epochs)"
Run --model dino --epochs 50 --train --num-workers $NW
Run --model dino --weights saved/dino.pt --evaluate --linear --num-workers $NW

Write-Host "==> DINO ablation: no centering (collapse) (50 epochs)"
Run --model dino --no-centering --epochs 50 --train --num-workers $NW
Run --model dino --weights saved/dino_no_centering.pt --evaluate --linear --num-workers $NW

Write-Host "==> DINO ablation: no local crops (50 epochs)"
Run --model dino --n-local 0 --epochs 50 --train --num-workers $NW
Run --model dino --weights saved/dino_no_local.pt --evaluate --linear --num-workers $NW

Write-Host "==> MAE main (mask 0.75, 50 epochs)"
Run --model mae --epochs 50 --train --num-workers $NW
Run --model mae --weights saved/mae_encoder.pt --evaluate --linear --num-workers $NW

Write-Host "==> MAE masking ablation (50 epochs each; mask=0.75 reuses the main run above)"
Run --model mae --mask-ratio 0.25 --epochs 50 --train --num-workers $NW
Run --model mae --mask-ratio 0.25 --epochs 50 --weights saved/mae_025_encoder.pt --evaluate --linear --num-workers $NW
Run --model mae --mask-ratio 0.50 --epochs 50 --train --num-workers $NW
Run --model mae --mask-ratio 0.50 --epochs 50 --weights saved/mae_050_encoder.pt --evaluate --linear --num-workers $NW

Write-Host "==> Generating all figures"
Run --figures --num-workers $NW

Write-Host ""
Write-Host "==> Done. Metrics -> results/results.json | Figures -> figures/"
