#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="../.venv-torch/bin/python"
export PYTHONUNBUFFERED=1
TRAIN_ROOT="/mnt/d/denoise/素材/训练素材"
VAL_ROOT="/mnt/d/denoise/素材/训练素材/不参加训练，用来验证训练模型效果"

echo "=== Rebuild references (median flicker alignment + ROI stats) ==="
"$PY" build_references.py "$TRAIN_ROOT" --output-dir cache

echo "=== Train Restormer v2 from scratch ==="
"$PY" train.py \
  --manifest cache/train_manifest.json \
  --output-dir checkpoints_v2 \
  --patch-size 128 \
  --micro-batch 16 \
  --accum-steps 1 \
  --iters-per-epoch 300 \
  --epochs 80 \
  --num-workers 3 \
  --lr 2e-4 \
  --mean-weight 1.0 \
  --gradient-weight 0.15 \
  --flat-smooth-weight 0.18 \
  --flat-roi-fraction 0.75 \
  --brightness-condition \
  --exposure-boost-ms 50 \
  --exposure-boost-sigma 15 \
  --fps20-boost 3.0 \
  --exposure-boost 3.0 \
  --flicker-boost 1.5

echo "=== Evaluate on held-out validation (frame 10) ==="
"$PY" evaluate.py \
  "$VAL_ROOT" \
  --checkpoint checkpoints_v2/best.pt \
  --output-dir eval_v2_validation_frame010 \
  --frame-index 10

echo "=== Done ==="
