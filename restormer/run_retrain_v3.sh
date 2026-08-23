#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="../.venv-torch/bin/python"
export PYTHONUNBUFFERED=1
TRAIN_ROOT="/mnt/d/denoise/素材/训练素材"
VAL_ROOT="/mnt/d/denoise/素材/训练素材/不参加训练，用来验证训练模型效果"

echo "=== Rebuild references + exposure brightness priors ==="
"$PY" build_references.py "$TRAIN_ROOT" --output-dir cache

echo "=== Train Restormer v3 (reduced flat bias + brightness offset + split eval) ==="
"$PY" train.py \
  --manifest cache/train_manifest.json \
  --exposure-priors cache/exposure_brightness_priors.json \
  --output-dir checkpoints_v3 \
  --patch-size 128 \
  --micro-batch 16 \
  --accum-steps 1 \
  --iters-per-epoch 300 \
  --epochs 80 \
  --num-workers 3 \
  --lr 2e-4 \
  --mean-weight 1.0 \
  --gradient-weight 0.15 \
  --flat-smooth-weight 0.06 \
  --flat-gradient-gate 0.04 \
  --flat-roi-fraction 0.45 \
  --brightness-condition \
  --brightness-offset \
  --exposure-boost-ms 50 \
  --exposure-boost-sigma 15 \
  --fps20-boost 3.0 \
  --exposure-boost 3.0 \
  --flicker-boost 1.5

echo "=== Evaluate with scene-type split ==="
"$PY" evaluate.py \
  "$VAL_ROOT" \
  --checkpoint checkpoints_v3/best.pt \
  --exposure-priors cache/exposure_brightness_priors.json \
  --output-dir eval_v3_validation_frame010 \
  --frame-index 10

echo "=== Done ==="
