#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="../.venv-torch/bin/python"
export PYTHONUNBUFFERED=1

echo "=== Fine-tune Restormer (flat-biased, stronger brightness/flat loss) ==="
"$PY" train.py \
  --manifest cache/train_manifest.json \
  --output-dir checkpoints_finetune \
  --load-weights checkpoints/best.pt \
  --patch-size 128 \
  --micro-batch 16 \
  --accum-steps 1 \
  --iters-per-epoch 300 \
  --epochs 20 \
  --num-workers 3 \
  --lr 5e-5 \
  --mean-weight 1.5 \
  --gradient-weight 0.15 \
  --flat-smooth-weight 0.10 \
  --flat-roi-fraction 0.70

echo "=== Evaluate on held-out validation (frame 10) ==="
"$PY" evaluate.py \
  "/mnt/d/denoise/素材/训练素材/不参加训练，用来验证训练模型效果" \
  --checkpoint checkpoints_finetune/best.pt \
  --output-dir eval_finetune_validation_frame010 \
  --frame-index 10

echo "=== Done ==="
