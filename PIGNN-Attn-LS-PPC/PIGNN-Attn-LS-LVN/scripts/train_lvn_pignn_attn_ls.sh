#!/usr/bin/env bash
# Train PIGNN-Attn-LS (VnFeat) on the LVN dataset.
#
# Equivalent to:
#   python -m train --config configs/lvn_pignn_attn_ls.yaml
# Any extra args are forwarded to the trainer and override YAML values.
#
# Example: 60 epochs, seed 43, with MLflow logging
#   ./scripts/train_lvn_pignn_attn_ls.sh --EPOCHS 60 --seed_value 43 --mlflow

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"

exec "$PYTHON" -m train \
  --config configs/lvn_pignn_attn_ls.yaml \
  "$@"
