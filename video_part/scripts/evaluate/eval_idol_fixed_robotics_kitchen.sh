#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
init_new_paths

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-}"
DATA_ROOT="${NEW_ROOT}/dataset/robotic_manipulation/kitchen"

DATASET_NAME="fixed_robotics_kitchen"
NUM_FRAMES=16
CLIPS_PER_VIDEO=2
BATCH_SIZE=16
LATENT_DIM=2048
Z_DIM=58
LAG=2
HIDDEN_DIM=128
MAX_SAMPLES=1000
GPU_ID="${GPU_ID:-3}"
REPR_MODE="${REPR_MODE:-agg}"
TOPK="${TOPK:-57}"
EPOCH="${EPOCH:-100}"
BETA="${BETA:-0.00025}"
GAMMA="${GAMMA:-0.0075}"
THETA="${THETA:-0.02}"

CKPT_PATH="${CKPT_PATH:-${NEW_ROOT}/training-runs/idol/idol_fixed_robotics_kitchen/checkpoints/vae_epoch${EPOCH}.pth}"
LOG_PATH="${LOG_PATH:-${NEW_ROOT}/training-runs/idol/idol_fixed_robotics_kitchen_eval_${REPR_MODE}.log}"

mkdir -p "$(dirname "${LOG_PATH}")"

if [[ -n "${CONDA_SH}" && -n "${CONDA_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" "${CODE_DIR}/evaluate.py" \
  --model idol \
  --repr_mode "${REPR_MODE}" \
  --topk "${TOPK}" \
  --dataset_name "${DATASET_NAME}" \
  --dataset_path "${DATA_ROOT}" \
  --num_frames "${NUM_FRAMES}" \
  --clips_per_video "${CLIPS_PER_VIDEO}" \
  --batch_size "${BATCH_SIZE}" \
  --latent_dim "${LATENT_DIM}" \
  --z_dim "${Z_DIM}" \
  --lag "${LAG}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --beta "${BETA}" \
  --gamma "${GAMMA}" \
  --theta "${THETA}" \
  --max_samples "${MAX_SAMPLES}" \
  --ckpt_path "${CKPT_PATH}" \
  > "${LOG_PATH}" 2>&1
