#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
init_new_paths

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-}"
LATENT_VIEW="${LATENT_VIEW:-front}"
LATENT_VIEW="$(normalize_fixed_robotics_latent_view "${LATENT_VIEW}")"
VIEW_SUFFIX="$(latent_view_suffix "${LATENT_VIEW}" "front")"
DATA_ROOT="${NEW_ROOT}/dataset/robotic_manipulation/kitchen"

DATASET_NAME="fixed_robotics_kitchen"
NUM_FRAMES=16
CLIPS_PER_VIDEO=2
BATCH_SIZE=16
LATENT_DIM=2048
Z_DIM=57
LAG=2
CONTEXT_FRAMES=3
HIDDEN_DIM=128
MAX_SAMPLES=1000
GPU_ID="${GPU_ID:-3}"
REPR_MODE="${REPR_MODE:-agg}"
R2_SELECT_MODE="${R2_SELECT_MODE:-best}"
TOPK="${TOPK:-57}"
EPOCH="${EPOCH:-100}"
BETA="${BETA:-0.00025}"
GAMMA="${GAMMA:-0.0075}"

RUN_NAME="caring_fixed_robotics_kitchen${VIEW_SUFFIX}"
CKPT_PATH="${CKPT_PATH:-${NEW_ROOT}/training-runs/caring/${RUN_NAME}/checkpoints/vae_epoch${EPOCH}.pth}"
LOG_PATH="${LOG_PATH:-${NEW_ROOT}/training-runs/caring/${RUN_NAME}_eval_${REPR_MODE}.log}"

mkdir -p "$(dirname "${LOG_PATH}")"

if [[ -n "${CONDA_SH}" && -n "${CONDA_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" "${CODE_DIR}/evaluate.py" \
  --model caring \
  --repr_mode "${REPR_MODE}" \
  --r2_select_mode "${R2_SELECT_MODE}" \
  --topk "${TOPK}" \
  --dataset_name "${DATASET_NAME}" \
  --dataset_path "${DATA_ROOT}" \
  --latent_view "${LATENT_VIEW}" \
  --num_frames "${NUM_FRAMES}" \
  --clips_per_video "${CLIPS_PER_VIDEO}" \
  --batch_size "${BATCH_SIZE}" \
  --latent_dim "${LATENT_DIM}" \
  --z_dim "${Z_DIM}" \
  --lag "${LAG}" \
  --context_frames "${CONTEXT_FRAMES}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --beta "${BETA}" \
  --gamma "${GAMMA}" \
  --max_samples "${MAX_SAMPLES}" \
  --ckpt_path "${CKPT_PATH}" \
  > "${LOG_PATH}" 2>&1
