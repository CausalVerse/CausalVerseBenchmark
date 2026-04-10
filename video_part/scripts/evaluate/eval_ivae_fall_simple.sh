#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
init_new_paths

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-}"
LATENT_VIEW="${LATENT_VIEW:-right}"
LATENT_VIEW="$(normalize_fall_latent_view "${LATENT_VIEW}")"
VIEW_SUFFIX="$(latent_view_suffix "${LATENT_VIEW}" "right")"

SRC_DATA_ROOT="${NEW_ROOT}/dataset/physical_simulation/fall_simple"

DATASET_NAME="fall"
NUM_FRAMES=16
CLIPS_PER_VIDEO=1
BATCH_SIZE=16
Z_DIM=10
MAX_SAMPLES=1000
GPU_ID="${GPU_ID:-3}"
AUX_NOISE_DIM="${AUX_NOISE_DIM:-0}"
AUX_NOISE_SCALE="${AUX_NOISE_SCALE:-1.0}"
REPR_MODE="${REPR_MODE:-agg}"
R2_SELECT_MODE="${R2_SELECT_MODE:-best}"
TOPK="${TOPK:-4}"
EPOCH="${EPOCH:-100}"

RUN_NAME="ivae_fall_simple${VIEW_SUFFIX}"
CKPT_PATH="${CKPT_PATH:-${NEW_ROOT}/training-runs/ivae/${RUN_NAME}/checkpoints/ivae_epoch${EPOCH}.pth}"
LOG_PATH="${LOG_PATH:-${NEW_ROOT}/training-runs/ivae/${RUN_NAME}_eval_${REPR_MODE}.log}"

mkdir -p "$(dirname "${LOG_PATH}")"

if [[ -n "${CONDA_SH}" && -n "${CONDA_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" "${CODE_DIR}/evaluate.py" \
  --model ivae \
  --repr_mode "${REPR_MODE}" \
  --r2_select_mode "${R2_SELECT_MODE}" \
  --topk "${TOPK}" \
  --dataset_name "${DATASET_NAME}" \
  --dataset_path "${SRC_DATA_ROOT}" \
  --latent_view "${LATENT_VIEW}" \
  --num_frames "${NUM_FRAMES}" \
  --clips_per_video "${CLIPS_PER_VIDEO}" \
  --batch_size "${BATCH_SIZE}" \
  --z_dim "${Z_DIM}" \
  --max_samples "${MAX_SAMPLES}" \
  --ckpt_path "${CKPT_PATH}" \
  --aux_noise_dim "${AUX_NOISE_DIM}" \
  --aux_noise_scale "${AUX_NOISE_SCALE}" \
  > "${LOG_PATH}" 2>&1
