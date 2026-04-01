#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
init_new_paths

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-}"
DATA_ROOT="${NEW_ROOT}/dataset/robotic_manipulation/kitchen"

DATASET_NAME="fixed_robotics_kitchen"
NUM_FRAMES=16
CLIPS_PER_VIDEO=2
BATCH_SIZE_PER_GPU=32
Z_DIM=57
LR=1e-4
EPOCHS=100
SAVE_INTERVAL=100
BETA=0.001
BETA_START=0.0
BETA_WARMUP_EPOCHS=50
AUX_NOISE_DIM="${AUX_NOISE_DIM:-0}"
AUX_NOISE_SCALE="${AUX_NOISE_SCALE:-1.0}"
USE_MULTI_GPU="${USE_MULTI_GPU:-1}"
GPU_IDS=(0 1 2 3 4 5 6 7)

RUN_NAME="ivae_fixed_robotics_kitchen"
LOG_DIR="${NEW_ROOT}/training-runs/ivae/${RUN_NAME}"
TRAIN_LOG="${LOG_DIR}/${RUN_NAME}.log"
TENSORBOARD_DIR="${LOG_DIR}/tensorboard"
SRC_SNAPSHOT_DIR="${LOG_DIR}/src"

mkdir -p "${LOG_DIR}" "${TENSORBOARD_DIR}" "${LOG_DIR}/checkpoints" "${LOG_DIR}/videos" "${SRC_SNAPSHOT_DIR}/lib"
cp -f "${CODE_DIR}/lib/ivae.py" "${SRC_SNAPSHOT_DIR}/lib/ivae.py"
cp -f "${CODE_DIR}/train_ivae.py" "${SRC_SNAPSHOT_DIR}/train_ivae.py"

if [[ -n "${CONDA_SH}" && -n "${CONDA_ENV}" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

echo "PYTHON_BIN=${PYTHON_BIN}"
echo "PYTHON_EXE=$(which "${PYTHON_BIN}")"

if [[ "${USE_MULTI_GPU}" == "1" ]]; then
  GPU_CSV="$(IFS=,; echo "${GPU_IDS[*]}")"
  nohup env CUDA_VISIBLE_DEVICES="${GPU_CSV}" TENSORBOARD_DIR="${TENSORBOARD_DIR}" \
    "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${#GPU_IDS[@]}" \
    "${CODE_DIR}/train_ivae.py" \
    --dataset_name "${DATASET_NAME}" \
    --dataset_path "${DATA_ROOT}" \
    --num_frames "${NUM_FRAMES}" \
    --clips_per_video "${CLIPS_PER_VIDEO}" \
    --batch_size "${BATCH_SIZE_PER_GPU}" \
    --z_dim "${Z_DIM}" \
    --lr "${LR}" \
    --epochs "${EPOCHS}" \
    --save_interval "${SAVE_INTERVAL}" \
    --log_dir "${LOG_DIR}" \
    --beta "${BETA}" \
    --beta_start "${BETA_START}" \
    --beta_warmup_epochs "${BETA_WARMUP_EPOCHS}" \
    --aux_noise_dim "${AUX_NOISE_DIM}" \
    --aux_noise_scale "${AUX_NOISE_SCALE}" \
    > "${TRAIN_LOG}" 2>&1 &
else
  GPU_ID="${GPU_ID:-0}"
  nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" TENSORBOARD_DIR="${TENSORBOARD_DIR}" \
    "${PYTHON_BIN}" "${CODE_DIR}/train_ivae.py" \
    --dataset_name "${DATASET_NAME}" \
    --dataset_path "${DATA_ROOT}" \
    --num_frames "${NUM_FRAMES}" \
    --clips_per_video "${CLIPS_PER_VIDEO}" \
    --batch_size "${BATCH_SIZE_PER_GPU}" \
    --z_dim "${Z_DIM}" \
    --lr "${LR}" \
    --epochs "${EPOCHS}" \
    --save_interval "${SAVE_INTERVAL}" \
    --log_dir "${LOG_DIR}" \
    --beta "${BETA}" \
    --beta_start "${BETA_START}" \
    --beta_warmup_epochs "${BETA_WARMUP_EPOCHS}" \
    --aux_noise_dim "${AUX_NOISE_DIM}" \
    --aux_noise_scale "${AUX_NOISE_SCALE}" \
    > "${TRAIN_LOG}" 2>&1 &
fi

echo "training submitted: ${TRAIN_LOG}"
