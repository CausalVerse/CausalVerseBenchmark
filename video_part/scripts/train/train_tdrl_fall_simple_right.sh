#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
init_new_paths

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-}"

SRC_DATA_ROOT="${NEW_ROOT}/dataset/physical_simulation/free_fall_simple"
VIEW_DATA_ROOT="${NEW_ROOT}/tmp/train_data/free_fall_simple_right"

DATASET_NAME="fall"
NUM_FRAMES=16
CLIPS_PER_VIDEO=1
BATCH_SIZE=16
LATENT_DIM=2048
Z_DIM=8
LAG=2
HIDDEN_DIM=128
LR=1e-4
EPOCHS=100
SAVE_INTERVAL=1000
BETA=0.00025
GAMMA=0.0075

USE_MULTI_GPU="${USE_MULTI_GPU:-1}"
GPU_IDS=(0 1 2 3 4 5 6 7)

RUN_NAME="tdrl_fall_simple_right"
LOG_DIR="${NEW_ROOT}/training-runs/tdrl/${RUN_NAME}"
TRAIN_LOG="${LOG_DIR}/${RUN_NAME}.log"
TENSORBOARD_DIR="${LOG_DIR}/tensorboard"
SRC_SNAPSHOT_DIR="${LOG_DIR}/src"

mkdir -p "${VIEW_DATA_ROOT}" "${LOG_DIR}" "${TENSORBOARD_DIR}" "${LOG_DIR}/checkpoints" "${LOG_DIR}/videos" "${SRC_SNAPSHOT_DIR}/lib"
cp -f "${CODE_DIR}/lib/tdrl.py" "${SRC_SNAPSHOT_DIR}/lib/tdrl.py"
cp -f "${CODE_DIR}/train_tdrl.py" "${SRC_SNAPSHOT_DIR}/train_tdrl.py"
ln -sfn "${SRC_DATA_ROOT}/meta" "${VIEW_DATA_ROOT}/meta"
ln -sfn "${SRC_DATA_ROOT}/video" "${VIEW_DATA_ROOT}/video"
mkdir -p "${VIEW_DATA_ROOT}/latents"
ln -sfn "${SRC_DATA_ROOT}/latents/right" "${VIEW_DATA_ROOT}/latents/front"

if [[ -n "${CONDA_SH}" && -n "${CONDA_ENV}" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

echo "PYTHON_BIN=${PYTHON_BIN}"
echo "PYTHON_EXE=$(which "${PYTHON_BIN}")"

echo "GPU_IDS=${GPU_IDS[*]}"
echo "USE_MULTI_GPU=${USE_MULTI_GPU}"

COMMON_ARGS=(
  --dataset_name "${DATASET_NAME}"
  --dataset_path "${VIEW_DATA_ROOT}"
  --num_frames "${NUM_FRAMES}"
  --clips_per_video "${CLIPS_PER_VIDEO}"
  --batch_size "${BATCH_SIZE}"
  --latent_dim "${LATENT_DIM}"
  --z_dim "${Z_DIM}"
  --lag "${LAG}"
  --hidden_dim "${HIDDEN_DIM}"
  --lr "${LR}"
  --epochs "${EPOCHS}"
  --save_interval "${SAVE_INTERVAL}"
  --log_dir "${LOG_DIR}"
  --beta "${BETA}"
  --gamma "${GAMMA}"
)

if [[ "${USE_MULTI_GPU}" == "1" ]]; then
  NUM_GPUS="${#GPU_IDS[@]}"
  CUDA_DEVICES="$(IFS=,; echo "${GPU_IDS[*]}")"
  nohup env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" TENSORBOARD_DIR="${TENSORBOARD_DIR}" \
    "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}" \
    "${CODE_DIR}/train_tdrl.py" \
    "${COMMON_ARGS[@]}" \
    > "${TRAIN_LOG}" 2>&1 &
else
  GPU_ID="${GPU_IDS[0]}"
  nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" TENSORBOARD_DIR="${TENSORBOARD_DIR}" \
    "${PYTHON_BIN}" "${CODE_DIR}/train_tdrl.py" \
    "${COMMON_ARGS[@]}" \
    > "${TRAIN_LOG}" 2>&1 &
fi

echo "training submitted: ${TRAIN_LOG}"
