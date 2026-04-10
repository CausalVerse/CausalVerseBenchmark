#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
init_new_paths

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-}"

DATASET_ROOT="${NEW_ROOT}/dataset/physical_simulation/fall_simple"
PREPROCESS_PY="${NEW_ROOT}/src/preprocess/preprocess_crl_latents.py"
VAE_PATH="${NEW_ROOT}/pretrained_models"

NUM_FRAMES=16
INTERVAL=30
MAX_START_FRAME=2
IMAGE_SIZE=512
CLIPS_PER_VIDEO=1
ENCODE_BATCH_SIZE=8
VIEW="${VIEW:-front}"
VIEW="$(normalize_fall_latent_view "${VIEW}")"
VIEW_SUFFIX="$(latent_view_suffix "${VIEW}" "front")"
DATASET_TYPE="physics"

GPU_IDS=(0 1 2 3 4 5 6 7)
NUM_SHARDS="${#GPU_IDS[@]}"

RUN_NAME="fall_simple_sdvae_latent${VIEW_SUFFIX}"
LATENT_ROOT="${DATASET_ROOT}/latent/${VIEW}"
LOG_DIR="${NEW_ROOT}/logs/preprocess/${RUN_NAME}"

mkdir -p "${LOG_DIR}"
mkdir -p "${LATENT_ROOT}"

if [[ -n "${CONDA_SH}" && -n "${CONDA_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

echo "PYTHON_BIN=${PYTHON_BIN}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[error] python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c "import diffusers, decord, torch"; then
  echo "[error] Missing Python packages in ${PYTHON_BIN}. Need at least: diffusers, decord, torch" >&2
  echo "[hint] Example: CONDA_SH=\$HOME/miniconda3/etc/profile.d/conda.sh CONDA_ENV=<env> bash $0" >&2
  exit 2
fi

echo "NEW_ROOT=${NEW_ROOT}"
echo "WORKSPACE_ROOT=${WORKSPACE_ROOT}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "VIEW=${VIEW}"
echo "LATENT_ROOT=${LATENT_ROOT}"
echo "LOG_DIR=${LOG_DIR}"
echo "GPU_IDS=${GPU_IDS[*]}"
echo "NUM_SHARDS=${NUM_SHARDS}"

for shard_index in "${!GPU_IDS[@]}"; do
  gpu_id="${GPU_IDS[$shard_index]}"
  log_path="${LOG_DIR}/shard_${shard_index}.log"

  nohup env CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${PYTHON_BIN}" "${PREPROCESS_PY}" \
    --dataset-root "${DATASET_ROOT}" \
    --dataset-type "${DATASET_TYPE}" \
    --view "${VIEW}" \
    --vae-path "${VAE_PATH}" \
    --latent-root "${LATENT_ROOT}" \
    --num-frames "${NUM_FRAMES}" \
    --interval "${INTERVAL}" \
    --max-start-frame "${MAX_START_FRAME}" \
    --image-size "${IMAGE_SIZE}" \
    --clips-per-video "${CLIPS_PER_VIDEO}" \
    --encode-batch-size "${ENCODE_BATCH_SIZE}" \
    --shard-index "${shard_index}" \
    --num-shards "${NUM_SHARDS}" \
    > "${log_path}" 2>&1 &

  echo "started shard=${shard_index} gpu=${gpu_id} log=${log_path}"
done

echo "all preprocess jobs submitted in background"
