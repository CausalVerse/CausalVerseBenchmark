#!/usr/bin/env bash
set -euo pipefail

# Run with the same effective parameters as the original script.
python3 export_causalverse_images.py \
  --repo-id "CausalVerse/CausalVerse_Image" \
  --hf-home "./.hf" \
  --raw-repo-dir "./CausalVerse_Image" \
  --image-root "./image" \
  --folder-case "upper" \
  --no-overwrite \
  --include-render-path-column \
  --download-allow-patterns "data/*.parquet" \
  --skip-download-if-local
