# Preprocess

Preprocessing exists because the video benchmarks operate on high-resolution clips, while the
temporal CRL pipeline is designed to train on a compact, shared latent representation instead of
raw pixels. Following the paper's video setup, preprocessing uses a pretrained VAE to convert each
video clip into reusable superpixel-style latents once, so downstream methods can be compared on
the same input space with much lower compute and more stable training. In practice, this step is
run once per dataset split or camera view, and the generated latents are then reused by all
downstream experiments.

## Pretrained VAE

The preprocessing code currently uses the image VAE
[`stabilityai/sd-vae-ft-mse`](https://huggingface.co/stabilityai/sd-vae-ft-mse) by default. This
matches the implementation in `src/preprocess/preprocess_crl_latents.py`, which loads the VAE
through `diffusers.AutoencoderKL.from_pretrained(...)` and assumes 512x512 inputs, producing
latents of shape `[T, 4, 64, 64]`.

The paper also discusses a video VAE option,
[`THUDM/CogVideoX-2b`](https://huggingface.co/THUDM/CogVideoX-2b), but that is not what the current
preprocessing script uses. The script in this directory only encodes with `AutoencoderKL`, so the
relevant default for preprocessing is `sd-vae-ft-mse`.

The current codebase uses the following path convention:

- `src/preprocess/preprocess_crl_latents.py` accepts `--vae-path`, so you can pass either a
  local directory or a Hugging Face model id directly.
- The helper scripts
  `scripts/preprocess/preprocess_fall_simple_latents.sh` and
  `scripts/preprocess/preprocess_fall_simple_latents_right.sh` default to
  `VAE_PATH="${REPO_ROOT}/pretrained_models"` (or the equivalent repository root path resolved by
  the helper script), so by default they expect the SD-VAE files to be available under
  `pretrained_models/`.
- Several downstream training scripts also assume a local image-VAE checkout for the non-`_vid`
  case, but this is not completely uniform across all files. For example, `src/train_ivae.py`,
  `src/train_tcl.py`, `src/train_tdrl.py`, `src/train_idol.py`, and `src/train_caring.py` load the
  image decoder from a repository-local `../pretrained_models` path, while other code may use a
  different local path string.

So if you use the shell scripts and later want training-time reconstructions or decoded samples to
work without code changes, the safest setup is to place a local snapshot of
`stabilityai/sd-vae-ft-mse` under `pretrained_models/`.

## Example

For the free-fall dataset, preprocessing can be launched through:

- `scripts/preprocess/preprocess_fall_simple_latents.sh`
- `scripts/preprocess/preprocess_fall_simple_latents_right.sh`

These scripts call the preprocessing code with the dataset-specific VAE checkpoint, camera view,
frame count, image size, shard count, and output location.

## Dataset Placement

The source dataset should stay under the repository dataset tree, for example:

- `dataset/physical_simulation/free_fall_simple/`

The generated latent files are also written back under that dataset tree, for example:

- `dataset/physical_simulation/free_fall_simple/latents/front/`
- `dataset/physical_simulation/free_fall_simple/latents/right/`

This keeps the preprocessed latents colocated with the original videos and metadata, matching the
directory layout expected by the downstream dataset loaders and training scripts.

When adapting preprocessing to a different dataset, the main things to verify are the dataset root
directory, the target latent output directory, the camera view or view subset, the number of
frames per clip, the temporal sampling interval, the maximum start frame policy, the VAE checkpoint
used for latent extraction, the number of shards or GPUs used during preprocessing, and similar
dataset-specific choices.

## Adapting to Other Datasets

For a new dataset, the usual workflow is:

1. Place the raw data under the repository dataset tree.
2. Decide where the latent output directory should live inside that dataset.
3. Create or adapt a shell script under `scripts/preprocess/` with the correct dataset-specific
   settings.
4. Make sure the corresponding dataset loader reads from the same latent directory layout.

The key requirement is consistency: preprocessing, dataset loading, and training must all agree on
the latent directory layout and on the clip settings used to generate those latents.
