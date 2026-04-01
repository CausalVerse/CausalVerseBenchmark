# Video Part

This directory contains the main temporal causal representation learning implementations for videos.
The current codebase includes `iVAE`, `TCL`, `CaRiNG`, `TDRL`, and `IDOL`.

The directory structure is organized by responsibility: `data_modules/` contains dataset wrappers
and dataset construction utilities, `lib/` contains the model implementations and reusable neural
components, `metrics/` contains the MCC and R² code used at evaluation time, and `preprocess/`
contains latent preprocessing utilities. The top-level files in this directory provide the main
training and evaluation entry points together with the shared frame-latent backbone.

## Example

Using `iVAE` as an example, training on the right-view latents of the
`physical_simulation/free_fall_simple` scene can be launched with:

```bash
python src/train_ivae.py \
  --dataset_name fall \
  --dataset_path dataset/physical_simulation/free_fall_simple \
  --latent_view right \
  --num_frames 16 \
  --clips_per_video 1 \
  --batch_size 32 \
  --z_dim 10 \
  --lr 1e-4 \
  --epochs 100 \
  --save_interval 200 \
  --log_dir training-runs/ivae/ivae_fall_simple_right \
  --beta 0.001 \
  --beta_start 0.0 \
  --beta_warmup_epochs 50
```

Evaluation can be launched with:

```bash
python src/evaluate.py \
  --model ivae \
  --repr_mode agg \
  --dataset_name fall \
  --dataset_path dataset/physical_simulation/free_fall_simple \
  --latent_view right \
  --num_frames 16 \
  --clips_per_video 1 \
  --batch_size 16 \
  --z_dim 10 \
  --max_samples 1000 \
  --ckpt_path training-runs/ivae/ivae_fall_simple_right/checkpoints/ivae_epoch100.pth
```

For latent-based training and evaluation, the dataset root should point to the scene directory
under the domain-level dataset tree, while the latent subdirectory is selected separately through
`--latent_view`. For example, `--dataset_path dataset/physical_simulation/free_fall_simple` with
`--latent_view right` reads from `dataset/physical_simulation/free_fall_simple/latents/right/`.

The evaluation entry point is intentionally flexible. Depending on the experiment, you can choose
which ground-truth causal factors to evaluate against within each dataset definition, whether to
use the full representation or an aggregated representation (`full` vs `agg`), and which
evaluation strategy to apply, such as the target GT subset, the top-k used by MCC, and the
corresponding regression setting for R².

More detailed configuration options can be found in the corresponding shell scripts under
`scripts/train/` and `scripts/evaluate/`.
