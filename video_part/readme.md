# Video Part

This directory is the lean code tree. It contains the same `scripts/` and `src/` structure as the
working tree, but it does not currently carry the data, model, log, or run directories by
default.

If you want to run the shell wrappers in `repo/` directly, you need to create or symlink the
expected asset folders under this directory first.

## Minimal Layout Expected By The Scripts

Currently present:

- `scripts/`
- `src/`

You need to add the following if you want the wrappers to work unchanged:

- `dataset/`
- `pretrained_models/`
- `logs/`
- `training-runs/`

The expected populated layout is currently organized under two domains:

- `dataset/physical_simulation/`
- `dataset/robotic_manipulation/`

The scenes currently shown below are only examples, not an exhaustive list:

- `dataset/physical_simulation/fall_simple/{video,meta,latent/<view>}`
- `dataset/robotic_manipulation/kitchen/{video,meta,latent/<view>}`
- `dataset/robotic_manipulation/study/{video,meta,latent/<view>}`

Additional scenes should follow the same pattern:

- `dataset/<domain>/<scene>/{video,meta,latent/<view>}`

Typical output layout after you start running jobs:

- `logs/preprocess/<run_name>/shard_*.log`
- `training-runs/<method>/<run_name>/<run_name>.log`
- `training-runs/<method>/<run_name>/checkpoints/*.pth`
- `training-runs/<method>/<run_name>_eval_<repr>.log`

If you already maintain assets under another directory, the simplest approach is to copy or symlink
those four top-level folders into `repo/` so that the shell wrappers can keep using their default
paths.

## What To Place Here

Before running the wrappers in `repo/`, place:

1. Raw scene data under `dataset/.../video` and `dataset/.../meta`
2. Pretrained VAE files under `pretrained_models/`
3. Optional precomputed latents under `dataset/.../latent/<view>`

Recommended `pretrained_models/` layout:

```text
pretrained_models/
├── config.json
└── diffusion_pytorch_model.{safetensors,bin}
```

Only one weights file is typically needed. The exact filename depends on the local snapshot you
downloaded, but the directory should be a valid `diffusers.AutoencoderKL.from_pretrained(...)`
checkpoint root.

PyTorch and torchvision are intentionally not pinned in `src/requirements.txt`, because the correct
installation depends on whether you use CPU, CUDA, or ROCm. Install the appropriate PyTorch build
for your platform first, then install the remaining Python packages from `src/requirements.txt`.

If latents are already available, training and evaluation can skip the preprocess stage.

## End-to-End Workflow

Once the asset directories are present, the standard workflow is:

1. Prepare raw data and the pretrained VAE under this root.
2. Preprocess raw videos into latents.
3. Train one CRL method on a chosen view.
4. Evaluate the resulting checkpoint.

### Example A: `fall_simple` on the `right` view

Preprocess:

```bash
CONDA_SH=$HOME/miniconda3/etc/profile.d/conda.sh \
CONDA_ENV=<env> \
VIEW=right \
bash scripts/preprocess/preprocess_fall_simple_latent.sh
```

This writes latents to:

- `dataset/physical_simulation/fall_simple/latent/right/`

Train, for example with CaRiNG:

```bash
LATENT_VIEW=right bash scripts/train/train_caring_fall_simple.sh
```

Outputs go to:

- `training-runs/caring/caring_fall_simple_right/`

Evaluate:

```bash
LATENT_VIEW=right EPOCH=100 \
bash scripts/evaluate/eval_caring_fall_simple.sh
```

This writes:

- `training-runs/caring/caring_fall_simple_right_eval_agg.log`

### Example B: `fixed_robotics_kitchen` on the `front` view

Preprocess:

```bash
VIEW=front bash scripts/preprocess/preprocess_fixed_robotics_kitchen_latent.sh
```

Train:

```bash
LATENT_VIEW=front bash scripts/train/train_tdrl_fixed_robotics_kitchen.sh
```

Evaluate:

```bash
LATENT_VIEW=front EPOCH=100 \
bash scripts/evaluate/eval_tdrl_fixed_robotics_kitchen.sh
```

## Common Knobs

The same wrapper-level controls are used here:

- Environment: `PYTHON_BIN`, `CONDA_SH`, `CONDA_ENV`
- View selection: `VIEW` for preprocess, `LATENT_VIEW` for train/evaluate
- GPU control: `GPU_ID`, `GPU_IDS`, `USE_MULTI_GPU`
- Training size/schedule: `BATCH_SIZE`, `LR`, `EPOCHS`, `SAVE_INTERVAL`
- Model-specific weights: `BETA`, `GAMMA`, `THETA`, `LAG`, `CONTEXT_FRAMES`, `HIDDEN_DIM`
- Evaluation controls: `REPR_MODE`, `R2_SELECT_MODE`, `TOPK`, `EPOCH`, `MAX_SAMPLES`, `CKPT_PATH`

Current view conventions:

- `fall_simple`: typically `front`, `left`, `right`, `bird`
- `fixed_robotics_*`: typically `front`, `side`, `bird`, `agent`, `robot0_eye_in_hand`

Current script defaults are not identical across all datasets. For example, the `fall_simple`
train/evaluate wrappers default to `LATENT_VIEW=right`, while the kitchen preprocess wrapper still
defaults to `VIEW=front`. Read the target script before launching a long run.

## Where To Read Next

Use these files as the next level of detail:

- `src/README.md`: code-level overview of datasets, models, and entry points
- `src/preprocess/README.md`: latent extraction setup and VAE placement
- `src/data_modules/README.md`: dataset loader expectations and directory conventions
- `src/metrics/README.md`: metric implementation notes
- `scripts/evaluate/README.md`: evaluation wrapper behavior
- `scripts/train/*.sh`: the authoritative source for per-method default hyperparameters
- `scripts/preprocess/*.sh`: the authoritative source for per-scene preprocessing defaults

For exact runtime defaults, always trust the specific shell script and Python entry point over this
summary document.
