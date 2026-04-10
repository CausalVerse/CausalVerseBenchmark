# Evaluate Scripts

This directory contains shell entry points for evaluating the video-based CRL methods in this
codebase. All wrappers call the same Python entry point:

- `src/evaluate.py`

## Models

The available evaluation targets are `ivae`, `tcl`, `caring`, `tdrl`, and `idol`.

## Common Parameters

The shell wrappers are configured through environment variables. The most important ones are:

- `LATENT_VIEW`: selects which latent cache to read, that is `dataset/.../latent/<view>`
- `REPR_MODE`: chooses which representation to evaluate, typically `agg` or `full`
- `EPOCH`: selects the checkpoint epoch used by the default `CKPT_PATH`
- `CKPT_PATH`: overrides the checkpoint file directly
- `LOG_PATH`: overrides the output log path
- `R2_SELECT_MODE`: chooses how the nonlinear regressor is selected, typically `best` or `last`
- `TOPK`: over-complete top-k `MCC`
- `MAX_SAMPLES`: caps the number of evaluation samples
- `GPU_ID`: selects the GPU used by the wrapper
- `PYTHON_BIN`, `CONDA_SH`, `CONDA_ENV`: control the Python executable and optional Conda activation

Some wrappers also expose model-specific arguments that must stay consistent with the checkpoint
being evaluated, for example:

- `BETA`, `GAMMA`, `THETA`
- `SEGMENT_SIZE`
- `AUX_NOISE_DIM`, `AUX_NOISE_SCALE`

If you override these, make sure they still match how the checkpoint was trained.

## Metric Reporting

Current evaluation logs report:

- `MCC`: top-k average over the best GT-to-latent Pearson correlations
- `NonLinear R² Score`: nonlinear regression R² used by the evaluation pipeline

## Main Scripts

For example:

```bash
bash scripts/evaluate/eval_ivae_fall_simple.sh
```

```bash
LATENT_VIEW=right EPOCH=100 bash scripts/evaluate/eval_caring_fall_simple.sh
```

```bash
LATENT_VIEW=front REPR_MODE=full TOPK=8 \
bash scripts/evaluate/eval_tdrl_fixed_robotics_kitchen.sh
```

```bash
CKPT_PATH=training-runs/ivae/ivae_fall_simple_right/checkpoints/ivae_epoch100.pth \
bash scripts/evaluate/eval_ivae_fall_simple.sh
```

For exact defaults, always read the specific `eval_*.sh` wrapper you are about to run.
