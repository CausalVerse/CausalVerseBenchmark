# Evaluate Scripts

This directory now uses a single evaluation entry point:

- `src/evaluate.py`

Legacy split entry points such as `evaluate7.py` and `evaluate7_2.py` are no longer needed.
Evaluation behavior is controlled through arguments.

## Unified Modes

`evaluate.py` supports:

```bash
--mcc_mode mean
--mcc_mode full
```

`--mcc_mode mean`:

- Reshapes each `2048 = 8 x 16 x 16` latent into `8 x 16 x 16`
- Applies spatial mean pooling
- Computes MCC from the resulting 8D representation

`--mcc_mode full`:

- Uses the original high-dimensional latent directly
- Computes MCC from the full representation
- Uses the current top-k correlation path instead of the older slow Hungarian-based path

## Model Selection

This codebase only keeps CRL evaluation entry points:

```bash
--model ivae
--model tcl
--model caring
--model tdrl
--model idol
```

## Current iVAE Setup

The current `iVAE` implementation is an engineering variant rather than a strict
paper reproduction:

- It uses the shared frame-latent VAE backbone
- The default auxiliary variable setup is `time only`
- Optional auxiliary noise is also supported:

```bash
--aux_noise_dim N
--aux_noise_scale S
```

If you trained with auxiliary noise enabled, keep the same configuration at evaluation time.

## Main Script

Primary evaluation script:

- `scripts/evaluate/eval_ivae_fall_simple_right.sh`

Default behavior:

- `EPOCH=100`
- `MCC_MODE=full`
- Computes both `MCC` and `R2`

Examples:

```bash
bash scripts/evaluate/eval_ivae_fall_simple_right.sh
```

```bash
MCC_MODE=mean bash scripts/evaluate/eval_ivae_fall_simple_right.sh
```

```bash
EPOCH=80 bash scripts/evaluate/eval_ivae_fall_simple_right.sh
```

## Debug Scripts

Two debug scripts are still kept for MCC-only inspection, and both call `evaluate.py`:

- `scripts/evaluate/eval_ivae_fall_simple_right_debug_mcc_mean.sh`
- `scripts/evaluate/eval_ivae_fall_simple_right_debug_mcc_full.sh`

They differ only in the `--mcc_mode` argument and both add:

```bash
--debug_mcc
```
