# Evaluate Scripts

This directory contains shell entry points for evaluating the video-based CRL methods in this codebase. All scripts call the same Python entry point: `src/evaluate.py`.

## Model Selection

The available evaluation targets are `ivae`, `tcl`, `caring`, `tdrl`, and `idol`.

## Main Scripts

For example:

```bash
bash scripts/evaluate/eval_ivae_fall_simple_right.sh
```

```bash
REPR_MODE=full bash scripts/evaluate/eval_ivae_fall_simple_right.sh
```

```bash
TOPK=8 EPOCH=80 bash scripts/evaluate/eval_ivae_fall_simple_right.sh
```
