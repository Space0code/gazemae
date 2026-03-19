# GazeMAE Core Code

This directory contains the original GazeMAE training and evaluation code used for paper-style experiments.

## Main Entry Points

- `train.py`: train autoencoder models
- `evaluate.py`: extract/evaluate representations from pretrained models
- `train_supervised.py`: train supervised CNN baselines on supported tasks
- `settings.py`: shared CLI options and defaults

## Run Notes

- Run commands from inside `gazemae/` (scripts use relative paths like `../data`, `../models`, `../generated-data`).
- Legacy dependencies are listed in `requirements_conda.txt`.

## Example Commands

```bash
# from repo root
cd gazemae

# Train velocity model (paper-style setting)
python train.py --signal-type vel -bs 128 -vt 2 -hz 500 --hierarchical --slice-time-windows 2s-overlap

# Evaluate pretrained velocity model on ETRA2019
python evaluate.py --model-vel vel-i8528 --signal-type vel -hz 500 -vt 2 --corpora ETRA2019
```
