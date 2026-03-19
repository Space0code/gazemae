# GazeMAE for TrustME

This repository combines:
- the original GazeMAE code from the ICPR 2020 paper,
- pretrained position/velocity checkpoints,
- a TrustME-focused pipeline that builds window-level embeddings from Tobii parquet files.

Paper: **GazeMAE: General Representations of Eye Movements using a Micro-Macro Autoencoder**  
Preprint: https://arxiv.org/abs/2009.02437

## Quickstart: Build Embeddings for TrustME Data

From the repository root:

```bash
CONDA_NO_PLUGINS=true conda run -n trust-me python trustME/build_gazemae_embeddings.py \
  --input-root data/TrustMe \
  --out-dir data/gazemae_outputs \
  --model-pos models/pos-i3738 \
  --model-vel models/vel-i8528 \
  --target-hz 500 \
  --window-seconds 3 \
  --batch-size 256 \
  --device auto
```

Input layout:
- `data/TrustMe/<subject>/tobii/*.parquet`

Required columns in each parquet:
- `window_id`, `TimeStamp`, `GazePointX`, `GazePointY`

Main outputs:
- `data/gazemae_outputs/trustme_gazemae_embeddings.parquet`
- `data/gazemae_outputs/trustme_gazemae_run_summary.json`

Detailed usage: [`trustME/README.md`](trustME/README.md)

## Repository Map

- [`trustME/README.md`](trustME/README.md): TrustME embedding pipeline (recommended starting point)
- [`gazemae/README.md`](gazemae/README.md): original training/evaluation scripts for paper corpora
- [`data/README.md`](data/README.md): expected dataset layout
- [`models/README.md`](models/README.md): pretrained checkpoints
- [`tests/initial/README.md`](tests/initial/README.md): saved baseline evaluation notes

## Legacy GazeMAE Training/Evaluation

If you want to reproduce paper-style experiments, work from `gazemae/` and follow:
- [`gazemae/README.md`](gazemae/README.md)

The legacy scripts expect data under `data/` and use relative paths from inside `gazemae/`.
