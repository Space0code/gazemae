# Data Layout

This folder stores raw eye-tracking inputs and generated embedding outputs.

## TrustME (used by `trustME/build_gazemae_embeddings.py`)

- Input root: `data/TrustMe/`
- Per-subject parquet files: `data/TrustMe/<subject>/tobii/*.parquet`
- Required columns: `window_id`, `TimeStamp`, `GazePointX`, `GazePointY`

## GazeMAE Paper Corpora (used by scripts in `gazemae/`)

- `data/ETRA2019/`
- `data/MIT-LOWRES/`
- Optional additional corpora supported by code:
  `data/Cerf2007-FIFA/`, `data/EMVIC2014/official_files/`

## Generated Outputs

- TrustME embedding outputs: `data/gazemae_outputs/`
- Intermediate preprocessed caches for legacy scripts are written under `generated-data/` (repo root).
