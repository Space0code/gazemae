# TrustME Embedding Pipeline

Use this folder to build GazeMAE embeddings for TrustME Tobii windows.

## Input Contract

Expected layout:
- `data/TrustMe/<subject>/tobii/*.parquet`

Required parquet columns:
- `window_id`
- `TimeStamp`
- `GazePointX`
- `GazePointY`

Optional window-level metadata (copied to output if present):
- `sleep_feedback`, `prompt_id`, `prompt_time`, `1`..`9`

## What The Script Does

`build_gazemae_embeddings.py`:
- groups rows by `window_id`,
- applies X/Y quality control (`-1`/NaN invalid, threshold-based filtering, interpolation),
- resamples each kept window to fixed length (`target_hz * window_seconds`),
- builds velocity from position,
- encodes position and velocity with pretrained GazeMAE checkpoints,
- writes parquet embeddings plus a JSON run summary.

## Run

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

Useful flags:
- `--subjects s_004_pk,s_005_ak`
- `--include-z-concat`
- `--max-files-per-subject`, `--max-windows-per-file` for quick smoke runs

## Outputs

- `trustme_gazemae_embeddings.parquet`  
  One row per kept window, with metadata and embedding columns:
  `z_pos_000..127`, `z_vel_000..127` (and optional `z_concat_000..255`).
- `trustme_gazemae_run_summary.json`  
  Config, model paths, counters, drop reasons, and output paths.

## Tests

```bash
CONDA_NO_PLUGINS=true conda run -n trust-me python -m trustME.test_build_gazemae_embeddings
```
