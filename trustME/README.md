# TrustMe GazeMAE Embeddings

This subproject computes window-level embeddings for TrustMe Tobii data using pretrained GazeMAE models for both position and velocity signals.

## What It Does

1. Loads `data/TrustMe/s_*/tobii/*.parquet`.
2. Groups rows by `window_id` (window key is `(subject, source_file, window_id)`).
3. Applies X/Y-only QC at native rate:
   1. Invalid row: `GazePointX == -1` or `GazePointY == -1` or NaN.
   2. Drop window if `valid_fraction < 0.3`.
   3. Drop window if `valid_count < 32`.
   4. Interpolate invalid rows (`linear`, `limit=max_invalid_frames-1`), then `ffill+bfill`.
   5. Drop if NaN remains after interpolation.
4. Resamples each kept window to fixed 3s at 500 Hz (`target_len=1500`) for X/Y position.
5. Builds velocity as `abs(diff(position)) / (1000/500)`, padded back to length 1500.
6. Encodes both signals with pretrained models (`pos-i3738`, `vel-i8528`).
7. Writes:
   1. `trustme_gazemae_embeddings.parquet`
   2. `trustme_gazemae_run_summary.json`

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

## Main CLI Options

- `--input-root` (default: `data/TrustMe`)
- `--out-dir` (default: `data/gazemae_outputs`)
- `--model-pos` (default: `models/pos-i3738`)
- `--model-vel` (default: `models/vel-i8528`)
- `--target-hz` (default: `500`)
- `--window-seconds` (default: `3`)
- `--max-invalid-frames` (default: `60`)
- `--min-valid-fraction` (default: `0.3`)
- `--min-valid-frames` (default: `32`)
- `--subjects` optional comma-separated subject filter
- `--batch-size`
- `--device` (`auto`, `cpu`, `cuda`, etc.)
- `--num-workers` reserved for future parallel loading
- `--include-z-concat` optional 256-d concat columns
- `--max-files-per-subject` optional debug/smoke cap
- `--max-windows-per-file` optional debug/smoke cap

## Output Schema

`trustme_gazemae_embeddings.parquet` contains one row per kept window:

- Metadata:
  - `window_uid`, `subject`, `source_file`, `window_id`, `window_id_str`
  - `orig_len`, `valid_count`, `valid_fraction`
  - `start_t`, `end_t`, `target_hz`, `window_seconds`, `target_len`
  - Optional copied window-level labels if present: `sleep_feedback`, `prompt_id`, `prompt_time`, `1`..`9`
- Embedding columns:
  - `z_pos_000` .. `z_pos_127`
  - `z_vel_000` .. `z_vel_127`
  - Optional `z_concat_000` .. `z_concat_255` if `--include-z-concat` is passed

`trustme_gazemae_run_summary.json` contains config, model paths, counts, drop reasons, and output paths.

## Tests

Run unit + smoke tests:

```bash
CONDA_NO_PLUGINS=true conda run -n trust-me python -m trustME.test_build_gazemae_embeddings
```

The smoke test uses one smallest parquet file per subject and verifies:

1. no NaNs in embedding columns,
2. 128-dim pos and vel embeddings,
3. row count consistency (`rows_written == total_kept`).
