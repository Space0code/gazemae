# TrustME GazeMAE Embeddings Output

This folder stores outputs from:

```bash
python trustME/build_gazemae_embeddings.py ...
```

## Files

- `trustme_gazemae_embeddings.parquet`: one row per kept TrustME window.
- `trustme_gazemae_run_summary.json`: run config, paths, counts, and drop reasons.

## Embedding Shape

Current file (`trustme_gazemae_embeddings.parquet`) contains:

- `132,350` rows (windows)
- `281` columns total
- `25` metadata columns
- `128` position embedding columns: `z_pos_000` ... `z_pos_127`
- `128` velocity embedding columns: `z_vel_000` ... `z_vel_127`
- optional (not present in this run): `z_concat_000` ... `z_concat_255` when `--include-z-concat` is used

## Parquet Schema (practical view)

Metadata columns include:

- identity: `window_uid`, `subject`, `source_file`, `window_id`, `window_id_str`
- QC/info: `orig_len`, `valid_count`, `valid_fraction`, `start_t`, `end_t`
- resampling config: `target_hz`, `window_seconds`, `target_len`
- copied optional labels (if present upstream): `sleep_feedback`, `prompt_id`, `prompt_time`, `1`..`9`

Embedding columns are float features:

- `z_pos_*`: latent vector from the position encoder
- `z_vel_*`: latent vector from the velocity encoder

## How To Read

### Pandas

```python
import pandas as pd

df = pd.read_parquet("data/gazemae_outputs/trustme_gazemae_embeddings.parquet")

z_pos_cols = [c for c in df.columns if c.startswith("z_pos_")]
z_vel_cols = [c for c in df.columns if c.startswith("z_vel_")]

X_pos = df[z_pos_cols].to_numpy(dtype="float32")  # [N, 128]
X_vel = df[z_vel_cols].to_numpy(dtype="float32")  # [N, 128]
X = df[z_pos_cols + z_vel_cols].to_numpy(dtype="float32")  # [N, 256]
meta = df[["window_uid", "subject", "prompt_id", "sleep_feedback"]]
```

### PyArrow (schema only)

```python
import pyarrow.parquet as pq

pf = pq.ParquetFile("data/gazemae_outputs/trustme_gazemae_embeddings.parquet")
print(pf.metadata.num_rows)
print(pf.schema.names[:20])
```

## How Embeddings Were Calculated (summary)

Pipeline in `trustME/build_gazemae_embeddings.py`:

1. Load windowed Tobii parquet rows (`data/TrustMe/<subject>/tobii/*.parquet`).
2. Group by `window_id`.
3. Mark invalid samples when `GazePointX`/`GazePointY` are `-1` or `NaN`.
4. Drop low-quality windows (`min_valid_fraction`, `min_valid_frames`).
5. Interpolate invalid X/Y (linear + edge fill).
6. Resample each kept window to fixed length (`target_hz * window_seconds`, default `500 * 3 = 1500`).
7. Build velocity from position (`abs(diff(pos)) / ms_per_sample`, padded to same length).
8. Encode both signals with pretrained checkpoints:
   - `models/pos-i3738` for position
   - `models/vel-i8528` for velocity
9. Write parquet + summary JSON.

## Approximate Network Architecture

The checkpoints are GazeMAE temporal convolutional autoencoders (`gazemae/network/`):

- Encoder: stacked residual 1D dilated convolution blocks (TCN-style), global average pooling over time.
- Bottleneck: linear + ReLU + BatchNorm latent projection.
- Hierarchical mode: two latent branches (`64 + 64`) concatenated to `128` dims.
- Decoder: causal dilated residual conv decoder trained to reconstruct the input signal.

At inference in this pipeline, only `network.encode(...)` is used to extract latent vectors (not reconstructions).
