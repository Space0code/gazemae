"""Build TrustMe window-level GazeMAE embeddings for position and velocity signals.

This script processes pre-windowed TrustMe Tobii parquet files, applies X/Y quality
control, upsamples each kept 3-second window to a fixed 500 Hz grid, and computes
embeddings from pretrained GazeMAE checkpoints.

Example
-------
python trustME/build_gazemae_embeddings.py \
  --input-root data/TrustMe \
  --out-dir data/gazemae_outputs \
  --model-pos models/pos-i3738 \
  --model-vel models/vel-i8528 \
  --target-hz 500 \
  --window-seconds 3 \
  --batch-size 256 \
  --device auto
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


LOGGER = logging.getLogger(__name__)
EXTRA_METADATA_COLUMNS: tuple[str, ...] = (
    "sleep_feedback",
    "prompt_id",
    "prompt_time",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
)
REQUIRED_COLUMNS: tuple[str, ...] = (
    "window_id",
    "TimeStamp",
    "GazePointX",
    "GazePointY",
)


@dataclass(frozen=True)
class BuildConfig:
    """Configuration for TrustMe embedding generation."""

    input_root: Path
    out_dir: Path
    model_pos: Path
    model_vel: Path
    target_hz: int = 500
    window_seconds: float = 3.0
    max_invalid_frames: int = 60
    min_valid_fraction: float = 0.3
    min_valid_frames: int = 32
    subjects: tuple[str, ...] = ()
    batch_size: int = 256
    device: str = "auto"
    num_workers: int = 0
    include_z_concat: bool = False
    max_files_per_subject: int = 0
    max_windows_per_file: int = 0
    verbose: bool = False


@dataclass(frozen=True)
class WindowProcessed:
    """Processed payload for one kept window."""

    metadata: dict[str, Any]
    pos_signal: np.ndarray
    vel_signal: np.ndarray


@dataclass
class BuildStats:
    """Run statistics for the embedding pipeline."""

    subjects_processed: int = 0
    parquet_files_processed: int = 0
    total_candidates: int = 0
    total_kept: int = 0
    total_dropped: int = 0
    drop_reason_counts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.drop_reason_counts is None:
            self.drop_reason_counts = {}

    def add_drop(self, reason: str) -> None:
        """Increment drop counters for the given reason."""
        self.total_dropped += 1
        self.drop_reason_counts[reason] = self.drop_reason_counts.get(reason, 0) + 1


def setup_logging(verbose: bool) -> None:
    """Initialize logging format and level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_subjects(raw_value: str | None) -> tuple[str, ...]:
    """Parse comma-separated subjects from CLI."""
    if not raw_value:
        return ()
    values = tuple(sorted({part.strip() for part in raw_value.split(",") if part.strip()}))
    return values


def resolve_device(device_arg: str) -> torch.device:
    """Resolve requested device string to a torch device."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def ensure_gazemae_import_path(repo_root: Path) -> None:
    """Add the local `gazemae/` directory to import path for checkpoint unpickling."""
    gazemae_module_root = repo_root / "gazemae"
    if str(gazemae_module_root) not in sys.path:
        sys.path.insert(0, str(gazemae_module_root))


def load_pretrained_network(model_path: Path, device: torch.device) -> torch.nn.Module:
    """Load pretrained GazeMAE checkpoint and return initialized network."""
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

    LOGGER.info("Loading checkpoint: %s", model_path)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    network = checkpoint.get("network", checkpoint.get("model"))
    if network is None:
        raise KeyError(f"Checkpoint does not contain 'network' or 'model': {model_path}")

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError(f"Checkpoint missing 'model_state_dict': {model_path}")
    network.load_state_dict(state_dict)
    network = network.to(device).eval()
    return network


def discover_subject_dirs(input_root: Path, subjects: tuple[str, ...]) -> list[Path]:
    """Discover subject directories under input root, optionally filtered."""
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    all_subjects = sorted([p for p in input_root.iterdir() if p.is_dir()])
    if not subjects:
        return all_subjects

    subject_set = set(subjects)
    filtered = [p for p in all_subjects if p.name in subject_set]
    missing = sorted(subject_set - {p.name for p in filtered})
    if missing:
        raise ValueError(f"Requested subjects not found under {input_root}: {missing}")
    return filtered


def discover_parquet_files(subject_dir: Path, max_files_per_subject: int = 0) -> list[Path]:
    """Return sorted parquet files from a subject's `tobii` directory."""
    tobii_dir = subject_dir / "tobii"
    if not tobii_dir.exists():
        raise FileNotFoundError(f"Missing Tobii directory for {subject_dir.name}: {tobii_dir}")

    parquet_files = sorted(tobii_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {tobii_dir}")

    if max_files_per_subject > 0:
        return parquet_files[:max_files_per_subject]
    return parquet_files


def get_parquet_columns(path: Path) -> list[str]:
    """Read parquet schema columns without loading full table."""
    try:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema.names)
    except Exception:
        try:
            import fastparquet

            return list(fastparquet.ParquetFile(path).columns)
        except Exception as exc:
            raise RuntimeError(f"Failed reading parquet schema for {path}") from exc


def window_id_to_str(window_id: Any) -> str:
    """Convert window id to stable compact string representation."""
    if isinstance(window_id, (int, np.integer)):
        return str(int(window_id))
    if isinstance(window_id, (float, np.floating)) and float(window_id).is_integer():
        return str(int(window_id))
    return str(window_id)


def scalar_from_window(window_df: pd.DataFrame, column: str, window_uid: str) -> Any:
    """Return stable scalar for a window-level metadata column."""
    if column not in window_df.columns:
        return np.nan
    non_null_values = window_df[column].dropna().unique()
    if len(non_null_values) > 1:
        raise ValueError(
            f"Conflicting values in column '{column}' for window {window_uid}: {non_null_values[:5]}"
        )
    if len(non_null_values) == 0:
        return np.nan
    return non_null_values[0]


def build_invalid_mask_xy(window_df: pd.DataFrame) -> np.ndarray:
    """Build invalid mask using X/Y sentinel and NaN checks."""
    signals = window_df.loc[:, ["GazePointX", "GazePointY"]]
    sentinel_invalid = (signals == -1.0).any(axis=1).to_numpy()
    nan_invalid = signals.isna().any(axis=1).to_numpy()
    return sentinel_invalid | nan_invalid


def interpolate_xy(signals: pd.DataFrame, invalid_mask: np.ndarray, max_invalid_frames: int) -> pd.DataFrame:
    """Interpolate invalid X/Y rows with linear interpolation then edge fill."""
    out = signals.astype(float).copy()
    out.loc[invalid_mask, :] = np.nan

    limit = max(max_invalid_frames - 1, 0)
    limit = min(limit, max(len(out) - 1, 0))
    if limit > 0:
        out = out.interpolate(method="linear", limit=limit)
    out = out.ffill().bfill()
    return out


def _interp_1d(old_x: np.ndarray, old_y: np.ndarray, new_x: np.ndarray) -> np.ndarray:
    """Interpolate a 1D signal with edge-value extrapolation."""
    if len(old_x) < 2:
        return np.full_like(new_x, fill_value=float(old_y[0]), dtype=np.float32)
    return np.interp(new_x, old_x, old_y).astype(np.float32)


def resample_xy_to_fixed_len(
    xy: np.ndarray,
    timestamps: np.ndarray | None,
    target_len: int,
    window_seconds: float,
) -> np.ndarray:
    """Resample [2, T] XY signal into deterministic fixed target length."""
    if xy.ndim != 2 or xy.shape[0] != 2:
        raise ValueError(f"Expected XY shape [2, T], got {xy.shape}")
    if xy.shape[1] < 1:
        raise ValueError("Cannot resample empty XY signal.")
    if target_len <= 0:
        raise ValueError(f"Invalid target length: {target_len}")

    if xy.shape[1] == 1:
        return np.repeat(xy.astype(np.float32), repeats=target_len, axis=1)

    old_axis: np.ndarray
    new_axis: np.ndarray
    use_timestamps = (
        timestamps is not None
        and len(timestamps) == xy.shape[1]
        and np.isfinite(timestamps).all()
    )
    if use_timestamps:
        shifted = timestamps.astype(np.float64) - float(timestamps[0])
        unique_axis, unique_indices = np.unique(shifted, return_index=True)
        if len(unique_axis) >= 2 and unique_axis[-1] > 0:
            old_axis = unique_axis
            xy = xy[:, unique_indices]
            new_axis = np.linspace(
                0.0,
                float(window_seconds) * 1000.0,
                num=target_len,
                endpoint=False,
                dtype=np.float64,
            )
        else:
            use_timestamps = False
    if not use_timestamps:
        old_axis = np.arange(xy.shape[1], dtype=np.float64)
        new_axis = np.linspace(0.0, float(xy.shape[1] - 1), num=target_len, dtype=np.float64)

    out = np.zeros((2, target_len), dtype=np.float32)
    out[0] = _interp_1d(old_axis, xy[0], new_axis)
    out[1] = _interp_1d(old_axis, xy[1], new_axis)
    return out


def build_velocity_from_position(pos_signal: np.ndarray, target_hz: int) -> np.ndarray:
    """Build velocity [2, T] from position [2, T] in px/ms with 1-step right padding."""
    if pos_signal.ndim != 2 or pos_signal.shape[0] != 2:
        raise ValueError(f"Expected position shape [2, T], got {pos_signal.shape}")
    if target_hz <= 0:
        raise ValueError(f"Invalid target Hz: {target_hz}")

    ms_per_sample = 1000.0 / float(target_hz)
    vel = np.abs(np.diff(pos_signal, axis=1)) / ms_per_sample
    vel = np.pad(vel, ((0, 0), (0, 1)), mode="constant", constant_values=0.0)
    return vel.astype(np.float32)


def preprocess_window(
    window_df: pd.DataFrame,
    *,
    subject: str,
    source_file: str,
    window_id: Any,
    target_hz: int,
    window_seconds: float,
    max_invalid_frames: int,
    min_valid_fraction: float,
    min_valid_frames: int,
) -> tuple[WindowProcessed | None, str]:
    """Preprocess one window; return processed payload or drop reason."""
    window_uid = f"{subject}|{source_file}|{window_id_to_str(window_id)}"
    if len(window_df) == 0:
        return None, "empty_window"

    invalid_mask = build_invalid_mask_xy(window_df)
    orig_len = int(len(window_df))
    valid_count = int((~invalid_mask).sum())
    valid_fraction = float(valid_count / orig_len)
    if valid_fraction < min_valid_fraction:
        return None, "low_valid_fraction"
    if valid_count < min_valid_frames:
        return None, "too_few_valid_frames"

    interpolated = interpolate_xy(
        window_df.loc[:, ["GazePointX", "GazePointY"]],
        invalid_mask=invalid_mask,
        max_invalid_frames=max_invalid_frames,
    )
    if interpolated.isna().any(axis=None):
        return None, "nan_after_interpolation"

    xy = interpolated.to_numpy(dtype=np.float32).T
    target_len = int(round(target_hz * window_seconds))
    timestamps = window_df["TimeStamp"].to_numpy(dtype=np.float64)
    pos_signal = resample_xy_to_fixed_len(
        xy=xy,
        timestamps=timestamps,
        target_len=target_len,
        window_seconds=window_seconds,
    )
    vel_signal = build_velocity_from_position(pos_signal, target_hz=target_hz)

    start_t = float(window_df["TimeStamp"].iloc[0]) if "TimeStamp" in window_df.columns else np.nan
    end_t = float(window_df["TimeStamp"].iloc[-1]) if "TimeStamp" in window_df.columns else np.nan
    metadata = {
        "window_uid": window_uid,
        "subject": subject,
        "source_file": source_file,
        "window_id": window_id,
        "window_id_str": window_id_to_str(window_id),
        "orig_len": orig_len,
        "valid_count": valid_count,
        "valid_fraction": valid_fraction,
        "start_t": start_t,
        "end_t": end_t,
        "target_hz": int(target_hz),
        "window_seconds": float(window_seconds),
        "target_len": int(target_len),
    }
    for col in EXTRA_METADATA_COLUMNS:
        metadata[col] = scalar_from_window(window_df, col, window_uid=window_uid)

    processed = WindowProcessed(metadata=metadata, pos_signal=pos_signal, vel_signal=vel_signal)
    return processed, ""


def encode_batch(network: torch.nn.Module, batch_np: np.ndarray, device: torch.device) -> np.ndarray:
    """Encode a batch of [N, 2, T] signals and return latent vectors."""
    if batch_np.ndim != 3 or batch_np.shape[1] != 2:
        raise ValueError(f"Expected batch shape [N, 2, T], got {batch_np.shape}")
    with torch.no_grad():
        batch = torch.tensor(batch_np, dtype=torch.float32, device=device)
        encoded = network.encode(batch)[0]
        return encoded.detach().cpu().numpy().astype(np.float32)


def _append_embeddings(
    records: list[dict[str, Any]],
    batch_meta: list[dict[str, Any]],
    pos_embeddings: np.ndarray,
    vel_embeddings: np.ndarray,
    include_z_concat: bool,
) -> None:
    """Append embedding values and metadata into flat record dictionaries."""
    if len(batch_meta) != len(pos_embeddings) or len(batch_meta) != len(vel_embeddings):
        raise ValueError("Batch metadata and embedding arrays are misaligned.")

    pos_dim = pos_embeddings.shape[1]
    vel_dim = vel_embeddings.shape[1]
    for idx, meta in enumerate(batch_meta):
        row = dict(meta)
        for d in range(pos_dim):
            row[f"z_pos_{d:03d}"] = float(pos_embeddings[idx, d])
        for d in range(vel_dim):
            row[f"z_vel_{d:03d}"] = float(vel_embeddings[idx, d])
        if include_z_concat:
            concat = np.concatenate([pos_embeddings[idx], vel_embeddings[idx]])
            for d, value in enumerate(concat):
                row[f"z_concat_{d:03d}"] = float(value)
        records.append(row)


def iter_grouped_windows(
    df: pd.DataFrame, max_windows_per_file: int = 0
) -> Iterable[tuple[Any, pd.DataFrame]]:
    """Iterate grouped windows deterministically with optional cap per file."""
    grouped = df.groupby("window_id", sort=False)
    count = 0
    for window_id, window_df in grouped:
        if max_windows_per_file > 0 and count >= max_windows_per_file:
            break
        count += 1
        yield window_id, window_df


def build_gazemae_embeddings(config: BuildConfig) -> tuple[Path, Path, BuildStats]:
    """Run full TrustMe preprocessing + GazeMAE embedding generation pipeline."""
    config.out_dir.mkdir(parents=True, exist_ok=True)
    stats = BuildStats()

    repo_root = Path(__file__).resolve().parents[1]
    ensure_gazemae_import_path(repo_root=repo_root)
    device = resolve_device(config.device)
    LOGGER.info("Using device: %s", device)

    network_pos = load_pretrained_network(config.model_pos, device=device)
    network_vel = load_pretrained_network(config.model_vel, device=device)
    LOGGER.info(
        "Loaded models. pos_latent=%d vel_latent=%d",
        int(network_pos.latent_size * (2 if network_pos.hierarchical else 1)),
        int(network_vel.latent_size * (2 if network_vel.hierarchical else 1)),
    )

    subject_dirs = discover_subject_dirs(config.input_root, config.subjects)
    stats.subjects_processed = len(subject_dirs)
    LOGGER.info("Found %d subjects to process.", len(subject_dirs))

    records: list[dict[str, Any]] = []
    batch_pos: list[np.ndarray] = []
    batch_vel: list[np.ndarray] = []
    batch_meta: list[dict[str, Any]] = []

    def flush_batch() -> None:
        if not batch_meta:
            return
        pos_np = np.stack(batch_pos, axis=0).astype(np.float32)
        vel_np = np.stack(batch_vel, axis=0).astype(np.float32)
        pos_embeddings = encode_batch(network_pos, pos_np, device=device)
        vel_embeddings = encode_batch(network_vel, vel_np, device=device)
        _append_embeddings(
            records=records,
            batch_meta=batch_meta,
            pos_embeddings=pos_embeddings,
            vel_embeddings=vel_embeddings,
            include_z_concat=config.include_z_concat,
        )
        batch_pos.clear()
        batch_vel.clear()
        batch_meta.clear()

    for subject_dir in subject_dirs:
        subject = subject_dir.name
        parquet_files = discover_parquet_files(
            subject_dir=subject_dir, max_files_per_subject=config.max_files_per_subject
        )
        LOGGER.info("Subject %s: %d parquet files", subject, len(parquet_files))

        for parquet_path in parquet_files:
            available_cols = set(get_parquet_columns(parquet_path))
            missing_required = sorted(set(REQUIRED_COLUMNS) - available_cols)
            if missing_required:
                raise ValueError(f"Missing required columns {missing_required} in {parquet_path}")

            read_cols = list(REQUIRED_COLUMNS)
            read_cols.extend([c for c in EXTRA_METADATA_COLUMNS if c in available_cols])
            df = pd.read_parquet(parquet_path, columns=read_cols)
            stats.parquet_files_processed += 1
            if df.empty:
                LOGGER.info("Skipping empty parquet file: %s", parquet_path)
                continue

            LOGGER.info(
                "Loaded %s rows=%d windows=%d",
                parquet_path.name,
                len(df),
                df["window_id"].nunique(dropna=False),
            )

            for window_id, window_df in iter_grouped_windows(
                df, max_windows_per_file=config.max_windows_per_file
            ):
                if pd.isna(window_id):
                    raise ValueError(f"Found NaN window_id in {parquet_path}")

                stats.total_candidates += 1
                processed, drop_reason = preprocess_window(
                    window_df=window_df,
                    subject=subject,
                    source_file=parquet_path.name,
                    window_id=window_id,
                    target_hz=config.target_hz,
                    window_seconds=config.window_seconds,
                    max_invalid_frames=config.max_invalid_frames,
                    min_valid_fraction=config.min_valid_fraction,
                    min_valid_frames=config.min_valid_frames,
                )
                if processed is None:
                    stats.add_drop(drop_reason)
                    continue

                batch_pos.append(processed.pos_signal)
                batch_vel.append(processed.vel_signal)
                batch_meta.append(processed.metadata)
                stats.total_kept += 1
                if len(batch_meta) >= config.batch_size:
                    flush_batch()

    flush_batch()

    out_embeddings_path = config.out_dir / "trustme_gazemae_embeddings.parquet"
    out_summary_path = config.out_dir / "trustme_gazemae_run_summary.json"

    if records:
        output_df = pd.DataFrame.from_records(records)
        output_df = output_df.sort_values("window_uid").reset_index(drop=True)
    else:
        output_df = pd.DataFrame(
            columns=[
                "window_uid",
                "subject",
                "source_file",
                "window_id",
                "window_id_str",
                "orig_len",
                "valid_count",
                "valid_fraction",
                "start_t",
                "end_t",
                "target_hz",
                "window_seconds",
                "target_len",
            ]
        )
    output_df.to_parquet(out_embeddings_path, index=False)

    summary = {
        "pipeline": "trustme_gazemae_embeddings",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            **asdict(config),
            "input_root": str(config.input_root),
            "out_dir": str(config.out_dir),
            "model_pos": str(config.model_pos),
            "model_vel": str(config.model_vel),
        },
        "models": {
            "model_pos": str(config.model_pos),
            "model_vel": str(config.model_vel),
        },
        "stats": asdict(stats),
        "outputs": {
            "embeddings_parquet": str(out_embeddings_path),
            "summary_json": str(out_summary_path),
            "rows_written": int(len(output_df)),
        },
    }
    out_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    LOGGER.info("Done. candidates=%d kept=%d dropped=%d", stats.total_candidates, stats.total_kept, stats.total_dropped)
    LOGGER.info("Embeddings parquet: %s", out_embeddings_path)
    LOGGER.info("Run summary: %s", out_summary_path)
    return out_embeddings_path, out_summary_path, stats


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for TrustMe GazeMAE embedding pipeline."""
    parser = argparse.ArgumentParser(
        description="Build TrustMe window-level GazeMAE embeddings (position + velocity)."
    )
    parser.add_argument("--input-root", type=Path, default=Path("data/TrustMe"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/gazemae_outputs"))
    parser.add_argument("--model-pos", type=Path, default=Path("models/pos-i3738"))
    parser.add_argument("--model-vel", type=Path, default=Path("models/vel-i8528"))
    parser.add_argument("--target-hz", type=int, default=500)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--max-invalid-frames", type=int, default=60)
    parser.add_argument("--min-valid-fraction", type=float, default=0.3)
    parser.add_argument("--min-valid-frames", type=int, default=32)
    parser.add_argument(
        "--subjects",
        type=str,
        default="",
        help="Optional comma-separated subject ids (e.g., s_004_pk,s_005_ak).",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Reserved for future parallel parquet loading. Kept for interface stability.",
    )
    parser.add_argument("--include-z-concat", action="store_true", default=False)
    parser.add_argument(
        "--max-files-per-subject",
        type=int,
        default=0,
        help="Optional debug/smoke cap on parquet files per subject (0 = all).",
    )
    parser.add_argument(
        "--max-windows-per-file",
        type=int,
        default=0,
        help="Optional debug/smoke cap on windows per parquet file (0 = all).",
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    config = BuildConfig(
        input_root=args.input_root,
        out_dir=args.out_dir,
        model_pos=args.model_pos,
        model_vel=args.model_vel,
        target_hz=args.target_hz,
        window_seconds=args.window_seconds,
        max_invalid_frames=args.max_invalid_frames,
        min_valid_fraction=args.min_valid_fraction,
        min_valid_frames=args.min_valid_frames,
        subjects=parse_subjects(args.subjects),
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        include_z_concat=args.include_z_concat,
        max_files_per_subject=args.max_files_per_subject,
        max_windows_per_file=args.max_windows_per_file,
        verbose=args.verbose,
    )
    build_gazemae_embeddings(config=config)


if __name__ == "__main__":
    main()
