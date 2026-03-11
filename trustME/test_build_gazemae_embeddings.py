"""Unit and smoke tests for TrustMe GazeMAE embedding pipeline.

Run:
CONDA_NO_PLUGINS=true conda run -n trust-me python trustME/test_build_gazemae_embeddings.py
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from trustME.build_gazemae_embeddings import (
        BuildConfig,
        build_gazemae_embeddings,
        build_invalid_mask_xy,
        build_velocity_from_position,
        interpolate_xy,
        preprocess_window,
        resample_xy_to_fixed_len,
    )
except ModuleNotFoundError:
    from build_gazemae_embeddings import (  # type: ignore
        BuildConfig,
        build_gazemae_embeddings,
        build_invalid_mask_xy,
        build_velocity_from_position,
        interpolate_xy,
        preprocess_window,
        resample_xy_to_fixed_len,
    )


class TestPreprocessing(unittest.TestCase):
    """Synthetic tests for preprocessing and signal construction."""

    def test_invalid_mask_detects_sentinel_and_nan(self) -> None:
        df = pd.DataFrame(
            {
                "GazePointX": [100.0, -1.0, 120.0, np.nan],
                "GazePointY": [200.0, 230.0, -1.0, 240.0],
            }
        )
        mask = build_invalid_mask_xy(df)
        self.assertEqual(mask.tolist(), [False, True, True, True])

    def test_interpolate_xy_fills_short_gaps(self) -> None:
        signals = pd.DataFrame(
            {
                "GazePointX": [100.0, np.nan, 120.0],
                "GazePointY": [200.0, np.nan, 240.0],
            }
        )
        invalid = np.array([False, True, False], dtype=bool)
        out = interpolate_xy(signals=signals, invalid_mask=invalid, max_invalid_frames=60)
        self.assertFalse(out.isna().any(axis=None))
        self.assertAlmostEqual(float(out.loc[1, "GazePointX"]), 110.0, places=5)
        self.assertAlmostEqual(float(out.loc[1, "GazePointY"]), 220.0, places=5)

    def test_resample_outputs_expected_length(self) -> None:
        xy = np.array([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]], dtype=np.float32)
        timestamps = np.array([0.0, 500.0, 1000.0], dtype=np.float64)
        out = resample_xy_to_fixed_len(
            xy=xy, timestamps=timestamps, target_len=1500, window_seconds=3.0
        )
        self.assertEqual(out.shape, (2, 1500))

    def test_velocity_padding_and_units(self) -> None:
        pos = np.array([[0.0, 2.0, 4.0], [10.0, 12.0, 18.0]], dtype=np.float32)
        vel = build_velocity_from_position(pos_signal=pos, target_hz=500)
        self.assertEqual(vel.shape, (2, 3))
        # ms_per_sample at 500 Hz = 2 ms
        np.testing.assert_allclose(vel[:, 0], np.array([1.0, 1.0]), rtol=1e-6)
        np.testing.assert_allclose(vel[:, 1], np.array([1.0, 3.0]), rtol=1e-6)
        np.testing.assert_allclose(vel[:, 2], np.array([0.0, 0.0]), rtol=1e-6)

    def test_preprocess_window_qc_drop_and_keep(self) -> None:
        keep_df = pd.DataFrame(
            {
                "window_id": [1, 1, 1, 1, 1],
                "TimeStamp": [0.0, 16.0, 33.0, 50.0, 66.0],
                "GazePointX": [100.0, -1.0, 102.0, 103.0, 104.0],
                "GazePointY": [200.0, -1.0, 202.0, 203.0, 204.0],
            }
        )
        processed, reason = preprocess_window(
            keep_df,
            subject="s_001",
            source_file="x.parquet",
            window_id=1,
            target_hz=500,
            window_seconds=3.0,
            max_invalid_frames=60,
            min_valid_fraction=0.3,
            min_valid_frames=2,
        )
        self.assertEqual(reason, "")
        self.assertIsNotNone(processed)
        assert processed is not None
        self.assertEqual(processed.pos_signal.shape, (2, 1500))
        self.assertEqual(processed.vel_signal.shape, (2, 1500))

        drop_df = pd.DataFrame(
            {
                "window_id": [1, 1, 1, 1],
                "TimeStamp": [0.0, 16.0, 33.0, 50.0],
                "GazePointX": [-1.0, -1.0, -1.0, 100.0],
                "GazePointY": [-1.0, -1.0, -1.0, 200.0],
            }
        )
        dropped, drop_reason = preprocess_window(
            drop_df,
            subject="s_001",
            source_file="x.parquet",
            window_id=2,
            target_hz=500,
            window_seconds=3.0,
            max_invalid_frames=60,
            min_valid_fraction=0.3,
            min_valid_frames=2,
        )
        self.assertIsNone(dropped)
        self.assertEqual(drop_reason, "low_valid_fraction")


class TestSmokeRealData(unittest.TestCase):
    """Smoke test on one smallest parquet per subject using real models and data."""

    @unittest.skipUnless(Path("data/TrustMe").exists(), "TrustMe data folder not found")
    @unittest.skipUnless(Path("models/pos-i3738").exists(), "Position model missing")
    @unittest.skipUnless(Path("models/vel-i8528").exists(), "Velocity model missing")
    def test_smoke_pipeline_one_file_per_subject(self) -> None:
        input_root = Path("data/TrustMe")
        subject_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])
        self.assertGreaterEqual(len(subject_dirs), 1)

        with tempfile.TemporaryDirectory(prefix="trustme_smoke_") as tmpdir:
            tmp_root = Path(tmpdir)
            tmp_input_root = tmp_root / "input"
            tmp_output_root = tmp_root / "output"
            tmp_input_root.mkdir(parents=True, exist_ok=True)

            for subject_dir in subject_dirs:
                source_tobii = subject_dir / "tobii"
                files = sorted(source_tobii.glob("*.parquet"))
                self.assertGreater(len(files), 0)
                # Largest files are more likely to contain at least a few
                # valid windows for embedding smoke checks.
                chosen_file = max(files, key=lambda p: p.stat().st_size)

                target_tobii = tmp_input_root / subject_dir.name / "tobii"
                target_tobii.mkdir(parents=True, exist_ok=True)
                link_path = target_tobii / chosen_file.name
                os.symlink(chosen_file.resolve(), link_path)

            config = BuildConfig(
                input_root=tmp_input_root,
                out_dir=tmp_output_root,
                model_pos=Path("models/pos-i3738"),
                model_vel=Path("models/vel-i8528"),
                target_hz=500,
                window_seconds=3.0,
                max_invalid_frames=60,
                min_valid_fraction=0.3,
                min_valid_frames=32,
                subjects=(),
                batch_size=32,
                device="auto",
                num_workers=0,
                include_z_concat=False,
                max_files_per_subject=0,
                max_windows_per_file=256,
                verbose=False,
            )

            embeddings_path, summary_path, stats = build_gazemae_embeddings(config)
            self.assertTrue(embeddings_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertGreaterEqual(stats.total_candidates, stats.total_kept)

            df = pd.read_parquet(embeddings_path)
            summary = json.loads(summary_path.read_text())

            self.assertEqual(int(summary["outputs"]["rows_written"]), int(stats.total_kept))
            self.assertEqual(int(summary["stats"]["total_kept"]), int(stats.total_kept))
            self.assertEqual(len(df), int(stats.total_kept))
            if len(df) == 0:
                # Keep strict-QC run checks above, then do a tiny relaxed run to
                # exercise embedding columns and NaN checks.
                relaxed_out = tmp_root / "output_relaxed"
                relaxed_config = BuildConfig(
                    input_root=tmp_input_root,
                    out_dir=relaxed_out,
                    model_pos=Path("models/pos-i3738"),
                    model_vel=Path("models/vel-i8528"),
                    target_hz=500,
                    window_seconds=3.0,
                    max_invalid_frames=60,
                    min_valid_fraction=0.0,
                    min_valid_frames=1,
                    subjects=(),
                    batch_size=16,
                    device="auto",
                    num_workers=0,
                    include_z_concat=False,
                    max_files_per_subject=0,
                    max_windows_per_file=32,
                    verbose=False,
                )
                relaxed_embeddings_path, _, relaxed_stats = build_gazemae_embeddings(relaxed_config)
                relaxed_df = pd.read_parquet(relaxed_embeddings_path)
                self.assertGreater(len(relaxed_df), 0)
                self.assertEqual(len(relaxed_df), int(relaxed_stats.total_kept))
                df = relaxed_df

            pos_cols = sorted([c for c in df.columns if c.startswith("z_pos_")])
            vel_cols = sorted([c for c in df.columns if c.startswith("z_vel_")])
            self.assertEqual(len(pos_cols), 128)
            self.assertEqual(len(vel_cols), 128)
            self.assertFalse(df[pos_cols].isna().any(axis=None))
            self.assertFalse(df[vel_cols].isna().any(axis=None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
