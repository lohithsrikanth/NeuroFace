from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np


def load_bbox_file(path: Path) -> pd.DataFrame:
    """
    Load a *_color_gt.txt BBox file.
    Columns: Frame, x1, y1, x2, y2
    """
    df = pd.read_csv(path, header=0)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_landmarks_file(path: Path) -> pd.DataFrame:
    """
    Load a *_color.txt landmarks file.
    Columns: Frame, x1, y1, .... , x68, y68 (137 columns total)
    """
    df = pd.read_csv(path, header=0)
    df.columns = [c.strip() for c in df.columns]
    return df


def get_bbox_for_frame(
    df_bbox: pd.DataFrame, frame_idx: int
) -> Tuple[float, float, float, float]:
    row = df_bbox.loc[df_bbox["Frame"] == frame_idx]
    if row.empty:
        raise KeyError(f"Frame {frame_idx} not found in bbox file.")
    row = row.iloc[0]
    return float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])


def get_landmarks_for_frame(df_lm: pd.DataFrame, frame_idx: int) -> np.ndarray:
    """
    Return landmarks as array of shape (68, 2) for given frame
    """
    row = df_lm.loc[df_lm["Frame"] == frame_idx]
    if row.empty:
        raise KeyError(f"Frame {frame_idx} not found in landmarks file.")
    row = row.iloc[0]

    coords = row.iloc[1:].to_numpy(dtype=float)
    assert coords.shape[0] == 136, "Expected 68(x, y) pairs"
    landmarks = coords.reshape(68, 2)
    return landmarks
