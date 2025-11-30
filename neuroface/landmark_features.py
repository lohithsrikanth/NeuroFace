from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Literal

import numpy as np
import pandas as pd

from .io_bbox_landmarks import (
    load_landmarks_file,
    load_bbox_file,
    get_bbox_for_frame,
)
from .config import NeuroFaceConfig

# These indices follow the common 68-point facial landmark convention
NOSE_TIP_IDX = 30 - 1          # 30 in 1-based → 29 in 0-based
JAW_CHIN_IDX = 9 - 1           # chin
UPPER_LIP_IDX = 52 - 1         # approx upper lip center
LOWER_LIP_IDX = 58 - 1         # approx lower lip center
MOUTH_LEFT_CORNER_IDX = 49 - 1
MOUTH_RIGHT_CORNER_IDX = 55 - 1

LEFT_EYE_IDXS = list(range(37 - 1, 42 - 1 + 1))   # 37–42
RIGHT_EYE_IDXS = list(range(43 - 1, 48 - 1 + 1))  # 43–48

# ----- Core normalization utilities --------
def normalize_landmarks(
        landmarks: np.ndarray,
        bbox: Optional[np.ndarray] = None,
        scale_mode: Literal["bbox_diag", "interpupil"] = "bbox_diag",
) -> np.ndarray:
    """
    landmarks: (68, 2) array in original image coordinates.
    bbox: optional [x1, y1, x2, y2] in same coordinate system.
    Steps:
      1) Translation: subtract nose tip so face is centered at origin.
      2) Scale: divide by either bbox diagonal or interpupil distance.
    """
    assert landmarks.shape == (68, 2), "Expected shape (68, 2)"

    # Translation: center on nose tip
    nose = landmarks[NOSE_TIP_IDX]
    centered = landmarks - nose[None, :]

    if scale_mode == "bbox_diag":
        if bbox is None:
            raise ValueError("bbox muse be provided when scale_mode='bbox_diag'")
        x1, y1, x2, y2 = bbox
        diag = float(np.sqrt((x2 -x1) ** 2 + (y2 - y1) ** 2))
        scale = diag if diag > 1e-6 else 1.0
    elif scale_mode == "interpupil":
        # pupil centers = mean of eye region landmarks
        left_eye_center = landmarks[LEFT_EYE_IDXS].mean(axis=0)
        right_eye_center = landmarks[RIGHT_EYE_IDXS].mean(axis=0)
        dist = float(np.linalg.norm(right_eye_center - left_eye_center))
        scale = dist if dist > 1e-6 else 1.0
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")
    
    normalized = centered / scale
    return normalized

def compute_static_shape_features(
        normalized_landmarks: np.ndarray
) -> np.ndarray:
    """
    Create static shape features from normalized landmarks.

    Features (per frame):
      - Flattened normalized landmarks (136 dims)
      - Mouth width (distance between corners)
      - Mouth opening (upper vs lower lip)
      - Jaw opening (chin vs nose tip)
    """
    assert normalized_landmarks.shape == (68, 2)
    feats: List[float] = []

    # 1) Flattened normalized landmarks
    flat = normalized_landmarks.flatten()  # (136,)
    feats.extend(flat.tolist())

    # 2) Key distances (all in normalized coordinate system)
    def dist(i: int, j: int) -> float:
        return float(np.linalg.norm(normalized_landmarks[i] - normalized_landmarks[j]))

    # Mouth width
    mouth_width = dist(MOUTH_LEFT_CORNER_IDX, MOUTH_RIGHT_CORNER_IDX)
    # Mouth opening (vertical distance between upper and lower lip)
    mouth_opening = dist(UPPER_LIP_IDX, LOWER_LIP_IDX)
    # Jaw opening (chin vs nose tip)
    jaw_opening = dist(JAW_CHIN_IDX, NOSE_TIP_IDX)

    feats.extend([mouth_width, mouth_opening, jaw_opening])

    return np.asarray(feats, dtype=np.float32)

# ------ Landmark feature extractor class --------
@dataclass
class LandmarkFeatureExtractorConfig:
    scale_mode: Literal["bbox_diag", "interpupil"] = "bbox_diag"

class LandmarkFeatureExtractor:
    """
    Extracts static shape features from landmarks synchronized with metadata.

    Expected metadata columns (frame-level):
        - task
        - frame_path
        - landmarks_path
        - bbox_path
        - frame_idx
        - subject_id
        - group
        - label_idx
        - split
    """
    def __init__(self, config: NeuroFaceConfig, lm_config: Optional[LandmarkFeatureExtractorConfig] = None):
        self.config = config
        self.lm_config = lm_config or LandmarkFeatureExtractorConfig()

    def _load_landmarks_for_video(self, lm_path: Path) -> pd.DataFrame:
        df = load_landmarks_file(lm_path)
        # Clean column names
        df.columns = [c.strip() for c in df.columns]
        return df
    
    def _load_bboxes_for_video(self, bbox_path: Path) -> pd.DataFrame:
        df = load_bbox_file(bbox_path)
        df.columns = [c.strip() for c in df.columns]
        return df
    
    def extract_frame_features_for_group(
        self,
        df_group: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        df_group: metadata subset for a single (group, subject, video_id, task) group

        Returns frame-level feature DataFrame:
          columns: subject_id, group, label_idx, split, frame_idx, video_id, feature_0..feature_(D-1)
        """
        if df_group.empty:
            return df_group

        lm_path = Path(df_group["landmarks_path"].iloc[0])
        bbox_path = Path(df_group["bbox_path"].iloc[0])

        df_lm = self._load_landmarks_for_video(lm_path)
        df_bbox = self._load_bboxes_for_video(bbox_path)

        records: List[Dict] = []

        for _, row in df_group.iterrows():
            frame_idx = int(row["frame_idx"])

            # Extract landmarks for this frame
            lm_row = df_lm.loc[df_lm["Frame"] == frame_idx]
            if lm_row.empty:
                continue
            lm_row = lm_row.iloc[0]
            coords = lm_row.iloc[1:].to_numpy(dtype=float)
            if coords.shape[0] != 136:
                raise ValueError(f"Expected 136 landmark coords, got {coords.shape[0]} in {lm_path}")
            landmarks = coords.reshape(68, 2)

            bbox = get_bbox_for_frame(df_bbox, frame_idx)  # (x1, y1, x2, y2)

            # Normalize + static features
            norm_landmarks = normalize_landmarks(
                landmarks,
                bbox=np.asarray(bbox),
                scale_mode=self.lm_config.scale_mode,
            )
            feat_vec = compute_static_shape_features(norm_landmarks)

            record = dict(
                subject_id=row["subject_id"],
                group=row["group"],
                label_idx=row["label_idx"],
                split=row["split"],
                video_id=row["video_id"],
                task=row["task"],
                frame_idx=frame_idx,
            )
            # Add feature_0..feature_(D-1)
            for i, val in enumerate(feat_vec):
                record[f"f_{i:03d}"] = float(val)

            records.append(record)

        return pd.DataFrame.from_records(records)
    
    def build_frame_feature_table(
        self,
        metadata_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Main entry point:
        Given the full metadata (one row per frame),
        return one row per frame with landmark-derived features
        """
        feature_dfs: List[pd.DataFrame] = []

        # Group by distinct (group, subject_id, video_id, task) for efficiency
        grouped = metadata_df.groupby(["group", "subject_id", "video_id", "task"])

        for _, df_group in grouped:
            fdf = self.extract_frame_features_for_group(df_group)
            if not fdf.empty:
                feature_dfs.append(fdf)

        if not feature_dfs:
            raise RuntimeError("No landmark features extracted; check metadata and file paths")
        
        df_features = pd.concat(feature_dfs, axis=0).reset_index(drop=True)
        return df_features
    
def build_subject_task_feature_table(
    frame_features_df: pd.DataFrame,
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate frame-level features to subject+task-level features.

    New behavior:
      - Aggregates frames by:
            subject_id, group, label_idx, split, task
      - Produces one row per (subject, task) pair.
      - Prevents mixing landmarks from different tasks.
    
    Returns DataFrame:
      columns:
        subject_id, group, label_idx, split, task, f_000..f_(D-1)
    """

    # Feature columns (f_000, f_001, ... )
    feature_cols = [c for c in frame_features_df.columns if c.startswith("f_")]

    # REQUIRED grouping columns, now includes task
    group_cols = ["subject_id", "group", "label_idx", "split", "task"]

    if agg == "mean":
        agg_df = (
            frame_features_df
            .groupby(group_cols)[feature_cols]
            .mean()
            .reset_index()
        )
    elif agg == "median":
        agg_df = (
            frame_features_df
            .groupby(group_cols)[feature_cols]
            .median()
            .reset_index()
        )
    else:
        raise ValueError(f"Unsupported aggregation: {agg}")

    return agg_df
