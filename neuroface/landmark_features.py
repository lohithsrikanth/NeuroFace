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

