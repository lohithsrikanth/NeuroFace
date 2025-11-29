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