from pathlib import Path
from typing import Optional
import os

import numpy as np
from PIL import Image
import pandas as pd

from .io_bbox_landmarks import load_bbox_file, get_bbox_for_frame
from .config import NeuroFaceConfig

def crop_face_from_frame(img: Image.Image,
                         bbox: tuple,
                        ) -> Image.Image:
    """
    Crop the face region from a full frame using (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = bbox
    # Ensure ints and valid box
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, img.width)
    y2 = min(y2, img.height)
    return img.crop((x1, y1, x2, y2))

def preprocess_single_frame(
        frame_path: Path,
        bbox_path: Path,
        frame_idx: int,
        image_size: int,
) -> Image.Image:
    """
    Load a frame, crop with BBox, resize to (image_size, image_size)
    """
    img = Image.open(frame_path).convert("RGB")
    df_bbox = load_bbox_file(bbox_path)
    bbox = get_bbox_for_frame(df_bbox, frame_idx)
    face_img = crop_face_from_frame(img, bbox)
    face_img = face_img.resize((image_size, image_size), Image.BICUBIC)
    return face_img

def offline_preprocess_all_frames(
        df: pd.DataFrame,
        config: NeuroFaceConfig,
        output_root: Path,
        overwrite: bool = False
) -> None:
    """
    Offline preprocessing: crop and resize all frames and save to disk:
        output_root / split / group / subject / video_id / frame_idx.jpg
    """
    output_root.mkdir(parents=True, exist_ok=True)

    # Group by (split, group, subject, video_id) to avoid re-reading bbox files too often
    grouped = df.groupby(["split", "group", "subject_id", "video_id"])

    for (split, group, subject_id, video_id), sub_df in grouped:
        # Load bbox file once per video
        bbox_path = Path(sub_df["bbox_path"].iloc[0])
        df_bbox = load_bbox_file(bbox_path)
        
        out_dir = output_root / split / group / subject_id / video_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for _, row in sub_df.iterrows():
            frame_path = Path(row["frame_path"])
            frame_idx = int(row["frame_idx"])

            out_path = out_dir / f"{frame_idx:03d}.jpg"
            if out_path.exists() and not overwrite:
                continue

            img = Image.open(frame_path).convert("RGB")
            bbox = get_bbox_for_frame(df_bbox, frame_idx)
            face_img = crop_face_from_frame(img, bbox)
            face_img = face_img.resize((config.image_size, config.image_size), Image.BICUBIC)
            face_img.save(out_path, quality=95)

