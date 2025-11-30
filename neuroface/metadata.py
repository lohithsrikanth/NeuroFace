from pathlib import Path
from typing import List, Dict, Optional
import re
import pandas as pd

from .config import NeuroFaceConfig

FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_frame_filename(stem: str) -> Dict[str, str]:
    """
    Parses the filename stem to extract subject, video, task, and frame information.

    Handles format: SubjectID_RepID_TaskName_Suffix_FrameIdx
    Example: 'A022_02_BBP_NORMAL_color.avi_184' (from .jpg)

    Returns:
        subject_id: 'A022'
        video_id:   'A022_02_BBP_NORMAL_color.avi'
        task:       'BBP_NORMAL'
        frame_idx:  184
    """
    # Split off the frame index (everything after the last underscore)
    video_id, frame_part = stem.rsplit("_", 1)

    # Extract strictly defined IDs
    frame_idx = int(frame_part)
    subject_id = video_id[:4]  # First 4 chars are always Subject ID

    # Extract the Task
    parts = video_id.split("_")

    # We grab everything between index 2 (after Rep ID) and the last element (Suffix)
    task = "_".join(parts[2:-1])

    filename = video_id.split(".")[0]

    return {
        "subject_id": subject_id,
        "video_id": video_id,
        "task": task,
        "frame_idx": frame_idx,
        "filename": filename,
    }


def build_metadata(
    config: NeuroFaceConfig, save_csv: Optional[Path] = None
) -> pd.DataFrame:
    """
    Scan the directory structure and build a metadata dataframe
    Columns:
        - group (ALS/PS/HC)
        - label_idx (0/1/2)
        - subject_id
        - task
        - video_id
        - frame_idx
        - frame_path
        - landmarks_path
        - bbox_path
    """
    records: List[Dict] = []

    base_path = config.data_root.parent

    for group, folder_name in config.group_dirs.items():
        group_dir = config.data_root / folder_name
        frames_dir = group_dir / "Frames"
        landmarks_dir = group_dir / "Landmarks_gt"
        bbox_dir = group_dir / "Bbox_gt"

        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames dir not found: {frames_dir}")

        for img_path in frames_dir.rglob("*"):
            if (
                not img_path.is_file()
                or img_path.suffix.lower() not in FRAME_EXTENSIONS
            ):
                continue

            stem = img_path.stem
            parsed = parse_frame_filename(stem)
            subject_id = parsed["subject_id"]
            task = parsed["task"]
            video_id = parsed["video_id"]
            frame_idx = parsed["frame_idx"]
            filename = parsed["filename"]

            # Landmarks & bbox txt filenames
            lm_file = landmarks_dir / f"{filename}.txt"
            bbox_file = bbox_dir / f"{filename}_gt.txt"

            if not lm_file.exists():
                raise FileNotFoundError(
                    f"Landmarks file not found for {video_id}: {lm_file}"
                )
            if not bbox_file.exists():
                raise FileNotFoundError(
                    f"BBox file not found for {video_id}: {bbox_file}"
                )

            rel_frame_path = img_path.relative_to(base_path)
            rel_lm_path = lm_file.relative_to(base_path)
            rel_bbox_path = bbox_file.relative_to(base_path)

            records.append(
                dict(
                    group=group,
                    label_idx=config.label_to_index[group],
                    subject_id=subject_id,
                    video_id=video_id,
                    task=task,
                    frame_idx=frame_idx,
                    frame_path=str(rel_frame_path),
                    landmarks_path=str(rel_lm_path),
                    bbox_path=str(rel_bbox_path),
                )
            )

    df = pd.DataFrame.from_records(records)

    df = df.sort_values(
        by=["group", "subject_id", "video_id", "frame_idx"]
    ).reset_index(drop=True)

    if save_csv is not None:
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)

    return df
