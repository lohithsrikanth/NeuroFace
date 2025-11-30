# scripts/build_landmark_features.py
import sys
from pathlib import Path
import pandas as pd

# Get the absolute path to the project root (.. from scripts/)
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from neuroface.config import default_config
from neuroface.landmark_features import (
    LandmarkFeatureExtractor,
    LandmarkFeatureExtractorConfig,
    build_subject_task_feature_table,
)

if __name__ == "__main__":
    DATA_ROOT = project_root / "Data"
    METADATA = project_root / "metadata"
    METADATA_CSV = Path( METADATA / "metadata_frames_with_splits.csv")

    FRAME_FEATURES_CSV = Path(METADATA / "landmark_frame_features.csv")
    SUBJECT_TASK_FEATURES_CSV = Path(METADATA / "landmark_subject_task_features.csv")

    config = default_config(DATA_ROOT)

    df = pd.read_csv(METADATA_CSV)

    df["frame_path"] = df["frame_path"].apply(lambda p: str(project_root / p))
    df["bbox_path"] = df["bbox_path"].apply(lambda p: str(project_root / p))
    df["landmarks_path"] = df["landmarks_path"].apply(lambda p: str(project_root / p))

    extractor = LandmarkFeatureExtractor(
        config=config,
        lm_config=LandmarkFeatureExtractorConfig(scale_mode="bbox_diag"),
    )

    print("Extracting frame-level landmark features...")
    frame_feat_df = extractor.build_frame_feature_table(df)
    FRAME_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame_feat_df.to_csv(FRAME_FEATURES_CSV, index=False)
    print(f"Saved frame-level features to {FRAME_FEATURES_CSV}")

    print("Aggregating to subject+task-level features...")
    subj_task_df = build_subject_task_feature_table(frame_feat_df, agg="mean")
    subj_task_df.to_csv(SUBJECT_TASK_FEATURES_CSV, index=False)
    print(f"Saved task-aware subject features to {SUBJECT_TASK_FEATURES_CSV}")
