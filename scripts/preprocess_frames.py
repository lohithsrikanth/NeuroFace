import sys
from pathlib import Path
import pandas as pd

# Get the absolute path to the project root (.. from scripts/)
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from neuroface.config import default_config
from neuroface.metadata import build_metadata
from neuroface.splits import make_subject_splits
from neuroface.image_preproc import offline_preprocess_all_frames

if __name__ == "__main__":
    DATA_ROOT = project_root / "Data"
    METADATA = project_root / "metadata"
    METADATA_CSV = Path(METADATA / "metadata_frames_with_splits.csv")
    OUTPUT_ROOT = Path(project_root / "processed_frames")

    config = default_config(DATA_ROOT)

    if METADATA_CSV.exists():
        df = pd.read_csv(METADATA_CSV)
    else:
        df = build_metadata(config)
        df = make_subject_splits(df, config)
        METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(METADATA_CSV, index=False)

    print("Converting relative paths to absolute paths for processing...")
    df["frame_path"] = df["frame_path"].apply(lambda p: str(project_root / p))
    df["bbox_path"] = df["bbox_path"].apply(lambda p: str(project_root / p))

    offline_preprocess_all_frames(df, config, OUTPUT_ROOT, overwrite=False)
    print("Offline preprocessing complete")