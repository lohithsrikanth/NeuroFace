import sys
from pathlib import Path

# Get the absolute path to the project root (.. from scripts/)
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import pandas as pd
from neuroface.splits import make_subject_splits
from neuroface.config import default_config
from neuroface.metadata import build_metadata

if __name__ == '__main__':
    DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"
    METADATA_CSV = (
        Path(__file__).resolve().parent.parent / "metadata" / "metadata_frames.csv"
    )

    METADATA = project_root / "metadata"

    config = default_config(DATA_ROOT)

    if not METADATA.exists():
        
        df = build_metadata(config, save_csv=METADATA_CSV)
        print(
            f"Metadata saved to {METADATA_CSV}, n={len(df)} frames, subjects={df['subject_id'].nunique()}"
        )
    else:
        df = pd.read_csv(METADATA_CSV)

    df = make_subject_splits(df, config)
    df.to_csv(METADATA / "metadata_frames_with_splits.csv", index=False)



