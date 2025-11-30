import sys
from pathlib import Path

# Get the absolute path to the project root (.. from scripts/)
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


from neuroface.config import default_config
from neuroface.metadata import build_metadata

if __name__ == "__main__":
    DATA_ROOT = project_root / "Data"
    OUTPUT_CSV = (
        project_root / "metadata" / "metadata_frames.csv"
    )

    config = default_config(DATA_ROOT)
    df = build_metadata(config, save_csv=OUTPUT_CSV)
    print(
        f"Metadata saved to {OUTPUT_CSV}, n={len(df)} frames, subjects={df['subject_id'].nunique()}"
    )
