from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class NeuroFaceConfig:
    # Root folder containing ALS / HC / PS directories
    data_root: Path

    group_dirs: Dict[str, str] = None   # label_name -> folder_name
    label_to_index: Dict[str, int] = None   # For ML models

    # Image preprocessing
    image_size: int = 224

    # Split fractions
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1

    def __post_init__(self):
        if self.group_dirs is None:
            self.group_dirs = {
                "ALS": "ALS",
                "HC": "HC",
                "PS": "PS"
            }

        if self.label_to_index is None:
            self.label_to_index = {"ALS": 0, "PS": 1, "HC": 2}

    @property
    def groups(self) -> List[str]:
        return list(self.group_dirs.keys())
    
def default_config(data_root: str) -> NeuroFaceConfig:
    return NeuroFaceConfig(data_root=Path(data_root))