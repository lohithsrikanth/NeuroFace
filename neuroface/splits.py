from typing import Tuple, Dict
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import NeuroFaceConfig

def make_subject_splits(
        df: pd.DataFrame,
        config: NeuroFaceConfig,
        random_state: int = 30
) -> pd.DataFrame:
    """
    Create subject-wise 80/10/10 splits, stratified by group.
    Returns the original df with a new column 'split' in {'train', 'val', 'test'}.
    """
    # One row per subject
    subj_df = df[["subject_id", "group"]].drop_duplicates()

    # Encode stratification labels
    strat_labels = subj_df["group"]

    # First split: train vs temp (val + test)
    train_subj, temp_subj = train_test_split(
        subj_df["subject_id"],
        test_size=(config.val_frac + config.test_frac),
        stratify=strat_labels,
        random_state=random_state
    )

    # Build temp df for second split
    temp_df = subj_df[subj_df["subject_id"].isin(temp_subj)]
    temp_strat = temp_df["group"]

    # relative val vs test sizes within the temp set
    val_ratio = config.val_frac / (config.val_frac + config.test_frac)

    val_subj, test_subj = train_test_split(
        temp_df["subject_id"],
        test_size=(1.0 - val_ratio),
        stratify=temp_strat,
        random_state=random_state
    )

    # Map subject -> split
    split_map: Dict[str, str] = {}
    for s in train_subj:
        split_map[s] = "train"
    for s in val_subj:
        split_map[s] = "val"
    for s in test_subj:
        split_map[s] = "test"

    df = df.copy()
    df["split"] = df["subject_id"].map(split_map)

    # Sanity check
    assert not df["split"].isna().any(), "Some subjects did not get a split assigned"

    return df


