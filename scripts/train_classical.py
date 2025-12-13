import sys
from pathlib import Path
import pandas as pd
from pprint import pprint

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from neuroface.config import default_config
from neuroface.classical_models import (
    ClassicalTaskAwareTrainer,
    build_dataset_splits_from_subject_task_features,
)

if __name__ == "__main__":
    DATA_ROOT = project_root / "Data"
    METADATA = project_root / "metadata"
    SUBJECT_TASK_FEATURES_CSV = Path(METADATA / "landmark_subject_task_features.csv")

    config = default_config(DATA_ROOT)

    subj_task_df = pd.read_csv(SUBJECT_TASK_FEATURES_CSV)

    data_splits = build_dataset_splits_from_subject_task_features(
        subj_task_features_df=subj_task_df,
        config=config,
    )

    trainer = ClassicalTaskAwareTrainer(config)

    print("Training Logistic Regression (task-aware)...")
    logres_result = trainer.train_logistic_regression(data_splits)
    print("Task-level metrics (LogReg):", logres_result.task_level_metrics)
    print("Subject-level metrics (LogReg):", logres_result.subject_level_metrics)

    print("\nTraining SVM (task-aware)...")
    svm_result = trainer.train_svm(data_splits)
    print("Task-level metrics (SVM):", svm_result.task_level_metrics)
    print("Subject-level metrics (SVM):", svm_result.subject_level_metrics)

    print("\nTraining Random Forest (task-aware)...")
    rf_result = trainer.train_random_forest(data_splits)
    print("Task-level metrics (RF):", rf_result.task_level_metrics)
    print("Subject-level metrics (RF):", rf_result.subject_level_metrics)
