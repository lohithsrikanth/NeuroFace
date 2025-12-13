from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from .config import NeuroFaceConfig
from .aggregation import (
    compute_classification_metrics_core,
    aggregate_subject_probas,
)

@dataclass
class DatasetSplitsTask:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    # meta for test-set (needed for subject-level aggregation)
    test_meta: pd.DataFrame
    label_names: List[str]

@dataclass
class TrainingResultTask:
    model: Any
    task_level_metrics: Dict[str, Any]
    subject_level_metrics: Dict[str, Any]
    subject_predictions: pd.DataFrame

# ----- Build dataset splits from subject + task feature table ------

def build_dataset_splits_from_subject_task_features(
    subj_task_features_df: pd.DataFrame,
    config: NeuroFaceConfig,
) -> DatasetSplitsTask:
    """
    Input DataFrame is the output of build_subject_feature_table where
    aggregation keys now include 'task'. So each row is (subject_id, task).

    Expected columns:
      - subject_id
      - group
      - label_idx
      - split ∈ {train, val, test}
      - task
      - f_000 ...
    """
    feature_cols = [c for c in subj_task_features_df.columns if c.startswith("f_")]

    def subset(split: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        df_split = subj_task_features_df[subj_task_features_df["split"] == split].copy()
        X = df_split[feature_cols].to_numpy(dtype=np.float32)
        y = df_split["label_idx"].to_numpy(dtype=int)
        return X, y, df_split

    X_train, y_train, train_meta = subset("train")
    X_val, y_val, val_meta = subset("val")
    X_test, y_test, test_meta = subset("test")

    label_names = sorted(config.label_to_index.keys(), key=lambda k: config.label_to_index[k])

    return DatasetSplitsTask(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        test_meta=test_meta,
        label_names=label_names,
    )


# ------------------------ Trainer class (task-aware) ------------------------


class ClassicalTaskAwareTrainer:
    """
    Train & evaluate Logistic Regression, SVM, and Random Forest
    on subject+task-level clinical (landmark) features.

    Produces:
      - Task-level metrics (sample = subject-task)
      - Subject-level metrics (aggregated across tasks of a subject)
    """

    def __init__(self, config: NeuroFaceConfig):
        self.config = config

    # ---- Generic evaluation helper ----

    def _evaluate(
        self,
        model: Any,
        data: DatasetSplitsTask,
    ) -> TrainingResultTask:
        # Task-level predictions
        y_pred = model.predict(data.X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(data.X_test)
        else:
            y_proba = None

        # Task-level metrics
        task_level_metrics = compute_classification_metrics_core(
            y_true=data.y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            label_names=data.label_names,
        )

        # Subject-level aggregation (requires y_proba)
        if y_proba is not None:
            subject_ids = data.test_meta["subject_id"].to_numpy()
            subj_agg = aggregate_subject_probas(
                y_proba=y_proba,
                y_true=data.y_test,
                subject_ids=subject_ids,
                label_names=data.label_names,
            )
            subject_metrics = subj_agg["metrics"]
            subject_predictions = subj_agg["subject_df"]
        else:
            subject_metrics = {}
            subject_predictions = pd.DataFrame()

        return TrainingResultTask(
            model=model,
            task_level_metrics=task_level_metrics,
            subject_level_metrics=subject_metrics,
            subject_predictions=subject_predictions,
        )

    # ---- Model-specific training ----

    def train_logistic_regression(
        self,
        data: DatasetSplitsTask,
        C: float = 1.0,
        max_iter: int = 1000,
        penalty: str = "l2",
        solver: str = "lbfgs",
    ) -> TrainingResultTask:
        clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            penalty=penalty,
            solver=solver,
            multi_class="multinomial",
        )
        clf.fit(data.X_train, data.y_train)
        return self._evaluate(clf, data)

    def train_svm(
        self,
        data: DatasetSplitsTask,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str = "scale",
    ) -> TrainingResultTask:
        clf = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True,  # so we can compute ROC-AUC and subject-level probs
        )
        clf.fit(data.X_train, data.y_train)
        return self._evaluate(clf, data)

    def train_random_forest(
        self,
        data: DatasetSplitsTask,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        random_state: int = 42,
    ) -> TrainingResultTask:
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        clf.fit(data.X_train, data.y_train)
        return self._evaluate(clf, data)
