from __future__ import annotations
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score
)

def aggregate_subject_probas(
    y_proba: np.ndarray,
    y_true: np.ndarray,
    subject_ids: np.ndarray,
    label_names: List[str],
) -> Dict[str, Any]:
    """
    Aggregate frame/task-level probabilities to subject-level:

    Inputs:
      y_proba: (N_samples, num_classes) predicted probabilities
      y_true:  (N_samples,) true labels (int) per sample
      subject_ids: (N_samples,) subject_id per sample (string or int)
      label_names: list of class names, ordered by class index

    Returns:
      dict with:
        - subject_df: DataFrame with per-subject probs & predictions
        - metrics: standard classification metrics at subject level
    """
    if y_proba is None:
        raise ValueError("y_proba must not be None for subject-level aggregation")
    
    num_classes = y_proba.shape[1]
    df = pd.DataFrame({
        "subject_id": subject_ids,
        "y_true": y_true,
    })

    # Add prob columns
    for c in range(num_classes):
        df[f"p_{c}"] = y_proba[:, c]

    # Group by subject, average probabilities
    grouped = df.groupby("subject_id")
    agg_records = []
    for subject_id, g in grouped:
        # Average probs
        mean_probas = g[[f"p_{c}" for c in range(num_classes)]].mean(axis=0).to_numpy()

        # True label: all rows for a subject should share the same label
        y_true_subj = int(g["y_true"].iloc[0])
        y_pred_subj = int(np.argmax(mean_probas))

        rec = {
            "subject_id": subject_id,
            "y_true": y_true_subj,
            "y_pred": y_pred_subj,
        }

        for c in range(num_classes):
            rec[f"p_{c}"] = float(mean_probas[c])
        agg_records.append(rec)

    subj_df = pd.DataFrame.from_records(agg_records)

    # Compute metrics at subject-level
    y_true_subj = subj_df["y_true"].to_numpy()
    y_pred_subj = subj_df["y_pred"].to_numpy()
    prob_cols = [f"p_{c}" for c in range(num_classes)]
    y_proba_subj = subj_df[prob_cols].to_numpy()

    metrics = compute_classification_metrics_core(
        y_true=y_true_subj,
        y_pred=y_pred_subj,
        y_proba=y_proba_subj,
        label_names=label_names,
    )

    return {
        "subject_df": subj_df,
        "metrics": metrics,
    }

def compute_classification_metrics_core(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        label_names: List[str],
) -> Dict[str, Any]:
    """
    Core metric computation used for both task-level and subject-level
    """
    metrics: Dict[str, Any] = {}

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(label_names))
    )

    metrics["per_class"] = {
        label_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(label_names))
    }

    if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == len(label_names):
        try:
            roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr")
            metrics["roc_auc_ovr"] = float(roc_auc)
        except ValueError:
            metrics["roc_auc_ovr"] = None
    else:
        metrics["roc_auc_ovr"] = None

    return metrics
