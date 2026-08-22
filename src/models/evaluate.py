"""
Infrastructure Intelligence - Model Evaluation Module
Calculates Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Confusion Matrix,
and handles threshold tuning for class-imbalanced infrastructure telemetry.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, classification_report
)


class ModelEvaluator:
    """Evaluates classification models with focus on recall-precision trade-offs."""

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Compute complete metric matrix at specified classification threshold."""
        y_pred = (y_probs >= threshold).astype(int)

        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        try:
            roc_auc = float(roc_auc_score(y_true, y_probs))
        except Exception:
            roc_auc = 0.0

        p_curve, r_curve, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = float(auc(r_curve, p_curve))

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        return {
            'threshold': round(float(threshold), 3),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(roc_auc, 4),
            'pr_auc': round(pr_auc, 4),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }

    @staticmethod
    def find_optimal_threshold(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        min_recall: float = 0.80
    ) -> Tuple[float, Dict[str, Any]]:
        """Find decision threshold maximizing F1 while enforcing minimum required recall."""
        thresholds = np.linspace(0.05, 0.95, 91)
        best_threshold = 0.5
        best_f1 = -1.0
        best_metrics = {}

        for th in thresholds:
            metrics = ModelEvaluator.calculate_metrics(y_true, y_probs, threshold=th)
            if metrics['recall'] >= min_recall and metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_threshold = th
                best_metrics = metrics

        # Fallback to max F1 if min_recall cannot be satisfied
        if not best_metrics:
            for th in thresholds:
                metrics = ModelEvaluator.calculate_metrics(y_true, y_probs, threshold=th)
                if metrics['f1_score'] > best_f1:
                    best_f1 = metrics['f1_score']
                    best_threshold = th
                    best_metrics = metrics

        return best_threshold, best_metrics

    @staticmethod
    def generate_comparison_table(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Create clean comparison DataFrame across multiple trained models."""
        rows = []
        for model_name, metrics in results_dict.items():
            rows.append({
                'Model': model_name,
                'Threshold': metrics['threshold'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc'],
                'PR-AUC': metrics['pr_auc'],
                'TP': metrics['true_positives'],
                'FP': metrics['false_positives'],
                'FN': metrics['false_negatives'],
                'TN': metrics['true_negatives']
            })
        df_comp = pd.DataFrame(rows).sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
        return df_comp
