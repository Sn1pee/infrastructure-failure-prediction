"""
Infrastructure Intelligence - Model Training & Hyperparameter Tuning Pipeline
Trains Logistic Regression, Decision Tree, Random Forest, and XGBoost models
with class imbalance handling, temporal CV splits, threshold tuning, and model export.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import xgboost as xgb

from src.features.feature_engineering import run_feature_engineering_pipeline, FeatureEngineer
from src.models.evaluate import ModelEvaluator


class InfrastructureModelTrainer:
    """Handles end-to-end model training, hyperparameter search, and artifact persistence."""

    def __init__(self, models_dir: str = r"C:\Users\monis\.gemini\antigravity\scratch\infrastructure-intelligence\models"):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = ""
        self.optimal_threshold = 0.5
        self.feature_names = []

    def fit_scaler(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit StandardScaler on training features and return scaled matrix."""
        self.feature_names = X_train.columns.tolist()
        return self.scaler.fit_transform(X_train)

    def transform_scaler(self, X: pd.DataFrame) -> np.ndarray:
        """Transform feature matrix using fitted scaler."""
        return self.scaler.transform(X)

    def train_models(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
        X_test: pd.DataFrame, y_test: pd.Series
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Train baseline, tree, and gradient boosting models and select top performer."""
        X_tr_scaled = self.fit_scaler(X_train)
        X_val_scaled = self.transform_scaler(X_val)
        X_te_scaled = self.transform_scaler(X_test)

        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos = neg_count / max(1, pos_count)

        models = {
            'Logistic Regression (Baseline)': LogisticRegression(
                class_weight='balanced', max_iter=1000, random_state=42
            ),
            'Decision Tree': DecisionTreeClassifier(
                class_weight='balanced', max_depth=8, min_samples_leaf=10, random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=120, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=42
            ),
            'XGBoost Classifier': xgb.XGBClassifier(
                n_estimators=150, max_depth=6, learning_rate=0.05,
                scale_pos_weight=scale_pos, eval_metric='logloss',
                random_state=42, n_jobs=-1
            )
        }

        print("\n--- Training Model Candidates ---")
        val_results = {}
        trained_instances = {}

        for name, model in models.items():
            print(f"Fitting {name}...")
            model.fit(X_tr_scaled, y_train)
            trained_instances[name] = model

            # Validate probabilities
            val_probs = model.predict_proba(X_val_scaled)[:, 1]
            opt_th, val_metric = ModelEvaluator.find_optimal_threshold(y_val.values, val_probs, min_recall=0.80)
            val_results[name] = {
                'model': model,
                'opt_threshold': opt_th,
                'val_metrics': val_metric
            }
            print(f"  -> {name} | Val F1: {val_metric['f1_score']:.4f} | Recall: {val_metric['recall']:.4f} | Opt Threshold: {opt_th:.2f}")

        # Hyperparameter Tuning on XGBoost using RandomizedSearchCV
        print("\n--- Hyperparameter Tuning XGBoost Candidate ---")
        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.03, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0]
        }

        xgb_base = xgb.XGBClassifier(
            scale_pos_weight=scale_pos, eval_metric='logloss', random_state=42, n_jobs=-1
        )
        tscv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(
            xgb_base, param_distributions=param_grid, n_iter=8,
            scoring='f1', cv=tscv, random_state=42, n_jobs=-1
        )
        search.fit(X_tr_scaled, y_train)

        tuned_xgb = search.best_estimator_
        tuned_val_probs = tuned_xgb.predict_proba(X_val_scaled)[:, 1]
        tuned_th, tuned_val_metric = ModelEvaluator.find_optimal_threshold(y_val.values, tuned_val_probs, min_recall=0.80)

        trained_instances['XGBoost (Tuned)'] = tuned_xgb
        val_results['XGBoost (Tuned)'] = {
            'model': tuned_xgb,
            'opt_threshold': tuned_th,
            'val_metrics': tuned_val_metric
        }
        print(f"  -> XGBoost (Tuned) Best Params: {search.best_params_}")
        print(f"  -> XGBoost (Tuned) | Val F1: {tuned_val_metric['f1_score']:.4f} | Recall: {tuned_val_metric['recall']:.4f}")

        # Test Set Final Evaluation across all models
        print("\n--- Final Test Set Evaluation ---")
        test_eval_results = {}

        for name, info in trained_instances.items():
            model = info
            opt_th = val_results[name]['opt_threshold']
            te_probs = model.predict_proba(X_te_scaled)[:, 1]
            te_metrics = ModelEvaluator.calculate_metrics(y_test.values, te_probs, threshold=opt_th)
            test_eval_results[name] = te_metrics

        comparison_df = ModelEvaluator.generate_comparison_table(test_eval_results)
        print("\n=== Model Benchmark Matrix (Test Set) ===")
        print(comparison_df.to_string(index=False))

        # Select overall best model (XGBoost Tuned or highest F1)
        best_row = comparison_df.iloc[0]
        self.best_model_name = str(best_row['Model'])
        self.best_model = trained_instances[self.best_model_name]
        self.optimal_threshold = float(best_row['Threshold'])

        print(f"\n[CHAMPION MODEL SELECTED]: {self.best_model_name} (Operating Threshold: {self.optimal_threshold:.2f})")

        # Save Artifacts
        self.save_artifacts(test_eval_results, comparison_df)

        return test_eval_results, comparison_df

    def save_artifacts(self, test_results: Dict[str, Any], comparison_df: pd.DataFrame) -> None:
        """Export serialized model, scaler, and metadata JSON."""
        # 1. Save Scaler
        scaler_path = os.path.join(self.models_dir, 'feature_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)

        # 2. Save Champion Model
        model_path = os.path.join(self.models_dir, 'xgboost_model.pkl')
        joblib.dump(self.best_model, model_path)

        # 3. Save Metadata JSON
        meta_path = os.path.join(self.models_dir, 'model_metadata.json')
        meta = {
            'champion_model': self.best_model_name,
            'optimal_threshold': self.optimal_threshold,
            'feature_names': self.feature_names,
            'test_benchmark': test_results[self.best_model_name],
            'all_models_summary': test_results
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=4)

        # 4. Save CSV comparison table
        comparison_df.to_csv(os.path.join(self.models_dir, 'model_comparison.csv'), index=False)

        print(f"Exported model artifacts to: {self.models_dir}")


def run_training_pipeline() -> None:
    """Execute feature pipeline, train candidate models, evaluate, and save artifacts."""
    train_df, val_df, test_df, df_featured, fe = run_feature_engineering_pipeline()

    X_train, y_train, feat_names = fe.prepare_dataset(train_df, is_train=True)
    X_val, y_val, _ = fe.prepare_dataset(val_df, is_train=False)
    X_test, y_test, _ = fe.prepare_dataset(test_df, is_train=False)

    trainer = InfrastructureModelTrainer()
    trainer.train_models(X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == "__main__":
    run_training_pipeline()
