"""
Infrastructure Intelligence - Model Predict & Risk Scoring Engine
Handles real-time and batch inference, feature transformation,
and categorizes risk scores into LOW, MEDIUM, and HIGH risk levels.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List, Tuple


class TelemetryPredictor:
    """Inference engine for IT infrastructure failure prediction and risk scoring."""

    def __init__(
        self,
        models_dir: str = r"C:\Users\monis\.gemini\antigravity\scratch\infrastructure-intelligence\models",
        low_threshold: float = 0.30,
        high_threshold: float = 0.70
    ):
        self.models_dir = models_dir
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        self.scaler_path = os.path.join(self.models_dir, 'feature_scaler.pkl')
        self.model_path = os.path.join(self.models_dir, 'xgboost_model.pkl')
        self.fe_path = os.path.join(self.models_dir, 'feature_engineer.pkl')
        self.meta_path = os.path.join(self.models_dir, 'model_metadata.json')

        self.scaler = None
        self.model = None
        self.feature_engineer = None
        self.metadata = {}
        self.operating_threshold = 0.5

        self.load_artifacts()

    def load_artifacts(self) -> None:
        """Load trained model, scaler, feature engineering pipeline, and metadata."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Train the model first.")

        self.scaler = joblib.load(self.scaler_path)
        self.model = joblib.load(self.model_path)
        self.feature_engineer = joblib.load(self.fe_path)

        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'r') as f:
                self.metadata = json.load(f)
            self.operating_threshold = self.metadata.get('optimal_threshold', 0.5)

    def calculate_risk_level(self, probability: float) -> str:
        """Classify failure probability into Low, Medium, or High Risk."""
        if probability < self.low_threshold:
            return "LOW RISK"
        elif probability < self.high_threshold:
            return "MEDIUM RISK"
        else:
            return "HIGH RISK"

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run batch inference on raw or preprocessed telemetry DataFrame."""
        df_featured = self.feature_engineer.create_features(df)
        X, _, _ = self.feature_engineer.prepare_dataset(df_featured, is_train=False)

        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        res_df = df.copy()
        res_df['failure_probability'] = np.round(probabilities, 4)
        res_df['risk_score'] = np.round(probabilities * 100.0, 2)
        res_df['risk_level'] = [self.calculate_risk_level(p) for p in probabilities]
        res_df['predicted_failure'] = (probabilities >= self.operating_threshold).astype(int)

        return res_df

    def predict_single(self, telemetry_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Run real-time inference on a single server telemetry record."""
        df_single = pd.DataFrame([telemetry_dict])
        res_df = self.predict_dataframe(df_single)
        row = res_df.iloc[0]

        return {
            'server_id': row.get('server_id', 'UNKNOWN'),
            'timestamp': str(row.get('timestamp', '')),
            'failure_probability': float(row['failure_probability']),
            'risk_score': float(row['risk_score']),
            'risk_level': str(row['risk_level']),
            'predicted_failure': int(row['predicted_failure']),
            'operating_threshold': self.operating_threshold
        }


if __name__ == "__main__":
    predictor = TelemetryPredictor()
    sample = {
        'timestamp': '2026-08-22 12:00:00',
        'server_id': 'SRV-001',
        'server_type': 'Database',
        'region': 'us-east-1',
        'cpu_usage': 94.5,
        'memory_usage': 96.2,
        'disk_usage': 82.0,
        'network_latency': 180.0,
        'packet_loss': 4.5,
        'request_rate': 4200.0,
        'error_rate': 45.0,
        'active_connections': 1800,
        'temperature': 88.0,
        'uptime_hours': 340.0,
        'workload_intensity': 1.85,
        'previous_failures': 2,
        'maintenance_flag': 0,
        'failure': 0
    }
    result = predictor.predict_single(sample)
    print("Sample Inference Result:", json.dumps(result, indent=2))
