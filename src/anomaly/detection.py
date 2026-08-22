"""
Infrastructure Intelligence - Anomaly Detection Module
Uses unsupervised Isolation Forest on telemetry metrics to detect unusual, outlier
system behaviors before explicit failures occur.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """Unsupervised Isolation Forest engine for telemetry anomaly detection."""

    ANOMALY_FEATURE_COLS = [
        'cpu_usage', 'memory_usage', 'disk_usage', 'network_latency',
        'packet_loss', 'request_rate', 'error_rate', 'active_connections',
        'temperature', 'workload_intensity'
    ]

    def __init__(
        self,
        contamination: float = 0.05,
        models_dir: str = r"C:\Users\monis\.gemini\antigravity\scratch\infrastructure-intelligence\models"
    ):
        self.contamination = contamination
        self.models_dir = models_dir
        self.model_path = os.path.join(self.models_dir, 'isolation_forest.pkl')
        self.scaler_path = os.path.join(self.models_dir, 'anomaly_scaler.pkl')

        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        )

    def fit(self, df: pd.DataFrame) -> None:
        """Fit scaler and Isolation Forest model on telemetry metrics."""
        os.makedirs(self.models_dir, exist_ok=True)
        X = df[self.ANOMALY_FEATURE_COLS].copy()
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled)

        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.model, self.model_path)
        print(f"Trained IsolationForest anomaly model (contamination={self.contamination}) and saved to: {self.model_path}")

    def load_model(self) -> None:
        """Load trained Isolation Forest model and scaler."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Anomaly model not found at {self.model_path}. Run fit() first.")
        self.scaler = joblib.load(self.scaler_path)
        self.model = joblib.load(self.model_path)

    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score observations and return dataframe with anomaly scores and binary flags."""
        X = df[self.ANOMALY_FEATURE_COLS].copy()
        X_scaled = self.scaler.transform(X)

        # decision_function returns negative for anomalies, positive for normal
        raw_scores = self.model.decision_function(X_scaled)
        predictions = self.model.predict(X_scaled)  # -1 for anomaly, 1 for normal

        # Convert raw decision score into normalized Anomaly Score (0 to 100, where 100 is highly anomalous)
        # Decision function typically spans [-0.3, +0.3]
        anomaly_scores = np.clip((0.2 - raw_scores) * 200.0, 0.0, 100.0)

        res_df = df.copy()
        res_df['anomaly_score'] = np.round(anomaly_scores, 2)
        res_df['is_anomaly'] = (predictions == -1).astype(int)

        return res_df


def run_anomaly_training(
    cleaned_csv_path: str = r"C:\Users\monis\.gemini\antigravity\scratch\infrastructure-intelligence\data\processed\telemetry_cleaned.csv"
) -> AnomalyDetector:
    """Train and evaluate unsupervised anomaly detector."""
    print("--- Starting Unsupervised Anomaly Detection Training ---")
    df = pd.read_csv(cleaned_csv_path)

    detector = AnomalyDetector(contamination=0.05)
    detector.fit(df)
    scored_df = detector.detect_anomalies(df)

    anom_count = scored_df['is_anomaly'].sum()
    print(f"Detected {anom_count:,} anomalies ({anom_count/len(df):.2%}) in telemetry dataset.")
    print(f"Top 5 Most Anomalous Servers:")
    top_anom = scored_df.sort_values(by='anomaly_score', ascending=False)[['timestamp', 'server_id', 'server_type', 'cpu_usage', 'memory_usage', 'network_latency', 'anomaly_score']].head(5)
    print(top_anom.to_string(index=False))

    return detector


if __name__ == "__main__":
    run_anomaly_training()
