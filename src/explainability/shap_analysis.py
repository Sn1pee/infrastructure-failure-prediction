"""
Infrastructure Intelligence - SHAP Explainable AI (XAI) Module
Computes global feature importance and local instance-level SHAP root cause explanations
for champion tree-based infrastructure failure prediction models.
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List, Tuple


class SHAPExplainer:
    """Generates SHAP global and local explanations for model predictions."""

    FEATURE_HUMAN_NAMES = {
        'cpu_usage': 'CPU Utilization (%)',
        'memory_usage': 'Memory Utilization (%)',
        'disk_usage': 'Disk Utilization (%)',
        'network_latency': 'Network Latency (ms)',
        'packet_loss': 'Packet Loss (%)',
        'request_rate': 'Request Rate (req/sec)',
        'error_rate': 'Application Error Rate (errors/sec)',
        'active_connections': 'Active Connection Count',
        'temperature': 'Server Temperature (°C)',
        'uptime_hours': 'System Uptime (Hours)',
        'workload_intensity': 'Workload Intensity Index',
        'previous_failures': 'Historical Failure Count',
        'maintenance_flag': 'Active Maintenance Status',
        'cpu_memory_stress': 'CPU-Memory Stress Interaction',
        'network_stress_score': 'Network Latency & Packet Loss Stress',
        'system_utilization_index': 'Overall Infrastructure Utilization Index',
        'error_to_request_ratio': 'Error-to-Request Ratio',
        'thermal_efficiency_delta': 'Thermal Heating Efficiency Delta',
        'conn_per_request': 'Connections-per-Request Ratio',
        'cpu_roll_mean_3h': 'CPU Usage 3-Hour Rolling Mean',
        'cpu_roll_std_3h': 'CPU Volatility 3-Hour Standard Deviation',
        'memory_roll_mean_3h': 'Memory Usage 3-Hour Rolling Mean',
        'latency_roll_mean_3h': 'Latency 3-Hour Rolling Mean',
        'error_roll_mean_3h': 'Error Rate 3-Hour Rolling Mean',
        'cpu_roll_mean_6h': 'CPU Usage 6-Hour Rolling Mean',
        'memory_roll_mean_6h': 'Memory Usage 6-Hour Rolling Mean',
        'error_roll_mean_6h': 'Error Rate 6-Hour Rolling Mean',
        'hour_of_day': 'Hour of Day',
        'day_of_week': 'Day of Week',
        'is_weekend': 'Is Weekend Flag'
    }

    def __init__(self, models_dir: str = r"C:\Users\monis\.gemini\antigravity\scratch\infrastructure-intelligence\models"):
        self.models_dir = models_dir
        self.model_path = os.path.join(self.models_dir, 'xgboost_model.pkl')
        self.scaler_path = os.path.join(self.models_dir, 'feature_scaler.pkl')

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.explainer = shap.TreeExplainer(self.model)

    def get_feature_name(self, raw_col: str) -> str:
        """Translate raw column name into clean human-readable title."""
        if raw_col in self.FEATURE_HUMAN_NAMES:
            return self.FEATURE_HUMAN_NAMES[raw_col]
        # Clean categorical names like server_type_Database
        cleaned = raw_col.replace('server_type_', 'Server Type: ').replace('region_', 'Region: ')
        return cleaned

    def compute_global_importance(self, X_scaled: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
        """Calculate mean absolute SHAP feature importance across entire dataset."""
        shap_vals = self.explainer.shap_values(X_scaled)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)

        df_imp = pd.DataFrame({
            'feature': feature_names,
            'human_feature': [self.get_feature_name(f) for f in feature_names],
            'importance': mean_abs_shap
        }).sort_values(by='importance', ascending=False).reset_index(drop=True)

        return df_imp

    def explain_instance(
        self,
        X_single_scaled: np.ndarray,
        X_single_raw: pd.DataFrame,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Generate instance-level SHAP explanation for a single server state."""
        shap_vals = self.explainer.shap_values(X_single_scaled)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        
        shap_vals = np.array(shap_vals)
        while shap_vals.ndim > 1:
            if shap_vals.ndim == 3:
                shap_vals = shap_vals[0, :, 1] if shap_vals.shape[2] > 1 else shap_vals[0, :, 0]
            elif shap_vals.ndim == 2:
                shap_vals = shap_vals[0]

        exp_val = self.explainer.expected_value
        if isinstance(exp_val, (list, np.ndarray)):
            base_value = float(exp_val[-1])
        else:
            base_value = float(exp_val)

        feature_names = X_single_raw.columns.tolist()

        contributions = []
        for idx, col in enumerate(feature_names):
            raw_val = X_single_raw.iloc[0][col]
            shap_val = float(shap_vals[idx])

            contributions.append({
                'feature': col,
                'human_feature': self.get_feature_name(col),
                'feature_value': raw_val,
                'shap_value': round(shap_val, 4),
                'impact': 'Increases Risk' if shap_val > 0 else 'Decreases Risk'
            })

        df_contrib = pd.DataFrame(contributions)

        # Top positive risk drivers
        top_positive = df_contrib.sort_values(by='shap_value', ascending=False).head(top_k).to_dict('records')

        # Top negative protective factors
        top_negative = df_contrib.sort_values(by='shap_value', ascending=True).head(top_k).to_dict('records')

        return {
            'base_value': round(base_value, 4),
            'top_positive_factors': top_positive,
            'top_negative_factors': top_negative,
            'all_contributions': df_contrib
        }


if __name__ == "__main__":
    from src.models.predict import TelemetryPredictor
    predictor = TelemetryPredictor()
    fe = predictor.feature_engineer

    sample_df = pd.DataFrame([{
        'timestamp': '2026-08-22 14:00:00',
        'server_id': 'SRV-002',
        'server_type': 'Web Server',
        'region': 'us-east-1',
        'cpu_usage': 92.0,
        'memory_usage': 94.5,
        'disk_usage': 80.0,
        'network_latency': 195.0,
        'packet_loss': 5.2,
        'request_rate': 4500.0,
        'error_rate': 65.0,
        'active_connections': 2100,
        'temperature': 89.0,
        'uptime_hours': 450.0,
        'workload_intensity': 2.1,
        'previous_failures': 3,
        'maintenance_flag': 0,
        'failure': 1
    }])

    df_feat = fe.create_features(sample_df)
    X, _, _ = fe.prepare_dataset(df_feat, is_train=False)
    X_scaled = predictor.scaler.transform(X)

    explainer = SHAPExplainer()
    explanation = explainer.explain_instance(X_scaled, X)
    print("Instance Explanation Top Risk Factors:")
    for factor in explanation['top_positive_factors']:
        print(f"  + {factor['human_feature']} (val: {factor['feature_value']}) -> SHAP impact: +{factor['shap_value']}")
