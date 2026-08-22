"""
Test case verifying SHAP explanation on scored telemetry state dictionary with prediction outputs.
"""

import pandas as pd
import pytest
from src.models.predict import TelemetryPredictor
from src.explainability.shap_analysis import SHAPExplainer


def test_shap_explanation_on_scored_state():
    predictor = TelemetryPredictor()
    explainer = SHAPExplainer()

    # Telemetry state dict containing output prediction & anomaly columns
    latest_state = {
        'timestamp': '2026-08-22 14:00:00',
        'server_id': 'SRV-001',
        'server_type': 'Database',
        'region': 'us-east-1',
        'cpu_usage': 95.0,
        'memory_usage': 92.0,
        'disk_usage': 80.0,
        'network_latency': 180.0,
        'packet_loss': 4.0,
        'request_rate': 4000.0,
        'error_rate': 45.0,
        'active_connections': 1500,
        'temperature': 88.0,
        'uptime_hours': 300.0,
        'workload_intensity': 1.8,
        'previous_failures': 2,
        'maintenance_flag': 0,
        'failure': 0,
        # Output columns that previously caused ValueError
        'failure_probability': 0.88,
        'risk_score': 88.0,
        'risk_level': 'HIGH RISK',
        'predicted_failure': 1,
        'anomaly_score': 74.5,
        'is_anomaly': 1
    }

    df_feat = predictor.feature_engineer.create_features(pd.DataFrame([latest_state]))
    X_single, _, _ = predictor.feature_engineer.prepare_dataset(df_feat, is_train=False)
    X_single_scaled = predictor.scaler.transform(X_single)

    explanation = explainer.explain_instance(X_single_scaled, X_single, top_k=5)
    assert 'top_positive_factors' in explanation
    assert len(explanation['top_positive_factors']) > 0
