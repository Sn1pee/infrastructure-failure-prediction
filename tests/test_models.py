"""
Infrastructure Intelligence - Pytest Model & Predictor Tests
"""

import pytest
import pandas as pd
from src.models.predict import TelemetryPredictor


def test_predictor_single_inference():
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

    res = predictor.predict_single(sample)
    assert 'failure_probability' in res
    assert 'risk_score' in res
    assert res['risk_level'] in ['LOW RISK', 'MEDIUM RISK', 'HIGH RISK']
    assert 0.0 <= res['failure_probability'] <= 1.0
