"""
Infrastructure Intelligence - Pytest Feature Engineering Tests
"""

import pandas as pd
import numpy as np
import pytest
from src.features.feature_engineering import FeatureEngineer, temporal_train_val_test_split


def test_create_features():
    fe = FeatureEngineer()
    df = pd.DataFrame([
        {
            'timestamp': f'2026-08-22 {h:02d}:00:00',
            'server_id': 'SRV-001',
            'server_type': 'Database',
            'region': 'us-east-1',
            'cpu_usage': 50.0 + h,
            'memory_usage': 60.0,
            'disk_usage': 40.0,
            'network_latency': 20.0,
            'packet_loss': 0.0,
            'request_rate': 500.0,
            'error_rate': 0.0,
            'active_connections': 100,
            'temperature': 55.0,
            'uptime_hours': float(h),
            'workload_intensity': 1.0,
            'previous_failures': 0,
            'maintenance_flag': 0,
            'failure': 0
        }
        for h in range(10)
    ])

    df_feat = fe.create_features(df)

    assert 'cpu_memory_stress' in df_feat.columns
    assert 'network_stress_score' in df_feat.columns
    assert 'cpu_roll_mean_3h' in df_feat.columns
    assert len(df_feat) == 10
