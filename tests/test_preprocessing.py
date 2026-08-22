"""
Infrastructure Intelligence - Pytest Preprocessing Tests
"""

import pandas as pd
import numpy as np
import pytest
from src.data.preprocessing import DataPreprocessor


def test_clean_telemetry():
    preprocessor = DataPreprocessor(db_path=":memory:")
    raw_df = pd.DataFrame([{
        'timestamp': '2026-08-22 12:00:00',
        'server_id': 'SRV-001',
        'server_type': 'Database',
        'region': 'us-east-1',
        'cpu_usage': 105.0,  # Needs clipping to 100
        'memory_usage': -5.0,  # Needs clipping to 0
        'disk_usage': 50.0,
        'network_latency': 40.0,
        'packet_loss': 0.0,
        'request_rate': 1000.0,
        'error_rate': 1.0,
        'active_connections': 100,
        'temperature': 50.0,
        'uptime_hours': 10.0,
        'workload_intensity': 1.0,
        'previous_failures': 0,
        'maintenance_flag': 0,
        'failure': 0
    }])

    cleaned = preprocessor.clean_telemetry(raw_df)
    assert cleaned['cpu_usage'].iloc[0] == 100.0
    assert cleaned['memory_usage'].iloc[0] == 0.0
    assert pd.api.types.is_datetime64_any_dtype(cleaned['timestamp'])
