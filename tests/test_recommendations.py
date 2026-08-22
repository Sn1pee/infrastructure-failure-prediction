"""
Infrastructure Intelligence - Pytest Risk Scoring & Recommendation Tests
"""

import pytest
from src.models.predict import TelemetryPredictor
from src.recommendations.recommendation_engine import RecommendationEngine


def test_risk_score_categorization():
    predictor = TelemetryPredictor()
    assert predictor.calculate_risk_level(0.10) == "LOW RISK"
    assert predictor.calculate_risk_level(0.50) == "MEDIUM RISK"
    assert predictor.calculate_risk_level(0.85) == "HIGH RISK"


def test_recommendation_rules():
    sample_telemetry = {
        'cpu_usage': 95.0,
        'memory_usage': 90.0,
        'network_latency': 200.0,
        'packet_loss': 5.0,
        'error_rate': 50.0,
        'temperature': 92.0,
        'previous_failures': 3
    }

    recs = RecommendationEngine.generate_recommendations(sample_telemetry, risk_level='HIGH RISK')
    assert len(recs) >= 3

    categories = [r['category'] for r in recs]
    assert 'CPU Saturation' in categories
    assert 'Memory Contention' in categories
    assert 'Network Degradation' in categories
