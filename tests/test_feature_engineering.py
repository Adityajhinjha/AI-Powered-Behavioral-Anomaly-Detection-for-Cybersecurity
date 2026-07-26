import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_engineering import extract_features
from src.utils import calculate_geo_velocity, is_off_hours, parse_timestamp


def test_geo_velocity_calculation():
    speed = calculate_geo_velocity(40.7128, -74.0060, 51.5074, -0.1278, 3600)
    assert speed > 500.0

    zero_speed = calculate_geo_velocity(40.7128, -74.0060, 40.7128, -74.0060, 100)
    assert zero_speed == 0.0

    invalid_speed = calculate_geo_velocity(40.7128, -74.0060, 51.5074, -0.1278, 0)
    assert invalid_speed == 0.0


def test_is_off_hours():
    assert is_off_hours(23) is True
    assert is_off_hours(3) is True
    assert is_off_hours(14) is False


def test_parse_timestamp():
    dt = parse_timestamp("2026-07-26 12:00:00")
    assert dt.year == 2026
    assert dt.month == 7


def test_extract_features_structure():
    raw_df = pd.DataFrame([{
        "log_id": "test_01",
        "entity_id": "user_001",
        "entity_type": "user",
        "timestamp": "2026-07-26 12:00:00",
        "source_ip": "192.168.1.10",
        "geo_location": "40.7128,-74.0060",
        "resource_accessed": "/api/reports",
        "auth_method": "password",
        "auth_success": 1,
        "session_duration": 300.0,
        "command_sequence": "login;query;export",
        "device_fingerprint": "Windows 11|00:1A:2B:3C:4D:5E|HTTPS",
        "label": "normal"
    }])

    feat_df = extract_features(raw_df)
    assert len(feat_df) == 1
    assert "hour_of_day" in feat_df.columns
    assert "geo_velocity" in feat_df.columns
    assert "auth_failure_rate_1h" in feat_df.columns
    assert "session_duration_zscore" in feat_df.columns
