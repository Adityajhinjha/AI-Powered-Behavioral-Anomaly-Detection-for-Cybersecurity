import os
import sys
import sqlite3
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DATABASE_PATH
from src.database import (
    init_db, get_connection, save_logs, save_alerts,
    fetch_all_logs, fetch_alerts, get_alert_stats, get_unique_entity_ids
)


def test_database_initialization():
    init_db()
    assert os.path.exists(DATABASE_PATH)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()

    assert "access_logs" in tables
    assert "security_alerts" in tables


def test_save_and_fetch_logs():
    init_db()
    dummy_logs = pd.DataFrame([{
        "log_id": "test_log_001",
        "entity_id": "test_user_1",
        "entity_type": "user",
        "timestamp": "2026-01-01 12:00:00",
        "source_ip": "192.168.1.1",
        "geo_location": "40.7128,-74.0060",
        "resource_accessed": "/api/test",
        "auth_method": "password",
        "auth_success": 1,
        "session_duration": 120.0,
        "command_sequence": "login;fetch",
        "device_fingerprint": "Windows|MAC|HTTPS",
        "label": "normal"
    }])
    save_logs(dummy_logs)

    fetched = fetch_all_logs(limit=10)
    assert len(fetched) > 0
    assert "entity_id" in fetched.columns


def test_get_alert_stats():
    init_db()
    stats = get_alert_stats()
    assert "total_logs" in stats
    assert "total_alerts" in stats
    assert "alert_rate" in stats
    assert "alerts_by_type" in stats


def test_get_unique_entity_ids():
    init_db()
    entities = get_unique_entity_ids()
    assert isinstance(entities, list)
