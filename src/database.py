import os
import sqlite3
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATABASE_PATH


def get_connection():
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS access_logs (
            log_id TEXT PRIMARY KEY,
            entity_id TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            source_ip TEXT,
            geo_location TEXT,
            resource_accessed TEXT,
            auth_method TEXT,
            auth_success INTEGER DEFAULT 1,
            session_duration REAL,
            command_sequence TEXT,
            device_fingerprint TEXT,
            label TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS security_alerts (
            alert_id TEXT PRIMARY KEY,
            log_id TEXT REFERENCES access_logs(log_id),
            entity_id TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            risk_score REAL NOT NULL,
            predicted_anomaly_type TEXT,
            shap_explanation TEXT,
            status TEXT DEFAULT 'New'
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_logs_entity
        ON access_logs(entity_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_logs_timestamp
        ON access_logs(timestamp)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_alerts_risk
        ON security_alerts(risk_score DESC)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_alerts_entity
        ON security_alerts(entity_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_alerts_status
        ON security_alerts(status)
    """)

    conn.commit()
    conn.close()
    print(f"[OK] Database initialized at: {DATABASE_PATH}")


def save_logs(df):
    conn = get_connection()
    df.to_sql("access_logs", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print(f"[OK] Saved {len(df)} access log records to database.")


def save_alerts(df):
    conn = get_connection()
    df.to_sql("security_alerts", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print(f"[OK] Saved {len(df)} security alerts to database.")


def fetch_all_logs(limit=None):
    conn = get_connection()
    query = "SELECT * FROM access_logs ORDER BY timestamp"
    params = []
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def fetch_alerts(risk_threshold=0.0, anomaly_type=None, status=None, limit=500):
    conn = get_connection()
    query = "SELECT * FROM security_alerts WHERE risk_score >= ?"
    params = [risk_threshold]

    if anomaly_type and anomaly_type != "All":
        query += " AND predicted_anomaly_type = ?"
        params.append(anomaly_type)

    if status and status != "All":
        query += " AND status = ?"
        params.append(status)

    query += " ORDER BY risk_score DESC LIMIT ?"
    params.append(limit)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def fetch_entity_history(entity_id):
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM access_logs WHERE entity_id = ? ORDER BY timestamp",
        conn,
        params=[entity_id],
    )
    conn.close()
    return df


def fetch_entity_alerts(entity_id):
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM security_alerts WHERE entity_id = ? ORDER BY risk_score DESC",
        conn,
        params=[entity_id],
    )
    conn.close()
    return df


def update_alert_status(alert_id, new_status):
    conn = get_connection()
    conn.execute(
        "UPDATE security_alerts SET status = ? WHERE alert_id = ?",
        (new_status, alert_id),
    )
    conn.commit()
    conn.close()


def get_alert_stats():
    conn = get_connection()
    cursor = conn.cursor()

    total_logs = cursor.execute("SELECT COUNT(*) FROM access_logs").fetchone()[0]
    total_alerts = cursor.execute("SELECT COUNT(*) FROM security_alerts").fetchone()[0]

    alert_rate = (total_alerts / total_logs * 100) if total_logs > 0 else 0

    type_df = pd.read_sql_query(
        "SELECT predicted_anomaly_type, COUNT(*) as count "
        "FROM security_alerts GROUP BY predicted_anomaly_type "
        "ORDER BY count DESC",
        conn,
    )

    status_df = pd.read_sql_query(
        "SELECT status, COUNT(*) as count "
        "FROM security_alerts GROUP BY status",
        conn,
    )

    if len(type_df) > 0:
        max_count = type_df["count"].max()
        threshold = max_count * 0.95
        top_types = type_df[type_df["count"] >= threshold]["predicted_anomaly_type"].tolist()
        top_type = " / ".join(top_types)
    else:
        top_type = "N/A"

    conn.close()

    return {
        "total_logs": total_logs,
        "total_alerts": total_alerts,
        "alert_rate": round(alert_rate, 2),
        "top_anomaly_type": top_type,
        "alerts_by_type": type_df,
        "alerts_by_status": status_df,
    }


def get_unique_entity_ids():
    conn = get_connection()
    df = pd.read_sql_query("SELECT DISTINCT entity_id FROM access_logs ORDER BY entity_id", conn)
    conn.close()
    return df["entity_id"].tolist()


def get_flagged_entities(threat_type=None):
    conn = get_connection()
    if threat_type and threat_type != "All Threat Types":
        db_threat = threat_type.lower().replace(" ", "_")
        query = (
            "SELECT entity_id, COUNT(*) as alert_count, MAX(risk_score) as max_risk "
            "FROM security_alerts WHERE predicted_anomaly_type = ? "
            "GROUP BY entity_id ORDER BY max_risk DESC, alert_count DESC"
        )
        df = pd.read_sql_query(query, conn, params=(db_threat,))
    else:
        query = (
            "SELECT entity_id, COUNT(*) as alert_count, MAX(risk_score) as max_risk "
            "FROM security_alerts GROUP BY entity_id ORDER BY max_risk DESC, alert_count DESC"
        )
        df = pd.read_sql_query(query, conn)
    conn.close()
    return df


if __name__ == "__main__":
    init_db()
