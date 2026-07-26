import os
import sys
import uuid
import json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    PROCESSED_DATA_PATH, ISOLATION_FOREST_PATH, XGBOOST_MODEL_PATH,
    SCALER_PATH, LABEL_ENCODER_PATH, RISK_SCORE_THRESHOLD,
)
from src.database import init_db, save_alerts
from src.explainability import generate_explanations

FEATURE_COLUMNS = [
    "hour_of_day", "day_of_week", "is_weekend", "is_off_hours",
    "time_since_last_login",
    "auth_success", "failed_auth_count_1h", "auth_failure_rate_1h",
    "unique_source_ips_1h", "total_events_1h",
    "geo_velocity", "geo_distance_from_home",
    "unique_resources_1h", "resource_diversity_ratio", "new_resource_flag",
    "fingerprint_changed", "new_device_flag",
    "session_duration", "session_duration_zscore", "command_sequence_length",
    "days_since_first_seen",
]


def load_models():
    print("  Loading model artifacts...")
    iso_forest = joblib.load(ISOLATION_FOREST_PATH)
    xgb_model = joblib.load(XGBOOST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    print("  -> All models loaded successfully")
    return iso_forest, xgb_model, scaler, le


def predict_batch(df=None):
    print("=" * 60)
    print("  PREDICTION PIPELINE")
    print("=" * 60)

    if df is None:
        print(f"\n[1/5] Loading processed features...")
        df = pd.read_csv(PROCESSED_DATA_PATH)
    else:
        print(f"\n[1/5] Using provided DataFrame...")
    print(f"  -> {len(df)} records")

    print("\n[2/5] Loading trained models...")
    iso_forest, xgb_model, scaler, le = load_models()

    X = df[FEATURE_COLUMNS].fillna(0).values
    X_scaled = scaler.transform(X)

    print("\n[3/5] Running Isolation Forest scoring...")
    raw_scores = iso_forest.decision_function(X_scaled)
    anomaly_scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    df["anomaly_score_if"] = anomaly_scores

    print("\n[4/5] Running XGBoost classification...")
    y_pred = xgb_model.predict(X)
    y_prob = xgb_model.predict_proba(X)

    df["predicted_class_idx"] = y_pred
    df["predicted_anomaly_type"] = le.inverse_transform(y_pred)
    df["risk_score"] = y_prob.max(axis=1)

    df["combined_risk_score"] = (
        0.4 * df["anomaly_score_if"] + 0.6 * df["risk_score"]
    )

    normal_mask = df["predicted_anomaly_type"] == "normal"
    df.loc[normal_mask, "combined_risk_score"] *= 0.3

    print("\n[5/5] Generating security alerts...")
    alert_mask = (
        (df["predicted_anomaly_type"] != "normal") |
        (df["combined_risk_score"] >= RISK_SCORE_THRESHOLD)
    )
    alert_df = df[alert_mask].copy()

    print(f"  -> {len(alert_df)} events flagged as alerts")
    print("  -> Generating SHAP explanations...")
    shap_explanations = generate_explanations(
        xgb_model, alert_df[FEATURE_COLUMNS].fillna(0).values, FEATURE_COLUMNS
    )

    alerts = pd.DataFrame({
        "alert_id": [str(uuid.uuid4()) for _ in range(len(alert_df))],
        "log_id": alert_df["log_id"].values,
        "entity_id": alert_df["entity_id"].values,
        "timestamp": alert_df["timestamp"].values,
        "risk_score": alert_df["combined_risk_score"].values.round(4),
        "predicted_anomaly_type": alert_df["predicted_anomaly_type"].values,
        "shap_explanation": shap_explanations,
        "status": "New",
    })

    alerts = alerts.sort_values("risk_score", ascending=False).reset_index(drop=True)

    init_db()
    save_alerts(alerts)

    print(f"\n{'=' * 60}")
    print(f"  PREDICTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n  Total events scored:     {len(df)}")
    print(f"  Alerts generated:        {len(alerts)}")
    print(f"  Alert rate:              {len(alerts) / len(df) * 100:.2f}%")
    print(f"\n  Alerts by type:")
    for atype, count in alerts["predicted_anomaly_type"].value_counts().items():
        print(f"    {atype:30s}  {count:5d}")
    print(f"\n  Risk score distribution:")
    print(f"    Critical (>=0.8):  {(alerts['risk_score'] >= 0.8).sum()}")
    print(f"    High (0.6-0.8):   {((alerts['risk_score'] >= 0.6) & (alerts['risk_score'] < 0.8)).sum()}")
    print(f"    Medium (0.4-0.6): {((alerts['risk_score'] >= 0.4) & (alerts['risk_score'] < 0.6)).sum()}")
    print(f"    Low (<0.4):       {(alerts['risk_score'] < 0.4).sum()}")

    return alerts


if __name__ == "__main__":
    predict_batch()
