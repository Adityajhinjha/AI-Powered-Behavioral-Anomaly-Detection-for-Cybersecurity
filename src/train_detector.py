import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    PROCESSED_DATA_PATH, ISOLATION_FOREST_PATH, SCALER_PATH, MODEL_DIR,
    ISOLATION_FOREST_CONTAMINATION, ISOLATION_FOREST_N_ESTIMATORS,
    ISOLATION_FOREST_RANDOM_STATE, COLD_START_DAYS,
)

DETECTOR_FEATURES = [
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


def train(data_path=None):
    print("=" * 60)
    print("  ISOLATION FOREST TRAINING")
    print("=" * 60)

    path = data_path or PROCESSED_DATA_PATH
    print(f"\n[1/5] Loading features from {path}...")
    df = pd.read_csv(path)
    print(f"  -> {len(df)} records loaded")

    print("\n[2/5] Preparing feature matrix...")
    X = df[DETECTOR_FEATURES].values
    labels = df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  -> {X_scaled.shape[1]} features scaled")

    print("\n[3/5] Training Isolation Forest on normal data...")
    normal_mask = labels == "normal"
    X_normal = X_scaled[normal_mask]
    print(f"  -> Training set: {X_normal.shape[0]} normal events")

    model = IsolationForest(
        contamination=ISOLATION_FOREST_CONTAMINATION,
        n_estimators=ISOLATION_FOREST_N_ESTIMATORS,
        random_state=ISOLATION_FOREST_RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X_normal)

    print("\n[4/5] Scoring all events...")
    raw_scores = model.decision_function(X_scaled)

    anomaly_scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())

    cold_start_mask = df["days_since_first_seen"].values < COLD_START_DAYS
    anomaly_scores[cold_start_mask] = np.maximum(anomaly_scores[cold_start_mask], 0.5)

    df["anomaly_score"] = anomaly_scores
    df["is_anomaly_if"] = (model.predict(X_scaled) == -1).astype(int)

    print("\n[5/5] Saving model artifacts...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, ISOLATION_FOREST_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"  -> Model saved: {ISOLATION_FOREST_PATH}")
    print(f"  -> Scaler saved: {SCALER_PATH}")

    print(f"\n{'=' * 60}")
    print(f"  ISOLATION FOREST RESULTS")
    print(f"{'=' * 60}")

    normal_detected = df[df["label"] == "normal"]["is_anomaly_if"].mean()
    anomaly_detected = df[df["label"] != "normal"]["is_anomaly_if"].mean()

    print(f"\n  Normal events flagged as anomaly:  {normal_detected:.2%}")
    print(f"  Attack events detected as anomaly: {anomaly_detected:.2%}")
    print(f"\n  Anomaly score statistics:")
    print(f"    Normal events  - mean: {df[df['label'] == 'normal']['anomaly_score'].mean():.4f}, "
          f"std: {df[df['label'] == 'normal']['anomaly_score'].std():.4f}")
    print(f"    Attack events  - mean: {df[df['label'] != 'normal']['anomaly_score'].mean():.4f}, "
          f"std: {df[df['label'] != 'normal']['anomaly_score'].std():.4f}")

    print(f"\n  Detection rate by attack type:")
    for attack_type in df[df["label"] != "normal"]["label"].unique():
        mask = df["label"] == attack_type
        rate = df[mask]["is_anomaly_if"].mean()
        avg_score = df[mask]["anomaly_score"].mean()
        print(f"    {attack_type:30s}  detected: {rate:.2%}  avg_score: {avg_score:.4f}")

    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\n  -> Updated features saved with anomaly scores")

    return model, scaler, df


if __name__ == "__main__":
    train()
