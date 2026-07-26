"""
Feature Engineering Pipeline.
Transforms raw access logs into ML-ready feature vectors for anomaly detection
and attack classification.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH,
    OFF_HOURS_START, OFF_HOURS_END, ROLLING_WINDOW_SECONDS,
)
from src.utils import calculate_geo_velocity, is_off_hours


def _parse_geo(geo_str):
    """Parse 'lat,lon' string to floats."""
    try:
        parts = str(geo_str).split(",")
        return float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        return 0.0, 0.0


def _parse_fingerprint(fp_str):
    """Parse 'OS|MAC|Protocol' fingerprint string."""
    try:
        parts = str(fp_str).split("|")
        return {
            "os": parts[0] if len(parts) > 0 else "unknown",
            "mac": parts[1] if len(parts) > 1 else "unknown",
            "protocol": parts[2] if len(parts) > 2 else "unknown",
        }
    except Exception:
        return {"os": "unknown", "mac": "unknown", "protocol": "unknown"}


def extract_features(df=None):
    """
    Main feature extraction pipeline.

    Args:
        df: Optional DataFrame of raw logs. If None, loads from RAW_DATA_PATH.

    Returns:
        DataFrame with engineered features + original label column.
    """
    print("=" * 60)
    print("  FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    # Load data
    if df is None:
        print(f"\n[1/7] Loading raw data from {RAW_DATA_PATH}...")
        df = pd.read_csv(RAW_DATA_PATH)
    else:
        print("\n[1/7] Using provided DataFrame...")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    print(f"  -> {len(df)} records loaded")

    # ─── Temporal Features ────────────────────────────────────────────────
    print("\n[2/7] Extracting temporal features...")
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Mon, 6=Sun
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_off_hours"] = df["hour_of_day"].apply(
        lambda h: int(is_off_hours(h, OFF_HOURS_START, OFF_HOURS_END))
    )

    # Time since last login per entity
    df["prev_timestamp"] = df.groupby("entity_id")["timestamp"].shift(1)
    df["time_since_last_login"] = (
        (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds().fillna(0)
    )

    # ─── Authentication Features ──────────────────────────────────────────
    print("[3/7] Extracting authentication features...")
    df["auth_success"] = df["auth_success"].astype(int)

    # Rolling 1-hour failed auth count per entity
    df_sorted = df.set_index("timestamp").sort_index()

    failed_counts = []
    total_counts = []
    unique_ips = []

    for entity_id, group in df.groupby("entity_id"):
        group = group.sort_values("timestamp")

        for idx, row in group.iterrows():
            window_start = row["timestamp"] - pd.Timedelta(seconds=ROLLING_WINDOW_SECONDS)
            window = group[
                (group["timestamp"] >= window_start) & (group["timestamp"] <= row["timestamp"])
            ]
            failed_counts.append((idx, window["auth_success"].eq(0).sum()))
            total_counts.append((idx, len(window)))
            unique_ips.append((idx, window["source_ip"].nunique()))

    failed_df = pd.DataFrame(failed_counts, columns=["idx", "failed_auth_count_1h"]).set_index("idx")
    total_df = pd.DataFrame(total_counts, columns=["idx", "total_events_1h"]).set_index("idx")
    ip_df = pd.DataFrame(unique_ips, columns=["idx", "unique_source_ips_1h"]).set_index("idx")

    df["failed_auth_count_1h"] = failed_df["failed_auth_count_1h"].values
    df["total_events_1h"] = total_df["total_events_1h"].values
    df["unique_source_ips_1h"] = ip_df["unique_source_ips_1h"].values
    df["auth_failure_rate_1h"] = np.where(
        df["total_events_1h"] > 0,
        df["failed_auth_count_1h"] / df["total_events_1h"],
        0,
    )

    # ─── Geographic Features ─────────────────────────────────────────────
    print("[4/7] Extracting geographic features...")
    df["lat"] = df["geo_location"].apply(lambda x: _parse_geo(x)[0])
    df["lon"] = df["geo_location"].apply(lambda x: _parse_geo(x)[1])

    # Previous location for geo-velocity
    df["prev_lat"] = df.groupby("entity_id")["lat"].shift(1)
    df["prev_lon"] = df.groupby("entity_id")["lon"].shift(1)

    # Geo-velocity: speed between consecutive logins
    df["geo_velocity"] = df.apply(
        lambda row: calculate_geo_velocity(
            row["prev_lat"], row["prev_lon"],
            row["lat"], row["lon"],
            row["time_since_last_login"]
        ) if pd.notna(row["prev_lat"]) and row["time_since_last_login"] > 0 else 0.0,
        axis=1,
    )

    # Distance from entity's most common location (home)
    entity_home = df.groupby("entity_id").agg(
        home_lat=("lat", "median"),
        home_lon=("lon", "median"),
    )
    df = df.merge(entity_home, on="entity_id", how="left")
    df["geo_distance_from_home"] = df.apply(
        lambda row: calculate_geo_velocity(
            row["home_lat"], row["home_lon"],
            row["lat"], row["lon"],
            3600  # Normalize to distance in km
        ) if pd.notna(row["home_lat"]) else 0.0,
        axis=1,
    )

    # ─── Resource Access Features ────────────────────────────────────────
    print("[5/7] Extracting resource access features...")

    # Unique resources accessed in rolling window per entity
    resource_counts = []
    for entity_id, group in df.groupby("entity_id"):
        group = group.sort_values("timestamp")
        for idx, row in group.iterrows():
            window_start = row["timestamp"] - pd.Timedelta(seconds=ROLLING_WINDOW_SECONDS)
            window = group[
                (group["timestamp"] >= window_start) & (group["timestamp"] <= row["timestamp"])
            ]
            resource_counts.append((idx, window["resource_accessed"].nunique()))

    res_df = pd.DataFrame(resource_counts, columns=["idx", "unique_resources_1h"]).set_index("idx")
    df["unique_resources_1h"] = res_df["unique_resources_1h"].values

    # Resource diversity ratio
    entity_resource_counts = df.groupby("entity_id")["resource_accessed"].transform("nunique")
    entity_total_events = df.groupby("entity_id")["resource_accessed"].transform("count")
    df["resource_diversity_ratio"] = entity_resource_counts / entity_total_events

    # New resource flag: did the entity access a resource they haven't used before?
    entity_resource_history = {}
    new_resource_flags = []
    for idx, row in df.iterrows():
        eid = row["entity_id"]
        res = row["resource_accessed"]
        if eid not in entity_resource_history:
            entity_resource_history[eid] = set()
        new_resource_flags.append(1 if res not in entity_resource_history[eid] else 0)
        entity_resource_history[eid].add(res)

    df["new_resource_flag"] = new_resource_flags

    # ─── Device Features ─────────────────────────────────────────────────
    print("[6/7] Extracting device features...")

    # Track fingerprint changes per entity
    df["prev_fingerprint"] = df.groupby("entity_id")["device_fingerprint"].shift(1)
    df["fingerprint_changed"] = (
        (df["device_fingerprint"] != df["prev_fingerprint"]) &
        (df["prev_fingerprint"].notna())
    ).astype(int)

    # New device flag
    entity_device_history = {}
    new_device_flags = []
    for idx, row in df.iterrows():
        eid = row["entity_id"]
        fp = row["device_fingerprint"]
        if eid not in entity_device_history:
            entity_device_history[eid] = set()
        new_device_flags.append(1 if fp not in entity_device_history[eid] else 0)
        entity_device_history[eid].add(fp)

    df["new_device_flag"] = new_device_flags

    # ─── Entity Profile Features ─────────────────────────────────────────
    print("[7/7] Extracting entity profile features...")

    # Session duration z-score (per entity)
    entity_session_stats = df.groupby("entity_id")["session_duration"].agg(["mean", "std"])
    df = df.merge(entity_session_stats, on="entity_id", how="left", suffixes=("", "_stats"))
    df["session_duration_zscore"] = np.where(
        df["std"] > 0,
        (df["session_duration"] - df["mean"]) / df["std"],
        0,
    )

    # Command sequence length
    df["command_sequence_length"] = df["command_sequence"].apply(
        lambda x: len(json.loads(x)) if pd.notna(x) and x else 0
    )

    # Days since first seen (cold-start indicator)
    entity_first_seen = df.groupby("entity_id")["timestamp"].min().rename("first_seen")
    df = df.merge(entity_first_seen, on="entity_id", how="left")
    df["days_since_first_seen"] = (df["timestamp"] - df["first_seen"]).dt.days

    # Login frequency deviation
    entity_daily_counts = df.groupby(["entity_id", df["timestamp"].dt.date]).size()
    entity_avg_daily = entity_daily_counts.groupby(level=0).mean().rename("avg_daily_logins")
    entity_std_daily = entity_daily_counts.groupby(level=0).std().rename("std_daily_logins").fillna(0)
    df = df.merge(entity_avg_daily, on="entity_id", how="left")
    df = df.merge(entity_std_daily, on="entity_id", how="left")

    # ─── Select Final Features ───────────────────────────────────────────
    feature_columns = [
        # Temporal
        "hour_of_day", "day_of_week", "is_weekend", "is_off_hours",
        "time_since_last_login",
        # Authentication
        "auth_success", "failed_auth_count_1h", "auth_failure_rate_1h",
        "unique_source_ips_1h", "total_events_1h",
        # Geographic
        "geo_velocity", "geo_distance_from_home",
        # Resource
        "unique_resources_1h", "resource_diversity_ratio", "new_resource_flag",
        # Device
        "fingerprint_changed", "new_device_flag",
        # Session & Profile
        "session_duration", "session_duration_zscore", "command_sequence_length",
        "days_since_first_seen",
    ]

    # Build output DataFrame
    features_df = df[feature_columns].copy()
    features_df["label"] = df["label"]
    features_df["entity_id"] = df["entity_id"]
    features_df["log_id"] = df["log_id"]
    features_df["timestamp"] = df["timestamp"]

    # Fill any remaining NaN values
    features_df[feature_columns] = features_df[feature_columns].fillna(0)

    # Save processed features
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    features_df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"\n{'=' * 60}")
    print(f"  FEATURE ENGINEERING COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n  Total features: {len(feature_columns)}")
    print(f"  Total records:  {len(features_df)}")
    print(f"  Saved to:       {PROCESSED_DATA_PATH}")
    print(f"\n  Feature list:")
    for i, col in enumerate(feature_columns, 1):
        print(f"    {i:2d}. {col}")

    return features_df


if __name__ == "__main__":
    extract_features()
