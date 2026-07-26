import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "synthetic_logs.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "features.csv")

DATABASE_PATH = os.path.join(BASE_DIR, "database", "anomaly_detector.db")

MODEL_DIR = os.path.join(BASE_DIR, "models")
ISOLATION_FOREST_PATH = os.path.join(MODEL_DIR, "isolation_forest.joblib")
XGBOOST_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_classifier.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

NUM_USERS = 500
NUM_SERVICE_ACCOUNTS = 50
NUM_EDGE_DEVICES = 30
TOTAL_EVENTS = 100_000
ANOMALY_RATE = 0.02
DATA_WINDOW_DAYS = 90

ATTACK_DISTRIBUTION = {
    "brute_force": 0.20,
    "impossible_travel": 0.15,
    "credential_stuffing": 0.20,
    "lateral_movement": 0.15,
    "device_spoofing": 0.15,
    "low_and_slow_exfiltration": 0.10,
    "insider_drift": 0.05,
}

OFF_HOURS_START = 22
OFF_HOURS_END = 6
ROLLING_WINDOW_SECONDS = 3600

ISOLATION_FOREST_CONTAMINATION = 0.02
ISOLATION_FOREST_N_ESTIMATORS = 200
ISOLATION_FOREST_RANDOM_STATE = 42

XGBOOST_N_ESTIMATORS = 300
XGBOOST_MAX_DEPTH = 6
XGBOOST_LEARNING_RATE = 0.1
XGBOOST_RANDOM_STATE = 42

RISK_SCORE_THRESHOLD = 0.70
COLD_START_DAYS = 7
TOP_SHAP_FEATURES = 5

ANOMALY_CLASSES = [
    "normal",
    "brute_force",
    "impossible_travel",
    "credential_stuffing",
    "lateral_movement",
    "device_spoofing",
    "low_and_slow_exfiltration",
    "insider_drift",
]
