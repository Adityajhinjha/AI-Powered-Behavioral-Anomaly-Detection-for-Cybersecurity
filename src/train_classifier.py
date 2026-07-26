import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, roc_auc_score
)
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    PROCESSED_DATA_PATH, XGBOOST_MODEL_PATH, LABEL_ENCODER_PATH,
    MODEL_DIR, ANOMALY_CLASSES,
    XGBOOST_N_ESTIMATORS, XGBOOST_MAX_DEPTH,
    XGBOOST_LEARNING_RATE, XGBOOST_RANDOM_STATE,
)

CLASSIFIER_FEATURES = [
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


def _compute_sample_weights(y_train, le):
    class_counts = np.bincount(y_train)
    total = len(y_train)
    n_classes = len(class_counts)
    weights = total / (n_classes * class_counts)

    sample_weights = np.array([weights[label] for label in y_train])
    return sample_weights


def train(data_path=None):
    print("=" * 60)
    print("  XGBOOST CLASSIFIER TRAINING")
    print("=" * 60)

    path = data_path or PROCESSED_DATA_PATH
    print(f"\n[1/6] Loading features from {path}...")
    df = pd.read_csv(path)
    print(f"  -> {len(df)} records loaded")

    print("\n[2/6] Preparing feature matrix and labels...")
    X = df[CLASSIFIER_FEATURES].values

    le = LabelEncoder()
    le.fit(ANOMALY_CLASSES)
    y = le.transform(df["label"].values)

    print(f"  -> {X.shape[1]} features, {len(le.classes_)} classes")
    print(f"  -> Class distribution:")
    for cls_name, cls_idx in zip(le.classes_, range(len(le.classes_))):
        count = (y == cls_idx).sum()
        pct = count / len(y) * 100
        print(f"      {cls_name:30s}  {count:6d}  ({pct:.2f}%)")

    print("\n[3/6] Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=XGBOOST_RANDOM_STATE, stratify=y
    )
    print(f"  -> Train: {len(X_train)}, Test: {len(X_test)}")

    sample_weights = _compute_sample_weights(y_train, le)

    print("\n[4/6] Training XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=XGBOOST_N_ESTIMATORS,
        max_depth=XGBOOST_MAX_DEPTH,
        learning_rate=XGBOOST_LEARNING_RATE,
        objective="multi:softprob",
        num_class=len(le.classes_),
        random_state=XGBOOST_RANDOM_STATE,
        eval_metric="mlogloss",
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    print("\n[5/6] Evaluating model performance...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True,
    )
    report_str = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
    )

    cm = confusion_matrix(y_test, y_pred)

    macro_f1 = f1_score(y_test, y_pred, average="macro")

    try:
        roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = 0.0

    risk_scores = y_prob.max(axis=1)
    top_1pct_threshold = np.percentile(risk_scores, 99)
    top_1pct_mask = risk_scores >= top_1pct_threshold
    top_1pct_labels = y_test[top_1pct_mask]
    normal_idx = le.transform(["normal"])[0]
    fpr_top1 = (top_1pct_labels == normal_idx).sum() / max(top_1pct_mask.sum(), 1)

    print("\n[6/6] Saving model artifacts...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, XGBOOST_MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"  -> Model saved: {XGBOOST_MODEL_PATH}")
    print(f"  -> Label encoder saved: {LABEL_ENCODER_PATH}")

    print(f"\n{'=' * 60}")
    print(f"  CLASSIFICATION RESULTS")
    print(f"{'=' * 60}")
    print(f"\n{report_str}")
    print(f"\n  Macro F1-Score:         {macro_f1:.4f}")
    print(f"  ROC-AUC (OvR, macro):   {roc_auc:.4f}")
    print(f"  FPR at top 1% budget:   {fpr_top1:.4f}")

    eval_metrics = {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": le.classes_.tolist(),
        "macro_f1": float(macro_f1),
        "roc_auc": float(roc_auc),
        "fpr_top1_pct": float(fpr_top1),
    }

    metrics_path = os.path.join(MODEL_DIR, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"\n  -> Evaluation metrics saved: {metrics_path}")

    return model, le, eval_metrics


if __name__ == "__main__":
    train()
