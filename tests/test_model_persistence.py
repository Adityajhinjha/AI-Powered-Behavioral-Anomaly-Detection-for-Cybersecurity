import os
import sys
import json
import joblib
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    ISOLATION_FOREST_PATH, XGBOOST_MODEL_PATH, SCALER_PATH,
    LABEL_ENCODER_PATH, MODEL_DIR
)
from src.predict import FEATURE_COLUMNS, load_models


def test_model_artifact_persistence():
    assert os.path.exists(ISOLATION_FOREST_PATH)
    assert os.path.exists(XGBOOST_MODEL_PATH)
    assert os.path.exists(SCALER_PATH)
    assert os.path.exists(LABEL_ENCODER_PATH)


def test_load_models_inference():
    iso_forest, xgb_model, scaler, le = load_models()

    assert iso_forest is not None
    assert xgb_model is not None
    assert scaler is not None
    assert le is not None

    dummy_X = np.zeros((1, len(FEATURE_COLUMNS)))
    dummy_scaled = scaler.transform(dummy_X)

    if_scores = iso_forest.decision_function(dummy_scaled)
    assert len(if_scores) == 1

    xgb_preds = xgb_model.predict(dummy_X)
    assert len(xgb_preds) == 1


def test_eval_metrics_json():
    metrics_path = os.path.join(MODEL_DIR, "eval_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "classification_report" in data
        assert "confusion_matrix" in data
        assert "class_names" in data
