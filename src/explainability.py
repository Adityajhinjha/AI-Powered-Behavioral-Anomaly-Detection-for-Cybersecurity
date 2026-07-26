import os
import sys
import json
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TOP_SHAP_FEATURES
from src.utils import format_shap_explanation


def _compute_shap_values(model, X):
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        return explainer.shap_values(X)
    except Exception:
        dmatrix = xgb.DMatrix(X)
        contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
        if len(contribs.shape) == 3:
            n_classes = contribs.shape[1]
            return [contribs[:, c, :-1] for c in range(n_classes)]
        else:
            return contribs[:, :-1]


def generate_explanations(model, X, feature_names):
    shap_values = _compute_shap_values(model, X)

    explanations = []
    for i in range(len(X)):
        if isinstance(shap_values, list):
            pred_probs = model.predict_proba(X[i:i+1])[0]
            pred_class = np.argmax(pred_probs)
            event_shap = shap_values[pred_class][i]
        else:
            event_shap = shap_values[i]

        top_indices = np.argsort(np.abs(event_shap))[::-1][:TOP_SHAP_FEATURES]

        explanation = {
            "top_features": [
                {
                    "feature": feature_names[idx],
                    "shap_value": round(float(event_shap[idx]), 4),
                    "feature_value": round(float(X[i][idx]), 4),
                }
                for idx in top_indices
                if abs(event_shap[idx]) > 0.001
            ],
            "summary": format_shap_explanation(
                feature_names, event_shap, TOP_SHAP_FEATURES
            ),
        }

        explanations.append(json.dumps(explanation))

    return explanations


def create_waterfall_chart(shap_explanation_json, title="Alert Explanation"):
    try:
        explanation = json.loads(shap_explanation_json)
    except (json.JSONDecodeError, TypeError):
        return _empty_chart("No explanation available")

    features = explanation.get("top_features", [])
    if not features:
        return _empty_chart("No significant features")

    names = [f["feature"] for f in features]
    values = [f["shap_value"] for f in features]
    feature_vals = [f["feature_value"] for f in features]

    colors = ["#ff4444" if v > 0 else "#4488ff" for v in values]
    labels = [f"{n}\n(val: {fv:.2f})" for n, fv in zip(names, feature_vals)]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="SHAP Value (Impact on Prediction)",
        yaxis_title="",
        height=max(300, len(features) * 60),
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )

    return fig


def create_summary_plot(model, X, feature_names, max_display=15):
    shap_values = _compute_shap_values(model, X)

    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    sorted_idx = np.argsort(mean_abs_shap)[::-1][:max_display]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_values = mean_abs_shap[sorted_idx]

    sorted_names = sorted_names[::-1]
    sorted_values = sorted_values[::-1]

    fig = go.Figure(go.Bar(
        x=sorted_values,
        y=sorted_names,
        orientation="h",
        marker=dict(
            color=sorted_values,
            colorscale="Viridis",
        ),
        text=[f"{v:.4f}" for v in sorted_values],
        textposition="outside",
    ))

    fig.update_layout(
        title=dict(text="Global Feature Importance (Mean |SHAP|)", font=dict(size=16)),
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="",
        height=max(400, max_display * 35),
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )

    return fig


def _empty_chart(message):
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False, font=dict(size=16, color="gray"),
    )
    fig.update_layout(
        height=200,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
