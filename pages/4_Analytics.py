import os
import sys
import json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.database import get_alert_stats, fetch_alerts


def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


CHART_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8", size=12),
    margin=dict(t=30, b=30, l=10, r=10),
)

PALETTE = ["#06b6d4", "#3b82f6", "#8b5cf6", "#f59e0b", "#ef4444", "#10b981", "#f97316"]


def load_model_metrics():
    """Load metrics JSON from models directory."""
    base = os.path.join(os.path.dirname(__file__), "..", "models")
    for fname in ["eval_metrics.json", "classifier_metrics.json"]:
        metrics_path = os.path.join(base, fname)
        if os.path.exists(metrics_path):
            with open(metrics_path, encoding="utf-8") as f:
                data = json.load(f)

                report = data.get("classification_report", {})
                if report and "accuracy" in report:
                    data.setdefault("accuracy", report["accuracy"])

                if report and "macro avg" in report:
                    data.setdefault("macro_f1", report["macro avg"].get("f1-score"))
                    data["balanced_accuracy"] = report["macro avg"].get("recall")

                if "per_class" not in data and report:
                    data["per_class"] = {
                        k: v for k, v in report.items()
                        if isinstance(v, dict) and k not in ("macro avg", "weighted avg")
                    }

                if "num_classes" not in data and "class_names" in data:
                    data["num_classes"] = len(data["class_names"])

                return data
    return None


def build_shap_global(df):
    """Aggregate SHAP values across all alerts to compute feature importance."""
    agg = {}
    for _, row in df.iterrows():
        shap_raw = row.get("shap_explanation") or row.get("shap_values")
        shap_dict = {}
        if isinstance(shap_raw, str):
            try:
                parsed = json.loads(shap_raw)
                if isinstance(parsed, dict):
                    if "top_features" in parsed:
                        shap_dict = {
                            tf["feature"]: tf["shap_value"]
                            for tf in parsed["top_features"]
                        }
                    else:
                        shap_dict = parsed
            except Exception:
                pass
        elif isinstance(shap_raw, dict):
            if "top_features" in shap_raw:
                shap_dict = {
                    tf["feature"]: tf["shap_value"]
                    for tf in shap_raw["top_features"]
                }
            else:
                shap_dict = shap_raw

        for feat, val in shap_dict.items():
            agg.setdefault(feat, []).append(abs(float(val)))

    if not agg:
        return pd.DataFrame()

    result = pd.DataFrame([
        {"feature": f, "mean_abs_shap": np.mean(vals)}
        for f, vals in agg.items()
    ]).sort_values("mean_abs_shap", ascending=False).head(15)

    result["feature"] = result["feature"].str.replace("_", " ").str.title()
    return result


def main():
    st.set_page_config(page_title="Analytics | CyberGuard SOC", page_icon=None, layout="wide")
    load_css()

    st.markdown(
        """
        <div class="page-header">
            <h1>Model Analytics</h1>
            <p>Performance metrics, evaluation results, and global feature importance</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        stats   = get_alert_stats()
        alerts  = fetch_alerts(risk_threshold=0.0, limit=50000)
        metrics = load_model_metrics()
    except Exception as e:
        st.error(f"Database error: {e}")
        return

    # ── KPI Row ───────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predictions", f"{stats['total_alerts']:,}")
    with col2:
        accuracy = metrics.get("balanced_accuracy", None) if metrics else None
        st.metric("Balanced Accuracy", f"{accuracy:.1%}" if accuracy else "—")
    with col3:
        f1 = metrics.get("macro_f1", None) if metrics else None
        st.metric("Macro F1 Score", f"{f1:.4f}" if f1 else "—")
    with col4:
        classes = metrics.get("num_classes", len(stats["alerts_by_type"])) if metrics else len(stats["alerts_by_type"])
        st.metric("Attack Classes", classes)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Confusion Matrix", "Per-Class Metrics", "SHAP Importance", "Model Config"]
    )

    # ─── Tab 1: Confusion Matrix ──────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-label">Predicted vs Actual Attack Classification</div>', unsafe_allow_html=True)

        if metrics and "confusion_matrix" in metrics:
            cm      = np.array(metrics["confusion_matrix"])
            classes = metrics.get("class_names", [f"Class {i}" for i in range(len(cm))])

            cm_norm = cm.astype(float)
            row_sums = cm_norm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_pct = cm_norm / row_sums

            fig_cm = go.Figure(go.Heatmap(
                z=cm_pct,
                x=[c.replace("_", " ").title() for c in classes],
                y=[c.replace("_", " ").title() for c in classes],
                text=cm,
                texttemplate="%{text}",
                textfont=dict(size=11, color="#f1f5f9"),
                colorscale=[[0, "rgba(17,24,39,1)"], [0.5, "rgba(6,182,212,0.35)"], [1, "#06b6d4"]],
                showscale=False,
            ))
            fig_cm.update_layout(
                **CHART_LAYOUT,
                height=480,
                xaxis=dict(title="Predicted", tickangle=-30, side="bottom", gridcolor="rgba(0,0,0,0)"),
                yaxis=dict(title="Actual", gridcolor="rgba(0,0,0,0)", autorange="reversed"),
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            # Fallback: approximate from alert data
            if len(alerts) > 0:
                type_counts = alerts["predicted_anomaly_type"].value_counts()
                st.info("Confusion matrix not found. Showing class distribution from live predictions.")

                fig_bar = px.bar(
                    type_counts.reset_index(),
                    x="predicted_anomaly_type",
                    y="count",
                    labels={"predicted_anomaly_type": "Attack Type", "count": "Count"},
                    color_discrete_sequence=PALETTE,
                )
                fig_bar.update_layout(**CHART_LAYOUT, height=350)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No predictions available. Run the pipeline first.")

    # ─── Tab 2: Per-Class Metrics ─────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-label">Per-Class Classification Report</div>', unsafe_allow_html=True)

        if metrics and "per_class" in metrics:
            per_class_df = pd.DataFrame(metrics["per_class"]).T
            per_class_df.index = [i.replace("_", " ").title() for i in per_class_df.index]
            per_class_df = per_class_df.rename(columns={
                "precision": "Precision",
                "recall":    "Recall",
                "f1-score":  "F1 Score",
                "support":   "Support",
            })

            class_list = ["All Classes"] + list(per_class_df.index)
            selected_class = st.selectbox(
                "Filter / Search Attack Class",
                class_list,
                key="analytics_class_filter",
            )
            if selected_class != "All Classes":
                disp_df = per_class_df.loc[[selected_class]]
            else:
                disp_df = per_class_df

            col_l, col_r = st.columns([3, 2])

            with col_l:
                st.dataframe(
                    disp_df[["Precision", "Recall", "F1 Score", "Support"]].round(4),
                    use_container_width=True,
                )

            with col_r:
                st.markdown('<div class="section-label">F1 Score by Class</div>', unsafe_allow_html=True)
                f1_series = disp_df["F1 Score"].dropna()
                fig_f1 = px.bar(
                    x=f1_series.index.tolist(),
                    y=f1_series.values.tolist(),
                    labels={"x": "Class", "y": "F1 Score"},
                    color=f1_series.values.tolist(),
                    color_continuous_scale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#10b981"]],
                )
                fig_f1.update_layout(**CHART_LAYOUT, height=350, coloraxis_showscale=False)
                fig_f1.update_layout(
                    xaxis=dict(tickangle=-35, gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(range=[0, 1.05], gridcolor="rgba(255,255,255,0.05)"),
                )
                st.plotly_chart(fig_f1, use_container_width=True)

        else:
            st.info("Per-class metrics not available. Ensure `classifier_metrics.json` is generated during training.")

    # ─── Tab 3: SHAP Global Importance ───────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-label">Global Feature Importance (SHAP)</div>', unsafe_allow_html=True)

        threat_shap_options = [
            "All Threat Types",
            "Brute Force",
            "Credential Stuffing",
            "Device Spoofing",
            "Impossible Travel",
            "Insider Drift",
            "Lateral Movement",
            "Low And Slow Exfiltration"
        ]
        sel_shap_threat = st.selectbox(
            "Filter Threat Class for SHAP Importance",
            threat_shap_options,
            key="analytics_shap_threat_filter",
        )
        filtered_alerts_for_shap = alerts
        if sel_shap_threat != "All Threat Types":
            norm = sel_shap_threat.lower().replace(" ", "_")
            filtered_alerts_for_shap = alerts[alerts["predicted_anomaly_type"].str.lower() == norm]

        with st.spinner("Aggregating SHAP values..."):
            shap_df = build_shap_global(filtered_alerts_for_shap)

        if len(shap_df) > 0:
            fig_shap = px.bar(
                shap_df,
                x="mean_abs_shap",
                y="feature",
                orientation="h",
                labels={"mean_abs_shap": "Mean |SHAP|", "feature": ""},
                color="mean_abs_shap",
                color_continuous_scale=[[0, "rgba(6,182,212,0.3)"], [1, "#06b6d4"]],
            )
            fig_shap.update_layout(
                **CHART_LAYOUT,
                height=520,
                coloraxis_showscale=False,
                yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            st.markdown('<div class="section-label">Interpretation</div>', unsafe_allow_html=True)
            st.markdown(
                """
                - Higher **Mean |SHAP|** indicates that feature has stronger average influence on predictions
                - Features shown are aggregated across all alerts; individual alert SHAP values are shown in the Live Alerts page
                - Red bars (on Live Alerts) increase anomaly probability; blue bars decrease it
                """
            )
        else:
            st.info("No SHAP values found in the database. Ensure the prediction pipeline ran with SHAP enabled.")

    # ─── Tab 4: Model Config ──────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-label">Model Configuration</div>', unsafe_allow_html=True)
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown(
                """
                <div class="card">
                    <div class="card-label">Anomaly Detector</div>
                    <table style="width:100%;font-size:13px;border-collapse:collapse;">
                        <tr><td style="color:#94a3b8;padding:5px 0;">Algorithm</td><td style="color:#f1f5f9;text-align:right;">Isolation Forest</td></tr>
                        <tr><td style="color:#94a3b8;padding:5px 0;">Estimators</td><td style="color:#f1f5f9;text-align:right;">200</td></tr>
                        <tr><td style="color:#94a3b8;padding:5px 0;">Contamination</td><td style="color:#f1f5f9;text-align:right;">5%</td></tr>
                        <tr><td style="color:#94a3b8;padding:5px 0;">Max Samples</td><td style="color:#f1f5f9;text-align:right;">auto</td></tr>
                        <tr><td style="color:#94a3b8;padding:5px 0;">Bootstrap</td><td style="color:#f1f5f9;text-align:right;">False</td></tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_r:
            st.markdown(
                """
                <div class="card">
                    <div class="card-label">Attack Classifier</div>
                    <table style="width:100%;font-size:13px;border-collapse:collapse;">
                        <tr><td style="color:#94a3b8;padding:5px 0;">Algorithm</td><td style="color:#f1f5f9;text-align:right;">Random Forest + XGBoost</td></tr>
                        <tr><td style="color:#94a3b8;padding:5px 0;">Estimators</td><td style="color:#f1f5f9;text-align:right;">300 / 200</td></tr>
                        <tr><td style="color:#94a3b8;padding:5px 0;">Explainability</td><td style="color:#f1f5f9;text-align:right;">SHAP TreeExplainer</td></tr>
                        <tr><td style="color:#94a3b8;padding:5px 0;">Classes</td><td style="color:#f1f5f9;text-align:right;">7 attack types</td></tr>
                        <tr><td style="color:#94a3b8;padding:5px 0;">Features</td><td style="color:#f1f5f9;text-align:right;">33 engineered</td></tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
else:
    main()
