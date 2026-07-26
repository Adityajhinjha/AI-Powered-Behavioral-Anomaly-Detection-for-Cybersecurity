import os
import sys
import json
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.database import fetch_alerts, update_alert_status


def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


RISK_BAND = {
    "Critical (>0.90)": 0.90,
    "High    (>0.70)":  0.70,
    "Medium  (>0.50)":  0.50,
    "All Alerts":       0.00,
}

STATUS_BADGE = {
    "New":           "badge-red",
    "Investigating": "badge-amber",
    "Closed":        "badge-green",
    "False Positive":"badge-purple",
}


def risk_badge(score: float) -> str:
    if score >= 0.90:
        return '<span class="badge badge-red">CRITICAL</span>'
    if score >= 0.70:
        return '<span class="badge badge-amber">HIGH</span>'
    if score >= 0.50:
        return '<span class="badge badge-blue">MEDIUM</span>'
    return '<span class="badge badge-muted">LOW</span>'


def main():
    st.set_page_config(page_title="Live Alerts | CyberGuard SOC", page_icon=None, layout="wide")
    load_css()

    st.markdown(
        """
        <div class="page-header">
            <h1>Live Alert Queue</h1>
            <p>Ranked security alerts for analyst triage — sorted by risk score descending</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Main Screen Filters ───────────────────────────────────────────────────
    st.markdown('<div class="section-label">Alert Filters &amp; Queue Controls</div>', unsafe_allow_html=True)
    fcol1, fcol2, fcol3, fcol4 = st.columns([1.2, 1.4, 1.6, 1.4])

    threat_options = [
        "All Threat Types",
        "Brute Force",
        "Credential Stuffing",
        "Device Spoofing",
        "Impossible Travel",
        "Insider Drift",
        "Lateral Movement",
        "Low And Slow Exfiltration"
    ]

    with fcol1:
        risk_label = st.selectbox(
            "Risk Threshold",
            list(RISK_BAND.keys()),
            index=1,
            key="main_risk_filter",
        )
        threshold = RISK_BAND[risk_label]

    with fcol2:
        sel_threat = st.selectbox(
            "Threat Category",
            threat_options,
            index=0,
            key="main_threat_filter",
        )

    with fcol3:
        status_filter = st.multiselect(
            "Status Filter",
            ["New", "Investigating", "Closed", "False Positive"],
            default=["New", "Investigating"],
            key="main_status_filter",
        )

    with fcol4:
        search_query = st.text_input(
            "Search Entity / Alert ID",
            placeholder="Search user_042 or alert ID...",
            key="main_alert_search",
        )

    max_results = st.select_slider(
        "Max Results",
        options=[25, 50, 100, 200, 500],
        value=100,
        key="main_max_results",
    )

    # ── Fetch data ────────────────────────────────────────────────────────────
    try:
        db_threat = None if sel_threat == "All Threat Types" else sel_threat.lower().replace(" ", "_")
        df = fetch_alerts(risk_threshold=threshold, anomaly_type=db_threat, limit=max_results)
    except Exception as e:
        st.error(f"Database error: {e}")
        return

    if status_filter:
        df = df[df["status"].isin(status_filter)]

    if search_query.strip():
        sq = search_query.strip().lower()
        df = df[
            df["entity_id"].astype(str).str.lower().str.contains(sq) |
            df["alert_id"].astype(str).str.lower().str.contains(sq) |
            df["predicted_anomaly_type"].astype(str).str.lower().str.contains(sq)
        ]

    if len(df) == 0:
        st.info("No alerts match the current filters.")
        return

    # ── Summary bar ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Alerts Shown", len(df))
    with col2:
        critical_count = len(df[df["risk_score"] >= 0.90])
        st.metric("Critical", critical_count)
    with col3:
        new_count = len(df[df["status"] == "New"])
        st.metric("Unreviewed", new_count)
    with col4:
        entities = df["entity_id"].nunique() if "entity_id" in df.columns else 0
        st.metric("Entities at Risk", f"{entities:,}")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Alert Queue</div>', unsafe_allow_html=True)

    # ── Alert Queue ───────────────────────────────────────────────────────────
    for _, alert in df.iterrows():
        alert_id   = str(alert.get("alert_id", ""))
        entity_id  = alert.get("entity_id", "—")
        risk_score = float(alert.get("risk_score", 0))
        att_type   = str(alert.get("predicted_anomaly_type", "Unknown")).replace("_", " ").title()
        ts         = str(alert.get("timestamp", ""))[:19]
        status     = alert.get("status", "New")
        status_cls = STATUS_BADGE.get(status, "badge-muted")

        shap_raw = alert.get("shap_explanation") or alert.get("shap_values")
        shap_data = {}
        if isinstance(shap_raw, str):
            try:
                parsed = json.loads(shap_raw)
                if isinstance(parsed, dict):
                    if "top_features" in parsed:
                        shap_data = {
                            tf["feature"]: tf["shap_value"]
                            for tf in parsed["top_features"]
                        }
                    else:
                        shap_data = parsed
            except Exception:
                pass
        elif isinstance(shap_raw, dict):
            if "top_features" in shap_raw:
                shap_data = {
                    tf["feature"]: tf["shap_value"]
                    for tf in shap_raw["top_features"]
                }
            else:
                shap_data = shap_raw

        short_id = alert_id[:8] if len(alert_id) >= 8 else alert_id
        label = f"ID {short_id}  ·  {entity_id}  ·  {att_type}  ·  Risk {risk_score:.3f}  ·  {ts}"

        with st.expander(label):
            c1, c2, c3 = st.columns([1, 1, 2])

            with c1:
                st.markdown('<div class="card-label">Status</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<span class="badge {status_cls}">{status}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("")
                new_status = st.selectbox(
                    "Update Status",
                    ["New", "Investigating", "Closed", "False Positive"],
                    index=["New", "Investigating", "Closed", "False Positive"].index(status),
                    key=f"sel_{alert_id}",
                    label_visibility="collapsed",
                )
                if st.button("Save", key=f"btn_{alert_id}"):
                    try:
                        update_alert_status(alert_id, new_status)
                        st.success("Updated")
                        st.rerun()
                    except Exception as ex:
                        st.error(str(ex))

            with c2:
                st.markdown('<div class="card-label">Risk Score</div>', unsafe_allow_html=True)
                st.markdown(risk_badge(risk_score), unsafe_allow_html=True)
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:1.8rem;'
                    f'font-weight:700;color:#f1f5f9;margin-top:8px;">{risk_score:.4f}</div>',
                    unsafe_allow_html=True,
                )

            with c3:
                st.markdown('<div class="card-label">SHAP Feature Attribution</div>', unsafe_allow_html=True)
                if shap_data:
                    top = sorted(shap_data.items(), key=lambda x: abs(float(x[1])), reverse=True)[:8]
                    feat_names = [f.replace("_", " ").title() for f, _ in top]
                    shap_vals  = [float(v) for _, v in top]

                    import plotly.graph_objects as go
                    colors = ["#ef4444" if v > 0 else "#06b6d4" for v in shap_vals]
                    fig = go.Figure(go.Bar(
                        x=shap_vals,
                        y=feat_names,
                        orientation="h",
                        marker_color=colors,
                        text=[f"{v:+.3f}" for v in shap_vals],
                        textposition="outside",
                        textfont=dict(size=10),
                    ))
                    fig.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter", color="#94a3b8", size=11),
                        height=220,
                        margin=dict(t=10, b=10, l=5, r=40),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=True,
                                   zerolinecolor="rgba(255,255,255,0.15)"),
                        yaxis=dict(gridcolor="rgba(0,0,0,0)", autorange="reversed"),
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"shap_chart_{alert_id}")
                else:
                    st.caption("SHAP values not available for this alert.")


if __name__ == "__main__":
    main()
else:
    main()
