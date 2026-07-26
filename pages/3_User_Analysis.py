import os
import sys
import json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import src.database as db
from src.database import (
    fetch_alerts, fetch_entity_history, fetch_entity_alerts, get_unique_entity_ids
)


def get_flagged_entities(threat_type=None):
    conn = db.get_connection()
    if threat_type and threat_type != "All Threat Types":
        db_threat = threat_type.lower().replace(" ", "_")
        query = (
            "SELECT entity_id, COUNT(*) as alert_count, MAX(risk_score) as max_risk, "
            "GROUP_CONCAT(DISTINCT predicted_anomaly_type) as attack_types "
            "FROM security_alerts WHERE predicted_anomaly_type = ? "
            "GROUP BY entity_id ORDER BY max_risk DESC, alert_count DESC"
        )
        df = pd.read_sql_query(query, conn, params=(db_threat,))
    else:
        query = (
            "SELECT entity_id, COUNT(*) as alert_count, MAX(risk_score) as max_risk, "
            "GROUP_CONCAT(DISTINCT predicted_anomaly_type) as attack_types "
            "FROM security_alerts GROUP BY entity_id ORDER BY max_risk DESC, alert_count DESC"
        )
        df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


CHART_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showline=False, zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", showline=False, zeroline=False),
    margin=dict(t=30, b=30, l=10, r=10),
)

PALETTE = ["#06b6d4", "#3b82f6", "#8b5cf6", "#f59e0b", "#ef4444", "#10b981", "#f97316"]


def main():
    st.set_page_config(page_title="User Analysis | CyberGuard SOC", page_icon=None, layout="wide")
    load_css()

    st.markdown(
        """
        <div class="page-header">
            <h1>User &amp; Entity Analysis</h1>
            <p>Investigate individual entity behavior, access history, and associated alerts</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Main Screen Entity Search & Flagged Cases Selection ───────────────────
    try:
        all_entities = get_unique_entity_ids()
    except Exception as e:
        st.error(f"Database error: {e}")
        return

    if not all_entities:
        st.info("No data found. Run the pipeline first.")
        return

    st.markdown('<div class="section-label">Entity Selection &amp; Search</div>', unsafe_allow_html=True)
    col_flagged, col_all = st.columns(2)

    selected_flagged_id = None
    selected_all_id = None

    with col_flagged:
        st.markdown('<div class="card-label">Flagged Cases (Filter by Threat Category)</div>', unsafe_allow_html=True)

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
        selected_threat = st.selectbox(
            "Filter Threat Category",
            threat_options,
            key="select_threat_type",
        )

        flagged_search = st.text_input(
            "Search Flagged Entity",
            placeholder="Type user_042 or risk...",
            key="flagged_search_input",
        )

        flagged_df = get_flagged_entities(selected_threat)

        if len(flagged_df) > 0:
            flagged_options = {}
            for _, row in flagged_df.iterrows():
                eid = str(row["entity_id"])
                acount = int(row["alert_count"])
                mrisk = float(row["max_risk"])
                label_text = f"{eid}  ·  {acount} Alert{'s' if acount > 1 else ''}  ·  {mrisk:.3f} Risk"
                if not flagged_search.strip() or flagged_search.strip().lower() in label_text.lower():
                    flagged_options[label_text] = eid

            if flagged_options:
                flagged_choice = st.selectbox(
                    "Select Flagged Entity",
                    list(flagged_options.keys()),
                    key="select_flagged",
                )
                selected_flagged_id = flagged_options[flagged_choice]
            else:
                st.info(f"No flagged entities matching '{flagged_search}'.")
        else:
            st.info(f"No flagged cases found for '{selected_threat}'.")

    with col_all:
        st.markdown('<div class="card-label">All Entities Search</div>', unsafe_allow_html=True)
        search_input = st.text_input(
            "Filter Entity ID",
            placeholder="Type user_042 or device_...",
            label_visibility="collapsed",
            key="search_input"
        )
        filtered = [e for e in all_entities if search_input.lower() in e.lower()] if search_input else all_entities
        selected_all_id = st.selectbox(
            "Select Any Entity",
            filtered[:200],
            key="select_all",
            label_visibility="collapsed",
        )

    mode = st.radio(
        "Selection Mode",
        ["Flagged Cases (High Risk)", "All Entities Search"],
        horizontal=True,
        key="entity_mode",
    )

    if mode == "Flagged Cases (High Risk)" and selected_flagged_id:
        selected_entity = selected_flagged_id
    else:
        selected_entity = selected_all_id

    if not selected_entity:
        st.info("Select an entity to begin investigation.")
        return

    # ── Filter data for entity ────────────────────────────────────────────────
    try:
        entity_logs = fetch_entity_history(selected_entity)
        entity_logs["timestamp"] = pd.to_datetime(entity_logs["timestamp"])
    except Exception as e:
        st.error(f"Error loading entity history: {e}")
        return

    try:
        entity_alerts = fetch_entity_alerts(selected_entity)
    except Exception:
        entity_alerts = pd.DataFrame()

    # ── Entity summary header ─────────────────────────────────────────────────
    total_events = len(entity_logs)
    total_alerts = len(entity_alerts)
    critical_alerts = len(entity_alerts[entity_alerts["risk_score"] >= 0.90]) if len(entity_alerts) > 0 else 0
    if total_alerts > 0 and "predicted_anomaly_type" in entity_alerts.columns:
        primary_threat = (
            entity_alerts["predicted_anomaly_type"]
            .value_counts()
            .index[0]
            .replace("_", " ")
            .title()
        )
    else:
        primary_threat = "None (Clean)"

    date_range = ""
    if total_events > 0:
        date_range = f"{entity_logs['timestamp'].min().date()} — {entity_logs['timestamp'].max().date()}"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events", f"{total_events:,}")
    with col2:
        st.metric("Security Alerts", total_alerts)
    with col3:
        st.metric("Critical Alerts", critical_alerts)
    with col4:
        st.metric("Primary Threat Vector", primary_threat)

    st.markdown(
        f'<div style="font-size:11px;color:#475569;margin-top:-8px;margin-bottom:20px;font-family:\'JetBrains Mono\',monospace;">'
        f'Entity: <span style="color:#06b6d4;">{selected_entity}</span>'
        + (f" &nbsp; · &nbsp; Date Range: {date_range}" if date_range else "")
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Activity Timeline ─────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Activity Timeline</div>', unsafe_allow_html=True)
    entity_logs["date"] = entity_logs["timestamp"].dt.date
    daily_events = entity_logs.groupby("date").size().reset_index(name="events")

    fig_timeline = px.bar(
        daily_events, x="date", y="events",
        labels={"date": "Date", "events": "Events"},
        color_discrete_sequence=["#06b6d4"],
    )

    # Overlay alerts as scatter
    if len(entity_alerts) > 0:
        entity_alerts["timestamp"] = pd.to_datetime(entity_alerts["timestamp"])
        entity_alerts["date_only"] = entity_alerts["timestamp"].dt.date
        daily_alerts = entity_alerts.groupby("date_only").size().reset_index(name="count")

        fig_timeline.add_scatter(
            x=daily_alerts["date_only"],
            y=daily_alerts["count"],
            mode="markers",
            marker=dict(color="#ef4444", size=8, symbol="diamond"),
            name="Alerts",
        )

    fig_timeline.update_layout(**CHART_LAYOUT, height=280)
    fig_timeline.update_layout(legend=dict(
        orientation="h", y=-0.3, x=0.5, xanchor="center",
        font=dict(size=11), bgcolor="rgba(0,0,0,0)",
    ))
    st.plotly_chart(fig_timeline, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Behavior Profile ──────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-label">Hourly Access Heatmap</div>', unsafe_allow_html=True)
        if total_events > 0:
            entity_logs["hour"]    = entity_logs["timestamp"].dt.hour
            entity_logs["weekday"] = entity_logs["timestamp"].dt.day_name()
            heatmap_data = entity_logs.groupby(["weekday", "hour"]).size().reset_index(name="count")

            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            heatmap_data["weekday"] = pd.Categorical(heatmap_data["weekday"], categories=day_order, ordered=True)
            heatmap_data = heatmap_data.sort_values("weekday")

            pivot = heatmap_data.pivot(index="weekday", columns="hour", values="count").fillna(0)

            fig_heat = go.Figure(go.Heatmap(
                z=pivot.values,
                x=list(range(24)),
                y=list(pivot.index),
                colorscale=[[0, "rgba(6,182,212,0.0)"], [0.5, "rgba(6,182,212,0.4)"], [1, "#06b6d4"]],
                showscale=False,
            ))
            fig_heat.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#94a3b8", size=11),
                height=240,
                margin=dict(t=10, b=30, l=80, r=10),
                xaxis=dict(title="Hour of Day", tickmode="linear", tick0=0, dtick=3, gridcolor="rgba(0,0,0,0)"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-label">Alert Type Breakdown</div>', unsafe_allow_html=True)
        if len(entity_alerts) > 0:
            type_counts = (
                entity_alerts["predicted_anomaly_type"]
                .value_counts()
                .reset_index()
            )
            type_counts.columns = ["type", "count"]
            type_counts["type"] = type_counts["type"].str.replace("_", " ").str.title()

            fig_bar = px.bar(
                type_counts, x="count", y="type",
                orientation="h",
                labels={"count": "Alerts", "type": ""},
                color_discrete_sequence=PALETTE,
                color="type",
            )
            fig_bar.update_layout(**CHART_LAYOUT, height=240, showlegend=False)
            fig_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No alerts recorded for this entity.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Alert History Table ───────────────────────────────────────────────────
    st.markdown('<div class="section-label">Alert History</div>', unsafe_allow_html=True)
    if len(entity_alerts) > 0:
        display = entity_alerts[
            ["timestamp", "risk_score", "predicted_anomaly_type", "status"]
        ].copy()
        display.columns = ["Timestamp", "Risk Score", "Attack Type", "Status"]
        display["Attack Type"] = display["Attack Type"].str.replace("_", " ").str.title()
        display["Risk Score"]  = display["Risk Score"].round(4)
        display = display.sort_values("Risk Score", ascending=False)
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.info("No alerts recorded for this entity.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Raw Log Sample ────────────────────────────────────────────────────────
    with st.expander("Raw Log Sample — Last 20 Events"):
        recent = entity_logs.sort_values("timestamp", ascending=False).head(20)
        st.dataframe(recent, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
else:
    main()
