import os
import sys
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

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
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showline=False, zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", showline=False, zeroline=False),
    margin=dict(t=30, b=30, l=10, r=10),
)

PALETTE = ["#06b6d4", "#3b82f6", "#8b5cf6", "#f59e0b", "#ef4444", "#10b981", "#f97316"]


def main():
    st.set_page_config(page_title="Dashboard | CyberGuard SOC", page_icon=None, layout="wide")
    load_css()

    # Header
    st.markdown(
        """
        <div class="page-header">
            <h1>Security Operations Dashboard</h1>
            <p>Real-time overview of system-wide security posture and threat landscape</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        stats = get_alert_stats()
    except Exception as e:
        st.error(f"Database not initialized. Run the data pipeline first.\n\nError: {e}")
        st.info(
            "Run: `python -m src.data_generator` → `python -m src.feature_engineering` "
            "→ `python -m src.train_detector` → `python -m src.train_classifier` → `python -m src.predict`"
        )
        return

    # ── KPI Row ───────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    card_style = (
        "background:#111827;border:1px solid rgba(255,255,255,0.07);border-radius:10px;"
        "padding:16px 18px;height:100px;display:flex;flex-direction:column;justify-content:center;"
    )

    with col1:
        st.markdown(
            f"""
            <div style="{card_style}">
                <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;">Total Access Events</div>
                <div style="font-size:1.4rem;font-weight:800;color:#f1f5f9;margin-top:4px;">{stats['total_logs']:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div style="{card_style}">
                <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;">Security Alerts</div>
                <div style="font-size:1.4rem;font-weight:800;color:#f1f5f9;margin-top:4px;">{stats['total_alerts']:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div style="{card_style}">
                <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;">Alert Rate</div>
                <div style="font-size:1.4rem;font-weight:800;color:#06b6d4;margin-top:4px;">{stats['alert_rate']:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        if "alerts_by_type" in stats and len(stats["alerts_by_type"]) > 0:
            type_counts = stats["alerts_by_type"]
            max_c = type_counts["count"].max()
            top_cats = type_counts[type_counts["count"] >= max_c * 0.95]["predicted_anomaly_type"].tolist()
            items = [t.replace("_", " ").title() for t in top_cats]
        else:
            top_raw = stats.get("top_anomaly_type", "N/A")
            items = [t.replace("_", " ").title() for t in top_raw.split(" / ")]

        stacked_html = "".join([f'<div style="font-size:0.88rem;font-weight:700;color:#f1f5f9;line-height:1.25;margin-top:2px;">{item}</div>' for item in items])

        st.markdown(
            f"""
            <div style="{card_style}">
                <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;">Top Threat Type</div>
                {stacked_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Row 1: Timeline + Donut ───────────────────────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown('<div class="section-label">Alert Volume Over Time</div>', unsafe_allow_html=True)
        alerts_df = fetch_alerts(risk_threshold=0.0, limit=10000)

        if len(alerts_df) > 0:
            alerts_df["timestamp"] = pd.to_datetime(alerts_df["timestamp"])
            alerts_df["date"] = alerts_df["timestamp"].dt.date
            daily = alerts_df.groupby(["date", "predicted_anomaly_type"]).size().reset_index(name="count")

            fig = px.area(
                daily,
                x="date", y="count",
                color="predicted_anomaly_type",
                labels={"date": "Date", "count": "Alerts", "predicted_anomaly_type": "Threat"},
                color_discrete_sequence=PALETTE,
            )
            fig.update_layout(**CHART_LAYOUT, height=380)
            fig.update_layout(
                legend=dict(
                    orientation="h", y=-0.25, x=0.5, xanchor="center",
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)",
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No alert data. Run the prediction pipeline first.")

    with col_right:
        st.markdown('<div class="section-label">Attack Distribution</div>', unsafe_allow_html=True)
        if len(stats["alerts_by_type"]) > 0:
            type_df = stats["alerts_by_type"].copy()
            type_df["predicted_anomaly_type"] = (
                type_df["predicted_anomaly_type"]
                .str.replace("_", " ")
                .str.title()
            )
            fig2 = px.pie(
                type_df,
                values="count",
                names="predicted_anomaly_type",
                hole=0.55,
                color_discrete_sequence=PALETTE,
            )
            fig2.update_traces(
                textposition="inside",
                textinfo="percent",
                texttemplate="%{percent:.1%}",
                textfont=dict(size=11, color="#ffffff", family="Inter, sans-serif"),
            )
            fig2.update_layout(
                **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
                showlegend=True,
                legend=dict(
                    orientation="h",
                    y=-0.22, x=0.5, xanchor="center",
                    font=dict(size=10, color="#94a3b8"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                height=380,
                margin=dict(t=10, b=50, l=10, r=10),
            )
            st.plotly_chart(fig2, use_container_width=True, key="dashboard_attack_dist_pie")
        else:
            st.info("No data available.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Row 2: Risk Histogram + Status Bar ───────────────────────────────────
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown('<div class="section-label">Risk Score Distribution</div>', unsafe_allow_html=True)
        if len(alerts_df) > 0:
            fig3 = px.histogram(
                alerts_df, x="risk_score", nbins=50,
                labels={"risk_score": "Risk Score"},
                color_discrete_sequence=["#06b6d4"],
            )
            fig3.add_vline(
                x=0.7, line_dash="dash", line_color="#ef4444", line_width=1.5,
                annotation_text="Threshold 0.7",
                annotation_font=dict(color="#ef4444", size=11),
                annotation_position="top right",
            )
            fig3.update_layout(**CHART_LAYOUT, height=320)
            fig3.update_layout(yaxis_title="Count")
            st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        st.markdown('<div class="section-label">Alert Status Breakdown</div>', unsafe_allow_html=True)
        if len(stats["alerts_by_status"]) > 0:
            status_colors = {
                "New":           "#ef4444",
                "Investigating": "#f59e0b",
                "Closed":        "#10b981",
                "False Positive":"#8b5cf6",
            }
            fig4 = go.Figure([
                go.Bar(
                    x=stats["alerts_by_status"]["status"],
                    y=stats["alerts_by_status"]["count"],
                    marker_color=[
                        status_colors.get(s, "#64748b")
                        for s in stats["alerts_by_status"]["status"]
                    ],
                    text=stats["alerts_by_status"]["count"],
                    textposition="outside",
                    textfont=dict(size=12, color="#f1f5f9"),
                    width=0.5,
                )
            ])
            fig4.update_layout(**CHART_LAYOUT, height=320, yaxis_title="Count")
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Critical Alerts Table ─────────────────────────────────────────────────
    st.markdown('<div class="section-label">Top 10 Critical Alerts</div>', unsafe_allow_html=True)
    critical = fetch_alerts(risk_threshold=0.7, limit=10)
    if len(critical) > 0:
        display = critical[["timestamp", "entity_id", "risk_score", "predicted_anomaly_type", "status"]].copy()
        display.columns = ["Timestamp", "Entity ID", "Risk Score", "Attack Type", "Status"]
        display["Attack Type"] = display["Attack Type"].str.replace("_", " ").str.title()
        display["Risk Score"] = display["Risk Score"].round(4)
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.info("No critical alerts found.")


if __name__ == "__main__":
    main()
else:
    main()
