import os
import sys
import json
import io
import streamlit as st
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.database import get_alert_stats, fetch_alerts


def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def load_model_metrics():
    base = os.path.join(os.path.dirname(__file__), "..", "models")
    for fname in ["eval_metrics.json", "classifier_metrics.json"]:
        path = os.path.join(base, fname)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                report = data.get("classification_report", {})
                if report and "accuracy" in report:
                    data.setdefault("accuracy", report["accuracy"])
                if report and "macro avg" in report:
                    data.setdefault("macro_f1", report["macro avg"].get("f1-score"))
                    data["balanced_accuracy"] = report["macro avg"].get("recall")
                return data
    return None


def generate_text_report(stats, metrics, alerts_df):
    """Build a plain-text executive report string."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "=" * 72,
        "  CYBERGUARD SOC — BEHAVIORAL ANOMALY DETECTION",
        "  EXECUTIVE SUMMARY REPORT",
        "=" * 72,
        f"  Generated: {now}",
        "",
        "SYSTEM OVERVIEW",
        "-" * 40,
        f"  Total Access Events Analyzed : {stats['total_logs']:,}",
        f"  Security Alerts Generated    : {stats['total_alerts']:,}",
        f"  Overall Alert Rate           : {stats['alert_rate']:.2f}%",
        f"  Top Threat Category          : {stats['top_anomaly_type'].replace('_', ' ').title()}",
        "",
    ]

    if metrics:
        lines += [
            "MODEL PERFORMANCE",
            "-" * 40,
            f"  Accuracy   : {metrics.get('accuracy', 'N/A')}",
            f"  Macro F1   : {metrics.get('macro_f1', 'N/A')}",
            f"  Classes    : {metrics.get('num_classes', 'N/A')}",
            "",
        ]

    lines += [
        "ALERT DISTRIBUTION BY TYPE",
        "-" * 40,
    ]
    for _, row in stats["alerts_by_type"].iterrows():
        label = str(row["predicted_anomaly_type"]).replace("_", " ").title()
        count = int(row["count"])
        bar   = "#" * min(40, int(count / max(stats["alerts_by_type"]["count"]) * 40))
        lines.append(f"  {label:<30} {count:>6}  {bar}")

    lines += [
        "",
        "STATUS BREAKDOWN",
        "-" * 40,
    ]
    for _, row in stats["alerts_by_status"].iterrows():
        lines.append(f"  {str(row['status']):<20} {int(row['count']):>6}")

    lines += [
        "",
        "THREAT METHODOLOGY",
        "-" * 40,
        "  Detection:      Isolation Forest (200 estimators, 5% contamination)",
        "  Classification: Random Forest + XGBoost ensemble (7 attack classes)",
        "  Explainability: SHAP TreeExplainer (per-alert feature attribution)",
        "  Features:       33 engineered behavioral features per event",
        "",
        "DETECTION TAXONOMY",
        "-" * 40,
        "  01  Brute Force          — High-velocity failed authentication attempts",
        "  02  Impossible Travel    — Geographically infeasible login sequences",
        "  03  Credential Stuffing  — Automated credential replay at scale",
        "  04  Lateral Movement     — East-west traversal across sensitive systems",
        "  05  Device Spoofing      — Anomalous device fingerprint changes",
        "  06  Low-and-Slow Exfil  — Sub-threshold data transfers over days",
        "  07  Insider Drift        — Gradual behavioral divergence from baseline",
        "",
        "LIMITATIONS",
        "-" * 40,
        "  - Trained exclusively on synthetic data; real-world deployment requires",
        "    retraining on organizational telemetry.",
        "  - Ground-truth labels are injected at generation time; alert classification",
        "    accuracy depends on class balance and feature coverage.",
        "  - SHAP values are approximate and should be combined with analyst judgment.",
        "",
        "=" * 72,
        "  END OF REPORT",
        "=" * 72,
    ]

    return "\n".join(lines)


def main():
    st.set_page_config(page_title="Reports | CyberGuard SOC", page_icon=None, layout="wide")
    load_css()

    st.markdown(
        """
        <div class="page-header">
            <h1>Reports</h1>
            <p>Generate and export executive summary reports for security leadership and compliance</p>
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

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    # ── Report Header ─────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="card" style="margin-bottom:20px;">
            <div class="card-label">Report Metadata</div>
            <div style="display:flex;gap:48px;flex-wrap:wrap;margin-top:6px;">
                <div>
                    <div style="font-size:11px;color:#475569;margin-bottom:2px;">Generated</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:13px;color:#f1f5f9;">{now_str}</div>
                </div>
                <div>
                    <div style="font-size:11px;color:#475569;margin-bottom:2px;">Events Analyzed</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:13px;color:#f1f5f9;">{stats['total_logs']:,}</div>
                </div>
                <div>
                    <div style="font-size:11px;color:#475569;margin-bottom:2px;">Alerts Generated</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:13px;color:#f1f5f9;">{stats['total_alerts']:,}</div>
                </div>
                <div>
                    <div style="font-size:11px;color:#475569;margin-bottom:2px;">Alert Rate</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:13px;color:#f1f5f9;">{stats['alert_rate']:.2f}%</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Report Filter Controls ────────────────────────────────────────────────
    st.markdown('<div class="section-label">Report Scope &amp; Search Filters</div>', unsafe_allow_html=True)
    fcol1, fcol2, fcol3 = st.columns([1.5, 1.5, 2])

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
        sel_threat = st.selectbox(
            "Threat Category Filter",
            threat_options,
            index=0,
            key="report_threat_filter",
        )

    risk_options = {
        "All Risk Levels": 0.0,
        "Critical (>0.90)": 0.90,
        "High (>0.70)": 0.70,
        "Medium (>0.50)": 0.50,
    }
    with fcol2:
        sel_risk_label = st.selectbox(
            "Risk Threshold Filter",
            list(risk_options.keys()),
            index=0,
            key="report_risk_filter",
        )
        min_risk = risk_options[sel_risk_label]

    with fcol3:
        search_query = st.text_input(
            "Search Entity / Alert ID",
            placeholder="Type user_042 or alert ID...",
            key="report_search_input",
        )

    # ── Filter alerts based on dropdown and search selections ─────────────────
    filtered_alerts = alerts.copy()
    if min_risk > 0:
        filtered_alerts = filtered_alerts[filtered_alerts["risk_score"] >= min_risk]
    if sel_threat != "All Threat Types":
        norm_threat = sel_threat.lower().replace(" ", "_")
        filtered_alerts = filtered_alerts[
            filtered_alerts["predicted_anomaly_type"].str.lower() == norm_threat
        ]
    if search_query.strip():
        sq = search_query.strip().lower()
        filtered_alerts = filtered_alerts[
            filtered_alerts["entity_id"].astype(str).str.lower().str.contains(sq) |
            filtered_alerts["alert_id"].astype(str).str.lower().str.contains(sq)
        ]

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Executive Summary", "Threat Breakdown", "Methodology", "Export"]
    )

    # ─── Tab 1: Executive Summary ─────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-label">Key Findings</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            filtered_rate = (len(filtered_alerts) / stats['total_logs'] * 100) if stats['total_logs'] > 0 else 0.0
            st.metric("Filtered Alert Rate", f"{filtered_rate:.2f}%")
        with col2:
            accuracy = metrics.get("balanced_accuracy", metrics.get("accuracy")) if metrics else None
            st.metric("Balanced Accuracy", f"{accuracy:.1%}" if accuracy else "—")
        with col3:
            if len(filtered_alerts) > 0:
                top_filtered = filtered_alerts["predicted_anomaly_type"].value_counts().index[0].replace("_", " ").title()
            else:
                top_filtered = "None"
            st.metric("Top Threat (Filtered)", top_filtered)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Alert Distribution</div>', unsafe_allow_html=True)
        if len(filtered_alerts) > 0:
            dist = filtered_alerts["predicted_anomaly_type"].value_counts().reset_index()
            dist.columns = ["Attack Type", "Count"]
            dist["Attack Type"] = dist["Attack Type"].str.replace("_", " ").str.title()
            dist["% Share"] = (dist["Count"] / dist["Count"].sum() * 100).round(1).astype(str) + "%"
            st.dataframe(dist, use_container_width=True, hide_index=True)
        else:
            st.info("No alerts match the selected search & dropdown filters.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Status Summary</div>', unsafe_allow_html=True)
        if len(filtered_alerts) > 0:
            s_df = filtered_alerts["status"].value_counts().reset_index()
            s_df.columns = ["Status", "Count"]
            st.dataframe(s_df, use_container_width=True, hide_index=True)

    # ─── Tab 2: Threat Breakdown ──────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-label">Detection Taxonomy</div>', unsafe_allow_html=True)
        threats = [
            ("Brute Force",
             "High-velocity failed authentication attempts targeting a single entity.",
             "Threshold-based velocity detection; rapid successive failures flagged."),
            ("Impossible Travel",
             "Login sequences from geographically infeasible locations within minutes.",
             "Haversine distance + time-delta calculation against prior location."),
            ("Credential Stuffing",
             "Large-scale automated credential replay across many distinct accounts.",
             "Cross-entity velocity and user-agent entropy analysis."),
            ("Lateral Movement",
             "Internal east-west traversal across sensitive or unusual resources.",
             "Unique resource count, privilege escalation patterns."),
            ("Device Spoofing",
             "Anomalous device fingerprint changes mid-session.",
             "Device ID entropy, device switch frequency per session."),
            ("Low-and-Slow Exfiltration",
             "Sub-threshold data transfers sustained over extended periods.",
             "Rolling byte-rate over 24-hour windows with drift detection."),
            ("Insider Drift",
             "Gradual behavioral divergence from an established normal baseline.",
             "Rolling z-score against per-entity 30-day baseline model."),
        ]

        for i, (name, desc, method) in enumerate(threats, 1):
            st.markdown(
                f"""
                <div class="card" style="margin-bottom:10px;">
                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                        <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                                     font-weight:700;color:#06b6d4;letter-spacing:0.1em;">
                            {i:02d}
                        </span>
                        <span style="font-size:15px;font-weight:700;color:#f1f5f9;">{name}</span>
                    </div>
                    <div style="font-size:13px;color:#94a3b8;margin-bottom:6px;">{desc}</div>
                    <div style="font-size:11.5px;color:#475569;font-style:italic;">{method}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ─── Tab 3: Methodology ───────────────────────────────────────────────────
    with tab3:
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<div class="section-label">Detection Pipeline</div>', unsafe_allow_html=True)
            steps = [
                ("01", "Log Ingestion", "Synthetic telemetry generation with controlled attack injections"),
                ("02", "Feature Engineering", "33 behavioral features extracted per event"),
                ("03", "Anomaly Detection", "Isolation Forest baseline model — unsupervised"),
                ("04", "Attack Classification", "Random Forest + XGBoost ensemble — supervised"),
                ("05", "Explainability", "SHAP TreeExplainer per prediction"),
                ("06", "SOC Interface", "Ranked alert queue with drill-down"),
            ]
            for step, title, desc in steps:
                st.markdown(
                    f"""
                    <div style="display:flex;gap:14px;margin-bottom:14px;align-items:flex-start;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;
                                    color:#06b6d4;min-width:24px;padding-top:2px;">{step}</div>
                        <div>
                            <div style="font-size:13px;font-weight:600;color:#f1f5f9;margin-bottom:3px;">{title}</div>
                            <div style="font-size:12px;color:#475569;">{desc}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_r:
            st.markdown('<div class="section-label">Model Specifications</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="card">
                    <div class="card-label" style="margin-bottom:12px;">Anomaly Detector</div>
                    <table style="width:100%;font-size:13px;border-collapse:collapse;margin-bottom:16px;">
                        <tr><td style="color:#64748b;padding:4px 0;">Algorithm</td><td style="color:#f1f5f9;text-align:right;">Isolation Forest</td></tr>
                        <tr><td style="color:#64748b;padding:4px 0;">Estimators</td><td style="color:#f1f5f9;text-align:right;">200</td></tr>
                        <tr><td style="color:#64748b;padding:4px 0;">Contamination</td><td style="color:#f1f5f9;text-align:right;">0.05</td></tr>
                        <tr><td style="color:#64748b;padding:4px 0;">Max Samples</td><td style="color:#f1f5f9;text-align:right;">auto</td></tr>
                    </table>
                    <div class="card-label" style="margin-bottom:12px;">Attack Classifier</div>
                    <table style="width:100%;font-size:13px;border-collapse:collapse;">
                        <tr><td style="color:#64748b;padding:4px 0;">Algorithm</td><td style="color:#f1f5f9;text-align:right;">RF + XGBoost</td></tr>
                        <tr><td style="color:#64748b;padding:4px 0;">Classes</td><td style="color:#f1f5f9;text-align:right;">7 attack types</td></tr>
                        <tr><td style="color:#64748b;padding:4px 0;">Features</td><td style="color:#f1f5f9;text-align:right;">33 engineered</td></tr>
                        <tr><td style="color:#64748b;padding:4px 0;">Explainability</td><td style="color:#f1f5f9;text-align:right;">SHAP TreeExplainer</td></tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Limitations</div>', unsafe_allow_html=True)
            limitations = [
                "Trained on synthetic data — real deployment requires retraining on organizational telemetry",
                "Ground-truth labels injected at generation time; classification depends on class balance",
                "SHAP values are approximate and should supplement, not replace, analyst judgment",
                "No real-time ingestion pipeline — batch prediction mode only in current version",
            ]
            for item in limitations:
                st.markdown(
                    f'<div style="font-size:12.5px;color:#475569;margin-bottom:8px;padding-left:12px;'
                    f'border-left:2px solid rgba(245,158,11,0.4);">{item}</div>',
                    unsafe_allow_html=True,
                )

    # ─── Tab 4: Export ────────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-label">Export Options</div>', unsafe_allow_html=True)

        col_e1, col_e2 = st.columns(2)

        with col_e1:
            st.markdown(
                """
                <div class="card" style="margin-bottom:14px;">
                    <div class="card-label">Text Report</div>
                    <p style="font-size:13px;color:#64748b;margin-top:6px;">
                        Download the complete executive summary as a plain-text file.
                        Suitable for email distribution and compliance documentation.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            report_text = generate_text_report(stats, metrics, filtered_alerts)
            st.download_button(
                label="Download .txt Report",
                data=report_text,
                file_name=f"cyberguard_soc_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
            )

        with col_e2:
            st.markdown(
                """
                <div class="card" style="margin-bottom:14px;">
                    <div class="card-label">CSV Alert Export</div>
                    <p style="font-size:13px;color:#64748b;margin-top:6px;">
                        Download filtered alerts as a structured CSV file.
                        Compatible with SIEM integration and data analysis tools.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if len(filtered_alerts) > 0:
                csv_buf = io.StringIO()
                filtered_alerts.drop(columns=["shap_values"], errors="ignore").to_csv(csv_buf, index=False)
                st.download_button(
                    label="Download Alerts CSV",
                    data=csv_buf.getvalue(),
                    file_name=f"cyberguard_alerts_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )
            else:
                st.info("No alert data available for export.")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Report Preview</div>', unsafe_allow_html=True)

        with st.expander("View Text Report"):
            report_text_preview = generate_text_report(stats, metrics, filtered_alerts)
            st.code(report_text_preview, language=None)


if __name__ == "__main__":
    main()
else:
    main()
