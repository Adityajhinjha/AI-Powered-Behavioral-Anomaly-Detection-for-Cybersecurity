import os
import sys
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))


def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


from config import DATABASE_PATH, ISOLATION_FOREST_PATH, PROCESSED_DATA_PATH


def check_pipeline_status():
    checks = {
        "database":    os.path.exists(DATABASE_PATH),
        "models":      os.path.exists(ISOLATION_FOREST_PATH),
        "predictions": os.path.exists(PROCESSED_DATA_PATH),
    }
    return checks


def main():
    st.set_page_config(
        page_title="Overview | CyberGuard SOC",
        page_icon="assets/favicon.ico",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_css()

    status = check_pipeline_status()
    all_ready = all(status.values())

    ready_chip = (
        '<span class="status-chip live"><span class="dot"></span>System Ready</span>'
        if all_ready
        else '<span class="status-chip" style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.25);color:#fcd34d;">Setup Required</span>'
    )

    st.markdown(
        f"""
        <div class="page-header" style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:12px;">
            <div>
                <h1 style="margin:0;font-size:1.75rem;font-weight:800;letter-spacing:-0.03em;color:#f1f5f9;">
                    CyberGuard SOC
                </h1>
                <p style="margin:6px 0 0;color:#94a3b8;font-size:0.92rem;">
                    Behavioral Anomaly Detection &nbsp;/&nbsp; Enterprise Security Operations
                </p>
            </div>
            <div style="padding-top:4px;">{ready_chip}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if all_ready:
        from src.database import get_alert_stats, fetch_alerts
        import pandas as pd

        try:
            stats = get_alert_stats()
            recent = fetch_alerts(risk_threshold=0.0, limit=5000)
        except Exception:
            stats = None
            recent = pd.DataFrame()

        if stats:
            st.markdown('<div class="section-label">Live System Metrics</div>', unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("Total Events", f"{stats['total_logs']:,}")
            with c2:
                st.metric("Security Alerts", f"{stats['total_alerts']:,}")
            with c3:
                critical = len(recent[recent["risk_score"] >= 0.90]) if len(recent) > 0 else 0
                st.metric("Critical Alerts", f"{critical:,}")
            with c4:
                st.metric("Alert Rate", f"{stats['alert_rate']:.2f}%")
            with c5:
                entities = recent["entity_id"].nunique() if len(recent) > 0 and "entity_id" in recent.columns else 0
                st.metric("Entities at Risk", f"{entities:,}")
        else:
            st.markdown('<div class="section-label">Pipeline Artifacts</div>', unsafe_allow_html=True)
            st.info("System ready but unable to load metrics.")
    else:
        st.markdown('<div class="section-label">Pipeline Artifacts</div>', unsafe_allow_html=True)

        icons = {"database": "DB", "models": "ML", "predictions": "OUT"}
        labels = {"database": "Database", "models": "ML Models", "predictions": "Predictions"}
        col1, col2, col3, col4 = st.columns(4)

        for col, (key, label) in zip([col1, col2, col3], status.items()):
            ok = status[key]
            color = "#10b981" if ok else "#ef4444"
            bg = "rgba(16,185,129,0.08)" if ok else "rgba(239,68,68,0.08)"
            border = "rgba(16,185,129,0.25)" if ok else "rgba(239,68,68,0.25)"
            state_text = "Ready" if ok else "Missing"
            with col:
                st.markdown(
                    f"""
                    <div style="background:{bg};border:1px solid {border};border-radius:10px;
                                padding:16px 18px;margin-bottom:12px;">
                        <div style="font-size:10px;font-weight:700;color:{color};text-transform:uppercase;
                                    letter-spacing:0.1em;font-family:'JetBrains Mono',monospace;">{icons[key]}</div>
                        <div style="font-size:1rem;font-weight:700;color:#f1f5f9;margin:6px 0 4px;">{label}</div>
                        <div style="font-size:11px;font-weight:600;color:{color};">{state_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col4:
            overall_color = "#f59e0b"
            overall_bg = "rgba(245,158,11,0.08)"
            overall_border = "rgba(245,158,11,0.25)"
            st.markdown(
                f"""
                <div style="background:{overall_bg};border:1px solid {overall_border};border-radius:10px;
                            padding:16px 18px;margin-bottom:12px;">
                    <div style="font-size:10px;font-weight:700;color:{overall_color};text-transform:uppercase;
                                letter-spacing:0.1em;font-family:'JetBrains Mono',monospace;">SYS</div>
                    <div style="font-size:1rem;font-weight:700;color:#f1f5f9;margin:6px 0 4px;">Overall Status</div>
                    <div style="font-size:11px;font-weight:600;color:{overall_color};">Setup Required</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.info(
            "System artifacts missing. Click the button below to initialize the pipeline directly, or run manually:\n"
            "`python -m src.data_generator` ➔ `python -m src.feature_engineering` "
            "➔ `python -m src.train_detector` ➔ `python -m src.train_classifier` ➔ `python -m src.predict`"
        )
        
        if st.button("🚀 Initialize Pipeline & Train Models", type="primary", use_container_width=True):
            status_container = st.empty()
            progress_bar = st.progress(0)
            
            try:
                status_container.info("Step 1/5: Generating synthetic telemetry data...")
                import src.data_generator as dg
                dg.generate_all()
                progress_bar.progress(20)
                
                status_container.info("Step 2/5: Engineering behavioral features...")
                import src.feature_engineering as fe
                fe.extract_features()
                progress_bar.progress(40)
                
                status_container.info("Step 3/5: Training Isolation Forest anomaly detector...")
                import src.train_detector as td
                td.train()
                progress_bar.progress(60)
                
                status_container.info("Step 4/5: Training XGBoost multi-class classifier...")
                import src.train_classifier as tc
                tc.train()
                progress_bar.progress(80)
                
                status_container.info("Step 5/5: Running predictions & SHAP explainability...")
                import src.predict as pred
                pred.predict_batch()
                progress_bar.progress(100)
                
                status_container.success("✅ System Pipeline Initialized Successfully!")
                st.rerun()
            except Exception as e:
                status_container.error(f"Error during initialization: {str(e)}")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Platform Capabilities</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-num">01 — DATA</div>
                <div class="feature-title">Synthetic Telemetry</div>
                <div class="feature-desc">Realistic access-log generation with injected attack patterns across 7 threat categories.</div>
            </div>
            <div class="feature-card">
                <div class="feature-num">02 — DETECT</div>
                <div class="feature-title">Anomaly Detection</div>
                <div class="feature-desc">Isolation Forest baseline modeling to surface statistically deviant behaviors in real time.</div>
            </div>
            <div class="feature-card">
                <div class="feature-num">03 — CLASSIFY</div>
                <div class="feature-title">Threat Classification</div>
                <div class="feature-desc">Random Forest + XGBoost ensemble for multi-class attack categorization with confidence scores.</div>
            </div>
            <div class="feature-card">
                <div class="feature-num">04 — EXPLAIN</div>
                <div class="feature-title">SHAP Explainability</div>
                <div class="feature-desc">Per-alert feature attribution so analysts understand every prediction — not just the label.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Detection Pipeline</div>', unsafe_allow_html=True)

    pipeline_steps = [
        ("Log Ingestion", "data_generator.py", "Synthetic telemetry with realistic\ndistributions & attack injections"),
        ("Feature Engineering", "feature_engineering.py", "33 behavioral features:\nvelocity, geo-distance, time deltas"),
        ("Anomaly Scoring", "train_detector.py", "Isolation Forest baseline\nanomaly score assignment"),
        ("Attack Classification", "train_classifier.py", "XGBoost multi-class\nthreat categorization"),
        ("SHAP + Triage", "predict.py", "Per-alert SHAP attribution\n& SOC alert queue"),
    ]

    node_x = [0.08, 0.28, 0.50, 0.72, 0.92]
    node_y = [0.5, 0.5, 0.5, 0.5, 0.5]
    node_colors = ["#06b6d4", "#3b82f6", "#8b5cf6", "#f59e0b", "#10b981"]

    fig_pipeline = go.Figure()

    for i in range(len(pipeline_steps) - 1):
        fig_pipeline.add_annotation(
            x=node_x[i + 1] - 0.04,
            y=0.5,
            ax=node_x[i] + 0.04,
            ay=0.5,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor="rgba(148,163,184,0.4)",
        )

    fig_pipeline.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=52,
            color=node_colors,
            line=dict(width=2, color="rgba(255,255,255,0.15)"),
            symbol="circle",
        ),
        text=[f"0{i+1}" for i in range(5)],
        textposition="middle center",
        textfont=dict(family="JetBrains Mono, monospace", size=14, color="#0a0d13", weight=700),
        hovertext=[f"<b>{s[0]}</b><br><i>{s[1]}</i><br>{s[2]}" for s in pipeline_steps],
        hoverinfo="text",
        showlegend=False,
    ))

    for i, (title, module, _) in enumerate(pipeline_steps):
        fig_pipeline.add_annotation(
            x=node_x[i], y=0.28,
            text=f"<b>{title}</b>",
            showarrow=False,
            font=dict(family="Inter, sans-serif", size=12, color="#f1f5f9"),
            xref="x", yref="y",
        )
        fig_pipeline.add_annotation(
            x=node_x[i], y=0.18,
            text=f"{module}",
            showarrow=False,
            font=dict(family="JetBrains Mono, monospace", size=9, color="#64748b"),
            xref="x", yref="y",
        )

    fig_pipeline.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(t=20, b=10, l=20, r=20),
        xaxis=dict(range=[-0.02, 1.02], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0.05, 0.75], showgrid=False, zeroline=False, visible=False),
    )
    st.plotly_chart(fig_pipeline, use_container_width=True, key="pipeline_flow")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Threat Coverage &amp; MITRE ATT&amp;CK Mapping</div>', unsafe_allow_html=True)
    threats = [
        ("Brute Force", "T1110", "#ef4444", "High-velocity failed auth attempts against a single entity."),
        ("Impossible Travel", "T1078", "#f59e0b", "Logins from geographically infeasible locations within minutes."),
        ("Credential Stuffing", "T1110.004", "#8b5cf6", "Large-scale automated credential replay across many accounts."),
        ("Lateral Movement", "T1021", "#3b82f6", "Internal east-west traversal across sensitive resources."),
        ("Device Spoofing", "T1200", "#06b6d4", "Anomalous device fingerprint changes mid-session."),
        ("Low-and-Slow Exfiltration", "T1041", "#10b981", "Below-threshold data transfers sustained over days."),
        ("Insider Drift", "T1078.003", "#ec4899", "Gradual behavioral shift from an established normal baseline."),
    ]
    cols = st.columns(4)
    for i, (name, mitre, color, desc) in enumerate(threats):
        with cols[i % 4]:
            st.markdown(
                f"""
                <div style="background:#111827;border:1px solid rgba(255,255,255,0.07);border-radius:10px;
                            padding:14px 16px;margin-bottom:10px;">
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
                        <span style="font-size:12px;font-weight:700;color:#f1f5f9;">{name}</span>
                        <span style="font-size:9px;font-weight:700;color:{color};background:rgba(255,255,255,0.05);
                                     padding:2px 6px;border-radius:4px;font-family:'JetBrains Mono',monospace;">{mitre}</span>
                    </div>
                    <div style="font-size:11.5px;color:#64748b;line-height:1.45;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if all_ready and 'recent' in dir() and len(recent) > 0:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Recent Critical Alerts</div>', unsafe_allow_html=True)
        critical_df = recent[recent["risk_score"] >= 0.90].head(6)
        if len(critical_df) > 0:
            for _, row in critical_df.iterrows():
                eid = str(row.get("entity_id", ""))
                rtype = str(row.get("predicted_anomaly_type", "")).replace("_", " ").title()
                risk = float(row.get("risk_score", 0))
                ts = str(row.get("timestamp", ""))
                badge_color = "#ef4444" if risk >= 0.95 else "#f59e0b"
                badge_bg = f"rgba({239 if risk >= 0.95 else 245},{68 if risk >= 0.95 else 158},{68 if risk >= 0.95 else 11},0.12)"
                st.markdown(
                    f"""
                    <div style="background:#111827;border:1px solid rgba(255,255,255,0.07);border-radius:10px;
                                padding:12px 16px;margin-bottom:6px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
                        <div style="display:flex;align-items:center;gap:14px;">
                            <span style="background:{badge_bg};color:{badge_color};font-size:10px;font-weight:700;
                                         padding:3px 8px;border-radius:6px;font-family:'JetBrains Mono',monospace;">{risk:.3f}</span>
                            <span style="font-size:13px;font-weight:600;color:#f1f5f9;">{eid}</span>
                            <span style="font-size:12px;color:#64748b;">{rtype}</span>
                        </div>
                        <span style="font-size:11px;color:#475569;font-family:'JetBrains Mono',monospace;">{ts}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
else:
    main()

