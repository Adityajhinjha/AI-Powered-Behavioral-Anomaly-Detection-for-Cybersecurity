# 🛡️ Behavioral Anomaly Detection System

🚀 **Live Streamlit App:** [https://ai-powered-behavioral-anomaly-detection-for-cybersecurity-sd.streamlit.app/](https://ai-powered-behavioral-anomaly-detection-for-cybersecurity-sd.streamlit.app/)  
💻 **GitHub Repository:** [https://github.com/Adityajhinjha/AI-Powered-Behavioral-Anomaly-Detection-for-Cybersecurity](https://github.com/Adityajhinjha/AI-Powered-Behavioral-Anomaly-Detection-for-Cybersecurity)

An end-to-end machine learning system that models "normal" access and connection behavior for users and devices, detects intrusions or compromised credential activity in near real-time, and classifies the type of anomaly with an explainable risk score — presented through an interactive SOC analyst dashboard.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Pipeline Steps](#-pipeline-steps)
- [Attack Taxonomy](#-attack-taxonomy)
- [Feature Engineering](#-feature-engineering)
- [Model Details](#-model-details)
- [Dashboard Pages](#-dashboard-pages)
- [Evaluation Metrics](#-evaluation-metrics)
- [Assumptions & Limitations](#-assumptions--limitations)
- [Future Improvements](#-future-improvements)

---

## 🎯 Problem Statement

Design and build a machine learning system that:
1. Models **"normal" access and connection behavior** for users and devices
2. **Detects intrusions** or compromised credential activity in near real-time
3. **Classifies the type of anomaly** (brute force, impossible travel, credential misuse, lateral movement, device spoofing)
4. Provides an **explainable risk score** with feature attribution

### Key Challenges Handled
- **Sequential & behavioral data** — access events over time, not static snapshots
- **Extreme class imbalance** — true intrusions are a tiny fraction of total events
- **Concept drift** — legitimate behavior evolves and should not be permanently flagged
- **Explainability** — SOC analysts need to know *why* an event was flagged
- **Cold-start problem** — scoring brand-new users or devices with no history

---

## 🏗️ Architecture

```
┌─────────────────────────┐
│   Synthetic Data        │  Faker + NumPy + Pandas
│   Generator             │  7 Attack Patterns
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Feature Engineering   │  21 ML Features
│                         │  Temporal, Geo, Auth, Device
└──────────┬──────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌────────────┐
│ Isolation│ │  XGBoost   │  Multi-class
│ Forest   │ │  Classifier│  8-class Classification
└────┬────┘ └─────┬──────┘
     │            │
     └──────┬─────┘
            ▼
┌─────────────────────────┐
│   SHAP Explainability   │  Per-alert Feature
│                         │  Attribution
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Streamlit Dashboard   │  5-Page SOC Analyst
│                         │  Interface
└─────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|:------|:-----------|
| Frontend & Dashboard | Streamlit, streamlit-option-menu |
| Data Generation | Pandas, NumPy, Faker |
| Anomaly Detection | Isolation Forest (scikit-learn) |
| Attack Classification | XGBoost |
| Explainability | SHAP |
| Visualization | Plotly |
| Geo-Distance | geopy |
| Database | SQLite |
| Model Storage | Joblib |

---

## 📁 Project Structure

```
├── app.py                         # Streamlit main landing page
├── config.py                      # Centralized configuration
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── synthetic_logs.csv     # Generated access logs
│   └── processed/
│       └── features.csv           # Engineered ML features
│
├── database/
│   └── anomaly_detector.db        # SQLite database
│
├── models/
│   ├── isolation_forest.joblib    # Baseline detector
│   ├── xgboost_classifier.joblib  # Attack classifier
│   ├── scaler.joblib              # Feature scaler
│   ├── label_encoder.joblib       # Class label encoder
│   └── eval_metrics.json          # Evaluation results
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py          # Synthetic log generator
│   ├── feature_engineering.py     # Feature extraction pipeline
│   ├── database.py                # SQLite helpers
│   ├── train_detector.py          # Isolation Forest training
│   ├── train_classifier.py        # XGBoost training
│   ├── predict.py                 # Inference pipeline
│   ├── explainability.py          # SHAP explanations
│   └── utils.py                   # Helper functions
│
├── pages/
│   ├── 1_Dashboard.py             # Executive Overview
│   ├── 2_Live_Alerts.py           # Alert Queue + SHAP
│   ├── 3_User_Analysis.py         # Entity Investigation
│   ├── 4_Analytics.py             # Model Performance
│   └── 5_Reports.py               # Summary & Export
│
└── assets/
    └── styles.css                 # Dark SOC theme
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
```bash
# Step 1: Generate synthetic access logs (~100K events)
python -m src.data_generator

# Step 2: Extract 21 ML features
python -m src.feature_engineering

# Step 3: Train Isolation Forest baseline detector
python -m src.train_detector

# Step 4: Train XGBoost multi-class classifier
python -m src.train_classifier

# Step 5: Run predictions and generate alerts with SHAP explanations
python -m src.predict
```

### 3. Launch the Dashboard
```bash
streamlit run app.py
```

---

## 🔄 Pipeline Steps

| Step | Module | Input | Output |
|:-----|:-------|:------|:-------|
| 1 | `data_generator.py` | Config parameters | `data/raw/synthetic_logs.csv` + SQLite |
| 2 | `feature_engineering.py` | Raw logs | `data/processed/features.csv` |
| 3 | `train_detector.py` | Processed features | `models/isolation_forest.joblib` |
| 4 | `train_classifier.py` | Processed features | `models/xgboost_classifier.joblib` |
| 5 | `predict.py` | Features + Models | Security alerts → SQLite |

---

## 🎭 Attack Taxonomy

| Attack Type | Description | Signal Type | Injection Rate |
|:------------|:------------|:------------|:---------------|
| **Brute Force** | Rapid repeated failed-auth attempts from one source | Anomaly | ~0.4% |
| **Impossible Travel** | Same entity, distant geos, implausible time gap | Anomaly | ~0.3% |
| **Credential Stuffing** | Many entity_ids, few source_ips, high failure rate | Anomaly | ~0.4% |
| **Lateral Movement** | Entity accesses unusual breadth of resources | Anomaly | ~0.3% |
| **Device Spoofing** | Device reappears with mismatched fingerprint | Anomaly | ~0.3% |
| **Low-and-Slow Exfiltration** | Gradual off-hours resource access over weeks | Anomaly | ~0.2% |
| **Insider Drift** | Legitimate entity slowly expanding privileges | Edge Case | ~0.1% |

---

## 🧪 Feature Engineering

**21 features** across 6 categories:

| Category | Features |
|:---------|:---------|
| **Temporal** | hour_of_day, day_of_week, is_weekend, is_off_hours, time_since_last_login |
| **Authentication** | auth_success, failed_auth_count_1h, auth_failure_rate_1h, unique_source_ips_1h, total_events_1h |
| **Geographic** | geo_velocity (km/h via geopy), geo_distance_from_home |
| **Resource** | unique_resources_1h, resource_diversity_ratio, new_resource_flag |
| **Device** | fingerprint_changed, new_device_flag |
| **Profile** | session_duration, session_duration_zscore, command_sequence_length, days_since_first_seen |

---

## 🤖 Model Details

### Stage 1: Isolation Forest (Baseline Detector)
- Trained on **normal-only** events (unsupervised)
- Produces anomaly score (0–1, higher = more anomalous)
- **Cold-start handling**: Entities with < 7 days history get elevated baseline score (0.5)
- **Concept drift**: Rolling 60-day training window

### Stage 2: XGBoost Classifier (Attack Classification)
- 8-class classification: `normal` + 7 attack types
- **Imbalance handling**: Inverse class frequency sample weights
- **Combined risk score**: `0.4 × IF_score + 0.6 × XGB_max_probability`
- Normal classifications get risk score reduced by 70%

### Explainability: SHAP TreeExplainer
- Per-alert top-5 feature attribution
- Human-readable explanations: *"Flagged due to geo_velocity (+0.42) + failed_auth_count_1h (+0.31)"*

---

## 📊 Dashboard Pages

| Page | Description |
|:-----|:------------|
| **📊 Dashboard** | Executive KPIs, alert timeline, attack distribution, risk histogram |
| **🚨 Live Alerts** | Ranked alert queue, SHAP waterfall drill-down, status management |
| **🔍 User Analysis** | Entity search, activity timeline, access heatmap, geo map |
| **📈 Analytics** | Confusion matrix, classification report, SHAP importance, F1 chart |
| **📄 Reports** | Full methodology report, CSV/TXT export |

---

## ⚠️ Assumptions & Limitations

**Assumptions:**
- Entity behavioral profiles are relatively stable over the observation window
- Attack patterns follow documented signatures
- Geographic coordinates are approximations based on IP-to-location mapping

**Known Limitations:**
- **Synthetic data** may not capture full real-world complexity
- **Batch-oriented** — real-time streaming would require Kafka/Flink
- **Single-node SQLite** limits concurrent write access
- **Cold-start** elevated scores may increase false positives during onboarding
- **Insider drift** is intentionally ambiguous for false-positive tuning

---

## 🔮 Future Improvements

- **Graph Neural Networks** for entity-resource relationship modeling
- **Real-time streaming** via Apache Kafka + Flink
- **LSTM/Transformer** sequence-aware command anomaly detection
- **Online learning** for adaptive baseline updates
- **Federated deployment** across distributed organizational units

---

## 📄 License

This project was developed as part of a cybersecurity assessment challenge.
