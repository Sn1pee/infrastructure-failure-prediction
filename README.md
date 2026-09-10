# ⚡ Infrastructure Intelligence: AI-Powered IT Infrastructure Failure Prediction & Root-Cause Analytics

> **End-to-End Data Science & Machine Learning Portfolio Project**  
> *Targeted for World Wide Technology (WWT) Data Science & Data Analytics Internship Application*

---

## 📌 Executive Summary & Business Problem

Large enterprise environments operate tens of thousands of physical servers, cloud instances, database clusters, and network switches. These systems continuously stream telemetry data (CPU load, memory pressure, disk I/O, network latency, error rates, and temperature).

Unplanned infrastructure downtime causes:
- Service outages and compromised customer experience
- Severe enterprise Service Level Agreement (SLA) financial penalties
- Reactive operational overhead for IT & DevOps engineering teams

**Infrastructure Intelligence** is an enterprise-grade end-to-end data science system that predicts infrastructure failures before they occur, detects anomalous system behavior, explains prediction root causes using SHAP, and delivers real-time actionable operational recommendations via an interactive Streamlit dashboard.

---

## 🏗️ End-to-End System Architecture

```
                                 ┌──────────────────────────────────────────────┐
                                 │   Synthetic Telemetry Generator (60,000 Rows)│
                                 └──────────────────────┬───────────────────────┘
                                                        │
                                                        ▼
                                 ┌──────────────────────────────────────────────┐
                                 │      ETL Pipeline & Data Validation          │
                                 └──────────────────────┬───────────────────────┘
                                                        │
                                    ┌───────────────────┴───────────────────┐
                                    ▼                                       ▼
                       ┌─────────────────────────┐             ┌─────────────────────────┐
                       │  SQLite Database        │             │  Feature Engineering    │
                       │  (sql/analytics.sql)    │             │  (Rolling 3h/6h, Stress)│
                       └─────────────────────────┘             └────────────┬────────────┘
                                                                            │
                                                        ┌───────────────────┴───────────────────┐
                                                        ▼                                       ▼
                                           ┌─────────────────────────┐             ┌─────────────────────────┐
                                           │ Supervised ML Models    │             │ Unsupervised Anomaly    │
                                           │ (XGBoost, RF, LR)       │             │ (Isolation Forest)      │
                                           └────────────┬────────────┘             └────────────┬────────────┘
                                                        │                                       │
                                                        ▼                                       ▼
                                           ┌─────────────────────────┐             ┌─────────────────────────┐
                                           │ SHAP Explainability     │             │ Risk Scoring Engine     │
                                           │ (Root Cause Analysis)   │             │ (LOW / MEDIUM / HIGH)   │
                                           └────────────┬────────────┘             └────────────┬────────────┘
                                                        │                                       │
                                                        └───────────────────┬───────────────────┘
                                                                            │
                                                                            ▼
                                                           ┌─────────────────────────────────┐
                                                           │ Streamlit Interactive Dashboard │
                                                           │ (5 Executive Analytics Views)   │
                                                           └─────────────────────────────────┘
```

---

## 📊 Dataset & Physical Correlations

The telemetry dataset consists of **60,000 observations** sampled hourly across **100 enterprise servers** over a 600-hour period.

### Telemetry Attributes:
- **Identifier & Categorical**: `timestamp`, `server_id`, `server_type` (Database, Web Server, App Server, Storage, Compute Node), `region` (us-east-1, us-west-2, eu-central-1, ap-southeast-1).
- **Physical Metrics**: `cpu_usage` (%), `memory_usage` (%), `disk_usage` (%), `network_latency` (ms), `packet_loss` (%), `request_rate` (req/s), `error_rate` (errors/s), `active_connections`, `temperature` (°C), `uptime_hours`.
- **Contextual Signals**: `workload_intensity`, `previous_failures`, `maintenance_flag`.
- **Target Variable**: `failure` (Binary 0 or 1, imbalanced ~6.5% positive failure rate).

---

## ⚙️ Data Engineering & Feature Engineering

### 1. Data Cleaning & Validation
- Standardized timestamps and enforced physiological bounds on continuous telemetry.
- Ingested cleaned data into an indexed SQLite database (`data/processed/infrastructure.db`).
- Executed production SQL analytical queries (`sql/analytics.sql`) calculating pre-failure telemetry statistics and high-risk leaderboards.

### 2. Feature Engineering
- **Rolling Time-Series Metrics**: 3-hour and 6-hour rolling mean and standard deviation for CPU, Memory, Latency, and Error Rate calculated per server.
- **Stress Interaction Terms**:
  - $\text{CPU-Memory Stress} = (\text{CPU} \times \text{Memory}) / 100$
  - $\text{Network Stress Score} = \text{Latency} \times (1 + \text{Packet Loss})$
  - $\text{System Utilization Index} = (\text{CPU} + \text{Memory} + \text{Disk}) / 3$
  - $\text{Thermal Efficiency Delta} = \text{Temperature} - (30 + 0.45 \times \text{CPU})$
- **Temporal Split**: Chronological 70% Train, 15% Validation, and 15% Test split to strictly prevent look-ahead data leakage.

---

## 🤖 Machine Learning Benchmarks

We evaluated four candidate classification architectures using class-imbalance techniques (`scale_pos_weight`, decision threshold optimization):

| Model | Decision Threshold | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier (Tuned)** | **0.48** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Random Forest** | 0.57 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Decision Tree** | 0.05 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Logistic Regression (Baseline)** | 0.94 | 0.9780 | 0.9762 | 0.9771 | 0.9995 | 0.9981 |

---

## 🔍 Explainable AI (SHAP) & Anomaly Detection

### SHAP TreeExplainer
Provides local instance-level feature attributions explaining *why* a specific server is considered high risk (e.g. Failure Probability 87% driven by +0.35 CPU saturation, +0.28 memory leak, +0.15 network latency).

### Isolation Forest Anomaly Detection
Unsupervised anomaly detection trained on continuous telemetry identifying unusual system behavior even when explicit failure labels have not been triggered.

---

## 🛠️ Installation & Local Execution

### Prerequisites
- Python 3.10+
- Git

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/your-username/infrastructure-intelligence.git
cd infrastructure-intelligence
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Data Pipeline & Train Models
```bash
python -m src.data.ingestion
python -m src.data.preprocessing
python -m src.features.feature_engineering
python -m src.models.train
python -m src.anomaly.detection
```

### 3. Run Pytest Suite
```bash
python -m pytest tests/ -v
```

### 4. Launch Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t infrastructure-intelligence .
```

### Run Docker Container
```bash
docker run -d -p 8501:8501 --name infra-app infrastructure-intelligence
```
Access dashboard at `http://localhost:8501`.

---

## 📂 Project Repository Structure

```
infrastructure-intelligence/
├── data/
│   ├── raw/                  # Raw telemetry CSV
│   └── processed/            # Cleaned data, features CSV, infrastructure.db
├── notebooks/
│   ├── run_eda.py            # EDA visualization script
│   └── eda_plots/            # Generated EDA plot outputs
├── src/
│   ├── data/                 # Ingestion & Preprocessing scripts
│   ├── features/             # Feature engineering & temporal split
│   ├── models/               # Model training, evaluation & inference engine
│   ├── explainability/       # SHAP XAI module
│   ├── anomaly/              # Isolation Forest anomaly detector
│   └── recommendations/      # Action recommendation engine
├── dashboard/
│   ├── app.py                # Streamlit executive dashboard
│   └── utils.py              # UI styling & Plotly chart helpers
├── sql/
│   ├── schema.sql            # SQLite DDL schema
│   └── analytics.sql         # Production SQL analytical queries
├── tests/                    # Pytest test suite
├── models/                   # Serialized model pickles & metadata
├── Dockerfile                # Docker container configuration
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```
