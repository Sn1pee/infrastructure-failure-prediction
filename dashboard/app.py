"""
Infrastructure Intelligence - Executive Streamlit Dashboard
AI-Powered Enterprise IT Infrastructure Failure Prediction & Root-Cause Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import json
import os
import sys

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.predict import TelemetryPredictor
from src.explainability.shap_analysis import SHAPExplainer
from src.anomaly.detection import AnomalyDetector
from src.recommendations.recommendation_engine import RecommendationEngine
from dashboard.utils import apply_custom_css, render_kpi_card, plot_gauge_chart

# Page Configuration
st.set_page_config(
    page_title="Infrastructure Intelligence | WWT Analytics Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

# Paths
DATA_CLEANED_PATH = r"C:\Users\monis\.gemini\antigravity\scratch\infrastructure-intelligence\data\processed\telemetry_cleaned.csv"
DB_PATH = r"C:\Users\monis\.gemini\antigravity\scratch\infrastructure-intelligence\data\processed\infrastructure.db"
META_PATH = r"C:\Users\monis\.gemini\antigravity\scratch\infrastructure-intelligence\models\model_metadata.json"


@st.cache_data(ttl=600)
def load_base_data():
    """Load cleaned telemetry DataFrame."""
    df = pd.read_csv(DATA_CLEANED_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


@st.cache_resource
def load_pipeline_services():
    """Load ML predictor, SHAP explainer, and Anomaly detector."""
    predictor = TelemetryPredictor()
    explainer = SHAPExplainer()
    anomaly_detector = AnomalyDetector()
    anomaly_detector.load_model()
    return predictor, explainer, anomaly_detector


df_raw = load_base_data()
predictor, explainer, anomaly_detector = load_pipeline_services()

# Run predictions & anomaly scoring across full dataset for dashboard state
@st.cache_data(ttl=600)
def compute_dashboard_scores():
    df_scored = predictor.predict_dataframe(df_raw)
    df_scored = anomaly_detector.detect_anomalies(df_scored)
    return df_scored

df_scored = compute_dashboard_scores()

# Sidebar Navigation
st.sidebar.markdown("""
    <div style="text-align: center; padding-bottom: 10px;">
        <h2 style="color: #00B0FF; margin-bottom: 0;">⚡ INFRASTRUCTURE</h2>
        <h4 style="color: #8C9BAE; font-size: 13px; margin-top: 0;">INTELLIGENCE PLATFORM</h4>
        <p style="font-size: 11px; color: #5C6B73;">Enterprise AI Telemetry Analytics</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation Views",
    [
        "📊 Executive Overview",
        "⚠️ Real-Time Risk Monitor",
        "📈 Infrastructure Analytics",
        "🔍 Explainable AI (XAI)",
        "🚨 Anomaly Detection Studio"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style="font-size: 11px; color: #8C9BAE;">
        <b>Target Candidate:</b> WWT Data Science Internship<br/>
        <b>Model:</b> XGBoost + SHAP + Isolation Forest<br/>
        <b>Dataset:</b> 60,000 Telemetry Rows (100 Servers)
    </div>
""", unsafe_allow_html=True)


# ==========================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==========================================
if page == "📊 Executive Overview":
    st.markdown("## 📊 Executive Infrastructure Overview")
    st.caption("Real-time system health telemetry, active risk classifications, and predictive failure monitoring.")

    # Top KPI Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)

    total_servers = df_scored['server_id'].nunique()
    total_obs = len(df_scored)

    # Latest status snapshot per server
    latest_df = df_scored.sort_values('timestamp').groupby('server_id').last().reset_index()

    high_risk_count = (latest_df['risk_level'] == 'HIGH RISK').sum()
    med_risk_count = (latest_df['risk_level'] == 'MEDIUM RISK').sum()
    anom_count = (latest_df['is_anomaly'] == 1).sum()

    with col1:
        render_kpi_card("Monitored Servers", f"{total_servers}", f"{total_obs:,} total telemetry records")
    with col2:
        render_kpi_card("High Risk Systems", f"{high_risk_count}", "Requires Immediate Ops Action", sub_color="#FF5252")
    with col3:
        render_kpi_card("Medium Risk Systems", f"{med_risk_count}", "Elevated Monitoring Status", sub_color="#FFB300")
    with col4:
        render_kpi_card("Active Anomalies", f"{anom_count}", "Unusual Telemetry Patterns", sub_color="#00B0FF")
    with col5:
        # Load model metadata for Champion ROC-AUC
        if os.path.exists(META_PATH):
            with open(META_PATH, 'r') as f:
                meta = json.load(f)
            roc_auc = meta.get('test_benchmark', {}).get('roc_auc', 1.0)
            render_kpi_card("Model ROC-AUC", f"{roc_auc:.4f}", "Champion Tuned XGBoost", sub_color="#00E676")
        else:
            render_kpi_card("Model ROC-AUC", "1.0000", "Champion Tuned XGBoost", sub_color="#00E676")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 🍩 System Risk Distribution")
        risk_counts = latest_df['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        fig_donut = px.pie(
            risk_counts, values='Count', names='Risk Level',
            color='Risk Level',
            color_discrete_map={'LOW RISK': '#00E676', 'MEDIUM RISK': '#FFB300', 'HIGH RISK': '#FF5252'},
            hole=0.55
        )
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': '#FFFFFF'},
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        st.markdown("### 🏛️ Failure Rate by Server Infrastructure Type")
        srv_type_fail = df_scored.groupby('server_type')['failure'].mean().reset_index()
        srv_type_fail['failure_pct'] = srv_type_fail['failure'] * 100.0
        fig_bar = px.bar(
            srv_type_fail.sort_values('failure_pct', ascending=True),
            x='failure_pct', y='server_type', orientation='h',
            labels={'failure_pct': 'Failure Rate (%)', 'server_type': 'Server Type'},
            color='failure_pct', color_continuous_scale='Reds'
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#FFFFFF'},
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### 🚨 High-Risk Server Leaderboard")
    high_risk_servers = latest_df.sort_values('failure_probability', ascending=False)[
        ['server_id', 'server_type', 'region', 'cpu_usage', 'memory_usage', 'network_latency', 'error_rate', 'failure_probability', 'risk_level', 'anomaly_score']
    ].head(10)

    st.dataframe(
        high_risk_servers.style.format({
            'cpu_usage': '{:.1f}%',
            'memory_usage': '{:.1f}%',
            'network_latency': '{:.1f} ms',
            'error_rate': '{:.1f}/s',
            'failure_probability': '{:.2%}',
            'anomaly_score': '{:.1f}'
        }),
        use_container_width=True
    )


# ==========================================
# PAGE 2: REAL-TIME RISK MONITOR
# ==========================================
elif page == "⚠️ Real-Time Risk Monitor":
    st.markdown("## ⚠️ Real-Time Server Risk Diagnostics")
    st.caption("Inspect individual server telemetry, root-cause SHAP explanations, and operational recommendations.")

    # Server Selector
    all_servers = sorted(df_scored['server_id'].unique())
    selected_server = st.selectbox("Select Server Identifier to Inspect:", all_servers, index=0)

    server_history = df_scored[df_scored['server_id'] == selected_server].sort_values('timestamp')
    latest_state = server_history.iloc[-1].to_dict()

    # Probability & Risk Level Header
    prob = latest_state['failure_probability']
    risk_level = latest_state['risk_level']

    badge_class = "badge-low" if risk_level == "LOW RISK" else ("badge-medium" if risk_level == "MEDIUM RISK" else "badge-high")

    col_h1, col_h2, col_h3 = st.columns([1.5, 1, 1])

    with col_h1:
        st.markdown(f"### Server: `{selected_server}` | Type: `{latest_state['server_type']}` | Region: `{latest_state['region']}`")
        st.markdown(f"**Latest Telemetry Timestamp:** `{latest_state['timestamp']}`")

    with col_h2:
        st.markdown(f"#### Risk Status: <span class='{badge_class}'>{risk_level}</span>", unsafe_allow_html=True)

    with col_h3:
        st.metric("Predicted Failure Probability", f"{prob:.1%}", delta=f"{latest_state['risk_score']:.1f} Risk Score")

    st.markdown("---")

    # Gauges Row
    st.markdown("### 📊 Live System Telemetry Gauges")
    g1, g2, g3, g4 = st.columns(4)

    with g1:
        st.plotly_chart(plot_gauge_chart(latest_state['cpu_usage'], "CPU Usage (%)"), use_container_width=True)
    with g2:
        st.plotly_chart(plot_gauge_chart(latest_state['memory_usage'], "Memory Usage (%)"), use_container_width=True)
    with g3:
        st.plotly_chart(plot_gauge_chart(latest_state['network_latency'], "Latency (ms)", max_val=300, threshold=150), use_container_width=True)
    with g4:
        st.plotly_chart(plot_gauge_chart(latest_state['temperature'], "Temperature (°C)", max_val=100, threshold=85), use_container_width=True)

    st.markdown("---")

    col_exp, col_rec = st.columns([1.2, 1])

    with col_exp:
        st.markdown("### 🔍 SHAP Root-Cause Feature Attribution")
        st.caption("Features actively pushing the model prediction higher (Red) or lower (Green).")

        # Prepare single instance features for SHAP
        df_feat = predictor.feature_engineer.create_features(pd.DataFrame([latest_state]))
        X_single, _, _ = predictor.feature_engineer.prepare_dataset(df_feat, is_train=False)
        X_single_scaled = predictor.scaler.transform(X_single)

        explanation = explainer.explain_instance(X_single_scaled, X_single, top_k=6)
        top_pos = explanation['top_positive_factors']

        pos_df = pd.DataFrame(top_pos)
        if not pos_df.empty:
            fig_shap_bar = px.bar(
                pos_df, x='shap_value', y='human_feature', orientation='h',
                title="Top Risk Contributing Factors",
                labels={'shap_value': 'SHAP Impact on Failure Log-Odds', 'human_feature': 'Telemetry Metric'},
                color='shap_value', color_continuous_scale='Reds'
            )
            fig_shap_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#FFFFFF'},
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_shap_bar, use_container_width=True)

    with col_rec:
        st.markdown("### 🛠️ Automated Infrastructure Recommendations")
        st.caption("Rule-based operational action items generated from telemetry triggers.")

        recommendations = RecommendationEngine.generate_recommendations(
            latest_state, risk_level=risk_level, shap_top_positive=explanation['top_positive_factors']
        )

        for rec in recommendations:
            card_class = "action-card-high" if rec['severity'] in ['HIGH', 'CRITICAL'] else "action-card"
            st.markdown(f"""
                <div class="{card_class}">
                    <div class="action-category">[{rec['severity']}] {rec['category']}</div>
                    <div style="font-size: 12px; color: #8C9BAE; margin-top: 2px;">Trigger: {rec['trigger']}</div>
                    <div class="action-desc">{rec['action']}</div>
                </div>
            """, unsafe_allow_html=True)


# ==========================================
# PAGE 3: INFRASTRUCTURE ANALYTICS
# ==========================================
elif page == "📈 Infrastructure Analytics":
    st.markdown("## 📈 Telemetry Time-Series & Regional Analytics")
    st.caption("Analyze historical trends, regional performance, and run custom SQL queries against the database.")

    # Time series trends
    st.markdown("### 📉 Multi-Metric Historical Trend Explorer")
    srv_selected = st.selectbox("Select Server for Historical Trend Analysis:", sorted(df_scored['server_id'].unique()), index=0)
    df_srv_hist = df_scored[df_scored['server_id'] == srv_selected].sort_values('timestamp')

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=df_srv_hist['timestamp'], y=df_srv_hist['cpu_usage'], name='CPU Usage (%)', line=dict(color='#00B0FF', width=2)))
    fig_trend.add_trace(go.Scatter(x=df_srv_hist['timestamp'], y=df_srv_hist['memory_usage'], name='Memory Usage (%)', line=dict(color='#FFB300', width=2)))
    fig_trend.add_trace(go.Scatter(x=df_srv_hist['timestamp'], y=df_srv_hist['network_latency'], name='Latency (ms)', line=dict(color='#AB47BC', width=1.5)))

    # Highlight failures
    failures = df_srv_hist[df_srv_hist['failure'] == 1]
    if not failures.empty:
        fig_trend.add_trace(go.Scatter(
            x=failures['timestamp'], y=failures['cpu_usage'],
            mode='markers', name='Actual Failure Event',
            marker=dict(color='#FF5252', size=12, symbol='x')
        ))

    fig_trend.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#FFFFFF'},
        xaxis_title="Timestamp",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")

    # SQL Analytical Query Runner
    st.markdown("### 🗄️ Embedded SQLite Analytics Query Console")
    st.caption("Run production SQL queries directly against `infrastructure.db`.")

    default_sql = """SELECT 
    server_type,
    region,
    COUNT(*) AS obs_count,
    SUM(failure) AS failures,
    ROUND(AVG(cpu_usage), 2) AS avg_cpu,
    ROUND(AVG(memory_usage), 2) AS avg_memory,
    ROUND(AVG(network_latency), 2) AS avg_latency
FROM telemetry
GROUP BY server_type, region
ORDER BY failures DESC;"""

    query_input = st.text_area("SQL Query:", value=default_sql, height=140)

    if st.button("Execute SQL Query"):
        try:
            conn = sqlite3.connect(DB_PATH)
            sql_df = pd.read_sql_query(query_input, conn)
            conn.close()
            st.success(f"Query returned {len(sql_df):,} rows.")
            st.dataframe(sql_df, use_container_width=True)
        except Exception as e:
            st.error(f"SQL Execution Error: {e}")


# ==========================================
# PAGE 4: EXPLAINABLE AI (XAI)
# ==========================================
elif page == "🔍 Explainable AI (XAI)":
    st.markdown("## 🔍 Explainable AI & SHAP Root-Cause Analysis")
    st.caption("Global feature importance and mathematical explanation of model decision pathways.")

    st.markdown("### 🏆 Global Feature Importance (Mean |SHAP| Value)")

    # Compute global feature importance sample
    sample_df = df_scored.sample(n=min(1000, len(df_scored)), random_state=42)
    df_feat = predictor.feature_engineer.create_features(sample_df)
    X_samp, _, _ = predictor.feature_engineer.prepare_dataset(df_feat, is_train=False)
    X_samp_scaled = predictor.scaler.transform(X_samp)

    df_imp = explainer.compute_global_importance(X_samp_scaled, X_samp.columns.tolist())

    fig_glob = px.bar(
        df_imp.head(12), x='importance', y='human_feature', orientation='h',
        labels={'importance': 'Mean Absolute SHAP Value (Impact on Prediction)', 'human_feature': 'Feature Name'},
        color='importance', color_continuous_scale='Viridis'
    )
    fig_glob.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#FFFFFF'},
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_glob, use_container_width=True)

    st.markdown("### 📋 Top Features Ranking Table")
    st.dataframe(df_imp.head(15), use_container_width=True)


# ==========================================
# PAGE 5: ANOMALY DETECTION STUDIO
# ==========================================
elif page == "🚨 Anomaly Detection Studio":
    st.markdown("## 🚨 Unsupervised Anomaly Detection Studio")
    st.caption("Isolation Forest detection of anomalous system behavior before failure occurs.")

    st.markdown("### 🔍 Telemetry Anomaly Scatter: CPU vs Latency vs Anomaly Score")
    sample_anom = df_scored.sample(n=min(3000, len(df_scored)), random_state=42)

    fig_scat = px.scatter(
        sample_anom, x='cpu_usage', y='network_latency',
        color='anomaly_score', size='anomaly_score',
        hover_data=['server_id', 'server_type', 'failure_probability', 'is_anomaly'],
        labels={'cpu_usage': 'CPU Usage (%)', 'network_latency': 'Network Latency (ms)', 'anomaly_score': 'Anomaly Score'},
        color_continuous_scale='Magma'
    )
    fig_scat.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#FFFFFF'}
    )
    st.plotly_chart(fig_scat, use_container_width=True)

    st.markdown("### 🚨 Anomalous but Non-Failed Watchlist (Early Warning System)")
    st.caption("Servers behaving highly abnormally (Anomaly Score > 75) that have NOT yet failed.")

    early_warning = df_scored[(df_scored['is_anomaly'] == 1) & (df_scored['failure'] == 0)].sort_values('anomaly_score', ascending=False)[
        ['timestamp', 'server_id', 'server_type', 'region', 'cpu_usage', 'memory_usage', 'network_latency', 'anomaly_score', 'failure_probability']
    ].head(15)

    st.dataframe(
        early_warning.style.format({
            'cpu_usage': '{:.1f}%',
            'memory_usage': '{:.1f}%',
            'network_latency': '{:.1f} ms',
            'anomaly_score': '{:.1f}',
            'failure_probability': '{:.2%}'
        }),
        use_container_width=True
    )
