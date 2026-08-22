"""
Infrastructure Intelligence - Streamlit Dashboard UI Utilities & Styling
Provides enterprise dark theme styling, metric cards, and Plotly visual helpers.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, List


def apply_custom_css():
    """Inject modern enterprise dark theme CSS."""
    st.markdown("""
        <style>
        /* Main Container Styling */
        .stApp {
            background-color: #0E1117;
            color: #E0E6ED;
        }

        /* Metric Cards */
        .kpi-card {
            background: linear-gradient(135deg, #1E232A 0%, #161A1F 100%);
            border: 1px solid #2C323B;
            border-radius: 10px;
            padding: 18px 22px;
            margin-bottom: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        .kpi-title {
            color: #8C9BAE;
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .kpi-value {
            color: #FFFFFF;
            font-size: 28px;
            font-weight: 700;
            margin-top: 4px;
        }
        .kpi-sub {
            font-size: 12px;
            margin-top: 4px;
        }

        /* Risk Level Badges */
        .badge-low {
            background-color: rgba(0, 230, 118, 0.15);
            color: #00E676;
            border: 1px solid #00E676;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 700;
            display: inline-block;
        }
        .badge-medium {
            background-color: rgba(255, 179, 0, 0.15);
            color: #FFB300;
            border: 1px solid #FFB300;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 700;
            display: inline-block;
        }
        .badge-high {
            background-color: rgba(255, 82, 82, 0.15);
            color: #FF5252;
            border: 1px solid #FF5252;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 700;
            display: inline-block;
        }

        /* Action Recommendation Cards */
        .action-card {
            background: #181C22;
            border-left: 4px solid #00B0FF;
            padding: 14px 18px;
            margin-bottom: 10px;
            border-radius: 4px;
        }
        .action-card-high {
            border-left: 4px solid #FF5252;
            background: rgba(255, 82, 82, 0.08);
        }
        .action-category {
            color: #FFFFFF;
            font-size: 14px;
            font-weight: 700;
        }
        .action-desc {
            color: #B0BEC5;
            font-size: 13px;
            margin-top: 4px;
        }
        </style>
    """, unsafe_allow_html=True)


def render_kpi_card(title: str, value: str, subtext: str = "", sub_color: str = "#8C9BAE"):
    """Render sleek metric card component."""
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub" style="color: {sub_color};">{subtext}</div>
        </div>
    """, unsafe_allow_html=True)


def plot_gauge_chart(value: float, title: str, min_val: float = 0, max_val: float = 100, threshold: float = 80):
    """Render custom Plotly gauge meter."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14, 'color': '#FFFFFF'}},
        number={'suffix': "%" if max_val == 100 else "", 'font': {'color': '#FFFFFF', 'size': 24}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickcolor': "#8C9BAE"},
            'bar': {'color': "#FF5252" if value >= threshold else "#00B0FF"},
            'bgcolor': "#1E232A",
            'borderwidth': 1,
            'bordercolor': "#2C323B",
            'steps': [
                {'range': [min_val, max_val * 0.6], 'color': 'rgba(0, 230, 118, 0.1)'},
                {'range': [max_val * 0.6, max_val * 0.85], 'color': 'rgba(255, 179, 0, 0.1)'},
                {'range': [max_val * 0.85, max_val], 'color': 'rgba(255, 82, 82, 0.1)'}
            ],
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#FFFFFF"},
        margin=dict(l=20, r=20, t=30, b=20),
        height=180
    )
    return fig
