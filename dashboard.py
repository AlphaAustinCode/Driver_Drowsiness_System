import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configure page layout
st.set_page_config(
    page_title="Driver Drowsiness System Dashboard",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    
    div[data-testid="stMetric"] {
        background-color: #1a1c24;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
    }
    
    .metric-alert-high {
        color: #FF4B4B !important;
        font-weight: 700;
    }

    h1, h2, h3 { 
        font-family: 'Inter', sans-serif; 
        font-weight: 600 !important; 
        letter-spacing: -0.5px; 
    }
    
    section[data-testid="stSidebar"] { 
        border-right: 1px solid rgba(255,255,255,0.05); 
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🚘 Driver Drowsiness System")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Dashboard", "Analytics", "Settings"])
st.sidebar.markdown("---")

# Database connection
import sqlite3
def get_db_connection():
    return sqlite3.connect('drowsiness.db')

# Load data functions
@st.cache_data(ttl=30)
def load_overview_stats():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Total Sessions
        cursor.execute("SELECT COUNT(id) FROM sessions")
        total_sessions = cursor.fetchone()[0] or 0
        
        # Total Alerts
        cursor.execute("SELECT COUNT(id) FROM alerts")
        total_alerts = cursor.fetchone()[0] or 0
        
        # Average EAR
        cursor.execute("SELECT AVG(ear_value) FROM frame_logs ORDER BY id DESC LIMIT 1000")
        avg_ear = cursor.fetchone()[0] or 0.28
        
        conn.close()
        
        return {
            "Total Sessions": total_sessions,
            "Total Alerts": total_alerts,
            "Avg. EAR": round(avg_ear, 3),
            "System Status": "Active" if total_sessions > 0 else "Inactive"
        }
    except Exception as e:
        st.error(f"Database error: {e}")
        return {"Total Sessions": 0, "Total Alerts": 0, "Avg. EAR": 0.0, "System Status": "Error"}

@st.cache_data(ttl=30)
def load_recent_alerts(limit=20):
    try:
        conn = get_db_connection()
        query = """
        SELECT 
            datetime(timestamp) as Time,
            alert_type as Type,
            severity as Severity
        FROM alerts
        ORDER BY timestamp DESC
        LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=[limit])
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_alert_trends():
    try:
        conn = get_db_connection()
        query = """
        SELECT 
            date(timestamp) as Date,
            alert_type,
            COUNT(*) as Count
        FROM alerts
        WHERE timestamp >= date('now', '-7 days')
        GROUP BY Date, alert_type
        ORDER BY Date
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

# Main content
if page == "Dashboard":
    st.title("🚘 Driver Drowsiness Dashboard")
    
    # KPIs
    stats = load_overview_stats()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sessions", stats["Total Sessions"])
    
    with col2:
        alerts = stats["Total Alerts"]
        value_color = 'var(--text-color)' if alerts <= 50 else '#FF4B4B'
        st.markdown(f'<div data-testid="stMetricValue" style="color: {value_color}; font-size: 1.8rem; font-weight: 600;">{alerts}</div>', unsafe_allow_html=True)
        st.markdown('<div style="color: gray; font-size: 0.9rem;">Total Alerts</div>', unsafe_allow_html=True)
    
    with col3:
        st.metric("Avg. EAR", f"{stats['Avg. EAR']:.3f}")
    
    with col4:
        status_color = "#22c55e" if stats["System Status"] == "Active" else "#FF4B4B"
        st.markdown(f'<div style="color: {status_color}; font-size: 1.8rem; font-weight: 600;">● {stats["System Status"]}</div>', unsafe_allow_html=True)
        st.markdown('<div style="color: gray; font-size: 0.9rem;">System Status</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts and Recent Alerts
    col_chart, col_alerts = st.columns([2, 1])
    
    with col_chart:
        st.subheader("Alert Trends (Last 7 Days)")
        df_trends = load_alert_trends()
        
        if not df_trends.empty:
            fig = px.area(
                df_trends, 
                x="Date", 
                y="Count", 
                color="alert_type",
                title="Alert Frequency Over Time",
                template="plotly_dark"
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No alert data available for the last 7 days.")
    
    with col_alerts:
        st.subheader("Recent Alerts")
        df_alerts = load_recent_alerts()
        
        if not df_alerts.empty:
            st.dataframe(df_alerts, use_container_width=True, hide_index=True)
        else:
            st.info("No recent alerts recorded.")
    
    # System Metrics
    st.subheader("System Metrics", divider="gray")
    
    # Create sample gauge charts
    def create_gauge(title, value, max_val, color):
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = value,
            title = {'text': title, 'font': {'size': 16, 'color': '#FAFAFA'}},
            gauge = {
                'axis': {'range': [0, max_val], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': color},
                'bgcolor': "#1E1E1E",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, max_val*0.5], 'color': 'rgba(255,255,255,0.02)'},
                    {'range': [max_val*0.5, max_val], 'color': 'rgba(255,255,255,0.08)'}
                ],
            }
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=200,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        return fig
    
    gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
    
    with gauge_col1:
        st.plotly_chart(create_gauge("Eye Aspect Ratio", 0.28, 0.5, "#00D4FF"), use_container_width=True)
    
    with gauge_col2:
        st.plotly_chart(create_gauge("Mouth Aspect Ratio", 0.3, 1.0, "#facc15"), use_container_width=True)
    
    with gauge_col3:
        st.plotly_chart(create_gauge("System Confidence", 85, 100, "#22c55e"), use_container_width=True)

elif page == "Analytics":
    st.title("📊 Analytics")
    st.info("Advanced analytics features coming soon...")
    
elif page == "Settings":
    st.title("⚙️ Settings")
    st.info("System settings and configuration options coming soon...")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Driver Drowsiness Detection System v1.0")
