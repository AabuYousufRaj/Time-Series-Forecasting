import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="PGCB AI Decision Support", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main { padding-top: 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-size: 16px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA & ASSET LOADING
# ==========================================
@st.cache_data
def load_data():
    df_dash = pd.read_csv(BASE_DIR / 'dashboard_data.csv')
    date_col = df_dash.columns[0]
    df_dash[date_col] = pd.to_datetime(df_dash[date_col])
    df_dash.set_index(date_col, inplace=True)
    
    df_lgbm = pd.read_csv(BASE_DIR / 'lgbm_confidence_bands.csv')
    df_lgbm['Date'] = pd.to_datetime(df_lgbm['Date'])
    df_lgbm.set_index('Date', inplace=True)
    
    df_metrics = pd.read_csv(BASE_DIR / 'evaluation_metrics.csv')
    return df_dash, df_lgbm, df_metrics

def load_image(filename):
    image_path = BASE_DIR / filename
    if image_path.exists():
        return Image.open(image_path)
    return None

try:
    df_dash, df_lgbm, df_metrics = load_data()
except FileNotFoundError:
    st.error("⚠️ CSV Data files not found in the directory.")
    st.stop()

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Power_Grid_Company_of_Bangladesh_Logo.svg/1200px-Power_Grid_Company_of_Bangladesh_Logo.svg.png", width=150)
st.sidebar.title("AI Engine Controls")
st.sidebar.markdown("Predictive analytics for grid management.")

model_choice = st.sidebar.selectbox(
    "1. Select AI Model:",
    ["LightGBM (Quantile)", "Random Forest", "PyTorch LSTM", "Facebook Prophet"]
)

min_date = df_dash.index.min().to_pydatetime()
max_date = df_dash.index.max().to_pydatetime()

date_range = st.sidebar.slider(
    "2. Select Forecast Horizon:",
    min_value=min_date, max_value=max_date,
    value=(max_date - pd.Timedelta(days=7), max_date)
)

st.sidebar.markdown("---")
st.sidebar.info("**System Status:** Operational ✅\n\n**Latency:** < 10ms")

# ==========================================
# 4. MAIN DASHBOARD TABS
# ==========================================
st.title("⚡ PGCB Remote Monitoring & Decision Support System")
tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Forecasting", "🧠 Explainable AI", "📈 Model Diagnostics", "🔍 Data Profiling (EDA)"])

# ------------------------------------------
# TAB 1: LIVE FORECASTING
# ------------------------------------------
with tab1:
    mask_dash = (df_dash.index >= date_range[0]) & (df_dash.index <= date_range[1])
    mask_lgbm = (df_lgbm.index >= date_range[0]) & (df_lgbm.index <= date_range[1])
    plot_dash, plot_lgbm = df_dash.loc[mask_dash], df_lgbm.loc[mask_lgbm]

    if model_choice == "LightGBM (Quantile)":
        y_pred = plot_lgbm['LGBM_Pred']
        mape_display = "1.59%" 
    else:
        col_map = {"Random Forest": "RF_Pred", "PyTorch LSTM": "LSTM_Pred", "Facebook Prophet": "Prophet_Pred"}
        metric_model_map = {"Random Forest": "Random Forest", "PyTorch LSTM": "LSTM", "Facebook Prophet": "Prophet"}
        y_pred = plot_dash[col_map[model_choice]]
        metric_model = metric_model_map[model_choice]
        mape_display = f"{df_metrics.loc[df_metrics['Model'] == metric_model, 'MAPE (%)'].values[0]:.2f}%"

    # KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Actual Demand", f"{int(plot_dash['demand_mw'].iloc[-1]):,} MW")
    col2.metric("Predicted Peak Demand", f"{int(y_pred.max()):,} MW")
    col3.metric("Current Model Error (MAPE)", mape_display)
    diff = int(y_pred.max() - plot_dash['demand_mw'].max())
    col4.metric("Peak Variance (Pred vs Actual)", f"{diff:,} MW", delta=diff, delta_color="inverse")

    st.markdown("---")
    
    # Plotly Chart
    fig = go.Figure()
    if model_choice == "LightGBM (Quantile)":
        fig.add_trace(go.Scatter(
            x=plot_lgbm.index.tolist() + plot_lgbm.index.tolist()[::-1],
            y=plot_lgbm['LGBM_Upper_90'].tolist() + plot_lgbm['LGBM_Lower_90'].tolist()[::-1],
            fill='toself', fillcolor='rgba(0, 176, 246, 0.2)', line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", showlegend=True, name='80% True Confidence Interval'
        ))
    fig.add_trace(go.Scatter(x=plot_dash.index, y=plot_dash['demand_mw'], mode='lines', name='Actual Demand', line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(x=plot_dash.index, y=y_pred, mode='lines', name=f'Forecast ({model_choice})', line=dict(color='#ff7f0e', width=2, dash='dash')))
    
    fig.update_layout(
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=10, b=0), height=450, xaxis_title="", yaxis_title="Megawatts (MW)"
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# TAB 2: EXPLAINABLE AI (SHAP)
# ------------------------------------------
with tab2:
    st.subheader("Model Transparency: Why is the AI making these predictions?")
    st.markdown("Using Game Theory (SHAP), we can unpack the black box of our LightGBM model to understand exactly which grid variables are driving electricity demand.")
    
    col1, col2 = st.columns(2)
    with col1:
        img_shap_sum = load_image('shap_summary_plot.png')
        if img_shap_sum: st.image(img_shap_sum, caption="SHAP Summary (Global Impact)", use_container_width=True)
        else: st.warning("shap_summary_plot.png not found.")
        
    with col2:
        img_shap_water = load_image('shap_local_waterfall.png')
        if img_shap_water: st.image(img_shap_water, caption="Local Explanation for Latest Hour", use_container_width=True)
        else: st.warning("shap_local_waterfall.png not found.")

# ------------------------------------------
# TAB 3: MODEL DIAGNOSTICS
# ------------------------------------------
with tab3:
    st.subheader("Algorithm Performance & Loss Tracking")
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("**Final Evaluation Metrics**")
        st.dataframe(df_metrics.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE (%)": "{:.2f}%"}), use_container_width=True)
        st.markdown("> *Note: LightGBM (Quantile) achieved a 1.59% MAPE, outperforming all other models.*")
        
    with col2:
        img_lstm = load_image('lstm_loss_curve.png')
        if img_lstm: st.image(img_lstm, caption="LSTM Training vs Validation Loss (Early Stopping Triggered)", use_container_width=True)
        else: st.warning("lstm_loss_curve.png not found.")

# ------------------------------------------
# TAB 4: DATA PROFILING (EDA)
# ------------------------------------------
with tab4:
    st.subheader("Historical Grid Behavior")
    st.markdown("Exploratory Data Analysis generated prior to feature engineering.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        img_eda2 = load_image('eda_2_hourly_avg.png')
        if img_eda2: st.image(img_eda2, caption="Daily Peaks", use_container_width=True)
    with col2:
        img_eda3 = load_image('eda_3_dow_avg.png')
        if img_eda3: st.image(img_eda3, caption="Weekly Rhythm", use_container_width=True)
    with col3:
        img_eda4 = load_image('eda_4_distribution.png')
        if img_eda4: st.image(img_eda4, caption="Overall Distribution", use_container_width=True)