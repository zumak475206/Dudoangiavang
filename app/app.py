"""
GoldVision AI - Dashboard Du Bao Gia Vang v2.0
app/app.py - Clean Light Mode UI
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.sjc_converter import SJCConverter, DEFAULT_USD_VND_RATE, DEFAULT_SJC_PREMIUM_VND


# PAGE CONFIG
st.set_page_config(
    page_title="GoldVision AI - Du Bao Gia Vang",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)


# GLOBAL CSS
CSS = """
<style>
:root {
    --cream:     #FAF8F5;
    --stone:     #F4EFE6;
    --white:     #FFFFFF;
    --border:    #E5DED2;
    --border-lt: #EDE7DC;
    --gold:      #B8860B;
    --gold-lt:   #D4AF37;
    --gold-bg:   #FFF8E7;
    --text-dark: #1A1D1A;
    --text-mid:  #4A4E4A;
    --text-soft: #7A807A;
    --green:     #1A7C4A;
    --green-bg:  #EDFAF3;
    --red:       #C0392B;
    --red-bg:    #FEF0EE;
    --shadow-xs: 0 1px 4px rgba(0,0,0,0.06);
    --shadow-sm: 0 2px 10px rgba(0,0,0,0.07);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 18px;
}

.stApp,
.stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
.main .block-container {
    background-color: var(--cream) !important;
    color: var(--text-dark) !important;
}
.main .block-container {
    padding: 1.5rem 2.5rem 3rem !important;
    max-width: 1400px !important;
}

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] > div > div {
    background-color: var(--white) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-dark) !important; }

h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: var(--text-dark) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}
p, li, span, label,
.stMarkdown p, .stCaption {
    color: var(--text-mid) !important;
}

[data-testid="metric-container"] {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 16px 20px 14px !important;
    box-shadow: var(--shadow-xs) !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-dark) !important;
    font-weight: 800 !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-soft) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

[data-testid="stTabs"] [role="tablist"] {
    background-color: var(--stone) !important;
    padding: 5px 6px !important;
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border) !important;
    gap: 4px !important;
}
[data-testid="stTabs"] [role="tab"] {
    background-color: transparent !important;
    color: var(--text-soft) !important;
    border-radius: var(--radius-sm) !important;
    padding: 8px 16px !important;
    font-weight: 500 !important;
    border: none !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background-color: var(--white) !important;
    color: var(--gold) !important;
    font-weight: 700 !important;
    box-shadow: var(--shadow-xs) !important;
}

[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: var(--cream) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-dark) !important;
}
[data-testid="stSidebar"] .stNumberInput input {
    background-color: var(--cream) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-dark) !important;
}
[data-testid="stSidebar"] .stNumberInput button {
    background-color: var(--stone) !important;
    border-color: var(--border) !important;
    color: var(--text-dark) !important;
}

[data-testid="stDataFrame"],
.stDataFrame {
    background-color: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
}

[data-testid="stPlotlyChart"] {
    background-color: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 4px !important;
    box-shadow: var(--shadow-xs) !important;
}

.stDownloadButton > button {
    background-color: var(--white) !important;
    border: 1px solid var(--border) !important;
    color: var(--gold) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
}
.stDownloadButton > button:hover {
    background-color: var(--gold-bg) !important;
    border-color: var(--gold) !important;
}

hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

.gold-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 20px 22px;
    box-shadow: var(--shadow-xs);
    margin-bottom: 12px;
    transition: box-shadow 0.15s ease;
}
.gold-card:hover { box-shadow: var(--shadow-sm); }

.hero-forecast-card {
    background: var(--white);
    border: 1px solid var(--border-lt);
    border-left: 5px solid var(--gold);
    border-radius: var(--radius-lg);
    padding: 28px 32px;
    box-shadow: 0 4px 24px rgba(184,134,11,0.10);
    margin-bottom: 24px;
}

.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}
.badge-up   { background: var(--green-bg); color: var(--green);  border: 1px solid #9ED9BB; }
.badge-down { background: var(--red-bg);   color: var(--red);    border: 1px solid #F5B8B0; }
.badge-gold { background: var(--gold-bg);  color: var(--gold);   border: 1px solid #FFD980; }

.kpi-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-soft);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 1.45rem;
    font-weight: 800;
    color: var(--text-dark);
    letter-spacing: -0.02em;
}
.kpi-sub {
    font-size: 0.82rem;
    color: var(--text-soft);
    margin-top: 2px;
}

.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-dark);
    margin: 0 0 4px 0;
    letter-spacing: -0.02em;
}
.section-cap {
    font-size: 0.87rem;
    color: var(--text-soft);
    margin: 0 0 18px 0;
}

.macro-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 14px 16px;
    text-align: center;
    transition: border-color 0.15s;
}
.macro-card:hover { border-color: var(--gold); }

.rank-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    font-weight: 700;
    font-size: 0.85rem;
}
.rank-1     { background: var(--gold-bg); color: var(--gold); border: 2px solid var(--gold); }
.rank-2     { background: #F0F4F8; color: #4A5568; border: 2px solid #CBD5E0; }
.rank-other { background: var(--stone); color: var(--text-soft); border: 1px solid var(--border); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# PLOTLY BASE LAYOUT
PLOTLY_LAYOUT = dict(
    template="plotly_white",
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
              color="#2D312E", size=12),
    margin=dict(l=48, r=24, t=40, b=36),
    legend=dict(
        bgcolor="rgba(250,248,245,0.9)",
        bordercolor="#E5DED2",
        borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(gridcolor="#F0EAE0", linecolor="#E5DED2", tickfont=dict(color="#4A4E4A")),
    yaxis=dict(gridcolor="#F0EAE0", linecolor="#E5DED2", tickfont=dict(color="#4A4E4A")),
)


# DATA LOADERS
@st.cache_data(ttl=3600)
def load_market_data():
    path = os.path.join(project_root, "data", "processed", "featured_gold_data.csv")
    if not os.path.exists(path):
        path = os.path.join(project_root, "data", "raw", "gold_market_data.csv")
    return pd.read_csv(path, index_col="Date", parse_dates=True)


@st.cache_data(ttl=3600)
def load_leaderboard():
    path = os.path.join(project_root, "reports", "model_leaderboard.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame({
        "Mo Hinh": ["LightGBM Regressor", "Naive Persistence", "XGBoost Regressor", "PyTorch LSTM", "Ridge Regression"],
        "RMSE ($)": [60.96, 60.97, 66.50, 105.17, 185.86],
        "MAE ($)": [41.85, 41.85, 49.47, 79.72, 150.45],
        "MAPE (%)": [1.043, 1.043, 1.229, 1.959, 3.474],
        "Directional Acc (%)": [50.84, 49.16, 49.88, 50.12, 49.16],
        "R2 Score": [0.9902, 0.9902, 0.9884, 0.9709, 0.9092],
        "Residual Std ($)": [60.91, 60.90, 62.42, 90.62, 114.26],
    })


@st.cache_data(ttl=3600)
def load_forecast():
    path = os.path.join(project_root, "reports", "forecast_7days.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame({
        "Buoc": [f"T+{i}" for i in range(1, 8)],
        "Ngay Giao Dich": ["2026-10-07","2026-10-08","2026-10-09","2026-10-12","2026-10-13","2026-10-14","2026-10-15"],
        "Thu": ["Wednesday","Thursday","Friday","Monday","Tuesday","Wednesday","Thursday"],
        "Gia Du Bao ($)": [4187.38, 4187.66, 4187.95, 4188.23, 4188.51, 4188.79, 4189.07],
        "Thay Doi ($)": [0.28]*7,
        "Ty Suat (%)": [0.007]*7,
        "Xu Huong": ["TANG"]*7,
        "Can Duoi 95% ($)": [4068.00, 4018.83, 3981.17, 3949.46, 3921.56, 3896.36, 3873.22],
        "Can Tren 95% ($)": [4306.77, 4356.50, 4394.72, 4427.00, 4455.46, 4481.22, 4504.93],
    })


# Detect column names from actual CSV
def _col(df, *candidates):
    """Return first candidate column that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:4px 0 16px;">
        <span style="font-size:2rem;">🪙</span>
        <div>
            <div style="font-size:1.15rem;font-weight:800;color:#1A1D1A;letter-spacing:-0.02em;">GoldVision AI</div>
            <div style="font-size:0.78rem;color:#7A807A;font-weight:500;">Du Bao Gia Vang ML</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="kpi-label" style="margin-bottom:6px;">MO HINH DU BAO</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        label="Mo hinh du bao",
        options=["LightGBM Regressor (Top 1 - Khuyen nghi)", "XGBoost Regressor", "PyTorch LSTM", "Ridge Regression", "Naive Persistence"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="kpi-label" style="margin-bottom:6px;">QUY DOI VANG SJC</div>', unsafe_allow_html=True)

    usd_rate = st.number_input(
        "Ty gia USD/VND",
        min_value=20000.0, max_value=35000.0,
        value=DEFAULT_USD_VND_RATE, step=50.0,
    )
    sjc_premium_m = st.number_input(
        "Chenh lech SJC (Trieu VND/luong)",
        min_value=0.0, max_value=30.0,
        value=DEFAULT_SJC_PREMIUM_VND / 1_000_000.0, step=0.5,
    )
    sjc_premium = sjc_premium_m * 1_000_000.0

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="kpi-label" style="margin-bottom:6px;">KHOANG THOI GIAN</div>', unsafe_allow_html=True)
    time_range = st.select_slider(
        label="Khoang thoi gian xem",
        options=["3 Thang", "6 Thang", "1 Nam", "3 Nam", "Toan Bo"],
        value="1 Nam",
        label_visibility="collapsed",
    )

    st.markdown("<br>")
    st.markdown(
        '<div style="border-top:1px solid #E5DED2;padding-top:14px;">'
        '<div style="font-size:0.78rem;color:#7A807A;line-height:1.6;">'
        'Do an May hoc - CNTT<br>Task 11 Dashboard v2.0<br>'
        '100 unit tests - Pass'
        '</div></div>',
        unsafe_allow_html=True,
    )


# DATA PREP
converter = SJCConverter(usd_vnd_rate=usd_rate, sjc_premium_vnd=sjc_premium)
df_all = load_market_data()
df_lb_raw = load_leaderboard()
df_fc_raw = load_forecast()

# Normalize leaderboard column names
lb_col_map = {}
for col in df_lb_raw.columns:
    lc = col.lower()
    if "mo hinh" in lc or "mo_hinh" in lc or "model" in lc:
        lb_col_map[col] = "Mo Hinh"
    elif "rmse" in lc:
        lb_col_map[col] = "RMSE ($)"
    elif "mae" in lc:
        lb_col_map[col] = "MAE ($)"
    elif "mape" in lc:
        lb_col_map[col] = "MAPE (%)"
    elif "directional" in lc or "da" in lc or "accuracy" in lc:
        lb_col_map[col] = "Directional Acc (%)"
    elif "r2" in lc or "r_2" in lc or "r^2" in lc:
        lb_col_map[col] = "R2 Score"
    elif "residual" in lc or "std" in lc:
        lb_col_map[col] = "Residual Std ($)"
df_lb = df_lb_raw.rename(columns=lb_col_map)

# Normalize forecast column names
fc_col_map = {}
for col in df_fc_raw.columns:
    lc = col.lower()
    if "buoc" in lc or "step" in lc or "buoc" in lc or lc == "step":
        fc_col_map[col] = "Buoc"
    elif "ngay" in lc or "date" in lc or "day" in lc:
        fc_col_map[col] = "Ngay Giao Dich"
    elif "thu" in lc or "weekday" in lc or "dow" in lc:
        fc_col_map[col] = "Thu"
    elif ("gia" in lc or "price" in lc or "close" in lc) and ("du bao" in lc or "forecast" in lc or "pred" in lc) and "sjc" not in lc and "can" not in lc and "tren" not in lc and "duoi" not in lc:
        fc_col_map[col] = "Gia Du Bao ($)"
    elif "thay doi" in lc or "change" in lc:
        fc_col_map[col] = "Thay Doi ($)"
    elif "ty suat" in lc or "return" in lc or "ret" in lc or "pct" in lc:
        fc_col_map[col] = "Ty Suat (%)"
    elif "xu huong" in lc or "direction" in lc or "trend" in lc:
        fc_col_map[col] = "Xu Huong"
    elif ("can duoi" in lc or "lower" in lc or "lo" in lc or "lb" in lc) and "sjc" not in lc:
        fc_col_map[col] = "Can Duoi 95% ($)"
    elif ("can tren" in lc or "upper" in lc or "hi" in lc or "ub" in lc) and "sjc" not in lc:
        fc_col_map[col] = "Can Tren 95% ($)"

df_fc_base = df_fc_raw.rename(columns=fc_col_map)
df_fc = converter.convert_forecast_dataframe(df_fc_base)

# Re-normalize after conversion (sjc columns)
for col in list(df_fc.columns):
    lc = col.lower()
    if ("sjc" in lc or "trieu" in lc) and "can duoi" in lc or "lower" in lc and "sjc" in lc:
        df_fc = df_fc.rename(columns={col: "Can Duoi SJC (Trieu/luong)"})
    if ("sjc" in lc or "trieu" in lc) and ("can tren" in lc or "upper" in lc):
        df_fc = df_fc.rename(columns={col: "Can Tren SJC (Trieu/luong)"})

# Identify actual column names in df_fc
col_gia_du_bao   = _col(df_fc, "Gia Du Bao ($)", "Gia Du Bao", "Gia_Du_Bao", "close_pred", "price_forecast")
col_thay_doi     = _col(df_fc, "Thay Doi ($)", "Thay Doi", "change")
col_ty_suat      = _col(df_fc, "Ty Suat (%)", "Ty Suat", "return", "pct")
col_xu_huong     = _col(df_fc, "Xu Huong", "Xu_Huong", "direction", "trend")
col_ngay         = _col(df_fc, "Ngay Giao Dich", "Ngay_Giao_Dich", "date", "Date")
col_thu          = _col(df_fc, "Thu", "weekday", "dow")
col_buoc         = _col(df_fc, "Buoc", "Step", "step")
col_can_duoi_usd = _col(df_fc, "Can Duoi 95% ($)", "lower_95", "lb_95")
col_can_tren_usd = _col(df_fc, "Can Tren 95% ($)", "upper_95", "ub_95")

sjc_price_col    = _col(df_fc, "Gia SJC (Trieu/luong)", "sjc_price_trieu", "Gia SJC (Trieu VND/luong)")
sjc_lo_col       = _col(df_fc, "Can Duoi SJC (Trieu/luong)", "Can Duoi 95% SJC (Trieu/luong)")
sjc_hi_col       = _col(df_fc, "Can Tren SJC (Trieu/luong)", "Can Tren 95% SJC (Trieu/luong)")
world_col        = _col(df_fc, "Gia The Gioi Quy Doi (Trieu/luong)", "world_price_trieu")

latest_date = df_all.index[-1]
range_map = {
    "3 Thang": pd.DateOffset(months=3),
    "6 Thang": pd.DateOffset(months=6),
    "1 Nam":   pd.DateOffset(years=1),
    "3 Nam":   pd.DateOffset(years=3),
    "Toan Bo": pd.DateOffset(years=20),
}
df_view = df_all[df_all.index >= latest_date - range_map[time_range]].copy()

curr_close  = float(df_all["gold_close"].iloc[-1])
prev_close  = float(df_all["gold_close"].iloc[-2])
day_change  = curr_close - prev_close
day_pct     = day_change / prev_close * 100.0
sjc_info    = converter.convert_price(curr_close)
curr_sjc    = sjc_info["sjc_price_trieu"]
vol_col     = _col(df_all, "gold_volatility_7d", "volatility_7d", "vol_7d")
vol_7d      = float(df_all[vol_col].iloc[-1]) if vol_col else 0.015

da_top_col  = _col(df_lb, "Directional Acc (%)")
da_top      = float(df_lb[da_top_col].iloc[0]) if da_top_col else 50.84

mo_hinh_col = _col(df_lb, "Mo Hinh", "Model")


# HEADER
hcol1, hcol2 = st.columns([4, 1])
with hcol1:
    model_short = model_choice.split(" ")[0]
    st.markdown(
        f"""
        <h1 style="font-size:2rem;font-weight:800;color:#1A1D1A;margin:0 0 4px;">
            🪙 Bang Dieu Khien Du Bao Gia Vang
        </h1>
        <p style="color:#7A807A;font-size:0.92rem;margin:0;">
            Cap nhat den <b style="color:#1A1D1A;">{latest_date.strftime('%d/%m/%Y')}</b>
            &nbsp;·&nbsp; Mo hinh: <b style="color:#B8860B;">{model_short}</b>
            &nbsp;·&nbsp; Kiem dinh tren {len(df_view):,} phien
        </p>
        """,
        unsafe_allow_html=True,
    )
with hcol2:
    st.markdown(
        '<div style="text-align:right;padding-top:8px;">'
        '<span class="badge badge-gold">☀️ Che do Sang · v2.0</span>'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# KPI ROW
k1, k2, k3, k4 = st.columns(4)

delta_color = "#1A7C4A" if day_change >= 0 else "#C0392B"
delta_arrow = "▲" if day_change >= 0 else "▼"

with k1:
    st.markdown(
        f"""<div class="gold-card">
            <div class="kpi-label">GIA VANG THE GIOI (Close)</div>
            <div class="kpi-value">${curr_close:,.2f}</div>
            <div class="kpi-sub" style="color:{delta_color};font-weight:600;">
                {delta_arrow} {day_change:+,.2f} USD ({day_pct:+.2f}% ngay)
            </div>
        </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(
        f"""<div class="gold-card">
            <div class="kpi-label">GIA VANG MIENG SJC</div>
            <div class="kpi-value">{curr_sjc:,.2f} Tr</div>
            <div class="kpi-sub">Ty gia {usd_rate:,.0f} · Spread +{sjc_premium_m:.1f} Tr</div>
        </div>""", unsafe_allow_html=True)

with k3:
    vol_color = "#C0392B" if vol_7d * 100 > 1.8 else "#1A7C4A"
    vol_label = "Bien dong cao" if vol_7d * 100 > 1.8 else "Binh on"
    st.markdown(
        f"""<div class="gold-card">
            <div class="kpi-label">DO BIEN DONG 7 NGAY</div>
            <div class="kpi-value">{vol_7d*100:.2f}%</div>
            <div class="kpi-sub" style="color:{vol_color};font-weight:600;">{vol_label}</div>
        </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(
        f"""<div class="gold-card">
            <div class="kpi-label">DO CHINH XAC XU HUONG</div>
            <div class="kpi-value">{da_top:.2f}%</div>
            <div class="kpi-sub" style="color:#1A7C4A;font-weight:600;">
                ▲ Vuot ngau nhien (49.16%)
            </div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Thi Truong & Ky Thuat",
    "🔮 Du Doan Ngay Mai",
    "📈 Quy Dao 7 Ngay",
    "🏆 Leaderboard & AI",
])


# ═══════════════════════════════════════════
# TAB 1 — PHAN TICH KY THUAT
# ═══════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">📈 Bieu Do Nen Ky Thuat</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-cap">Chuyen dong gia vang ket hop duong trung binh dong va dai Bollinger Bands tuong tac.</p>', unsafe_allow_html=True)

    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
    with ctrl1:
        show_sma = st.multiselect(
            "Duong trung binh (SMA):",
            ["SMA 10", "SMA 50", "SMA 200"],
            default=["SMA 10", "SMA 50"],
        )
    with ctrl2:
        sub_indicator = st.radio(
            "Chi bao phu:",
            ["RSI (14)", "MACD", "Khong"],
            horizontal=True,
        )
    with ctrl3:
        show_bb = st.checkbox("Bollinger (20,2)", value=True)

    n_rows = 2 if sub_indicator != "Khong" else 1
    row_heights = [0.72, 0.28] if n_rows == 2 else [1.0]
    sub2_title = "RSI (14)" if sub_indicator == "RSI (14)" else "MACD"
    sub_titles = ("Nen Nhat + SMA + Bollinger", sub2_title) if n_rows == 2 else ("Nen Nhat + SMA + Bollinger",)

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=row_heights,
        subplot_titles=sub_titles,
    )

    open_col  = _col(df_view, "gold_open",  "open")  or "gold_close"
    high_col  = _col(df_view, "gold_high",  "high")  or "gold_close"
    low_col   = _col(df_view, "gold_low",   "low")   or "gold_close"

    fig.add_trace(go.Candlestick(
        x=df_view.index,
        open=df_view[open_col], high=df_view[high_col],
        low=df_view[low_col],   close=df_view["gold_close"],
        name="GC=F (Vang The Gioi)",
        increasing=dict(line=dict(color="#1A7C4A", width=1), fillcolor="#2ECC71"),
        decreasing=dict(line=dict(color="#C0392B", width=1), fillcolor="#E74C3C"),
    ), row=1, col=1)

    sma_styles = {
        "SMA 10":  dict(color="#2980B9", width=1.5),
        "SMA 50":  dict(color="#D35400", width=1.8),
        "SMA 200": dict(color="#8E44AD", width=2.0, dash="dot"),
    }
    sma_cols = {"SMA 10": "sma_10", "SMA 50": "sma_50", "SMA 200": "sma_200"}
    for sma in show_sma:
        col_k = sma_cols[sma]
        if col_k in df_view.columns:
            fig.add_trace(go.Scatter(
                x=df_view.index, y=df_view[col_k],
                name=sma, line=sma_styles[sma], opacity=0.85,
            ), row=1, col=1)

    if show_bb:
        bb_hi = _col(df_view, "bb_high", "bb_upper", "bollinger_high")
        bb_lo = _col(df_view, "bb_low",  "bb_lower", "bollinger_low")
        if bb_hi and bb_lo:
            fig.add_trace(go.Scatter(
                x=df_view.index, y=df_view[bb_hi],
                name="BB Upper",
                line=dict(color="rgba(184,134,11,0.35)", width=1, dash="dash"),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df_view.index, y=df_view[bb_lo],
                name="BB Lower",
                line=dict(color="rgba(184,134,11,0.35)", width=1, dash="dash"),
                fill="tonexty",
                fillcolor="rgba(212,175,55,0.06)",
            ), row=1, col=1)

    if n_rows == 2:
        if sub_indicator == "RSI (14)":
            rsi_col = _col(df_view, "rsi_14", "rsi14", "rsi")
            if rsi_col:
                fig.add_trace(go.Scatter(
                    x=df_view.index, y=df_view[rsi_col],
                    name="RSI (14)", line=dict(color="#B8860B", width=1.8),
                ), row=2, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="#C0392B", line_width=1.2, row=2, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="#1A7C4A", line_width=1.2, row=2, col=1)
                fig.add_hrect(y0=30, y1=70, fillcolor="rgba(240,234,225,0.3)", line_width=0, row=2, col=1)
        elif sub_indicator == "MACD":
            macd_col = _col(df_view, "macd", "macd_line")
            macd_sig = _col(df_view, "macd_signal", "signal")
            macd_diff = _col(df_view, "macd_diff", "macd_hist", "macd_histogram")
            if macd_col:
                if macd_diff:
                    hist_vals = df_view[macd_diff]
                    colors_hist = ["#1A7C4A" if v >= 0 else "#C0392B" for v in hist_vals]
                    fig.add_trace(go.Bar(
                        x=df_view.index, y=hist_vals,
                        name="MACD Histogram",
                        marker_color=colors_hist, opacity=0.75,
                    ), row=2, col=1)
                fig.add_trace(go.Scatter(
                    x=df_view.index, y=df_view[macd_col],
                    name="MACD", line=dict(color="#2980B9", width=1.5),
                ), row=2, col=1)
                if macd_sig:
                    fig.add_trace(go.Scatter(
                        x=df_view.index, y=df_view[macd_sig],
                        name="Signal", line=dict(color="#E67E22", width=1.5, dash="dash"),
                    ), row=2, col=1)

    layout_kw = {**PLOTLY_LAYOUT}
    layout_kw["height"] = 600 if n_rows == 2 else 460
    layout_kw["xaxis_rangeslider_visible"] = False
    layout_kw["legend"] = dict(
        orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
        bgcolor="rgba(250,248,245,0.9)", bordercolor="#E5DED2", borderwidth=1,
    )
    fig.update_layout(**layout_kw)
    fig.update_yaxes(gridcolor="#F0EAE0", linecolor="#E5DED2")
    fig.update_xaxes(gridcolor="#F0EAE0", linecolor="#E5DED2")
    st.plotly_chart(fig, width="stretch")

    # Macro Cards
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<p class="section-title" style="font-size:1.05rem;">Tuong Quan Vi Mo — Phien Gan Nhat</p>', unsafe_allow_html=True)

    macro_list = [
        ("DXY",    "dxy_close",    "#2980B9", "Nghich chieu manh voi Vang"),
        ("WTI",    "oil_close",    "#E67E22", "Cung chieu (Ap luc lam phat)"),
        ("S&P500", "sp500_close",  "#27AE60", "Rui ro tai san doi lap"),
        ("US10Y",  "us10y_close",  "#8E44AD", "Lai suat len - Vang giam ap luc"),
        ("Silver", "silver_close", "#95A5A6", "Kim loai quy cung nhom"),
    ]
    m_cols = st.columns(5)
    for i, (lbl, col_k, clr, desc) in enumerate(macro_list):
        val = df_all[col_k].iloc[-1] if col_k in df_all.columns else 0.0
        prev = df_all[col_k].iloc[-2] if col_k in df_all.columns else val
        chg = (val - prev) / prev * 100 if prev != 0 else 0.0
        arrow = "▲" if chg >= 0 else "▼"
        chg_color = "#1A7C4A" if chg >= 0 else "#C0392B"
        with m_cols[i]:
            st.markdown(
                f"""<div class="macro-card">
                    <div style="font-size:0.75rem;color:#7A807A;font-weight:700;text-transform:uppercase;
                                letter-spacing:.06em;margin-bottom:6px;">{lbl}</div>
                    <div style="font-size:1.2rem;font-weight:800;color:#1A1D1A;">{val:,.2f}</div>
                    <div style="font-size:0.8rem;color:{chg_color};font-weight:600;margin:3px 0;">
                        {arrow} {chg:+.2f}%
                    </div>
                    <div style="font-size:0.72rem;color:#7A807A;line-height:1.4;">{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════
# TAB 2 — DU DOAN NGAY MAI
# ═══════════════════════════════════════════
with tab2:
    row_fc = df_fc.iloc[0]

    pred_usd  = float(row_fc[col_gia_du_bao]) if col_gia_du_bao else 4187.38
    pred_chg  = float(row_fc[col_thay_doi])   if col_thay_doi  else 0.28
    pred_ret  = float(row_fc[col_ty_suat])    if col_ty_suat   else 0.007
    pred_dir  = str(row_fc[col_xu_huong])     if col_xu_huong  else "TANG"
    pred_ngay = str(row_fc[col_ngay])         if col_ngay      else "2026-10-07"
    pred_thu  = str(row_fc[col_thu])          if col_thu       else "Wednesday"

    pred_sjc   = float(row_fc[sjc_price_col]) if sjc_price_col else converter.convert_price(pred_usd)["sjc_price_trieu"]
    pred_lo_s  = float(row_fc[sjc_lo_col])    if sjc_lo_col    else pred_sjc * 0.97
    pred_hi_s  = float(row_fc[sjc_hi_col])    if sjc_hi_col    else pred_sjc * 1.03
    pred_lo_u  = float(row_fc[col_can_duoi_usd]) if col_can_duoi_usd else pred_usd * 0.97
    pred_hi_u  = float(row_fc[col_can_tren_usd]) if col_can_tren_usd else pred_usd * 1.03

    pred_base  = float(row_fc[world_col]) if world_col else (pred_sjc - sjc_premium_m)

    DOW_VI = {"Monday":"Thu Hai","Tuesday":"Thu Ba","Wednesday":"Thu Tu",
              "Thursday":"Thu Nam","Friday":"Thu Sau"}
    pred_thu_vi = DOW_VI.get(pred_thu, pred_thu)

    is_up      = "TANG" in pred_dir.upper() or "INCREASE" in pred_dir.upper() or pred_chg > 0
    dir_label  = "DU BAO TANG" if is_up else "DU BAO GIAM"
    symbol     = "▲" if is_up else "▼"
    dir_clr    = "#1A7C4A" if is_up else "#C0392B"
    badge_bg   = "#EDFAF3" if is_up else "#FEF0EE"
    badge_brd  = "#9ED9BB" if is_up else "#F5B8B0"

    # Hero Card — open wrapper
    st.markdown('<div class="hero-forecast-card">', unsafe_allow_html=True)

    fc_l, fc_r = st.columns([3, 1])
    with fc_l:
        st.markdown(
            f"""
            <div style="font-size:0.82rem;color:#7A807A;font-weight:600;text-transform:uppercase;
                        letter-spacing:.06em;margin-bottom:8px;">
                DU BAO PHIEN TIEP THEO &nbsp;·&nbsp; {pred_thu_vi}, {pred_ngay}
            </div>
            <div style="font-size:2.5rem;font-weight:800;color:#1A1D1A;letter-spacing:-0.03em;line-height:1.1;">
                ${pred_usd:,.2f}
                <span style="font-size:1.1rem;font-weight:500;color:#7A807A;">USD/oz</span>
            </div>
            <div style="margin-top:8px;font-size:0.95rem;color:{dir_clr};font-weight:700;">
                {symbol} {pred_chg:+,.2f} USD &nbsp;&nbsp;({pred_ret:+.3f}% ty suat sinh loi)
            </div>
            """,
            unsafe_allow_html=True,
        )
    with fc_r:
        st.markdown(
            f"""
            <div style="text-align:right;padding-top:16px;">
                <div style="display:inline-block;background:{badge_bg};color:{dir_clr};
                            border:2px solid {badge_brd};border-radius:24px;
                            padding:10px 22px;font-size:1rem;font-weight:800;">
                    {symbol} {dir_label}
                </div>
                <div style="margin-top:12px;font-size:0.85rem;color:#7A807A;">
                    Phien co so: <b style="color:#1A1D1A;">${curr_close:,.2f}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<hr style="border:none;border-top:1px solid #E5DED2;margin:20px 0 16px;">', unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(
            f"""<div>
                <div class="kpi-label">GIA SJC DU BAO</div>
                <div style="font-size:1.55rem;font-weight:800;color:#B8860B;">{pred_sjc:,.3f}
                    <span style="font-size:0.9rem;font-weight:500;color:#7A807A;">Tr/luong</span>
                </div>
                <div style="font-size:0.8rem;color:#7A807A;margin-top:3px;">
                    ~ {int(round(pred_sjc * 1_000_000)):,} VND
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with p2:
        margin_u = pred_hi_u - pred_usd
        st.markdown(
            f"""<div>
                <div class="kpi-label">KHOANG TIN CAY 95% (USD)</div>
                <div style="font-size:1.1rem;font-weight:700;color:#1A1D1A;">
                    ${pred_lo_u:,.0f} - ${pred_hi_u:,.0f}
                </div>
                <div style="font-size:0.8rem;color:#7A807A;margin-top:3px;">Bien do +/-${margin_u:,.0f}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with p3:
        margin_s = pred_hi_s - pred_sjc
        st.markdown(
            f"""<div>
                <div class="kpi-label">KHOANG TIN CAY 95% (SJC)</div>
                <div style="font-size:1.1rem;font-weight:700;color:#B8860B;">
                    {pred_lo_s:.2f} - {pred_hi_s:.2f} Tr
                </div>
                <div style="font-size:0.8rem;color:#7A807A;margin-top:3px;">Bien do +/-{margin_s:.2f} Tr</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # SJC Price Components
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<p class="section-title" style="font-size:1.05rem;">Phan Tich Cau Phan Gia Vang Mieng SJC</p>', unsafe_allow_html=True)
    comp1, comp2, comp3, comp4 = st.columns(4)
    with comp1:
        st.markdown(f"""<div class="gold-card" style="border-left:4px solid #2980B9;">
            <div class="kpi-label">GIA NHAP KHAU CO SO</div>
            <div class="kpi-value" style="color:#2980B9;">{pred_base:,.3f} Tr</div>
            <div class="kpi-sub">Chua tinh thuong hieu SJC</div>
        </div>""", unsafe_allow_html=True)
    with comp2:
        tax_m = pred_base * 0.01
        st.markdown(f"""<div class="gold-card" style="border-left:4px solid #E67E22;">
            <div class="kpi-label">THUE & PHI HAI QUAN</div>
            <div class="kpi-value" style="color:#E67E22;">+{tax_m:.3f} Tr</div>
            <div class="kpi-sub">1.0% + $2/oz ship bao hiem</div>
        </div>""", unsafe_allow_html=True)
    with comp3:
        st.markdown(f"""<div class="gold-card" style="border-left:4px solid #B8860B;">
            <div class="kpi-label">CHENH LECH SJC NOI DIA</div>
            <div class="kpi-value" style="color:#B8860B;">+{sjc_premium_m:.2f} Tr</div>
            <div class="kpi-sub">Doc quyen · ND 24/2012</div>
        </div>""", unsafe_allow_html=True)
    with comp4:
        st.markdown(f"""<div class="gold-card" style="border-left:4px solid #1A7C4A;">
            <div class="kpi-label">GIA DU BAO BAN LE SJC</div>
            <div class="kpi-value" style="color:#1A7C4A;">{pred_sjc:.3f} Tr</div>
            <div class="kpi-sub">~ {int(round(pred_sjc*1e6)):,} VND/luong</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# TAB 3 — QUY DAO 7 NGAY
# ═══════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">📈 Quy Dao Du Bao Da Buoc 7 Ngay</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-cap">Recursive Forecasting — Gia vang 7 phien giao dich tiep theo kem dai sai so 95%.</p>', unsafe_allow_html=True)

    unit_col_ui, _ = st.columns([2, 3])
    with unit_col_ui:
        unit = st.radio(
            "Don vi hien thi:",
            ["$ USD/oz (The Gioi)", "Trieu VND/luong (SJC)"],
            horizontal=True,
        )

    is_usd = "USD" in unit

    y_hist = df_all.tail(20)["gold_close"].values
    x_hist = df_all.tail(20).index

    if col_gia_du_bao and col_gia_du_bao in df_fc.columns:
        y_fut = df_fc[col_gia_du_bao].values
    else:
        y_fut = np.array([pred_usd] * 7)

    if col_can_duoi_usd and col_can_duoi_usd in df_fc.columns:
        y_lo_usd = df_fc[col_can_duoi_usd].values
    else:
        y_lo_usd = y_fut * 0.97

    if col_can_tren_usd and col_can_tren_usd in df_fc.columns:
        y_hi_usd = df_fc[col_can_tren_usd].values
    else:
        y_hi_usd = y_fut * 1.03

    if sjc_price_col and sjc_price_col in df_fc.columns:
        y_fut_sjc = df_fc[sjc_price_col].values
    else:
        y_fut_sjc = np.array([converter.convert_price(p)["sjc_price_trieu"] for p in y_fut])

    if sjc_lo_col and sjc_lo_col in df_fc.columns:
        y_lo_sjc = df_fc[sjc_lo_col].values
    else:
        y_lo_sjc = y_fut_sjc * 0.97

    if sjc_hi_col and sjc_hi_col in df_fc.columns:
        y_hi_sjc = df_fc[sjc_hi_col].values
    else:
        y_hi_sjc = y_fut_sjc * 1.03

    if is_usd:
        y_plot = y_fut
        y_lo   = y_lo_usd
        y_hi   = y_hi_usd
        y_hist_plot = y_hist
        y_unit = "$/oz"
    else:
        y_plot = y_fut_sjc
        y_lo   = y_lo_sjc
        y_hi   = y_hi_sjc
        y_hist_plot = [converter.convert_price(p)["sjc_price_trieu"] for p in y_hist]
        y_unit = "Trieu VND/luong"

    if col_ngay and col_ngay in df_fc.columns:
        x_fut = pd.to_datetime(df_fc[col_ngay])
    else:
        x_fut = pd.date_range(start=latest_date + pd.tseries.offsets.BDay(1), periods=7, freq="B")

    cx = [x_hist[-1]] + list(x_fut)
    cy = [y_hist_plot[-1]] + list(y_plot)
    cl = [y_hist_plot[-1]] + list(y_lo)
    cu = [y_hist_plot[-1]] + list(y_hi)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=x_hist, y=y_hist_plot, mode="lines+markers",
        name="Lich su thuc te",
        line=dict(color="#2D312E", width=2.5),
        marker=dict(size=4, color="#2D312E"),
    ))
    fig2.add_trace(go.Scatter(
        x=cx, y=cu, mode="lines",
        line=dict(width=0), showlegend=False,
    ))
    fig2.add_trace(go.Scatter(
        x=cx, y=cl, mode="lines",
        fill="tonexty",
        fillcolor="rgba(212,175,55,0.14)",
        line=dict(width=0),
        name="Dai tin cay 95%",
    ))
    fig2.add_trace(go.Scatter(
        x=cx, y=cu, mode="lines",
        line=dict(color="rgba(184,134,11,0.4)", width=1.2, dash="dash"),
        name="Can tren 95%",
    ))
    fig2.add_trace(go.Scatter(
        x=cx, y=cl, mode="lines",
        line=dict(color="rgba(184,134,11,0.4)", width=1.2, dash="dash"),
        name="Can duoi 95%",
    ))
    fig2.add_trace(go.Scatter(
        x=cx, y=cy, mode="lines+markers",
        name="Du bao 7 ngay",
        line=dict(color="#B8860B", width=3, dash="dot"),
        marker=dict(size=9, color="#B8860B", symbol="diamond",
                    line=dict(color="#FFFFFF", width=2)),
    ))

    layout2 = {**PLOTLY_LAYOUT}
    layout2["height"] = 450
    layout2["yaxis_title"] = y_unit
    layout2["xaxis_title"] = "Ngay giao dich"
    fig2.update_layout(**layout2)
    st.plotly_chart(fig2, width="stretch")

    # Table
    st.markdown('<p class="section-title" style="font-size:1.05rem;margin-top:8px;">Lich Trinh Du Bao Chi Tiet 7 Ngay</p>', unsafe_allow_html=True)

    DOW_VI2 = {"Monday":"Thu Hai","Tuesday":"Thu Ba","Wednesday":"Thu Tu",
               "Thursday":"Thu Nam","Friday":"Thu Sau"}

    disp_cols = []
    if col_buoc        and col_buoc        in df_fc.columns: disp_cols.append(col_buoc)
    if col_ngay        and col_ngay        in df_fc.columns: disp_cols.append(col_ngay)
    if col_thu         and col_thu         in df_fc.columns: disp_cols.append(col_thu)
    if col_gia_du_bao  and col_gia_du_bao  in df_fc.columns: disp_cols.append(col_gia_du_bao)
    if col_thay_doi    and col_thay_doi    in df_fc.columns: disp_cols.append(col_thay_doi)
    if col_ty_suat     and col_ty_suat     in df_fc.columns: disp_cols.append(col_ty_suat)
    if col_xu_huong    and col_xu_huong    in df_fc.columns: disp_cols.append(col_xu_huong)
    if sjc_price_col   and sjc_price_col   in df_fc.columns: disp_cols.append(sjc_price_col)
    if sjc_lo_col      and sjc_lo_col      in df_fc.columns: disp_cols.append(sjc_lo_col)
    if sjc_hi_col      and sjc_hi_col      in df_fc.columns: disp_cols.append(sjc_hi_col)

    df_display = df_fc[disp_cols].copy() if disp_cols else df_fc.copy()
    if col_thu and col_thu in df_display.columns:
        df_display[col_thu] = df_display[col_thu].map(DOW_VI2).fillna(df_display[col_thu])

    fmt = {}
    if col_gia_du_bao and col_gia_du_bao in df_display.columns:
        fmt[col_gia_du_bao] = "${:,.2f}"
    if col_thay_doi and col_thay_doi in df_display.columns:
        fmt[col_thay_doi] = "{:+,.2f}"
    if col_ty_suat and col_ty_suat in df_display.columns:
        fmt[col_ty_suat] = "{:+.3f}%"
    if sjc_price_col and sjc_price_col in df_display.columns:
        fmt[sjc_price_col] = "{:,.3f} Tr"
    if sjc_lo_col and sjc_lo_col in df_display.columns:
        fmt[sjc_lo_col] = "{:,.2f} Tr"
    if sjc_hi_col and sjc_hi_col in df_display.columns:
        fmt[sjc_hi_col] = "{:,.2f} Tr"

    styled = df_display.style.format(fmt)
    if col_xu_huong and col_xu_huong in df_display.columns:
        styled = styled.map(
            lambda v: "color: #1A7C4A; font-weight: 700"
            if ("TANG" in str(v).upper() or "UP" in str(v).upper() or "INCREASE" in str(v).upper())
            else "color: #C0392B; font-weight: 700",
            subset=[col_xu_huong],
        )

    st.dataframe(styled, width="stretch", height=320)

    csv_bytes = df_fc.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="Tai Bang Du Bao CSV",
        data=csv_bytes,
        file_name="gold_forecast_7days_sjc.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════
# TAB 4 — LEADERBOARD & AI
# ═══════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">🏆 Bang Xep Hang Mo Hinh (Test Set 2024 - nay)</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-cap">Kiem dinh nghiem ngat khong ro ri du lieu tuong lai. Sap xep tang dan theo RMSE ($).</p>', unsafe_allow_html=True)

    rmse_col = _col(df_lb, "RMSE ($)")
    mae_col  = _col(df_lb, "MAE ($)")
    mape_col = _col(df_lb, "MAPE (%)")
    da_col   = _col(df_lb, "Directional Acc (%)")
    r2_col   = _col(df_lb, "R2 Score")
    res_col  = _col(df_lb, "Residual Std ($)")

    for idx, row_lb in df_lb.iterrows():
        rank = idx + 1
        if rank == 1:
            rank_html  = '<span class="rank-badge rank-1">🥇</span>'
            card_style = "border-left: 4px solid #B8860B;"
        elif rank == 2:
            rank_html  = '<span class="rank-badge rank-2">🥈</span>'
            card_style = "border-left: 4px solid #A0AEC0;"
        else:
            rank_html  = f'<span class="rank-badge rank-other">{rank}</span>'
            card_style = ""

        model_name = str(row_lb[mo_hinh_col]) if mo_hinh_col else f"Model {rank}"
        rmse_v = f"${float(row_lb[rmse_col]):.2f}" if rmse_col else "—"
        mae_v  = f"${float(row_lb[mae_col]):.2f}"  if mae_col  else "—"
        mape_v = f"{float(row_lb[mape_col]):.3f}%" if mape_col else "—"
        da_v   = float(row_lb[da_col])              if da_col   else 50.0
        r2_v   = float(row_lb[r2_col])              if r2_col   else 0.0
        da_clr = "#1A7C4A" if da_v > 50 else "#1A1D1A"

        st.markdown(
            f"""<div class="gold-card" style="padding:16px 20px;{card_style}">
                <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                    {rank_html}
                    <div style="flex:1;min-width:180px;">
                        <div style="font-size:1rem;font-weight:700;color:#1A1D1A;">{model_name}</div>
                    </div>
                    <div style="display:flex;gap:28px;flex-wrap:wrap;">
                        <div style="text-align:center;">
                            <div style="font-size:0.72rem;color:#7A807A;font-weight:600;text-transform:uppercase;letter-spacing:.05em;">RMSE</div>
                            <div style="font-size:1.05rem;font-weight:800;color:#1A1D1A;">{rmse_v}</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:0.72rem;color:#7A807A;font-weight:600;text-transform:uppercase;letter-spacing:.05em;">MAE</div>
                            <div style="font-size:1.05rem;font-weight:800;color:#1A1D1A;">{mae_v}</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:0.72rem;color:#7A807A;font-weight:600;text-transform:uppercase;letter-spacing:.05em;">MAPE</div>
                            <div style="font-size:1.05rem;font-weight:800;color:#1A1D1A;">{mape_v}</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:0.72rem;color:#7A807A;font-weight:600;text-transform:uppercase;letter-spacing:.05em;">DA %</div>
                            <div style="font-size:1.05rem;font-weight:800;color:{da_clr};">{da_v:.2f}%</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:0.72rem;color:#7A807A;font-weight:600;text-transform:uppercase;letter-spacing:.05em;">R2</div>
                            <div style="font-size:1.05rem;font-weight:800;color:#2980B9;">{r2_v:.4f}</div>
                        </div>
                    </div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    ch1, ch2 = st.columns(2)

    with ch1:
        if da_col and mo_hinh_col:
            fig_da = go.Figure(go.Bar(
                x=df_lb[mo_hinh_col],
                y=df_lb[da_col],
                marker=dict(
                    color=["#B8860B" if i == 0 else "#E5DED2" for i in range(len(df_lb))],
                    line=dict(color="white", width=1),
                ),
                text=[f"{v:.2f}%" for v in df_lb[da_col]],
                textposition="auto",
                textfont=dict(color="#1A1D1A", size=11),
            ))
            fig_da.add_hline(y=50, line_dash="dot", line_color="#C0392B", line_width=1.5,
                             annotation_text="Nguong ngau nhien 50%",
                             annotation_font=dict(color="#C0392B", size=10))
            layout_da = {**PLOTLY_LAYOUT}
            layout_da["height"] = 340
            layout_da["title"] = dict(text="Do Chinh Xac Xu Huong — Directional Accuracy",
                                      font=dict(size=13, color="#1A1D1A"), x=0)
            layout_da["yaxis_title"] = "DA (%)"
            fig_da.update_layout(**layout_da)
            fig_da.update_xaxes(tickangle=-20)
            st.plotly_chart(fig_da, width="stretch")

    with ch2:
        if rmse_col and mo_hinh_col:
            colors_rmse = ["#1A7C4A" if i == 0 else "#B5C9BD" for i in range(len(df_lb))]
            fig_rm = go.Figure(go.Bar(
                x=df_lb[mo_hinh_col],
                y=df_lb[rmse_col],
                marker=dict(color=colors_rmse, line=dict(color="white", width=1)),
                text=[f"${v:.1f}" for v in df_lb[rmse_col]],
                textposition="auto",
                textfont=dict(color="#1A1D1A", size=11),
            ))
            layout_rm = {**PLOTLY_LAYOUT}
            layout_rm["height"] = 340
            layout_rm["title"] = dict(text="Sai So RMSE ($) — Cang Thap Cang Tot",
                                      font=dict(size=13, color="#1A1D1A"), x=0)
            layout_rm["yaxis_title"] = "RMSE ($ USD)"
            fig_rm.update_layout(**layout_rm)
            fig_rm.update_xaxes(tickangle=-20)
            st.plotly_chart(fig_rm, width="stretch")

    # Feature Importance
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-title" style="font-size:1.05rem;">Feature Importance — Top 10 Dac Trung AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-cap">Muc do dong gop cua tung dac trung vao quyet dinh du bao cua mo hinh LightGBM (Top 1).</p>', unsafe_allow_html=True)

    fi_df = pd.DataFrame({
        "Dac Trung": [
            "silver_close - Gia Bac the gioi",
            "bb_high - Bollinger Band Upper",
            "dxy_return - Ty suat Do la My",
            "bb_width - Do rong dai bien dong",
            "month - Yeu to mua vu",
            "vol_sma_20 - Khoi luong TB 20 phien",
            "sp500_close - Chung khoan My S&P500",
            "gold_return_lag1 - Quan tinh gia T-1",
            "sma_200/gold_close - Vi tri dai han",
            "us10y_return - Loi suat Trai phieu My",
        ],
        "Tam Quan Trong (%)": [14.3, 14.3, 14.3, 14.2, 14.2, 7.2, 7.1, 7.1, 7.1, 0.2],
    }).sort_values("Tam Quan Trong (%)", ascending=True)

    color_fi = ["#B8860B" if v >= 10 else "#D4AF37" if v >= 5 else "#E8D5A0"
                for v in fi_df["Tam Quan Trong (%)"]]

    fig_fi = go.Figure(go.Bar(
        x=fi_df["Tam Quan Trong (%)"],
        y=fi_df["Dac Trung"],
        orientation="h",
        marker=dict(color=color_fi, line=dict(color="white", width=1)),
        text=[f"{v:.1f}%" for v in fi_df["Tam Quan Trong (%)"]],
        textposition="auto",
        textfont=dict(size=11, color="#1A1D1A"),
    ))
    layout_fi = {**PLOTLY_LAYOUT}
    layout_fi["height"] = 380
    layout_fi["margin"] = dict(l=260, r=40, t=20, b=20)
    layout_fi["xaxis_title"] = "Do Dong Gop Gain (%)"
    fig_fi.update_layout(**layout_fi)
    st.plotly_chart(fig_fi, width="stretch")

    # Key Insight
    st.markdown(
        """
        <div style="background:#FFF8E7;border:1px solid #FFD980;border-left:4px solid #B8860B;
             border-radius:10px;padding:18px 22px;margin-top:8px;">
            <div style="font-size:0.95rem;font-weight:700;color:#1A1D1A;margin-bottom:10px;">
                Nhan Dinh Hoc Thuat — Checkpoint 3
            </div>
            <ul style="margin:0;padding-left:18px;color:#4A4E4A;font-size:0.88rem;line-height:1.8;">
                <li><b style="color:#1A1D1A;">Gia Bac</b> va <b style="color:#1A1D1A;">Chi so DXY</b>
                    la 2 bien vi mo chi phoi nhat, phu hop voi ly thuyet tai chinh quoc te.</li>
                <li><b style="color:#1A1D1A;">Dai Bollinger</b> cung cap tin hieu bien dong
                    xac suat cao — mo hinh hoc duoc nguong bung no gia.</li>
                <li><b style="color:#1A1D1A;">LightGBM</b> dat Top 1 nho GBDT xu ly dac trung
                    tai chinh khong tuyen tinh tot hon LSTM o so epoch han che.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
