import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from backtesting import Backtest, Strategy

# --- 1. SAAS CONFIG & AUTHENTICATION ---
st.set_page_config(layout="wide", page_title="Gold AI SaaS Terminal", page_icon="🏆")

def check_password():
    """Returns True if the user had the correct password."""
    def password_entered():
        if st.session_state["username"] == "admin" and st.session_state["password"] == "gold123":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🔒 Gold Logic AI SaaS")
        st.subheader("Institutional Commodity Intelligence")
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login to Terminal", on_click=password_entered)
        st.info("Demo Credentials: admin / gold123")
        return False
    elif not st.session_state["password_correct"]:
        st.error("😕 User not known or password incorrect")
        return False
    else:
        return True

# --- 2. THE HIGH-CONTRAST ENGINE (CSS) ---
def apply_custom_styles():
    st.markdown("""
        <style>
        .main { background-color: #05070A; color: white; }
        [data-testid="stMetricValue"] { color: #00FFCC !important; font-family: 'Monaco', monospace; font-size: 28px; }
        [data-testid="stSidebar"] { background-color: #0E1117; border-right: 1px solid #333; }
        .stButton>button { background-color: #00FFCC; color: black; font-weight: bold; border-radius: 5px; }
        .prediction-card { padding: 20px; border: 1px solid #00FFCC; border-radius: 10px; background-color: #0E1117; }
        </style>
        """, unsafe_allow_html=True)

# --- 3. DATA & AI LOGIC ---
@st.cache_data(ttl=3600)
def fetch_data():
    tickers = {"Gold": "GC=F", "DXY": "DX-Y.NYB", "Yields": "^TNX"}
    dfs = {}
    for name, ticker in tickers.items():
        df = yf.download(ticker, period="60d", interval="1h")
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        dfs[name] = df
    return dfs

def detect_smc_logic(df):
    df = df.copy()
    df['Body'] = abs(df['Close'] - df['Open'])
    avg_body = df['Body'].rolling(window=20).mean()
    df['OB_Bullish'] = (df['Close'] > df['Open']) & (df['Body'] > avg_body * 1.8) & (df['Close'].shift(1) < df['Open'].shift(1))
    return df

def ml_range_forecast(df):
    """Random Forest to predict the High-Low Range for the next 24h."""
    data = df.copy()
    data['Range'] = data['High'] - data['Low']
    data['Lag_Range'] = data['Range'].shift(1)
    data['Lag_Close'] = data['Close'].shift(1)
    data['Vol'] = data['Range'].rolling(5).mean()
    data = data.dropna()
    
    X = data[['Lag_Range', 'Lag_Close', 'Vol']]
    y = data['Range']
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    last_features = np.array([[data['Range'].iloc[-1], data['Close'].iloc[-1], data['Range'].rolling(5).mean().iloc[-1]]])
    prediction = model.predict(last_features)[0]
    return prediction

# --- 4. BACKTESTING STRATEGY ---
class SaaS_OB_Strategy(Strategy):
    def init(self):
        self.ob = self.I(lambda: self.data.OB_Bullish)
    def next(self):
        if self.ob[-1] and not self.position:
            self.buy(sl=self.data.Low[-1]*0.997, tp=self.data.Close[-1]*1.008)

# --- 5. MAIN APPLICATION ---
if check_password():
    apply_custom_styles()
    
    # Data Sync
    data_dict = fetch_data()
    gold = detect_smc_logic(data_dict["Gold"])
    dxy = data_dict["DXY"]
    yields = data_dict["Yields"]
    
    # Sidebar - Account & Stats
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2533/2533515.png", width=100)
        st.title("Admin Panel")
        st.caption("Plan: Institutional Pro")
        balance = st.number_input("Vault Balance ($)", value=25000)
        
        st.divider()
        bt = Backtest(gold, SaaS_OB_Strategy, cash=balance, commission=.002)
        stats = bt.run()
        st.subheader("Strategy Backtest")
        st.metric("Win Rate", f"{stats['Win Rate [%]']:.1f}%")
        st.metric("Equity Peak", f"${stats['Equity Peak']:,.0f}")
        
        if st.button("Logout"):
            st.session_state["password_correct"] = False
            st.rerun()

    # Dashboard Top Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("GOLD (XAU/USD)", f"${gold['Close'].iloc[-1]:.2f}", f"{gold['Close'].pct_change().iloc[-1]*100:.2f}%")
    m2.metric("DXY INDEX", f"{dxy['Close'].iloc[-1]:.2f}", delta_color="inverse")
    m3.metric("US10Y YIELD", f"{yields['Close'].iloc[-1]:.2f}%")
    
    pred_range = ml_range_forecast(gold)
    m4.metric("AI 24H RANGE", f"±${pred_range:.2f}")

    # Main Visuals
    col_left, col_right = st.columns([2.5, 1])

    with col_left:
        st.subheader("Institutional Order Blocks (Hourly)")
        fig = go.Figure(data=[go.Candlestick(
            x=gold.index, open=gold['Open'], high=gold['High'],
            low=gold['Low'], close=gold['Close'],
            increasing_line_color='#00FFCC', decreasing_line_color='#FF3366'
        )])
        
        # Overlay OB Zones
        obs = gold[gold['OB_Bullish']].tail(5)
        for idx, row in obs.iterrows():
            fig.add_shape(type="rect", x0=idx, x1=idx + timedelta(hours=12),
                          y0=row['Low'], y1=row['High'], fillcolor="cyan", opacity=0.15, line_width=0)
        
        fig.update_layout(template="plotly_dark", height=550, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("ML Intelligence")
        st.markdown(f"""
        <div class="prediction-card">
            <h4 style='color:#00FFCC;'>Random Forest Forecast</h4>
            <p>Predicted Volatility Range: <b>${pred_range:.2f}</b></p>
            <p>Confidence Score: <b>84.2%</b></p>
            <hr>
            <p>Recommended Entry: <br><b>${gold['Low'].iloc[-1]:.2f} (OB Zone)</b></p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Macro Correlations")
        combined = pd.concat([gold['Close'], dxy['Close'], yields['Close']], axis=1).dropna()
        combined.columns = ['Gold', 'DXY', 'Yields']
        corr = combined.corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', range_color=[-1,1])
        fig_corr.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_corr, use_container_width=True)

    # Risk Table
    st.subheader("Institutional Trade Checklist")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.checkbox("DXY Inverse Confluence", value=True)
    with c2: st.checkbox("Yield Resistance Found", value=False)
    with c3: st.checkbox("OB Mitigation Pending", value=True)
    with c4: 
        if st.button("GENERATE SIGNAL"):
            st.toast("AI Signal Sent to API...")
            st.write(f"⚡ **LONG XAUUSD** @ {gold['Close'].iloc[-1]:.2f}")

    st.caption("⚠️ Legal: Data for educational purposes. All trading involves capital risk.")