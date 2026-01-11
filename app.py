import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
import requests
from io import StringIO
import concurrent.futures
import time
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -------------------------------------------------
# Configuration
# -------------------------------------------------
st.set_page_config(layout="wide", page_title="OBV Divergence Master")
st.title("📊 OBV Divergence Screener (Dual-View)")

# -------------------------------------------------
# 0. Data Fetching Functions
# -------------------------------------------------
@st.cache_data(ttl=86400)
def fetch_nifty50_stocks():
    return ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC", "SBIN",
            "BHARTIARTL", "KOTAKBANK", "BAJFINANCE", "LT", "ASIANPAINT", "AXISBANK", "MARUTI",
            "SUNPHARMA", "TITAN", "ULTRACEMCO", "NESTLEIND", "WIPRO", "DMART", "HCLTECH",
            "TECHM", "POWERGRID", "BAJAJFINSV", "NTPC", "TATAMOTORS", "COALINDIA", "ADANIPORTS",
            "ONGC", "HINDALCO", "GRASIM", "JSWSTEEL", "CIPLA", "DRREDDY", "EICHERMOT",
            "BRITANNIA", "APOLLOHOSP", "DIVISLAB", "GODREJCP", "PIDILITIND", "BERGEPAINT",
            "INDUSINDBK", "BANKBARODA", "SIEMENS", "DLF", "BAJAJ-AUTO", "TATASTEEL", "ADANIENT", "VEDL"]

@st.cache_data(ttl=86400)
def fetch_nse500_stocks():
    try:
        url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=30, verify=False)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            return df['Symbol'].tolist()
    except:
        pass
    return fetch_nifty50_stocks() # Fallback

@st.cache_data(ttl=86400)
def fetch_all_nse_stocks():
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=30, verify=False)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            return df['SYMBOL'].tolist()
    except:
        pass
    return fetch_nifty50_stocks() # Fallback

@st.cache_data(ttl=86400)
def fetch_all_bse_stocks():
    try:
        url = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?Group=&Scripcode=&industry=&segment=Equity&status=Active"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=30, verify=False)
        if response.status_code == 200:
            data = response.json()
            return [item['scrip_cd'] for item in data.get('Table', [])]
    except:
        pass
    return ["500325", "532540", "500180"] # Very minimal fallback

# -------------------------------------------------
# 1. Logic & Math (The Improved "Synced" Engine)
# -------------------------------------------------
def calculate_obv(df):
    """Vectorized, fast OBV calculation"""
    df = df.copy()
    df['Change'] = df['Close'].diff()
    df['OBV_Vol'] = np.where(df['Change'] > 0, df['Volume'], 
                             np.where(df['Change'] < 0, -df['Volume'], 0))
    df['OBV'] = df['OBV_Vol'].cumsum().fillna(0)
    return df

def get_pivots(data, order=5):
    """Finds peaks and valleys using Scipy"""
    highs_idx = argrelextrema(data.values, np.greater, order=order)[0]
    lows_idx = argrelextrema(data.values, np.less, order=order)[0]
    return highs_idx, lows_idx

def detect_divergence(df, order=5):
    """
    Detects divergence by syncing Price Pivots with OBV timestamps.
    Returns a list of divergence dictionaries.
    """
    price_highs_idx, price_lows_idx = get_pivots(df['Close'], order)
    results = []

    # --- Bearish Divergence (Price Higher High, OBV Lower High) ---
    if len(price_highs_idx) >= 2:
        recent_peaks = price_highs_idx[-5:] # Check recent 5 peaks
        for i in range(len(recent_peaks) - 1, 0, -1):
            curr_idx = recent_peaks[i]
            prev_idx = recent_peaks[i-1]
            
            if df['Close'].iloc[curr_idx] > df['Close'].iloc[prev_idx]: # Price HH
                if df['OBV'].iloc[curr_idx] < df['OBV'].iloc[prev_idx]: # OBV LH
                    results.append({
                        "Type": "Bearish",
                        "P1_Idx": prev_idx, "P2_Idx": curr_idx,
                        "P1_Date": df.index[prev_idx], "P2_Date": df.index[curr_idx],
                        "P1_Price": df['Close'].iloc[prev_idx], "P2_Price": df['Close'].iloc[curr_idx],
                        "P1_OBV": df['OBV'].iloc[prev_idx], "P2_OBV": df['OBV'].iloc[curr_idx]
                    })
                    break # Stop after finding the most recent one

    # --- Bullish Divergence (Price Lower Low, OBV Higher Low) ---
    if len(price_lows_idx) >= 2:
        recent_lows = price_lows_idx[-5:]
        for i in range(len(recent_lows) - 1, 0, -1):
            curr_idx = recent_lows[i]
            prev_idx = recent_lows[i-1]
            
            if df['Close'].iloc[curr_idx] < df['Close'].iloc[prev_idx]: # Price LL
                if df['OBV'].iloc[curr_idx] > df['OBV'].iloc[prev_idx]: # OBV HL
                    results.append({
                        "Type": "Bullish",
                        "P1_Idx": prev_idx, "P2_Idx": curr_idx,
                        "P1_Date": df.index[prev_idx], "P2_Date": df.index[curr_idx],
                        "P1_Price": df['Close'].iloc[prev_idx], "P2_Price": df['Close'].iloc[curr_idx],
                        "P1_OBV": df['OBV'].iloc[prev_idx], "P2_OBV": df['OBV'].iloc[curr_idx]
                    })
                    break 
                    
    return results, price_highs_idx, price_lows_idx

# -------------------------------------------------
# 2. Visualization Options
# -------------------------------------------------

def plot_static_matplotlib(df, divs, price_highs, price_lows, ticker):
    """The 'Classic' Matplotlib Look"""
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # 1. Price Chart
    ax1.plot(df.index, df['Close'], color='black', linewidth=1, label='Price')
    # Plot pivots dots
    ax1.scatter(df.index[price_highs], df['Close'].iloc[price_highs], color='red', s=10, alpha=0.5)
    ax1.scatter(df.index[price_lows], df['Close'].iloc[price_lows], color='green', s=10, alpha=0.5)
    ax1.set_title(f"{ticker} - Price Action")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel("Price")

    # 2. OBV Chart
    ax2.plot(df.index, df['OBV'], color='purple', linewidth=1, label='OBV')
    ax2.set_title("On-Balance Volume (OBV)")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel("Volume")

    # 3. Draw Divergence Lines
    for d in divs:
        color = 'green' if d['Type'] == 'Bullish' else 'red'
        # Draw on Price
        ax1.plot([d['P1_Date'], d['P2_Date']], [d['P1_Price'], d['P2_Price']], 
                 color=color, linewidth=2, linestyle='--')
        # Draw on OBV
        ax2.plot([d['P1_Date'], d['P2_Date']], [d['P1_OBV'], d['P2_OBV']], 
                 color=color, linewidth=2, linestyle='--')
        
        # Add Label
        ax1.annotate(d['Type'], (d['P2_Date'], d['P2_Price']), 
                     xytext=(10, 0), textcoords='offset points', color=color, weight='bold')

    plt.tight_layout()
    st.pyplot(fig)

def plot_interactive_plotly(df, divs, ticker):
    """The 'Modern' Plotly Look"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3],
                        subplot_titles=(f"{ticker} Price", "OBV"))

    # Price Candle
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                 low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    
    # OBV Line
    fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], name='OBV', 
                             line=dict(color='purple')), row=2, col=1)

    # Divergence Lines
    for d in divs:
        c = 'green' if d['Type'] == 'Bullish' else 'red'
        fig.add_trace(go.Scatter(x=[d['P1_Date'], d['P2_Date']], y=[d['P1_Price'], d['P2_Price']],
                                 mode='lines', line=dict(color=c, width=2, dash='dot'), 
                                 name=f"{d['Type']} Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[d['P1_Date'], d['P2_Date']], y=[d['P1_OBV'], d['P2_OBV']],
                                 mode='lines', line=dict(color=c, width=2, dash='dot'), 
                                 name=f"{d['Type']} OBV"), row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False, height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with st.sidebar:
    st.header("⚙️ Settings")
    chart_style = st.radio("Chart Style", ["Static (Matplotlib)", "Interactive (Plotly)"])
    
    if st.checkbox("Use Preset List", value=True):
        dataset_choice = st.sidebar.selectbox(
            "Select Dataset", 
            ["NIFTY 50", "NSE 500", "All NSE Stocks", "All BSE Stocks"], 
            index=1
        )
        if dataset_choice == "NIFTY 50": symbols_raw = fetch_nifty50_stocks()
        elif dataset_choice == "NSE 500": symbols_raw = fetch_nse500_stocks()
        elif dataset_choice == "All NSE Stocks": symbols_raw = fetch_all_nse_stocks()
        else: symbols_raw = fetch_all_bse_stocks()
        
        suffix = ".BO" if dataset_choice == "All BSE Stocks" else ".NS"
        tickers = [f"{s}{suffix}" for s in symbols_raw]
        
        num_to_scan = st.slider("Number of stocks to scan", 5, len(tickers), min(100, len(tickers)))
        tickers = tickers[:num_to_scan]
    else:
        tickers_input = st.text_area("Stocks (comma sep)", "RELIANCE.NS, TCS.NS, HDFCBANK.NS, TATAMOTORS.NS, INFY.NS")
        tickers = [t.strip().upper() for t in tickers_input.split(',')]

    period = st.selectbox("Lookback Period", ["3mo", "6mo", "1y", "2y"], index=1)
    sensitivity = st.slider("Pivot Sensitivity", 3, 20, 5, help="Higher = Fewer, stronger pivots")
    max_batch_size = st.slider("Download Batch Size", 20, 100, 50, help="Number of stocks to download in one go")

if st.button("🚀 Run Screener"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results_container = []

    # Batch Fetching Implementation
    total_tickers = len(tickers)
    for start_idx in range(0, total_tickers, max_batch_size):
        batch = tickers[start_idx:start_idx + max_batch_size]
        status_text.text(f"📥 Downloading batch {start_idx//max_batch_size + 1}: {len(batch)} tickers...")
        
        try:
            # Download entire batch at once
            all_df = yf.download(batch, period=period, interval="1d", progress=False, group_by='ticker')
            
            for ticker in batch:
                try:
                    # Handle single vs multi-ticker dataframes
                    if len(batch) > 1:
                        df = all_df[ticker]
                    else:
                        df = all_df
                        
                    if df.empty or 'Close' not in df.columns:
                        continue
                    
                    # Clean data (handle MultiIndex if necessary, though group_by='ticker' usually handles it)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                        
                    df = calculate_obv(df)
                    divs, ph, pl = detect_divergence(df, order=sensitivity)
                    results_container.append((ticker, df, divs, ph, pl))
                except Exception as e:
                    # Skip problematic stocks silently in batch
                    continue
                    
        except Exception as e:
            st.error(f"Error downloading batch: {e}")
        
        progress_bar.progress(min((start_idx + max_batch_size) / total_tickers, 1.0))

    status_text.success(f"✅ Scanning Complete! Analyzed {len(results_container)} stocks. Found {len([r for r in results_container if r[2]])} stocks with signals.")

    # Sort results to show signals first
    results_container.sort(key=lambda x: len(x[2]), reverse=True)

    for ticker, df, divs, ph, pl in results_container:
        if divs:
            st.subheader(f"🎯 {ticker}: {divs[0]['Type']} Divergence Detected")
            d = divs[0]
            st.caption(f"Found between {d['P1_Date'].date()} and {d['P2_Date'].date()}")
            
            if chart_style == "Static (Matplotlib)":
                plot_static_matplotlib(df, divs, ph, pl, ticker)
            else:
                plot_interactive_plotly(df, divs, ticker)
            st.divider()
        else:
            with st.expander(f"⚪ {ticker} (No Signal)"):
                st.write("No clear divergence found.")
                # Lazy loading: Only render chart if user checks this box
                if st.checkbox(f"Show chart for {ticker}", key=f"show_{ticker}"):
                    if chart_style == "Static (Matplotlib)":
                        plot_static_matplotlib(df, [], ph, pl, ticker)
                    else:
                        plot_interactive_plotly(df, [], ticker)
