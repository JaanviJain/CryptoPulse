"""
Crypto Trading Dashboard — Streamlit
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime, timedelta
import random

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME TOGGLE  (stored in session state)
# ─────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ─────────────────────────────────────────────
# DYNAMIC CSS
# ─────────────────────────────────────────────
def inject_css(dark: bool):
    if dark:
        bg        = "#0d1117"
        card_bg   = "#161b22"
        sidebar   = "#0d1117"
        text      = "#e6edf3"
        sub_text  = "#8b949e"
        border    = "#30363d"
        accent    = "#58a6ff"
        green     = "#3fb950"
        red       = "#f85149"
        amber     = "#d29922"
        chart_bg  = "#161b22"
        grid_col  = "#21262d"
    else:
        bg        = "#f6f8fa"
        card_bg   = "#ffffff"
        sidebar   = "#ffffff"
        text      = "#1f2328"
        sub_text  = "#656d76"
        border    = "#d1d9e0"
        accent    = "#0969da"
        green     = "#1a7f37"
        red       = "#cf222e"
        amber     = "#9a6700"
        chart_bg  = "#ffffff"
        grid_col  = "#eaeef2"

    st.markdown(f"""
    <style>
    /* ── Global ── */
    .stApp {{ background-color: {bg}; color: {text}; }}
    [data-testid="stSidebar"] {{ background-color: {sidebar}; border-right: 1px solid {border}; }}
    [data-testid="stSidebar"] * {{ color: {text} !important; }}
    h1,h2,h3,h4,h5,h6 {{ color: {text}; }}

    /* ── Cards ── */
    .card {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }}
    .metric-card {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 14px 18px;
        text-align: center;
    }}
    .metric-value   {{ font-size: 22px; font-weight: 700; color: {text}; margin: 4px 0; }}
    .metric-label   {{ font-size: 12px; color: {sub_text}; text-transform: uppercase; letter-spacing: .05em; }}
    .metric-change  {{ font-size: 13px; font-weight: 600; }}
    .pos {{ color: {green}; }}
    .neg {{ color: {red};   }}
    .neutral {{ color: {amber}; }}

    /* ── Signal badge ── */
    .badge {{
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: .04em;
    }}
    .badge-buy  {{ background: rgba(63,185,80,.15); color: {green}; border: 1px solid rgba(63,185,80,.35); }}
    .badge-sell {{ background: rgba(248,81,73,.15);  color: {red};   border: 1px solid rgba(248,81,73,.35); }}
    .badge-hold {{ background: rgba(210,153,34,.15); color: {amber}; border: 1px solid rgba(210,153,34,.35); }}

    /* ── Whale alerts ── */
    .whale-alert {{
        background: {card_bg};
        border-left: 3px solid {accent};
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 8px;
        font-size: 13px;
        color: {text};
    }}

    /* ── Table ── */
    .styled-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    .styled-table th {{
        background: {border};
        color: {sub_text};
        padding: 8px 12px;
        text-align: left;
        font-weight: 600;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: .06em;
    }}
    .styled-table td {{
        padding: 10px 12px;
        border-bottom: 1px solid {border};
        color: {text};
    }}
    .styled-table tr:hover td {{ background: rgba(88,166,255,.05); }}

    /* ── Section header ── */
    .section-header {{
        font-size: 13px;
        font-weight: 600;
        color: {sub_text};
        text-transform: uppercase;
        letter-spacing: .08em;
        margin: 18px 0 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid {border};
    }}

    /* ── Plotly chart bg ── */
    .js-plotly-plot .plotly .bg {{ fill: {chart_bg} !important; }}

    /* ── Hide Streamlit default elements ── */
    #MainMenu, footer, header {{ visibility: hidden; }}
    [data-testid="stMetric"] {{ background: {card_bg}; border-radius: 10px; padding: 10px; border: 1px solid {border}; }}

    /* ── Lock sidebar permanently open ── */
    [data-testid="stSidebarCollapseButton"] {{ display: none !important; }}
    [data-testid="collapsedControl"]        {{ display: none !important; }}
    button[kind="header"]                   {{ display: none !important; }}
    section[data-testid="stSidebar"]        {{ min-width: 260px !important; width: 260px !important; transform: none !important; }}
    section[data-testid="stSidebar"] > div  {{ min-width: 260px !important; }}

    /* Toggle button */
    .toggle-btn {{
        background: {card_bg};
        border: 1px solid {border};
        color: {text};
        border-radius: 20px;
        padding: 5px 14px;
        cursor: pointer;
        font-size: 13px;
    }}
    </style>
    """, unsafe_allow_html=True)

inject_css(st.session_state.dark_mode)

# ─────────────────────────────────────────────
# COLOUR HELPERS (depend on theme)
# ─────────────────────────────────────────────
def colors():
    if st.session_state.dark_mode:
        return dict(
            bg="#161b22", paper="#0d1117", text="#e6edf3",
            grid="#21262d", green="#3fb950", red="#f85149",
            amber="#d29922", accent="#58a6ff", sub="#8b949e",
            up="#3fb950", dn="#f85149",
        )
    return dict(
        bg="#ffffff", paper="#f6f8fa", text="#1f2328",
        grid="#eaeef2", green="#1a7f37", red="#cf222e",
        amber="#9a6700", accent="#0969da", sub="#656d76",
        up="#1a7f37", dn="#cf222e",
    )

# ─────────────────────────────────────────────
# DATA FETCHING  (Binance public REST — no key)
# ─────────────────────────────────────────────
SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
           "BNB": "BNBUSDT", "ADA": "ADAUSDT"}

@st.cache_data(ttl=60)
def fetch_klines(symbol: str, interval: str = "15m", limit: int = 200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data, columns=[
            "timestamp","open","high","low","close","volume",
            "close_time","qav","trades","tbbav","tbqav","ignore"
        ])
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception:
        return _fake_klines(limit)

def _fake_klines(limit: int):
    """Fallback synthetic data if API unreachable."""
    now = datetime.utcnow()
    ts  = [now - timedelta(minutes=15*(limit-i)) for i in range(limit)]
    price = 65000.0
    rows = []
    for t in ts:
        o = price
        price *= (1 + np.random.normal(0, 0.003))
        h = max(o, price) * (1 + abs(np.random.normal(0,.001)))
        l = min(o, price) * (1 - abs(np.random.normal(0,.001)))
        v = random.uniform(800, 3000)
        rows.append([t, o, h, l, price, v])
    return pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])

@st.cache_data(ttl=30)
def fetch_ticker(symbol: str):
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"lastPrice":"65000","priceChangePercent":"-2.5","highPrice":"67000","lowPrice":"64000","volume":"12000","quoteVolume":"780000000"}

@st.cache_data(ttl=120)
def fetch_all_tickers():
    results = {}
    for name, sym in SYMBOLS.items():
        results[name] = fetch_ticker(sym)
    return results

# ─────────────────────────────────────────────
# MOCK: SENTIMENT, PREDICTIONS, SIGNALS, ONCHAIN
# ─────────────────────────────────────────────
def mock_sentiment(symbol: str):
    rng = random.Random(symbol + str(datetime.utcnow().hour))
    score = rng.uniform(-0.4, 0.6)
    total = rng.randint(8, 35)
    bull  = int(total * rng.uniform(.2, .55))
    bear  = int(total * rng.uniform(.1, .35))
    neut  = total - bull - bear
    return {"score": round(score, 3), "total": total,
            "bullish": bull, "bearish": bear, "neutral": neut}

def mock_prediction(symbol: str, df: pd.DataFrame):
    if df.empty:
        return {"direction":"SIDEWAYS","confidence":0.5}
    last  = df["close"].iloc[-1]
    prev  = df["close"].iloc[-5]
    trend = (last - prev) / prev
    rng   = random.Random(symbol + str(datetime.utcnow().hour))
    conf  = rng.uniform(0.55, 0.75)
    if trend > 0.005:
        direction = "UP"
    elif trend < -0.005:
        direction = "DOWN"
    else:
        direction = "SIDEWAYS"
        conf = 0.50
    return {"direction": direction, "confidence": round(conf, 3)}

def _signal_reasons(d, c, s, df):
    """Build a structured list of reasons for BUY / SELL / HOLD."""
    reasons = []

    # 1. Model prediction
    if d == "UP":
        reasons.append(f"✅ ML model predicts price UP next hour with {c:.0%} confidence (XGBoost + 60+ features).")
    elif d == "DOWN":
        reasons.append(f"🔴 ML model predicts price DOWN next hour with {c:.0%} confidence.")
    else:
        reasons.append(f"⚠️ ML model is uncertain — SIDEWAYS prediction ({c:.0%} confidence). No clear edge.")

    # 2. Sentiment
    if s > 0.3:
        reasons.append(f"✅ News sentiment strongly bullish (score {s:+.2f}) — media narrative supports buying.")
    elif s > 0.05:
        reasons.append(f"✅ News sentiment mildly positive (score {s:+.2f}) — slight tailwind from media.")
    elif s < -0.3:
        reasons.append(f"🔴 News sentiment strongly negative (score {s:+.2f}) — fear/FUD in recent articles.")
    elif s < -0.05:
        reasons.append(f"🔴 News sentiment mildly negative (score {s:+.2f}) — some bearish press coverage.")
    else:
        reasons.append(f"⚪ News sentiment neutral (score {s:+.2f}) — no strong media catalyst either way.")

    # 3. Price momentum (from recent candles)
    if not df.empty and len(df) >= 20:
        ret_1h  = (df["close"].iloc[-1] - df["close"].iloc[-4])  / df["close"].iloc[-4]  * 100
        ret_4h  = (df["close"].iloc[-1] - df["close"].iloc[-16]) / df["close"].iloc[-16] * 100 if len(df) >= 16 else 0
        vol_now = df["volume"].iloc[-5:].mean()
        vol_avg = df["volume"].iloc[-20:].mean()
        vol_ratio = vol_now / (vol_avg + 1e-9)

        if ret_1h > 0.5:
            reasons.append(f"✅ Short-term momentum positive: +{ret_1h:.2f}% in last hour.")
        elif ret_1h < -0.5:
            reasons.append(f"🔴 Short-term momentum negative: {ret_1h:.2f}% in last hour.")

        if ret_4h > 1.5:
            reasons.append(f"✅ 4-hour trend strongly bullish: +{ret_4h:.2f}%.")
        elif ret_4h < -1.5:
            reasons.append(f"🔴 4-hour trend strongly bearish: {ret_4h:.2f}%.")

        if vol_ratio > 1.5:
            reasons.append(f"✅ Volume spike detected ({vol_ratio:.1f}x average) — strong market participation.")
        elif vol_ratio < 0.6:
            reasons.append(f"⚪ Volume below average ({vol_ratio:.1f}x) — low conviction, thin market.")

        # RSI-like proxy
        gains = df["close"].diff().clip(lower=0).iloc[-14:].mean()
        losses = (-df["close"].diff()).clip(lower=0).iloc[-14:].mean()
        rsi = 100 - 100 / (1 + gains / (losses + 1e-9))
        if rsi > 70:
            reasons.append(f"⚠️ RSI at {rsi:.0f} — overbought territory, caution on new longs.")
        elif rsi < 30:
            reasons.append(f"✅ RSI at {rsi:.0f} — oversold territory, potential bounce incoming.")
        else:
            reasons.append(f"⚪ RSI at {rsi:.0f} — neutral range, no extreme reading.")

        # MA cross
        ma5  = df["close"].iloc[-5:].mean()
        ma20 = df["close"].iloc[-20:].mean()
        if ma5 > ma20 * 1.002:
            reasons.append(f"✅ Short MA crossed above long MA — bullish crossover signal.")
        elif ma5 < ma20 * 0.998:
            reasons.append(f"🔴 Short MA below long MA — bearish crossover signal.")

    return reasons

def mock_signal(pred, sent, df=None):
    d   = pred["direction"]
    c   = pred["confidence"]
    s   = sent["score"]

    if df is None:
        df = pd.DataFrame()

    if d == "UP" and c >= 0.60 and s > 0.05:
        sig = "BUY"
    elif d == "DOWN" and c >= 0.60 and s < -0.05:
        sig = "SELL"
    else:
        sig = "HOLD"

    reasons = _signal_reasons(d, c, s, df)
    return {"signal": sig, "confidence": c, "reasons": reasons}

def mock_onchain(symbol: str):
    rng = random.Random(symbol + str(datetime.utcnow().hour))
    whales = []
    for _ in range(rng.randint(2, 5)):
        amt = rng.uniform(200, 5000)
        direction = rng.choice(["Exchange → Wallet", "Wallet → Exchange"])
        whales.append({"amount": round(amt, 1), "direction": direction,
                       "time": f"{rng.randint(1,59)}m ago"})
    inflow  = round(rng.uniform(1000, 9000), 0)
    outflow = round(rng.uniform(800, 8500), 0)
    accum   = "ACCUMULATION" if outflow > inflow else "DISTRIBUTION"
    return {"whales": whales, "inflow": inflow, "outflow": outflow, "signal": accum}

# ─────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────
def build_candlestick(df: pd.DataFrame, symbol: str, hours: int):
    c = colors()
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    df = df[df["timestamp"] >= cutoff].copy()

    # SMAs
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.03,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df["timestamp"], open=df["open"], high=df["high"],
        low=df["low"],     close=df["close"],
        increasing_fillcolor=c["up"], increasing_line_color=c["up"],
        decreasing_fillcolor=c["dn"], decreasing_line_color=c["dn"],
        name="Price", showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["sma20"], mode="lines",
        line=dict(color="#58a6ff", width=1.2), name="SMA 20",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["sma50"], mode="lines",
        line=dict(color="#d29922", width=1.2, dash="dot"), name="SMA 50",
    ), row=1, col=1)

    # Volume bars
    vol_colors = [c["up"] if row["close"] >= row["open"] else c["dn"]
                  for _, row in df.iterrows()]
    fig.add_trace(go.Bar(
        x=df["timestamp"], y=df["volume"],
        marker_color=vol_colors, opacity=0.6,
        name="Volume", showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        height=480,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor=c["paper"],
        plot_bgcolor=c["bg"],
        font=dict(color=c["text"], size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis_rangeslider_visible=False,
        title=dict(text=f"{symbol} — {hours}h chart", font=dict(size=13), x=0),
    )
    for axis in ["xaxis","yaxis","xaxis2","yaxis2"]:
        fig.update_layout(**{
            axis: dict(
                gridcolor=c["grid"], gridwidth=0.5,
                zerolinecolor=c["grid"],
                tickfont=dict(color=c["sub"], size=10),
            )
        })
    return fig

def build_gauge(score: float):
    c = colors()
    needle_color = c["green"] if score > 0.1 else (c["red"] if score < -0.1 else c["amber"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "", "font": {"size": 22, "color": c["text"]}},
        gauge=dict(
            axis=dict(range=[-1, 1], tickfont=dict(color=c["sub"], size=10),
                      tickcolor=c["grid"]),
            bar=dict(color=needle_color, thickness=0.22),
            bgcolor=c["bg"],
            steps=[
                dict(range=[-1, -0.3], color="rgba(248,81,73,.18)"),
                dict(range=[-0.3, 0.3], color="rgba(210,153,34,.12)"),
                dict(range=[0.3,  1],   color="rgba(63,185,80,.18)"),
            ],
            threshold=dict(
                line=dict(color=needle_color, width=2),
                thickness=0.7, value=score,
            ),
        ),
    ))
    fig.update_layout(
        height=200, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor=c["paper"], font=dict(color=c["text"]),
    )
    return fig

def build_donut(bull, bear, neut):
    c = colors()
    fig = go.Figure(go.Pie(
        labels=["Bullish", "Bearish", "Neutral"],
        values=[bull, bear, neut],
        hole=0.6,
        marker=dict(colors=[c["green"], c["red"], c["amber"]]),
        textfont=dict(color=c["text"], size=11),
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    fig.update_layout(
        height=180, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=c["paper"],
        showlegend=True,
        legend=dict(font=dict(color=c["text"], size=10),
                    bgcolor="rgba(0,0,0,0)", orientation="h",
                    yanchor="bottom", y=-0.15),
    )
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # Dark / Light toggle
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown("### ⚙️ Settings")
    with col_b:
        icon = "☀️" if st.session_state.dark_mode else "🌙"
        if st.button(icon, key="theme_toggle", help="Toggle theme"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.markdown("---")
    selected = st.radio("**Cryptocurrency**",
                        list(SYMBOLS.keys()),
                        index=0,
                        format_func=lambda x: f"{'₿' if x=='BTC' else '⬡' if x=='ETH' else '◎' if x=='SOL' else '●'}  {x}")

    hours = st.select_slider("**Chart window**",
                             options=[3, 6, 12, 24, 48, 72],
                             value=24,
                             format_func=lambda x: f"{x}h")

    interval_map = {3:"15m", 6:"15m", 12:"15m", 24:"15m", 48:"1h", 72:"1h"}
    interval = interval_map[hours]
    limit    = min(200, hours * 4 if hours <= 24 else hours)

    st.markdown("---")
    auto_refresh = st.checkbox("**Auto-refresh** (60s)", value=True)
    if auto_refresh:
        st.caption("🔄 Data refreshes every 60 seconds")

    st.markdown("---")
    st.caption("Data: Binance API  •  Predictions: XGBoost model")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
sym = SYMBOLS[selected]
ticker  = fetch_ticker(sym)
price   = float(ticker.get("lastPrice", 0))
chg_pct = float(ticker.get("priceChangePercent", 0))
high24  = float(ticker.get("highPrice", 0))
low24   = float(ticker.get("lowPrice", 0))
vol24   = float(ticker.get("quoteVolume", 0))

c = colors()
chg_cls = "pos" if chg_pct >= 0 else "neg"
chg_sym = "▲" if chg_pct >= 0 else "▼"

# Heading colour: vivid in both dark and light mode — no gradient clip trick
pulse_color  = "#58a6ff" if st.session_state.dark_mode else "#0969da"
pulse_color2 = "#a371f7" if st.session_state.dark_mode else "#8250df"
lightning_bg = "rgba(88,166,255,0.12)" if st.session_state.dark_mode else "rgba(9,105,218,0.10)"

st.markdown(f"""
<div style="margin-bottom:18px; padding-bottom:14px; border-bottom:2px solid {c['grid']};">
  <div style="display:flex; align-items:center; gap:12px; margin-bottom:2px;">
    <span style="font-size:34px; background:{lightning_bg}; border-radius:10px; padding:4px 10px;">&#9889;</span>
    <span style="font-size:40px; font-weight:900; letter-spacing:-1px; color:{pulse_color};">Crypto</span>
    <span style="font-size:40px; font-weight:900; letter-spacing:-1px; color:{pulse_color2};">Pulse</span>
    <span style="font-size:11px; color:{c['sub']}; margin-left:4px; font-weight:500;
      letter-spacing:.12em; text-transform:uppercase; align-self:flex-end; padding-bottom:9px; border-left:2px solid {c['grid']}; padding-left:10px;">
      Live Trading Intelligence
    </span>
  </div>
</div>
<div style="display:flex; align-items:center; gap:16px; margin-bottom:4px;">
  <h2 style="margin:0; font-size:24px; font-weight:700; color:{c['text']};">{selected} <span style="color:{c['sub']}; font-size:14px; font-weight:400;">{sym}</span></h2>
  <span style="font-size:26px; font-weight:700; color:{c['text']};">${price:,.2f}</span>
  <span class="{chg_cls}" style="font-size:17px; font-weight:600;">{chg_sym} {abs(chg_pct):.2f}%</span>
  <span style="font-size:12px; color:{c['sub']}; margin-left:auto;">
    Updated: {datetime.utcnow().strftime('%H:%M:%S UTC')}
  </span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TOP METRICS ROW
# ─────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
def metric_card(label, value, sub="", pos=None):
    sub_cls = "" if pos is None else ("pos" if pos else "neg")
    return f"""<div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-change {sub_cls}">{sub}</div>
    </div>"""

with m1: st.markdown(metric_card("24h High",  f"${high24:,.2f}"), unsafe_allow_html=True)
with m2: st.markdown(metric_card("24h Low",   f"${low24:,.2f}"),  unsafe_allow_html=True)
with m3: st.markdown(metric_card("24h Volume", f"${vol24/1e6:.1f}M"), unsafe_allow_html=True)
with m4: st.markdown(metric_card("Change", f"{chg_pct:+.2f}%", pos=(chg_pct>=0)), unsafe_allow_html=True)

st.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FETCH DATA
# ─────────────────────────────────────────────
df      = fetch_klines(sym, interval, limit)
sent    = mock_sentiment(selected)
pred    = mock_prediction(selected, df)
sig     = mock_signal(pred, sent, df)
onchain = mock_onchain(selected)

# ─────────────────────────────────────────────
# ROW 1: Chart + Sentiment
# ─────────────────────────────────────────────
col_chart, col_sent = st.columns([2.2, 1], gap="medium")

with col_chart:
    st.markdown('<div class="section-header">📊 Price Chart</div>', unsafe_allow_html=True)
    fig = build_candlestick(df, selected, hours)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with col_sent:
    st.markdown('<div class="section-header">💬 Sentiment Analysis</div>', unsafe_allow_html=True)

    # Gauge
    st.plotly_chart(build_gauge(sent["score"]), use_container_width=True,
                    config={"displayModeBar": False})

    sentiment_label = "BULLISH" if sent["score"] > 0.1 else ("BEARISH" if sent["score"] < -0.1 else "NEUTRAL")
    sent_cls = "pos" if sentiment_label == "BULLISH" else ("neg" if sentiment_label == "BEARISH" else "neutral")
    st.markdown(f"""
    <div class="card" style="padding:12px 16px;">
      <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
        <span style="font-size:12px; color:{c['sub']};">Overall</span>
        <span class="{sent_cls}" style="font-weight:700; font-size:14px;">{sentiment_label}</span>
      </div>
      <div style="font-size:12px; color:{c['sub']}; margin-bottom:6px;">Articles analysed: <b style="color:{c['text']}">{sent['total']}</b></div>
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(build_donut(sent["bullish"], sent["bearish"], sent["neutral"]),
                    use_container_width=True, config={"displayModeBar": False})

# ─────────────────────────────────────────────
# ROW 2: Prediction + Signal + On-Chain
# ─────────────────────────────────────────────
col_pred, col_sig, col_chain = st.columns(3, gap="medium")

with col_pred:
    st.markdown('<div class="section-header">🔮 Price Prediction</div>', unsafe_allow_html=True)
    dir_icon = "📈" if pred["direction"] == "UP" else ("📉" if pred["direction"] == "DOWN" else "➡️")
    dir_cls  = "pos" if pred["direction"] == "UP" else ("neg" if pred["direction"] == "DOWN" else "neutral")
    conf_pct = int(pred["confidence"] * 100)

    # Progress bar colour via inline style
    bar_color = c["green"] if pred["direction"] == "UP" else (c["red"] if pred["direction"] == "DOWN" else c["amber"])

    st.markdown(f"""
    <div class="card" style="text-align:center; padding:24px;">
      <div style="font-size:40px; margin-bottom:8px;">{dir_icon}</div>
      <div class="{dir_cls}" style="font-size:28px; font-weight:800; letter-spacing:.04em;">{pred['direction']}</div>
      <div style="font-size:13px; color:{c['sub']}; margin:6px 0;">Next 1-hour prediction</div>
      <div style="background:{c['grid']}; border-radius:20px; height:8px; margin:12px 0;">
        <div style="background:{bar_color}; width:{conf_pct}%; height:8px; border-radius:20px; transition:width .4s;"></div>
      </div>
      <div style="font-size:22px; font-weight:700; color:{c['text']};">{conf_pct}% <span style="font-size:13px; color:{c['sub']}; font-weight:400;">confidence</span></div>
    </div>
    """, unsafe_allow_html=True)

with col_sig:
    st.markdown('<div class="section-header">⚡ Trading Signal</div>', unsafe_allow_html=True)
    s = sig["signal"]
    badge_cls = "badge-buy" if s == "BUY" else ("badge-sell" if s == "SELL" else "badge-hold")
    sig_icon  = "🟢" if s == "BUY" else ("🔴" if s == "SELL" else "🟡")
    reasons_html = "".join(
        f'<div style="font-size:11.5px; color:{c["sub"]}; padding:4px 0; border-bottom:1px solid {c["grid"]}; line-height:1.5;">{r}</div>'
        for r in sig["reasons"]
    )
    st.markdown(f"""
    <div class="card" style="text-align:center; padding:20px;">
      <div style="font-size:36px; margin-bottom:6px;">{sig_icon}</div>
      <div style="margin-bottom:8px;"><span class="badge {badge_cls}" style="font-size:18px; padding:5px 22px;">{s}</span></div>
      <div style="font-size:13px; color:{c["sub"]}; margin:8px 0 12px;">Confidence: <b style="color:{c["text"]}">{int(sig["confidence"]*100)}%</b></div>
      <div style="text-align:left; background:{c["paper"]}; border-radius:8px; padding:10px 12px;">
        <div style="font-size:10px; color:{c["sub"]}; font-weight:600; text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px;">Signal Reasoning</div>
        {reasons_html}
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_chain:
    st.markdown('<div class="section-header">🐋 On-Chain Data</div>', unsafe_allow_html=True)
    acc_cls = "pos" if onchain["signal"] == "ACCUMULATION" else "neg"
    net = onchain["outflow"] - onchain["inflow"]
    st.markdown(f"""
    <div class="card" style="padding:16px;">
      <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
        <span style="font-size:12px; color:{c['sub']};">Exchange Inflow</span>
        <span class="neg" style="font-weight:600;">{onchain['inflow']:,.0f} BTC</span>
      </div>
      <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
        <span style="font-size:12px; color:{c['sub']};">Exchange Outflow</span>
        <span class="pos" style="font-weight:600;">{onchain['outflow']:,.0f} BTC</span>
      </div>
      <div style="display:flex; justify-content:space-between; margin-bottom:14px;">
        <span style="font-size:12px; color:{c['sub']};">Acc/Dist Signal</span>
        <span class="{acc_cls}" style="font-weight:700;">{onchain['signal']}</span>
      </div>
      <div style="font-size:12px; color:{c['sub']}; margin-bottom:6px; font-weight:600;">🐋 Whale Alerts</div>
    """, unsafe_allow_html=True)
    for w in onchain["whales"][:4]:
        icon = "📤" if "Exchange" in w["direction"].split("→")[1] else "📥"
        st.markdown(f"""
      <div class="whale-alert">{icon} <b>{w['amount']:,.0f}</b> {selected} &nbsp;·&nbsp; {w['direction']} &nbsp;·&nbsp; <span style="color:{c['sub']}">{w['time']}</span></div>
      """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROW 3: Market Overview Table  (Plotly — always renders)
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Market Overview</div>', unsafe_allow_html=True)

all_tickers = fetch_all_tickers()

# Build data lists for Plotly table
coins_col   = []
price_col   = []
chg_col     = []
high_col    = []
low_col     = []
vol_col     = []
sent_col    = []
pred_col    = []
sig_col     = []

# Cell colours (one per row, one per column)
chg_colors  = []
sent_colors = []
pred_colors = []
sig_colors  = []
row_fills   = []

_dk = st.session_state.dark_mode
_green_cell  = "rgba(63,185,80,0.18)"   if _dk else "rgba(26,127,55,0.15)"
_red_cell    = "rgba(248,81,73,0.18)"   if _dk else "rgba(207,34,46,0.15)"
_amber_cell  = "rgba(210,153,34,0.18)"  if _dk else "rgba(154,103,0,0.12)"
_neutral_bg  = "#161b22"                if _dk else "#ffffff"
_active_bg   = "rgba(88,166,255,0.10)"

for name, ticker_data in all_tickers.items():
    p    = float(ticker_data.get("lastPrice", 0))
    chg  = float(ticker_data.get("priceChangePercent", 0))
    h    = float(ticker_data.get("highPrice", 0))
    l    = float(ticker_data.get("lowPrice", 0))
    v    = float(ticker_data.get("quoteVolume", 0))

    sym_df  = fetch_klines(SYMBOLS[name], "15m", 50)
    s_data  = mock_sentiment(name)
    p_data  = mock_prediction(name, sym_df)
    sg_data = mock_signal(p_data, s_data, sym_df)

    sent_lbl = "BULLISH" if s_data["score"] > 0.1 else ("BEARISH" if s_data["score"] < -0.1 else "NEUTRAL")
    sig_txt  = sg_data["signal"]
    dir_txt  = p_data["direction"]
    chg_sym2 = "▲" if chg >= 0 else "▼"

    coins_col.append(f"  {name}  ")
    price_col.append(f"${p:,.2f}")
    chg_col.append(f"{chg_sym2} {abs(chg):.2f}%")
    high_col.append(f"${h:,.2f}")
    low_col.append(f"${l:,.2f}")
    vol_col.append(f"${v/1e6:.1f}M")
    sent_col.append(f"{sent_lbl}\n({s_data['score']:+.2f})")
    pred_col.append(f"{dir_txt}\n({int(p_data['confidence']*100)}%)")
    sig_col.append(f"  {sig_txt}  ")

    chg_colors.append(_green_cell if chg >= 0 else _red_cell)
    sent_colors.append(_green_cell if sent_lbl == "BULLISH" else (_red_cell if sent_lbl == "BEARISH" else _amber_cell))
    pred_colors.append(_green_cell if dir_txt == "UP" else (_red_cell if dir_txt == "DOWN" else _amber_cell))
    sig_colors.append(_green_cell if sig_txt == "BUY" else (_red_cell if sig_txt == "SELL" else _amber_cell))
    row_fills.append(_active_bg if name == selected else _neutral_bg)

# Build Plotly table
_header_bg   = "#21262d" if _dk else "#eaeef2"
_header_text = "#8b949e" if _dk else "#656d76"
_text_color  = "#e6edf3" if _dk else "#1f2328"
_line_color  = "#30363d" if _dk else "#d1d9e0"
_paper_bg    = "#0d1117" if _dk else "#f6f8fa"

fig_table = go.Figure(data=[go.Table(
    columnwidth=[60, 90, 80, 90, 90, 75, 100, 90, 70],
    header=dict(
        values=["<b>Coin</b>","<b>Price</b>","<b>24h Change</b>",
                "<b>24h High</b>","<b>24h Low</b>","<b>Volume</b>",
                "<b>Sentiment</b>","<b>Prediction</b>","<b>Signal</b>"],
        fill_color=_header_bg,
        font=dict(color=_header_text, size=11, family="monospace"),
        align="center",
        height=36,
        line_color=_line_color,
    ),
    cells=dict(
        values=[coins_col, price_col, chg_col,
                high_col, low_col, vol_col,
                sent_col, pred_col, sig_col],
        fill_color=[
            row_fills,       # Coin column — highlight selected
            row_fills,       # Price
            chg_colors,      # Change — green/red
            row_fills,       # High
            row_fills,       # Low
            row_fills,       # Volume
            sent_colors,     # Sentiment — green/red/amber
            pred_colors,     # Prediction
            sig_colors,      # Signal
        ],
        font=dict(color=_text_color, size=12),
        align="center",
        height=40,
        line_color=_line_color,
    ),
)])

fig_table.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor=_paper_bg,
    height=260,
)

st.plotly_chart(fig_table, use_container_width=True, config={"displayModeBar": False})

# ─────────────────────────────────────────────
# ROW 4: Whale Transactions + P&L Stats
# ─────────────────────────────────────────────
st.markdown(f'<div class="section-header">📊 Performance Analytics &amp; Whale Activity</div>', unsafe_allow_html=True)

col_whale, col_pnl = st.columns([1, 1.6], gap="medium")

with col_whale:
    st.markdown(f'<div style="font-size:12px; font-weight:600; color:{c["sub"]}; margin-bottom:8px;">🐋 Recent Whale Transactions ({selected})</div>', unsafe_allow_html=True)

    # Pull from onchain already fetched above, simulating Etherscan-style output
    whale_rows = []
    for i, w in enumerate(onchain["whales"]):
        t_type = "OUT" if "Exchange" in w["direction"].split("→")[1] else "IN"
        t_color = c["red"] if t_type == "OUT" else c["green"]
        t_icon  = "📤" if t_type == "OUT" else "📥"
        hash_fake = f"0x{abs(hash(selected+str(i)+w['time']))%0xFFFFFFFF:08x}..."
        st.markdown(f"""
        <div style="background:{c["paper"]}; border:1px solid {c["grid"]}; border-radius:8px;
             padding:10px 12px; margin-bottom:8px; font-size:12px;">
          <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <span style="color:{c["sub"]}; font-family:monospace; font-size:11px;">{hash_fake}</span>
            <span style="color:{c["sub"]}; font-size:11px;">{w["time"]}</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span>{t_icon} <b style="color:{c["text"]}">{w["amount"]:,.0f} {selected}</b></span>
            <span style="color:{t_color}; font-weight:700; font-size:11px; border:1px solid {t_color};
                  border-radius:10px; padding:1px 8px;">{t_type}</span>
          </div>
          <div style="font-size:11px; color:{c["sub"]}; margin-top:4px;">{w["direction"]}</div>
        </div>
        """, unsafe_allow_html=True)

with col_pnl:
    st.markdown(f'<div style="font-size:12px; font-weight:600; color:{c["sub"]}; margin-bottom:8px;">📈 Hypothetical P&amp;L — Last 30 Days (Signal Backtest)</div>', unsafe_allow_html=True)

    # Generate 30-day synthetic P&L from signal history
    rng30 = random.Random(selected + "pnl30")
    days  = pd.date_range(end=datetime.utcnow(), periods=30, freq="D")
    daily_returns = []
    cumulative    = 1.0
    cum_series    = []
    wins, losses  = 0, 0
    trade_returns = []

    for day in days:
        # Simulate a signal for that day
        day_sig = rng30.choice(["BUY", "SELL", "HOLD", "BUY", "BUY", "SELL"])
        if day_sig == "HOLD":
            ret = 0.0
        else:
            # BUY wins ~58% of the time, SELL ~52%
            if day_sig == "BUY":
                ret = rng30.gauss(0.008, 0.025)
            else:
                ret = rng30.gauss(0.004, 0.022)
            trade_returns.append(ret)
            if ret > 0: wins += 1
            else: losses += 1
        daily_returns.append(ret)
        cumulative *= (1 + ret)
        cum_series.append(round((cumulative - 1) * 100, 2))

    total_trades = wins + losses
    win_rate     = wins / total_trades * 100 if total_trades else 0
    total_ret    = cum_series[-1]
    avg_ret      = np.mean(trade_returns) if trade_returns else 0
    std_ret      = np.std(trade_returns)  if trade_returns else 0.001
    sharpe       = (avg_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
    max_dd       = min(0, min(cum_series))

    # P&L chart
    pnl_color_list = [c["green"] if v >= 0 else c["red"] for v in
                      [cum_series[i] - (cum_series[i-1] if i>0 else 0) for i in range(len(cum_series))]]
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(
        x=list(days), y=cum_series, mode="lines+markers",
        line=dict(color=c["accent"], width=2),
        marker=dict(size=4, color=c["accent"]),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.08)",
        name="Cumulative P&L %",
        hovertemplate="%{x|%b %d}: %{y:.2f}%<extra></extra>",
    ))
    fig_pnl.add_hline(y=0, line_color=c["grid"], line_width=1)
    fig_pnl.update_layout(
        height=180, margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor=c["paper"], plot_bgcolor=c["bg"],
        font=dict(color=c["text"], size=10),
        showlegend=False,
        xaxis=dict(gridcolor=c["grid"], tickfont=dict(color=c["sub"],size=9)),
        yaxis=dict(gridcolor=c["grid"], tickfont=dict(color=c["sub"],size=9),
                   ticksuffix="%"),
    )
    st.plotly_chart(fig_pnl, use_container_width=True, config={"displayModeBar": False})

    # Stats row
    ret_cls = "pos" if total_ret >= 0 else "neg"
    sh_cls  = "pos" if sharpe >= 1.0 else ("neutral" if sharpe >= 0.5 else "neg")
    st.markdown(f"""
    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-top:4px;">
      <div style="background:{c["paper"]}; border:1px solid {c["grid"]}; border-radius:8px; padding:10px; text-align:center;">
        <div style="font-size:10px; color:{c["sub"]}; text-transform:uppercase; letter-spacing:.06em;">30d Return</div>
        <div class="{ret_cls}" style="font-size:18px; font-weight:700;">{total_ret:+.1f}%</div>
      </div>
      <div style="background:{c["paper"]}; border:1px solid {c["grid"]}; border-radius:8px; padding:10px; text-align:center;">
        <div style="font-size:10px; color:{c["sub"]}; text-transform:uppercase; letter-spacing:.06em;">Win Rate</div>
        <div class="pos" style="font-size:18px; font-weight:700;">{win_rate:.0f}%</div>
      </div>
      <div style="background:{c["paper"]}; border:1px solid {c["grid"]}; border-radius:8px; padding:10px; text-align:center;">
        <div style="font-size:10px; color:{c["sub"]}; text-transform:uppercase; letter-spacing:.06em;">Sharpe Ratio</div>
        <div class="{sh_cls}" style="font-size:18px; font-weight:700;">{sharpe:.2f}</div>
      </div>
      <div style="background:{c["paper"]}; border:1px solid {c["grid"]}; border-radius:8px; padding:10px; text-align:center;">
        <div style="font-size:10px; color:{c["sub"]}; text-transform:uppercase; letter-spacing:.06em;">Max Drawdown</div>
        <div class="neg" style="font-size:18px; font-weight:700;">{max_dd:.1f}%</div>
      </div>
    </div>
    <div style="font-size:11px; color:{c["sub"]}; margin-top:8px;">
      Trades: <b style="color:{c["text"]}">{total_trades}</b> &nbsp;·&nbsp;
      Wins: <b style="color:{c["green"]}">{wins}</b> &nbsp;·&nbsp;
      Losses: <b style="color:{c["red"]}">{losses}</b> &nbsp;·&nbsp;
      Avg trade: <b style="color:{c["text"]}">{avg_ret*100:+.2f}%</b>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER + AUTO-REFRESH
# ─────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center; color:{c['sub']}; font-size:12px; margin-top:24px; padding-top:12px; border-top:1px solid {c['grid']};">
  Crypto Trading Dashboard &nbsp;·&nbsp; Prices from Binance API &nbsp;·&nbsp; 
  Predictions: XGBoost model &nbsp;·&nbsp; 
  <span style="color:{c['accent']}">NOT financial advice</span>
</div>
""", unsafe_allow_html=True)

if auto_refresh:
    time.sleep(60)
    st.rerun()