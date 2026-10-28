import numpy as np
import pandas as pd
import requests
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

# NOMAN TRADING BOT — PAPER ONLY. No API keys and no real orders.
BASE = "https://fapi.binance.com"
st.set_page_config(page_title="Noman Trading Bot", page_icon="📈", layout="wide")
st.title("📈 Noman Trading Bot")
st.caption("Binance USDⓈ-M Futures | USDT perpetual scanner | Demo only")

@st.cache_resource
def session():
    s = requests.Session()
    s.headers.update({"User-Agent": "NomanTradingBot/3.0"})
    return s

def api(path, params=None):
    r = session().get(BASE + path, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=1800)
def symbols():
    x = api("/fapi/v1/exchangeInfo")
    return [a["symbol"] for a in x["symbols"]
            if a.get("quoteAsset") == "USDT" and a.get("contractType") == "PERPETUAL"
            and a.get("status") == "TRADING"]

@st.cache_data(ttl=45)
def tickers():
    return api("/fapi/v1/ticker/24hr")

def candles(symbol, tf, limit=100):
    raw = api("/fapi/v1/klines", {"symbol": symbol, "interval": tf, "limit": limit})
    names = ["t","open","high","low","close","volume","ct","qv","trades","taker_buy_base","taker_buy_quote","ignore"]
    d = pd.DataFrame(raw, columns=names)
    for c in ["open","high","low","close","volume","qv","taker_buy_base"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.iloc[:-1].reset_index(drop=True)  # completed candles only

def indicators(d):
    d = d.copy()
    d["ema12"] = d.close.ewm(span=12, adjust=False).mean()
    d["ema44"] = d.close.ewm(span=44, adjust=False).mean()
    ch = d.close.diff()
    gain = ch.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-ch.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = 100 - 100/(1+rs)
    macd = d.close.ewm(span=12, adjust=False).mean() - d.close.ewm(span=26, adjust=False).mean()
    d["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    d["vol_avg"] = d.volume.rolling(20).mean()
    d["vol_ratio"] = d.volume / d.vol_avg.replace(0, np.nan)
    d["buy_ratio"] = d.taker_buy_base / d.volume.replace(0, np.nan)
    return d

def trend(symbol, tf):
    d = indicators(candles(symbol, tf, 80)).dropna()
    if len(d) < 2: return "N/A"
    x = d.iloc[-1]
    if x.close > x.ema12 > x.ema44 and x.macd_hist > 0: return "UP"
    if x.close < x.ema12 < x.ema44 and x.macd_hist < 0: return "DOWN"
    return "SIDE"

def scan(symbol, tf, vol24):
    try:
        d = indicators(candles(symbol, tf, 100)).dropna()
        if len(d) < 22: return None
        x, p = d.iloc[-1], d.iloc[-2]
        br = float(x.buy_ratio) if pd.notna(x.buy_ratio) else .5
        vr = float(x.vol_ratio) if pd.notna(x.vol_ratio) else 0
        h1, h4 = trend(symbol, "1h"), trend(symbol, "4h")
        long_score = sum([
            x.close > x.ema12, x.ema12 > x.ema44, x.rsi > 52,
            x.macd_hist > 0, br > .55, vr >= 1.5 and x.close > p.close,
            h1 == "UP", h4 == "UP"])
        short_score = sum([
            x.close < x.ema12, x.ema12 < x.ema44, x.rsi < 48,
            x.macd_hist < 0, br < .45, vr >= 1.5 and x.close < p.close,
            h1 == "DOWN", h4 == "DOWN"])
        sig = "WAIT"
        if long_score >= 6 and long_score >= short_score + 2: sig = "LONG"
        elif short_score >= 6 and short_score >= long_score + 2: sig = "SHORT"
        pressure = "Buyers" if br > .55 else ("Sellers" if br < .45 else "Balanced")
        return {"Coin":symbol, "Signal":sig, "Long score":f"{long_score}/8",
                "Short score":f"{short_score}/8", "Price":float(x.close),
                "RSI":round(float(x.rsi),1), "Volume / 20 avg":round(vr,2),
                "Taker buy %":round(br*100,1), "Pressure":pressure,
                "1h trend":h1, "4h trend":h4, "24h Volume USDT":float(vol24)}
    except Exception:
        return None

def two_minute(symbol):
    # Binance Futures has no native 2m interval; aggregate closed 1m candles.
    raw = api("/fapi/v1/klines", {"symbol":symbol, "interval":"1m", "limit":8})
    d = pd.DataFrame(raw, columns=["t","o","h","l","c","v","ct","qv","n","tb","tq","ig"])
    for c in ["o","h","l","c","v"]: d[c] = pd.to_numeric(d[c], errors="coerce")
    d["t"] = pd.to_datetime(d.t, unit="ms", utc=True)
    d = d.iloc[:-1].copy()
    d["bucket"] = d.t.dt.floor("2min")
    return d.groupby("bucket").agg(open=("o","first"), close=("c","last")).reset_index()

st.sidebar.header("Market scanner")
tf = st.sidebar.selectbox("Signal timeframe", ["15m","1h","4h"])
scope = st.sidebar.radio("Coins", ["All USDT perpetuals","Top 50 by volume"])
workers = st.sidebar.selectbox("Parallel requests", [2,3,4], index=1)
st.sidebar.caption("All-contract scan may take time. Requests are limited to reduce API pressure.")

st.sidebar.header("Paper trade")
sym = st.sidebar.text_input("Coin (example ETHUSDT)", "ETHUSDT").upper().strip()
side = st.sidebar.selectbox("Direction", ["SHORT","LONG"])
margin = st.sidebar.number_input("Margin ($)", min_value=.1, value=2.0, step=.5)
lev = st.sidebar.number_input("Leverage (x)", min_value=1, max_value=125, value=10)
if "paper" not in st.session_state: st.session_state.paper = None
c1, c2 = st.sidebar.columns(2)
if c1.button("Start demo"):
    try:
        px = float(api("/fapi/v1/ticker/price", {"symbol":sym})["price"])
        st.session_state.paper = {"symbol":sym,"side":side,"entry":px,"margin":margin,
                                  "lev":lev,"peak":0.0,"closed":False}
    except Exception as e: st.sidebar.error(str(e))
if c2.button("Reset"):
    st.session_state.paper = None

st.subheader("🔎 Binance Futures scanner")
try:
    eligible = symbols()
    td = pd.DataFrame(tickers())
    td = td[td.symbol.isin(eligible)].copy()
    td["quoteVolume"] = pd.to_numeric(td.quoteVolume, errors="coerce")
    td = td.sort_values("quoteVolume", ascending=False)
    selected = td.symbol.head(50).tolist() if scope.startswith("Top") else td.symbol.tolist()
    vm = td.set_index("symbol").quoteVolume.to_dict()
    st.write(f"Eligible contracts: **{len(eligible)}** | Scanning: **{len(selected)}**")
    bar, status = st.progress(0), st.empty()
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(scan,s,tf,vm.get(s,0)) for s in selected]
        for i, f in enumerate(as_completed(futures), 1):
            z = f.result()
            if z: rows.append(z)
            if i % 8 == 0 or i == len(futures):
                bar.progress(i/len(futures)); status.caption(f"Scanned {i}/{len(futures)}")
    if rows:
        out = pd.DataFrame(rows).sort_values("24h Volume USDT", ascending=False)
        st.dataframe(out, use_container_width=True, hide_index=True)
        st.caption("Scores are rule-based confluence, not win probabilities. WAIT means no strong agreement.")
    else: st.warning("No results returned; try again later.")
except Exception as e:
    st.error(f"Scanner error: {e}")
    st.info("Binance public API may be busy or rate-limited.")

st.subheader("🛡️ Smart Profit Lock — Demo only")
st.write("Profit milestones: $5 / $10 / $15. A completed 2-minute reversal is used as the exit confirmation.")
p = st.session_state.paper
if p is None:
    st.info("Choose a coin and direction in the sidebar, then press Start demo.")
else:
    try:
        price = float(api("/fapi/v1/ticker/price", {"symbol":p["symbol"]})["price"])
        notional = p["margin"] * p["lev"]
        qty = notional / p["entry"]
        pnl = ((price-p["entry"]) if p["side"]=="LONG" else (p["entry"]-price))*qty
        p["peak"] = max(p["peak"], pnl)
        peak = p["peak"]
        floor = 15.0 if peak >= 15 else (10.0 if peak >= 10 else (5.0 if peak >= 5 else 0.0))
        bars = two_minute(p["symbol"])
        reversal = False
        if len(bars) >= 2:
            last, prev = bars.iloc[-1], bars.iloc[-2]
            reversal = (last.close > last.open and last.close > prev.close) if p["side"]=="SHORT" else (last.close < last.open and last.close < prev.close)
        if not p["closed"] and floor and reversal and pnl >= floor:
            p["closed"] = True
            p["exit_note"] = f"2m reversal; estimated paper PnL ${pnl:.2f}"
        a,b,c = st.columns(3)
        a.metric("Entry",f"{p['entry']:.8g}"); b.metric("Current",f"{price:.8g}"); c.metric("PnL",f"${pnl:.2f}")
        d,e,f = st.columns(3)
        d.metric("Peak PnL",f"${peak:.2f}"); e.metric("Profit floor",f"${floor:.2f}"); f.metric("Position value",f"${notional:.2f}")
        st.write(f"**{p['symbol']} {p['side']}** | Quantity: {qty:.8g}")
        if p["closed"]: st.success("Demo position closed: "+p["exit_note"])
        elif floor: st.info(f"Profit lock active. 2m reversal detected: {reversal}.")
        else: st.info("Profit lock starts after estimated PnL reaches $5.")
        st.caption("Paper simulation only. Peak tracking may reset if app sleeps/restarts. Fees, funding, slippage, liquidation and actual order execution are not simulated.")
    except Exception as e: st.error(f"Paper monitor error: {e}")

st.divider()
st.warning("Futures are high risk. This scanner does not guarantee signals or profits and never places real orders.")
