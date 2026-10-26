
import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timezone

st.set_page_config(page_title="Noman Trading Bot", page_icon="📈", layout="wide")
st.title("📈 Noman Trading Bot")
st.caption("DEMO / PAPER TRADING ONLY — কোনো বাস্তব অর্ডার পাঠানো হয় না")

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT"]

with st.sidebar:
    st.header("Bot settings")
    symbols = st.multiselect("যে কয়েন স্ক্যান করবে", DEFAULT_SYMBOLS, default=["BTCUSDT", "ETHUSDT"])
    interval = st.selectbox("Signal timeframe", ["15m", "1h", "4h"], index=1)
    risk_pct = st.number_input("প্রতি ট্রেডে সর্বোচ্চ ঝুঁকি (%)", min_value=0.1, max_value=2.0, value=2.0, step=0.1)
    paper_balance = st.number_input("Demo balance (USDT)", min_value=10.0, value=100.0, step=10.0)
    refresh = st.button("🔄 Scan market", use_container_width=True)

st.info("এই সংস্করণে 2% হলো প্রতি ট্রেডে নির্ধারিত সর্বোচ্চ account-risk budget; এটি লাভ বা নিরাপত্তার নিশ্চয়তা নয়। বট কেবল সিগন্যাল ও paper-trade উদাহরণ দেখায়।")

@st.cache_data(ttl=45)
def get_klines(symbol, interval, limit=180):
    url = "https://fapi.binance.com/fapi/v1/klines"
    r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=12)
    r.raise_for_status()
    raw = r.json()
    df = pd.DataFrame(raw, columns=["open_time","open","high","low","close","volume","close_time","qav","trades","taker_base","taker_quote","ignore"])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def indicators(df):
    c = df["close"]
    df = df.copy()
    df["ema12"] = c.ewm(span=12, adjust=False).mean()
    df["ema44"] = c.ewm(span=44, adjust=False).mean()
    df["ema99"] = c.ewm(span=99, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - signal
    return df

def evaluate(df):
    # Ignore the currently-forming candle.
    d = indicators(df.iloc[:-1].copy())
    a, b = d.iloc[-1], d.iloc[-2]
    bullish = a.ema12 > a.ema44 > a.ema99 and 45 <= a.rsi <= 68 and a.macd_hist > 0 and b.macd_hist <= a.macd_hist
    bearish = a.ema12 < a.ema44 < a.ema99 and 32 <= a.rsi <= 55 and a.macd_hist < 0 and b.macd_hist >= a.macd_hist
    if bullish:
        side, reason = "LONG", "EMA trend bullish, RSI ও MACD confirmation"
    elif bearish:
        side, reason = "SHORT", "EMA trend bearish, RSI ও MACD confirmation"
    else:
        side, reason = "WAIT", "সব entry condition একসঙ্গে মেলেনি"
    return side, reason, a

if not symbols:
    st.warning("কমপক্ষে একটি symbol নির্বাচন করুন।")
else:
    results = []
    with st.spinner("Binance Futures public market data পড়া হচ্ছে..."):
        for symbol in symbols:
            try:
                df = get_klines(symbol, interval)
                side, reason, row = evaluate(df)
                price = float(row.close)
                # Illustrative 1.5% stop distance. Position sizing is informational only.
                stop_distance = price * 0.015
                risk_cash = paper_balance * risk_pct / 100
                qty = risk_cash / stop_distance if side != "WAIT" else 0
                results.append({
                    "Symbol": symbol, "Signal": side, "Last closed price": price,
                    "RSI": round(float(row.rsi), 1), "EMA12": float(row.ema12),
                    "EMA44": float(row.ema44), "EMA99": float(row.ema99),
                    "MACD hist": float(row.macd_hist), "Reason": reason,
                    "Illustrative qty": round(qty, 6) if qty else "—"
                })
            except Exception as e:
                results.append({"Symbol": symbol, "Signal": "ERROR", "Reason": str(e)[:140]})
    st.subheader(f"Market scan · {interval}")
    st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
    st.caption("দাম ও সূচক সর্বশেষ সম্পূর্ণ candle থেকে। Signal কোনো financial advice নয়; live order execution নিষ্ক্রিয়।")

st.divider()
st.subheader("Risk rules (demo)")
st.markdown("""
- প্রতি ট্রেডে ঝুঁকির বাজেট: Demo balance-এর সর্বোচ্চ 2%।
- উদাহরণ stop distance: entry থেকে 1.5%; বাস্তব বাজারের volatility অনুযায়ী এটি যথেষ্ট নাও হতে পারে।
- একই symbol-এ একাধিক signal এলে আগে open-position / exposure check দরকার।
- Fees, funding, slippage, partial fills ও liquidation এই প্রাথমিক সংস্করণে সিমুলেট করা হয় না।
""")
st.caption("Data source: Binance USDⓈ-M Futures public klines API. Built for learning and paper review.")
