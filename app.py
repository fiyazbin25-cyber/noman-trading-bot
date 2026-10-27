import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Noman Trading Bot — DEMO / PAPER ONLY
# No API keys, no order placement, no real trading.

BASE = "https://fapi.binance.com"
st.set_page_config(page_title="Noman Trading Bot", page_icon="📈", layout="wide")
st.title("📈 Noman Trading Bot")
st.caption("Binance Futures public market data • Demo only • No real orders")

session = requests.Session()
session.headers.update({"User-Agent": "NomanTradingBot/1.0"})


def api_get(path, params=None, timeout=12):
    response = session.get(BASE + path, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=3600)
def get_symbols():
    info = api_get("/fapi/v1/exchangeInfo")
    return [
        s["symbol"] for s in info["symbols"]
        if s.get("quoteAsset") == "USDT"
        and s.get("contractType") == "PERPETUAL"
        and s.get("status") == "TRADING"
    ]


@st.cache_data(ttl=30)
def get_tickers():
    return api_get("/fapi/v1/ticker/24hr")


@st.cache_data(ttl=15)
def get_klines(symbol, interval="15m", limit=180):
    raw = api_get("/fapi/v1/klines", {
        "symbol": symbol, "interval": interval, "limit": limit
    })
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(raw, columns=cols)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    # Ignore the still-forming last candle.
    if len(df) > 1:
        df = df.iloc[:-1].copy()
    return df.reset_index(drop=True)


def indicators(df):
    d = df.copy()
    d["ema12"] = d["close"].ewm(span=12, adjust=False).mean()
    d["ema44"] = d["close"].ewm(span=44, adjust=False).mean()
    delta = d["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = 100 - (100 / (1 + rs))
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    d["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - d["close"].shift()).abs(),
        (d["low"] - d["close"].shift()).abs()
    ], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    return d


def make_signal(df):
    d = indicators(df).dropna()
    if len(d) < 5:
        return "WAIT", d.iloc[-1]
    x = d.iloc[-1]
    if x["close"] > x["ema12"] > x["ema44"] and x["rsi"] > 52 and x["macd_hist"] > 0:
        return "LONG", x
    if x["close"] < x["ema12"] < x["ema44"] and x["rsi"] < 48 and x["macd_hist"] < 0:
        return "SHORT", x
    return "WAIT", x


def two_minute_reversal(symbol):
    # Binance Futures has no native 2m interval. Aggregate completed 1m candles.
    raw = api_get("/fapi/v1/klines", {"symbol": symbol, "interval": "1m", "limit": 12})
    cols = ["t", "o", "h", "l", "c", "v", "ct", "qv", "n", "tb", "tq", "ig"]
    d = pd.DataFrame(raw, columns=cols)
    for c in ["o", "h", "l", "c", "v"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["t"] = pd.to_datetime(d["t"], unit="ms", utc=True)
    d = d.iloc[:-1].copy()  # discard current incomplete 1m candle
    d["bucket"] = d["t"].dt.floor("2min")
    agg = d.groupby("bucket").agg(
        open=("o", "first"), high=("h", "max"), low=("l", "min"),
        close=("c", "last"), volume=("v", "sum")
    ).reset_index()
    if len(agg) < 3:
        return None, None, None
    return agg.iloc[-2], agg.iloc[-1], agg


def fmt(x):
    return f"{x:.8g}"


# Sidebar controls
st.sidebar.header("Scanner settings")
top_n = st.sidebar.selectbox("কয়েন সংখ্যা", [10, 15, 20, 25], index=1)
timeframe = st.sidebar.selectbox("Signal timeframe", ["15m", "1h", "4h"], index=0)
st.sidebar.caption("সিগন্যালের জন্য বন্ধ হওয়া ক্যান্ডেল ব্যবহার করা হয়।")

st.sidebar.header("Demo position settings")
symbol_input = st.sidebar.text_input("কয়েন (যেমন BTCUSDT)", value="BTCUSDT").upper().strip()
side = st.sidebar.selectbox("Demo position", ["SHORT", "LONG"])
entry = st.sidebar.number_input("Entry price", min_value=0.00000001, value=0.20, format="%.8f")
margin = st.sidebar.number_input("Margin ($)", min_value=0.1, value=2.0, step=0.5)
leverage = st.sidebar.number_input("Leverage (x)", min_value=1, max_value=125, value=10, step=1)
tp1 = st.sidebar.number_input("Profit lock starts ($)", min_value=0.1, value=5.0, step=1.0)
tp2 = st.sidebar.number_input("Next target ($)", min_value=0.1, value=10.0, step=1.0)
tp3 = st.sidebar.number_input("Final target ($)", min_value=0.1, value=15.0, step=1.0)
giveback = st.sidebar.slider("Peak profit থেকে কত % ফেরত এলে বন্ধ করবে", 5, 80, 20, 5)
st.sidebar.caption("এগুলো কেবল ডেমো হিসাব। ফি, funding, slippage ও liquidation ধরা হয়নি।")

if "paper_peak" not in st.session_state:
    st.session_state.paper_peak = 0.0
if "paper_closed" not in st.session_state:
    st.session_state.paper_closed = False
if "paper_last_symbol" not in st.session_state:
    st.session_state.paper_last_symbol = ""

if st.sidebar.button("ডেমো ট্রেড নতুন করে শুরু / Reset"):
    st.session_state.paper_peak = 0.0
    st.session_state.paper_closed = False
    st.session_state.paper_last_symbol = symbol_input
    st.rerun()

if st.session_state.paper_last_symbol != symbol_input:
    st.session_state.paper_peak = 0.0
    st.session_state.paper_closed = False
    st.session_state.paper_last_symbol = symbol_input

# Scanner
st.subheader("Market scanner")
try:
    tickers = get_tickers()
    symbols = set(get_symbols())
    tdf = pd.DataFrame(tickers)
    tdf = tdf[tdf["symbol"].isin(symbols)].copy()
    tdf["quoteVolume"] = pd.to_numeric(tdf["quoteVolume"], errors="coerce")
    tdf = tdf.nlargest(top_n, "quoteVolume")
    rows = []
    for sym in tdf["symbol"].tolist():
        try:
            candles = get_klines(sym, timeframe, 160)
            signal, last = make_signal(candles)
            rows.append({
                "Coin": sym,
                "Signal": signal,
                "Price": fmt(float(last["close"])),
                "RSI": round(float(last["rsi"]), 1),
                "EMA 12": fmt(float(last["ema12"])),
                "EMA 44": fmt(float(last["ema44"])),
                "24h Volume (USDT)": round(float(
                    tdf.loc[tdf["symbol"] == sym, "quoteVolume"].iloc[0]
                ), 0)
            })
        except Exception:
            continue
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning("এখন সিগন্যাল পাওয়া যায়নি। কিছুক্ষণ পর আবার চেষ্টা করুন।")
except Exception as e:
    st.error(f"Market data load হয়নি: {e}")
    st.info("ইন্টারনেট বা Binance public API সাময়িক ব্যস্ত থাকতে পারে।")

# Paper trade monitor
st.subheader("🛡️ Smart Profit Lock — Demo monitor")
st.write("এটি তোমার দেওয়া Entry, Side, Margin ও Leverage দিয়ে কল্পিত লাভ হিসাব করে। কোনো অর্ডার দেয় না।")

try:
    sym = symbol_input
    if not sym.endswith("USDT"):
        st.warning("কয়েনের নাম USDT দিয়ে শেষ করো, যেমন ETHUSDT।")
    else:
        price_data = api_get("/fapi/v1/ticker/price", {"symbol": sym})
        current_price = float(price_data["price"])
        notional = margin * leverage
        qty = notional / entry
        if side == "LONG":
            pnl = (current_price - entry) * qty
        else:
            pnl = (entry - current_price) * qty

        st.session_state.paper_peak = max(st.session_state.paper_peak, pnl)
        peak = st.session_state.paper_peak
        lock_active = peak >= tp1
        close_now = False
        reason = "এখনও লাভের সুরক্ষা চালু হয়নি।"

        if not st.session_state.paper_closed and lock_active:
            protected_floor = max(tp1, peak * (1 - giveback / 100))
            if pnl <= protected_floor:
                close_now = True
                reason = f"লাভ peak ${peak:.2f} থেকে কমে সুরক্ষার সীমা ${protected_floor:.2f}-এ এসেছে।"
            elif peak >= tp3 and pnl >= tp3:
                reason = f"TP3 স্তর ${tp3:.2f} পার হয়েছে; রিভার্সাল হলে লাভ সুরক্ষিত করবে।"
            elif peak >= tp2 and pnl >= tp2:
                reason = f"TP2 স্তর ${tp2:.2f} পার হয়েছে; আরও লাভের সুযোগ রাখছে।"
            else:
                reason = f"Profit Lock চালু। সর্বোচ্চ দেখা লাভ ${peak:.2f}; ফেরত সীমা ${protected_floor:.2f}।"

        if close_now:
            st.session_state.paper_closed = True

        c1, c2, c3 = st.columns(3)
        c1.metric("Current price", fmt(current_price))
        c2.metric("Estimated PnL", f"${pnl:.2f}")
        c3.metric("Highest PnL seen", f"${peak:.2f}")

        st.write(f"আনুমানিক position value: **${notional:.2f}** | পরিমাণ: **{qty:.6g} {sym.replace('USDT','')}**")
        if st.session_state.paper_closed:
            st.error("ডেমো নিয়ম অনুযায়ী ট্রেড বন্ধ হয়েছে। " + reason)
        elif lock_active:
            st.success(reason)
        else:
            st.info(reason)

        st.caption(
            "নোট: এই ডেমো মনিটর অ্যাপ খোলা ও refresh চলাকালীন peak profit মনে রাখে। "
            "অ্যাপ বন্ধ/রিস্টার্ট হলে হিসাব reset হতে পারে। ২ মিনিটের candle Binance-এর 1m candle "
            "দুটি জুড়ে বিশ্লেষণ করা হয়; এই সংস্করণে live reversal trigger সরল peak-profit giveback "
            "নিয়ম দিয়ে সিমুলেট করা হয়েছে।"
        )
except Exception as e:
    st.error(f"ডেমো মনিটর চালানো যায়নি: {e}")

st.divider()
st.caption("ঝুঁকি সতর্কতা: Futures-এ margin হারাতে পারো। এটি শেখা ও পরীক্ষার জন্য, আর্থিক পরামর্শ নয়।")
