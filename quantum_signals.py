#!/usr/bin/env python3
"""
Quantum Pro — signal bot (Python port of the TradingView strategy's confluence engine).

Runs statelessly on a cron (e.g. GitHub Actions), evaluates the LAST CLOSED candle
for each configured symbol/timeframe, and pushes BUY/SELL alerts to Telegram.

⚠ Not byte-identical to the Pine version (different feed / EMA seeding / rounding).
   It ports the 5-bucket confluence + A/B/C tier logic faithfully. Validate parity
   against TradingView and paper-trade before trusting it with capital.
"""
import os, json, time, sys, datetime
import numpy as np
import pandas as pd
import requests

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (edit config.json, or rely on these defaults — they mirror the Pine defaults)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "data_host": "https://data-api.binance.vision",  # public Binance data (no auth, no US geo-block)
    "symbols": ["BTCUSDT"],
    "timeframe": "15m",              # fallback when `timeframes` is not set
    "timeframes": ["15m", "1h", "4h", "1d", "1w"],   # scan each per symbol (1m excluded: unusable on cron)
    "htf": ["1h", "4h", "1d"],       # fallback only; HTF is derived per-TF via higher_tfs()
    "limit": 600,                    # candles to fetch (warmup for EMA200 / VP)
    # signal gate (bucket confluence)
    "minBuckets": 3,
    "maxOpposite": 1,
    "tradeTier": "A + B",            # "A only" | "A + B" | "A + B + C"
    # filters
    "minADX": 18.0,
    "minVolumeRatio": 1.0,
    "maxVolatilityPct": 6.0,
    "minVolatilityPct": 0.0,         # low-vol floor (0 = off)
    "minCandleBodyRatio": 0.3,
    # order flow / VP
    "enableImbalance": True,
    "imbalanceThreshold": 2.0,
    "enableVolumeProfile": True,
    "vpLength": 150,
    "vpBins": 50,
    "valueAreaPct": 70.0,
    # entry-quality
    "usePullback": False,
    # risk / levels
    "atrMultSL": 1.5,
    "maxStopPct": 5.0,
    "tp1R": 1.0,
    "tp2R": 2.0,
    "useStructuralStop": True,
    "swingStopLen": 10,
    # signal spacing (mirror Pine: no repeat direction until opposite fires, + cooldown bars)
    "allowRepeatDirection": False,
    "cooldownBars": 4,
    "heartbeat": True,               # once-a-day "bot alive" post so the channel is never silent
}

INTERVAL_MS = {"1m":60_000,"3m":180_000,"5m":300_000,"15m":900_000,"30m":1_800_000,
               "1h":3_600_000,"2h":7_200_000,"4h":14_400_000,"6h":21_600_000,
               "8h":28_800_000,"12h":43_200_000,"1d":86_400_000,"3d":259_200_000,
               "1w":604_800_000,"1M":2_592_000_000}

# Coarse ladder for higher-timeframe trend confirmation (picks the next 3 above the signal TF)
HTF_LADDER = ["15m", "1h", "4h", "1d", "1w", "1M"]

STATE_FILE = "state.json"


def higher_tfs(cfg):
    tf = cfg["timeframe"]
    if tf in HTF_LADDER:
        return HTF_LADDER[HTF_LADDER.index(tf)+1:][:3]
    return cfg.get("htf", ["1h", "4h", "1d"])


def load_config():
    cfg = dict(DEFAULTS)
    if os.path.exists("config.json"):
        with open("config.json") as f:
            cfg.update(json.load(f))
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
_KLINE_CACHE = {}

def fetch_klines(host, symbol, interval, limit):
    """Return a DataFrame of CLOSED candles only (drops the still-forming one). Cached per run, retries transient errors."""
    ck = (symbol, interval, limit)
    if ck in _KLINE_CACHE:
        return _KLINE_CACHE[ck]
    url = f"{host}/api/v3/klines"
    rows = None; last = None
    for attempt in range(3):
        try:
            r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=20)
            r.raise_for_status()
            rows = r.json(); break
        except Exception as e:
            last = e; time.sleep(1.5 * (attempt + 1))
    if rows is None:
        raise last
    df = pd.DataFrame(rows, columns=["openTime","open","high","low","close","volume",
                                     "closeTime","qav","trades","tbav","tqav","ignore"])
    for c in ("open","high","low","close","volume"):
        df[c] = df[c].astype(float)
    df["closeTime"] = df["closeTime"].astype("int64")
    now_ms = int(time.time() * 1000)
    df = df[df["closeTime"] <= now_ms].reset_index(drop=True)   # keep only closed bars
    _KLINE_CACHE[ck] = df
    return df


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS  (Wilder smoothing = RMA; matches TradingView's ta.rma/atr/rsi/dmi)
# ─────────────────────────────────────────────────────────────────────────────
def ema(s, n):  return s.ewm(span=n, adjust=False).mean()
def sma(s, n):  return s.rolling(n).mean()
def rma(s, n):  return s.ewm(alpha=1/n, adjust=False).mean()

def true_range(df):
    pc = df["close"].shift(1)
    return pd.concat([df["high"]-df["low"], (df["high"]-pc).abs(), (df["low"]-pc).abs()], axis=1).max(axis=1)

def atr(df, n=14): return rma(true_range(df), n)

def rsi(s, n=14):
    d = s.diff()
    up = rma(d.clip(lower=0), n)
    dn = rma((-d).clip(lower=0), n)
    rs = up / dn.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50)

def macd(s, fast=12, slow=26, sig=9):
    line = ema(s, fast) - ema(s, slow)
    signal = ema(line, sig)
    return line, signal, line - signal

def dmi(df, n=14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm  = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(df)
    atr_ = rma(tr, n)
    plus_di  = 100 * rma(pd.Series(plus_dm, index=df.index), n) / atr_
    minus_di = 100 * rma(pd.Series(minus_dm, index=df.index), n) / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = rma(dx.fillna(0), n)
    return plus_di, minus_di, adx

def supertrend(df, factor=3.0, period=10):
    """Returns direction series: -1 = bullish (price above), +1 = bearish (matches Pine ta.supertrend)."""
    hl2 = (df["high"] + df["low"]) / 2
    a = atr(df, period)
    upper = (hl2 + factor*a).values
    lower = (hl2 - factor*a).values
    close = df["close"].values
    n = len(df)
    fu = upper.copy(); fl = lower.copy(); dir_ = np.ones(n)
    for i in range(1, n):
        fl[i] = lower[i] if (lower[i] > fl[i-1] or close[i-1] < fl[i-1]) else fl[i-1]
        fu[i] = upper[i] if (upper[i] < fu[i-1] or close[i-1] > fu[i-1]) else fu[i-1]
        if close[i] > fu[i-1]:
            dir_[i] = -1
        elif close[i] < fl[i-1]:
            dir_[i] = 1
        else:
            dir_[i] = dir_[i-1]
    return pd.Series(dir_, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURE (SMC): confirmed pivots → BOS / CHoCH state machine (ports the Pine logic)
# ─────────────────────────────────────────────────────────────────────────────
def structure(df, L=6):
    high = df["high"].values; low = df["low"].values; n = len(df)
    # confirmed pivot value indexed by the CONFIRMATION bar (pivot bar + L), like ta.pivothigh
    ph = [np.nan]*n; pl = [np.nan]*n
    for c in range(2*L, n):
        i = c - L
        if high[i] == max(high[i-L:i+L+1]): ph[c] = high[i]
        if low[i]  == min(low[i-L:i+L+1]):  pl[c] = low[i]
    bosBull = np.zeros(n, bool); bosBear = np.zeros(n, bool)
    chochBull = np.zeros(n, bool); chochBear = np.zeros(n, bool)
    trend = 0; lastH = prevH = lastL = prevL = np.nan
    trend_series = np.zeros(n, int)
    for c in range(n):
        if not np.isnan(ph[c]):
            prevH, lastH = lastH, ph[c]
            if not np.isnan(prevH):
                if lastH > prevH: bosBull[c] = True;  trend = 1
                elif trend == 1:  chochBear[c] = True; trend = -1
        if not np.isnan(pl[c]):
            prevL, lastL = lastL, pl[c]
            if not np.isnan(prevL):
                if lastL < prevL: bosBear[c] = True;  trend = -1
                elif trend == -1: chochBull[c] = True; trend = 1
        trend_series[c] = trend
    return bosBull, bosBear, chochBull, chochBear, trend_series


def volume_profile(df, length, bins, va_pct):
    """POC / VAH / VAL over the last `length` closed bars. Returns (poc, vah, val) or (nan,nan,nan)."""
    w = df.iloc[-length:] if len(df) >= length else df
    lo, hi = w["low"].min(), w["high"].max()
    if hi <= lo: return np.nan, np.nan, np.nan
    edges = np.linspace(lo, hi, bins+1)
    idx = np.clip(np.digitize(w["close"].values, edges) - 1, 0, bins-1)
    vol = np.zeros(bins)
    np.add.at(vol, idx, w["volume"].values)
    poc_i = int(vol.argmax())
    poc = (edges[poc_i] + edges[poc_i+1]) / 2
    target = vol.sum() * va_pct/100
    cum = vol[poc_i]; l = r = poc_i
    while cum < target and (l > 0 or r < bins-1):
        lv = vol[l-1] if l > 0 else -1
        rv = vol[r+1] if r < bins-1 else -1
        if rv >= lv and r < bins-1: r += 1; cum += rv
        elif l > 0:                 l -= 1; cum += lv
        else: break
    return poc, edges[r+1], edges[l]


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL ENGINE  (5 confluence buckets + A/B/C tiers — ported from the strategy)
# ─────────────────────────────────────────────────────────────────────────────
def htf_bull_count(cfg, host, symbol):
    """No-repaint HTF trend on the timeframes above the signal TF: last CLOSED HTF bar, close vs EMA21.
    Returns (bull, total) where total = HTFs actually evaluated (varies by signal TF)."""
    bull = total = 0
    for tf in higher_tfs(cfg):
        try:
            d = fetch_klines(host, symbol, tf, 300)
            if len(d) < 25: continue
            total += 1
            if d["close"].iloc[-1] > ema(d["close"], 21).iloc[-1]: bull += 1
        except Exception as e:
            print(f"  [htf {tf}] fetch failed: {e}")
    return bull, total

def compute_signal(cfg, host, symbol, df):
    if len(df) < 210:
        return None
    close, openp, high, low, vol = df["close"], df["open"], df["high"], df["low"], df["volume"]
    a = atr(df, 14)
    atrPct = a / close * 100
    ema50, ema200, ema21 = ema(close,50), ema(close,200), ema(close,21)
    r = rsi(close, 14)
    macdL, macdS, macdH = macd(close)
    diP, diM, adx = dmi(df, 14)
    stDir = supertrend(df, 3.0, 10)
    volSMA = sma(vol, 20)
    volRatio = (vol / volSMA).replace([np.inf, np.nan], 1.0)
    rng = (high - low)
    bodyPct = (close - openp).abs() / rng.replace(0, np.nan)
    candleQual = bodyPct.fillna(0) >= cfg["minCandleBodyRatio"]
    bullCandle = (close > openp) & candleQual
    bearCandle = (close < openp) & candleQual

    # order flow (candle-approx CVD + imbalance)
    upVol   = np.where(close > openp, vol, vol * (close - low)  / rng.replace(0, np.nan))
    downVol = np.where(close < openp, vol, vol * (high - close) / rng.replace(0, np.nan))
    upVol = pd.Series(upVol, index=df.index).fillna(0)
    downVol = pd.Series(downVol, index=df.index).fillna(0)
    cvd = (upVol - downVol).cumsum()
    cvdMA = sma(cvd, 20)
    upSum = upVol + upVol.shift(1).fillna(0) + upVol.shift(2).fillna(0)
    downSum = downVol + downVol.shift(1).fillna(0) + downVol.shift(2).fillna(0)
    ratio = (upSum / downSum.replace(0, np.nan)).fillna(999)
    bullImb = cfg["enableImbalance"] & (ratio > cfg["imbalanceThreshold"]) & (volRatio > 1.5)
    bearImb = cfg["enableImbalance"] & (ratio < 1/cfg["imbalanceThreshold"]) & (volRatio > 1.5)

    bosBull, bosBear, chochBull, chochBear, trend_s = structure(df, 6)
    # pattern events (last-bar scope is all we need, but compute arrays for lookback)
    bullEng = (close.shift(1) < openp.shift(1)) & (close > openp) & (close > high.shift(1)) & (openp < low.shift(1))
    bearEng = (close.shift(1) > openp.shift(1)) & (close < openp) & (close < low.shift(1)) & (openp > high.shift(1))
    recentHigh = high.rolling(15).max(); recentLow = low.rolling(15).min()
    stopHuntBelow = (low < recentLow.shift(1)) & (close > recentLow.shift(1)) & (close > openp)
    stopHuntAbove = (high > recentHigh.shift(1)) & (close < recentHigh.shift(1)) & (close < openp)

    structLookback = 4 if INTERVAL_MS.get(cfg["timeframe"], 0) < 3_600_000 else 6
    bullEvt = pd.Series(bosBull, index=df.index) | pd.Series(chochBull, index=df.index) | bullEng | stopHuntBelow
    bearEvt = pd.Series(bosBear, index=df.index) | pd.Series(chochBear, index=df.index) | bearEng | stopHuntAbove
    recentBull = bool(bullEvt.iloc[-(structLookback+1):].any())
    recentBear = bool(bearEvt.iloc[-(structLookback+1):].any())

    # volume profile → location
    poc = vah = val = np.nan
    if cfg["enableVolumeProfile"]:
        poc, vah, val = volume_profile(df, cfg["vpLength"], cfg["vpBins"], cfg["valueAreaPct"])

    # ── evaluate LAST CLOSED bar ──
    i = -1
    htfBull, htfTotal = htf_bull_count(cfg, host, symbol)
    htfBear = htfTotal - htfBull
    need = min(2, htfTotal) if htfTotal > 0 else 99   # need 2 aligned when >=2 HTFs exist, else all available
    c = float(close.iloc[i]); st = float(stDir.iloc[i]); trend = int(trend_s[i])

    vTrend = 1 if (htfBull >= need and c > ema50.iloc[i] and st == -1) else (-1 if (htfBear >= need and c < ema50.iloc[i] and st == 1) else 0)
    vMom   = 1 if (macdH.iloc[i] > 0 and diP.iloc[i] > diM.iloc[i] and r.iloc[i] < 78) else (-1 if (macdH.iloc[i] < 0 and diM.iloc[i] > diP.iloc[i] and r.iloc[i] > 22) else 0)
    vStruct = 1 if (recentBull and not recentBear) else (-1 if (recentBear and not recentBull) else 0)
    flowBull = (cvd.iloc[i] > cvdMA.iloc[i]) and (bool(bullImb.iloc[i]) or (volRatio.iloc[i] >= cfg["minVolumeRatio"] and bool(bullCandle.iloc[i])))
    flowBear = (cvd.iloc[i] < cvdMA.iloc[i]) and (bool(bearImb.iloc[i]) or (volRatio.iloc[i] >= cfg["minVolumeRatio"] and bool(bearCandle.iloc[i])))
    vFlow  = 1 if flowBull else (-1 if flowBear else 0)
    locBull = (not np.isnan(val)) and c >= val and c <= poc and trend == 1
    locBear = (not np.isnan(vah)) and c <= vah and c >= poc and trend == -1
    vLoc   = 1 if locBull else (-1 if locBear else 0)

    bullBk = sum(v == 1 for v in (vTrend, vMom, vStruct, vFlow, vLoc))
    bearBk = sum(v == -1 for v in (vTrend, vMom, vStruct, vFlow, vLoc))

    scoreBull = htfBull*3 + (4 if c>ema50.iloc[i] else 0) + (3 if c>ema200.iloc[i] else 0) + (4 if macdH.iloc[i]>0 else 0) + (5 if r.iloc[i]<35 else 0) + (6 if bool(bullImb.iloc[i]) else 0) + (6 if recentBull else 0) + (8 if bool(stopHuntBelow.iloc[i]) else 0) + (3 if cvd.iloc[i]>cvdMA.iloc[i] else 0)
    scoreBear = htfBear*3 + (4 if c<ema50.iloc[i] else 0) + (3 if c<ema200.iloc[i] else 0) + (4 if macdH.iloc[i]<0 else 0) + (5 if r.iloc[i]>65 else 0) + (6 if bool(bearImb.iloc[i]) else 0) + (6 if recentBear else 0) + (8 if bool(stopHuntAbove.iloc[i]) else 0) + (3 if cvd.iloc[i]<cvdMA.iloc[i] else 0)
    isSwing = INTERVAL_MS.get(cfg["timeframe"], 0) >= 3_600_000
    minScoreBase = 18 + (2 if isSwing else 0)
    scoreDiffBase = 6
    minScore  = int(minScoreBase * 0.8) if adx.iloc[i] > 30 else minScoreBase
    scoreDiff = int(scoreDiffBase * 0.7) if adx.iloc[i] > 30 else scoreDiffBase

    def tier_of(bk, sc, opp):
        edge = sc >= minScore and sc > opp + scoreDiff
        if bk >= 4 and edge: return 1
        if bk >= 3 and sc >= minScore: return 2
        if bk >= cfg["minBuckets"]: return 3
        return 0

    # filters (correlation / killzone / spread default off → True)
    volatilityOk = cfg["minVolatilityPct"] <= atrPct.iloc[i] <= cfg["maxVolatilityPct"]
    volOk = volRatio.iloc[i] >= cfg["minVolumeRatio"]
    trendStrength = adx.iloc[i] > cfg["minADX"]
    baseFilters = volatilityOk and volOk and trendStrength

    # pullback (optional)
    pbLong = (not cfg["usePullback"]) or (low.iloc[-3:].min() <= ema21.iloc[i] and c > ema21.iloc[i])
    pbShort = (not cfg["usePullback"]) or (high.iloc[-3:].max() >= ema21.iloc[i] and c < ema21.iloc[i])

    tierCap = 1 if cfg["tradeTier"] == "A only" else (2 if cfg["tradeTier"] == "A + B" else 3)
    rawBull = bullBk >= cfg["minBuckets"] and bearBk <= cfg["maxOpposite"] and bullBk > bearBk
    rawBear = bearBk >= cfg["minBuckets"] and bullBk <= cfg["maxOpposite"] and bearBk > bullBk
    tBull = tier_of(bullBk, scoreBull, scoreBear) if rawBull else 0
    tBear = tier_of(bearBk, scoreBear, scoreBull) if rawBear else 0

    side = tier = buckets = None
    if baseFilters and pbLong and 0 < tBull <= tierCap:
        side, tier, buckets = "LONG", tBull, bullBk
    elif baseFilters and pbShort and 0 < tBear <= tierCap:
        side, tier, buckets = "SHORT", tBear, bearBk
    if side is None:
        return None

    # entry / stop / targets (structural stop + maxStopPct cap)
    entry = c; av = float(a.iloc[i])
    if side == "LONG":
        atrSL = entry - av*cfg["atrMultSL"]
        structSL = float(low.iloc[-cfg["swingStopLen"]:].min()) - av*0.1
        rawSL = min(atrSL, structSL) if cfg["useStructuralStop"] else atrSL
        sl = max(rawSL, entry*(1 - cfg["maxStopPct"]/100))
        rDist = entry - sl
        tp1 = entry + rDist*cfg["tp1R"]; tp2 = entry + rDist*cfg["tp2R"]
    else:
        atrSL = entry + av*cfg["atrMultSL"]
        structSL = float(high.iloc[-cfg["swingStopLen"]:].max()) + av*0.1
        rawSL = max(atrSL, structSL) if cfg["useStructuralStop"] else atrSL
        sl = min(rawSL, entry*(1 + cfg["maxStopPct"]/100))
        rDist = sl - entry
        tp1 = entry - rDist*cfg["tp1R"]; tp2 = entry - rDist*cfg["tp2R"]

    return {
        "side": side, "tier": "ABC"[tier-1], "buckets": buckets,
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
        "rr": cfg["tp2R"],
        "votes": {"trend": vTrend, "mom": vMom, "struct": vStruct, "flow": vFlow, "loc": vLoc},
        "adx": float(adx.iloc[i]), "rsi": float(r.iloc[i]),
        "htfBull": htfBull, "htfTotal": htfTotal, "atrPct": float(atrPct.iloc[i]),
        "bar_time": int(df["closeTime"].iloc[-1]),
        "dir": 1 if side == "LONG" else -1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM + STATE
# ─────────────────────────────────────────────────────────────────────────────
def send_telegram(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN"); chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("  [telegram] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — printing instead:")
        print(text); return
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": text, "parse_mode": "HTML",
                            "disable_web_page_preview": True}, timeout=20)
    if not r.ok:
        print(f"  [telegram] send failed {r.status_code}: {r.text}")


def build_message(sig, sym, tf):
    """Rich HTML Telegram card: icons, full labels, %-distances, R:R, bucket breakdown, context."""
    is_long = sig["side"] == "LONG"
    head = "🟢🟢🟢" if is_long else "🔴🔴🔴"
    verb = "LONG · BUY" if is_long else "SHORT · SELL"
    entry = sig["entry"]
    def pct(level):
        return f"{(level/entry - 1)*100:+.2f}%"
    want = 1 if is_long else -1
    def mark(v):  # ✅ agrees with signal, ❌ opposes, ➖ neutral
        return "✅" if v == want else ("❌" if v == -want else "➖")
    v = sig["votes"]
    tier_word = {"A": "A · Strong", "B": "B · Solid", "C": "C · Weak"}.get(sig["tier"], sig["tier"])
    ts = datetime.datetime.fromtimestamp(sig["bar_time"]/1000, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return "\n".join([
        f"{head}",
        f"<b>{verb}</b>   ·   <b>{sym}</b>   ·   {tf}",
        f"🏆 Tier <b>{tier_word}</b>    🎯 Confluence <b>{sig['buckets']}/5</b>",
        "━━━━━━━━━━━━━━━━━━",
        f"💰 <b>Entry</b>              <code>{fmt(entry)}</code>",
        f"🛑 <b>Stop Loss</b>       <code>{fmt(sig['sl'])}</code>   ({pct(sig['sl'])})",
        f"✅ <b>Take Profit 1</b>  <code>{fmt(sig['tp1'])}</code>   ({pct(sig['tp1'])})",
        f"🚀 <b>Take Profit 2</b>  <code>{fmt(sig['tp2'])}</code>   ({pct(sig['tp2'])})",
        "━━━━━━━━━━━━━━━━━━",
        f"⚖️ <b>Risk / Reward</b>   1 : {sig['rr']:.1f}",
        f"📊 <b>Confluence</b>",
        f"    {mark(v['trend'])} Trend    {mark(v['mom'])} Momentum    {mark(v['struct'])} Structure",
        f"    {mark(v['flow'])} Flow     {mark(v['loc'])} Location",
        f"📈 ADX <b>{sig['adx']:.0f}</b>   ·   RSI <b>{sig['rsi']:.0f}</b>   ·   HTF <b>{sig['htfBull']}/{sig['htfTotal']}</b> bull   ·   ATR <b>{sig['atrPct']:.2f}%</b>",
        "━━━━━━━━━━━━━━━━━━",
        f"⚡ <b>Quantum Pro Signals</b>   ·   🕐 {ts}",
    ])

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f: return json.load(f)
        except Exception: pass
    return {}

def save_state(st):
    with open(STATE_FILE, "w") as f: json.dump(st, f, indent=2)

def fmt(x):
    return f"{x:,.6f}".rstrip("0").rstrip(".") if x < 100 else f"{x:,.2f}"


def build_worklist(cfg):
    """Explicit (symbol, timeframe) pairs if `pairs` is set, else symbols × timeframes."""
    if cfg.get("pairs"):
        return [(p["symbol"], p["timeframe"]) for p in cfg["pairs"]]
    tfs = cfg.get("timeframes") or [cfg["timeframe"]]
    return [(s, tf) for tf in tfs for s in cfg["symbols"]]


def main():
    cfg = load_config()
    host = cfg["data_host"]
    state = load_state()
    changed = False
    sent = 0

    for sym, tf in build_worklist(cfg):
        cfg["timeframe"] = tf                          # compute_signal reads this (regime / structLookback)
        interval_ms = INTERVAL_MS.get(tf, 900_000)
        key = f"{sym}|{tf}"
        try:
            df = fetch_klines(host, sym, tf, cfg["limit"])
            sig = compute_signal(cfg, host, sym, df)
        except Exception as e:
            print(f"[{sym} {tf}] error: {e}"); continue

        if not sig:
            print(f"[{sym} {tf}] no signal"); continue

        prev = state.get(key, {})
        # de-dup: one alert per bar; mirror Pine's no-flip-flop + cooldown
        if prev.get("last_bar") == sig["bar_time"]:
            print(f"[{sym} {tf}] already alerted this bar"); continue
        cooldown_ok = (sig["bar_time"] - prev.get("last_bar", 0)) >= cfg["cooldownBars"] * interval_ms
        same_dir = prev.get("last_dir") == sig["dir"]
        if not cfg["allowRepeatDirection"] and same_dir:
            print(f"[{sym} {tf}] {sig['side']} suppressed (same direction as last signal)"); continue
        if prev and not cooldown_ok:
            print(f"[{sym} {tf}] {sig['side']} suppressed (cooldown)"); continue

        send_telegram(build_message(sig, sym, tf))
        print(f"[{sym} {tf}] SENT: {sig['side']} [{sig['tier']}] {sig['buckets']}/5")
        state[key] = {"last_bar": sig["bar_time"], "last_dir": sig["dir"]}
        changed = True; sent += 1

    # daily heartbeat (once per UTC day) so the channel is never silent on quiet days
    if cfg.get("heartbeat", True):
        today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        if state.get("_heartbeat_day") != today:
            n = len(build_worklist(cfg))
            send_telegram(f"✅ <b>Quantum Pro</b> — online · monitoring <b>{n}</b> pairs · {today}\n"
                          f"<i>Scanning 24/7. A signal card is posted the moment a setup triggers.</i>")
            state["_heartbeat_day"] = today
            changed = True
            print(f"[heartbeat] sent for {today}")

    if changed:
        save_state(state)
    print(f"done. {sent} sent." + ("" if changed else " (state unchanged)"))


if __name__ == "__main__":
    sys.exit(main())
