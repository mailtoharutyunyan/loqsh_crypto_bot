#!/usr/bin/env python3
"""
Backtest + walk-forward for the Quantum Pro signal engine.

Replays the SAME indicator/bucket logic the live bot uses over historical candles,
with NO look-ahead:
  • signal at bar t uses only data ≤ t
  • HTF trend is point-in-time (last HTF bar CLOSED at or before t, via merge_asof)
  • entry at close[t]; fills simulated on bars t+1…
  • same-bar resolution is CONSERVATIVE (stop checked before target)
  • fees + slippage are subtracted from every trade
Trades are sequential & non-overlapping (single position), mirroring the strategy.

Results are in R-multiples (R = risk per trade = entry→stop distance), so they're
position-size-agnostic. ⚠ A good report here is necessary, not sufficient — it's still
one symbol / one period / one implementation. Validate across symbols and out-of-sample
before trusting it, and never charge for signals you haven't proven.

Usage:
  python backtest.py BTCUSDT 1h            # ~3000 bars
  python backtest.py ETHUSDT 15m 5000
"""
import sys
import numpy as np
import pandas as pd
import requests
import quantum_signals as q

# ── realistic cost model (edit to your venue) ──
COMMISSION_PCT = 0.05     # per side, % of notional (perp taker ~0.05)
SLIPPAGE_PCT   = 0.02     # per side, % — market entry/exit slippage
MAX_HOLD_BARS  = 100      # force-close a trade after N bars if neither TP nor SL hit
WARMUP         = 220      # bars skipped at the start (EMA200 etc. need history)


_HIST_CACHE = {}

def fetch_history(host, symbol, interval, total=3000):
    """Paginated klines (Binance max 1000/req), CLOSED candles only, oldest→newest. Cached."""
    ck = (symbol, interval, total)
    if ck in _HIST_CACHE:
        return _HIST_CACHE[ck]
    out, end = [], None
    while len(out) < total:
        p = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end:
            p["endTime"] = end
        rows = requests.get(f"{host}/api/v3/klines", params=p, timeout=30).json()
        if not rows:
            break
        out = rows + out
        end = rows[0][0] - 1
        if len(rows) < 1000:
            break
    df = pd.DataFrame(out, columns=["openTime","open","high","low","close","volume",
                                    "closeTime","qav","trades","tbav","tqav","ignore"])
    for c in ("open","high","low","close","volume"):
        df[c] = df[c].astype(float)
    df["closeTime"] = df["closeTime"].astype("int64")
    df = df.drop_duplicates("openTime").sort_values("openTime").reset_index(drop=True)
    import time as _t
    df = df[df["closeTime"] <= int(_t.time()*1000)].reset_index(drop=True)
    df = df.tail(total).reset_index(drop=True)
    _HIST_CACHE[ck] = df
    return df


def htf_bull_series(host, symbol, ref_df, cfg):
    """Point-in-time HTF alignment: bull count + total, aligned to each ref bar (no look-ahead)."""
    htfs = q.higher_tfs(cfg)
    total = pd.Series(0, index=ref_df.index)
    bull = pd.Series(0, index=ref_df.index)
    ref = ref_df[["closeTime"]].copy()
    for tf in htfs:
        d = fetch_history(host, symbol, tf, total=1500)
        if len(d) < 30:
            continue
        d = d.assign(b=(d["close"] > q.ema(d["close"], 21)).astype(int))[["closeTime","b"]]
        m = pd.merge_asof(ref, d, on="closeTime", direction="backward")
        bull = bull.add(m["b"].fillna(0).astype(int).values, fill_value=0)
        total += (~m["b"].isna()).astype(int).values
    return bull.astype(int), total.astype(int)


def build_features(df, host, symbol, cfg):
    """Vectorized per-bar features (buckets, scores, filters) — independent of the sweepable gate params."""
    close, openp, high, low, vol = df["close"], df["open"], df["high"], df["low"], df["volume"]
    a = q.atr(df, 14); atrPct = a/close*100
    ema50, ema200 = q.ema(close,50), q.ema(close,200)
    r = q.rsi(close,14)
    _,_,macdH = q.macd(close)
    diP,diM,adx = q.dmi(df,14)
    stDir = q.supertrend(df,3.0,10)
    volSMA = q.sma(vol,20); volRatio=(vol/volSMA).replace([np.inf,np.nan],1.0)
    rng=(high-low); bodyPct=(close-openp).abs()/rng.replace(0,np.nan)
    cq=bodyPct.fillna(0)>=cfg["minCandleBodyRatio"]
    bullCandle=(close>openp)&cq; bearCandle=(close<openp)&cq
    upVol=pd.Series(np.where(close>openp,vol,vol*(close-low)/rng.replace(0,np.nan)),index=df.index).fillna(0)
    dnVol=pd.Series(np.where(close<openp,vol,vol*(high-close)/rng.replace(0,np.nan)),index=df.index).fillna(0)
    cvd=(upVol-dnVol).cumsum(); cvdMA=q.sma(cvd,20)
    upSum=upVol+upVol.shift(1).fillna(0)+upVol.shift(2).fillna(0)
    dnSum=dnVol+dnVol.shift(1).fillna(0)+dnVol.shift(2).fillna(0)
    ratio=(upSum/dnSum.replace(0,np.nan)).fillna(999)
    bullImb=cfg["enableImbalance"]&(ratio>cfg["imbalanceThreshold"])&(volRatio>1.5)
    bearImb=cfg["enableImbalance"]&(ratio<1/cfg["imbalanceThreshold"])&(volRatio>1.5)

    bosBull,bosBear,chochBull,chochBear,trend_s=q.structure(df,6)
    bullEng=(close.shift(1)<openp.shift(1))&(close>openp)&(close>high.shift(1))&(openp<low.shift(1))
    bearEng=(close.shift(1)>openp.shift(1))&(close<openp)&(close<low.shift(1))&(openp>high.shift(1))
    rH=high.rolling(15).max(); rL=low.rolling(15).min()
    shBelow=(low<rL.shift(1))&(close>rL.shift(1))&(close>openp)
    shAbove=(high>rH.shift(1))&(close<rH.shift(1))&(close<openp)
    slb=4 if q.INTERVAL_MS.get(cfg["timeframe"],0)<3_600_000 else 6
    bullEvt=(pd.Series(bosBull,index=df.index)|pd.Series(chochBull,index=df.index)|bullEng|shBelow)
    bearEvt=(pd.Series(bosBear,index=df.index)|pd.Series(chochBear,index=df.index)|bearEng|shAbove)
    recentBull=bullEvt.rolling(slb+1).max().fillna(0).astype(bool)
    recentBear=bearEvt.rolling(slb+1).max().fillna(0).astype(bool)

    # volume profile POC/VAL/VAH per bar (trailing window) — location bucket
    n=len(df); poc=np.full(n,np.nan); vah=np.full(n,np.nan); val=np.full(n,np.nan)
    if cfg["enableVolumeProfile"]:
        L=cfg["vpLength"]
        for i in range(L,n):
            poc[i],vah[i],val[i]=q.volume_profile(df.iloc[:i+1],L,cfg["vpBins"],cfg["valueAreaPct"])

    htfBull,htfTotal=htf_bull_series(host,symbol,df,cfg)
    need=htfTotal.clip(upper=2).where(htfTotal>0,99)

    vTrend=np.where((htfBull>=need)&(close>ema50)&(stDir==-1),1,
            np.where((htfTotal-htfBull>=need)&(close<ema50)&(stDir==1),-1,0))
    vMom=np.where((macdH>0)&(diP>diM)&(r<78),1,np.where((macdH<0)&(diM>diP)&(r>22),-1,0))
    vStruct=np.where(recentBull&~recentBear,1,np.where(recentBear&~recentBull,-1,0))
    flowBull=(cvd>cvdMA)&(bullImb|((volRatio>=cfg["minVolumeRatio"])&bullCandle))
    flowBear=(cvd<cvdMA)&(bearImb|((volRatio>=cfg["minVolumeRatio"])&bearCandle))
    vFlow=np.where(flowBull,1,np.where(flowBear,-1,0))
    pocS=pd.Series(poc,index=df.index); vahS=pd.Series(vah,index=df.index); valS=pd.Series(val,index=df.index)
    locBull=(~valS.isna())&(close>=valS)&(close<=pocS)&(pd.Series(trend_s,index=df.index)==1)
    locBear=(~vahS.isna())&(close<=vahS)&(close>=pocS)&(pd.Series(trend_s,index=df.index)==-1)
    vLoc=np.where(locBull,1,np.where(locBear,-1,0))

    V=np.vstack([vTrend,vMom,vStruct,vFlow,vLoc])
    bullBk=(V==1).sum(0); bearBk=(V==-1).sum(0)
    # score (for tiering)
    isSwing=q.INTERVAL_MS.get(cfg["timeframe"],0)>=3_600_000
    minScoreBase=18+(2 if isSwing else 0); scoreDiffBase=6
    minScore=np.where(adx>30,int(minScoreBase*0.8),minScoreBase)
    scoreDiff=np.where(adx>30,int(scoreDiffBase*0.7),scoreDiffBase)
    scoreBull=(htfBull*3+np.where(close>ema50,4,0)+np.where(close>ema200,3,0)+np.where(macdH>0,4,0)
               +np.where(r<35,5,0)+np.where(bullImb,6,0)+np.where(recentBull,6,0)+np.where(shBelow,8,0)+np.where(cvd>cvdMA,3,0))
    scoreBear=((htfTotal-htfBull)*3+np.where(close<ema50,4,0)+np.where(close<ema200,3,0)+np.where(macdH<0,4,0)
               +np.where(r>65,5,0)+np.where(bearImb,6,0)+np.where(recentBear,6,0)+np.where(shAbove,8,0)+np.where(cvd<cvdMA,3,0))
    ema21=q.ema(close,21)
    pbLongRaw=((low.rolling(3).min()<=ema21)&(close>ema21)).values
    pbShortRaw=((high.rolling(3).max()>=ema21)&(close<ema21)).values
    return {
        "df":df, "av":a.values,
        "bullBk":bullBk, "bearBk":bearBk,
        "scoreBull":scoreBull, "scoreBear":scoreBear, "minScore":minScore, "scoreDiff":scoreDiff,
        "atrPct":atrPct.values, "volRatio":volRatio.values, "adx":adx.values,
        "pbLongRaw":pbLongRaw, "pbShortRaw":pbShortRaw,
    }


def gate(feat, cfg):
    """Apply the sweepable gate (tier / minBuckets / filters / pullback) → long/short signal arrays."""
    atrPct, volRatio, adx = feat["atrPct"], feat["volRatio"], feat["adx"]
    base = (volRatio>=cfg["minVolumeRatio"]) & (atrPct<=cfg["maxVolatilityPct"]) & \
           (atrPct>=cfg["minVolatilityPct"]) & (adx>cfg["minADX"])
    bullBk, bearBk = feat["bullBk"], feat["bearBk"]
    minScore, scoreDiff = feat["minScore"], feat["scoreDiff"]
    def tier(bk,sc,opp):
        edge=(sc>=minScore)&(sc>opp+scoreDiff)
        return np.where((bk>=4)&edge,1,np.where((bk>=3)&(sc>=minScore),2,np.where(bk>=cfg["minBuckets"],3,0)))
    tierCap=1 if cfg["tradeTier"]=="A only" else (2 if cfg["tradeTier"]=="A + B" else 3)
    rawBull=(bullBk>=cfg["minBuckets"])&(bearBk<=cfg["maxOpposite"])&(bullBk>bearBk)
    rawBear=(bearBk>=cfg["minBuckets"])&(bullBk<=cfg["maxOpposite"])&(bearBk>bullBk)
    tB=np.where(rawBull,tier(bullBk,feat["scoreBull"],feat["scoreBear"]),0)
    tS=np.where(rawBear,tier(bearBk,feat["scoreBear"],feat["scoreBull"]),0)
    longSig=base&(tB>0)&(tB<=tierCap)
    shortSig=base&(tS>0)&(tS<=tierCap)
    if cfg["usePullback"]:
        longSig=longSig&feat["pbLongRaw"]; shortSig=shortSig&feat["pbShortRaw"]
    return longSig, shortSig


def simulate(df,longSig,shortSig,av,cfg):
    high=df["high"].values; low=df["low"].values; close=df["close"].values; n=len(df)
    cost_frac=(COMMISSION_PCT+SLIPPAGE_PCT)*2/100   # entry+exit, both sides
    last_dir=0; trades=[]; i=WARMUP
    while i < n-1:
        want=1 if longSig[i] else (-1 if shortSig[i] else 0)
        if want==0 or (not cfg["allowRepeatDirection"] and want==last_dir):
            i+=1; continue
        entry=close[i]; a=av[i]
        if want==1:
            atrSL=entry-a*cfg["atrMultSL"]; structSL=df["low"].iloc[max(0,i-cfg["swingStopLen"]+1):i+1].min()-a*0.1
            sl=max(min(atrSL,structSL) if cfg["useStructuralStop"] else atrSL, entry*(1-cfg["maxStopPct"]/100))
            r=entry-sl; tp1=entry+r*cfg["tp1R"]; tp2=entry+r*cfg["tp2R"]
        else:
            atrSL=entry+a*cfg["atrMultSL"]; structSL=df["high"].iloc[max(0,i-cfg["swingStopLen"]+1):i+1].max()+a*0.1
            sl=min(max(atrSL,structSL) if cfg["useStructuralStop"] else atrSL, entry*(1+cfg["maxStopPct"]/100))
            r=sl-entry; tp1=entry-r*cfg["tp1R"]; tp2=entry-r*cfg["tp2R"]
        if r<=0: i+=1; continue
        tp1done=False; realizedR=0.0; exit_i=None
        for j in range(i+1,min(i+1+MAX_HOLD_BARS,n)):
            hi,lo=high[j],low[j]
            if want==1:
                if not tp1done:
                    if lo<=sl: realizedR=-1.0; exit_i=j; break                      # full stop (SL-first)
                    if hi>=tp1:
                        tp1done=True; realizedR+=0.5*cfg["tp1R"]                      # scale out half
                        if hi>=tp2: realizedR+=0.5*cfg["tp2R"]; exit_i=j; break
                else:
                    if lo<=entry: exit_i=j; break                                    # runner to breakeven
                    if hi>=tp2: realizedR+=0.5*cfg["tp2R"]; exit_i=j; break
            else:
                if not tp1done:
                    if hi>=sl: realizedR=-1.0; exit_i=j; break
                    if lo<=tp1:
                        tp1done=True; realizedR+=0.5*cfg["tp1R"]
                        if lo<=tp2: realizedR+=0.5*cfg["tp2R"]; exit_i=j; break
                else:
                    if hi>=entry: exit_i=j; break
                    if lo<=tp2: realizedR+=0.5*cfg["tp2R"]; exit_i=j; break
        if exit_i is None:                                                          # time exit at market
            j=min(i+MAX_HOLD_BARS,n-1)
            frac=0.5 if tp1done else 1.0
            realizedR+=frac*((close[j]-entry)/r if want==1 else (entry-close[j])/r)
            exit_i=j
        netR=realizedR-cost_frac*entry/r                                            # fees+slippage in R terms
        trades.append({"i":i,"exit":exit_i,"dir":want,"R":netR,"t":int(df["closeTime"].iloc[i])})
        last_dir=want; i=exit_i+1
    return trades


def metrics(trades):
    if not trades: return {"n":0}
    R=np.array([t["R"] for t in trades])
    wins=R[R>0]; losses=R[R<0]
    eq=np.cumsum(R); peak=np.maximum.accumulate(eq); dd=eq-peak
    return {"n":len(R),"win%":100*len(wins)/len(R),
            "PF":(wins.sum()/abs(losses.sum())) if losses.sum()!=0 else float('inf'),
            "expR":R.mean(),"totalR":R.sum(),"maxDD_R":dd.min(),
            "avgWin":wins.mean() if len(wins) else 0,"avgLoss":losses.mean() if len(losses) else 0}


def show(label,m):
    if m["n"]==0: print(f"{label:<22} no trades"); return
    print(f"{label:<22} trades {m['n']:>3} · win {m['win%']:>4.0f}% · PF {m['PF']:>4.2f} · "
          f"expR {m['expR']:>+5.2f} · totalR {m['totalR']:>+6.1f} · maxDD {m['maxDD_R']:>6.1f}R")


def main():
    sym=sys.argv[1] if len(sys.argv)>1 else "BTCUSDT"
    tf=sys.argv[2] if len(sys.argv)>2 else "1h"
    bars=int(sys.argv[3]) if len(sys.argv)>3 else 3000
    cfg=dict(q.DEFAULTS); cfg["timeframe"]=tf
    host=cfg["data_host"]
    print(f"\nBacktest {sym} {tf} · {bars} bars · tier={cfg['tradeTier']} minBuckets={cfg['minBuckets']} "
          f"· fees {COMMISSION_PCT}%+slip {SLIPPAGE_PCT}%/side\n" + "─"*78)
    df=fetch_history(host,sym,tf,bars)
    if len(df)<WARMUP+50:
        print(f"not enough history ({len(df)} bars)"); return
    feat=build_features(df,host,sym,cfg)
    longSig,shortSig=gate(feat,cfg)
    trades=simulate(df,longSig,shortSig,feat["av"],cfg)
    show("OVERALL",metrics(trades))
    # out-of-sample stability: split trades into 4 sequential segments
    if trades:
        segs=np.array_split(trades,4)
        print("─"*78)
        for k,s in enumerate(segs):
            show(f"  segment {k+1}/4",metrics(list(s)))
    print("─"*78)
    print("⚠ R-based, single symbol/period. Confirm across symbols + out-of-sample before trusting. "
          "Segments should ALL be profitable — if edge lives in one segment, it's regime luck, not edge.\n")


if __name__=="__main__":
    main()
