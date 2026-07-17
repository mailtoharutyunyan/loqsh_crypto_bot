#!/usr/bin/env python3
"""
Parameter sweep — find the config with the most ROBUST out-of-sample edge.

For each symbol × timeframe it builds features once, then tests every gate combo
(tier × minBuckets × pullback) via the same no-lookahead backtest. A (symbol,tf,combo)
is "robust" only if it has enough trades AND stays profitable across most time segments
(not one lucky window). Ranks global combos and prints a recommended config.

Usage: python sweep.py
"""
import json, itertools
import numpy as np
import quantum_signals as q
import backtest as bt

BASKET = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","ADAUSDT","AVAXUSDT","LINKUSDT"]
TFS    = ["15m","1h","4h","1d"]
BARS   = 2500
COMBOS = [dict(tradeTier=t, minBuckets=b, usePullback=p)
          for t in ("A only","A + B") for b in (3,4) for p in (False,True)]
MIN_TRADES = 10          # ignore samples too small to mean anything
ROBUST_PF  = 1.2
ROBUST_SEG = 3           # of 4 segments must be net positive


def eval_combo(feat, cfg):
    longSig, shortSig = bt.gate(feat, cfg)
    trades = bt.simulate(feat["df"], longSig, shortSig, feat["av"], cfg)
    m = bt.metrics(trades)
    if m["n"] == 0:
        return m, 0
    segs = np.array_split(trades, 4)
    green = sum(1 for s in segs if len(s) and bt.metrics(list(s))["totalR"] > 0)
    return m, green


def main():
    host = q.DEFAULTS["data_host"]
    # 1) build features once per (symbol, tf)
    feats = {}
    for sym in BASKET:
        for tf in TFS:
            cfg = dict(q.DEFAULTS); cfg["timeframe"] = tf
            try:
                df = bt.fetch_history(host, sym, tf, BARS)
                if len(df) < bt.WARMUP + 60:
                    continue
                feats[(sym, tf)] = bt.build_features(df, host, sym, cfg)
                print(f"  built {sym} {tf} ({len(df)} bars)")
            except Exception as e:
                print(f"  skip {sym} {tf}: {e}")

    # 2) evaluate every combo over every (symbol, tf)
    rows = []            # (combo_idx, sym, tf, m, green)
    for ci, combo in enumerate(COMBOS):
        for (sym, tf), feat in feats.items():
            cfg = dict(q.DEFAULTS); cfg["timeframe"] = tf; cfg.update(combo)
            m, green = eval_combo(feat, cfg)
            rows.append((ci, sym, tf, m, green))

    # 3) aggregate per global combo
    print("\n" + "="*84)
    print("GLOBAL COMBO RANKING (across all symbols × timeframes)")
    print("="*84)
    agg = []
    for ci, combo in enumerate(COMBOS):
        rs = [r for r in rows if r[0]==ci and r[3]["n"]>=MIN_TRADES]
        if not rs:
            continue
        tot_trades = sum(r[3]["n"] for r in rs)
        tot_R      = sum(r[3]["totalR"] for r in rs)
        pos_pairs  = sum(1 for r in rs if r[3]["totalR"]>0)
        robust     = sum(1 for r in rs if r[4]>=ROBUST_SEG and r[3]["PF"]>=ROBUST_PF and r[3]["n"]>=MIN_TRADES)
        agg.append((robust, tot_R, pos_pairs, len(rs), tot_trades, ci, combo))
    agg.sort(reverse=True)
    for robust, tot_R, pos_pairs, npairs, tot_trades, ci, combo in agg:
        label = f"{combo['tradeTier']:<7} b{combo['minBuckets']} pb={int(combo['usePullback'])}"
        print(f"  {label:<20} robust {robust:>2}/{npairs:<2} pairs · net+ {pos_pairs}/{npairs} · "
              f"totalR {tot_R:>+6.1f} · {tot_trades} trades")

    if not agg:
        print("\nNo combo produced enough trades to judge. Edge unproven — do NOT monetize.")
        return

    # 4) best combo + per-pair robust breakdown
    best = agg[0]; bci, bcombo = best[5], best[6]
    print("\n" + "="*84)
    print(f"BEST COMBO: {bcombo}  → per-pair breakdown (★ = robust)")
    print("="*84)
    robust_pairs = []
    for r in sorted([r for r in rows if r[0]==bci], key=lambda x:-x[3].get("totalR",-999)):
        _,sym,tf,m,green = r
        if m["n"]==0: continue
        star = "★" if (green>=ROBUST_SEG and m["PF"]>=ROBUST_PF and m["n"]>=MIN_TRADES) else " "
        if star=="★": robust_pairs.append((sym,tf))
        print(f"  {star} {sym:<9} {tf:<4} trades {m['n']:>3} · win {m['win%']:>3.0f}% · "
              f"PF {m['PF']:>4.2f} · totalR {m['totalR']:>+6.1f} · segsGreen {green}/4")

    print("\n" + "="*84)
    if robust_pairs:
        syms = sorted({s for s,_ in robust_pairs}); tfs = sorted({t for _,t in robust_pairs}, key=lambda x:TFS.index(x))
        rec = {"symbols":syms, "timeframes":tfs, "tradeTier":bcombo["tradeTier"],
               "minBuckets":bcombo["minBuckets"], "usePullback":bcombo["usePullback"], "minVolatilityPct":0.0}
        print("RECOMMENDED config.json (only robust symbols/timeframes kept):")
        print(json.dumps(rec, indent=2))
        with open("config.recommended.json","w") as f: json.dump(rec,f,indent=2)
        print("\nWrote config.recommended.json")
    else:
        print("⚠ NO robust (symbol,tf) pairs at any combo — edge is not proven out-of-sample.")
        print("  Recommendation: do NOT sell signals. The engine is ~break-even after costs.")
    print("="*84)


if __name__=="__main__":
    main()
