"""High-win-rate research harness with a pre-registered protocol (no tuning on test data).

Families (the classic high-win-rate designs: small target, wider stop):
  dip_trend  - buy short-term dips (RSI-2 extreme) inside a higher-timeframe trend, sell rips in a downtrend
  range_fade - in a ranging market (low ADX/ER), fade closes outside the Bollinger band back to the mean

Execution (realistic, Delta fees incl. 18% GST): limit entry at the signal close that must be traded THROUGH
within 3 candles (maker 0.02%); take-profit is a resting limit (maker); stop / time exit are market orders
(taker 0.05% + 0.05% slippage).

Protocol:
  coins split into DEV and TEST halves (alternating by liquidity rank)
  time: DEV = first 30 days; T1/T2/T3 = the next three 10-day windows
  1) parameters chosen on DEV coins x DEV time only
  2) locked; then scored on T1, T2, T3 for ALL coins (first 40 trades of each window, chronological)
  pass = win rate >= 70% AND return >= +10% on capital in every window
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import indicators as ta
from .client import RESOLUTION_SECONDS
from .config import Settings
from .data import candles_df
from .strategy import regime_frame

MAKER = 0.0002 * 1.18
TAKER = 0.0005 * 1.18 + 0.0005
WAIT = 3

GRID2 = {   # round 2: rebuilt on DEV data only - strong clean trends, BTC agreement, more symmetric exits
    "dip_trend2": {"rsi_n": [2, 3], "rsi_lvl": [10, 20], "adx_min": [25, 35], "er_min": [0.0, 0.3],
                   "btc": [True, False], "tp_atr": [0.8, 1.2], "sl_atr": [1.0, 1.5], "hold": [24]},
}

GRID = {
    "dip_trend": {"rsi_n": [2, 3], "rsi_lvl": [10, 20], "tp_atr": [0.6, 1.0], "sl_atr": [1.5, 2.5], "hold": [24, 48]},
    "range_fade": {"bb_k": [2.0, 2.5], "rsi_lvl": [25, 30], "tp_mode": ["mid", "half"], "sl_atr": [1.5, 2.5], "hold": [24, 48]},
}


def signals(family: str, e: pd.DataFrame, reg: pd.DataFrame, p: dict) -> pd.DataFrame:
    c = e["close"]
    a = ta.atr(e, 14)
    out = pd.DataFrame(index=e.index)
    if family == "dip_trend2":
        r = ta.rsi(c, p["rsi_n"])
        strong = (reg["adx"] >= p["adx_min"]) & (reg["er"] >= p["er_min"])
        trend = reg["direction"]
        up_ok = (reg["btc_dir"] >= 0) if p["btc"] else True
        dn_ok = (reg["btc_dir"] <= 0) if p["btc"] else True
        out["long"] = strong & (trend > 0) & (r < p["rsi_lvl"]) & up_ok
        out["short"] = strong & (trend < 0) & (r > 100 - p["rsi_lvl"]) & dn_ok
        out["tp_dist"] = p["tp_atr"] * a
    elif family == "dip_trend":
        r = ta.rsi(c, p["rsi_n"])
        trend = reg["direction"]
        out["long"] = (trend > 0) & (r < p["rsi_lvl"])
        out["short"] = (trend < 0) & (r > 100 - p["rsi_lvl"])
        out["tp_dist"] = p["tp_atr"] * a
    else:
        mid, sd = c.rolling(20).mean(), c.rolling(20).std()
        r = ta.rsi(c, 14)
        ranging = (reg["adx"] < 20)
        out["long"] = ranging & (c < mid - p["bb_k"] * sd) & (r < p["rsi_lvl"])
        out["short"] = ranging & (c > mid + p["bb_k"] * sd) & (r > 100 - p["rsi_lvl"])
        full = (c - mid).abs()
        out["tp_dist"] = full if p["tp_mode"] == "mid" else full / 2
    out["sl_dist"] = p["sl_atr"] * a
    out["hold"] = p["hold"]
    return out


def simulate(e: pd.DataFrame, sig: pd.DataFrame) -> list[dict]:
    """One position at a time per coin. Returns trades with R = net P&L / stop distance."""
    o, h, lo, c = (e[k].values for k in ("open", "high", "low", "close"))
    L, S = sig["long"].fillna(False).values, sig["short"].fillna(False).values
    TP, SL, H = sig["tp_dist"].values, sig["sl_dist"].values, int(sig["hold"].iloc[0])
    trades, i, n = [], 60, len(e)
    while i < n - 1:
        if not (L[i] or S[i]) or not (SL[i] > 0 and TP[i] > 0):
            i += 1
            continue
        side = 1 if L[i] else -1
        lim = c[i]
        fill = None
        for k in range(i + 1, min(i + 1 + WAIT, n)):
            if (lo[k] < lim) if side > 0 else (h[k] > lim):
                fill = k
                break
        if fill is None:
            i += 1
            continue
        entry = min(lim, o[fill]) if side > 0 else max(lim, o[fill])
        stop, tgt = entry - side * SL[i], entry + side * TP[i]
        px, j, maker_exit, reason = None, fill, False, "time"
        for j in range(fill, min(fill + H, n)):
            hit_sl = lo[j] <= stop if side > 0 else h[j] >= stop
            hit_tp = h[j] > tgt if side > 0 else lo[j] < tgt
            if hit_sl:
                px, reason = (min(stop, o[j]) if side > 0 else max(stop, o[j])), "stop"
                break
            if hit_tp:
                px, maker_exit, reason = tgt, True, "target"
                break
        if px is None:
            px = c[j]
        gross = side * (px - entry)
        cost = MAKER * entry + (MAKER if maker_exit else TAKER) * px
        trades.append({"time": e.index[i], "r": (gross - cost) / SL[i], "reason": reason,
                       "ret_pct": (gross - cost) / entry * 100, "i": i, "side": side,
                       "sl_pct": SL[i] / c[i] * 100, "tp_pct": TP[i] / c[i] * 100})
        i = j + 1
    return trades


def equity_return(rs: list[float], risk: float) -> float:
    eq = 1.0
    for r in rs:
        eq *= 1 + risk * r
    return eq - 1


def load(client, products, syms, s: Settings, tf: str, rtf: str, days: int):
    data = {}
    btc_r = candles_df(client, "BTCUSD", rtf, int(days * 86400 / RESOLUTION_SECONDS[rtf]) + 300)
    btc_f = regime_frame(btc_r, s)[["direction"]].rename(columns={"direction": "btc_dir"})
    btc_f["avail"] = btc_f.index + pd.Timedelta(seconds=RESOLUTION_SECONDS[rtf])
    for sym in syms:
        e = candles_df(client, sym, tf, int(days * 86400 / RESOLUTION_SECONDS[tf]) + 100)
        r = candles_df(client, sym, rtf, int(days * 86400 / RESOLUTION_SECONDS[rtf]) + 300)
        if len(e) < 1000 or len(r) < 300:
            continue
        rf = regime_frame(r, s)[["direction", "adx", "er"]].copy()
        rf["avail"] = rf.index + pd.Timedelta(seconds=RESOLUTION_SECONDS[rtf])
        left = pd.DataFrame({"avail": e.index + pd.Timedelta(seconds=RESOLUTION_SECONDS[tf])})
        m = pd.merge_asof(left, rf.sort_values("avail"), on="avail", direction="backward")
        m = pd.merge_asof(m, btc_f.sort_values("avail"), on="avail", direction="backward")
        reg = pd.DataFrame({"direction": m["direction"].fillna(0).values, "adx": m["adx"].values,
                            "er": m["er"].values, "btc_dir": m["btc_dir"].fillna(0).values}, index=e.index)
        data[sym] = (e, reg)
    return data


def run_protocol(client, products, s: Settings, tf="5m", rtf="1h", days=60, coins=16, log=print,
                 grid=None, fresh_test_coins=False) -> dict:
    from dataclasses import replace
    from .scanner import liquid_universe
    grid = grid or GRID
    ranked = [x for x, _ in liquid_universe(client, products, replace(s, universe_size=60, min_turnover_usd=500_000))]
    base = ranked[:coins]
    test_extra = ranked[coins:2 * coins] if fresh_test_coins else []
    data = load(client, products, base + test_extra, s, tf, rtf, days)
    dev_syms = [x for x in base[0::2] if x in data]
    test_syms = [x for x in test_extra if x in data] if fresh_test_coins else [x for x in base if x in data]
    syms = list(data)
    t_end = min(e.index[-1] for e, _ in data.values())
    t0 = t_end - pd.Timedelta(days=days)
    dev_end = t0 + pd.Timedelta(days=30)
    windows = [(dev_end + pd.Timedelta(days=10 * k), dev_end + pd.Timedelta(days=10 * (k + 1))) for k in range(3)]
    log(f"{len(syms)} coins ({len(dev_syms)} dev), {tf} entries, dev {t0:%d %b}-{dev_end:%d %b}, "
        f"tests {', '.join(f'{a:%d %b}-{b:%d %b}' for a, b in windows)}")

    def trades_for(family, p, sym_list, start, end):
        out = []
        for sym in sym_list:
            e, reg = data[sym]
            out += [t for t in simulate(e, signals(family, e, reg, p)) if start <= t["time"] < end]
        return sorted(out, key=lambda t: t["time"])

    best = None
    tested = 0
    for family, fgrid in grid.items():
        keys = list(fgrid)
        for vals in itertools.product(*(fgrid[k] for k in keys)):
            tested += 1
            p = dict(zip(keys, vals))
            tr = trades_for(family, p, dev_syms, t0, dev_end)
            if len(tr) < 40:
                continue
            rs = np.array([t["r"] for t in tr])
            wr, exp = (rs > 0).mean(), rs.mean()
            cand = {"family": family, "params": p, "dev_n": len(rs), "dev_win": wr, "dev_exp_r": exp}
            # objective: highest expectancy among configs that reach a 70% win rate on DEV
            if wr >= 0.70 and (best is None or exp > best["dev_exp_r"]):
                best = cand
    if best is None:
        return {"passed": False, "reason": "no configuration reached a 70% win rate on development data", "windows": []}
    # risk sized on DEV so that 10 trades of average expectancy make ~11%
    exp = best["dev_exp_r"]
    best["risk"] = float(np.clip(0.011 / exp, 0.005, 0.08)) if exp > 0 else 0.01
    log(f"{tested} configurations searched on DEV; test coins: {'fresh (never used)' if fresh_test_coins else 'all'} ({len(test_syms)})")
    log(f"locked: {best['family']} {best['params']} | dev {best['dev_n']} trades, win {best['dev_win']:.0%}, "
        f"{best['dev_exp_r']:+.3f}R/trade, risk {best['risk']:.1%}")
    results = []
    for k, (a, b) in enumerate(windows, 1):
        tr = trades_for(best["family"], best["params"], test_syms, a, b)[:40]
        rs = [t["r"] for t in tr]
        res = {"window": f"T{k} {a:%d %b}-{b:%d %b}", "trades": len(rs),
               "wins": int(sum(r > 0 for r in rs)), "win_rate": float(np.mean([r > 0 for r in rs])) if rs else 0.0,
               "exp_r": float(np.mean(rs)) if rs else 0.0, "return": equity_return(rs, best["risk"])}
        res["pass"] = res["trades"] >= 30 and res["win_rate"] >= 0.70 and res["return"] >= 0.10
        results.append(res)
        log(f"  {res['window']}: {res['wins']}/{res['trades']} wins ({res['win_rate']:.0%}), "
            f"{res['exp_r']:+.3f}R/trade, return {res['return']:+.1%} -> {'PASS' if res['pass'] else 'FAIL'}")
    return {"passed": all(r["pass"] for r in results), "locked": best, "windows": results}
