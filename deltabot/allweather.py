"""All-weather portfolio: the evidence-based core of the bot.

Two return sources that tend to fail at different times, combined with risk controls:

  1. CARRY (market-neutral): long spot + short perpetual on BTC/ETH/SOL/XRP (the coins Delta India
     lists spot for) while funding pays; price moves cancel, funding is collected.
  2. TREND (time-series momentum): long/short a basket of coins in the direction of their 20/60/120-day
     moves, each position sized by its volatility, portfolio volatility-targeted.

Every setting below was fixed in advance from the published literature - nothing is tuned on the
backtest. History: Binance USDT-M public data (daily prices + funding) from 2019, costs at Delta's
rates (taker 0.05% + 18% GST + 0.05% slippage; spot 0.1% + GST).

Universe for TREND = the 20 largest coins by market cap in January 2020, INCLUDING later losers
(EOS, DASH, NEO, IOTA, ...) to avoid survivorship bias. A coin enters when its futures market exists
and is dropped when its data stops.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import ROOT

CACHE = ROOT / "data" / "binance"
UNIVERSE_2020 = ["BTC", "ETH", "XRP", "BCH", "LTC", "BNB", "EOS", "XTZ", "LINK", "XLM", "ADA", "TRX", "ETC",
                 "DASH", "ZEC", "ATOM", "NEO", "ONT", "IOTA", "VET"]
CARRY_COINS = ["BTC", "ETH", "SOL", "XRP"]          # Delta India has spot for these

# ---- fixed, literature-based settings (NOT tuned)
LOOKBACKS = (20, 60, 120)          # days, time-series momentum ensemble
VOL_WINDOW = 60                    # days for volatility estimates
TREND_TARGET_VOL = 0.20            # 20% annualised for the trend sleeve
MAX_GROSS = 2.0                    # trend sleeve gross exposure cap
NO_TRADE_BAND = 0.25               # skip rebalances smaller than 25% of the target position
PERP_COST = 0.0005 * 1.18 + 0.0005  # per side, on traded notional
CARRY_ENTRY = 0.00005              # enter when 7-day avg funding > 0.005% / 8h (covers costs within weeks)
CARRY_EXIT = 0.0                   # exit when it turns negative
CARRY_HEDGE_FRACTION = 0.75        # 1 unit capital -> 0.75 spot + 0.25 perp margin (3x)
CARRY_SWITCH_COST = (0.001 + 0.0002) * 1.18   # spot taker + perp maker, per side
DD_BRAKE, DD_RELEASE = 0.20, 0.10  # halve trend risk below -20% drawdown until back above -10%
WEIGHTS = {"carry": 0.5, "trend": 0.5}


# ---------------------------------------------------------------- data
def _get(url, params, tries=4):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 400:
                return None
        except requests.RequestException:
            pass
        time.sleep(1 + 2 * k)
    return None


def binance_daily(coin: str, start="2019-09-01") -> pd.Series | None:
    path = CACHE / f"{coin}_1d.csv"
    if path.exists() and path.stat().st_size > 50 and time.time() - path.stat().st_mtime < 6 * 3600:
        s = pd.read_csv(path, index_col=0, parse_dates=True)["close"]
        return s if len(s) else None
    rows, t = [], int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    while True:
        d = _get("https://fapi.binance.com/fapi/v1/klines", {"symbol": f"{coin}USDT", "interval": "1d", "startTime": t, "limit": 1500})
        if not d:
            break
        rows += d
        if len(d) < 1500:
            break
        t = d[-1][0] + 86400000
    CACHE.mkdir(parents=True, exist_ok=True)
    if not rows:                         # download failed: keep any older cache rather than wiping it
        if path.exists() and path.stat().st_size > 50:
            s = pd.read_csv(path, index_col=0, parse_dates=True)["close"]
            return s if len(s) else None
        return None
    s = pd.Series({pd.Timestamp(r[0], unit="ms", tz="UTC"): float(r[4]) for r in rows}, name="close").sort_index()
    s = s[s.index < pd.Timestamp.now(tz="UTC").normalize()]          # drop today's unfinished candle
    s.to_frame().to_csv(path)
    return s


def binance_funding(coin: str, start="2019-09-01") -> pd.Series | None:
    """Daily SUM of funding rates (fraction of notional paid by longs to shorts that day)."""
    path = CACHE / f"{coin}_funding.csv"
    if path.exists() and path.stat().st_size > 50 and time.time() - path.stat().st_mtime < 6 * 3600:
        s = pd.read_csv(path, index_col=0, parse_dates=True)["funding"]
        return s if len(s) else None
    rows, t = [], int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    while True:
        d = _get("https://fapi.binance.com/fapi/v1/fundingRate", {"symbol": f"{coin}USDT", "startTime": t, "limit": 1000})
        if not d:
            break
        rows += d
        if len(d) < 1000:
            break
        t = int(d[-1]["fundingTime"]) + 1
    CACHE.mkdir(parents=True, exist_ok=True)
    if not rows:                         # download failed: keep any older cache rather than wiping it
        if path.exists() and path.stat().st_size > 50:
            s = pd.read_csv(path, index_col=0, parse_dates=True)["funding"]
            return s if len(s) else None
        return None
    s = pd.Series({pd.Timestamp(int(r["fundingTime"]), unit="ms", tz="UTC"): float(r["fundingRate"]) for r in rows})
    s = s.sort_index().resample("1D").sum().rename("funding")
    s.to_frame().to_csv(path)
    return s


def load_all(log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    coins = sorted(set(UNIVERSE_2020) | set(CARRY_COINS))
    px, fd = {}, {}
    for c in coins:
        p, f = binance_daily(c), binance_funding(c)
        if p is not None and len(p) > 200:
            px[c] = p
            fd[c] = f if f is not None else pd.Series(dtype=float)
        log(f"  {c:<5} prices {len(p) if p is not None else 0:>5} days | funding {len(f) if f is not None else 0:>5} days")
    prices = pd.DataFrame(px).sort_index()
    funding = pd.DataFrame(fd).reindex(prices.index).fillna(0.0)
    return prices, funding


# ---------------------------------------------------------------- sleeves
def trend_weights(prices: pd.DataFrame, universe=UNIVERSE_2020) -> pd.DataFrame:
    """Target weights decided at each daily close (fraction of trend-sleeve capital per coin, signed)."""
    cols = [c for c in universe if c in prices]
    p = prices[cols]
    rets = p.pct_change()
    sig = sum(np.sign(p / p.shift(L) - 1) for L in LOOKBACKS) / len(LOOKBACKS)
    vol = rets.rolling(VOL_WINDOW).std() * np.sqrt(365)
    active = p.notna() & p.shift(max(LOOKBACKS)).notna() & vol.notna() & (vol > 0)
    n_active = active.sum(axis=1).replace(0, np.nan)
    raw = (sig * (TREND_TARGET_VOL / vol)).where(active, 0.0).div(n_active, axis=0).fillna(0.0)
    # portfolio-level vol targeting on the raw book (uses only past returns)
    raw_ret = (raw.shift(1) * rets.fillna(0)).sum(axis=1)
    realised = raw_ret.rolling(VOL_WINDOW).std() * np.sqrt(365)
    scale = (TREND_TARGET_VOL / realised).clip(upper=3.0).shift(1).fillna(1.0)
    target = raw.mul(scale, axis=0)
    gross = target.abs().sum(axis=1)
    target = target.div((gross / MAX_GROSS).clip(lower=1.0), axis=0)
    # no-trade band
    w = target.copy()
    vals, prev = target.values, np.zeros(target.shape[1])
    out = np.zeros_like(vals)
    for i in range(len(vals)):
        t = vals[i]
        move = np.abs(t - prev) > NO_TRADE_BAND * np.maximum(np.abs(t), 1e-9)
        prev = np.where(move | (t == 0), t, prev)
        out[i] = prev
    return pd.DataFrame(out, index=target.index, columns=cols)


def trend_sleeve(prices: pd.DataFrame, funding: pd.DataFrame, universe=UNIVERSE_2020) -> pd.DataFrame:
    w = trend_weights(prices, universe)
    cols = list(w.columns)
    rets = prices[cols].pct_change()
    held = w.shift(1).fillna(0.0)                               # decided at close t, earns t -> t+1
    pnl = (held * rets.fillna(0)).sum(axis=1)
    fund = -(held * funding[cols].reindex(held.index).fillna(0)).sum(axis=1)   # longs pay, shorts receive
    cost = (w.diff().abs().fillna(w.abs()) * PERP_COST).sum(axis=1).shift(1).fillna(0)
    return pd.DataFrame({"ret": pnl + fund - cost, "price": pnl, "funding": fund, "cost": cost,
                         "gross": held.abs().sum(axis=1), "net": held.sum(axis=1), "n": (held != 0).sum(axis=1)})


def carry_states(prices: pd.DataFrame, funding: pd.DataFrame, coins=CARRY_COINS) -> pd.DataFrame:
    """True where the carry trade is on for a coin at each daily close (with entry/exit hysteresis)."""
    cols = [c for c in coins if c in prices]
    f = funding[cols]
    have = prices[cols].notna() & (f.abs().rolling(7).sum() > 0)
    ma7 = f.rolling(7).mean() / 3                                 # daily sums -> average per 8h period
    on = pd.DataFrame(False, index=f.index, columns=cols)
    state = np.zeros(len(cols), dtype=bool)
    for i, (idx, row) in enumerate(ma7.iterrows()):
        h = have.iloc[i].values
        state = np.where(~h, False, np.where(state, row.values > CARRY_EXIT, row.values > CARRY_ENTRY))
        on.iloc[i] = state
    return on


def carry_sleeve(prices: pd.DataFrame, funding: pd.DataFrame, coins=CARRY_COINS) -> pd.DataFrame:
    on = carry_states(prices, funding, coins)
    cols = list(on.columns)
    f = funding[cols]
    n_on = on.sum(axis=1).replace(0, np.nan)
    w = on.astype(float).div(n_on, axis=0).fillna(0.0) * CARRY_HEDGE_FRACTION
    held = w.shift(1).fillna(0.0)
    fund = (held * f.fillna(0)).sum(axis=1)                      # short perp receives funding
    cost = (w.diff().abs().fillna(w.abs()) * CARRY_SWITCH_COST).sum(axis=1).shift(1).fillna(0)
    return pd.DataFrame({"ret": fund - cost, "funding": fund, "cost": cost, "invested": held.sum(axis=1),
                         "coins_on": (held > 0).sum(axis=1)})


def portfolio(carry: pd.DataFrame, trend: pd.DataFrame) -> pd.DataFrame:
    idx = carry.index.union(trend.index)
    c, t = carry["ret"].reindex(idx).fillna(0), trend["ret"].reindex(idx).fillna(0)
    eq, peak, braked, out = 1.0, 1.0, False, []
    for i in range(len(idx)):
        dd = eq / peak - 1
        braked = (dd < -DD_BRAKE) or (braked and dd < -DD_RELEASE)
        r = WEIGHTS["carry"] * c.iloc[i] + WEIGHTS["trend"] * t.iloc[i] * (0.5 if braked else 1.0)
        eq *= 1 + r
        peak = max(peak, eq)
        out.append((r, braked))
    return pd.DataFrame(out, index=idx, columns=["ret", "braked"])


# ---------------------------------------------------------------- stats
def stats(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 365
    monthly = (1 + r).resample("ME").prod() - 1
    dd = eq / eq.cummax() - 1
    return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else 0.0,
            "vol": float(r.std() * np.sqrt(365)),
            "sharpe": float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else 0.0,
            "max_dd": float(dd.min()), "pos_months": float((monthly > 0).mean()),
            "worst_month": float(monthly.min()), "best_month": float(monthly.max()), "total": float(eq.iloc[-1] - 1),
            "start": str(r.index[0].date()), "end": str(r.index[-1].date())}


def by_regime(r: pd.Series, btc: pd.Series) -> dict:
    """Monthly results grouped by what BTC did that month: bull > +10%, bear < -10%, else sideways."""
    m = (1 + r).resample("ME").prod() - 1
    b = btc.resample("ME").last().pct_change().reindex(m.index)
    reg = pd.cut(b, [-10, -0.10, 0.10, 10], labels=["bear", "sideways", "bull"])
    out = {}
    for name in ("bull", "sideways", "bear"):
        g = m[reg == name]
        out[name] = {"months": int(len(g)), "avg": float(g.mean()) if len(g) else 0.0,
                     "positive": float((g > 0).mean()) if len(g) else 0.0}
    return out


def run(log=print) -> dict:
    log("Downloading 2019-2026 daily prices + funding (Binance public data)...")
    prices, funding = load_all(log)
    trend = trend_sleeve(prices, funding)
    carry = carry_sleeve(prices, funding)
    start = max(trend.index[trend["gross"] > 0][0], carry.index[carry["invested"] > 0][0])
    trend, carry = trend[trend.index >= start], carry[carry.index >= start]
    port = portfolio(carry, trend)
    btc = prices["BTC"]
    btc_ret = btc.pct_change().reindex(port.index).fillna(0)
    res = {"period": f"{start.date()} - {port.index[-1].date()}",
           "portfolio": stats(port["ret"]), "carry": stats(carry["ret"]), "trend": stats(trend["ret"]),
           "btc_hold": stats(btc_ret),
           "regimes": {k: by_regime(v, btc) for k, v in
                       {"portfolio": port["ret"], "carry": carry["ret"], "trend": trend["ret"], "btc_hold": btc_ret}.items()},
           "years": {}}
    for y in sorted(set(port.index.year)):
        sel = port.index.year == y
        res["years"][int(y)] = {k: float((1 + v[sel]).prod() - 1) for k, v in
                                {"portfolio": port["ret"], "carry": carry["ret"].reindex(port.index).fillna(0),
                                 "trend": trend["ret"].reindex(port.index).fillna(0), "btc_hold": btc_ret}.items()}
    res["series"] = {"portfolio": port["ret"], "carry": carry, "trend": trend}
    return res
