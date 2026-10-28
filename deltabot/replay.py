"""Portfolio replay: runs the bot's rules over past data on ONE shared account
(max positions, shared equity, daily loss limit, kill switch) - like the live bot would have.

Same fill rules as backtest.py: signal at bar close -> fill next bar open with slippage + fees,
exchange-side SL/TP checked intrabar (stop first if both touched).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .client import RESOLUTION_SECONDS, DeltaClient
from .config import Settings
from .data import Product, candles_df
from .risk import position_size
from .scanner import liquid_universe
from .strategy import Signal, entry_frame, regime_frame

log = logging.getLogger(__name__)


@dataclass
class ReplayResult:
    trades: list[dict] = field(default_factory=list)
    curve: list[tuple[int, float]] = field(default_factory=list)     # (unix, equity USD)
    candles: dict[str, pd.DataFrame] = field(default_factory=dict)
    skipped_small: int = 0


def _aligned_regime(edf: pd.DataFrame, rdf: pd.DataFrame, s: Settings) -> pd.DataFrame:
    r = regime_frame(rdf, s)[["direction", "score"]].copy()
    r["avail"] = r.index + pd.Timedelta(seconds=RESOLUTION_SECONDS[s.regime_tf])
    e = pd.DataFrame({"avail": edf.index + pd.Timedelta(seconds=RESOLUTION_SECONDS[s.entry_tf])})
    m = pd.merge_asof(e, r.sort_values("avail"), on="avail", direction="backward")
    return pd.DataFrame({"direction": m["direction"].fillna(0).astype(int).values,
                         "score": m["score"].fillna(0).values}, index=edf.index)


def run_replay(client: DeltaClient, products: dict[str, Product], s: Settings, days: int,
               start_usd: float, progress=lambda msg, pct: None) -> ReplayResult:
    res = ReplayResult()
    bar_s = RESOLUTION_SECONDS[s.entry_tf]
    entry_bars = int(days * 86400 / bar_s)
    regime_bars = int(days * 86400 / RESOLUTION_SECONDS[s.regime_tf]) + s.regime_ema_slow + 50
    universe = [sym for sym, _ in liquid_universe(client, products, s)]

    feats: dict[str, dict] = {}
    for n, sym in enumerate(universe, 1):
        progress(f"Downloading {sym} ({n}/{len(universe)})", 5 + 70 * n / len(universe))
        try:
            edf = candles_df(client, sym, s.entry_tf, entry_bars + 100)
            rdf = candles_df(client, sym, s.regime_tf, regime_bars)
        except Exception as e:
            log.warning("replay: %s failed: %s", sym, e)
            continue
        if len(edf) < 300 or len(rdf) < s.regime_ema_slow + 50:
            continue
        reg = _aligned_regime(edf, rdf, s)
        f = entry_frame(edf, s)
        res.candles[sym] = edf
        feats[sym] = dict(df=edf, f=f, reg=reg)

    if not feats:
        return res
    progress("Simulating trades", 80)
    timeline = sorted(set().union(*[set(v["df"].index) for v in feats.values()]))
    start_ts = timeline[-1] - pd.Timedelta(days=days)
    timeline = [t for t in timeline if t >= start_ts]
    warm = max(s.ema_slow, s.donchian, s.atr_len) + 5
    max_bars = int(s.max_hold_hours * 3600 / bar_s)

    cash, peak, day, day_start, killed = start_usd, start_usd, None, start_usd, False
    pos: dict[str, dict] = {}
    pend_entry: dict[str, Signal] = {}
    pend_exit: dict[str, str] = {}

    def row(sym, ts):
        d = feats[sym]["df"]
        return d.loc[ts] if ts in d.index else None

    def close(sym, px, reason, ts):
        nonlocal cash
        p, prod = pos.pop(sym), products[sym]
        long = p["side"] == "buy"
        px = px * (1 - s.slippage) if long else px * (1 + s.slippage)
        gross = (1 if long else -1) * (px - p["entry"]) * p["size"] * prod.contract_value
        fee = px * p["size"] * prod.contract_value * s.fee_rate
        cash += gross - fee
        res.trades.append(dict(symbol=sym, side=p["side"], size=p["size"], entry=p["entry"], exit=px,
                               stop=p["stop"], target=p["target"], entry_time=int(p["time"].timestamp()),
                               exit_time=int(ts.timestamp()), pnl=gross - p["fee"] - fee, reason=reason,
                               score=p["score"]))

    def equity_at(ts):
        eq = cash
        for sym, p in pos.items():
            r = row(sym, ts)
            if r is not None:
                p["last"] = r["close"]
            eq += (1 if p["side"] == "buy" else -1) * (p["last"] - p["entry"]) * p["size"] * products[sym].contract_value
        return eq

    for ts in timeline:
        # --- fills at this bar's open
        for sym, reason in list(pend_exit.items()):
            r = row(sym, ts)
            if r is not None and sym in pos:
                close(sym, r["open"], reason, ts)
                pend_exit.pop(sym)
        eq_now = equity_at(ts)
        open_notional = sum(products[k].notional(p["size"], p["entry"]) for k, p in pos.items())
        for sym, sig in list(pend_entry.items()):
            pend_entry.pop(sym)
            r = row(sym, ts)
            if r is None or len(pos) >= s.max_positions:
                continue
            size = position_size(products[sym], sig, eq_now, open_notional, s)
            if size < 1:
                res.skipped_small += 1
                continue
            long = sig.side == "buy"
            px = r["open"] * (1 + s.slippage) if long else r["open"] * (1 - s.slippage)
            fee = px * size * products[sym].contract_value * s.fee_rate
            cash -= fee
            pos[sym] = dict(side=sig.side, size=size, entry=px, stop=sig.stop, target=sig.target, time=ts,
                            fee=fee, last=px, bar=0, score=getattr(sig, "_score", 0.0))
            open_notional += products[sym].notional(size, px)

        # --- intrabar exchange-side SL/TP
        for sym in list(pos):
            r = row(sym, ts)
            if r is None:
                continue
            p, long = pos[sym], pos[sym]["side"] == "buy"
            hit_sl = r["low"] <= p["stop"] if long else r["high"] >= p["stop"]
            hit_tp = r["high"] >= p["target"] if long else r["low"] <= p["target"]
            if hit_sl:
                lvl = min(p["stop"], r["open"]) if long else max(p["stop"], r["open"])
                close(sym, lvl, "stop_loss", ts)
                pend_exit.pop(sym, None)
            elif hit_tp:
                close(sym, p["target"], "take_profit", ts)
                pend_exit.pop(sym, None)

        # --- risk limits on mark-to-market equity
        eq = equity_at(ts)
        res.curve.append((int(ts.timestamp()), eq))
        peak = max(peak, eq)
        d = ts.strftime("%Y-%m-%d")
        if d != day:
            day, day_start = d, eq
        if eq <= peak * (1 - s.max_drawdown) and not killed:
            killed = True
            for sym in list(pos):
                pend_exit[sym] = "kill_switch"
        can_open = not killed and eq > day_start * (1 - s.daily_loss_limit)

        # --- close-of-bar decisions
        for sym, p in pos.items():
            i = feats[sym]["df"].index.get_indexer([ts])[0]
            if i < 0:
                continue
            p["bar"] += 1
            f = feats[sym]["f"].iloc[i]
            if (f["long_exit"] if p["side"] == "buy" else f["short_exit"]):
                pend_exit.setdefault(sym, "trend_break")
            elif p["bar"] >= max_bars:
                pend_exit.setdefault(sym, "time_stop")
        if not can_open:
            continue
        slots = s.max_positions - len(pos)
        if slots <= 0:
            continue
        ideas = []
        for sym, v in feats.items():
            if sym in pos or sym in pend_entry:
                continue
            i = v["df"].index.get_indexer([ts])[0]
            if i < warm:
                continue
            reg, f = v["reg"].iloc[i], v["f"].iloc[i]
            dirn, a, c = int(reg["direction"]), float(f["atr"]), float(v["df"]["close"].iloc[i])
            if dirn == 0 or not np.isfinite(a) or a <= 0:
                continue
            if dirn > 0 and f["long_entry"]:
                sig = Signal("buy", c, c - s.sl_atr * a, c + s.tp_atr * a, a)
            elif dirn < 0 and f["short_entry"]:
                sig = Signal("sell", c, c + s.sl_atr * a, c - s.tp_atr * a, a)
            else:
                continue
            sig._score = float(reg["score"])
            ideas.append((float(reg["score"]), sym, sig))
        for _, sym, sig in sorted(ideas, key=lambda x: x[0], reverse=True)[:slots]:
            pend_entry[sym] = sig

    last = timeline[-1]
    for sym in list(pos):
        close(sym, pos[sym]["last"], "end_of_replay", last)
    if res.curve:
        res.curve[-1] = (res.curve[-1][0], cash)
    progress("Done", 100)
    return res
