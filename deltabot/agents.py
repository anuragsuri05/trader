"""A team of trading agents, each with its own style and timeframe.

Every agent is vectorized: `frame()` returns per-bar signal columns, so the SAME code drives the
live team (last closed bar) and the per-agent backtest (all bars).

Columns returned by frame():
  long / short          entry signal on this bar's close
  sl_dist / tp_dist     stop-loss / take-profit distance from the close (price units)
  score                 how strong the setup is (used to rank ideas)
  exit_long/exit_short  close an open position on this bar's close
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ta
from .client import RESOLUTION_SECONDS
from .config import Settings
from .strategy import entry_frame, regime_frame


def regime_on(e: pd.DataFrame, r: pd.DataFrame, s: Settings, regime_tf: str, entry_tf: str) -> pd.DataFrame:
    """Regime columns (direction, score, adx) known at each entry bar's close - closed bars only."""
    rf = regime_frame(r, s)[["direction", "score", "adx"]].copy()
    rf["avail"] = rf.index + pd.Timedelta(seconds=RESOLUTION_SECONDS[regime_tf])
    left = pd.DataFrame({"avail": e.index + pd.Timedelta(seconds=RESOLUTION_SECONDS[entry_tf])})
    m = pd.merge_asof(left, rf.sort_values("avail"), on="avail", direction="backward")
    return pd.DataFrame({"direction": m["direction"].fillna(0).astype(int).values,
                         "score": m["score"].fillna(0).values,
                         "adx": m["adx"].values}, index=e.index)


@dataclass
class Agent:
    name: str
    label: str
    style: str
    regime_tf: str
    entry_tf: str
    max_positions: int = 2
    max_hold_bars: int = 96
    symbols: list | None = None          # restrict to these coins (planner); None = whole universe

    def frame(self, e: pd.DataFrame, reg: pd.DataFrame, s: Settings) -> pd.DataFrame:
        raise NotImplementedError


class TrendBreakout(Agent):
    """Close breaks the 20-bar high (low) while the higher timeframe trends up (down)."""

    def explain(self, e, reg, s) -> dict | None:
        """What this bot is waiting for on the latest closed bar (None = coin not trending)."""
        d = int(reg["direction"].iloc[-1])
        if d == 0:
            return None
        f = entry_frame(e, s).iloc[-1]
        c, a = float(e["close"].iloc[-1]), float(f["atr"])
        if not a > 0:
            return None
        up = d > 0
        level = float(f["dc_high"] if up else f["dc_low"])
        dist = (level - c) / a if up else (c - level) / a
        blockers = []
        if up and not f["ema_fast"] > f["ema_slow"]:
            blockers.append("fast EMA below slow")
        if not up and not f["ema_fast"] < f["ema_slow"]:
            blockers.append("fast EMA above slow")
        if up and f["rsi"] >= s.rsi_max_long:
            blockers.append(f"RSI {f['rsi']:.0f} too high")
        if not up and f["rsi"] <= s.rsi_min_short:
            blockers.append(f"RSI {f['rsi']:.0f} too low")
        sig = max(4, -int(np.floor(np.log10(abs(level)))) + 4) if level > 0 else 6
        text = (f"{'UP' if up else 'DOWN'} trend · needs close {'above' if up else 'below'} "
                f"{round(level, sig)} ({max(dist, 0):.1f} ATR away)")
        if blockers:
            text += " · waiting: " + ", ".join(blockers)
        return {"dist": max(dist, 0) + (1 if blockers else 0), "text": text}

    def frame(self, e, reg, s):
        f = entry_frame(e, s)
        out = pd.DataFrame(index=e.index)
        out["long"] = (reg["direction"] > 0) & f["long_entry"]
        out["short"] = (reg["direction"] < 0) & f["short_entry"]
        out["sl_dist"] = s.sl_atr * f["atr"]
        out["tp_dist"] = s.tp_atr * f["atr"]
        out["score"] = reg["score"]
        out["exit_long"], out["exit_short"] = f["long_exit"], f["short_exit"]
        return out


class Pullback(Agent):
    """In an uptrend, buy when price dips to the EMA20 and the bar closes back above it (mirror for shorts)."""

    def frame(self, e, reg, s):
        f = entry_frame(e, s)
        c, o, lo, hi = e["close"], e["open"], e["low"], e["high"]
        ef, es, r = f["ema_fast"], f["ema_slow"], f["rsi"]
        out = pd.DataFrame(index=e.index)
        out["long"] = ((reg["direction"] > 0) & (ef > es) & (lo <= ef) & (c > ef) & (c > o)
                       & r.between(40, 65))
        out["short"] = ((reg["direction"] < 0) & (ef < es) & (hi >= ef) & (c < ef) & (c < o)
                        & r.between(35, 60))
        out["sl_dist"] = 2.0 * f["atr"]
        out["tp_dist"] = 3.0 * f["atr"]
        out["score"] = reg["score"]
        out["exit_long"], out["exit_short"] = c < es, c > es
        return out


class MeanReversion(Agent):
    """Only in NON-trending markets (higher-timeframe ADX < 20): fade closes outside the
    Bollinger bands with extreme RSI, exit back at the 20-bar average."""

    def frame(self, e, reg, s):
        c = e["close"]
        mid = c.rolling(20).mean()
        sd = c.rolling(20).std()
        a = ta.atr(e, s.atr_len)
        r = ta.rsi(c, 14)
        ranging = reg["adx"] < 20
        out = pd.DataFrame(index=e.index)
        out["long"] = ranging & (c < mid - 2 * sd) & (r < 28)
        out["short"] = ranging & (c > mid + 2 * sd) & (r > 72)
        out["sl_dist"] = 1.5 * a
        out["tp_dist"] = ((c - mid).abs()).clip(lower=0.75 * a)          # back to the mean
        out["score"] = ((50 - r).abs() / 50).clip(0, 1)
        out["exit_long"], out["exit_short"] = c >= mid, c <= mid
        return out


class MomentumBurst(Agent):
    """A big candle on 2.5x normal volume, in the higher-timeframe trend direction, closing near its extreme."""

    def frame(self, e, reg, s):
        c, o, h, lo, v = (e[k] for k in ("close", "open", "high", "low", "volume"))
        a = ta.atr(e, s.atr_len)
        vol_ratio = v / v.rolling(20).mean().shift(1)
        body = (c - o).abs()
        rng = (h - lo).replace(0, np.nan)
        ef = ta.ema(c, s.ema_fast)
        out = pd.DataFrame(index=e.index)
        strong = (vol_ratio > 2.5) & (body > a)
        out["long"] = strong & (reg["direction"] > 0) & (c > o) & ((c - lo) / rng > 0.7)
        out["short"] = strong & (reg["direction"] < 0) & (c < o) & ((h - c) / rng > 0.7)
        out["sl_dist"] = 1.5 * a
        out["tp_dist"] = 3.0 * a
        out["score"] = (vol_ratio / 5).clip(0, 1).fillna(0)
        out["exit_long"], out["exit_short"] = c < ef, c > ef
        return out


class MomentumChaser(Agent):
    """TARGET MODE (paper only): no waiting for a breakout. Rides whatever direction the coin is
    moving on its own chart right now: fast EMA on the right side of the slow EMA, price on the right
    side of the fast EMA, and the last 3 candles moving that way. Tight stop, quick target."""

    def frame(self, e, reg, s):
        f = entry_frame(e, s)
        c = e["close"]
        ef, es, r, a = f["ema_fast"], f["ema_slow"], f["rsi"], f["atr"]
        up_mom, dn_mom = c > c.shift(3), c < c.shift(3)
        out = pd.DataFrame(index=e.index)
        out["long"] = (ef > es) & (c > ef) & up_mom & r.between(50, 80)
        out["short"] = (ef < es) & (c < ef) & dn_mom & r.between(20, 50)
        out["sl_dist"] = 1.5 * a
        out["tp_dist"] = 2.5 * a
        out["score"] = ((ef - es).abs() / a).clip(0, 3).fillna(0) / 3
        out["exit_long"], out["exit_short"] = c < es, c > es
        return out

    def explain(self, e, reg, s) -> dict | None:
        f = entry_frame(e, s).iloc[-1]
        c, c3 = float(e["close"].iloc[-1]), float(e["close"].iloc[-4])
        up = f["ema_fast"] > f["ema_slow"]
        side = "UP" if up else "DOWN"
        missing = []
        if up and not c > f["ema_fast"]:
            missing.append("price back above EMA20")
        if not up and not c < f["ema_fast"]:
            missing.append("price back below EMA20")
        if up and not c > c3:
            missing.append("3-candle momentum up")
        if not up and not c < c3:
            missing.append("3-candle momentum down")
        if up and not 50 <= f["rsi"] <= 80:
            missing.append(f"RSI 50-80 (now {f['rsi']:.0f})")
        if not up and not 20 <= f["rsi"] <= 50:
            missing.append(f"RSI 20-50 (now {f['rsi']:.0f})")
        text = f"{side} on its chart · " + ("ready" if not missing else "needs " + ", ".join(missing))
        return {"dist": len(missing), "text": text}


def target_team() -> list[Agent]:
    """Aggressive target-chasing line-up (paper only). Trades far more often; expect bigger swings."""
    return [
        MomentumChaser("chase1", "Chaser 1m", "Target mode: rides 1-minute momentum", "15m", "1m", max_positions=2, max_hold_bars=30),
        MomentumChaser("chase3", "Chaser 3m", "Target mode: rides 3-minute momentum", "15m", "3m", max_positions=2, max_hold_bars=40),
        MomentumChaser("chase5", "Chaser 5m", "Target mode: rides 5-minute momentum", "1h", "5m", max_positions=2, max_hold_bars=48),
        MomentumChaser("chase15", "Chaser 15m", "Target mode: rides 15-minute momentum", "1h", "15m", max_positions=2, max_hold_bars=48),
    ]


def default_team() -> list[Agent]:
    """The arena line-up. Breakouts were the only style with a positive result after realistic
    limit-order costs (walk-forward, 12 coins, 10 and 30 days); the other styles lost even at near-zero cost."""
    return [
        TrendBreakout("scalp1", "Scalper 1m", "1-minute scalps. Tested negative (fees beat the moves): on probation",
                      "15m", "1m", max_hold_bars=30),
        TrendBreakout("brk3", "Breakout 3m", "Fast breakouts, limit-order entries", "15m", "3m", max_hold_bars=120),
        TrendBreakout("brk5", "Breakout 5m", "Breakouts in the hourly trend", "1h", "5m", max_hold_bars=96),
        TrendBreakout("brk15", "Breakout 15m", "Slower breakouts in the hourly trend", "1h", "15m", max_hold_bars=96),
    ]


def classic_team() -> list[Agent]:
    """The original 5-style team (kept for comparison; 3 of these styles showed no edge)."""
    return [
        TrendBreakout("trend", "Trend Rider", "Breakout in the trend direction", "1h", "15m", max_hold_bars=96),
        TrendBreakout("scalper", "Scalper", "Fast breakouts", "15m", "5m", max_hold_bars=72),
        Pullback("pullback", "Pullback Buyer", "Buys dips inside a trend", "1h", "15m", max_hold_bars=96),
        MeanReversion("meanrev", "Mean Reverter", "Fades extremes in sideways markets", "1h", "5m", max_hold_bars=36),
        MomentumBurst("momentum", "Momentum Hunter", "Rides sudden volume spikes", "1h", "5m", max_hold_bars=24),
    ]


# ---------------------------------------------------------------- per-agent backtest
def simulate_agent(e: pd.DataFrame, reg: pd.DataFrame, agent: Agent, contract_value: float,
                   s: Settings, start: float) -> tuple[list[dict], float]:
    """Same fill rules as backtest.simulate: signal at close -> next open, SL/TP intrabar (stop first)."""
    from .risk import position_size
    from .data import Product
    from .strategy import Signal

    f = agent.frame(e, reg, s)
    prod = Product(0, "x", contract_value, 1e-9, s.fee_rate, 100)
    o, h, l, c = (e[k].values for k in ("open", "high", "low", "close"))
    L, S, SL, TP = f["long"].values, f["short"].values, f["sl_dist"].values, f["tp_dist"].values
    XL, XS = f["exit_long"].values, f["exit_short"].values
    cash, pos, trades, pending_exit = start, None, [], None
    warm = 60

    def close(px, reason, i):
        nonlocal cash, pos
        long = pos["side"] == "buy"
        px = px * (1 - s.slippage) if long else px * (1 + s.slippage)
        gross = (1 if long else -1) * (px - pos["entry"]) * pos["size"] * contract_value
        fee = px * pos["size"] * contract_value * s.fee_rate
        cash += gross - fee
        trades.append(dict(side=pos["side"], pnl=gross - fee - pos["fee"], reason=reason, bars=i - pos["bar"],
                           time=e.index[pos["bar"]]))
        pos = None

    for i in range(warm, len(e)):
        if pos and pending_exit:
            close(o[i], pending_exit, i)
            pending_exit = None
        if pos is None and (L[i - 1] or S[i - 1]) and np.isfinite(SL[i - 1]) and SL[i - 1] > 0:
            long = bool(L[i - 1])
            ref = c[i - 1]
            sig = Signal("buy" if long else "sell", ref, ref - SL[i - 1] if long else ref + SL[i - 1],
                         ref + TP[i - 1] if long else ref - TP[i - 1], SL[i - 1])
            size = position_size(prod, sig, cash, 0, s)
            if size >= 1:
                px = o[i] * (1 + s.slippage) if long else o[i] * (1 - s.slippage)
                fee = px * size * contract_value * s.fee_rate
                cash -= fee
                pos = dict(side=sig.side, size=size, entry=px, stop=sig.stop, target=sig.target, bar=i, fee=fee)
        if pos:
            long = pos["side"] == "buy"
            hit_sl = l[i] <= pos["stop"] if long else h[i] >= pos["stop"]
            hit_tp = h[i] >= pos["target"] if long else l[i] <= pos["target"]
            if hit_sl:
                close(min(pos["stop"], o[i]) if long else max(pos["stop"], o[i]), "stop_loss", i)
            elif hit_tp:
                close(pos["target"], "take_profit", i)
        if pos:
            if XL[i] if pos["side"] == "buy" else XS[i]:
                pending_exit = "signal_exit"
            elif i - pos["bar"] >= agent.max_hold_bars:
                pending_exit = "time_stop"
    if pos:
        close(c[-1], "end", len(e) - 1)
    return trades, cash
