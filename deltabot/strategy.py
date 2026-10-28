"""Trading logic shared by the live bot and the backtester.

Two layers:
  1. Regime (higher timeframe, default 4h): is this market trending, and which way?
     Used by the scanner to pick which markets to trade.
  2. Entry/exit (lower timeframe, default 1h): breakout in the trend direction,
     ATR stop-loss and take-profit, exit early when the short-term trend breaks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ta
from .config import Settings


# ------------------------------------------------------------------ regime
def regime_frame(df: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Per-bar regime features. direction: +1 uptrend, -1 downtrend, 0 no trade."""
    out = pd.DataFrame(index=df.index)
    c = df["close"]
    ef = ta.ema(c, s.regime_ema_fast)
    es = ta.ema(c, s.regime_ema_slow)
    out["adx"] = ta.adx(df, 14)
    out["er"] = ta.efficiency_ratio(c, 48)
    out["atr_pct"] = ta.atr(df, s.atr_len) / c
    slope = ef - ef.shift(5)
    up = (c > ef) & (ef > es) & (slope > 0)
    down = (c < ef) & (ef < es) & (slope < 0)
    trending = (out["adx"] >= s.adx_min) & (out["er"] >= s.er_min) & (out["atr_pct"] < 0.04)
    out["direction"] = np.where(up & trending, 1, np.where(down & trending, -1, 0))
    # score in [0, 1]: blend of trend strength and trend cleanliness
    out["score"] = (0.5 * (out["adx"] / 50).clip(0, 1) + 0.5 * out["er"].clip(0, 1)).fillna(0)
    out.loc[out["direction"] == 0, "score"] = 0.0
    return out


# ------------------------------------------------------------- entry / exit
def entry_frame(df: pd.DataFrame, s: Settings) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c = df["close"]
    out["ema_fast"] = ta.ema(c, s.ema_fast)
    out["ema_slow"] = ta.ema(c, s.ema_slow)
    out["atr"] = ta.atr(df, s.atr_len)
    out["rsi"] = ta.rsi(c, 14)
    # Donchian channel of the PREVIOUS n bars (excludes current bar -> no look-ahead)
    out["dc_high"] = df["high"].rolling(s.donchian).max().shift(1)
    out["dc_low"] = df["low"].rolling(s.donchian).min().shift(1)
    out["long_entry"] = (c > out["dc_high"]) & (out["ema_fast"] > out["ema_slow"]) & (out["rsi"] < s.rsi_max_long)
    out["short_entry"] = (c < out["dc_low"]) & (out["ema_fast"] < out["ema_slow"]) & (out["rsi"] > s.rsi_min_short)
    out["long_exit"] = c < out["ema_slow"]
    out["short_exit"] = c > out["ema_slow"]
    return out


@dataclass
class Signal:
    side: str          # "buy" | "sell"
    price: float       # reference price (last close)
    stop: float
    target: float
    atr: float


def entry_signal(df: pd.DataFrame, direction: int, s: Settings) -> Signal | None:
    """Check the last CLOSED bar for an entry in `direction`."""
    if direction == 0 or len(df) < max(s.ema_slow, s.donchian) + 5:
        return None
    f = entry_frame(df, s).iloc[-1]
    price, a = float(df["close"].iloc[-1]), float(f["atr"])
    if not np.isfinite(a) or a <= 0:
        return None
    if direction > 0 and f["long_entry"]:
        return Signal("buy", price, price - s.sl_atr * a, price + s.tp_atr * a, a)
    if direction < 0 and f["short_entry"]:
        return Signal("sell", price, price + s.sl_atr * a, price - s.tp_atr * a, a)
    return None


def should_exit(df: pd.DataFrame, side: str, s: Settings) -> bool:
    """Early exit when the short-term trend breaks against the position."""
    if len(df) < s.ema_slow + 5:
        return False
    f = entry_frame(df, s).iloc[-1]
    return bool(f["long_exit"]) if side == "buy" else bool(f["short_exit"])
