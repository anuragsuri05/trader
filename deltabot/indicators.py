"""Technical indicators on pandas OHLC frames (columns: open, high, low, close, volume)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing."""
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return rma(tr, n)


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain = rma(delta.clip(lower=0), n)
    loss = rma(-delta.clip(upper=0), n)
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr_s = atr(df, n)
    plus_di = 100 * rma(plus_dm, n) / tr_s
    minus_di = 100 * rma(minus_dm, n) / tr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return rma(dx.fillna(0), n)


def efficiency_ratio(s: pd.Series, n: int) -> pd.Series:
    """Kaufman efficiency ratio: 1 = straight-line trend, 0 = pure noise."""
    change = (s - s.shift(n)).abs()
    noise = s.diff().abs().rolling(n).sum()
    return change / noise.replace(0, np.nan)
