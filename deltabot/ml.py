"""Trained trade filter ("meta-labeling").

1. Candidate trades come from the existing rule bots (momentum chaser + breakout signals).
2. Each candidate is labeled by what really happened: enter at the next candle's open (market order,
   taker fee + slippage), stop 1.5 ATR, target 2.5 ATR, max hold 48 candles, stop assumed first if both
   are touched in one candle. Result is measured in R (multiples of the stop distance) after costs.
3. A gradient-boosting model learns which market conditions make a candidate a winner.
4. Strict time split: train on the oldest 60%, choose the probability threshold on the next 20%,
   and report the untouched newest 20%. Overlapping trades on the same coin are skipped so results
   match what a bot holding one position per coin could actually do.

The SAME `features()` function is used for training and for live trading.
"""
from __future__ import annotations

import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import indicators as ta
from .agents import MomentumChaser, TrendBreakout, regime_on
from .client import RESOLUTION_SECONDS
from .config import ROOT, Settings
from .strategy import entry_frame

MODEL_DIR = ROOT / "models"
SL_ATR, TP_ATR, MAX_HOLD = 1.5, 2.5, 48
COST_PER_SIDE = 0.0005 * 1.18 + 0.0005          # taker fee incl. GST + slippage
MAKER_COST = 0.0002 * 1.18                       # limit-order entry, no slippage
LIMIT_WAIT = 3                                   # candles a limit entry stays valid

FEATURES = [
    "side", "tf_min", "ema_gap", "px_vs_ema20", "px_vs_ema50", "ret3", "ret12", "ret48", "rsi_c", "rsi_abs",
    "atr_pct", "adx_e", "er_e", "adx_r", "reg_dir", "dc_pos", "log_vol", "body", "upper_wick",
    "lower_wick", "hour_sin", "hour_cos", "btc_ret12", "btc_trend", "is_breakout", "is_chaser",
]


def _sig_frames(e: pd.DataFrame, reg: pd.DataFrame, s: Settings, tf: str):
    chase = MomentumChaser("c", "", "", "1h", tf).frame(e, reg, s)
    brk = TrendBreakout("b", "", "", "1h", tf).frame(e, reg, s)
    return chase, brk


def features(e: pd.DataFrame, reg: pd.DataFrame, btc: pd.DataFrame | None, s: Settings, tf: str) -> pd.DataFrame:
    """Per-bar features for BOTH sides; candidates and side are picked afterwards."""
    f = entry_frame(e, s)
    c, o, h, lo, v = (e[k] for k in ("close", "open", "high", "low", "volume"))
    a = f["atr"].replace(0, np.nan)
    rng = (h - lo).replace(0, np.nan)
    out = pd.DataFrame(index=e.index)
    out["tf_min"] = RESOLUTION_SECONDS[tf] / 60
    out["ema_gap"] = (f["ema_fast"] - f["ema_slow"]) / a
    out["px_vs_ema20"] = (c - f["ema_fast"]) / a
    out["px_vs_ema50"] = (c - f["ema_slow"]) / a
    out["ret3"] = (c - c.shift(3)) / a
    out["ret12"] = (c - c.shift(12)) / a
    out["ret48"] = (c - c.shift(48)) / a
    out["rsi_c"] = f["rsi"] - 50
    out["atr_pct"] = a / c * 100
    out["adx_e"] = ta.adx(e, 14)
    out["er_e"] = ta.efficiency_ratio(c, 20)
    out["adx_r"] = reg["adx"]
    out["reg_dir"] = reg["direction"]
    hi20, lo20 = h.rolling(20).max().shift(1), lo.rolling(20).min().shift(1)
    out["dc_pos"] = ((c - lo20) / (hi20 - lo20).replace(0, np.nan)) - 0.5
    out["log_vol"] = np.log((v / v.rolling(20).mean().shift(1)).replace(0, np.nan))
    out["body"] = (c - o) / a
    out["upper_wick"] = (h - np.maximum(c, o)) / rng
    out["lower_wick"] = (np.minimum(c, o) - lo) / rng
    hour = e.index.hour + e.index.minute / 60
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    if btc is not None and len(btc):
        bc = btc["close"].reindex(e.index, method="ffill")
        out["btc_ret12"] = (bc / bc.shift(12) - 1) * 100
        be20, be50 = ta.ema(btc["close"], 20).reindex(e.index, method="ffill"), ta.ema(btc["close"], 50).reindex(e.index, method="ffill")
        out["btc_trend"] = np.sign(be20 - be50)
    else:
        out["btc_ret12"], out["btc_trend"] = 0.0, 0.0
    chase, brk = _sig_frames(e, reg, s, tf)
    out["_long_c"], out["_short_c"] = chase["long"], chase["short"]
    out["_long_b"], out["_short_b"] = brk["long"], brk["short"]
    out["_atr"] = f["atr"]
    return out


DIRECTIONAL = ["ema_gap", "px_vs_ema20", "px_vs_ema50", "ret3", "ret12", "ret48", "rsi_c", "reg_dir", "dc_pos",
               "body", "btc_ret12", "btc_trend"]


def candidates(feat: pd.DataFrame) -> pd.DataFrame:
    """One row per (bar, side) where a rule bot would trade. Directional features are flipped for shorts
    so the model learns 'aligned with the trade' rather than 'up'."""
    rows = []
    for side, lc, lb in ((1, "_long_c", "_long_b"), (-1, "_short_c", "_short_b")):
        m = feat[lc].fillna(False) | feat[lb].fillna(False)
        sub = feat[m].copy()
        if sub.empty:
            continue
        sub["side"] = side
        sub["is_chaser"] = sub[lc].astype(float)
        sub["is_breakout"] = sub[lb].astype(float)
        for col in DIRECTIONAL:
            sub[col] = sub[col] * side
        upper, lower = sub["upper_wick"].copy(), sub["lower_wick"].copy()
        if side < 0:                                 # the wick against the trade
            sub["upper_wick"], sub["lower_wick"] = lower, upper
        rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=FEATURES)
    out = pd.concat(rows).sort_index()
    out["rsi_abs"] = out["rsi_c"].abs()
    return out


def label_trades(e: pd.DataFrame, cand: pd.DataFrame, entry_mode: str = "market") -> pd.DataFrame:
    """Simulate each candidate, 1.5 ATR stop, 2.5 ATR target, max 48 candles.
    market: enter at the next open (taker + slippage).
    limit:  resting order at the signal close; fills only if price trades THROUGH it within 3 candles
            (maker fee, no slippage); unfilled candidates are dropped."""
    o, h, lo, c = (e[k].values for k in ("open", "high", "low", "close"))
    pos = e.index.get_indexer(cand.index)
    r_out, exit_i = [], []
    for p, side, atr in zip(pos, cand["side"].values, cand["_atr"].values):
        i = p + 1
        if i >= len(e) or not atr > 0:
            r_out.append(np.nan); exit_i.append(p); continue
        entry_cost = COST_PER_SIDE
        if entry_mode == "limit":
            lim, filled = c[p], None
            for k in range(i, min(i + LIMIT_WAIT, len(e))):
                if (lo[k] < lim) if side > 0 else (h[k] > lim):
                    filled = k; break
            if filled is None:
                r_out.append(np.nan); exit_i.append(p); continue
            i, entry, entry_cost = filled, (min(lim, o[filled]) if side > 0 else max(lim, o[filled])), MAKER_COST
        else:
            entry = o[i]
        stop = entry - side * SL_ATR * atr
        target = entry + side * TP_ATR * atr
        px, j = None, i
        for j in range(i, min(i + MAX_HOLD, len(e))):
            hit_sl = lo[j] <= stop if side > 0 else h[j] >= stop
            hit_tp = h[j] >= target if side > 0 else lo[j] <= target
            if hit_sl:
                px = min(stop, o[j]) if side > 0 else max(stop, o[j]); break
            if hit_tp:
                px = target; break
        if px is None:
            px = c[j]
        gross = side * (px - entry)
        exit_cost = MAKER_COST if px == target else COST_PER_SIDE      # target = resting limit order
        cost = entry_cost * entry + exit_cost * px
        r_out.append((gross - cost) / (SL_ATR * atr))
        exit_i.append(j)
    cand = cand.copy()
    cand["r"] = r_out
    cand["exit_pos"] = exit_i
    cand["entry_pos"] = pos
    return cand.dropna(subset=["r"])


@dataclass
class TrainResult:
    threshold: float
    report: dict
    model: object


def no_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Keep trades a one-position-per-coin bot could actually take."""
    keep, busy_until = [], {}
    for idx, row in df.sort_values("time").iterrows():
        key = (row["sym"], row["tf"])
        if row["entry_pos"] <= busy_until.get(key, -1):
            continue
        keep.append(idx)
        busy_until[key] = row["exit_pos"]
    return df.loc[keep]


def _stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"trades": 0, "win_rate": 0.0, "avg_r": 0.0, "total_r": 0.0}
    return {"trades": int(len(df)), "win_rate": float((df["r"] > 0).mean()), "avg_r": float(df["r"].mean()),
            "total_r": float(df["r"].sum())}


def train(data: pd.DataFrame, log=print) -> TrainResult:
    from sklearn.ensemble import HistGradientBoostingClassifier

    data = data.sort_values("time").reset_index(drop=True)
    n = len(data)
    t1, t2 = data["time"].iloc[int(n * 0.6)], data["time"].iloc[int(n * 0.8)]
    gap = pd.Timedelta(hours=12)                                  # embargo between splits
    tr = data[data["time"] < t1 - gap]
    va = data[(data["time"] >= t1) & (data["time"] < t2 - gap)]
    te = data[data["time"] >= t2]
    log(f"train {len(tr)} candidates ({tr['time'].min():%d %b} - {tr['time'].max():%d %b}), "
        f"validate {len(va)}, test {len(te)} ({te['time'].min():%d %b} - {te['time'].max():%d %b})")
    model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250, min_samples_leaf=80,
                                           l2_regularization=1.0, random_state=0)
    model.fit(tr[FEATURES], (tr["r"] > 0).astype(int))
    va = va.assign(p=model.predict_proba(va[FEATURES])[:, 1])
    te = te.assign(p=model.predict_proba(te[FEATURES])[:, 1])

    best_th, best_total = None, -math.inf
    for th in np.arange(0.30, 0.71, 0.02):
        st = _stats(no_overlap(va[va["p"] >= th]))
        if st["trades"] >= 40 and st["total_r"] > best_total:
            best_th, best_total = float(th), st["total_r"]
    if best_th is None:
        best_th = 0.5
    report = {
        "validation_all": _stats(no_overlap(va)), "validation_model": _stats(no_overlap(va[va["p"] >= best_th])),
        "test_all": _stats(no_overlap(te)), "test_model": _stats(no_overlap(te[te["p"] >= best_th])),
        "threshold": best_th,
        "test_by_tf": {tf: _stats(no_overlap(g[g["p"] >= best_th])) for tf, g in te.groupby("tf")},
        "test_by_week": {str(k.date()): _stats(no_overlap(g[g["p"] >= best_th]))
                         for k, g in te.groupby(pd.Grouper(key="time", freq="W"))},
    }
    return TrainResult(best_th, report, model)


def build_dataset(client, products, symbols: list[str], s: Settings, days: int,
                  tfs=(("5m", "1h"), ("15m", "1h")), entry_mode: str = "market", log=print) -> pd.DataFrame:
    from .data import candles_df
    frames = []
    btc_cache = {}
    for tf, rtf in tfs:
        bars = int(days * 86400 / RESOLUTION_SECONDS[tf]) + 100
        rbars = int(days * 86400 / RESOLUTION_SECONDS[rtf]) + s.regime_ema_slow + 60
        btc_cache[tf] = candles_df(client, "BTCUSD", tf, bars)
        for sym in symbols:
            try:
                e = candles_df(client, sym, tf, bars)
                r = candles_df(client, sym, rtf, rbars)
            except Exception as ex:
                log(f"  skip {sym} {tf}: {ex}")
                continue
            if len(e) < 500 or len(r) < s.regime_ema_slow + 50:
                continue
            reg = regime_on(e, r, s, rtf, tf)
            feat = features(e, reg, btc_cache[tf], s, tf)
            cand = label_trades(e, candidates(feat), entry_mode)
            if cand.empty:
                continue
            cand["sym"], cand["tf"], cand["time"] = sym, tf, cand.index
            frames.append(cand)
        log(f"  {tf}: {sum(len(f) for f in frames if f['tf'].iloc[0] == tf)} candidate trades")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save(result: TrainResult, meta: dict) -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / "trade_filter.pkl"
    with path.open("wb") as fh:
        pickle.dump({"model": result.model, "threshold": result.threshold, "features": FEATURES,
                     "meta": meta, "report": result.report, "trained_at": time.time()}, fh)
    (MODEL_DIR / "trade_filter_report.json").write_text(json.dumps({"meta": meta, **result.report}, indent=2, default=str))
    return path


def load() -> dict | None:
    path = MODEL_DIR / "trade_filter.pkl"
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)
