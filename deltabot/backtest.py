"""Backtest the exact same strategy code on Delta Exchange history (no API key needed).

Rules mirror the live bot: signal on a closed bar, fill at next bar open with slippage + fees,
SL/TP checked intrabar (stop assumed first if both touched), trend-break and time-stop exits.
Each symbol is simulated independently with its own starting equity.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .client import RESOLUTION_SECONDS, DeltaClient
from .config import Settings
from .data import Product, candles_df, unix_seconds
from .risk import position_size
from .strategy import Signal, entry_frame, regime_frame


def align_regime(entry_df: pd.DataFrame, regime_df: pd.DataFrame, s: Settings) -> pd.Series:
    """Direction known at each entry bar's close, using only CLOSED regime bars."""
    r = regime_frame(regime_df, s)[["direction"]].copy()
    r["avail"] = r.index + pd.Timedelta(seconds=RESOLUTION_SECONDS[s.regime_tf])
    e = pd.DataFrame({"avail": entry_df.index + pd.Timedelta(seconds=RESOLUTION_SECONDS[s.entry_tf])})
    m = pd.merge_asof(e.sort_values("avail"), r.sort_values("avail"), on="avail", direction="backward")
    return pd.Series(m["direction"].fillna(0).astype(int).values, index=entry_df.index)


def simulate(df: pd.DataFrame, direction: pd.Series, product: Product, s: Settings, start_equity: float):
    f = entry_frame(df, s)
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    bar_s = RESOLUTION_SECONDS[s.entry_tf]
    max_bars = int(s.max_hold_hours * 3600 / bar_s)
    cv, cost = product.contract_value, s.fee_rate

    cash, pos, trades, curve = start_equity, None, [], []
    warm = max(s.ema_slow, s.donchian, s.atr_len) + 5
    pending_exit = None
    for i in range(warm, len(df)):
        # 1) fills scheduled at this bar's open
        if pos and pending_exit:
            px = o[i] * (1 - s.slippage) if pos["side"] == "buy" else o[i] * (1 + s.slippage)
            cash, pos = _close(pos, px, pending_exit, cash, cv, cost, trades, df.index[i])
            pending_exit = None
        if pos is None and pending_exit is None:
            sig = _signal_at(f, c, direction, i - 1, s)
            if sig:
                size = position_size(product, sig, cash, 0, s)
                if size >= 1:
                    px = o[i] * (1 + s.slippage) if sig.side == "buy" else o[i] * (1 - s.slippage)
                    cash -= px * size * cv * cost
                    pos = dict(side=sig.side, size=size, entry=px, stop=sig.stop, target=sig.target,
                               bar=i, time=df.index[i])
        # 2) intrabar stop-loss / take-profit
        if pos:
            long = pos["side"] == "buy"
            hit_sl = l[i] <= pos["stop"] if long else h[i] >= pos["stop"]
            hit_tp = h[i] >= pos["target"] if long else l[i] <= pos["target"]
            if hit_sl or hit_tp:
                level = pos["stop"] if hit_sl else pos["target"]
                # gap through the level -> fill at the open
                if hit_sl:
                    level = min(level, o[i]) if long else max(level, o[i])
                px = level * (1 - s.slippage) if long else level * (1 + s.slippage)
                cash, pos = _close(pos, px, "stop_loss" if hit_sl else "take_profit", cash, cv, cost, trades, df.index[i])
        # 3) close-of-bar exit checks (fill next open)
        if pos:
            long = pos["side"] == "buy"
            if (f["long_exit"].iloc[i] if long else f["short_exit"].iloc[i]):
                pending_exit = "trend_break"
            elif i - pos["bar"] >= max_bars:
                pending_exit = "time_stop"
        # mark to market
        mtm = 0.0
        if pos:
            sign = 1 if pos["side"] == "buy" else -1
            mtm = sign * (c[i] - pos["entry"]) * pos["size"] * cv
        curve.append(cash + mtm)
    if pos:
        cash, pos = _close(pos, c[-1], "end_of_test", cash, cv, cost, trades, df.index[-1])
        curve[-1] = cash
    return trades, pd.Series(curve, index=df.index[warm:])


def simulate_options(df: pd.DataFrame, direction: pd.Series, s: Settings, start_equity: float,
                     contract_value: float = 0.001, iv_mult: float = 1.15, half_spread: float = 0.015):
    """MODEL-BASED options backtest: same entry/exit signals, but buys a call (uptrend) or put
    (downtrend) priced with Black-Scholes. IV = 30-day realized vol x iv_mult (options usually trade
    above realized vol). Pays half_spread on each side plus fees. Strike chosen by target delta,
    expiry = opt_target_dte. Real fills on thin option books will differ."""
    from .options import bs_price, option_fee, option_size, strike_for_delta

    f = entry_frame(df, s)
    c = df["close"].values
    bar_s = RESOLUTION_SECONDS[s.entry_tf]
    bars_year = 365 * 86400 / bar_s
    rv = np.log(df["close"]).diff().rolling(int(30 * 86400 / bar_s)).std() * np.sqrt(bars_year)
    iv = (rv * iv_mult).clip(0.2, 2.0).values
    times = np.array(unix_seconds(df.index))
    max_bars = int(s.max_hold_hours * 3600 / bar_s)

    cash, pos, trades, curve = start_equity, None, [], []
    warm = max(s.ema_slow, s.donchian, s.atr_len, int(30 * 86400 / bar_s)) + 5
    for i in range(warm, len(df)):
        if pos:
            yrs = (pos["expiry"] - times[i]) / (365 * 86400)
            mark = bs_price(c[i], pos["k"], yrs, iv[i], pos["call"])
            bid = mark * (1 - half_spread)
            reason = None
            if yrs * 365 * 24 < s.opt_exit_before_expiry_h:
                reason = "near_expiry"
            elif bid <= pos["stop"]:
                reason = "premium_stop"
            elif bid >= pos["target"]:
                reason = "premium_target"
            elif f["long_exit"].iloc[i] if pos["call"] else f["short_exit"].iloc[i]:
                reason = "trend_break"
            elif i - pos["bar"] >= max_bars:
                reason = "time_stop"
            if reason:
                fee = option_fee(bid, c[i], pos["size"], contract_value, s)
                proceeds = bid * pos["size"] * contract_value - fee
                cash += proceeds
                trades.append(dict(entry_time=pos["time"], exit_time=df.index[i], side="call" if pos["call"] else "put",
                                   size=pos["size"], entry=pos["entry"], exit=bid,
                                   pnl=proceeds - pos["cost"], reason=reason))
                pos = None
        if pos is None and np.isfinite(iv[i]):
            sig = _signal_at(f, c, direction, i, s)
            if sig:
                call = sig.side == "buy"
                yrs = s.opt_target_dte / 365
                k = strike_for_delta(c[i], yrs, iv[i], s.opt_target_delta, call)
                ask = bs_price(c[i], k, yrs, iv[i], call) * (1 + half_spread)
                size = option_size(ask, contract_value, cash, s)
                if size >= 1:
                    fee = option_fee(ask, c[i], size, contract_value, s)
                    cost = ask * size * contract_value + fee
                    cash -= cost
                    pos = dict(call=call, k=k, expiry=times[i] + s.opt_target_dte * 86400, size=size,
                               entry=ask, cost=cost, stop=ask * (1 - s.opt_stop_loss),
                               target=ask * (1 + s.opt_take_profit), bar=i, time=df.index[i])
        value = 0.0
        if pos:
            yrs = (pos["expiry"] - times[i]) / (365 * 86400)
            value = bs_price(c[i], pos["k"], yrs, iv[i], pos["call"]) * pos["size"] * contract_value
        curve.append(cash + value)
    if pos:   # mark the still-open option at model value
        value_end = curve[-1] - cash
        trades.append(dict(entry_time=pos["time"], exit_time=df.index[-1], side="call" if pos["call"] else "put",
                           size=pos["size"], entry=pos["entry"], exit=None, pnl=value_end - pos["cost"],
                           reason="end_of_test"))
    return trades, pd.Series(curve, index=df.index[warm:])


def _signal_at(f: pd.DataFrame, c, direction: pd.Series, i: int, s: Settings) -> Signal | None:
    d = direction.iloc[i]
    a = f["atr"].iloc[i]
    if d == 0 or not np.isfinite(a) or a <= 0:
        return None
    price = c[i]
    if d > 0 and f["long_entry"].iloc[i]:
        return Signal("buy", price, price - s.sl_atr * a, price + s.tp_atr * a, a)
    if d < 0 and f["short_entry"].iloc[i]:
        return Signal("sell", price, price + s.sl_atr * a, price - s.tp_atr * a, a)
    return None


def _close(pos, px, reason, cash, cv, cost, trades, ts):
    sign = 1 if pos["side"] == "buy" else -1
    gross = sign * (px - pos["entry"]) * pos["size"] * cv
    fee = px * pos["size"] * cv * cost
    cash += gross - fee
    trades.append(dict(entry_time=pos["time"], exit_time=ts, side=pos["side"], size=pos["size"],
                       entry=pos["entry"], exit=px, pnl=gross - fee, reason=reason))
    return cash, None


def stats(trades: list[dict], curve: pd.Series, start: float) -> dict:
    pnl = np.array([t["pnl"] for t in trades]) if trades else np.array([])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    dd = (curve / curve.cummax() - 1).min() if len(curve) else 0
    return {
        "trades": len(trades),
        "win_rate": float(len(wins) / len(pnl)) if len(pnl) else 0.0,
        "return": float(curve.iloc[-1] / start - 1) if len(curve) else 0.0,
        "max_dd": float(dd),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf") if len(wins) else 0.0,
        "avg_trade": float(pnl.mean()) if len(pnl) else 0.0,
    }


def run_backtest(client: DeltaClient, products: dict[str, Product], symbols: list[str], days: int,
                 s: Settings, start_equity: float = 1000.0, instrument: str = "perpetual") -> pd.DataFrame:
    rows = []
    entry_bars = int(days * 86400 / RESOLUTION_SECONDS[s.entry_tf])
    regime_bars = int(days * 86400 / RESOLUTION_SECONDS[s.regime_tf]) + s.regime_ema_slow + 50
    for sym in symbols:
        if sym not in products or products[sym].kind != "perpetual":
            print(f"  skip {sym}: not a live linear perpetual")
            continue
        opt_cv = next((p.contract_value for p in products.values()
                       if p.kind == "option" and p.underlying == products[sym].underlying), None)
        if instrument == "options" and opt_cv is None:
            print(f"  skip {sym}: no options listed on {products[sym].underlying}")
            continue
        t0 = time.time()
        edf = candles_df(client, sym, s.entry_tf, entry_bars)
        rdf = candles_df(client, sym, s.regime_tf, regime_bars)
        if len(edf) < 500 or len(rdf) < s.regime_ema_slow + 50:
            print(f"  skip {sym}: not enough history ({len(edf)} bars)")
            continue
        direction = align_regime(edf, rdf, s)
        if instrument == "options":
            trades, curve = simulate_options(edf, direction, s, start_equity, opt_cv)
        else:
            trades, curve = simulate(edf, direction, products[sym], s, start_equity)
        st = stats(trades, curve, start_equity)
        st["buy_hold"] = float(edf["close"].iloc[-1] / edf["close"].iloc[0] - 1)
        st["symbol"] = sym
        st["days"] = round((edf.index[-1] - edf.index[0]).total_seconds() / 86400)
        rows.append(st)
        print(f"  {sym:<12} trades={st['trades']:<4} win={st['win_rate']:.0%} ret={st['return']:+.1%} "
              f"maxDD={st['max_dd']:.1%} PF={st['profit_factor']:.2f} buy&hold={st['buy_hold']:+.1%} "
              f"({time.time() - t0:.1f}s)")
    return pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame()
