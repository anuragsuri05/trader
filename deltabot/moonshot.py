"""Moonshot mode (PAPER ONLY): all-in, maximum useful leverage, one goal - reach the target.

Every trade puts the WHOLE balance up as margin. A losing trade is a liquidation (balance ~0, session over);
a winning trade multiplies the balance by (1 + step). Chaining wins compounds toward the target.

Before it trades, `analyse()` replays the entry signal on recent real 1-minute Delta candles for every
leverage and step size, with real costs, and picks the combination with the best chance of hitting the
target. Fees and slippage are charged on the full leveraged position, which is why the maximum leverage
is usually NOT the best choice.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from .client import RESOLUTION_SECONDS
from .data import candles_df

TAKER = 0.0005 * 1.18          # fee per side incl. GST
SLIP = 0.0003                  # per side on liquid perps
MM = 0.004                     # maintenance margin (fraction of position) [verify per product on Delta]
LEVERAGES = [5, 10, 20, 25, 50, 75, 100, 150, 200]
STEPS = [0.5, 1.0, 2.0]        # balance gain per winning trade: +50%, x2, x3
MAX_MIN = 240                  # a trade that hits neither side is closed after 4 hours


def cost_rt() -> float:
    return 2 * (TAKER + SLIP)


def liq_move(lev: float) -> float:
    """Adverse price move that wipes the margin (fees included)."""
    return max(0.0002, 1 / lev - MM - cost_rt())


def tp_move(lev: float, step: float) -> float:
    """Favourable price move needed for the balance to grow by `step` after costs."""
    return step / lev + cost_rt()


def signal(df: pd.DataFrame) -> int:
    """+1 / -1 momentum burst on the last closed 1m candle, 0 = no trade."""
    if len(df) < 25:
        return 0
    c, v = df["close"].values, df["volume"].values
    r3 = c[-1] / c[-4] - 1
    vol = v[-1] / (v[-21:-1].mean() + 1e-12)
    rng = (df["high"].values[-20:] - df["low"].values[-20:]).mean() / c[-1]
    if vol < 1.5 or abs(r3) < 1.5 * rng:
        return 0
    return 1 if r3 > 0 else -1


def replay(df: pd.DataFrame, lev: float, step: float) -> list[float]:
    """Balance multipliers of every trade the signal would have taken (0 = liquidated)."""
    out, i, n = [], 25, len(df)
    o, h, lo, c = (df[k].values for k in ("open", "high", "low", "close"))
    lm, tm = liq_move(lev), tp_move(lev, step)
    while i < n - 1:
        side = signal(df.iloc[i - 25:i + 1])
        if not side:
            i += 1
            continue
        entry = o[i + 1]
        liq = entry * (1 - side * lm)
        tgt = entry * (1 + side * tm)
        mult, j = None, i + 1
        for j in range(i + 1, min(i + 1 + MAX_MIN, n)):
            if (lo[j] <= liq) if side > 0 else (h[j] >= liq):
                mult = 0.0
                break
            if (h[j] >= tgt) if side > 0 else (lo[j] <= tgt):
                mult = 1 + step
                break
        if mult is None:                                   # timed out: close at market
            move = side * (c[j] / entry - 1) - cost_rt()
            mult = max(0.0, 1 + lev * move)
        out.append(mult)
        i = j + 1
    return out


def chance(mults: list[float], start: float, target: float, max_trades: int, sims: int = 20000, seed: int = 0) -> dict:
    if not mults:
        return {"p_target": 0.0, "p_bust": 0.0, "median": start}
    rng = np.random.default_rng(seed)
    pool = np.array(mults)
    goal = target / start
    hit = bust = 0
    finals = np.empty(sims)
    for s in range(sims):
        eq = 1.0
        for _ in range(max_trades):
            eq *= pool[rng.integers(len(pool))]
            if eq >= goal:
                hit += 1
                break
            if eq < 0.02:
                bust += 1
                break
        finals[s] = eq
    return {"p_target": hit / sims, "p_bust": bust / sims, "median": float(np.median(finals) * start)}


def analyse(client, products, symbols: list[str], start: float, target: float, minutes: float,
            days: int = 5, progress=lambda m, p: None) -> dict:
    t0 = time.time()
    data = {}
    for k, sym in enumerate(symbols, 1):
        progress(f"Downloading 1-minute candles: {sym}", 5 + 40 * k / len(symbols))
        try:
            df = candles_df(client, sym, "1m", days * 1440)
        except Exception:
            continue
        if len(df) > 1000:
            data[sym] = df
    rows = []
    combos = [(L, st) for L in LEVERAGES for st in STEPS]
    for k, (lev, step) in enumerate(combos, 1):
        progress(f"Testing {lev}x leverage, +{step:.0%} per win", 45 + 50 * k / len(combos))
        allowed = [s for s in data if products[s].max_leverage >= lev]
        mults, minutes_used = [], 0.0
        for sym in allowed:
            m = replay(data[sym], lev, step)
            mults += m
        if not mults:
            continue
        signals_per_min = len(mults) / (len(allowed) * len(data[allowed[0]]))      # per coin-minute
        trades_in_session = max(1, int(minutes * signals_per_min * len(allowed)))  # watching all coins
        ch = chance(mults, start, target, trades_in_session)
        wins = sum(1 for x in mults if x > 1)
        rows.append({"leverage": lev, "step": step, "trades_measured": len(mults), "win_rate": wins / len(mults),
                     "liquidated": sum(1 for x in mults if x == 0) / len(mults),
                     "liq_move_pct": liq_move(lev) * 100, "tp_move_pct": tp_move(lev, step) * 100,
                     "fees_pct_of_balance": cost_rt() * lev * 100,
                     "trades_in_session": trades_in_session, **ch})
    rows.sort(key=lambda r: (r["p_target"], -r["p_bust"]), reverse=True)
    best = rows[0] if rows else None
    progress("Done", 100)
    return {"time": time.time(), "seconds": round(time.time() - t0, 1), "start": start, "target": target,
            "multiple": target / start, "minutes": minutes, "coins": list(data), "days": days,
            "best": best, "table": rows}


# ====================================================================== live paper engine
class MoonshotBot:
    """Paper-only engine: one all-in leveraged position at a time, checked every 5 seconds."""

    def __init__(self, s, analysis: dict | None = None):
        import logging
        from .broker import PaperBroker
        from .client import DeltaClient
        from .data import load_products
        from .notify import Notifier
        from .risk import RiskManager
        from .scanner import liquid_universe
        from .state import State
        if s.is_live:
            raise RuntimeError("Moonshot mode is paper-only")
        self.log = logging.getLogger("deltabot.moonshot")
        self.s = s
        self.client = DeltaClient(s.base_url)
        self.notify = Notifier(s)
        self.products = load_products(self.client)
        self.state = State.load("paper_moonshot")
        self.state.mode = "paper_moonshot"
        self.broker = PaperBroker(self.client, s, self.state, self.products)
        self.risk = RiskManager(s, self.state)
        self.candidates = []
        self.symbols = [x for x, _ in liquid_universe(self.client, self.products, s)[:8]]
        self.analysis = analysis
        best = (analysis or {}).get("best") or {}
        self.lev = float(best.get("leverage", 20))
        self.step = float(best.get("step", 1.0))
        self._c1: dict = {}

    # dashboard compatibility
    def new_session(self):
        self.state.trades.clear()
        self.state.paper_cash = self.s.paper_start_equity
        self.state.extra = {"moon": {"trades": [], "busted": False, "wins": 0}}
        self.state.halted_reason = ""
        self.state.save()
        self.notify(f"Moonshot armed: {self.lev:.0f}x leverage, all-in, +{self.step:.0%} per win")

    def equity(self) -> float:
        return max(0.0, self.broker.equity())

    def _open(self, sym: str, side: int, mark: float) -> None:
        from .strategy import Signal
        p = self.products[sym]
        eq = self.equity()
        lev = min(self.lev, p.max_leverage)
        size = math.floor(eq * lev / (p.contract_value * mark))
        if size < 1:
            return
        liq = mark * (1 - side * liq_move(lev))
        tgt = mark * (1 + side * tp_move(lev, self.step))
        sig = Signal("buy" if side > 0 else "sell", mark, liq, tgt, 0)
        self.broker.open(p, sig, size, signal_symbol=sym)
        t = self.state.trades[sym]
        t.agent, t.hold_until = "moonshot", time.time() + MAX_MIN * 60
        self.notify(f"MOONSHOT {'LONG' if side > 0 else 'SHORT'} {sym}: {size} contracts, {lev:.0f}x, all-in "
                    f"{eq:.2f} USD | liquidation {liq:.6g} | target {tgt:.6g}")

    def _close(self, sym: str, px: float, reason: str) -> None:
        t = self.state.trades[sym]
        pnl = self.broker.close(self.products[sym], reason, px)
        if reason == "liquidated" or self.state.paper_cash < 0.01:
            pnl -= self.state.paper_cash          # the exchange keeps the remaining maintenance margin
            self.state.paper_cash = 0.0
        moon = self.state.extra.setdefault("moon", {"trades": [], "busted": False, "wins": 0})
        moon["trades"].append({"time": time.time(), "sym": sym, "side": t.side, "reason": reason, "pnl": pnl,
                               "equity_after": self.equity()})
        if reason == "target":
            moon["wins"] += 1
        self.notify(f"MOONSHOT {sym} {reason.upper()}: P&L {pnl:+.2f} USD, balance {self.equity():.2f} USD")
        if self.equity() < 0.5:
            moon["busted"] = True
            self.state.halted_reason = "session: LIQUIDATED - the balance is gone (that is the usual outcome of this mode)"
            self.notify("MOONSHOT BUSTED: account liquidated")

    def step_once(self) -> None:
        st, s = self.state, self.s
        if st.halted_reason:
            return
        marks = {t["symbol"]: float(t.get("mark_price") or 0) for t in self.client.tickers("perpetual_futures")}
        self.broker.set_marks(marks)
        for sym, t in list(st.trades.items()):
            m = marks.get(sym) or t.entry_price
            long = t.side == "buy"
            if (m <= t.stop) if long else (m >= t.stop):
                self._close(sym, t.stop, "liquidated")
            elif (m >= t.target) if long else (m <= t.target):
                self._close(sym, t.target, "target")
            elif time.time() >= t.hold_until:
                self._close(sym, m, "timeout")
        eq = self.equity()
        if s.target_equity and eq >= s.target_equity and not st.halted_reason:
            for sym in list(st.trades):
                self._close(sym, marks.get(sym), "target_reached")
            st.halted_reason = f"target: REACHED {eq:.2f} USD"
            self.notify("MOONSHOT HIT THE TARGET")
        elif s.session_end and time.time() >= s.session_end and not st.halted_reason:
            for sym in list(st.trades):
                self._close(sym, marks.get(sym), "session_end")
            st.halted_reason = "session: time is up"
        if st.halted_reason or st.trades:
            st.save()
            return
        best = None
        for sym in self.symbols:                       # strongest momentum burst right now
            last = self._c1.get(sym)
            if not last or time.time() - last[0] > 55:
                try:
                    self._c1[sym] = (time.time(), candles_df(self.client, sym, "1m", 40))
                except Exception:
                    continue
            df = self._c1[sym][1]
            sd = signal(df)
            if sd:
                strength = abs(df["close"].values[-1] / df["close"].values[-4] - 1)
                if not best or strength > best[2]:
                    best = (sym, sd, strength)
        st.last_scan = time.time()
        if best and marks.get(best[0]):
            self._open(best[0], best[1], marks[best[0]])
        st.save()

    def run(self, stop=None) -> None:
        while not (stop and stop.is_set()):
            try:
                self.step_once()
            except Exception as e:
                self.log.exception("moonshot: %s", e)
            (stop.wait if stop else time.sleep)(5)

    def moon_view(self, rate: float) -> dict:
        moon = self.state.extra.get("moon", {})
        pos = []
        for sym, t in list(self.state.trades.items()):
            m = self.broker.marks.get(sym) or t.entry_price
            pos.append({"sym": sym, "side": t.side, "entry": t.entry_price, "liq": t.stop, "target": t.target,
                        "mark": m, "size": t.size})
        return {"leverage": self.lev, "step": self.step, "analysis": self.analysis, "positions": pos,
                "trades": moon.get("trades", [])[-10:], "wins": moon.get("wins", 0), "busted": moon.get("busted", False)}
