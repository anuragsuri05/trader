"""Day Trader pipeline for Delta Exchange (PAPER).

Method adapted from Jake Schincariol's day-trader-skill (MIT, github.com/Jakeschincariol/day-trader-skill),
ported from US stocks to crypto perpetuals:

  scanner -> plan (exit) -> Time Machine -> budget -> risk check -> YOUR APPROVAL -> paper order -> alerts

* Session: crypto never closes, so the "open" is the US equity open (13:30 UTC by default), when crypto
  volume peaks. Opening range = first 15 minutes; a setup triggers on the first 5-minute CLOSE beyond it
  within the entry window.
* Plan: stop at the opening range's midpoint, target = R x the risk (1R default), flat by session end.
* Time Machine: 10,000 replays of the rest of the session, each 5-minute bar drawn from the coin's own bars
  at the same time of day (+/-2 bars) over the last 60 days, plus the same rule re-run on every past day.
  Unlike the original (stocks, no commissions) every R here is AFTER Delta's round-trip costs.
* Nothing trades without an explicit approval in the dashboard.
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .allweather import PERP_COST
from .broker import PaperBroker
from .client import DeltaClient
from .data import candles_df, load_products
from .notify import Notifier
from .scanner import liquid_universe
from .state import State
from .strategy import Signal

MODE = "paper_daytrader"
BAR = pd.Timedelta(minutes=5)


@dataclass
class DTConfig:
    session_start_utc: str = "13:30"
    or_minutes: int = 15
    entry_window_min: int = 120          # setups only count this long after the open
    session_hours: float = 6.5           # flat by open + 6.5h (like a stock session)
    target_r: float = 1.0
    max_loss_pct: float = 0.01           # max loss per trade, fraction of equity (the "budget")
    daily_loss_pct: float = 0.03
    max_trades_day: int = 3
    max_exposure_x: float = 3.0
    coins: int = 15
    replays: int = 10_000
    seed: int = 19
    window: int = 2


def session_open(day: pd.Timestamp, cfg: DTConfig) -> pd.Timestamp:
    h, m = map(int, cfg.session_start_utc.split(":"))
    return day.normalize() + pd.Timedelta(hours=h, minutes=m)


def n_or(cfg: DTConfig) -> int:
    return cfg.or_minutes // 5


def n_session(cfg: DTConfig) -> int:
    return int(cfg.session_hours * 12)


def day_bars(df: pd.DataFrame, day: pd.Timestamp, cfg: DTConfig) -> pd.DataFrame:
    o = session_open(day, cfg)
    return df[(df.index >= o) & (df.index < o + n_session(cfg) * BAR)]


def orb_trigger(bars: pd.DataFrame, cfg: DTConfig, cutoff_slot: int | None = None) -> dict | None:
    """First 5-minute close beyond the opening range inside the entry window."""
    k = n_or(cfg)
    if len(bars) <= k:
        return None
    rng = bars.iloc[:k]
    hi, lo = float(rng["high"].max()), float(rng["low"].min())
    last = min(len(bars), k + cfg.entry_window_min // 5)
    if cutoff_slot is not None:
        last = min(last, cutoff_slot)
    for i in range(k, last):
        c = float(bars["close"].iloc[i])
        if c > hi or c < lo:
            side = 1 if c > hi else -1
            stop = (hi + lo) / 2
            risk = abs(c - stop)
            if risk <= 0:
                return None
            return {"slot": i, "time": bars.index[i], "side": side, "entry": c, "stop": stop,
                    "target": c + side * cfg.target_r * risk, "risk": risk, "or_high": hi, "or_low": lo}
    return None


def cost_r(entry: float, risk: float) -> float:
    return 2 * PERP_COST * entry / risk


def walk(bars: pd.DataFrame, start: int, side: int, entry: float, stop: float, target: float, risk: float, tr: float):
    """Real outcome from bar `start` on: stop checked first, then target, else the last close."""
    for j in range(start, len(bars)):
        h, l = float(bars["high"].iloc[j]), float(bars["low"].iloc[j])
        if (l <= stop) if side > 0 else (h >= stop):
            return "stop", -1.0 - cost_r(entry, risk)
        if (h >= target) if side > 0 else (l <= target):
            return "target", tr - cost_r(entry, risk)
    last = float(bars["close"].iloc[-1])
    return "neither", side * (last - entry) / risk - cost_r(entry, risk)


def time_machine(df: pd.DataFrame, plan: dict, cfg: DTConfig) -> dict:
    """10,000 replays of the rest of the session + the same rule on every past session."""
    today = plan["time"].normalize()
    days = sorted({t.normalize() for t in df.index if t.normalize() < today})[-60:]
    N = n_session(cfg)
    ratios = {}                                          # slot -> list of (o, h, l, c) ratios to previous close
    history = []
    for d in days:
        b = day_bars(df, d, cfg)
        if len(b) < N:
            continue                                     # full sessions only
        prev = df[df.index < b.index[0]]["close"]
        if prev.empty:
            continue
        pc = float(prev.iloc[-1])
        for s in range(N):
            row = b.iloc[s]
            ratios.setdefault(s, []).append((row["open"] / pc, row["high"] / pc, row["low"] / pc, row["close"] / pc))
            pc = float(row["close"])
        trig = orb_trigger(b, cfg)
        if trig:
            outcome, r = walk(b, trig["slot"] + 1, trig["side"], trig["entry"], trig["stop"], trig["target"],
                              trig["risk"], cfg.target_r)
            history.append({"day": str(d.date()), "side": trig["side"], "outcome": outcome, "r": r})
    pool = {s: np.array(v) for s, v in ratios.items()}
    rng = np.random.default_rng(cfg.seed)
    M = cfg.replays
    side, entry, stop, target, risk = plan["side"], plan["entry"], plan["stop"], plan["target"], plan["risk"]
    price = np.full(M, entry)
    state = np.zeros(M, dtype=int)                       # 0 open, 1 target, -1 stop
    for s in range(plan["slot"] + 1, N):
        cand = [pool[x] for x in range(max(0, s - cfg.window), min(N, s + cfg.window + 1)) if x in pool]
        if not cand:
            continue
        p = np.concatenate(cand)
        pick = p[rng.integers(len(p), size=M)]
        hi, lo, cl = price * pick[:, 1], price * pick[:, 2], price * pick[:, 3]
        open_ = state == 0
        hit_stop = open_ & ((lo <= stop) if side > 0 else (hi >= stop))
        hit_tgt = open_ & ~hit_stop & ((hi >= target) if side > 0 else (lo <= target))
        state[hit_stop], state[hit_tgt] = -1, 1
        price = np.where(state == 0, cl, price)
    cr = cost_r(entry, risk)
    r = np.where(state == 1, cfg.target_r, np.where(state == -1, -1.0, side * (price - entry) / risk)) - cr
    decided = (state != 0).sum()
    p_tgt = float((state == 1).mean())
    tf_decided = float((state == 1).sum() / decided) if decided else 0.0
    be = 1 / (1 + cfg.target_r)
    gap = (tf_decided - be) * 100
    if len(pool.get(0, [])) < 20:
        verdict = "Not enough history to judge"
    elif abs(gap) < 5:
        verdict = "Close to a coin flip" if cfg.target_r == 1 else "Close to break-even"
    elif abs(gap) < 15:
        verdict = "Leans toward the target" if gap > 0 else "Leans toward the stop"
    else:
        verdict = "Strongly toward the target" if gap > 0 else "Strongly toward the stop"
    hist_r = [h["r"] for h in history]
    return {"replays": M, "sessions": len(pool.get(0, [])), "target_first": p_tgt, "stop_first": float((state == -1).mean()),
            "neither": float((state == 0).mean()), "avg_r": float(r.mean()), "cost_r": cr,
            "target_share_of_decided": tf_decided, "break_even": be, "verdict": verdict,
            "history": {"fired": len(history), "targets": sum(h["outcome"] == "target" for h in history),
                        "stops": sum(h["outcome"] == "stop" for h in history),
                        "neither": sum(h["outcome"] == "neither" for h in history),
                        "avg_r": float(np.mean(hist_r)) if hist_r else 0.0, "days": history[-12:]}}


class DayTraderBot:
    def __init__(self, s, cfg: DTConfig | None = None):
        if s.is_live:
            raise RuntimeError("Day Trader is paper-only")
        self.s, self.cfg = s, cfg or DTConfig()
        self.client = DeltaClient(s.base_url)
        self.notify = Notifier(s)
        self.products = load_products(self.client)
        self.state = State.load(MODE)
        self.state.mode = MODE
        self.broker = PaperBroker(self.client, s, self.state, self.products)
        self.candidates = []
        self.state.extra.setdefault("dt", {"tickets": {}, "seen": [], "log": []})
        self._cache: dict = {}

    # ------------------------------------------------------------ helpers
    @property
    def dt(self) -> dict:
        return self.state.extra["dt"]

    def new_session(self) -> None:
        self.state.trades.clear()
        self.state.paper_cash = self.s.paper_start_equity
        self.state.extra["dt"] = {"tickets": {}, "seen": [], "log": []}
        self.state.halted_reason = ""
        self.state.save()

    def say(self, text: str) -> None:
        self.dt["log"].append([time.time(), text])
        self.dt["log"] = self.dt["log"][-60:]
        self.notify(f"DAY TRADER: {text}")

    def bars(self, sym: str, days: int = 61) -> pd.DataFrame:
        hit = self._cache.get(sym)
        if hit and time.time() - hit[0] < 240:
            return hit[1]
        df = candles_df(self.client, sym, "5m", days * 288)
        self._cache[sym] = (time.time(), df)
        return df

    def today_trades(self) -> list[dict]:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        return [t for t in self.dt["tickets"].values() if t.get("approved_at") and
                time.strftime("%Y-%m-%d", time.gmtime(t["approved_at"])) == today]

    # ------------------------------------------------------------ pipeline
    def scan(self, now: pd.Timestamp | None = None) -> list[dict]:
        """Scanner: coins whose 5-minute close broke the opening range in today's entry window."""
        now = now or pd.Timestamp.now(tz="UTC")
        cfg = self.cfg
        o = session_open(now, cfg)
        if now < o + cfg.or_minutes * pd.Timedelta(minutes=1) + BAR:
            return []
        out = []
        for sym, _ in liquid_universe(self.client, self.products, self.s)[:cfg.coins]:
            df = self.bars(sym)
            b = day_bars(df, now, cfg)
            trig = orb_trigger(b, cfg)
            if trig:
                out.append({"symbol": sym, **trig})
        return out

    def build_ticket(self, sym: str, trig: dict) -> dict:
        cfg = self.cfg
        df = self.bars(sym)
        tm = time_machine(df, trig, cfg)
        prod = self.products[sym]
        equity = self.broker.equity()
        max_loss = cfg.max_loss_pct * equity
        per_contract = (trig["risk"] + 2 * PERP_COST * trig["entry"]) * prod.contract_value
        size = math.floor(max_loss / per_contract) if per_contract > 0 else 0
        loss_at_stop = size * per_contract
        notional = size * prod.contract_value * trig["entry"]
        o = session_open(trig["time"], cfg)
        checks = []

        def chk(name, ok, why):
            checks.append({"check": name, "ok": bool(ok), "why": why})
        chk("size", size >= 1, f"{size} contracts" if size >= 1 else "1 contract would lose more than the max loss")
        chk("max loss", loss_at_stop <= max_loss + 1e-9, f"{loss_at_stop:.2f} <= {max_loss:.2f} USD at the stop")
        chk("reward/risk", cfg.target_r >= 1.0, f"target {cfg.target_r:g}R")
        realized_today = sum(t.get("pnl", 0.0) for t in self.today_trades())
        chk("daily loss limit", realized_today > -cfg.daily_loss_pct * equity, f"today {realized_today:+.2f} USD")
        chk("trades per day", len(self.today_trades()) < cfg.max_trades_day, f"{len(self.today_trades())} of {cfg.max_trades_day} used")
        open_notional = sum(self.products[k].notional(t.size, t.entry_price) for k, t in self.state.trades.items())
        chk("exposure", open_notional + notional <= cfg.max_exposure_x * equity, f"{(open_notional + notional) / max(equity, 1e-9):.1f}x equity")
        chk("entry window", trig["time"] < o + pd.Timedelta(minutes=cfg.or_minutes + cfg.entry_window_min), "inside the window")
        chk("enough history", tm["sessions"] >= 20, f"{tm['sessions']} past sessions (need 20)")
        chk("positive odds after fees", tm["avg_r"] > 0,
            f"Time Machine average {tm['avg_r']:+.2f}R after {tm['cost_r']:.2f}R fees"
            + ("" if tm["avg_r"] > 0 else " - expected to lose; approving is allowed but not advised"))
        chk("not already in", sym not in self.state.trades, "no open position" if sym not in self.state.trades else "already holding")
        return {"id": uuid.uuid4().hex[:8], "symbol": sym, "side": "buy" if trig["side"] > 0 else "sell",
                "entry": trig["entry"], "stop": trig["stop"], "target": trig["target"], "or_high": trig["or_high"],
                "or_low": trig["or_low"], "trigger_time": int(trig["time"].timestamp()),
                "flat_by": int((o + pd.Timedelta(hours=cfg.session_hours)).timestamp()),
                "size": size, "loss_at_stop": loss_at_stop, "notional": notional, "tm": tm, "checks": checks,
                "risk_pass": all(c["ok"] for c in checks if c["check"] != "positive odds after fees"),
                "warning": next((c["why"] for c in checks if c["check"] == "positive odds after fees" and not c["ok"]), ""), "status": "awaiting approval", "created": time.time()}

    def approve(self, tid: str) -> dict:
        t = self.dt["tickets"].get(tid)
        if not t or t["status"] != "awaiting approval":
            return {"ok": False, "error": "ticket not awaiting approval"}
        if not t["risk_pass"]:
            return {"ok": False, "error": "risk check BLOCKED this ticket"}
        if time.time() > t["flat_by"]:
            t["status"] = "expired"
            return {"ok": False, "error": "session already over"}
        prod = self.products[t["symbol"]]
        m = float(self.broker.quote(t["symbol"])["mark"])
        side = 1 if t["side"] == "buy" else -1
        if (m <= t["stop"]) if side > 0 else (m >= t["stop"]):
            t["status"] = "expired"
            return {"ok": False, "error": "price is already through the stop"}
        sig = Signal(t["side"], m, t["stop"], t["target"], 0)
        fill = self.broker.open(prod, sig, t["size"], signal_symbol=t["symbol"])
        tr = self.state.trades[t["symbol"]]
        tr.agent, tr.hold_until = "daytrader", float(t["flat_by"])
        t.update(status="open", approved_at=time.time(), fill=fill)
        self.say(f"APPROVED {t['side'].upper()} {t['size']} {t['symbol']} @ {fill:.6g} (stop {t['stop']:.6g}, target {t['target']:.6g})")
        self.state.save()
        return {"ok": True, "fill": fill}

    def reject(self, tid: str) -> dict:
        t = self.dt["tickets"].get(tid)
        if t and t["status"] == "awaiting approval":
            t["status"] = "rejected"
            self.state.save()
        return {"ok": True}

    def monitor(self) -> None:
        """Alerts: stop, target, and flat by the session end."""
        for sym, tr in list(self.state.trades.items()):
            q = self.broker.quote(sym)["mark"]
            long = tr.side == "buy"
            reason, px = None, q
            if tr.hold_until and time.time() >= tr.hold_until:
                reason = "flat_by_session_end"
            elif (q <= tr.stop) if long else (q >= tr.stop):
                reason, px = "stop", tr.stop
            elif (q >= tr.target) if long else (q <= tr.target):
                reason, px = "target", tr.target
            if reason:
                pnl = self.broker.close(self.products[sym], reason, px, maker=reason == "target")
                for t in self.dt["tickets"].values():
                    if t["symbol"] == sym and t["status"] == "open":
                        t.update(status=f"closed: {reason}", pnl=pnl)
                self.say(f"{sym} {reason.replace('_', ' ')}: P&L {pnl:+.2f} USD")

    def step(self) -> None:
        tk = self.client.tickers("perpetual_futures")
        self.broker.set_marks({t["symbol"]: float(t.get("mark_price") or 0) for t in tk})
        self.monitor()
        for trig in self.scan():
            key = f"{trig['symbol']}|{trig['time'].date()}"
            if key in self.dt["seen"]:
                continue
            self.dt["seen"].append(key)
            self.dt["seen"] = self.dt["seen"][-300:]
            t = self.build_ticket(trig["symbol"], trig)
            self.dt["tickets"][t["id"]] = t
            self.say(f"NEW SETUP {t['symbol']} {t['side'].upper()} @ {t['entry']:.6g}: Time Machine "
                     f"{t['tm']['target_first']:.0%} target / {t['tm']['stop_first']:.0%} stop -> {t['tm']['verdict']}; "
                     f"risk {'PASS' if t['risk_pass'] else 'BLOCK'}. Waiting for your approval.")
        for t in self.dt["tickets"].values():
            if t["status"] == "awaiting approval" and time.time() > t["flat_by"]:
                t["status"] = "expired"
        self.state.last_scan = time.time()
        self.state.save()

    def run(self, stop=None) -> None:
        nxt = 0.0
        while not (stop and stop.is_set()):
            try:
                if time.time() >= nxt:
                    self.step()
                    nxt = time.time() + 60
                else:
                    tk = self.client.tickers("perpetual_futures")
                    self.broker.set_marks({t["symbol"]: float(t.get("mark_price") or 0) for t in tk})
                    self.monitor()
                    self.state.save()
            except Exception as e:
                import logging
                logging.getLogger("deltabot.daytrader").exception("day trader: %s", e)
            (stop.wait if stop else time.sleep)(5)

    def dt_view(self, rate: float) -> dict:
        tickets = sorted(self.dt["tickets"].values(), key=lambda t: -t["created"])[:20]
        now = pd.Timestamp.now(tz="UTC")
        o = session_open(now, self.cfg)
        return {"session_open": int(o.timestamp()), "or_end": int((o + pd.Timedelta(minutes=self.cfg.or_minutes)).timestamp()),
                "window_end": int((o + pd.Timedelta(minutes=self.cfg.or_minutes + self.cfg.entry_window_min)).timestamp()),
                "flat_by": int((o + pd.Timedelta(hours=self.cfg.session_hours)).timestamp()),
                "tickets": tickets, "log": self.dt["log"][-20:], "cfg": self.cfg.__dict__}
