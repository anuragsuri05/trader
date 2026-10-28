"""Survival agent (PAPER ONLY) - "pay for yourself or you die", built on Delta Exchange.

The agent starts with a small bankroll and must pay rent (default $200/month, like an AI subscription)
out of its profits. If its balance reaches $0 it dies.

Desks:
  CARRY  - market-neutral funding carry (the all-weather carry sleeve)           } core: steady income
  TREND  - time-series momentum across coins (the all-weather trend sleeve)      }
  SCOUT  - Claude Fable 5.1 + web search: reads live news / X sentiment every 30 min
  QUANT  - ChatGPT reads the numbers (1m-1h data, 5-second prices) every 10 min

Rules for the AI desks (from the post): they must state a probability that the target is hit before the
stop; code computes the break-even probability from the stop/target distances and costs and only trades
when the edge is >= 8 points (the "mispricing > 8%" rule), sized by the Kelly formula, capped at 6% of
the bankroll at risk. A desk that loses 25% of its budget is shut down and its money moves to the core
("learns to survive"). Below 50% of the starting balance the AI desks pause and only the core trades.
"""
from __future__ import annotations

import collections
import csv
import json
import logging
import math
import os
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path

from .ai_traders import HOLDS, find_claude_cli
from .allweather import PERP_COST
from .allweather_live import AllWeatherBot
from .broker import PaperBroker
from .claude_trader import ClaudeTrader, snapshot_text
from .client import RESOLUTION_SECONDS, DeltaClient
from .data import candles_df, load_products
from .notify import Notifier
from .scanner import liquid_universe
from .state import STATE_DIR, State
from .strategy import Signal

log = logging.getLogger("deltabot.survival")
MODE = "paper_survival"
CORE_SHARE = 0.6
KELLY_CAP, EDGE_MIN = 0.06, 0.08
DESK_DEATH = 0.25
SURVIVAL_LINE = 0.5
DESKS = {
    "scout": {"label": "SCOUT", "who": "Claude Fable 5.1 + web search", "backend": "claude", "model": "claude-fable-5-1", "every": 1800},
    "quant": {"label": "QUANT", "who": "ChatGPT (gpt-5.2)", "backend": "openai", "model": "gpt-5.2", "every": 600},
}

PROMPT = """You are {label}, a desk of an autonomous PAPER-trading agent on Delta Exchange (crypto perpetual futures).
The agent started with a small bankroll, pays rent every day and dies if its balance reaches $0, so every
trade must genuinely be worth taking.

For each trade you propose, give the probability (0-1) that the TARGET is hit before the STOP. Code then
computes the break-even probability from your stop/target distances and trading costs (~0.2% round trip)
and only trades if your probability beats break-even by at least 8 points; size = Kelly fraction capped at
6% of the bankroll at risk. Overconfident probabilities lose money fast - be calibrated. Proposing nothing
is fine.

hold = how long the trade may stay open: "15s", "1m", "5m", "15m" or "1h" (closed automatically after).
You may also list symbols of YOUR open positions to close now in `close`.
{extra}
Keep `market_read` under 80 words and each `reason` to one or two sentences citing evidence."""

SCOUT_EXTRA = """You can search the web. Before deciding, check the latest news and X (Twitter) sentiment for the coins
in the snapshot (and the market overall): listings, hacks, unlocks, ETF flows, liquidations, big wallet
moves, trending narratives. Trade only where sentiment gives a real, timely edge."""
QUANT_EXTRA = """You cannot browse. Decide from the numbers in the snapshot only."""

SCHEMA = {"type": "object", "properties": {
    "market_read": {"type": "string"},
    "trades": {"type": "array", "items": {"type": "object", "properties": {
        "symbol": {"type": "string"}, "side": {"type": "string", "enum": ["buy", "sell"]},
        "stop_price": {"type": "number"}, "target_price": {"type": "number"},
        "p_target_first": {"type": "number"}, "hold": {"type": "string", "enum": list(HOLDS)},
        "reason": {"type": "string"}},
        "required": ["symbol", "side", "stop_price", "target_price", "p_target_first", "hold", "reason"],
        "additionalProperties": False}},
    "close": {"type": "array", "items": {"type": "string"}}},
    "required": ["market_read", "trades", "close"], "additionalProperties": False}


def kelly(p: float, entry: float, stop: float, target: float) -> dict:
    """Break-even probability, edge and Kelly fraction for a stop/target bet after round-trip costs."""
    cost = 2 * PERP_COST * entry
    sl, tp = abs(entry - stop), abs(target - entry)
    b = (tp - cost) / (sl + cost) if sl + cost > 0 else 0.0
    if b <= 0:
        return {"b": b, "p_be": 1.0, "edge": -1.0, "f": 0.0}
    p_be = 1 / (1 + b)
    f = (p * b - (1 - p)) / b
    return {"b": b, "p_be": p_be, "edge": p - p_be, "f": max(0.0, f)}


class Shim:
    def __init__(self, bot):
        self.bot, self.marks = bot, {}

    def equity(self) -> float:
        return self.bot.equity()

    def quote(self, symbol: str) -> dict:
        m = self.marks.get(symbol) or 0.0
        return {"mark": m, "bid": m, "ask": m, "spot": m}


class SurvivalBot:
    def __init__(self, s, rent_per_day: float = 200 / 30):
        if s.is_live:
            raise RuntimeError("The survival agent is paper-only")
        self.s, self.rent_per_day = s, rent_per_day
        self.client = DeltaClient(s.base_url)
        self.notify = Notifier(s)
        self.products = load_products(self.client)
        self.state = State.load(MODE)                     # AI-desk ledger (shows in the dashboard tables)
        self.state.mode = MODE
        self.ai = PaperBroker(self.client, replace(s, paper_start_equity=s.paper_start_equity * (1 - CORE_SHARE)),
                              self.state, self.products)
        self.core = AllWeatherBot(replace(s, paper_start_equity=s.paper_start_equity * CORE_SHARE), "balanced",
                                  mode=MODE + "_core", committee=False)
        self.broker = Shim(self)
        self.candidates = []
        self.lock = threading.RLock()
        self.cli = find_claude_cli()
        self.busy = {k: False for k in DESKS}
        self.ticks: dict[str, collections.deque] = {}
        self._cache: dict = {}
        self._universe: list[str] = []
        if "sv" not in self.state.extra:
            self._fresh()

    # ------------------------------------------------------------ ledgers
    def _fresh(self) -> None:
        start = self.s.paper_start_equity
        self.state.trades.clear()
        self.state.paper_cash = start * (1 - CORE_SHARE)
        self.state.halted_reason = ""
        each = start * (1 - CORE_SHARE) / len(DESKS)
        self.state.extra = {"sv": {"start": start, "born": time.time(), "rent": 0.0, "last_tick": time.time(),
                                   "alive": True, "died": "", "survival_mode": False,
                                   "desks": {k: {"alive": True, "budget": each, "died": "", "last": {}, "next_at": 0.0,
                                                 "calls": 0, "tokens": 0} for k in DESKS},
                                   "events": []}}
        self.core._fresh(start * CORE_SHARE)

    @property
    def sv(self) -> dict:
        return self.state.extra["sv"]

    def equity(self) -> float:
        return self.core.equity() + self.ai.equity() - self.sv["rent"]

    def new_session(self) -> None:
        self._fresh()
        self.core.state.save()
        self.state.save()
        self.event("born", f"Agent born with {self.s.paper_start_equity:.2f} USD. Rent {self.rent_per_day * 30:.0f} USD/month. "
                           f"Pay for yourself or die.")

    def event(self, kind: str, text: str) -> None:
        self.sv["events"].append([time.time(), kind, text])
        self.sv["events"] = self.sv["events"][-80:]
        self.notify(f"SURVIVAL: {text}")

    # ------------------------------------------------------------ market data
    def universe(self) -> list[str]:
        if not self._universe or time.time() - self._cache.get("_u", 0) > 1800:
            self._universe = [x for x, _ in liquid_universe(self.client, self.products, self.s)[:12]]
            self._cache["_u"] = time.time()
        return self._universe

    def candles(self, sym: str, tf: str):
        step = RESOLUTION_SECONDS[tf]
        key, bar = (sym, tf), time.time() // step * step
        hit = self._cache.get(key)
        if hit and hit[0] >= bar:
            return hit[1]
        df = candles_df(self.client, sym, tf, 300)
        self._cache[key] = (bar, df)
        return df

    def _tickers(self) -> None:
        tk = self.client.tickers("perpetual_futures")
        marks = {t["symbol"]: float(t.get("mark_price") or t.get("close") or 0) for t in tk}
        self.broker.marks = self.ai.marks = marks
        self.ai.marks_time = time.time()
        self.core.broker.marks = marks
        self.core.funding_now = {t["symbol"]: float(t.get("funding_rate") or 0) / 100 for t in tk}
        for sym in self._universe:
            if marks.get(sym):
                self.ticks.setdefault(sym, collections.deque(maxlen=60)).append(marks[sym])

    # ------------------------------------------------------------ desk P&L
    def desk_pnl(self) -> dict:
        out = {k: {"realized": 0.0, "unrealized": 0.0, "trades": 0, "wins": 0} for k in DESKS}
        path = STATE_DIR / f"trades_{MODE}.csv"
        if path.exists():
            with path.open() as f:
                for r in csv.DictReader(f):
                    a = r.get("agent") or ""
                    if a in out:
                        try:
                            p = float(r["pnl"] or 0)
                        except ValueError:
                            continue
                        out[a]["realized"] += p
                        out[a]["trades"] += 1
                        out[a]["wins"] += p > 0
        for sym, t in list(self.state.trades.items()):
            if t.agent in out:
                m = self.ai.marks.get(sym) or t.entry_price
                out[t.agent]["unrealized"] += (1 if t.side == "buy" else -1) * (m - t.entry_price) * t.size * self.products[sym].contract_value
        for v in out.values():
            v["pnl"] = v["realized"] + v["unrealized"]
        return out

    # ------------------------------------------------------------ AI desks
    def _snapshot(self, desk: str) -> str:
        from .indicators import atr as _atr, rsi as _rsi
        coins = []
        for sym in self.universe():
            try:
                c1, c5 = self.candles(sym, "1m"), self.candles(sym, "5m")
                snap = ClaudeTrader.coin_snapshot(sym, c5, self.candles(sym, "15m"), self.candles(sym, "1h"), self.s, [],
                                                  self.products[sym].contract_value)
            except Exception as e:
                log.warning("snapshot %s: %s", sym, e)
                continue
            if not snap or len(c1) < 30:
                continue
            snap["closes_1m"] = [float(x) for x in c1["close"].iloc[-15:]]
            snap["rsi_1m"] = round(float(_rsi(c1["close"], 14).iloc[-1]), 1)
            snap["atr_pct_1m"] = round(float(_atr(c1, 14).iloc[-1] / c1["close"].iloc[-1] * 100), 3)
            snap["ticks_5s"] = list(self.ticks.get(sym, []))[-24:]
            coins.append(snap)
        with self.lock:
            d = self.sv["desks"][desk]
            pnl = self.desk_pnl()[desk]
            mine = [{"sym": k, "side": t.side, "entry": t.entry_price, "stop": t.stop, "target": t.target,
                     "mark": self.ai.marks.get(k), "seconds_left": round(t.hold_until - time.time()) if t.hold_until else None}
                    for k, t in list(self.state.trades.items()) if t.agent == desk]
            account = {"agent_balance_usd": round(self.equity(), 2), "agent_start_usd": round(self.sv["start"], 2),
                       "rent_paid_so_far_usd": round(self.sv["rent"], 2), "rent_per_day_usd": round(self.rent_per_day, 2),
                       "my_desk_budget_usd": round(d["budget"], 2), "my_desk_pnl_usd": round(pnl["pnl"], 3),
                       "desk_shut_down_if_pnl_below_usd": round(-DESK_DEATH * d["budget"], 2), "my_open_positions": mine,
                       "max_risk_per_trade_usd": round(KELLY_CAP * self.equity(), 2)}
        return snapshot_text(coins, account)

    def _ask(self, desk: str, text: str) -> tuple[dict | None, str]:
        cfg = DESKS[desk]
        prompt = PROMPT.format(label=cfg["label"], extra=SCOUT_EXTRA if desk == "scout" else QUANT_EXTRA)
        try:
            if cfg["backend"] == "claude":
                if not self.cli:
                    return None, "Claude Code CLI not found"
                env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
                p = subprocess.run([self.cli, "-p", "--output-format", "json", "--no-session-persistence", "--model", cfg["model"],
                                    "--tools", "WebSearch", "--allowedTools", "WebSearch", "--system-prompt", prompt,
                                    "--json-schema", json.dumps(SCHEMA)], input=text, capture_output=True, text=True,
                                   timeout=900, env=env, cwd=STATE_DIR)
                out = json.loads(p.stdout)
                if out.get("is_error"):
                    return None, "Claude: " + str(out.get("result"))[:200]
                return out.get("structured_output") or json.loads(out.get("result") or "{}"), ""
            if not os.environ.get("OPENAI_API_KEY"):
                return None, "no OPENAI_API_KEY"
            from openai import OpenAI
            r = OpenAI(timeout=180, max_retries=2).chat.completions.create(
                model=cfg["model"], messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
                response_format={"type": "json_schema", "json_schema": {"name": "desk", "strict": True, "schema": SCHEMA}})
            with self.lock:
                self.sv["desks"][desk]["tokens"] += r.usage.total_tokens if r.usage else 0
            return json.loads(r.choices[0].message.content or "{}"), ""
        except Exception as e:
            return None, f"{type(e).__name__}: {str(e)[:200]}"

    def _desk_turn(self, desk: str) -> None:
        try:
            text = self._snapshot(desk)
            t0 = time.time()
            data, err = self._ask(desk, text)
            rec = {"time": time.time(), "seconds": round(time.time() - t0, 1), "read": (data or {}).get("market_read", ""),
                   "error": err, "trades": []}
            if data:
                with self.lock:
                    for sym in data.get("close", []):
                        t = self.state.trades.get(sym)
                        if t and t.agent == desk:
                            pnl = self.ai.close(self.products[sym], "desk_close", self.ai.marks.get(sym))
                            rec["trades"].append({"symbol": sym, "verdict": f"closed, P&L {pnl:+.3f} USD"})
                    for tr in data.get("trades", []):
                        rec["trades"].append({**tr, "verdict": self._apply(desk, tr)})
            with self.lock:
                d = self.sv["desks"][desk]
                d["last"], d["calls"] = rec, d["calls"] + 1
                d["next_at"] = time.time() + DESKS[desk]["every"]
                summary = err or ("; ".join(f"{x.get('side', '').upper()} {x['symbol']}: {x['verdict']}" for x in rec["trades"])
                                  or "watched, no trade worth it")
                self.event(desk, f"{DESKS[desk]['label']} ({rec['seconds']}s): {summary}")
                self.state.save()
        except Exception as e:
            log.exception("desk %s: %s", desk, e)
            self.sv["desks"][desk]["next_at"] = time.time() + 300
        finally:
            self.busy[desk] = False

    def _apply(self, desk: str, tr: dict) -> str:
        sym = tr["symbol"]
        if sym not in self.universe():
            return "rejected: not in the scanned list"
        if sym in self.state.trades:
            return "rejected: already holding this coin"
        m = self.ai.marks.get(sym)
        if not m:
            return "rejected: no live price"
        long = tr["side"] == "buy"
        if (tr["stop_price"] >= m) if long else (tr["stop_price"] <= m):
            return "rejected: stop on the wrong side of the price"
        if (tr["target_price"] <= m) if long else (tr["target_price"] >= m):
            return "rejected: target on the wrong side of the price"
        p = min(0.99, max(0.0, float(tr["p_target_first"])))
        k = kelly(p, m, tr["stop_price"], tr["target_price"])
        if k["edge"] < EDGE_MIN:
            return f"no trade: p={p:.2f} vs break-even {k['p_be']:.2f} (edge {k['edge'] * 100:+.0f} pts < 8)"
        frac = min(k["f"], KELLY_CAP)
        risk_usd = frac * self.equity()
        prod = self.products[sym]
        per_contract = (abs(m - tr["stop_price"]) + 2 * PERP_COST * m) * prod.contract_value
        size = math.floor(risk_usd / per_contract) if per_contract > 0 else 0
        budget = self.sv["desks"][desk]["budget"]
        size = min(size, math.floor(5 * budget / (prod.contract_value * m)))       # max 5x the desk budget in exposure
        if size < 1:
            return f"rejected: 1 contract would risk more than the Kelly size ({risk_usd:.2f} USD)"
        sig = Signal(tr["side"], m, tr["stop_price"], tr["target_price"], 0)
        fill = self.ai.open(prod, sig, size, signal_symbol=sym)
        t = self.state.trades[sym]
        t.agent, t.entry_tf, t.hold_until = desk, "1m", time.time() + HOLDS[tr["hold"]]
        actual = size * per_contract
        capped = " (capped by desk exposure)" if actual < 0.99 * risk_usd else ""
        return (f"{'BOUGHT' if long else 'SOLD'} {size} @ {fill:g} | p {p:.2f} vs b/e {k['p_be']:.2f}, "
                f"Kelly {k['f'] * 100:.1f}% -> risking {actual:.2f} USD ({actual / self.equity() * 100:.1f}% of bankroll)"
                f"{capped}, hold {tr['hold']}")

    # ------------------------------------------------------------ survival rules
    def _monitor_positions(self) -> None:
        for sym, t in list(self.state.trades.items()):
            m = self.ai.marks.get(sym)
            if not m:
                continue
            long = t.side == "buy"
            reason, px = None, m
            if t.hold_until and time.time() >= t.hold_until:
                reason = "hold_time_up"
            elif (m <= t.stop) if long else (m >= t.stop):
                reason, px = "stop_loss", t.stop
            elif (m >= t.target) if long else (m <= t.target):
                reason, px = "take_profit", t.target
            if reason:
                pnl = self.ai.close(self.products[sym], reason, px, maker=reason == "take_profit")
                self.event(t.agent, f"{DESKS.get(t.agent, {}).get('label', t.agent)} {sym} {reason.replace('_', ' ')}: P&L {pnl:+.3f} USD")

    def _rules(self) -> None:
        sv = self.sv
        now = time.time()
        sv["rent"] += self.rent_per_day * (now - sv["last_tick"]) / 86400
        sv["last_tick"] = now
        eq = self.equity()
        if eq <= 0 and sv["alive"]:
            sv["alive"], sv["died"] = False, f"balance hit $0 after {(now - sv['born']) / 3600:.1f} hours"
            for sym in list(self.state.trades):
                self.ai.close(self.products[sym], "agent_died", self.ai.marks.get(sym))
            self.core.aw["pos"], self.core.aw["carry"] = {}, {}
            self.state.halted_reason = "session: AGENT DIED - the balance reached $0"
            self.event("death", f"AGENT DIED: {sv['died']}. Rent paid {sv['rent']:.2f} USD.")
            return
        pnl = self.desk_pnl()
        for k, d in sv["desks"].items():
            if d["alive"] and pnl[k]["pnl"] <= -DESK_DEATH * d["budget"]:
                d["alive"], d["died"] = False, f"lost {-pnl[k]['pnl']:.2f} USD (25% of its budget)"
                for sym, t in list(self.state.trades.items()):
                    if t.agent == k:
                        self.ai.close(self.products[sym], "desk_shut_down", self.ai.marks.get(sym))
                left = max(0.0, d["budget"] + pnl[k]["pnl"])
                self.state.paper_cash -= left
                self.core.aw["cash"] += left
                self.event(k, f"{DESKS[k]['label']} SHUT DOWN ({d['died']}); {left:.2f} USD moved to the core desks")
        was = sv["survival_mode"]
        sv["survival_mode"] = eq < SURVIVAL_LINE * sv["start"] or (was and eq < 0.6 * sv["start"])
        if sv["survival_mode"] != was:
            self.event("mode", "SURVIVAL MODE ON: balance below 50% - AI desks paused, only carry + trend trade" if sv["survival_mode"]
                       else "Survival mode off: AI desks back to work")

    # ------------------------------------------------------------ loop
    def run(self, stop=None) -> None:
        self.universe()
        next_min = 0.0
        while not (stop and stop.is_set()):
            try:
                with self.lock:
                    self._tickers()
                    if self.sv["alive"]:
                        self._monitor_positions()
                if self.sv["alive"] and time.time() >= next_min:
                    last = self.core.aw["last_rebalance"]
                    if not last or time.time() >= (last // 86400 + 1) * 86400 + 300:
                        self.core.rebalance()
                    with self.lock:
                        self.core.accrue()
                        self._rules()
                        self.state.save()
                    next_min = time.time() + 60
                if self.sv["alive"] and not self.sv["survival_mode"]:
                    for k, d in self.sv["desks"].items():
                        if d["alive"] and not self.busy[k] and time.time() >= d["next_at"]:
                            self.busy[k] = True
                            threading.Thread(target=self._desk_turn, args=(k,), daemon=True, name=f"desk-{k}").start()
            except Exception as e:
                log.exception("survival loop: %s", e)
            (stop.wait if stop else time.sleep)(5)

    # ------------------------------------------------------------ view
    def sv_view(self, rate: float) -> dict:
        sv, pnl = self.sv, self.desk_pnl()
        core_eq = self.core.equity()
        aw = self.core.aw
        desks = {"carry": {"label": "CARRY", "who": "funding carry (market-neutral)", "alive": sv["alive"],
                           "pnl": aw["fund_carry"], "detail": f"{sum(1 for v in aw.get('carry_info', {}).values() if v['on'])} coins on"},
                 "trend": {"label": "TREND", "who": "momentum across 14+ coins", "alive": sv["alive"],
                           "pnl": core_eq - CORE_SHARE * sv["start"] - aw["fund_carry"],
                           "detail": f"{len(aw.get('targets', {}))} positions"}}
        for k, d in sv["desks"].items():
            last = d.get("last") or {}
            desks[k] = {"label": DESKS[k]["label"], "who": DESKS[k]["who"], "alive": d["alive"] and sv["alive"],
                        "pnl": pnl[k]["pnl"], "trades": pnl[k]["trades"], "wins": pnl[k]["wins"], "died": d["died"],
                        "busy": self.busy[k], "next_in": max(0, round(d["next_at"] - time.time())), "calls": d["calls"],
                        "read": last.get("read", ""), "error": last.get("error", ""), "decisions": last.get("trades", [])}
        trades = sum(v["trades"] for v in pnl.values())
        wins = sum(v["wins"] for v in pnl.values())
        return {"alive": sv["alive"], "died": sv["died"], "survival_mode": sv["survival_mode"], "start": sv["start"],
                "equity": self.equity(), "rent": sv["rent"], "rent_per_day": self.rent_per_day, "born": sv["born"],
                "trades": trades, "win_rate": wins / trades if trades else None, "desks": desks,
                "events": sv["events"][-25:]}
